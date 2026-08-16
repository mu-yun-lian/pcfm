from __future__ import annotations

import base64
import csv
import copy
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .applicability import PredictionRefusedError
from .assistant import AssistantEngine
from .contracts import Observation, Scenario
from .cognitive_workbench import (
    CognitiveWorkbench,
    CognitiveWorkbenchError,
    load_builtin_hawley_case,
)
from .conversation_mvp import ConversationError, ConversationWorkbench
from .decision_evidence_v1 import decision_evidence_bundle_from_dict
from .demo_people import DEMO_PEOPLE, DEMO_SEED_VERSION
from .evaluation import evaluate_probability_array, report_to_dict
from .expression_renderer import (
    ExpressionRenderer,
    ExpressionRendererError,
    builtin_expression_profile_path,
)
from .ledger import EventLedger, VerificationAuthority, observation_payload
from .model_services import ModelServiceError, ModelServiceManager
from .public_search import BingRssPublicSearch, PublicSearchError
from .storage import load_bundle, save_bundle
from .synthetic import FEATURE_NAMES, generate_population_dataset
from .workflow import (
    fit_person_model,
    load_event_ledger_jsonl,
    predict_with_bundle,
    save_event_ledger_jsonl,
    update_person_model,
)


PRODUCT_FORMAT = "pcfm-local-product-v1"
VERIFIER_ID = "pcfm-local-product"
MINIMUM_PROFILE_SAMPLES = 50
MINIMUM_VALIDATION_SAMPLES = 100


class ProductError(ValueError):
    """An error that is safe to show to an ordinary product user."""


REASON_TEXT = {
    "independent_validation_required": "还没有独立验证数据。",
    "insufficient_person_validation_samples": "独立验证样本不足 100 条。",
    "insufficient_personalization_uplift": "人物模型没有稳定优于群体基线。",
    "personalization_uplift_not_significant": "人物化提升的统计证据还不稳定。",
    "calibration_error_too_high": "概率校准误差过高。",
    "mechanism_misspecification_suspected": "现有线性模型可能遗漏了重要数值结构。",
    "temporal_behavior_drift_suspected": "不同时段的选择规律可能发生变化。",
    "temporal_stability_not_assessed": "时间稳定性数据不足。",
    "feature_distribution_shift": "新情境的数值特征超出了历史数据范围。",
    "local_support_gap": "历史数据中缺少与这个新情境足够接近的样本。",
    "prediction_time_required": "需要提供预测时间。",
    "prediction_precedes_reference_data": "预测时间早于模型参考数据。",
    "stale_model": "模型距离最近参考数据已经过久，需要更新。",
    "unvalidated_domain_label": "这个情境类型没有在适用域数据中验证。",
    "unvalidated_option_pair": "这一组选项文字没有在适用域数据中验证。",
    "unvalidated_context": "这一类情境说明没有在适用域数据中验证。",
    "model_validation_unvalidated": "模型尚未通过独立验证。",
    "model_validation_failed": "模型没有通过独立验证。",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ProductError("日期时间格式无效。") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProductError("日期时间必须包含时区。")
    return parsed


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: object | None = None) -> object:
    if not path.exists():
        if default is not None:
            return default
        raise ProductError(f"本地文件不存在：{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str) -> str:
    normalized = "-".join(str(value).strip().lower().split())
    safe = "".join(
        char
        for char in normalized
        if char.isascii() and (char.isalnum() or char in "-_")
    )
    return safe[:40] or uuid.uuid4().hex[:12]


def _as_choice(value: object, option_a: str, option_b: str) -> int:
    text = str(value).strip()
    if text in {"0", "A", "a", option_a}:
        return 0
    if text in {"1", "B", "b", option_b}:
        return 1
    raise ProductError("真实选择必须是 A、B、0、1 或对应的选项文字。")


def _reason_text(reason: str) -> str:
    return REASON_TEXT.get(reason, f"模型门禁原因：{reason}")


class ProductService:
    """Local product integration around the existing PCFM workflow."""

    def __init__(
        self,
        data_dir: Path,
        *,
        seed_example: bool = True,
        seed_demos: bool = False,
        public_search: object | None | bool = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.people_dir = self.data_dir / "people"
        self.people_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.data_dir / "archived_people"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.model_services = ModelServiceManager(self.data_dir)
        self.cognitive = CognitiveWorkbench(self.people_dir)
        self.conversation = ConversationWorkbench(
            self.people_dir, model_services=self.model_services
        )
        if public_search is None:
            configured_provider = os.environ.get("PCFM_PUBLIC_SEARCH_PROVIDER", "").strip().casefold()
            self.public_search = (
                BingRssPublicSearch() if configured_provider == "bing_rss" else None
            )
        else:
            self.public_search = None if public_search is False else public_search
        self.expression_renderer = ExpressionRenderer(
            builtin_expression_profile_path()
        )
        self.expression_records_path = self.data_dir / "expression_renders.json"
        if not self.expression_records_path.exists():
            _write_json(self.expression_records_path, [])
        self.assistant = AssistantEngine(self, self.data_dir / "assistant_state.json")
        self._lock = threading.RLock()
        if seed_example and not any(self.people_dir.iterdir()):
            self._seed_example()
        if seed_example:
            self._seed_cognitive_example()
        demo_marker = self.data_dir / "demo_seed.json"
        if seed_demos and not demo_marker.exists():
            seeded = self._seed_conversation_demos()
            _write_json(
                demo_marker,
                {
                    "version": DEMO_SEED_VERSION,
                    "seeded_person_ids": seeded,
                    "created_at": _utc_now(),
                },
            )
        self._refresh_demo_metadata()
        if seed_demos:
            self._refresh_demo_sources()
        for path in sorted(self.people_dir.glob("*/person.json")):
            self._conversation_call(
                self.conversation.migrate_evidence_contract, path.parent.name
            )

    def _refresh_demo_metadata(self) -> None:
        """Refresh built-in navigation metadata without touching evidence or chat."""
        for definition in DEMO_PEOPLE:
            person_id = str(definition["person_id"])
            path = self.people_dir / person_id / "person.json"
            if not path.exists():
                continue
            raw = _read_json(path, {})
            if not isinstance(raw, dict) or not raw.get("is_demo"):
                continue
            raw["recommended_questions"] = copy.deepcopy(
                definition.get("recommended_questions", [])
            )
            raw["demo_metadata_version"] = DEMO_SEED_VERSION
            _write_json(path, raw)

    def _refresh_demo_sources(self) -> None:
        """Add newly bundled verified sources to existing demo people once."""
        for definition in DEMO_PEOPLE:
            person_id = str(definition["person_id"])
            person_path = self.people_dir / person_id / "person.json"
            if not person_path.exists():
                continue
            person = _read_json(person_path, {})
            if not isinstance(person, dict) or not person.get("is_demo"):
                continue
            existing_by_url = {
                str(item.get("source_url", "")).rstrip("/"): str(
                    item.get("source_id", "")
                )
                for item in self._conversation_call(
                    self.conversation.sources, person_id
                )
            }
            for source_spec in definition["sources"]:
                source_url = str(source_spec["source_url"])
                normalized_url = source_url.rstrip("/")
                if normalized_url in existing_by_url:
                    aliases = list(source_spec.get("entity_aliases", []))
                    if aliases:
                        self._conversation_call(
                            self.conversation.merge_source_entity_aliases,
                            person_id,
                            existing_by_url[normalized_url],
                            aliases,
                        )
                    continue
                source = self._conversation_call(
                    self.conversation.add_text_source,
                    person_id,
                    title=str(source_spec["title"]),
                    text=str(source_spec["text"]),
                    speaker=str(source_spec["speaker"]),
                    source_date=str(source_spec["source_date"]),
                    dataset_role=str(source_spec["dataset_role"]),
                    source_type="built_in_verified_demo",
                    source_url=source_url,
                    content_authenticity=str(source_spec["content_authenticity"]),
                    source_locator=str(source_spec["source_locator"]),
                    source_context=str(source_spec["source_context"]),
                    original_language=str(source_spec["original_language"]),
                    entity_aliases=list(source_spec.get("entity_aliases", [])),
                )
                self._conversation_call(
                    self.conversation.review_source,
                    person_id,
                    str(source["source_id"]),
                    "confirmed",
                )
                existing_by_url[normalized_url] = str(source["source_id"])

    # ---------- bounded expression rendering ----------

    def capabilities(self) -> dict[str, object]:
        provider = self.public_search
        return {
            "public_search": {
                "available": provider is not None,
                "status": "configured" if provider is not None else "not_configured",
                "provider": getattr(provider, "provider_id", None),
                "candidate_policy": "unverified_reference_only",
            },
            "model_services": {
                "supported_protocols": [
                    "openai_native",
                    "openai_compatible",
                    "anthropic",
                    "gemini",
                    "ollama",
                    "custom_compatible",
                ],
                "configured_count": len(self.model_services.public_state()["services"]),
                "secret_storage": "windows_dpapi_server_only",
            },
        }

    def model_service_state(self) -> dict[str, object]:
        return self.model_services.public_state()

    def save_model_service(self, payload: Mapping[str, object]) -> dict[str, object]:
        try:
            return self.model_services.save_service(payload)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def delete_model_service(self, service_id: str) -> None:
        try:
            self.model_services.delete_service(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def reveal_model_service_key(self, service_id: str) -> str:
        try:
            return self.model_services.reveal_api_key(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def test_model_service(
        self, service_id: str, model_id: str = ""
    ) -> dict[str, object]:
        try:
            return self.model_services.test_connection(service_id, model_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def refresh_model_service_models(self, service_id: str) -> list[str]:
        try:
            return self.model_services.refresh_models(service_id)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def set_model_role(self, role: str, model_ref: str) -> dict[str, object]:
        try:
            return self.model_services.set_role(role, model_ref)
        except ModelServiceError as error:
            raise ProductError(str(error)) from error

    def select_dialogue_model(self, person_id: str, model_ref: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            try:
                if model_ref:
                    self.model_services.resolve_model_ref(model_ref)
                return self.conversation.select_dialogue_model(person_id, model_ref)
            except (ModelServiceError, ConversationError) as error:
                raise ProductError(str(error)) from error

    def expression_profile(self) -> dict[str, object]:
        return self.expression_renderer.profile_summary()

    def expression_records(self) -> list[dict[str, object]]:
        raw = _read_json(self.expression_records_path, [])
        if not isinstance(raw, list):
            raise ProductError("表达渲染记录文件损坏。")
        return [dict(item) for item in raw]

    def render_expression(
        self,
        contract: Mapping[str, object],
        *,
        include_adversarial_probe: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            try:
                result = self.expression_renderer.render(
                    contract,
                    include_adversarial_probe=include_adversarial_probe,
                )
            except ExpressionRendererError as error:
                raise ProductError(str(error)) from error
            records = self.expression_records()
            records.append(result)
            _write_json(self.expression_records_path, records[-200:])
            return result

    # ---------- conversation-first product ----------

    def _conversation_call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except ConversationError as error:
            raise ProductError(str(error)) from error

    def create_conversation_person(
        self,
        *,
        name: str,
        aliases: Sequence[str] = (),
        language: str = "zh",
        description: str = "",
        time_start: str = "",
        time_end: str = "",
        source_mode: str = "user_provided",
        identity_note: str = "",
        focus_domain: str = "",
        avatar: str = "",
        notes: str = "",
    ) -> dict[str, object]:
        with self._lock:
            if source_mode not in {"user_provided", "system_search"}:
                raise ProductError("请选择系统自动搜索或用户自行提供资料。")
            if source_mode == "system_search" and self.public_search is None:
                raise ProductError("系统自动搜索公开资料暂未配置，请选择自行提供原始资料。")
            person = self.create_person(
                name=name,
                description=description,
                feature_names=("evidence_overlap", "intercept"),
            )
            person_id = str(person["person_id"])
            raw_person = self._require_person(person_id)
            raw_person.update(
                {
                    "identity_note": str(identity_note).strip(),
                    "focus_domain": str(focus_domain).strip(),
                    "avatar": str(avatar).strip(),
                    "notes": str(notes).strip(),
                    "collection": {
                        "mode": source_mode,
                        "status": "search_ready" if source_mode == "system_search" else "awaiting_user_materials",
                        "message": (
                            "搜索服务已配置；结果只会进入待审核候选资料。"
                            if source_mode == "system_search"
                            else "等待用户提供原始资料。"
                        ),
                    },
                }
            )
            _write_json(self._person_path(person_id), raw_person)
            self._conversation_call(
                self.conversation.configure,
                person_id,
                aliases=aliases,
                language=language,
                time_start=time_start,
                time_end=time_end,
                source_mode=source_mode,
                identity_note=identity_note,
                focus_domain=focus_domain,
            )
            if source_mode == "system_search":
                self.collect_public_sources(person_id)
            return self.get_person(person_id)

    def collect_public_sources(self, person_id: str) -> dict[str, object]:
        """Collect public source candidates. Discovery never creates training truth."""
        with self._lock:
            person = self._require_person(person_id)
            profile = self._conversation_call(self.conversation.profile, person_id)
            provider_id = getattr(self.public_search, "provider_id", "configured-provider")
            if self.public_search is None:
                collection = {
                    "mode": "system_search",
                    "status": "search_service_not_configured",
                    "message": "公开资料搜索服务未配置，当前不会伪装成已经搜索。",
                }
            else:
                try:
                    results = self.public_search.search(
                        person_name=str(person["name"]),
                        identity_note=str(person.get("identity_note", "")),
                        language=str(profile.get("language", "zh")),
                        limit=8,
                    )
                except Exception as error:
                    collection = {
                        "mode": "system_search",
                        "status": "temporarily_unavailable",
                        "message": f"公开资料搜索暂时失败：{error}",
                        "provider": provider_id,
                    }
                    results = []
                else:
                    for result in results:
                        text = "\n\n".join(
                            value
                            for value in (
                                str(result.get("title", "")),
                                str(result.get("snippet", "")),
                            )
                            if value
                        )
                        try:
                            self.conversation.add_text_source(
                                person_id,
                                title=str(result.get("title", "公开资料候选")),
                                text=text or str(result.get("url", "")),
                                speaker="",
                                source_date=str(result.get("published_at", "")),
                                dataset_role="reference_only",
                                source_type="system_search_result",
                                source_url=str(result.get("url", "")),
                                source_format="search_result",
                                content_authenticity="search_snippet",
                                source_locator=f"search result rank {result.get('provider_rank', '')}",
                                source_context="自动搜索候选；尚未访问原文或确认说话人。",
                            )
                        except ConversationError:
                            continue
                    collection = {
                        "mode": "system_search",
                        "status": "candidates_found" if results else "no_candidates",
                        "message": (
                            f"找到 {len(results)} 条公开资料候选，均需核验原文后才能训练。"
                            if results
                            else "未找到可用的公开资料候选。"
                        ),
                        "provider": provider_id,
                        "candidate_count": len(results),
                    }
            person["collection"] = collection
            _write_json(self._person_path(person_id), person)
            profile["collection"] = collection
            _write_json(
                self._person_dir(person_id) / "conversation_profile.json", profile
            )
            return collection

    def conversation_summary(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.summary, person_id)

    def start_new_conversation(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.start_new_conversation, person_id
            )

    def list_sessions(self, person_id: str) -> list[dict[str, object]]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.list_sessions, person_id)

    def create_session(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.create_session, person_id)

    def switch_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.switch_session, person_id, session_id)

    def rename_session(self, person_id: str, session_id: str, title: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.rename_session, person_id, session_id, title)

    def delete_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.delete_session, person_id, session_id)

    def add_conversation_text_source(
        self,
        person_id: str,
        *,
        title: str,
        text: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        source_url: str = "",
        original_language: str = "",
        translation_of: str = "",
        speaker_scope: str = "single_speaker_entire_document",
        entity_aliases: Sequence[str] = (),
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.add_text_source,
                person_id,
                title=title,
                text=text,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                source_url=source_url,
                original_language=original_language,
                translation_of=translation_of,
                speaker_scope=speaker_scope,
                entity_aliases=entity_aliases,
            )

    def add_conversation_file_source(
        self,
        person_id: str,
        *,
        filename: str,
        content_base64: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        speaker_scope: str = "single_speaker_entire_document",
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.add_file_source,
                person_id,
                filename=filename,
                content_base64=content_base64,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                speaker_scope=speaker_scope,
            )

    def add_conversation_url_source(
        self,
        person_id: str,
        *,
        url: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        speaker_scope: str = "single_speaker_entire_document",
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.add_url_source,
                person_id,
                url=url,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                speaker_scope=speaker_scope,
            )

    def review_conversation_source(
        self, person_id: str, source_id: str, decision: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.review_source, person_id, source_id, decision
            )

    def extract_conversation_response_candidates(
        self, person_id: str, source_id: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.extract_response_event_candidates,
                person_id,
                source_id,
            )

    def review_conversation_response_candidate(
        self,
        person_id: str,
        source_id: str,
        candidate_id: str,
        decision: str,
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.review_response_event_candidate,
                person_id,
                source_id,
                candidate_id,
                decision,
            )

    def send_conversation_message(
        self,
        person_id: str,
        text: str,
        *,
        reality_lookup_requested: bool = False,
        dialogue_model_ref: str = "",
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.send_message,
                person_id,
                text,
                reality_lookup_requested=reality_lookup_requested,
                dialogue_model_ref=dialogue_model_ref,
            )

    def find_conversation_reality_answer(
        self, person_id: str, message_id: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.find_reality_answer, person_id, message_id
            )

    def create_optimization_candidate(
        self,
        person_id: str,
        message_id: str,
        *,
        allow_retry: bool = False,
        comparison_candidate_id: str = "",
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.create_optimization_candidate,
                person_id,
                message_id,
                allow_retry=allow_retry,
                comparison_candidate_id=comparison_candidate_id,
            )

    def review_optimization_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.review_optimization_candidate,
                person_id,
                candidate_id,
                decision,
            )

    def review_optimization_style_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.review_optimization_style_candidate,
                person_id,
                candidate_id,
                decision,
            )

    def rollback_conversation_version(
        self, person_id: str, version_number: int
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.rollback_version, person_id, version_number
            )

    def record_conversation_feedback(
        self, person_id: str, message_id: str, value: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.feedback, person_id, message_id, value
            )

    # ---------- person and local persistence ----------

    def _person_dir(self, person_id: str) -> Path:
        if not person_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in person_id):
            raise ProductError("人物编号无效。")
        path = (self.people_dir / person_id).resolve()
        if path.parent != self.people_dir:
            raise ProductError("人物编号无效。")
        return path

    def _person_path(self, person_id: str) -> Path:
        return self._person_dir(person_id) / "person.json"

    def _archived_person_dir(self, person_id: str) -> Path:
        if not person_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in person_id):
            raise ProductError("人物编号无效。")
        path = (self.archive_dir / person_id).resolve()
        if path.parent != self.archive_dir:
            raise ProductError("人物编号无效。")
        return path

    def _require_person(self, person_id: str) -> dict[str, object]:
        raw = _read_json(self._person_path(person_id))
        if not isinstance(raw, dict):
            raise ProductError("人物文件损坏。")
        return dict(raw)

    def _history(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "history.json", [])
        if not isinstance(raw, list):
            raise ProductError("历史数据文件损坏。")
        return [dict(item) for item in raw]

    def _predictions(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "predictions.json", [])
        if not isinstance(raw, list):
            raise ProductError("预测记录文件损坏。")
        return [dict(item) for item in raw]

    def _versions(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "versions.json", [])
        if not isinstance(raw, list):
            raise ProductError("模型版本文件损坏。")
        return [dict(item) for item in raw]

    def _authority(self, person: Mapping[str, object]) -> VerificationAuthority:
        try:
            secret = bytes.fromhex(str(person["local_verifier_secret"]))
        except (KeyError, ValueError) as error:
            raise ProductError("人物的本地完整性密钥损坏。") from error
        return VerificationAuthority({VERIFIER_ID: secret})

    def list_people(self) -> list[dict[str, object]]:
        with self._lock:
            result = []
            for path in sorted(self.people_dir.glob("*/person.json")):
                person = dict(_read_json(path))
                person_id = str(person["person_id"])
                history = self._history(person_id)
                versions = self._versions(person_id)
                conversation = self._conversation_call(
                    self.conversation.summary, person_id
                )
                messages = list(conversation["messages"])
                result.append(
                    {
                        "person_id": person["person_id"],
                        "name": person["name"],
                        "description": person.get("description", ""),
                        "avatar": person.get("avatar", ""),
                        "identity_note": person.get("identity_note", ""),
                        "focus_domain": person.get("focus_domain", ""),
                        "collection": person.get(
                            "collection", conversation["profile"].get("collection", {})
                        ),
                        "feature_names": person["feature_names"],
                        "is_example": bool(person.get("is_example")),
                        "is_demo": bool(person.get("is_demo")),
                        "sample_count": len(history),
                        "model_version_count": len(versions),
                        "model_status": versions[-1]["validation_status"] if versions else "not_trained",
                        "model_kind": "behavior_baseline_logistic",
                        "cognitive_status": self.cognitive.summary(str(person["person_id"]))["status"],
                        "conversation_status": conversation["status"],
                        "conversation_status_text": conversation["status_text"],
                        "conversation_version": conversation["active_version"],
                        "source_count": conversation["source_counts"]["confirmed"],
                        "message_count": len(messages),
                        "last_message": messages[-1]["text"] if messages else "",
                        "language": conversation["profile"]["language"],
                        "style_profile_id": conversation["profile"]["style_profile_id"],
                    }
                )
            return result

    def create_person(
        self,
        *,
        name: str,
        description: str,
        feature_names: Sequence[str],
        person_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            clean_name = str(name).strip()
            names = tuple(str(item).strip() for item in feature_names if str(item).strip())
            if not clean_name:
                raise ProductError("请填写人物名称。")
            if not names or len(set(names)) != len(names):
                raise ProductError("至少需要一个不重复的数值影响项名称。")
            identifier = _slug(person_id or f"person-{uuid.uuid4().hex[:12]}")
            directory = self._person_dir(identifier)
            if directory.exists():
                raise ProductError("这个人物编号已经存在。")
            directory.mkdir(parents=True)
            person = {
                "format": PRODUCT_FORMAT,
                "person_id": identifier,
                "name": clean_name,
                "description": str(description).strip(),
                "feature_names": list(names),
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "is_example": False,
                "local_verifier_secret": secrets.token_hex(32),
            }
            _write_json(directory / "person.json", person)
            _write_json(directory / "history.json", [])
            _write_json(directory / "predictions.json", [])
            _write_json(directory / "versions.json", [])
            return self.get_person(identifier)

    def update_person(self, person_id: str, changes: Mapping[str, object]) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            if "name" in changes:
                name = str(changes["name"]).strip()
                if not name:
                    raise ProductError("人物名称不能为空。")
                person["name"] = name
            if "description" in changes:
                person["description"] = str(changes["description"]).strip()
            for key in ("identity_note", "focus_domain", "avatar", "notes"):
                if key in changes:
                    person[key] = str(changes[key]).strip()
            if "source_mode" in changes:
                source_mode = str(changes["source_mode"]).strip()
                if source_mode not in {"user_provided", "system_search"}:
                    raise ProductError("请选择系统自动搜索或用户自行提供资料。")
                if source_mode == "system_search" and self.public_search is None:
                    raise ProductError("系统自动搜索公开资料暂未配置，请选择自行提供原始资料。")
                person["collection"] = {
                    "mode": source_mode,
                    "status": "search_ready" if source_mode == "system_search" else "awaiting_user_materials",
                    "message": (
                        "搜索服务已配置；结果只会进入待审核候选资料。"
                        if source_mode == "system_search"
                        else "等待用户提供原始资料。"
                    ),
                }
            if "feature_names" in changes:
                if self._history(person_id) or self._versions(person_id):
                    raise ProductError("已有数据或模型时不能修改数值影响项；请新建人物。")
                names = tuple(str(item).strip() for item in changes["feature_names"] if str(item).strip())
                if not names or len(set(names)) != len(names):
                    raise ProductError("数值影响项不能为空或重复。")
                person["feature_names"] = list(names)
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            if any(
                key in changes
                for key in (
                    "aliases", "language", "time_start", "time_end",
                    "source_mode", "identity_note", "focus_domain",
                    "generation_params",
                )
            ):
                profile = self._conversation_call(
                    self.conversation.profile, person_id
                )
                self._conversation_call(
                    self.conversation.configure,
                    person_id,
                    aliases=changes.get("aliases", profile.get("aliases", [])),
                    language=str(changes.get("language", profile.get("language", "zh"))),
                    time_start=str(
                        changes.get(
                            "time_start", profile.get("time_scope", {}).get("start", "")
                        )
                    ),
                    time_end=str(
                        changes.get(
                            "time_end", profile.get("time_scope", {}).get("end", "")
                        )
                    ),
                    style_profile_id=str(profile.get("style_profile_id", "neutral_v1")),
                    source_mode=str(
                        changes.get(
                            "source_mode",
                            profile.get("collection", {}).get("mode", "user_provided"),
                        )
                    ),
                    identity_note=str(
                        changes.get("identity_note", profile.get("identity_note", ""))
                    ),
                    focus_domain=str(
                        changes.get("focus_domain", profile.get("focus_domain", ""))
                    ),
                    generation_params=dict(
                        changes.get(
                            "generation_params",
                            profile.get("generation_params", {}),
                        )
                    ),
                )
            return self.get_person(person_id)

    def delete_person(self, person_id: str) -> None:
        with self._lock:
            directory = self._person_dir(person_id)
            if not (directory / "person.json").exists():
                raise ProductError("人物不存在。")
            target = self._archived_person_dir(person_id)
            if target.exists():
                raise ProductError("归档中已经存在同编号人物。")
            person = self._require_person(person_id)
            person["archived_at"] = _utc_now()
            _write_json(directory / "person.json", person)
            directory.rename(target)

    def set_avatar(self, person_id: str, data_url: str) -> dict[str, object]:
        """保存人物头像（本地文件），data_url 形如 data:image/png;base64,...；传空则移除。"""
        with self._lock:
            person = self._require_person(person_id)
            if not str(data_url).strip():
                for old_ext in ("png", "jpg", "jpeg", "webp", "gif"):
                    old = self._person_dir(person_id) / f"avatar.{old_ext}"
                    if old.exists():
                        old.unlink(missing_ok=True)
                person["avatar"] = ""
                person["updated_at"] = _utc_now()
                _write_json(self._person_path(person_id), person)
                return self.get_person(person_id)
            header = "image/png"
            encoded = str(data_url).strip()
            if "," in encoded:
                header, encoded = encoded.split(",", 1)
                header = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            try:
                raw = base64.b64decode(encoded)
            except Exception as error:
                raise ProductError("头像数据无效。") from error
            if len(raw) > 2 * 1024 * 1024:
                raise ProductError("头像图片不能超过 2 MB。")
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(header, "png")
            for old_ext in ("png", "jpg", "jpeg", "webp", "gif"):
                old = self._person_dir(person_id) / f"avatar.{old_ext}"
                if old.exists() and old_ext != ext:
                    old.unlink(missing_ok=True)
            path = self._person_dir(person_id) / f"avatar.{ext}"
            path.write_bytes(raw)
            person["avatar"] = f"/api/people/{person_id}/avatar"
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            return self.get_person(person_id)

    def get_avatar(self, person_id: str) -> tuple[bytes, str]:
        """返回头像字节与 MIME；无头像时抛 ProductError。"""
        person = self._require_person(person_id)
        for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp"), ("gif", "image/gif")):
            path = self._person_dir(person_id) / f"avatar.{ext}"
            if path.exists():
                return path.read_bytes(), mime
        raise ProductError("该人物还没有头像。")

    def list_archived_people(self) -> list[dict[str, object]]:
        with self._lock:
            result = []
            for path in sorted(self.archive_dir.glob("*/person.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict):
                    conversation_sources = _read_json(path.parent / "conversation_sources.json", [])
                    conversation_messages = _read_json(path.parent / "conversation_messages.json", [])
                    conversation_versions = _read_json(path.parent / "conversation_versions.json", [])
                    result.append(
                        {
                            "person_id": raw.get("person_id"),
                            "name": raw.get("name"),
                            "archived_at": raw.get("archived_at"),
                            "avatar": raw.get("avatar", ""),
                            "source_count": len(conversation_sources) if isinstance(conversation_sources, list) else 0,
                            "message_count": len(conversation_messages) if isinstance(conversation_messages, list) else 0,
                            "version_count": len(conversation_versions) if isinstance(conversation_versions, list) else 0,
                        }
                    )
            return result

    def restore_person(self, person_id: str) -> dict[str, object]:
        with self._lock:
            source = self._archived_person_dir(person_id)
            target = self._person_dir(person_id)
            if not (source / "person.json").exists():
                raise ProductError("归档人物不存在。")
            if target.exists():
                raise ProductError("人物库中已经存在同编号人物。")
            person = dict(_read_json(source / "person.json", {}))
            person.pop("archived_at", None)
            _write_json(source / "person.json", person)
            source.rename(target)
            return self.get_person(person_id)

    def permanently_delete_archived_person(
        self, person_id: str, *, expected_name: str
    ) -> None:
        with self._lock:
            if (self._person_dir(person_id) / "person.json").exists():
                raise ProductError("只能永久删除已经在归档中的人物。")
            target = self._archived_person_dir(person_id)
            if not (target / "person.json").exists():
                raise ProductError("归档人物不存在。")
            archived = dict(_read_json(target / "person.json", {}))
            if str(expected_name).strip() != str(archived.get("name", "")):
                raise ProductError("永久删除前必须输入完全一致的人物名称。")
            shutil.rmtree(target)

    # ---------- ordinary and advanced data import ----------

    def _normalize_history_record(
        self,
        person: Mapping[str, object],
        raw: Mapping[str, object],
        *,
        provenance: str = "human_record",
        role: str = "pending",
    ) -> dict[str, object]:
        option_a = str(raw.get("option_a", raw.get("option_0", ""))).strip()
        option_b = str(raw.get("option_b", raw.get("option_1", ""))).strip()
        if not option_a or not option_b or option_a == option_b:
            raise ProductError("每条记录需要两个不同且非空的选项。")
        observed_at = str(raw.get("observed_at", raw.get("date", ""))).strip()
        _parse_time(observed_at)
        scenario_id = str(raw.get("scenario_id", "")).strip() or uuid.uuid4().hex
        features_raw = raw.get("features")
        if isinstance(features_raw, str):
            try:
                features_raw = json.loads(features_raw)
            except json.JSONDecodeError as error:
                raise ProductError("features 列必须是 JSON 对象。") from error
        if not isinstance(features_raw, Mapping):
            features_raw = {name: raw.get(name) for name in person["feature_names"]}
        features: dict[str, float] = {}
        for name in person["feature_names"]:
            try:
                value = float(features_raw[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ProductError(f"数值影响项“{name}”缺失或不是数字。") from error
            if not math.isfinite(value):
                raise ProductError(f"数值影响项“{name}”必须是有限数字。")
            features[str(name)] = value
        context_raw = raw.get("context", {})
        if isinstance(context_raw, str):
            try:
                context_raw = json.loads(context_raw) if context_raw.strip() else {}
            except json.JSONDecodeError as error:
                raise ProductError("context 列必须是 JSON 对象。") from error
        if not isinstance(context_raw, Mapping):
            raise ProductError("情境补充信息必须是键值对象。")
        actual_raw = raw.get("actual_choice", raw.get("choice"))
        if actual_raw is None:
            raise ProductError("每条历史记录都需要真实选择。")
        return {
            "record_id": str(raw.get("record_id", "")).strip() or uuid.uuid4().hex,
            "scenario_id": scenario_id,
            "observed_at": observed_at,
            "question": str(raw.get("question", raw.get("situation", ""))).strip(),
            "option_a": option_a,
            "option_b": option_b,
            "actual_choice": _as_choice(actual_raw, option_a, option_b),
            "features": features,
            "domain": str(raw.get("domain", "structured_choice")).strip() or "structured_choice",
            "context": {str(key): str(value) for key, value in context_raw.items()},
            "confidence": float(raw["confidence"]) if raw.get("confidence") not in {None, ""} else None,
            "reaction_time_ms": float(raw["reaction_time_ms"]) if raw.get("reaction_time_ms") not in {None, ""} else None,
            "provenance": provenance,
            "role": role,
        }

    def import_history(
        self,
        person_id: str,
        payload: object,
        *,
        input_format: str,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            if input_format == "csv":
                if not isinstance(payload, str):
                    raise ProductError("CSV 内容无效。")
                rows = list(csv.DictReader(io.StringIO(payload.lstrip("\ufeff"))))
            elif input_format == "json":
                parsed = json.loads(payload) if isinstance(payload, str) else payload
                if isinstance(parsed, Mapping) and "records" in parsed:
                    parsed = parsed["records"]
                if not isinstance(parsed, list):
                    raise ProductError("JSON 必须是记录数组，或包含 records 数组。")
                rows = parsed
            elif input_format == "form":
                rows = [payload]
            else:
                raise ProductError("不支持的数据格式。")
            if not rows:
                raise ProductError("没有可导入的记录。")
            normalized = [self._normalize_history_record(person, dict(row)) for row in rows]
            history = self._history(person_id)
            scenario_ids = {str(item["scenario_id"]) for item in history}
            for record in normalized:
                if record["scenario_id"] in scenario_ids:
                    raise ProductError(f"情境编号重复：{record['scenario_id']}")
                scenario_ids.add(str(record["scenario_id"]))
            history.extend(normalized)
            _write_json(self._person_dir(person_id) / "history.json", history)
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            return {"imported_count": len(normalized), "sample_count": len(history)}

    def import_decision_evidence(
        self,
        person_id: str,
        bundle_data: Mapping[str, object],
        verification_keys: Mapping[str, str],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            authority = VerificationAuthority(
                {str(key): str(value).encode("utf-8") for key, value in verification_keys.items()}
            )
            bundle = decision_evidence_bundle_from_dict(bundle_data, authority)
            summary = {
                "artifact_digest": bundle.artifact_digest,
                "record_count": len(bundle.records),
                "source_count": len(bundle.sources),
                "training_authorized": bundle.training_authorized,
                "message": "高级证据包已验证并保存，但不会自动进入普通训练。",
            }
            directory = self._person_dir(person_id) / "advanced_evidence"
            _write_json(directory / f"{bundle.artifact_digest}.json", bundle.to_dict())
            _write_json(directory / f"{bundle.artifact_digest}.summary.json", summary)
            return summary

    # ---------- conversion, partitioning and ledgers ----------

    def _observation(
        self,
        person_id: str,
        record: Mapping[str, object],
    ) -> Observation:
        features = dict(record["features"])
        return Observation(
            person_id=person_id,
            scenario=Scenario(
                scenario_id=str(record["scenario_id"]),
                features=tuple(float(features[name]) for name in features),
                feature_names=tuple(str(name) for name in features),
                options=(str(record["option_a"]), str(record["option_b"])),
                domain=str(record.get("domain", "structured_choice")),
                context={
                    "question": str(record.get("question", "")),
                    **{str(key): str(value) for key, value in dict(record.get("context", {})).items()},
                },
            ),
            actual_choice=int(record["actual_choice"]),
            confidence=(float(record["confidence"]) if record.get("confidence") is not None else None),
            reaction_time_ms=(float(record["reaction_time_ms"]) if record.get("reaction_time_ms") is not None else None),
            provenance=str(record.get("provenance", "human_record")),
        )

    def _partition_history(
        self, history: Sequence[Mapping[str, object]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        explicit = all(str(item.get("role", "pending")) in {"training", "update", "applicability", "validation"} for item in history)
        if explicit:
            training = [dict(item) for item in history if item["role"] in {"training", "update"}]
            applicability = [dict(item) for item in history if item["role"] == "applicability"]
            validation = [dict(item) for item in history if item["role"] == "validation"]
            return training, applicability, validation
        ordered = sorted((dict(item) for item in history), key=lambda item: (_parse_time(str(item["observed_at"])), str(item["scenario_id"])))
        count = len(ordered)
        if count >= 200:
            return ordered[:-150], ordered[-150:-100], ordered[-100:]
        if count >= 100:
            return ordered[:-50], ordered[-50:], []
        return ordered, [], []

    def _sign_records(
        self,
        observations: Sequence[tuple[Observation, Mapping[str, object]]],
        authority: VerificationAuthority,
        prefix: str,
    ) -> EventLedger:
        records = []
        for index, (observation, raw) in enumerate(observations):
            evidence = observation_payload(observation)
            observed_at = str(raw["observed_at"])
            verified_at = max(_parse_time(observed_at), datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            records.append(
                authority.sign(
                    event_id=f"{prefix}:{index:06d}:{observation.person_id}:{observation.scenario.scenario_id}",
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=_canonical_hash(evidence),
                    verifier_id=VERIFIER_ID,
                    verified_at=verified_at,
                )
            )
        return EventLedger.verify(records, authority)

    def _example_population(
        self,
        person_id: str,
    ) -> tuple[list[tuple[Observation, dict[str, object]]], list[dict[str, object]]]:
        people, source, _target = generate_population_dataset(
            seed=401,
            person_count=24,
            source_trials=140,
            target_trials=1,
            heterogeneity_scale=1.5,
        )
        history = self._history(person_id)
        training, _app, _validation = self._partition_history(history)
        target_items = [(self._observation(person_id, item), item) for item in training]
        reference_time = str(training[0]["observed_at"])
        for person in people[1:]:
            for item in source[person.person_id]:
                transformed = Observation(
                    person_id=f"example-reference-{person.person_id}",
                    scenario=Scenario(
                        scenario_id=f"reference-{item.scenario.scenario_id}",
                        features=item.scenario.features,
                        feature_names=FEATURE_NAMES,
                        options=("执行方案", "暂不执行"),
                        domain="日常方案选择",
                        context={"question": "合成群体参考题"},
                    ),
                    actual_choice=item.actual_choice,
                    confidence=item.confidence,
                    reaction_time_ms=item.reaction_time_ms,
                    provenance="synthetic_ground_truth",
                )
                target_items.append((transformed, {"observed_at": reference_time}))
        return target_items, training

    def _real_population(
        self,
        target_person: Mapping[str, object],
        target_training: list[dict[str, object]],
    ) -> tuple[list[tuple[Observation, dict[str, object]]], bool]:
        target_id = str(target_person["person_id"])
        items = [(self._observation(target_id, record), record) for record in target_training]
        people_used = 1
        for other in self.list_people():
            other_id = str(other["person_id"])
            if other_id == target_id or tuple(other["feature_names"]) != tuple(target_person["feature_names"]):
                continue
            other_training, _app, _validation = self._partition_history(self._history(other_id))
            if len(other_training) < MINIMUM_PROFILE_SAMPLES:
                continue
            for record in other_training:
                items.append((self._observation(other_id, record), record))
            people_used += 1
        return items, people_used >= 2

    # ---------- model training, prediction and update ----------

    def train(self, person_id: str) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            history = self._history(person_id)
            training, applicability, validation = self._partition_history(history)
            if len(training) < MINIMUM_PROFILE_SAMPLES:
                raise ProductError(
                    f"当前只有 {len(history)} 条可用历史记录；现有模型至少需要 {MINIMUM_PROFILE_SAMPLES} 条训练记录。"
                )
            assigned_roles = {
                str(item["record_id"]): role
                for role, records in (
                    ("training", training),
                    ("applicability", applicability),
                    ("validation", validation),
                )
                for item in records
            }
            for item in history:
                record_id = str(item["record_id"])
                if record_id in assigned_roles:
                    item["role"] = assigned_roles[record_id]
            _write_json(self._person_dir(person_id) / "history.json", history)
            authority = self._authority(person)
            reference_mode = "real_multi_person"
            if person.get("is_example"):
                population_items, training = self._example_population(person_id)
                real_population = True
                reference_mode = "synthetic_demo_population"
            else:
                population_items, real_population = self._real_population(person, training)
                if not real_population:
                    raise ProductError(
                        "行为基线证据不足：至少需要另一个具有相同数值字段、且样本充足的真实参照人物。系统不会再自动注入合成人群。"
                    )
            training_ledger = self._sign_records(population_items, authority, "training")
            applicability_ledger = (
                self._sign_records(
                    [(self._observation(person_id, item), item) for item in applicability],
                    authority,
                    "applicability",
                )
                if applicability
                else None
            )
            validation_ledger = (
                self._sign_records(
                    [(self._observation(person_id, item), item) for item in validation],
                    authority,
                    "validation",
                )
                if validation and real_population
                else None
            )
            bundle = fit_person_model(
                training_ledger,
                authority,
                applicability_ledger=applicability_ledger,
                validation_ledger=validation_ledger,
                person_id=person_id,
                feature_names=tuple(str(name) for name in person["feature_names"]),
            )
            version = self._save_version(
                person_id,
                bundle,
                training_ledger,
                applicability_ledger,
                validation_ledger,
                source="train",
                reference_mode=reference_mode,
            )
            return self._version_public(version, bundle)

    def _save_version(
        self,
        person_id: str,
        bundle,
        training_ledger: EventLedger,
        applicability_ledger: EventLedger | None,
        validation_ledger: EventLedger | None,
        *,
        source: str,
        reference_mode: str,
    ) -> dict[str, object]:
        versions = self._versions(person_id)
        number = len(versions) + 1
        directory = self._person_dir(person_id)
        model_rel = f"models/model-{number:04d}.json"
        training_rel = f"ledgers/training-{number:04d}.jsonl"
        app_rel = f"ledgers/applicability-{number:04d}.jsonl" if applicability_ledger else None
        val_rel = f"ledgers/validation-{number:04d}.jsonl" if validation_ledger else None
        save_bundle(directory / model_rel, bundle)
        save_event_ledger_jsonl(directory / training_rel, training_ledger)
        if applicability_ledger:
            save_event_ledger_jsonl(directory / str(app_rel), applicability_ledger)
        if validation_ledger:
            save_event_ledger_jsonl(directory / str(val_rel), validation_ledger)
        validation = bundle.manifest.validation
        version = {
            "version": number,
            "created_at": _utc_now(),
            "source": source,
            "model_id": bundle.manifest.model_id,
            "parent_model_id": bundle.manifest.parent_model_id,
            "model_path": model_rel,
            "training_ledger_path": training_rel,
            "applicability_ledger_path": app_rel,
            "validation_ledger_path": val_rel,
            "training_sample_count": bundle.representation.observation_count,
            "validation_sample_count": validation.sample_count,
            "validation_status": validation.status,
            "personal_nll": validation.personal_nll,
            "calibration_error": validation.calibration_error,
            "validation_reasons": list(validation.reasons),
            "reference_mode": reference_mode,
        }
        versions.append(version)
        _write_json(directory / "versions.json", versions)
        return version

    def _latest_bundle(self, person_id: str):
        versions = self._versions(person_id)
        if not versions:
            raise ProductError("这个人物还没有训练模型。")
        latest = versions[-1]
        bundle = load_bundle(self._person_dir(person_id) / str(latest["model_path"]))
        return latest, bundle

    def _scenario_from_input(
        self, person: Mapping[str, object], raw: Mapping[str, object]
    ) -> Scenario:
        features_raw = raw.get("features", {})
        if not isinstance(features_raw, Mapping):
            raise ProductError("数值影响项格式无效。")
        names = tuple(str(name) for name in person["feature_names"])
        try:
            values = tuple(float(features_raw[name]) for name in names)
        except (KeyError, TypeError, ValueError) as error:
            raise ProductError("请填写全部数值影响项，并确保都是数字。") from error
        option_a = str(raw.get("option_a", "")).strip()
        option_b = str(raw.get("option_b", "")).strip()
        if not option_a or not option_b:
            raise ProductError("请填写两个候选选项。")
        context = raw.get("context", {})
        if not isinstance(context, Mapping):
            context = {}
        return Scenario(
            scenario_id=str(raw.get("scenario_id", "")).strip() or f"prediction-{uuid.uuid4().hex}",
            features=values,
            feature_names=names,
            options=(option_a, option_b),
            domain=str(raw.get("domain", "structured_choice")).strip() or "structured_choice",
            context={"question": str(raw.get("question", "")).strip(), **{str(key): str(value) for key, value in context.items()}},
        )

    def predict(
        self,
        person_id: str,
        raw: Mapping[str, object],
        *,
        diagnostic_override: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            version, bundle = self._latest_bundle(person_id)
            scenario = self._scenario_from_input(person, raw)
            prediction_at = str(raw.get("prediction_at", "")).strip() or _utc_now()
            record: dict[str, object] = {
                "prediction_id": uuid.uuid4().hex,
                "created_at": _utc_now(),
                "prediction_at": prediction_at,
                "person_id": person_id,
                "model_version": version["version"],
                "model_id": version["model_id"],
                "scenario": {
                    "scenario_id": scenario.scenario_id,
                    "question": scenario.context.get("question", ""),
                    "option_a": scenario.options[0],
                    "option_b": scenario.options[1],
                    "features": dict(zip(scenario.feature_names, scenario.features, strict=True)),
                    "domain": scenario.domain,
                    "context": dict(scenario.context),
                },
                "actual_choice": None,
                "updated_model_version": None,
            }
            try:
                prediction = predict_with_bundle(
                    bundle,
                    scenario,
                    prediction_at=prediction_at,
                    validation_override=diagnostic_override,
                    applicability_override=False,
                )
                effective_weights = np.asarray(bundle.population_model.weights) + np.asarray(bundle.adapter.delta_weights)
                contributions = [
                    {
                        "name": name,
                        "feature_value": float(value),
                        "effective_weight": float(weight),
                        "logit_contribution": float(value * weight),
                    }
                    for name, value, weight in zip(
                        bundle.population_model.feature_names,
                        scenario.ordered_features(bundle.population_model.feature_names),
                        effective_weights,
                        strict=True,
                    )
                ]
                contributions.sort(key=lambda item: abs(float(item["logit_contribution"])), reverse=True)
                p_b = prediction.probability_option_1
                record.update(
                    {
                        "status": "predicted",
                        "probability_a": 1.0 - p_b,
                        "probability_b": p_b,
                        "predicted_choice": prediction.predicted_choice,
                        "probability_lower_95_b": prediction.probability_lower_95,
                        "probability_upper_95_b": prediction.probability_upper_95,
                        "applicability_status": prediction.applicability_status,
                        "applicability_warnings": list(prediction.applicability_warnings),
                        "applicability_warning_text": [_reason_text(item) for item in prediction.applicability_warnings],
                        "ood_score": prediction.ood_score,
                        "ood_threshold": prediction.ood_threshold,
                        "local_ood_score": prediction.local_ood_score,
                        "local_ood_threshold": prediction.local_ood_threshold,
                        "validation_status": prediction.validation_status,
                        "gate_overrides": list(prediction.gate_overrides),
                        "diagnostic_override": bool(prediction.gate_overrides),
                        "influences": contributions,
                        "influence_notice": "这些只是模型中的数值贡献，不代表真实信念、价值观或心理机制。",
                    }
                )
            except PredictionRefusedError as error:
                record.update(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                        "reason_text": [_reason_text(item) for item in error.reasons],
                        "ood_score": error.ood_score,
                        "ood_threshold": error.ood_threshold,
                        "local_ood_score": error.local_ood_score,
                        "local_ood_threshold": error.local_ood_threshold,
                    }
                )
            predictions = self._predictions(person_id)
            predictions.append(record)
            _write_json(self._person_dir(person_id) / "predictions.json", predictions)
            return record

    def record_outcome(
        self,
        person_id: str,
        prediction_id: str,
        actual_choice: object,
        observed_at: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            predictions = self._predictions(person_id)
            try:
                record = next(item for item in predictions if item["prediction_id"] == prediction_id)
            except StopIteration as error:
                raise ProductError("预测记录不存在。") from error
            if record.get("actual_choice") is not None:
                raise ProductError("这条预测已经录入真实结果。")
            scenario_raw = dict(record["scenario"])
            choice = _as_choice(actual_choice, str(scenario_raw["option_a"]), str(scenario_raw["option_b"]))
            record["actual_choice"] = choice
            record["outcome_recorded_at"] = _utc_now()
            if record.get("status") != "predicted":
                record["comparison"] = {"status": "prediction_was_refused"}
                _write_json(self._person_dir(person_id) / "predictions.json", predictions)
                return record
            versions = self._versions(person_id)
            latest = versions[-1]
            if int(record["model_version"]) != int(latest["version"]):
                raise ProductError("这条预测来自旧模型。为避免分叉，请在最新模型上重新预测。")
            directory = self._person_dir(person_id)
            authority = self._authority(person)
            bundle = load_bundle(directory / str(latest["model_path"]))
            training_ledger = load_event_ledger_jsonl(directory / str(latest["training_ledger_path"]), authority)
            app_path = latest.get("applicability_ledger_path")
            val_path = latest.get("validation_ledger_path")
            applicability_ledger = load_event_ledger_jsonl(directory / str(app_path), authority) if app_path else None
            validation_ledger = load_event_ledger_jsonl(directory / str(val_path), authority) if val_path else None
            outcome_time = observed_at or _utc_now()
            _parse_time(outcome_time)
            history_record = self._normalize_history_record(
                person,
                {
                    **scenario_raw,
                    "observed_at": outcome_time,
                    "actual_choice": choice,
                    "record_id": f"prediction-{prediction_id}",
                },
                role="update",
            )
            observation = self._observation(person_id, history_record)
            verified_at = max(_parse_time(outcome_time), datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            outcome = authority.sign(
                event_id=f"outcome:{prediction_id}",
                observation=observation,
                observed_at=outcome_time,
                evidence_hash=_canonical_hash(observation_payload(observation)),
                verifier_id=VERIFIER_ID,
                verified_at=verified_at,
            )
            updated = update_person_model(
                bundle,
                training_ledger,
                outcome,
                authority,
                applicability_ledger=applicability_ledger,
                validation_ledger=validation_ledger,
            )
            new_version = self._save_version(
                person_id,
                updated.bundle,
                updated.ledger,
                applicability_ledger,
                validation_ledger,
                source="prediction_outcome_update",
                reference_mode=str(latest["reference_mode"]),
            )
            probability = float(record["probability_b"])
            selected_probability = probability if choice == 1 else 1.0 - probability
            record["comparison"] = {
                "correct": int(record["predicted_choice"]) == choice,
                "selected_probability": selected_probability,
                "nll": -math.log(max(selected_probability, 1e-9)),
            }
            record["updated_model_version"] = new_version["version"]
            history = self._history(person_id)
            history.append(history_record)
            _write_json(directory / "history.json", history)
            _write_json(directory / "predictions.json", predictions)
            return record

    # ---------- views, metrics and portability ----------

    def _version_public(self, version: Mapping[str, object], bundle=None) -> dict[str, object]:
        reasons = list(version.get("validation_reasons", []))
        result = {key: value for key, value in version.items() if not str(key).endswith("_path")}
        result["validation_reason_text"] = [_reason_text(str(item)) for item in reasons]
        result["model_kind"] = "behavior_baseline_logistic"
        result["display_name"] = "行为基线模型（Logistic）"
        if version.get("reference_mode") == "synthetic_compatibility_only":
            result["model_explanation"] = "只有一个真实人物可用于当前特征结构，因此群体参照是合成兼容数据；模型只能作诊断使用，不能声称人物化已验证。"
        elif version.get("reference_mode") == "synthetic_demo_population":
            result["model_explanation"] = "这是内置合成演示的验证结果，只证明软件流程可运行，不代表真实人物效果。"
        else:
            result["model_explanation"] = "模型使用了本地多个同特征结构人物作为群体参照。"
        return result

    def _prediction_metrics(self, person_id: str) -> dict[str, object] | None:
        completed = [item for item in self._predictions(person_id) if item.get("status") == "predicted" and item.get("actual_choice") is not None]
        if not completed:
            return None
        observations = []
        probabilities = []
        for item in completed:
            raw = dict(item["scenario"])
            person = self._require_person(person_id)
            scenario = self._scenario_from_input(person, raw)
            observations.append(Observation(person_id=person_id, scenario=scenario, actual_choice=int(item["actual_choice"])))
            probabilities.append(float(item["probability_b"]))
        return report_to_dict(evaluate_probability_array(observations, probabilities))

    def get_person(self, person_id: str) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            history = self._history(person_id)
            training, applicability, validation = self._partition_history(history)
            versions = self._versions(person_id)
            predictions = self._predictions(person_id)
            public_person = {key: value for key, value in person.items() if key != "local_verifier_secret"}
            public_person.update(
                {
                    "sample_count": len(history),
                    "partition_counts": {
                        "training": len(training),
                        "applicability": len(applicability),
                        "validation": len(validation),
                    },
                    "data_sufficiency": self._data_sufficiency(len(training), len(applicability), len(validation)),
                    "predictions": predictions,
                    "prediction_metrics": self._prediction_metrics(person_id),
                    "versions": [self._version_public(item) for item in versions],
                    "latest_model": self._version_public(versions[-1]) if versions else None,
                    "history_preview": history[-20:],
                    "advanced_evidence_count": len(list((self._person_dir(person_id) / "advanced_evidence").glob("*.summary.json"))) if (self._person_dir(person_id) / "advanced_evidence").exists() else 0,
                    "behavior_baseline": {
                        "display_name": "行为基线模型（Logistic）",
                        "status": self._version_public(versions[-1])["validation_status"] if versions else "not_trained",
                        "notice": "它只拟合结构化历史选择，不等同于人物认知模型。",
                    },
                    "cognitive": self.cognitive.summary(person_id),
                    "conversation": self._conversation_call(
                        self.conversation.summary, person_id
                    ),
                }
            )
            return public_person

    # ---------- evidence-constrained cognitive workbench ----------

    def _cognitive_call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except CognitiveWorkbenchError as error:
            raise ProductError(str(error)) from error

    def add_cognitive_evidence(
        self, person_id: str, raw: Mapping[str, object]
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.add_evidence, person_id, raw)

    def review_cognitive_evidence(
        self, person_id: str, evidence_id: str, decision: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.review_evidence,
                person_id,
                evidence_id,
                decision,
            )

    def generate_cognitive_card(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.generate_card, person_id)

    def draft_cognitive_scenario(
        self, person_id: str, text: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.draft_scenario, person_id, text)

    def confirm_cognitive_scenario(
        self,
        person_id: str,
        scenario_id: str,
        changes: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.confirm_scenario,
                person_id,
                scenario_id,
                changes,
            )

    def _behavior_baseline_for_cognitive(
        self, person_id: str, scenario: Mapping[str, object]
    ) -> dict[str, object]:
        versions = self._versions(person_id)
        if not versions:
            return {
                "status": "insufficient_evidence",
                "reason": "没有可比较的行为基线版本。",
                "model_kind": "behavior_baseline_logistic",
            }
        latest = versions[-1]
        if latest.get("reference_mode") != "real_multi_person":
            return {
                "status": "insufficient_evidence",
                "reason": "现有行为基线没有真实多人物参照，不能用于普通人物推演。",
                "model_kind": "behavior_baseline_logistic",
            }
        person = self._require_person(person_id)
        factor_values = dict(scenario.get("factor_values", {}))
        if set(map(str, person["feature_names"])) != set(map(str, factor_values)):
            return {
                "status": "incompatible_structure",
                "reason": "行为基线数值字段与本次认知情境因素不同。",
                "model_kind": "behavior_baseline_logistic",
            }
        raw = {
            "scenario_id": f"behavior-{scenario['scenario_id']}",
            "question": scenario["original_text"],
            "option_a": scenario["option_a"],
            "option_b": scenario["option_b"],
            "features": factor_values,
            "domain": scenario["domain"],
            "context": {"decision_type": scenario["decision_type"]},
            "prediction_at": scenario["prediction_at"],
        }
        baseline = self.predict(person_id, raw, diagnostic_override=False)
        return {
            "status": baseline["status"],
            "predicted_choice": baseline.get("predicted_choice"),
            "probability_a": baseline.get("probability_a"),
            "probability_b": baseline.get("probability_b"),
            "reasons": baseline.get("reasons", []),
            "model_kind": "behavior_baseline_logistic",
        }

    def predict_cognitive_scenario(
        self, person_id: str, scenario_id: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            scenario = self._cognitive_call(
                self.cognitive.get_scenario, person_id, scenario_id
            )
            baseline = self._behavior_baseline_for_cognitive(person_id, scenario)
            return self._cognitive_call(
                self.cognitive.predict,
                person_id,
                scenario_id,
                behavior_baseline=baseline,
            )

    def record_cognitive_outcome(
        self,
        person_id: str,
        prediction_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.record_outcome,
                person_id,
                prediction_id,
                payload,
            )

    def _data_sufficiency(self, training: int, applicability: int, validation: int) -> dict[str, object]:
        if training < MINIMUM_PROFILE_SAMPLES:
            message = f"还差 {MINIMUM_PROFILE_SAMPLES - training} 条训练记录，才能建立当前模型的适用范围。"
            level = "insufficient"
        elif applicability < MINIMUM_PROFILE_SAMPLES:
            message = "可以训练诊断模型，但还没有独立的适用域校准数据。"
            level = "diagnostic"
        elif validation < MINIMUM_VALIDATION_SAMPLES:
            message = f"可以训练，但还差 {MINIMUM_VALIDATION_SAMPLES - validation} 条独立验证记录，不能声称模型已验证。"
            level = "diagnostic"
        else:
            message = "样本数量达到训练、适用域校准和独立验证的最低数量门槛；是否通过仍取决于 NLL、校准和时间稳定性。"
            level = "validation_ready"
        return {"level": level, "message": message}

    def export_person(self, person_id: str) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            directory = self._person_dir(person_id)
            files: dict[str, str] = {}
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    files[path.relative_to(directory).as_posix()] = path.read_text(encoding="utf-8")
            return {
                "format": PRODUCT_FORMAT,
                "exported_at": _utc_now(),
                "person_id": person_id,
                "files": files,
                "notice": "导出包包含此人物的本地完整性密钥，请像备份文件一样保管。",
            }

    def import_product_export(self, payload: Mapping[str, object], *, replace: bool = False) -> dict[str, object]:
        with self._lock:
            if payload.get("format") != PRODUCT_FORMAT or not isinstance(payload.get("files"), Mapping):
                raise ProductError("不是受支持的 PCFM 人物导出文件。")
            person_id = str(payload.get("person_id", ""))
            directory = self._person_dir(person_id)
            backup = self.people_dir / f"_restore-{uuid.uuid4().hex}"
            if directory.exists():
                if not replace:
                    raise ProductError("同编号人物已经存在；确认覆盖后才能加载。")
                directory.rename(backup)
            directory.mkdir(parents=True)
            try:
                for relative, text in dict(payload["files"]).items():
                    target = (directory / str(relative)).resolve()
                    if directory not in target.parents:
                        raise ProductError("导出文件包含不安全的路径。")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(text), encoding="utf-8")
                person = self._require_person(person_id)
                if person.get("format") != PRODUCT_FORMAT:
                    raise ProductError("人物文件版本不受支持。")
                authority = self._authority(person)
                for version in self._versions(person_id):
                    def local_path(value: object) -> Path:
                        resolved = (directory / str(value)).resolve()
                        if directory not in resolved.parents:
                            raise ProductError("备份中的模型路径不安全。")
                        return resolved

                    bundle = load_bundle(local_path(version["model_path"]))
                    ledger = load_event_ledger_jsonl(
                        local_path(version["training_ledger_path"]),
                        authority,
                    )
                    target_records = ledger.records_for_person(person_id)
                    target_ids = tuple(
                        record.event_id
                        for record in sorted(
                            target_records,
                            key=lambda record: record.event_id,
                        )
                    )
                    if (
                        target_ids != bundle.manifest.person_event_ids
                        or EventLedger.snapshot_hash(target_records)
                        != bundle.manifest.person_data_hash
                    ):
                        raise ProductError("备份中的模型与训练历史不一致。")
                    for ledger_key in (
                        "applicability_ledger_path",
                        "validation_ledger_path",
                    ):
                        if version.get(ledger_key):
                            load_event_ledger_jsonl(
                                local_path(version[ledger_key]),
                                authority,
                            )
                result = self.get_person(person_id)
                if backup.exists():
                    shutil.rmtree(backup)
                return result
            except Exception:
                if directory.exists():
                    shutil.rmtree(directory)
                if backup.exists():
                    backup.rename(directory)
                raise

    def csv_template(self, person_id: str) -> str:
        person = self._require_person(person_id)
        fields = ["observed_at", "scenario_id", "question", "option_a", "option_b", "choice", "domain", *person["feature_names"]]
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerow([_utc_now(), "example-001", "示例情境", "选项A", "选项B", "A", "structured_choice", *([0.0] * len(person["feature_names"]))])
        return stream.getvalue()

    # ---------- built-in demonstrator ----------

    def _seed_conversation_demos(self) -> list[str]:
        seeded: list[str] = []
        for spec in DEMO_PEOPLE:
            person_id = str(spec["person_id"])
            if self._person_dir(person_id).exists() or (self.archive_dir / person_id).exists():
                continue
            self.create_person(
                name=str(spec["name"]),
                description=str(spec["description"]),
                feature_names=("evidence_overlap", "intercept"),
                person_id=person_id,
            )
            person = self._require_person(person_id)
            person.update(
                {
                    "identity_note": str(spec["identity_note"]),
                    "focus_domain": str(spec["focus_domain"]),
                    "avatar": str(spec["avatar"]),
                    "notes": "内置演示人物；不得视为现实人物预测准确率证明。",
                    "is_example": True,
                    "is_demo": True,
                    "demo_seed_version": DEMO_SEED_VERSION,
                    "demo_status": "exploratory_accuracy_not_validated",
                    "recommended_questions": list(spec["recommended_questions"]),
                    "collection": {
                        "mode": "user_provided",
                        "status": "verified_demo_materials_loaded",
                        "message": "已载入可追溯的一手演示资料；预测仍属探索性，准确性尚未验证。",
                    },
                }
            )
            _write_json(self._person_path(person_id), person)
            self._conversation_call(
                self.conversation.configure,
                person_id,
                aliases=list(spec["aliases"]),
                language=str(spec["language"]),
                time_start=str(spec["time_start"]),
                time_end=str(spec["time_end"]),
                source_mode="user_provided",
                identity_note=str(spec["identity_note"]),
                focus_domain=str(spec["focus_domain"]),
            )
            for source_spec in spec["sources"]:
                source = self._conversation_call(
                    self.conversation.add_text_source,
                    person_id,
                    title=str(source_spec["title"]),
                    text=str(source_spec["text"]),
                    speaker=str(source_spec["speaker"]),
                    source_date=str(source_spec["source_date"]),
                    dataset_role=str(source_spec["dataset_role"]),
                    source_type="built_in_verified_demo",
                    source_url=str(source_spec["source_url"]),
                    content_authenticity=str(source_spec["content_authenticity"]),
                    source_locator=str(source_spec["source_locator"]),
                    source_context=str(source_spec["source_context"]),
                    original_language=str(source_spec["original_language"]),
                    entity_aliases=list(source_spec.get("entity_aliases", [])),
                )
                self._conversation_call(
                    self.conversation.review_source,
                    person_id,
                    str(source["source_id"]),
                    "confirmed",
                )
            seeded.append(person_id)
        return seeded

    def _seed_example(self) -> None:
        identifier = "example-person"
        directory = self._person_dir(identifier)
        directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).replace(microsecond=0)
        people, source, target = generate_population_dataset(
            seed=401,
            person_count=24,
            source_trials=140,
            target_trials=441,
            heterogeneity_scale=1.5,
        )
        target_person = people[0]
        history: list[dict[str, object]] = []
        groups = [
            (source[target_person.person_id], "training", now - timedelta(days=100)),
            (target[target_person.person_id][:220], "applicability", now - timedelta(days=70)),
            (target[target_person.person_id][220:330], "validation", now - timedelta(days=40)),
            (target[target_person.person_id][330:440], "validation", now - timedelta(days=20)),
        ]
        for observations, role, timestamp in groups:
            for index, observation in enumerate(observations):
                history.append(
                    {
                        "record_id": uuid.uuid4().hex,
                        "scenario_id": f"example-{role}-{len(history):04d}",
                        "observed_at": timestamp.isoformat().replace("+00:00", "Z"),
                        "question": "在当前条件下是否执行这项方案？",
                        "option_a": "执行方案",
                        "option_b": "暂不执行",
                        "actual_choice": observation.actual_choice,
                        "features": dict(zip(FEATURE_NAMES, observation.scenario.features, strict=True)),
                        "domain": "日常方案选择",
                        "context": {},
                        "confidence": observation.confidence,
                        "reaction_time_ms": observation.reaction_time_ms,
                        "provenance": "synthetic_ground_truth",
                        "role": role,
                    }
                )
        suggested = target[target_person.person_id][440]
        person = {
            "format": PRODUCT_FORMAT,
            "person_id": identifier,
            "name": "示例人物：林澄（合成数据）",
            "description": "用于直接体验完整流程。所有记录均为程序生成，不代表任何真实人物。",
            "feature_names": list(FEATURE_NAMES),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "is_example": True,
            "local_verifier_secret": secrets.token_hex(32),
            "suggested_scenario": {
                "scenario_id": "example-new-choice",
                "question": "在当前条件下是否执行这项方案？",
                "option_a": "执行方案",
                "option_b": "暂不执行",
                "features": dict(zip(FEATURE_NAMES, suggested.scenario.features, strict=True)),
                "domain": "日常方案选择",
                "context": {},
                "prediction_at": now.isoformat().replace("+00:00", "Z"),
            },
            "suggested_actual_choice": suggested.actual_choice,
        }
        _write_json(directory / "person.json", person)
        _write_json(directory / "history.json", history)
        _write_json(directory / "predictions.json", [])
        _write_json(directory / "versions.json", [])

    def _seed_cognitive_example(self) -> None:
        case = load_builtin_hawley_case()
        raw_person = dict(case["person"])
        identifier = str(raw_person["person_id"])
        directory = self._person_dir(identifier)
        if not (directory / "person.json").exists():
            directory.mkdir(parents=True, exist_ok=True)
            person = {
                "format": PRODUCT_FORMAT,
                "person_id": identifier,
                "name": raw_person["name"],
                "description": raw_person["description"],
                "feature_names": [],
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "is_example": False,
                "is_cognitive_case": True,
                "local_verifier_secret": secrets.token_hex(32),
            }
            _write_json(directory / "person.json", person)
            _write_json(directory / "history.json", [])
            _write_json(directory / "predictions.json", [])
            _write_json(directory / "versions.json", [])
        self.cognitive.seed_case(identifier, case)
