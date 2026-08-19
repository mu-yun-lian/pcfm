from __future__ import annotations

from .conversation._shared import *  # noqa: F401, F403
from .conversation._shared import (  # noqa: F401
    _canonical_hash,
    _decode_web_bytes,
    _derivation_view,
    _extract_html,
    _extract_pdf,
    _extract_qa,
    _is_short_reference,
    _json_mapping,
    _localize_view,
    _read_json,
    _segments,
    _similarity,
    _structured_rows_text,
    _text_hash,
    _tokens,
    _utc_now,
    _write_json,
)
from .conversation.session_store import SessionStoreMixin
from .conversation.sources import SourcesMixin
from .conversation.extraction import ExtractionMixin
from .conversation.version_builder import VersionBuilderMixin
from .conversation.rendering import RenderingMixin
from .conversation.verdict import VerdictMixin
from .conversation.composing import ComposingMixin
from .conversation.message_pipeline import MessagePipelineMixin
from .conversation.reality_lookup import RealityLookupMixin
from .conversation.optimization import OptimizationMixin
from .conversation.read_path import ReadPathMixin
from .conversation.summary import SummaryMixin


class ConversationWorkbench(
    SessionStoreMixin,
    SourcesMixin,
    ExtractionMixin,
    VersionBuilderMixin,
    RenderingMixin,
    VerdictMixin,
    ComposingMixin,
    MessagePipelineMixin,
    RealityLookupMixin,
    OptimizationMixin,
    SummaryMixin,
    ReadPathMixin,
):
    def __init__(
        self, people_dir: Path, *, model_services: ModelServiceManager | None = None
    ) -> None:
        self.people_dir = Path(people_dir).resolve()
        self._renderers: dict[str, object] = {
            "steve_jobs_v1": ExpressionRenderer(builtin_expression_profile_path()),
            "neutral_v1": None,
        }
        self._legacy_predictor = ResponsePredictionKernel()
        self._predictor = ResponsePredictionKernelV2()
        self._simulation_predictor = SimulationKernelV5()
        self._model_services = model_services
        self._report_cache: dict[tuple, object] = {}
        self._progress: dict[str, object] = {}

    def processing_progress(self, person_id: str) -> dict[str, object]:
        progress = self._progress.get(person_id)
        if isinstance(progress, dict):
            return dict(progress)
        return {"active": False}

    @staticmethod
    def _file_tag(path: Path) -> tuple:
        try:
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)
        except OSError:
            return (0, 0)

    def _simulation_files_tag(self, person_id: str, version_number: int) -> tuple:
        return (
            self._file_tag(self._simulation_model_path(person_id, int(version_number))),
            self._file_tag(self._path(person_id, "conversation_sources.json")),
            self._file_tag(self._path(person_id, "conversation_profile.json")),
            self._file_tag(self._path(person_id, "conversation_versions.json")),
        )

    def _person_dir(self, person_id: str) -> Path:
        path = (self.people_dir / person_id).resolve()
        if path.parent != self.people_dir or not (path / "person.json").exists():
            raise ConversationError("人物不存在。")
        return path

    def _path(self, person_id: str, name: str) -> Path:
        return self._person_dir(person_id) / name

    def _person(self, person_id: str) -> dict[str, object]:
        raw = _read_json(self._path(person_id, "person.json"), {})
        if not isinstance(raw, dict):
            raise ConversationError("人物文件损坏。")
        return dict(raw)

    def _list(self, person_id: str, name: str) -> list[dict[str, object]]:
        raw = _read_json(self._path(person_id, name), [])
        if not isinstance(raw, list):
            raise ConversationError(f"{name} 文件损坏。")
        return [dict(item) for item in raw]

    def _state(self, person_id: str) -> dict[str, object]:
        raw = _read_json(
            self._path(person_id, "conversation_state.json"),
            {"schema_version": SCHEMA_VERSION, "active_version": None, "rollback_history": []},
        )
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise ConversationError("对话状态版本不受支持。")
        return dict(raw)

    def select_dialogue_model(
        self, person_id: str, model_ref: str
    ) -> dict[str, object]:
        state = self._state(person_id)
        state["dialogue_model_ref"] = str(model_ref).strip()
        state["dialogue_model_selected_at"] = _utc_now()
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return copy.deepcopy(state)

    def start_new_conversation(self, person_id: str) -> dict[str, object]:
        """新建一个空会话并设为活跃；旧的归档到 conversation_archives 已由会话文件取代。"""
        self.profile(person_id)
        return self.create_session(person_id)

    def _telemetry(self, person_id: str) -> dict[str, int]:
        raw = _read_json(
            self._path(person_id, "conversation_telemetry.json"),
            {
                "content_retrieval_calls": 0,
                "content_prediction_calls": 0,
                "content_generation_llm_calls": 0,
                "content_planning_llm_calls": 0,
                "validation_llm_calls": 0,
                "reality_lookup_requests": 0,
                "reality_local_search_calls": 0,
                "reality_online_search_calls": 0,
            },
        )
        return {str(key): int(value) for key, value in dict(raw).items()}

    def _save_telemetry(self, person_id: str, value: Mapping[str, int]) -> None:
        _write_json(self._path(person_id, "conversation_telemetry.json"), dict(value))

    @staticmethod
    def _conversation_context(
        profile: Mapping[str, object],
        messages: Sequence[Mapping[str, object]],
        current_text: str,
    ) -> dict[str, object]:
        """Recompute durable dialogue state from raw messages; summaries are caches."""
        topics: list[dict[str, object]] = []
        message_topic: dict[str, str] = {}
        last_user_topic = ""
        for item in messages:
            message_id = str(item.get("message_id", ""))
            role = str(item.get("role", ""))
            text = str(item.get("text", "")).strip()
            if not message_id or not text:
                continue
            if role == "user":
                ranked = sorted(
                    (
                        (_similarity(text, str(topic["anchor_text"])), str(topic["topic_id"]), topic)
                        for topic in topics
                    ),
                    key=lambda value: (-value[0], value[1]),
                )
                if ranked and ranked[0][0] >= 0.22:
                    topic = ranked[0][2]
                else:
                    topic = {
                        "topic_id": f"topic-{message_id}",
                        "anchor_text": text[:500],
                        "message_ids": [],
                        "last_touched_index": 0,
                    }
                    topics.append(topic)
                topic["message_ids"].append(message_id)
                topic["last_touched_index"] = len(message_topic)
                last_user_topic = str(topic["topic_id"])
                message_topic[message_id] = last_user_topic
            elif role == "assistant" and last_user_topic:
                topic = next(value for value in topics if value["topic_id"] == last_user_topic)
                topic["message_ids"].append(message_id)
                message_topic[message_id] = last_user_topic
        active = None
        if topics:
            if not _is_short_reference(current_text):
                ranked = sorted(
                    (
                        (_similarity(current_text, str(topic["anchor_text"])), str(topic["topic_id"]), topic)
                        for topic in topics
                    ),
                    key=lambda value: (-value[0], value[1]),
                )
                if ranked and ranked[0][0] >= 0.18:
                    active = ranked[0][2]
            if active is None:
                active = max(topics, key=lambda value: int(value["last_touched_index"]))
        assistant_commitments = [
            {
                "message_id": str(item.get("message_id", "")),
                "claims": [
                    str(claim.get("text", ""))
                    for claim in dict(item.get("structured_prediction") or {}).get("claims", [])
                    if claim.get("text")
                ],
                "person_prediction_status": str(item.get("person_prediction_status", "")),
                "context_only_not_person_evidence": True,
            }
            for item in messages
            if item.get("role") == "assistant" and item.get("message_id")
        ][-12:]
        public_topics = [
            {
                "topic_id": str(topic["topic_id"]),
                "anchor_text": str(topic["anchor_text"]),
                "message_ids": list(map(str, topic["message_ids"])),
                "last_touched_index": int(topic["last_touched_index"]),
            }
            for topic in topics[-24:]
        ]
        context = {
            "schema_version": "pcfm-conversation-state-v1",
            "current_delta": current_text,
            "current_topic": str(active["anchor_text"]) if active else current_text[:500],
            "active_topic_id": str(active["topic_id"]) if active else "",
            "active_topic_message_ids": list(map(str, active["message_ids"])) if active else [],
            "topic_threads": public_topics,
            "relationship": str(profile.get("relationship", "public_user")),
            "occasion": str(profile.get("occasion", "ordinary_chat")),
            "time_stage": copy.deepcopy(profile.get("time_scope", {})),
            "assistant_commitments": assistant_commitments,
            "recent_context": [
                {
                    "message_id": str(item.get("message_id", "")),
                    "role": str(item.get("role", "")),
                    "text": str(item.get("text", "")),
                }
                for item in messages[-6:]
            ],
            "prior_claims": [
                claim
                for item in assistant_commitments
                for claim in item["claims"]
            ][-12:],
            "generated_dialogue_is_fitting_evidence": False,
        }
        context["context_digest"] = _canonical_hash(context)
        return context

    def configure(
        self,
        person_id: str,
        *,
        aliases: Sequence[str],
        language: str,
        time_start: str = "",
        time_end: str = "",
        style_profile_id: str | None = None,
        source_mode: str = "user_provided",
        identity_note: str = "",
        focus_domain: str = "",
        generation_params: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        person = self._person(person_id)
        name = str(person["name"])
        selected_style = style_profile_id or (
            "steve_jobs_v1" if name.casefold() in {"steve jobs", "史蒂夫·乔布斯", "史蒂夫 乔布斯"} else "neutral_v1"
        )
        if selected_style not in self._renderers:
            raise ConversationError("表达包不存在。")
        if source_mode not in {"user_provided", "system_search"}:
            raise ConversationError("资料来源方式必须是用户提供或系统自动搜索。")
        # 生成参数随人物变化：默认按表达包取温度，显式 generation_params 可覆盖。
        temperature = CHARACTER_GENERATION_TEMPERATURE.get(
            selected_style, DEFAULT_GENERATION_TEMPERATURE
        )
        if generation_params:
            try:
                temperature = float(dict(generation_params).get("temperature", temperature))
            except (TypeError, ValueError):
                pass
        temperature = min(max(temperature, 0.0), 2.0)
        profile = {
            "schema_version": SCHEMA_VERSION,
            "person_id": person_id,
            "aliases": sorted({str(item).strip() for item in aliases if str(item).strip()}),
            "language": str(language).strip() or "zh",
            "time_scope": {"start": str(time_start).strip(), "end": str(time_end).strip()},
            "style_profile_id": selected_style,
            "content_model_kind": "pcfm_conversation_conditioned_response_simulation_v5",
            "response_accuracy_status": "not_assessed",
            "identity_note": str(identity_note).strip(),
            "focus_domain": str(focus_domain).strip(),
            "generation_params": {"temperature": temperature},
            "collection": {
                "mode": source_mode,
                "status": "search_ready" if source_mode == "system_search" else "awaiting_user_materials",
                "message": (
                    "搜索服务已配置；结果只会进入待审核候选资料。"
                    if source_mode == "system_search"
                    else "可粘贴或上传原始资料，系统会自动提取响应事件候选。"
                ),
            },
            "created_at": _utc_now(),
        }
        _write_json(self._path(person_id, "conversation_profile.json"), profile)
        for name, default in (
            ("conversation_sources.json", []),
            ("conversation_messages.json", []),
            ("conversation_versions.json", []),
            ("optimization_candidates.json", []),
            ("conversation_telemetry.json", self._telemetry(person_id)),
            ("conversation_state.json", self._state(person_id)),
        ):
            path = self._path(person_id, name)
            if not path.exists():
                _write_json(path, default)
        return profile

    def profile(self, person_id: str) -> dict[str, object]:
        raw = _read_json(self._path(person_id, "conversation_profile.json"), {})
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            person = self._person(person_id)
            return self.configure(
                person_id,
                aliases=[],
                language="zh",
                style_profile_id="neutral_v1",
            )
        return dict(raw)

    def _generation_temperature(self, person_id: str) -> float:
        """读取当前人物画像里的生成温度（随人物变化），越界/非法回退默认值。

        旧画像没有 generation_params 字段时，回退到该表达包的人物默认温度，
        而不是全局默认，保证「温度随人物变化」对既有人物同样生效。
        """
        profile = self.profile(person_id)
        style_id = str(profile.get("style_profile_id", ""))
        fallback = CHARACTER_GENERATION_TEMPERATURE.get(
            style_id, DEFAULT_GENERATION_TEMPERATURE
        )
        params = dict(profile.get("generation_params") or {})
        try:
            value = float(params.get("temperature", fallback))
        except (TypeError, ValueError):
            value = fallback
        return min(max(value, 0.0), 2.0)
