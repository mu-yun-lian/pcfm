from __future__ import annotations

import logging

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _as_choice,
    _canonical_hash,
    _parse_time,
    _read_json,
    _reason_text,
    _slug,
    _utc_now,
    _write_json,
)
from .person_service import PersonServiceMixin
from .conversation_service import ConversationServiceMixin
from .source_service import SourceServiceMixin
from .extraction_service import ExtractionServiceMixin
from .model_service_admin import ModelServiceAdminMixin
from .version_service import VersionServiceMixin
from .archive_service import ArchiveServiceMixin
from .job_service import JobServiceMixin
from .train_service import TrainServiceMixin
from .prediction_service import PredictionServiceMixin
from .cognitive_service import CognitiveServiceMixin


class PcfmService(
    PersonServiceMixin,
    ConversationServiceMixin,
    SourceServiceMixin,
    ExtractionServiceMixin,
    ModelServiceAdminMixin,
    VersionServiceMixin,
    ArchiveServiceMixin,
    JobServiceMixin,
    TrainServiceMixin,
    PredictionServiceMixin,
    CognitiveServiceMixin,
):
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
            # 默认启用免 key 的 Bing RSS 搜索；设 PCFM_PUBLIC_SEARCH_PROVIDER=none 可关闭。
            configured_provider = os.environ.get("PCFM_PUBLIC_SEARCH_PROVIDER", "bing_rss").strip().casefold()
            self.public_search = (
                BingRssPublicSearch() if configured_provider != "none" else None
            )
        else:
            self.public_search = None if public_search is False else public_search
        self.wikipedia = WikipediaCollector()
        self.expression_renderer = ExpressionRenderer(
            builtin_expression_profile_path()
        )
        self.expression_records_path = self.data_dir / "expression_renders.json"
        if not self.expression_records_path.exists():
            _write_json(self.expression_records_path, [])
        self.assistant = AssistantEngine(self, self.data_dir / "assistant_state.json")
        self._lock = threading.RLock()
        self._locks_guard = threading.Lock()
        self._person_locks: dict[str, threading.RLock] = {}
        self.db = Database(self.data_dir / "pcfm.db")
        self.person_repo = PersonRepository(self.db)
        self.source_repo = SourceRepository(self.db)
        self.version_repo = VersionRepository(self.db)
        self.session_repo = SessionRepository(self.db)
        self.message_repo = MessageRepository(self.db)
        self.state_repo = ConversationStateRepository(self.db)
        self.conversation._source_repo = self.source_repo
        self.conversation._version_repo = self.version_repo
        self.job_store = JobStore(self.db)
        self.job_runner = JobRunner(self.job_store, max_workers=2)
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
            try:
                result = self._conversation_call(
                    self.conversation.migrate_evidence_contract, path.parent.name
                )
                if result.get("changed"):
                    self._sync_sources_to_sqlite(path.parent.name)
                    self._sync_versions_to_sqlite(path.parent.name)
            except Exception:
                # 单个人物迁移失败(含 conversation 数据损坏)不阻断启动
                continue

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
            changed = False
            for source_spec in definition["sources"]:
                source_url = str(source_spec["source_url"])
                normalized_url = source_url.rstrip("/")
                if normalized_url in existing_by_url:
                    aliases = list(source_spec.get("entity_aliases", []))
                    if aliases:
                        merged = self._conversation_call(
                            self.conversation.merge_source_entity_aliases,
                            person_id,
                            existing_by_url[normalized_url],
                            aliases,
                        )
                        changed = changed or bool(merged)
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
                changed = True
            if changed:
                self._sync_sources_to_sqlite(person_id)
                self._sync_versions_to_sqlite(person_id)

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
                "secret_storage": "windows_dpapi_server_only" if os.name == "nt" else "environment_only",
            },
        }

    def processing_progress(self, person_id: str) -> dict[str, object]:
        self._require_person(person_id)
        return self.conversation.processing_progress(person_id)

    def close(self) -> None:
        self.job_runner.shutdown(wait=False)
        self.db.close()

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

    def _sync_sources_to_sqlite(self, person_id: str) -> None:
        """把 conversation_sources.json 的资料元数据镜像到 source 表; 失败不影响主流程。"""
        try:
            self._ensure_person_in_sqlite(person_id)
            sources = self._conversation_call(
                self.conversation._list, person_id, "conversation_sources.json"
            )
            with self.db.transaction():
                self.source_repo.delete_by_person_no_commit(person_id)
                for item in sources:
                    record = dict(item)
                    record["person_id"] = person_id
                    self.source_repo.upsert_no_commit(record)
        except Exception:
            logging.getLogger("pcfm").warning(
                "sync_sources_to_sqlite failed person_id=%s", person_id, exc_info=True
            )

    def _sync_versions_to_sqlite(self, person_id: str) -> None:
        """把版本元数据 + 对话状态镜像到 SQLite, 用事务保证 version 与 active_version 原子一致。"""
        try:
            self._ensure_person_in_sqlite(person_id)
            versions = self._conversation_call(
                self.conversation._list, person_id, "conversation_versions.json"
            )
            state = self._conversation_call(self.conversation._state, person_id)
            state_record = {
                "active_version": state.get("active_version"),
                "active_session_id": state.get("active_session_id", ""),
                "dialogue_model_ref": state.get("dialogue_model_ref", ""),
                "updated_at": str(state.get("updated_at", "")),
            }
            with self.db.transaction():
                self.version_repo.delete_by_person_no_commit(person_id)
                for item in versions:
                    record = dict(item)
                    record["person_id"] = person_id
                    self.version_repo.upsert_no_commit(record)
                self.state_repo.upsert_no_commit(person_id, state_record)
        except Exception:
            logging.getLogger("pcfm").warning(
                "sync_versions_to_sqlite failed person_id=%s", person_id, exc_info=True
            )

    def _sync_sessions_to_sqlite(self, person_id: str) -> None:
        """把会话文件元数据镜像到 session 表; 失败不影响主流程。"""
        try:
            self._ensure_person_in_sqlite(person_id)
            active_id = self._conversation_call(self.conversation._active_session_id, person_id)
            sessions = self._conversation_call(self.conversation.list_sessions, person_id)
            with self.db.transaction():
                self.session_repo.delete_by_person_no_commit(person_id)
                for item in sessions:
                    record = dict(item)
                    record["person_id"] = person_id
                    record["active"] = str(item.get("session_id", "")) == str(active_id)
                    self.session_repo.upsert_no_commit(record)
        except Exception:
            logging.getLogger("pcfm").warning(
                "sync_sessions_to_sqlite failed person_id=%s", person_id, exc_info=True
            )

    def _sync_messages_to_sqlite(self, person_id: str) -> None:
        """把活动会话消息镜像到 message 表; 失败不影响主流程。"""
        try:
            self._ensure_person_in_sqlite(person_id)
            session_id = self._conversation_call(self.conversation._active_session_id, person_id)
            session = self._conversation_call(self.conversation._read_session, person_id, session_id)
            # 先确保会话存在(满足 message 外键)
            with self.db.transaction():
                session_record = dict(session)
                session_record["person_id"] = person_id
                session_record["active"] = True
                self.session_repo.upsert_no_commit(session_record)
                self.message_repo.delete_by_session_no_commit(session_id)
                for item in session.get("messages", []):
                    record = dict(item)
                    record["session_id"] = session_id
                    self.message_repo.upsert_no_commit(record)
        except Exception:
            logging.getLogger("pcfm").warning(
                "sync_messages_to_sqlite failed person_id=%s", person_id, exc_info=True
            )

    def _ensure_person_in_sqlite(self, person_id: str) -> None:
        """确保人物在 person 表(演示人物直接写文件未走 person_repo), 满足外键。"""
        try:
            if self.person_repo.get(person_id) is None:
                self.person_repo.upsert(self._require_person(person_id))
        except Exception:
            pass

    def _cognitive_call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except CognitiveWorkbenchError as error:
            raise ProductError(str(error)) from error

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
            self.person_repo.upsert(person)
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
            self._sync_sources_to_sqlite(person_id)
            self._sync_versions_to_sqlite(person_id)
            self._sync_sessions_to_sqlite(person_id)
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
