from __future__ import annotations

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
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



class SummaryMixin:
    def _baseline_files_tag(self) -> tuple:
        tags: list[tuple] = []
        for path in sorted(self.people_dir.glob("*/response_models/*.json")):
            tags.append(("rm", path.name, self._file_tag(path)[0], self._file_tag(path)[1]))
        for path in sorted(self.people_dir.glob("*/conversation_state.json")):
            tags.append(("st", path.parent.name, self._file_tag(path)[0], self._file_tag(path)[1]))
        for path in sorted(self.people_dir.glob("*/conversation_sources.json")):
            tags.append(("src", path.parent.name, self._file_tag(path)[0], self._file_tag(path)[1]))
        return tuple(tags)

    def _baseline_report(
        self, person_id: str, active_version: int | None
    ) -> dict[str, object]:
        if active_version is None:
            return {
                "status": "not_assessed",
                "reason": "active_response_model_required",
                "sample_count": 0,
            }
        cache_key = (
            "baseline_report",
            str(person_id),
            int(active_version),
            self._baseline_files_tag(),
        )
        cached = self._report_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        try:
            baseline_artifact = self._response_model(person_id, int(active_version))
        except ConversationError:
            result = {
                "status": "not_assessed",
                "reason": "v2_baseline_unavailable_v5_runtime_unaffected",
                "sample_count": 0,
            }
            self._report_cache[cache_key] = copy.deepcopy(result)
            return result
        holdout_events = [
            dict(event)
            for source in self._list(person_id, "conversation_sources.json")
            if source.get("review_status") == "confirmed"
            and source.get("dataset_role") == "final_holdout"
            for event in source.get("response_events", [])
            if event.get("label_status")
            == "confirmed_response_weak_semantic_labels"
            and event.get("data_role") == "sealed_final_validation"
        ]
        wrong_artifacts: list[dict[str, object]] = []
        for path in sorted(self.people_dir.glob("*/conversation_state.json")):
            other_id = path.parent.name
            if other_id == person_id:
                continue
            other_state = _read_json(path, {})
            if isinstance(other_state, Mapping) and other_state.get("active_version"):
                try:
                    wrong_artifacts.append(
                        self._response_model(
                            other_id, int(other_state["active_version"])
                        )
                    )
                except ConversationError:
                    continue
        result = self._predictor.compare_baselines(
            baseline_artifact,
            holdout_events,
            wrong_person_artifacts=wrong_artifacts,
        )
        self._report_cache[cache_key] = copy.deepcopy(result)
        return result

    def _simulation_validation_report(
        self, person_id: str, active_version: int | None
    ) -> dict[str, object]:
        if active_version is None:
            return {
                "status": "not_assessed",
                "reason": "active_simulation_v5_required",
                "sample_count": 0,
                "accuracy_claim": "none",
            }
        holdouts = [
            copy.deepcopy(source)
            for source in self._list(person_id, "conversation_sources.json")
            if source.get("review_status") == "confirmed"
            and source.get("dataset_role") == "final_holdout"
        ]
        return self._simulation_predictor.evaluate(
            self._simulation_model(person_id, active_version), holdouts
        )

    def summary(self, person_id: str, *, light: bool = False, full_messages: bool = False) -> dict[str, object]:
        profile = self.profile(person_id)
        state = self._read_state(person_id)
        active = state.get("active_version")
        if light:
            sources: list[dict[str, object]] = []
            confirmed: list[dict[str, object]] = []
            repo = getattr(self, "_source_repo", None)
            source_counts: dict[str, int] | None = None
            if repo is not None:
                try:
                    source_counts = repo.source_counts(person_id)
                except Exception:
                    source_counts = None
            json_sources = self._list(person_id, "conversation_sources.json")
            if source_counts is None or (
                source_counts.get("total", 0) == 0 and json_sources
            ):
                # SQLite 与 JSON 真相源不一致（或读取异常）→ 回退 JSON 重算 + 触发自愈，
                # 避免旧版本升级且来源未再变动的人物持续显示 0。
                source_counts = {
                    "total": len(json_sources),
                    "confirmed": sum(item.get("review_status") == "confirmed" for item in json_sources),
                    "pending": sum(item.get("review_status") == "pending" for item in json_sources),
                    "model_source": sum(item.get("review_status") == "confirmed" and item.get("dataset_role") == "model_source" for item in json_sources),
                    "final_holdout": sum(item.get("review_status") == "confirmed" and item.get("dataset_role") == "final_holdout" for item in json_sources),
                }
                sync = getattr(self, "_sync_callback", None)
                if sync and sync.get("sources") and repo is not None:
                    try:
                        sync["sources"](person_id)
                    except Exception:
                        pass
        else:
            sources = self._list(person_id, "conversation_sources.json")
            confirmed = [item for item in sources if item.get("review_status") == "confirmed"]
            source_counts = {
                "total": len(sources),
                "confirmed": len(confirmed),
                "pending": sum(item.get("review_status") == "pending" for item in sources),
                "model_source": sum(item.get("review_status") == "confirmed" and item.get("dataset_role") == "model_source" for item in sources),
                "final_holdout": sum(item.get("review_status") == "confirmed" and item.get("dataset_role") == "final_holdout" for item in sources),
            }
        versions = self._read_versions(person_id)
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        raw_messages = session.get("messages", [])
        messages = [dict(raw_messages[-1])] if (light and not full_messages and raw_messages) else [dict(item) for item in raw_messages]
        candidates = self._list(person_id, "optimization_candidates.json")
        if light:
            baseline_report = {
                "status": "not_assessed",
                "reason": "list_view_light",
                "sample_count": 0,
            }
            simulation_validation = {
                "status": "not_assessed",
                "reason": "list_view_light",
                "sample_count": 0,
                "accuracy_claim": "none",
            }
            active_model: dict[str, object] = {}
        else:
            baseline_report = self._baseline_report(
                person_id, int(active) if active is not None else None
            )
            simulation_validation = self._simulation_validation_report(
                person_id, int(active) if active is not None else None
            )
            active_model = self._simulation_model(person_id, int(active)) if active else {}
        if active:
            profile = self.profile(person_id)
        reviewed_model = dict(active_model.get("reviewed_public_model") or {})
        status = "exploratory" if active else "insufficient_evidence"
        return {
            "schema_version": SCHEMA_VERSION,
            "profile": profile,
            "status": status,
            "status_text": (
                "人物公开响应模型可用（探索性，准确性未验证）"
                if active
                else "尚未建立人物模型；普通问题可由所选大模型以通用模式回答"
            ),
            "messages": messages,
            "sources": [] if light else [self._source_public(item) for item in sources],
            "source_counts": source_counts,
            "active_version": active,
            "dialogue_model_ref": str(state.get("dialogue_model_ref", "")),
            "dialogue_state": copy.deepcopy(dict(session.get("dialogue_state") or {})),
            "session_id": session_id,
            "session_title": str(session.get("title", "")),
            "active_session_id": session_id,
            "dialogue_model_status": (
                "尚未选择对话模型；使用确定性证据计划与本地表达层"
                if not state.get("dialogue_model_ref")
                else "该对话模型尚未在当前人物固定测试集上验证"
            ),
            "versions": versions,
            "optimization_candidates": candidates,
            "telemetry": self._telemetry(person_id),
            "metrics": {
                "content_holdout_agreement": simulation_validation.get(
                    "covered_direction_accuracy", "not_assessed"
                ),
                "correct_person_uplift": "not_assessed",
                "confidence_calibration": "not_assessed",
                "fact_source_support": 1.0 if messages and all(item.get("evidence") for item in messages if item.get("role") == "assistant" and item.get("status") == "answered") else "not_assessed",
                "style_blind_test": "not_assessed",
                "style_semantic_preservation": "structural_gate_only",
                "out_of_scope_handling": "general_assisted_without_person_stance",
            },
            "public_response_model": {
                "schema_version": active_model.get("schema_version"),
                "event_frame_count": len(active_model.get("event_frames", [])),
                "preference_atom_count": len(reviewed_model.get("preference_atoms", [])),
                "value_atom_count": len(active_model.get("value_atoms", [])),
                "preference_structure_count": len(
                    active_model.get("orientation_index", [])
                ),
                "value_orientation_count": len(
                    active_model.get("value_orientation_index", [])
                ),
                "cross_domain_preference_count": sum(
                    item.get("status") == "cross_domain_public_preference"
                    for item in active_model.get("orientation_index", [])
                ),
                "conversation_conditioning": "full_history_state_plus_current_delta",
                "knowledge_claim_count": len(active_model.get("knowledge_claims", [])),
                "knowledge_boundary": "person_used_reason_not_complete_inner_knowledge",
                "accuracy_status": simulation_validation.get(
                    "status", "not_assessed"
                ),
            },
            "network_collection": {
                "direct_url_import": "available",
                "search_by_person_name": "not_configured",
                "notice": "当前没有可替换的联网搜索后端；系统不会伪装成已经自动搜集。",
            },
            "model_components": active_model.get("components", []),
            "baseline_report": baseline_report,
            "simulation_validation": simulation_validation,
        }
