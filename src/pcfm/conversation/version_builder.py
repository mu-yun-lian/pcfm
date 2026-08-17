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



class VersionBuilderMixin:
    def _backup_before_version(self, person_id: str, version_number: int) -> None:
        """版本创建前备份元数据，保留最近 5 份；失败不阻断。"""
        try:
            backup_dir = self._person_dir(person_id) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            snapshot = {
                "person_id": person_id,
                "version": version_number,
                "created_at": _utc_now(),
                "conversation_versions": self._list(person_id, "conversation_versions.json"),
                "conversation_state": _read_json(self._path(person_id, "conversation_state.json"), {}),
            }
            _write_json(backup_dir / f"pre-version-{version_number}.json", snapshot)
            backups = sorted(
                backup_dir.glob("pre-version-*.json"),
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
            for old in backups[5:]:
                old.unlink(missing_ok=True)
        except Exception:
            pass

    def _version_source_ids(self, person_id: str, version_number: int | None = None) -> list[str]:
        versions = self._list(person_id, "conversation_versions.json")
        state = self._state(person_id)
        target = version_number if version_number is not None else state.get("active_version")
        for version in versions:
            if int(version["version"]) == int(target or -1):
                return [str(value) for value in version.get("source_ids", [])]
        return []

    def _response_model_path(self, person_id: str, version_number: int) -> Path:
        return self._path(
            person_id, f"response_models/response-model-v{int(version_number)}.json"
        )

    def _simulation_model_path(self, person_id: str, version_number: int) -> Path:
        return self._path(
            person_id, f"simulation_models/simulation-model-v{int(version_number)}.json"
        )

    def _style_artifact_path(self, person_id: str, version_number: int) -> Path:
        return self._path(
            person_id, f"style_profiles/style-profile-v{int(version_number)}.json"
        )

    def _distill_surface_style(
        self, person_id: str, version_number: int, events: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        answers = [str(event.get("actual_response", "")).strip() for event in events]
        answers = [value for value in answers if value]
        sentences = [sentence for answer in answers for sentence in _segments(answer)]
        total_chars = sum(len(value) for value in answers)
        punctuation = {
            mark: sum(value.count(mark) for value in answers)
            for mark in (".", ",", ";", ":", "?", "!", "。", "，", "；", "：", "？", "！")
        }
        observed_rules: list[dict[str, object]] = []
        connector_counts: list[tuple[int, str, list[str]]] = []
        for connector in SAFE_SURFACE_CONNECTORS:
            matching_ids = [
                str(event["event_id"])
                for event, answer in zip(events, [str(item.get("actual_response", "")).strip() for item in events], strict=True)
                if answer.casefold().startswith(connector.casefold())
                or re.search(rf"[.!?。！？]\s*{re.escape(connector)}", answer, flags=re.IGNORECASE)
            ]
            if len(matching_ids) >= 2:
                connector_counts.append((len(matching_ids), connector, matching_ids))
        connector_counts.sort(key=lambda item: (-item[0], SAFE_SURFACE_CONNECTORS.index(item[1])))
        for index, (count, connector, event_ids) in enumerate(connector_counts[:2]):
            operation = "prefix_first_claim" if index == 0 else "prefix_first_reason"
            observed_rules.append(
                {
                    "rule_id": f"observed-connector-{index + 1}",
                    "category": "A_surface",
                    "review_status": "confirmed_from_verified_responses",
                    "operation": operation,
                    "prefix": connector,
                    "observed_count": count,
                    "provenance_event_ids": sorted(set(event_ids)),
                    "classification_reason": "repeated sentence-opening connector; carries no person fact or stance",
                }
            )
        artifact = {
            "schema_version": "pcfm-person-surface-style-v2",
            "person_id": person_id,
            "version": int(version_number),
            "created_at": _utc_now(),
            "source_event_ids": sorted(str(event["event_id"]) for event in events),
            "source_event_digest": _canonical_hash(
                sorted((str(event["event_id"]), str(event["content_hash"])) for event in events)
            ),
            "content_fields_excluded": [
                "beliefs", "values", "positions", "facts", "decisions", "memories"
            ],
            "surface_statistics": {
                "sample_count": len(answers),
                "mean_answer_characters": round(total_chars / max(len(answers), 1), 3),
                "mean_sentence_characters": round(
                    sum(len(str(item.get("text", ""))) for item in sentences)
                    / max(len(sentences), 1),
                    3,
                ),
                "question_mark_rate": round(sum("?" in value or "？" in value for value in answers) / max(len(answers), 1), 4),
                "exclamation_mark_rate": round(sum("!" in value or "！" in value for value in answers) / max(len(answers), 1), 4),
                "first_person_rate": round(sum(bool(re.search(r"\b(i|we|my|our)\b|我|我们", value.casefold())) for value in answers) / max(len(answers), 1), 4),
                "punctuation_counts": punctuation,
            },
            "surface_rules": observed_rules,
            "provenance": [
                {
                    "event_id": str(event["event_id"]),
                    "content_hash": str(event["content_hash"]),
                    "evidence_role": "verified_person_response_parameter_training",
                }
                for event in events
            ],
            "runtime_protocol": "observed_surface_connectors_over_exact_locked_segments",
            "validation_status": (
                "rendering_enabled_semantic_gate_required"
                if observed_rules
                else "style_material_ready_rendering_not_enabled"
            ),
        }
        artifact["artifact_hash"] = _canonical_hash(artifact)
        _write_json(self._style_artifact_path(person_id, version_number), artifact)
        return artifact

    def _trainable_events(
        self, person_id: str, source_ids: Sequence[str]
    ) -> list[dict[str, object]]:
        return [
            dict(event)
            for source in self._source_records(person_id, source_ids)
            for event in source.get("response_events", [])
            if event.get("label_status")
            == "confirmed_response_weak_semantic_labels"
            and event.get("data_role") == "parameter_training"
        ]

    def _population_events(
        self, person_id: str
    ) -> tuple[list[dict[str, object]], int]:
        events: list[dict[str, object]] = []
        people: set[str] = set()
        for path in sorted(self.people_dir.glob("*/conversation_sources.json")):
            other_id = path.parent.name
            if other_id == person_id:
                continue
            raw = _read_json(path, [])
            if not isinstance(raw, list):
                continue
            person_events = [
                dict(event)
                for source in raw
                if isinstance(source, Mapping)
                and source.get("review_status") == "confirmed"
                for event in source.get("response_events", [])
                if event.get("label_status")
                == "confirmed_response_weak_semantic_labels"
                and event.get("data_role") == "parameter_training"
            ]
            if person_events:
                people.add(other_id)
                events.extend(person_events)
        return events, len(people)

    def _fit_response_model(
        self,
        person_id: str,
        *,
        version_number: int,
        source_ids: Sequence[str],
    ) -> dict[str, object]:
        profile = self.profile(person_id)
        events = self._trainable_events(person_id, source_ids)
        population_events, population_people = self._population_events(person_id)
        try:
            artifact = self._predictor.fit(
                person_id=person_id,
                version=version_number,
                events=events,
                population_events=population_events,
                population_people=population_people,
                scope={
                    "focus_domain": profile.get("focus_domain", ""),
                    "language": profile.get("language", ""),
                    "time_scope": profile.get("time_scope", {}),
                    "identity_note": profile.get("identity_note", ""),
                },
            )
        except ResponsePredictionError as error:
            raise ConversationError(
                "已确认资料中没有可训练的本人公开表达事件；资料仍会保留为证据候选。"
            ) from error
        _write_json(self._response_model_path(person_id, version_number), artifact)
        return artifact

    def _response_model(
        self, person_id: str, version_number: int
    ) -> dict[str, object]:
        path = self._response_model_path(person_id, version_number)
        if not path.exists():
            artifact = self._fit_response_model(
                person_id,
                version_number=version_number,
                source_ids=self._version_source_ids(person_id, version_number),
            )
            versions = self._list(person_id, "conversation_versions.json")
            for version in versions:
                if int(version["version"]) == int(version_number):
                    version["response_model_path"] = str(
                        path.relative_to(self._person_dir(person_id)).as_posix()
                    )
                    version["response_model_hash"] = artifact["artifact_hash"]
                    version["content_model_kind"] = "pcfm_unified_response_predictor_v2"
            _write_json(self._path(person_id, "conversation_versions.json"), versions)
            return artifact
        raw = _read_json(path, {})
        if not isinstance(raw, dict):
            raise ConversationError("人物响应模型文件损坏。")
        if (
            raw.get("schema_version") != MODEL_SCHEMA_V2
            or dict(raw.get("feature_schema") or {}).get("public_response_model")
            != "episode_tendency_knowledge_v2"
            or "overall_tendencies" not in raw
            or "event_relations" not in raw
        ):
            migrated_from = str(raw.get("schema_version", "unknown"))
            artifact = self._fit_response_model(
                person_id,
                version_number=version_number,
                source_ids=self._version_source_ids(person_id, version_number),
            )
            versions = self._list(person_id, "conversation_versions.json")
            for version in versions:
                if int(version["version"]) == int(version_number):
                    version["content_model_kind"] = "pcfm_unified_response_predictor_v2"
                    version["response_model_hash"] = artifact["artifact_hash"]
                    version["artifact_migration"] = {
                        "from_schema": migrated_from,
                        "to_schema": MODEL_SCHEMA_V2,
                        "method": "refit_from_reviewed_version_sources",
                        "migrated_at": _utc_now(),
                    }
            _write_json(self._path(person_id, "conversation_versions.json"), versions)
            return artifact
        try:
            self._predictor.verify(raw)
        except ResponsePredictionError as error:
            raise ConversationError("人物响应模型完整性校验失败。") from error
        return dict(raw)

    def _reviewed_sources_for_simulation_v4(
        self, person_id: str, source_ids: Sequence[str]
    ) -> list[dict[str, object]]:
        """Read reviewed sources; only V4 reviewed frames carry inferred semantics."""
        allowed = set(map(str, source_ids))
        reviewed = []
        for raw in self._list(person_id, "conversation_sources.json"):
            if (
                raw.get("review_status") != "confirmed"
                or str(raw.get("source_id", "")) not in allowed
            ):
                continue
            reviewed.append(self._simulation_source_view(raw))
        return reviewed

    @staticmethod
    def _simulation_source_view(raw: Mapping[str, object]) -> dict[str, object]:
        source = copy.deepcopy(dict(raw))
        optimization_id = str(source.get("optimization_candidate_id", ""))
        if optimization_id:
            source["qas"] = []
            source["segments"] = []
            source["reviewed_event_frames_v4"] = [
                item
                for item in source.get("reviewed_event_frames_v4", [])
                if item.get("optimization_candidate_id") == optimization_id
            ]
        return source

    def _fit_simulation_model(
        self,
        person_id: str,
        *,
        version_number: int,
        source_ids: Sequence[str],
    ) -> dict[str, object]:
        profile = self.profile(person_id)
        person = self._person(person_id)
        try:
            artifact = self._simulation_predictor.fit(
                person_id=person_id,
                version=version_number,
                reviewed_sources=self._reviewed_sources_for_simulation_v4(
                    person_id, source_ids
                ),
                scope={
                    "focus_domain": profile.get("focus_domain", ""),
                    "language": profile.get("language", ""),
                    "time_scope": profile.get("time_scope", {}),
                    "identity_note": profile.get("identity_note", ""),
                    # 供确定性身份兜底：identity_note 为空时用人物名称+描述回答「你是谁」
                    "person_name": str(person.get("name", "")),
                    "person_description": str(person.get("description", "")),
                },
            )
        except SimulationV5Error as error:
            raise ConversationError(
                "No eligible reviewed response episode is available for simulation V5."
            ) from error
        _write_json(self._simulation_model_path(person_id, version_number), artifact)
        return artifact

    def _simulation_model(
        self, person_id: str, version_number: int
    ) -> dict[str, object]:
        cache_key = (
            "simulation_model",
            str(person_id),
            int(version_number),
            self._simulation_files_tag(person_id, version_number),
        )
        cached = self._report_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        path = self._simulation_model_path(person_id, version_number)
        raw = _read_json(path, {}) if path.exists() else {}
        needs_refit = (
            not isinstance(raw, dict)
            or raw.get("schema_version") != MODEL_SCHEMA_V5
            or raw.get("model_build") != MODEL_BUILD_V5
        )
        if not needs_refit:
            try:
                self._simulation_predictor.verify(raw)
            except SimulationV5Error as error:
                raise ConversationError(
                    "Simulation V5 artifact integrity validation failed."
                ) from error
            profile = self.profile(person_id)
            person = self._person(person_id)
            try:
                recomputed = self._simulation_predictor.fit(
                    person_id=person_id,
                    version=version_number,
                    reviewed_sources=self._reviewed_sources_for_simulation_v4(
                        person_id,
                        self._version_source_ids(person_id, version_number),
                    ),
                    scope={
                        "focus_domain": profile.get("focus_domain", ""),
                        "language": profile.get("language", ""),
                        "time_scope": profile.get("time_scope", {}),
                        "identity_note": profile.get("identity_note", ""),
                        "person_name": str(person.get("name", "")),
                        "person_description": str(person.get("description", "")),
                    },
                )
            except SimulationV5Error as error:
                raise ConversationError(
                    "Simulation V5 no longer recomputes from the version's reviewed source bytes."
                ) from error
            if (
                recomputed.get("semantic_model_digest")
                != raw.get("semantic_model_digest")
            ):
                raise ConversationError(
                    "Simulation V5 no longer recomputes from the version's reviewed source bytes."
                )
        if needs_refit:
            artifact = self._fit_simulation_model(
                person_id,
                version_number=version_number,
                source_ids=self._version_source_ids(person_id, version_number),
            )
        else:
            artifact = dict(raw)
        versions = self._list(person_id, "conversation_versions.json")
        changed = False
        for version in versions:
            if int(version["version"]) == int(version_number) and (
                version.get("content_model_kind")
                != "pcfm_conversation_conditioned_response_simulation_v5"
                or version.get("simulation_model_hash")
                != artifact["artifact_hash"]
            ):
                version["content_model_kind"] = "pcfm_conversation_conditioned_response_simulation_v5"
                version["simulation_model_path"] = str(
                    path.relative_to(self._person_dir(person_id)).as_posix()
                )
                version["simulation_model_hash"] = artifact["artifact_hash"]
                version["active_components"] = artifact["active_components"]
                version["components"] = artifact["components"]
                version["v2_response_model_role"] = "frozen_baseline_only"
                version["simulation_v3_role"] = "frozen_baseline_only"
                changed = True
        if changed:
            _write_json(self._path(person_id, "conversation_versions.json"), versions)
        profile = self.profile(person_id)
        if (
            profile.get("content_model_kind")
            != "pcfm_conversation_conditioned_response_simulation_v5"
        ):
            profile["content_model_kind"] = (
                "pcfm_conversation_conditioned_response_simulation_v5"
            )
            profile["response_accuracy_status"] = "not_assessed"
            _write_json(
                self._path(person_id, "conversation_profile.json"), profile
            )
        self._report_cache[cache_key] = copy.deepcopy(artifact)
        return artifact

    def _create_version(
        self,
        person_id: str,
        *,
        source_ids: Sequence[str],
        reason: str,
        validation_status: str,
        parent_version: int | None,
        update_style: bool = True,
    ) -> dict[str, object]:
        versions = self._list(person_id, "conversation_versions.json")
        profile = self.profile(person_id)
        version_number = len(versions) + 1
        parent = next(
            (
                item
                for item in versions
                if parent_version is not None
                and int(item["version"]) == int(parent_version)
            ),
            None,
        )
        try:
            baseline_artifact = self._fit_response_model(
                person_id,
                version_number=version_number,
                source_ids=source_ids,
            )
            baseline_status = "available_frozen_baseline_only"
        except ConversationError:
            baseline_artifact = None
            baseline_status = "unavailable_not_required_by_v4"
        baseline_model_path = self._response_model_path(person_id, version_number)
        artifact = self._fit_simulation_model(
            person_id,
            version_number=version_number,
            source_ids=source_ids,
        )
        model_path = self._simulation_model_path(person_id, version_number)
        if update_style or parent is None:
            style_artifact = self._distill_surface_style(
                person_id, version_number, self._trainable_events(person_id, source_ids)
            )
            style_path = self._style_artifact_path(person_id, version_number)
            style_status = (
                "rendering_enabled_exploratory"
                if style_artifact["surface_rules"]
                else "style_material_ready_rendering_not_enabled"
            )
            style_revision = int(parent.get("style_revision", 0)) + 1 if parent else 1
        else:
            style_artifact = {
                "artifact_hash": parent["style_artifact_hash"],
                "surface_rules": [],
            }
            style_path = self._person_dir(person_id) / str(parent["style_artifact_path"])
            style_status = "unchanged_separate_review_required"
            style_revision = int(parent.get("style_revision", 1))
        version = {
            "schema_version": SCHEMA_VERSION,
            "version": version_number,
            "parent_version": parent_version,
            "created_at": _utc_now(),
            "reason": reason,
            "source_ids": sorted(set(map(str, source_ids))),
            "source_set_digest": _canonical_hash(sorted(set(map(str, source_ids)))),
            "content_model_kind": "pcfm_conversation_conditioned_response_simulation_v5",
            "simulation_model_path": str(
                model_path.relative_to(self._person_dir(person_id)).as_posix()
            ),
            "simulation_model_hash": artifact["artifact_hash"],
            "response_model_path": (
                str(baseline_model_path.relative_to(self._person_dir(person_id)).as_posix())
                if baseline_artifact is not None
                else None
            ),
            "response_model_hash": (
                baseline_artifact["artifact_hash"]
                if baseline_artifact is not None
                else None
            ),
            "v2_response_model_role": "frozen_baseline_only",
            "simulation_v3_role": "frozen_baseline_only",
            "v2_response_model_status": baseline_status,
            "active_components": artifact["active_components"],
            "components": artifact["components"],
            "style_profile_id": profile["style_profile_id"],
            "content_revision": int(parent.get("content_revision", 0)) + 1 if parent else 1,
            "style_revision": style_revision,
            "content_update_status": "applied_exploratory",
            "style_update_status": style_status,
            "style_artifact_path": str(
                style_path.relative_to(self._person_dir(person_id)).as_posix()
            ),
            "style_artifact_hash": style_artifact["artifact_hash"],
            "validation_status": validation_status,
            "response_accuracy_status": "not_assessed",
        }
        self._backup_before_version(person_id, int(version["version"]))
        versions.append(version)
        _write_json(self._path(person_id, "conversation_versions.json"), versions)
        state = self._state(person_id)
        state["active_version"] = version["version"]
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return version

    def _create_style_only_version(
        self, person_id: str, *, candidate_id: str
    ) -> dict[str, object]:
        versions = self._list(person_id, "conversation_versions.json")
        state = self._state(person_id)
        parent = next(
            item
            for item in versions
            if int(item["version"]) == int(state.get("active_version") or -1)
        )
        version_number = len(versions) + 1
        source_ids = [str(value) for value in parent.get("source_ids", [])]
        style_artifact = self._distill_surface_style(
            person_id,
            version_number,
            self._trainable_events(person_id, source_ids),
        )
        style_path = self._style_artifact_path(person_id, version_number)
        version = {
            **copy.deepcopy(parent),
            "version": version_number,
            "parent_version": int(parent["version"]),
            "created_at": _utc_now(),
            "reason": f"style optimization candidate {candidate_id}",
            "content_revision": int(parent.get("content_revision", 1)),
            "style_revision": int(parent.get("style_revision", 1)) + 1,
            "content_update_status": "unchanged",
            "style_update_status": (
                "rendering_enabled_exploratory"
                if style_artifact["surface_rules"]
                else "style_material_ready_rendering_not_enabled"
            ),
            "style_artifact_path": str(
                style_path.relative_to(self._person_dir(person_id)).as_posix()
            ),
            "style_artifact_hash": style_artifact["artifact_hash"],
            "validation_status": "style_source_integrity_and_semantic_gate_passed_accuracy_not_assessed",
        }
        self._backup_before_version(person_id, int(version["version"]))
        versions.append(version)
        _write_json(self._path(person_id, "conversation_versions.json"), versions)
        state["active_version"] = version_number
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return copy.deepcopy(version)
