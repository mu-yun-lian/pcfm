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



class OptimizationMixin:
    def create_optimization_candidate(
        self,
        person_id: str,
        message_id: str,
        *,
        allow_retry: bool = False,
        comparison_candidate_id: str = "",
    ) -> dict[str, object]:
        _messages, message = self._find_message(person_id, message_id)
        comparison = message.get("comparison")
        if not isinstance(comparison, Mapping) or comparison.get("status") != "candidate_found":
            raise ConversationError("这条回答还没有可用的现实回答对照。")
        reality_candidates = [
            dict(value)
            for value in comparison.get("reality_candidates", [])
            if isinstance(value, Mapping)
        ]
        if not reality_candidates:
            reality_candidates = [
                {
                    "comparison_candidate_id": "legacy-single-candidate",
                    "answer": comparison.get("reality_answer", ""),
                    "source_id": comparison.get("source_id", ""),
                }
            ]
        selected_id = str(
            comparison_candidate_id or comparison.get("selected_candidate_id") or ""
        )
        if not selected_id and len(reality_candidates) > 1:
            raise ConversationError("请先选择一条确实与本轮问题相近的现实回答。")
        selected = next(
            (
                value
                for value in reality_candidates
                if value.get("comparison_candidate_id") == selected_id
            ),
            reality_candidates[0] if len(reality_candidates) == 1 else None,
        )
        if selected is None:
            raise ConversationError("所选现实回答候选不存在。")
        candidates = self._list(person_id, "optimization_candidates.json")
        existing = [
            item
            for item in candidates
            if item.get("message_id") == message_id
            and item.get("source_id") == selected.get("source_id")
            and item.get("comparison_candidate_id")
            == selected.get("comparison_candidate_id")
        ]
        if existing and not (
            allow_retry and existing[-1].get("status") == "failed_validation"
        ):
            raise ConversationError("这条现实回答已经进入优化候选。")
        answer = str(selected["answer"])
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"optimization-{uuid.uuid4().hex[:12]}",
            "person_id": person_id,
            "message_id": message_id,
            "comparison_id": comparison["comparison_id"],
            "comparison_candidate_id": selected["comparison_candidate_id"],
            "source_id": selected["source_id"],
            "source_event_id": str(selected.get("event_id", "")),
            "source_content_hash": next(
                item["content_hash"]
                for item in self._list(person_id, "conversation_sources.json")
                if item["source_id"] == selected["source_id"]
            ),
            "status": "pending",
            "created_at": _utc_now(),
            "active_version_before": self._state(person_id).get("active_version"),
            "content_extraction": {
                "speech_act": "answer",
                "claims": [answer],
                "reasons": [],
                "facts": [],
                "experiences": [],
                "uncertainties": [],
            },
            "surface_extraction": {
                "sentence_count": len(re.findall(r"[.!?。！？]+", answer)) or 1,
                "token_count": len(answer.split()),
                "status": "pending_separate_style_review",
            },
            "validation_reasons": [],
            "new_version": None,
        }
        candidates.append(candidate)
        _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
        return copy.deepcopy(candidate)

    def _speaker_matches(self, person_id: str, source: Mapping[str, object]) -> bool:
        person = self._person(person_id)
        profile = self.profile(person_id)
        allowed = {
            str(person["name"]).casefold(),
            *(str(value).casefold() for value in profile.get("aliases", [])),
        }
        return str(source.get("speaker", "")).casefold() in allowed

    def _holdout_score(
        self,
        person_id: str,
        source_ids: Sequence[str],
        holdouts: Sequence[Mapping[str, object]],
        additional_events: Sequence[Mapping[str, object]] = (),
    ) -> float:
        events = [
            *self._trainable_events(person_id, source_ids),
            *(copy.deepcopy(dict(event)) for event in additional_events),
        ]
        holdout_events = [
            dict(event)
            for source in holdouts
            for event in source.get("response_events", [])
            if event.get("label_status")
            == "confirmed_response_weak_semantic_labels"
            and event.get("data_role") == "sealed_final_validation"
        ]
        if not events or not holdout_events:
            return 0.0
        population_events, population_people = self._population_events(person_id)
        artifact = self._predictor.fit(
            person_id=person_id,
            version=0,
            events=events,
            population_events=population_events,
            population_people=population_people,
            scope=self.profile(person_id),
        )
        report = self._predictor.evaluate(artifact, holdout_events)
        if report.get("status") == "not_assessed":
            return 0.0
        return round(
            0.25 * float(report["speech_act_accuracy"])
            + 0.25 * float(report["stance_accuracy"])
            + 0.5 * float(report["mean_claim_support"]),
            6,
        )

    def review_optimization_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        if decision not in {"confirmed", "reference_only", "not_same_question"}:
            raise ConversationError("优化审核结果无效。")
        candidates = self._list(person_id, "optimization_candidates.json")
        try:
            candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
        except StopIteration as error:
            raise ConversationError("优化候选不存在。") from error
        if candidate.get("status") != "pending":
            raise ConversationError("这条候选已经处理。")
        if decision == "reference_only":
            candidate["status"] = "reference_saved"
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        if decision == "not_same_question":
            candidate["status"] = "rejected_not_same_question"
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)

        sources = self._list(person_id, "conversation_sources.json")
        source = next(item for item in sources if item["source_id"] == candidate["source_id"])
        reasons = []
        if source.get("review_status") != "confirmed":
            reasons.append("source_not_confirmed")
        if source.get("content_hash") != candidate.get("source_content_hash"):
            reasons.append("source_content_changed")
        if source.get("dataset_role") == "final_holdout":
            reasons.append("sealed_holdout_cannot_train")
        if not self._speaker_matches(person_id, source):
            reasons.append("speaker_not_confirmed")
        source_time = str(source.get("source_date", "")).strip()
        if not source_time:
            reasons.append("reality_response_time_missing")
        active_ids = self._version_source_ids(person_id)
        if source["source_id"] in active_ids:
            reasons.append("source_already_in_model")
        holdouts = [
            item
            for item in sources
            if item.get("review_status") == "confirmed"
            and item.get("dataset_role") == "final_holdout"
            and item.get("source_id") != source.get("source_id")
            and any(
                event.get("label_status") == "confirmed_response_weak_semantic_labels"
                and event.get("data_role") == "sealed_final_validation"
                for event in item.get("response_events", [])
            )
        ]
        if not holdouts:
            reasons.append("independent_holdout_required")
        else:
            active_training_times = [
                str(item.get("source_date", "")).strip()
                for item in sources
                if item.get("source_id") in active_ids
                and item.get("dataset_role") == "model_source"
            ]
            chronology = [source_time, *active_training_times]
            if any(not value for value in chronology):
                reasons.append("training_time_order_unverified")
            else:
                training_cutoff = max(chronology)
                holdout_times = [
                    str(item.get("source_date", "")).strip()
                    for item in holdouts
                ]
                if any(not value for value in holdout_times):
                    reasons.append("holdout_time_missing")
                elif any(value <= training_cutoff for value in holdout_times):
                    reasons.append("holdout_not_strictly_later_than_training")
        if reasons:
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = reasons
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        selected_event_id = str(candidate.get("source_event_id", ""))
        if not selected_event_id:
            claims = list(dict(candidate.get("content_extraction") or {}).get("claims", []))
            selected_answer = str(claims[0]) if claims else ""
            selected_event_id = next(
                (
                    str(event.get("event_id", ""))
                    for event in source.get("response_events", [])
                    if str(event.get("actual_response", "")) == selected_answer
                ),
                "",
            )
        stored_event = next(
            (
                copy.deepcopy(dict(event))
                for event in source.get("response_events", [])
                if str(event.get("event_id", "")) == selected_event_id
            ),
            None,
        )
        person = self._person(person_id)
        profile = self.profile(person_id)
        recomputed_source = copy.deepcopy(source)
        recomputed_source["response_events"] = response_events_from_source(
            recomputed_source
        )
        recomputed_source["response_events"] = review_response_events(
            recomputed_source,
            str(person["name"]),
            [str(value) for value in profile.get("aliases", [])],
        )
        selected_event = next(
            (
                copy.deepcopy(dict(event))
                for event in recomputed_source["response_events"]
                if str(event.get("event_id", "")) == selected_event_id
            ),
            None,
        )
        selected_answer = str(
            next(
                iter(dict(candidate.get("content_extraction") or {}).get("claims", [])),
                "",
            )
        )
        if selected_event is None:
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = ["selected_reality_event_missing"]
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        if (
            stored_event is None
            or stored_event.get("content_hash") != selected_event.get("content_hash")
            or stored_event.get("actual_response") != selected_event.get("actual_response")
            or selected_answer != selected_event.get("actual_response")
        ):
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = ["selected_event_recompute_mismatch"]
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        allowed_uses = set(
            map(
                str,
                dict(
                    dict(selected_event.get("event_atom") or {}).get(
                        "completeness"
                    )
                    or {}
                ).get("allowed_uses", []),
            )
        )
        if "reality_optimization_training" not in allowed_uses:
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = [
                "selected_event_not_eligible_for_reality_optimization"
            ]
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        selected_event["data_role"] = "parameter_training"
        before = self._holdout_score(person_id, active_ids, holdouts)
        proposed_ids = [*active_ids, str(source["source_id"])]
        after = self._holdout_score(
            person_id, active_ids, holdouts, additional_events=[selected_event]
        )
        if after + 1e-12 < before:
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = ["holdout_regression"]
            candidate["holdout_before"] = before
            candidate["holdout_after"] = after
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        source["role_history"].append(
            {
                "from": source["dataset_role"],
                "to": "model_source",
                "reason": candidate_id,
                "changed_at": _utc_now(),
                "removed_from_independent_evaluation": True,
            }
        )
        source["dataset_role"] = "model_source"
        source["optimization_candidate_id"] = candidate_id
        source["optimization_selected_event_id"] = selected_event_id
        source["response_events"] = recomputed_source["response_events"]
        for event in source.get("response_events", []):
            event["data_role"] = (
                "parameter_training"
                if str(event.get("event_id", "")) == selected_event_id
                else "external_reality_comparison"
            )
        direct_question = str(
            selected_event.get("trigger")
            or selected_event.get("full_context")
            or source.get("source_context")
            or source.get("title")
            or "public response"
        )
        direct_response = str(selected_event.get("actual_response", "")).strip()
        direct_hash = _canonical_hash([direct_question, direct_response])
        if direct_response and direct_hash not in {
            str(item.get("content_hash", ""))
            for item in source.get("reviewed_event_frames_v4", [])
        }:
            source.setdefault("reviewed_event_frames_v4", []).append(
                {
                    "schema_version": REVIEWED_EVENT_SCHEMA_V4,
                    "review_status": "confirmed",
                    "origin": "reality_optimization_direct_evidence",
                    "question": direct_question,
                    "response": direct_response,
                    "source_locator": str(
                        selected_event.get("source_locator")
                        or source.get("source_locator")
                        or "reviewed reality response"
                    ),
                    "speaker_role": "public_speaker",
                    "audience": "unknown",
                    "domain_ids": [],
                    "conditions": [],
                    "reasons": [],
                    "tradeoffs": [],
                    "demonstrated_claim_spans": [],
                    "optimization_candidate_id": candidate_id,
                    "reviewed_at": _utc_now(),
                    "content_hash": direct_hash,
                }
            )
        v5_before = (
            self._simulation_predictor.evaluate(
                self._simulation_model(
                    person_id, int(self._state(person_id)["active_version"])
                ),
                holdouts,
            )
            if self._state(person_id).get("active_version")
            else {
                "status": "not_assessed",
                "reason": "active_simulation_v5_required",
                "sample_count": 0,
                "accuracy_claim": "none",
            }
        )
        try:
            candidate_artifact = self._simulation_predictor.fit(
                person_id=person_id,
                version=0,
                reviewed_sources=[
                    *self._reviewed_sources_for_simulation_v4(person_id, active_ids),
                    self._simulation_source_view(source),
                ],
                scope=self.profile(person_id),
            )
            v5_after = self._simulation_predictor.evaluate(candidate_artifact, holdouts)
        except SimulationV5Error:
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = ["simulation_v5_candidate_recompute_failed"]
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)

        def v5_score(report: Mapping[str, object]) -> float | None:
            accuracy = report.get("covered_direction_accuracy")
            if report.get("status") != "assessed_exploratory" or not isinstance(
                accuracy, (int, float)
            ):
                return None
            return float(report.get("coverage", 0.0)) * float(accuracy)

        before_v5_score = v5_score(v5_before)
        after_v5_score = v5_score(v5_after)
        candidate["simulation_v5_holdout_before"] = v5_before
        candidate["simulation_v5_holdout_after"] = v5_after
        if v5_after.get("status") == "invalid_holdout_leakage" or (
            before_v5_score is not None
            and (after_v5_score is None or after_v5_score + 1e-12 < before_v5_score)
        ):
            candidate["status"] = "failed_validation"
            candidate["validation_reasons"] = ["simulation_v5_holdout_regression"]
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        _write_json(self._path(person_id, "conversation_sources.json"), sources)
        state = self._state(person_id)
        version = self._create_version(
            person_id,
            source_ids=proposed_ids,
            reason=f"optimization candidate {candidate_id}",
            validation_status=(
                "selected_event_integrity_and_v5_holdout_non_regression_passed_exploratory_accuracy"
                if after_v5_score is not None
                else "selected_event_integrity_passed_v5_accuracy_not_assessed"
            ),
            parent_version=state.get("active_version"),
            update_style=False,
        )
        candidate["status"] = "accepted_exploratory"
        candidate["validation_reasons"] = []
        candidate["holdout_before"] = before
        candidate["holdout_after"] = after
        candidate["new_version"] = version["version"]
        candidate["style_update_status"] = "pending_separate_style_review"
        _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
        return copy.deepcopy(candidate)

    def review_optimization_style_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        if decision not in {"confirmed", "rejected"}:
            raise ConversationError("表达样本审核结果无效。")
        candidates = self._list(person_id, "optimization_candidates.json")
        try:
            candidate = next(
                item for item in candidates if item["candidate_id"] == candidate_id
            )
        except StopIteration as error:
            raise ConversationError("优化候选不存在。") from error
        surface = candidate.get("surface_extraction")
        if not isinstance(surface, dict) or surface.get("status") != "pending_separate_style_review":
            raise ConversationError("这条表达样本不在待独立审核状态。")
        if candidate.get("status") != "accepted_exploratory":
            raise ConversationError("内容候选尚未通过，不能审核表达样本。")
        if decision == "rejected":
            surface["status"] = "rejected"
            candidate["style_update_status"] = "rejected_separately"
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        sources = self._list(person_id, "conversation_sources.json")
        source = next(
            item for item in sources if item["source_id"] == candidate["source_id"]
        )
        reasons: list[str] = []
        if source.get("review_status") != "confirmed":
            reasons.append("style_source_not_confirmed")
        if not self._speaker_matches(person_id, source):
            reasons.append("style_speaker_not_confirmed")
        if source.get("dataset_role") == "final_holdout":
            reasons.append("final_holdout_cannot_train_style")
        if source.get("content_hash") != candidate.get("source_content_hash"):
            reasons.append("style_source_content_changed")
        if reasons:
            surface["status"] = "failed_validation"
            surface["validation_reasons"] = reasons
            candidate["style_update_status"] = "failed_validation"
            _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
            return copy.deepcopy(candidate)
        version = self._create_style_only_version(
            person_id, candidate_id=candidate_id
        )
        surface["status"] = "accepted_exploratory"
        surface["style_version"] = version["style_revision"]
        surface["model_version"] = version["version"]
        surface["validation_reasons"] = []
        candidate["style_update_status"] = version["style_update_status"]
        _write_json(self._path(person_id, "optimization_candidates.json"), candidates)
        return copy.deepcopy(candidate)

    def rollback_version(self, person_id: str, version_number: int) -> dict[str, object]:
        versions = self._list(person_id, "conversation_versions.json")
        target = next(
            (item for item in versions if int(item["version"]) == int(version_number)),
            None,
        )
        if target is None:
            raise ConversationError("目标版本不存在。")
        if target.get("validation_status") == "invalidated_evidence_contract":
            raise ConversationError("该版本已因证据契约不合格而失效，不能回滚为当前版本。")
        state = self._state(person_id)
        previous = state.get("active_version")
        state["active_version"] = int(version_number)
        state.setdefault("rollback_history", []).append(
            {"from": previous, "to": int(version_number), "at": _utc_now()}
        )
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return copy.deepcopy(state)

    def feedback(self, person_id: str, message_id: str, value: str) -> dict[str, object]:
        if value not in {"helpful", "not_helpful", "incorrect", "unsafe"}:
            raise ConversationError("反馈类型无效。")
        messages, message = self._find_message(person_id, message_id)
        message["feedback"] = {"value": value, "created_at": _utc_now()}
        self._save_active_messages(person_id, messages)
        return copy.deepcopy(message)
