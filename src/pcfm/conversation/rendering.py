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



class RenderingMixin:
    def _best_support(
        self, question: str, sources: Sequence[Mapping[str, object]]
    ) -> dict[str, object] | None:
        candidates: list[dict[str, object]] = []
        for source in sources:
            for qa in source.get("qas", []):
                score = _similarity(question, str(qa["question"]))
                candidates.append(
                    {
                        "kind": "qa",
                        "score": score,
                        "text": str(qa["answer"]),
                        "matched_question": str(qa["question"]),
                        "source": source,
                        "locator": qa["locator"],
                        "qa_id": qa["qa_id"],
                    }
                )
            for segment in source.get("segments", []):
                score = _similarity(question, str(segment["text"]))
                candidates.append(
                    {
                        "kind": "segment",
                        "score": score,
                        "text": str(segment["text"]),
                        "matched_question": "",
                        "source": source,
                        "locator": segment["locator"],
                        "qa_id": None,
                    }
                )
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                float(item["score"]),
                item["kind"] == "qa",
                str(item["source"]["source_id"]),
                str(item["locator"]),
            ),
            reverse=True,
        )
        return candidates[0]

    def _reality_support_candidates(
        self, question: str, sources: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        values: list[dict[str, object]] = []
        for source in sources:
            for event in source.get("response_events", []):
                if event.get("label_status") != "confirmed_response_weak_semantic_labels":
                    continue
                trigger = str(event.get("question") or event.get("trigger") or source.get("title", ""))
                answer = str(event.get("actual_response", ""))
                if not answer:
                    continue
                score = max(
                    _similarity(question, trigger),
                    0.7 * _similarity(question, " ".join((trigger, answer))),
                )
                if score < 0.45:
                    continue
                values.append(
                    {
                        "comparison_candidate_id": f"reality-{_canonical_hash([source['source_id'], event['event_id']])[:16]}",
                        "score": score,
                        "answer": answer,
                        "question": trigger,
                        "source_id": str(source["source_id"]),
                        "source_title": str(source["title"]),
                        "source_url": str(source.get("source_url", "")),
                        "source_date": str(source.get("source_date", "")),
                        "speaker": str(source.get("speaker", "")),
                        "locator": str(event.get("source_locator", "")),
                        "event_id": str(event["event_id"]),
                        "qa_id": "",
                    }
                )
        values.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["source_id"]),
                str(item["event_id"]),
            )
        )
        return values[:5]

    @staticmethod
    def _protected_numbers(text: str) -> list[str]:
        return sorted(set(re.findall(r"\b\d+(?:[.,]\d+)*(?:-%|%)?|\b\d{4}-\d{2}-\d{2}\b", text)))

    def _render_reply(
        self, person_id: str, contract: Mapping[str, object]
    ) -> tuple[str, str, dict[str, object]]:
        profile = self.profile(person_id)
        neutral = "\n".join(
            str(item["text"])
            for field in ("claims", "reasons", "memories", "uncertainties")
            for item in contract[field]
        )
        configured_renderer = self._renderers.get(str(profile["style_profile_id"]))
        if configured_renderer is not None and not isinstance(
            configured_renderer, ExpressionRenderer
        ):
            try:
                probe = configured_renderer.render(contract)
                selected = dict(probe.get("selected", {}))
            except Exception:
                selected = {"status": "rejected"}
            if selected.get("status") != "passed":
                return neutral, "neutral_fallback", {
                    "status": "failed_returned_neutral",
                    "selected_intensity": "neutral",
                }
        state = self._state(person_id)
        active_version = state.get("active_version")
        active_record = next(
            (
                item
                for item in self._list(person_id, "conversation_versions.json")
                if active_version is not None
                and int(item.get("version", -1)) == int(active_version)
            ),
            None,
        )
        style_path = (
            self._person_dir(person_id) / str(active_record["style_artifact_path"])
            if active_record and active_record.get("style_artifact_path")
            else None
        )
        if style_path is not None and style_path.exists():
            style_artifact = _read_json(style_path, {})
            if isinstance(style_artifact, Mapping):
                try:
                    result = render_person_surface_style(contract, style_artifact)
                except Exception as error:
                    result = {
                        "status": "rejected",
                        "changed": False,
                        "reasons": [f"semantic_gate_error:{type(error).__name__}"],
                        "checks": {},
                        "used_rules": [],
                    }
                common_gate = {
                    "status": result.get("status"),
                    "changed": bool(result.get("changed", False)),
                    "selected_intensity": "observed_surface_only",
                    "style_artifact_hash": style_artifact.get("artifact_hash"),
                    "style_profile_status": result.get("profile_status"),
                    "checks": result.get("checks", {}),
                    "reasons": result.get("reasons", []),
                    "used_rules": result.get("used_rules", []),
                }
                if result.get("status") == "passed" and result.get("changed"):
                    return str(result["text"]), "person_style_applied", common_gate
                if result.get("status") == "neutral":
                    return neutral, "neutral_expression", common_gate
                return neutral, "neutral_fallback", {
                    **common_gate,
                    "status": "failed_returned_neutral",
                    "selected_intensity": "neutral",
                }
        renderer = self._renderers.get(str(profile["style_profile_id"]))
        if renderer is None:
            return neutral, "neutral_no_validated_profile", {
                "status": "not_run_neutral_profile", "selected_intensity": "neutral"
            }
        try:
            result = renderer.render(contract)
        except (ExpressionRendererError, Exception):
            return neutral, "neutral_fallback", {
                "status": "failed_returned_neutral",
                "selected_intensity": "neutral",
            }
        selected = dict(result.get("selected", {}))
        if selected.get("status") != "passed":
            return str(result.get("neutral_text", neutral)), "neutral_fallback", {
                "status": "failed_returned_neutral",
                "selected_intensity": "neutral",
            }
        return str(selected.get("text", neutral)), "styled_semantic_gate_passed", {
            "status": str(result.get("semantic_preservation", {}).get("status", "passed")),
            "selected_intensity": selected.get("intensity", "neutral"),
        }

    def _model_semantic_query_plan(
        self,
        *,
        model_ref: str,
        text: str,
        history: Sequence[Mapping[str, object]],
        artifact: Mapping[str, object],
        conversation_context: Mapping[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Let a model understand context and propose IDs; V5 chooses the person direction."""
        if not model_ref:
            return {}, {
                "status": "not_configured_deterministic_closed_aliases_only",
                "authority": "none",
                "model_calls": 0,
            }
        if self._predictor.is_ordinary_dialogue(text):
            return {}, {
                "status": "ordinary_dialogue_handled_without_semantic_model",
                "authority": "content_free_dialogue_manager",
                "model_calls": 0,
            }
        if self._model_services is None:
            raise ConversationError(
                "Model service manager is unavailable; no silent semantic fallback was used."
            )
        ranked_events = sorted(
            artifact.get("event_frames", []),
            key=lambda frame: (
                -_similarity(
                    text,
                    " ".join(
                        (
                            str(dict(frame["decision_frame"]).get("trigger", "")),
                            str(dict(frame["observed_response"]).get("verbatim", "")),
                        )
                    ),
                ),
                str(frame["event_frame_id"]),
            ),
        )[:12]
        event_candidates = [
            {
                "event_frame_id": frame["event_frame_id"],
                "question": dict(frame["decision_frame"]).get("trigger", ""),
                "occasion": dict(frame.get("episode_context") or {}).get(
                    "occasion", ""
                ),
                "response_excerpt": str(
                    dict(frame["observed_response"]).get("verbatim", "")
                )[:500],
                "domain_ids": frame.get("domain_tags", []),
                "conditions": dict(frame["decision_frame"]).get("conditions", []),
            }
            for frame in ranked_events
        ]
        structure_candidates = ([
            {
                "orientation_id": item["orientation_id"],
                "protected_interest_id": item["protected_interest_id"],
                "accepted_cost_id": item["accepted_cost_id"],
                "domains": item.get("primary_domains", []),
                "conditions": item.get("conditions", []),
                "status": item.get("status", ""),
            }
            for item in artifact.get("orientation_index", [])
        ] + [
            {
                "orientation_id": item["orientation_id"],
                "interest_id": item["interest_id"],
                "domains": item.get("primary_domains", []),
                "status": item.get("status", ""),
            }
            for item in artifact.get("value_orientation_index", [])
        ])[:20]
        semantic_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Produce a grounded semantic routing candidate, not a person stance. "
                            "Return JSON with resolved_message_ids, domain_ids, scenario_effects, "
                            "selected_event_ids, selected_structure_ids, question_scope, and "
                            "target_entity. question_scope is one of narrow (a specific judgment "
                            "equivalent to a historical question), wide (a broad evaluation needing "
                            "multi-dimensional synthesis), or composite (several sub-questions). "
                            "target_entity is the evaluated object copied from the message, or empty. "
                            "A scenario effect has "
                            "an allow-listed interest_id, one effect (advances, constrains, "
                            "threatens, or neutral), and scenario_span copied exactly from the "
                            "current or resolved message. It describes the scenario, not what the person "
                            "prefers. Resolve references only to supplied real message IDs and select "
                            "only supplied event/orientation IDs. Never return a person stance, "
                            "answer, value ranking, biography, or invented fact. Use empty arrays "
                            "when uncertain."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "message": text,
                                "conversation_messages": [
                                    {
                                        "message_id": item.get("message_id"),
                                        "role": item.get("role"),
                                        "text": str(item.get("text", ""))[:600],
                                    }
                                    for item in history[-12:]
                                    if item.get("role") in {"user", "assistant"}
                                ],
                                "conversation_state": copy.deepcopy(
                                    dict(conversation_context)
                                ),
                                "allowed_interests": {
                                    interest_id: definition["label_zh"]
                                    for interest_id, definition in INTERESTS.items()
                                },
                                "allowed_domain_ids": sorted(DOMAIN_ALIASES),
                                "event_candidates": event_candidates,
                                "structure_candidates": structure_candidates,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ]
        compatibility_retry = False
        try:
            service, model_id = self._model_services.resolve_model_ref(model_ref)
            response = self._model_services.invoke(
                str(service["service_id"]),
                model_id,
                semantic_messages,
                structured=True,
                temperature=0.0,
                max_tokens=4096,
            )
        except ModelServiceError as structured_error:
            try:
                response = self._model_services.invoke(
                    str(service["service_id"]),
                    model_id,
                    semantic_messages,
                    structured=False,
                    temperature=0.0,
                    max_tokens=4096,
                )
                compatibility_retry = True
            except ModelServiceError as retry_error:
                raise ConversationError(
                    "Structured semantic routing failed and the same-model JSON "
                    f"compatibility retry also failed: {retry_error}"
                ) from structured_error
        try:
            parsed = _json_mapping(response["text"])
        except json.JSONDecodeError as error:
            raise ConversationError(
                "The selected model did not return a valid semantic query plan; no silent fallback was used."
            ) from error
        return parsed, {
            "status": (
                "candidate_proposed_after_same_model_json_compatibility_retry"
                if compatibility_retry
                else "candidate_proposed_pending_code_validation"
            ),
            "authority": "routing_candidate_only_no_stance_authority",
            "model_calls": 2 if compatibility_retry else 1,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
            "fallback_used": False,
            "same_model_json_compatibility_retry": compatibility_retry,
            "event_candidate_count": len(event_candidates),
            "structure_candidate_count": len(structure_candidates),
        }

    def _style_hints(self, person_id: str) -> list[str]:
        """Read the active style artifact's confirmed surface connectors plus the
        sealed expression profile's surface rules for the selected style.

        Both sources are style-only (openers/connectors/tone markers) — they never
        carry value content, facts, or stance. The render layer uses them to shape
        wording after the prediction layer has fixed the value judgment.
        """
        seen: list[str] = []
        state = self._state(person_id)
        ver = state.get("active_version")
        if ver:
            record = next(
                (
                    v
                    for v in self._list(person_id, "conversation_versions.json")
                    if int(v.get("version", -1)) == int(ver)
                ),
                None,
            )
            if record and record.get("style_artifact_path"):
                style_path = self._person_dir(person_id) / str(record["style_artifact_path"])
                if style_path.exists():
                    artifact = _read_json(style_path, {})
                    if isinstance(artifact, Mapping):
                        for rule in artifact.get("surface_rules", []):
                            prefix = str(rule.get("prefix", "")).strip() if isinstance(rule, Mapping) else ""
                            if prefix and prefix not in seen:
                                seen.append(prefix)
        # 密封表达包（如 steve_jobs_v1）里的表面规则更丰富；只取开场/连接词，不带价值内容。
        profile = self.profile(person_id)
        style_id = str(profile.get("style_profile_id", ""))
        renderer = self._renderers.get(style_id)
        if renderer is not None:
            rules = getattr(renderer, "rules", {})
            if isinstance(rules, Mapping):
                for rule in rules.values():
                    if not isinstance(rule, Mapping):
                        continue
                    prefix = str(rule.get("prefix", "")).strip()
                    if prefix and prefix not in seen:
                        seen.append(prefix)
        return seen
