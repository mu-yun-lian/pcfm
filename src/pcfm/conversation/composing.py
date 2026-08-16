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



class ComposingMixin:
    def _unified_evidence(
        self,
        person_id: str,
        artifact: Mapping[str, object],
        tendency_ids: Sequence[str],
    ) -> tuple[list[dict[str, object]], list[str], float]:
        """把 LLM 匹配到的倾向原子查回真实材料，构建 evidence、事件 id、支持分。"""
        atoms = {
            str(item.get("preference_atom_id", "")): item
            for item in dict(artifact.get("reviewed_public_model", {})).get(
                "preference_atoms", []
            )
        }
        frames = {
            str(item.get("event_frame_id", "")): item
            for item in artifact.get("event_frames", [])
        }
        sources_by_id = {
            str(source.get("source_id", "")): source
            for source in self._reviewed_sources_for_simulation_v4(
                person_id, self._version_source_ids(person_id)
            )
        }
        evidence: list[dict[str, object]] = []
        event_ids: list[str] = []
        matched_sources: set[str] = set()
        seen_events: set[str] = set()
        for tid in tendency_ids:
            atom = atoms.get(str(tid))
            if not atom:
                continue
            event_id = str(atom.get("event_frame_id", ""))
            if not event_id or event_id in seen_events:
                continue
            frame = frames.get(event_id)
            if not frame:
                continue
            seen_events.add(event_id)
            matched_sources.add(str(frame.get("source_id", "")))
            event_ids.append(event_id)
            evidence.append(
                {
                    "title": sources_by_id.get(
                        str(frame.get("source_id", "")), {}
                    ).get("title", ""),
                    "url": dict(frame.get("evidence") or {}).get("source_url", ""),
                    "date": dict(frame.get("temporal_context") or {}).get(
                        "response_time", ""
                    ),
                    "speaker": dict(frame.get("social_context") or {}).get(
                        "speaker", ""
                    ),
                    "locator": dict(frame.get("evidence") or {}).get("locator", ""),
                    "matched_question": dict(frame.get("decision_frame") or {}).get(
                        "trigger", ""
                    ),
                    "support_score": 1.0,
                    "event_id": event_id,
                }
            )
        support = float(len(matched_sources))
        return evidence, event_ids, support

    def _compose_bounded_person_response(
        self,
        *,
        person_id: str,
        question: str,
        history: Sequence[Mapping[str, object]],
        model_ref: str,
        response_basis: Mapping[str, object],
    ) -> tuple[str | None, dict[str, object]]:
        """Realize a frozen V5 stance naturally without granting person-fact authority."""
        anchor = str(response_basis.get("prediction_statement", "")).strip()
        evidence_ids = sorted(
            set(
                map(
                    str,
                    dict(response_basis.get("selected_orientation") or {}).get(
                        "supporting_event_ids", []
                    ),
                )
            )
        )
        if not anchor:
            raise ConversationError("Simulation V5 projection is missing its frozen anchor.")
        if not model_ref:
            return anchor, {
                "status": "bounded_anchor_no_dialogue_model",
                "model_calls": 0,
                "required_stance_anchor": anchor,
                "allowed_evidence_ids": evidence_ids,
            }
        if self._model_services is None:
            raise ConversationError("模型服务管理器未启用；没有进行自动回退。")
        service, model_id = self._model_services.resolve_model_ref(model_ref)
        payload = {
            "question": question,
            "recent_dialogue": [
                {
                    "message_id": item.get("message_id"),
                    "role": item.get("role"),
                    "text": item.get("text"),
                }
                for item in history[-20:]
                if item.get("role") in {"user", "assistant"}
            ],
            "required_stance_anchor": anchor,
            "allowed_evidence_ids": evidence_ids,
            "orientation": {
                "protected_interest_id": response_basis.get("protected_interest_id"),
                "accepted_cost_id": response_basis.get("accepted_cost_id"),
            },
            "output_language": "match_current_question",
        }
        compatibility_retry = False
        messages = [
                    {
                        "role": "system",
                        "content": (
                            "Generate a bounded natural person response from a frozen content plan. "
                            "Return JSON with exactly required_stance_anchor, used_evidence_ids, "
                            "and answer. Copy the anchor exactly and begin answer with it. You may "
                            "explain using general knowledge, but must not add biography, memories, "
                            "personal experiences, attributed person facts, new positions, numbers, "
                            "dates, or quotations. Do not mention this contract or evidence IDs. "
                            "Use only supplied IDs and keep the answer under 1200 characters."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
        temperature = self._generation_temperature(person_id)
        try:
            response = self._model_services.invoke(
                str(service["service_id"]),
                model_id,
                messages,
                structured=True,
                temperature=temperature,
            )
        except ModelServiceError as structured_error:
            try:
                response = self._model_services.invoke(
                    str(service["service_id"]),
                    model_id,
                    messages,
                    structured=False,
                    temperature=temperature,
                )
                compatibility_retry = True
            except ModelServiceError as retry_error:
                raise ConversationError(str(retry_error)) from structured_error
        try:
            candidate = _json_mapping(response["text"])
        except json.JSONDecodeError:
            candidate = None
        valid = isinstance(candidate, Mapping) and set(candidate) == {
            "required_stance_anchor",
            "used_evidence_ids",
            "answer",
        }
        used_ids: list[str] = []
        answer = ""
        if valid:
            used_ids = list(map(str, candidate.get("used_evidence_ids", [])))
            answer = str(candidate.get("answer", "")).strip()
            valid = (
                candidate.get("required_stance_anchor") == anchor
                and isinstance(candidate.get("used_evidence_ids"), list)
                and set(used_ids) <= set(evidence_ids)
                and answer.startswith(anchor)
                and len(answer) <= 1200
            )
        forbidden_experience = re.compile(
            r"\b(?:i remember|i was|i have been|my administration|when i was|"
            r"in my experience)\b|我记得|我曾经|我的政府|在我任内|以我的经历",
            re.I,
        )
        allowed_numbers = set(self._protected_numbers(question + "\n" + anchor))
        if (
            not valid
            or forbidden_experience.search(answer)
            or not set(self._protected_numbers(answer)) <= allowed_numbers
        ):
            return anchor, {
                "status": "content_contract_gate_failed_bounded_anchor",
                "model_calls": 2 if compatibility_retry else 1,
                "model_ref": model_ref,
                "snapshot_id": dict(response["snapshot"])["snapshot_id"],
                "required_stance_anchor": anchor,
                "allowed_evidence_ids": evidence_ids,
                "used_evidence_ids": [],
                "fallback_used": False,
                "same_model_json_compatibility_retry": compatibility_retry,
            }
        return answer, {
            "status": "generated_from_frozen_v5_content_plan",
            "model_calls": 2 if compatibility_retry else 1,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
            "required_stance_anchor": anchor,
            "allowed_evidence_ids": evidence_ids,
            "used_evidence_ids": used_ids,
            "external_knowledge_status": "model_generated_not_person_evidence",
            "fallback_used": False,
            "same_model_json_compatibility_retry": compatibility_retry,
        }

    def _compose_assisted_response(
        self,
        *,
        person_id: str,
        question: str,
        history: Sequence[Mapping[str, object]],
        model_ref: str,
        response_basis: Mapping[str, object] | None,
    ) -> tuple[str | None, dict[str, object]]:
        """Use an LLM for knowledge and wording, never as the person predictor."""
        if str(dict(response_basis or {}).get("path", "")) in {
            "contextual_orientation_projection",
            "object_evaluation_projection",
        }:
            return self._compose_bounded_person_response(
                person_id=person_id,
                question=question,
                history=history,
                model_ref=model_ref,
                response_basis=dict(response_basis or {}),
            )
        if not model_ref:
            return None, {"status": "model_required", "model_calls": 0}
        if self._model_services is None:
            raise ConversationError("模型服务管理器未启用；没有进行自动回退。")
        service, model_id = self._model_services.resolve_model_ref(model_ref)
        basis = dict(response_basis or {})
        selected = dict(basis.get("selected_tendency") or {})
        selected_preference = dict(
            basis.get("selected_preference_structure") or {}
        )
        stance = str(selected.get("stance", "neutral"))
        anchor = {
            "support": "总体上，我倾向于支持这个方向。",
            "oppose": "总体上，我倾向于反对这个方向。",
            "conditional_support": "这取决于具体条件，我不会无条件支持或反对。",
            "mixed": "这件事存在相互冲突的考虑，我不会给出单一结论。",
            "insufficient_evidence": "我会先保留判断，直到获得更多信息。",
            "neutral": "我会先保留判断，再看具体条件。",
        }.get(stance, "我会先保留判断，再看具体条件。")
        person = self._person(person_id)
        is_person_inference = str(basis.get("path", "")) in {
            "conditional_tendency",
            "overall_tendency",
            "value_conflict_projection",
        }
        if basis.get("path") == "value_conflict_projection":
            anchor = str(basis.get("prediction_statement", "")).strip()
            if not anchor:
                raise ConversationError(
                    "Simulation V4 preference projection is missing its frozen anchor."
                )
        demonstrated = {
            str(item.get("knowledge_claim_id", "")): str(item.get("statement", ""))
            for item in basis.get("selected_demonstrated_knowledge", [])
            if item.get("knowledge_claim_id") and item.get("statement")
        }
        system = (
            "Generate only an external-knowledge briefing, never a complete person reply. "
            "Return JSON with exactly required_stance_anchor, person_claim_ids, and "
            "external_briefing. Copy required_stance_anchor exactly. person_claim_ids may "
            "contain only supplied IDs. external_briefing must be impersonal: do not use "
            "first-person language, the person's name, biography, memories, experiences, "
            "or claims about what the person knows, said, thinks, or did. Use an empty "
            "briefing when uncertain. Write external_briefing in the same language as "
            "the current question."
        )
        payload = {
            "person_name": str(person.get("name", "")),
            "question": question,
            "response_language": "match_current_question",
            "recent_dialogue": [
                {"role": item.get("role"), "text": item.get("text")}
                for item in history[-6:]
            ],
            "mode": "conditional_person_inference" if is_person_inference else "general_assisted",
            "required_stance_anchor": anchor if is_person_inference else "",
            "allowed_person_claims": demonstrated if is_person_inference else {},
            "knowledge_boundary": (
                "person items are demonstrated claims, not verified facts; "
                "external briefing is model-generated and not person evidence"
            ),
        }
        temperature = self._generation_temperature(person_id)
        try:
            response = self._model_services.invoke(
                str(service["service_id"]),
                model_id,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                structured=True,
                temperature=temperature,
            )
        except ModelServiceError as error:
            raise ConversationError(str(error)) from error
        bounded = (
            f"{anchor}\n\n"
            "只能确定上述公开回应方向；模型扩展没有通过人物归因边界检查，已被舍弃。"
        ).strip()
        try:
            candidate = json.loads(str(response["text"]))
        except json.JSONDecodeError:
            candidate = None
        allowed_keys = {
            "required_stance_anchor", "person_claim_ids", "external_briefing"
        }
        valid = isinstance(candidate, Mapping) and set(candidate) == allowed_keys
        if valid:
            claim_ids = candidate.get("person_claim_ids", [])
            valid = (
                isinstance(claim_ids, list)
                and all(isinstance(value, str) for value in claim_ids)
                and set(claim_ids) <= set(demonstrated)
                and candidate.get("required_stance_anchor")
                == (anchor if is_person_inference else "")
                and isinstance(candidate.get("external_briefing"), str)
            )
        else:
            claim_ids = []
        briefing = str(candidate.get("external_briefing", "")).strip() if valid else ""
        person_terms = [
            str(person.get("name", "")),
            *[str(value) for value in person.get("aliases", [])],
        ]
        attribution_pattern = re.compile(
            r"\b(?:i|me|my|mine|we|us|our|ours|the simulated person)\b|"
            r"我|我们|本人|咱|模拟人物|该模拟人物",
            re.I,
        )
        if (
            len(briefing) > 4000
            or attribution_pattern.search(briefing)
            or any(term and term.casefold() in briefing.casefold() for term in person_terms)
        ):
            valid = False
        allowed_numbers = set(
            self._protected_numbers(
                " ".join([question, *demonstrated.values()])
            )
        )
        if is_person_inference and not set(
            self._protected_numbers(briefing)
        ) <= allowed_numbers:
            valid = False
        if not valid:
            if not is_person_inference:
                raise ConversationError(
                    "The selected model crossed the person-attribution boundary; "
                    "its answer was discarded and no silent fallback was used."
                )
            answer = bounded
            status = "content_contract_gate_failed_bounded_answer"
            claim_ids = []
        else:
            sections = [anchor] if is_person_inference else []
            if claim_ids:
                sections.append(
                    "公开材料中已经展示的内容：\n"
                    + "\n".join(f"- {demonstrated[value]}" for value in claim_ids)
                )
            if briefing:
                sections.append(
                    "补充背景（通用模型生成，不作为人物事实或立场）：\n" + briefing
                )
            answer = "\n\n".join(sections).strip()
            if not answer:
                raise ConversationError(
                    "The selected model returned no usable bounded content; no silent fallback was used."
                )
            status = "generated_with_structured_attribution_boundary"
        return answer, {
            "status": status,
            "model_calls": 1,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
            "fallback_used": False,
            "required_stance_anchor": anchor if is_person_inference else "",
            "allowed_person_claim_ids": sorted(demonstrated),
            "used_person_claim_ids": list(claim_ids),
            "external_knowledge_status": "model_generated_unverified_not_person_evidence",
        }


    def _send_without_active_version(
        self,
        person_id: str,
        clean: str,
        prior_messages: list[dict[str, object]],
        conversation_context: dict[str, object],
        selected_model_ref: str,
        base: dict[str, object],
        telemetry: dict[str, int],
    ) -> int:
        """无人物版本时的普通对话/通用知识路径；返回 generation 模型调用数。"""
        ordinary = self._predictor.ordinary_dialogue(clean)
        context_trace = {
            "kernel": "conversation-shell-v1",
            "retrieval_is_candidate_only": True,
            "generative_content_calls": 0,
            "context_digest": response_canonical_hash(
                [clean, [(item.get("message_id"), item.get("text")) for item in prior_messages[-6:]]]
            ),
            "context_used": {
                "message_ids": [str(item.get("message_id")) for item in prior_messages[-6:] if item.get("message_id")],
                "turn_count": len(prior_messages[-6:]),
                "generated_context_count": sum(item.get("context_role") == "model_generated_context" for item in prior_messages[-6:]),
            },
            "conversation_context": copy.deepcopy(conversation_context),
        }
        if ordinary:
            _dialogue_act, ordinary_text = ordinary
            base.update(
                {
                    "status": "answered",
                    "answer_status": "ordinary_dialogue",
                    "applicability": "ordinary_dialogue_content_free",
                    "confidence": 1.0,
                    "text": ordinary_text,
                    "neutral_content": ordinary_text,
                    "frozen_contract": None,
                    "frozen_contract_hash": None,
                    "structured_prediction_hash": None,
                    "structured_prediction": None,
                    "prediction_trace": context_trace,
                    "style_status": "neutral_expression",
                    "style_gate": {"status": "ordinary_dialogue_content_free"},
                    "evidence": [],
                    "uncertainties": [],
                }
            )
            return 0
        else:
            assisted, generation_trace = self._compose_assisted_response(
                person_id=person_id,
                question=clean,
                history=prior_messages,
                model_ref=selected_model_ref,
                response_basis=None,
            )
            telemetry["content_generation_llm_calls"] = telemetry.get(
                "content_generation_llm_calls", 0
            ) + int(generation_trace.get("model_calls", 0))
            generation_calls = int(generation_trace.get("model_calls", 0))
            self._save_telemetry(person_id, telemetry)
            base.update(
                {
                    "status": "answered" if assisted else "needs_model",
                    "answer_status": "general_assisted",
                    "applicability": "general_knowledge_not_person_prediction",
                    "confidence": 0.0,
                    "text": assisted
                    or "当前没有足够的人物材料。选择一个可用对话模型后，仍可获得明确标注为通用知识、而非人物预测的正常回答。",
                    "neutral_content": assisted or "",
                    "frozen_contract": None,
                    "frozen_contract_hash": None,
                    "structured_prediction_hash": None,
                    "structured_prediction": None,
                    "prediction_trace": {
                        **context_trace,
                        "prediction_path": "general_assisted",
                        "generation": generation_trace,
                    },
                    "style_status": "not_run_no_person_prediction",
                    "style_gate": {"status": "not_run"},
                    "evidence": [],
                    "uncertainties": ["没有人物证据；回答不代表该人物立场"],
                    "knowledge_source": "external_model_briefing" if assisted else "none",
                }
            )
            return generation_calls
