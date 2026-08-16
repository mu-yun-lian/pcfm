from __future__ import annotations

from ..jobs import JobCancelled

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



class MessagePipelineMixin:
    def send_message(
        self,
        person_id: str,
        text: str,
        *,
        reality_lookup_requested: bool = False,
        dialogue_model_ref: str = "",
        cancel_event: object = None,
        progress: object = None,
    ) -> dict[str, object]:
        self.profile(person_id)

        def _check_cancel() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled()

        clean = str(text).strip()
        if not clean:
            raise ConversationError("请输入消息。")
        _check_cancel()
        messages = self._active_messages(person_id)
        prior_messages = copy.deepcopy(messages)
        profile = self.profile(person_id)
        conversation_context = self._conversation_context(
            profile, prior_messages, clean
        )
        conversation_context["dynamic_state"] = {
            "status": "inactive",
            "reason": "no_compatible_verified_temporal_response_outcomes",
            "model_generated_dialogue_used_for_update": False,
        }
        user_message = {
            "schema_version": SCHEMA_VERSION,
            "message_id": f"message-{uuid.uuid4().hex[:12]}",
            "person_id": person_id,
            "role": "user",
            "text": clean,
            "context_role": "user_input_context",
            "created_at": _utc_now(),
        }
        messages.append(user_message)
        # 尽早持久化用户消息：即使生成被取消，用户消息也保留
        self._save_active_messages(person_id, messages)
        telemetry = self._telemetry(person_id)
        telemetry["content_retrieval_calls"] += 1
        telemetry["content_prediction_calls"] += 1
        telemetry.setdefault("content_generation_llm_calls", 0)
        self._save_telemetry(person_id, telemetry)
        state = self._state(person_id)
        if dialogue_model_ref:
            self.select_dialogue_model(person_id, dialogue_model_ref)
            state = self._state(person_id)
        selected_model_ref = str(
            dialogue_model_ref
            or state.get("dialogue_model_ref", "")
            or (
                self._model_services.roles().get("default_dialogue", "")
                if self._model_services is not None
                else ""
            )
        )
        model_snapshot = None
        if selected_model_ref and self._model_services is not None:
            try:
                model_snapshot = self._model_services.snapshot(selected_model_ref)
            except ModelServiceError as error:
                raise ConversationError(str(error)) from error
        active_version = state.get("active_version")
        base = {
            "schema_version": SCHEMA_VERSION,
            "message_id": f"message-{uuid.uuid4().hex[:12]}",
            "person_id": person_id,
            "role": "assistant",
            "context_role": "model_generated_context",
            "created_at": _utc_now(),
            "model_version": active_version,
            "model_kind": "pcfm_conversation_conditioned_response_simulation_v5",
            "dialogue_model_ref": selected_model_ref,
            "dialogue_model_snapshot_id": (
                model_snapshot.get("snapshot_id") if model_snapshot else None
            ),
            "dialogue_model_provider": (
                model_snapshot.get("provider") if model_snapshot else None
            ),
            "dialogue_model_id": (
                model_snapshot.get("model_id") if model_snapshot else None
            ),
            "model_fallback_used": False,
            "response_accuracy_status": "not_assessed",
            "person_prediction_status": "not_available",
            "knowledge_source": "none",
            "reality_lookup_requested": bool(reality_lookup_requested),
            "reality_lookup_status": "scheduled" if reality_lookup_requested else "not_requested",
            "comparison": None,
            "feedback": None,
        }
        planning_calls = 0
        generation_calls = 0
        validation_calls = 0
        if active_version is None:
            generation_calls = self._send_without_active_version(
                person_id, clean, prior_messages, conversation_context,
                selected_model_ref, base, telemetry,
            )

        else:
            artifact = self._simulation_model(person_id, int(active_version))
            query_plan: dict[str, object] = {}
            planner_trace = {
                "status": "not_needed_deterministic_person_route_succeeded",
                "authority": "none",
                "model_calls": 0,
            }
            predicted = self._simulation_predictor.predict(
                artifact,
                text=clean,
                history=prior_messages,
                conversation_context=conversation_context,
                query_plan=query_plan,
            )
            # 推导类问题：一次 LLM 统一路径，内部完成 理解→匹配→推导→组织；代码只守门
            derivation_statuses = {
                "general_assisted",
                "object_evaluation_projection_answer",
                "orientation_projection_answer",
                "tendency_answer",
                "preference_structure_answer",
                "refused",
                "clarification_needed",
            }
            if selected_model_ref and predicted["answer_status"] in derivation_statuses:
                unified = self._unified_person_response(
                    person_id=person_id,
                    text=clean,
                    history=prior_messages,
                    conversation_context=conversation_context,
                    model_ref=selected_model_ref,
                    artifact=artifact,
                )
                gate_ok, gate_reason = self._gate_unified_response(unified, artifact)
                structured = dict(predicted["structured_prediction"])
                render_trace: dict[str, object] = {"status": "not_run", "model_calls": 0}
                if unified["status"] == "ok" and gate_ok:
                    plain_answer = unified["answer"]
                    stance = unified["stance"]
                    tendency_ids = [str(v) for v in unified.get("tendency_ids", [])]
                    evidence, evidence_event_ids, support = self._unified_evidence(
                        person_id, artifact, tendency_ids
                    )
                    # 渲染层独立：只改措辞/语气/语言，立场与内容已在推导层锁定
                    answer_text, render_trace = self._render_person_answer(
                        person_id=person_id,
                        content=plain_answer,
                        model_ref=selected_model_ref,
                        is_chinese=bool(re.search(r"[\u4e00-\u9fff]", clean)),
                    )
                    person_prediction_status = (
                        "stance_atom_derived_depth_external"
                        if tendency_ids
                        else "not_available"
                    )
                    base.update(
                        {
                            "status": "answered",
                            "answer_status": predicted["answer_status"],
                            "question_type": unified["question_type"],
                            "applicability": "unified_person_response",
                            "confidence": support,
                            "text": answer_text,
                            "neutral_content": plain_answer,
                            "frozen_contract": None,
                            "frozen_contract_hash": None,
                            "structured_prediction_hash": None,
                            "structured_prediction": {
                                "schema_version": "pcfm-unified-person-response-v1",
                                "person_id": person_id,
                                "speech_act": {"label": "direct_answer", "probability": 0.0},
                                "stance": {"label": stance, "probability": 0.0},
                                "claims": [{"id": "claim-unified-" + _text_hash(answer_text)[:16], "text": answer_text}],
                                "reasons": [],
                                "memories": [],
                                "uncertainties": [],
                                "answer_status": predicted["answer_status"],
                                "confidence": support,
                                "applicability": "unified_person_response",
                                "refusal_reasons": [],
                                "evidence_refs": [],
                                "evidence_event_ids": evidence_event_ids,
                                "response_basis": {
                                    "path": "unified_person_response",
                                    "person_prediction_status": person_prediction_status,
                                    "question_type": unified["question_type"],
                                    "tendency_ids": tendency_ids,
                                },
                            },
                            "prediction_trace": {
                                "kernel": "simulation-v5",
                                "prediction_path": "unified_person_response",
                                "generation": {
                                    "status": "unified_person_response",
                                    "model_calls": int(unified.get("model_calls", 0)),
                                },
                            },
                            "style_status": str(render_trace.get("status", "")),
                            "style_gate": {
                                "status": str(render_trace.get("status", "")),
                                "changed": answer_text != plain_answer,
                            },
                            "evidence": evidence,
                            "uncertainties": [],
                            "knowledge_source": "atom_stance_plus_external_depth",
                            "external_depth": bool(tendency_ids),
                            "person_prediction_status": person_prediction_status,
                        }
                    )
                else:
                    base.update(
                        {
                            "status": "answered",
                            "answer_status": predicted["answer_status"],
                            "applicability": "unified_gate_failed",
                            "confidence": 0.0,
                            "text": "我会先保留判断。",
                            "neutral_content": "我会先保留判断。",
                            "prediction_trace": {"kernel": "simulation-v5", "prediction_path": "unified_gate_failed"},
                            "style_status": "unified_gate_failed",
                            "style_gate": {"status": "unified_gate_failed", "reason": gate_reason},
                            "evidence": [],
                            "uncertainties": [],
                        }
                    )
                generation_calls = int(unified.get("model_calls", 0)) + int(
                    render_trace.get("model_calls", 0)
                )
                base["model_usage"] = {
                    "selected_model_ref": selected_model_ref,
                    "planning_calls": 0,
                    "generation_calls": generation_calls,
                    "validation_calls": 0,
                    "total_calls": generation_calls,
                    "status": "used" if generation_calls else "not_selected",
                    "fallback_used": False,
                }
                _check_cancel()
                messages.append(base)
                self._save_active_messages(person_id, messages, dialogue_state=self._conversation_context(profile, messages, ""))
                return copy.deepcopy(base)
            needs_semantic_help = predicted["answer_status"] in {
                "general_assisted",
                "clarification_needed",
            }
            refusal_reasons = set(
                map(
                    str,
                    dict(predicted.get("structured_prediction") or {}).get(
                        "refusal_reasons", []
                    ),
                )
            )
            needs_semantic_help = needs_semantic_help or (
                predicted["answer_status"] == "refused"
                and bool(
                    refusal_reasons
                    & {
                        "person_opinion_evidence_required",
                        "local_support_gap",
                        "out_of_domain",
                    }
                )
            )
            if selected_model_ref and needs_semantic_help:
                try:
                    query_plan, planner_trace = self._model_semantic_query_plan(
                        model_ref=selected_model_ref,
                        text=clean,
                        history=prior_messages,
                        artifact=artifact,
                        conversation_context=conversation_context,
                    )
                except ConversationError as error:
                    planner_trace = {
                        "status": "failed_disclosed_deterministic_semantics_only",
                        "authority": "none",
                        "model_calls": 0,
                        "error": str(error),
                    }
                predicted = self._simulation_predictor.predict(
                    artifact,
                    text=clean,
                    history=prior_messages,
                    conversation_context=conversation_context,
                    query_plan=query_plan,
                )
            planning_calls = int(planner_trace.get("model_calls", 0))
            telemetry["content_planning_llm_calls"] = telemetry.get(
                "content_planning_llm_calls", 0
            ) + planning_calls
            self._save_telemetry(person_id, telemetry)
            predicted["prediction_trace"]["semantic_query_plan"] = planner_trace
            structured = dict(predicted["structured_prediction"])
            contract = dict(predicted["renderer_contract"])
            digest_probe = dict(structured)
            declared_structured_digest = str(digest_probe.pop("content_digest", ""))
            if response_canonical_hash(digest_probe) != declared_structured_digest:
                raise ConversationError("人物结构化预测完整性检查失败。")
            if response_canonical_hash(contract) != str(
                predicted["renderer_contract_digest"]
            ):
                raise ConversationError("冻结表达合同完整性检查失败。")
            base["answer_status"] = predicted["answer_status"]
            basis = dict(structured.get("response_basis") or {})
            if basis.get("person_prediction_status"):
                base["person_prediction_status"] = basis["person_prediction_status"]
            elif predicted["answer_status"] == "direct_answer":
                base["person_prediction_status"] = "direct_historical_evidence"
            elif predicted["answer_status"] in {"composite_answer", "partial_answer"}:
                base["person_prediction_status"] = "similar_event_inference"
            if predicted["answer_status"] in {"refused", "clarification_needed"}:
                reasons = [str(value) for value in structured["refusal_reasons"]]
                if predicted["answer_status"] == "clarification_needed":
                    refusal_text = "我还不能确定你指的是前面的哪件事。请再说明一下对象或问题。"
                elif "out_of_domain" in reasons:
                    refusal_text = "这个问题超出当前人物模型已经覆盖的领域，我不能用通用模型替人物补充观点。"
                elif "local_support_gap" in reasons:
                    refusal_text = "当前人物资料与这个问题的局部证据不足，我只能停止在这里，不能拼出一个看似确定的回答。"
                elif "person_opinion_evidence_required" in reasons:
                    refusal_text = (
                        "这个问题要求的是当前人物的评价，但现有人物事件和公开取向"
                        "不足以形成该评价；系统不会用通用百科内容替代人物观点。"
                    )
                else:
                    refusal_text = "当前证据不足以形成可靠回答。"
                base.update(
                    {
                        "status": "clarification" if predicted["answer_status"] == "clarification_needed" else "refused",
                        "applicability": reasons[0] if reasons else "prediction_refused",
                        "confidence": 0.0,
                        "text": refusal_text,
                        "neutral_content": "",
                        "frozen_contract": None,
                        "frozen_contract_hash": predicted["renderer_contract_digest"],
                        "structured_prediction_hash": predicted["content_digest"],
                        "structured_prediction": structured,
                        "prediction_trace": predicted["prediction_trace"],
                        "style_status": "not_run_refused",
                        "style_gate": {"status": "not_run"},
                        "evidence": [],
                        "uncertainties": reasons,
                    }
                )
            else:
                generated_neutral: str | None = None
                if predicted["answer_status"] == "ordinary_dialogue":
                    rendered = str(contract.get("ordinary_dialogue_text", ""))
                    style_status = "neutral_expression"
                    style_gate = {
                        "status": "ordinary_dialogue_content_free",
                        "changed": False,
                    }
                    render_calls = 0
                elif predicted["answer_status"] == "general_assisted":
                    rendered, generation_trace = self._compose_assisted_response(
                        person_id=person_id,
                        question=clean,
                        history=prior_messages,
                        model_ref=selected_model_ref,
                        response_basis=None,
                    )
                    rendered = rendered or (
                        "当前没有已验证并选用的对话模型，因此无法生成通用知识回答。"
                        "请在输入框的模型菜单中选择‘验证并使用’；系统不会把通用模型内容"
                        "冒充为人物预测。"
                    )
                    generated_neutral = rendered
                    style_status = "not_run_no_person_prediction"
                    style_gate = {"status": "not_run_no_person_prediction"}
                    render_calls = int(generation_trace.get("model_calls", 0))
                    validation_calls = 0
                    base["status"] = "answered" if render_calls else "needs_model"
                    base["knowledge_source"] = (
                        "external_model_briefing" if render_calls else "none"
                    )
                elif predicted["answer_status"] in {
                    "tendency_answer",
                    "preference_structure_answer",
                    "orientation_projection_answer",
                    "object_evaluation_projection_answer",
                }:
                    # 一次 LLM 调用内部完成 理解→匹配→推导→组织；代码只守门，不参与推导
                    unified = self._unified_person_response(
                        person_id=person_id,
                        text=clean,
                        history=prior_messages,
                        conversation_context=conversation_context,
                        model_ref=selected_model_ref,
                        artifact=artifact,
                    )
                    gate_ok, gate_reason = self._gate_unified_response(unified, artifact)
                    generation_trace = {
                        "status": (
                            "unified_person_response" if gate_ok else "unified_gate_failed"
                        ),
                        "model_calls": int(unified.get("model_calls", 0)),
                        "gate_reason": gate_reason,
                    }
                    render_trace: dict[str, object] = {"status": "not_run", "model_calls": 0}
                    if unified["status"] == "ok" and gate_ok:
                        plain = unified["answer"]
                        rendered, render_trace = self._render_person_answer(
                            person_id=person_id,
                            content=plain,
                            model_ref=selected_model_ref,
                            is_chinese=bool(re.search(r"[\u4e00-\u9fff]", clean)),
                        )
                        style_status = str(render_trace.get("status", ""))
                        style_gate = {
                            "status": str(render_trace.get("status", "")),
                            "changed": rendered != plain,
                        }
                        base["knowledge_source"] = "atom_stance_plus_external_depth"
                        base["external_depth"] = bool(unified.get("tendency_ids"))
                        base["person_prediction_status"] = (
                            "stance_atom_derived_depth_external"
                            if unified.get("tendency_ids")
                            else "not_available"
                        )
                        structured["stance"] = {
                            "label": unified["stance"],
                            "probability": 0.0,
                        }
                        structured["response_basis"] = {
                            "path": "unified_person_response",
                            "person_prediction_status": base["person_prediction_status"],
                            "question_type": unified["question_type"],
                            "tendency_ids": unified["tendency_ids"],
                        }
                    else:
                        rendered = str(basis.get("prediction_statement", "")).strip() or "我会先保留判断。"
                        style_status = "unified_gate_failed"
                        style_gate = {
                            "status": "unified_gate_failed",
                            "reason": gate_reason,
                        }
                        base["knowledge_source"] = "none"
                    generated_neutral = rendered
                    render_calls = int(unified.get("model_calls", 0)) + int(
                        render_trace.get("model_calls", 0)
                    )
                    validation_calls = 0
                else:
                    # Direct and similar-event answers already contain frozen,
                    # evidence-backed wording and therefore already exhibit the
                    # person's observed surface. Preserve it byte-for-byte rather
                    # than adding another inferred style marker.
                    rendered = "\n".join(
                        str(item["text"])
                        for field in ("claims", "reasons", "memories", "uncertainties")
                        for item in contract[field]
                    )
                    style_status = "source_verbatim_person_style"
                    style_gate = {
                        "status": "passed_source_verbatim",
                        "changed": False,
                    }
                    render_calls = 0
                    validation_calls = 0
                if predicted["answer_status"] == "ordinary_dialogue":
                    validation_calls = 0
                generation_calls = render_calls
                telemetry["content_generation_llm_calls"] = telemetry.get(
                    "content_generation_llm_calls", 0
                ) + render_calls
                telemetry["validation_llm_calls"] = telemetry.get(
                    "validation_llm_calls", 0
                ) + validation_calls
                self._save_telemetry(person_id, telemetry)
                neutral = "\n".join(
                    str(item["text"])
                    for field in ("claims", "reasons", "memories", "uncertainties")
                    for item in contract[field]
                )
                if predicted["answer_status"] in {
                    "tendency_answer",
                    "preference_structure_answer",
                    "orientation_projection_answer",
                    "general_assisted",
                }:
                    neutral = generated_neutral or rendered
                    predicted["prediction_trace"]["generation"] = generation_trace
                evidence_by_event: dict[str, dict[str, object]] = {}
                sources_by_id = {
                    str(source["source_id"]): source
                    for source in self._reviewed_sources_for_simulation_v4(
                        person_id, self._version_source_ids(person_id)
                    )
                }
                for frame in artifact.get("event_frames", []):
                    event_id = str(frame["event_frame_id"])
                    source = sources_by_id.get(str(frame["source_id"]), {})
                    evidence_by_event[event_id] = {
                        "source_id": frame["source_id"],
                        "title": source.get("title", ""),
                        "url": dict(frame["evidence"]).get("source_url", ""),
                        "date": dict(frame["temporal_context"]).get(
                            "response_time", ""
                        ),
                        "speaker": dict(frame["social_context"]).get("speaker", ""),
                        "locator": dict(frame["evidence"]).get("locator", ""),
                        "matched_question": dict(frame["decision_frame"]).get(
                            "trigger", ""
                        ),
                        "support_score": structured.get("confidence", 0.0),
                        "event_id": event_id,
                    }
                evidence = [
                    evidence_by_event[event_id]
                    for event_id in structured.get("evidence_event_ids", [])
                    if event_id in evidence_by_event
                ]
                base.update(
                    {
                        "status": (
                            "needs_model"
                            if predicted["answer_status"] == "general_assisted"
                            and not render_calls
                            else "answered"
                        ),
                        "answer_status": predicted["answer_status"],
                        "applicability": structured["applicability"],
                        "confidence": structured["confidence"],
                        "text": rendered,
                        "neutral_content": neutral,
                        "frozen_contract": contract,
                        "frozen_contract_hash": predicted["renderer_contract_digest"],
                        "structured_prediction_hash": predicted["content_digest"],
                        "structured_prediction": structured,
                        "prediction_trace": predicted["prediction_trace"],
                        "style_status": style_status,
                        "style_gate": style_gate,
                        "evidence": evidence,
                        "uncertainties": [str(item["text"]) for item in structured["uncertainties"]],
                    }
                )
        total_model_calls = planning_calls + generation_calls + validation_calls
        base["model_usage"] = {
            "selected_model_ref": selected_model_ref,
            "planning_calls": planning_calls,
            "generation_calls": generation_calls,
            "validation_calls": validation_calls,
            "total_calls": total_model_calls,
            "status": (
                "used"
                if total_model_calls
                else "selected_but_not_needed"
                if selected_model_ref
                else "not_selected"
            ),
            "fallback_used": False,
        }
        _check_cancel()
        messages.append(base)
        self._save_active_messages(person_id, messages, dialogue_state=self._conversation_context(profile, messages, ""))
        return copy.deepcopy(base)

    def _find_message(self, person_id: str, message_id: str) -> tuple[list[dict[str, object]], dict[str, object]]:
        messages = self._active_messages(person_id)
        try:
            message = next(item for item in messages if item["message_id"] == message_id)
        except StopIteration as error:
            raise ConversationError("消息不存在。") from error
        return messages, message
