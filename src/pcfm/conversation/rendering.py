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

    def _unified_person_response(
        self,
        *,
        person_id: str,
        text: str,
        history: Sequence[Mapping[str, object]],
        conversation_context: Mapping[str, object],
        model_ref: str,
        artifact: Mapping[str, object],
    ) -> dict[str, object]:
        """One LLM call that internally performs understand->match->derive->compose.
        Code only gates afterwards; it does not participate in derivation."""
        if not model_ref:
            return {
                "status": "no_model",
                "model_calls": 0,
                "question_type": "",
                "stance": "neutral",
                "tendency_ids": [],
                "answer": "",
            }
        if self._model_services is None:
            raise ConversationError("模型服务管理器未启用；没有进行自动回退。")
        service, model_id = self._model_services.resolve_model_ref(model_ref)
        identity_note = str(
            dict(artifact.get("scope", {})).get("identity_note", "")
        ).strip()
        # 身份兜底：identity_note 为空时用人物名称+描述，避免「你是谁」退化成「我就是我」。
        if not identity_note:
            person = self._person(person_id)
            identity_note = "; ".join(
                part
                for part in (
                    str(person.get("name", "")).strip(),
                    str(person.get("description", "")).strip(),
                )
                if part
            )
        # 领域倾向画像：领域 → 该领域的价值倾向，供 LLM 定位领域并匹配。
        # 这是建模阶段预构建的整体倾向（分领域），回答时按领域"查"，不现算。
        # 只给抽象倾向（剥掉逐字原话 reason），原话走 evidence 回填，避免照抄/漏专名。
        domain_tendency_index = _derivation_view(artifact.get("domain_tendency_index", {}))
        # 回答语言随问题语言：中文问题必须中文回答，不被英文材料带偏。
        is_chinese = bool(re.search(r"[\u4e00-\u9fff]", str(text)))
        response_language = "Chinese" if is_chinese else "English"
        # 中文问题时把抽象字段也本地化成中文，使 LLM 上下文以中文为主。
        if is_chinese:
            domain_tendency_index = _localize_view(domain_tendency_index)
        payload = {
            "person_identity": identity_note,
            "question": text,
            "response_language": response_language,
            "conversation_messages": [
                {"role": item.get("role"), "text": str(item.get("text", ""))[:600]}
                for item in history[-12:]
                if item.get("role") in {"user", "assistant"}
            ],
            "conversation_state": copy.deepcopy(dict(conversation_context or {})),
            "domain_tendency_index": domain_tendency_index,
            "allowed_tendency_types": (
                sorted(TENDENCY_TYPE_ZH.values())
                if is_chinese
                else sorted(TENDENCY_TYPES)
            ),
            "allowed_interests": (
                sorted(str(value.get("label_zh", "")) for value in INTERESTS.values())
                if is_chinese
                else sorted(INTERESTS)
            ),
            "allowed_stances": sorted(STANCES),
            "allowed_event_structure_types": (
                sorted(EVENT_STRUCTURE_ZH.values())
                if is_chinese
                else sorted(EVENT_STRUCTURE_TYPES)
            ),
        }
        system = (
            "You are generating a first-person response for a modeled real person. "
            "domain_tendency_index is this person's value-tendency profile: "
            "domain → event_structure_type (the subdivision) → { dominant_value, "
            "opposes (object categories), supports (object categories), atoms }. "
            "Each atom is an abstract tendency: direction (support/oppose/mixed), "
            "target category, protected interest, accepted cost, and tendency_type. "
            "When response_language is Chinese these fields are already translated "
            "to Chinese — read them in Chinese and answer in Chinese. "
            "To answer in three steps. STEP 1 — LOCKED stance: identify the "
            "question's domain AND event-structure subdivision; read that "
            "subdivision's tendency; match atoms by target CATEGORY plus direction "
            "(a company/organization question uses target 'organization', never "
            "'individual'; a person question uses 'individual'). This determines your "
            "stance and value ranking — it is locked and you must not change it. "
            "STEP 2 — DEPTH: expand the answer with depth and specificity using your "
            "own general knowledge — concrete argumentation, examples, and context — "
            "but stay WITHIN the locked stance: never contradict the matched atoms' "
            "direction or value ranking. Do NOT invent this person's own biography, "
            "memories, or specific life events (that is their specific history, not "
            "general knowledge). STEP 3 — CONTENT: write it first-person, in "
            "response_language, PLAIN and clear — express the locked stance and the "
            "depth, but do NOT add stylistic flourish, characteristic tone, or "
            "opening/connector phrases (wording will be styled separately). Return "
            "JSON with exactly question_type, stance, tendency_ids, and answer. "
            "CONTEXT: conversation_state.current_topic is the topic being discussed "
            "RIGHT NOW. A short follow-up (why / what evidence / continue / more "
            "detail / 为什么 / 证据 / 继续 / 具体点) refers to the IMMEDIATELY "
            "PRECEDING assistant answer in the current topic — resolve it there, "
            "never against an earlier topic. If the question asks for evidence of "
            "your last claim, give the reasoning behind THAT last claim. "
            "question_type is one of identity, self_evaluation, object_evaluation, "
            "policy_stance, factual, ordinary_dialogue, or direct_historical. "
            "For identity questions (question_type=identity), answer ONLY from "
            "person_identity — that field is authoritative for who this person is; "
            "do not invent a name, role, or biography beyond it. "
            "stance must be one of the ENGLISH allowed_stances "
            "(support/oppose/neutral/conditional_support/mixed/insufficient_evidence) "
            "— even though the atom fields may be in Chinese, map the direction back "
            "to its English stance value. tendency_ids lists only the atom_ids values "
            "from the domain_tendency_index entries you actually relied on. "
            "answer is the person's first-person reply, under 1200 characters. "
            "response_language tells you which language to write in. If "
            "response_language is Chinese, the WHOLE answer MUST be Chinese — never "
            "paste an English phrase into a Chinese answer. When no tendency atom "
            "applies, set stance to insufficient_evidence and still write a natural, "
            "deep first-person reply without any meta-commentary about evidence or "
            "data availability."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        temperature = self._generation_temperature(person_id)
        compatibility_retry = False
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
        model_calls = 2 if compatibility_retry else 1
        # 逐字复刻原话时用更强指令重试一次：不缩成「保留判断」，也不放任照抄。
        verbatim_retried = False
        if isinstance(candidate, Mapping) and self._answer_copies_atom_reason(
            str(candidate.get("answer", "")), artifact
        ):
            retry_system = system + (
                " The answer you produced copies a source sentence verbatim. "
                "Rewrite it as a NEW first-person sentence expressing the same "
                "tendency — do not paste the atom's reason text."
            )
            retry_messages = [
                {"role": "system", "content": retry_system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            try:
                retry_response = self._model_services.invoke(
                    str(service["service_id"]),
                    model_id,
                    retry_messages,
                    structured=True,
                    temperature=temperature,
                )
                retry_candidate = _json_mapping(retry_response["text"])
                if isinstance(retry_candidate, Mapping):
                    candidate = retry_candidate
                    model_calls += 1
                    verbatim_retried = True
            except (ModelServiceError, json.JSONDecodeError):
                pass
        if not isinstance(candidate, Mapping):
            return {
                "status": "unparseable",
                "model_calls": model_calls,
                "question_type": "",
                "stance": "neutral",
                "tendency_ids": [],
                "answer": "",
            }
        return {
            "status": "ok",
            "question_type": str(candidate.get("question_type", "")).strip(),
            "stance": str(candidate.get("stance", "")).strip(),
            "tendency_ids": [str(v) for v in candidate.get("tendency_ids", [])],
            "answer": str(candidate.get("answer", "")).strip(),
            "model_calls": model_calls,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
            "fallback_used": False,
            "same_model_json_compatibility_retry": compatibility_retry,
            "verbatim_reason_retry": verbatim_retried,
        }

    def _render_person_answer(
        self,
        *,
        person_id: str,
        content: str,
        model_ref: str,
        is_chinese: bool,
    ) -> tuple[str, dict[str, object]]:
        """渲染层：只改措辞/语气/语言，不改立场、事实、论证。

        推导层产出的是平实正文（立场+深度已锁），这里把它换成人物口吻。
        任何失败都回退到平实正文，不缩成「保留判断」。
        """
        plain = str(content).strip()
        if not plain:
            return plain, {"status": "empty_content_no_render", "model_calls": 0}
        if not model_ref or self._model_services is None:
            return plain, {"status": "not_run_no_dialogue_model", "model_calls": 0}
        try:
            service, model_id = self._model_services.resolve_model_ref(model_ref)
        except ModelServiceError:
            return plain, {"status": "render_model_unavailable", "model_calls": 0}
        person = self._person(person_id)
        name = str(person.get("name", "")).strip() or "this person"
        language = "Chinese" if is_chinese else "English"
        system = (
            f"You are rephrasing an answer spoken by {name}. Rewrite the supplied "
            "content in their characteristic speaking style — sharp, direct, "
            "first-person, short punchy sentences, confident — in "
            f"{language}. ONLY reword: do NOT change the stance, claims, facts, "
            "reasoning, numbers, or names; do NOT add or remove content. If it is "
            "already in that style, return it nearly unchanged. Return JSON with "
            "exactly answer."
        )
        payload = {
            "content": plain,
            "language": language,
            "person_name": name,
        }
        messages = [
            {"role": "system", "content": system},
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
        except ModelServiceError:
            return plain, {"status": "render_failed_fallback_plain", "model_calls": 0}
        try:
            candidate = _json_mapping(response["text"])
        except json.JSONDecodeError:
            candidate = None
        rendered = (
            str(candidate.get("answer", "")).strip()
            if isinstance(candidate, Mapping)
            else ""
        )
        if not rendered or len(rendered) > 1200:
            return plain, {"status": "render_invalid_fallback_plain", "model_calls": 1}
        # 语言守门：中文问必须中文答，否则回退平实正文（平实正文已在推导层按语言写好）
        if is_chinese and not re.search(r"[\u4e00-\u9fff]", rendered):
            return plain, {"status": "render_language_mismatch_fallback_plain", "model_calls": 1}
        return rendered, {
            "status": "rendered",
            "model_calls": 1,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
        }

    def _gate_unified_response(
        self,
        unified: Mapping[str, object],
        artifact: Mapping[str, object],
    ) -> tuple[bool, str]:
        """Code gate: only validates the LLM output, never derives."""
        stance = str(unified.get("stance", "")).strip()
        if stance not in STANCES:
            return False, "stance_not_in_closed_vocabulary"
        question_type = str(unified.get("question_type", "")).strip()
        if not question_type:
            return False, "question_type_empty"
        valid_ids: set[str] = {
            str(item.get("preference_atom_id", ""))
            for item in dict(artifact.get("reviewed_public_model", {})).get(
                "preference_atoms", []
            )
        }
        valid_ids |= {
            str(item.get("orientation_id", ""))
            for item in artifact.get("orientation_index", [])
        }
        valid_ids |= {
            str(item.get("orientation_id", ""))
            for item in artifact.get("value_orientation_index", [])
        }
        valid_ids.discard("")
        tendency_ids = {str(value) for value in unified.get("tendency_ids", [])}
        if not tendency_ids <= valid_ids:
            return False, "tendency_id_not_in_artifact"
        # 锁死立场：评价类原子的 direction 与输出 stance 不能矛盾。
        # 这是「内核可证伪」的硬门：把原子方向反过来，答案立场必须跟着翻。
        atoms_by_id = {
            str(item.get("preference_atom_id", "")): item
            for item in dict(artifact.get("reviewed_public_model", {})).get(
                "preference_atoms", []
            )
        }
        evaluation_directions: set[str] = set()
        for tid in tendency_ids:
            atom = atoms_by_id.get(tid)
            if not atom:
                continue
            if str(atom.get("tendency_type", "")) not in EVALUATION_TENDENCY_TYPES:
                continue
            direction = str(atom.get("direction", "")).strip()
            if direction:
                evaluation_directions.add(direction)
        if evaluation_directions == {"oppose"} and stance == "support":
            return False, "stance_contradicts_oppose_atom"
        if evaluation_directions == {"support"} and stance == "oppose":
            return False, "stance_contradicts_support_atom"
        answer = str(unified.get("answer", "")).strip()
        if not answer:
            return False, "empty_answer"
        if len(answer) > 1200:
            return False, "answer_too_long"
        forbidden_experience = re.compile(
            r"\b(?:i remember|i was|i have been|my administration|when i was|"
            r"in my experience)\b|我记得|我曾经|我的政府|在我任内|以我的经历",
            re.I,
        )
        if forbidden_experience.search(answer):
            return False, "forbidden_experience"
        # 第一人称交给风格层处理，不据此拒绝（它曾导致同一问题时而答时而拒）
        return True, ""

    @staticmethod
    def _answer_copies_atom_reason(answer: str, artifact: Mapping[str, object]) -> bool:
        """answer 是否近乎逐字复刻了某个倾向原子的原话（reason）。

        只检测长连续片段（>=20 个规范字符），避免误伤正常改写中偶然共用的短词。
        """
        def _norm(value: str) -> str:
            return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(value).casefold())

        reasons: list[str] = []
        for domain_cells in dict(artifact.get("domain_tendency_index", {})).values():
            if not isinstance(domain_cells, Mapping):
                continue
            for cell in domain_cells.values():
                if not isinstance(cell, Mapping):
                    continue
                for atom in cell.get("atoms", []):
                    if isinstance(atom, Mapping) and atom.get("reason"):
                        reasons.append(str(atom["reason"]))
        for item in dict(artifact.get("reviewed_public_model", {})).get(
            "preference_atoms", []
        ):
            if not isinstance(item, Mapping):
                continue
            for reason in item.get("reasons", []) or []:
                if reason:
                    reasons.append(str(reason))
        answer_norm = _norm(answer)
        for reason in reasons:
            reason_norm = _norm(reason)
            if len(reason_norm) >= 20 and reason_norm in answer_norm:
                return True
        return False
