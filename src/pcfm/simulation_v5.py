from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .response_prediction import EVALUATION_TENDENCY_TYPES
from .response_prediction_v2 import ResponsePredictionKernelV2
from .simulation_v4 import (
    DOMAIN_ALIASES,
    INTERESTS,
    MODEL_SCHEMA_V4,
    SimulationKernelV4,
    SimulationV4Error,
    _domains,
    _hash,
    _interest_mentions,
    _contains,
    _normal,
    _similarity,
    _terms,
)


MODEL_SCHEMA_V5 = "pcfm-conversation-conditioned-model-v5"
PREDICTION_SCHEMA_V5 = "pcfm-conversation-conditioned-prediction-v5"
FROZEN_CONTRACT_V5 = "pcfm-frozen-content-contract-v5"
KERNEL_ID_V5 = "simulation-v5"
MODEL_BUILD_V5 = "reviewed-source-entity-index-v3"

SCENARIO_EFFECTS = frozenset({"advances", "constrains", "threatens", "neutral"})
MODEL_AUTHORITY_FIELDS = frozenset(
    {"predicted_stance", "person_stance", "direction", "answer", "person_answer"}
)


class SimulationV5Error(ValueError):
    pass


def _is_short_reference(text: str) -> bool:
    clean = str(text).strip().casefold()
    return bool(
        re.fullmatch(
            r"(?:why|why is that|what about(?: that| it)?|and that|continue|go on|"
            r"为什么|为什么呢|那呢|这个呢|那个呢|继续|接着说)[?？!！\s]*",
            clean,
        )
    )


def _is_person_opinion_request(text: str) -> bool:
    clean = str(text).strip().casefold()
    return bool(
        re.search(
            r"(?:what do you think (?:of|about)|what is your opinion (?:of|on|about)|"
            r"how do you view|do you think you are|do you consider yourself|"
            r"你认为.+怎么样|你怎么看|你的看法|你如何评价|"
            r"你认为你是|你认为自己|你觉得自己|你是不是|你算不算)",
            clean,
        )
    )


def _matches_reviewed_entity_alias(
    text: str, frame: Mapping[str, object]
) -> bool:
    clean = _normal(str(text))
    aliases = dict(frame.get("episode_context") or {}).get("entity_aliases", [])
    return any(_normal(str(alias)) in clean for alias in aliases if str(alias).strip())


def _filtered_source(source: Mapping[str, object]) -> dict[str, object]:
    """Keep navigation chunks out of the response-episode model."""
    item = copy.deepcopy(dict(source))
    item["segments"] = []
    return item


def _episode_frames(
    frames: Sequence[Mapping[str, object]],
    sources: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_source = {str(item.get("source_id", "")): dict(item) for item in sources}
    reviewed_by_hash = {
        str(item.get("content_hash") or _hash([item.get("question", ""), item.get("response", "")])): dict(item)
        for source in sources
        for item in source.get("reviewed_event_frames_v4", [])
        if isinstance(item, Mapping) and item.get("review_status") == "confirmed"
    }
    enriched = []
    for raw in frames:
        frame = copy.deepcopy(dict(raw))
        source = by_source.get(str(frame.get("source_id", "")), {})
        evidence_hash = str(dict(frame.get("evidence") or {}).get("content_hash", ""))
        reviewed = reviewed_by_hash.get(evidence_hash, {})
        response_time = str(dict(frame["temporal_context"]).get("response_time", ""))
        trigger_span = str(reviewed.get("trigger_span", ""))
        context_span = str(reviewed.get("context_span", ""))
        interlocutor = str(reviewed.get("interlocutor", ""))
        occasion = str(reviewed.get("occasion") or "").strip()
        if not occasion:
            occasion = " | ".join(
                value
                for value in (
                    str(source.get("title", "")).strip(),
                    str(source.get("source_context", "")).strip(),
                )
                if value
            )
        missing = [
            name
            for name, value in (
                ("response_time", response_time),
                ("interlocutor", interlocutor),
                ("occasion", occasion),
                ("available_information", source.get("available_information", "")),
            )
            if not str(value).strip()
        ]
        frame["episode_context"] = {
            "atomicity": "independently_traceable_response_episode_not_context_free_sentence",
            "trigger_window": trigger_span or str(dict(frame["decision_frame"])["trigger"]),
            "trigger_grounding_status": str(
                reviewed.get(
                    "trigger_grounding_status",
                    "exact_confirmed_question_answer"
                    if frame.get("origin") == "confirmed_direct_evidence_only"
                    else "reviewed_semantic_summary_not_exact_span",
                )
            ),
            "context_window": context_span,
            "context_grounding_status": str(
                reviewed.get("context_grounding_status", "source_metadata_only_or_unknown")
            ),
            "response_span": str(dict(frame["observed_response"])["verbatim"]),
            "response_grounding_status": "exact_reviewed_source_span",
            "response_time": response_time or "unknown",
            "interlocutor": interlocutor or "unknown",
            "occasion": occasion or "unknown",
            "entity_aliases": sorted(
                {
                    str(value).strip()
                    for value in source.get("entity_aliases", [])
                    if str(value).strip()
                }
            ),
            "available_information": str(source.get("available_information", "")) or "unknown",
            "missing_fields": missing,
            "unknowns_are_not_model_filled": True,
            "temporal_aggregation_eligible": not bool("response_time" in missing),
        }
        enriched.append(frame)
    return enriched


def _value_atoms(frames: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Extract observable public salience only; never infer a private value."""
    atoms: list[dict[str, object]] = []
    for raw in frames:
        frame = dict(raw)
        response = str(dict(frame["observed_response"]).get("verbatim", ""))
        seen: set[str] = set()
        for mention in _interest_mentions(response):
            interest_id = str(mention["interest_id"])
            if interest_id in seen:
                continue
            seen.add(interest_id)
            atoms.append(
                {
                    "value_atom_id": f"value-v5-{_hash([frame['event_frame_id'], interest_id])[:16]}",
                    "event_frame_id": str(frame["event_frame_id"]),
                    "source_id": str(frame["source_id"]),
                    "source_lineage": str(frame["source_lineage"]),
                    "interest_id": interest_id,
                    "evidence_span": str(mention["span"]),
                    "domain_tags": list(map(str, frame.get("domain_tags", []))),
                    "role": str(dict(frame["social_context"]).get("speaker_role", "unknown")),
                    "response_time": str(
                        dict(frame["temporal_context"]).get("response_time", "")
                    ),
                    "status": "deterministic_explicit_public_salience",
                    "interpretation_boundary": (
                        "observable_public_wording_not_private_value_or_direction"
                    ),
                }
            )
    return sorted(atoms, key=lambda item: str(item["value_atom_id"]))


def _value_orientations(
    atoms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for raw in atoms:
        item = copy.deepcopy(dict(raw))
        grouped.setdefault(str(item["interest_id"]), []).append(item)
    orientations: list[dict[str, object]] = []
    for interest_id, values in sorted(grouped.items()):
        lineages = sorted({str(item["source_lineage"]) for item in values})
        domains = sorted(
            {
                str(domain)
                for item in values
                for domain in item.get("domain_tags", [])
            }
        )
        event_ids = sorted({str(item["event_frame_id"]) for item in values})
        times = sorted(
            str(item["response_time"])
            for item in values
            if str(item.get("response_time", ""))
        )
        if len(lineages) >= 2 and len(domains) >= 2:
            status = "cross_domain_public_salience"
        elif len(lineages) >= 2 or len(event_ids) >= 2:
            status = "repeated_public_salience"
        else:
            status = "single_source_public_salience"
        support = min(
            0.7,
            0.2
            + 0.08 * min(3, len(event_ids))
            + 0.08 * min(3, len(lineages))
            + 0.04 * min(3, len(domains)),
        )
        orientations.append(
            {
                "orientation_id": f"salience-v5-{_hash(interest_id)[:16]}",
                "interest_id": interest_id,
                "supporting_event_ids": event_ids,
                "supporting_atom_ids": sorted(
                    str(item["value_atom_id"]) for item in values
                ),
                "independent_source_count": len(lineages),
                "domain_count": len(domains),
                "primary_domains": domains,
                "role_scope": sorted({str(item["role"]) for item in values}),
                "temporal_scope": {
                    "start": times[0] if times else "unknown",
                    "end": times[-1] if times else "unknown",
                    "dated_event_count": len(times),
                },
                "evidence_spans": sorted(
                    {str(item["evidence_span"]) for item in values}
                ),
                "status": status,
                "support": support,
                "confidence_kind": (
                    "public_salience_support_not_preference_or_accuracy_probability"
                ),
            }
        )
    return orientations


class SimulationKernelV5:
    kernel_id = KERNEL_ID_V5

    def __init__(self) -> None:
        self._v4 = SimulationKernelV4()

    def fit(
        self,
        *,
        person_id: str,
        version: int,
        reviewed_sources: Sequence[Mapping[str, object]],
        scope: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        filtered = [_filtered_source(value) for value in reviewed_sources]
        try:
            reviewed = self._v4.fit(
                person_id=person_id,
                version=version,
                reviewed_sources=filtered,
                scope=scope,
            )
        except SimulationV4Error as error:
            raise SimulationV5Error("no_eligible_reviewed_episode") from error
        reviewed_ids = {
            str(frame["event_frame_id"])
            for frame in reviewed.get("event_frames", [])
            if frame.get("origin")
            in {"reviewed_semantic_event", "confirmed_direct_evidence_only"}
        }
        if not reviewed_ids:
            raise SimulationV5Error("no_eligible_reviewed_episode")
        orientation_index = []
        for raw in reviewed.get("preference_structures", []):
            item = copy.deepcopy(dict(raw))
            if item.get("status") not in {
                "cross_domain_public_preference",
                "repeated_domain_public_preference",
            }:
                continue
            orientation_index.append(
                {
                    "orientation_id": str(item["preference_structure_id"]),
                    "tendency_types": list(map(str, item.get("tendency_types", []))),
                    "directions": list(map(str, item.get("directions", []))),
                    "protected_interest_id": str(item["protected_interest_id"]),
                    "accepted_cost_id": str(item["accepted_cost_id"]),
                    "supporting_event_ids": list(map(str, item["supporting_event_ids"])),
                    "independent_source_count": int(item["independent_source_count"]),
                    "domain_count": int(item["domain_count"]),
                    "primary_domains": list(map(str, item.get("primary_domains", []))),
                    "role_scope": list(map(str, item.get("role_scope", []))),
                    "temporal_scope": copy.deepcopy(dict(item["temporal_scope"])),
                    "conditions": list(map(str, item.get("conditions", []))),
                    "reasons": list(map(str, item.get("reasons", []))),
                    "counterevidence_event_ids": list(
                        map(str, item.get("counterevidence_event_ids", []))
                    ),
                    "counterevidence_domains": list(
                        map(str, item.get("counterevidence_domains", []))
                    ),
                    "status": str(item["status"]),
                    "confidence_kind": "repeated_public_evidence_not_accuracy_probability",
                }
            )
        episode_frames = _episode_frames(
            list(reviewed.get("event_frames", [])), filtered
        )
        value_atoms = _value_atoms(episode_frames)
        value_orientation_index = _value_orientations(value_atoms)
        payload = {
            "reviewed_public_model": reviewed,
            "event_frames": episode_frames,
            "orientation_index": sorted(
                orientation_index, key=lambda value: str(value["orientation_id"])
            ),
            "value_atoms": value_atoms,
            "value_orientation_index": value_orientation_index,
            "knowledge_claims": copy.deepcopy(
                list(reviewed.get("knowledge_claims", []))
            ),
        }
        artifact: dict[str, object] = {
            "schema_version": MODEL_SCHEMA_V5,
            "kernel": KERNEL_ID_V5,
            "model_build": MODEL_BUILD_V5,
            "person_id": person_id,
            "version": int(version),
            "scope": copy.deepcopy(dict(scope or {})),
            **payload,
            "semantic_model_digest": _hash(
                {
                    "reviewed_semantic_model_digest": reviewed["semantic_model_digest"],
                    "event_frames": payload["event_frames"],
                    "orientation_index": payload["orientation_index"],
                    "value_atoms": payload["value_atoms"],
                    "value_orientation_index": payload["value_orientation_index"],
                    "knowledge_claims": payload["knowledge_claims"],
                }
            ),
            "active_components": [
                "reviewed_response_episodes_v5",
                "conversation_state_v1",
                "public_orientation_projection_v1",
                "event_public_salience_v1",
                "bounded_natural_response_v1",
            ],
            "components": [
                {"component_id": "reviewed_response_episodes_v5", "status": "active"},
                {"component_id": "conversation_state_v1", "status": "active"},
                {
                    "component_id": "public_orientation_projection_v1",
                    "status": "active_exploratory_accuracy_not_assessed",
                },
                {
                    "component_id": "event_public_salience_v1",
                    "status": "active_exploratory_not_private_value",
                },
                {"component_id": "simulation_v4", "status": "frozen_evidence_submodel"},
                {"component_id": "response_kernel_v2", "status": "frozen_baseline_only"},
            ],
            "validation_status": "implemented_exploratory_accuracy_not_assessed",
            "accuracy_claim": "none",
        }
        artifact["artifact_hash"] = _hash(artifact)
        return artifact

    @staticmethod
    def verify(artifact: Mapping[str, object]) -> None:
        if (
            artifact.get("schema_version") != MODEL_SCHEMA_V5
            or artifact.get("kernel") != KERNEL_ID_V5
            or artifact.get("model_build") != MODEL_BUILD_V5
        ):
            raise SimulationV5Error("unsupported_simulation_v5_schema")
        value = copy.deepcopy(dict(artifact))
        declared = str(value.pop("artifact_hash", ""))
        if _hash(value) != declared:
            raise SimulationV5Error("simulation_v5_integrity_failed")
        payload = {
            "reviewed_semantic_model_digest": str(
                dict(artifact.get("reviewed_public_model") or {}).get(
                    "semantic_model_digest", ""
                )
            ),
            "event_frames": copy.deepcopy(artifact.get("event_frames")),
            "orientation_index": copy.deepcopy(artifact.get("orientation_index")),
            "value_atoms": copy.deepcopy(artifact.get("value_atoms")),
            "value_orientation_index": copy.deepcopy(
                artifact.get("value_orientation_index")
            ),
            "knowledge_claims": copy.deepcopy(artifact.get("knowledge_claims")),
        }
        if _hash(payload) != artifact.get("semantic_model_digest"):
            raise SimulationV5Error("simulation_v5_semantic_payload_mismatch")
        reviewed = dict(artifact.get("reviewed_public_model") or {})
        if reviewed.get("schema_version") != MODEL_SCHEMA_V4:
            raise SimulationV5Error("simulation_v5_reviewed_submodel_invalid")
        SimulationKernelV4.verify(reviewed)

    @staticmethod
    def _query(
        text: str,
        history: Sequence[Mapping[str, object]],
        conversation_context: Mapping[str, object] | None,
        query_plan: Mapping[str, object] | None,
    ) -> dict[str, object]:
        plan = dict(query_plan or {})
        history_by_id = {
            str(item.get("message_id")): item
            for item in history
            if item.get("message_id") and item.get("role") in {"user", "assistant"}
        }
        rejected = sorted(field for field in MODEL_AUTHORITY_FIELDS if field in plan)
        resolved_ids = []
        for raw_id in plan.get("resolved_message_ids", []):
            message_id = str(raw_id)
            if message_id not in history_by_id:
                rejected.append(message_id)
            elif message_id not in resolved_ids:
                resolved_ids.append(message_id)
        ambiguous_reference = False
        if _is_short_reference(text) and not resolved_ids:
            context_ids = [
                str(value)
                for value in dict(conversation_context or {}).get(
                    "active_topic_message_ids", []
                )
                if str(value) in history_by_id
            ]
            if context_ids:
                resolved_ids = context_ids
            else:
                user_ids = [
                    str(item["message_id"])
                    for item in history
                    if item.get("role") == "user" and item.get("message_id")
                ]
                if len(user_ids) == 1:
                    resolved_ids = user_ids
                elif user_ids:
                    ambiguous_reference = True
        resolved_messages = [history_by_id[value] for value in resolved_ids]
        context_evidence_event_ids = sorted(
            {
                str(event_id)
                for item in resolved_messages
                if item.get("role") == "assistant"
                for event_id in dict(item.get("structured_prediction") or {}).get(
                    "evidence_event_ids", []
                )
            }
        )
        combined = "\n".join(
            [
                *(str(item.get("text", "")) for item in resolved_messages),
                text,
            ]
        ).strip()
        deterministic_mentions = _interest_mentions(combined)
        scenario_effects = []
        seen_effects = set()
        for raw in plan.get("scenario_effects", []):
            if not isinstance(raw, Mapping):
                rejected.append("invalid_scenario_effect")
                continue
            interest_id = str(raw.get("interest_id", ""))
            effect = str(raw.get("effect", ""))
            scenario_span = str(raw.get("scenario_span", "")).strip()
            if (
                interest_id not in INTERESTS
                or effect not in SCENARIO_EFFECTS
                or not _contains(combined, scenario_span)
            ):
                rejected.append(f"scenario_effect:{interest_id}:{effect}")
                continue
            key = (interest_id, effect)
            if key not in seen_effects:
                scenario_effects.append(
                    {
                        "interest_id": interest_id,
                        "effect": effect,
                        "scenario_span": scenario_span,
                        "origin": "model_semantic_candidate_not_person_stance",
                    }
                )
                seen_effects.add(key)
        domain_ids = set(_domains(combined))
        for raw in plan.get("domain_ids", []):
            domain_id = str(raw)
            if domain_id in DOMAIN_ALIASES:
                domain_ids.add(domain_id)
            else:
                rejected.append(f"domain:{domain_id}")
        role = "private" if re.search(
            r"\b(?:private|personal|family|friend)\b|私人|家庭|家人|朋友",
            combined,
            re.I,
        ) else "public"
        years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", combined)]
        question_scope = str(plan.get("question_scope", "")).strip()
        if question_scope not in {"narrow", "wide", "composite"}:
            question_scope = ""
        # 人物评价类开放式问题（"你认为 X 怎么样"）本质是宽评价，确定性默认 wide，
        # 避免在 LLM 语义规划被跳过时落到检索。
        if not question_scope and _is_person_opinion_request(text):
            question_scope = "wide"
        target_entity = str(plan.get("target_entity", "")).strip()
        return {
            "query": text,
            "combined_query": combined,
            "question_scope": question_scope,
            "target_entity": target_entity,
            "resolved_message_ids": resolved_ids,
            "resolved_message_roles": [str(item.get("role", "")) for item in resolved_messages],
            "context_evidence_event_ids": context_evidence_event_ids,
            "ambiguous_reference": ambiguous_reference,
            "domain_ids": sorted(domain_ids),
            "option_mentions": deterministic_mentions,
            "option_ids": sorted(
                {str(value["interest_id"]) for value in deterministic_mentions}
            ),
            "scenario_effects": scenario_effects,
            "role": role,
            "mentioned_years": years,
            "current_time_marker": bool(
                re.search(r"\b(?:today|now|currently)\b|现在|如今|当前|今天", combined, re.I)
            ),
            "relative_future_marker": bool(
                re.search(
                    r"\b(?:future|tomorrow|next\s+(?:year|month|week))\b|未来|明年|下个月|下周|以后",
                    combined,
                    re.I,
                )
            ),
            "selected_event_ids": sorted(
                {str(value) for value in plan.get("selected_event_ids", [])}
            ),
            "selected_structure_ids": sorted(
                {str(value) for value in plan.get("selected_structure_ids", [])}
            ),
            "rejected_fields": sorted(set(rejected)),
            "semantic_mapping_status": (
                "model_candidate_validated_to_closed_ids"
                if scenario_effects or plan.get("resolved_message_ids")
                else "deterministic_only"
            ),
        }

    def predict(
        self,
        artifact: Mapping[str, object],
        *,
        text: str,
        history: Sequence[Mapping[str, object]],
        conversation_context: Mapping[str, object] | None = None,
        query_plan: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.verify(artifact)
        clean = str(text).strip()
        if not clean:
            raise SimulationV5Error("message_required")
        query = self._query(clean, history, conversation_context, query_plan)
        allowed_event_ids = {
            str(item["event_frame_id"]) for item in artifact.get("event_frames", [])
        }
        allowed_structure_ids = {
            str(item["orientation_id"]) for item in artifact.get("orientation_index", [])
        } | {
            str(item["orientation_id"])
            for item in artifact.get("value_orientation_index", [])
        }
        invalid_events = [
            value for value in query["selected_event_ids"] if value not in allowed_event_ids
        ]
        invalid_structures = [
            value
            for value in query["selected_structure_ids"]
            if value not in allowed_structure_ids
        ]
        query["selected_event_ids"] = [
            value for value in query["selected_event_ids"] if value in allowed_event_ids
        ]
        query["selected_structure_ids"] = [
            value
            for value in query["selected_structure_ids"]
            if value in allowed_structure_ids
        ]
        query["rejected_fields"] = sorted(
            set(query["rejected_fields"])
            | {f"event:{value}" for value in invalid_events}
            | {f"structure:{value}" for value in invalid_structures}
        )
        trace = {
            "kernel": KERNEL_ID_V5,
            "context_digest": _hash(
                [
                    clean,
                    [
                        (item.get("message_id"), item.get("role"), item.get("text"))
                        for item in history
                    ],
                    dict(conversation_context or {}),
                ]
            ),
            "conversation_context": copy.deepcopy(dict(conversation_context or {})),
            "resolved_context_message_ids": list(query["resolved_message_ids"]),
            "resolved_context_turns": len(query["resolved_message_ids"]),
            "rejected_query_plan_fields": list(query["rejected_fields"]),
            "generative_content_calls": 0,
            "context_used": {
                "message_ids": [
                    str(item.get("message_id")) for item in history if item.get("message_id")
                ],
                "turn_count": len(history),
                "generated_context_count": sum(
                    item.get("context_role") == "model_generated_context"
                    for item in history
                ),
                "generated_context_is_fitting_evidence": False,
            },
        }
        ordinary = ResponsePredictionKernelV2.ordinary_dialogue(clean)
        if ordinary:
            return self._result(
                artifact,
                answer_status="ordinary_dialogue",
                speech_act=ordinary[0],
                stance="neutral",
                claims=[],
                reasons=[],
                uncertainties=[],
                evidence_event_ids=[],
                response_basis={
                    "path": "ordinary_dialogue",
                    "person_prediction_status": "not_applicable",
                    "query_frame": query,
                },
                applicability="ordinary_dialogue_content_free",
                support=1.0,
                trace={**trace, "prediction_path": "ordinary_dialogue"},
                ordinary_text=ordinary[1],
            )
        if query["ambiguous_reference"]:
            return self._result(
                artifact,
                answer_status="clarification_needed",
                speech_act="clarification",
                stance="neutral",
                claims=[],
                reasons=[],
                uncertainties=["conversation_reference_ambiguous"],
                evidence_event_ids=[],
                response_basis={
                    "path": "clarification",
                    "person_prediction_status": "not_available",
                    "query_frame": query,
                },
                applicability="conversation_reference_ambiguous",
                support=0.0,
                trace={**trace, "prediction_path": "clarification"},
                refusal_reasons=["conversation_reference_ambiguous"],
            )
        frames = [copy.deepcopy(dict(value)) for value in artifact.get("event_frames", [])]
        ranked = sorted(
            (
                (
                    _similarity(
                        str(query["combined_query"]),
                        str(dict(frame["decision_frame"])["trigger"]),
                    ),
                    str(frame["event_frame_id"]),
                    frame,
                )
                for frame in frames
            ),
            key=lambda value: (-value[0], value[1]),
        )
        exact = next(
            (
                frame
                for _, _, frame in ranked
                if _normal(str(dict(frame["decision_frame"])["trigger"]))
                == _normal(str(query["combined_query"]))
            ),
            None,
        )
        if exact is None and ranked:
            best_score = ranked[0][0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score >= 0.9 and best_score - second_score >= 0.15:
                exact = ranked[0][2]
        if exact is not None:
            response = str(dict(exact["observed_response"])["verbatim"])
            return self._result(
                artifact,
                answer_status="direct_answer",
                speech_act="direct_answer",
                stance="neutral",
                claims=[response],
                reasons=list(map(str, dict(exact["observed_response"]).get("reasons", []))),
                uncertainties=["这是历史公开回答，不是对新情境的预测。"],
                evidence_event_ids=[str(exact["event_frame_id"])],
                response_basis={
                    "path": "direct_historical_response",
                    "person_prediction_status": "direct_historical_evidence",
                    "query_frame": query,
                    "selected_demonstrated_knowledge": [
                        item
                        for item in artifact.get("knowledge_claims", [])
                        if item.get("event_frame_id") == exact["event_frame_id"]
                    ],
                    "knowledge_boundary": "exact_publicly_demonstrated_claims_only_not_complete_person_knowledge",
                },
                applicability="exact_reviewed_episode",
                support=1.0,
                trace={
                    **trace,
                    "prediction_path": "direct_event",
                    "selected_event_ids": [str(exact["event_frame_id"])],
                },
            )
        if _is_short_reference(clean) and query["context_evidence_event_ids"]:
            context_frames = [
                frame
                for frame in frames
                if str(frame["event_frame_id"])
                in set(map(str, query["context_evidence_event_ids"]))
            ]
            if context_frames:
                reasons = [
                    str(value)
                    for frame in context_frames
                    for value in dict(frame["observed_response"]).get("reasons", [])
                ]
                claims = reasons or [
                    str(dict(context_frames[0]["observed_response"])["verbatim"])
                ]
                event_ids = [
                    str(frame["event_frame_id"]) for frame in context_frames
                ]
                return self._result(
                    artifact,
                    answer_status="direct_answer",
                    speech_act="explain" if reasons else "direct_answer",
                    stance="neutral",
                    claims=claims,
                    reasons=[],
                    uncertainties=["回答沿用上一轮已经选中的人物证据；生成对话本身不是人物证据。"],
                    evidence_event_ids=event_ids,
                    response_basis={
                        "path": "resolved_conversation_evidence_followup",
                        "person_prediction_status": "direct_historical_evidence",
                        "query_frame": query,
                    },
                    applicability="resolved_prior_evidence_context",
                    support=1.0,
                    trace={
                        **trace,
                        "prediction_path": "resolved_context_evidence",
                        "selected_event_ids": event_ids,
                    },
                )
        evaluation = self._object_evaluation_projection(artifact, query)
        if evaluation is not None:
            return self._result(
                artifact,
                answer_status="object_evaluation_projection_answer",
                speech_act="direct_answer",
                stance=str(evaluation["stance"]),
                claims=[str(evaluation["statement"])],
                reasons=list(map(str, evaluation["reasons"])),
                uncertainties=[
                    "这些是历史公开表态的多维归纳，不构成对当前情境的确定预测。"
                ],
                evidence_event_ids=list(map(str, evaluation["evidence_event_ids"])),
                response_basis={
                    "path": "object_evaluation_projection",
                    "person_prediction_status": "evaluation_tendency_projection",
                    "query_frame": query,
                    "projection_kind": evaluation["projection_kind"],
                    "dimensions": evaluation["dimensions"],
                    "target": evaluation["target"],
                    "prediction_statement": str(evaluation["statement"]),
                },
                applicability=str(evaluation["applicability"]),
                support=float(evaluation["support"]),
                trace={**trace, "prediction_path": "object_evaluation_projection"},
            )
        projection = self._orientation_projection(artifact, query)
        if projection is None:
            projection = self._value_orientation_projection(artifact, query)
        if projection is not None:
            return self._result(
                artifact,
                answer_status="orientation_projection_answer",
                speech_act="direct_answer",
                stance=str(projection["stance"]),
                claims=[str(projection["statement"])],
                reasons=list(map(str, projection["reasons"])),
                uncertainties=[
                    "这是基于重复公开取向和当前情境映射的探索性预测，不代表人物真实内心，也不是准确率概率。"
                ],
                evidence_event_ids=list(map(str, projection["evidence_event_ids"])),
                response_basis={
                    "path": "contextual_orientation_projection",
                    "person_prediction_status": "public_orientation_projection",
                    "query_frame": query,
                    "selected_orientation": projection["orientation"],
                    "protected_interest_id": projection["protected_interest_id"],
                    "accepted_cost_id": projection["accepted_cost_id"],
                    "projection_kind": projection["projection_kind"],
                    "prediction_statement": projection["statement"],
                    "selected_demonstrated_knowledge": [],
                    "knowledge_boundary": "external_background_cannot_choose_person_direction",
                },
                applicability=str(projection["applicability"]),
                support=float(projection["support"]),
                trace={
                    **trace,
                    "prediction_path": "contextual_orientation_projection",
                    "selected_event_ids": list(map(str, projection["evidence_event_ids"])),
                },
            )
        selected_ids = set(query["selected_event_ids"])
        selected_frames = [
            frame
            for frame in frames
            if str(frame["event_frame_id"]) in selected_ids
            and (
                not query["domain_ids"]
                or set(map(str, frame.get("domain_tags", [])))
                & set(map(str, query["domain_ids"]))
            )
        ]
        if not selected_frames and query["domain_ids"]:
            domains = set(map(str, query["domain_ids"]))
            query_terms = _terms(str(query["combined_query"]))
            related = []
            for frame in frames:
                if not domains & set(map(str, frame.get("domain_tags", []))):
                    continue
                candidate_text = (
                    f"{dict(frame['decision_frame']).get('trigger', '')} "
                    f"{dict(frame['observed_response']).get('verbatim', '')}"
                )
                if len(query_terms & _terms(candidate_text)) < 2:
                    continue
                score = _similarity(str(query["combined_query"]), candidate_text)
                if score >= 0.15:
                    related.append((score, str(frame["event_frame_id"]), frame))
            related.sort(key=lambda value: (-value[0], value[1]))
            selected_frames = [value[2] for value in related[:3]]
        if not selected_frames and _is_person_opinion_request(clean):
            entity_related = [
                frame
                for frame in frames
                if _matches_reviewed_entity_alias(
                    str(query["combined_query"]), frame
                )
            ]
            if len(entity_related) == 1:
                selected_frames = entity_related
        if selected_frames:
            event_ids = [str(frame["event_frame_id"]) for frame in selected_frames[:3]]
            return self._result(
                artifact,
                answer_status="similar_event_evidence_answer",
                speech_act="direct_answer",
                stance="neutral",
                claims=[
                    str(dict(frame["observed_response"])["verbatim"])
                    for frame in selected_frames[:3]
                ],
                reasons=[],
                uncertainties=["这些是相关历史公开回应，不能单独确定新问题的立场。"],
                evidence_event_ids=event_ids,
                response_basis={
                    "path": "similar_event_evidence",
                    "person_prediction_status": "analogical_evidence_not_new_stance",
                    "query_frame": query,
                },
                applicability="related_reviewed_episodes_without_new_stance",
                support=0.0,
                trace={
                    **trace,
                    "prediction_path": "similar_event_evidence",
                    "selected_event_ids": event_ids,
                },
            )
        if _is_person_opinion_request(clean):
            return self._result(
                artifact,
                answer_status="refused",
                speech_act="refusal",
                stance="neutral",
                claims=[],
                reasons=[],
                uncertainties=[
                    "No applicable person event or public orientation supports this requested opinion."
                ],
                evidence_event_ids=[],
                response_basis={
                    "path": "person_opinion_evidence_required",
                    "person_prediction_status": "not_available",
                    "query_frame": query,
                },
                applicability="person_opinion_evidence_required",
                support=0.0,
                trace={**trace, "prediction_path": "person_opinion_evidence_required"},
                refusal_reasons=["person_opinion_evidence_required"],
            )
        return self._result(
            artifact,
            answer_status="general_assisted",
            speech_act="direct_answer",
            stance="neutral",
            claims=[],
            reasons=[],
            uncertainties=["没有找到可适用的人物事件或重复公开取向。"],
            evidence_event_ids=[],
            response_basis={
                "path": "general_assisted",
                "person_prediction_status": "not_available",
                "query_frame": query,
                "person_prediction_refusal_reasons": [
                    "no_applicable_reviewed_person_structure"
                ],
                "external_knowledge_policy": "allowed_if_disclosed_not_person_stance",
                "selected_demonstrated_knowledge": [],
                "knowledge_boundary": "no_person_claim_selected_external_model_content_is_not_person_knowledge",
            },
            applicability="general_knowledge_not_person_prediction",
            support=0.0,
            trace={**trace, "prediction_path": "general_assisted"},
        )

    @staticmethod
    def _orientation_projection(
        artifact: Mapping[str, object], query: Mapping[str, object]
    ) -> dict[str, object] | None:
        effects = {
            str(item["interest_id"]): str(item["effect"])
            for item in query.get("scenario_effects", [])
        }
        explicit_options = set(map(str, query.get("option_ids", [])))
        domains = set(map(str, query.get("domain_ids", [])))
        selected = set(map(str, query.get("selected_structure_ids", [])))
        current_year = datetime.now(timezone.utc).year
        candidates = []
        weight = {"advances": 1.0, "neutral": 0.0, "constrains": -1.0, "threatens": -1.0}
        for raw in artifact.get("orientation_index", []):
            item = copy.deepcopy(dict(raw))
            orientation_id = str(item["orientation_id"])
            if selected and orientation_id not in selected:
                continue
            protected = str(item["protected_interest_id"])
            cost = str(item["accepted_cost_id"])
            pair = {protected, cost}
            mapped = pair <= set(effects)
            explicit = pair <= explicit_options
            if not mapped and not explicit:
                continue
            if query.get("role") == "private" and not any(
                "private" in str(value) or "personal" in str(value)
                for value in item.get("role_scope", [])
            ):
                continue
            evidence_end = str(dict(item["temporal_scope"]).get("end", ""))
            evidence_year = int(evidence_end[:4]) if re.match(r"^\d{4}", evidence_end) else None
            target_years = list(map(int, query.get("mentioned_years", [])))
            if query.get("current_time_marker"):
                target_years.append(current_year)
            if query.get("relative_future_marker"):
                continue
            if evidence_year is not None and target_years and max(target_years) > evidence_year:
                continue
            if item.get("status") != "cross_domain_public_preference" and not (
                domains & set(map(str, item.get("primary_domains", [])))
            ):
                continue
            counter_domains = set(map(str, item.get("counterevidence_domains", [])))
            if item.get("counterevidence_event_ids") and (
                not domains or bool(domains & counter_domains)
            ):
                continue
            alignment = (
                weight.get(effects.get(protected, "neutral"), 0.0)
                - weight.get(effects.get(cost, "neutral"), 0.0)
                if mapped
                else 0.5
            )
            support = min(
                0.75,
                0.35
                + 0.08 * min(3, int(item["independent_source_count"]))
                + 0.05 * min(3, int(item["domain_count"]))
                + (0.08 if mapped else 0.03),
            )
            candidates.append((abs(alignment), support, orientation_id, alignment, item))
        if not candidates:
            return None
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2]))
        _, support, _, alignment, orientation = candidates[0]
        protected = str(orientation["protected_interest_id"])
        cost = str(orientation["accepted_cost_id"])
        if alignment > 0:
            stance = "support"
        elif alignment < 0:
            stance = "oppose"
        else:
            stance = "conditional_support"
        Chinese = bool(re.search(r"[\u4e00-\u9fff]", str(query["query"])))
        protected_label = (
            str(INTERESTS[protected]["label_zh"]) if Chinese else protected.replace("_", " ")
        )
        cost_label = str(INTERESTS[cost]["label_zh"]) if Chinese else cost.replace("_", " ")
        if Chinese:
            if stance == "support":
                statement = f"我更可能支持这个方向，因为它更符合{protected_label}，即使会牺牲一部分{cost_label}。"
            elif stance == "oppose":
                statement = f"我更可能反对这个方向，因为它损害了{protected_label}，却把{cost_label}放在了前面。"
            else:
                statement = f"我会先看具体条件，但在二者冲突时更可能优先考虑{protected_label}而不是{cost_label}。"
        else:
            if stance == "support":
                statement = f"I would support this direction because it advances {protected_label}, even at some cost to {cost_label}."
            elif stance == "oppose":
                statement = f"I would oppose this direction because it threatens {protected_label} while prioritizing {cost_label}."
            else:
                statement = f"I would look at the conditions, but I would generally prioritize {protected_label} over {cost_label}."
        return {
            "stance": stance,
            "statement": statement,
            "protected_interest_id": protected,
            "accepted_cost_id": cost,
            "evidence_event_ids": list(map(str, orientation["supporting_event_ids"])),
            "reasons": [
                f"The orientation appears in {orientation['independent_source_count']} independent source lineages across {orientation['domain_count']} recorded domains."
            ],
            "support": support,
            "orientation": orientation,
            "projection_kind": "reviewed_tradeoff_pair",
            "applicability": "matched_repeated_public_orientation",
        }

    @staticmethod
    def _value_orientation_projection(
        artifact: Mapping[str, object], query: Mapping[str, object]
    ) -> dict[str, object] | None:
        effects = {
            str(item["interest_id"]): str(item["effect"])
            for item in query.get("scenario_effects", [])
        }
        domains = set(map(str, query.get("domain_ids", [])))
        selected = set(map(str, query.get("selected_structure_ids", [])))
        candidates = []
        for raw in artifact.get("value_orientation_index", []):
            item = copy.deepcopy(dict(raw))
            orientation_id = str(item["orientation_id"])
            if selected and orientation_id not in selected:
                continue
            interest_id = str(item["interest_id"])
            effect = effects.get(interest_id, "neutral")
            if effect == "neutral":
                continue
            if query.get("role") == "private":
                continue
            item_domains = set(map(str, item.get("primary_domains", [])))
            cross_domain = item.get("status") == "cross_domain_public_salience"
            if not cross_domain and not selected and (
                not domains or not (domains & item_domains)
            ):
                continue
            stance = "support" if effect == "advances" else "oppose"
            support = min(0.58, float(item.get("support", 0.0)) * 0.9)
            candidates.append((support, orientation_id, stance, effect, item))
        if not candidates:
            return None
        candidates.sort(key=lambda value: (-value[0], value[1]))
        support, _, stance, effect, orientation = candidates[0]
        interest_id = str(orientation["interest_id"])
        Chinese = bool(re.search(r"[\u4e00-\u9fff]", str(query["query"])))
        label = (
            str(INTERESTS[interest_id]["label_zh"])
            if Chinese
            else interest_id.replace("_", " ")
        )
        if Chinese:
            statement = (
                f"仅根据已记录的公开表达，我会支持这个方向，因为它有利于{label}。"
                if stance == "support"
                else f"仅根据已记录的公开表达，我会反对这个方向，因为它损害或限制了{label}。"
            )
        else:
            statement = (
                f"Based only on the recorded public evidence, I would support this direction because it advances {label}."
                if stance == "support"
                else f"Based only on the recorded public evidence, I would oppose this direction because it threatens or constrains {label}."
            )
        return {
            "stance": stance,
            "statement": statement,
            "protected_interest_id": interest_id,
            "accepted_cost_id": "not_identified",
            "evidence_event_ids": list(
                map(str, orientation["supporting_event_ids"])
            ),
            "reasons": [
                "This is a low-confidence projection from explicit public salience; it is not a private value claim."
            ],
            "support": support,
            "orientation": orientation,
            "projection_kind": "event_public_salience",
            "applicability": "matched_event_public_salience_low_confidence",
            "scenario_effect": effect,
        }

    @staticmethod
    def _object_evaluation_projection(
        artifact: Mapping[str, object], query: Mapping[str, object]
    ) -> dict[str, object] | None:
        """Wide evaluation problems: derive a multi-dimensional direction from
        evaluation-class tendency atoms instead of falling back to retrieval."""
        if query.get("question_scope") != "wide":
            return None
        # 有利益取舍结构（scenario_effects）时走取向投影，不做对象评价投影
        if query.get("scenario_effects"):
            return None
        target = str(query.get("target_entity", "")).strip()
        domains = set(map(str, query.get("domain_ids", [])))
        # 从单事件偏好原子（preference_atoms）读取评价类倾向，单来源也能推导
        atoms = list(
            dict(artifact.get("reviewed_public_model", {})).get("preference_atoms", [])
        )
        grouped: dict[str, dict[str, object]] = {}
        for raw in atoms:
            item = dict(raw)
            tendency_type = str(item.get("tendency_type", ""))
            direction = str(item.get("direction", ""))
            if tendency_type not in EVALUATION_TENDENCY_TYPES:
                continue
            if direction not in {"support", "oppose", "mixed", "conditional_support"}:
                continue
            atom_target = str(item.get("target", "")).strip()
            if target and atom_target and (
                target.casefold() not in atom_target.casefold()
                and atom_target.casefold() not in target.casefold()
                and _similarity(target, atom_target) < 0.4
            ):
                continue
            atom_domains = set(map(str, item.get("domain_tags", [])))
            if domains and atom_domains and not (domains & atom_domains):
                continue
            interest_id = str(item.get("protected_interest_id", ""))
            slot = grouped.setdefault(
                interest_id,
                {"directions": [], "evidence_event_ids": [], "tendency_types": set()},
            )
            slot["directions"].append(direction)
            slot["evidence_event_ids"].append(str(item.get("event_frame_id", "")))
            slot["tendency_types"].add(tendency_type)
        dimensions = [
            {
                "interest_id": interest_id,
                "directions": slot["directions"],
                "tendency_types": sorted(slot["tendency_types"]),
                "evidence_event_ids": sorted(set(slot["evidence_event_ids"])),
                "independent_source_count": len(set(slot["evidence_event_ids"])),
            }
            for interest_id, slot in sorted(grouped.items())
        ]
        Chinese = bool(re.search(r"[\u4e00-\u9fff]", str(query["query"])))
        if not dimensions:
            statement = (
                (
                    f"关于{target}，我已有的公开表态不足以形成可靠的总体评价，"
                    "因此我不会把某一条孤立说法当成完整看法。"
                )
                if target and Chinese
                else (
                    f"About {target}, my recorded public statements are not enough to form a reliable overall evaluation, "
                    "so I would not turn a single isolated remark into a full view."
                )
                if target
                else (
                    "我已有的公开表态不足以形成可靠的总体评价，"
                    "因此我不会把某一条孤立说法当成完整看法。"
                )
                if Chinese
                else "My recorded public statements are not enough to form a reliable overall evaluation, so I would not turn a single isolated remark into a full view."
            )
            return {
                "stance": "neutral",
                "statement": statement,
                "dimensions": [],
                "evidence_event_ids": [],
                "support": 0.0,
                "projection_kind": "object_evaluation_insufficient_evidence",
                "applicability": "wide_evaluation_insufficient_evaluation_tendencies",
                "reasons": [],
                "target": target,
            }
        overall_support = 0
        overall_oppose = 0
        for dim in dimensions:
            dirs = dim["directions"]
            dim["oppose_count"] = int(dirs.count("oppose"))
            dim["support_count"] = int(dirs.count("support"))
            dim["mixed_count"] = int(dirs.count("mixed"))
            dim["stance"] = (
                "oppose"
                if dim["oppose_count"] > dim["support_count"]
                else "support"
                if dim["support_count"] > dim["oppose_count"]
                else "mixed"
            )
            overall_support += dim["support_count"]
            overall_oppose += dim["oppose_count"]
        if overall_oppose > overall_support:
            stance = "oppose"
        elif overall_support > overall_oppose:
            stance = "support"
        else:
            stance = "mixed"
        lines = []
        for dim in dimensions:
            label = str(INTERESTS[dim["interest_id"]]["label_zh"]) if Chinese else str(dim["interest_id"]).replace("_", " ")
            direction_word = {
                "oppose": "更可能持批评/反对",
                "support": "更可能持支持",
                "mixed": "态度混合",
            }.get(str(dim["stance"]), "态度不明") if Chinese else str(dim["stance"])
            lines.append(f"{label}：{direction_word}")
        prefix = f"关于{target}，我基于已记录的公开表态，主要落在几个维度：" if target and Chinese else f"About {target}, based on my recorded public statements, the view falls on several dimensions:" if target else ("我基于已记录的公开表态，主要落在几个维度：" if Chinese else "Based on my recorded public statements, the view falls on several dimensions:")
        statement = prefix + "\n" + "\n".join(lines) + (
            "\n\n（这些是历史公开表态，不构成对当前情境的确定预测。）" if Chinese else "\n\n(These are historical public statements, not a determinate prediction for the current situation.)"
        )
        return {
            "stance": stance,
            "statement": statement,
            "dimensions": dimensions,
            "evidence_event_ids": sorted(
                {eid for dim in dimensions for eid in dim["evidence_event_ids"]}
            ),
            "support": 0.0,
            "projection_kind": "object_evaluation_projection",
            "applicability": "wide_object_evaluation_from_public_tendencies",
            "reasons": [],
            "target": target,
        }

    def evaluate(
        self,
        artifact: Mapping[str, object],
        holdout_sources: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        self.verify(artifact)
        training_ids = {
            str(item.get("source_id", ""))
            for item in dict(artifact["reviewed_public_model"]).get(
                "source_identities", []
            )
        }
        leakage = sorted(
            {
                str(item.get("source_id", ""))
                for item in holdout_sources
                if str(item.get("source_id", "")) in training_ids
            }
        )
        if leakage:
            return {
                "status": "invalid_holdout_leakage",
                "sample_count": 0,
                "holdout_leakage_source_ids": leakage,
                "accuracy_claim": "none",
            }
        return {
            "status": "not_assessed_full_conversation_holdout_required",
            "sample_count": 0,
            "coverage": 0.0,
            "covered_direction_accuracy": None,
            "accuracy_claim": "none",
        }

    @staticmethod
    def _result(
        artifact: Mapping[str, object],
        *,
        answer_status: str,
        speech_act: str,
        stance: str,
        claims: Sequence[str],
        reasons: Sequence[str],
        uncertainties: Sequence[str],
        evidence_event_ids: Sequence[str],
        response_basis: Mapping[str, object],
        applicability: str,
        support: float,
        trace: Mapping[str, object],
        ordinary_text: str = "",
        refusal_reasons: Sequence[str] = (),
    ) -> dict[str, object]:
        def units(kind: str, values: Sequence[str]) -> list[dict[str, object]]:
            return [
                {
                    "id": f"{kind}-v5-{_hash([kind, value])[:16]}",
                    "text": str(value),
                    "evidence_event_id": (
                        str(evidence_event_ids[min(index, len(evidence_event_ids) - 1)])
                        if evidence_event_ids
                        else ""
                    ),
                    "probability": 0.0,
                }
                for index, value in enumerate(values)
            ]
        structured: dict[str, object] = {
            "schema_version": PREDICTION_SCHEMA_V5,
            "person_id": str(artifact["person_id"]),
            "speech_act": {"label": speech_act, "probability": 0.0},
            "speech_act_distribution": [],
            "stance": {"label": stance, "probability": 0.0},
            "stance_distribution": [],
            "claims": units("claim", claims),
            "reasons": units("reason", reasons),
            "memories": [],
            "uncertainties": units("uncertainty", uncertainties),
            "answer_status": answer_status,
            "confidence": round(float(support), 6),
            "confidence_kind": "evidence_support_not_accuracy_probability",
            "applicability": applicability,
            "refusal_reasons": list(map(str, refusal_reasons)),
            "evidence_refs": [],
            "evidence_event_ids": list(map(str, evidence_event_ids)),
            "response_basis": copy.deepcopy(dict(response_basis)),
            "active_components": copy.deepcopy(list(artifact["active_components"])),
            "components": copy.deepcopy(list(artifact["components"])),
            "model_version": f"{artifact['person_id']}-simulation-v5-{artifact['version']}",
            "model_validity": "implemented_exploratory_accuracy_not_assessed",
            "valid_scope": copy.deepcopy(dict(artifact.get("scope", {}))),
        }
        contract = {
            "schema_version": FROZEN_CONTRACT_V5,
            "speech_act": speech_act,
            "stance": stance,
            "answer_status": answer_status,
            "refusal_status": "refused" if refusal_reasons else "not_refused",
            "ordinary_dialogue_text": ordinary_text,
            "claims": [{"id": item["id"], "text": item["text"]} for item in structured["claims"]],
            "reasons": [{"id": item["id"], "text": item["text"]} for item in structured["reasons"]],
            "memories": [],
            "uncertainties": [
                {"id": item["id"], "text": item["text"]}
                for item in structured["uncertainties"]
            ],
            "protected_entities": [],
            "protected_numbers": sorted(
                set(re.findall(r"\b\d+(?:[.,]\d+)*\b", " ".join([*claims, *reasons])))
            ),
            "protected_dates": sorted(
                set(re.findall(r"\b\d{4}(?:-\d{2}-\d{2})?\b", " ".join([*claims, *reasons])))
            ),
            "protected_quotes": [],
            "evidence_refs": list(map(str, evidence_event_ids)),
            "confidence": structured["confidence"],
            "style_mode": "interview_public",
        }
        structured["renderer_contract_digest"] = _hash(contract)
        content_digest = _hash(structured)
        structured["content_digest"] = content_digest
        return {
            "schema_version": PREDICTION_SCHEMA_V5,
            "status": "answered",
            "answer_status": answer_status,
            "structured_prediction": structured,
            "renderer_contract": contract,
            "content_digest": content_digest,
            "renderer_contract_digest": _hash(contract),
            "prediction_trace": copy.deepcopy(dict(trace)),
        }
