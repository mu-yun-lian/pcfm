from __future__ import annotations

import copy
import math
import re
from typing import Mapping, Sequence

from .response_prediction import (
    APPLICABILITY_STOPWORDS,
    SPEECH_ACTS,
    STANCES,
    ResponsePredictionError,
    ResponsePredictionKernel,
    _feature_vector,
    applicability_similarity,
    canonical_hash,
    classify_event_type,
    classify_event_types,
    lexical_similarity,
    tokens,
)


MODEL_SCHEMA_V2 = "pcfm-unified-response-model-v2"
PREDICTION_SCHEMA_V2 = "pcfm-structured-response-v2"
KERNEL_ID_V2 = "pcfm-response-kernel-v2"
FROZEN_CONTRACT_V3 = "pcfm-frozen-content-contract-v3"
ANSWER_STATUSES = frozenset(
    {
        "ordinary_dialogue",
        "direct_answer",
        "composite_answer",
        "partial_answer",
        "tendency_answer",
        "general_assisted",
        "clarification_needed",
        "refused",
    }
)


def _normalized_text(value: str) -> str:
    return " ".join(tokens(value))


def _unit_id(kind: str, text: str) -> str:
    return f"{kind}-{canonical_hash([kind, _normalized_text(text)])[:16]}"


def _ordinary_dialogue(text: str) -> tuple[str, str] | None:
    normalized = _normalized_text(text)
    compact = re.sub(r"[\s，。！？,.!?]+", "", str(text).casefold())
    if compact in {"你好", "您好", "嗨", "hello", "hi", "hey"}:
        return "greeting", "你好。你想从哪件事开始聊？"
    if compact in {"谢谢", "感谢", "thankyou", "thanks", "thx"}:
        return "thanks", "不客气。"
    if compact in {"接着说", "继续", "请继续", "goon", "continue"}:
        return "continue", "可以，我们接着谈刚才的话题。你希望我展开哪一部分？"
    if normalized in {"ok", "okay", "好的", "好"}:
        return "acknowledge", "好。"
    return None


def _needs_reference_resolution(text: str) -> bool:
    compact = _normalized_text(text)
    short = len(tokens(text)) <= 6
    return bool(
        re.fullmatch(r"(?:why|what about(?: that| it)?|and that|that|it|this)[ ?]*", compact)
        or re.search(r"\b(?:about it|about that|do about it)\b", compact)
        or (short and re.search(r"为什么|那件事|这个|那个|它|这件事|那呢", str(text)))
    )


def _protected_values(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    entities = sorted(
        set(
            re.findall(
                r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)+\b", text
            )
        )
    )
    numbers = sorted(set(re.findall(r"\b\d+(?:[.,]\d+)*(?:-%|%)?", text)))
    dates = sorted(set(re.findall(r"\b\d{4}(?:-\d{2}-\d{2})?\b", text)))
    quotes = sorted(
        set(
            value
            for pair in re.findall(r'"([^"\n]+)"|“([^”\n]+)”', text)
            for value in pair
            if value
        )
    )
    return entities, numbers, dates, quotes


def _public_response_views(
    events: Sequence[Mapping[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    episodes: list[dict[str, object]] = []
    tendency_groups: dict[tuple[str, str], dict[str, object]] = {}
    overall_groups: dict[str, dict[str, object]] = {}
    knowledge: dict[str, dict[str, object]] = {}
    for event in events:
        event_id = str(event.get("event_id", ""))
        atom = dict(event.get("event_atom") or {})
        event_type = str(atom.get("event_type", "general"))
        completeness = dict(atom.get("completeness") or {})
        allowed_uses = set(map(str, completeness.get("allowed_uses", [])))
        source_lineage = str(
            event.get("near_duplicate_of") or event.get("source_id") or event_id
        )
        evidence_weight = {
            "verbatim_transcript": 1.0,
            "verified_translation": 0.85,
        }.get(str(event.get("content_authenticity", "")), 0.6)
        episodes.append(
            {
                "episode_id": f"episode-{canonical_hash([event_id, event.get('content_hash')])[:16]}",
                "event_id": event_id,
                "event_type": event_type,
                "topic_terms": list(atom.get("topic_terms", [])),
                "trigger": str(event.get("trigger", "")),
                "public_response": str(event.get("actual_response", "")),
                "claims": list(event.get("claims", [])),
                "reasons": list(event.get("reasons", [])),
                "context": {
                    "occasion": str(event.get("occasion", "")),
                    "observed_at": str(event.get("observed_at", "")),
                    "full_context": str(event.get("full_context", "")),
                },
                "completeness": completeness,
                "allowed_uses": sorted(allowed_uses),
                "source_lineage": source_lineage,
                "evidence_weight": evidence_weight,
                "confidence_kind": "source_evidence_strength_not_prediction_accuracy",
            }
        )
        for raw in event.get("tendency_atoms", []):
            if not isinstance(raw, Mapping):
                continue
            if "conditional_tendency" not in allowed_uses:
                continue
            stance = str(raw.get("stance", "neutral"))
            key = (event_type, stance)
            group = tendency_groups.setdefault(
                key,
                {
                    "tendency_id": f"tendency-{canonical_hash(key)[:16]}",
                    "kind": "conditional_public_response_pattern",
                    "event_type": event_type,
                    "event_types": list(
                        map(str, atom.get("event_types") or [event_type])
                    ),
                    "stance": stance,
                    "supporting_event_ids": [],
                    "representative_outcomes": [],
                    "targets": [],
                    "conditions": [],
                    "tradeoffs": [],
                    "exceptions": [],
                    "counterevidence_event_ids": [],
                    "supporting_domains": [],
                    "source_lineages": [],
                    "source_weights": {},
                    "temporal_points": [],
                    "status": "observed_pattern_not_inner_value",
                },
            )
            group["supporting_event_ids"].append(event_id)
            group["supporting_domains"].append(event_type)
            group["source_lineages"].append(source_lineage)
            group["source_weights"][source_lineage] = max(
                evidence_weight,
                float(group["source_weights"].get(source_lineage, 0.0)),
            )
            observed_at = str(event.get("observed_at", "")).strip()
            if observed_at:
                group["temporal_points"].append(observed_at)
            for field in ("target", "conditions", "tradeoffs", "exceptions"):
                values = raw.get(field, [])
                if field == "target":
                    values = [values]
                    destination = "targets"
                else:
                    destination = field
                for value in values if isinstance(values, (list, tuple)) else []:
                    clean = str(value).strip()
                    if clean and clean not in group[destination]:
                        group[destination].append(clean)
            outcome = str(raw.get("expressed_outcome", "")).strip()
            if outcome and outcome not in group["representative_outcomes"]:
                group["representative_outcomes"].append(outcome)
            overall = overall_groups.setdefault(
                stance,
                {
                    "tendency_id": f"overall-tendency-{canonical_hash(stance)[:16]}",
                    "kind": "overall_public_response_pattern",
                    "event_type": "overall",
                    "event_types": [],
                    "stance": stance,
                    "supporting_event_ids": [],
                    "supporting_domains": [],
                    "representative_outcomes": [],
                    "targets": [],
                    "conditions": [],
                    "tradeoffs": [],
                    "exceptions": [],
                    "counterevidence_event_ids": [],
                    "source_lineages": [],
                    "source_weights": {},
                    "temporal_points": [],
                    "status": "aggregated_public_pattern_not_inner_value",
                },
            )
            overall["supporting_event_ids"].append(event_id)
            overall["supporting_domains"].append(event_type)
            overall["event_types"] = sorted(
                set(overall["event_types"])
                | set(map(str, atom.get("event_types") or [event_type]))
            )
            overall["source_lineages"].append(source_lineage)
            overall["source_weights"][source_lineage] = max(
                evidence_weight,
                float(overall["source_weights"].get(source_lineage, 0.0)),
            )
            if observed_at:
                overall["temporal_points"].append(observed_at)
            for source_field, destination in (
                ("target", "targets"),
                ("conditions", "conditions"),
                ("tradeoffs", "tradeoffs"),
                ("exceptions", "exceptions"),
            ):
                values = raw.get(source_field, [])
                if source_field == "target":
                    values = [values]
                for value in values if isinstance(values, (list, tuple)) else []:
                    clean = str(value).strip()
                    if clean and clean not in overall[destination]:
                        overall[destination].append(clean)
            if outcome and outcome not in overall["representative_outcomes"]:
                overall["representative_outcomes"].append(outcome)
        for raw in event.get("knowledge_items", []):
            if not isinstance(raw, Mapping):
                continue
            statement = str(raw.get("statement", "")).strip()
            if not statement:
                continue
            item_id = f"knowledge-{canonical_hash(statement.casefold())[:16]}"
            item = knowledge.setdefault(
                item_id,
                {
                    "knowledge_id": item_id,
                    "statement": statement,
                    "status": "publicly_used_claim_not_verified_fact",
                    "knowledge_kind": str(
                        raw.get(
                            "knowledge_kind",
                            "person_demonstrated_claim_not_verified_fact",
                        )
                    ),
                    "temporal_status": str(
                        raw.get(
                            "temporal_status",
                            completeness.get("temporal_status", "unknown"),
                        )
                    ),
                    "supporting_event_ids": [],
                    "supporting_domains": [],
                    "source_lineages": [],
                    "source_weights": {},
                },
            )
            item["supporting_event_ids"].append(event_id)
            item["supporting_domains"].append(event_type)
            item["source_lineages"].append(source_lineage)
            item["source_weights"][source_lineage] = max(
                evidence_weight,
                float(item["source_weights"].get(source_lineage, 0.0)),
            )
    tendencies = list(tendency_groups.values())
    overall_tendencies = list(overall_groups.values())

    by_domain: dict[str, list[dict[str, object]]] = {}
    for value in tendencies:
        by_domain.setdefault(str(value["event_type"]), []).append(value)
    for values in by_domain.values():
        for value in values:
            value["counterevidence_event_ids"] = sorted(
                {
                    str(event_id)
                    for other in values
                    if other["stance"] != value["stance"]
                    for event_id in other["supporting_event_ids"]
                }
            )
    for value in overall_tendencies:
        value["counterevidence_event_ids"] = sorted(
            {
                str(event_id)
                for other in overall_tendencies
                if other["stance"] != value["stance"]
                for event_id in other["supporting_event_ids"]
            }
        )

    for value in [*tendencies, *overall_tendencies]:
        value["supporting_event_ids"] = sorted(set(value["supporting_event_ids"]))
        value["supporting_domains"] = sorted(set(value["supporting_domains"]))
        value["support_count"] = len(value["supporting_event_ids"])
        value["domain_count"] = len(value["supporting_domains"])
        value["source_lineages"] = sorted(set(value["source_lineages"]))
        value["independent_source_count"] = len(value["source_lineages"])
        temporal_points = sorted(set(value.pop("temporal_points")))
        value["temporal_scope"] = {
            "start": temporal_points[0] if temporal_points else "unknown",
            "end": temporal_points[-1] if temporal_points else "unknown",
            "dated_event_count": len(temporal_points),
        }
        value["evidence_strength"] = round(
            min(
                1.0,
                sum(float(weight) for weight in value.pop("source_weights").values())
                / 3.0,
            ),
            6,
        )
        value["confidence_kind"] = "recurring_evidence_strength_not_prediction_accuracy"
        value["representative_outcomes"] = value["representative_outcomes"][:4]
        for field in ("targets", "conditions", "tradeoffs", "exceptions"):
            value[field] = value[field][:8]
        if value["counterevidence_event_ids"]:
            value["status"] = "contradicted_public_pattern_requires_context_split"
        elif value["independent_source_count"] >= 2 and value["support_count"] >= 2:
            value["status"] = "recurring_public_pattern"
        else:
            value["status"] = "single_source_or_single_event_candidate"
    for value in knowledge.values():
        value["supporting_event_ids"] = sorted(set(value["supporting_event_ids"]))
        value["supporting_domains"] = sorted(set(value["supporting_domains"]))
        value["support_count"] = len(value["supporting_event_ids"])
        value["source_lineages"] = sorted(set(value["source_lineages"]))
        value["independent_source_count"] = len(value["source_lineages"])
        value["evidence_strength"] = round(
            min(
                1.0,
                sum(float(weight) for weight in value.pop("source_weights").values())
                / 3.0,
            ),
            6,
        )
        value["confidence_kind"] = "public_use_evidence_strength_not_factual_truth"
    relations: list[dict[str, object]] = []
    for left_index, left in enumerate(episodes):
        if len(relations) >= 2000:
            break
        left_topics = set(map(str, left.get("topic_terms", [])))
        for right in episodes[left_index + 1 :]:
            right_topics = set(map(str, right.get("topic_terms", [])))
            shared_topics = sorted(left_topics & right_topics)
            same_type = left.get("event_type") == right.get("event_type")
            if not same_type and not shared_topics:
                continue
            relations.append(
                {
                    "relation_id": f"event-relation-{canonical_hash([left['event_id'], right['event_id']])[:16]}",
                    "left_event_id": left["event_id"],
                    "right_event_id": right["event_id"],
                    "relation_type": "same_event_type" if same_type else "shared_topic_terms",
                    "shared_topic_terms": shared_topics[:8],
                    "status": "observable_cross_event_link_not_causal_claim",
                }
            )
            if len(relations) >= 2000:
                break
    cross_domain = [
        {
            **copy.deepcopy(value),
            "kind": "cross_domain_recurring_public_response_pattern",
            "status": "implicit_pattern_candidate_not_inner_value",
        }
        for value in overall_tendencies
        if int(value.get("domain_count", 0)) >= 2
    ]
    return (
        episodes,
        sorted(tendencies, key=lambda item: item["tendency_id"]),
        sorted(overall_tendencies, key=lambda item: item["tendency_id"]),
        sorted(cross_domain, key=lambda item: item["tendency_id"]),
        relations,
        sorted(knowledge.values(), key=lambda item: item["knowledge_id"]),
    )


class ResponsePredictionKernelV2:
    """One deterministic PCFM court over evidence-bounded model proposals."""

    kernel_id = KERNEL_ID_V2

    def __init__(self) -> None:
        self._v1 = ResponsePredictionKernel()

    @staticmethod
    def is_ordinary_dialogue(text: str) -> bool:
        return _ordinary_dialogue(text) is not None

    @staticmethod
    def ordinary_dialogue(text: str) -> tuple[str, str] | None:
        return _ordinary_dialogue(text)

    def fit(
        self,
        *,
        person_id: str,
        version: int,
        events: Sequence[Mapping[str, object]],
        population_events: Sequence[Mapping[str, object]] = (),
        population_people: int = 0,
        scope: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        base = self._v1.fit(
            person_id=person_id,
            version=version,
            events=events,
            population_events=population_events,
            population_people=population_people,
            scope=scope,
        )
        base.pop("artifact_hash", None)
        (
            episodes,
            tendencies,
            overall_tendencies,
            cross_domain_tendencies,
            event_relations,
            knowledge,
        ) = _public_response_views(events)
        base["schema_version"] = MODEL_SCHEMA_V2
        base["kernel"] = KERNEL_ID_V2
        base["feature_schema"] = {
            **dict(base["feature_schema"]),
            "response_plan_protocol": "evidence_unit_allowlist_v1",
            "ordinary_dialogue_policy": "content_free_dialogue_manager_v1",
            "public_response_model": "episode_tendency_knowledge_v2",
            "event_atom_schema": "pcfm-response-event-v2",
            "retrieval_policy": "semantic_domain_target_gate_v2",
        }
        base["episode_bundles"] = episodes
        base["conditional_tendencies"] = tendencies
        base["overall_tendencies"] = overall_tendencies
        base["cross_domain_tendency_candidates"] = cross_domain_tendencies
        base["event_relations"] = event_relations
        base["demonstrated_knowledge"] = knowledge
        base["active_components"] = [
            *list(base["active_components"]),
            "pcfm_episode_bundle_v1",
            "pcfm_conditional_tendency_v1",
            "pcfm_overall_public_tendency_v1",
            "pcfm_cross_event_relations_v1",
            "pcfm_demonstrated_knowledge_v1",
        ]
        base["migration_policy"] = "v1_artifacts_must_be_refit_from_reviewed_sources"
        base["artifact_hash"] = canonical_hash(base)
        return base

    @staticmethod
    def verify(artifact: Mapping[str, object]) -> None:
        if artifact.get("schema_version") != MODEL_SCHEMA_V2 or artifact.get("kernel") != KERNEL_ID_V2:
            raise ResponsePredictionError("unsupported_response_model_v2_schema")
        payload = copy.deepcopy(dict(artifact))
        digest = str(payload.pop("artifact_hash", ""))
        if canonical_hash(payload) != digest:
            raise ResponsePredictionError("response_model_v2_integrity_failed")

    @staticmethod
    def reseal_for_test(artifact: Mapping[str, object]) -> dict[str, object]:
        value = copy.deepcopy(dict(artifact))
        value.pop("artifact_hash", None)
        value["artifact_hash"] = canonical_hash(value)
        return value

    @staticmethod
    def _as_v1(artifact: Mapping[str, object]) -> dict[str, object]:
        value = copy.deepcopy(dict(artifact))
        value.pop("artifact_hash", None)
        value["schema_version"] = "pcfm-unified-response-model-v1"
        value["kernel"] = "pcfm-response-kernel-v1"
        value["artifact_hash"] = canonical_hash(value)
        return value

    def evaluate(
        self, artifact: Mapping[str, object], holdout_events: Sequence[Mapping[str, object]]
    ) -> dict[str, object]:
        self.verify(artifact)
        return self._v1.evaluate(self._as_v1(artifact), holdout_events)

    def compare_baselines(
        self,
        artifact: Mapping[str, object],
        holdout_events: Sequence[Mapping[str, object]],
        *,
        wrong_person_artifacts: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        self.verify(artifact)
        return self._v1.compare_baselines(
            self._as_v1(artifact),
            holdout_events,
            wrong_person_artifacts=[self._as_v1(value) for value in wrong_person_artifacts],
        )

    @staticmethod
    def _event_units(event: Mapping[str, object]) -> list[dict[str, object]]:
        units: list[dict[str, object]] = []
        for field, kind in (
            ("claims", "claim"),
            ("reasons", "reason"),
            ("memories", "memory"),
            ("uncertainties", "uncertainty"),
        ):
            for text in event.get(field, []):
                clean = str(text).strip()
                if clean:
                    units.append(
                        {
                            "unit_id": _unit_id(kind, clean),
                            "kind": kind,
                            "text": clean,
                            "event_id": str(event.get("event_id", "")),
                            "event_content_id": f"event-content-{canonical_hash([event.get('question'), event.get('actual_response')])[:16]}",
                        }
                    )
        return units

    @staticmethod
    def _context_query(
        text: str, history: Sequence[Mapping[str, object]]
    ) -> tuple[str, list[str]]:
        if not _needs_reference_resolution(text):
            return text, []
        ids: list[str] = []
        parts = [text]
        for item in reversed(history[-8:]):
            if item.get("role") == "user" and item.get("text"):
                parts.append(str(item["text"]))
                if item.get("message_id"):
                    ids.append(str(item["message_id"]))
                break
        return " ".join(parts), ids

    def recall(
        self,
        artifact: Mapping[str, object],
        *,
        text: str,
        history: Sequence[Mapping[str, object]],
        limit: int = 6,
    ) -> dict[str, object]:
        self.verify(artifact)
        query, resolved_ids = self._context_query(text, history)
        query_event_types = set(classify_event_types(query))
        query_terms = {
            value
            for value in tokens(query)
            if value not in APPLICABILITY_STOPWORDS and len(value) > 1
        }
        reference_followup = bool(resolved_ids)
        candidates: list[dict[str, object]] = []
        for event in artifact["events"]:
            question = str(event.get("question", ""))
            atom = dict(event.get("event_atom") or {})
            event_types = set(
                map(str, atom.get("event_types") or [atom.get("event_type", "general")])
            )
            event_terms = set(map(str, atom.get("topic_terms", [])))
            searchable = " ".join(
                [
                    question,
                    str(event.get("actual_response", "")),
                    " ".join(str(value) for value in event.get("claims", [])),
                    " ".join(str(value) for value in event.get("reasons", [])),
                    str(atom.get("event_type", "")),
                    " ".join(str(value) for value in atom.get("topic_terms", [])),
                ]
            )
            current = lexical_similarity(text, searchable)
            context = lexical_similarity(query, searchable)
            applicability = applicability_similarity(query, searchable)
            exact = _normalized_text(text) == _normalized_text(question)
            score = max(
                current,
                (0.25 * current + 0.75 * context)
                if reference_followup
                else (0.7 * current + 0.3 * context),
            )
            domain_match = bool(query_event_types & event_types)
            topic_overlap = len(query_terms & event_terms) / max(
                1, len(query_terms | event_terms)
            )
            general_scope = query_event_types == {"general"} and event_types == {"general"}
            eligible_same_event = bool(
                exact
                or current >= 0.78
                or (
                    reference_followup
                    and score >= 0.30
                    and applicability >= 0.30
                )
                or (
                    (domain_match or topic_overlap >= 0.12 or general_scope)
                    and score >= (0.15 if domain_match else 0.18)
                    and applicability >= (0.09 if domain_match else 0.18)
                )
            )
            candidates.append(
                {
                    "event": copy.deepcopy(dict(event)),
                    "units": self._event_units(event),
                    "score": score,
                    "applicability_support": applicability,
                    "direct_match": bool(exact or current >= 0.78),
                    "eligible_same_event": eligible_same_event,
                    "scope_gate": {
                        "query_event_types": sorted(query_event_types),
                        "event_types": sorted(event_types),
                        "domain_match": domain_match,
                        "topic_overlap": round(topic_overlap, 6),
                    },
                    "event_content_id": f"event-content-{canonical_hash([event.get('question'), event.get('actual_response')])[:16]}",
                }
            )
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["event_content_id"]),
            )
        )
        return {
            "query": query,
            "query_analysis": {
                "event_types": sorted(query_event_types),
                "target_terms": sorted(query_terms)[:24],
                "context_message_ids": resolved_ids,
            },
            "resolved_context_message_ids": resolved_ids,
            "candidates": candidates[:limit],
        }

    @staticmethod
    def candidate_payload(recall: Mapping[str, object]) -> list[dict[str, object]]:
        return [
            {
                "event_content_id": item["event_content_id"],
                "question": item["event"].get("question", ""),
                "occasion": item["event"].get("occasion", ""),
                "observed_at": item["event"].get("observed_at", ""),
                "units": [
                    {key: unit[key] for key in ("unit_id", "kind", "text")}
                    for unit in item["units"]
                ],
            }
            for item in recall.get("candidates", [])
            if item.get("eligible_same_event")
        ]

    def predict(
        self,
        artifact: Mapping[str, object],
        *,
        text: str,
        history: Sequence[Mapping[str, object]],
        conversation_context: Mapping[str, object] | None = None,
        proposed_plans: Sequence[Mapping[str, object]] = (),
        planner_trace: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.verify(artifact)
        clean = str(text).strip()
        if not clean:
            raise ResponsePredictionError("message_required")
        ordinary = _ordinary_dialogue(clean)
        context_records = [dict(value) for value in history[-8:] if value.get("role") in {"user", "assistant"}]
        common_trace = {
            "retrieval_is_candidate_only": True,
            "generative_content_calls": 0,
            "context_digest": canonical_hash(
                [
                    clean,
                    [
                        (
                            item.get("message_id"),
                            item.get("role"),
                            item.get("context_role"),
                            item.get("text"),
                        )
                        for item in context_records
                    ],
                ]
            ),
            "context_used": {
                "message_ids": [
                    str(item.get("message_id"))
                    for item in context_records
                    if item.get("message_id")
                ],
                "turn_count": len(context_records),
                "generated_context_count": sum(
                    item.get("context_role") == "model_generated_context"
                    for item in context_records
                ),
            },
            "conversation_context": copy.deepcopy(
                dict(conversation_context or {})
            ),
        }
        vector = _feature_vector(clean, context_records, conversation_context)
        speech_distribution = self._v1._head_predict(
            dict(artifact["speech_act_head"]), SPEECH_ACTS, vector
        )
        stance_distribution = self._v1._head_predict(
            dict(artifact["stance_head"]), STANCES, vector
        )
        if ordinary:
            dialogue_act, safe_text = ordinary
            structured = self._structured(
                artifact,
                speech_distribution,
                stance_distribution,
                answer_status="ordinary_dialogue",
                claims=[], reasons=[], memories=[], uncertainties=[],
                evidence_refs=[], evidence_event_ids=[],
                confidence=1.0,
                applicability="ordinary_dialogue_content_free",
                refusal_reasons=[],
            )
            contract = self._contract(structured, ordinary_text=safe_text)
            return self._result(
                structured,
                contract,
                status="answered",
                trace={**common_trace,
                    "kernel": KERNEL_ID_V2,
                    "dialogue_act": dialogue_act,
                    "retrieval_calls": 0,
                    "planner_calls": 0,
                    "resolved_context_turns": 0,
                    "selected_event_ids": [],
                    "selected_event_content_ids": [],
                    "planner": dict(planner_trace or {}),
                },
            )
        recall = self.recall(artifact, text=clean, history=context_records)
        candidates = list(recall["candidates"])
        resolved_ids = list(recall["resolved_context_message_ids"])
        if _needs_reference_resolution(clean) and not resolved_ids:
            structured = self._structured(
                artifact, speech_distribution, stance_distribution,
                answer_status="clarification_needed",
                claims=[], reasons=[], memories=[],
                uncertainties=[self._unit("uncertainty", "需要明确你指的是前面的哪件事。", "")],
                evidence_refs=[], evidence_event_ids=[], confidence=0.0,
                applicability="missing_conversation_reference",
                refusal_reasons=["missing_conversation_reference"],
            )
            return self._result(structured, self._contract(structured), status="clarification", trace={**common_trace, "kernel": KERNEL_ID_V2, "retrieval_calls": 1, "planner_calls": 0, "resolved_context_turns": 0, "selected_event_ids": [], "selected_event_content_ids": [], "planner": dict(planner_trace or {})})
        supported = [item for item in candidates if item.get("eligible_same_event")]
        direct = next((item for item in supported if item["direct_match"]), None)
        if not supported:
            conditional_tendencies = list(artifact.get("conditional_tendencies", []))
            overall_tendencies = list(artifact.get("overall_tendencies", []))
            if conditional_tendencies or overall_tendencies:
                query_event_types = classify_event_types(str(recall["query"]))
                query_event_type = query_event_types[0]
                scoped = [
                    dict(value)
                    for value in conditional_tendencies
                    if set(map(str, value.get("event_types") or [value.get("event_type")]))
                    & set(query_event_types)
                ]
                pool = scoped or [dict(value) for value in overall_tendencies]
                if not pool:
                    pool = [dict(value) for value in conditional_tendencies]
                ranked_tendencies = sorted(
                    pool,
                    key=lambda value: (
                        bool(value.get("counterevidence_event_ids")),
                        -int(value.get("support_count", 0)),
                        -int(value.get("independent_source_count", 0)),
                        str(value.get("tendency_id", "")),
                    ),
                )
                selected = ranked_tendencies[0]
                competing = [
                    value
                    for value in ranked_tendencies[1:]
                    if value.get("stance") != selected.get("stance")
                    and int(value.get("support_count", 0))
                    >= max(1, int(selected.get("support_count", 0)) - 1)
                ]
                if competing:
                    selected = {
                        "tendency_id": f"tendency-conflict-{canonical_hash([selected, competing])[:12]}",
                        "kind": "conflicting_conditional_public_response_patterns",
                        "event_type": query_event_type if scoped else "cross_domain",
                        "stance": "mixed",
                        "supporting_event_ids": sorted(
                            {
                                str(event_id)
                                for value in [selected, *competing]
                                for event_id in value.get("supporting_event_ids", [])
                            }
                        ),
                        "representative_outcomes": [
                            str(outcome)
                            for value in [selected, *competing]
                            for outcome in value.get("representative_outcomes", [])
                        ][:6],
                        "support_count": sum(
                            int(value.get("support_count", 0))
                            for value in [selected, *competing]
                        ),
                        "status": "context_split_required_not_averaged",
                    }
                evidence_event_ids = [
                    str(value)
                    for value in selected.get("supporting_event_ids", [])
                ]
                selected_knowledge = self._relevant_knowledge(
                    artifact,
                    query=str(recall["query"]),
                    event_types=query_event_types,
                )
                uncertainty = self._unit(
                    "uncertainty",
                    (
                        "没有找到足够相似的具体事件；以下依据同领域公开倾向作条件性推断。"
                        if scoped
                        else "没有找到同类事件；以下只能依据人物整体公开倾向作条件性推断。"
                    ),
                    "",
                )
                structured = self._structured(
                    artifact,
                    speech_distribution,
                    stance_distribution,
                    answer_status="tendency_answer",
                    claims=[],
                    reasons=[],
                    memories=[],
                    uncertainties=[uncertainty],
                    evidence_refs=[],
                    evidence_event_ids=evidence_event_ids,
                    confidence=min(
                        0.48,
                        0.12 + 0.36 * float(selected.get("evidence_strength", 0.0)),
                    ),
                    applicability="overall_public_tendency_only",
                    refusal_reasons=[],
                )
                structured["response_basis"] = {
                    "path": "domain_tendency" if scoped else "overall_tendency",
                    "person_prediction_status": "conditional_inference",
                    "query_event_type": query_event_type,
                    "scope_match": "same_event_type" if scoped else "cross_domain_only",
                    "selected_tendency": selected,
                    "selected_demonstrated_knowledge": selected_knowledge,
                    "knowledge_boundary": "person_demonstrated_claim_not_verified_fact",
                    "evidence_strength": float(selected.get("evidence_strength", 0.0)),
                    "competing_tendency_count": len(competing),
                    "external_knowledge_policy": "allowed_if_disclosed_not_person_memory",
                }
                return self._result(
                    structured,
                    self._contract(structured),
                    status="answered",
                    trace={
                        **common_trace,
                        "kernel": KERNEL_ID_V2,
                        "retrieval_calls": 1,
                        "planner_calls": int(bool(planner_trace)),
                        "resolved_context_turns": len(resolved_ids),
                        "selected_event_ids": evidence_event_ids,
                        "selected_event_content_ids": [],
                        "prediction_path": "domain_tendency" if scoped else "overall_tendency",
                        "planner": dict(planner_trace or {}),
                    },
                )
            structured = self._structured(
                artifact, speech_distribution, stance_distribution,
                answer_status="refused", claims=[], reasons=[], memories=[], uncertainties=[],
                evidence_refs=[], evidence_event_ids=[], confidence=0.0,
                applicability="out_of_domain", refusal_reasons=["out_of_domain"],
            )
            return self._result(structured, self._contract(structured), status="refused", trace={**common_trace, "kernel": KERNEL_ID_V2, "retrieval_calls": 1, "planner_calls": int(bool(planner_trace)), "resolved_context_turns": len(resolved_ids), "selected_event_ids": [], "selected_event_content_ids": [], "planner": dict(planner_trace or {})})
        unit_lookup = {
            unit["unit_id"]: {**unit, "support": float(item["score"]), "applicability_support": float(item["applicability_support"])}
            for item in supported
            for unit in item["units"]
        }
        proposed_ids: set[str] = set()
        for plan in proposed_plans:
            for key in ("claim_ids", "reason_ids", "memory_ids", "uncertainty_ids"):
                proposed_ids.update(str(value) for value in plan.get(key, []))
        proposed_ids.intersection_update(unit_lookup)
        if direct:
            chosen_events = [direct]
            chosen_ids = {unit["unit_id"] for unit in direct["units"]}
        else:
            best_score = float(supported[0]["score"])
            relative_floor = 0.75 if resolved_ids else 0.45
            chosen_events = [
                item
                for item in supported
                if float(item["score"]) >= max(0.08, best_score * relative_floor)
            ][:3]
            allowed_ids = {unit["unit_id"] for item in chosen_events for unit in item["units"]}
            chosen_ids = (proposed_ids & allowed_ids) or allowed_ids
        ranked = sorted(
            (unit_lookup[value] for value in chosen_ids if value in unit_lookup),
            key=lambda unit: (-float(unit["support"]), str(unit["unit_id"])),
        )
        top_speech = speech_distribution[0]
        top_stance = stance_distribution[0]
        claims = [self._selected(unit) for unit in ranked if unit["kind"] == "claim"][:4]
        reasons = [self._selected(unit) for unit in ranked if unit["kind"] == "reason"][:3]
        memories = [self._selected(unit) for unit in ranked if unit["kind"] == "memory"][:2]
        uncertainties = [self._selected(unit) for unit in ranked if unit["kind"] == "uncertainty"][:2]
        evidence_refs = sorted({str(unit["unit_id"]) for unit in ranked})
        evidence_event_ids = sorted({str(unit["event_id"]) for unit in ranked if unit["event_id"]})
        if direct:
            answer_status = "direct_answer"
        elif len({unit["event_content_id"] for unit in ranked}) >= 2 and claims:
            answer_status = "composite_answer"
        else:
            answer_status = "partial_answer"
            uncertainties.append(self._unit("uncertainty", "现有资料只能支持部分回答。", ""))
        if not claims and not reasons:
            answer_status = "refused"
        confidence = 0.0 if answer_status == "refused" else min(
            0.82,
            0.12 + max(float(item["score"]) for item in chosen_events)
            * math.sqrt(float(top_speech["probability"]) * float(top_stance["probability"])),
        )
        structured = self._structured(
            artifact, speech_distribution, stance_distribution,
            answer_status=answer_status,
            claims=claims, reasons=reasons, memories=memories, uncertainties=uncertainties,
            evidence_refs=evidence_refs, evidence_event_ids=evidence_event_ids,
            confidence=confidence,
            applicability="exploratory" if answer_status != "refused" else "local_support_gap",
            refusal_reasons=[] if answer_status != "refused" else ["local_support_gap"],
        )
        structured["response_basis"] = {
            "path": answer_status,
            "person_prediction_status": "direct_historical_evidence"
            if direct
            else "similar_event_inference",
            "query_analysis": copy.deepcopy(dict(recall["query_analysis"])),
            "selected_demonstrated_knowledge": self._relevant_knowledge(
                artifact,
                query=str(recall["query"]),
                event_types=classify_event_types(str(recall["query"])),
            ),
            "knowledge_boundary": "person_demonstrated_claim_not_verified_fact",
        }
        related_event_ids = self._related_event_ids(
            artifact, set(evidence_event_ids)
        )
        trace = {**common_trace,
            "kernel": KERNEL_ID_V2,
            "retrieval_calls": 1,
            "planner_calls": int(bool(planner_trace)),
            "planner": dict(planner_trace or {}),
            "resolved_context_turns": len(resolved_ids),
            "resolved_context_message_ids": resolved_ids,
            "candidate_event_ids": [str(item["event"]["event_id"]) for item in candidates],
            "selected_event_ids": evidence_event_ids,
            "related_event_ids": related_event_ids,
            "selected_event_content_ids": sorted({str(unit["event_content_id"]) for unit in ranked}),
            "direct_match": bool(direct),
            "model_proposal_is_advisory": True,
            "pcfm_final_court": ["speech_act", "stance", "claims", "reasons", "answer_status"],
        }
        return self._result(structured, self._contract(structured), status="refused" if answer_status == "refused" else "answered", trace=trace)

    @staticmethod
    def _relevant_knowledge(
        artifact: Mapping[str, object], *, query: str, event_types: Sequence[str]
    ) -> list[dict[str, object]]:
        event_type_set = set(map(str, event_types))
        ranked: list[tuple[float, str, dict[str, object]]] = []
        for raw in artifact.get("demonstrated_knowledge", []):
            item = copy.deepcopy(dict(raw))
            domain_match = bool(
                event_type_set & set(map(str, item.get("supporting_domains", [])))
            )
            similarity = lexical_similarity(query, str(item.get("statement", "")))
            if not domain_match and similarity < 0.18:
                continue
            score = similarity + (0.2 if domain_match else 0.0)
            ranked.append((score, str(item.get("knowledge_id", "")), item))
        ranked.sort(key=lambda value: (-value[0], value[1]))
        return [item for _, _, item in ranked[:4]]

    @staticmethod
    def _related_event_ids(
        artifact: Mapping[str, object], selected_event_ids: set[str]
    ) -> list[str]:
        related: set[str] = set()
        for raw in artifact.get("event_relations", []):
            relation = dict(raw)
            left = str(relation.get("left_event_id", ""))
            right = str(relation.get("right_event_id", ""))
            if left in selected_event_ids and right not in selected_event_ids:
                related.add(right)
            if right in selected_event_ids and left not in selected_event_ids:
                related.add(left)
        return sorted(related)

    @staticmethod
    def _unit(kind: str, text: str, event_id: str) -> dict[str, object]:
        return {"id": _unit_id(kind, text), "text": text, "evidence_ref": _unit_id(kind, text), "evidence_event_id": event_id, "probability": 0.0}

    @staticmethod
    def _selected(unit: Mapping[str, object]) -> dict[str, object]:
        return {
            "id": str(unit["unit_id"]),
            "text": str(unit["text"]),
            "evidence_ref": str(unit["unit_id"]),
            "evidence_event_id": str(unit["event_id"]),
            "probability": round(float(unit["support"]), 6),
        }

    @staticmethod
    def _structured(
        artifact: Mapping[str, object],
        speech_distribution: Sequence[Mapping[str, object]],
        stance_distribution: Sequence[Mapping[str, object]],
        *, answer_status: str, claims: list[dict[str, object]], reasons: list[dict[str, object]],
        memories: list[dict[str, object]], uncertainties: list[dict[str, object]],
        evidence_refs: list[str], evidence_event_ids: list[str], confidence: float,
        applicability: str, refusal_reasons: list[str],
    ) -> dict[str, object]:
        if answer_status not in ANSWER_STATUSES:
            raise ResponsePredictionError("invalid_answer_status")
        return {
            "schema_version": PREDICTION_SCHEMA_V2,
            "person_id": artifact["person_id"],
            "domain": str(dict(artifact.get("scope", {})).get("focus_domain", "unknown")),
            "speech_act": copy.deepcopy(dict(speech_distribution[0])),
            "speech_act_distribution": copy.deepcopy(list(speech_distribution)),
            "stance": copy.deepcopy(dict(stance_distribution[0])),
            "stance_distribution": copy.deepcopy(list(stance_distribution)),
            "claims": claims, "reasons": reasons, "memories": memories,
            "uncertainties": uncertainties, "answer_status": answer_status,
            "confidence": round(float(confidence), 6),
            "confidence_kind": "uncalibrated_exploratory_support",
            "applicability": applicability, "refusal_reasons": refusal_reasons,
            "evidence_refs": evidence_refs, "evidence_event_ids": evidence_event_ids,
            "active_components": copy.deepcopy(list(artifact["active_components"])),
            "components": copy.deepcopy(list(artifact["components"])),
            "model_version": f"{artifact['person_id']}-v{artifact['version']}",
            "model_validity": "exploratory_accuracy_not_assessed",
            "valid_scope": copy.deepcopy(dict(artifact.get("scope", {}))),
        }

    @staticmethod
    def _contract(structured: Mapping[str, object], ordinary_text: str = "") -> dict[str, object]:
        locked = "\n".join(
            str(item["text"])
            for field in ("claims", "reasons", "memories", "uncertainties")
            for item in structured[field]
        )
        entities, numbers, dates, quotes = _protected_values(locked)
        return {
            "schema_version": FROZEN_CONTRACT_V3,
            "speech_act": str(dict(structured["speech_act"])["label"]),
            "stance": str(dict(structured["stance"])["label"]),
            "answer_status": structured["answer_status"],
            "refusal_status": "prediction_refused" if structured["answer_status"] == "refused" else "not_refused",
            "ordinary_dialogue_text": ordinary_text,
            "claims": [{"id": item["id"], "text": item["text"]} for item in structured["claims"]],
            "reasons": [{"id": item["id"], "text": item["text"]} for item in structured["reasons"]],
            "memories": [{"id": item["id"], "text": item["text"]} for item in structured["memories"]],
            "uncertainties": [{"id": item["id"], "text": item["text"]} for item in structured["uncertainties"]],
            "protected_entities": entities, "protected_numbers": numbers,
            "protected_dates": dates, "protected_quotes": quotes,
            "evidence_refs": list(structured["evidence_refs"]),
            "confidence": structured["confidence"], "style_mode": "interview_public",
        }

    @staticmethod
    def _result(structured: dict[str, object], contract: dict[str, object], *, status: str, trace: dict[str, object]) -> dict[str, object]:
        structured["renderer_contract_digest"] = canonical_hash(contract)
        structured_digest = canonical_hash(structured)
        structured["content_digest"] = structured_digest
        return {
            "schema_version": PREDICTION_SCHEMA_V2,
            "status": status,
            "answer_status": structured["answer_status"],
            "structured_prediction": structured,
            "renderer_contract": contract,
            "content_digest": structured_digest,
            "renderer_contract_digest": canonical_hash(contract),
            "prediction_trace": trace,
        }
