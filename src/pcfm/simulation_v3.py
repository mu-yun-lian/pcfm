from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence


EVENT_SCHEMA_V3 = "pcfm-public-decision-frame-v3"
PREFERENCE_ATOM_SCHEMA_V3 = "pcfm-public-preference-atom-v3"
MODEL_SCHEMA_V3 = "pcfm-simulation-model-v3"
PREDICTION_SCHEMA_V3 = "pcfm-simulation-prediction-v3"
KERNEL_ID_V3 = "simulation-v3"
FROZEN_CONTRACT_V3 = "pcfm-frozen-content-contract-v3"
ELIGIBLE_SOURCE_AUTHENTICITY = frozenset(
    {"verbatim_transcript", "verified_quote", "first_party_public_statement"}
)


class SimulationV3Error(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _hash(value: object) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _terms(value: str) -> tuple[str, ...]:
    result: list[str] = []
    for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value).casefold()):
        result.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
            result.extend(token[index : index + 2] for index in range(len(token) - 1))
    return tuple(result)


STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "be",
        "because",
        "before",
        "come",
        "comes",
        "do",
        "does",
        "for",
        "how",
        "i",
        "in",
        "is",
        "it",
        "must",
        "of",
        "or",
        "our",
        "protect",
        "should",
        "the",
        "this",
        "to",
        "we",
        "what",
        "when",
        "would",
    }
)


DOMAIN_KEYWORDS: dict[str, frozenset[str]] = {
    "health": frozenset({"health", "hospital", "patient", "medical", "medicine"}),
    "technology": frozenset({"technology", "software", "system", "ai", "algorithm"}),
    "product": frozenset({"product", "launch", "prototype", "design", "team"}),
    "governance": frozenset({"government", "policy", "law", "regulation", "president"}),
    "economics": frozenset({"market", "price", "business", "company", "growth"}),
    "social": frozenset({"rights", "fairness", "society", "responsibility"}),
    "risk": frozenset({"risk", "safety", "harm", "failure", "irreversible", "costly"}),
    "space": frozenset({"space", "nasa", "astronaut", "mission", "launch"}),
    "aviation": frozenset({"aviation", "aircraft", "flight", "pilot"}),
    "personal": frozenset({"family", "friend", "relationship", "private"}),
}

PRIMARY_DOMAIN_PRIORITY = (
    "health",
    "aviation",
    "space",
    "governance",
    "economics",
    "product",
    "technology",
    "personal",
    "social",
    "risk",
    "general",
)


def _domain_tags(*values: str) -> list[str]:
    words = set(_terms(" ".join(values)))
    tags = sorted(
        domain for domain, keywords in DOMAIN_KEYWORDS.items() if words & keywords
    )
    return tags or ["general"]


def _primary_domain(tags: Sequence[str]) -> str:
    available = set(tags)
    return next(
        (domain for domain in PRIMARY_DOMAIN_PRIORITY if domain in available),
        "general",
    )


def _decision_type(question: str, has_tradeoff: bool) -> str:
    lowered = question.casefold()
    if has_tradeoff:
        return "priority_tradeoff"
    if re.search(r"\b(?:should|whether|would|do you agree|what matters)\b", lowered):
        return "evaluation_or_choice"
    if re.search(r"\b(?:why|how|what prompted|describe)\b", lowered):
        return "explanation_or_recollection"
    return "public_response_other"


def _event_scale(*values: str) -> str:
    words = set(_terms(" ".join(values)))
    if words & {"government", "policy", "law", "country", "nation", "president"}:
        return "public_policy_or_societal"
    if words & {"company", "team", "hospital", "nasa", "organization", "studio"}:
        return "organizational"
    if words & {"family", "friend", "private", "personal", "individual"}:
        return "individual"
    return "unknown_scale"


def _similarity(left: str, right: str) -> float:
    left_terms = {item for item in _terms(left) if item not in STOPWORDS}
    right_terms = {item for item in _terms(right) if item not in STOPWORDS}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / math.sqrt(
        len(left_terms) * len(right_terms)
    )


def _sentences(value: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？])\s+|\n+", str(value).strip())
        if item.strip()
    ]


def _clean_interest(value: str) -> str:
    clean = re.sub(r"\s+", " ", str(value).casefold()).strip(" .,:;!?\"'")
    clean = re.sub(
        r"^(?:we|i|they|a team|the team)\s+(?:should|must|would|need to|has to)\s+",
        "",
        clean,
    )
    clean = re.sub(r"^(?:protect|prioritize|choose|favor)\s+", "", clean)
    clean = re.sub(r"\s+(?:because|since|when|if)\b.*$", "", clean)
    tokens = [token for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", clean) if token not in STOPWORDS]
    return " ".join(tokens[:8]) or "unknown"


def _extract_reasons(answer: str) -> list[str]:
    reasons: list[str] = []
    for sentence in _sentences(answer):
        match = re.search(r"\b(?:because|since|due to)\s+(.+?)(?:[.;]|$)", sentence, re.I)
        if match:
            reasons.append(match.group(1).strip())
    return reasons


def _extract_conditions(*values: str) -> list[str]:
    conditions: list[str] = []
    for value in values:
        for sentence in _sentences(value):
            match = re.search(
                r"\b(?:if|when|unless|provided that)\s+(.+?)(?:[,.;]|$)",
                sentence,
                re.I,
            )
            if match:
                conditions.append(match.group(0).strip(" ,.;"))
    return sorted(set(conditions))


def _extract_tradeoffs(answer: str) -> list[dict[str, object]]:
    patterns = (
        re.compile(
            r"\bprotect\s+(?P<preferred>[^,.;!?]+?)\s+before\s+(?P<cost>[^,.;!?]+?)(?=\s+because\b|[,.;!?]|$)",
            re.I,
        ),
        re.compile(
            r"\b(?P<preferred>[a-z][a-z0-9 '\-]{0,60}?)\s+(?:must|should|has to|needs to)\s+come\s+before\s+(?P<cost>[^,.;!?]+?)(?=\s+because\b|[,.;!?]|$)",
            re.I,
        ),
        re.compile(
            r"\bprioriti[sz]e\s+(?P<preferred>[^,.;!?]+?)\s+over\s+(?P<cost>[^,.;!?]+?)(?=\s+because\b|[,.;!?]|$)",
            re.I,
        ),
        re.compile(
            r"\b(?P<preferred>[^,.;!?]+?)\s+rather than\s+(?P<cost>[^,.;!?]+?)(?=\s+because\b|[,.;!?]|$)",
            re.I,
        ),
    )
    results: list[dict[str, object]] = []
    for pattern in patterns:
        for match in pattern.finditer(answer):
            preferred = _clean_interest(match.group("preferred"))
            cost = _clean_interest(match.group("cost"))
            if "unknown" in {preferred, cost} or preferred == cost:
                continue
            results.append(
                {
                    "protected_interest": preferred,
                    "accepted_cost": cost,
                    "verbatim_tradeoff_span": match.group(0).strip(),
                    "extraction_status": "explicit_tradeoff_span",
                }
            )
    unique: dict[tuple[str, str], dict[str, object]] = {}
    for result in results:
        unique[(str(result["protected_interest"]), str(result["accepted_cost"]))] = result
    return [unique[key] for key in sorted(unique)]


def _extract_question_options(question: str) -> list[str]:
    patterns = (
        r"\bprioriti[sz]e\s+(.+?)\s+or\s+(.+?)(?:\?|$)",
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"\bweigh\s+(.+?)\s+against\s+(.+?)(?:\?|$)",
        r"\b(.+?)\s+(?:versus|vs\.)\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, re.I)
        if match:
            return [_clean_interest(match.group(1)), _clean_interest(match.group(2))]
    return []


def _source_units(source: Mapping[str, object]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for index, raw in enumerate(source.get("qas", []), start=1):
        if not isinstance(raw, Mapping):
            continue
        question = str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        if question and answer:
            units.append(
                {
                    "question": question,
                    "response": answer,
                    "locator": str(raw.get("locator", f"qa:{index}")),
                    "unit_kind": "question_answer_turn",
                }
            )
    if units:
        return units
    for index, raw in enumerate(source.get("segments", []), start=1):
        text = str(raw.get("text", "") if isinstance(raw, Mapping) else raw).strip()
        if len(text) >= 24:
            units.append(
                {
                    "question": str(source.get("source_context") or source.get("title") or "public statement"),
                    "response": text,
                    "locator": str(raw.get("locator", f"segment:{index}")) if isinstance(raw, Mapping) else f"segment:{index}",
                    "unit_kind": "attributable_narrative_span",
                }
            )
    return units


def _frame_from_unit(
    source: Mapping[str, object],
    unit: Mapping[str, str],
    tradeoffs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    question = str(unit["question"])
    response = str(unit["response"])
    source_id = str(source["source_id"])
    first_tradeoff = dict(tradeoffs[0]) if tradeoffs else {}
    preferred = str(first_tradeoff.get("protected_interest", "unknown"))
    cost = str(first_tradeoff.get("accepted_cost", "unknown"))
    tradeoff_spans = [
        str(item.get("verbatim_tradeoff_span", "")) for item in tradeoffs
    ]
    frame_id = f"frame-{_hash([source_id, unit['locator'], question, response])[:16]}"
    reasons = _extract_reasons(response)
    conditions = _extract_conditions(question, response)
    domains = _domain_tags(
        question,
        response,
        str(source.get("title", "")),
        str(source.get("source_context", "")),
    )
    return {
        "schema_version": EVENT_SCHEMA_V3,
        "event_frame_id": frame_id,
        "person_id": str(source.get("person_id", "")),
        "source_id": source_id,
        "source_lineage": str(source.get("near_duplicate_of") or source_id),
        "source_role": str(source.get("dataset_role", "")),
        "temporal_context": {
            "response_time": str(source.get("source_date", "")),
            "time_source": "reviewed_source_metadata",
            "precision": "day_or_coarser",
        },
        "social_context": {
            "speaker": str(source.get("speaker", "unknown")) or "unknown",
            "speaker_role": str(source.get("speaker_role", "public_speaker")),
            "audience": str(source.get("audience", "unknown")),
            "occasion": str(source.get("source_context") or source.get("title") or "unknown"),
        },
        "decision_frame": {
            "trigger": question,
            "target": question,
            "stakeholders": [],
            "available_options": _extract_question_options(question),
            "constraints": conditions,
            "stakes": [],
            "known_information": reasons,
            "preferred_interest": preferred,
            "accepted_cost": cost,
            "tradeoff_status": "explicit" if tradeoffs else "not_observed",
            "observed_tradeoffs": [copy.deepcopy(dict(item)) for item in tradeoffs],
        },
        "observed_response": {
            "verbatim": response,
            "reasons": reasons,
            "conditions": conditions,
            "exceptions": [],
        },
        "domain_tags": domains,
        "event_classification": {
            "primary_domain": _primary_domain(domains),
            "secondary_tags": [
                domain for domain in domains if domain != _primary_domain(domains)
            ],
            "decision_type": _decision_type(question, bool(tradeoffs)),
            "event_scale": _event_scale(
                question, response, str(source.get("source_context", ""))
            ),
            "classification_method": "deterministic_reviewable_keyword_taxonomy_v1",
        },
        "topic_terms": sorted(
            {
                item
                for item in _terms(f"{question} {response}")
                if item not in STOPWORDS and len(item) > 1
            }
        )[:32],
        "evidence": {
            "span_status": "exact_source_span",
            "question_span": question,
            "response_span": response,
            "tradeoff_span": tradeoff_spans[0] if tradeoff_spans else "",
            "tradeoff_spans": tradeoff_spans,
            "locator": str(unit["locator"]),
            "unit_kind": str(unit["unit_kind"]),
            "content_hash": _hash([question, response]),
            "source_url": str(source.get("source_url", "")),
        },
        "unknown_fields": [
            name
            for value, name in (
                (source.get("audience"), "audience"),
                (conditions, "conditions"),
                (preferred != "unknown", "preferred_interest"),
                (cost != "unknown", "accepted_cost"),
            )
            if not value
        ],
        "evidence_support": {
            "exact_attributable_span": True,
            "time_present": bool(str(source.get("source_date", "")).strip()),
            "speaker_attributed": bool(str(source.get("speaker", "")).strip()),
            "explicit_tradeoff_count": len(tradeoffs),
            "context_completeness": round(
                sum(
                    bool(value)
                    for value in (
                        source.get("source_date"),
                        source.get("speaker"),
                        source.get("source_context"),
                        source.get("source_locator"),
                        conditions,
                    )
                )
                / 5,
                3,
            ),
            "meaning": "descriptive_evidence_completeness_not_accuracy_probability",
        },
    }


def _preference_atoms(frame: Mapping[str, object]) -> list[dict[str, object]]:
    decision = dict(frame["decision_frame"])
    frame_id = str(frame["event_frame_id"])
    observed = dict(frame["observed_response"])
    results: list[dict[str, object]] = []
    for tradeoff in decision.get("observed_tradeoffs", []):
        protected = str(tradeoff.get("protected_interest", "unknown"))
        cost = str(tradeoff.get("accepted_cost", "unknown"))
        if "unknown" in {protected, cost}:
            continue
        results.append(
            {
                "schema_version": PREFERENCE_ATOM_SCHEMA_V3,
                "preference_atom_id": f"preference-{_hash([frame_id, protected, cost])[:16]}",
                "event_frame_id": frame_id,
                "source_id": str(frame["source_id"]),
                "source_lineage": str(frame["source_lineage"]),
                "protected_interest": protected,
                "accepted_cost": cost,
                "decision_rule": f"prioritize {protected} over {cost}",
                "conditions": list(observed.get("conditions", [])),
                "reasons": list(observed.get("reasons", [])),
                "exceptions": list(observed.get("exceptions", [])),
                "domain_tags": list(frame.get("domain_tags", [])),
                "primary_domain": str(
                    dict(frame.get("event_classification", {})).get(
                        "primary_domain", "general"
                    )
                ),
                "role": str(dict(frame["social_context"]).get("speaker_role", "unknown")),
                "response_time": str(dict(frame["temporal_context"])["response_time"]),
                "evidence_span": str(tradeoff.get("verbatim_tradeoff_span", "")),
                "support_weight": float(
                    dict(frame.get("evidence_support", {})).get(
                        "context_completeness", 0.0
                    )
                ),
                "support_weight_meaning": "context_completeness_not_accuracy_probability",
                "status": "single_event_candidate",
                "semantic_claim": "observable_public_tradeoff_not_inner_value",
            }
        )
    return results


def _preference_structures(atoms: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for raw in atoms:
        atom = copy.deepcopy(dict(raw))
        grouped.setdefault(
            (str(atom["protected_interest"]), str(atom["accepted_cost"])), []
        ).append(atom)
    structures: list[dict[str, object]] = []
    for (protected, cost), values in sorted(grouped.items()):
        values.sort(key=lambda item: str(item["preference_atom_id"]))
        lineages = sorted({str(item["source_lineage"]) for item in values})
        domains = sorted({str(item.get("primary_domain", "general")) for item in values})
        domain_tags = sorted(
            {str(domain) for item in values for domain in item.get("domain_tags", [])}
        )
        times = sorted({str(item["response_time"]) for item in values})
        roles = sorted({str(item.get("role", "unknown")) for item in values})
        support_by_lineage = {
            lineage: max(
                float(item.get("support_weight", 0.0))
                for item in values
                if str(item["source_lineage"]) == lineage
            )
            for lineage in lineages
        }
        counter_ids = sorted(
            str(item["event_frame_id"])
            for item in grouped.get((cost, protected), [])
        )
        counter_contexts = [
            {
                "event_frame_id": str(item["event_frame_id"]),
                "conditions": list(item.get("conditions", [])),
                "domain_tags": list(item.get("domain_tags", [])),
                "primary_domain": str(item.get("primary_domain", "general")),
                "reasons": list(item.get("reasons", [])),
            }
            for item in grouped.get((cost, protected), [])
        ]
        if len(values) >= 2 and len(lineages) >= 2 and len(domains) >= 2:
            status = "cross_domain_public_preference"
        elif len(values) >= 2:
            status = "repeated_public_preference"
        else:
            status = "single_event_candidate"
        structures.append(
            {
                "preference_structure_id": f"structure-{_hash([protected, cost])[:16]}",
                "kind": "observable_public_preference_structure",
                "protected_interest": protected,
                "accepted_cost": cost,
                "decision_rule": f"prioritize {protected} over {cost}",
                "supporting_atom_ids": [str(item["preference_atom_id"]) for item in values],
                "supporting_event_ids": [str(item["event_frame_id"]) for item in values],
                "counterevidence_event_ids": counter_ids,
                "counterevidence_contexts": counter_contexts,
                "independent_source_count": len(lineages),
                "source_lineages": lineages,
                "domain_count": len(domains),
                "primary_domains": domains,
                "domain_tags": domain_tags,
                "temporal_scope": {
                    "start": times[0],
                    "end": times[-1],
                    "dated_event_count": len(times),
                },
                "role_scope": roles,
                "support_summary": {
                    "weighted_independent_support": round(
                        sum(support_by_lineage.values()), 3
                    ),
                    "independent_lineage_weights": support_by_lineage,
                    "counterevidence_count": len(counter_ids),
                    "meaning": "descriptive_support_not_prediction_probability",
                },
                "conditions": sorted(
                    {str(value) for item in values for value in item.get("conditions", [])}
                ),
                "reasons": sorted(
                    {str(value) for item in values for value in item.get("reasons", [])}
                ),
                "exceptions": sorted(
                    {str(value) for item in values for value in item.get("exceptions", [])}
                ),
                "status": status,
                "conflict_status": "context_split_required" if counter_ids else "no_recorded_counterexample",
                "support_status": "descriptive_counts_not_prediction_probability",
            }
        )
    return structures


def _preference_relations(
    atoms: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    relations: list[dict[str, object]] = []
    ordered = sorted(atoms, key=lambda item: str(item["preference_atom_id"]))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            left_pair = (
                str(left["protected_interest"]),
                str(left["accepted_cost"]),
            )
            right_pair = (
                str(right["protected_interest"]),
                str(right["accepted_cost"]),
            )
            if left_pair == right_pair:
                relation = "reinforces_same_public_tradeoff"
                status = "observed_relation"
            elif left_pair == tuple(reversed(right_pair)):
                relation = "reverses_public_tradeoff"
                status = "observed_counterevidence"
            elif left_pair[0] == right_pair[0]:
                relation = "shared_priority_different_accepted_cost"
                status = "latent_structure_candidate_not_runtime_eligible"
            elif left_pair[1] == right_pair[1]:
                relation = "competing_priority_against_shared_cost"
                status = "latent_structure_candidate_not_runtime_eligible"
            elif left["event_frame_id"] == right["event_frame_id"]:
                relation = "co_observed_in_one_event"
                status = "observed_relation"
            else:
                continue
            relations.append(
                {
                    "relation_id": f"relation-{_hash([left['preference_atom_id'], right['preference_atom_id'], relation])[:16]}",
                    "left_preference_atom_id": str(left["preference_atom_id"]),
                    "right_preference_atom_id": str(right["preference_atom_id"]),
                    "relation": relation,
                    "status": status,
                    "source_lineages": sorted(
                        {str(left["source_lineage"]), str(right["source_lineage"])}
                    ),
                    "interpretation_boundary": (
                        "relation_between_observed_public_tradeoffs_not_hidden_value_fact"
                    ),
                }
            )
    return relations


class SimulationKernelV3:
    kernel_id = KERNEL_ID_V3

    def fit(
        self,
        *,
        person_id: str,
        version: int,
        reviewed_sources: Sequence[Mapping[str, object]],
        scope: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        frames: list[dict[str, object]] = []
        rejected: dict[str, list[str]] = {
            "not_reviewed_model_source": [],
            "unattributable_or_unverified_source": [],
            "missing_time": [],
            "missing_attributable_span": [],
        }
        source_identities: list[dict[str, object]] = []
        for raw in sorted(reviewed_sources, key=lambda item: str(item.get("source_id", ""))):
            source = dict(raw)
            source_id = str(source.get("source_id", ""))
            if (
                source.get("review_status") != "confirmed"
                or source.get("dataset_role") != "model_source"
                or str(source.get("person_id", person_id)) != person_id
            ):
                rejected["not_reviewed_model_source"].append(source_id)
                continue
            if (
                source.get("speaker_scope") == "mixed_speakers"
                or source.get("content_authenticity")
                not in ELIGIBLE_SOURCE_AUTHENTICITY
            ):
                rejected["unattributable_or_unverified_source"].append(source_id)
                continue
            if not str(source.get("source_date", "")).strip():
                rejected["missing_time"].append(source_id)
                continue
            units = _source_units(source)
            if not units:
                rejected["missing_attributable_span"].append(source_id)
                continue
            source_identities.append(
                {
                    "source_id": source_id,
                    "source_lineage": str(source.get("near_duplicate_of") or source_id),
                    "content_hash": str(source.get("content_hash") or _hash(units)),
                    "response_time": str(source.get("source_date")),
                    "review_status": "confirmed",
                    "dataset_role": "model_source",
                }
            )
            for unit in units:
                tradeoffs = _extract_tradeoffs(unit["response"])
                frames.append(_frame_from_unit(source, unit, tradeoffs))
        frames.sort(key=lambda item: str(item["event_frame_id"]))
        if not frames:
            raise SimulationV3Error("no_eligible_reviewed_raw_source_frames")
        atoms = [value for frame in frames for value in _preference_atoms(frame)]
        atoms.sort(key=lambda item: str(item["preference_atom_id"]))
        structures = _preference_structures(atoms)
        relations = _preference_relations(atoms)
        knowledge = sorted(
            [
                {
                    "knowledge_claim_id": f"knowledge-{_hash([frame['event_frame_id'], statement])[:16]}",
                    "statement": statement,
                    "event_frame_id": frame["event_frame_id"],
                    "source_id": frame["source_id"],
                    "domain_tags": frame["domain_tags"],
                    "status": "person_publicly_used_claim_not_verified_fact",
                }
                for frame in frames
                for statement in _sentences(
                    str(dict(frame["observed_response"])["verbatim"])
                )
                if len(statement) >= 12
            ],
            key=lambda item: str(item["knowledge_claim_id"]),
        )
        semantic_payload = {
            "source_identities": sorted(source_identities, key=lambda item: str(item["source_id"])),
            "event_frames": frames,
            "preference_atoms": atoms,
            "preference_structures": structures,
            "preference_relations": relations,
            "knowledge_claims": knowledge,
            "rejected_sources": {key: sorted(values) for key, values in rejected.items()},
        }
        artifact: dict[str, object] = {
            "schema_version": MODEL_SCHEMA_V3,
            "kernel": KERNEL_ID_V3,
            "input_contract": "reviewed_raw_sources_v1",
            "person_id": person_id,
            "version": int(version),
            "created_at": _utc_now(),
            "scope": copy.deepcopy(dict(scope or {})),
            **semantic_payload,
            "semantic_model_digest": _hash(semantic_payload),
            "components": [
                {"component_id": "raw_source_decision_frames_v3", "status": "active"},
                {"component_id": "public_preference_atoms_v3", "status": "active"},
                {"component_id": "preference_structure_graph_v3", "status": "active"},
                {"component_id": "conflict_query_mapper_v3", "status": "active"},
                {"component_id": "response_kernel_v2", "status": "baseline_only"},
            ],
            "active_components": [
                "raw_source_decision_frames_v3",
                "public_preference_atoms_v3",
                "preference_structure_graph_v3",
                "conflict_query_mapper_v3",
            ],
            "validation_status": "implemented_exploratory_accuracy_not_assessed",
            "accuracy_claim": "none",
        }
        artifact["artifact_hash"] = _hash(artifact)
        return artifact

    @staticmethod
    def verify(artifact: Mapping[str, object]) -> None:
        if artifact.get("schema_version") != MODEL_SCHEMA_V3 or artifact.get("kernel") != KERNEL_ID_V3:
            raise SimulationV3Error("unsupported_simulation_v3_schema")
        value = copy.deepcopy(dict(artifact))
        digest = str(value.pop("artifact_hash", ""))
        if _hash(value) != digest:
            raise SimulationV3Error("simulation_v3_integrity_failed")

    def evaluate(
        self,
        artifact: Mapping[str, object],
        holdout_sources: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Score only explicit, later-time holdout tradeoffs; never infer labels."""
        self.verify(artifact)
        training_times = [
            str(item.get("response_time", ""))
            for item in artifact.get("source_identities", [])
            if item.get("response_time")
        ]
        training_end = max(training_times, default="")
        cases: list[dict[str, object]] = []
        rejected = {
            "not_confirmed_final_holdout": [],
            "not_strictly_later": [],
            "no_explicit_tradeoff_label": [],
        }
        training_source_ids = {
            str(item.get("source_id"))
            for item in artifact.get("source_identities", [])
        }
        for raw in sorted(
            holdout_sources, key=lambda item: str(item.get("source_id", ""))
        ):
            source = dict(raw)
            source_id = str(source.get("source_id", ""))
            if (
                source.get("review_status") != "confirmed"
                or source.get("dataset_role") != "final_holdout"
            ):
                rejected["not_confirmed_final_holdout"].append(source_id)
                continue
            source_time = str(source.get("source_date", ""))
            if not source_time or not training_end or source_time <= training_end:
                rejected["not_strictly_later"].append(source_id)
                continue
            for unit in _source_units(source):
                expected = _extract_tradeoffs(unit["response"])
                if not expected:
                    rejected["no_explicit_tradeoff_label"].append(
                        f"{source_id}:{unit['locator']}"
                    )
                    continue
                prediction = self.predict(
                    artifact, text=unit["question"], history=[]
                )
                basis = dict(
                    dict(prediction["structured_prediction"]).get(
                        "response_basis", {}
                    )
                )
                predicted_pair = (
                    str(basis.get("protected_interest", "")),
                    str(basis.get("accepted_cost", "")),
                )
                expected_pairs = {
                    (
                        str(item["protected_interest"]),
                        str(item["accepted_cost"]),
                    )
                    for item in expected
                }
                cases.append(
                    {
                        "source_id": source_id,
                        "locator": unit["locator"],
                        "question": unit["question"],
                        "expected_tradeoffs": sorted(expected_pairs),
                        "answer_status": prediction["answer_status"],
                        "predicted_tradeoff": predicted_pair,
                        "person_prediction_made": prediction["answer_status"]
                        == "preference_structure_answer",
                        "direction_correct": predicted_pair in expected_pairs,
                    }
                )
        leakage = sorted(training_source_ids & {str(item.get("source_id")) for item in holdout_sources})
        if not cases:
            return {
                "status": "not_assessed",
                "reason": "no_strictly_later_explicit_tradeoff_holdout_cases",
                "sample_count": 0,
                "training_end": training_end,
                "holdout_leakage_source_ids": leakage,
                "rejected": rejected,
                "accuracy_claim": "none",
            }
        covered = [item for item in cases if item["person_prediction_made"]]
        correct = [item for item in covered if item["direction_correct"]]
        return {
            "status": "assessed_exploratory" if not leakage else "invalid_holdout_leakage",
            "sample_count": len(cases),
            "coverage": round(len(covered) / len(cases), 6),
            "covered_direction_accuracy": (
                round(len(correct) / len(covered), 6) if covered else "not_assessed"
            ),
            "selective_accuracy_note": "accuracy_is_conditional_on_person_prediction_coverage",
            "training_end": training_end,
            "holdout_leakage_source_ids": leakage,
            "cases": cases,
            "rejected": rejected,
            "accuracy_claim": "exploratory_only_not_guaranteed",
        }

    @staticmethod
    def _relevant_knowledge(
        artifact: Mapping[str, object], query: str, domains: set[str]
    ) -> list[dict[str, object]]:
        ranked: list[tuple[float, str, dict[str, object]]] = []
        for raw in artifact.get("knowledge_claims", []):
            item = copy.deepcopy(dict(raw))
            overlap = bool(domains & set(map(str, item.get("domain_tags", []))))
            similarity = _similarity(query, str(item.get("statement", "")))
            if overlap or similarity >= 0.2:
                ranked.append(
                    (similarity + (0.2 if overlap else 0.0), str(item["knowledge_claim_id"]), item)
                )
        ranked.sort(key=lambda value: (-value[0], value[1]))
        return [item for _, _, item in ranked[:4]]

    def predict(
        self,
        artifact: Mapping[str, object],
        *,
        text: str,
        history: Sequence[Mapping[str, object]],
        conversation_context: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        self.verify(artifact)
        clean = str(text).strip()
        if not clean:
            raise SimulationV3Error("message_required")
        context_trace = {
            "context_digest": _hash(
                [
                    clean,
                    [
                        (item.get("message_id"), item.get("role"), item.get("text"))
                        for item in history[-8:]
                    ],
                ]
            ),
            "context_used": {
                "message_ids": [
                    str(item.get("message_id"))
                    for item in history[-8:]
                    if item.get("message_id")
                ],
                "turn_count": len(history[-8:]),
                "generated_context_count": sum(
                    item.get("context_role") == "model_generated_context"
                    for item in history[-8:]
                ),
                "generated_context_is_fitting_evidence": False,
            },
            "conversation_context": copy.deepcopy(
                dict(conversation_context or {})
            ),
            "selected_event_ids": [],
            "related_event_ids": [],
            "resolved_context_turns": 0,
            "generative_content_calls": 0,
        }
        compact = re.sub(r"\W+", "", clean.casefold())
        if compact in {
            "hi", "hello", "hey", "你好", "您好", "thanks", "thankyou", "谢谢",
            "continue", "继续", "接着说",
        }:
            if compact in {"thanks", "thankyou", "谢谢"}:
                safe = "不客气。"
            elif compact in {"continue", "继续", "接着说"}:
                safe = "可以。你想继续前面的哪一件事？"
            else:
                safe = "你好。你想从哪件事开始聊？"
            return self._result(
                artifact,
                answer_status="ordinary_dialogue",
                claims=[],
                reasons=[],
                uncertainties=[],
                evidence_event_ids=[],
                response_basis={"path": "ordinary_dialogue", "person_prediction_status": "not_applicable"},
                applicability="ordinary_dialogue_content_free",
                confidence=1.0,
                ordinary_text=safe,
                trace={**context_trace, "kernel": KERNEL_ID_V3, "prediction_path": "ordinary_dialogue"},
            )
        query = clean
        resolved_ids: list[str] = []
        if re.fullmatch(r"\s*(?:why|what about that|为什么|那呢)[?？]?\s*", clean, re.I):
            for item in reversed(history[-8:]):
                prior_text = str(item.get("text", "")).strip()
                if (
                    item.get("role") == "user"
                    and prior_text
                    and not re.fullmatch(
                        r"\s*(?:why|what about that|为什么|那呢)[?？]?\s*",
                        prior_text,
                        re.I,
                    )
                ):
                    query = str(item["text"])
                    if item.get("message_id"):
                        resolved_ids.append(str(item["message_id"]))
                    break
            if not resolved_ids and not any(item.get("role") == "user" for item in history[-8:]):
                return self._result(
                    artifact,
                    answer_status="clarification_needed",
                    claims=[], reasons=[], uncertainties=["请说明你指的是前面的哪件事。"],
                    evidence_event_ids=[],
                    response_basis={"path": "clarification", "person_prediction_status": "not_available"},
                    applicability="missing_conversation_reference", confidence=0.0,
                    trace={**context_trace, "kernel": KERNEL_ID_V3, "prediction_path": "clarification"},
                )
        domains = set(_domain_tags(query))
        ranked_frames = sorted(
            (
                (
                    _similarity(query, str(dict(frame["decision_frame"])["trigger"])),
                    str(frame["event_frame_id"]),
                    copy.deepcopy(dict(frame)),
                )
                for frame in artifact.get("event_frames", [])
            ),
            key=lambda value: (-value[0], value[1]),
        )
        exact = next(
            (
                frame
                for _, _, frame in ranked_frames
                if " ".join(_terms(str(dict(frame["decision_frame"])["trigger"])))
                == " ".join(_terms(query))
            ),
            None,
        )
        direct_match = exact
        direct_match_kind = "exact_reviewed_event"
        if direct_match is None and ranked_frames:
            best_score, _, best_frame = ranked_frames[0]
            next_score = ranked_frames[1][0] if len(ranked_frames) > 1 else 0.0
            if best_score >= 0.88 and best_score - next_score >= 0.1:
                direct_match = best_frame
                direct_match_kind = "near_identical_reviewed_question"
        if direct_match is not None:
            response = str(dict(direct_match["observed_response"])["verbatim"])
            reasons = list(dict(direct_match["observed_response"]).get("reasons", []))
            return self._result(
                artifact,
                answer_status="direct_answer",
                claims=[response],
                reasons=reasons,
                uncertainties=["这是历史公开回答，不是对新情境的预测。"],
                evidence_event_ids=[str(direct_match["event_frame_id"])],
                response_basis={
                    "path": "direct_historical_response",
                    "person_prediction_status": "direct_historical_evidence",
                    "query_frame": self._query_frame(query, resolved_ids),
                    "selected_event_frame_ids": [str(direct_match["event_frame_id"])],
                    "retrieval_match_kind": direct_match_kind,
                    "selected_demonstrated_knowledge": [
                        copy.deepcopy(dict(item))
                        for item in artifact.get("knowledge_claims", [])
                        if str(item.get("event_frame_id", ""))
                        == str(direct_match["event_frame_id"])
                    ],
                    "knowledge_boundary": "person_publicly_used_claim_not_verified_fact",
                },
                applicability=direct_match_kind,
                confidence=1.0 if exact is not None else float(ranked_frames[0][0]),
                trace={
                    **context_trace,
                    "kernel": KERNEL_ID_V3,
                    "prediction_path": "direct_event",
                    "retrieval_match_kind": direct_match_kind,
                    "selected_event_ids": [str(direct_match["event_frame_id"])],
                    "resolved_context_message_ids": resolved_ids,
                    "resolved_context_turns": len(resolved_ids),
                },
            )
        options = _extract_question_options(query)
        eligible: list[tuple[float, str, dict[str, object]]] = []
        for raw in artifact.get("preference_structures", []):
            structure = copy.deepcopy(dict(raw))
            protected = str(structure["protected_interest"])
            cost = str(structure["accepted_cost"])
            status = str(structure["status"])
            cross_domain_eligible = status == "cross_domain_public_preference"
            same_domain_eligible = (
                len(structure.get("supporting_event_ids", [])) >= 2
                and bool(domains & set(map(str, structure.get("domain_tags", []))))
            )
            option_match = bool(options) and {
                _clean_interest(options[0]), _clean_interest(options[1])
            } == {protected, cost}
            structure_text = f"{protected} {cost} {' '.join(map(str, structure.get('reasons', [])))}"
            semantic = _similarity(query, structure_text)
            if structure.get("conflict_status") == "context_split_required":
                support_context_score = _similarity(
                    query,
                    " ".join(
                        [
                            *map(str, structure.get("conditions", [])),
                            *map(str, structure.get("domain_tags", [])),
                            *map(str, structure.get("reasons", [])),
                        ]
                    ),
                )
                counter_context_score = max(
                    (
                        _similarity(
                            query,
                            " ".join(
                                [
                                    *map(str, item.get("conditions", [])),
                                    *map(str, item.get("domain_tags", [])),
                                    *map(str, item.get("reasons", [])),
                                ]
                            ),
                        )
                        for item in structure.get("counterevidence_contexts", [])
                    ),
                    default=0.0,
                )
                if support_context_score < counter_context_score + 0.15:
                    continue
                structure["conflict_resolution"] = {
                    "status": "support_context_more_similar_than_countercontext",
                    "support_context_score": round(support_context_score, 6),
                    "counter_context_score": round(counter_context_score, 6),
                    "margin_required": 0.15,
                }
            if option_match and (cross_domain_eligible or same_domain_eligible):
                score = 1.0
            elif same_domain_eligible and semantic >= 0.18:
                score = semantic
            else:
                continue
            eligible.append((score, str(structure["preference_structure_id"]), structure))
        eligible.sort(key=lambda value: (-value[0], value[1]))
        if eligible:
            score, _, selected = eligible[0]
            protected = str(selected["protected_interest"])
            cost = str(selected["accepted_cost"])
            conditions = list(selected.get("conditions", []))
            condition_text = f" when {'; '.join(conditions[:2])}" if conditions else " under comparable conditions"
            statement = (
                f"Based on repeated public responses, this person would more likely "
                f"prioritize {protected} over {cost}{condition_text}."
            )
            reason = (
                f"The structure is supported by {len(selected['supporting_event_ids'])} "
                f"events from {selected['independent_source_count']} independent source lineages."
            )
            query_years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)]
            evidence_end = str(dict(selected.get("temporal_scope", {})).get("end", ""))
            evidence_end_year = int(evidence_end[:4]) if re.match(r"^\d{4}", evidence_end) else None
            temporal_extrapolation = bool(
                query_years
                and evidence_end_year is not None
                and max(query_years) > evidence_end_year
            )
            uncertainty = "This is an uncalibrated projection of public behavior, not a claim about private values."
            if temporal_extrapolation:
                uncertainty += " The question is later than the supporting evidence window, so temporal stability is unknown."
            return self._result(
                artifact,
                answer_status="preference_structure_answer",
                claims=[statement], reasons=[reason],
                uncertainties=[uncertainty],
                evidence_event_ids=list(map(str, selected["supporting_event_ids"])),
                response_basis={
                    "path": "value_conflict_projection",
                    "person_prediction_status": "public_preference_projection",
                    "query_frame": self._query_frame(query, resolved_ids),
                    "selected_preference_structure": selected,
                    "protected_interest": protected,
                    "accepted_cost": cost,
                    "prediction_statement": statement,
                    "temporal_applicability": {
                        "status": (
                            "later_than_evidence_window"
                            if temporal_extrapolation
                            else "within_or_unspecified_relative_to_evidence_window"
                        ),
                        "query_years": query_years,
                        "evidence_window": copy.deepcopy(
                            dict(selected.get("temporal_scope", {}))
                        ),
                    },
                    "selected_demonstrated_knowledge": self._relevant_knowledge(artifact, query, domains),
                    "knowledge_boundary": "person_publicly_used_claim_not_verified_fact",
                },
                applicability="matched_public_preference_structure",
                confidence=max(
                    0.0,
                    min(
                        0.7,
                        0.35
                        + 0.1 * len(selected["supporting_event_ids"])
                        + 0.05 * int(score == 1.0)
                        - 0.15 * int(temporal_extrapolation),
                    ),
                ),
                trace={
                    **context_trace,
                    "kernel": KERNEL_ID_V3,
                    "prediction_path": "preference_structure",
                    "selected_event_ids": list(
                        map(str, selected["supporting_event_ids"])
                    ),
                    "resolved_context_message_ids": resolved_ids,
                    "resolved_context_turns": len(resolved_ids),
                },
            )
        compound_query = bool(
            re.search(r"\b(?:and|also|together|both)\b|以及|同时|和", query, re.I)
        )
        analogous_ranked = sorted(
            (
                (
                    _similarity(
                        query,
                        f"{dict(frame['decision_frame'])['trigger']} "
                        f"{dict(frame['observed_response'])['verbatim']}",
                    ),
                    str(frame["event_frame_id"]),
                    copy.deepcopy(dict(frame)),
                )
                for frame in artifact.get("event_frames", [])
            ),
            key=lambda value: (-value[0], value[1]),
        )
        analogous = [
            frame
            for score, _, frame in analogous_ranked
            if score >= (0.1 if compound_query else 0.45)
        ][:3]
        if (compound_query and len(analogous) >= 2) or (
            not compound_query and analogous
        ):
            event_ids = [str(frame["event_frame_id"]) for frame in analogous]
            observed_claims = [
                str(dict(frame["observed_response"])["verbatim"])
                for frame in analogous
            ]
            return self._result(
                artifact,
                answer_status="similar_event_evidence_answer",
                claims=observed_claims,
                reasons=[],
                uncertainties=[
                    "These are related historical public responses, not a newly inferred person stance."
                ],
                evidence_event_ids=event_ids,
                response_basis={
                    "path": "similar_event_evidence",
                    "person_prediction_status": "analogical_evidence_not_new_stance",
                    "query_frame": self._query_frame(query, resolved_ids),
                    "selected_event_frame_ids": event_ids,
                    "selected_demonstrated_knowledge": self._relevant_knowledge(
                        artifact, query, domains
                    ),
                    "knowledge_boundary": "person_publicly_used_claim_not_verified_fact",
                },
                applicability="related_public_events_without_new_stance",
                confidence=0.0,
                trace={
                    **context_trace,
                    "kernel": KERNEL_ID_V3,
                    "prediction_path": "similar_event_evidence",
                    "selected_event_ids": event_ids,
                    "resolved_context_message_ids": resolved_ids,
                    "resolved_context_turns": len(resolved_ids),
                },
            )
        return self._result(
            artifact,
            answer_status="general_assisted",
            claims=[], reasons=[],
            uncertainties=["No eligible direct event or repeated public preference structure matched this question."],
            evidence_event_ids=[],
            response_basis={
                "path": "general_assisted",
                "person_prediction_status": "not_available",
                "query_frame": self._query_frame(query, resolved_ids),
                "external_knowledge_policy": "allowed_if_disclosed_not_person_knowledge",
            },
            applicability="general_knowledge_not_person_prediction",
            confidence=0.0,
            trace={**context_trace, "kernel": KERNEL_ID_V3, "prediction_path": "general_assisted", "resolved_context_message_ids": resolved_ids},
        )

    @staticmethod
    def _query_frame(query: str, resolved_ids: Sequence[str]) -> dict[str, object]:
        return {
            "query": query,
            "domain_tags": _domain_tags(query),
            "options": _extract_question_options(query),
            "target_terms": sorted({item for item in _terms(query) if item not in STOPWORDS and len(item) > 1})[:24],
            "resolved_context_message_ids": list(resolved_ids),
            "mentioned_years": [
                int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", query)
            ],
            "unknowns_are_not_model_filled": True,
        }

    @staticmethod
    def _result(
        artifact: Mapping[str, object], *, answer_status: str,
        claims: Sequence[str], reasons: Sequence[str], uncertainties: Sequence[str],
        evidence_event_ids: Sequence[str], response_basis: Mapping[str, object],
        applicability: str, confidence: float, trace: Mapping[str, object],
        ordinary_text: str = "",
    ) -> dict[str, object]:
        def units(kind: str, values: Sequence[str]) -> list[dict[str, object]]:
            return [
                {
                    "id": f"{kind}-{_hash([kind, value])[:16]}",
                    "text": str(value),
                    "evidence_ref": f"{kind}-{_hash([kind, value])[:16]}",
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
            "schema_version": PREDICTION_SCHEMA_V3,
            "person_id": str(artifact["person_id"]),
            "speech_act": {"label": "direct_answer", "probability": 0.0},
            "speech_act_distribution": [],
            "stance": {"label": "conditional_support" if answer_status == "preference_structure_answer" else "neutral", "probability": 0.0},
            "stance_distribution": [],
            "claims": units("claim", claims),
            "reasons": units("reason", reasons),
            "memories": [],
            "uncertainties": units("uncertainty", uncertainties),
            "answer_status": answer_status,
            "confidence": round(float(confidence), 6),
            "confidence_kind": "uncalibrated_evidence_support_not_accuracy",
            "applicability": applicability,
            "refusal_reasons": [],
            "evidence_refs": [],
            "evidence_event_ids": list(evidence_event_ids),
            "response_basis": copy.deepcopy(dict(response_basis)),
            "active_components": copy.deepcopy(list(artifact["active_components"])),
            "components": copy.deepcopy(list(artifact["components"])),
            "model_version": f"{artifact['person_id']}-simulation-v3-{artifact['version']}",
            "model_validity": "implemented_exploratory_accuracy_not_assessed",
            "valid_scope": copy.deepcopy(dict(artifact.get("scope", {}))),
        }
        contract = {
            "schema_version": FROZEN_CONTRACT_V3,
            "speech_act": str(dict(structured["speech_act"])["label"]),
            "stance": str(dict(structured["stance"])["label"]),
            "answer_status": answer_status,
            "refusal_status": "not_refused",
            "ordinary_dialogue_text": ordinary_text,
            "claims": [{"id": item["id"], "text": item["text"]} for item in structured["claims"]],
            "reasons": [{"id": item["id"], "text": item["text"]} for item in structured["reasons"]],
            "memories": [],
            "uncertainties": [{"id": item["id"], "text": item["text"]} for item in structured["uncertainties"]],
            "protected_entities": [],
            "protected_numbers": sorted(set(re.findall(r"\b\d+(?:[.,]\d+)*\b", " ".join([*claims, *reasons])))),
            "protected_dates": sorted(set(re.findall(r"\b\d{4}(?:-\d{2}-\d{2})?\b", " ".join([*claims, *reasons])))),
            "protected_quotes": [],
            "evidence_refs": [],
            "confidence": structured["confidence"],
            "style_mode": "interview_public",
        }
        structured["renderer_contract_digest"] = _hash(contract)
        content_digest = _hash(structured)
        structured["content_digest"] = content_digest
        return {
            "schema_version": PREDICTION_SCHEMA_V3,
            "status": "answered" if answer_status not in {"clarification_needed"} else "clarification",
            "answer_status": answer_status,
            "structured_prediction": structured,
            "renderer_contract": contract,
            "content_digest": content_digest,
            "renderer_contract_digest": _hash(contract),
            "prediction_trace": copy.deepcopy(dict(trace)),
        }
