from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .response_prediction import EVALUATION_TENDENCY_TYPES, TRADEOFF_TENDENCY_TYPES


REVIEWED_EVENT_SCHEMA_V4 = "pcfm-reviewed-public-response-event-v4"
MODEL_SCHEMA_V4 = "pcfm-simulation-model-v4"
PREDICTION_SCHEMA_V4 = "pcfm-simulation-prediction-v4"
FROZEN_CONTRACT_V4 = "pcfm-frozen-content-contract-v4"
KERNEL_ID_V4 = "simulation-v4"

ELIGIBLE_SOURCE_AUTHENTICITY = frozenset(
    {"verbatim_transcript", "verified_quote", "first_party_public_statement"}
)

INTERESTS: dict[str, dict[str, object]] = {
    "safety": {"label_zh": "安全与避免伤害", "aliases": ("safety", "safe", "harm prevention", "avoid harm", "安全", "避免伤害", "降低伤害")},
    "speed": {"label_zh": "速度与效率", "aliases": ("speed", "fast", "faster", "quick", "quickly", "efficiency", "速度", "快速", "效率")},
    "evidence_quality": {"label_zh": "证据质量", "aliases": ("evidence", "proof", "verification", "证据", "验证", "可靠证据")},
    "innovation": {"label_zh": "创新", "aliases": ("innovation", "innovate", "experiment", "创新", "实验", "探索")},
    "privacy": {"label_zh": "隐私", "aliases": ("privacy", "confidentiality", "隐私", "保密")},
    "transparency": {"label_zh": "透明度", "aliases": ("transparency", "openness", "disclosure", "透明", "公开", "披露")},
    "fairness": {"label_zh": "公平", "aliases": ("fairness", "equity", "equal treatment", "公平", "公正", "平等")},
    "economic_growth": {"label_zh": "经济增长", "aliases": ("growth", "economic growth", "prosperity", "增长", "经济增长", "繁荣")},
    "cost_control": {"label_zh": "成本控制", "aliases": ("cost", "affordability", "budget", "成本", "可负担", "预算")},
    "individual_freedom": {"label_zh": "个人自由", "aliases": ("freedom", "liberty", "autonomy", "自由", "自主", "个人选择")},
    "collective_welfare": {"label_zh": "公共福祉", "aliases": ("public welfare", "common good", "collective welfare", "公共福祉", "共同利益", "集体利益")},
    "stability": {"label_zh": "稳定性", "aliases": ("stability", "continuity", "predictability", "稳定", "连续性", "可预期")},
    "accountability": {"label_zh": "责任与问责", "aliases": ("accountability", "responsibility", "oversight", "问责", "责任", "监督")},
    "quality": {"label_zh": "质量", "aliases": ("quality", "reliability", "craft", "质量", "可靠性", "品质")},
    "inclusion": {"label_zh": "包容与参与", "aliases": ("inclusion", "participation", "access", "包容", "参与", "可及性")},
    "environment": {"label_zh": "环境保护", "aliases": ("environment", "climate", "sustainability", "环境", "气候", "可持续")},
    "national_security": {"label_zh": "国家安全", "aliases": ("national security", "defense", "security", "国家安全", "国防")},
    "diplomacy": {"label_zh": "外交合作", "aliases": ("diplomacy", "cooperation", "alliance", "外交", "合作", "联盟")},
    "rule_of_law": {"label_zh": "法治与程序", "aliases": ("rule of law", "due process", "procedure", "法治", "正当程序", "程序")},
    "competence": {
        "label_zh": "胜任能力与判断力",
        "aliases": (
            "competence", "competent", "fitness", "unfit", "prepared",
            "unprepared", "judgment", "temperament", "basic knowledge",
            "胜任", "能力", "判断力", "准备充分",
        ),
    },
    "institutional_norms": {
        "label_zh": "制度规范与宪政传统",
        "aliases": (
            "institutional norms", "norms and rules", "constitutional traditions",
            "democratic system", "checks and balances", "制度规范", "宪政传统",
            "民主制度", "权力制衡",
        ),
    },
}

DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "health": ("health", "hospital", "patient", "medical", "医疗", "医院", "患者", "健康"),
    "technology": ("technology", "software", "system", "algorithm", "ai", "技术", "软件", "系统", "算法", "人工智能"),
    "product": ("product", "launch", "prototype", "team", "产品", "发布", "原型", "团队"),
    "governance": (
        "government", "policy", "law", "regulation", "president", "election",
        "constitutional", "democratic", "政府", "政策", "法律", "监管", "总统",
        "选举", "宪法", "民主",
    ),
    "economics": ("market", "price", "business", "economy", "市场", "价格", "商业", "经济"),
    "social": ("rights", "society", "community", "权利", "社会", "社区"),
    "space": ("space", "nasa", "astronaut", "太空", "航天", "宇航员"),
    "aviation": ("aviation", "aircraft", "flight", "pilot", "航空", "飞机", "飞行", "飞行员"),
    "environment": ("environment", "climate", "energy", "环境", "气候", "能源"),
    "education": ("education", "school", "student", "教育", "学校", "学生"),
    "personal": ("family", "friend", "relationship", "private", "家庭", "朋友", "关系", "私人"),
}


class SimulationV4Error(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _normal(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", str(value).casefold()))


def _terms(value: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value).casefold()))
    for token in tuple(terms):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 1:
            terms.update(token[index : index + 2] for index in range(len(token) - 1))
    return terms


def _similarity(left: str, right: str) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / math.sqrt(len(left_terms) * len(right_terms))


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle.strip()) and needle.casefold() in haystack.casefold()


def _alias_in_text(text: str, alias: str) -> bool:
    if re.fullmatch(r"[a-z0-9 ]+", alias):
        return bool(re.search(rf"\b{re.escape(alias)}\b", text, re.I))
    return alias in text


def _interest_mentions(text: str) -> list[dict[str, str]]:
    mentions: dict[tuple[str, str], dict[str, str]] = {}
    for interest_id, definition in INTERESTS.items():
        for alias in definition["aliases"]:
            if _alias_in_text(text, str(alias)):
                mentions[(interest_id, str(alias).casefold())] = {
                    "span": str(alias),
                    "interest_id": interest_id,
                    "origin": "deterministic_closed_alias",
                }
    return [mentions[key] for key in sorted(mentions)]


def _span_maps_to_interest(span: str, interest_id: str) -> bool:
    if interest_id not in INTERESTS:
        return False
    return any(
        _alias_in_text(span, str(alias)) or _alias_in_text(str(alias), span)
        for alias in INTERESTS[interest_id]["aliases"]
    )


def _span_maps_to_domain(span: str, domain_id: str) -> bool:
    if domain_id not in DOMAIN_ALIASES:
        return False
    return any(
        _alias_in_text(span, str(alias)) or _alias_in_text(str(alias), span)
        for alias in DOMAIN_ALIASES[domain_id]
    )


def _domains(text: str) -> list[str]:
    return sorted(
        domain
        for domain, aliases in DOMAIN_ALIASES.items()
        if any(_alias_in_text(text, alias) for alias in aliases)
    )


def _source_units(source: Mapping[str, object]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for index, raw in enumerate(source.get("qas", []), start=1):
        if not isinstance(raw, Mapping):
            continue
        question = str(raw.get("question", "")).strip()
        response = str(raw.get("answer", "")).strip()
        if question and response:
            units.append(
                {
                    "question": question,
                    "response": response,
                    "locator": str(raw.get("locator", f"qa:{index}")),
                    "unit_kind": "confirmed_question_answer",
                }
            )
    if units or source.get("speaker_scope") == "mixed_speakers":
        return units
    for index, raw in enumerate(source.get("segments", []), start=1):
        text = str(raw.get("text", "") if isinstance(raw, Mapping) else raw).strip()
        if len(text) >= 24:
            units.append(
                {
                    "question": str(source.get("source_context") or source.get("title") or "public statement"),
                    "response": text,
                    "locator": str(raw.get("locator", f"segment:{index}")) if isinstance(raw, Mapping) else f"segment:{index}",
                    "unit_kind": "confirmed_single_speaker_span",
                }
            )
    return units


def _frame(
    source: Mapping[str, object],
    *,
    question: str,
    response: str,
    locator: str,
    origin: str,
    domain_ids: Sequence[str] = (),
    speaker_role: str = "public_speaker",
    audience: str = "unknown",
    conditions: Sequence[str] = (),
    reasons: Sequence[str] = (),
    tradeoffs: Sequence[Mapping[str, object]] = (),
    demonstrated_claim_spans: Sequence[str] = (),
) -> dict[str, object]:
    source_id = str(source["source_id"])
    event_id = f"event-v4-{_hash([source_id, locator, question, response])[:16]}"
    clean_domains = sorted({str(value) for value in domain_ids if str(value) in DOMAIN_ALIASES})
    if not clean_domains:
        clean_domains = _domains(f"{question} {response} {source.get('title', '')}") or ["general"]
    return {
        "schema_version": REVIEWED_EVENT_SCHEMA_V4,
        "event_frame_id": event_id,
        "person_id": str(source.get("person_id", "")),
        "source_id": source_id,
        "source_lineage": str(source.get("near_duplicate_of") or source_id),
        "source_role": str(source.get("dataset_role", "")),
        "origin": origin,
        "temporal_context": {
            "response_time": str(source.get("source_date", "")),
            "time_source": "reviewed_source_metadata",
        },
        "social_context": {
            "speaker": str(source.get("speaker", "unknown")),
            "speaker_role": speaker_role or "unknown",
            "audience": audience or "unknown",
        },
        "decision_frame": {
            "trigger": question,
            "conditions": sorted({str(value) for value in conditions if str(value).strip()}),
            "observed_tradeoffs": [copy.deepcopy(dict(value)) for value in tradeoffs],
        },
        "observed_response": {
            "verbatim": response,
            "reasons": sorted({str(value) for value in reasons if str(value).strip()}),
        },
        "domain_tags": clean_domains,
        "demonstrated_claim_spans": sorted(
            {str(value) for value in demonstrated_claim_spans if _contains(response, str(value))}
        ),
        "evidence": {
            "span_status": "exact_reviewed_source_span",
            "question_span": question,
            "response_span": response,
            "locator": locator,
            "content_hash": _hash([question, response]),
            "source_url": str(source.get("source_url", "")),
        },
    }


def _reviewed_frames(source: Mapping[str, object]) -> tuple[list[dict[str, object]], list[str]]:
    frames: list[dict[str, object]] = []
    rejected: list[str] = []
    source_text = str(source.get("text", ""))
    for index, raw in enumerate(source.get("reviewed_event_frames_v4", []), start=1):
        if not isinstance(raw, Mapping) or raw.get("review_status") != "confirmed":
            continue
        item = dict(raw)
        question = str(item.get("question", "")).strip()
        response = str(item.get("response", "")).strip()
        locator = str(item.get("source_locator", "")).strip()
        if not question or not response or not locator:
            rejected.append(f"reviewed:{index}:missing_exact_event_fields")
            continue
        if source_text and not _contains(source_text, response):
            rejected.append(f"reviewed:{index}:response_not_in_source")
            continue
        clean_tradeoffs: list[dict[str, object]] = []
        evidence_text = f"{question}\n{response}"
        for tradeoff_index, raw_tradeoff in enumerate(item.get("tradeoffs", []), start=1):
            if not isinstance(raw_tradeoff, Mapping):
                continue
            tradeoff = dict(raw_tradeoff)
            protected = str(tradeoff.get("protected_interest_id", ""))
            cost = str(tradeoff.get("accepted_cost_id", ""))
            protected_span = str(tradeoff.get("protected_interest_span", "")).strip()
            cost_span = str(tradeoff.get("accepted_cost_span", "")).strip()
            evidence_span = str(tradeoff.get("evidence_span", "")).strip()
            tendency_type = str(tradeoff.get("tendency_type", ""))
            is_evaluation = tendency_type in EVALUATION_TENDENCY_TYPES
            is_tradeoff = tendency_type in TRADEOFF_TENDENCY_TYPES
            valid = (
                protected in INTERESTS
                and (
                    (is_tradeoff and cost in INTERESTS and protected != cost)
                    or (is_evaluation and (not cost or cost in INTERESTS))
                )
                and _contains(evidence_text, protected_span)
                and (not cost_span or _contains(evidence_text, cost_span))
                and _contains(evidence_text, evidence_span)
            )
            if not valid:
                rejected.append(f"reviewed:{index}:tradeoff:{tradeoff_index}:ungrounded")
                continue
            clean_tradeoffs.append(
                {
                    "tendency_type": str(tradeoff.get("tendency_type", "")),
                    "direction": str(tradeoff.get("direction", "")),
                    "target": str(tradeoff.get("target", "")),
                    "protected_interest_id": protected,
                    "accepted_cost_id": cost,
                    "protected_interest_span": protected_span,
                    "accepted_cost_span": cost_span,
                    "evidence_span": evidence_span,
                    "semantic_status": "human_confirmed_model_candidate",
                }
            )
        frames.append(
            _frame(
                source,
                question=question,
                response=response,
                locator=locator,
                origin="reviewed_semantic_event",
                domain_ids=[str(value) for value in item.get("domain_ids", [])],
                speaker_role=str(item.get("speaker_role", "public_speaker")),
                audience=str(item.get("audience", "unknown")),
                conditions=[str(value) for value in item.get("conditions", [])],
                reasons=[str(value) for value in item.get("reasons", [])],
                tradeoffs=clean_tradeoffs,
                demonstrated_claim_spans=[str(value) for value in item.get("demonstrated_claim_spans", [])],
            )
        )
    return frames, rejected


def _preference_atoms(frames: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    atoms: list[dict[str, object]] = []
    rejected: list[str] = []
    for frame in frames:
        tradeoffs = list(dict(frame["decision_frame"]).get("observed_tradeoffs", []))
        if not tradeoffs:
            continue
        response_time = str(dict(frame["temporal_context"]).get("response_time", ""))
        if not response_time:
            rejected.append(f"{frame['event_frame_id']}:missing_time_for_preference")
            continue
        if not _valid_date(response_time):
            rejected.append(f"{frame['event_frame_id']}:invalid_time_for_preference")
            continue
        for tradeoff in tradeoffs:
            protected = str(tradeoff["protected_interest_id"])
            cost = str(tradeoff["accepted_cost_id"])
            atoms.append(
                {
                    "preference_atom_id": f"preference-v4-{_hash([frame['event_frame_id'], protected, cost])[:16]}",
                    "tendency_type": str(tradeoff.get("tendency_type", "")),
                    "direction": str(tradeoff.get("direction", "")),
                    "target": str(tradeoff.get("target", "")),
                    "event_frame_id": str(frame["event_frame_id"]),
                    "source_id": str(frame["source_id"]),
                    "source_lineage": str(frame["source_lineage"]),
                    "protected_interest_id": protected,
                    "accepted_cost_id": cost,
                    "conditions": list(dict(frame["decision_frame"]).get("conditions", [])),
                    "reasons": list(dict(frame["observed_response"]).get("reasons", [])),
                    "domain_tags": list(frame.get("domain_tags", [])),
                    "role": str(dict(frame["social_context"]).get("speaker_role", "unknown")),
                    "response_time": response_time,
                    "evidence_span": str(tradeoff["evidence_span"]),
                    "status": "reviewed_observable_public_tradeoff",
                    "interpretation_boundary": "public_response_pattern_not_private_value",
                }
            )
    atoms.sort(key=lambda item: str(item["preference_atom_id"]))
    return atoms, rejected


def _structures(atoms: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for raw in atoms:
        item = copy.deepcopy(dict(raw))
        grouped.setdefault(
            (str(item["protected_interest_id"]), str(item["accepted_cost_id"])), []
        ).append(item)
    structures: list[dict[str, object]] = []
    for pair, values in sorted(grouped.items()):
        protected, cost = pair
        lineages = sorted({str(item["source_lineage"]) for item in values})
        domains = sorted({str(domain) for item in values for domain in item.get("domain_tags", [])})
        roles = sorted({str(item.get("role", "unknown")) for item in values})
        times = sorted({str(item["response_time"]) for item in values})
        counters = grouped.get((cost, protected), [])
        if len(lineages) < 2:
            status = "insufficient_independent_evidence"
        elif len(domains) >= 2:
            status = "cross_domain_public_preference"
        else:
            status = "repeated_domain_public_preference"
        structures.append(
            {
                "preference_structure_id": f"structure-v4-{_hash(pair)[:16]}",
                "tendency_types": sorted(
                    {str(item.get("tendency_type", "")) for item in values if item.get("tendency_type")}
                ),
                "directions": sorted(
                    {str(item.get("direction", "")) for item in values if item.get("direction")}
                ),
                "protected_interest_id": protected,
                "accepted_cost_id": cost,
                "protected_interest_label": str(INTERESTS[protected]["label_zh"]),
                "accepted_cost_label": (
                    str(INTERESTS[cost]["label_zh"]) if cost in INTERESTS else "not_applicable"
                ),
                "supporting_event_ids": sorted(str(item["event_frame_id"]) for item in values),
                "supporting_atom_ids": sorted(str(item["preference_atom_id"]) for item in values),
                "independent_source_count": len(lineages),
                "source_lineages": lineages,
                "domain_count": len(domains),
                "primary_domains": domains,
                "role_scope": roles,
                "temporal_scope": {"start": times[0], "end": times[-1], "dated_event_count": len(times)},
                "conditions": sorted({str(value) for item in values for value in item.get("conditions", [])}),
                "reasons": sorted({str(value) for item in values for value in item.get("reasons", [])}),
                "counterevidence_event_ids": sorted(str(item["event_frame_id"]) for item in counters),
                "counterevidence_domains": sorted({str(domain) for item in counters for domain in item.get("domain_tags", [])}),
                "status": status,
                "support_status": "descriptive_independent_evidence_not_accuracy_probability",
            }
        )
    return structures


class SimulationKernelV4:
    kernel_id = KERNEL_ID_V4

    def fit(
        self,
        *,
        person_id: str,
        version: int,
        reviewed_sources: Sequence[Mapping[str, object]],
        scope: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        frames_by_hash: dict[str, dict[str, object]] = {}
        source_identities: list[dict[str, object]] = []
        rejected_sources: dict[str, list[str]] = {
            "not_reviewed_model_source": [],
            "unverified_source": [],
            "invalid_reviewed_semantics": [],
            "mixed_speaker_raw_units": [],
        }
        for raw in sorted(reviewed_sources, key=lambda item: str(item.get("source_id", ""))):
            source = copy.deepcopy(dict(raw))
            source_id = str(source.get("source_id", ""))
            if (
                source.get("review_status") != "confirmed"
                or source.get("dataset_role") != "model_source"
                or str(source.get("person_id", person_id)) != person_id
            ):
                rejected_sources["not_reviewed_model_source"].append(source_id)
                continue
            if source.get("content_authenticity") not in ELIGIBLE_SOURCE_AUTHENTICITY:
                rejected_sources["unverified_source"].append(source_id)
                continue
            source_identities.append(
                {
                    "source_id": source_id,
                    "source_lineage": str(source.get("near_duplicate_of") or source_id),
                    "content_hash": str(source.get("content_hash") or _hash([source.get("text", ""), source.get("qas", []), source.get("segments", [])])),
                    "response_time": str(source.get("source_date", "")),
                    "review_status": "confirmed",
                    "dataset_role": "model_source",
                }
            )
            reviewed, rejected = _reviewed_frames(source)
            rejected_sources["invalid_reviewed_semantics"].extend(f"{source_id}:{value}" for value in rejected)
            for frame in reviewed:
                frames_by_hash[str(dict(frame["evidence"])["content_hash"])] = frame
            if source.get("speaker_scope") == "mixed_speakers":
                if not reviewed:
                    rejected_sources["mixed_speaker_raw_units"].append(source_id)
                continue
            for unit in _source_units(source):
                direct = _frame(
                    source,
                    question=unit["question"],
                    response=unit["response"],
                    locator=unit["locator"],
                    origin="confirmed_direct_evidence_only",
                )
                frames_by_hash.setdefault(str(dict(direct["evidence"])["content_hash"]), direct)
        frames = sorted(frames_by_hash.values(), key=lambda item: str(item["event_frame_id"]))
        if not frames:
            raise SimulationV4Error("no_eligible_reviewed_event_frames")
        atoms, rejected_atoms = _preference_atoms(frames)
        structures = _structures(atoms)
        knowledge = sorted(
            [
                {
                    "knowledge_claim_id": f"knowledge-v4-{_hash([frame['event_frame_id'], span])[:16]}",
                    "statement": span,
                    "event_frame_id": str(frame["event_frame_id"]),
                    "source_id": str(frame["source_id"]),
                    "domain_tags": list(frame.get("domain_tags", [])),
                    "status": "exact_publicly_demonstrated_claim_not_complete_person_knowledge",
                }
                for frame in frames
                for span in frame.get("demonstrated_claim_spans", [])
            ],
            key=lambda item: str(item["knowledge_claim_id"]),
        )
        semantic_payload = {
            "source_identities": sorted(source_identities, key=lambda item: str(item["source_id"])),
            "event_frames": frames,
            "preference_atoms": atoms,
            "preference_structures": structures,
            "knowledge_claims": knowledge,
            "rejected_sources": {key: sorted(values) for key, values in rejected_sources.items()},
            "rejected_preference_atoms": sorted(rejected_atoms),
        }
        artifact: dict[str, object] = {
            "schema_version": MODEL_SCHEMA_V4,
            "kernel": KERNEL_ID_V4,
            "input_contract": "reviewed_event_frames_and_direct_evidence_v4",
            "person_id": person_id,
            "version": int(version),
            "created_at": _utc_now(),
            "scope": copy.deepcopy(dict(scope or {})),
            **semantic_payload,
            "semantic_model_digest": _hash(semantic_payload),
            "components": [
                {"component_id": "reviewed_event_frames_v4", "status": "active"},
                {"component_id": "lineage_bound_public_preferences_v4", "status": "active"},
                {"component_id": "validated_semantic_query_plan_v4", "status": "active"},
                {"component_id": "simulation_v3", "status": "frozen_baseline_only"},
            ],
            "active_components": [
                "reviewed_event_frames_v4",
                "lineage_bound_public_preferences_v4",
                "validated_semantic_query_plan_v4",
            ],
            "validation_status": "implemented_exploratory_accuracy_not_assessed",
            "accuracy_claim": "none",
        }
        artifact["artifact_hash"] = _hash(artifact)
        return artifact

    @staticmethod
    def verify(artifact: Mapping[str, object]) -> None:
        if artifact.get("schema_version") != MODEL_SCHEMA_V4 or artifact.get("kernel") != KERNEL_ID_V4:
            raise SimulationV4Error("unsupported_simulation_v4_schema")
        value = copy.deepcopy(dict(artifact))
        declared = str(value.pop("artifact_hash", ""))
        if _hash(value) != declared:
            raise SimulationV4Error("simulation_v4_integrity_failed")

    @staticmethod
    def _query(
        text: str,
        history: Sequence[Mapping[str, object]],
        query_plan: Mapping[str, object] | None,
    ) -> dict[str, object]:
        plan = dict(query_plan or {})
        history_by_id = {
            str(item.get("message_id")): item
            for item in history[-12:]
            if item.get("message_id") and item.get("role") == "user"
        }
        resolved_ids: list[str] = []
        rejected: list[str] = []
        resolved_texts: list[str] = []
        for raw_id in plan.get("resolved_message_ids", []):
            message_id = str(raw_id)
            item = history_by_id.get(message_id)
            if item is None:
                rejected.append(message_id)
                continue
            if message_id not in resolved_ids:
                resolved_ids.append(message_id)
                resolved_texts.append(str(item.get("text", "")))
        pure_followup = bool(
            re.fullmatch(
                r"\s*(?:why|what about that|continue|go on|为什么|那呢|继续|接着说)[?？\s]*",
                text,
                re.I,
            )
        )
        if pure_followup and not resolved_ids:
            for item in reversed(history[-12:]):
                prior = str(item.get("text", "")).strip()
                if item.get("role") == "user" and item.get("message_id") and prior:
                    resolved_ids.append(str(item["message_id"]))
                    resolved_texts.append(prior)
                    break
        combined = (
            resolved_texts[-1]
            if pure_followup and resolved_texts
            else "\n".join([*resolved_texts, text]).strip()
        )
        mentions = _interest_mentions(combined)
        known = {(item["span"].casefold(), item["interest_id"]) for item in mentions}
        for raw in plan.get("option_mentions", []):
            if not isinstance(raw, Mapping):
                rejected.append("invalid_option_mention")
                continue
            span = str(raw.get("span", "")).strip()
            interest_id = str(raw.get("interest_id", "")).strip()
            if not _contains(combined, span) or not _span_maps_to_interest(span, interest_id):
                rejected.append(f"option:{span or '<missing>'}:{interest_id or '<missing>'}")
                continue
            key = (span.casefold(), interest_id)
            if key not in known:
                mentions.append({"span": span, "interest_id": interest_id, "origin": "validated_model_semantic_candidate"})
                known.add(key)
        domain_ids = set(_domains(combined))
        for raw in plan.get("domain_mentions", []):
            if not isinstance(raw, Mapping):
                rejected.append("invalid_domain_mention")
                continue
            span = str(raw.get("span", "")).strip()
            domain_id = str(raw.get("domain_id", "")).strip()
            if not _contains(combined, span) or not _span_maps_to_domain(span, domain_id):
                rejected.append(f"domain_span:{span or '<missing>'}:{domain_id or '<missing>'}")
                continue
            domain_ids.add(domain_id)
        for value in plan.get("domain_ids", []):
            domain_id = str(value)
            if domain_id not in DOMAIN_ALIASES:
                rejected.append(f"domain:{domain_id}")
            elif domain_id not in domain_ids:
                rejected.append(f"domain:{domain_id}:ungrounded")
        selected_structure_ids = sorted({str(value) for value in plan.get("selected_structure_ids", [])})
        selected_event_ids = sorted({str(value) for value in plan.get("selected_event_ids", [])})
        role = "private" if re.search(r"\b(?:private|personal|family|father|mother|friend)\b|私人|家庭|家人|朋友", combined, re.I) else "unknown"
        if role == "unknown" and str(plan.get("role", "")) in {"public", "private", "unknown"}:
            role = str(plan.get("role"))
        years = [int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", combined)]
        current_marker = bool(re.search(r"\b(?:today|now|currently)\b|现在|如今|当前|今天", combined, re.I))
        relative_future_marker = bool(
            re.search(
                r"\b(?:future|tomorrow|next\s+(?:year|month|week)|going forward)\b|"
                r"未来|明年|下个月|下周|以后|今后",
                combined,
                re.I,
            )
        )
        return {
            "query": text,
            "combined_query": combined,
            "domain_ids": sorted(domain_ids),
            "option_mentions": sorted(mentions, key=lambda item: (item["interest_id"], item["span"])),
            "option_ids": sorted({str(item["interest_id"]) for item in mentions}),
            "role": role,
            "mentioned_years": years,
            "current_time_marker": current_marker,
            "relative_future_marker": relative_future_marker,
            "resolved_message_ids": resolved_ids,
            "selected_structure_ids": selected_structure_ids,
            "selected_event_ids": selected_event_ids,
            "rejected_fields": rejected,
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
            raise SimulationV4Error("message_required")
        query = self._query(clean, history, query_plan)
        allowed_structure_ids = {
            str(item["preference_structure_id"])
            for item in artifact.get("preference_structures", [])
        }
        allowed_event_ids = {
            str(item["event_frame_id"]) for item in artifact.get("event_frames", [])
        }
        invalid_structure_ids = [
            value
            for value in query["selected_structure_ids"]
            if value not in allowed_structure_ids
        ]
        invalid_event_ids = [
            value for value in query["selected_event_ids"] if value not in allowed_event_ids
        ]
        query["selected_structure_ids"] = [
            value
            for value in query["selected_structure_ids"]
            if value in allowed_structure_ids
        ]
        query["selected_event_ids"] = [
            value for value in query["selected_event_ids"] if value in allowed_event_ids
        ]
        query["rejected_fields"].extend(
            [
                *[f"structure:{value}" for value in invalid_structure_ids],
                *[f"event:{value}" for value in invalid_event_ids],
            ]
        )
        trace = {
            "kernel": KERNEL_ID_V4,
            "context_digest": _hash([clean, [(item.get("message_id"), item.get("role"), item.get("text")) for item in history[-12:]]]),
            "conversation_context": copy.deepcopy(dict(conversation_context or {})),
            "resolved_context_message_ids": list(query["resolved_message_ids"]),
            "resolved_context_turns": len(query["resolved_message_ids"]),
            "rejected_query_plan_fields": list(query["rejected_fields"]),
            "generative_content_calls": 0,
            "context_used": {
                "message_ids": [
                    str(item.get("message_id"))
                    for item in history[-12:]
                    if item.get("message_id")
                ],
                "turn_count": len(history[-12:]),
                "generated_context_count": sum(
                    item.get("context_role") == "model_generated_context"
                    for item in history[-12:]
                ),
                "generated_context_is_fitting_evidence": False,
            },
        }
        compact = re.sub(r"\W+", "", clean.casefold())
        ordinary = {
            "hi": "你好。你想从哪件事开始聊？",
            "hello": "你好。你想从哪件事开始聊？",
            "你好": "你好。你想从哪件事开始聊？",
            "thanks": "不客气。",
            "thankyou": "不客气。",
            "谢谢": "不客气。",
            "接着说": "可以。你想继续前面的哪一件事？",
            "continue": "可以。你想继续前面的哪一件事？",
            "goon": "可以。你想继续前面的哪一件事？",
            "浣犲ソ": "你好。你想从哪件事开始聊？",
            "璋㈣阿": "不客气。",
            "鎺ョ潃璇": "可以。你想继续前面的哪一件事？",
        }.get(compact)
        if ordinary:
            return self._result(
                artifact,
                answer_status="ordinary_dialogue",
                claims=[],
                reasons=[],
                uncertainties=[],
                evidence_event_ids=[],
                response_basis={"path": "ordinary_dialogue", "person_prediction_status": "not_applicable"},
                applicability="ordinary_dialogue_content_free",
                support=1.0,
                trace={**trace, "prediction_path": "ordinary_dialogue"},
                ordinary_text=ordinary,
            )
        ranked = sorted(
            [
                (
                    _similarity(str(query["combined_query"]), str(dict(frame["decision_frame"])["trigger"])),
                    str(frame["event_frame_id"]),
                    copy.deepcopy(dict(frame)),
                )
                for frame in artifact.get("event_frames", [])
            ],
            key=lambda value: (-value[0], value[1]),
        )
        exact = next(
            (frame for _, _, frame in ranked if _normal(str(dict(frame["decision_frame"])["trigger"])) == _normal(str(query["combined_query"]))),
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
                claims=[response],
                reasons=list(dict(exact["observed_response"]).get("reasons", [])),
                uncertainties=["这是历史公开回答，不是对新情境的预测。"],
                evidence_event_ids=[str(exact["event_frame_id"])],
                response_basis={
                    "path": "direct_historical_response",
                    "person_prediction_status": "direct_historical_evidence",
                    "query_frame": query,
                    "selected_event_frame_ids": [str(exact["event_frame_id"])],
                    "selected_demonstrated_knowledge": [item for item in artifact.get("knowledge_claims", []) if item.get("event_frame_id") == exact["event_frame_id"]],
                    "knowledge_boundary": "exact_publicly_demonstrated_claims_only_not_complete_person_knowledge",
                },
                applicability="exact_or_near_identical_reviewed_event",
                support=1.0,
                trace={**trace, "prediction_path": "direct_event", "selected_event_ids": [str(exact["event_frame_id"])]},
            )
        option_ids = set(map(str, query["option_ids"]))
        domains = set(map(str, query["domain_ids"]))
        candidates: list[tuple[float, str, dict[str, object]]] = []
        blocked_reasons: set[str] = set()
        current_year = datetime.now(timezone.utc).year
        for raw in artifact.get("preference_structures", []):
            structure = copy.deepcopy(dict(raw))
            if structure.get("status") not in {"cross_domain_public_preference", "repeated_domain_public_preference"}:
                continue
            pair = {str(structure["protected_interest_id"]), str(structure["accepted_cost_id"])}
            explicit_pair = pair <= option_ids
            if not explicit_pair:
                continue
            if query["role"] == "private" and not any("private" in str(value) or "personal" in str(value) for value in structure.get("role_scope", [])):
                blocked_reasons.add("role_transfer_not_supported")
                continue
            evidence_end = str(dict(structure["temporal_scope"]).get("end", ""))
            evidence_year = int(evidence_end[:4]) if re.match(r"^\d{4}", evidence_end) else None
            target_years = list(query["mentioned_years"])
            if query["current_time_marker"]:
                target_years.append(current_year)
            if query["relative_future_marker"]:
                blocked_reasons.add("later_than_evidence_window")
                continue
            if evidence_year is not None and target_years and max(target_years) > evidence_year:
                blocked_reasons.add("later_than_evidence_window")
                continue
            cross_domain = structure["status"] == "cross_domain_public_preference"
            if not cross_domain and not (domains & set(map(str, structure.get("primary_domains", [])))):
                blocked_reasons.add("domain_transfer_not_supported")
                continue
            counter_domains = set(map(str, structure.get("counterevidence_domains", [])))
            if structure.get("counterevidence_event_ids") and (not domains or bool(domains & counter_domains)):
                blocked_reasons.add("unresolved_contextual_counterevidence")
                continue
            score = (
                0.4
                + 0.08 * min(3, int(structure["independent_source_count"]))
                + 0.06 * min(3, int(structure["domain_count"]))
                + 0.08 * int(explicit_pair)
            )
            candidates.append((score, str(structure["preference_structure_id"]), structure))
        candidates.sort(key=lambda value: (-value[0], value[1]))
        if candidates:
            support, _, selected = candidates[0]
            protected = str(selected["protected_interest_id"])
            cost = str(selected["accepted_cost_id"])
            if re.search(r"[\u4e00-\u9fff]", clean):
                statement = f"根据重复且独立的公开回应，在相近公开条件下，这个人更可能优先考虑{INTERESTS[protected]['label_zh']}，而不是{INTERESTS[cost]['label_zh']}。"
            else:
                statement = f"Based on repeated independent public responses, this person would more likely prioritize {protected} over {cost} under comparable public conditions."
            return self._result(
                artifact,
                answer_status="preference_structure_answer",
                claims=[statement],
                reasons=[f"The direction is supported by {selected['independent_source_count']} independent source lineages across {selected['domain_count']} recorded domains."],
                uncertainties=["这是未经准确率校准的公开行为推断，不代表人物真实内心，也不证明未来保持稳定。"],
                evidence_event_ids=list(map(str, selected["supporting_event_ids"])),
                response_basis={
                    "path": "value_conflict_projection",
                    "person_prediction_status": "public_preference_projection",
                    "query_frame": query,
                    "selected_preference_structure": selected,
                    "protected_interest_id": protected,
                    "accepted_cost_id": cost,
                    "prediction_statement": statement,
                    "selected_demonstrated_knowledge": [],
                    "knowledge_boundary": "exact_public_claims_only_general_model_knowledge_is_external",
                },
                applicability="matched_reviewed_public_preference_structure",
                support=min(0.75, support),
                trace={**trace, "prediction_path": "preference_structure", "selected_event_ids": list(map(str, selected["supporting_event_ids"]))},
            )
        planned_events = [event_id for event_id in query["selected_event_ids"] if event_id in allowed_event_ids]
        if planned_events:
            selected_frames = [frame for frame in artifact.get("event_frames", []) if str(frame["event_frame_id"]) in planned_events and (not domains or domains & set(map(str, frame.get("domain_tags", []))))]
            if selected_frames:
                return self._result(
                    artifact,
                    answer_status="similar_event_evidence_answer",
                    claims=[str(dict(frame["observed_response"])["verbatim"]) for frame in selected_frames[:3]],
                    reasons=[],
                    uncertainties=["这些是相关历史公开回应，不是对新问题推导出的新人物立场。"],
                    evidence_event_ids=[str(frame["event_frame_id"]) for frame in selected_frames[:3]],
                    response_basis={
                        "path": "similar_event_evidence",
                        "person_prediction_status": "analogical_evidence_not_new_stance",
                        "query_frame": query,
                        "person_prediction_refusal_reasons": sorted(blocked_reasons),
                    },
                    applicability="related_reviewed_events_without_new_stance",
                    support=0.0,
                    trace={**trace, "prediction_path": "similar_event_evidence", "selected_event_ids": [str(frame["event_frame_id"]) for frame in selected_frames[:3]]},
                )
        compound = bool(
            re.search(r"\b(?:and|also|both|together)\b|以及|同时|并且", clean, re.I)
        )
        related_ranked: list[tuple[float, str, dict[str, object]]] = []
        query_terms = _terms(str(query["combined_query"]))
        for raw in (artifact.get("event_frames", []) if domains else []):
            frame = copy.deepcopy(dict(raw))
            if domains and not (domains & set(map(str, frame.get("domain_tags", [])))):
                continue
            candidate_text = (
                f"{dict(frame['decision_frame']).get('trigger', '')} "
                f"{dict(frame['observed_response']).get('verbatim', '')}"
            )
            if len(query_terms & _terms(candidate_text)) < 2:
                continue
            score = _similarity(str(query["combined_query"]), candidate_text)
            if score >= 0.15:
                related_ranked.append(
                    (score, str(frame["event_frame_id"]), frame)
                )
        related_ranked.sort(key=lambda value: (-value[0], value[1]))
        related = [frame for _, _, frame in related_ranked[:3]]
        if related and (not compound or len(related) >= 2):
            event_ids = [str(frame["event_frame_id"]) for frame in related]
            return self._result(
                artifact,
                answer_status="similar_event_evidence_answer",
                claims=[
                    str(dict(frame["observed_response"])["verbatim"])
                    for frame in related
                ],
                reasons=[],
                uncertainties=[
                    "这些是代码门禁筛选出的相关历史公开回应，不是对新问题推导出的新人物立场。"
                ],
                evidence_event_ids=event_ids,
                response_basis={
                    "path": "similar_event_evidence",
                    "person_prediction_status": "analogical_evidence_not_new_stance",
                    "query_frame": query,
                    "person_prediction_refusal_reasons": sorted(blocked_reasons),
                },
                applicability="related_reviewed_events_without_new_stance",
                support=0.0,
                trace={
                    **trace,
                    "prediction_path": "similar_event_evidence",
                    "selected_event_ids": event_ids,
                },
            )
        return self._result(
            artifact,
            answer_status="general_assisted",
            claims=[],
            reasons=[],
            uncertainties=["没有找到可适用的直接事件或重复公开倾向；通用回答不得归因于该人物。"],
            evidence_event_ids=[],
            response_basis={
                "path": "general_assisted",
                "person_prediction_status": "not_available",
                "query_frame": query,
                "person_prediction_refusal_reasons": sorted(blocked_reasons) or ["no_applicable_reviewed_person_structure"],
                "external_knowledge_policy": "allowed_if_disclosed_not_person_knowledge",
            },
            applicability="general_knowledge_not_person_prediction",
            support=0.0,
            trace={**trace, "prediction_path": "general_assisted"},
        )

    def evaluate(
        self,
        artifact: Mapping[str, object],
        holdout_sources: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        self.verify(artifact)
        training_end = max(
            (
                str(item.get("response_time", ""))
                for item in artifact.get("source_identities", [])
                if _valid_date(str(item.get("response_time", "")))
            ),
            default="",
        )
        training_source_ids = {
            str(item.get("source_id", ""))
            for item in artifact.get("source_identities", [])
        }
        holdout_source_ids = {
            str(item.get("source_id", "")) for item in holdout_sources
        }
        leakage = sorted(training_source_ids & holdout_source_ids)
        cases: list[dict[str, object]] = []
        rejected: dict[str, list[str]] = {"not_confirmed_final_holdout": [], "not_strictly_later": [], "no_reviewed_tradeoff": []}
        for raw in sorted(holdout_sources, key=lambda item: str(item.get("source_id", ""))):
            source = dict(raw)
            source_id = str(source.get("source_id", ""))
            if source.get("review_status") != "confirmed" or source.get("dataset_role") != "final_holdout":
                rejected["not_confirmed_final_holdout"].append(source_id)
                continue
            source_time = str(source.get("source_date", ""))
            if not training_end or not source_time or source_time <= training_end:
                rejected["not_strictly_later"].append(source_id)
                continue
            frames, _ = _reviewed_frames(source)
            for frame in frames:
                tradeoffs = list(dict(frame["decision_frame"]).get("observed_tradeoffs", []))
                if not tradeoffs:
                    rejected["no_reviewed_tradeoff"].append(str(frame["event_frame_id"]))
                    continue
                prediction = self.predict(artifact, text=str(dict(frame["decision_frame"])["trigger"]), history=[])
                basis = dict(dict(prediction["structured_prediction"]).get("response_basis", {}))
                predicted = (str(basis.get("protected_interest_id", "")), str(basis.get("accepted_cost_id", "")))
                expected = {(str(item["protected_interest_id"]), str(item["accepted_cost_id"])) for item in tradeoffs}
                made = prediction["answer_status"] == "preference_structure_answer"
                cases.append({"source_id": source_id, "event_frame_id": frame["event_frame_id"], "person_prediction_made": made, "direction_correct": made and predicted in expected})
        if not cases:
            return {"status": "not_assessed", "reason": "no_strictly_later_reviewed_tradeoff_cases", "sample_count": 0, "training_end": training_end, "holdout_leakage_source_ids": leakage, "rejected": rejected, "accuracy_claim": "none"}
        covered = [item for item in cases if item["person_prediction_made"]]
        correct = [item for item in covered if item["direction_correct"]]
        return {
            "status": "invalid_holdout_leakage" if leakage else "assessed_exploratory",
            "sample_count": len(cases),
            "coverage": round(len(covered) / len(cases), 6),
            "covered_direction_accuracy": round(len(correct) / len(covered), 6) if covered else "not_assessed",
            "training_end": training_end,
            "holdout_leakage_source_ids": leakage,
            "cases": cases,
            "rejected": rejected,
            "accuracy_claim": "exploratory_only_not_guaranteed",
        }

    @staticmethod
    def _result(
        artifact: Mapping[str, object],
        *,
        answer_status: str,
        claims: Sequence[str],
        reasons: Sequence[str],
        uncertainties: Sequence[str],
        evidence_event_ids: Sequence[str],
        response_basis: Mapping[str, object],
        applicability: str,
        support: float,
        trace: Mapping[str, object],
        ordinary_text: str = "",
    ) -> dict[str, object]:
        def units(kind: str, values: Sequence[str]) -> list[dict[str, object]]:
            return [
                {
                    "id": f"{kind}-v4-{_hash([kind, value])[:16]}",
                    "text": str(value),
                    "evidence_event_id": str(evidence_event_ids[min(index, len(evidence_event_ids) - 1)]) if evidence_event_ids else "",
                    "probability": 0.0,
                }
                for index, value in enumerate(values)
            ]
        structured: dict[str, object] = {
            "schema_version": PREDICTION_SCHEMA_V4,
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
            "confidence": round(float(support), 6),
            "confidence_kind": "evidence_support_not_accuracy_probability",
            "applicability": applicability,
            "refusal_reasons": [],
            "evidence_refs": [],
            "evidence_event_ids": list(evidence_event_ids),
            "response_basis": copy.deepcopy(dict(response_basis)),
            "active_components": copy.deepcopy(list(artifact["active_components"])),
            "components": copy.deepcopy(list(artifact["components"])),
            "model_version": f"{artifact['person_id']}-simulation-v4-{artifact['version']}",
            "model_validity": "implemented_exploratory_accuracy_not_assessed",
            "valid_scope": copy.deepcopy(dict(artifact.get("scope", {}))),
        }
        contract = {
            "schema_version": FROZEN_CONTRACT_V4,
            "speech_act": "direct_answer",
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
            "schema_version": PREDICTION_SCHEMA_V4,
            "status": "answered",
            "answer_status": answer_status,
            "structured_prediction": structured,
            "renderer_contract": contract,
            "content_digest": content_digest,
            "renderer_contract_digest": _hash(contract),
            "prediction_trace": copy.deepcopy(dict(trace)),
        }
