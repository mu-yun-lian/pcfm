from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np

from .contracts import Observation, PersonalAdapter, Scenario
from .core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    PopulationPriorEstimator,
)
from .interfaces import PopulationModel


EVENT_SCHEMA = "pcfm-response-event-v2"
MODEL_SCHEMA = "pcfm-unified-response-model-v1"
PREDICTION_SCHEMA = "pcfm-structured-response-v1"
KERNEL_ID = "pcfm-response-kernel-v1"
FEATURE_DIMENSION = 72
FEATURE_NAMES = tuple(f"response_feature_{index:03d}" for index in range(FEATURE_DIMENSION))

APPLICABILITY_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "can", "could", "did", "do", "does",
        "for", "how", "i", "in", "is", "it", "of", "on", "or", "our",
        "should", "that", "the", "this", "to", "was", "we", "what",
        "when", "where", "which", "who", "why", "with", "would", "you",
        "your",
    }
)

SPEECH_ACTS = (
    "direct_answer",
    "challenge_then_answer",
    "rebut",
    "reframe",
    "ask_followup",
    "evade",
    "refuse",
    "admit_unknown",
)
STANCES = (
    "support",
    "oppose",
    "neutral",
    "conditional_support",
    "mixed",
    "insufficient_evidence",
)

# 公开反应倾向的 8 类封闭分类（operationized，非心理标签）。
# 每个伴随倾向原子（tradeoff）都必须属于其中一类。
TENDENCY_TYPES = (
    "object_evaluation",            # 对对象的支持/反对/混合评价
    "principle_priority",           # 行为原则和优先级
    "conditional_policy_preference",  # 条件性政策偏好
    "means_ends",                   # 手段—目的判断
    "responsibility_attribution",   # 责任归属
    "risk_tolerance",               # 风险容忍度
    "rule_procedure_tradeoff",      # 对规则/程序/结果/公平的权衡
    "behavior_evaluation",          # 对人物行为方式的评价
)

# 取舍类倾向（有 preferred/sacrificed 双方）；评价类倾向（有 target+direction）
TRADEOFF_TENDENCY_TYPES = frozenset(
    {
        "principle_priority",
        "conditional_policy_preference",
        "means_ends",
        "risk_tolerance",
        "rule_procedure_tradeoff",
    }
)
EVALUATION_TENDENCY_TYPES = frozenset(
    {
        "object_evaluation",
        "behavior_evaluation",
        "responsibility_attribution",
    }
)

DATA_ROLE_MAP = {
    "model_source": "parameter_training",
    "applicability_reference": "applicability_calibration",
    "feature_discovery": "feature_discovery",
    "candidate_selection": "candidate_selection",
    "final_holdout": "sealed_final_validation",
    "post_deployment_monitoring": "post_deployment_monitoring",
    "reference_only": "external_reality_comparison",
}

TRAINABLE_AUTHENTICITY = frozenset(
    {"verbatim_transcript", "verified_quote", "verified_translation"}
)

EVENT_TYPE_KEYWORDS = {
    "product_strategy": {"product", "launch", "release", "design", "prototype", "产品", "发布", "设计", "原型"},
    "technology_science": {"technology", "science", "software", "ai", "space", "技术", "科学", "软件", "人工智能", "太空"},
    "economics_business": {"market", "business", "company", "price", "growth", "市场", "商业", "公司", "价格", "增长"},
    "governance_policy": {"policy", "government", "law", "regulation", "public", "政策", "政府", "法律", "监管", "公共"},
    "ethics_social": {"fair", "rights", "society", "responsibility", "公平", "权利", "社会", "责任"},
    "risk_crisis": {"risk", "safety", "crisis", "uncertain", "风险", "安全", "危机", "不确定"},
    "personal_relations": {"family", "friend", "team", "relationship", "家庭", "朋友", "团队", "关系"},
}

# Keep valid UTF-8 Chinese vocabulary separate from legacy imported literals.
EVENT_TYPE_KEYWORDS["product_strategy"].update({"产品", "发布", "设计", "原型"})
EVENT_TYPE_KEYWORDS["technology_science"].update(
    {
        "技术",
        "科学",
        "软件",
        "人工智能",
        "太空",
        "artificial intelligence",
        "nasa",
        "astronaut",
    }
)
EVENT_TYPE_KEYWORDS["economics_business"].update(
    {"市场", "商业", "公司", "价格", "增长"}
)
EVENT_TYPE_KEYWORDS["governance_policy"].update(
    {"政策", "政府", "法律", "监管", "公共"}
)
EVENT_TYPE_KEYWORDS["ethics_social"].update(
    {"公平", "权利", "社会", "责任"}
)
EVENT_TYPE_KEYWORDS["risk_crisis"].update(
    {"风险", "安全", "危机", "不确定"}
)
EVENT_TYPE_KEYWORDS["personal_relations"].update(
    {"家庭", "朋友", "团队", "关系"}
)


class ResponsePredictionError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_hash(value: object) -> str:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def tokens(value: str) -> tuple[str, ...]:
    """Return boundary-safe English words and searchable Chinese terms.

    Chinese was previously emitted one character at a time and then discarded
    by the topic-term builder.  Keep the full run and stable bi/tri-grams so
    related phrases can overlap without adding a tokenizer dependency.
    """
    result: list[str] = []
    for item in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value).casefold()):
        if not re.fullmatch(r"[\u4e00-\u9fff]+", item):
            result.append(item)
            continue
        result.append(item)
        if len(item) > 1:
            result.extend(item[index : index + 2] for index in range(len(item) - 1))
        if len(item) > 2:
            result.extend(item[index : index + 3] for index in range(len(item) - 2))
    return tuple(result)


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(
        len(left_tokens) * len(right_tokens)
    )


def applicability_similarity(left: str, right: str) -> float:
    left_tokens = {value for value in tokens(left) if value not in APPLICABILITY_STOPWORDS}
    right_tokens = {value for value in tokens(right) if value not in APPLICABILITY_STOPWORDS}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / math.sqrt(
        len(left_tokens) * len(right_tokens)
    )


def _sentences(text: str) -> list[str]:
    return [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？])\s+|\n+", str(text).strip())
        if item.strip()
    ]


def _speech_act(answer: str) -> str:
    lowered = answer.casefold()
    if re.search(r"\b(i do not know|i don't know|不知道|不清楚)\b", lowered):
        return "admit_unknown"
    if re.search(r"\b(cannot discuss|can't discuss|won't discuss|拒绝回答|不便回答)\b", lowered):
        return "refuse"
    if "?" in answer and len(answer) < 100:
        return "ask_followup"
    if re.search(r"\b(the premise|your premise|that assumption|前提|假设)\b", lowered):
        return "challenge_then_answer"
    if re.search(r"\b(the real question|actually|换句话说|真正的问题)\b", lowered):
        return "reframe"
    if re.match(r"\s*(no|wrong|不|不是|错)", lowered):
        return "rebut"
    return "direct_answer"


def _stance(answer: str) -> str:
    lowered = answer.casefold()
    conditional = re.search(
        r"\b(if|unless|provided|only after|when)\b|如果|除非|只有|取决于", lowered
    )
    positive = bool(
        re.search(r"\b(yes|should|must|support|agree)\b|应该|必须|支持|赞成", lowered)
    )
    negative = bool(
        re.search(r"\b(no|not|never|oppose|disagree|shouldn't|should not)\b|反对|不应|不能|不要", lowered)
    )
    if positive and negative:
        return "mixed"
    if conditional:
        return "conditional_support"
    if negative:
        return "oppose"
    if positive:
        return "support"
    return "neutral"


def _content_units(answer: str) -> tuple[list[str], list[str], list[str]]:
    claims: list[str] = []
    reasons: list[str] = []
    uncertainties: list[str] = []
    for sentence in _sentences(answer):
        lowered = sentence.casefold()
        if re.search(r"\b(because|since|therefore|thus|due to)\b|因为|由于|所以|因此", lowered):
            reasons.append(sentence)
        else:
            claims.append(sentence)
        if re.search(r"\b(may|might|uncertain|unknown|not sure|depends)\b|可能|不确定|尚不清楚|取决于", lowered):
            uncertainties.append(sentence)
    if not claims and reasons:
        claims.append(reasons.pop(0))
    return claims, reasons, uncertainties


def _keyword_matches(keyword: str, raw: str, combined: set[str]) -> bool:
    normalized = keyword.casefold().strip()
    if not normalized:
        return False
    if re.search(r"[\u4e00-\u9fff]", normalized):
        return any(normalized in token for token in combined)
    if " " in normalized:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", raw))
    return normalized in combined


def classify_event_types(*values: str) -> tuple[str, ...]:
    raw = " ".join(values).casefold()
    combined = set(tokens(raw))
    scored = sorted(
        (
            (
                sum(
                    1
                    for keyword in keywords
                    if _keyword_matches(keyword, raw, combined)
                ),
                event_type,
            )
            for event_type, keywords in EVENT_TYPE_KEYWORDS.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    return tuple(event_type for score, event_type in scored if score > 0) or ("general",)


def _event_type(*values: str) -> str:
    return classify_event_types(*values)[0]


def classify_event_type(text: str) -> str:
    return _event_type(text)


def _time_precision(value: str) -> str:
    clean = str(value).strip()
    if not clean:
        return "unknown"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ].*)?", clean):
        return "day_or_finer"
    if re.fullmatch(r"\d{4}-\d{2}", clean):
        return "month"
    if re.fullmatch(r"\d{4}", clean):
        return "year"
    return "unverified_text"


def _event_completeness(
    *, question: str, answer: str, source: Mapping[str, object]
) -> dict[str, object]:
    response_time = str(source.get("source_date", "")).strip()
    context = str(source.get("source_context", "")).strip()
    missing = [
        name
        for value, name in (
            (question, "trigger"),
            (question, "target"),
            (answer, "public_response"),
            (context, "context"),
            (response_time, "response_time"),
        )
        if not str(value).strip()
    ]
    allowed = ["source_reference", "style_evidence"]
    if question and answer:
        allowed.extend(["direct_response_evidence", "topic_retrieval"])
    if question and answer and context and response_time:
        allowed.extend(
            [
                "response_head_training",
                "conditional_tendency",
                "temporal_prediction",
                "sealed_temporal_validation",
                "reality_optimization_training",
            ]
        )
    prohibited = [
        use
        for condition, use in (
            (bool(response_time), "temporal_prediction"),
            (bool(response_time), "sealed_temporal_validation"),
            (bool(response_time), "reality_optimization_training"),
            (bool(question and context), "conditional_tendency"),
        )
        if not condition
    ]
    return {
        "missing_fields": missing,
        "temporal_status": "known" if response_time else "unknown",
        "context_status": "known" if context else "incomplete",
        "allowed_uses": sorted(set(allowed)),
        "prohibited_uses": sorted(set(prohibited)),
        "unknowns_are_not_model_filled": True,
    }


def _event_semantics(
    *, question: str, answer: str, source: Mapping[str, object]
) -> dict[str, object]:
    claims, reasons, uncertainties = _content_units(answer)
    stance = _stance(answer)
    event_types = classify_event_types(
        question,
        answer,
        str(source.get("title", "")),
        str(source.get("source_context", "")),
    )
    event_type = event_types[0]
    topic_terms = sorted(
        {
            value
            for value in tokens(" ".join((question, answer)))
            if value not in APPLICABILITY_STOPWORDS and len(value) > 1
        }
    )[:24]
    completeness = _event_completeness(
        question=question, answer=answer, source=source
    )
    target = question.strip() or "unknown"
    response_time = str(source.get("source_date", "")).strip()
    tendency_atoms = []
    if claims:
        tendency_atoms.append(
            {
                "kind": "observable_public_response_tendency",
                "target": target,
                "direction": stance,
                "condition": {
                    "event_type": event_type,
                    "event_types": list(event_types),
                    "occasion": str(source.get("source_context", ""))
                    or str(source.get("title", "")),
                    "role": "public_speaker",
                    "observed_at": str(source.get("source_date", "")),
                },
                "stance": stance,
                "conditions": [
                    value
                    for value in (
                        str(source.get("source_context", "")).strip(),
                        *uncertainties,
                    )
                    if value
                ],
                "tradeoffs": list(reasons),
                "exceptions": list(uncertainties),
                "domain": event_type,
                "temporal_scope": {
                    "response_time": response_time or "unknown",
                    "precision": _time_precision(response_time),
                },
                "expressed_outcome": claims[0],
                "supporting_event_ids": [],
                "counterevidence_event_ids": [],
                "status": "single_event_candidate",
            }
        )
    knowledge_items = [
        {
            "statement": value,
            "knowledge_kind": "person_demonstrated_claim_not_verified_fact",
            "domain": event_type,
            "temporal_status": completeness["temporal_status"],
            "status": "publicly_used_claim_not_verified_fact",
        }
        for value in [*reasons, *claims]
    ]
    return {
        "speech_act": _speech_act(answer),
        "stance": stance,
        "claims": claims,
        "reasons": reasons,
        "uncertainties": uncertainties,
        "event_atom": {
            "event_type": event_type,
            "event_types": list(event_types),
            "topic_terms": topic_terms,
            "trigger": question.strip() or "unknown",
            "target": target,
            "role": str(source.get("speaker_role", "public_speaker")),
            "audience": str(source.get("audience", "unknown")),
            "occasion": str(source.get("source_context", ""))
            or str(source.get("title", ""))
            or "unknown",
            "known_information": str(
                source.get("available_information", "unknown")
            ),
            "constraints": list(source.get("constraints", [])),
            "stakes": list(source.get("stakes", [])),
            "temporal_context": {
                "response_time": response_time or "unknown",
                "publication_time": str(
                    source.get("publication_date", "unknown")
                ),
                "precision": _time_precision(response_time),
                "time_source": "source_metadata" if response_time else "unknown",
            },
            "context": str(source.get("source_context", "")),
            "public_response": answer,
            "completeness": completeness,
        },
        "tendency_atoms": tendency_atoms,
        "knowledge_items": knowledge_items,
    }


def _narrative_responses(source: Mapping[str, object]) -> list[tuple[str, str, str]]:
    """Return attributable public statements without requiring a Q/A layout."""
    if source.get("speaker_scope") == "mixed_speakers":
        return []
    if source.get("qas"):
        return []
    raw_segments = source.get("segments", [])
    segments = [
        str(value.get("text", "")).strip()
        for value in raw_segments
        if isinstance(value, Mapping)
    ]
    if not segments:
        segments = [str(source.get("text", "")).strip()]
    title = str(source.get("title", "")).strip()
    context = str(source.get("source_context", "")).strip()
    trigger = context or title or "public statement"
    results: list[tuple[str, str, str]] = []
    for index, segment in enumerate(segments[:40], start=1):
        compact = re.sub(r"\s+", " ", segment).strip()
        if len(compact) < 24 or len(compact) > 4000:
            continue
        results.append((trigger, compact, f"segment:{index}"))
    return results


def response_events_from_source(source: Mapping[str, object]) -> list[dict[str, object]]:
    """Create attributable response candidates without treating inference as fact."""
    events: list[dict[str, object]] = []
    person_id = str(source["person_id"])
    source_id = str(source["source_id"])
    source_locator = str(source.get("source_locator", "")).strip()
    source_context = str(source.get("source_context", "")).strip()
    authenticity = str(source.get("content_authenticity", "unverified_material"))
    qas = [] if source.get("speaker_scope") == "mixed_speakers" else source.get("qas", [])
    for index, qa in enumerate(qas):
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            continue
        semantics = _event_semantics(question=question, answer=answer, source=source)
        event = {
            "schema_version": EVENT_SCHEMA,
            "event_id": f"response-{canonical_hash([source_id, index, question, answer])[:16]}",
            "person_id": person_id,
            "observed_at": str(source.get("source_date", "")),
            "occasion": str(source.get("title", "")),
            "interlocutor": "unknown",
            "context_visibility": "unknown",
            "full_context": source_context,
            "trigger": question,
            "question": question,
            "actual_response": answer,
            "speaker": str(source.get("speaker", "")),
            "speaker_role": "claimed_subject",
            "attribution": "direct_person_expression_candidate",
            "source_id": source_id,
            "source_url": str(source.get("source_url", "")),
            "source_locator": "; ".join(
                value
                for value in (
                    source_locator,
                    str(qa.get("locator", f"qa:{index + 1}")),
                )
                if value
            ),
            "available_information": "not_recorded",
            "later_correction": "not_recorded",
            "related_action": "not_recorded",
            "data_role": DATA_ROLE_MAP.get(
                str(source.get("dataset_role", "")), "feature_discovery"
            ),
            "origin": "human_supplied_source",
            "content_authenticity": authenticity,
            "original_language": str(source.get("original_language", "")),
            "translation_of": str(source.get("translation_of", "")),
            "content_hash": canonical_hash([question, answer]),
            "near_duplicate_of": source.get("near_duplicate_of"),
            "label_status": "pending_source_review",
            "label_method": "deterministic_public_response_atoms_v2",
            "speech_act": semantics["speech_act"],
            "stance": semantics["stance"],
            "claims": semantics["claims"],
            "reasons": semantics["reasons"],
            "memories": [],
            "uncertainties": semantics["uncertainties"],
            "event_atom": semantics["event_atom"],
            "tendency_atoms": semantics["tendency_atoms"],
            "knowledge_items": semantics["knowledge_items"],
        }
        events.append(event)
    for index, (trigger, answer, locator) in enumerate(
        _narrative_responses(source), start=len(events)
    ):
        semantics = _event_semantics(question=trigger, answer=answer, source=source)
        events.append(
            {
                "schema_version": EVENT_SCHEMA,
                "event_id": f"response-{canonical_hash([source_id, index, trigger, answer])[:16]}",
                "person_id": person_id,
                "observed_at": str(source.get("source_date", "")),
                "occasion": str(source.get("title", "")),
                "interlocutor": "public_audience",
                "context_visibility": "source_context_only",
                "full_context": source_context,
                "trigger": trigger,
                "question": trigger,
                "actual_response": answer,
                "speaker": str(source.get("speaker", "")),
                "speaker_role": "claimed_subject",
                "attribution": "direct_person_expression_candidate",
                "source_id": source_id,
                "source_url": str(source.get("source_url", "")),
                "source_locator": "; ".join(
                    value for value in (source_locator, locator) if value
                ),
                "available_information": "not_recorded",
                "later_correction": "not_recorded",
                "related_action": "not_recorded",
                "data_role": DATA_ROLE_MAP.get(
                    str(source.get("dataset_role", "")), "feature_discovery"
                ),
                "origin": "human_supplied_unstructured_source",
                "content_authenticity": authenticity,
                "original_language": str(source.get("original_language", "")),
                "translation_of": str(source.get("translation_of", "")),
                "content_hash": canonical_hash([trigger, answer]),
                "near_duplicate_of": source.get("near_duplicate_of"),
                "label_status": "pending_source_review",
                "label_method": "deterministic_public_response_atoms_v2",
                "speech_act": semantics["speech_act"],
                "stance": semantics["stance"],
                "claims": semantics["claims"],
                "reasons": semantics["reasons"],
                "memories": [],
                "uncertainties": semantics["uncertainties"],
                "event_atom": semantics["event_atom"],
                "tendency_atoms": semantics["tendency_atoms"],
                "knowledge_items": semantics["knowledge_items"],
            }
        )
    if not events:
        preview = str(source.get("text_preview", "") or source.get("text", ""))[:500]
        if preview.strip():
            events.append(
                {
                    "schema_version": EVENT_SCHEMA,
                    "event_id": f"response-{canonical_hash([source_id, preview])[:16]}",
                    "person_id": person_id,
                    "observed_at": str(source.get("source_date", "")),
                    "occasion": str(source.get("title", "")),
                    "interlocutor": "unknown",
                    "context_visibility": "unknown",
                    "full_context": source_context,
                    "trigger": "",
                    "question": "",
                    "actual_response": "",
                    "speaker": str(source.get("speaker", "")),
                    "speaker_role": "unknown",
                    "attribution": "unstructured_material_candidate",
                    "source_id": source_id,
                    "source_url": str(source.get("source_url", "")),
                    "source_locator": source_locator or "document",
                    "available_information": "not_recorded",
                    "later_correction": "not_recorded",
                    "related_action": "not_recorded",
                    "data_role": "feature_discovery",
                    "origin": "human_supplied_source",
                    "content_authenticity": authenticity,
                    "original_language": str(source.get("original_language", "")),
                    "translation_of": str(source.get("translation_of", "")),
                    "content_hash": canonical_hash(preview),
                    "near_duplicate_of": source.get("near_duplicate_of"),
                    "label_status": "unverified_candidate",
                    "label_method": "no_response_label_extracted",
                    "speech_act": None,
                    "stance": None,
                    "claims": [],
                    "reasons": [],
                    "memories": [],
                    "uncertainties": [],
                }
            )
    return events


def review_response_events(
    source: Mapping[str, object], person_name: str, aliases: Sequence[str]
) -> list[dict[str, object]]:
    allowed = {person_name.casefold(), *(str(item).casefold() for item in aliases)}
    speaker_matches = str(source.get("speaker", "")).casefold() in allowed
    authenticity = str(source.get("content_authenticity", "unverified_material"))
    provenance_available = bool(
        str(source.get("source_url", "")).strip()
        or str(source.get("filename", "")).strip()
    ) and bool(str(source.get("source_locator", "")).strip())
    translation_traceable = (
        authenticity != "verified_translation"
        or bool(str(source.get("translation_of", "")).strip())
    )
    reviewed: list[dict[str, object]] = []
    for raw in source.get("response_events", []):
        event = copy.deepcopy(dict(raw))
        if (
            speaker_matches
            and event.get("actual_response")
            and event.get("attribution") == "direct_person_expression_candidate"
            and authenticity in TRAINABLE_AUTHENTICITY
            and provenance_available
            and translation_traceable
        ):
            event["speaker_role"] = "person_self"
            event["attribution"] = "direct_person_expression"
            event["label_status"] = "confirmed_response_weak_semantic_labels"
            completeness = dict(
                dict(event.get("event_atom") or {}).get("completeness") or {}
            )
            allowed_uses = set(map(str, completeness.get("allowed_uses", [])))
            event["training_rejection_reasons"] = []
            if "response_head_training" not in allowed_uses:
                event["data_role"] = "feature_discovery"
                if completeness.get("temporal_status") != "known":
                    event["training_rejection_reasons"].append(
                        "response_time_missing"
                    )
                if completeness.get("context_status") != "known":
                    event["training_rejection_reasons"].append(
                        "event_context_incomplete"
                    )
        else:
            event["label_status"] = "unverified_candidate"
            event["training_rejection_reasons"] = [
                reason
                for condition, reason in (
                    (speaker_matches, "speaker_identity_not_confirmed"),
                    (authenticity in TRAINABLE_AUTHENTICITY, "not_verbatim_or_verified_translation"),
                    (provenance_available, "source_or_locator_missing"),
                    (translation_traceable, "translation_missing_original_reference"),
                    (bool(event.get("actual_response")), "no_explicit_response"),
                )
                if not condition
            ]
        reviewed.append(event)
    return reviewed


def _feature_vector(
    text: str,
    history: Sequence[Mapping[str, object]],
    conversation_context: Mapping[str, object] | None = None,
) -> np.ndarray:
    vector = np.zeros(FEATURE_DIMENSION, dtype=np.float64)
    vector[0] = 1.0
    inputs: list[tuple[str, float]] = [(str(text), 1.0)]
    for distance, message in enumerate(reversed(history[-6:]), start=1):
        role_weight = 0.55 if message.get("role") == "user" else 0.32
        inputs.append((str(message.get("text", "")), role_weight / distance))
    if conversation_context:
        for name in ("current_topic", "relationship", "occasion", "time_stage"):
            value = str(conversation_context.get(name, "")).strip()
            if value:
                inputs.append((value, 0.25))
    for value, weight in inputs:
        for token in tokens(value):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = 6 + int.from_bytes(digest[:2], "big") % (FEATURE_DIMENSION - 6)
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[bucket] += sign * weight
    current_tokens = tokens(text)
    vector[1] = min(len(current_tokens), 50) / 50.0
    vector[2] = 1.0 if "?" in text or "？" in text else 0.0
    vector[3] = min(len(history), 10) / 10.0
    vector[4] = 1.0 if re.search(r"\b(it|that|this|they|he|she)\b|这|那|它|他|她", text.casefold()) else 0.0
    vector[5] = 1.0 if any(item.get("role") == "assistant" for item in history) else 0.0
    norm = float(np.linalg.norm(vector[1:]))
    if norm > 0:
        vector[1:] /= norm
    return vector


def _context_query(text: str, history: Sequence[Mapping[str, object]]) -> str:
    parts = [str(text)]
    if re.search(r"\b(it|that|this|they|he|she)\b|这|那|它|他|她", text.casefold()):
        for message in reversed(history):
            if message.get("role") == "user" and message.get("text"):
                parts.append(str(message["text"]))
                break
    return " ".join(parts)


def _head_fit(
    events: Sequence[Mapping[str, object]],
    labels: Sequence[str],
    label_field: str,
    population_events: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def observations(rows: Sequence[Mapping[str, object]], label: str) -> list[Observation]:
        return [
            Observation(
                person_id=str(item["person_id"]),
                scenario=Scenario(
                    scenario_id=f"{item['event_id']}:{label}",
                    features=tuple(float(value) for value in _feature_vector(str(item["question"]), [])),
                    feature_names=FEATURE_NAMES,
                    options=(f"not_{label}", label),
                    domain="person_response_head",
                    context={"label_field": label_field},
                ),
                actual_choice=int(item.get(label_field) == label),
                provenance="verified_direct_response_event",
            )
            for item in rows
        ]

    fitted: dict[str, object] = {}
    for label in labels:
        population_observations = observations(population_events, label)
        population_people = {item.person_id for item in population_observations}
        if len(population_people) >= 2 and len(population_observations) >= 8:
            population_model = PopulationPriorEstimator(FEATURE_NAMES).fit(
                population_observations
            )
        else:
            population_model = PopulationModel(
                weights=tuple(0.0 for _ in FEATURE_NAMES),
                covariance=tuple(
                    tuple(1.5 if row == column else 0.0 for column in range(FEATURE_DIMENSION))
                    for row in range(FEATURE_DIMENSION)
                ),
                feature_names=FEATURE_NAMES,
                model_version="fixed-zero-population-prior-v1",
            )
        representation = MapPersonEncoder().fit(
            str(events[0]["person_id"]), observations(events, label), population_model
        )
        adapter = IdentityAdapterGenerator().generate(representation, population_model)
        fitted[label] = {
            "population_weights": list(population_model.weights),
            "population_covariance": [list(row) for row in population_model.covariance],
            "population_model_version": population_model.model_version,
            "delta_weights": list(adapter.delta_weights),
            "parameter_covariance": [list(row) for row in representation.covariance],
            "representation_version": representation.representation_version,
            "adapter_version": adapter.adapter_version,
            "observation_count": representation.observation_count,
        }
    return fitted


def _component_manifest(population_people: int, population_events: int) -> list[dict[str, object]]:
    population_ready = population_people >= 2 and population_events >= 8
    return [
        {"component_id": "response_event_pipeline", "status": "active"},
        {
            "component_id": "population_prior",
            "status": "active" if population_ready else "experimental",
            "reason": "pooled_verified_people_available" if population_ready else "insufficient_multi_person_pool",
        },
        {"component_id": "pcfm_core_map_person_encoder", "status": "active"},
        {"component_id": "pcfm_core_identity_adapter", "status": "active"},
        {"component_id": "pcfm_core_decision_integrator", "status": "active"},
        {"component_id": "retrieval_candidate_recall", "status": "active"},
        {"component_id": "text_applicability_guard", "status": "active"},
        {"component_id": "expression_renderer", "status": "active"},
        {"component_id": "dynamic_state", "status": "experimental", "reason": "no_compatible_verified_temporal_response_outcomes"},
        {"component_id": "cognitive_features", "status": "registered"},
        {"component_id": "mechanism_correction", "status": "experimental"},
        {"component_id": "empirical_bayes", "status": "experimental"},
        {"component_id": "hypernetwork", "status": "rejected", "reason": "prior_candidate_failed_common_gate"},
        {"component_id": "person_issue", "status": "rejected", "reason": "prior_candidate_failed_common_gate"},
        {"component_id": "joint_core", "status": "rejected", "reason": "prior_candidate_failed_common_gate"},
    ]


class ResponsePredictionKernel:
    """The sole fit/predict/evaluate kernel for conversational response content."""

    kernel_id = KERNEL_ID

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
        trainable = [
            copy.deepcopy(dict(item))
            for item in events
            if item.get("label_status") == "confirmed_response_weak_semantic_labels"
            and item.get("data_role") == "parameter_training"
            and item.get("question")
            and item.get("actual_response")
        ]
        trainable.sort(key=lambda item: str(item["event_id"]))
        if not trainable:
            raise ResponsePredictionError("no_trainable_direct_response_events")
        population = [
            dict(item)
            for item in population_events
            if item.get("label_status") == "confirmed_response_weak_semantic_labels"
            and item.get("data_role") == "parameter_training"
        ]
        population.sort(key=lambda item: (str(item.get("person_id", "")), str(item["event_id"])))
        components = _component_manifest(population_people, len(population))
        artifact: dict[str, object] = {
            "schema_version": MODEL_SCHEMA,
            "kernel": KERNEL_ID,
            "person_id": person_id,
            "version": int(version),
            "created_at": _utc_now(),
            "event_ids": sorted(str(item["event_id"]) for item in trainable),
            "event_digest": canonical_hash(
                sorted((str(item["event_id"]), str(item["content_hash"])) for item in trainable)
            ),
            "events": trainable,
            "feature_schema": {
                "encoder": "fixed_hash_context_encoder_v1",
                "dimension": FEATURE_DIMENSION,
                "generative_model": False,
            },
            "speech_act_head": _head_fit(
                trainable, SPEECH_ACTS, "speech_act", population
            ),
            "stance_head": _head_fit(trainable, STANCES, "stance", population),
            "components": components,
            "active_components": [
                str(item["component_id"])
                for item in components
                if item["status"] == "active"
            ],
            "scope": dict(scope or {}),
            "validation_status": "not_assessed",
            "calibration_status": "not_assessed",
            "accuracy_claim": "exploratory_only",
        }
        artifact["artifact_hash"] = canonical_hash(artifact)
        return artifact

    @staticmethod
    def verify(artifact: Mapping[str, object]) -> None:
        if artifact.get("schema_version") != MODEL_SCHEMA or artifact.get("kernel") != KERNEL_ID:
            raise ResponsePredictionError("unsupported_response_model_schema")
        payload = dict(artifact)
        digest = str(payload.pop("artifact_hash", ""))
        if canonical_hash(payload) != digest:
            raise ResponsePredictionError("response_model_integrity_failed")

    @staticmethod
    def _head_predict(
        model: Mapping[str, object], labels: Sequence[str], vector: np.ndarray
    ) -> list[dict[str, object]]:
        values: list[tuple[str, float]] = []
        for label in labels:
            item = dict(model[label])
            population_model = PopulationModel(
                weights=tuple(float(value) for value in item["population_weights"]),
                covariance=tuple(tuple(float(value) for value in row) for row in item["population_covariance"]),
                feature_names=FEATURE_NAMES,
                model_version=str(item["population_model_version"]),
            )
            adapter = PersonalAdapter(
                person_id="response-person",
                delta_weights=tuple(float(value) for value in item["delta_weights"]),
                adapter_version=str(item["adapter_version"]),
                representation_version=str(item["representation_version"]),
            )
            scenario = Scenario(
                scenario_id=f"runtime:{label}",
                features=tuple(float(value) for value in vector),
                feature_names=FEATURE_NAMES,
                options=(f"not_{label}", label),
                domain="person_response_head",
            )
            probability = DecisionIntegrator().predict(
                scenario,
                population_model,
                adapter,
                parameter_covariance=tuple(
                    tuple(float(value) for value in row)
                    for row in item["parameter_covariance"]
                ),
            ).probability_option_1
            values.append((label, max(probability, 1e-9)))
        total = sum(value for _, value in values)
        return [
            {"label": label, "probability": value / total}
            for label, value in sorted(values, key=lambda item: (-item[1], item[0]))
        ]

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
            raise ResponsePredictionError("message_required")
        context_records = [
            dict(item)
            for item in history[-6:]
            if item.get("role") in {"user", "assistant"}
        ]
        vector = _feature_vector(clean, context_records, conversation_context)
        speech_distribution = self._head_predict(
            dict(artifact["speech_act_head"]), SPEECH_ACTS, vector
        )
        stance_distribution = self._head_predict(
            dict(artifact["stance_head"]), STANCES, vector
        )
        query = _context_query(clean, context_records)
        pronoun_followup = query != clean
        previous_user_text = ""
        if pronoun_followup:
            previous_user_text = next(
                (
                    str(item.get("text", ""))
                    for item in reversed(context_records)
                    if item.get("role") == "user" and item.get("text")
                ),
                "",
            )
        candidates: list[dict[str, object]] = []
        for event in artifact["events"]:
            current_similarity = lexical_similarity(clean, str(event["question"]))
            applicability_support = applicability_similarity(
                query if pronoun_followup else clean,
                str(event["question"]),
            )
            historical_similarity = (
                lexical_similarity(previous_user_text, str(event["question"]))
                if previous_user_text
                else 0.0
            )
            similarity = (
                0.82 * current_similarity + 0.18 * historical_similarity
                if pronoun_followup
                else current_similarity
            )
            speech_match = next(
                item["probability"]
                for item in speech_distribution
                if item["label"] == event["speech_act"]
            )
            stance_match = next(
                item["probability"]
                for item in stance_distribution
                if item["label"] == event["stance"]
            )
            candidates.append(
                {
                    "event": event,
                    "similarity": similarity,
                    "applicability_support": applicability_support,
                    "score": 0.72 * similarity + 0.16 * speech_match + 0.12 * stance_match,
                }
            )
        candidates.sort(
            key=lambda item: (
                -float(item["score"]),
                str(item["event"]["event_id"]),
            )
        )
        best_support = float(candidates[0]["similarity"]) if candidates else 0.0
        best_applicability_support = max(
            (float(item["applicability_support"]) for item in candidates),
            default=0.0,
        )
        refusal_reasons: list[str] = []
        if not candidates:
            refusal_reasons.append("insufficient_evidence")
        elif best_support < 0.12 or best_applicability_support <= 0.0:
            refusal_reasons.append("out_of_domain")
        if candidates:
            best_score = float(candidates[0]["score"])
            relative_support_floor = max(0.12, best_support * 0.55)
            selected_events = [
                item
                for item in candidates
                if float(item["similarity"]) >= relative_support_floor
                and float(item["score"]) >= best_score * 0.75
            ][:3]
        else:
            selected_events = []
        claims: list[dict[str, object]] = []
        reasons: list[dict[str, object]] = []
        uncertainties: list[dict[str, object]] = []
        used_text: set[str] = set()
        for candidate in selected_events:
            event = candidate["event"]
            for field, target, prefix, limit in (
                ("claims", claims, "C", 3),
                ("reasons", reasons, "R", 2),
                ("uncertainties", uncertainties, "U", 2),
            ):
                for value in event.get(field, []):
                    text_value = str(value).strip()
                    if not text_value or text_value in used_text or len(target) >= limit:
                        continue
                    used_text.add(text_value)
                    target.append(
                        {
                            "id": f"{prefix}{len(target) + 1}",
                            "text": text_value,
                            "probability": round(float(candidate["score"]), 6),
                            "evidence_ref": str(event["event_id"]),
                        }
                    )
        if not claims and not refusal_reasons:
            refusal_reasons.append("local_support_gap")
        refused = bool(refusal_reasons)
        top_speech = speech_distribution[0]
        top_stance = stance_distribution[0]
        confidence = 0.0 if refused else min(
            0.85,
            best_support
            * math.sqrt(float(top_speech["probability"]) * float(top_stance["probability"]))
            + 0.12,
        )
        content = {
            "person_id": artifact["person_id"],
            "domain": str(dict(artifact.get("scope", {})).get("focus_domain", "unknown")),
            "speech_act": copy.deepcopy(top_speech),
            "speech_act_distribution": speech_distribution,
            "stance": copy.deepcopy(top_stance),
            "stance_distribution": stance_distribution,
            "claims": claims if not refused else [],
            "reasons": reasons if not refused else [],
            "memories": [],
            "uncertainties": uncertainties,
            "confidence": round(confidence, 6),
            "confidence_kind": "uncalibrated_posterior_support",
            "applicability": (
                "prediction_refused" if refused else "exploratory"
            ),
            "refusal_reasons": refusal_reasons,
            "evidence_refs": sorted(
                {
                    str(item["evidence_ref"])
                    for item in [*claims, *reasons, *uncertainties]
                }
            ),
            "active_components": list(artifact["active_components"]),
            "components": copy.deepcopy(artifact["components"]),
            "model_version": f"{artifact['person_id']}-v{artifact['version']}",
            "model_validity": "exploratory_accuracy_not_assessed",
            "valid_scope": copy.deepcopy(artifact.get("scope", {})),
            "candidate_set_size": len(candidates),
        }
        locked_text = " ".join(
            item["text"]
            for item in [
                *content["claims"],
                *content["reasons"],
                *content["uncertainties"],
            ]
        )
        protected_dates = sorted(
            set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", locked_text))
        )
        protected_entities = sorted(
            set(
                re.findall(
                    r"\b[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)+\b",
                    locked_text,
                )
            )
        )
        protected_quotes = sorted(
            set(
                value
                for pair in re.findall(r'"([^"\n]+)"|“([^”\n]+)”', locked_text)
                for value in pair
                if value
            )
        )
        renderer_contract = {
            "schema_version": "pcfm-frozen-content-contract-v2",
            "speech_act": str(top_speech["label"]),
            "stance": str(top_stance["label"]),
            "refusal_status": "prediction_refused" if refused else "not_refused",
            "claims": [{"id": item["id"], "text": item["text"]} for item in content["claims"]],
            "reasons": [{"id": item["id"], "text": item["text"]} for item in content["reasons"]],
            "memories": [],
            "uncertainties": [
                {"id": item["id"], "text": item["text"]}
                for item in content["uncertainties"]
            ],
            "protected_entities": protected_entities,
            "protected_numbers": sorted(
                set(
                    re.findall(
                        r"\b\d+(?:[.,]\d+)*(?:-%|%)?|\b\d{4}-\d{2}-\d{2}\b",
                        locked_text,
                    )
                )
            ),
            "protected_dates": protected_dates,
            "protected_quotes": protected_quotes,
            "confidence": content["confidence"],
            "style_mode": "interview_public",
        }
        renderer_digest = canonical_hash(renderer_contract)
        content["renderer_contract_digest"] = renderer_digest
        content_digest = canonical_hash(content)
        content["content_digest"] = content_digest
        return {
            "schema_version": PREDICTION_SCHEMA,
            "status": "refused" if refused else "answered",
            "structured_prediction": content,
            "renderer_contract": renderer_contract,
            "content_digest": content_digest,
            "renderer_contract_digest": renderer_digest,
            "prediction_trace": {
                "kernel": KERNEL_ID,
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
                "conversation_context": copy.deepcopy(dict(conversation_context or {})),
                "candidate_event_ids": [
                    str(item["event"]["event_id"]) for item in selected_events
                ],
                "best_local_support": round(best_support, 6),
                "best_applicability_support": round(best_applicability_support, 6),
                "context_retrieval_weight": 0.18 if pronoun_followup else 0.0,
            },
        }

    def evaluate(
        self,
        artifact: Mapping[str, object],
        holdout_events: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        predictions = []
        for event in holdout_events:
            if not event.get("question") or not event.get("actual_response"):
                continue
            result = self.predict(artifact, text=str(event["question"]), history=[])
            structured = result["structured_prediction"]
            predictions.append(
                {
                    "event_id": event["event_id"],
                    "speech_act_correct": structured["speech_act"]["label"] == event.get("speech_act"),
                    "stance_correct": structured["stance"]["label"] == event.get("stance"),
                    "claim_support": lexical_similarity(
                        " ".join(item["text"] for item in structured["claims"]),
                        str(event["actual_response"]),
                    ),
                    "refused": result["status"] == "refused",
                }
            )
        if not predictions:
            return {"status": "not_assessed", "sample_count": 0}
        return {
            "status": "exploratory",
            "sample_count": len(predictions),
            "speech_act_accuracy": sum(item["speech_act_correct"] for item in predictions) / len(predictions),
            "stance_accuracy": sum(item["stance_correct"] for item in predictions) / len(predictions),
            "mean_claim_support": sum(float(item["claim_support"]) for item in predictions) / len(predictions),
            "coverage": sum(not item["refused"] for item in predictions) / len(predictions),
            "records": predictions,
        }

    def compare_baselines(
        self,
        artifact: Mapping[str, object],
        holdout_events: Sequence[Mapping[str, object]],
        *,
        wrong_person_artifacts: Sequence[Mapping[str, object]] = (),
    ) -> dict[str, object]:
        """Score every available baseline on the exact same holdout events."""
        self.verify(artifact)
        usable = [
            dict(item)
            for item in holdout_events
            if item.get("question") and item.get("actual_response")
        ]
        if not usable:
            return {
                "status": "not_assessed",
                "reason": "sealed_final_validation_required",
                "sample_count": 0,
            }
        correct = self.evaluate(artifact, usable)
        population = copy.deepcopy(dict(artifact))
        for head_name in ("speech_act_head", "stance_head"):
            for label, item in population[head_name].items():
                item["weights"] = list(item["population_weights"])
        population.pop("artifact_hash", None)
        population["artifact_hash"] = canonical_hash(population)
        population_report = self.evaluate(population, usable)

        training = [dict(item) for item in artifact["events"]]
        speech_frequency = max(
            SPEECH_ACTS,
            key=lambda label: (
                sum(item.get("speech_act") == label for item in training),
                -SPEECH_ACTS.index(label),
            ),
        )
        stance_frequency = max(
            STANCES,
            key=lambda label: (
                sum(item.get("stance") == label for item in training),
                -STANCES.index(label),
            ),
        )
        frequency_records = []
        retrieval_records = []
        for event in usable:
            nearest = max(
                training,
                key=lambda item: (
                    lexical_similarity(str(event["question"]), str(item["question"])),
                    str(item["event_id"]),
                ),
            )
            retrieval_records.append(
                {
                    "speech_act_correct": nearest.get("speech_act") == event.get("speech_act"),
                    "stance_correct": nearest.get("stance") == event.get("stance"),
                    "claim_support": lexical_similarity(
                        str(nearest.get("actual_response", "")),
                        str(event["actual_response"]),
                    ),
                }
            )
            frequency_records.append(
                {
                    "speech_act_correct": speech_frequency == event.get("speech_act"),
                    "stance_correct": stance_frequency == event.get("stance"),
                    "claim_support": 0.0,
                }
            )

        def aggregate(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
            count = len(records)
            return {
                "status": "exploratory",
                "sample_count": count,
                "speech_act_accuracy": sum(bool(item["speech_act_correct"]) for item in records) / count,
                "stance_accuracy": sum(bool(item["stance_correct"]) for item in records) / count,
                "mean_claim_support": sum(float(item["claim_support"]) for item in records) / count,
            }

        wrong_reports = []
        for wrong in wrong_person_artifacts:
            try:
                self.verify(wrong)
            except ResponsePredictionError:
                continue
            wrong_reports.append(
                {
                    "person_id": wrong["person_id"],
                    "report": self.evaluate(wrong, usable),
                }
            )
        return {
            "status": "exploratory_not_confirmatory",
            "sample_count": len(usable),
            "correct_person": correct,
            "population": population_report,
            "retrieval": aggregate(retrieval_records),
            "person_history_frequency": aggregate(frequency_records),
            "wrong_people": wrong_reports,
            "recent_dynamic_population": {
                "status": "not_assessed",
                "reason": "compatible_time_ordered_population_outcomes_required",
            },
            "release_gate": "not_passed",
            "notice": "Software execution is not evidence that the correct-person model outperforms baselines.",
        }
