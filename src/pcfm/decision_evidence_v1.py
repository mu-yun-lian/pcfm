from __future__ import annotations

"""Decision-Context-Rationale 证据合同 v1。

边界：签名采用 HMAC-SHA256（对称 MAC），仅适用于「单一受信第三方持钥」的原型；
不提供不可否认性、不能防持钥方篡改。验签密钥必须带外获得，不得随包提交；
`created_at` 由调用方给定、仅作自证，不能证明外部时间或材料完备性。
生产须换非对称签名 / 透明日志 / 外部时间戳服务。
"""

import hashlib
import json
from math import isfinite
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .ledger import VerificationAuthority


CONFIG_VERSION = "decision-evidence-config-v1"
BUNDLE_VERSION = "decision-context-rationale-evidence-v1"
MODULE_STATUS = "implemented_unintegrated"
MAX_RATIONALE_LAG_HOURS = 168.0
MAX_RATIONALE_LEAD_HOURS = 24.0
NEAR_DUPLICATE_THRESHOLD = 0.92

ALLOWED_PROVENANCE = frozenset(
    {
        "official_primary",
        "human_primary",
        "archival_primary_snapshot",
        "verified_external_consequence",
    }
)
ALLOWED_ROLES = frozenset(
    {
        "candidate_discovery",
        "parameter_fitting",
        "applicability_calibration",
        "candidate_selection",
        "sealed_confirmation",
        "post_deployment_monitoring",
        "external_utility_evaluation",
    }
)
DEVELOPMENT_ROLES = frozenset(
    {
        "candidate_discovery",
        "parameter_fitting",
        "applicability_calibration",
        "candidate_selection",
    }
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a SHA-256 hex digest"
        ) from error


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{label} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clean_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _text_shingles(value: str, width: int = 3) -> frozenset[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return frozenset()
    if len(cleaned) <= width:
        return frozenset({cleaned})
    return frozenset(
        cleaned[index : index + width]
        for index in range(len(cleaned) - width + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class DecisionEvidenceRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(sorted(set(str(reason) for reason in reasons)))
        super().__init__(
            "decision_evidence_refused:" + ",".join(self.reasons)
        )


@dataclass(frozen=True)
class DecisionEvidenceConfig:
    max_rationale_lag_hours: float = MAX_RATIONALE_LAG_HOURS
    artifact_version: str = CONFIG_VERSION

    def __post_init__(self) -> None:
        value = float(self.max_rationale_lag_hours)
        object.__setattr__(self, "max_rationale_lag_hours", value)
        if not isfinite(value):
            raise ValueError("max_rationale_lag_hours must be finite")
        if value <= 0:
            raise ValueError("max_rationale_lag_hours must be positive")
        if value > MAX_RATIONALE_LAG_HOURS:
            raise ValueError(
                "max_rationale_lag_hours exceeds the immutable hard ceiling"
            )
        if self.artifact_version != CONFIG_VERSION:
            raise ValueError("unsupported decision evidence config version")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "max_rationale_lag_hours": self.max_rationale_lag_hours,
        }


def decision_evidence_config_from_dict(
    data: Mapping[str, object],
) -> DecisionEvidenceConfig:
    expected = {"artifact_version", "max_rationale_lag_hours"}
    if set(data) != expected:
        raise ValueError("decision evidence config fields do not match v1")
    return DecisionEvidenceConfig(
        artifact_version=str(data["artifact_version"]),
        max_rationale_lag_hours=float(data["max_rationale_lag_hours"]),
    )


@dataclass(frozen=True)
class DecisionOption:
    option_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.option_id or not self.text.strip():
            raise ValueError("decision option id and text are required")

    def to_dict(self) -> dict[str, str]:
        return {"option_id": self.option_id, "text": self.text}


def decision_option_from_dict(
    data: Mapping[str, object],
) -> DecisionOption:
    if set(data) != {"option_id", "text"}:
        raise ValueError("decision option fields do not match v1")
    return DecisionOption(
        option_id=str(data["option_id"]),
        text=str(data["text"]),
    )


@dataclass(frozen=True)
class EvidenceCitation:
    source_id: str
    quote: str

    def __post_init__(self) -> None:
        if not self.source_id or not self.quote.strip():
            raise ValueError("evidence citation source and quote are required")

    def to_dict(self) -> dict[str, str]:
        return {"source_id": self.source_id, "quote": self.quote}


def evidence_citation_from_dict(
    data: Mapping[str, object],
) -> EvidenceCitation:
    if set(data) != {"source_id", "quote"}:
        raise ValueError("evidence citation fields do not match v1")
    return EvidenceCitation(
        source_id=str(data["source_id"]),
        quote=str(data["quote"]),
    )


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    source_locator: str
    provenance: str
    published_at: str
    captured_at: str
    linked_event_id: str
    content: str
    content_digest: str
    author_person_id: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("source_id", self.source_id),
            ("source_locator", self.source_locator),
            ("linked_event_id", self.linked_event_id),
            ("content", self.content),
        ):
            if not str(value).strip():
                raise ValueError(f"{label} is required")
        published = _parse_timestamp(self.published_at, "published_at")
        captured = _parse_timestamp(self.captured_at, "captured_at")
        if captured < published:
            raise ValueError("captured_at cannot precede published_at")
        _require_digest(self.content_digest, "content_digest")
        if self.content_digest != _content_digest(self.content):
            raise ValueError("content_digest does not match source content")
        if self.author_person_id is not None and not self.author_person_id:
            raise ValueError("author_person_id cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_locator": self.source_locator,
            "provenance": self.provenance,
            "published_at": self.published_at,
            "captured_at": self.captured_at,
            "linked_event_id": self.linked_event_id,
            "content": self.content,
            "content_digest": self.content_digest,
            "author_person_id": self.author_person_id,
        }


def source_snapshot_from_dict(
    data: Mapping[str, object],
) -> SourceSnapshot:
    expected = {
        "source_id",
        "source_locator",
        "provenance",
        "published_at",
        "captured_at",
        "linked_event_id",
        "content",
        "content_digest",
        "author_person_id",
    }
    if set(data) != expected:
        raise ValueError("source snapshot fields do not match v1")
    author = data["author_person_id"]
    return SourceSnapshot(
        source_id=str(data["source_id"]),
        source_locator=str(data["source_locator"]),
        provenance=str(data["provenance"]),
        published_at=str(data["published_at"]),
        captured_at=str(data["captured_at"]),
        linked_event_id=str(data["linked_event_id"]),
        content=str(data["content"]),
        content_digest=str(data["content_digest"]),
        author_person_id=None if author is None else str(author),
    )


def _canonical_citations(
    values: Sequence[EvidenceCitation],
) -> tuple[EvidenceCitation, ...]:
    converted = tuple(values)
    keys = [(item.source_id, item.quote) for item in converted]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate evidence citation")
    return tuple(
        sorted(converted, key=lambda item: (item.source_id, item.quote))
    )


@dataclass(frozen=True)
class DecisionEvidenceRecord:
    event_id: str
    person_id: str
    domain: str
    task: str
    decision_at: str
    question_text: str
    options: tuple[DecisionOption, ...]
    chosen_option_id: str
    evidence_role: str
    role_assigned_at: str
    role_assignment_reference: str
    context_citations: tuple[EvidenceCitation, ...]
    choice_citations: tuple[EvidenceCitation, ...]
    rationale_citations: tuple[EvidenceCitation, ...] = ()
    consequence_citations: tuple[EvidenceCitation, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("event_id", self.event_id),
            ("person_id", self.person_id),
            ("domain", self.domain),
            ("task", self.task),
            ("question_text", self.question_text),
            ("chosen_option_id", self.chosen_option_id),
        ):
            if not str(value).strip():
                raise ValueError(f"{label} is required")
        _parse_timestamp(self.decision_at, "decision_at")
        _parse_timestamp(self.role_assigned_at, "role_assigned_at")
        _require_digest(
            self.role_assignment_reference,
            "role_assignment_reference",
        )
        options = tuple(self.options)
        object.__setattr__(self, "options", options)
        if len(options) < 2:
            raise ValueError("at least two decision options are required")
        option_ids = [item.option_id for item in options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("decision option ids must be unique")
        normalized_option_texts = [_clean_text(item.text) for item in options]
        if len(set(normalized_option_texts)) != len(
            normalized_option_texts
        ):
            raise ValueError("decision option texts must be distinct")
        if self.chosen_option_id not in option_ids:
            raise ValueError("chosen_option_id is not a declared option")
        if self.evidence_role not in ALLOWED_ROLES:
            raise ValueError("unsupported decision evidence role")
        for name in (
            "context_citations",
            "choice_citations",
            "rationale_citations",
            "consequence_citations",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_citations(getattr(self, name)),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "person_id": self.person_id,
            "domain": self.domain,
            "task": self.task,
            "decision_at": self.decision_at,
            "question_text": self.question_text,
            "options": [item.to_dict() for item in self.options],
            "chosen_option_id": self.chosen_option_id,
            "evidence_role": self.evidence_role,
            "role_assigned_at": self.role_assigned_at,
            "role_assignment_reference": self.role_assignment_reference,
            "context_citations": [
                item.to_dict() for item in self.context_citations
            ],
            "choice_citations": [
                item.to_dict() for item in self.choice_citations
            ],
            "rationale_citations": [
                item.to_dict() for item in self.rationale_citations
            ],
            "consequence_citations": [
                item.to_dict() for item in self.consequence_citations
            ],
        }


def decision_evidence_record_from_dict(
    data: Mapping[str, object],
) -> DecisionEvidenceRecord:
    expected = {
        "event_id",
        "person_id",
        "domain",
        "task",
        "decision_at",
        "question_text",
        "options",
        "chosen_option_id",
        "evidence_role",
        "role_assigned_at",
        "role_assignment_reference",
        "context_citations",
        "choice_citations",
        "rationale_citations",
        "consequence_citations",
    }
    if set(data) != expected:
        raise ValueError("decision evidence record fields do not match v1")

    def citations(name: str) -> tuple[EvidenceCitation, ...]:
        return tuple(
            evidence_citation_from_dict(dict(item))
            for item in data[name]
        )

    return DecisionEvidenceRecord(
        event_id=str(data["event_id"]),
        person_id=str(data["person_id"]),
        domain=str(data["domain"]),
        task=str(data["task"]),
        decision_at=str(data["decision_at"]),
        question_text=str(data["question_text"]),
        options=tuple(
            decision_option_from_dict(dict(item))
            for item in data["options"]
        ),
        chosen_option_id=str(data["chosen_option_id"]),
        evidence_role=str(data["evidence_role"]),
        role_assigned_at=str(data["role_assigned_at"]),
        role_assignment_reference=str(
            data["role_assignment_reference"]
        ),
        context_citations=citations("context_citations"),
        choice_citations=citations("choice_citations"),
        rationale_citations=citations("rationale_citations"),
        consequence_citations=citations("consequence_citations"),
    )


def _record_sort_key(
    record: DecisionEvidenceRecord,
) -> tuple[datetime, str, str]:
    return (
        _parse_timestamp(record.decision_at, "decision_at"),
        record.person_id,
        record.event_id,
    )


def _role_counts(
    records: Sequence[DecisionEvidenceRecord],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(record.evidence_role for record in records)
    return tuple(sorted((role, int(count)) for role, count in counts.items()))


def _decision_design_text(record: DecisionEvidenceRecord) -> str:
    return "|".join(
        (
            record.person_id,
            record.domain,
            record.task,
            record.question_text,
            *(option.text for option in record.options),
        )
    )


def _decision_design_digest(record: DecisionEvidenceRecord) -> str:
    return hashlib.sha256(
        _clean_text(_decision_design_text(record)).encode("utf-8")
    ).hexdigest()


def _citation_payload_without_identifiers(
    citations: Sequence[EvidenceCitation],
    sources: Mapping[str, SourceSnapshot],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "quote": citation.quote,
                "source_content_digest": sources[
                    citation.source_id
                ].content_digest,
                "source_provenance": sources[
                    citation.source_id
                ].provenance,
                "source_published_at": sources[
                    citation.source_id
                ].published_at,
            }
            for citation in citations
            if citation.source_id in sources
        ),
        key=lambda item: _canonical_json(item),
    )


def _content_fingerprint(
    records: Sequence[DecisionEvidenceRecord],
    sources: Mapping[str, SourceSnapshot],
) -> str:
    payload = []
    for record in records:
        payload.append(
            {
                "person_id": record.person_id,
                "domain": record.domain,
                "task": record.task,
                "decision_at": record.decision_at,
                "question_text": record.question_text,
                "options": [
                    {"option_id": item.option_id, "text": item.text}
                    for item in record.options
                ],
                "chosen_option_id": record.chosen_option_id,
                "evidence_role": record.evidence_role,
                "role_assigned_at": record.role_assigned_at,
                "role_assignment_reference": (
                    record.role_assignment_reference
                ),
                "context": _citation_payload_without_identifiers(
                    record.context_citations,
                    sources,
                ),
                "choice": _citation_payload_without_identifiers(
                    record.choice_citations,
                    sources,
                ),
                "rationale": _citation_payload_without_identifiers(
                    record.rationale_citations,
                    sources,
                ),
                "consequence": _citation_payload_without_identifiers(
                    record.consequence_citations,
                    sources,
                ),
            }
        )
    return _digest_json(sorted(payload, key=_canonical_json))


@dataclass(frozen=True)
class DecisionEvidenceSummary:
    status: str
    artifact_digest: str
    record_count: int
    source_count: int
    role_counts: tuple[tuple[str, int], ...]
    rationale_record_count: int
    consequence_record_count: int
    content_fingerprint: str
    training_authorized: bool = False
    semantic_claims_authorized: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "artifact_digest": self.artifact_digest,
            "record_count": self.record_count,
            "source_count": self.source_count,
            "role_counts": dict(self.role_counts),
            "rationale_record_count": self.rationale_record_count,
            "consequence_record_count": self.consequence_record_count,
            "content_fingerprint": self.content_fingerprint,
            "training_authorized": self.training_authorized,
            "semantic_claims_authorized": list(
                self.semantic_claims_authorized
            ),
        }


@dataclass(frozen=True)
class DecisionEvidenceBundle:
    config: DecisionEvidenceConfig
    created_at: str
    verifier_id: str
    sources: tuple[SourceSnapshot, ...]
    records: tuple[DecisionEvidenceRecord, ...]
    role_counts: tuple[tuple[str, int], ...]
    artifact_digest: str
    signature: str
    schema_version: str = BUNDLE_VERSION
    module_status: str = MODULE_STATUS
    training_authorized: bool = False
    semantic_claims_authorized: tuple[str, ...] = ()
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _parse_timestamp(self.created_at, "created_at")
        if not self.verifier_id:
            raise ValueError("verifier_id is required")
        if self.schema_version != BUNDLE_VERSION:
            raise ValueError("unsupported decision evidence bundle version")
        if self.module_status != MODULE_STATUS:
            raise ValueError("unsupported decision evidence module status")
        if self.training_authorized:
            raise ValueError("decision evidence cannot authorize training")
        if self.semantic_claims_authorized:
            raise ValueError(
                "decision evidence cannot authorize semantic claims"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError("unsupported decision evidence signature method")
        sources = tuple(sorted(self.sources, key=lambda item: item.source_id))
        records = tuple(sorted(self.records, key=_record_sort_key))
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "records", records)
        counts = tuple(
            sorted((str(role), int(count)) for role, count in self.role_counts)
        )
        object.__setattr__(self, "role_counts", counts)
        _require_digest(self.artifact_digest, "artifact_digest")
        _require_digest(self.signature, "signature")
        if self.artifact_digest != _digest_json(self._content_dict()):
            raise ValueError("artifact_digest does not match bundle content")

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "module_status": self.module_status,
            "training_authorized": self.training_authorized,
            "semantic_claims_authorized": list(
                self.semantic_claims_authorized
            ),
            "config": self.config.to_dict(),
            "created_at": self.created_at,
            "verifier_id": self.verifier_id,
            "signature_method": self.signature_method,
            "sources": [item.to_dict() for item in self.sources],
            "records": [item.to_dict() for item in self.records],
            "role_counts": dict(self.role_counts),
        }

    def signed_payload(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "artifact_digest": self.artifact_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.signed_payload(), "signature": self.signature}


def _validate_bundle_content(
    *,
    sources: Sequence[SourceSnapshot],
    records: Sequence[DecisionEvidenceRecord],
    config: DecisionEvidenceConfig,
    created_at: str,
) -> tuple[list[str], str]:
    reasons: list[str] = []
    if not sources:
        reasons.append("source_evidence_required")
    if not records:
        reasons.append("decision_records_required")
    source_ids = [source.source_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        reasons.append("duplicate_source_id")
    event_ids = [record.event_id for record in records]
    if len(set(event_ids)) != len(event_ids):
        reasons.append("duplicate_event_id")
    source_map = {source.source_id: source for source in sources}
    created = _parse_timestamp(created_at, "created_at")
    referenced_sources: set[str] = set()

    for source in sources:
        if source.provenance not in ALLOWED_PROVENANCE:
            reasons.append("source_provenance_not_allowed")
        if _parse_timestamp(source.captured_at, "captured_at") > created:
            reasons.append("source_captured_after_bundle_creation")

    for record in records:
        decision = _parse_timestamp(record.decision_at, "decision_at")
        role_assigned = _parse_timestamp(
            record.role_assigned_at,
            "role_assigned_at",
        )
        if decision > created:
            reasons.append("decision_after_bundle_creation")
        if role_assigned > created:
            reasons.append("role_assigned_after_bundle_creation")
        if (
            record.evidence_role == "sealed_confirmation"
            and role_assigned > decision
        ):
            reasons.append("sealed_role_assigned_after_decision")
        if not record.context_citations:
            reasons.append("context_evidence_required")
        if not record.choice_citations:
            reasons.append("choice_evidence_required")
        if not record.rationale_citations and not record.consequence_citations:
            reasons.append("explanation_evidence_required")

        groups = (
            ("context", record.context_citations),
            ("choice", record.choice_citations),
            ("rationale", record.rationale_citations),
            ("consequence", record.consequence_citations),
        )
        for kind, citations in groups:
            for citation in citations:
                referenced_sources.add(citation.source_id)
                source = source_map.get(citation.source_id)
                if source is None:
                    reasons.append("citation_source_missing")
                    continue
                if source.linked_event_id != record.event_id:
                    reasons.append("source_event_link_mismatch")
                if citation.quote not in source.content:
                    reasons.append("citation_quote_not_in_source")
                published = _parse_timestamp(
                    source.published_at,
                    "published_at",
                )
                if kind == "context" and published > decision:
                    reasons.append("context_not_available_prechoice")
                elif kind == "choice" and published < decision:
                    reasons.append("choice_evidence_precedes_decision")
                elif kind == "rationale":
                    lower = decision - timedelta(
                        hours=MAX_RATIONALE_LEAD_HOURS
                    )
                    upper = decision + timedelta(
                        hours=config.max_rationale_lag_hours
                    )
                    if not lower <= published <= upper:
                        reasons.append("rationale_outside_time_window")
                    if source.author_person_id != record.person_id:
                        reasons.append("rationale_wrong_person")
                    if source.provenance not in {
                        "official_primary",
                        "human_primary",
                        "archival_primary_snapshot",
                    }:
                        reasons.append("rationale_provenance_invalid")
                elif kind == "consequence":
                    if published < decision:
                        reasons.append("consequence_precedes_decision")
                    if source.provenance != "verified_external_consequence":
                        reasons.append("consequence_provenance_invalid")

        context_contents = "\n".join(
            source_map[citation.source_id].content
            for citation in record.context_citations
            if citation.source_id in source_map
        )
        if record.question_text not in context_contents:
            reasons.append("question_text_not_in_context_source")
        for option in record.options:
            if option.text not in context_contents:
                reasons.append("option_text_not_in_context_source")
        choice_contents = "\n".join(
            source_map[citation.source_id].content
            for citation in record.choice_citations
            if citation.source_id in source_map
        )
        chosen_text = next(
            option.text
            for option in record.options
            if option.option_id == record.chosen_option_id
        )
        if chosen_text not in choice_contents:
            reasons.append("chosen_option_not_in_choice_source")

    if set(source_ids) - referenced_sources:
        reasons.append("unreferenced_source")

    roles = {record.evidence_role for record in records}
    if len(roles) < 2:
        reasons.append("at_least_two_evidence_roles_required")
    if not roles & DEVELOPMENT_ROLES:
        reasons.append("development_role_required")
    if "sealed_confirmation" not in roles:
        reasons.append("sealed_confirmation_role_required")

    development_times = [
        _parse_timestamp(record.decision_at, "decision_at")
        for record in records
        if record.evidence_role in DEVELOPMENT_ROLES
    ]
    confirmation_times = [
        _parse_timestamp(record.decision_at, "decision_at")
        for record in records
        if record.evidence_role == "sealed_confirmation"
    ]
    if (
        development_times
        and confirmation_times
        and min(confirmation_times) <= max(development_times)
    ):
        reasons.append("sealed_confirmation_not_later_than_development")

    scoped_records: dict[
        tuple[str, str, str],
        list[DecisionEvidenceRecord],
    ] = {}
    for record in records:
        scope = (record.person_id, record.domain, record.task)
        scoped_records.setdefault(scope, []).append(record)
    for scoped in scoped_records.values():
        scoped_development = [
            _parse_timestamp(record.decision_at, "decision_at")
            for record in scoped
            if record.evidence_role in DEVELOPMENT_ROLES
        ]
        scoped_confirmation = [
            _parse_timestamp(record.decision_at, "decision_at")
            for record in scoped
            if record.evidence_role == "sealed_confirmation"
        ]
        if not scoped_development or not scoped_confirmation:
            reasons.append("scope_role_coverage_incomplete")
        elif min(scoped_confirmation) <= max(scoped_development):
            reasons.append(
                "scope_confirmation_not_later_than_development"
            )

    for left_index, left in enumerate(records):
        left_digest = _decision_design_digest(left)
        left_shingles = _text_shingles(_decision_design_text(left))
        for right in records[left_index + 1 :]:
            if (
                left.person_id,
                left.domain,
                left.task,
            ) != (
                right.person_id,
                right.domain,
                right.task,
            ):
                continue
            right_digest = _decision_design_digest(right)
            if left_digest == right_digest:
                if left.evidence_role != right.evidence_role:
                    reasons.append("decision_design_replay")
                else:
                    reasons.append("near_duplicate_decision_content")
                continue
            similarity = _jaccard(
                left_shingles,
                _text_shingles(_decision_design_text(right)),
            )
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                reasons.append("near_duplicate_decision_content")

    return reasons, _content_fingerprint(records, source_map)


def _bundle_content_dict(
    *,
    config: DecisionEvidenceConfig,
    created_at: str,
    verifier_id: str,
    sources: Sequence[SourceSnapshot],
    records: Sequence[DecisionEvidenceRecord],
    role_counts: Sequence[tuple[str, int]],
) -> dict[str, object]:
    return {
        "schema_version": BUNDLE_VERSION,
        "module_status": MODULE_STATUS,
        "training_authorized": False,
        "semantic_claims_authorized": [],
        "config": config.to_dict(),
        "created_at": created_at,
        "verifier_id": verifier_id,
        "signature_method": "hmac-sha256",
        "sources": [item.to_dict() for item in sources],
        "records": [item.to_dict() for item in records],
        "role_counts": dict(role_counts),
    }


def create_decision_evidence_bundle(
    *,
    sources: Sequence[SourceSnapshot],
    records: Sequence[DecisionEvidenceRecord],
    created_at: str,
    authority: VerificationAuthority,
    verifier_id: str,
    config: DecisionEvidenceConfig | None = None,
) -> DecisionEvidenceBundle:
    selected_config = config or DecisionEvidenceConfig()
    canonical_sources = tuple(sorted(sources, key=lambda item: item.source_id))
    canonical_records = tuple(sorted(records, key=_record_sort_key))
    reasons, _ = _validate_bundle_content(
        sources=canonical_sources,
        records=canonical_records,
        config=selected_config,
        created_at=created_at,
    )
    if reasons:
        raise DecisionEvidenceRefusedError(reasons)
    counts = _role_counts(canonical_records)
    content = _bundle_content_dict(
        config=selected_config,
        created_at=created_at,
        verifier_id=verifier_id,
        sources=canonical_sources,
        records=canonical_records,
        role_counts=counts,
    )
    artifact_digest = _digest_json(content)
    payload = {**content, "artifact_digest": artifact_digest}
    signature = authority.sign_payload(payload, verifier_id)
    bundle = DecisionEvidenceBundle(
        config=selected_config,
        created_at=created_at,
        verifier_id=verifier_id,
        sources=canonical_sources,
        records=canonical_records,
        role_counts=counts,
        artifact_digest=artifact_digest,
        signature=signature,
    )
    validate_decision_evidence_bundle(bundle, authority)
    return bundle


def validate_decision_evidence_bundle(
    bundle: DecisionEvidenceBundle,
    authority: VerificationAuthority,
) -> DecisionEvidenceSummary:
    reasons: list[str] = []
    try:
        authority.verify_payload(
            bundle.signed_payload(),
            bundle.verifier_id,
            bundle.signature,
        )
    except ValueError:
        reasons.append("bundle_signature_invalid")
    expected_digest = _digest_json(bundle._content_dict())
    if expected_digest != bundle.artifact_digest:
        reasons.append("artifact_digest_mismatch")
    # created_at 自证边界：不得在未来（允许少量时钟漂移）。
    try:
        created_at_time = _parse_timestamp(bundle.created_at, "created_at")
    except ValueError as error:
        reasons.append(str(error))
    else:
        if created_at_time > datetime.now(timezone.utc) + timedelta(minutes=5):
            reasons.append("created_at_in_future")
    content_reasons, content_fingerprint = _validate_bundle_content(
        sources=bundle.sources,
        records=bundle.records,
        config=bundle.config,
        created_at=bundle.created_at,
    )
    reasons.extend(content_reasons)
    expected_counts = _role_counts(bundle.records)
    if bundle.role_counts != expected_counts:
        reasons.append("role_counts_recomputation_mismatch")
    if reasons:
        raise DecisionEvidenceRefusedError(reasons)
    return DecisionEvidenceSummary(
        status="admission_contract_satisfied",
        artifact_digest=bundle.artifact_digest,
        record_count=len(bundle.records),
        source_count=len(bundle.sources),
        role_counts=expected_counts,
        rationale_record_count=sum(
            bool(record.rationale_citations) for record in bundle.records
        ),
        consequence_record_count=sum(
            bool(record.consequence_citations) for record in bundle.records
        ),
        content_fingerprint=content_fingerprint,
    )


def decision_evidence_bundle_from_dict(
    data: Mapping[str, object],
    authority: VerificationAuthority,
) -> DecisionEvidenceBundle:
    expected = {
        "schema_version",
        "module_status",
        "training_authorized",
        "semantic_claims_authorized",
        "config",
        "created_at",
        "verifier_id",
        "signature_method",
        "sources",
        "records",
        "role_counts",
        "artifact_digest",
        "signature",
    }
    if set(data) != expected:
        raise ValueError("decision evidence bundle fields do not match v1")
    raw_counts = data["role_counts"]
    if not isinstance(raw_counts, Mapping):
        raise ValueError("role_counts must be an object")
    bundle = DecisionEvidenceBundle(
        schema_version=str(data["schema_version"]),
        module_status=str(data["module_status"]),
        training_authorized=bool(data["training_authorized"]),
        semantic_claims_authorized=tuple(
            str(value) for value in data["semantic_claims_authorized"]
        ),
        config=decision_evidence_config_from_dict(
            dict(data["config"])
        ),
        created_at=str(data["created_at"]),
        verifier_id=str(data["verifier_id"]),
        signature_method=str(data["signature_method"]),
        sources=tuple(
            source_snapshot_from_dict(dict(item))
            for item in data["sources"]
        ),
        records=tuple(
            decision_evidence_record_from_dict(dict(item))
            for item in data["records"]
        ),
        role_counts=tuple(
            (str(role), int(count))
            for role, count in raw_counts.items()
        ),
        artifact_digest=str(data["artifact_digest"]),
        signature=str(data["signature"]),
    )
    validate_decision_evidence_bundle(bundle, authority)
    return bundle


def save_decision_evidence_bundle(
    path: Path,
    bundle: DecisionEvidenceBundle,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            bundle.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_decision_evidence_bundle(
    path: Path,
    authority: VerificationAuthority,
) -> DecisionEvidenceBundle:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("decision evidence bundle root must be an object")
    return decision_evidence_bundle_from_dict(raw, authority)
