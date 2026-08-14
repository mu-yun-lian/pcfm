from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .ledger import VerificationAuthority
from .tyler_source_v1 import (
    DECISION_AXES,
    TylerSourceArtifact,
)


PACKET_SCHEMA = "tyler-annotation-packet-v1"
SUBMISSION_SCHEMA = "tyler-annotation-submission-v1"
ADJUDICATION_SCHEMA = "tyler-adjudication-packet-v1"
DATASET_SCHEMA = "tyler-adjudicated-dataset-v1"
CODEBOOK_VERSION = "tyler-annotation-codebook-v1"
MINIMUM_CANDIDATES = 5
MINIMUM_EXACT_AGREEMENT = 2.0 / 3.0
MINIMUM_DISPOSITION_KAPPA = 0.40
DIAGNOSTIC_END = datetime.fromisoformat(
    "2026-07-31T23:59:59+00:00"
)
TRAINING_END = datetime.fromisoformat(
    "2024-12-31T23:59:59+00:00"
)
PROTOCOL_VALIDATION_END = datetime.fromisoformat(
    "2025-12-31T23:59:59+00:00"
)
DISPOSITIONS = frozenset(
    {
        "clear_in_scope_stance",
        "clear_out_of_scope_stance",
        "not_a_stance",
        "uncertain_attribution",
        "insufficient_context",
    }
)
DIRECTIONS = frozenset(
    {
        "toward_pole_1",
        "toward_pole_2",
        "mixed_or_conditional",
    }
)
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high"})
AXIS_POLES = {
    "ai_acceleration_vs_risk_regulation": {
        "pole_1": "AI acceleration",
        "pole_2": "risk regulation",
    },
    "market_mechanisms_vs_government_intervention": {
        "pole_1": "market mechanisms",
        "pole_2": "government intervention",
    },
    "technological_progress_vs_employment_displacement": {
        "pole_1": "technological progress",
        "pole_2": "employment-displacement protection",
    },
    "state_capacity_vs_individual_liberty": {
        "pole_1": "state capacity",
        "pole_2": "individual liberty",
    },
    "short_term_social_cost_vs_long_term_growth": {
        "pole_1": "accepting short-term social cost",
        "pole_2": "protecting against short-term cost",
    },
}
CODEBOOK = {
    "version": CODEBOOK_VERSION,
    "dispositions": sorted(DISPOSITIONS),
    "directions": sorted(DIRECTIONS),
    "confidence_levels": sorted(CONFIDENCE_LEVELS),
    "axes": {
        axis: AXIS_POLES[axis]
        for axis in DECISION_AXES
    },
    "rules": {
        "all_labels_require_evidence": True,
        "only_clear_in_scope_allows_axes": True,
        "confidence_is_annotator_confidence": True,
        "titles_and_urls_are_not_evidence_units": True,
    },
}


class AnnotationRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__(
            "Tyler annotation refused: "
            + ", ".join(self.reasons)
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
    return parsed


def _role_for_timestamp(value: str) -> str:
    published = _parse_timestamp(value, "published_at")
    if published <= TRAINING_END:
        return "training_discovery"
    if published <= PROTOCOL_VALIDATION_END:
        return "protocol_validation"
    if published <= DIAGNOSTIC_END:
        return "retrospective_diagnostic"
    raise AnnotationRefusedError(("future_registration_required",))


CODEBOOK_DIGEST = _digest_json(CODEBOOK)


@dataclass(frozen=True)
class AnnotationUnit:
    unit_id: str
    source_unit_sha256: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "source_unit_sha256": self.source_unit_sha256,
            "text": self.text,
        }


@dataclass(frozen=True)
class AnnotationItem:
    candidate_id: str
    title: str
    canonical_url: str
    published_at: str
    evidence_role: str
    units: tuple[AnnotationUnit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "canonical_url": self.canonical_url,
            "published_at": self.published_at,
            "evidence_role": self.evidence_role,
            "units": [unit.to_dict() for unit in self.units],
        }


@dataclass(frozen=True)
class AnnotationPacket:
    schema_version: str
    slot: str
    person_id: str
    source_artifact_digest: str
    candidate_set_digest: str
    codebook_version: str
    codebook_digest: str
    codebook: Mapping[str, object]
    created_at: str
    items: tuple[AnnotationItem, ...]
    artifact_digest: str

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "slot": self.slot,
            "person_id": self.person_id,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_set_digest": self.candidate_set_digest,
            "codebook_version": self.codebook_version,
            "codebook_digest": self.codebook_digest,
            "codebook": dict(self.codebook),
            "created_at": self.created_at,
            "items": [item.to_dict() for item in self.items],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True)
class AnnotationLabel:
    candidate_id: str
    disposition: str
    axis_directions: tuple[tuple[str, str], ...]
    evidence_unit_ids: tuple[str, ...]
    counterevidence_unit_ids: tuple[str, ...]
    annotator_confidence: str
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "disposition": self.disposition,
            "axis_directions": [
                [axis, direction]
                for axis, direction in self.axis_directions
            ],
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "counterevidence_unit_ids": list(
                self.counterevidence_unit_ids
            ),
            "annotator_confidence": self.annotator_confidence,
            "rationale": self.rationale,
        }

    def agreement_payload(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "disposition": self.disposition,
            "axis_directions": [
                [axis, direction]
                for axis, direction in self.axis_directions
            ],
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "counterevidence_unit_ids": list(
                self.counterevidence_unit_ids
            ),
        }


@dataclass(frozen=True)
class AnnotationSubmission:
    schema_version: str
    packet_artifact_digest: str
    candidate_set_digest: str
    slot: str
    annotator_id: str
    completed_at: str
    labels: tuple[AnnotationLabel, ...]
    artifact_digest: str
    signature_method: str
    signature: str

    def _base_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "packet_artifact_digest": self.packet_artifact_digest,
            "candidate_set_digest": self.candidate_set_digest,
            "slot": self.slot,
            "annotator_id": self.annotator_id,
            "completed_at": self.completed_at,
            "labels": [label.to_dict() for label in self.labels],
        }

    def signed_payload(self) -> dict[str, object]:
        return {
            **self._base_dict(),
            "artifact_digest": self.artifact_digest,
            "signature_method": self.signature_method,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class DisagreementItem:
    candidate_id: str
    evidence_role: str
    allowed_unit_ids: tuple[str, ...]
    label_a: AnnotationLabel
    label_b: AnnotationLabel

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_role": self.evidence_role,
            "allowed_unit_ids": list(self.allowed_unit_ids),
            "label_a": self.label_a.to_dict(),
            "label_b": self.label_b.to_dict(),
        }


@dataclass(frozen=True)
class AdjudicationPacket:
    schema_version: str
    source_artifact_digest: str
    candidate_set_digest: str
    codebook_digest: str
    packet_a_digest: str
    packet_b_digest: str
    submission_a_digest: str
    submission_b_digest: str
    annotator_ids: tuple[str, str]
    created_at: str
    exact_agreement: float
    disposition_kappa: float
    reliability_passed: bool
    evidence_roles: tuple[tuple[str, str], ...]
    allowed_units: tuple[tuple[str, tuple[str, ...]], ...]
    agreed_labels: tuple[AnnotationLabel, ...]
    disagreements: tuple[DisagreementItem, ...]
    artifact_digest: str

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_set_digest": self.candidate_set_digest,
            "codebook_digest": self.codebook_digest,
            "packet_a_digest": self.packet_a_digest,
            "packet_b_digest": self.packet_b_digest,
            "submission_a_digest": self.submission_a_digest,
            "submission_b_digest": self.submission_b_digest,
            "annotator_ids": list(self.annotator_ids),
            "created_at": self.created_at,
            "exact_agreement": self.exact_agreement,
            "disposition_kappa": self.disposition_kappa,
            "reliability_passed": self.reliability_passed,
            "evidence_roles": [
                [candidate_id, role]
                for candidate_id, role in self.evidence_roles
            ],
            "allowed_units": [
                [candidate_id, list(unit_ids)]
                for candidate_id, unit_ids in self.allowed_units
            ],
            "agreed_labels": [
                label.to_dict() for label in self.agreed_labels
            ],
            "disagreements": [
                item.to_dict() for item in self.disagreements
            ],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "artifact_digest": self.artifact_digest,
        }


@dataclass(frozen=True)
class AdjudicationResolution:
    candidate_id: str
    final_label: AnnotationLabel
    rationale: str


@dataclass(frozen=True)
class AdjudicatedRecord:
    candidate_id: str
    evidence_role: str
    final_label: AnnotationLabel
    training_eligible: bool
    resolution_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_role": self.evidence_role,
            "final_label": self.final_label.to_dict(),
            "training_eligible": self.training_eligible,
            "resolution_source": self.resolution_source,
        }


@dataclass(frozen=True)
class AdjudicatedDataset:
    schema_version: str
    adjudication_packet_digest: str
    source_artifact_digest: str
    candidate_set_digest: str
    codebook_digest: str
    adjudicator_id: str
    adjudicated_at: str
    exact_agreement: float
    disposition_kappa: float
    records: tuple[AdjudicatedRecord, ...]
    training_eligible_count: int
    artifact_digest: str
    signature_method: str
    signature: str

    def _base_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "adjudication_packet_digest": (
                self.adjudication_packet_digest
            ),
            "source_artifact_digest": self.source_artifact_digest,
            "candidate_set_digest": self.candidate_set_digest,
            "codebook_digest": self.codebook_digest,
            "adjudicator_id": self.adjudicator_id,
            "adjudicated_at": self.adjudicated_at,
            "exact_agreement": self.exact_agreement,
            "disposition_kappa": self.disposition_kappa,
            "records": [record.to_dict() for record in self.records],
            "training_eligible_count": self.training_eligible_count,
        }

    def signed_payload(self) -> dict[str, object]:
        return {
            **self._base_dict(),
            "artifact_digest": self.artifact_digest,
            "signature_method": self.signature_method,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


def _unit_id(
    candidate_id: str,
    source_unit_sha256: str,
    occurrence: int,
) -> str:
    return _digest_json(
        {
            "candidate_id": candidate_id,
            "source_unit_sha256": source_unit_sha256,
            "occurrence": occurrence,
        }
    )


def _items_from_source(
    source: TylerSourceArtifact,
) -> tuple[AnnotationItem, ...]:
    items = []
    for post in source.posts:
        if post.candidate_status != "needs_human_annotation":
            continue
        role = _role_for_timestamp(post.published_at)
        units = tuple(
            AnnotationUnit(
                unit_id=_unit_id(post.post_id, unit.sha256, index),
                source_unit_sha256=unit.sha256,
                text=unit.text,
            )
            for index, unit in enumerate(post.authored_prose)
        )
        if not units:
            raise AnnotationRefusedError(("candidate_without_units",))
        items.append(
            AnnotationItem(
                candidate_id=post.post_id,
                title=post.title,
                canonical_url=post.canonical_url,
                published_at=post.published_at,
                evidence_role=role,
                units=units,
            )
        )
    return tuple(sorted(items, key=lambda item: item.candidate_id))


def _candidate_set_digest(
    items: Sequence[AnnotationItem],
) -> str:
    return _digest_json(
        [item.to_dict() for item in sorted(
            items,
            key=lambda item: item.candidate_id,
        )]
    )


def _packet_for_slot(
    source: TylerSourceArtifact,
    items: Sequence[AnnotationItem],
    *,
    slot: str,
    created_at: str,
) -> AnnotationPacket:
    ordered_a = tuple(
        sorted(
            items,
            key=lambda item: _digest_json(
                {"slot": "A", "candidate_id": item.candidate_id}
            ),
        )
    )
    ordered = ordered_a if slot == "A" else tuple(reversed(ordered_a))
    candidate_digest = _candidate_set_digest(items)
    unsigned = {
        "schema_version": PACKET_SCHEMA,
        "slot": slot,
        "person_id": source.person_id,
        "source_artifact_digest": source.artifact_digest,
        "candidate_set_digest": candidate_digest,
        "codebook_version": CODEBOOK_VERSION,
        "codebook_digest": CODEBOOK_DIGEST,
        "codebook": CODEBOOK,
        "created_at": created_at,
        "items": ordered,
    }
    serializable = {
        **unsigned,
        "codebook": dict(CODEBOOK),
        "items": [item.to_dict() for item in ordered],
    }
    return AnnotationPacket(
        **unsigned,
        artifact_digest=_digest_json(serializable),
    )


def create_annotation_packets(
    source: TylerSourceArtifact,
    *,
    created_at: str,
) -> tuple[AnnotationPacket, AnnotationPacket]:
    created = _parse_timestamp(created_at, "created_at")
    if created < _parse_timestamp(source.collected_at, "source collected_at"):
        raise AnnotationRefusedError(
            ("packet_precedes_source_collection",)
        )
    items = _items_from_source(source)
    if len(items) < MINIMUM_CANDIDATES:
        raise AnnotationRefusedError(("insufficient_candidates",))
    return (
        _packet_for_slot(
            source,
            items,
            slot="A",
            created_at=created_at,
        ),
        _packet_for_slot(
            source,
            items,
            slot="B",
            created_at=created_at,
        ),
    )


def verify_annotation_packet(
    packet: AnnotationPacket,
    source: TylerSourceArtifact,
) -> bool:
    if packet.schema_version != PACKET_SCHEMA:
        raise ValueError(f"schema_version must be {PACKET_SCHEMA}")
    if packet.slot not in {"A", "B"}:
        raise ValueError("slot must be A or B")
    if packet.source_artifact_digest != source.artifact_digest:
        raise ValueError("source_artifact_digest mismatch")
    if packet.codebook_version != CODEBOOK_VERSION:
        raise ValueError("codebook_version mismatch")
    if packet.codebook_digest != CODEBOOK_DIGEST:
        raise ValueError("codebook_digest mismatch")
    if dict(packet.codebook) != CODEBOOK:
        raise ValueError("codebook mismatch")
    _require_digest(packet.artifact_digest, "artifact_digest")
    if packet.artifact_digest != _digest_json(packet._unsigned_dict()):
        raise ValueError("artifact_digest mismatch")
    expected = create_annotation_packets(
        source,
        created_at=packet.created_at,
    )[0 if packet.slot == "A" else 1]
    if packet != expected:
        raise ValueError("source packet recomputation mismatch")
    return True


def _canonicalize_label(label: AnnotationLabel) -> AnnotationLabel:
    return AnnotationLabel(
        candidate_id=label.candidate_id,
        disposition=label.disposition,
        axis_directions=tuple(sorted(label.axis_directions)),
        evidence_unit_ids=tuple(sorted(set(label.evidence_unit_ids))),
        counterevidence_unit_ids=tuple(
            sorted(set(label.counterevidence_unit_ids))
        ),
        annotator_confidence=label.annotator_confidence,
        rationale=label.rationale.strip(),
    )


def _validate_label(
    label: AnnotationLabel,
    *,
    candidate_id: str,
    allowed_unit_ids: Sequence[str],
) -> None:
    if label.candidate_id != candidate_id:
        raise AnnotationRefusedError(("candidate_label_mismatch",))
    if label.disposition not in DISPOSITIONS:
        raise AnnotationRefusedError(("invalid_disposition",))
    if label.annotator_confidence not in CONFIDENCE_LEVELS:
        raise AnnotationRefusedError(("invalid_annotator_confidence",))
    if not label.rationale.strip():
        raise AnnotationRefusedError(("missing_rationale",))
    if not label.evidence_unit_ids:
        raise AnnotationRefusedError(("missing_evidence",))
    allowed = set(allowed_unit_ids)
    cited = set(label.evidence_unit_ids)
    counter = set(label.counterevidence_unit_ids)
    if not cited.issubset(allowed) or not counter.issubset(allowed):
        raise AnnotationRefusedError(("invalid_evidence_unit",))
    if cited.intersection(counter):
        raise AnnotationRefusedError(("evidence_counterevidence_overlap",))
    axes = tuple(axis for axis, _ in label.axis_directions)
    if len(axes) != len(set(axes)):
        raise AnnotationRefusedError(("duplicate_axis",))
    for axis, direction in label.axis_directions:
        if axis not in DECISION_AXES:
            raise AnnotationRefusedError(("invalid_axis",))
        if direction not in DIRECTIONS:
            raise AnnotationRefusedError(("invalid_direction",))
    if label.disposition == "clear_in_scope_stance":
        if not label.axis_directions:
            raise AnnotationRefusedError(("axis_required",))
    elif label.axis_directions:
        raise AnnotationRefusedError(("axis_not_allowed",))


def create_annotation_submission(
    packet: AnnotationPacket,
    *,
    labels: Sequence[AnnotationLabel],
    annotator_id: str,
    completed_at: str,
    authority: VerificationAuthority,
) -> AnnotationSubmission:
    completed = _parse_timestamp(completed_at, "completed_at")
    if completed < _parse_timestamp(packet.created_at, "packet created_at"):
        raise AnnotationRefusedError(("submission_precedes_packet",))
    if not annotator_id:
        raise AnnotationRefusedError(("missing_annotator_id",))
    by_candidate = {item.candidate_id: item for item in packet.items}
    if len(labels) != len(by_candidate):
        raise AnnotationRefusedError(("incomplete_coverage",))
    normalized = tuple(
        sorted(
            (_canonicalize_label(label) for label in labels),
            key=lambda label: label.candidate_id,
        )
    )
    if len({label.candidate_id for label in normalized}) != len(
        normalized
    ):
        raise AnnotationRefusedError(("duplicate_candidate_label",))
    if {label.candidate_id for label in normalized} != set(by_candidate):
        raise AnnotationRefusedError(("incomplete_coverage",))
    for label in normalized:
        item = by_candidate[label.candidate_id]
        _validate_label(
            label,
            candidate_id=item.candidate_id,
            allowed_unit_ids=tuple(
                unit.unit_id for unit in item.units
            ),
        )
    base = {
        "schema_version": SUBMISSION_SCHEMA,
        "packet_artifact_digest": packet.artifact_digest,
        "candidate_set_digest": packet.candidate_set_digest,
        "slot": packet.slot,
        "annotator_id": annotator_id,
        "completed_at": completed_at,
        "labels": [label.to_dict() for label in normalized],
    }
    artifact_digest = _digest_json(base)
    signed_payload = {
        **base,
        "artifact_digest": artifact_digest,
        "signature_method": "hmac-sha256",
    }
    signature = authority.sign_payload(
        signed_payload,
        annotator_id,
    )
    return AnnotationSubmission(
        schema_version=SUBMISSION_SCHEMA,
        packet_artifact_digest=packet.artifact_digest,
        candidate_set_digest=packet.candidate_set_digest,
        slot=packet.slot,
        annotator_id=annotator_id,
        completed_at=completed_at,
        labels=normalized,
        artifact_digest=artifact_digest,
        signature_method="hmac-sha256",
        signature=signature,
    )


def verify_annotation_submission(
    submission: AnnotationSubmission,
    packet: AnnotationPacket,
    authority: VerificationAuthority,
) -> bool:
    if submission.schema_version != SUBMISSION_SCHEMA:
        raise ValueError(
            f"schema_version must be {SUBMISSION_SCHEMA}"
        )
    if (
        submission.packet_artifact_digest != packet.artifact_digest
        or submission.candidate_set_digest
        != packet.candidate_set_digest
        or submission.slot != packet.slot
    ):
        raise AnnotationRefusedError(("packet_submission_mismatch",))
    if submission.signature_method != "hmac-sha256":
        raise ValueError("unsupported signature_method")
    _require_digest(submission.artifact_digest, "artifact_digest")
    if submission.artifact_digest != _digest_json(
        submission._base_dict()
    ):
        raise ValueError("artifact_digest mismatch")
    _parse_timestamp(submission.completed_at, "completed_at")
    if _parse_timestamp(
        submission.completed_at,
        "completed_at",
    ) < _parse_timestamp(packet.created_at, "packet created_at"):
        raise ValueError("submission precedes packet")
    by_candidate = {item.candidate_id: item for item in packet.items}
    if (
        len(submission.labels) != len(by_candidate)
        or tuple(
            sorted(
                submission.labels,
                key=lambda label: label.candidate_id,
            )
        )
        != submission.labels
        or {label.candidate_id for label in submission.labels}
        != set(by_candidate)
    ):
        raise ValueError("submission coverage mismatch")
    for label in submission.labels:
        item = by_candidate[label.candidate_id]
        if label != _canonicalize_label(label):
            raise ValueError("label is not canonical")
        _validate_label(
            label,
            candidate_id=item.candidate_id,
            allowed_unit_ids=tuple(
                unit.unit_id for unit in item.units
            ),
        )
    authority.verify_payload(
        submission.signed_payload(),
        submission.annotator_id,
        submission.signature,
    )
    return True


def _cohen_kappa(
    labels_a: Sequence[str],
    labels_b: Sequence[str],
) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("kappa requires equal non-empty label arrays")
    count = len(labels_a)
    observed = sum(
        left == right
        for left, right in zip(labels_a, labels_b, strict=True)
    ) / count
    categories = set(labels_a).union(labels_b)
    expected = sum(
        (labels_a.count(category) / count)
        * (labels_b.count(category) / count)
        for category in categories
    )
    if expected >= 1.0 - 1e-15:
        return 1.0 if observed >= 1.0 - 1e-15 else 0.0
    return (observed - expected) / (1.0 - expected)


def create_adjudication_packet(
    packet_a: AnnotationPacket,
    packet_b: AnnotationPacket,
    submission_a: AnnotationSubmission,
    submission_b: AnnotationSubmission,
    *,
    source: TylerSourceArtifact,
    authority: VerificationAuthority,
) -> AdjudicationPacket:
    verify_annotation_packet(packet_a, source)
    verify_annotation_packet(packet_b, source)
    if packet_a.slot != "A" or packet_b.slot != "B":
        raise AnnotationRefusedError(("packet_slot_mismatch",))
    if (
        packet_a.source_artifact_digest
        != packet_b.source_artifact_digest
        or packet_a.candidate_set_digest
        != packet_b.candidate_set_digest
        or packet_a.codebook_digest != packet_b.codebook_digest
    ):
        raise AnnotationRefusedError(("packet_lineage_mismatch",))
    verify_annotation_submission(
        submission_a,
        packet_a,
        authority,
    )
    verify_annotation_submission(
        submission_b,
        packet_b,
        authority,
    )
    if submission_a.annotator_id == submission_b.annotator_id:
        raise AnnotationRefusedError(
            ("annotators_must_be_distinct",)
        )
    labels_a = {
        label.candidate_id: label for label in submission_a.labels
    }
    labels_b = {
        label.candidate_id: label for label in submission_b.labels
    }
    item_map = {
        item.candidate_id: item for item in packet_a.items
    }
    candidate_ids = tuple(sorted(item_map))
    agreed = []
    disagreements = []
    for candidate_id in candidate_ids:
        left = labels_a[candidate_id]
        right = labels_b[candidate_id]
        if left.agreement_payload() == right.agreement_payload():
            agreed.append(left)
        else:
            item = item_map[candidate_id]
            disagreements.append(
                DisagreementItem(
                    candidate_id=candidate_id,
                    evidence_role=item.evidence_role,
                    allowed_unit_ids=tuple(
                        unit.unit_id for unit in item.units
                    ),
                    label_a=left,
                    label_b=right,
                )
            )
    exact = len(agreed) / len(candidate_ids)
    kappa = _cohen_kappa(
        [labels_a[candidate].disposition for candidate in candidate_ids],
        [labels_b[candidate].disposition for candidate in candidate_ids],
    )
    reliability = (
        exact >= MINIMUM_EXACT_AGREEMENT
        and kappa >= MINIMUM_DISPOSITION_KAPPA
    )
    created_at = max(
        submission_a.completed_at,
        submission_b.completed_at,
        key=lambda value: _parse_timestamp(value, "completed_at"),
    )
    unsigned = {
        "schema_version": ADJUDICATION_SCHEMA,
        "source_artifact_digest": packet_a.source_artifact_digest,
        "candidate_set_digest": packet_a.candidate_set_digest,
        "codebook_digest": packet_a.codebook_digest,
        "packet_a_digest": packet_a.artifact_digest,
        "packet_b_digest": packet_b.artifact_digest,
        "submission_a_digest": submission_a.artifact_digest,
        "submission_b_digest": submission_b.artifact_digest,
        "annotator_ids": (
            submission_a.annotator_id,
            submission_b.annotator_id,
        ),
        "created_at": created_at,
        "exact_agreement": exact,
        "disposition_kappa": kappa,
        "reliability_passed": reliability,
        "evidence_roles": tuple(
            (candidate_id, item_map[candidate_id].evidence_role)
            for candidate_id in candidate_ids
        ),
        "allowed_units": tuple(
            (
                candidate_id,
                tuple(
                    unit.unit_id
                    for unit in item_map[candidate_id].units
                ),
            )
            for candidate_id in candidate_ids
        ),
        "agreed_labels": tuple(agreed),
        "disagreements": tuple(disagreements),
    }
    serializable = {
        **unsigned,
        "annotator_ids": list(unsigned["annotator_ids"]),
        "evidence_roles": [
            list(value) for value in unsigned["evidence_roles"]
        ],
        "allowed_units": [
            [candidate_id, list(unit_ids)]
            for candidate_id, unit_ids in unsigned["allowed_units"]
        ],
        "agreed_labels": [
            label.to_dict() for label in unsigned["agreed_labels"]
        ],
        "disagreements": [
            item.to_dict() for item in unsigned["disagreements"]
        ],
    }
    return AdjudicationPacket(
        **unsigned,
        artifact_digest=_digest_json(serializable),
    )


def _validate_adjudication_packet(
    packet: AdjudicationPacket,
) -> None:
    if packet.schema_version != ADJUDICATION_SCHEMA:
        raise ValueError(
            f"schema_version must be {ADJUDICATION_SCHEMA}"
        )
    if packet.artifact_digest != _digest_json(packet._unsigned_dict()):
        raise ValueError("adjudication artifact_digest mismatch")
    if packet.codebook_digest != CODEBOOK_DIGEST:
        raise ValueError("adjudication codebook_digest mismatch")
    roles = dict(packet.evidence_roles)
    allowed = dict(packet.allowed_units)
    if (
        len(roles) != len(packet.evidence_roles)
        or len(allowed) != len(packet.allowed_units)
        or set(roles) != set(allowed)
    ):
        raise ValueError("adjudication candidate metadata mismatch")
    agreed_ids = {
        label.candidate_id for label in packet.agreed_labels
    }
    disagreement_ids = {
        item.candidate_id for item in packet.disagreements
    }
    if (
        len(agreed_ids) != len(packet.agreed_labels)
        or len(disagreement_ids) != len(packet.disagreements)
        or agreed_ids.intersection(disagreement_ids)
        or agreed_ids.union(disagreement_ids) != set(roles)
    ):
        raise ValueError("adjudication label coverage mismatch")
    disposition_a = []
    disposition_b = []
    for label in packet.agreed_labels:
        _validate_label(
            label,
            candidate_id=label.candidate_id,
            allowed_unit_ids=allowed[label.candidate_id],
        )
        disposition_a.append(label.disposition)
        disposition_b.append(label.disposition)
    for item in packet.disagreements:
        if (
            item.evidence_role != roles[item.candidate_id]
            or item.allowed_unit_ids != allowed[item.candidate_id]
        ):
            raise ValueError("disagreement metadata mismatch")
        _validate_label(
            item.label_a,
            candidate_id=item.candidate_id,
            allowed_unit_ids=item.allowed_unit_ids,
        )
        _validate_label(
            item.label_b,
            candidate_id=item.candidate_id,
            allowed_unit_ids=item.allowed_unit_ids,
        )
        if (
            item.label_a.agreement_payload()
            == item.label_b.agreement_payload()
        ):
            raise ValueError("false disagreement")
        disposition_a.append(item.label_a.disposition)
        disposition_b.append(item.label_b.disposition)
    count = len(packet.agreed_labels) + len(packet.disagreements)
    if count < MINIMUM_CANDIDATES:
        raise ValueError("adjudication packet is too small")
    expected_exact = len(packet.agreed_labels) / count
    if abs(packet.exact_agreement - expected_exact) > 1e-12:
        raise ValueError("exact_agreement mismatch")
    expected_kappa = _cohen_kappa(
        disposition_a,
        disposition_b,
    )
    if abs(packet.disposition_kappa - expected_kappa) > 1e-12:
        raise ValueError("disposition_kappa mismatch")
    expected_passed = (
        packet.exact_agreement >= MINIMUM_EXACT_AGREEMENT
        and packet.disposition_kappa
        >= MINIMUM_DISPOSITION_KAPPA
    )
    if packet.reliability_passed != expected_passed:
        raise ValueError("reliability_passed mismatch")


def verify_adjudication_packet(
    adjudication: AdjudicationPacket,
    packet_a: AnnotationPacket,
    packet_b: AnnotationPacket,
    submission_a: AnnotationSubmission,
    submission_b: AnnotationSubmission,
    *,
    source: TylerSourceArtifact,
    authority: VerificationAuthority,
) -> bool:
    _validate_adjudication_packet(adjudication)
    expected = create_adjudication_packet(
        packet_a,
        packet_b,
        submission_a,
        submission_b,
        source=source,
        authority=authority,
    )
    if adjudication != expected:
        raise ValueError("adjudication raw-evidence recomputation mismatch")
    return True


def finalize_adjudicated_dataset(
    adjudication: AdjudicationPacket,
    *,
    resolutions: Sequence[AdjudicationResolution],
    adjudicator_id: str,
    adjudicated_at: str,
    authority: VerificationAuthority,
) -> AdjudicatedDataset:
    _validate_adjudication_packet(adjudication)
    if not adjudication.reliability_passed:
        raise AnnotationRefusedError(("reliability_floor_failed",))
    if adjudicator_id in adjudication.annotator_ids:
        raise AnnotationRefusedError(
            ("adjudicator_must_be_independent",)
        )
    adjudicated = _parse_timestamp(
        adjudicated_at,
        "adjudicated_at",
    )
    if adjudicated < _parse_timestamp(
        adjudication.created_at,
        "adjudication created_at",
    ):
        raise AnnotationRefusedError(
            ("adjudication_precedes_submissions",)
        )
    disagreement_map = {
        item.candidate_id: item
        for item in adjudication.disagreements
    }
    if (
        len(resolutions) != len(disagreement_map)
        or {resolution.candidate_id for resolution in resolutions}
        != set(disagreement_map)
    ):
        reason = (
            "unresolved_disagreements"
            if len(resolutions) < len(disagreement_map)
            else "invalid_resolution_set"
        )
        raise AnnotationRefusedError((reason,))
    resolution_labels = {}
    for resolution in resolutions:
        if not resolution.rationale.strip():
            raise AnnotationRefusedError(
                ("missing_adjudicator_rationale",)
            )
        disagreement = disagreement_map[resolution.candidate_id]
        label = _canonicalize_label(resolution.final_label)
        _validate_label(
            label,
            candidate_id=resolution.candidate_id,
            allowed_unit_ids=disagreement.allowed_unit_ids,
        )
        resolution_labels[resolution.candidate_id] = label
    roles = dict(adjudication.evidence_roles)
    final_labels = {
        label.candidate_id: label
        for label in adjudication.agreed_labels
    }
    final_labels.update(resolution_labels)
    records = tuple(
        AdjudicatedRecord(
            candidate_id=candidate_id,
            evidence_role=roles[candidate_id],
            final_label=final_labels[candidate_id],
            training_eligible=(
                roles[candidate_id] == "training_discovery"
                and final_labels[candidate_id].disposition
                == "clear_in_scope_stance"
            ),
            resolution_source=(
                "adjudicator"
                if candidate_id in resolution_labels
                else "annotator_agreement"
            ),
        )
        for candidate_id in sorted(final_labels)
    )
    training_count = sum(
        record.training_eligible for record in records
    )
    base = {
        "schema_version": DATASET_SCHEMA,
        "adjudication_packet_digest": adjudication.artifact_digest,
        "source_artifact_digest": adjudication.source_artifact_digest,
        "candidate_set_digest": adjudication.candidate_set_digest,
        "codebook_digest": adjudication.codebook_digest,
        "adjudicator_id": adjudicator_id,
        "adjudicated_at": adjudicated_at,
        "exact_agreement": adjudication.exact_agreement,
        "disposition_kappa": adjudication.disposition_kappa,
        "records": [record.to_dict() for record in records],
        "training_eligible_count": training_count,
    }
    artifact_digest = _digest_json(base)
    signed_payload = {
        **base,
        "artifact_digest": artifact_digest,
        "signature_method": "hmac-sha256",
    }
    signature = authority.sign_payload(
        signed_payload,
        adjudicator_id,
    )
    return AdjudicatedDataset(
        schema_version=DATASET_SCHEMA,
        adjudication_packet_digest=adjudication.artifact_digest,
        source_artifact_digest=adjudication.source_artifact_digest,
        candidate_set_digest=adjudication.candidate_set_digest,
        codebook_digest=adjudication.codebook_digest,
        adjudicator_id=adjudicator_id,
        adjudicated_at=adjudicated_at,
        exact_agreement=adjudication.exact_agreement,
        disposition_kappa=adjudication.disposition_kappa,
        records=records,
        training_eligible_count=training_count,
        artifact_digest=artifact_digest,
        signature_method="hmac-sha256",
        signature=signature,
    )


def verify_adjudicated_dataset(
    dataset: AdjudicatedDataset,
    adjudication: AdjudicationPacket,
    authority: VerificationAuthority,
) -> bool:
    _validate_adjudication_packet(adjudication)
    if dataset.schema_version != DATASET_SCHEMA:
        raise ValueError(f"schema_version must be {DATASET_SCHEMA}")
    if (
        dataset.adjudication_packet_digest
        != adjudication.artifact_digest
        or dataset.source_artifact_digest
        != adjudication.source_artifact_digest
        or dataset.candidate_set_digest
        != adjudication.candidate_set_digest
        or dataset.codebook_digest != adjudication.codebook_digest
    ):
        raise ValueError("dataset lineage mismatch")
    if dataset.signature_method != "hmac-sha256":
        raise ValueError("unsupported signature_method")
    if dataset.artifact_digest != _digest_json(dataset._base_dict()):
        raise ValueError("artifact_digest mismatch")
    roles = dict(adjudication.evidence_roles)
    allowed = dict(adjudication.allowed_units)
    agreed = {
        label.candidate_id: label
        for label in adjudication.agreed_labels
    }
    disagreement_ids = {
        item.candidate_id
        for item in adjudication.disagreements
    }
    if (
        tuple(
            sorted(
                dataset.records,
                key=lambda record: record.candidate_id,
            )
        )
        != dataset.records
        or {record.candidate_id for record in dataset.records}
        != set(roles)
    ):
        raise ValueError("dataset record coverage mismatch")
    for record in dataset.records:
        if record.evidence_role != roles[record.candidate_id]:
            raise ValueError("evidence_role mismatch")
        if record.candidate_id in agreed:
            if record.final_label != agreed[record.candidate_id]:
                raise ValueError("agreed label changed")
            if record.resolution_source != "annotator_agreement":
                raise ValueError("agreed resolution_source mismatch")
        elif record.candidate_id in disagreement_ids:
            if record.resolution_source != "adjudicator":
                raise ValueError(
                    "disagreement resolution_source mismatch"
                )
        else:
            raise ValueError("unknown adjudicated candidate")
        _validate_label(
            record.final_label,
            candidate_id=record.candidate_id,
            allowed_unit_ids=allowed[record.candidate_id],
        )
        expected_training = (
            record.evidence_role == "training_discovery"
            and record.final_label.disposition
            == "clear_in_scope_stance"
        )
        if record.training_eligible != expected_training:
            raise ValueError("training_eligible mismatch")
    if dataset.training_eligible_count != sum(
        record.training_eligible for record in dataset.records
    ):
        raise ValueError("training_eligible_count mismatch")
    if (
        dataset.exact_agreement != adjudication.exact_agreement
        or dataset.disposition_kappa
        != adjudication.disposition_kappa
    ):
        raise ValueError("reliability metric mismatch")
    authority.verify_payload(
        dataset.signed_payload(),
        dataset.adjudicator_id,
        dataset.signature,
    )
    return True


def _unit_from_dict(data: Mapping[str, object]) -> AnnotationUnit:
    return AnnotationUnit(
        unit_id=str(data["unit_id"]),
        source_unit_sha256=str(data["source_unit_sha256"]),
        text=str(data["text"]),
    )


def _item_from_dict(data: Mapping[str, object]) -> AnnotationItem:
    return AnnotationItem(
        candidate_id=str(data["candidate_id"]),
        title=str(data["title"]),
        canonical_url=str(data["canonical_url"]),
        published_at=str(data["published_at"]),
        evidence_role=str(data["evidence_role"]),
        units=tuple(
            _unit_from_dict(dict(unit))
            for unit in data["units"]
        ),
    )


def _label_from_dict(data: Mapping[str, object]) -> AnnotationLabel:
    return AnnotationLabel(
        candidate_id=str(data["candidate_id"]),
        disposition=str(data["disposition"]),
        axis_directions=tuple(
            (str(value[0]), str(value[1]))
            for value in data["axis_directions"]
        ),
        evidence_unit_ids=tuple(
            str(value) for value in data["evidence_unit_ids"]
        ),
        counterevidence_unit_ids=tuple(
            str(value)
            for value in data["counterevidence_unit_ids"]
        ),
        annotator_confidence=str(data["annotator_confidence"]),
        rationale=str(data["rationale"]),
    )


def annotation_packet_from_dict(
    data: Mapping[str, object],
) -> AnnotationPacket:
    if data.get("schema_version") != PACKET_SCHEMA:
        raise ValueError(f"schema_version must be {PACKET_SCHEMA}")
    packet = AnnotationPacket(
        schema_version=str(data["schema_version"]),
        slot=str(data["slot"]),
        person_id=str(data["person_id"]),
        source_artifact_digest=str(data["source_artifact_digest"]),
        candidate_set_digest=str(data["candidate_set_digest"]),
        codebook_version=str(data["codebook_version"]),
        codebook_digest=str(data["codebook_digest"]),
        codebook=dict(data["codebook"]),
        created_at=str(data["created_at"]),
        items=tuple(
            _item_from_dict(dict(item))
            for item in data["items"]
        ),
        artifact_digest=str(data["artifact_digest"]),
    )
    if packet.artifact_digest != _digest_json(packet._unsigned_dict()):
        raise ValueError("artifact_digest mismatch")
    return packet


def annotation_submission_from_dict(
    data: Mapping[str, object],
) -> AnnotationSubmission:
    if data.get("schema_version") != SUBMISSION_SCHEMA:
        raise ValueError(
            f"schema_version must be {SUBMISSION_SCHEMA}"
        )
    submission = AnnotationSubmission(
        schema_version=str(data["schema_version"]),
        packet_artifact_digest=str(data["packet_artifact_digest"]),
        candidate_set_digest=str(data["candidate_set_digest"]),
        slot=str(data["slot"]),
        annotator_id=str(data["annotator_id"]),
        completed_at=str(data["completed_at"]),
        labels=tuple(
            _label_from_dict(dict(label))
            for label in data["labels"]
        ),
        artifact_digest=str(data["artifact_digest"]),
        signature_method=str(data["signature_method"]),
        signature=str(data["signature"]),
    )
    if submission.artifact_digest != _digest_json(
        submission._base_dict()
    ):
        raise ValueError("artifact_digest mismatch")
    return submission


def adjudicated_dataset_from_dict(
    data: Mapping[str, object],
) -> AdjudicatedDataset:
    if data.get("schema_version") != DATASET_SCHEMA:
        raise ValueError(f"schema_version must be {DATASET_SCHEMA}")
    records = tuple(
        AdjudicatedRecord(
            candidate_id=str(record["candidate_id"]),
            evidence_role=str(record["evidence_role"]),
            final_label=_label_from_dict(
                dict(record["final_label"])
            ),
            training_eligible=bool(record["training_eligible"]),
            resolution_source=str(record["resolution_source"]),
        )
        for raw_record in data["records"]
        for record in (dict(raw_record),)
    )
    dataset = AdjudicatedDataset(
        schema_version=str(data["schema_version"]),
        adjudication_packet_digest=str(
            data["adjudication_packet_digest"]
        ),
        source_artifact_digest=str(data["source_artifact_digest"]),
        candidate_set_digest=str(data["candidate_set_digest"]),
        codebook_digest=str(data["codebook_digest"]),
        adjudicator_id=str(data["adjudicator_id"]),
        adjudicated_at=str(data["adjudicated_at"]),
        exact_agreement=float(data["exact_agreement"]),
        disposition_kappa=float(data["disposition_kappa"]),
        records=records,
        training_eligible_count=int(data["training_eligible_count"]),
        artifact_digest=str(data["artifact_digest"]),
        signature_method=str(data["signature_method"]),
        signature=str(data["signature"]),
    )
    if dataset.artifact_digest != _digest_json(dataset._base_dict()):
        raise ValueError("artifact_digest mismatch")
    return dataset


def adjudication_packet_from_dict(
    data: Mapping[str, object],
) -> AdjudicationPacket:
    if data.get("schema_version") != ADJUDICATION_SCHEMA:
        raise ValueError(
            f"schema_version must be {ADJUDICATION_SCHEMA}"
        )
    disagreements = tuple(
        DisagreementItem(
            candidate_id=str(item["candidate_id"]),
            evidence_role=str(item["evidence_role"]),
            allowed_unit_ids=tuple(
                str(value) for value in item["allowed_unit_ids"]
            ),
            label_a=_label_from_dict(dict(item["label_a"])),
            label_b=_label_from_dict(dict(item["label_b"])),
        )
        for raw_item in data["disagreements"]
        for item in (dict(raw_item),)
    )
    packet = AdjudicationPacket(
        schema_version=str(data["schema_version"]),
        source_artifact_digest=str(data["source_artifact_digest"]),
        candidate_set_digest=str(data["candidate_set_digest"]),
        codebook_digest=str(data["codebook_digest"]),
        packet_a_digest=str(data["packet_a_digest"]),
        packet_b_digest=str(data["packet_b_digest"]),
        submission_a_digest=str(data["submission_a_digest"]),
        submission_b_digest=str(data["submission_b_digest"]),
        annotator_ids=tuple(
            str(value) for value in data["annotator_ids"]
        ),
        created_at=str(data["created_at"]),
        exact_agreement=float(data["exact_agreement"]),
        disposition_kappa=float(data["disposition_kappa"]),
        reliability_passed=bool(data["reliability_passed"]),
        evidence_roles=tuple(
            (str(value[0]), str(value[1]))
            for value in data["evidence_roles"]
        ),
        allowed_units=tuple(
            (
                str(value[0]),
                tuple(str(unit_id) for unit_id in value[1]),
            )
            for value in data["allowed_units"]
        ),
        agreed_labels=tuple(
            _label_from_dict(dict(label))
            for label in data["agreed_labels"]
        ),
        disagreements=disagreements,
        artifact_digest=str(data["artifact_digest"]),
    )
    _validate_adjudication_packet(packet)
    return packet


def save_annotation_packet(
    packet: AnnotationPacket,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            packet.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def annotation_submission_template(
    packet: AnnotationPacket,
) -> dict[str, object]:
    return {
        "schema_version": "tyler-annotation-submission-template-v1",
        "packet_artifact_digest": packet.artifact_digest,
        "candidate_set_digest": packet.candidate_set_digest,
        "codebook_digest": packet.codebook_digest,
        "slot": packet.slot,
        "instructions": (
            "Complete every blank label independently. Cite only "
            "available_unit_ids. This template is not a valid signed "
            "submission until processed by create_annotation_submission."
        ),
        "labels": [
            {
                "candidate_id": item.candidate_id,
                "disposition": "",
                "axis_directions": [],
                "evidence_unit_ids": [],
                "counterevidence_unit_ids": [],
                "annotator_confidence": "",
                "rationale": "",
                "available_unit_ids": [
                    unit.unit_id for unit in item.units
                ],
            }
            for item in packet.items
        ],
    }


def save_annotation_submission_template(
    packet: AnnotationPacket,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            annotation_submission_template(packet),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_annotation_packet(path: Path) -> AnnotationPacket:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("annotation packet must be a JSON object")
    return annotation_packet_from_dict(raw)


def save_annotation_submission(
    submission: AnnotationSubmission,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            submission.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_annotation_submission(path: Path) -> AnnotationSubmission:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "annotation submission must be a JSON object"
        )
    return annotation_submission_from_dict(raw)


def save_adjudicated_dataset(
    dataset: AdjudicatedDataset,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dataset.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_adjudication_packet(
    packet: AdjudicationPacket,
    path: Path,
) -> None:
    _validate_adjudication_packet(packet)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            packet.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_adjudication_packet(path: Path) -> AdjudicationPacket:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "adjudication packet must be a JSON object"
        )
    return adjudication_packet_from_dict(raw)


def load_adjudicated_dataset(path: Path) -> AdjudicatedDataset:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            "adjudicated dataset must be a JSON object"
        )
    return adjudicated_dataset_from_dict(raw)
