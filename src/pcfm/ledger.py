from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

from .contracts import Observation


ALLOWED_VERIFIED_SOURCES = frozenset(
    {
        "human_record",
        "synthetic_ground_truth",
    }
)


def _parse_iso8601(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error


def observation_payload(observation: Observation) -> dict[str, object]:
    return {
        "person_id": observation.person_id,
        "scenario": {
            "scenario_id": observation.scenario.scenario_id,
            "features": {
                name: value
                for name, value in zip(
                    observation.scenario.feature_names,
                    observation.scenario.features,
                    strict=True,
                )
            },
            "options": list(observation.scenario.options),
            "domain": observation.scenario.domain,
            "context": dict(sorted(observation.scenario.context.items())),
        },
        "actual_choice": observation.actual_choice,
        "confidence": observation.confidence,
        "reaction_time_ms": observation.reaction_time_ms,
        "provenance": observation.provenance,
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    observation: Observation
    observed_at: str
    evidence_hash: str
    verifier_id: str
    verified_at: str
    signature: str
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if not self.verifier_id:
            raise ValueError("verifier_id is required")
        observed_at = _parse_iso8601(self.observed_at, "observed_at")
        verified_at = _parse_iso8601(self.verified_at, "verified_at")
        if verified_at < observed_at:
            raise ValueError("verified_at cannot precede observed_at")
        _require_sha256(self.evidence_hash, "evidence_hash")
        _require_sha256(self.signature, "signature")
        if self.signature_method != "hmac-sha256":
            raise ValueError("unsupported event signature method")

    def signed_payload(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "observation": observation_payload(self.observation),
            "observed_at": self.observed_at,
            "evidence_hash": self.evidence_hash,
            "verifier_id": self.verifier_id,
            "verified_at": self.verified_at,
            "signature_method": self.signature_method,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


class VerificationAuthority:
    def __init__(self, keys: Mapping[str, bytes]) -> None:
        converted = {
            str(verifier_id): bytes(secret)
            for verifier_id, secret in keys.items()
        }
        if not converted or any(not key or not secret for key, secret in converted.items()):
            raise ValueError("at least one non-empty verifier key is required")
        self._keys = converted

    def sign(
        self,
        *,
        event_id: str,
        observation: Observation,
        observed_at: str,
        evidence_hash: str,
        verifier_id: str,
        verified_at: str,
    ) -> EventRecord:
        if observation.provenance not in ALLOWED_VERIFIED_SOURCES:
            raise ValueError("event provenance is not allowed for verified training")
        if verifier_id not in self._keys:
            raise ValueError("unknown verifier_id")
        unsigned = {
            "event_id": event_id,
            "observation": observation_payload(observation),
            "observed_at": observed_at,
            "evidence_hash": evidence_hash,
            "verifier_id": verifier_id,
            "verified_at": verified_at,
            "signature_method": "hmac-sha256",
        }
        signature = hmac.new(
            self._keys[verifier_id],
            _canonical_json(unsigned),
            hashlib.sha256,
        ).hexdigest()
        return EventRecord(
            event_id=event_id,
            observation=observation,
            observed_at=observed_at,
            evidence_hash=evidence_hash,
            verifier_id=verifier_id,
            verified_at=verified_at,
            signature=signature,
        )

    @property
    def verifier_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def sign_payload(
        self,
        payload: Mapping[str, object],
        verifier_id: str,
    ) -> str:
        if verifier_id not in self._keys:
            raise ValueError("unknown verifier_id")
        return hmac.new(
            self._keys[verifier_id],
            _canonical_json(dict(payload)),
            hashlib.sha256,
        ).hexdigest()

    def verify_payload(
        self,
        payload: Mapping[str, object],
        verifier_id: str,
        signature: str,
    ) -> None:
        if verifier_id not in self._keys:
            raise ValueError("unknown verifier_id")
        _require_sha256(signature, "signature")
        expected = self.sign_payload(payload, verifier_id)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("payload signature verification failed")

    def verify(self, record: EventRecord) -> None:
        if record.observation.provenance not in ALLOWED_VERIFIED_SOURCES:
            raise ValueError("event provenance is not allowed for verified training")
        secret = self._keys.get(record.verifier_id)
        if secret is None:
            raise ValueError("unknown verifier_id")
        expected = hmac.new(
            secret,
            _canonical_json(record.signed_payload()),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, record.signature):
            raise ValueError("event signature verification failed")


@dataclass(frozen=True)
class EventLedger:
    records: tuple[EventRecord, ...]

    @classmethod
    def verify(
        cls,
        records: Sequence[EventRecord],
        authority: VerificationAuthority,
    ) -> EventLedger:
        converted = tuple(records)
        if not converted:
            raise ValueError("event ledger must not be empty")
        event_ids = [record.event_id for record in converted]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("duplicate event_id in event ledger")
        trial_ids = [
            (
                record.observation.person_id,
                record.observation.scenario.scenario_id,
            )
            for record in converted
        ]
        if len(set(trial_ids)) != len(trial_ids):
            raise ValueError("duplicate person/scenario trial in event ledger")
        for record in converted:
            authority.verify(record)
        return cls(records=converted)

    def records_for_person(self, person_id: str) -> tuple[EventRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.observation.person_id == person_id
        )

    def observations(self) -> tuple[Observation, ...]:
        return tuple(record.observation for record in self.records)

    def append(
        self,
        record: EventRecord,
        authority: VerificationAuthority,
    ) -> EventLedger:
        if any(existing.event_id == record.event_id for existing in self.records):
            raise ValueError(f"event_id {record.event_id} already exists")
        return self.verify(self.records + (record,), authority)

    @staticmethod
    def snapshot_hash(records: Sequence[EventRecord]) -> str:
        canonical_records = [
            record.to_dict()
            for record in sorted(records, key=lambda item: item.event_id)
        ]
        return hashlib.sha256(_canonical_json(canonical_records)).hexdigest()

    def person_snapshot_hash(self, person_id: str) -> str:
        records = self.records_for_person(person_id)
        if not records:
            raise ValueError(f"no events found for person {person_id}")
        return self.snapshot_hash(records)
