from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import Scenario
from .interfaces import CognitiveModule
from .ledger import EventLedger, EventRecord, VerificationAuthority
from .math_utils import logistic_normal_probability, sigmoid
from .storage import (
    PersonModelBundle,
    scenario_design_hash,
    trial_key_hash,
)


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
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 digest") from error


def _binary_loss(
    probabilities: np.ndarray,
    choices: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    return -(
        choices * np.log(clipped)
        + (1.0 - choices) * np.log(1.0 - clipped)
    )


def _expected_calibration_error(
    probabilities: np.ndarray,
    choices: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (
            (probabilities >= lower)
            & (
                probabilities <= upper
                if index == bins - 1
                else probabilities < upper
            )
        )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(probabilities[mask]))
                - float(np.mean(choices[mask]))
            )
    return error


def _hac_standard_error(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("inf")
    centered = values - np.mean(values)
    sample_count = len(centered)
    maximum_lag = min(
        sample_count - 1,
        max(1, int(round(sample_count ** (1.0 / 3.0)))),
    )
    long_run_variance = float(centered @ centered / sample_count)
    for lag in range(1, maximum_lag + 1):
        covariance = float(
            centered[lag:] @ centered[:-lag] / sample_count
        )
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    return sqrt(max(long_run_variance, 0.0) / sample_count)


def _iid_standard_error(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("inf")
    return float(np.std(values, ddof=1) / sqrt(len(values)))


def _confirmation_standard_error(
    values: np.ndarray,
    records: Sequence[EventRecord],
) -> tuple[float, str]:
    timestamps = tuple(
        _parse_timestamp(record.observed_at, "observed_at")
        for record in records
    )
    if len(set(timestamps)) == len(timestamps):
        return _hac_standard_error(values), "time_ordered_hac"
    return _iid_standard_error(values), "iid"


def _option_signature(scenario: Scenario) -> str:
    return json.dumps(
        list(scenario.options),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _context_signature(scenario: Scenario) -> str:
    return json.dumps(
        sorted(
            (str(name), str(value))
            for name, value in scenario.context.items()
            if name != "prediction_at"
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


class MechanismRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "mechanism comparison refused: "
            + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class EvidenceWindow:
    start_at: str
    end_at: str
    expected_event_count: int

    def __post_init__(self) -> None:
        start = _parse_timestamp(self.start_at, "start_at")
        end = _parse_timestamp(self.end_at, "end_at")
        if start >= end:
            raise ValueError(
                "mechanism evidence window start must precede end"
            )
        if self.expected_event_count <= 0:
            raise ValueError(
                "mechanism expected_event_count must be positive"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "start_at": self.start_at,
            "end_at": self.end_at,
            "expected_event_count": self.expected_event_count,
        }


_TERM_ARITY = {
    "intercept": 0,
    "linear": 1,
    "absolute": 1,
    "quadratic": 1,
    "interaction": 2,
}


@dataclass(frozen=True)
class MechanismTerm:
    term_id: str
    kind: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        object.__setattr__(self, "feature_names", names)
        if not self.term_id:
            raise ValueError("mechanism term_id is required")
        if self.kind not in _TERM_ARITY:
            raise ValueError("unsupported mechanism term kind")
        if len(names) != _TERM_ARITY[self.kind]:
            raise ValueError(
                "mechanism term feature arity is invalid"
            )
        if any(not name for name in names):
            raise ValueError(
                "mechanism term feature names must be non-empty"
            )
        if self.kind == "interaction" and names[0] == names[1]:
            raise ValueError(
                "interaction term requires two distinct features"
            )

    def signature(self) -> tuple[str, tuple[str, ...]]:
        names = (
            tuple(sorted(self.feature_names))
            if self.kind == "interaction"
            else self.feature_names
        )
        return self.kind, names

    def evaluate(
        self,
        scenario: Scenario,
        model_feature_names: tuple[str, ...],
    ) -> float:
        ordered = scenario.ordered_features(model_feature_names)
        by_name = dict(
            zip(model_feature_names, ordered, strict=True)
        )
        if self.kind == "intercept":
            return 1.0
        first = by_name[self.feature_names[0]]
        if self.kind == "linear":
            return first
        if self.kind == "absolute":
            return abs(first)
        if self.kind == "quadratic":
            return first * first
        return first * by_name[self.feature_names[1]]

    def to_dict(self) -> dict[str, object]:
        return {
            "term_id": self.term_id,
            "kind": self.kind,
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True)
class MechanismHypothesis:
    hypothesis_id: str
    terms: tuple[MechanismTerm, ...]

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        object.__setattr__(self, "terms", terms)
        if not self.hypothesis_id or not terms:
            raise ValueError(
                "mechanism hypothesis identity and terms are required"
            )
        term_ids = tuple(term.term_id for term in terms)
        if len(set(term_ids)) != len(term_ids):
            raise ValueError(
                "mechanism hypothesis term ids must be unique"
            )
        signatures = tuple(term.signature() for term in terms)
        if len(set(signatures)) != len(signatures):
            raise ValueError(
                "mechanism hypothesis terms must be structurally unique"
            )

    def structural_signature(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(sorted(term.signature() for term in self.terms))

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "terms": [term.to_dict() for term in self.terms],
        }


@dataclass(frozen=True)
class MechanismComparisonConfig:
    minimum_samples_per_split: int = 100
    maximum_hypotheses: int = 12
    maximum_terms_per_hypothesis: int = 16
    l2_precision: float = 1.0
    minimum_confirmation_nll_uplift: float = 0.01
    maximum_confirmation_calibration_error: float = 0.15
    confidence_z: float = 1.96
    maximum_report_age_days: float = 180.0
    model_version: str = "preregistered-mechanism-comparison-v2"

    def __post_init__(self) -> None:
        if self.minimum_samples_per_split < 100:
            raise ValueError(
                "minimum_samples_per_split must be at least 100"
            )
        if self.maximum_hypotheses < 2:
            raise ValueError(
                "maximum_hypotheses must be at least two"
            )
        if self.maximum_terms_per_hypothesis <= 0:
            raise ValueError(
                "maximum_terms_per_hypothesis must be positive"
            )
        if not isfinite(self.l2_precision) or self.l2_precision <= 0:
            raise ValueError(
                "mechanism comparison positive values are invalid"
            )
        if not isfinite(self.confidence_z) or self.confidence_z < 1.96:
            raise ValueError(
                "confidence_z must be at least 1.96"
            )
        if (
            not isfinite(self.minimum_confirmation_nll_uplift)
            or self.minimum_confirmation_nll_uplift < 0.01
        ):
            raise ValueError(
                "minimum_confirmation_nll_uplift must be at least 0.01"
            )
        if (
            not isfinite(
                self.maximum_confirmation_calibration_error
            )
            or not 0
            < self.maximum_confirmation_calibration_error
            <= 0.15
        ):
            raise ValueError(
                "maximum_confirmation_calibration_error must be "
                "in (0, 0.15]"
            )
        if (
            not isfinite(self.maximum_report_age_days)
            or not 0 < self.maximum_report_age_days <= 180
        ):
            raise ValueError(
                "maximum_report_age_days must be in (0, 180]"
            )
        if (
            self.model_version
            != "preregistered-mechanism-comparison-v2"
        ):
            raise ValueError(
                "unsupported mechanism comparison model_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_samples_per_split": (
                self.minimum_samples_per_split
            ),
            "maximum_hypotheses": self.maximum_hypotheses,
            "maximum_terms_per_hypothesis": (
                self.maximum_terms_per_hypothesis
            ),
            "l2_precision": self.l2_precision,
            "minimum_confirmation_nll_uplift": (
                self.minimum_confirmation_nll_uplift
            ),
            "maximum_confirmation_calibration_error": (
                self.maximum_confirmation_calibration_error
            ),
            "confidence_z": self.confidence_z,
            "maximum_report_age_days": self.maximum_report_age_days,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class MechanismComparisonPlan:
    base_model_id: str
    person_id: str
    registered_at: str
    discovery_window: EvidenceWindow
    selection_window: EvidenceWindow
    confirmation_window: EvidenceWindow
    hypotheses: tuple[MechanismHypothesis, ...]
    config: MechanismComparisonConfig
    verifier_id: str
    artifact_version: str = "pcfm-mechanism-plan-v2"
    plan_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _require_digest(self.base_model_id, "base_model_id")
        registered = _parse_timestamp(
            self.registered_at,
            "registered_at",
        )
        hypotheses = tuple(self.hypotheses)
        object.__setattr__(self, "hypotheses", hypotheses)
        if not self.person_id or not self.verifier_id:
            raise ValueError(
                "mechanism plan identity is required"
            )
        if not 2 <= len(hypotheses) <= self.config.maximum_hypotheses:
            raise ValueError(
                "mechanism plan hypothesis count is invalid"
            )
        hypothesis_ids = tuple(
            hypothesis.hypothesis_id for hypothesis in hypotheses
        )
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError(
                "mechanism hypothesis ids must be unique"
            )
        structural_signatures = tuple(
            hypothesis.structural_signature()
            for hypothesis in hypotheses
        )
        if len(set(structural_signatures)) != len(
            structural_signatures
        ):
            raise ValueError(
                "mechanism hypotheses must be structurally unique"
            )
        if any(
            len(hypothesis.terms)
            > self.config.maximum_terms_per_hypothesis
            for hypothesis in hypotheses
        ):
            raise ValueError(
                "mechanism hypothesis has too many terms"
            )
        windows = (
            self.discovery_window,
            self.selection_window,
            self.confirmation_window,
        )
        if any(
            window.expected_event_count
            < self.config.minimum_samples_per_split
            for window in windows
        ):
            raise ValueError(
                "mechanism evidence window has too few events"
            )
        discovery_start = _parse_timestamp(
            self.discovery_window.start_at,
            "discovery_start_at",
        )
        discovery_end = _parse_timestamp(
            self.discovery_window.end_at,
            "discovery_end_at",
        )
        selection_start = _parse_timestamp(
            self.selection_window.start_at,
            "selection_start_at",
        )
        selection_end = _parse_timestamp(
            self.selection_window.end_at,
            "selection_end_at",
        )
        confirmation_start = _parse_timestamp(
            self.confirmation_window.start_at,
            "confirmation_start_at",
        )
        if not (
            registered < discovery_start
            and discovery_end < selection_start
            and selection_end < confirmation_start
        ):
            raise ValueError(
                "mechanism evidence windows must be preregistered "
                "and strictly ordered"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported mechanism plan signature method"
            )
        expected = self.digest()
        if self.plan_id:
            _require_digest(self.plan_id, "plan_id")
            if self.plan_id != expected:
                raise ValueError(
                    "mechanism plan_id does not match content"
                )
        else:
            object.__setattr__(self, "plan_id", expected)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "base_model_id": self.base_model_id,
            "person_id": self.person_id,
            "registered_at": self.registered_at,
            "discovery_window": self.discovery_window.to_dict(),
            "selection_window": self.selection_window.to_dict(),
            "confirmation_window": (
                self.confirmation_window.to_dict()
            ),
            "hypotheses": [
                hypothesis.to_dict()
                for hypothesis in self.hypotheses
            ],
            "config": self.config.to_dict(),
            "verifier_id": self.verifier_id,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "plan_id": self.plan_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("mechanism plan is unsigned")
        authority.verify_payload(
            self.signed_payload(),
            self.verifier_id,
            self.signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class MechanismCandidateFit:
    hypothesis_id: str
    term_ids: tuple[str, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    discovery_nll: float
    selection_nll: float
    selection_nll_uplift: float

    def __post_init__(self) -> None:
        dimension = len(self.term_ids)
        vectors = (
            self.centers,
            self.scales,
            self.coefficients,
        )
        if (
            not self.hypothesis_id
            or dimension == 0
            or len(set(self.term_ids)) != dimension
            or any(len(vector) != dimension for vector in vectors)
        ):
            raise ValueError(
                "mechanism candidate fit dimensions are invalid"
            )
        if any(
            not isfinite(value)
            for vector in vectors
            for value in vector
        ) or any(value <= 0 for value in self.scales):
            raise ValueError(
                "mechanism candidate fit values are invalid"
            )
        covariance = np.asarray(
            self.covariance,
            dtype=np.float64,
        )
        if (
            covariance.shape != (dimension, dimension)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, atol=1e-10)
            or np.min(np.linalg.eigvalsh(covariance)) <= 0
        ):
            raise ValueError(
                "mechanism candidate covariance is invalid"
            )
        metrics = (
            self.discovery_nll,
            self.selection_nll,
            self.selection_nll_uplift,
        )
        if not all(isfinite(value) for value in metrics):
            raise ValueError(
                "mechanism candidate metrics are invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "term_ids": list(self.term_ids),
            "centers": list(self.centers),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "covariance": [
                list(row) for row in self.covariance
            ],
            "discovery_nll": self.discovery_nll,
            "selection_nll": self.selection_nll,
            "selection_nll_uplift": (
                self.selection_nll_uplift
            ),
        }


@dataclass(frozen=True)
class MechanismComparisonReport:
    plan_id: str
    base_model_id: str
    person_id: str
    evaluated_at: str
    discovery_event_ids: tuple[str, ...]
    discovery_data_hash: str
    selection_event_ids: tuple[str, ...]
    selection_data_hash: str
    confirmation_event_ids: tuple[str, ...]
    confirmation_data_hash: str
    candidate_fits: tuple[MechanismCandidateFit, ...]
    selected_hypothesis_id: str
    confirmation_base_nll: float
    confirmation_nll: float
    confirmation_nll_uplift: float
    confirmation_nll_uplift_standard_error: float
    confirmation_nll_uplift_ci_lower: float
    confirmation_nll_uplift_ci_upper: float
    confirmation_calibration_error: float
    confirmation_standard_error_method: str
    supported_domains: tuple[str, ...]
    supported_option_signatures: tuple[str, ...]
    supported_context_signatures: tuple[str, ...]
    selected_term_minimums: tuple[float, ...]
    selected_term_maximums: tuple[float, ...]
    status: str
    reasons: tuple[str, ...]
    verifier_id: str
    interpretation: str = "predictive_structure_only"
    artifact_version: str = "pcfm-mechanism-report-v2"
    report_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        for value, label in (
            (self.plan_id, "plan_id"),
            (self.base_model_id, "base_model_id"),
            (self.discovery_data_hash, "discovery_data_hash"),
            (self.selection_data_hash, "selection_data_hash"),
            (
                self.confirmation_data_hash,
                "confirmation_data_hash",
            ),
        ):
            _require_digest(value, label)
        _parse_timestamp(self.evaluated_at, "evaluated_at")
        if (
            not self.person_id
            or not self.selected_hypothesis_id
            or not self.verifier_id
        ):
            raise ValueError(
                "mechanism report identity is required"
            )
        for event_ids, label in (
            (self.discovery_event_ids, "discovery"),
            (self.selection_event_ids, "selection"),
            (self.confirmation_event_ids, "confirmation"),
        ):
            if (
                not event_ids
                or len(set(event_ids)) != len(event_ids)
            ):
                raise ValueError(
                    f"mechanism {label} event ids are invalid"
                )
        fits = tuple(self.candidate_fits)
        object.__setattr__(self, "candidate_fits", fits)
        fit_ids = tuple(fit.hypothesis_id for fit in fits)
        if (
            not fits
            or len(set(fit_ids)) != len(fit_ids)
            or self.selected_hypothesis_id not in fit_ids
        ):
            raise ValueError(
                "mechanism candidate fits are invalid"
            )
        metrics = (
            self.confirmation_base_nll,
            self.confirmation_nll,
            self.confirmation_nll_uplift,
            self.confirmation_nll_uplift_standard_error,
            self.confirmation_nll_uplift_ci_lower,
            self.confirmation_nll_uplift_ci_upper,
            self.confirmation_calibration_error,
        )
        if not all(isfinite(value) for value in metrics):
            raise ValueError(
                "mechanism confirmation metrics are invalid"
            )
        if self.confirmation_nll_uplift_standard_error < 0:
            raise ValueError(
                "mechanism confirmation standard error is invalid"
            )
        if not 0 <= self.confirmation_calibration_error <= 1:
            raise ValueError(
                "mechanism confirmation calibration error is invalid"
            )
        if not np.isclose(
            self.confirmation_nll_uplift,
            self.confirmation_base_nll - self.confirmation_nll,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "mechanism confirmation uplift is inconsistent"
            )
        if (
            self.confirmation_nll_uplift_ci_lower
            > self.confirmation_nll_uplift
            or self.confirmation_nll_uplift
            > self.confirmation_nll_uplift_ci_upper
        ):
            raise ValueError(
                "mechanism confirmation interval is invalid"
            )
        if self.confirmation_standard_error_method not in {
            "iid",
            "time_ordered_hac",
        }:
            raise ValueError(
                "unsupported confirmation standard error method"
            )
        for values, label in (
            (self.supported_domains, "domain"),
            (self.supported_option_signatures, "option"),
            (self.supported_context_signatures, "context"),
        ):
            if (
                not values
                or any(not value for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    f"mechanism supported {label} scope is invalid"
                )
        selected_fit = next(
            fit
            for fit in fits
            if fit.hypothesis_id == self.selected_hypothesis_id
        )
        if (
            len(self.selected_term_minimums)
            != len(selected_fit.term_ids)
            or len(self.selected_term_maximums)
            != len(selected_fit.term_ids)
            or any(
                not isfinite(value)
                for value in (
                    self.selected_term_minimums
                    + self.selected_term_maximums
                )
            )
            or any(
                lower > upper
                for lower, upper in zip(
                    self.selected_term_minimums,
                    self.selected_term_maximums,
                    strict=True,
                )
            )
        ):
            raise ValueError(
                "mechanism selected term support is invalid"
            )
        if self.status not in {
            "supported_candidate",
            "no_supported_candidate",
        }:
            raise ValueError(
                "unsupported mechanism report status"
            )
        if (
            self.interpretation != "predictive_structure_only"
            or "causal_interpretation_not_identified"
            not in self.reasons
        ):
            raise ValueError(
                "mechanism report must preserve non-causal scope"
            )
        if self.status == "supported_candidate" and (
            "confirmation_support_not_established" in self.reasons
        ):
            raise ValueError(
                "supported mechanism report has refusal reason"
            )
        if self.status == "no_supported_candidate" and (
            "confirmation_support_not_established"
            not in self.reasons
        ):
            raise ValueError(
                "unsupported mechanism report lacks reason"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported mechanism report signature method"
            )
        expected = self.digest()
        if self.report_id:
            _require_digest(self.report_id, "report_id")
            if self.report_id != expected:
                raise ValueError(
                    "mechanism report_id does not match content"
                )
        else:
            object.__setattr__(self, "report_id", expected)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "plan_id": self.plan_id,
            "base_model_id": self.base_model_id,
            "person_id": self.person_id,
            "evaluated_at": self.evaluated_at,
            "discovery_event_ids": list(
                self.discovery_event_ids
            ),
            "discovery_data_hash": self.discovery_data_hash,
            "selection_event_ids": list(
                self.selection_event_ids
            ),
            "selection_data_hash": self.selection_data_hash,
            "confirmation_event_ids": list(
                self.confirmation_event_ids
            ),
            "confirmation_data_hash": self.confirmation_data_hash,
            "candidate_fits": [
                fit.to_dict() for fit in self.candidate_fits
            ],
            "selected_hypothesis_id": (
                self.selected_hypothesis_id
            ),
            "confirmation_base_nll": self.confirmation_base_nll,
            "confirmation_nll": self.confirmation_nll,
            "confirmation_nll_uplift": (
                self.confirmation_nll_uplift
            ),
            "confirmation_nll_uplift_standard_error": (
                self.confirmation_nll_uplift_standard_error
            ),
            "confirmation_nll_uplift_ci_lower": (
                self.confirmation_nll_uplift_ci_lower
            ),
            "confirmation_nll_uplift_ci_upper": (
                self.confirmation_nll_uplift_ci_upper
            ),
            "confirmation_calibration_error": (
                self.confirmation_calibration_error
            ),
            "confirmation_standard_error_method": (
                self.confirmation_standard_error_method
            ),
            "supported_domains": list(self.supported_domains),
            "supported_option_signatures": list(
                self.supported_option_signatures
            ),
            "supported_context_signatures": list(
                self.supported_context_signatures
            ),
            "selected_term_minimums": list(
                self.selected_term_minimums
            ),
            "selected_term_maximums": list(
                self.selected_term_maximums
            ),
            "status": self.status,
            "reasons": list(self.reasons),
            "verifier_id": self.verifier_id,
            "interpretation": self.interpretation,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "report_id": self.report_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("mechanism report is unsigned")
        authority.verify_payload(
            self.signed_payload(),
            self.verifier_id,
            self.signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class MechanismPrediction:
    scenario_id: str
    person_id: str
    probability_option_1: float
    predicted_choice: int
    base_probability_option_1: float
    probability_lower_95: float
    probability_upper_95: float
    logit_standard_deviation: float
    selected_hypothesis_id: str
    mechanism_plan_id: str
    mechanism_report_id: str
    applicability_status: str
    model_form_uncertainty_status: str = (
        "candidate_selection_not_quantified"
    )
    uncertainty_scope: str = (
        "conditional_gaussian_parameter_approximation"
    )
    interpretation: str = "predictive_structure_only"
    model_version: str = "mechanism-corrected-logit-v1"

    def __post_init__(self) -> None:
        probabilities = (
            self.probability_option_1,
            self.base_probability_option_1,
            self.probability_lower_95,
            self.probability_upper_95,
        )
        if any(
            not isfinite(value) or not 0.0 <= value <= 1.0
            for value in probabilities
        ):
            raise ValueError(
                "mechanism prediction probabilities are invalid"
            )
        if not (
            self.probability_lower_95
            <= self.probability_option_1
            <= self.probability_upper_95
        ):
            raise ValueError(
                "mechanism prediction interval is invalid"
            )
        if (
            self.predicted_choice not in (0, 1)
            or not isfinite(self.logit_standard_deviation)
            or self.logit_standard_deviation <= 0
        ):
            raise ValueError(
                "mechanism prediction values are invalid"
            )
        for value, label in (
            (self.mechanism_plan_id, "mechanism_plan_id"),
            (self.mechanism_report_id, "mechanism_report_id"),
        ):
            _require_digest(value, label)
        if (
            not self.scenario_id
            or not self.person_id
            or not self.selected_hypothesis_id
        ):
            raise ValueError(
                "mechanism prediction identity is required"
            )
        if (
            self.model_form_uncertainty_status
            != "candidate_selection_not_quantified"
            or self.uncertainty_scope
            != "conditional_gaussian_parameter_approximation"
            or self.interpretation != "predictive_structure_only"
        ):
            raise ValueError(
                "mechanism prediction must expose uncertainty scope"
            )


def _raw_term_matrix(
    records: Sequence[EventRecord],
    hypothesis: MechanismHypothesis,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            [
                term.evaluate(
                    record.observation.scenario,
                    feature_names,
                )
                for term in hypothesis.terms
            ]
            for record in records
        ],
        dtype=np.float64,
    )


def _base_logits(
    records: Sequence[EventRecord],
    bundle: PersonModelBundle,
) -> np.ndarray:
    weights = np.asarray(
        bundle.representation.latent_mean,
        dtype=np.float64,
    )
    return np.asarray(
        [
            np.asarray(
                record.observation.scenario.ordered_features(
                    bundle.representation.feature_names
                ),
                dtype=np.float64,
            )
            @ weights
            for record in records
        ],
        dtype=np.float64,
    )


def _base_feature_matrix(
    records: Sequence[EventRecord],
    bundle: PersonModelBundle,
) -> np.ndarray:
    return np.asarray(
        [
            record.observation.scenario.ordered_features(
                bundle.representation.feature_names
            )
            for record in records
        ],
        dtype=np.float64,
    )


def _base_logit_variances(
    records: Sequence[EventRecord],
    bundle: PersonModelBundle,
) -> np.ndarray:
    features = _base_feature_matrix(records, bundle)
    covariance = np.asarray(
        bundle.representation.covariance,
        dtype=np.float64,
    )
    return np.maximum(
        np.einsum(
            "ij,jk,ik->i",
            features,
            covariance,
            features,
        ),
        0.0,
    )


def _predictive_probabilities(
    logit_means: np.ndarray,
    logit_variances: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        logistic_normal_probability(
            logit_means,
            logit_variances,
        ),
        dtype=np.float64,
    )


def _choices(records: Sequence[EventRecord]) -> np.ndarray:
    return np.asarray(
        [record.observation.actual_choice for record in records],
        dtype=np.float64,
    )


def _fit_offset_logistic(
    features: np.ndarray,
    choices: np.ndarray,
    offsets: np.ndarray,
    l2_precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = features.shape[1]
    identity = np.eye(dimension, dtype=np.float64)
    weights = np.zeros(dimension, dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        logits = offsets + features @ candidate
        return float(
            np.sum(
                np.logaddexp(0.0, logits)
                - choices * logits
            )
            + 0.5 * l2_precision * (candidate @ candidate)
        )

    for _ in range(100):
        logits = offsets + features @ weights
        probabilities = np.asarray(sigmoid(logits))
        variance = np.clip(
            probabilities * (1.0 - probabilities),
            1e-8,
            None,
        )
        gradient = (
            features.T @ (probabilities - choices)
            + l2_precision * weights
        )
        hessian = (
            features.T @ (features * variance[:, None])
            + l2_precision * identity
            + 1e-9 * identity
        )
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step, ord=2) < 1e-8:
            break
        current = objective(weights)
        scale = 1.0
        accepted = False
        for _ in range(25):
            candidate = weights - scale * step
            if objective(candidate) <= (
                current + 1e-12 * max(1.0, abs(current))
            ):
                weights = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError(
                "mechanism optimizer could not find a safe step"
            )
        if np.linalg.norm(scale * step, ord=2) < 1e-8:
            break
    probabilities = np.asarray(
        sigmoid(offsets + features @ weights)
    )
    variance = np.clip(
        probabilities * (1.0 - probabilities),
        1e-8,
        None,
    )
    hessian = (
        features.T @ (features * variance[:, None])
        + l2_precision * identity
        + 1e-9 * identity
    )
    covariance = np.linalg.inv(hessian)
    return weights, covariance


def _fit_candidate(
    bundle: PersonModelBundle,
    hypothesis: MechanismHypothesis,
    discovery_records: tuple[EventRecord, ...],
    selection_records: tuple[EventRecord, ...],
    config: MechanismComparisonConfig,
) -> MechanismCandidateFit:
    feature_names = bundle.representation.feature_names
    discovery_raw = _raw_term_matrix(
        discovery_records,
        hypothesis,
        feature_names,
    )
    selection_raw = _raw_term_matrix(
        selection_records,
        hypothesis,
        feature_names,
    )
    centers = np.mean(discovery_raw, axis=0)
    scales = np.std(discovery_raw, axis=0)
    for index, term in enumerate(hypothesis.terms):
        if term.kind == "intercept":
            centers[index] = 0.0
            scales[index] = 1.0
    scales = np.where(scales < 1e-8, 1.0, scales)
    discovery_features = (discovery_raw - centers) / scales
    selection_features = (selection_raw - centers) / scales
    discovery_offsets = _base_logits(
        discovery_records,
        bundle,
    )
    selection_offsets = _base_logits(
        selection_records,
        bundle,
    )
    discovery_base_variances = _base_logit_variances(
        discovery_records,
        bundle,
    )
    selection_base_variances = _base_logit_variances(
        selection_records,
        bundle,
    )
    discovery_choices = _choices(discovery_records)
    selection_choices = _choices(selection_records)
    coefficients, covariance = _fit_offset_logistic(
        discovery_features,
        discovery_choices,
        discovery_offsets,
        config.l2_precision,
    )
    discovery_candidate_variances = (
        discovery_base_variances
        + np.einsum(
            "ij,jk,ik->i",
            discovery_features,
            covariance,
            discovery_features,
        )
    )
    selection_candidate_variances = (
        selection_base_variances
        + np.einsum(
            "ij,jk,ik->i",
            selection_features,
            covariance,
            selection_features,
        )
    )
    discovery_probabilities = _predictive_probabilities(
        discovery_offsets + discovery_features @ coefficients,
        discovery_candidate_variances,
    )
    selection_probabilities = _predictive_probabilities(
        selection_offsets + selection_features @ coefficients,
        selection_candidate_variances,
    )
    selection_base_probabilities = _predictive_probabilities(
        selection_offsets,
        selection_base_variances,
    )
    discovery_nll = float(
        np.mean(
            _binary_loss(
                discovery_probabilities,
                discovery_choices,
            )
        )
    )
    selection_nll = float(
        np.mean(
            _binary_loss(
                selection_probabilities,
                selection_choices,
            )
        )
    )
    selection_base_nll = float(
        np.mean(
            _binary_loss(
                selection_base_probabilities,
                selection_choices,
            )
        )
    )
    return MechanismCandidateFit(
        hypothesis_id=hypothesis.hypothesis_id,
        term_ids=tuple(
            term.term_id for term in hypothesis.terms
        ),
        centers=tuple(float(value) for value in centers),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(
            float(value) for value in coefficients
        ),
        covariance=tuple(
            tuple(float(value) for value in row)
            for row in covariance
        ),
        discovery_nll=discovery_nll,
        selection_nll=selection_nll,
        selection_nll_uplift=(
            selection_base_nll - selection_nll
        ),
    )


def _base_model_mechanism_eligibility_reasons(
    bundle: PersonModelBundle,
) -> tuple[str, ...]:
    validation = bundle.manifest.validation
    if validation.status == "passed":
        return ()
    if validation.status == "unvalidated":
        return ("base_model_independent_validation_required",)
    reasons = []
    if not validation.personalization_passed:
        reasons.append("base_model_personalization_not_validated")
    if (
        validation.temporal_stability_status != "stable"
        or validation.temporal_drift_detected
    ):
        reasons.append("base_model_temporal_stability_not_validated")
    mechanism_only_failure = (
        validation.personalization_passed
        and not validation.mechanism_adequacy_passed
        and validation.temporal_stability_status == "stable"
        and not validation.temporal_drift_detected
        and set(validation.reasons)
        == {"mechanism_misspecification_suspected"}
    )
    if not mechanism_only_failure and not reasons:
        reasons.append("base_model_failure_not_mechanism_repairable")
    return tuple(reasons)


def create_mechanism_comparison_plan(
    bundle: PersonModelBundle,
    hypotheses: Sequence[MechanismHypothesis],
    authority: VerificationAuthority,
    *,
    verifier_id: str,
    registered_at: str,
    discovery_window: EvidenceWindow,
    selection_window: EvidenceWindow,
    confirmation_window: EvidenceWindow,
    config: MechanismComparisonConfig | None = None,
) -> MechanismComparisonPlan:
    eligibility_reasons = _base_model_mechanism_eligibility_reasons(
        bundle
    )
    if eligibility_reasons:
        raise MechanismRefusedError(eligibility_reasons)
    parsed_registered = _parse_timestamp(
        registered_at,
        "registered_at",
    )
    valid_through = (
        bundle.manifest.applicability_profile.valid_through
    )
    if (
        valid_through is not None
        and parsed_registered
        <= _parse_timestamp(valid_through, "valid_through")
    ):
        raise MechanismRefusedError(
            ("mechanism_plan_not_after_base_evidence",)
        )
    typed_hypotheses = tuple(hypotheses)
    model_features = set(bundle.representation.feature_names)
    if any(
        name not in model_features
        for hypothesis in typed_hypotheses
        for term in hypothesis.terms
        for name in term.feature_names
    ):
        raise MechanismRefusedError(
            ("mechanism_term_feature_unknown",)
        )
    unsigned = MechanismComparisonPlan(
        base_model_id=bundle.manifest.model_id,
        person_id=bundle.manifest.person_id,
        registered_at=registered_at,
        discovery_window=discovery_window,
        selection_window=selection_window,
        confirmation_window=confirmation_window,
        hypotheses=typed_hypotheses,
        config=config or MechanismComparisonConfig(),
        verifier_id=verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        verifier_id,
    )
    return replace(unsigned, signature=signature)


def _records_in_window(
    records: tuple[EventRecord, ...],
    window: EvidenceWindow,
) -> bool:
    start = _parse_timestamp(window.start_at, "window_start")
    end = _parse_timestamp(window.end_at, "window_end")
    return all(
        start
        <= _parse_timestamp(record.observed_at, "observed_at")
        <= end
        for record in records
    )


def _verify_evidence(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
) -> tuple[
    tuple[EventRecord, ...],
    tuple[EventRecord, ...],
    tuple[EventRecord, ...],
]:
    reasons = []
    verified_ledgers = []
    labels = (
        "discovery",
        "selection",
        "confirmation",
    )
    supplied = (
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
    )
    windows = (
        plan.discovery_window,
        plan.selection_window,
        plan.confirmation_window,
    )
    for label, ledger, window in zip(
        labels,
        supplied,
        windows,
        strict=True,
    ):
        try:
            verified = EventLedger.verify(
                ledger.records,
                authority,
            )
        except ValueError:
            reasons.append(
                f"{label}_evidence_signature_invalid"
            )
            verified_ledgers.append(ledger)
            continue
        records = verified.records
        verified_ledgers.append(verified)
        if len(records) != window.expected_event_count:
            reasons.append(f"{label}_event_count_mismatch")
        if any(
            record.observation.person_id != plan.person_id
            for record in records
        ):
            reasons.append(f"{label}_person_mismatch")
        if not _records_in_window(records, window):
            reasons.append(f"{label}_outside_registered_window")
    if any(not ledger.records for ledger in verified_ledgers):
        raise MechanismRefusedError(reasons)
    ordered_records = tuple(
        tuple(
            sorted(
                ledger.records,
                key=lambda record: (
                    _parse_timestamp(
                        record.observed_at,
                        "observed_at",
                    ),
                    record.event_id,
                ),
            )
        )
        for ledger in verified_ledgers
    )
    all_records = tuple(
        record
        for records in ordered_records
        for record in records
    )
    event_ids = tuple(record.event_id for record in all_records)
    trial_hashes = tuple(
        trial_key_hash(
            record.observation.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in all_records
    )
    design_hashes = tuple(
        scenario_design_hash(
            record.observation.person_id,
            record.observation.scenario,
        )
        for record in all_records
    )
    if (
        len(set(event_ids)) != len(event_ids)
        or len(set(trial_hashes)) != len(trial_hashes)
        or len(set(design_hashes)) != len(design_hashes)
    ):
        reasons.append("mechanism_evidence_overlap")
    if set(trial_hashes) & set(
        bundle.manifest.lineage_trial_hashes
    ):
        reasons.append("mechanism_evidence_reuses_base_trial")
    if set(design_hashes) & set(
        bundle.manifest.lineage_design_hashes
    ):
        reasons.append("mechanism_evidence_reuses_base_design")
    registered = _parse_timestamp(
        plan.registered_at,
        "registered_at",
    )
    if any(
        _parse_timestamp(record.observed_at, "observed_at")
        <= registered
        for record in all_records
    ):
        reasons.append("mechanism_evidence_precedes_plan")
    discovery_records, selection_records, confirmation_records = (
        ordered_records
    )
    if (
        max(
            _parse_timestamp(record.observed_at, "observed_at")
            for record in discovery_records
        )
        >= min(
            _parse_timestamp(record.observed_at, "observed_at")
            for record in selection_records
        )
        or max(
            _parse_timestamp(record.observed_at, "observed_at")
            for record in selection_records
        )
        >= min(
            _parse_timestamp(record.observed_at, "observed_at")
            for record in confirmation_records
        )
    ):
        reasons.append("mechanism_evidence_time_order_invalid")
    for record in all_records:
        try:
            record.observation.scenario.ordered_features(
                bundle.representation.feature_names
            )
            assessment = (
                bundle.manifest.applicability_profile.assess(
                    record.observation.scenario,
                    prediction_at=record.observed_at,
                )
            )
        except ValueError:
            reasons.append(
                "mechanism_evidence_feature_schema_mismatch"
            )
            continue
        if assessment.reasons:
            reasons.append(
                "mechanism_evidence_outside_applicability"
            )
        if assessment.warnings:
            reasons.append(
                "mechanism_evidence_cross_domain_unvalidated"
            )
    if reasons:
        raise MechanismRefusedError(reasons)
    return (
        tuple(discovery_records),
        tuple(selection_records),
        tuple(confirmation_records),
    )


def compare_mechanisms(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
) -> MechanismComparisonReport:
    try:
        plan.verify(authority)
    except ValueError as error:
        raise MechanismRefusedError(
            ("mechanism_plan_signature_invalid",)
        ) from error
    reasons = []
    if plan.base_model_id != bundle.manifest.model_id:
        reasons.append("mechanism_base_model_mismatch")
    if plan.person_id != bundle.manifest.person_id:
        reasons.append("mechanism_person_mismatch")
    reasons.extend(
        _base_model_mechanism_eligibility_reasons(bundle)
    )
    model_features = set(bundle.representation.feature_names)
    if any(
        name not in model_features
        for hypothesis in plan.hypotheses
        for term in hypothesis.terms
        for name in term.feature_names
    ):
        reasons.append("mechanism_term_feature_unknown")
    if reasons:
        raise MechanismRefusedError(reasons)
    (
        discovery_records,
        selection_records,
        confirmation_records,
    ) = _verify_evidence(
        bundle,
        plan,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    try:
        fits = tuple(
            _fit_candidate(
                bundle,
                hypothesis,
                discovery_records,
                selection_records,
                plan.config,
            )
            for hypothesis in plan.hypotheses
        )
    except (RuntimeError, np.linalg.LinAlgError) as error:
        raise MechanismRefusedError(
            ("mechanism_candidate_fit_failed",)
        ) from error
    selected_fit = min(
        fits,
        key=lambda fit: (
            fit.selection_nll,
            fit.hypothesis_id,
        ),
    )
    selected_hypothesis = next(
        hypothesis
        for hypothesis in plan.hypotheses
        if hypothesis.hypothesis_id
        == selected_fit.hypothesis_id
    )
    confirmation_raw = _raw_term_matrix(
        confirmation_records,
        selected_hypothesis,
        bundle.representation.feature_names,
    )
    centers = np.asarray(selected_fit.centers)
    scales = np.asarray(selected_fit.scales)
    coefficients = np.asarray(selected_fit.coefficients)
    confirmation_features = (
        confirmation_raw - centers
    ) / scales
    confirmation_offsets = _base_logits(
        confirmation_records,
        bundle,
    )
    confirmation_base_variances = _base_logit_variances(
        confirmation_records,
        bundle,
    )
    correction_covariance = np.asarray(
        selected_fit.covariance,
        dtype=np.float64,
    )
    confirmation_candidate_variances = (
        confirmation_base_variances
        + np.einsum(
            "ij,jk,ik->i",
            confirmation_features,
            correction_covariance,
            confirmation_features,
        )
    )
    confirmation_choices = _choices(confirmation_records)
    base_probabilities = _predictive_probabilities(
        confirmation_offsets,
        confirmation_base_variances,
    )
    candidate_probabilities = _predictive_probabilities(
        confirmation_offsets
        + confirmation_features @ coefficients,
        confirmation_candidate_variances,
    )
    base_losses = _binary_loss(
        base_probabilities,
        confirmation_choices,
    )
    candidate_losses = _binary_loss(
        candidate_probabilities,
        confirmation_choices,
    )
    paired_uplift = base_losses - candidate_losses
    base_nll = float(np.mean(base_losses))
    candidate_nll = float(np.mean(candidate_losses))
    uplift = float(np.mean(paired_uplift))
    calibration_error = _expected_calibration_error(
        candidate_probabilities,
        confirmation_choices,
    )
    standard_error, standard_error_method = (
        _confirmation_standard_error(
            paired_uplift,
            confirmation_records,
        )
    )
    lower = uplift - plan.config.confidence_z * standard_error
    upper = uplift + plan.config.confidence_z * standard_error
    supported = (
        uplift
        >= plan.config.minimum_confirmation_nll_uplift
        and lower > 0
        and calibration_error
        <= plan.config.maximum_confirmation_calibration_error
    )
    report_reasons = [
        "causal_interpretation_not_identified",
        "temporal_vs_structural_not_identified",
    ]
    if not supported:
        report_reasons.append(
            "confirmation_support_not_established"
        )
    if (
        calibration_error
        > plan.config.maximum_confirmation_calibration_error
    ):
        report_reasons.append(
            "confirmation_calibration_not_established"
        )
    evaluated_at = max(
        confirmation_records,
        key=lambda record: _parse_timestamp(
            record.verified_at,
            "verified_at",
        ),
    ).verified_at
    split_records = (
        discovery_records,
        selection_records,
        confirmation_records,
    )
    supported_domains = tuple(
        sorted(
            set.intersection(
                *(
                    {
                        record.observation.scenario.domain
                        for record in records
                    }
                    for records in split_records
                )
            )
        )
    )
    supported_option_signatures = tuple(
        sorted(
            set.intersection(
                *(
                    {
                        _option_signature(
                            record.observation.scenario
                        )
                        for record in records
                    }
                    for records in split_records
                )
            )
        )
    )
    supported_context_signatures = tuple(
        sorted(
            set.intersection(
                *(
                    {
                        _context_signature(
                            record.observation.scenario
                        )
                        for record in records
                    }
                    for records in split_records
                )
            )
        )
    )
    if (
        not supported_domains
        or not supported_option_signatures
        or not supported_context_signatures
    ):
        raise MechanismRefusedError(
            ("mechanism_evidence_metadata_scope_empty",)
        )
    all_selected_terms = _raw_term_matrix(
        tuple(
            record
            for records in split_records
            for record in records
        ),
        selected_hypothesis,
        bundle.representation.feature_names,
    )
    unsigned = MechanismComparisonReport(
        plan_id=plan.plan_id,
        base_model_id=bundle.manifest.model_id,
        person_id=plan.person_id,
        evaluated_at=evaluated_at,
        discovery_event_ids=tuple(
            record.event_id for record in discovery_records
        ),
        discovery_data_hash=EventLedger.snapshot_hash(
            discovery_records
        ),
        selection_event_ids=tuple(
            record.event_id for record in selection_records
        ),
        selection_data_hash=EventLedger.snapshot_hash(
            selection_records
        ),
        confirmation_event_ids=tuple(
            record.event_id for record in confirmation_records
        ),
        confirmation_data_hash=EventLedger.snapshot_hash(
            confirmation_records
        ),
        candidate_fits=fits,
        selected_hypothesis_id=selected_fit.hypothesis_id,
        confirmation_base_nll=base_nll,
        confirmation_nll=candidate_nll,
        confirmation_nll_uplift=uplift,
        confirmation_nll_uplift_standard_error=standard_error,
        confirmation_nll_uplift_ci_lower=lower,
        confirmation_nll_uplift_ci_upper=upper,
        confirmation_calibration_error=calibration_error,
        confirmation_standard_error_method=standard_error_method,
        supported_domains=supported_domains,
        supported_option_signatures=(
            supported_option_signatures
        ),
        supported_context_signatures=(
            supported_context_signatures
        ),
        selected_term_minimums=tuple(
            float(value)
            for value in np.min(all_selected_terms, axis=0)
        ),
        selected_term_maximums=tuple(
            float(value)
            for value in np.max(all_selected_terms, axis=0)
        ),
        status=(
            "supported_candidate"
            if supported
            else "no_supported_candidate"
        ),
        reasons=tuple(report_reasons),
        verifier_id=plan.verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        plan.verifier_id,
    )
    return replace(unsigned, signature=signature)


def verify_mechanism_report(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    report: MechanismComparisonReport,
) -> None:
    try:
        report.verify(authority)
    except ValueError as error:
        raise MechanismRefusedError(
            ("mechanism_report_signature_invalid",)
        ) from error
    recomputed = compare_mechanisms(
        bundle,
        plan,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    if recomputed != report:
        raise MechanismRefusedError(
            ("mechanism_report_derivation_mismatch",)
        )


def _predict_with_verified_mechanism(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    scenario: Scenario,
    *,
    prediction_at: str,
) -> MechanismPrediction:
    if report.status != "supported_candidate":
        raise MechanismRefusedError(
            ("mechanism_candidate_not_supported",)
        )
    prediction_time = _parse_timestamp(
        prediction_at,
        "prediction_at",
    )
    if prediction_time <= _parse_timestamp(
        report.evaluated_at,
        "evaluated_at",
    ):
        raise MechanismRefusedError(
            ("mechanism_prediction_precedes_confirmation",)
        )
    report_age_days = (
        prediction_time
        - _parse_timestamp(report.evaluated_at, "evaluated_at")
    ).total_seconds() / 86400.0
    if report_age_days > plan.config.maximum_report_age_days:
        raise MechanismRefusedError(
            ("mechanism_report_expired",)
        )
    assessment = bundle.manifest.applicability_profile.assess(
        scenario,
        prediction_at=prediction_at,
    )
    if assessment.reasons:
        raise MechanismRefusedError(
            (
                "mechanism_prediction_outside_base_applicability",
                *assessment.reasons,
            )
        )
    if (
        scenario.domain not in report.supported_domains
        or _option_signature(scenario)
        not in report.supported_option_signatures
        or _context_signature(scenario)
        not in report.supported_context_signatures
    ):
        raise MechanismRefusedError(
            ("mechanism_transfer_unvalidated",)
        )
    selected_hypothesis = next(
        hypothesis
        for hypothesis in plan.hypotheses
        if hypothesis.hypothesis_id
        == report.selected_hypothesis_id
    )
    selected_fit = next(
        fit
        for fit in report.candidate_fits
        if fit.hypothesis_id
        == report.selected_hypothesis_id
    )
    raw_terms = np.asarray(
        [
            term.evaluate(
                scenario,
                bundle.representation.feature_names,
            )
            for term in selected_hypothesis.terms
        ],
        dtype=np.float64,
    )
    if any(
        value < lower - 1e-12 or value > upper + 1e-12
        for value, lower, upper in zip(
            raw_terms,
            report.selected_term_minimums,
            report.selected_term_maximums,
            strict=True,
        )
    ):
        raise MechanismRefusedError(
            ("mechanism_local_term_support_gap",)
        )
    standardized_terms = (
        raw_terms - np.asarray(selected_fit.centers)
    ) / np.asarray(selected_fit.scales)
    base_features = np.asarray(
        scenario.ordered_features(
            bundle.representation.feature_names
        ),
        dtype=np.float64,
    )
    base_weights = np.asarray(
        bundle.representation.latent_mean,
        dtype=np.float64,
    )
    coefficients = np.asarray(
        selected_fit.coefficients,
        dtype=np.float64,
    )
    logit_mean = float(
        base_features @ base_weights
        + standardized_terms @ coefficients
    )
    base_covariance = np.asarray(
        bundle.representation.covariance,
        dtype=np.float64,
    )
    correction_covariance = np.asarray(
        selected_fit.covariance,
        dtype=np.float64,
    )
    logit_variance = max(
        float(
            base_features @ base_covariance @ base_features
            + standardized_terms
            @ correction_covariance
            @ standardized_terms
        ),
        0.0,
    )
    logit_standard_deviation = sqrt(logit_variance)
    probability = float(
        _predictive_probabilities(
            np.asarray([logit_mean]),
            np.asarray([logit_variance]),
        )[0]
    )
    base_logit_mean = float(base_features @ base_weights)
    base_logit_variance = max(
        float(
            base_features
            @ base_covariance
            @ base_features
        ),
        0.0,
    )
    base_probability = float(
        _predictive_probabilities(
            np.asarray([base_logit_mean]),
            np.asarray([base_logit_variance]),
        )[0]
    )
    lower = float(
        sigmoid(
            logit_mean - 1.96 * logit_standard_deviation
        )
    )
    upper = float(
        sigmoid(
            logit_mean + 1.96 * logit_standard_deviation
        )
    )
    return MechanismPrediction(
        scenario_id=scenario.scenario_id,
        person_id=plan.person_id,
        probability_option_1=probability,
        predicted_choice=int(probability >= 0.5),
        base_probability_option_1=base_probability,
        probability_lower_95=lower,
        probability_upper_95=upper,
        logit_standard_deviation=logit_standard_deviation,
        selected_hypothesis_id=(
            report.selected_hypothesis_id
        ),
        mechanism_plan_id=plan.plan_id,
        mechanism_report_id=report.report_id,
        applicability_status=assessment.status,
    )


def predict_with_mechanism(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    scenario: Scenario,
    *,
    prediction_at: str,
) -> MechanismPrediction:
    verify_mechanism_report(
        bundle,
        plan,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
        report,
    )
    return _predict_with_verified_mechanism(
        bundle,
        plan,
        report,
        scenario,
        prediction_at=prediction_at,
    )


@dataclass
class MechanismDistiller(CognitiveModule):
    config: MechanismComparisonConfig = MechanismComparisonConfig()
    module_id: str = "preregistered-mechanism-distiller"
    module_version: str = "preregistered-mechanism-comparison-v2"

    def required_inputs(self) -> tuple[str, ...]:
        return (
            "validated_person_model",
            "preregistered_structural_hypotheses",
            "signed_discovery_ledger",
            "signed_selection_ledger",
            "signed_confirmation_ledger",
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": "implemented",
            "selection_uses_confirmation_outcomes": False,
            "causal_interpretation": "not_identified",
            "interpretation": "predictive_structure_only",
            "config": self.config.to_dict(),
        }


def _term_from_dict(data: dict[str, object]) -> MechanismTerm:
    return MechanismTerm(
        term_id=str(data["term_id"]),
        kind=str(data["kind"]),
        feature_names=tuple(
            str(value) for value in data["feature_names"]
        ),
    )


def _hypothesis_from_dict(
    data: dict[str, object],
) -> MechanismHypothesis:
    return MechanismHypothesis(
        hypothesis_id=str(data["hypothesis_id"]),
        terms=tuple(
            _term_from_dict(dict(value))
            for value in data["terms"]
        ),
    )


def _window_from_dict(data: dict[str, object]) -> EvidenceWindow:
    return EvidenceWindow(
        start_at=str(data["start_at"]),
        end_at=str(data["end_at"]),
        expected_event_count=int(data["expected_event_count"]),
    )


def _config_from_dict(
    data: dict[str, object],
) -> MechanismComparisonConfig:
    expected = set(MechanismComparisonConfig().to_dict())
    if set(data) != expected:
        raise ValueError(
            "mechanism comparison config fields do not match this version"
        )
    return MechanismComparisonConfig(
        minimum_samples_per_split=int(
            data["minimum_samples_per_split"]
        ),
        maximum_hypotheses=int(data["maximum_hypotheses"]),
        maximum_terms_per_hypothesis=int(
            data["maximum_terms_per_hypothesis"]
        ),
        l2_precision=float(data["l2_precision"]),
        minimum_confirmation_nll_uplift=float(
            data["minimum_confirmation_nll_uplift"]
        ),
        maximum_confirmation_calibration_error=float(
            data["maximum_confirmation_calibration_error"]
        ),
        confidence_z=float(data["confidence_z"]),
        maximum_report_age_days=float(
            data["maximum_report_age_days"]
        ),
        model_version=str(data["model_version"]),
    )


def mechanism_plan_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> MechanismComparisonPlan:
    if data.get("artifact_version") != "pcfm-mechanism-plan-v2":
        raise ValueError(
            "unsupported mechanism plan artifact version"
        )
    plan = MechanismComparisonPlan(
        base_model_id=str(data["base_model_id"]),
        person_id=str(data["person_id"]),
        registered_at=str(data["registered_at"]),
        discovery_window=_window_from_dict(
            dict(data["discovery_window"])
        ),
        selection_window=_window_from_dict(
            dict(data["selection_window"])
        ),
        confirmation_window=_window_from_dict(
            dict(data["confirmation_window"])
        ),
        hypotheses=tuple(
            _hypothesis_from_dict(dict(value))
            for value in data["hypotheses"]
        ),
        config=_config_from_dict(dict(data["config"])),
        verifier_id=str(data["verifier_id"]),
        artifact_version=str(data["artifact_version"]),
        plan_id=str(data["plan_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    plan.verify(authority)
    return plan


def _candidate_fit_from_dict(
    data: dict[str, object],
) -> MechanismCandidateFit:
    return MechanismCandidateFit(
        hypothesis_id=str(data["hypothesis_id"]),
        term_ids=tuple(str(value) for value in data["term_ids"]),
        centers=tuple(float(value) for value in data["centers"]),
        scales=tuple(float(value) for value in data["scales"]),
        coefficients=tuple(
            float(value) for value in data["coefficients"]
        ),
        covariance=tuple(
            tuple(float(value) for value in row)
            for row in data["covariance"]
        ),
        discovery_nll=float(data["discovery_nll"]),
        selection_nll=float(data["selection_nll"]),
        selection_nll_uplift=float(
            data["selection_nll_uplift"]
        ),
    )


def mechanism_report_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> MechanismComparisonReport:
    if data.get("artifact_version") != "pcfm-mechanism-report-v2":
        raise ValueError(
            "unsupported mechanism report artifact version"
        )
    report = MechanismComparisonReport(
        plan_id=str(data["plan_id"]),
        base_model_id=str(data["base_model_id"]),
        person_id=str(data["person_id"]),
        evaluated_at=str(data["evaluated_at"]),
        discovery_event_ids=tuple(
            str(value) for value in data["discovery_event_ids"]
        ),
        discovery_data_hash=str(data["discovery_data_hash"]),
        selection_event_ids=tuple(
            str(value) for value in data["selection_event_ids"]
        ),
        selection_data_hash=str(data["selection_data_hash"]),
        confirmation_event_ids=tuple(
            str(value)
            for value in data["confirmation_event_ids"]
        ),
        confirmation_data_hash=str(
            data["confirmation_data_hash"]
        ),
        candidate_fits=tuple(
            _candidate_fit_from_dict(dict(value))
            for value in data["candidate_fits"]
        ),
        selected_hypothesis_id=str(
            data["selected_hypothesis_id"]
        ),
        confirmation_base_nll=float(
            data["confirmation_base_nll"]
        ),
        confirmation_nll=float(data["confirmation_nll"]),
        confirmation_nll_uplift=float(
            data["confirmation_nll_uplift"]
        ),
        confirmation_nll_uplift_standard_error=float(
            data["confirmation_nll_uplift_standard_error"]
        ),
        confirmation_nll_uplift_ci_lower=float(
            data["confirmation_nll_uplift_ci_lower"]
        ),
        confirmation_nll_uplift_ci_upper=float(
            data["confirmation_nll_uplift_ci_upper"]
        ),
        confirmation_calibration_error=float(
            data["confirmation_calibration_error"]
        ),
        confirmation_standard_error_method=str(
            data["confirmation_standard_error_method"]
        ),
        supported_domains=tuple(
            str(value) for value in data["supported_domains"]
        ),
        supported_option_signatures=tuple(
            str(value)
            for value in data["supported_option_signatures"]
        ),
        supported_context_signatures=tuple(
            str(value)
            for value in data["supported_context_signatures"]
        ),
        selected_term_minimums=tuple(
            float(value)
            for value in data["selected_term_minimums"]
        ),
        selected_term_maximums=tuple(
            float(value)
            for value in data["selected_term_maximums"]
        ),
        status=str(data["status"]),
        reasons=tuple(str(value) for value in data["reasons"]),
        verifier_id=str(data["verifier_id"]),
        interpretation=str(data["interpretation"]),
        artifact_version=str(data["artifact_version"]),
        report_id=str(data["report_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    report.verify(authority)
    return report


def save_mechanism_comparison_plan(
    path: Path,
    plan: MechanismComparisonPlan,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            plan.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_mechanism_comparison_plan(
    path: Path,
    authority: VerificationAuthority,
) -> MechanismComparisonPlan:
    return mechanism_plan_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )


def save_mechanism_comparison_report(
    path: Path,
    report: MechanismComparisonReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_mechanism_comparison_report(
    path: Path,
    authority: VerificationAuthority,
) -> MechanismComparisonReport:
    return mechanism_report_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )
