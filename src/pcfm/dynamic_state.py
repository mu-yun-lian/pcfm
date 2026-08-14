from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from math import exp, isfinite, log, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import Prediction, Scenario
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
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error


def _binary_loss(probability: float, choice: int) -> float:
    clipped = min(max(probability, 1e-9), 1.0 - 1e-9)
    return -(
        choice * log(clipped)
        + (1 - choice) * log(1.0 - clipped)
    )


def _logistic_normal_probability(
    logit_mean: float,
    logit_variance: float,
) -> float:
    return float(
        logistic_normal_probability(
            logit_mean,
            logit_variance,
        )
    )


def _hac_mean_interval(
    values: np.ndarray,
) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("-inf"), float("inf")
    centered = values - mean
    gamma_zero = float(centered @ centered / len(values))
    maximum_lag = min(
        len(values) - 1,
        max(1, int(round(len(values) ** (1.0 / 3.0)))),
    )
    long_run_variance = gamma_zero
    for lag in range(1, maximum_lag + 1):
        covariance = float(
            centered[lag:] @ centered[:-lag] / len(values)
        )
        weight = 1.0 - lag / (maximum_lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    conservative_variance = max(
        long_run_variance,
        0.25 * gamma_zero,
        1e-12,
    )
    standard_error = sqrt(conservative_variance / len(values))
    return (
        mean,
        mean - 1.96 * standard_error,
        mean + 1.96 * standard_error,
    )


def _state_evidence_status(
    mean: float,
    variance: float,
    config: DynamicStateConfig,
    *,
    event_count: int,
) -> str:
    if event_count < config.minimum_state_evidence_events:
        return "insufficient_evidence"
    standard_deviation = sqrt(variance)
    lower = mean - config.evidence_z * standard_deviation
    upper = mean + config.evidence_z * standard_deviation
    if (
        abs(mean) >= config.minimum_effect
        and (lower > 0.0 or upper < 0.0)
    ):
        return "latent_shift_detected"
    return "no_detectable_shift"


@dataclass(frozen=True)
class DynamicStateConfig:
    half_life_days: float = 14.0
    stationary_variance: float = 0.5
    initial_variance: float = 0.25
    minimum_samples: int = 80
    minimum_state_evidence_events: int = 20
    minimum_effect: float = 0.25
    evidence_z: float = 2.576
    minimum_consecutive_detections: int = 4
    minimum_nll_uplift: float = 0.005
    sequential_alpha: float = 0.05
    maximum_prediction_gap_days: float = 14.0
    model_version: str = "continuous-time-logit-state-v2"

    def __post_init__(self) -> None:
        positive = (
            self.half_life_days,
            self.stationary_variance,
            self.initial_variance,
            self.minimum_effect,
            self.evidence_z,
            self.maximum_prediction_gap_days,
        )
        if any(not isfinite(value) or value <= 0 for value in positive):
            raise ValueError("dynamic state scales must be positive")
        if self.minimum_samples < 20:
            raise ValueError(
                "dynamic state minimum_samples must be at least 20"
            )
        if not 1 <= self.minimum_state_evidence_events:
            raise ValueError(
                "minimum_state_evidence_events must be positive"
            )
        if not 2 <= self.minimum_consecutive_detections:
            raise ValueError(
                "minimum_consecutive_detections must be at least two"
            )
        if (
            not isfinite(self.minimum_nll_uplift)
            or self.minimum_nll_uplift < 0
        ):
            raise ValueError("minimum_nll_uplift must be non-negative")
        if (
            not isfinite(self.sequential_alpha)
            or not 0.0 < self.sequential_alpha < 1.0
        ):
            raise ValueError(
                "sequential_alpha must be between zero and one"
            )
        if not self.model_version:
            raise ValueError("dynamic state model_version is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "half_life_days": self.half_life_days,
            "stationary_variance": self.stationary_variance,
            "initial_variance": self.initial_variance,
            "minimum_samples": self.minimum_samples,
            "minimum_state_evidence_events": (
                self.minimum_state_evidence_events
            ),
            "minimum_effect": self.minimum_effect,
            "evidence_z": self.evidence_z,
            "minimum_consecutive_detections": (
                self.minimum_consecutive_detections
            ),
            "minimum_nll_uplift": self.minimum_nll_uplift,
            "sequential_alpha": self.sequential_alpha,
            "maximum_prediction_gap_days": (
                self.maximum_prediction_gap_days
            ),
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class DynamicStatePlan:
    base_model_id: str
    person_id: str
    registered_at: str
    monitoring_start_at: str
    monitoring_end_at: str
    expected_event_count: int
    config: DynamicStateConfig
    verifier_id: str
    plan_version: str = "pcfm-dynamic-state-plan-v1"
    plan_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _require_digest(self.base_model_id, "base_model_id")
        if not self.person_id or not self.verifier_id:
            raise ValueError("dynamic state plan identity is required")
        registered = _parse_timestamp(
            self.registered_at,
            "registered_at",
        )
        start = _parse_timestamp(
            self.monitoring_start_at,
            "monitoring_start_at",
        )
        end = _parse_timestamp(
            self.monitoring_end_at,
            "monitoring_end_at",
        )
        if not registered < start <= end:
            raise ValueError(
                "dynamic state plan must be registered before its "
                "monitoring window"
            )
        if self.expected_event_count <= 0:
            raise ValueError(
                "dynamic state expected_event_count must be positive"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported dynamic state plan signature method"
            )
        expected_id = self.digest()
        if self.plan_id:
            _require_digest(self.plan_id, "plan_id")
            if self.plan_id != expected_id:
                raise ValueError(
                    "dynamic state plan_id does not match content"
                )
        else:
            object.__setattr__(self, "plan_id", expected_id)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "plan_version": self.plan_version,
            "base_model_id": self.base_model_id,
            "person_id": self.person_id,
            "registered_at": self.registered_at,
            "monitoring_start_at": self.monitoring_start_at,
            "monitoring_end_at": self.monitoring_end_at,
            "expected_event_count": self.expected_event_count,
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
            raise ValueError("dynamic state plan is unsigned")
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
class DynamicStatePoint:
    event_id: str
    observed_at: str
    actual_choice: int
    static_probability: float
    prequential_probability: float
    prior_state_mean: float
    prior_state_variance: float
    posterior_state_mean: float
    posterior_state_variance: float
    posterior_lower: float
    posterior_upper: float
    evidence_status: str

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("dynamic state point event_id is required")
        _parse_timestamp(self.observed_at, "observed_at")
        if self.actual_choice not in (0, 1):
            raise ValueError(
                "dynamic state point actual_choice must be binary"
            )
        if not (
            0.0 <= self.static_probability <= 1.0
            and 0.0 <= self.prequential_probability <= 1.0
        ):
            raise ValueError("dynamic state probabilities are invalid")
        numeric = (
            self.prior_state_mean,
            self.prior_state_variance,
            self.posterior_state_mean,
            self.posterior_state_variance,
            self.posterior_lower,
            self.posterior_upper,
        )
        if not all(isfinite(value) for value in numeric):
            raise ValueError("dynamic state point values must be finite")
        if (
            self.prior_state_variance <= 0
            or self.posterior_state_variance <= 0
            or self.posterior_lower > self.posterior_upper
        ):
            raise ValueError("dynamic state point variance is invalid")
        if self.evidence_status not in {
            "insufficient_evidence",
            "no_detectable_shift",
            "latent_shift_detected",
        }:
            raise ValueError("unsupported dynamic state evidence status")

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "observed_at": self.observed_at,
            "actual_choice": self.actual_choice,
            "static_probability": self.static_probability,
            "prequential_probability": self.prequential_probability,
            "prior_state_mean": self.prior_state_mean,
            "prior_state_variance": self.prior_state_variance,
            "posterior_state_mean": self.posterior_state_mean,
            "posterior_state_variance": self.posterior_state_variance,
            "posterior_lower": self.posterior_lower,
            "posterior_upper": self.posterior_upper,
            "evidence_status": self.evidence_status,
        }


@dataclass(frozen=True)
class DynamicStateReport:
    base_model_id: str
    person_id: str
    plan_id: str
    evidence_event_ids: tuple[str, ...]
    evidence_data_hash: str
    config: DynamicStateConfig
    points: tuple[DynamicStatePoint, ...]
    status: str
    reasons: tuple[str, ...]
    static_nll: float
    dynamic_prequential_nll: float
    nll_uplift: float
    nll_uplift_ci_lower: float
    nll_uplift_ci_upper: float
    maximum_detection_run: int
    final_log_e_value: float
    maximum_log_e_value: float
    config_status: str
    interpretation_status: str
    last_observed_at: str
    verifier_id: str
    artifact_version: str = "pcfm-dynamic-state-v2"
    artifact_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _require_digest(self.base_model_id, "base_model_id")
        _require_digest(self.plan_id, "plan_id")
        _require_digest(self.evidence_data_hash, "evidence_data_hash")
        if (
            not self.person_id
            or not self.points
            or not self.verifier_id
        ):
            raise ValueError("dynamic state report identity is required")
        if len(self.evidence_event_ids) != len(self.points):
            raise ValueError(
                "dynamic state event ids must align with points"
            )
        if self.evidence_event_ids != tuple(
            point.event_id for point in self.points
        ):
            raise ValueError(
                "dynamic state event ids and points differ"
            )
        if len(set(self.evidence_event_ids)) != len(
            self.evidence_event_ids
        ):
            raise ValueError("dynamic state event ids must be unique")
        if self.status not in {
            "not_assessed",
            "no_prequential_residual_signal",
            "prequential_residual_signal",
        }:
            raise ValueError("unsupported dynamic state report status")
        if self.status == "not_assessed":
            if "insufficient_dynamic_state_samples" not in self.reasons:
                raise ValueError(
                    "unassessed dynamic state requires a sample reason"
                )
        elif self.reasons:
            raise ValueError(
                "assessed dynamic state report cannot contain reasons"
            )
        metrics = (
            self.static_nll,
            self.dynamic_prequential_nll,
            self.nll_uplift,
            self.nll_uplift_ci_lower,
            self.nll_uplift_ci_upper,
            self.final_log_e_value,
            self.maximum_log_e_value,
        )
        if not all(isfinite(value) for value in metrics):
            raise ValueError("dynamic state report metrics must be finite")
        if (
            self.static_nll < 0
            or self.dynamic_prequential_nll < 0
            or self.nll_uplift_ci_lower > self.nll_uplift_ci_upper
            or self.maximum_detection_run < 0
            or self.maximum_log_e_value < 0
        ):
            raise ValueError("dynamic state report metrics are invalid")
        if self.interpretation_status not in {
            "no_causal_interpretation",
            "unidentified_latent_shift",
        }:
            raise ValueError(
                "unsupported dynamic state interpretation status"
            )
        if (
            self.status == "prequential_residual_signal"
            and self.interpretation_status
            != "unidentified_latent_shift"
        ):
            raise ValueError(
                "validated state must remain causally unidentified"
            )
        if self.config_status != "signed_preregistered":
            raise ValueError(
                "unsupported dynamic state config status"
            )
        if self.last_observed_at != self.points[-1].observed_at:
            raise ValueError(
                "dynamic state last observation does not match points"
            )
        if not self.artifact_version:
            raise ValueError("dynamic state artifact version is required")
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported dynamic state report signature method"
            )
        self._validate_derived_metrics()
        expected = self.digest()
        if self.artifact_id:
            _require_digest(self.artifact_id, "artifact_id")
            if self.artifact_id != expected:
                raise ValueError(
                    "dynamic state artifact_id does not match content"
                )
        else:
            object.__setattr__(self, "artifact_id", expected)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _validate_derived_metrics(self) -> None:
        static_losses = np.asarray(
            [
                _binary_loss(
                    point.static_probability,
                    point.actual_choice,
                )
                for point in self.points
            ],
            dtype=np.float64,
        )
        dynamic_losses = np.asarray(
            [
                _binary_loss(
                    point.prequential_probability,
                    point.actual_choice,
                )
                for point in self.points
            ],
            dtype=np.float64,
        )
        paired = static_losses - dynamic_losses
        uplift, ci_lower, ci_upper = _hac_mean_interval(paired)
        cumulative = np.cumsum(paired)
        final_log_e = float(cumulative[-1])
        maximum_log_e = max(0.0, float(np.max(cumulative)))
        current_run = 0
        maximum_run = 0
        previous_direction = 0
        for point in self.points:
            direction = (
                1
                if (
                    point.evidence_status
                    == "latent_shift_detected"
                    and point.posterior_state_mean > 0
                )
                else (
                    -1
                    if point.evidence_status
                    == "latent_shift_detected"
                    else 0
                )
            )
            if direction and direction == previous_direction:
                current_run += 1
            elif direction:
                current_run = 1
            else:
                current_run = 0
            previous_direction = direction
            maximum_run = max(maximum_run, current_run)
        expected_values = (
            (self.static_nll, float(np.mean(static_losses))),
            (
                self.dynamic_prequential_nll,
                float(np.mean(dynamic_losses)),
            ),
            (self.nll_uplift, uplift),
            (self.nll_uplift_ci_lower, ci_lower),
            (self.nll_uplift_ci_upper, ci_upper),
            (self.final_log_e_value, final_log_e),
            (self.maximum_log_e_value, maximum_log_e),
        )
        if any(
            not np.isclose(actual, expected, rtol=1e-10, atol=1e-12)
            for actual, expected in expected_values
        ) or self.maximum_detection_run != maximum_run:
            raise ValueError(
                "dynamic state derived metrics do not match points"
            )
        enough_samples = len(self.points) >= self.config.minimum_samples
        signal = (
            enough_samples
            and self.nll_uplift
            >= self.config.minimum_nll_uplift
            and self.maximum_log_e_value
            >= -log(self.config.sequential_alpha)
            and self.maximum_detection_run
            >= self.config.minimum_consecutive_detections
        )
        expected_status = (
            "not_assessed"
            if not enough_samples
            else (
                "prequential_residual_signal"
                if signal
                else "no_prequential_residual_signal"
            )
        )
        if self.status != expected_status:
            raise ValueError(
                "dynamic state status contradicts derived evidence"
            )

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "base_model_id": self.base_model_id,
            "person_id": self.person_id,
            "plan_id": self.plan_id,
            "evidence_event_ids": list(self.evidence_event_ids),
            "evidence_data_hash": self.evidence_data_hash,
            "config": self.config.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "status": self.status,
            "reasons": list(self.reasons),
            "static_nll": self.static_nll,
            "dynamic_prequential_nll": (
                self.dynamic_prequential_nll
            ),
            "nll_uplift": self.nll_uplift,
            "nll_uplift_ci_lower": self.nll_uplift_ci_lower,
            "nll_uplift_ci_upper": self.nll_uplift_ci_upper,
            "maximum_detection_run": self.maximum_detection_run,
            "final_log_e_value": self.final_log_e_value,
            "maximum_log_e_value": self.maximum_log_e_value,
            "config_status": self.config_status,
            "interpretation_status": self.interpretation_status,
            "last_observed_at": self.last_observed_at,
            "verifier_id": self.verifier_id,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "artifact_id": self.artifact_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("dynamic state report is unsigned")
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


class DynamicStateRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__(
            "dynamic state refused: " + ", ".join(self.reasons)
        )


@dataclass
class DynamicStateTracker(CognitiveModule):
    config: DynamicStateConfig = DynamicStateConfig()
    module_id: str = "continuous-time-logit-state-tracker"
    module_version: str = "continuous-time-logit-state-v2"

    def required_inputs(self) -> tuple[str, ...]:
        return (
            "validated_person_model",
            "signed_preregistered_state_plan",
            "signed_future_event_ledger",
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": "implemented",
            "causal_interpretation": "not_identified",
            "config": self.config.to_dict(),
        }

    def infer(
        self,
        bundle: PersonModelBundle,
        ledger: EventLedger,
        authority: VerificationAuthority,
        plan: DynamicStatePlan,
    ) -> DynamicStateReport:
        ledger = EventLedger.verify(ledger.records, authority)
        try:
            plan.verify(authority)
        except ValueError as error:
            raise DynamicStateRefusedError(
                ("dynamic_state_plan_signature_invalid",)
            ) from error
        reasons = []
        if plan.base_model_id != bundle.manifest.model_id:
            reasons.append("dynamic_state_plan_model_mismatch")
        if plan.person_id != bundle.manifest.person_id:
            reasons.append("dynamic_state_plan_person_mismatch")
        if plan.config != self.config:
            reasons.append("dynamic_state_plan_config_mismatch")
        validation_status = bundle.manifest.validation.status
        if validation_status != "passed":
            reasons.append(
                f"base_model_validation_{validation_status}"
            )
        person_records = ledger.records_for_person(
            bundle.manifest.person_id
        )
        if len(person_records) != len(ledger.records):
            reasons.append("state_evidence_wrong_person")
        used_event_ids = (
            set(bundle.manifest.population_event_ids)
            | set(bundle.manifest.person_event_ids)
            | set(bundle.manifest.applicability_event_ids)
            | set(bundle.manifest.validation.validation_event_ids)
        )
        if used_event_ids & {
            record.event_id for record in ledger.records
        }:
            reasons.append("state_evidence_reuses_model_event")
        if set(bundle.manifest.lineage_trial_hashes) & {
            trial_key_hash(
                record.observation.person_id,
                record.observation.scenario.scenario_id,
            )
            for record in ledger.records
        }:
            reasons.append("state_evidence_reuses_model_scenario")
        if set(bundle.manifest.lineage_design_hashes) & {
            scenario_design_hash(
                record.observation.person_id,
                record.observation.scenario,
            )
            for record in ledger.records
        }:
            reasons.append("state_evidence_reuses_model_design")
        if reasons:
            raise DynamicStateRefusedError(reasons)

        ordered = tuple(
            sorted(
                person_records,
                key=lambda record: _parse_timestamp(
                    record.observed_at,
                    "observed_at",
                ),
            )
        )
        timestamps = tuple(
            _parse_timestamp(record.observed_at, "observed_at")
            for record in ordered
        )
        if len(ordered) != plan.expected_event_count:
            raise DynamicStateRefusedError(
                ("state_evidence_count_differs_from_plan",)
            )
        if (
            ordered[0].observed_at != plan.monitoring_start_at
            or ordered[-1].observed_at
            != plan.monitoring_end_at
        ):
            raise DynamicStateRefusedError(
                ("state_evidence_window_differs_from_plan",)
            )
        if any(
            right <= left
            for left, right in zip(
                timestamps,
                timestamps[1:],
                strict=False,
            )
        ):
            raise DynamicStateRefusedError(
                ("state_evidence_timestamps_not_strictly_increasing",)
            )
        reference_time = _parse_timestamp(
            bundle.manifest.applicability_profile.valid_through
            or bundle.manifest.training_cutoff,
            "base_reference_time",
        )
        if any(timestamp <= reference_time for timestamp in timestamps):
            raise DynamicStateRefusedError(
                ("state_evidence_precedes_base_reference",)
            )

        for record in ordered:
            assessment = (
                bundle.manifest.applicability_profile.assess(
                    record.observation.scenario,
                    prediction_at=record.observed_at,
                )
            )
            if assessment.reasons or assessment.warnings:
                raise DynamicStateRefusedError(
                    ("state_evidence_outside_applicability",)
                )

        stable_weights = np.asarray(
            bundle.population_model.weights,
            dtype=np.float64,
        ) + np.asarray(
            bundle.adapter.delta_weights,
            dtype=np.float64,
        )
        parameter_covariance = np.asarray(
            bundle.representation.covariance,
            dtype=np.float64,
        )
        points = []
        static_losses = []
        dynamic_losses = []
        posterior_mean = 0.0
        posterior_variance = self.config.initial_variance
        previous_time: datetime | None = None
        current_run = 0
        maximum_run = 0
        previous_direction = 0

        for index, (record, timestamp) in enumerate(
            zip(ordered, timestamps, strict=True)
        ):
            features = np.asarray(
                record.observation.scenario.ordered_features(
                    bundle.population_model.feature_names
                ),
                dtype=np.float64,
            )
            base_logit = float(features @ stable_weights)
            parameter_variance = max(
                float(
                    features
                    @ parameter_covariance
                    @ features
                ),
                0.0,
            )
            if previous_time is None:
                elapsed_days = (
                    timestamp - reference_time
                ).total_seconds() / 86400.0
                persistence = exp(
                    -log(2.0)
                    * elapsed_days
                    / self.config.half_life_days
                )
                prior_mean = 0.0
                prior_variance = (
                    persistence**2
                    * self.config.initial_variance
                    + self.config.stationary_variance
                    * (1.0 - persistence**2)
                )
            else:
                elapsed_days = (
                    timestamp - previous_time
                ).total_seconds() / 86400.0
                persistence = exp(
                    -log(2.0)
                    * elapsed_days
                    / self.config.half_life_days
                )
                prior_mean = persistence * posterior_mean
                prior_variance = (
                    persistence**2 * posterior_variance
                    + self.config.stationary_variance
                    * (1.0 - persistence**2)
                )
            static_probability = _logistic_normal_probability(
                base_logit,
                parameter_variance,
            )
            prequential_probability = _logistic_normal_probability(
                base_logit + prior_mean,
                parameter_variance + prior_variance,
            )
            choice = record.observation.actual_choice
            static_losses.append(
                _binary_loss(static_probability, choice)
            )
            dynamic_losses.append(
                _binary_loss(prequential_probability, choice)
            )

            parameter_scale = sqrt(
                1.0 + np.pi * parameter_variance / 8.0
            )
            state = prior_mean
            for _ in range(30):
                probability = float(
                    sigmoid((base_logit + state) / parameter_scale)
                )
                gradient = (
                    (probability - choice) / parameter_scale
                    + (state - prior_mean) / prior_variance
                )
                hessian = (
                    probability
                    * (1.0 - probability)
                    / parameter_scale**2
                    + 1.0 / prior_variance
                )
                step = gradient / hessian
                state -= step
                if abs(step) < 1e-10:
                    break
            posterior_mean = float(state)
            posterior_probability = float(
                sigmoid((base_logit + posterior_mean) / parameter_scale)
            )
            posterior_hessian = (
                posterior_probability
                * (1.0 - posterior_probability)
                / parameter_scale**2
                + 1.0 / prior_variance
            )
            posterior_variance = float(1.0 / posterior_hessian)
            posterior_standard_deviation = sqrt(
                posterior_variance
            )
            lower = (
                posterior_mean
                - self.config.evidence_z
                * posterior_standard_deviation
            )
            upper = (
                posterior_mean
                + self.config.evidence_z
                * posterior_standard_deviation
            )
            evidence_status = _state_evidence_status(
                posterior_mean,
                posterior_variance,
                self.config,
                event_count=index + 1,
            )
            direction = (
                1
                if (
                    evidence_status == "latent_shift_detected"
                    and posterior_mean > 0
                )
                else (
                    -1
                    if evidence_status == "latent_shift_detected"
                    else 0
                )
            )
            if direction and direction == previous_direction:
                current_run += 1
            elif direction:
                current_run = 1
            else:
                current_run = 0
            previous_direction = direction
            maximum_run = max(maximum_run, current_run)
            points.append(
                DynamicStatePoint(
                    event_id=record.event_id,
                    observed_at=record.observed_at,
                    actual_choice=choice,
                    static_probability=static_probability,
                    prequential_probability=prequential_probability,
                    prior_state_mean=prior_mean,
                    prior_state_variance=prior_variance,
                    posterior_state_mean=posterior_mean,
                    posterior_state_variance=posterior_variance,
                    posterior_lower=lower,
                    posterior_upper=upper,
                    evidence_status=evidence_status,
                )
            )
            previous_time = timestamp

        paired_uplift = np.asarray(
            static_losses,
            dtype=np.float64,
        ) - np.asarray(dynamic_losses, dtype=np.float64)
        uplift, ci_lower, ci_upper = _hac_mean_interval(
            paired_uplift
        )
        cumulative_log_e = np.cumsum(paired_uplift)
        final_log_e = float(cumulative_log_e[-1])
        maximum_log_e = max(
            0.0,
            float(np.max(cumulative_log_e)),
        )
        enough_samples = len(points) >= self.config.minimum_samples
        residual_signal = (
            enough_samples
            and uplift >= self.config.minimum_nll_uplift
            and maximum_log_e
            >= -log(self.config.sequential_alpha)
            and maximum_run
            >= self.config.minimum_consecutive_detections
        )
        if not enough_samples:
            status = "not_assessed"
            report_reasons = (
                "insufficient_dynamic_state_samples",
            )
        elif residual_signal:
            status = "prequential_residual_signal"
            report_reasons = ()
        else:
            status = "no_prequential_residual_signal"
            report_reasons = ()
        interpretation = (
            "unidentified_latent_shift"
            if residual_signal
            else "no_causal_interpretation"
        )
        unsigned = DynamicStateReport(
            base_model_id=bundle.manifest.model_id,
            person_id=bundle.manifest.person_id,
            plan_id=plan.plan_id,
            evidence_event_ids=tuple(
                point.event_id for point in points
            ),
            evidence_data_hash=EventLedger.snapshot_hash(ordered),
            config=self.config,
            points=tuple(points),
            status=status,
            reasons=report_reasons,
            static_nll=float(np.mean(static_losses)),
            dynamic_prequential_nll=float(
                np.mean(dynamic_losses)
            ),
            nll_uplift=uplift,
            nll_uplift_ci_lower=ci_lower,
            nll_uplift_ci_upper=ci_upper,
            maximum_detection_run=maximum_run,
            final_log_e_value=final_log_e,
            maximum_log_e_value=maximum_log_e,
            config_status="signed_preregistered",
            interpretation_status=interpretation,
            last_observed_at=points[-1].observed_at,
            verifier_id=plan.verifier_id,
        )
        signature = authority.sign_payload(
            unsigned.signed_payload(),
            plan.verifier_id,
        )
        return replace(unsigned, signature=signature)


def create_dynamic_state_plan(
    bundle: PersonModelBundle,
    authority: VerificationAuthority,
    *,
    verifier_id: str,
    registered_at: str,
    monitoring_start_at: str,
    monitoring_end_at: str,
    expected_event_count: int,
    config: DynamicStateConfig | None = None,
) -> DynamicStatePlan:
    unsigned = DynamicStatePlan(
        base_model_id=bundle.manifest.model_id,
        person_id=bundle.manifest.person_id,
        registered_at=registered_at,
        monitoring_start_at=monitoring_start_at,
        monitoring_end_at=monitoring_end_at,
        expected_event_count=expected_event_count,
        config=config or DynamicStateConfig(),
        verifier_id=verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        verifier_id,
    )
    return replace(unsigned, signature=signature)


def infer_dynamic_state(
    bundle: PersonModelBundle,
    ledger: EventLedger,
    authority: VerificationAuthority,
    plan: DynamicStatePlan,
) -> DynamicStateReport:
    return DynamicStateTracker(
        config=plan.config
    ).infer(bundle, ledger, authority, plan)


def _propagate_state(
    report: DynamicStateReport,
    prediction_at: str,
) -> tuple[float, float]:
    prediction_time = _parse_timestamp(
        prediction_at,
        "prediction_at",
    )
    reference_time = _parse_timestamp(
        report.last_observed_at,
        "last_observed_at",
    )
    elapsed_days = (
        prediction_time - reference_time
    ).total_seconds() / 86400.0
    if elapsed_days <= 0:
        raise DynamicStateRefusedError(
            ("prediction_not_after_state_evidence",)
        )
    if elapsed_days > report.config.maximum_prediction_gap_days:
        raise DynamicStateRefusedError(("stale_dynamic_state",))
    persistence = exp(
        -log(2.0)
        * elapsed_days
        / report.config.half_life_days
    )
    latest = report.points[-1]
    mean = persistence * latest.posterior_state_mean
    variance = (
        persistence**2 * latest.posterior_state_variance
        + report.config.stationary_variance
        * (1.0 - persistence**2)
    )
    return float(mean), float(variance)


def verify_dynamic_state_report(
    bundle: PersonModelBundle,
    report: DynamicStateReport,
    evidence_ledger: EventLedger,
    authority: VerificationAuthority,
    plan: DynamicStatePlan,
) -> None:
    try:
        plan.verify(authority)
        report.verify(authority)
    except ValueError as error:
        raise DynamicStateRefusedError(
            ("dynamic_state_signature_invalid",)
        ) from error
    if report.plan_id != plan.plan_id:
        raise DynamicStateRefusedError(
            ("dynamic_state_plan_mismatch",)
        )
    recomputed = infer_dynamic_state(
        bundle,
        evidence_ledger,
        authority,
        plan,
    )
    if recomputed != report:
        raise DynamicStateRefusedError(
            ("dynamic_state_derivation_mismatch",)
        )


def predict_with_dynamic_state(
    bundle: PersonModelBundle,
    report: DynamicStateReport,
    scenario: Scenario,
    authority: VerificationAuthority,
    plan: DynamicStatePlan,
    evidence_ledger: EventLedger,
    *,
    prediction_at: str,
    validation_override: bool = False,
    applicability_override: bool = False,
    state_override: bool = False,
) -> Prediction:
    verify_dynamic_state_report(
        bundle,
        report,
        evidence_ledger,
        authority,
        plan,
    )
    if report.base_model_id != bundle.manifest.model_id:
        raise DynamicStateRefusedError(
            ("dynamic_state_base_model_mismatch",)
        )
    if report.person_id != bundle.manifest.person_id:
        raise DynamicStateRefusedError(
            ("dynamic_state_person_mismatch",)
        )
    state_reason = f"dynamic_state_{report.status}"
    if (
        report.status != "prequential_residual_signal"
        and not state_override
    ):
        raise DynamicStateRefusedError((state_reason,))
    from .workflow import predict_with_bundle

    base = predict_with_bundle(
        bundle,
        scenario,
        prediction_at=prediction_at,
        validation_override=validation_override,
        applicability_override=applicability_override,
    )
    transfer_reason = "dynamic_state_transfer_unvalidated"
    if (
        base.applicability_status != "in_distribution"
        and not state_override
    ):
        raise DynamicStateRefusedError((transfer_reason,))
    state_mean, state_variance = _propagate_state(
        report,
        prediction_at,
    )
    current_evidence_status = _state_evidence_status(
        state_mean,
        state_variance,
        report.config,
        event_count=len(report.points),
    )
    features = np.asarray(
        scenario.ordered_features(
            bundle.population_model.feature_names
        ),
        dtype=np.float64,
    )
    stable_weights = np.asarray(
        bundle.population_model.weights,
        dtype=np.float64,
    ) + np.asarray(
        bundle.adapter.delta_weights,
        dtype=np.float64,
    )
    parameter_covariance = np.asarray(
        bundle.representation.covariance,
        dtype=np.float64,
    )
    base_logit = float(features @ stable_weights)
    parameter_variance = max(
        float(features @ parameter_covariance @ features),
        0.0,
    )
    total_variance = parameter_variance + state_variance
    probability = _logistic_normal_probability(
        base_logit + state_mean,
        total_variance,
    )
    overrides = list(base.gate_overrides)
    if (
        state_override
        and report.status != "prequential_residual_signal"
    ):
        overrides.append(state_reason)
    if (
        state_override
        and base.applicability_status != "in_distribution"
    ):
        overrides.append(transfer_reason)
    any_override = bool(overrides)
    interval_allowed = (
        base.probability_lower_95 is not None
        and not any_override
    )
    standard_deviation = sqrt(total_variance)
    lower = (
        float(
            sigmoid(
                base_logit
                + state_mean
                - 1.96 * standard_deviation
            )
        )
        if interval_allowed
        else None
    )
    upper = (
        float(
            sigmoid(
                base_logit
                + state_mean
                + 1.96 * standard_deviation
            )
        )
        if interval_allowed
        else None
    )
    return replace(
        base,
        probability_option_1=probability,
        predicted_choice=int(probability >= 0.5),
        active_modules=base.active_modules + ("dynamic_state",),
        probability_lower_95=lower,
        probability_upper_95=upper,
        logit_standard_deviation=standard_deviation,
        model_form_uncertainty_status=(
            "unquantified_override"
            if any_override
            else base.model_form_uncertainty_status
        ),
        gate_overrides=tuple(overrides),
        dynamic_state_status=(
            "overridden"
            if state_override
            and (
                report.status != "prequential_residual_signal"
                or base.applicability_status != "in_distribution"
            )
            else "prequential_residual_signal"
        ),
        dynamic_state_mean=state_mean,
        dynamic_state_standard_deviation=sqrt(state_variance),
        dynamic_state_reference_time=report.last_observed_at,
        dynamic_state_artifact_id=report.artifact_id,
        dynamic_state_current_evidence_status=(
            current_evidence_status
        ),
    )


def dynamic_state_report_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> DynamicStateReport:
    if data.get("artifact_version") != "pcfm-dynamic-state-v2":
        raise ValueError("unsupported dynamic state artifact version")
    config = dynamic_state_config_from_dict(dict(data["config"]))
    points = tuple(
        DynamicStatePoint(
            event_id=str(item["event_id"]),
            observed_at=str(item["observed_at"]),
            actual_choice=int(item["actual_choice"]),
            static_probability=float(item["static_probability"]),
            prequential_probability=float(
                item["prequential_probability"]
            ),
            prior_state_mean=float(item["prior_state_mean"]),
            prior_state_variance=float(
                item["prior_state_variance"]
            ),
            posterior_state_mean=float(
                item["posterior_state_mean"]
            ),
            posterior_state_variance=float(
                item["posterior_state_variance"]
            ),
            posterior_lower=float(item["posterior_lower"]),
            posterior_upper=float(item["posterior_upper"]),
            evidence_status=str(item["evidence_status"]),
        )
        for item in (dict(value) for value in data["points"])
    )
    report = DynamicStateReport(
        base_model_id=str(data["base_model_id"]),
        person_id=str(data["person_id"]),
        plan_id=str(data["plan_id"]),
        evidence_event_ids=tuple(
            str(value) for value in data["evidence_event_ids"]
        ),
        evidence_data_hash=str(data["evidence_data_hash"]),
        config=config,
        points=points,
        status=str(data["status"]),
        reasons=tuple(str(value) for value in data["reasons"]),
        static_nll=float(data["static_nll"]),
        dynamic_prequential_nll=float(
            data["dynamic_prequential_nll"]
        ),
        nll_uplift=float(data["nll_uplift"]),
        nll_uplift_ci_lower=float(data["nll_uplift_ci_lower"]),
        nll_uplift_ci_upper=float(data["nll_uplift_ci_upper"]),
        maximum_detection_run=int(data["maximum_detection_run"]),
        final_log_e_value=float(data["final_log_e_value"]),
        maximum_log_e_value=float(data["maximum_log_e_value"]),
        config_status=str(data["config_status"]),
        interpretation_status=str(data["interpretation_status"]),
        last_observed_at=str(data["last_observed_at"]),
        verifier_id=str(data["verifier_id"]),
        artifact_version=str(data["artifact_version"]),
        artifact_id=str(data["artifact_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    report.verify(authority)
    return report


def dynamic_state_config_from_dict(
    config_data: dict[str, object],
) -> DynamicStateConfig:
    return DynamicStateConfig(
        half_life_days=float(config_data["half_life_days"]),
        stationary_variance=float(
            config_data["stationary_variance"]
        ),
        initial_variance=float(config_data["initial_variance"]),
        minimum_samples=int(config_data["minimum_samples"]),
        minimum_state_evidence_events=int(
            config_data["minimum_state_evidence_events"]
        ),
        minimum_effect=float(config_data["minimum_effect"]),
        evidence_z=float(config_data["evidence_z"]),
        minimum_consecutive_detections=int(
            config_data["minimum_consecutive_detections"]
        ),
        minimum_nll_uplift=float(
            config_data["minimum_nll_uplift"]
        ),
        sequential_alpha=float(config_data["sequential_alpha"]),
        maximum_prediction_gap_days=float(
            config_data["maximum_prediction_gap_days"]
        ),
        model_version=str(config_data["model_version"]),
    )


def save_dynamic_state_report(
    path: Path,
    report: DynamicStateReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def dynamic_state_plan_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> DynamicStatePlan:
    if data.get("plan_version") != "pcfm-dynamic-state-plan-v1":
        raise ValueError("unsupported dynamic state plan version")
    plan = DynamicStatePlan(
        base_model_id=str(data["base_model_id"]),
        person_id=str(data["person_id"]),
        registered_at=str(data["registered_at"]),
        monitoring_start_at=str(data["monitoring_start_at"]),
        monitoring_end_at=str(data["monitoring_end_at"]),
        expected_event_count=int(data["expected_event_count"]),
        config=dynamic_state_config_from_dict(dict(data["config"])),
        verifier_id=str(data["verifier_id"]),
        plan_version=str(data["plan_version"]),
        plan_id=str(data["plan_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    plan.verify(authority)
    return plan


def save_dynamic_state_plan(
    path: Path,
    plan: DynamicStatePlan,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            plan.to_dict(),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_dynamic_state_plan(
    path: Path,
    authority: VerificationAuthority,
) -> DynamicStatePlan:
    return dynamic_state_plan_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )


def load_dynamic_state_report(
    path: Path,
    authority: VerificationAuthority,
) -> DynamicStateReport:
    return dynamic_state_report_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )
