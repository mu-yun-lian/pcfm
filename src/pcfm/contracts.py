from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Sequence


def _finite_tuple(values: Sequence[float], label: str) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError(f"{label} must not be empty")
    if not all(isfinite(value) for value in converted):
        raise ValueError(f"{label} must contain only finite values")
    return converted


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    features: tuple[float, ...]
    feature_names: tuple[str, ...]
    options: tuple[str, str] = ("A", "B")
    domain: str = "structured_choice"
    context: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        features = _finite_tuple(self.features, "features")
        names = tuple(str(name) for name in self.feature_names)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", names)
        if len(names) != len(features):
            raise ValueError("feature_names and features dimensions must match")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("feature_names must be non-empty and unique")
        options = tuple(str(value) for value in self.options)
        domain = str(self.domain)
        context = {
            str(name): str(value)
            for name, value in self.context.items()
        }
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "context", context)
        if (
            len(options) != 2
            or not all(options)
            or options[0] == options[1]
        ):
            raise ValueError("exactly two distinct options are required")
        if not domain:
            raise ValueError("scenario domain is required")
        if any(not name for name in context):
            raise ValueError("scenario context names must be non-empty")

    def ordered_features(self, expected_names: Sequence[str]) -> tuple[float, ...]:
        expected = tuple(expected_names)
        if set(self.feature_names) != set(expected) or len(expected) != len(
            self.feature_names
        ):
            raise ValueError("scenario feature schema does not match the model")
        by_name = dict(zip(self.feature_names, self.features, strict=True))
        return tuple(by_name[name] for name in expected)


@dataclass(frozen=True)
class Observation:
    person_id: str
    scenario: Scenario
    actual_choice: int
    confidence: float | None = None
    reaction_time_ms: float | None = None
    provenance: str = "human_record"

    def __post_init__(self) -> None:
        if not self.person_id:
            raise ValueError("person_id is required")
        if self.actual_choice not in (0, 1):
            raise ValueError("actual_choice must be 0 or 1")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.reaction_time_ms is not None and self.reaction_time_ms <= 0:
            raise ValueError("reaction_time_ms must be positive")
        if not self.provenance:
            raise ValueError("provenance is required")


@dataclass(frozen=True)
class PersonRepresentation:
    person_id: str
    latent_mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    representation_version: str
    observation_count: int
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = _finite_tuple(self.latent_mean, "latent_mean")
        object.__setattr__(self, "latent_mean", mean)
        if self.observation_count <= 0:
            raise ValueError("observation_count must be positive")
        if len(self.feature_names) != len(mean):
            raise ValueError("feature_names and latent_mean dimensions must match")
        if len(self.covariance) != len(mean):
            raise ValueError("covariance dimensions must match latent_mean")
        for row in self.covariance:
            if len(row) != len(mean) or not all(isfinite(float(value)) for value in row):
                raise ValueError("covariance must be a finite square matrix")


@dataclass(frozen=True)
class PersonalAdapter:
    person_id: str
    delta_weights: tuple[float, ...]
    adapter_version: str
    representation_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_weights",
            _finite_tuple(self.delta_weights, "delta_weights"),
        )


@dataclass(frozen=True)
class Prediction:
    scenario_id: str
    person_id: str
    probability_option_1: float
    predicted_choice: int
    active_modules: tuple[str, ...]
    model_version: str
    probability_lower_95: float | None = None
    probability_upper_95: float | None = None
    logit_standard_deviation: float | None = None
    applicability_status: str = "not_assessed"
    applicability_warnings: tuple[str, ...] = ()
    ood_score: float | None = None
    ood_threshold: float | None = None
    local_ood_score: float | None = None
    local_ood_threshold: float | None = None
    model_form_uncertainty_status: str = "not_assessed"
    validation_status: str = "not_assessed"
    gate_overrides: tuple[str, ...] = ()
    dynamic_state_status: str = "not_assessed"
    dynamic_state_mean: float | None = None
    dynamic_state_standard_deviation: float | None = None
    dynamic_state_reference_time: str | None = None
    dynamic_state_artifact_id: str | None = None
    dynamic_state_current_evidence_status: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability_option_1 <= 1.0:
            raise ValueError("probability_option_1 must be between 0 and 1")
        if self.predicted_choice not in (0, 1):
            raise ValueError("predicted_choice must be 0 or 1")
        interval = (self.probability_lower_95, self.probability_upper_95)
        if any(value is not None for value in interval):
            if any(value is None for value in interval):
                raise ValueError("probability interval must have both bounds")
            lower, upper = interval
            if not 0.0 <= lower <= upper <= 1.0:
                raise ValueError("probability interval bounds are invalid")
        if (
            self.logit_standard_deviation is not None
            and self.logit_standard_deviation < 0
        ):
            raise ValueError("logit_standard_deviation must be non-negative")
        if self.applicability_status not in {
            "not_assessed",
            "in_distribution",
            "cross_domain_extrapolation",
            "overridden",
        }:
            raise ValueError("unsupported prediction applicability status")
        if any(not warning for warning in self.applicability_warnings):
            raise ValueError("applicability warnings must be non-empty strings")
        if (
            self.applicability_status == "cross_domain_extrapolation"
            and not self.applicability_warnings
        ):
            raise ValueError(
                "cross-domain prediction requires an applicability warning"
            )
        if (self.ood_score is None) != (self.ood_threshold is None):
            raise ValueError("OOD score and threshold must be reported together")
        if self.ood_score is not None and (
            not isfinite(self.ood_score)
            or not isfinite(self.ood_threshold)
            or self.ood_score < 0
            or self.ood_threshold <= 0
        ):
            raise ValueError("prediction OOD diagnostics are invalid")
        if (self.local_ood_score is None) != (
            self.local_ood_threshold is None
        ):
            raise ValueError(
                "local OOD score and threshold must be reported together"
            )
        if self.local_ood_score is not None and (
            not isfinite(self.local_ood_score)
            or not isfinite(self.local_ood_threshold)
            or self.local_ood_score < 0
            or self.local_ood_threshold <= 0
        ):
            raise ValueError("prediction local OOD diagnostics are invalid")
        if (
            self.applicability_status != "not_assessed"
            and (
                self.ood_score is None
                or self.local_ood_score is None
            )
        ):
            raise ValueError("assessed prediction requires OOD diagnostics")
        if self.model_form_uncertainty_status not in {
            "not_assessed",
            "within_supported_metadata",
            "unquantified_extrapolation",
            "unquantified_override",
        }:
            raise ValueError("unsupported model-form uncertainty status")
        if self.validation_status not in {
            "not_assessed",
            "passed",
            "failed",
            "unvalidated",
        }:
            raise ValueError("unsupported prediction validation status")
        if any(not value for value in self.gate_overrides):
            raise ValueError("gate overrides must be non-empty strings")
        if (
            self.applicability_status == "cross_domain_extrapolation"
            and not self.gate_overrides
            and self.model_form_uncertainty_status
            != "unquantified_extrapolation"
        ):
            raise ValueError(
                "cross-domain prediction must expose model-form uncertainty"
            )
        if (
            self.gate_overrides
            and self.model_form_uncertainty_status
            != "unquantified_override"
        ):
            raise ValueError(
                "overridden prediction must expose override uncertainty"
            )
        if (
            not self.gate_overrides
            and self.model_form_uncertainty_status
            == "unquantified_override"
        ):
            raise ValueError(
                "override uncertainty requires an explicit gate override"
            )
        if self.dynamic_state_status not in {
            "not_assessed",
            "prequential_residual_signal",
            "overridden",
        }:
            raise ValueError("unsupported dynamic state status")
        dynamic_values = (
            self.dynamic_state_mean,
            self.dynamic_state_standard_deviation,
            self.dynamic_state_reference_time,
            self.dynamic_state_artifact_id,
            self.dynamic_state_current_evidence_status,
        )
        if self.dynamic_state_status == "not_assessed":
            if any(value is not None for value in dynamic_values):
                raise ValueError(
                    "unassessed dynamic state cannot expose state values"
                )
        else:
            if any(value is None for value in dynamic_values):
                raise ValueError(
                    "assessed dynamic state requires complete state values"
                )
            if (
                not isfinite(float(self.dynamic_state_mean))
                or not isfinite(
                    float(self.dynamic_state_standard_deviation)
                )
                or float(self.dynamic_state_standard_deviation) <= 0
                or not self.dynamic_state_reference_time
                or not self.dynamic_state_artifact_id
            ):
                raise ValueError("dynamic state values are invalid")
            if self.dynamic_state_current_evidence_status not in {
                "no_detectable_shift",
                "latent_shift_detected",
            }:
                raise ValueError(
                    "unsupported current dynamic state evidence"
                )
        if (
            self.dynamic_state_status == "overridden"
            and self.model_form_uncertainty_status
            != "unquantified_override"
        ):
            raise ValueError(
                "overridden dynamic state requires override uncertainty"
            )


@dataclass(frozen=True)
class EvaluationReport:
    sample_count: int
    negative_log_likelihood: float
    brier_score: float
    accuracy: float
    expected_calibration_error: float
