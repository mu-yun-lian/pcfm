from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import isfinite, sqrt
from statistics import NormalDist

import numpy as np

from .contracts import Observation, Scenario
from .ledger import EventRecord


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


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


@dataclass(frozen=True)
class ApplicabilityProfile:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    inverse_covariance: tuple[tuple[float, ...], ...]
    squared_distance_threshold: float
    reference_features: tuple[tuple[float, ...], ...]
    local_neighbor_count: int
    local_squared_distance_threshold: float
    reference_sample_count: int
    calibration_sample_count: int
    supported_domains: tuple[str, ...]
    supported_option_signatures: tuple[str, ...]
    supported_context_signatures: tuple[str, ...]
    valid_through: str | None
    maximum_age_days: float
    calibration_safety_factor: float
    model_version: str = "hybrid-applicability-v2"

    def __post_init__(self) -> None:
        dimension = len(self.feature_names)
        if dimension == 0 or len(set(self.feature_names)) != dimension:
            raise ValueError("applicability feature names must be unique")
        if len(self.center) != dimension or not all(
            isfinite(value) for value in self.center
        ):
            raise ValueError("applicability center dimensions are invalid")
        inverse = np.asarray(self.inverse_covariance, dtype=np.float64)
        if (
            inverse.shape != (dimension, dimension)
            or not np.all(np.isfinite(inverse))
            or not np.allclose(inverse, inverse.T, atol=1e-8)
            or np.min(np.linalg.eigvalsh(inverse)) <= 0
        ):
            raise ValueError(
                "applicability inverse covariance must be positive definite"
            )
        if (
            not isfinite(self.squared_distance_threshold)
            or self.squared_distance_threshold <= 0
            or not isfinite(self.local_squared_distance_threshold)
            or self.local_squared_distance_threshold <= 0
        ):
            raise ValueError("applicability thresholds must be positive")
        if self.reference_sample_count != len(self.reference_features):
            raise ValueError(
                "applicability reference count does not match features"
            )
        if self.reference_sample_count <= dimension:
            raise ValueError("applicability profile has too few references")
        if self.calibration_sample_count <= 0:
            raise ValueError("applicability calibration sample is empty")
        if any(
            len(row) != dimension
            or not all(isfinite(value) for value in row)
            for row in self.reference_features
        ):
            raise ValueError("applicability reference features are invalid")
        if not 1 <= self.local_neighbor_count < self.reference_sample_count:
            raise ValueError("local neighbor count is invalid")
        if (
            not self.supported_domains
            or any(not domain for domain in self.supported_domains)
            or len(set(self.supported_domains)) != len(
                self.supported_domains
            )
        ):
            raise ValueError("supported domains must be non-empty and unique")
        for signatures, label in (
            (self.supported_option_signatures, "option"),
            (self.supported_context_signatures, "context"),
        ):
            if (
                not signatures
                or any(not value for value in signatures)
                or len(set(signatures)) != len(signatures)
            ):
                raise ValueError(
                    f"supported {label} signatures must be non-empty and unique"
                )
        if self.valid_through is not None:
            _parse_timestamp(self.valid_through)
        if not isfinite(self.maximum_age_days) or self.maximum_age_days <= 0:
            raise ValueError("maximum model age must be positive")
        if (
            not isfinite(self.calibration_safety_factor)
            or self.calibration_safety_factor < 1.0
        ):
            raise ValueError("calibration safety factor must be at least one")

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "inverse_covariance": [
                list(row) for row in self.inverse_covariance
            ],
            "squared_distance_threshold": self.squared_distance_threshold,
            "reference_features": [
                list(row) for row in self.reference_features
            ],
            "local_neighbor_count": self.local_neighbor_count,
            "local_squared_distance_threshold": (
                self.local_squared_distance_threshold
            ),
            "reference_sample_count": self.reference_sample_count,
            "calibration_sample_count": self.calibration_sample_count,
            "supported_domains": list(self.supported_domains),
            "supported_option_signatures": list(
                self.supported_option_signatures
            ),
            "supported_context_signatures": list(
                self.supported_context_signatures
            ),
            "valid_through": self.valid_through,
            "maximum_age_days": self.maximum_age_days,
            "calibration_safety_factor": self.calibration_safety_factor,
            "model_version": self.model_version,
        }

    def digest(self) -> str:
        rendered = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(rendered).hexdigest()

    def assess(
        self,
        scenario: Scenario,
        *,
        prediction_at: str | None = None,
    ) -> ScenarioApplicability:
        features = np.asarray(
            scenario.ordered_features(self.feature_names),
            dtype=np.float64,
        )
        center = np.asarray(self.center, dtype=np.float64)
        inverse = np.asarray(
            self.inverse_covariance,
            dtype=np.float64,
        )
        centered = features - center
        squared_distance = max(
            float(centered @ inverse @ centered),
            0.0,
        )
        references = np.asarray(
            self.reference_features,
            dtype=np.float64,
        )
        differences = references - features
        local_distances = np.einsum(
            "ij,jk,ik->i",
            differences,
            inverse,
            differences,
        )
        neighbor_index = self.local_neighbor_count - 1
        local_squared_distance = max(
            float(
                np.partition(local_distances, neighbor_index)[
                    neighbor_index
                ]
            ),
            0.0,
        )

        reasons = []
        warnings = []
        if squared_distance > self.squared_distance_threshold:
            reasons.append("feature_distribution_shift")
        if (
            local_squared_distance
            > self.local_squared_distance_threshold
        ):
            reasons.append("local_support_gap")
        if scenario.domain not in self.supported_domains:
            warnings.append("unvalidated_domain_label")
        if (
            _option_signature(scenario)
            not in self.supported_option_signatures
        ):
            warnings.append("unvalidated_option_pair")
        if (
            _context_signature(scenario)
            not in self.supported_context_signatures
        ):
            warnings.append("unvalidated_context")

        if self.valid_through is not None:
            if prediction_at is None:
                reasons.append("prediction_time_required")
            else:
                prediction_time = _parse_timestamp(prediction_at)
                reference_time = _parse_timestamp(self.valid_through)
                age_days = (
                    prediction_time - reference_time
                ).total_seconds() / 86400.0
                if age_days < 0:
                    reasons.append("prediction_precedes_reference_data")
                elif age_days > self.maximum_age_days:
                    reasons.append("stale_model")

        return ScenarioApplicability(
            status=(
                "refused"
                if reasons
                else (
                    "cross_domain_extrapolation"
                    if warnings
                    else "in_distribution"
                )
            ),
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            ood_score=squared_distance,
            ood_threshold=self.squared_distance_threshold,
            local_ood_score=local_squared_distance,
            local_ood_threshold=self.local_squared_distance_threshold,
        )


@dataclass(frozen=True)
class ScenarioApplicability:
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    ood_score: float
    ood_threshold: float
    local_ood_score: float
    local_ood_threshold: float


class PredictionRefusedError(ValueError):
    def __init__(
        self,
        reasons: Sequence[str],
        assessment: ScenarioApplicability | None = None,
    ) -> None:
        self.reasons = tuple(reasons)
        self.ood_score = (
            assessment.ood_score if assessment is not None else None
        )
        self.ood_threshold = (
            assessment.ood_threshold if assessment is not None else None
        )
        self.local_ood_score = (
            assessment.local_ood_score
            if assessment is not None
            else None
        )
        self.local_ood_threshold = (
            assessment.local_ood_threshold
            if assessment is not None
            else None
        )
        super().__init__("prediction refused: " + ", ".join(self.reasons))


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile, method="higher"))


def fit_applicability_profile(
    observations: Sequence[Observation],
    feature_names: Sequence[str],
    *,
    valid_through: str | None = None,
    maximum_age_days: float = 180.0,
    calibration_safety_factor: float = 1.25,
) -> ApplicabilityProfile:
    names = tuple(feature_names)
    minimum_references = max(50, 5 * len(names))
    if len(observations) < minimum_references:
        raise ValueError(
            f"at least {minimum_references} applicability references "
            "are required"
        )
    all_features = np.asarray(
        [
            observation.scenario.ordered_features(names)
            for observation in observations
        ],
        dtype=np.float64,
    )
    stable_order = sorted(
        range(len(observations)),
        key=lambda index: hashlib.sha256(
            (
                observations[index].person_id
                + "\0"
                + observations[index].scenario.scenario_id
            ).encode("utf-8")
        ).digest(),
    )
    calibration_count = len(stable_order) // 2
    calibration_indices = stable_order[:calibration_count]
    reference_indices = stable_order[calibration_count:]
    features = all_features[reference_indices]
    calibration_features = all_features[calibration_indices]
    center = np.mean(features, axis=0)
    covariance = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.95 * covariance + 0.05 * diagonal
    regularization = max(
        float(np.median(np.diag(covariance))) * 1e-6,
        1e-8,
    )
    covariance = covariance + np.eye(len(names)) * regularization
    inverse = np.linalg.inv(covariance)
    calibration_centered = calibration_features - center
    distances = np.einsum(
        "ij,jk,ik->i",
        calibration_centered,
        inverse,
        calibration_centered,
    )
    dimension = len(names)
    normal_quantile_999 = 3.090232306167813
    chi_square_999 = dimension * (
        1.0
        - 2.0 / (9.0 * dimension)
        + normal_quantile_999 * sqrt(2.0 / (9.0 * dimension))
    ) ** 3
    empirical_995 = _higher_quantile(distances, 0.995)
    global_threshold = (
        max(chi_square_999, empirical_995)
        * calibration_safety_factor
    )

    neighbor_count = min(5, len(features) - 1)
    pairwise_differences = (
        calibration_features[:, None, :] - features[None, :, :]
    )
    pairwise_distances = np.einsum(
        "ijk,kl,ijl->ij",
        pairwise_differences,
        inverse,
        pairwise_differences,
    )
    local_reference_scores = np.partition(
        pairwise_distances,
        neighbor_count - 1,
        axis=1,
    )[:, neighbor_count - 1]
    local_threshold = _higher_quantile(
        local_reference_scores,
        0.995,
    )
    local_threshold = (
        max(local_threshold, 1e-8)
        * calibration_safety_factor
    )

    return ApplicabilityProfile(
        feature_names=names,
        center=tuple(float(value) for value in center),
        inverse_covariance=tuple(
            tuple(float(value) for value in row)
            for row in inverse
        ),
        squared_distance_threshold=float(global_threshold),
        reference_features=tuple(
            tuple(float(value) for value in row)
            for row in features
        ),
        local_neighbor_count=neighbor_count,
        local_squared_distance_threshold=float(local_threshold),
        reference_sample_count=len(features),
        calibration_sample_count=len(calibration_features),
        supported_domains=tuple(
            sorted(
                {
                    observation.scenario.domain
                    for observation in observations
                }
            )
        ),
        supported_option_signatures=tuple(
            sorted({_option_signature(o.scenario) for o in observations})
        ),
        supported_context_signatures=tuple(
            sorted({_context_signature(o.scenario) for o in observations})
        ),
        valid_through=valid_through,
        maximum_age_days=float(maximum_age_days),
        calibration_safety_factor=float(calibration_safety_factor),
    )


@dataclass(frozen=True)
class TemporalStability:
    status: str
    drift_detected: bool
    maximum_score_z: float | None
    maximum_score_effect: float | None
    critical_score_z: float | None
    early_nll: float | None
    late_nll: float | None
    early_sample_count: int
    late_sample_count: int

    def __post_init__(self) -> None:
        if self.status not in {"stable", "unstable", "not_assessed"}:
            raise ValueError("unsupported temporal stability status")
        if self.drift_detected != (self.status == "unstable"):
            raise ValueError("temporal stability status is contradictory")


def _not_assessed_temporal() -> TemporalStability:
    return TemporalStability(
        status="not_assessed",
        drift_detected=False,
        maximum_score_z=None,
        maximum_score_effect=None,
        critical_score_z=None,
        early_nll=None,
        late_nll=None,
        early_sample_count=0,
        late_sample_count=0,
    )


def _log_loss(
    probabilities: np.ndarray,
    choices: np.ndarray,
) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return -float(
        np.mean(
            choices * np.log(clipped)
            + (1.0 - choices) * np.log(1.0 - clipped)
        )
    )


def _two_sample_score_test(
    scores: np.ndarray,
    split: int,
) -> tuple[float, float]:
    early = scores[:split]
    late = scores[split:]
    difference = np.mean(late, axis=0) - np.mean(early, axis=0)
    standard_error = np.sqrt(
        np.var(early, axis=0, ddof=1) / len(early)
        + np.var(late, axis=0, ddof=1) / len(late)
    )
    z_scores = np.divide(
        np.abs(difference),
        standard_error,
        out=np.zeros_like(difference),
        where=standard_error > 1e-12,
    )
    deterministic_nonzero = (
        (standard_error <= 1e-12)
        & (np.abs(difference) > 1e-12)
    )
    z_scores[deterministic_nonzero] = 1e6
    return (
        min(float(np.max(z_scores)), 1e6),
        float(np.max(np.abs(difference))),
    )


def assess_temporal_stability(
    records: Sequence[EventRecord],
    probabilities: Sequence[float],
    feature_names: Sequence[str],
    *,
    minimum_window_samples: int = 50,
    familywise_error_rate: float = 0.05,
    minimum_score_effect: float = 0.05,
) -> TemporalStability:
    if len(records) != len(probabilities):
        raise ValueError("temporal predictions must align with records")
    if len(records) < 2 * minimum_window_samples:
        return _not_assessed_temporal()
    ordered = sorted(
        zip(records, probabilities, strict=True),
        key=lambda item: _parse_timestamp(item[0].observed_at),
    )
    timestamps = [
        _parse_timestamp(record.observed_at) for record, _ in ordered
    ]
    split_candidates = [
        index
        for index in range(
            minimum_window_samples,
            len(ordered) - minimum_window_samples + 1,
        )
        if timestamps[index - 1] < timestamps[index]
    ]
    if not split_candidates:
        return _not_assessed_temporal()

    ordered_records = [record for record, _ in ordered]
    ordered_probabilities = np.asarray(
        [probability for _, probability in ordered],
        dtype=np.float64,
    )
    choices = np.asarray(
        [record.observation.actual_choice for record in ordered_records],
        dtype=np.float64,
    )
    features = np.asarray(
        [
            record.observation.scenario.ordered_features(feature_names)
            for record in ordered_records
        ],
        dtype=np.float64,
    )
    scale = np.std(features, axis=0, ddof=1)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (features - np.mean(features, axis=0)) / scale
    design = np.column_stack(
        [np.ones(len(standardized), dtype=np.float64), standardized]
    )
    scores = (
        choices - ordered_probabilities
    )[:, None] * design

    test_count = len(split_candidates) * design.shape[1]
    adjusted_tail = familywise_error_rate / (2.0 * test_count)
    critical_z = max(
        3.3,
        NormalDist().inv_cdf(1.0 - adjusted_tail),
    )
    candidates = []
    for split in split_candidates:
        z_value, effect = _two_sample_score_test(scores, split)
        candidates.append((z_value, effect, split))
    observed_z, observed_effect, best_split = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    drift_detected = (
        observed_z > critical_z
        and observed_effect > minimum_score_effect
    )
    return TemporalStability(
        status="unstable" if drift_detected else "stable",
        drift_detected=drift_detected,
        maximum_score_z=observed_z,
        maximum_score_effect=observed_effect,
        critical_score_z=critical_z,
        early_nll=_log_loss(
            ordered_probabilities[:best_split],
            choices[:best_split],
        ),
        late_nll=_log_loss(
            ordered_probabilities[best_split:],
            choices[best_split:],
        ),
        early_sample_count=best_split,
        late_sample_count=len(ordered) - best_split,
    )
