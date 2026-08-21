from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .contracts import EvaluationReport, Observation
from .math_utils import sigmoid


ProbabilityFunction = Callable[[Observation], float]


def evaluate_predictions(
    observations: Sequence[Observation],
    probability_for: ProbabilityFunction,
    *,
    calibration_bins: int = 10,
) -> EvaluationReport:
    probabilities = np.asarray(
        [
            min(max(float(probability_for(observation)), 1e-9), 1.0 - 1e-9)
            for observation in observations
        ],
        dtype=np.float64,
    )
    return evaluate_probability_array(
        observations,
        probabilities,
        calibration_bins=calibration_bins,
    )


def evaluate_probability_array(
    observations: Sequence[Observation],
    probabilities: Sequence[float],
    *,
    calibration_bins: int = 10,
) -> EvaluationReport:
    if not observations:
        raise ValueError("at least one observation is required")
    if len(observations) != len(probabilities):
        raise ValueError("predictions must align with observations")
    probabilities_array = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-9,
        1.0 - 1e-9,
    )
    choices = np.asarray(
        [observation.actual_choice for observation in observations],
        dtype=np.float64,
    )
    nll = -float(
        np.mean(
            choices * np.log(probabilities_array)
            + (1.0 - choices) * np.log(1.0 - probabilities_array)
        )
    )
    brier = float(np.mean((probabilities_array - choices) ** 2))
    accuracy = float(np.mean((probabilities_array >= 0.5) == choices))

    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for index in range(calibration_bins):
        lower = edges[index]
        upper = edges[index + 1]
        if index == calibration_bins - 1:
            mask = (
                (probabilities_array >= lower)
                & (probabilities_array <= upper)
            )
        else:
            mask = (
                (probabilities_array >= lower)
                & (probabilities_array < upper)
            )
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(probabilities_array[mask]))
            - float(np.mean(choices[mask]))
        )

    return EvaluationReport(
        sample_count=len(observations),
        negative_log_likelihood=nll,
        brier_score=brier,
        accuracy=accuracy,
        expected_calibration_error=ece,
    )


def aggregate_reports(reports: Sequence[EvaluationReport]) -> EvaluationReport:
    if not reports:
        raise ValueError("at least one report is required")
    total = sum(report.sample_count for report in reports)

    def weighted(attribute: str) -> float:
        return sum(
            report.sample_count * float(getattr(report, attribute))
            for report in reports
        ) / total

    return EvaluationReport(
        sample_count=total,
        negative_log_likelihood=weighted("negative_log_likelihood"),
        brier_score=weighted("brier_score"),
        accuracy=weighted("accuracy"),
        expected_calibration_error=weighted("expected_calibration_error"),
    )


def report_to_dict(report: EvaluationReport) -> dict[str, float | int]:
    return {
        "sample_count": report.sample_count,
        "negative_log_likelihood": report.negative_log_likelihood,
        "brier_score": report.brier_score,
        "accuracy": report.accuracy,
        "expected_calibration_error": report.expected_calibration_error,
    }


def _binary_log_loss(
    probabilities: np.ndarray,
    choices: np.ndarray,
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return -(
        choices * np.log(clipped)
        + (1.0 - choices) * np.log(1.0 - clipped)
    )


def _nonlinear_features(
    observations: Sequence[Observation],
    feature_names: Sequence[str],
) -> np.ndarray:
    features = np.asarray(
        [
            observation.scenario.ordered_features(feature_names)
            for observation in observations
        ],
        dtype=np.float64,
    )
    columns = [features**2, np.abs(features)]
    interactions = [
        features[:, left] * features[:, right]
        for left in range(features.shape[1])
        for right in range(left + 1, features.shape[1])
    ]
    if interactions:
        columns.append(np.column_stack(interactions))
    return np.column_stack(columns)


def _fit_offset_logistic(
    features: np.ndarray,
    choices: np.ndarray,
    offsets: np.ndarray,
    *,
    l2_precision: float = 2.0,
) -> np.ndarray:
    weights = np.zeros(features.shape[1], dtype=np.float64)
    identity = np.eye(features.shape[1], dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        logits = offsets + features @ candidate
        return float(
            np.sum(np.logaddexp(0.0, logits) - choices * logits)
            + 0.5 * l2_precision * candidate @ candidate
        )

    for _ in range(100):
        probabilities = np.asarray(sigmoid(offsets + features @ weights))
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
        if np.linalg.norm(step) < 1e-8:
            break
        current = objective(weights)
        scale = 1.0
        for _ in range(25):
            candidate = weights - scale * step
            if objective(candidate) <= current + 1e-12 * max(1.0, current):
                weights = candidate
                break
            scale *= 0.5
        else:
            break
    return weights


def cross_fitted_mechanism_probe(
    observations: Sequence[Observation],
    base_probabilities: Sequence[float],
    feature_names: Sequence[str],
) -> float | None:
    if len(observations) < 20:
        return None
    nonlinear = _nonlinear_features(observations, feature_names)
    choices = np.asarray(
        [observation.actual_choice for observation in observations],
        dtype=np.float64,
    )
    base = np.clip(
        np.asarray(base_probabilities, dtype=np.float64),
        1e-6,
        1.0 - 1e-6,
    )
    offsets = np.log(base / (1.0 - base))
    corrected = np.empty_like(base)
    indices = np.arange(len(observations))
    for fold in (0, 1):
        test_mask = indices % 2 == fold
        train_mask = ~test_mask
        train_mean = np.mean(nonlinear[train_mask], axis=0)
        train_scale = np.std(nonlinear[train_mask], axis=0)
        train_scale = np.where(train_scale < 1e-8, 1.0, train_scale)
        standardized = (nonlinear - train_mean) / train_scale
        weights = _fit_offset_logistic(
            standardized[train_mask],
            choices[train_mask],
            offsets[train_mask],
        )
        corrected[test_mask] = np.asarray(
            sigmoid(
                offsets[test_mask]
                + standardized[test_mask] @ weights
            )
        )
    base_nll = float(np.mean(_binary_log_loss(base, choices)))
    corrected_nll = float(np.mean(_binary_log_loss(corrected, choices)))
    return base_nll - corrected_nll


def assess_person_validation(
    observations: Sequence[Observation],
    personal_probabilities: Sequence[float],
    population_probabilities: Sequence[float],
    feature_names: Sequence[str],
    *,
    minimum_nll_uplift: float = 0.01,
    maximum_calibration_error: float = 0.15,
    minimum_samples: int = 100,
    maximum_mechanism_probe_uplift: float = 0.01,
) -> dict[str, object]:
    if len(observations) != len(personal_probabilities) or len(
        observations
    ) != len(population_probabilities):
        raise ValueError("validation predictions must align with observations")
    personal_report = evaluate_probability_array(
        observations,
        personal_probabilities,
    )
    population_report = evaluate_probability_array(
        observations,
        population_probabilities,
    )
    choices = np.asarray(
        [observation.actual_choice for observation in observations],
        dtype=np.float64,
    )
    personal = np.asarray(personal_probabilities, dtype=np.float64)
    population = np.asarray(population_probabilities, dtype=np.float64)
    paired_uplift = (
        _binary_log_loss(population, choices)
        - _binary_log_loss(personal, choices)
    )
    uplift = float(np.mean(paired_uplift))
    if len(paired_uplift) > 1:
        standard_error = float(
            np.std(paired_uplift, ddof=1)
            / np.sqrt(len(paired_uplift))
        )
    else:
        standard_error = float("inf")
    ci_lower = uplift - 1.96 * standard_error
    ci_upper = uplift + 1.96 * standard_error
    mechanism_probe_uplift = cross_fitted_mechanism_probe(
        observations,
        personal_probabilities,
        feature_names,
    )

    personalization_reasons = []
    if len(observations) < minimum_samples:
        personalization_reasons.append("insufficient_person_validation_samples")
    if uplift < minimum_nll_uplift:
        personalization_reasons.append("insufficient_personalization_uplift")
    if ci_lower <= 0:
        personalization_reasons.append("personalization_uplift_not_significant")
    if (
        personal_report.expected_calibration_error
        > maximum_calibration_error
    ):
        personalization_reasons.append("calibration_error_too_high")

    mechanism_reasons = []
    if mechanism_probe_uplift is None:
        # 样本不足（<20）无法评估机制充分性 → 标 not_assessed，不冒充通过。
        mechanism_adequacy_passed: bool | None = None
    elif mechanism_probe_uplift > maximum_mechanism_probe_uplift:
        mechanism_reasons.append("mechanism_misspecification_suspected")
        mechanism_adequacy_passed = False
    else:
        mechanism_adequacy_passed = True

    return {
        "personal_report": personal_report,
        "population_report": population_report,
        "nll_uplift": uplift,
        "nll_uplift_ci_lower": ci_lower,
        "nll_uplift_ci_upper": ci_upper,
        "personalization_passed": not personalization_reasons,
        "personalization_reasons": personalization_reasons,
        "mechanism_probe_nll_uplift": mechanism_probe_uplift,
        "mechanism_adequacy_passed": mechanism_adequacy_passed,
        "mechanism_reasons": mechanism_reasons,
        "passed": not personalization_reasons and not mechanism_reasons,
    }
