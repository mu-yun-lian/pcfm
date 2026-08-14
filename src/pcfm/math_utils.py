from __future__ import annotations

from math import pi

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def sigmoid(values: FloatArray | float) -> FloatArray | float:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    if np.isscalar(values):
        return float(result)
    return result


def logistic_normal_probability(
    logit_means: FloatArray | float,
    logit_variances: FloatArray | float,
) -> FloatArray | float:
    means = np.asarray(logit_means, dtype=np.float64)
    variances = np.asarray(logit_variances, dtype=np.float64)
    if (
        not np.all(np.isfinite(means))
        or not np.all(np.isfinite(variances))
    ):
        raise ValueError("logit moments must be finite")
    probability = sigmoid(
        means
        / np.sqrt(
            1.0 + pi * np.maximum(variances, 0.0) / 8.0
        )
    )
    if np.isscalar(logit_means) and np.isscalar(logit_variances):
        return float(probability)
    return np.asarray(probability, dtype=np.float64)


def _negative_log_posterior(
    features: FloatArray,
    choices: FloatArray,
    weights: FloatArray,
    prior_mean: FloatArray,
    prior_precision: FloatArray,
) -> float:
    logits = features @ weights
    difference = weights - prior_mean
    return float(
        np.sum(np.logaddexp(0.0, logits) - choices * logits)
        + 0.5 * difference @ prior_precision @ difference
    )


def fit_map_logistic(
    features: FloatArray,
    choices: FloatArray,
    prior_mean: FloatArray,
    prior_precision: FloatArray,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[FloatArray, FloatArray]:
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    if choices.ndim != 1 or choices.shape[0] != features.shape[0]:
        raise ValueError("choices must be a vector aligned with features")
    if prior_mean.shape != (features.shape[1],):
        raise ValueError("prior_mean has the wrong dimension")
    if prior_precision.shape != (features.shape[1], features.shape[1]):
        raise ValueError("prior_precision has the wrong dimension")

    weights = prior_mean.astype(np.float64, copy=True)
    identity = np.eye(features.shape[1], dtype=np.float64)

    for _ in range(max_iterations):
        probabilities = np.asarray(sigmoid(features @ weights))
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = (
            features.T @ (probabilities - choices)
            + prior_precision @ (weights - prior_mean)
        )
        hessian = features.T @ (features * variance[:, None]) + prior_precision
        hessian = hessian + identity * 1e-9
        newton_step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(newton_step, ord=2) < tolerance:
            break
        current_objective = _negative_log_posterior(
            features,
            choices,
            weights,
            prior_mean,
            prior_precision,
        )
        step_scale = 1.0
        accepted = False
        for _ in range(25):
            candidate = weights - step_scale * newton_step
            candidate_objective = _negative_log_posterior(
                features,
                choices,
                candidate,
                prior_mean,
                prior_precision,
            )
            numerical_tolerance = 1e-12 * max(1.0, abs(current_objective))
            if np.isfinite(candidate_objective) and candidate_objective <= (
                current_objective + numerical_tolerance
            ):
                weights = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            raise RuntimeError("MAP logistic optimizer could not find a safe step")
        if np.linalg.norm(step_scale * newton_step, ord=2) < tolerance:
            break

    probabilities = np.asarray(sigmoid(features @ weights))
    variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
    hessian = features.T @ (features * variance[:, None]) + prior_precision
    covariance = np.linalg.inv(hessian + identity * 1e-9)
    return weights, covariance
