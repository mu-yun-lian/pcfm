from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from math import exp, isfinite, log
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcfm.math_utils import logistic_normal_probability, sigmoid


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _as_matrix(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must be a finite matrix")
    return matrix


def _as_choices(value: object, count: int) -> np.ndarray:
    choices = np.asarray(value, dtype=np.float64)
    if choices.shape != (count,) or not np.all(np.isin(choices, (0.0, 1.0))):
        raise ValueError("choices must be a binary vector")
    return choices


def negative_log_likelihood(
    choices: object,
    probabilities: object,
) -> float:
    y = np.asarray(choices, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("choices and probabilities must be equal vectors")
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("choices must be binary")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(np.mean(-(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped))))


def _fit_ridge_logistic(
    features: np.ndarray,
    choices: np.ndarray,
    offsets: np.ndarray,
    precision: float,
    *,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if precision <= 0 or not isfinite(precision):
        raise ValueError("ridge precision must be positive")
    count, dimension = features.shape
    if choices.shape != (count,) or offsets.shape != (count,):
        raise ValueError("ridge logistic inputs are misaligned")
    identity = np.eye(dimension, dtype=np.float64)
    weights = np.zeros(dimension, dtype=np.float64)

    def objective(candidate: np.ndarray) -> float:
        logits = offsets + features @ candidate
        return float(
            np.sum(np.logaddexp(0.0, logits) - choices * logits)
            + 0.5 * precision * (candidate @ candidate)
        )

    for _ in range(maximum_iterations):
        logits = offsets + features @ weights
        probabilities = np.asarray(sigmoid(logits), dtype=np.float64)
        variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
        gradient = features.T @ (probabilities - choices) + precision * weights
        hessian = (
            features.T @ (features * variance[:, None])
            + precision * identity
            + 1e-9 * identity
        )
        step = np.linalg.solve(hessian, gradient)
        if np.linalg.norm(step) < tolerance:
            break
        current = objective(weights)
        scale = 1.0
        accepted = False
        for _ in range(30):
            proposal = weights - scale * step
            if objective(proposal) <= current + 1e-12 * max(1.0, abs(current)):
                weights = proposal
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError("joint-core optimizer could not find a safe step")
        if np.linalg.norm(scale * step) < tolerance:
            break

    logits = offsets + features @ weights
    probabilities = np.asarray(sigmoid(logits), dtype=np.float64)
    variance = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
    hessian = (
        features.T @ (features * variance[:, None])
        + precision * identity
        + 1e-9 * identity
    )
    covariance = np.linalg.inv(hessian)
    return weights, covariance


@dataclass(frozen=True)
class JointCoreModel:
    scenario_feature_names: tuple[str, ...]
    environment_feature_names: tuple[str, ...]
    global_weights: tuple[float, ...]
    global_covariance: tuple[tuple[float, ...], ...]
    person_weights: dict[str, tuple[float, ...]]
    person_covariances: dict[str, tuple[tuple[float, ...], ...]]
    stable_person_precision: float
    model_version: str = "joint-person-core-candidate-v1"
    model_id: str = ""

    def __post_init__(self) -> None:
        scenario_count = len(self.scenario_feature_names)
        environment_count = len(self.environment_feature_names)
        global_count = scenario_count + environment_count
        if (
            scenario_count == 0
            or len(set(self.scenario_feature_names)) != scenario_count
            or len(set(self.environment_feature_names)) != environment_count
            or set(self.scenario_feature_names) & set(self.environment_feature_names)
        ):
            raise ValueError("joint-core feature names are invalid")
        global_weights = np.asarray(self.global_weights, dtype=np.float64)
        global_covariance = np.asarray(self.global_covariance, dtype=np.float64)
        if (
            global_weights.shape != (global_count,)
            or global_covariance.shape != (global_count, global_count)
            or not np.all(np.isfinite(global_weights))
            or not np.all(np.isfinite(global_covariance))
        ):
            raise ValueError("joint-core global parameter dimensions are invalid")
        if not self.person_weights or set(self.person_weights) != set(
            self.person_covariances
        ):
            raise ValueError("joint-core person parameters are incomplete")
        for person_id in self.person_weights:
            weights = np.asarray(self.person_weights[person_id], dtype=np.float64)
            covariance = np.asarray(
                self.person_covariances[person_id], dtype=np.float64
            )
            if (
                not person_id
                or weights.shape != (scenario_count,)
                or covariance.shape != (scenario_count, scenario_count)
                or not np.all(np.isfinite(weights))
                or not np.all(np.isfinite(covariance))
            ):
                raise ValueError("joint-core person parameter dimensions are invalid")
        expected_id = hashlib.sha256(_canonical_json(self._unsigned_dict())).hexdigest()
        if self.model_id and self.model_id != expected_id:
            raise ValueError("joint-core model identity mismatch")
        object.__setattr__(self, "model_id", expected_id)

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "scenario_feature_names": list(self.scenario_feature_names),
            "environment_feature_names": list(self.environment_feature_names),
            "global_weights": list(self.global_weights),
            "global_covariance": [list(row) for row in self.global_covariance],
            "person_weights": {
                key: list(self.person_weights[key]) for key in sorted(self.person_weights)
            },
            "person_covariances": {
                key: [list(row) for row in self.person_covariances[key]]
                for key in sorted(self.person_covariances)
            },
            "stable_person_precision": self.stable_person_precision,
            "model_version": self.model_version,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "model_id": self.model_id}

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> JointCoreModel:
        return cls(
            scenario_feature_names=tuple(
                str(name) for name in value["scenario_feature_names"]
            ),
            environment_feature_names=tuple(
                str(name) for name in value["environment_feature_names"]
            ),
            global_weights=tuple(float(item) for item in value["global_weights"]),
            global_covariance=tuple(
                tuple(float(item) for item in row)
                for row in value["global_covariance"]
            ),
            person_weights={
                str(key): tuple(float(item) for item in values)
                for key, values in dict(value["person_weights"]).items()
            },
            person_covariances={
                str(key): tuple(
                    tuple(float(item) for item in row) for row in values
                )
                for key, values in dict(value["person_covariances"]).items()
            },
            stable_person_precision=float(value["stable_person_precision"]),
            model_version=str(value["model_version"]),
            model_id=str(value["model_id"]),
        )

    def probabilities(
        self,
        person_id: str,
        scenario_features: object,
        environment_features: object,
        *,
        include_person: bool = True,
        include_environment: bool = True,
    ) -> np.ndarray:
        scenario = _as_matrix(scenario_features, "scenario_features")
        environment = _as_matrix(environment_features, "environment_features")
        if (
            scenario.shape[0] != environment.shape[0]
            or scenario.shape[1] != len(self.scenario_feature_names)
            or environment.shape[1] != len(self.environment_feature_names)
        ):
            raise ValueError("joint-core prediction feature dimensions are invalid")
        if person_id not in self.person_weights:
            raise ValueError(f"unknown person {person_id}")
        global_weights = np.asarray(self.global_weights)
        scenario_count = scenario.shape[1]
        means = scenario @ global_weights[:scenario_count]
        global_design = np.concatenate(
            (
                scenario,
                environment if include_environment else np.zeros_like(environment),
            ),
            axis=1,
        )
        if include_environment:
            means = means + environment @ global_weights[scenario_count:]
        if include_person:
            means = means + scenario @ np.asarray(self.person_weights[person_id])
        global_covariance = np.asarray(self.global_covariance)
        variances = np.einsum(
            "ij,jk,ik->i", global_design, global_covariance, global_design
        )
        if include_person:
            person_covariance = np.asarray(self.person_covariances[person_id])
            variances = variances + np.einsum(
                "ij,jk,ik->i", scenario, person_covariance, scenario
            )
        return np.asarray(
            logistic_normal_probability(means, np.maximum(variances, 0.0)),
            dtype=np.float64,
        )

    def logits_and_variances(
        self,
        person_id: str,
        scenario_features: object,
        environment_features: object,
        *,
        include_person: bool = True,
        include_environment: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        scenario = _as_matrix(scenario_features, "scenario_features")
        environment = _as_matrix(environment_features, "environment_features")
        if scenario.shape[0] != environment.shape[0]:
            raise ValueError("joint-core prediction rows are misaligned")
        scenario_count = scenario.shape[1]
        global_weights = np.asarray(self.global_weights)
        means = scenario @ global_weights[:scenario_count]
        environment_used = environment if include_environment else np.zeros_like(environment)
        if include_environment:
            means = means + environment @ global_weights[scenario_count:]
        if include_person:
            means = means + scenario @ np.asarray(self.person_weights[person_id])
        design = np.concatenate((scenario, environment_used), axis=1)
        variances = np.einsum(
            "ij,jk,ik->i",
            design,
            np.asarray(self.global_covariance),
            design,
        )
        if include_person:
            variances = variances + np.einsum(
                "ij,jk,ik->i",
                scenario,
                np.asarray(self.person_covariances[person_id]),
                scenario,
            )
        return means, np.maximum(variances, 0.0)


def fit_joint_core(
    scenario_features: object,
    environment_features: object,
    choices: object,
    person_ids: Sequence[str],
    *,
    scenario_feature_names: Sequence[str],
    environment_feature_names: Sequence[str],
    stable_person_precision: float,
    global_l2_precision: float = 1.0,
    coordinate_passes: int = 6,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> JointCoreModel:
    scenario = _as_matrix(scenario_features, "scenario_features")
    environment = _as_matrix(environment_features, "environment_features")
    if scenario.shape[0] != environment.shape[0]:
        raise ValueError("joint-core fitting rows are misaligned")
    y = _as_choices(choices, scenario.shape[0])
    people = tuple(str(value) for value in person_ids)
    if len(people) != len(y) or any(not value for value in people):
        raise ValueError("joint-core person IDs are invalid")
    scenario_names = tuple(str(value) for value in scenario_feature_names)
    environment_names = tuple(str(value) for value in environment_feature_names)
    if scenario.shape[1] != len(scenario_names) or environment.shape[1] != len(
        environment_names
    ):
        raise ValueError("joint-core feature names do not match matrices")
    if coordinate_passes <= 0:
        raise ValueError("coordinate_passes must be positive")
    unique_people = tuple(sorted(set(people)))
    masks = {
        person_id: np.asarray([value == person_id for value in people])
        for person_id in unique_people
    }
    global_features = np.concatenate((scenario, environment), axis=1)
    global_weights, global_covariance = _fit_ridge_logistic(
        global_features,
        y,
        np.zeros(len(y)),
        global_l2_precision,
        maximum_iterations=maximum_iterations,
        tolerance=tolerance,
    )
    person_weights = {
        person_id: np.zeros(scenario.shape[1], dtype=np.float64)
        for person_id in unique_people
    }
    person_covariances = {
        person_id: np.eye(scenario.shape[1], dtype=np.float64)
        / stable_person_precision
        for person_id in unique_people
    }
    for _ in range(coordinate_passes):
        global_offsets = global_features @ global_weights
        for person_id in unique_people:
            mask = masks[person_id]
            weights, covariance = _fit_ridge_logistic(
                scenario[mask],
                y[mask],
                global_offsets[mask],
                stable_person_precision,
                maximum_iterations=maximum_iterations,
                tolerance=tolerance,
            )
            person_weights[person_id] = weights
            person_covariances[person_id] = covariance
        person_offsets = np.asarray(
            [
                scenario[index] @ person_weights[people[index]]
                for index in range(len(y))
            ],
            dtype=np.float64,
        )
        global_weights, global_covariance = _fit_ridge_logistic(
            global_features,
            y,
            person_offsets,
            global_l2_precision,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )
    return JointCoreModel(
        scenario_feature_names=scenario_names,
        environment_feature_names=environment_names,
        global_weights=tuple(float(value) for value in global_weights),
        global_covariance=tuple(
            tuple(float(value) for value in row) for row in global_covariance
        ),
        person_weights={
            key: tuple(float(value) for value in person_weights[key])
            for key in unique_people
        },
        person_covariances={
            key: tuple(
                tuple(float(value) for value in row)
                for row in person_covariances[key]
            )
            for key in unique_people
        },
        stable_person_precision=stable_person_precision,
    )


@dataclass(frozen=True)
class StateConfig:
    half_life_days: float
    stationary_variance: float
    initial_variance: float | None = None
    update_maximum_iterations: int = 50
    update_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        initial = (
            self.stationary_variance
            if self.initial_variance is None
            else self.initial_variance
        )
        if (
            not isfinite(self.half_life_days)
            or self.half_life_days <= 0
            or not isfinite(self.stationary_variance)
            or self.stationary_variance <= 0
            or not isfinite(initial)
            or initial <= 0
            or self.update_maximum_iterations <= 0
            or self.update_tolerance <= 0
        ):
            raise ValueError("state configuration values are invalid")
        object.__setattr__(self, "initial_variance", float(initial))


@dataclass(frozen=True)
class PrequentialStateResult:
    probabilities: np.ndarray
    prior_state_means: np.ndarray
    prior_state_variances: np.ndarray
    final_state_mean: float
    final_state_variance: float
    final_timestamp: str


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("state timestamps must include a timezone")
    return parsed


def run_prequential_state(
    base_logits: object,
    base_variances: object,
    choices: object,
    timestamps: Sequence[str],
    config: StateConfig,
    *,
    initial_state_mean: float = 0.0,
    initial_state_variance: float | None = None,
    previous_timestamp: str | None = None,
) -> PrequentialStateResult:
    logits = np.asarray(base_logits, dtype=np.float64)
    variances = np.asarray(base_variances, dtype=np.float64)
    if logits.ndim != 1 or variances.shape != logits.shape:
        raise ValueError("state logits and variances must be equal vectors")
    y = _as_choices(choices, len(logits))
    if len(timestamps) != len(logits) or len(logits) == 0:
        raise ValueError("state timestamps must match non-empty logits")
    parsed = tuple(_parse_timestamp(value) for value in timestamps)
    if any(right <= left for left, right in zip(parsed, parsed[1:])):
        raise ValueError("state timestamps must be strictly increasing")
    previous_time = (
        _parse_timestamp(previous_timestamp) if previous_timestamp is not None else None
    )
    if previous_time is not None and parsed[0] <= previous_time:
        raise ValueError("state sequence must follow previous_timestamp")
    posterior_mean = float(initial_state_mean)
    posterior_variance = float(
        config.initial_variance
        if initial_state_variance is None
        else initial_state_variance
    )
    if posterior_variance <= 0 or not isfinite(posterior_variance):
        raise ValueError("initial state variance must be positive")
    probabilities = []
    prior_means = []
    prior_variances = []
    for index, timestamp in enumerate(parsed):
        if previous_time is None:
            prior_mean = posterior_mean
            prior_variance = posterior_variance
        else:
            elapsed_days = (timestamp - previous_time).total_seconds() / 86400.0
            persistence = exp(-log(2.0) * elapsed_days / config.half_life_days)
            prior_mean = persistence * posterior_mean
            prior_variance = (
                persistence * persistence * posterior_variance
                + config.stationary_variance * (1.0 - persistence * persistence)
            )
        probability = float(
            logistic_normal_probability(
                logits[index] + prior_mean,
                max(variances[index] + prior_variance, 0.0),
            )
        )
        probabilities.append(probability)
        prior_means.append(prior_mean)
        prior_variances.append(prior_variance)

        state = prior_mean
        for _ in range(config.update_maximum_iterations):
            likelihood_probability = float(sigmoid(logits[index] + state))
            gradient = (
                likelihood_probability
                - y[index]
                + (state - prior_mean) / prior_variance
            )
            hessian = (
                max(likelihood_probability * (1.0 - likelihood_probability), 1e-8)
                + 1.0 / prior_variance
            )
            step = gradient / hessian
            state -= step
            if abs(step) < config.update_tolerance:
                break
        likelihood_probability = float(sigmoid(logits[index] + state))
        posterior_mean = state
        posterior_variance = 1.0 / (
            max(likelihood_probability * (1.0 - likelihood_probability), 1e-8)
            + 1.0 / prior_variance
        )
        previous_time = timestamp
    return PrequentialStateResult(
        probabilities=np.asarray(probabilities, dtype=np.float64),
        prior_state_means=np.asarray(prior_means, dtype=np.float64),
        prior_state_variances=np.asarray(prior_variances, dtype=np.float64),
        final_state_mean=float(posterior_mean),
        final_state_variance=float(posterior_variance),
        final_timestamp=timestamps[-1],
    )
