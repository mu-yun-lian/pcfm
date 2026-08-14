from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import hashlib
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .hypernetwork_v1 import (
    _canonical_digest,
    _canonical_records,
    _choices_for_canonical_records,
    _matrix_for_canonical_records,
    _matrix_to_tuple,
    _training_evidence_id,
    _validate_record_scope,
)
from .math_utils import fit_map_logistic, sigmoid
from .person_choice_benchmark import (
    BENCHMARK_FEATURE_NAMES,
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkMetric,
    BenchmarkRecord,
    LogisticChoiceModel,
    _fit_population_model,
    _metric,
    generate_benchmark_dataset,
    run_person_choice_benchmark,
)


_PARAMETER_DIMENSION = len(BENCHMARK_FEATURE_NAMES) + 1
_AUDIT_SEEDS = (8101, 8102, 8103, 8104, 8105)
_SHRINKAGE_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)
_NEAR_DUPLICATE_DISTANCE = 1e-6


class EmpiricalBayesInvalidError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "empirical Bayes invalid: "
            + ", ".join(self.reasons)
        )


def _validate_empirical_bayes_scope(
    records: Sequence[BenchmarkRecord],
    *,
    allowed_roles: set[str],
) -> None:
    try:
        _validate_record_scope(
            records,
            allowed_roles=allowed_roles,
        )
    except ValueError as error:
        reasons = getattr(
            error,
            "reasons",
            ("record_scope_invalid",),
        )
        raise EmpiricalBayesInvalidError(reasons) from error


@dataclass(frozen=True)
class EmpiricalBayesConfig:
    weak_meta_prior_precision: float = 0.1
    eigenvalue_floor: float = 0.02
    shrinkage_candidates: tuple[float, ...] = (
        _SHRINKAGE_CANDIDATES
    )
    support_sizes: tuple[int, ...] = (0, 16, 32, 64)
    minimum_primary_nll_gain: float = 0.005
    minimum_wrong_person_nll_gain: float = 0.01
    minimum_mean_nll_gain: float = 0.005
    maximum_primary_nll: float = 0.58
    maximum_temporal_nll_excess: float = 0.02
    maximum_null_nll_difference: float = 0.04
    maximum_head_delta_norm: float = 6.0
    minimum_passing_seeds: int = 4
    audit_seeds: tuple[int, ...] = _AUDIT_SEEDS
    null_control_seed: int = 8110
    misspecification_seed: int = 8120
    config_version: str = "anisotropic-empirical-bayes-config-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shrinkage_candidates",
            tuple(self.shrinkage_candidates),
        )
        object.__setattr__(
            self,
            "support_sizes",
            tuple(self.support_sizes),
        )
        object.__setattr__(
            self,
            "audit_seeds",
            tuple(self.audit_seeds),
        )
        if self.weak_meta_prior_precision != 0.1:
            raise ValueError(
                "weak meta prior precision is fixed at 0.1"
            )
        if self.eigenvalue_floor != 0.02:
            raise ValueError(
                "eigenvalue floor is fixed at 0.02"
            )
        if (
            self.shrinkage_candidates
            != _SHRINKAGE_CANDIDATES
        ):
            raise ValueError(
                "covariance shrinkage candidates are fixed"
            )
        if self.support_sizes != (0, 16, 32, 64):
            raise ValueError(
                "empirical Bayes support sizes are fixed"
            )
        if self.audit_seeds != _AUDIT_SEEDS:
            raise ValueError(
                "empirical Bayes audit seeds are fixed"
            )
        if self.null_control_seed != 8110:
            raise ValueError(
                "null control seed is fixed at 8110"
            )
        if self.misspecification_seed != 8120:
            raise ValueError(
                "misspecification seed is fixed at 8120"
            )
        if not 4 <= self.minimum_passing_seeds <= 5:
            raise ValueError(
                "minimum passing seeds cannot be below 4"
            )
        if (
            not isfinite(self.minimum_primary_nll_gain)
            or self.minimum_primary_nll_gain < 0.005
        ):
            raise ValueError(
                "primary gain cannot weaken the 0.005 floor"
            )
        if (
            not isfinite(
                self.minimum_wrong_person_nll_gain
            )
            or self.minimum_wrong_person_nll_gain < 0.01
        ):
            raise ValueError(
                "wrong-person gain cannot weaken the 0.01 floor"
            )
        if (
            not isfinite(self.minimum_mean_nll_gain)
            or self.minimum_mean_nll_gain < 0.005
        ):
            raise ValueError(
                "mean gain cannot weaken the 0.005 floor"
            )
        if (
            not isfinite(self.maximum_primary_nll)
            or not 0 < self.maximum_primary_nll <= 0.58
        ):
            raise ValueError(
                "primary NLL ceiling cannot exceed 0.58"
            )
        if (
            not isfinite(self.maximum_temporal_nll_excess)
            or self.maximum_temporal_nll_excess > 0.02
        ):
            raise ValueError(
                "temporal NLL excess cannot exceed 0.02"
            )
        if (
            not isfinite(self.maximum_null_nll_difference)
            or not 0 <= self.maximum_null_nll_difference <= 0.04
        ):
            raise ValueError(
                "null NLL difference cannot exceed 0.04"
            )
        if (
            not isfinite(self.maximum_head_delta_norm)
            or not 0 < self.maximum_head_delta_norm <= 6.0
        ):
            raise ValueError(
                "head delta norm cap must be in (0, 6]"
            )
        if (
            self.config_version
            != "anisotropic-empirical-bayes-config-v1"
        ):
            raise ValueError(
                "unsupported empirical Bayes config version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "weak_meta_prior_precision": (
                self.weak_meta_prior_precision
            ),
            "eigenvalue_floor": self.eigenvalue_floor,
            "shrinkage_candidates": list(
                self.shrinkage_candidates
            ),
            "support_sizes": list(self.support_sizes),
            "minimum_primary_nll_gain": (
                self.minimum_primary_nll_gain
            ),
            "minimum_wrong_person_nll_gain": (
                self.minimum_wrong_person_nll_gain
            ),
            "minimum_mean_nll_gain": (
                self.minimum_mean_nll_gain
            ),
            "maximum_primary_nll": self.maximum_primary_nll,
            "maximum_temporal_nll_excess": (
                self.maximum_temporal_nll_excess
            ),
            "maximum_null_nll_difference": (
                self.maximum_null_nll_difference
            ),
            "maximum_head_delta_norm": (
                self.maximum_head_delta_norm
            ),
            "minimum_passing_seeds": (
                self.minimum_passing_seeds
            ),
            "audit_seeds": list(self.audit_seeds),
            "null_control_seed": self.null_control_seed,
            "misspecification_seed": (
                self.misspecification_seed
            ),
            "config_version": self.config_version,
        }


def _config_from_dict(
    data: Mapping[str, object],
) -> EmpiricalBayesConfig:
    return EmpiricalBayesConfig(
        weak_meta_prior_precision=float(
            data["weak_meta_prior_precision"]
        ),
        eigenvalue_floor=float(data["eigenvalue_floor"]),
        shrinkage_candidates=tuple(
            float(value)
            for value in data["shrinkage_candidates"]
        ),
        support_sizes=tuple(
            int(value) for value in data["support_sizes"]
        ),
        minimum_primary_nll_gain=float(
            data["minimum_primary_nll_gain"]
        ),
        minimum_wrong_person_nll_gain=float(
            data["minimum_wrong_person_nll_gain"]
        ),
        minimum_mean_nll_gain=float(
            data["minimum_mean_nll_gain"]
        ),
        maximum_primary_nll=float(
            data["maximum_primary_nll"]
        ),
        maximum_temporal_nll_excess=float(
            data["maximum_temporal_nll_excess"]
        ),
        maximum_null_nll_difference=float(
            data["maximum_null_nll_difference"]
        ),
        maximum_head_delta_norm=float(
            data["maximum_head_delta_norm"]
        ),
        minimum_passing_seeds=int(
            data["minimum_passing_seeds"]
        ),
        audit_seeds=tuple(
            int(value) for value in data["audit_seeds"]
        ),
        null_control_seed=int(data["null_control_seed"]),
        misspecification_seed=int(
            data["misspecification_seed"]
        ),
        config_version=str(data["config_version"]),
    )


def _clip_positive_definite(
    matrix: np.ndarray,
    floor: float,
) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    result = (
        eigenvectors
        @ np.diag(clipped)
        @ eigenvectors.T
    )
    return 0.5 * (result + result.T)


def _bounded_weights(
    weights: np.ndarray,
    population: np.ndarray,
    maximum_norm: float,
) -> np.ndarray:
    delta = weights - population
    norm = float(np.linalg.norm(delta))
    if norm <= maximum_norm:
        return weights
    return (
        population
        + delta * (maximum_norm / norm)
    )


def _fit_adapted_weights(
    records: Sequence[BenchmarkRecord],
    population: np.ndarray,
    precision: np.ndarray,
    *,
    maximum_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        return (
            population.copy(),
            np.linalg.inv(precision),
        )
    features = _matrix_for_canonical_records(records)
    choices = _choices_for_canonical_records(records)
    try:
        weights, covariance = fit_map_logistic(
            features,
            choices,
            population,
            precision,
        )
    except (
        np.linalg.LinAlgError,
        RuntimeError,
        ValueError,
    ) as error:
        raise EmpiricalBayesInvalidError(
            ("map_optimizer_failure",)
        ) from error
    return (
        _bounded_weights(
            weights,
            population,
            maximum_norm,
        ),
        covariance,
    )


@dataclass(frozen=True)
class AnisotropicPriorArtifact:
    training_evidence_id: str
    config: EmpiricalBayesConfig
    population_weights: tuple[float, ...]
    prior_covariance: tuple[tuple[float, ...], ...]
    prior_precision: tuple[tuple[float, ...], ...]
    selected_shrinkage: float
    validation_nll: float
    meta_person_count: int
    artifact_version: str = "anisotropic-prior-artifact-v1"
    artifact_id: str = ""

    def __post_init__(self) -> None:
        reasons = []
        if (
            self.artifact_version
            != "anisotropic-prior-artifact-v1"
        ):
            reasons.append("unsupported_artifact_version")
        if not self.training_evidence_id:
            reasons.append("training_evidence_id_required")
        if len(self.population_weights) != _PARAMETER_DIMENSION:
            reasons.append("population_dimension_mismatch")
        covariance = np.asarray(
            self.prior_covariance,
            dtype=np.float64,
        )
        precision = np.asarray(
            self.prior_precision,
            dtype=np.float64,
        )
        expected_shape = (
            _PARAMETER_DIMENSION,
            _PARAMETER_DIMENSION,
        )
        if covariance.shape != expected_shape:
            reasons.append("covariance_shape_mismatch")
        if precision.shape != expected_shape:
            reasons.append("precision_shape_mismatch")
        if (
            not np.all(np.isfinite(covariance))
            or not np.all(np.isfinite(precision))
            or not np.all(
                np.isfinite(
                    np.asarray(
                        self.population_weights,
                        dtype=np.float64,
                    )
                )
            )
            or not isfinite(self.validation_nll)
        ):
            reasons.append("non_finite_artifact_parameter")
        if covariance.shape == expected_shape:
            if not np.allclose(
                covariance,
                covariance.T,
                atol=1e-12,
                rtol=0.0,
            ):
                reasons.append("covariance_not_symmetric")
            elif np.min(np.linalg.eigvalsh(covariance)) <= 0:
                reasons.append("covariance_not_positive_definite")
        if precision.shape == expected_shape:
            if not np.allclose(
                precision,
                precision.T,
                atol=1e-12,
                rtol=0.0,
            ):
                reasons.append("precision_not_symmetric")
            elif np.min(np.linalg.eigvalsh(precision)) <= 0:
                reasons.append("precision_not_positive_definite")
        if (
            covariance.shape == expected_shape
            and precision.shape == expected_shape
            and not np.allclose(
                precision @ covariance,
                np.eye(_PARAMETER_DIMENSION),
                atol=1e-9,
                rtol=1e-9,
            )
        ):
            reasons.append("precision_covariance_mismatch")
        if (
            self.selected_shrinkage
            not in self.config.shrinkage_candidates
        ):
            reasons.append("unregistered_shrinkage")
        if self.meta_person_count < 8:
            reasons.append("insufficient_meta_people")
        if reasons:
            raise EmpiricalBayesInvalidError(reasons)
        expected_id = _canonical_digest(
            self._identity_payload()
        )
        if self.artifact_id:
            if self.artifact_id != expected_id:
                raise EmpiricalBayesInvalidError(
                    ("artifact_id_content_mismatch",)
                )
        else:
            object.__setattr__(self, "artifact_id", expected_id)

    def _identity_payload(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "training_evidence_id": self.training_evidence_id,
            "config": self.config.to_dict(),
            "feature_names": list(BENCHMARK_FEATURE_NAMES),
            "population_weights": list(self.population_weights),
            "prior_covariance": [
                list(row) for row in self.prior_covariance
            ],
            "prior_precision": [
                list(row) for row in self.prior_precision
            ],
            "selected_shrinkage": self.selected_shrinkage,
            "validation_nll": self.validation_nll,
            "meta_person_count": self.meta_person_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "artifact_id": self.artifact_id,
        }

    def adapted_model(
        self,
        support_records: Sequence[BenchmarkRecord],
    ) -> LogisticChoiceModel:
        support = tuple(support_records)
        if support:
            _validate_empirical_bayes_scope(
                support,
                allowed_roles={"support"},
            )
            if len({record.person_id for record in support}) != 1:
                raise EmpiricalBayesInvalidError(
                    ("support_contains_multiple_people",)
                )
            if len(support) not in self.config.support_sizes:
                raise EmpiricalBayesInvalidError(
                    ("unsupported_support_size",)
                )
            designs = [
                record.design_hash() for record in support
            ]
            if len(designs) != len(set(designs)):
                raise EmpiricalBayesInvalidError(
                    ("replayed_support_design",)
                )
            raw_features = np.asarray(
                [
                    record.observation.scenario.ordered_features(
                        BENCHMARK_FEATURE_NAMES
                    )
                    for record in support
                ],
                dtype=np.float64,
            )
            differences = (
                raw_features[:, None, :]
                - raw_features[None, :, :]
            )
            distances = np.linalg.norm(
                differences,
                axis=2,
            )
            np.fill_diagonal(distances, np.inf)
            if float(np.min(distances)) <= (
                _NEAR_DUPLICATE_DISTANCE
            ):
                raise EmpiricalBayesInvalidError(
                    ("near_duplicate_support_design",)
                )
        population = np.asarray(
            self.population_weights,
            dtype=np.float64,
        )
        if not support:
            weights = population
        else:
            weights, _ = _fit_adapted_weights(
                support,
                population,
                np.asarray(
                    self.prior_precision,
                    dtype=np.float64,
                ),
                maximum_norm=(
                    self.config.maximum_head_delta_norm
                ),
            )
        return LogisticChoiceModel(
            weights=tuple(float(value) for value in weights),
            model_version=(
                "anisotropic-empirical-bayes-head-v1:"
                + self.artifact_id
            ),
        )


def empirical_bayes_artifact_from_dict(
    data: Mapping[str, object],
) -> AnisotropicPriorArtifact:
    return AnisotropicPriorArtifact(
        training_evidence_id=str(data["training_evidence_id"]),
        config=_config_from_dict(dict(data["config"])),
        population_weights=tuple(
            float(value) for value in data["population_weights"]
        ),
        prior_covariance=tuple(
            tuple(float(value) for value in row)
            for row in data["prior_covariance"]
        ),
        prior_precision=tuple(
            tuple(float(value) for value in row)
            for row in data["prior_precision"]
        ),
        selected_shrinkage=float(
            data["selected_shrinkage"]
        ),
        validation_nll=float(data["validation_nll"]),
        meta_person_count=int(data["meta_person_count"]),
        artifact_version=str(data["artifact_version"]),
        artifact_id=str(data["artifact_id"]),
    )


def _estimate_between_person_covariance(
    dataset: BenchmarkDataset,
    population: np.ndarray,
    config: EmpiricalBayesConfig,
) -> np.ndarray:
    weights = []
    posterior_covariances = []
    weak_precision = (
        np.eye(_PARAMETER_DIMENSION, dtype=np.float64)
        * config.weak_meta_prior_precision
    )
    for person_id in sorted(dataset.meta_train_person_ids):
        records = dataset.records_for(
            person_id,
            "meta_train",
        )
        person_weights, covariance = _fit_adapted_weights(
            records,
            population,
            weak_precision,
            maximum_norm=config.maximum_head_delta_norm,
        )
        weights.append(person_weights)
        posterior_covariances.append(covariance)
    weight_matrix = np.asarray(weights, dtype=np.float64)
    raw_covariance = np.cov(
        weight_matrix,
        rowvar=False,
        ddof=1,
    )
    noise = np.mean(
        np.asarray(posterior_covariances),
        axis=0,
    )
    return _clip_positive_definite(
        raw_covariance - noise,
        config.eigenvalue_floor,
    )


def _candidate_covariance(
    between: np.ndarray,
    shrinkage: float,
    floor: float,
) -> np.ndarray:
    diagonal = np.diag(np.diag(between))
    return _clip_positive_definite(
        (1.0 - shrinkage) * between
        + shrinkage * diagonal,
        floor,
    )


def _validation_score(
    dataset: BenchmarkDataset,
    population: np.ndarray,
    precision: np.ndarray,
    config: EmpiricalBayesConfig,
) -> float:
    losses = []
    for person_id in sorted(dataset.validation_person_ids):
        ordered = _canonical_records(
            dataset.records_for(person_id, "validation")
        )
        query = ordered[-32:]
        for support_size in (16, 32, 64):
            support = ordered[:support_size]
            weights, _ = _fit_adapted_weights(
                support,
                population,
                precision,
                maximum_norm=config.maximum_head_delta_norm,
            )
            features = _matrix_for_canonical_records(query)
            choices = _choices_for_canonical_records(query)
            probabilities = np.asarray(
                sigmoid(features @ weights),
                dtype=np.float64,
            )
            clipped = np.clip(
                probabilities,
                1e-9,
                1.0 - 1e-9,
            )
            losses.append(
                -float(
                    np.mean(
                        choices * np.log(clipped)
                        + (1.0 - choices)
                        * np.log(1.0 - clipped)
                    )
                )
            )
    if not losses:
        raise EmpiricalBayesInvalidError(
            ("validation_people_required",)
        )
    return float(np.mean(losses))


def fit_anisotropic_prior(
    dataset: BenchmarkDataset,
    config: EmpiricalBayesConfig | None = None,
) -> AnisotropicPriorArtifact:
    resolved = config or EmpiricalBayesConfig()
    _validate_empirical_bayes_scope(
        dataset.records,
        allowed_roles={
            "meta_train",
            "validation",
            "support",
            "scenario_test",
            "temporal_test",
            "ood_test",
        },
    )
    population_model = _fit_population_model(dataset)
    population = np.asarray(
        population_model.weights,
        dtype=np.float64,
    )
    between = _estimate_between_person_covariance(
        dataset,
        population,
        resolved,
    )
    selected_shrinkage = None
    selected_covariance = None
    selected_precision = None
    selected_nll = float("inf")
    for shrinkage in resolved.shrinkage_candidates:
        covariance = _candidate_covariance(
            between,
            shrinkage,
            resolved.eigenvalue_floor,
        )
        precision = np.linalg.inv(covariance)
        score = _validation_score(
            dataset,
            population,
            precision,
            resolved,
        )
        if score < selected_nll - 1e-12:
            selected_shrinkage = shrinkage
            selected_covariance = covariance
            selected_precision = precision
            selected_nll = score
    if (
        selected_shrinkage is None
        or selected_covariance is None
        or selected_precision is None
    ):
        raise EmpiricalBayesInvalidError(
            ("no_covariance_candidate_selected",)
        )
    return AnisotropicPriorArtifact(
        training_evidence_id=_training_evidence_id(dataset),
        config=resolved,
        population_weights=population_model.weights,
        prior_covariance=_matrix_to_tuple(
            selected_covariance
        ),
        prior_precision=_matrix_to_tuple(
            selected_precision
        ),
        selected_shrinkage=float(selected_shrinkage),
        validation_nll=selected_nll,
        meta_person_count=len(dataset.meta_train_person_ids),
    )


def verify_anisotropic_prior_artifact(
    artifact: AnisotropicPriorArtifact,
    dataset: BenchmarkDataset,
    config: EmpiricalBayesConfig | None = None,
) -> bool:
    resolved = config or artifact.config
    if artifact.training_evidence_id != _training_evidence_id(
        dataset
    ):
        raise EmpiricalBayesInvalidError(
            ("training_evidence_mismatch",)
        )
    repeated = fit_anisotropic_prior(
        dataset,
        resolved,
    )
    if repeated.to_dict() != artifact.to_dict():
        raise EmpiricalBayesInvalidError(
            ("artifact_recomputation_mismatch",)
        )
    return True


@dataclass(frozen=True)
class EmpiricalBayesBenchmarkReport:
    seed: int
    dataset_id: str
    artifact: AnisotropicPriorArtifact
    metrics: tuple[BenchmarkMetric, ...]
    acceptance_checks: tuple[tuple[str, bool], ...]
    single_seed_status: str
    maximum_metric_recalculation_error: float
    record_count: int
    report_version: str = "anisotropic-empirical-bayes-report-v1"
    interpretation: str = (
        "synthetic_anisotropic_population_prior_adaptation"
    )
    non_claims: tuple[str, ...] = (
        "prior_axes_have_no_psychological_semantics",
        "real_person_validity_not_established",
        "causal_mechanism_not_identified",
    )

    def __post_init__(self) -> None:
        if (
            self.report_version
            != "anisotropic-empirical-bayes-report-v1"
        ):
            raise EmpiricalBayesInvalidError(
                ("unsupported_report_version",)
            )
        if self.single_seed_status not in {
            "single_seed_pass",
            "single_seed_fail",
        }:
            raise EmpiricalBayesInvalidError(
                ("invalid_single_seed_status",)
            )

    def metric(
        self,
        baseline: str,
        split: str,
        support_size: int,
    ):
        matches = [
            item.report
            for item in self.metrics
            if (
                item.baseline == baseline
                and item.split == split
                and item.support_size == support_size
            )
        ]
        if len(matches) != 1:
            raise KeyError(
                "empirical Bayes metric is absent or ambiguous"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_version": self.report_version,
            "seed": self.seed,
            "dataset_id": self.dataset_id,
            "artifact": self.artifact.to_dict(),
            "interpretation": self.interpretation,
            "non_claims": list(self.non_claims),
            "single_seed_status": self.single_seed_status,
            "acceptance_checks": dict(
                self.acceptance_checks
            ),
            "metrics": [
                metric.to_dict() for metric in self.metrics
            ],
            "maximum_metric_recalculation_error": (
                self.maximum_metric_recalculation_error
            ),
            "resource_usage": {
                "record_count": self.record_count,
                "parameter_dimension": _PARAMETER_DIMENSION,
                "covariance_scalars": (
                    _PARAMETER_DIMENSION
                    * (_PARAMETER_DIMENSION + 1)
                    // 2
                ),
                "selected_shrinkage": (
                    self.artifact.selected_shrinkage
                ),
                "device": "cpu",
                "dependencies": ["numpy"],
            },
        }


def run_empirical_bayes_benchmark(
    dataset: BenchmarkDataset,
    config: EmpiricalBayesConfig | None = None,
    *,
    artifact: AnisotropicPriorArtifact | None = None,
) -> EmpiricalBayesBenchmarkReport:
    resolved = config or EmpiricalBayesConfig()
    fitted = (
        artifact
        if artifact is not None
        else fit_anisotropic_prior(dataset, resolved)
    )
    if fitted.config != resolved:
        raise EmpiricalBayesInvalidError(
            ("artifact_config_mismatch",)
        )
    if fitted.training_evidence_id != _training_evidence_id(
        dataset
    ):
        raise EmpiricalBayesInvalidError(
            ("training_evidence_mismatch",)
        )
    base_report = run_person_choice_benchmark(
        dataset,
        train_neural=False,
    )
    metrics = list(base_report.metrics)
    errors = [base_report.maximum_metric_recalculation_error]
    people = tuple(sorted(dataset.test_person_ids))
    for support_size in dataset.config.support_sizes:
        adapted = {}
        wrong_adapted = {}
        for index, person_id in enumerate(people):
            support = dataset.records_for(
                person_id,
                "support",
            )[:support_size]
            wrong_person = people[(index + 1) % len(people)]
            wrong_support = dataset.records_for(
                wrong_person,
                "support",
            )[:support_size]
            adapted[person_id] = fitted.adapted_model(support)
            wrong_adapted[person_id] = fitted.adapted_model(
                wrong_support
            )
        for split in (
            "scenario_test",
            "temporal_test",
            "ood_test",
        ):
            records = tuple(
                record
                for person_id in people
                for record in dataset.records_for(
                    person_id,
                    split,
                )
            )
            adapted_probabilities = np.concatenate(
                [
                    adapted[person_id].probabilities(
                        dataset.records_for(person_id, split)
                    )
                    for person_id in people
                ]
            )
            wrong_probabilities = np.concatenate(
                [
                    wrong_adapted[person_id].probabilities(
                        dataset.records_for(person_id, split)
                    )
                    for person_id in people
                ]
            )
            for baseline, probabilities in (
                (
                    "anisotropic_empirical_bayes_map",
                    adapted_probabilities,
                ),
                (
                    "wrong_person_anisotropic_empirical_bayes_map",
                    wrong_probabilities,
                ),
            ):
                metric, error = _metric(
                    baseline,
                    split,
                    support_size,
                    records,
                    probabilities,
                )
                metrics.append(metric)
                errors.append(error)
    temporary = EmpiricalBayesBenchmarkReport(
        seed=dataset.config.seed,
        dataset_id=dataset.dataset_id,
        artifact=fitted,
        metrics=tuple(metrics),
        acceptance_checks=(),
        single_seed_status="single_seed_fail",
        maximum_metric_recalculation_error=max(errors),
        record_count=len(dataset.records),
    )
    adapted_scenario = temporary.metric(
        "anisotropic_empirical_bayes_map",
        "scenario_test",
        64,
    )
    isotropic_scenario = temporary.metric(
        "personal_map_logistic",
        "scenario_test",
        64,
    )
    wrong_scenario = temporary.metric(
        "wrong_person_anisotropic_empirical_bayes_map",
        "scenario_test",
        64,
    )
    adapted_temporal = temporary.metric(
        "anisotropic_empirical_bayes_map",
        "temporal_test",
        64,
    )
    isotropic_temporal = temporary.metric(
        "personal_map_logistic",
        "temporal_test",
        64,
    )
    checks = (
        (
            "beats_isotropic_map",
            (
                isotropic_scenario.negative_log_likelihood
                - adapted_scenario.negative_log_likelihood
                >= resolved.minimum_primary_nll_gain
            ),
        ),
        (
            "beats_wrong_person_support",
            (
                wrong_scenario.negative_log_likelihood
                - adapted_scenario.negative_log_likelihood
                >= resolved.minimum_wrong_person_nll_gain
            ),
        ),
        (
            "absolute_primary_adequacy",
            (
                adapted_scenario.negative_log_likelihood
                <= resolved.maximum_primary_nll
            ),
        ),
        (
            "temporal_not_materially_worse",
            (
                adapted_temporal.negative_log_likelihood
                - isotropic_temporal.negative_log_likelihood
                <= resolved.maximum_temporal_nll_excess
            ),
        ),
    )
    return EmpiricalBayesBenchmarkReport(
        seed=temporary.seed,
        dataset_id=temporary.dataset_id,
        artifact=temporary.artifact,
        metrics=temporary.metrics,
        acceptance_checks=checks,
        single_seed_status=(
            "single_seed_pass"
            if all(value for _, value in checks)
            else "single_seed_fail"
        ),
        maximum_metric_recalculation_error=(
            temporary.maximum_metric_recalculation_error
        ),
        record_count=temporary.record_count,
    )


def generate_stable_misspecified_dataset(
    seed: int = 8120,
) -> BenchmarkDataset:
    base = generate_benchmark_dataset(
        BenchmarkConfig.smoke(seed=seed)
    )
    records = []
    for record in base.records:
        features = np.asarray(
            record.observation.scenario.ordered_features(
                BENCHMARK_FEATURE_NAMES
            ),
            dtype=np.float64,
        )
        logit = float(
            3.5 * features[0] * features[1]
            + 0.25 * features[3]
        )
        probability = float(sigmoid(logit))
        digest = hashlib.sha256(
            (
                f"{seed}|{record.design_hash()}|"
                "stable-misspecification"
            ).encode("utf-8")
        ).digest()
        uniform = int.from_bytes(
            digest[:8],
            "big",
        ) / float(2**64)
        choice = int(uniform < probability)
        records.append(
            replace(
                record,
                observation=replace(
                    record.observation,
                    actual_choice=choice,
                ),
                true_probability=probability,
            )
        )
    return BenchmarkDataset(
        config=base.config,
        records=tuple(records),
        meta_train_person_ids=base.meta_train_person_ids,
        validation_person_ids=base.validation_person_ids,
        test_person_ids=base.test_person_ids,
    )


@dataclass(frozen=True)
class EmpiricalBayesSeedAudit:
    config: EmpiricalBayesConfig
    seed_reports: tuple[EmpiricalBayesBenchmarkReport, ...]
    passing_seed_count: int
    mean_primary_nll_gain: float
    null_control_nll_difference: float
    null_control_passed: bool
    misspecification_primary_nll: float
    misspecification_rejected: bool
    candidate_status: str
    audit_version: str = "anisotropic-empirical-bayes-seed-audit-v1"

    def __post_init__(self) -> None:
        if (
            self.audit_version
            != "anisotropic-empirical-bayes-seed-audit-v1"
        ):
            raise EmpiricalBayesInvalidError(
                ("unsupported_seed_audit_version",)
            )
        if self.candidate_status not in {
            "accepted_candidate",
            "rejected_candidate",
        }:
            raise EmpiricalBayesInvalidError(
                ("invalid_candidate_status",)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": self.audit_version,
            "candidate_status": self.candidate_status,
            "audit_seeds": list(self.config.audit_seeds),
            "passing_seed_count": self.passing_seed_count,
            "required_passing_seed_count": (
                self.config.minimum_passing_seeds
            ),
            "mean_primary_nll_gain": (
                self.mean_primary_nll_gain
            ),
            "minimum_mean_nll_gain": (
                self.config.minimum_mean_nll_gain
            ),
            "null_control_nll_difference": (
                self.null_control_nll_difference
            ),
            "null_control_passed": self.null_control_passed,
            "misspecification_primary_nll": (
                self.misspecification_primary_nll
            ),
            "misspecification_rejected": (
                self.misspecification_rejected
            ),
            "seed_reports": [
                report.to_dict() for report in self.seed_reports
            ],
            "non_claims": [
                "synthetic_evidence_only",
                "no_psychological_semantics",
                "no_real_person_validity",
            ],
            "resource_usage": {
                "record_count_per_seed": 3408,
                "parameter_dimension": _PARAMETER_DIMENSION,
                "covariance_scalars": 21,
                "device": "cpu",
                "dependencies": ["numpy"],
            },
            "device": "cpu",
        }


def run_empirical_bayes_seed_audit(
    config: EmpiricalBayesConfig | None = None,
) -> EmpiricalBayesSeedAudit:
    resolved = config or EmpiricalBayesConfig()
    reports = []
    gains = []
    for seed in resolved.audit_seeds:
        dataset = generate_benchmark_dataset(
            BenchmarkConfig.smoke(seed=seed)
        )
        report = run_empirical_bayes_benchmark(
            dataset,
            resolved,
        )
        reports.append(report)
        adapted = report.metric(
            "anisotropic_empirical_bayes_map",
            "scenario_test",
            64,
        )
        isotropic = report.metric(
            "personal_map_logistic",
            "scenario_test",
            64,
        )
        gains.append(
            isotropic.negative_log_likelihood
            - adapted.negative_log_likelihood
        )
    null_dataset = generate_benchmark_dataset(
        replace(
            BenchmarkConfig.smoke(
                seed=resolved.null_control_seed
            ),
            heterogeneity_scale=0.0,
        )
    )
    null_report = run_empirical_bayes_benchmark(
        null_dataset,
        resolved,
    )
    null_adapted = null_report.metric(
        "anisotropic_empirical_bayes_map",
        "scenario_test",
        64,
    )
    null_population = null_report.metric(
        "population_logistic",
        "scenario_test",
        64,
    )
    null_difference = abs(
        null_adapted.negative_log_likelihood
        - null_population.negative_log_likelihood
    )
    null_passed = (
        null_difference
        <= resolved.maximum_null_nll_difference
    )
    misspecified_dataset = (
        generate_stable_misspecified_dataset(
            resolved.misspecification_seed
        )
    )
    misspecified_report = run_empirical_bayes_benchmark(
        misspecified_dataset,
        resolved,
    )
    misspecified_metric = misspecified_report.metric(
        "anisotropic_empirical_bayes_map",
        "scenario_test",
        64,
    )
    misspecification_rejected = not dict(
        misspecified_report.acceptance_checks
    )["absolute_primary_adequacy"]
    passing = sum(
        report.single_seed_status == "single_seed_pass"
        for report in reports
    )
    mean_gain = float(np.mean(gains))
    status = (
        "accepted_candidate"
        if (
            passing >= resolved.minimum_passing_seeds
            and mean_gain >= resolved.minimum_mean_nll_gain
            and null_passed
            and misspecification_rejected
        )
        else "rejected_candidate"
    )
    return EmpiricalBayesSeedAudit(
        config=resolved,
        seed_reports=tuple(reports),
        passing_seed_count=passing,
        mean_primary_nll_gain=mean_gain,
        null_control_nll_difference=null_difference,
        null_control_passed=null_passed,
        misspecification_primary_nll=(
            misspecified_metric.negative_log_likelihood
        ),
        misspecification_rejected=(
            misspecification_rejected
        ),
        candidate_status=status,
    )
