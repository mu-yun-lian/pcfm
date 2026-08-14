from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .math_utils import sigmoid
from .person_choice_benchmark import (
    BENCHMARK_FEATURE_NAMES,
    BenchmarkDataset,
    BenchmarkMetric,
    BenchmarkRecord,
    LogisticChoiceModel,
    _fit_population_model,
    _metric,
    generate_benchmark_dataset,
    run_person_choice_benchmark,
    BenchmarkConfig,
)


_HEAD_DIMENSION = len(BENCHMARK_FEATURE_NAMES) + 1
_FIXED_AUDIT_SEEDS = (7301, 7302, 7303, 7304, 7305)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class HyperNetworkInvalidError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "hypernetwork invalid: " + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class HyperNetworkConfig:
    rank: int = 3
    maximum_epochs: int = 80
    learning_rate: float = 0.03
    regularization: float = 1e-3
    support_shrinkage: float = 16.0
    episode_support_sizes: tuple[int, ...] = (16, 32, 64)
    minimum_primary_nll_gain: float = 0.01
    maximum_primary_nll: float = 0.60
    maximum_temporal_nll_excess: float = 0.02
    maximum_null_nll_difference: float = 0.04
    maximum_head_delta_norm: float = 4.0
    minimum_passing_seeds: int = 4
    audit_seeds: tuple[int, ...] = _FIXED_AUDIT_SEEDS
    null_control_seed: int = 7310
    maximum_trainable_scalars: int = 128
    config_version: str = "support-set-hypernetwork-config-v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_support_sizes",
            tuple(self.episode_support_sizes),
        )
        object.__setattr__(
            self,
            "audit_seeds",
            tuple(self.audit_seeds),
        )
        if self.rank != 3:
            raise ValueError("hypernetwork v1 rank is fixed at 3")
        if self.maximum_epochs != 80:
            raise ValueError(
                "hypernetwork v1 maximum_epochs is fixed at 80"
            )
        if self.episode_support_sizes != (16, 32, 64):
            raise ValueError(
                "hypernetwork v1 episode support sizes are fixed"
            )
        if self.audit_seeds != _FIXED_AUDIT_SEEDS:
            raise ValueError(
                "hypernetwork v1 audit seeds are fixed"
            )
        if self.minimum_passing_seeds < 4:
            raise ValueError(
                "minimum_passing_seeds cannot be below 4"
            )
        if self.minimum_passing_seeds > len(self.audit_seeds):
            raise ValueError(
                "minimum_passing_seeds exceeds audit seed count"
            )
        if (
            not isfinite(self.minimum_primary_nll_gain)
            or self.minimum_primary_nll_gain < 0.01
        ):
            raise ValueError(
                "primary NLL gain cannot weaken the 0.01 floor"
            )
        if (
            not isfinite(self.maximum_primary_nll)
            or self.maximum_primary_nll > 0.60
            or self.maximum_primary_nll <= 0
        ):
            raise ValueError(
                "primary NLL ceiling cannot exceed 0.60"
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
            or self.maximum_null_nll_difference > 0.04
            or self.maximum_null_nll_difference < 0
        ):
            raise ValueError(
                "null NLL difference cannot exceed 0.04"
            )
        if (
            not isfinite(self.maximum_head_delta_norm)
            or self.maximum_head_delta_norm > 4.0
            or self.maximum_head_delta_norm <= 0
        ):
            raise ValueError(
                "head delta norm cap must be in (0, 4]"
            )
        if (
            not isfinite(self.learning_rate)
            or self.learning_rate != 0.03
            or not isfinite(self.regularization)
            or self.regularization != 1e-3
            or not isfinite(self.support_shrinkage)
            or self.support_shrinkage != 16.0
        ):
            raise ValueError(
                "hypernetwork v1 optimization settings are fixed"
            )
        if self.null_control_seed != 7310:
            raise ValueError(
                "hypernetwork v1 null control seed is fixed"
            )
        trainable_scalars = self.trainable_scalar_count
        if (
            self.maximum_trainable_scalars < trainable_scalars
            or self.maximum_trainable_scalars > 128
        ):
            raise ValueError(
                "maximum_trainable_scalars must cover the fixed "
                "generator and cannot exceed 128"
            )
        if (
            self.config_version
            != "support-set-hypernetwork-config-v1"
        ):
            raise ValueError(
                "unsupported hypernetwork config version"
            )

    @property
    def trainable_scalar_count(self) -> int:
        return (
            _HEAD_DIMENSION * self.rank
            + self.rank * _HEAD_DIMENSION
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "maximum_epochs": self.maximum_epochs,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "support_shrinkage": self.support_shrinkage,
            "episode_support_sizes": list(
                self.episode_support_sizes
            ),
            "minimum_primary_nll_gain": (
                self.minimum_primary_nll_gain
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
            "maximum_trainable_scalars": (
                self.maximum_trainable_scalars
            ),
            "config_version": self.config_version,
        }


def _canonical_records(
    records: Sequence[BenchmarkRecord],
) -> tuple[BenchmarkRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.design_hash(),
                record.observation.actual_choice,
            ),
        )
    )


def _validate_record_scope(
    records: Sequence[BenchmarkRecord],
    *,
    allowed_roles: set[str],
) -> None:
    reasons = []
    for record in records:
        scenario = record.observation.scenario
        if record.role not in allowed_roles:
            reasons.append("record_role_out_of_scope")
        if scenario.domain != "person_choice_v1":
            reasons.append("record_domain_out_of_scope")
        if scenario.options != ("A", "B"):
            reasons.append("record_options_out_of_scope")
        if scenario.context != {"role": record.role}:
            reasons.append("record_context_out_of_scope")
        if (
            record.observation.provenance
            != "synthetic_ground_truth"
        ):
            reasons.append("record_provenance_out_of_scope")
    if reasons:
        raise HyperNetworkInvalidError(reasons)


def _matrix_for_canonical_records(
    records: Sequence[BenchmarkRecord],
) -> np.ndarray:
    ordered = _canonical_records(records)
    raw = np.asarray(
        [
            record.observation.scenario.ordered_features(
                BENCHMARK_FEATURE_NAMES
            )
            for record in ordered
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        (raw, np.ones(len(raw), dtype=np.float64))
    )


def _choices_for_canonical_records(
    records: Sequence[BenchmarkRecord],
) -> np.ndarray:
    return np.asarray(
        [
            record.observation.actual_choice
            for record in _canonical_records(records)
        ],
        dtype=np.float64,
    )


def _support_summary(
    records: Sequence[BenchmarkRecord],
    population_weights: np.ndarray,
    *,
    shrinkage: float,
) -> np.ndarray:
    if not records:
        return np.zeros(_HEAD_DIMENSION, dtype=np.float64)
    features = _matrix_for_canonical_records(records)
    choices = _choices_for_canonical_records(records)
    probabilities = np.asarray(
        sigmoid(features @ population_weights),
        dtype=np.float64,
    )
    count = len(records)
    scale = np.sqrt(count / (count + shrinkage))
    return (
        scale
        * features.T
        @ (choices - probabilities)
        / count
    )


def _bounded_delta(
    left: np.ndarray,
    right: np.ndarray,
    summary: np.ndarray,
    maximum_norm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    latent = right @ summary
    raw_delta = left @ latent
    norm = float(np.linalg.norm(raw_delta))
    if norm <= maximum_norm:
        return raw_delta, latent, 1.0
    return (
        raw_delta * (maximum_norm / norm),
        latent,
        maximum_norm / norm,
    )


def _pull_back_bounded_delta_gradient(
    raw_delta: np.ndarray,
    gradient: np.ndarray,
    maximum_norm: float,
) -> np.ndarray:
    norm = float(np.linalg.norm(raw_delta))
    if norm <= maximum_norm:
        return gradient
    unit = raw_delta / norm
    return (
        maximum_norm
        / norm
        * (gradient - unit * float(unit @ gradient))
    )


def _matrix_to_tuple(
    matrix: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(value) for value in row)
        for row in matrix
    )


@dataclass(frozen=True)
class SupportSetHyperNetworkArtifact:
    training_evidence_id: str
    config: HyperNetworkConfig
    population_weights: tuple[float, ...]
    left_matrix: tuple[tuple[float, ...], ...]
    right_matrix: tuple[tuple[float, ...], ...]
    selected_epoch: int
    validation_nll: float
    artifact_version: str = "support-set-hypernetwork-artifact-v1"
    artifact_id: str = ""

    def __post_init__(self) -> None:
        reasons = []
        if (
            self.artifact_version
            != "support-set-hypernetwork-artifact-v1"
        ):
            reasons.append("unsupported_artifact_version")
        if not self.training_evidence_id:
            reasons.append("training_evidence_id_required")
        if len(self.population_weights) != _HEAD_DIMENSION:
            reasons.append("population_head_dimension_mismatch")
        left = np.asarray(self.left_matrix, dtype=np.float64)
        right = np.asarray(self.right_matrix, dtype=np.float64)
        if left.shape != (_HEAD_DIMENSION, self.config.rank):
            reasons.append("left_matrix_shape_mismatch")
        if right.shape != (self.config.rank, _HEAD_DIMENSION):
            reasons.append("right_matrix_shape_mismatch")
        if (
            not np.all(
                np.isfinite(
                    np.asarray(
                        self.population_weights,
                        dtype=np.float64,
                    )
                )
            )
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or not isfinite(self.validation_nll)
        ):
            reasons.append("non_finite_artifact_parameter")
        if not 1 <= self.selected_epoch <= self.config.maximum_epochs:
            reasons.append("selected_epoch_out_of_range")
        if reasons:
            raise HyperNetworkInvalidError(reasons)
        expected_id = _canonical_digest(
            self._identity_payload()
        )
        if self.artifact_id:
            if self.artifact_id != expected_id:
                raise HyperNetworkInvalidError(
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
            "left_matrix": [
                list(row) for row in self.left_matrix
            ],
            "right_matrix": [
                list(row) for row in self.right_matrix
            ],
            "selected_epoch": self.selected_epoch,
            "validation_nll": self.validation_nll,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "artifact_id": self.artifact_id,
        }

    def generated_model(
        self,
        support_records: Sequence[BenchmarkRecord],
    ) -> LogisticChoiceModel:
        support = tuple(support_records)
        if support:
            _validate_record_scope(
                support,
                allowed_roles={"support"},
            )
            people = {record.person_id for record in support}
            if len(people) != 1:
                raise HyperNetworkInvalidError(
                    ("support_contains_multiple_people",)
                )
            if len(support) not in self.config.episode_support_sizes:
                raise HyperNetworkInvalidError(
                    ("unsupported_support_size",)
                )
            designs = [record.design_hash() for record in support]
            if len(designs) != len(set(designs)):
                raise HyperNetworkInvalidError(
                    ("replayed_support_design",)
                )
        population = np.asarray(
            self.population_weights,
            dtype=np.float64,
        )
        summary = _support_summary(
            support,
            population,
            shrinkage=self.config.support_shrinkage,
        )
        delta, _, _ = _bounded_delta(
            np.asarray(self.left_matrix, dtype=np.float64),
            np.asarray(self.right_matrix, dtype=np.float64),
            summary,
            self.config.maximum_head_delta_norm,
        )
        generated = population + delta
        return LogisticChoiceModel(
            weights=tuple(
                float(value) for value in generated
            ),
            model_version=(
                "support-set-hypernetwork-head-v1:"
                + self.artifact_id
            ),
        )


def hypernetwork_artifact_from_dict(
    data: Mapping[str, object],
) -> SupportSetHyperNetworkArtifact:
    raw_config = dict(data["config"])
    config = HyperNetworkConfig(
        rank=int(raw_config["rank"]),
        maximum_epochs=int(raw_config["maximum_epochs"]),
        learning_rate=float(raw_config["learning_rate"]),
        regularization=float(raw_config["regularization"]),
        support_shrinkage=float(
            raw_config["support_shrinkage"]
        ),
        episode_support_sizes=tuple(
            int(value)
            for value in raw_config["episode_support_sizes"]
        ),
        minimum_primary_nll_gain=float(
            raw_config["minimum_primary_nll_gain"]
        ),
        maximum_primary_nll=float(
            raw_config["maximum_primary_nll"]
        ),
        maximum_temporal_nll_excess=float(
            raw_config["maximum_temporal_nll_excess"]
        ),
        maximum_null_nll_difference=float(
            raw_config["maximum_null_nll_difference"]
        ),
        maximum_head_delta_norm=float(
            raw_config["maximum_head_delta_norm"]
        ),
        minimum_passing_seeds=int(
            raw_config["minimum_passing_seeds"]
        ),
        audit_seeds=tuple(
            int(value) for value in raw_config["audit_seeds"]
        ),
        null_control_seed=int(
            raw_config["null_control_seed"]
        ),
        maximum_trainable_scalars=int(
            raw_config["maximum_trainable_scalars"]
        ),
        config_version=str(raw_config["config_version"]),
    )
    return SupportSetHyperNetworkArtifact(
        training_evidence_id=str(data["training_evidence_id"]),
        config=config,
        population_weights=tuple(
            float(value) for value in data["population_weights"]
        ),
        left_matrix=tuple(
            tuple(float(value) for value in row)
            for row in data["left_matrix"]
        ),
        right_matrix=tuple(
            tuple(float(value) for value in row)
            for row in data["right_matrix"]
        ),
        selected_epoch=int(data["selected_epoch"]),
        validation_nll=float(data["validation_nll"]),
        artifact_version=str(data["artifact_version"]),
        artifact_id=str(data["artifact_id"]),
    )


def _training_evidence_id(dataset: BenchmarkDataset) -> str:
    records = tuple(
        record
        for record in dataset.records
        if record.role in {"meta_train", "validation"}
    )
    return _canonical_digest(
        {
            "dataset_version": dataset.dataset_version,
            "benchmark_config": dataset.config.to_dict(),
            "roles": ["meta_train", "validation"],
            "records": [
                record.to_dict()
                for record in _canonical_records(records)
            ],
        }
    )


def _episode_permutation(
    count: int,
    *,
    seed: int,
    person_id: str,
    epoch: int,
    purpose: str,
) -> np.ndarray:
    material = (
        f"{seed}|{person_id}|{epoch}|{purpose}"
    ).encode("utf-8")
    local_seed = int.from_bytes(
        hashlib.sha256(material).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(local_seed)
    return rng.permutation(count)


def _episode(
    records: Sequence[BenchmarkRecord],
    *,
    support_size: int,
    seed: int,
    person_id: str,
    epoch: int,
    purpose: str,
) -> tuple[
    tuple[BenchmarkRecord, ...],
    tuple[BenchmarkRecord, ...],
]:
    ordered = _canonical_records(records)
    if len(ordered) <= support_size:
        raise HyperNetworkInvalidError(
            ("episode_has_no_query_records",)
        )
    permutation = _episode_permutation(
        len(ordered),
        seed=seed,
        person_id=person_id,
        epoch=epoch,
        purpose=purpose,
    )
    support_indices = permutation[:support_size]
    query_indices = permutation[support_size:]
    return (
        tuple(ordered[index] for index in support_indices),
        tuple(ordered[index] for index in query_indices),
    )


class _MatrixAdam:
    def __init__(
        self,
        parameters: Mapping[str, np.ndarray],
    ) -> None:
        self.first = {
            name: np.zeros_like(value)
            for name, value in parameters.items()
        }
        self.second = {
            name: np.zeros_like(value)
            for name, value in parameters.items()
        }
        self.step = 0

    def update(
        self,
        parameters: dict[str, np.ndarray],
        gradients: Mapping[str, np.ndarray],
        *,
        learning_rate: float,
    ) -> None:
        self.step += 1
        for name, gradient in gradients.items():
            clipped = np.clip(gradient, -5.0, 5.0)
            self.first[name] = (
                0.9 * self.first[name] + 0.1 * clipped
            )
            self.second[name] = (
                0.999 * self.second[name]
                + 0.001 * clipped * clipped
            )
            first = self.first[name] / (
                1.0 - 0.9**self.step
            )
            second = self.second[name] / (
                1.0 - 0.999**self.step
            )
            parameters[name] -= (
                learning_rate
                * first
                / (np.sqrt(second) + 1e-8)
            )


def _validation_nll(
    dataset: BenchmarkDataset,
    population: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    config: HyperNetworkConfig,
) -> float:
    losses = []
    for person_id in dataset.validation_person_ids:
        records = dataset.records_for(
            person_id,
            "validation",
        )
        for support_size in config.episode_support_sizes:
            support, query = _episode(
                records,
                support_size=support_size,
                seed=dataset.config.seed,
                person_id=person_id,
                epoch=0,
                purpose="validation",
            )
            summary = _support_summary(
                support,
                population,
                shrinkage=config.support_shrinkage,
            )
            delta, _, _ = _bounded_delta(
                left,
                right,
                summary,
                config.maximum_head_delta_norm,
            )
            features = _matrix_for_canonical_records(query)
            choices = _choices_for_canonical_records(query)
            probabilities = np.asarray(
                sigmoid(features @ (population + delta)),
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
        raise HyperNetworkInvalidError(
            ("validation_people_required",)
        )
    return float(np.mean(losses))


def fit_support_set_hypernetwork(
    dataset: BenchmarkDataset,
    config: HyperNetworkConfig | None = None,
) -> SupportSetHyperNetworkArtifact:
    resolved = config or HyperNetworkConfig()
    _validate_record_scope(
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
    rng = np.random.default_rng(dataset.config.seed + 17011)
    parameters = {
        "left": rng.normal(
            0.0,
            0.25,
            size=(_HEAD_DIMENSION, resolved.rank),
        ),
        "right": np.zeros(
            (resolved.rank, _HEAD_DIMENSION),
            dtype=np.float64,
        ),
    }
    optimizer = _MatrixAdam(parameters)
    best_nll = float("inf")
    best_epoch = 0
    best_left = parameters["left"].copy()
    best_right = parameters["right"].copy()
    meta_people = tuple(sorted(dataset.meta_train_person_ids))
    for epoch in range(1, resolved.maximum_epochs + 1):
        batch_start = (
            (epoch - 1) * 4
        ) % len(meta_people)
        batch_people = tuple(
            meta_people[
                (batch_start + offset) % len(meta_people)
            ]
            for offset in range(4)
        )
        for batch_offset, person_id in enumerate(batch_people):
            person_index = (
                batch_start + batch_offset
            ) % len(meta_people)
            records = dataset.records_for(
                person_id,
                "meta_train",
            )
            support_size = resolved.episode_support_sizes[
                (epoch + person_index)
                % len(resolved.episode_support_sizes)
            ]
            support, query = _episode(
                records,
                support_size=support_size,
                seed=dataset.config.seed,
                person_id=person_id,
                epoch=epoch,
                purpose="meta_train",
            )
            summary = _support_summary(
                support,
                population,
                shrinkage=resolved.support_shrinkage,
            )
            left = parameters["left"]
            right = parameters["right"]
            latent = right @ summary
            raw_delta = left @ latent
            norm = float(np.linalg.norm(raw_delta))
            if norm <= resolved.maximum_head_delta_norm:
                delta = raw_delta
            else:
                delta = (
                    raw_delta
                    * resolved.maximum_head_delta_norm
                    / norm
                )
            query_features = _matrix_for_canonical_records(
                query
            )
            query_choices = _choices_for_canonical_records(
                query
            )
            probabilities = np.asarray(
                sigmoid(
                    query_features @ (population + delta)
                ),
                dtype=np.float64,
            )
            weight_gradient = (
                query_features.T
                @ (probabilities - query_choices)
                / len(query)
            )
            raw_gradient = _pull_back_bounded_delta_gradient(
                raw_delta,
                weight_gradient,
                resolved.maximum_head_delta_norm,
            )
            left_before = left.copy()
            gradients = {
                "left": (
                    np.outer(raw_gradient, latent)
                    + resolved.regularization * left
                ),
                "right": (
                    np.outer(
                        left_before.T @ raw_gradient,
                        summary,
                    )
                    + resolved.regularization * right
                ),
            }
            optimizer.update(
                parameters,
                gradients,
                learning_rate=resolved.learning_rate,
            )
        if epoch % 4 != 0:
            continue
        validation_nll = _validation_nll(
            dataset,
            population,
            parameters["left"],
            parameters["right"],
            resolved,
        )
        if validation_nll < best_nll - 1e-12:
            best_nll = validation_nll
            best_epoch = epoch
            best_left = parameters["left"].copy()
            best_right = parameters["right"].copy()
    return SupportSetHyperNetworkArtifact(
        training_evidence_id=_training_evidence_id(dataset),
        config=resolved,
        population_weights=population_model.weights,
        left_matrix=_matrix_to_tuple(best_left),
        right_matrix=_matrix_to_tuple(best_right),
        selected_epoch=best_epoch,
        validation_nll=best_nll,
    )


def verify_support_set_hypernetwork_artifact(
    artifact: SupportSetHyperNetworkArtifact,
    dataset: BenchmarkDataset,
    config: HyperNetworkConfig | None = None,
) -> bool:
    resolved = config or artifact.config
    if artifact.training_evidence_id != _training_evidence_id(
        dataset
    ):
        raise HyperNetworkInvalidError(
            ("training_evidence_mismatch",)
        )
    repeated = fit_support_set_hypernetwork(
        dataset,
        resolved,
    )
    if repeated.to_dict() != artifact.to_dict():
        raise HyperNetworkInvalidError(
            ("artifact_recomputation_mismatch",)
        )
    return True


@dataclass(frozen=True)
class HyperNetworkBenchmarkReport:
    seed: int
    dataset_id: str
    artifact: SupportSetHyperNetworkArtifact
    metrics: tuple[BenchmarkMetric, ...]
    acceptance_checks: tuple[tuple[str, bool], ...]
    single_seed_status: str
    maximum_metric_recalculation_error: float
    record_count: int
    embedding_trained: bool
    report_version: str = "support-set-hypernetwork-report-v1"
    interpretation: str = (
        "synthetic_support_conditioned_weight_generation"
    )
    non_claims: tuple[str, ...] = (
        "generated_weights_have_no_psychological_semantics",
        "real_person_validity_not_established",
        "causal_mechanism_not_identified",
    )

    def __post_init__(self) -> None:
        if (
            self.report_version
            != "support-set-hypernetwork-report-v1"
        ):
            raise HyperNetworkInvalidError(
                ("unsupported_report_version",)
            )
        if self.single_seed_status not in {
            "single_seed_pass",
            "single_seed_fail",
        }:
            raise HyperNetworkInvalidError(
                ("invalid_single_seed_status",)
            )
        if (
            not isfinite(
                self.maximum_metric_recalculation_error
            )
            or self.maximum_metric_recalculation_error < 0
        ):
            raise HyperNetworkInvalidError(
                ("invalid_metric_recalculation_error",)
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
                "hypernetwork metric is absent or ambiguous"
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
            "embedding_trained": self.embedding_trained,
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
                "hypernetwork_trainable_scalars": (
                    self.artifact.config.trainable_scalar_count
                ),
                "maximum_epochs": (
                    self.artifact.config.maximum_epochs
                ),
                "selected_epoch": (
                    self.artifact.selected_epoch
                ),
                "device": "cpu",
                "dependencies": ["numpy"],
            },
        }


def run_hypernetwork_benchmark(
    dataset: BenchmarkDataset,
    config: HyperNetworkConfig | None = None,
    *,
    artifact: SupportSetHyperNetworkArtifact | None = None,
    train_embedding: bool,
) -> HyperNetworkBenchmarkReport:
    resolved = config or HyperNetworkConfig()
    fitted = (
        artifact
        if artifact is not None
        else fit_support_set_hypernetwork(dataset, resolved)
    )
    if fitted.config != resolved:
        raise HyperNetworkInvalidError(
            ("artifact_config_mismatch",)
        )
    if fitted.training_evidence_id != _training_evidence_id(
        dataset
    ):
        raise HyperNetworkInvalidError(
            ("training_evidence_mismatch",)
        )
    base_report = run_person_choice_benchmark(
        dataset,
        train_neural=train_embedding,
    )
    metrics = list(base_report.metrics)
    errors = [base_report.maximum_metric_recalculation_error]
    people = tuple(sorted(dataset.test_person_ids))
    for support_size in dataset.config.support_sizes:
        generated_models = {}
        for person_id in people:
            support = dataset.records_for(
                person_id,
                "support",
            )[:support_size]
            generated_models[person_id] = (
                fitted.generated_model(support)
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
            probabilities = np.concatenate(
                [
                    generated_models[
                        person_id
                    ].probabilities(
                        dataset.records_for(person_id, split)
                    )
                    for person_id in people
                ]
            )
            metric, error = _metric(
                "support_set_hypernetwork",
                split,
                support_size,
                records,
                probabilities,
            )
            metrics.append(metric)
            errors.append(error)
    temporary = HyperNetworkBenchmarkReport(
        seed=dataset.config.seed,
        dataset_id=dataset.dataset_id,
        artifact=fitted,
        metrics=tuple(metrics),
        acceptance_checks=(),
        single_seed_status="single_seed_fail",
        maximum_metric_recalculation_error=max(errors),
        record_count=len(dataset.records),
        embedding_trained=train_embedding,
    )
    generated_scenario = temporary.metric(
        "support_set_hypernetwork",
        "scenario_test",
        64,
    )
    personal_scenario = temporary.metric(
        "personal_map_logistic",
        "scenario_test",
        64,
    )
    generated_temporal = temporary.metric(
        "support_set_hypernetwork",
        "temporal_test",
        64,
    )
    personal_temporal = temporary.metric(
        "personal_map_logistic",
        "temporal_test",
        64,
    )
    beats_embedding = False
    if train_embedding:
        embedding_scenario = temporary.metric(
            "person_embedding_mlp",
            "scenario_test",
            64,
        )
        beats_embedding = (
            embedding_scenario.negative_log_likelihood
            - generated_scenario.negative_log_likelihood
            >= resolved.minimum_primary_nll_gain
        )
    checks = (
        (
            "beats_personal_map",
            (
                personal_scenario.negative_log_likelihood
                - generated_scenario.negative_log_likelihood
                >= resolved.minimum_primary_nll_gain
            ),
        ),
        ("beats_person_embedding", beats_embedding),
        (
            "absolute_primary_adequacy",
            (
                generated_scenario.negative_log_likelihood
                <= resolved.maximum_primary_nll
            ),
        ),
        (
            "temporal_not_materially_worse",
            (
                generated_temporal.negative_log_likelihood
                - personal_temporal.negative_log_likelihood
                <= resolved.maximum_temporal_nll_excess
            ),
        ),
    )
    return HyperNetworkBenchmarkReport(
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
        embedding_trained=temporary.embedding_trained,
    )


@dataclass(frozen=True)
class HyperNetworkSeedAudit:
    config: HyperNetworkConfig
    seed_reports: tuple[HyperNetworkBenchmarkReport, ...]
    null_control_nll_difference: float
    null_control_passed: bool
    passing_seed_count: int
    candidate_status: str
    audit_version: str = "support-set-hypernetwork-seed-audit-v1"

    def __post_init__(self) -> None:
        if (
            self.audit_version
            != "support-set-hypernetwork-seed-audit-v1"
        ):
            raise HyperNetworkInvalidError(
                ("unsupported_seed_audit_version",)
            )
        if self.candidate_status not in {
            "accepted_candidate",
            "rejected_candidate",
        }:
            raise HyperNetworkInvalidError(
                ("invalid_candidate_status",)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_version": self.audit_version,
            "candidate_status": self.candidate_status,
            "passing_seed_count": self.passing_seed_count,
            "required_passing_seed_count": (
                self.config.minimum_passing_seeds
            ),
            "null_control_nll_difference": (
                self.null_control_nll_difference
            ),
            "null_control_passed": self.null_control_passed,
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
                "hypernetwork_trainable_scalars": (
                    self.config.trainable_scalar_count
                ),
                "maximum_epochs_per_seed": (
                    self.config.maximum_epochs
                ),
                "device": "cpu",
                "dependencies": ["numpy"],
            },
            "device": "cpu",
        }


def run_hypernetwork_seed_audit(
    config: HyperNetworkConfig | None = None,
) -> HyperNetworkSeedAudit:
    resolved = config or HyperNetworkConfig()
    reports = []
    for seed in resolved.audit_seeds:
        dataset = generate_benchmark_dataset(
            BenchmarkConfig.smoke(seed=seed)
        )
        reports.append(
            run_hypernetwork_benchmark(
                dataset,
                resolved,
                train_embedding=True,
            )
        )
    null_dataset = generate_benchmark_dataset(
        BenchmarkConfig(
            seed=resolved.null_control_seed,
            heterogeneity_scale=0.0,
        )
    )
    null_report = run_hypernetwork_benchmark(
        null_dataset,
        resolved,
        train_embedding=False,
    )
    generated = null_report.metric(
        "support_set_hypernetwork",
        "scenario_test",
        64,
    )
    population = null_report.metric(
        "population_logistic",
        "scenario_test",
        64,
    )
    null_difference = abs(
        generated.negative_log_likelihood
        - population.negative_log_likelihood
    )
    null_passed = (
        null_difference
        <= resolved.maximum_null_nll_difference
    )
    passing = sum(
        report.single_seed_status == "single_seed_pass"
        for report in reports
    )
    status = (
        "accepted_candidate"
        if (
            passing >= resolved.minimum_passing_seeds
            and null_passed
        )
        else "rejected_candidate"
    )
    return HyperNetworkSeedAudit(
        config=resolved,
        seed_reports=tuple(reports),
        null_control_nll_difference=null_difference,
        null_control_passed=null_passed,
        passing_seed_count=passing,
        candidate_status=status,
    )
