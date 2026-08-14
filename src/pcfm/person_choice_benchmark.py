from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from .contracts import EvaluationReport, Observation, Scenario
from .evaluation import evaluate_probability_array, report_to_dict
from .math_utils import fit_map_logistic, sigmoid


BENCHMARK_FEATURE_NAMES = (
    "reward_gain",
    "loss_risk",
    "delay",
    "control",
    "fairness",
)
BENCHMARK_ROLES = frozenset(
    {
        "meta_train",
        "validation",
        "support",
        "scenario_test",
        "temporal_test",
        "ood_test",
    }
)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            "benchmark timestamp must be ISO-8601"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            "benchmark timestamp must include a timezone"
        )
    return parsed


class BenchmarkInvalidError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "benchmark invalid: " + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 7301
    meta_train_person_count: int = 20
    validation_person_count: int = 4
    test_person_count: int = 6
    meta_train_trials: int = 96
    validation_trials: int = 96
    support_trials: int = 64
    scenario_test_trials: int = 48
    temporal_test_trials: int = 48
    ood_test_trials: int = 24
    support_sizes: tuple[int, ...] = (0, 16, 32, 64)
    heterogeneity_scale: float = 1.35
    personal_prior_precision: float = 4.0
    neural_hidden_dimension: int = 16
    person_embedding_dimension: int = 8
    neural_epochs: int = 80
    neural_batch_size: int = 64
    maximum_records: int = 5000
    config_version: str = "person-choice-benchmark-config-v1"

    def __post_init__(self) -> None:
        if (
            self.meta_train_person_count < 8
            or self.validation_person_count < 2
            or self.test_person_count < 4
        ):
            raise ValueError(
                "benchmark person counts are below hard floors"
            )
        if (
            self.meta_train_trials < 64
            or self.validation_trials < 32
            or self.support_trials < 32
            or self.scenario_test_trials < 32
            or self.temporal_test_trials < 32
            or self.ood_test_trials < 16
        ):
            raise ValueError(
                "benchmark trial counts are below hard floors"
            )
        support_sizes = tuple(self.support_sizes)
        object.__setattr__(self, "support_sizes", support_sizes)
        if (
            not support_sizes
            or support_sizes[0] != 0
            or tuple(sorted(set(support_sizes))) != support_sizes
            or any(
                value != 0 and value < 8
                for value in support_sizes
            )
            or support_sizes[-1] != self.support_trials
        ):
            raise ValueError(
                "support sizes must be unique, ordered, start at zero, "
                "and end at support_trials"
            )
        if (
            not isfinite(self.heterogeneity_scale)
            or self.heterogeneity_scale < 0
            or not isfinite(self.personal_prior_precision)
            or self.personal_prior_precision <= 0
        ):
            raise ValueError(
                "benchmark distribution parameters are invalid"
            )
        if not 4 <= self.neural_hidden_dimension <= 64:
            raise ValueError(
                "neural hidden dimension must be in [4, 64]"
            )
        if not 2 <= self.person_embedding_dimension <= 32:
            raise ValueError(
                "person embedding dimension must be in [2, 32]"
            )
        if not 10 <= self.neural_epochs <= 200:
            raise ValueError(
                "neural epochs must be in [10, 200]"
            )
        if not 16 <= self.neural_batch_size <= 256:
            raise ValueError(
                "neural batch size must be in [16, 256]"
            )
        if self.maximum_records > 50_000:
            raise ValueError(
                "benchmark maximum_records exceeds the local safety cap"
            )
        if self.estimated_record_count > self.maximum_records:
            raise ValueError(
                "benchmark configuration exceeds maximum_records"
            )
        if (
            self.config_version
            != "person-choice-benchmark-config-v1"
        ):
            raise ValueError(
                "unsupported benchmark config version"
            )

    @property
    def estimated_record_count(self) -> int:
        return (
            self.meta_train_person_count * self.meta_train_trials
            + self.validation_person_count * self.validation_trials
            + self.test_person_count
            * (
                self.support_trials
                + self.scenario_test_trials
                + self.temporal_test_trials
                + self.ood_test_trials
            )
        )

    @classmethod
    def smoke(cls, *, seed: int = 7301) -> BenchmarkConfig:
        return cls(seed=seed)

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "meta_train_person_count": (
                self.meta_train_person_count
            ),
            "validation_person_count": (
                self.validation_person_count
            ),
            "test_person_count": self.test_person_count,
            "meta_train_trials": self.meta_train_trials,
            "validation_trials": self.validation_trials,
            "support_trials": self.support_trials,
            "scenario_test_trials": self.scenario_test_trials,
            "temporal_test_trials": self.temporal_test_trials,
            "ood_test_trials": self.ood_test_trials,
            "support_sizes": list(self.support_sizes),
            "heterogeneity_scale": self.heterogeneity_scale,
            "personal_prior_precision": (
                self.personal_prior_precision
            ),
            "neural_hidden_dimension": (
                self.neural_hidden_dimension
            ),
            "person_embedding_dimension": (
                self.person_embedding_dimension
            ),
            "neural_epochs": self.neural_epochs,
            "neural_batch_size": self.neural_batch_size,
            "maximum_records": self.maximum_records,
            "config_version": self.config_version,
        }


@dataclass(frozen=True)
class BenchmarkRecord:
    record_id: str
    observation: Observation
    observed_at: str
    role: str
    true_probability: float

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("benchmark record_id is required")
        _parse_timestamp(self.observed_at)
        if self.role not in BENCHMARK_ROLES:
            raise ValueError("unsupported benchmark record role")
        if (
            not isfinite(self.true_probability)
            or not 0 <= self.true_probability <= 1
        ):
            raise ValueError(
                "benchmark true probability is invalid"
            )
        if (
            self.observation.scenario.feature_names
            != BENCHMARK_FEATURE_NAMES
        ):
            raise ValueError(
                "benchmark feature schema does not match v1"
            )

    @property
    def person_id(self) -> str:
        return self.observation.person_id

    def design_hash(self) -> str:
        return _canonical_digest(
            {
                "person_id": self.person_id,
                "features": list(
                    self.observation.scenario.features
                ),
                "options": list(
                    self.observation.scenario.options
                ),
                "domain": self.observation.scenario.domain,
            }
        )

    def to_dict(self) -> dict[str, object]:
        scenario = self.observation.scenario
        return {
            "record_id": self.record_id,
            "person_id": self.person_id,
            "scenario_id": scenario.scenario_id,
            "features": list(scenario.features),
            "feature_names": list(scenario.feature_names),
            "options": list(scenario.options),
            "domain": scenario.domain,
            "context": dict(sorted(scenario.context.items())),
            "actual_choice": self.observation.actual_choice,
            "confidence": self.observation.confidence,
            "reaction_time_ms": (
                self.observation.reaction_time_ms
            ),
            "provenance": self.observation.provenance,
            "observed_at": self.observed_at,
            "role": self.role,
            "true_probability": self.true_probability,
        }


@dataclass(frozen=True)
class BenchmarkDataset:
    config: BenchmarkConfig
    records: tuple[BenchmarkRecord, ...]
    meta_train_person_ids: tuple[str, ...]
    validation_person_ids: tuple[str, ...]
    test_person_ids: tuple[str, ...]
    dataset_version: str = "person-choice-dataset-v1"
    dataset_id: str = ""

    def __post_init__(self) -> None:
        records = tuple(
            sorted(
                self.records,
                key=lambda item: (
                    item.person_id,
                    item.role,
                    item.observation.scenario.scenario_id,
                    item.record_id,
                ),
            )
        )
        object.__setattr__(self, "records", records)
        groups = (
            tuple(self.meta_train_person_ids),
            tuple(self.validation_person_ids),
            tuple(self.test_person_ids),
        )
        (
            meta_train_ids,
            validation_ids,
            test_ids,
        ) = groups
        object.__setattr__(
            self,
            "meta_train_person_ids",
            tuple(sorted(meta_train_ids)),
        )
        object.__setattr__(
            self,
            "validation_person_ids",
            tuple(sorted(validation_ids)),
        )
        object.__setattr__(
            self,
            "test_person_ids",
            tuple(sorted(test_ids)),
        )
        reasons = []
        if self.dataset_version != "person-choice-dataset-v1":
            reasons.append("unsupported_dataset_version")
        if (
            len(meta_train_ids)
            != self.config.meta_train_person_count
            or len(validation_ids)
            != self.config.validation_person_count
            or len(test_ids)
            != self.config.test_person_count
        ):
            reasons.append("person_group_count_mismatch")
        if (
            len(set(meta_train_ids))
            != len(meta_train_ids)
            or len(set(validation_ids))
            != len(validation_ids)
            or len(set(test_ids)) != len(test_ids)
            or set(meta_train_ids) & set(validation_ids)
            or set(meta_train_ids) & set(test_ids)
            or set(validation_ids) & set(test_ids)
        ):
            reasons.append("person_groups_not_disjoint")
        all_people = set().union(*map(set, groups))
        if {
            record.person_id for record in records
        } != all_people:
            reasons.append("record_person_group_mismatch")
        record_ids = tuple(record.record_id for record in records)
        scenario_keys = tuple(
            (
                record.person_id,
                record.observation.scenario.scenario_id,
            )
            for record in records
        )
        if len(set(record_ids)) != len(record_ids):
            reasons.append("duplicate_record_id")
        if len(set(scenario_keys)) != len(scenario_keys):
            reasons.append("duplicate_person_scenario")
        designs = tuple(record.design_hash() for record in records)
        if len(set(designs)) != len(designs):
            reasons.append("replayed_or_relabelled_design")
        expected_counts = {
            "meta_train": self.config.meta_train_trials,
            "validation": self.config.validation_trials,
            "support": self.config.support_trials,
            "scenario_test": self.config.scenario_test_trials,
            "temporal_test": self.config.temporal_test_trials,
            "ood_test": self.config.ood_test_trials,
        }
        for person_id in meta_train_ids:
            if self._role_counts(person_id) != {
                "meta_train": expected_counts["meta_train"]
            }:
                reasons.append("meta_train_role_mismatch")
        for person_id in validation_ids:
            if self._role_counts(person_id) != {
                "validation": expected_counts["validation"]
            }:
                reasons.append("validation_role_mismatch")
        for person_id in test_ids:
            if self._role_counts(person_id) != {
                role: expected_counts[role]
                for role in (
                    "support",
                    "scenario_test",
                    "temporal_test",
                    "ood_test",
                )
            }:
                reasons.append("test_role_mismatch")
                continue
            support_times = [
                _parse_timestamp(item.observed_at)
                for item in records
                if (
                    item.person_id == person_id
                    and item.role == "support"
                )
            ]
            temporal_times = [
                _parse_timestamp(item.observed_at)
                for item in records
                if (
                    item.person_id == person_id
                    and item.role == "temporal_test"
                )
            ]
            if max(support_times) >= min(temporal_times):
                reasons.append("temporal_test_not_after_support")
        if len(records) != self.config.estimated_record_count:
            reasons.append("record_count_mismatch")
        if reasons:
            raise BenchmarkInvalidError(reasons)
        expected_id = self.digest()
        if self.dataset_id:
            if self.dataset_id != expected_id:
                raise BenchmarkInvalidError(
                    ("dataset_id_content_mismatch",)
                )
        else:
            object.__setattr__(self, "dataset_id", expected_id)

    def _role_counts(self, person_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.person_id == person_id:
                counts[record.role] = counts.get(record.role, 0) + 1
        return counts

    def records_for(
        self,
        person_id: str,
        role: str,
    ) -> tuple[BenchmarkRecord, ...]:
        if role not in BENCHMARK_ROLES:
            raise ValueError("unsupported benchmark role")
        return tuple(
            record
            for record in self.records
            if record.person_id == person_id and record.role == role
        )

    def digest(self) -> str:
        return _canonical_digest(
            {
                "dataset_version": self.dataset_version,
                "config": self.config.to_dict(),
                "meta_train_person_ids": list(
                    sorted(self.meta_train_person_ids)
                ),
                "validation_person_ids": list(
                    sorted(self.validation_person_ids)
                ),
                "test_person_ids": list(
                    sorted(self.test_person_ids)
                ),
                "records": [
                    record.to_dict() for record in self.records
                ],
            }
        )


def _sample_features(
    rng: np.random.Generator,
    role: str,
) -> np.ndarray:
    raw = rng.normal(
        0.0,
        1.0,
        size=len(BENCHMARK_FEATURE_NAMES),
    )
    if role in {"meta_train", "validation", "support"}:
        return raw
    if role == "scenario_test":
        transform = np.asarray(
            [
                [0.90, 0.18, 0.00, 0.00, 0.00],
                [-0.12, 0.95, 0.15, 0.00, 0.00],
                [0.00, 0.00, 0.90, 0.20, 0.00],
                [0.10, 0.00, 0.00, 0.90, 0.18],
                [0.00, 0.10, 0.00, 0.00, 0.95],
            ],
            dtype=np.float64,
        )
        return transform @ raw + np.asarray(
            [0.15, -0.10, 0.10, -0.08, 0.12]
        )
    if role == "temporal_test":
        return raw + np.asarray(
            [0.20, -0.15, 0.12, -0.10, 0.10]
        )
    if role == "ood_test":
        direction = np.asarray(
            [1.0, -1.0, 1.0, -1.0, 1.0]
        )
        return raw * 0.6 + 3.5 * direction
    raise ValueError("unsupported feature role")


def _record_time(role: str) -> str:
    return {
        "meta_train": "2026-01-01T00:00:00Z",
        "validation": "2026-01-15T00:00:00Z",
        "support": "2026-02-01T00:00:00Z",
        "scenario_test": "2026-02-15T00:00:00Z",
        "temporal_test": "2026-03-01T00:00:00Z",
        "ood_test": "2026-03-01T00:00:00Z",
    }[role]


def _generate_records(
    rng: np.random.Generator,
    *,
    person_id: str,
    weights: np.ndarray,
    intercept: float,
    role: str,
    count: int,
) -> tuple[BenchmarkRecord, ...]:
    records = []
    for index in range(count):
        features = _sample_features(rng, role)
        probability = float(
            sigmoid(float(features @ weights + intercept))
        )
        choice = int(rng.random() < probability)
        scenario_id = f"{person_id}-{role}-{index:04d}"
        observation = Observation(
            person_id=person_id,
            scenario=Scenario(
                scenario_id=scenario_id,
                features=tuple(float(value) for value in features),
                feature_names=BENCHMARK_FEATURE_NAMES,
                domain="person_choice_v1",
                context={"role": role},
            ),
            actual_choice=choice,
            confidence=0.5 + abs(probability - 0.5),
            reaction_time_ms=1200.0,
            provenance="synthetic_ground_truth",
        )
        records.append(
            BenchmarkRecord(
                record_id=f"record-{scenario_id}",
                observation=observation,
                observed_at=_record_time(role),
                role=role,
                true_probability=probability,
            )
        )
    return tuple(records)


def generate_benchmark_dataset(
    config: BenchmarkConfig,
) -> BenchmarkDataset:
    rng = np.random.default_rng(config.seed)
    total_people = (
        config.meta_train_person_count
        + config.validation_person_count
        + config.test_person_count
    )
    population_weights = np.asarray(
        [1.20, -1.30, -0.65, 0.85, 0.60],
        dtype=np.float64,
    )
    person_scale = np.asarray(
        [0.85, 0.85, 0.60, 0.75, 0.65],
        dtype=np.float64,
    )
    people = []
    for index in range(total_people):
        person_id = f"benchmark-person-{index:03d}"
        weights = population_weights + rng.normal(
            0.0,
            person_scale * config.heterogeneity_scale,
        )
        intercept = float(
            rng.normal(0.0, 0.35 * config.heterogeneity_scale)
        )
        people.append((person_id, weights, intercept))
    meta_end = config.meta_train_person_count
    validation_end = meta_end + config.validation_person_count
    meta_people = people[:meta_end]
    validation_people = people[meta_end:validation_end]
    test_people = people[validation_end:]
    records = []
    for person_id, weights, intercept in meta_people:
        records.extend(
            _generate_records(
                rng,
                person_id=person_id,
                weights=weights,
                intercept=intercept,
                role="meta_train",
                count=config.meta_train_trials,
            )
        )
    for person_id, weights, intercept in validation_people:
        records.extend(
            _generate_records(
                rng,
                person_id=person_id,
                weights=weights,
                intercept=intercept,
                role="validation",
                count=config.validation_trials,
            )
        )
    test_roles = (
        ("support", config.support_trials),
        ("scenario_test", config.scenario_test_trials),
        ("temporal_test", config.temporal_test_trials),
        ("ood_test", config.ood_test_trials),
    )
    for person_id, weights, intercept in test_people:
        for role, count in test_roles:
            records.extend(
                _generate_records(
                    rng,
                    person_id=person_id,
                    weights=weights,
                    intercept=intercept,
                    role=role,
                    count=count,
                )
            )
    return BenchmarkDataset(
        config=config,
        records=tuple(records),
        meta_train_person_ids=tuple(
            person_id for person_id, _, _ in meta_people
        ),
        validation_person_ids=tuple(
            person_id for person_id, _, _ in validation_people
        ),
        test_person_ids=tuple(
            person_id for person_id, _, _ in test_people
        ),
    )


def _ordered_records(
    records: Sequence[BenchmarkRecord],
) -> tuple[BenchmarkRecord, ...]:
    return tuple(
        sorted(
            records,
            key=lambda item: (
                item.person_id,
                item.observation.scenario.scenario_id,
                item.record_id,
            ),
        )
    )


def _design_matrix(
    records: Sequence[BenchmarkRecord],
) -> np.ndarray:
    ordered = _ordered_records(records)
    features = np.asarray(
        [
            record.observation.scenario.ordered_features(
                BENCHMARK_FEATURE_NAMES
            )
            for record in ordered
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        (features, np.ones(len(features), dtype=np.float64))
    )


def _choices(
    records: Sequence[BenchmarkRecord],
) -> np.ndarray:
    return np.asarray(
        [
            record.observation.actual_choice
            for record in _ordered_records(records)
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class LogisticChoiceModel:
    weights: tuple[float, ...]
    model_version: str

    def probabilities(
        self,
        records: Sequence[BenchmarkRecord],
    ) -> np.ndarray:
        return np.asarray(
            sigmoid(
                _design_matrix(records)
                @ np.asarray(self.weights, dtype=np.float64)
            ),
            dtype=np.float64,
        )


def _fit_population_model(
    dataset: BenchmarkDataset,
) -> LogisticChoiceModel:
    records = tuple(
        record
        for record in dataset.records
        if record.role == "meta_train"
    )
    dimension = len(BENCHMARK_FEATURE_NAMES) + 1
    weights, _ = fit_map_logistic(
        _design_matrix(records),
        _choices(records),
        np.zeros(dimension, dtype=np.float64),
        np.eye(dimension, dtype=np.float64) * 0.1,
    )
    return LogisticChoiceModel(
        weights=tuple(float(value) for value in weights),
        model_version="population-logistic-v1",
    )


def _fit_person_model(
    population: LogisticChoiceModel,
    records: Sequence[BenchmarkRecord],
    *,
    prior_precision: float,
) -> LogisticChoiceModel:
    if not records:
        return LogisticChoiceModel(
            weights=population.weights,
            model_version="personal-map-logistic-v1",
        )
    prior = np.asarray(population.weights, dtype=np.float64)
    weights, _ = fit_map_logistic(
        _design_matrix(records),
        _choices(records),
        prior,
        np.eye(len(prior), dtype=np.float64)
        * prior_precision,
    )
    return LogisticChoiceModel(
        weights=tuple(float(value) for value in weights),
        model_version="personal-map-logistic-v1",
    )


class _Adam:
    def __init__(self, parameters: Mapping[str, np.ndarray]) -> None:
        self.m = {
            name: np.zeros_like(value)
            for name, value in parameters.items()
        }
        self.v = {
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
            self.m[name] = (
                0.9 * self.m[name] + 0.1 * clipped
            )
            self.v[name] = (
                0.999 * self.v[name]
                + 0.001 * clipped * clipped
            )
            corrected_m = self.m[name] / (
                1.0 - 0.9**self.step
            )
            corrected_v = self.v[name] / (
                1.0 - 0.999**self.step
            )
            parameters[name] -= (
                learning_rate
                * corrected_m
                / (np.sqrt(corrected_v) + 1e-8)
            )


@dataclass
class PersonEmbeddingMLP:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    parameters: dict[str, np.ndarray]
    train_person_index: dict[str, int]
    model_version: str = "person-embedding-mlp-v1"

    @classmethod
    def fit(
        cls,
        dataset: BenchmarkDataset,
    ) -> PersonEmbeddingMLP:
        config = dataset.config
        records = _ordered_records(
            tuple(
                record
                for record in dataset.records
                if record.role == "meta_train"
            )
        )
        raw = _design_matrix(records)[:, :-1]
        mean = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        features = (raw - mean) / scale
        choices = _choices(records)
        person_index = {
            person_id: index
            for index, person_id in enumerate(
                dataset.meta_train_person_ids
            )
        }
        person_rows = np.asarray(
            [person_index[record.person_id] for record in records],
            dtype=np.int64,
        )
        rng = np.random.default_rng(config.seed + 991)
        feature_dimension = features.shape[1]
        hidden = config.neural_hidden_dimension
        embedding_dimension = config.person_embedding_dimension
        parameters = {
            "feature_weights": rng.normal(
                0.0,
                0.25 / np.sqrt(feature_dimension),
                size=(feature_dimension, hidden),
            ),
            "embedding_weights": rng.normal(
                0.0,
                0.25 / np.sqrt(embedding_dimension),
                size=(embedding_dimension, hidden),
            ),
            "hidden_bias": np.zeros(hidden, dtype=np.float64),
            "output_weights": rng.normal(
                0.0,
                0.20 / np.sqrt(hidden),
                size=hidden,
            ),
            "output_bias": np.zeros(1, dtype=np.float64),
            "embeddings": rng.normal(
                0.0,
                0.03,
                size=(
                    len(person_index),
                    embedding_dimension,
                ),
            ),
        }
        optimizer = _Adam(parameters)
        indices = np.arange(len(records))
        regularization = 1e-4
        for _ in range(config.neural_epochs):
            rng.shuffle(indices)
            for start in range(
                0,
                len(indices),
                config.neural_batch_size,
            ):
                batch = indices[
                    start : start + config.neural_batch_size
                ]
                x = features[batch]
                y = choices[batch]
                person_batch = person_rows[batch]
                embeddings = parameters["embeddings"][
                    person_batch
                ]
                hidden_pre = (
                    x @ parameters["feature_weights"]
                    + embeddings
                    @ parameters["embedding_weights"]
                    + parameters["hidden_bias"]
                )
                hidden_value = np.tanh(hidden_pre)
                probabilities = np.asarray(
                    sigmoid(
                        hidden_value
                        @ parameters["output_weights"]
                        + parameters["output_bias"][0]
                    )
                )
                logit_gradient = (
                    probabilities - y
                ) / len(batch)
                output_weights = parameters[
                    "output_weights"
                ].copy()
                embedding_weights = parameters[
                    "embedding_weights"
                ].copy()
                hidden_gradient = (
                    logit_gradient[:, None]
                    * output_weights[None, :]
                    * (1.0 - hidden_value * hidden_value)
                )
                embedding_gradient_rows = (
                    hidden_gradient @ embedding_weights.T
                )
                gradients = {
                    "feature_weights": (
                        x.T @ hidden_gradient
                        + regularization
                        * parameters["feature_weights"]
                    ),
                    "embedding_weights": (
                        embeddings.T @ hidden_gradient
                        + regularization
                        * parameters["embedding_weights"]
                    ),
                    "hidden_bias": np.sum(
                        hidden_gradient,
                        axis=0,
                    ),
                    "output_weights": (
                        hidden_value.T @ logit_gradient
                        + regularization
                        * parameters["output_weights"]
                    ),
                    "output_bias": np.asarray(
                        [np.sum(logit_gradient)]
                    ),
                    "embeddings": np.zeros_like(
                        parameters["embeddings"]
                    ),
                }
                np.add.at(
                    gradients["embeddings"],
                    person_batch,
                    embedding_gradient_rows,
                )
                gradients["embeddings"] += (
                    regularization
                    * parameters["embeddings"]
                    / len(records)
                )
                optimizer.update(
                    parameters,
                    gradients,
                    learning_rate=0.01,
                )
        return cls(
            feature_mean=mean,
            feature_scale=scale,
            parameters=parameters,
            train_person_index=person_index,
        )

    def adapt_embedding(
        self,
        records: Sequence[BenchmarkRecord],
    ) -> np.ndarray:
        dimension = self.parameters["embeddings"].shape[1]
        embedding = np.zeros(dimension, dtype=np.float64)
        if not records:
            return embedding
        ordered = _ordered_records(records)
        raw = _design_matrix(ordered)[:, :-1]
        features = (
            raw - self.feature_mean
        ) / self.feature_scale
        choices = _choices(ordered)
        first = np.zeros_like(embedding)
        second = np.zeros_like(embedding)
        embedding_weights = self.parameters[
            "embedding_weights"
        ]
        output_weights = self.parameters["output_weights"]
        for step in range(1, 81):
            hidden_pre = (
                features @ self.parameters["feature_weights"]
                + embedding @ embedding_weights
                + self.parameters["hidden_bias"]
            )
            hidden_value = np.tanh(hidden_pre)
            probabilities = np.asarray(
                sigmoid(
                    hidden_value @ output_weights
                    + self.parameters["output_bias"][0]
                )
            )
            logit_gradient = (
                probabilities - choices
            ) / len(ordered)
            hidden_gradient = (
                logit_gradient[:, None]
                * output_weights[None, :]
                * (1.0 - hidden_value * hidden_value)
            )
            gradient = (
                np.sum(
                    hidden_gradient @ embedding_weights.T,
                    axis=0,
                )
                + 1e-3 * embedding
            )
            first = 0.9 * first + 0.1 * gradient
            second = (
                0.999 * second + 0.001 * gradient * gradient
            )
            embedding -= (
                0.03
                * (first / (1.0 - 0.9**step))
                / (
                    np.sqrt(second / (1.0 - 0.999**step))
                    + 1e-8
                )
            )
        return embedding

    def probabilities(
        self,
        records: Sequence[BenchmarkRecord],
        embedding: np.ndarray,
    ) -> np.ndarray:
        raw = _design_matrix(records)[:, :-1]
        features = (
            raw - self.feature_mean
        ) / self.feature_scale
        hidden = np.tanh(
            features @ self.parameters["feature_weights"]
            + embedding @ self.parameters["embedding_weights"]
            + self.parameters["hidden_bias"]
        )
        return np.asarray(
            sigmoid(
                hidden @ self.parameters["output_weights"]
                + self.parameters["output_bias"][0]
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class BenchmarkMetric:
    baseline: str
    split: str
    support_size: int
    report: EvaluationReport

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "split": self.split,
            "support_size": self.support_size,
            **report_to_dict(self.report),
        }


@dataclass(frozen=True)
class PersonChoiceBenchmarkReport:
    dataset_id: str
    metrics: tuple[BenchmarkMetric, ...]
    record_count: int
    neural_trained: bool
    maximum_metric_recalculation_error: float
    interpretation: str = "synthetic_benchmark_only"
    non_claims: tuple[str, ...] = (
        "neural_representation_has_no_psychological_semantics",
        "real_person_validity_not_established",
        "causal_mechanism_not_identified",
    )
    required_next_baseline: str = (
        "support_set_hypernetwork_low_rank_adapter"
    )
    hypernetwork_must_beat: tuple[str, ...] = (
        "personal_map_logistic",
        "person_embedding_mlp",
    )
    report_version: str = "person-choice-benchmark-report-v1"

    def metric(
        self,
        baseline: str,
        split: str,
        support_size: int,
    ) -> EvaluationReport:
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
                "benchmark metric is absent or ambiguous"
            )
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_version": self.report_version,
            "dataset_id": self.dataset_id,
            "interpretation": self.interpretation,
            "non_claims": list(self.non_claims),
            "neural_trained": self.neural_trained,
            "metrics": [
                metric.to_dict() for metric in self.metrics
            ],
            "maximum_metric_recalculation_error": (
                self.maximum_metric_recalculation_error
            ),
            "required_next_baseline": (
                self.required_next_baseline
            ),
            "hypernetwork_must_beat": list(
                self.hypernetwork_must_beat
            ),
            "resource_usage": {
                "record_count": self.record_count,
                "trainable_neural_scalars": (
                    0
                    if not self.neural_trained
                    else "less_than_2000"
                ),
                "device": "cpu",
                "dependencies": ["numpy"],
            },
        }


def _metric(
    baseline: str,
    split: str,
    support_size: int,
    records: Sequence[BenchmarkRecord],
    probabilities: Sequence[float],
) -> tuple[BenchmarkMetric, float]:
    ordered = _ordered_records(records)
    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )
    observations = tuple(
        record.observation for record in ordered
    )
    report = evaluate_probability_array(
        observations,
        probability_array,
    )
    choices = _choices(ordered)
    clipped = np.clip(
        probability_array,
        1e-9,
        1.0 - 1e-9,
    )
    independent_nll = -float(
        np.mean(
            choices * np.log(clipped)
            + (1.0 - choices) * np.log(1.0 - clipped)
        )
    )
    return (
        BenchmarkMetric(
            baseline=baseline,
            split=split,
            support_size=support_size,
            report=report,
        ),
        abs(independent_nll - report.negative_log_likelihood),
    )


def run_person_choice_benchmark(
    dataset: BenchmarkDataset,
    *,
    train_neural: bool,
) -> PersonChoiceBenchmarkReport:
    population = _fit_population_model(dataset)
    neural = (
        PersonEmbeddingMLP.fit(dataset)
        if train_neural
        else None
    )
    metrics = []
    errors = []
    people = tuple(sorted(dataset.test_person_ids))
    for support_size in dataset.config.support_sizes:
        personal_models = {}
        neural_embeddings = {}
        for person_id in people:
            support = dataset.records_for(
                person_id,
                "support",
            )[:support_size]
            personal_models[person_id] = _fit_person_model(
                population,
                support,
                prior_precision=(
                    dataset.config.personal_prior_precision
                ),
            )
            if neural is not None:
                neural_embeddings[person_id] = (
                    neural.adapt_embedding(support)
                )
        wrong_models = {
            person_id: personal_models[
                people[(index + 1) % len(people)]
            ]
            for index, person_id in enumerate(people)
        }
        for split in (
            "scenario_test",
            "temporal_test",
            "ood_test",
        ):
            evaluation_records = tuple(
                record
                for person_id in people
                for record in dataset.records_for(
                    person_id,
                    split,
                )
            )
            ordered = _ordered_records(evaluation_records)
            population_probabilities = population.probabilities(
                ordered
            )
            personal_probabilities = np.concatenate(
                [
                    personal_models[person_id].probabilities(
                        dataset.records_for(person_id, split)
                    )
                    for person_id in people
                ]
            )
            wrong_probabilities = np.concatenate(
                [
                    wrong_models[person_id].probabilities(
                        dataset.records_for(person_id, split)
                    )
                    for person_id in people
                ]
            )
            for baseline, probabilities in (
                (
                    "population_logistic",
                    population_probabilities,
                ),
                (
                    "personal_map_logistic",
                    personal_probabilities,
                ),
                (
                    "wrong_person_logistic",
                    wrong_probabilities,
                ),
            ):
                entry, error = _metric(
                    baseline,
                    split,
                    support_size,
                    ordered,
                    probabilities,
                )
                metrics.append(entry)
                errors.append(error)
            if neural is not None:
                neural_probabilities = np.concatenate(
                    [
                        neural.probabilities(
                            dataset.records_for(
                                person_id,
                                split,
                            ),
                            neural_embeddings[person_id],
                        )
                        for person_id in people
                    ]
                )
                entry, error = _metric(
                    "person_embedding_mlp",
                    split,
                    support_size,
                    ordered,
                    neural_probabilities,
                )
                metrics.append(entry)
                errors.append(error)
    return PersonChoiceBenchmarkReport(
        dataset_id=dataset.dataset_id,
        metrics=tuple(metrics),
        record_count=len(dataset.records),
        neural_trained=train_neural,
        maximum_metric_recalculation_error=max(
            errors,
            default=0.0,
        ),
    )
