from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import Observation, Scenario
from .math_utils import sigmoid


FEATURE_NAMES = (
    "reward_gain",
    "loss_risk",
    "delay",
    "control",
    "fairness",
    "social_approval",
    "default_bias",
)


@dataclass(frozen=True)
class SyntheticPerson:
    person_id: str
    true_weights: tuple[float, ...]


def make_people(
    rng: np.random.Generator,
    count: int,
    heterogeneity_scale: float = 1.0,
) -> tuple[SyntheticPerson, ...]:
    if count < 2:
        raise ValueError("at least two people are required")
    if heterogeneity_scale < 0:
        raise ValueError("heterogeneity_scale must be non-negative")
    population_mean = np.asarray(
        [1.15, -1.25, -0.65, 0.85, 0.55, 0.35, 0.0],
        dtype=np.float64,
    )
    population_scale = np.asarray(
        [0.65, 0.70, 0.50, 0.70, 0.55, 0.50, 0.35],
        dtype=np.float64,
    )
    people = []
    for index in range(count):
        weights = population_mean + rng.normal(
            loc=0.0,
            scale=population_scale * heterogeneity_scale,
        )
        people.append(
            SyntheticPerson(
                person_id=f"person-{index:03d}",
                true_weights=tuple(float(value) for value in weights),
            )
        )
    return tuple(people)


def _sample_features(
    rng: np.random.Generator,
    condition: str,
) -> np.ndarray:
    base = rng.normal(0.0, 1.0, size=len(FEATURE_NAMES) - 1)
    if condition == "source":
        transform = np.asarray(
            [
                [1.0, 0.15, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.10, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.10, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.10],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        shifted = transform @ base
    elif condition == "target":
        transform = np.asarray(
            [
                [0.8, -0.25, 0.0, 0.15, 0.0, 0.0],
                [0.2, 0.9, 0.15, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.1, -0.20, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.9, 0.20, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.8, 0.25],
                [0.1, 0.0, 0.0, 0.0, -0.15, 1.0],
            ],
            dtype=np.float64,
        )
        shifted = transform @ base + np.asarray(
            [0.35, -0.20, 0.25, -0.30, 0.15, 0.20],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"unknown condition: {condition}")
    return np.concatenate([shifted, np.ones(1, dtype=np.float64)])


def generate_observations(
    rng: np.random.Generator,
    person: SyntheticPerson,
    count: int,
    condition: str,
) -> tuple[Observation, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    weights = np.asarray(person.true_weights, dtype=np.float64)
    observations = []
    for index in range(count):
        features = _sample_features(rng, condition)
        probability = float(sigmoid(features @ weights))
        choice = int(rng.random() < probability)
        confidence = 0.5 + abs(probability - 0.5)
        entropy = -(
            probability * np.log(max(probability, 1e-9))
            + (1.0 - probability) * np.log(max(1.0 - probability, 1e-9))
        )
        reaction_time_ms = 950.0 + 2100.0 * entropy + rng.normal(0.0, 80.0)
        observations.append(
            Observation(
                person_id=person.person_id,
                scenario=Scenario(
                    scenario_id=f"{person.person_id}-{condition}-{index:04d}",
                    features=tuple(float(value) for value in features),
                    feature_names=FEATURE_NAMES,
                    domain=f"synthetic_{condition}",
                    context={"condition": condition},
                ),
                actual_choice=choice,
                confidence=float(confidence),
                reaction_time_ms=float(max(reaction_time_ms, 100.0)),
                provenance="synthetic_ground_truth",
            )
        )
    return tuple(observations)


def generate_population_dataset(
    *,
    seed: int,
    person_count: int,
    source_trials: int,
    target_trials: int,
    heterogeneity_scale: float = 1.0,
) -> tuple[
    tuple[SyntheticPerson, ...],
    dict[str, tuple[Observation, ...]],
    dict[str, tuple[Observation, ...]],
]:
    rng = np.random.default_rng(seed)
    people = make_people(
        rng,
        person_count,
        heterogeneity_scale=heterogeneity_scale,
    )
    source: dict[str, tuple[Observation, ...]] = {}
    target: dict[str, tuple[Observation, ...]] = {}
    for person in people:
        source[person.person_id] = generate_observations(
            rng,
            person,
            source_trials,
            "source",
        )
        target[person.person_id] = generate_observations(
            rng,
            person,
            target_trials,
            "target",
        )
    return people, source, target


def _generate_misspecified_observations(
    rng: np.random.Generator,
    person: SyntheticPerson,
    count: int,
    condition: str,
) -> tuple[Observation, ...]:
    observations = []
    for index in range(count):
        features = _sample_features(rng, condition)
        score = (
            2.8 * features[0] * features[3]
            - 2.4 * features[1] * features[4]
            + 1.8 * (abs(features[2]) - 0.7)
        )
        probability = float(sigmoid(score))
        choice = int(rng.random() < probability)
        confidence = 0.5 + abs(probability - 0.5)
        observations.append(
            Observation(
                person_id=person.person_id,
                scenario=Scenario(
                    scenario_id=(
                        f"{person.person_id}-misspecified-"
                        f"{condition}-{index:04d}"
                    ),
                    features=tuple(float(value) for value in features),
                    feature_names=FEATURE_NAMES,
                    domain=f"synthetic_misspecified_{condition}",
                    context={
                        "condition": condition,
                        "mechanism": "nonlinear_interactions",
                    },
                ),
                actual_choice=choice,
                confidence=float(confidence),
                reaction_time_ms=1500.0,
                provenance="synthetic_ground_truth",
            )
        )
    return tuple(observations)


def generate_misspecified_dataset(
    *,
    seed: int,
    person_count: int,
    source_trials: int,
    target_trials: int,
) -> tuple[
    tuple[SyntheticPerson, ...],
    dict[str, tuple[Observation, ...]],
    dict[str, tuple[Observation, ...]],
]:
    rng = np.random.default_rng(seed)
    people = make_people(rng, person_count, heterogeneity_scale=0.0)
    source = {
        person.person_id: _generate_misspecified_observations(
            rng,
            person,
            source_trials,
            "source",
        )
        for person in people
    }
    target = {
        person.person_id: _generate_misspecified_observations(
            rng,
            person,
            target_trials,
            "target",
        )
        for person in people
    }
    return people, source, target
