from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

import numpy as np

from .contracts import (
    Observation,
    PersonalAdapter,
    PersonRepresentation,
    Prediction,
    Scenario,
)
from .interfaces import AdapterGenerator, PersonEncoder, PopulationModel, Predictor
from .ledger import EventRecord, VerificationAuthority
from .math_utils import (
    fit_map_logistic,
    logistic_normal_probability,
    sigmoid,
)


def _matrix_from_observations(
    observations: Sequence[Observation],
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    if not observations:
        raise ValueError("at least one observation is required")
    features = np.asarray(
        [
            item.scenario.ordered_features(feature_names)
            for item in observations
        ],
        dtype=np.float64,
    )
    choices = np.asarray(
        [item.actual_choice for item in observations],
        dtype=np.float64,
    )
    return features, choices


@dataclass
class PopulationPriorEstimator:
    feature_names: tuple[str, ...]
    l2_precision: float = 0.15
    initial_person_variance: float = 0.5
    min_person_variance: float = 1e-6
    max_person_variance: float = 4.0
    eb_iterations: int = 40
    eb_damping: float = 0.5
    eb_tolerance: float = 1e-5
    model_version: str = "population-laplace-eb-logit-v3"

    def fit(self, observations: Sequence[Observation]) -> PopulationModel:
        features, choices = _matrix_from_observations(
            observations,
            self.feature_names,
        )
        if features.shape[1] != len(self.feature_names):
            raise ValueError("feature_names do not match observation dimensions")
        if self.initial_person_variance <= 0:
            raise ValueError("initial_person_variance must be positive")
        if not 0 < self.min_person_variance < self.max_person_variance:
            raise ValueError("person variance bounds are invalid")
        if self.eb_iterations <= 0:
            raise ValueError("eb_iterations must be positive")
        if not 0 < self.eb_damping <= 1:
            raise ValueError("eb_damping must be between zero and one")
        dimension = features.shape[1]
        pooled_weights, _mean_estimation_covariance = fit_map_logistic(
            features,
            choices,
            np.zeros(dimension, dtype=np.float64),
            np.eye(dimension, dtype=np.float64) * self.l2_precision,
        )
        by_person: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            by_person[observation.person_id].append(observation)
        if len(by_person) < 2:
            raise ValueError(
                "at least two people are required to estimate person variance"
            )

        weights = pooled_weights
        between_person_variance = np.full(
            dimension,
            self.initial_person_variance,
            dtype=np.float64,
        )
        for _ in range(self.eb_iterations):
            posterior_means = []
            posterior_variances = []
            prior_precision = np.diag(
                1.0 / np.clip(
                    between_person_variance,
                    self.min_person_variance,
                    None,
                )
            )
            for person_observations in by_person.values():
                person_features, person_choices = _matrix_from_observations(
                    person_observations,
                    self.feature_names,
                )
                person_weights, person_covariance = fit_map_logistic(
                    person_features,
                    person_choices,
                    weights,
                    prior_precision,
                )
                posterior_means.append(person_weights)
                posterior_variances.append(np.diag(person_covariance))

            stacked_means = np.asarray(posterior_means, dtype=np.float64)
            stacked_variances = np.asarray(
                posterior_variances,
                dtype=np.float64,
            )
            proposed_weights = np.mean(stacked_means, axis=0)
            proposed_variance = np.mean(
                stacked_variances
                + (stacked_means - proposed_weights) ** 2,
                axis=0,
            )
            proposed_variance = np.clip(
                proposed_variance,
                self.min_person_variance,
                self.max_person_variance,
            )
            new_weights = (
                (1.0 - self.eb_damping) * weights
                + self.eb_damping * proposed_weights
            )
            new_variance = (
                (1.0 - self.eb_damping) * between_person_variance
                + self.eb_damping * proposed_variance
            )
            change = max(
                float(np.max(np.abs(new_weights - weights))),
                float(
                    np.max(
                        np.abs(
                            np.log(new_variance)
                            - np.log(between_person_variance)
                        )
                    )
                ),
            )
            weights = new_weights
            between_person_variance = new_variance
            if change < self.eb_tolerance:
                break
        person_covariance = np.diag(between_person_variance)
        return PopulationModel(
            weights=tuple(float(value) for value in weights),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in person_covariance
            ),
            feature_names=self.feature_names,
            model_version=self.model_version,
        )


@dataclass
class MapPersonEncoder(PersonEncoder):
    prior_precision_scale: float = 1.0
    representation_version: str = "map-person-v1"

    def fit(
        self,
        person_id: str,
        observations: Sequence[Observation],
        population_model: PopulationModel,
    ) -> PersonRepresentation:
        person_observations = [
            observation
            for observation in observations
            if observation.person_id == person_id
        ]
        if not person_observations:
            raise ValueError(f"no observations found for person {person_id}")
        features, choices = _matrix_from_observations(
            person_observations,
            population_model.feature_names,
        )
        prior_mean = np.asarray(population_model.weights, dtype=np.float64)
        population_covariance = np.asarray(
            population_model.covariance,
            dtype=np.float64,
        )
        regularized_covariance = population_covariance + np.eye(
            population_covariance.shape[0],
            dtype=np.float64,
        ) * 1e-4
        prior_precision = (
            np.linalg.inv(regularized_covariance) * self.prior_precision_scale
        )
        weights, covariance = fit_map_logistic(
            features,
            choices,
            prior_mean,
            prior_precision,
        )
        return PersonRepresentation(
            person_id=person_id,
            latent_mean=tuple(float(value) for value in weights),
            covariance=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            representation_version=self.representation_version,
            observation_count=len(person_observations),
            feature_names=population_model.feature_names,
        )


@dataclass
class IdentityAdapterGenerator(AdapterGenerator):
    adapter_version: str = "identity-adapter-v1"

    def generate(
        self,
        representation: PersonRepresentation,
        population_model: PopulationModel,
    ) -> PersonalAdapter:
        if representation.feature_names != population_model.feature_names:
            raise ValueError("representation and population model features differ")
        delta = np.asarray(representation.latent_mean) - np.asarray(
            population_model.weights
        )
        return PersonalAdapter(
            person_id=representation.person_id,
            delta_weights=tuple(float(value) for value in delta),
            adapter_version=self.adapter_version,
            representation_version=representation.representation_version,
        )


@dataclass
class DecisionIntegrator(Predictor):
    model_version: str = "decision-integrator-v1"

    def predict(
        self,
        scenario: Scenario,
        population_model: PopulationModel,
        adapter: PersonalAdapter,
        *,
        parameter_covariance: tuple[tuple[float, ...], ...] | None = None,
    ) -> Prediction:
        if len(adapter.delta_weights) != len(population_model.weights):
            raise ValueError("adapter and model feature dimensions differ")
        weights = np.asarray(population_model.weights) + np.asarray(
            adapter.delta_weights
        )
        features = np.asarray(
            scenario.ordered_features(population_model.feature_names)
        )
        logit_mean = float(features @ weights)
        lower = None
        upper = None
        logit_standard_deviation = None
        active_modules = ["population_prior", "person_adapter"]
        if parameter_covariance is None:
            probability = float(sigmoid(logit_mean))
        else:
            covariance = np.asarray(
                parameter_covariance,
                dtype=np.float64,
            )
            expected_shape = (len(weights), len(weights))
            if covariance.shape != expected_shape:
                raise ValueError("parameter covariance dimensions differ")
            logit_variance = max(float(features @ covariance @ features), 0.0)
            logit_standard_deviation = float(np.sqrt(logit_variance))
            probability = float(
                logistic_normal_probability(
                    logit_mean,
                    logit_variance,
                )
            )
            lower = float(
                sigmoid(logit_mean - 1.96 * logit_standard_deviation)
            )
            upper = float(
                sigmoid(logit_mean + 1.96 * logit_standard_deviation)
            )
            active_modules.append("parameter_uncertainty")
        return Prediction(
            scenario_id=scenario.scenario_id,
            person_id=adapter.person_id,
            probability_option_1=probability,
            predicted_choice=int(probability >= 0.5),
            active_modules=tuple(active_modules),
            model_version=self.model_version,
            probability_lower_95=lower,
            probability_upper_95=upper,
            logit_standard_deviation=logit_standard_deviation,
        )

    def predict_population(
        self,
        scenario: Scenario,
        population_model: PopulationModel,
        person_id: str = "population",
    ) -> Prediction:
        adapter = PersonalAdapter(
            person_id=person_id,
            delta_weights=tuple(0.0 for _ in population_model.weights),
            adapter_version="population-zero-adapter",
            representation_version="population",
        )
        return self.predict(scenario, population_model, adapter)


@dataclass(frozen=True)
class UpdatedPersonModel:
    representation: PersonRepresentation
    adapter: PersonalAdapter
    observations: tuple[Observation, ...]


@dataclass
class ModelUpdater:
    encoder: PersonEncoder
    adapter_generator: AdapterGenerator

    def update(
        self,
        person_id: str,
        existing_observations: Sequence[Observation],
        outcome: EventRecord,
        authority: VerificationAuthority,
        population_model: PopulationModel,
    ) -> UpdatedPersonModel:
        authority.verify(outcome)
        observation = outcome.observation
        if observation.person_id != person_id:
            raise ValueError("outcome person_id does not match the model")
        if any(
            observation.person_id != person_id
            for observation in existing_observations
        ):
            raise ValueError(
                "existing_observations must contain only the target person"
            )
        combined = tuple(existing_observations) + (observation,)
        representation = self.encoder.fit(person_id, combined, population_model)
        adapter = self.adapter_generator.generate(
            representation,
            population_model,
        )
        return UpdatedPersonModel(
            representation=representation,
            adapter=adapter,
            observations=combined,
        )
