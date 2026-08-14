from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from .contracts import (
    Observation,
    PersonalAdapter,
    PersonRepresentation,
    Prediction,
    Scenario,
)


@dataclass(frozen=True)
class PopulationModel:
    weights: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    feature_names: tuple[str, ...]
    model_version: str

    def __post_init__(self) -> None:
        dimension = len(self.weights)
        if dimension == 0 or len(self.feature_names) != dimension:
            raise ValueError("population model feature dimensions do not match")
        if len(set(self.feature_names)) != dimension:
            raise ValueError("population model feature names must be unique")
        if not all(isfinite(float(value)) for value in self.weights):
            raise ValueError("population model weights must be finite")
        if len(self.covariance) != dimension:
            raise ValueError("population model covariance dimensions do not match")
        if any(
            len(row) != dimension
            or not all(isfinite(float(value)) for value in row)
            for row in self.covariance
        ):
            raise ValueError("population model covariance must be finite and square")


class PersonEncoder(ABC):
    @abstractmethod
    def fit(
        self,
        person_id: str,
        observations: Sequence[Observation],
        population_model: PopulationModel,
    ) -> PersonRepresentation:
        """Infer a person representation from verified source-task behavior."""


class AdapterGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        representation: PersonRepresentation,
        population_model: PopulationModel,
    ) -> PersonalAdapter:
        """Generate person-specific model parameters."""


class Predictor(ABC):
    @abstractmethod
    def predict(
        self,
        scenario: Scenario,
        population_model: PopulationModel,
        adapter: PersonalAdapter,
        *,
        parameter_covariance: tuple[tuple[float, ...], ...] | None = None,
    ) -> Prediction:
        """Predict a structured choice without a language model."""


class CognitiveModule(ABC):
    module_id: str
    module_version: str

    @abstractmethod
    def required_inputs(self) -> tuple[str, ...]:
        """Declare inputs without reading another module's private state."""

    @abstractmethod
    def diagnostics(self) -> dict[str, object]:
        """Return machine-readable diagnostics."""
