from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from math import isfinite, pi, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .contracts import Scenario
from .interfaces import CognitiveModule
from .ledger import EventLedger, VerificationAuthority
from .math_utils import logistic_normal_probability, sigmoid
from .storage import (
    PersonModelBundle,
    scenario_design_hash,
    trial_key_hash,
)


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error


def _scenario_payload(scenario: Scenario) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "features": list(scenario.features),
        "feature_names": list(scenario.feature_names),
        "options": list(scenario.options),
        "domain": scenario.domain,
        "context": dict(sorted(scenario.context.items())),
    }


def _scenario_from_dict(data: dict[str, object]) -> Scenario:
    raw_features = data["features"]
    if isinstance(raw_features, dict):
        feature_names = tuple(str(name) for name in raw_features)
        features = tuple(float(value) for value in raw_features.values())
    else:
        feature_names = tuple(
            str(name) for name in data["feature_names"]
        )
        features = tuple(float(value) for value in raw_features)
    return Scenario(
        scenario_id=str(data["scenario_id"]),
        features=features,
        feature_names=feature_names,
        options=tuple(str(value) for value in data.get("options", ("A", "B"))),
        domain=str(data.get("domain", "structured_choice")),
        context={
            str(name): str(value)
            for name, value in dict(data.get("context", {})).items()
        },
    )


class ActiveExperimentRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "active experiment refused: " + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class ActiveExperimentConfig:
    minimum_information_gain: float = 1e-12
    quadrature_points: int = 48
    model_version: str = "gaussian-mutual-information-v2"

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_information_gain)
            or self.minimum_information_gain < 1e-12
        ):
            raise ValueError(
                "minimum_information_gain must be at least 1e-12"
            )
        if not 16 <= self.quadrature_points <= 128:
            raise ValueError(
                "quadrature_points must be between 16 and 128"
            )
        if self.model_version != "gaussian-mutual-information-v2":
            raise ValueError(
                "unsupported active experiment model_version"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_information_gain": (
                self.minimum_information_gain
            ),
            "quadrature_points": self.quadrature_points,
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class GaussianBinaryInformation:
    predictive_probability: float
    mutual_information: float
    logit_choice_covariance: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.predictive_probability <= 1.0:
            raise ValueError(
                "predictive_probability must be between zero and one"
            )
        if (
            not isfinite(self.mutual_information)
            or self.mutual_information < 0
            or not isfinite(self.logit_choice_covariance)
        ):
            raise ValueError(
                "Gaussian binary information values are invalid"
            )


def _binary_entropy(probability: np.ndarray | float):
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return -(
        clipped * np.log(clipped)
        + (1.0 - clipped) * np.log(1.0 - clipped)
    )


def gaussian_binary_information(
    *,
    logit_mean: float,
    logit_variance: float,
    quadrature_points: int = 48,
) -> GaussianBinaryInformation:
    if (
        not isfinite(logit_mean)
        or not isfinite(logit_variance)
        or logit_variance < 0
    ):
        raise ValueError("logit moments are invalid")
    if not 16 <= quadrature_points <= 128:
        raise ValueError(
            "quadrature_points must be between 16 and 128"
        )
    if logit_variance <= 1e-15:
        probability = float(sigmoid(logit_mean))
        return GaussianBinaryInformation(
            predictive_probability=probability,
            mutual_information=0.0,
            logit_choice_covariance=0.0,
        )
    nodes, weights = np.polynomial.hermite.hermgauss(
        quadrature_points
    )
    logits = (
        logit_mean
        + sqrt(2.0 * logit_variance) * nodes
    )
    probabilities = np.asarray(sigmoid(logits), dtype=np.float64)
    normalized_weights = weights / sqrt(pi)
    predictive = float(normalized_weights @ probabilities)
    conditional_entropy = float(
        normalized_weights @ _binary_entropy(probabilities)
    )
    mutual_information = max(
        float(_binary_entropy(predictive)) - conditional_entropy,
        0.0,
    )
    covariance = float(
        normalized_weights
        @ ((logits - logit_mean) * probabilities)
    )
    return GaussianBinaryInformation(
        predictive_probability=predictive,
        mutual_information=mutual_information,
        logit_choice_covariance=covariance,
    )


@dataclass(frozen=True)
class ExperimentSelection:
    rank: int
    scenario: Scenario
    expected_choice_probability: float
    expected_information_gain: float
    cumulative_information_gain: float

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("active experiment rank must be positive")
        if not 0.0 <= self.expected_choice_probability <= 1.0:
            raise ValueError(
                "active experiment probability must be between zero and one"
            )
        if (
            not isfinite(self.expected_information_gain)
            or self.expected_information_gain < 0
            or not isfinite(self.cumulative_information_gain)
            or self.cumulative_information_gain < 0
        ):
            raise ValueError(
                "active experiment information gain is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "scenario": _scenario_payload(self.scenario),
            "expected_choice_probability": (
                self.expected_choice_probability
            ),
            "expected_information_gain": (
                self.expected_information_gain
            ),
            "cumulative_information_gain": (
                self.cumulative_information_gain
            ),
        }


@dataclass(frozen=True)
class ActiveExperimentPlan:
    base_model_id: str
    predictive_model_id: str
    predictive_model_version: str
    parameter_dimension: int
    person_id: str
    created_at: str
    candidate_pool_hash: str
    candidate_count: int
    selection_count: int
    config: ActiveExperimentConfig
    selections: tuple[ExperimentSelection, ...]
    prior_log_determinant: float
    expected_posterior_log_determinant: float
    expected_posterior_covariance: tuple[tuple[float, ...], ...]
    total_expected_information_gain: float
    expected_covariance_entropy_reduction: float
    verifier_id: str
    artifact_version: str = "pcfm-active-experiment-v3"
    plan_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    @property
    def selection_mode(self) -> str:
        if self.selection_count == 1:
            return "adaptive_single_step"
        return "outcome_blind_batch_approximation"

    def __post_init__(self) -> None:
        _require_digest(self.base_model_id, "base_model_id")
        _require_digest(
            self.predictive_model_id,
            "predictive_model_id",
        )
        _require_digest(self.candidate_pool_hash, "candidate_pool_hash")
        _parse_timestamp(self.created_at, "created_at")
        if (
            not self.person_id
            or not self.verifier_id
            or not self.predictive_model_version
        ):
            raise ValueError("active experiment identity is required")
        if self.parameter_dimension <= 0:
            raise ValueError(
                "active experiment parameter dimension must be positive"
            )
        if self.candidate_count <= 0:
            raise ValueError(
                "active experiment candidate_count must be positive"
            )
        if (
            self.selection_count <= 0
            or self.selection_count != len(self.selections)
            or self.selection_count > self.candidate_count
        ):
            raise ValueError(
                "active experiment selection count is invalid"
            )
        if tuple(item.rank for item in self.selections) != tuple(
            range(1, self.selection_count + 1)
        ):
            raise ValueError(
                "active experiment ranks must be consecutive"
            )
        selected_ids = tuple(
            item.scenario.scenario_id for item in self.selections
        )
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError(
                "active experiment selections must be unique"
            )
        running_gain = 0.0
        for item in self.selections:
            running_gain += item.expected_information_gain
            if not np.isclose(
                item.cumulative_information_gain,
                running_gain,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    "active experiment cumulative gain is inconsistent"
                )
        if not np.isclose(
            self.total_expected_information_gain,
            running_gain,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "active experiment total gain is inconsistent"
            )
        covariance = np.asarray(
            self.expected_posterior_covariance,
            dtype=np.float64,
        )
        if (
            covariance.shape
            != (self.parameter_dimension, self.parameter_dimension)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, atol=1e-10)
            or np.min(np.linalg.eigvalsh(covariance)) <= 0
        ):
            raise ValueError(
                "active experiment posterior covariance is invalid"
            )
        sign, posterior_log_determinant = np.linalg.slogdet(covariance)
        metrics = (
            self.prior_log_determinant,
            self.expected_posterior_log_determinant,
            self.total_expected_information_gain,
            self.expected_covariance_entropy_reduction,
        )
        if sign <= 0 or not all(isfinite(value) for value in metrics):
            raise ValueError(
                "active experiment determinant metrics are invalid"
            )
        if not np.isclose(
            self.expected_posterior_log_determinant,
            posterior_log_determinant,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "active experiment posterior determinant is inconsistent"
            )
        determinant_gain = 0.5 * (
            self.prior_log_determinant
            - self.expected_posterior_log_determinant
        )
        if not np.isclose(
            self.expected_covariance_entropy_reduction,
            determinant_gain,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError(
                "active experiment covariance entropy is inconsistent"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported active experiment signature method"
            )
        expected = self.digest()
        if self.plan_id:
            _require_digest(self.plan_id, "plan_id")
            if self.plan_id != expected:
                raise ValueError(
                    "active experiment plan_id does not match content"
                )
        else:
            object.__setattr__(self, "plan_id", expected)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "base_model_id": self.base_model_id,
            "predictive_model_id": self.predictive_model_id,
            "predictive_model_version": (
                self.predictive_model_version
            ),
            "parameter_dimension": self.parameter_dimension,
            "person_id": self.person_id,
            "created_at": self.created_at,
            "candidate_pool_hash": self.candidate_pool_hash,
            "candidate_count": self.candidate_count,
            "selection_count": self.selection_count,
            "selection_mode": self.selection_mode,
            "config": self.config.to_dict(),
            "selections": [
                selection.to_dict() for selection in self.selections
            ],
            "prior_log_determinant": self.prior_log_determinant,
            "expected_posterior_log_determinant": (
                self.expected_posterior_log_determinant
            ),
            "expected_posterior_covariance": [
                list(row)
                for row in self.expected_posterior_covariance
            ],
            "total_expected_information_gain": (
                self.total_expected_information_gain
            ),
            "expected_covariance_entropy_reduction": (
                self.expected_covariance_entropy_reduction
            ),
            "verifier_id": self.verifier_id,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "plan_id": self.plan_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("active experiment plan is unsigned")
        authority.verify_payload(
            self.signed_payload(),
            self.verifier_id,
            self.signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.signed_payload(),
            "signature": self.signature,
        }


@dataclass(frozen=True)
class ActiveExperimentUpdate:
    bundle: PersonModelBundle
    ledger: EventLedger
    plan_id: str
    result_event_ids: tuple[str, ...]
    result_data_hash: str
    prior_log_determinant: float
    posterior_log_determinant: float
    realized_covariance_entropy_reduction: float

    def __post_init__(self) -> None:
        _require_digest(self.plan_id, "plan_id")
        _require_digest(self.result_data_hash, "result_data_hash")
        if (
            not self.result_event_ids
            or len(set(self.result_event_ids))
            != len(self.result_event_ids)
        ):
            raise ValueError(
                "active experiment result event ids are invalid"
            )
        metrics = (
            self.prior_log_determinant,
            self.posterior_log_determinant,
            self.realized_covariance_entropy_reduction,
        )
        if not all(isfinite(value) for value in metrics):
            raise ValueError(
                "active experiment update metrics are invalid"
            )
        expected = 0.5 * (
            self.prior_log_determinant
            - self.posterior_log_determinant
        )
        if not np.isclose(
            self.realized_covariance_entropy_reduction,
            expected,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "active experiment realized reduction is inconsistent"
            )
        if self.plan_id not in self.bundle.manifest.experiment_plan_ids:
            raise ValueError(
                "updated model does not bind the experiment plan"
            )


def _normalize_candidates(
    candidates: Sequence[Scenario],
    feature_names: tuple[str, ...],
) -> tuple[Scenario, ...]:
    normalized = []
    for scenario in candidates:
        ordered = scenario.ordered_features(feature_names)
        normalized.append(
            Scenario(
                scenario_id=scenario.scenario_id,
                features=ordered,
                feature_names=feature_names,
                options=scenario.options,
                domain=scenario.domain,
                context=dict(scenario.context),
            )
        )
    return tuple(
        sorted(normalized, key=lambda scenario: scenario.scenario_id)
    )


def _validate_candidate_pool(
    bundle: PersonModelBundle,
    candidates: Sequence[Scenario],
    *,
    created_at: str,
    selection_count: int,
    require_passed_base: bool,
) -> tuple[Scenario, ...]:
    _parse_timestamp(created_at, "created_at")
    reasons = []
    if (
        require_passed_base
        and bundle.manifest.validation.status != "passed"
    ):
        reasons.append(
            "base_model_validation_"
            + bundle.manifest.validation.status
        )
    if not candidates:
        reasons.append("candidate_pool_empty")
    if any(not isinstance(item, Scenario) for item in candidates):
        reasons.append("candidate_not_scenario")
    if reasons:
        raise ActiveExperimentRefusedError(reasons)

    typed_candidates = tuple(candidates)
    if selection_count <= 0:
        reasons.append("selection_count_not_positive")
    if selection_count > len(typed_candidates):
        reasons.append("selection_count_exceeds_candidates")
    scenario_ids = tuple(
        scenario.scenario_id for scenario in typed_candidates
    )
    if len(set(scenario_ids)) != len(scenario_ids):
        reasons.append("candidate_scenario_ids_not_unique")
    lineage = set(bundle.manifest.lineage_trial_hashes)
    if lineage & {
        trial_key_hash(bundle.manifest.person_id, scenario_id)
        for scenario_id in scenario_ids
    }:
        reasons.append("candidate_reuses_model_scenario")
    lineage_designs = set(bundle.manifest.lineage_design_hashes)
    if lineage_designs & {
        scenario_design_hash(
            bundle.manifest.person_id,
            scenario,
        )
        for scenario in typed_candidates
    }:
        reasons.append("candidate_reuses_model_design")

    normalized = []
    for scenario in typed_candidates:
        try:
            ordered = scenario.ordered_features(
                bundle.population_model.feature_names
            )
        except ValueError:
            reasons.append("candidate_feature_schema_mismatch")
            continue
        normalized_scenario = Scenario(
            scenario_id=scenario.scenario_id,
            features=ordered,
            feature_names=bundle.population_model.feature_names,
            options=scenario.options,
            domain=scenario.domain,
            context=dict(scenario.context),
        )
        assessment = bundle.manifest.applicability_profile.assess(
            normalized_scenario,
            prediction_at=created_at,
        )
        if assessment.reasons:
            reasons.append("candidate_outside_applicability")
        if assessment.warnings:
            reasons.append("candidate_cross_domain_unvalidated")
        normalized.append(normalized_scenario)
    design_hashes = tuple(
        scenario_design_hash(
            bundle.manifest.person_id,
            scenario,
        )
        for scenario in normalized
    )
    if len(set(design_hashes)) != len(design_hashes):
        reasons.append("candidate_designs_not_unique")
    if reasons:
        raise ActiveExperimentRefusedError(reasons)
    return tuple(
        sorted(
            normalized,
            key=lambda scenario: scenario.scenario_id,
        )
    )


def _deployed_gaussian_probability(
    logit_mean: float,
    logit_variance: float,
) -> float:
    return float(
        logistic_normal_probability(
            logit_mean,
            logit_variance,
        )
    )


def _build_gaussian_view_plan(
    *,
    bundle: PersonModelBundle,
    normalized_candidates: tuple[Scenario, ...],
    design_vectors: dict[str, np.ndarray],
    parameter_mean: np.ndarray,
    parameter_covariance: np.ndarray,
    predictive_model_id: str,
    predictive_model_version: str,
    authority: VerificationAuthority,
    verifier_id: str,
    created_at: str,
    selection_count: int,
    config: ActiveExperimentConfig,
) -> ActiveExperimentPlan:
    _require_digest(predictive_model_id, "predictive_model_id")
    mean = np.asarray(parameter_mean, dtype=np.float64)
    covariance = np.asarray(
        parameter_covariance,
        dtype=np.float64,
    )
    covariance = 0.5 * (covariance + covariance.T)
    dimension = len(mean)
    candidate_ids = {
        scenario.scenario_id for scenario in normalized_candidates
    }
    if (
        mean.ndim != 1
        or not np.all(np.isfinite(mean))
        or covariance.shape != (dimension, dimension)
        or not np.all(np.isfinite(covariance))
        or np.min(np.linalg.eigvalsh(covariance)) <= 0
        or set(design_vectors) != candidate_ids
        or any(
            np.asarray(vector).shape != (dimension,)
            or not np.all(np.isfinite(vector))
            for vector in design_vectors.values()
        )
    ):
        raise ActiveExperimentRefusedError(
            ("gaussian_predictive_view_invalid",)
        )
    prior_sign, prior_log_determinant = np.linalg.slogdet(
        covariance
    )
    if prior_sign <= 0:
        raise ActiveExperimentRefusedError(
            ("parameter_covariance_not_positive_definite",)
        )

    remaining = list(normalized_candidates)
    selections = []
    cumulative_gain = 0.0
    for rank in range(1, selection_count + 1):
        scored = []
        for scenario in remaining:
            design = np.asarray(
                design_vectors[scenario.scenario_id],
                dtype=np.float64,
            )
            logit_mean = float(design @ mean)
            projected_variance = max(
                float(design @ covariance @ design),
                0.0,
            )
            information = gaussian_binary_information(
                logit_mean=logit_mean,
                logit_variance=projected_variance,
                quadrature_points=config.quadrature_points,
            )
            deployed_probability = _deployed_gaussian_probability(
                logit_mean,
                projected_variance,
            )
            scored.append(
                (
                    -information.mutual_information,
                    scenario.scenario_id,
                    scenario,
                    deployed_probability,
                    information.predictive_probability,
                    information.logit_choice_covariance,
                    projected_variance,
                )
            )
        (
            negative_gain,
            _scenario_id,
            selected,
            deployed_probability,
            quadrature_probability,
            logit_choice_covariance,
            projected_variance,
        ) = min(scored, key=lambda item: (item[0], item[1]))
        gain = -negative_gain
        if gain < config.minimum_information_gain:
            raise ActiveExperimentRefusedError(
                ("insufficient_expected_information_gain",)
            )
        design = np.asarray(
            design_vectors[selected.scenario_id],
            dtype=np.float64,
        )
        covariance_design = covariance @ design
        outcome_variance = (
            quadrature_probability
            * (1.0 - quadrature_probability)
        )
        if projected_variance > 1e-15 and outcome_variance > 1e-15:
            parameter_choice_covariance = (
                covariance_design
                * logit_choice_covariance
                / projected_variance
            )
            covariance = covariance - (
                np.outer(
                    parameter_choice_covariance,
                    parameter_choice_covariance,
                )
                / outcome_variance
            )
        covariance = 0.5 * (covariance + covariance.T)
        if (
            not np.all(np.isfinite(covariance))
            or np.min(np.linalg.eigvalsh(covariance)) <= 0
        ):
            raise ActiveExperimentRefusedError(
                ("expected_posterior_covariance_invalid",)
            )
        cumulative_gain += gain
        selections.append(
            ExperimentSelection(
                rank=rank,
                scenario=selected,
                expected_choice_probability=deployed_probability,
                expected_information_gain=gain,
                cumulative_information_gain=cumulative_gain,
            )
        )
        remaining.remove(selected)

    posterior_sign, posterior_log_determinant = np.linalg.slogdet(
        covariance
    )
    if posterior_sign <= 0:
        raise ActiveExperimentRefusedError(
            ("expected_posterior_covariance_invalid",)
        )
    candidate_pool_hash = _digest_json(
        [
            _scenario_payload(scenario)
            for scenario in normalized_candidates
        ]
    )
    covariance_entropy_reduction = 0.5 * (
        float(prior_log_determinant)
        - float(posterior_log_determinant)
    )
    unsigned = ActiveExperimentPlan(
        base_model_id=bundle.manifest.model_id,
        predictive_model_id=predictive_model_id,
        predictive_model_version=predictive_model_version,
        parameter_dimension=dimension,
        person_id=bundle.manifest.person_id,
        created_at=created_at,
        candidate_pool_hash=candidate_pool_hash,
        candidate_count=len(normalized_candidates),
        selection_count=selection_count,
        config=config,
        selections=tuple(selections),
        prior_log_determinant=float(prior_log_determinant),
        expected_posterior_log_determinant=float(
            posterior_log_determinant
        ),
        expected_posterior_covariance=tuple(
            tuple(float(value) for value in row)
            for row in covariance
        ),
        total_expected_information_gain=cumulative_gain,
        expected_covariance_entropy_reduction=(
            covariance_entropy_reduction
        ),
        verifier_id=verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        verifier_id,
    )
    return replace(unsigned, signature=signature)


@dataclass
class ActiveExperimentPlanner(CognitiveModule):
    config: ActiveExperimentConfig = ActiveExperimentConfig()
    module_id: str = "bayesian-active-experiment-planner"
    module_version: str = "gaussian-mutual-information-v2"

    def required_inputs(self) -> tuple[str, ...]:
        return (
            "validated_person_model",
            "candidate_scenarios_without_outcomes",
            "planning_timestamp",
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": "implemented",
            "objective": "posterior_predictive_mutual_information",
            "outcomes_used_for_selection": False,
            "config": self.config.to_dict(),
        }

    def plan(
        self,
        bundle: PersonModelBundle,
        candidates: Sequence[Scenario],
        authority: VerificationAuthority,
        *,
        verifier_id: str,
        created_at: str,
        selection_count: int,
    ) -> ActiveExperimentPlan:
        normalized_candidates = _validate_candidate_pool(
            bundle,
            candidates,
            created_at=created_at,
            selection_count=selection_count,
            require_passed_base=True,
        )
        parameter_mean = np.asarray(
            bundle.representation.latent_mean,
            dtype=np.float64,
        )
        parameter_covariance = np.asarray(
            bundle.representation.covariance,
            dtype=np.float64,
        )
        design_vectors = {
            scenario.scenario_id: np.asarray(
                scenario.features,
                dtype=np.float64,
            )
            for scenario in normalized_candidates
        }
        return _build_gaussian_view_plan(
            bundle=bundle,
            normalized_candidates=normalized_candidates,
            design_vectors=design_vectors,
            parameter_mean=parameter_mean,
            parameter_covariance=parameter_covariance,
            predictive_model_id=bundle.manifest.model_id,
            predictive_model_version="stable-person-model-v1",
            authority=authority,
            verifier_id=verifier_id,
            created_at=created_at,
            selection_count=selection_count,
            config=self.config,
        )


def create_active_experiment_plan(
    bundle: PersonModelBundle,
    candidates: Sequence[Scenario],
    authority: VerificationAuthority,
    *,
    verifier_id: str,
    created_at: str,
    selection_count: int,
    config: ActiveExperimentConfig | None = None,
) -> ActiveExperimentPlan:
    planner = ActiveExperimentPlanner(
        config=config or ActiveExperimentConfig()
    )
    return planner.plan(
        bundle,
        candidates,
        authority,
        verifier_id=verifier_id,
        created_at=created_at,
        selection_count=selection_count,
    )


def create_next_active_experiment_plan(
    bundle: PersonModelBundle,
    candidates: Sequence[Scenario],
    authority: VerificationAuthority,
    *,
    verifier_id: str,
    created_at: str,
    config: ActiveExperimentConfig | None = None,
) -> ActiveExperimentPlan:
    """Plan one experiment so the next choice can use its outcome."""
    return create_active_experiment_plan(
        bundle,
        candidates,
        authority,
        verifier_id=verifier_id,
        created_at=created_at,
        selection_count=1,
        config=config,
    )


def verify_active_experiment_plan(
    bundle: PersonModelBundle,
    candidates: Sequence[Scenario],
    authority: VerificationAuthority,
    plan: ActiveExperimentPlan,
) -> None:
    try:
        plan.verify(authority)
    except ValueError as error:
        raise ActiveExperimentRefusedError(
            ("active_experiment_signature_invalid",)
        ) from error
    try:
        recomputed = create_active_experiment_plan(
            bundle,
            candidates,
            authority,
            verifier_id=plan.verifier_id,
            created_at=plan.created_at,
            selection_count=plan.selection_count,
            config=plan.config,
        )
    except (ValueError, ActiveExperimentRefusedError) as error:
        raise ActiveExperimentRefusedError(
            ("active_experiment_derivation_mismatch",)
        ) from error
    if recomputed != plan:
        raise ActiveExperimentRefusedError(
            ("active_experiment_derivation_mismatch",)
        )


def verify_active_experiment_results(
    plan: ActiveExperimentPlan,
    ledger: EventLedger,
    authority: VerificationAuthority,
) -> EventLedger:
    try:
        plan.verify(authority)
    except ValueError as error:
        raise ActiveExperimentRefusedError(
            ("active_experiment_signature_invalid",)
        ) from error
    try:
        verified = EventLedger.verify(ledger.records, authority)
    except ValueError as error:
        raise ActiveExperimentRefusedError(
            ("experiment_result_signature_invalid",)
        ) from error
    reasons = []
    if len(verified.records) != plan.selection_count:
        reasons.append("experiment_result_count_mismatch")
    if any(
        record.observation.person_id != plan.person_id
        for record in verified.records
    ):
        reasons.append("experiment_result_person_mismatch")
    planned = {
        item.scenario.scenario_id: item.scenario
        for item in plan.selections
    }
    actual_ids = {
        record.observation.scenario.scenario_id
        for record in verified.records
    }
    if actual_ids != set(planned):
        reasons.append("experiment_result_scenario_set_mismatch")
    created_at = _parse_timestamp(plan.created_at, "created_at")
    if any(
        _parse_timestamp(record.observed_at, "observed_at")
        <= created_at
        for record in verified.records
    ):
        reasons.append("experiment_result_precedes_plan")
    for record in verified.records:
        observed = record.observation.scenario
        expected = planned.get(observed.scenario_id)
        if expected is None:
            continue
        try:
            normalized = Scenario(
                scenario_id=observed.scenario_id,
                features=observed.ordered_features(
                    expected.feature_names
                ),
                feature_names=expected.feature_names,
                options=observed.options,
                domain=observed.domain,
                context=dict(observed.context),
            )
        except ValueError:
            reasons.append("experiment_result_scenario_mismatch")
            continue
        if _scenario_payload(normalized) != _scenario_payload(expected):
            reasons.append("experiment_result_scenario_mismatch")
    if reasons:
        raise ActiveExperimentRefusedError(reasons)
    return verified


def apply_active_experiment_results(
    bundle: PersonModelBundle,
    training_ledger: EventLedger,
    applicability_ledger: EventLedger,
    future_validation_ledger: EventLedger,
    candidates: Sequence[Scenario],
    plan: ActiveExperimentPlan,
    result_ledger: EventLedger,
    authority: VerificationAuthority,
) -> ActiveExperimentUpdate:
    if plan.predictive_model_id != bundle.manifest.model_id:
        raise ActiveExperimentRefusedError(
            ("composite_active_plan_requires_composite_update",)
        )
    verify_active_experiment_plan(
        bundle,
        candidates,
        authority,
        plan,
    )
    verified_results = verify_active_experiment_results(
        plan,
        result_ledger,
        authority,
    )
    return _apply_verified_experiment_results(
        bundle,
        training_ledger,
        applicability_ledger,
        future_validation_ledger,
        plan,
        verified_results,
        authority,
    )


def _apply_verified_experiment_results(
    bundle: PersonModelBundle,
    training_ledger: EventLedger,
    applicability_ledger: EventLedger,
    future_validation_ledger: EventLedger,
    plan: ActiveExperimentPlan,
    verified_results: EventLedger,
    authority: VerificationAuthority,
) -> ActiveExperimentUpdate:
    try:
        verified_validation = EventLedger.verify(
            future_validation_ledger.records,
            authority,
        )
    except ValueError as error:
        raise ActiveExperimentRefusedError(
            ("future_validation_signature_invalid",)
        ) from error
    latest_result = max(
        _parse_timestamp(record.observed_at, "observed_at")
        for record in verified_results.records
    )
    if any(
        _parse_timestamp(record.observed_at, "observed_at")
        <= latest_result
        for record in verified_validation.records
    ):
        raise ActiveExperimentRefusedError(
            ("validation_not_after_experiment_results",)
        )
    prior_covariance = np.asarray(
        bundle.representation.covariance,
        dtype=np.float64,
    )
    prior_sign, prior_log_determinant = np.linalg.slogdet(
        prior_covariance
    )
    if prior_sign <= 0:
        raise ActiveExperimentRefusedError(
            ("parameter_covariance_not_positive_definite",)
        )
    current_bundle = bundle
    current_ledger = EventLedger.verify(
        training_ledger.records,
        authority,
    )
    from .workflow import update_person_model

    ordered_results = tuple(
        sorted(
            verified_results.records,
            key=lambda record: (
                _parse_timestamp(record.observed_at, "observed_at"),
                record.event_id,
            ),
        )
    )
    try:
        for record in ordered_results:
            update = update_person_model(
                current_bundle,
                current_ledger,
                record,
                authority,
                applicability_ledger=applicability_ledger,
                validation_ledger=verified_validation,
                experiment_plan_id=plan.plan_id,
            )
            current_bundle = update.bundle
            current_ledger = update.ledger
    except ValueError as error:
        raise ActiveExperimentRefusedError(
            ("active_experiment_model_update_failed",)
        ) from error
    posterior_covariance = np.asarray(
        current_bundle.representation.covariance,
        dtype=np.float64,
    )
    posterior_sign, posterior_log_determinant = np.linalg.slogdet(
        posterior_covariance
    )
    if posterior_sign <= 0:
        raise ActiveExperimentRefusedError(
            ("updated_parameter_covariance_invalid",)
        )
    return ActiveExperimentUpdate(
        bundle=current_bundle,
        ledger=current_ledger,
        plan_id=plan.plan_id,
        result_event_ids=tuple(
            record.event_id for record in ordered_results
        ),
        result_data_hash=EventLedger.snapshot_hash(ordered_results),
        prior_log_determinant=float(prior_log_determinant),
        posterior_log_determinant=float(
            posterior_log_determinant
        ),
        realized_covariance_entropy_reduction=0.5
        * (
            float(prior_log_determinant)
            - float(posterior_log_determinant)
        ),
    )


def active_experiment_config_from_dict(
    data: dict[str, object],
) -> ActiveExperimentConfig:
    expected = set(ActiveExperimentConfig().to_dict())
    if set(data) != expected:
        raise ValueError(
            "active experiment config fields do not match this version"
        )
    return ActiveExperimentConfig(
        minimum_information_gain=float(
            data["minimum_information_gain"]
        ),
        quadrature_points=int(data["quadrature_points"]),
        model_version=str(data["model_version"]),
    )


def active_experiment_plan_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> ActiveExperimentPlan:
    if data.get("artifact_version") != "pcfm-active-experiment-v3":
        raise ValueError(
            "unsupported active experiment artifact version"
        )
    selection_count = int(data["selection_count"])
    expected_selection_mode = (
        "adaptive_single_step"
        if selection_count == 1
        else "outcome_blind_batch_approximation"
    )
    if data.get("selection_mode") != expected_selection_mode:
        raise ValueError(
            "active experiment selection_mode is inconsistent"
        )
    plan = ActiveExperimentPlan(
        base_model_id=str(data["base_model_id"]),
        predictive_model_id=str(data["predictive_model_id"]),
        predictive_model_version=str(
            data["predictive_model_version"]
        ),
        parameter_dimension=int(data["parameter_dimension"]),
        person_id=str(data["person_id"]),
        created_at=str(data["created_at"]),
        candidate_pool_hash=str(data["candidate_pool_hash"]),
        candidate_count=int(data["candidate_count"]),
        selection_count=selection_count,
        config=active_experiment_config_from_dict(
            dict(data["config"])
        ),
        selections=tuple(
            ExperimentSelection(
                rank=int(item["rank"]),
                scenario=_scenario_from_dict(dict(item["scenario"])),
                expected_choice_probability=float(
                    item["expected_choice_probability"]
                ),
                expected_information_gain=float(
                    item["expected_information_gain"]
                ),
                cumulative_information_gain=float(
                    item["cumulative_information_gain"]
                ),
            )
            for raw_item in data["selections"]
            for item in (dict(raw_item),)
        ),
        prior_log_determinant=float(data["prior_log_determinant"]),
        expected_posterior_log_determinant=float(
            data["expected_posterior_log_determinant"]
        ),
        expected_posterior_covariance=tuple(
            tuple(float(value) for value in row)
            for row in data["expected_posterior_covariance"]
        ),
        total_expected_information_gain=float(
            data["total_expected_information_gain"]
        ),
        expected_covariance_entropy_reduction=float(
            data["expected_covariance_entropy_reduction"]
        ),
        verifier_id=str(data["verifier_id"]),
        artifact_version=str(data["artifact_version"]),
        plan_id=str(data["plan_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    plan.verify(authority)
    return plan


def save_active_experiment_plan(
    path: Path,
    plan: ActiveExperimentPlan,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            plan.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_active_experiment_plan(
    path: Path,
    authority: VerificationAuthority,
) -> ActiveExperimentPlan:
    return active_experiment_plan_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )
