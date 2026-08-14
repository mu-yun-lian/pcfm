from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Sequence

import numpy as np

from .active_experiment import (
    ActiveExperimentConfig,
    ActiveExperimentPlan,
    ActiveExperimentRefusedError,
    ActiveExperimentUpdate,
    _apply_verified_experiment_results,
    _build_gaussian_view_plan,
    _validate_candidate_pool,
    verify_active_experiment_results,
)
from .contracts import Scenario
from .ledger import EventLedger, VerificationAuthority
from .mechanism import (
    MechanismComparisonPlan,
    MechanismComparisonReport,
    MechanismRefusedError,
    _predict_with_verified_mechanism,
    verify_mechanism_report,
)
from .storage import (
    PersonModelBundle,
    scenario_design_hash,
    trial_key_hash,
)


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            f"{label} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


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
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a SHA-256 digest"
        ) from error


class CompositeModelRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "composite model refused: " + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class CompositeModelArtifact:
    base_model_id: str
    person_id: str
    created_at: str
    valid_through: str
    mechanism_plan_id: str
    mechanism_report_id: str
    mechanism_discovery_data_hash: str
    mechanism_selection_data_hash: str
    mechanism_confirmation_data_hash: str
    selected_hypothesis_id: str
    component_mode: str
    verifier_id: str
    interpretation: str = "validated_composite_predictive_view"
    uncertainty_scope: str = (
        "block_diagonal_conditional_gaussian"
    )
    artifact_version: str = "pcfm-composite-model-v1"
    composite_model_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        for value, label in (
            (self.base_model_id, "base_model_id"),
            (self.mechanism_plan_id, "mechanism_plan_id"),
            (self.mechanism_report_id, "mechanism_report_id"),
            (
                self.mechanism_discovery_data_hash,
                "mechanism_discovery_data_hash",
            ),
            (
                self.mechanism_selection_data_hash,
                "mechanism_selection_data_hash",
            ),
            (
                self.mechanism_confirmation_data_hash,
                "mechanism_confirmation_data_hash",
            ),
        ):
            _require_digest(value, label)
        created = _parse_timestamp(self.created_at, "created_at")
        valid_through = _parse_timestamp(
            self.valid_through,
            "valid_through",
        )
        if created >= valid_through:
            raise ValueError(
                "composite model must be created before expiry"
            )
        if (
            not self.person_id
            or not self.selected_hypothesis_id
            or not self.verifier_id
        ):
            raise ValueError(
                "composite model identity is required"
            )
        if self.component_mode != "stable_plus_confirmed_mechanism":
            raise ValueError(
                "unsupported composite component mode"
            )
        if (
            self.interpretation
            != "validated_composite_predictive_view"
            or self.uncertainty_scope
            != "block_diagonal_conditional_gaussian"
        ):
            raise ValueError(
                "composite model interpretation is invalid"
            )
        if self.signature_method != "hmac-sha256":
            raise ValueError(
                "unsupported composite signature method"
            )
        expected = self.digest()
        if self.composite_model_id:
            _require_digest(
                self.composite_model_id,
                "composite_model_id",
            )
            if self.composite_model_id != expected:
                raise ValueError(
                    "composite_model_id does not match content"
                )
        else:
            object.__setattr__(
                self,
                "composite_model_id",
                expected,
            )
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "base_model_id": self.base_model_id,
            "person_id": self.person_id,
            "created_at": self.created_at,
            "valid_through": self.valid_through,
            "mechanism_plan_id": self.mechanism_plan_id,
            "mechanism_report_id": self.mechanism_report_id,
            "mechanism_discovery_data_hash": (
                self.mechanism_discovery_data_hash
            ),
            "mechanism_selection_data_hash": (
                self.mechanism_selection_data_hash
            ),
            "mechanism_confirmation_data_hash": (
                self.mechanism_confirmation_data_hash
            ),
            "selected_hypothesis_id": (
                self.selected_hypothesis_id
            ),
            "component_mode": self.component_mode,
            "verifier_id": self.verifier_id,
            "interpretation": self.interpretation,
            "uncertainty_scope": self.uncertainty_scope,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {
            **self._content_dict(),
            "composite_model_id": self.composite_model_id,
        }

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("composite model is unsigned")
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
class CompositePrediction:
    scenario_id: str
    person_id: str
    probability_option_1: float
    predicted_choice: int
    probability_lower_95: float
    probability_upper_95: float
    logit_standard_deviation: float
    predictive_model_id: str
    base_model_id: str
    mechanism_plan_id: str
    mechanism_report_id: str
    selected_hypothesis_id: str
    active_components: tuple[str, ...]
    applicability_status: str
    validation_status: str
    uncertainty_scope: str
    interpretation: str
    model_version: str = "composite-predictive-view-v1"

    def __post_init__(self) -> None:
        probabilities = (
            self.probability_option_1,
            self.probability_lower_95,
            self.probability_upper_95,
        )
        if (
            any(
                not isfinite(value) or not 0 <= value <= 1
                for value in probabilities
            )
            or not self.probability_lower_95
            <= self.probability_option_1
            <= self.probability_upper_95
        ):
            raise ValueError(
                "composite prediction probabilities are invalid"
            )
        if (
            self.predicted_choice not in (0, 1)
            or not isfinite(self.logit_standard_deviation)
            or self.logit_standard_deviation <= 0
        ):
            raise ValueError(
                "composite prediction values are invalid"
            )
        for value, label in (
            (self.predictive_model_id, "predictive_model_id"),
            (self.base_model_id, "base_model_id"),
            (self.mechanism_plan_id, "mechanism_plan_id"),
            (self.mechanism_report_id, "mechanism_report_id"),
        ):
            _require_digest(value, label)
        if (
            not self.scenario_id
            or not self.person_id
            or not self.selected_hypothesis_id
            or self.active_components
            != ("stable_person_model", "confirmed_mechanism")
        ):
            raise ValueError(
                "composite prediction identity is invalid"
            )


@dataclass(frozen=True)
class CompositeActiveExperimentUpdate:
    base_update: ActiveExperimentUpdate
    invalidated_composite_model_id: str
    invalidated_mechanism_report_id: str
    status: str = "base_updated_composite_invalidated"
    required_next_action: str = (
        "preregister_and_confirm_mechanism_against_updated_base"
    )

    def __post_init__(self) -> None:
        _require_digest(
            self.invalidated_composite_model_id,
            "invalidated_composite_model_id",
        )
        _require_digest(
            self.invalidated_mechanism_report_id,
            "invalidated_mechanism_report_id",
        )
        if (
            self.status != "base_updated_composite_invalidated"
            or self.required_next_action
            != "preregister_and_confirm_mechanism_against_updated_base"
        ):
            raise ValueError(
                "composite update invalidation contract is invalid"
            )
        if (
            self.base_update.plan_id
            not in self.base_update.bundle.manifest.experiment_plan_ids
        ):
            raise ValueError(
                "updated base does not bind composite experiment plan"
            )


def _composite_valid_through(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
) -> datetime:
    applicability = bundle.manifest.applicability_profile
    base_reference = _parse_timestamp(
        applicability.valid_through
        or bundle.manifest.training_cutoff,
        "base_reference_time",
    )
    base_expiry = base_reference + timedelta(
        days=applicability.maximum_age_days
    )
    mechanism_expiry = _parse_timestamp(
        report.evaluated_at,
        "mechanism_evaluated_at",
    ) + timedelta(days=plan.config.maximum_report_age_days)
    return min(base_expiry, mechanism_expiry)


def _verify_component(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
) -> None:
    try:
        verify_mechanism_report(
            bundle,
            plan,
            discovery_ledger,
            selection_ledger,
            confirmation_ledger,
            authority,
            report,
        )
    except (MechanismRefusedError, ValueError) as error:
        raise CompositeModelRefusedError(
            ("composite_component_derivation_mismatch",)
        ) from error
    if report.status != "supported_candidate":
        raise CompositeModelRefusedError(
            ("mechanism_candidate_not_supported",)
        )


def create_composite_model(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    *,
    verifier_id: str,
    created_at: str,
    dynamic_state_artifact_id: str | None = None,
) -> CompositeModelArtifact:
    if dynamic_state_artifact_id is not None:
        _require_digest(
            dynamic_state_artifact_id,
            "dynamic_state_artifact_id",
        )
        raise CompositeModelRefusedError(
            ("dynamic_state_not_reinferred_for_composite",)
        )
    _verify_component(
        bundle,
        plan,
        report,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    unsigned = _unsigned_composite_model(
        bundle,
        plan,
        report,
        verifier_id=verifier_id,
        created_at=created_at,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        verifier_id,
    )
    return replace(unsigned, signature=signature)


def _unsigned_composite_model(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    *,
    verifier_id: str,
    created_at: str,
) -> CompositeModelArtifact:
    reasons = []
    if report.base_model_id != bundle.manifest.model_id:
        reasons.append("composite_base_model_mismatch")
    if report.person_id != bundle.manifest.person_id:
        reasons.append("composite_person_mismatch")
    if verifier_id != report.verifier_id:
        reasons.append("composite_verifier_mismatch")
    created = _parse_timestamp(created_at, "created_at")
    evaluated = _parse_timestamp(
        report.evaluated_at,
        "mechanism_evaluated_at",
    )
    valid_through = _composite_valid_through(
        bundle,
        plan,
        report,
    )
    if created <= evaluated:
        reasons.append(
            "composite_created_before_mechanism_confirmation"
        )
    if created >= valid_through:
        reasons.append("composite_model_expired")
    if reasons:
        raise CompositeModelRefusedError(reasons)
    return CompositeModelArtifact(
        base_model_id=bundle.manifest.model_id,
        person_id=bundle.manifest.person_id,
        created_at=created_at,
        valid_through=_timestamp(valid_through),
        mechanism_plan_id=plan.plan_id,
        mechanism_report_id=report.report_id,
        mechanism_discovery_data_hash=(
            report.discovery_data_hash
        ),
        mechanism_selection_data_hash=(
            report.selection_data_hash
        ),
        mechanism_confirmation_data_hash=(
            report.confirmation_data_hash
        ),
        selected_hypothesis_id=(
            report.selected_hypothesis_id
        ),
        component_mode="stable_plus_confirmed_mechanism",
        verifier_id=verifier_id,
    )


def verify_composite_model(
    bundle: PersonModelBundle,
    artifact: CompositeModelArtifact,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
) -> None:
    try:
        artifact.verify(authority)
    except ValueError as error:
        raise CompositeModelRefusedError(
            ("composite_artifact_signature_invalid",)
        ) from error
    _verify_component(
        bundle,
        plan,
        report,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    recomputed = _unsigned_composite_model(
        bundle,
        plan,
        report,
        verifier_id=artifact.verifier_id,
        created_at=artifact.created_at,
    )
    if recomputed != replace(artifact, signature=""):
        raise CompositeModelRefusedError(
            ("composite_artifact_derivation_mismatch",)
        )


def predict_with_composite_model(
    bundle: PersonModelBundle,
    artifact: CompositeModelArtifact,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    scenario: Scenario,
    *,
    prediction_at: str,
) -> CompositePrediction:
    verify_composite_model(
        bundle,
        artifact,
        plan,
        report,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    prediction_time = _parse_timestamp(
        prediction_at,
        "prediction_at",
    )
    if prediction_time <= _parse_timestamp(
        artifact.created_at,
        "created_at",
    ):
        raise CompositeModelRefusedError(
            ("composite_prediction_precedes_artifact",)
        )
    if prediction_time > _parse_timestamp(
        artifact.valid_through,
        "valid_through",
    ):
        raise CompositeModelRefusedError(
            ("composite_model_expired",)
        )
    try:
        prediction = _predict_with_verified_mechanism(
            bundle,
            plan,
            report,
            scenario,
            prediction_at=prediction_at,
        )
    except MechanismRefusedError as error:
        raise CompositeModelRefusedError(
            error.reasons
        ) from error
    validation_status = (
        "passed"
        if bundle.manifest.validation.status == "passed"
        else "mechanism_only_failure_repaired"
    )
    return CompositePrediction(
        scenario_id=scenario.scenario_id,
        person_id=artifact.person_id,
        probability_option_1=prediction.probability_option_1,
        predicted_choice=prediction.predicted_choice,
        probability_lower_95=prediction.probability_lower_95,
        probability_upper_95=prediction.probability_upper_95,
        logit_standard_deviation=(
            prediction.logit_standard_deviation
        ),
        predictive_model_id=artifact.composite_model_id,
        base_model_id=artifact.base_model_id,
        mechanism_plan_id=artifact.mechanism_plan_id,
        mechanism_report_id=artifact.mechanism_report_id,
        selected_hypothesis_id=(
            artifact.selected_hypothesis_id
        ),
        active_components=(
            "stable_person_model",
            "confirmed_mechanism",
        ),
        applicability_status=prediction.applicability_status,
        validation_status=validation_status,
        uncertainty_scope=artifact.uncertainty_scope,
        interpretation=artifact.interpretation,
    )


def _mechanism_joint_view(
    bundle: PersonModelBundle,
    plan: MechanismComparisonPlan,
    report: MechanismComparisonReport,
    candidates: tuple[Scenario, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fit = next(
        candidate
        for candidate in report.candidate_fits
        if candidate.hypothesis_id
        == report.selected_hypothesis_id
    )
    hypothesis = next(
        candidate
        for candidate in plan.hypotheses
        if candidate.hypothesis_id
        == report.selected_hypothesis_id
    )
    base_mean = np.asarray(
        bundle.representation.latent_mean,
        dtype=np.float64,
    )
    correction_mean = np.asarray(
        fit.coefficients,
        dtype=np.float64,
    )
    mean = np.concatenate((base_mean, correction_mean))
    base_covariance = np.asarray(
        bundle.representation.covariance,
        dtype=np.float64,
    )
    correction_covariance = np.asarray(
        fit.covariance,
        dtype=np.float64,
    )
    covariance = np.zeros(
        (len(mean), len(mean)),
        dtype=np.float64,
    )
    base_dimension = len(base_mean)
    covariance[:base_dimension, :base_dimension] = (
        base_covariance
    )
    covariance[base_dimension:, base_dimension:] = (
        correction_covariance
    )
    centers = np.asarray(fit.centers, dtype=np.float64)
    scales = np.asarray(fit.scales, dtype=np.float64)
    designs = {}
    for scenario in candidates:
        base = np.asarray(
            scenario.ordered_features(
                bundle.representation.feature_names
            ),
            dtype=np.float64,
        )
        raw_terms = np.asarray(
            [
                term.evaluate(
                    scenario,
                    bundle.representation.feature_names,
                )
                for term in hypothesis.terms
            ],
            dtype=np.float64,
        )
        correction = (raw_terms - centers) / scales
        designs[scenario.scenario_id] = np.concatenate(
            (base, correction)
        )
    return mean, covariance, designs


def create_composite_active_experiment_plan(
    bundle: PersonModelBundle,
    artifact: CompositeModelArtifact,
    mechanism_plan: MechanismComparisonPlan,
    mechanism_report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    candidates: Sequence[Scenario],
    *,
    verifier_id: str,
    created_at: str,
    selection_count: int,
    config: ActiveExperimentConfig | None = None,
) -> ActiveExperimentPlan:
    verify_composite_model(
        bundle,
        artifact,
        mechanism_plan,
        mechanism_report,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
    )
    normalized = _validate_candidate_pool(
        bundle,
        candidates,
        created_at=created_at,
        selection_count=selection_count,
        require_passed_base=False,
    )
    mechanism_records = tuple(
        record
        for ledger in (
            discovery_ledger,
            selection_ledger,
            confirmation_ledger,
        )
        for record in ledger.records
    )
    mechanism_trial_hashes = {
        trial_key_hash(
            bundle.manifest.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in mechanism_records
    }
    mechanism_design_hashes = {
        scenario_design_hash(
            bundle.manifest.person_id,
            record.observation.scenario,
        )
        for record in mechanism_records
    }
    candidate_trial_hashes = {
        trial_key_hash(
            bundle.manifest.person_id,
            scenario.scenario_id,
        )
        for scenario in normalized
    }
    candidate_design_hashes = {
        scenario_design_hash(
            bundle.manifest.person_id,
            scenario,
        )
        for scenario in normalized
    }
    replay_reasons = []
    if mechanism_trial_hashes & candidate_trial_hashes:
        replay_reasons.append(
            "candidate_reuses_mechanism_scenario"
        )
    if mechanism_design_hashes & candidate_design_hashes:
        replay_reasons.append(
            "candidate_reuses_mechanism_design"
        )
    if replay_reasons:
        raise CompositeModelRefusedError(replay_reasons)
    for scenario in normalized:
        try:
            _predict_with_verified_mechanism(
                bundle,
                mechanism_plan,
                mechanism_report,
                scenario,
                prediction_at=created_at,
            )
        except MechanismRefusedError as error:
            raise CompositeModelRefusedError(
                error.reasons
            ) from error
    mean, covariance, designs = _mechanism_joint_view(
        bundle,
        mechanism_plan,
        mechanism_report,
        normalized,
    )
    try:
        return _build_gaussian_view_plan(
            bundle=bundle,
            normalized_candidates=normalized,
            design_vectors=designs,
            parameter_mean=mean,
            parameter_covariance=covariance,
            predictive_model_id=artifact.composite_model_id,
            predictive_model_version="composite-predictive-view-v1",
            authority=authority,
            verifier_id=verifier_id,
            created_at=created_at,
            selection_count=selection_count,
            config=config or ActiveExperimentConfig(),
        )
    except ActiveExperimentRefusedError as error:
        raise CompositeModelRefusedError(
            error.reasons
        ) from error


def verify_composite_active_experiment_plan(
    bundle: PersonModelBundle,
    artifact: CompositeModelArtifact,
    mechanism_plan: MechanismComparisonPlan,
    mechanism_report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    candidates: Sequence[Scenario],
    plan: ActiveExperimentPlan,
) -> None:
    try:
        plan.verify(authority)
    except ValueError as error:
        raise CompositeModelRefusedError(
            ("composite_active_experiment_signature_invalid",)
        ) from error
    try:
        recomputed = create_composite_active_experiment_plan(
            bundle,
            artifact,
            mechanism_plan,
            mechanism_report,
            discovery_ledger,
            selection_ledger,
            confirmation_ledger,
            authority,
            candidates,
            verifier_id=plan.verifier_id,
            created_at=plan.created_at,
            selection_count=plan.selection_count,
            config=plan.config,
        )
    except (
        ValueError,
        CompositeModelRefusedError,
        ActiveExperimentRefusedError,
    ) as error:
        raise CompositeModelRefusedError(
            ("composite_active_experiment_derivation_mismatch",)
        ) from error
    if recomputed != plan:
        raise CompositeModelRefusedError(
            ("composite_active_experiment_derivation_mismatch",)
        )


def apply_composite_active_experiment_results(
    bundle: PersonModelBundle,
    artifact: CompositeModelArtifact,
    mechanism_plan: MechanismComparisonPlan,
    mechanism_report: MechanismComparisonReport,
    discovery_ledger: EventLedger,
    selection_ledger: EventLedger,
    confirmation_ledger: EventLedger,
    authority: VerificationAuthority,
    training_ledger: EventLedger,
    applicability_ledger: EventLedger,
    future_validation_ledger: EventLedger,
    candidates: Sequence[Scenario],
    active_plan: ActiveExperimentPlan,
    result_ledger: EventLedger,
) -> CompositeActiveExperimentUpdate:
    verify_composite_active_experiment_plan(
        bundle,
        artifact,
        mechanism_plan,
        mechanism_report,
        discovery_ledger,
        selection_ledger,
        confirmation_ledger,
        authority,
        candidates,
        active_plan,
    )
    try:
        verified_future_validation = EventLedger.verify(
            future_validation_ledger.records,
            authority,
        )
    except ValueError as error:
        raise CompositeModelRefusedError(
            ("future_validation_signature_invalid",)
        ) from error
    mechanism_records = tuple(
        record
        for ledger in (
            discovery_ledger,
            selection_ledger,
            confirmation_ledger,
        )
        for record in ledger.records
    )
    future_records = verified_future_validation.records
    mechanism_trials = {
        trial_key_hash(
            bundle.manifest.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in mechanism_records
    }
    future_trials = {
        trial_key_hash(
            bundle.manifest.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in future_records
    }
    mechanism_designs = {
        scenario_design_hash(
            bundle.manifest.person_id,
            record.observation.scenario,
        )
        for record in mechanism_records
    }
    future_designs = {
        scenario_design_hash(
            bundle.manifest.person_id,
            record.observation.scenario,
        )
        for record in future_records
    }
    validation_replay_reasons = []
    if mechanism_trials & future_trials:
        validation_replay_reasons.append(
            "future_validation_reuses_mechanism_scenario"
        )
    if mechanism_designs & future_designs:
        validation_replay_reasons.append(
            "future_validation_reuses_mechanism_design"
        )
    if validation_replay_reasons:
        raise CompositeModelRefusedError(
            validation_replay_reasons
        )
    try:
        verified_results = verify_active_experiment_results(
            active_plan,
            result_ledger,
            authority,
        )
        base_update = _apply_verified_experiment_results(
            bundle,
            training_ledger,
            applicability_ledger,
            verified_future_validation,
            active_plan,
            verified_results,
            authority,
        )
    except ActiveExperimentRefusedError as error:
        raise CompositeModelRefusedError(
            error.reasons
        ) from error
    return CompositeActiveExperimentUpdate(
        base_update=base_update,
        invalidated_composite_model_id=(
            artifact.composite_model_id
        ),
        invalidated_mechanism_report_id=(
            mechanism_report.report_id
        ),
    )


def composite_model_from_dict(
    data: dict[str, object],
    authority: VerificationAuthority,
) -> CompositeModelArtifact:
    if data.get("artifact_version") != "pcfm-composite-model-v1":
        raise ValueError(
            "unsupported composite model artifact version"
        )
    artifact = CompositeModelArtifact(
        base_model_id=str(data["base_model_id"]),
        person_id=str(data["person_id"]),
        created_at=str(data["created_at"]),
        valid_through=str(data["valid_through"]),
        mechanism_plan_id=str(data["mechanism_plan_id"]),
        mechanism_report_id=str(data["mechanism_report_id"]),
        mechanism_discovery_data_hash=str(
            data["mechanism_discovery_data_hash"]
        ),
        mechanism_selection_data_hash=str(
            data["mechanism_selection_data_hash"]
        ),
        mechanism_confirmation_data_hash=str(
            data["mechanism_confirmation_data_hash"]
        ),
        selected_hypothesis_id=str(
            data["selected_hypothesis_id"]
        ),
        component_mode=str(data["component_mode"]),
        verifier_id=str(data["verifier_id"]),
        interpretation=str(data["interpretation"]),
        uncertainty_scope=str(data["uncertainty_scope"]),
        artifact_version=str(data["artifact_version"]),
        composite_model_id=str(data["composite_model_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )
    artifact.verify(authority)
    return artifact


def save_composite_model(
    path: Path,
    artifact: CompositeModelArtifact,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_composite_model(
    path: Path,
    authority: VerificationAuthority,
) -> CompositeModelArtifact:
    return composite_model_from_dict(
        json.loads(path.read_text(encoding="utf-8")),
        authority,
    )
