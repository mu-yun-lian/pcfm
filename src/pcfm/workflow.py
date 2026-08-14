from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

from .applicability import (
    ApplicabilityProfile,
    PredictionRefusedError,
    assess_temporal_stability,
    fit_applicability_profile,
)
from .contracts import (
    Observation,
    PersonalAdapter,
    PersonRepresentation,
    Prediction,
    Scenario,
)
from .core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    ModelUpdater,
    PopulationPriorEstimator,
)
from .evaluation import assess_person_validation
from .ledger import EventLedger, EventRecord, VerificationAuthority
from .interfaces import PopulationModel
from .storage import (
    ModelManifest,
    ModelValidation,
    PersonModelBundle,
    feature_schema_hash,
    manifest_model_id,
    model_content_hash,
    scenario_design_hash,
    trial_key_hash,
)


def observation_from_dict(data: dict[str, object]) -> Observation:
    scenario_data = dict(data["scenario"])
    raw_features = scenario_data["features"]
    if isinstance(raw_features, dict):
        feature_names = tuple(str(name) for name in raw_features)
        features = tuple(float(raw_features[name]) for name in raw_features)
    else:
        feature_names = tuple(
            str(name) for name in scenario_data["feature_names"]
        )
        features = tuple(float(value) for value in raw_features)
    scenario = Scenario(
        scenario_id=str(scenario_data["scenario_id"]),
        features=features,
        feature_names=feature_names,
        options=tuple(scenario_data.get("options", ("A", "B"))),
        domain=str(scenario_data.get("domain", "structured_choice")),
        context=dict(scenario_data.get("context", {})),
    )
    return Observation(
        person_id=str(data["person_id"]),
        scenario=scenario,
        actual_choice=int(data["actual_choice"]),
        confidence=(
            float(data["confidence"])
            if data.get("confidence") is not None
            else None
        ),
        reaction_time_ms=(
            float(data["reaction_time_ms"])
            if data.get("reaction_time_ms") is not None
            else None
        ),
        provenance=str(data.get("provenance", "human_record")),
    )


def load_observations_jsonl(path: Path) -> tuple[Observation, ...]:
    observations = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            observations.append(observation_from_dict(json.loads(raw_line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid observation on line {line_number}: {error}"
            ) from error
    if not observations:
        raise ValueError("observation file is empty")
    return tuple(observations)


def event_record_from_dict(data: dict[str, object]) -> EventRecord:
    return EventRecord(
        event_id=str(data["event_id"]),
        observation=observation_from_dict(dict(data["observation"])),
        observed_at=str(data["observed_at"]),
        evidence_hash=str(data["evidence_hash"]),
        verifier_id=str(data["verifier_id"]),
        verified_at=str(data["verified_at"]),
        signature=str(data["signature"]),
        signature_method=str(data.get("signature_method", "hmac-sha256")),
    )


def load_verification_authority(path: Path) -> VerificationAuthority:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("verification key file must contain a JSON object")
    return VerificationAuthority(
        {
            str(verifier_id): str(secret).encode("utf-8")
            for verifier_id, secret in raw.items()
        }
    )


def load_event_ledger_jsonl(
    path: Path,
    authority: VerificationAuthority,
) -> EventLedger:
    records = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            records.append(event_record_from_dict(json.loads(raw_line)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid event on line {line_number}: {error}"
            ) from error
    try:
        return EventLedger.verify(records, authority)
    except ValueError as error:
        raise ValueError(f"invalid event ledger: {error}") from error


def save_event_ledger_jsonl(path: Path, ledger: EventLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        for record in ledger.records
    )
    path.write_text(rendered + "\n", encoding="utf-8")


def _require_target_only(
    ledger: EventLedger,
    person_id: str,
    label: str,
) -> tuple[EventRecord, ...]:
    records = ledger.records_for_person(person_id)
    if len(records) != len(ledger.records):
        raise ValueError(f"{label} ledger must contain only the target person")
    return records


def _ensure_ledgers_disjoint(
    labelled_ledgers: Sequence[tuple[str, EventLedger]],
) -> None:
    for left_index, (left_name, left) in enumerate(labelled_ledgers):
        left_event_ids = {record.event_id for record in left.records}
        left_trials = {
            (
                record.observation.person_id,
                record.observation.scenario.scenario_id,
            )
            for record in left.records
        }
        for right_name, right in labelled_ledgers[left_index + 1 :]:
            if left_event_ids & {
                record.event_id for record in right.records
            }:
                raise ValueError(
                    f"{left_name} and {right_name} event ids overlap"
                )
            if left_trials & {
                (
                    record.observation.person_id,
                    record.observation.scenario.scenario_id,
                )
                for record in right.records
            }:
                raise ValueError(
                    f"{left_name} and {right_name} scenarios overlap"
                )


def _latest_observed_at(
    ledgers: Sequence[EventLedger],
) -> str:
    records = tuple(
        record for ledger in ledgers for record in ledger.records
    )
    latest = max(
        records,
        key=lambda record: datetime.fromisoformat(
            record.observed_at.replace("Z", "+00:00")
        ),
    )
    return latest.observed_at


def fit_person_model(
    ledger: EventLedger,
    authority: VerificationAuthority,
    *,
    applicability_ledger: EventLedger | None = None,
    validation_ledger: EventLedger | None = None,
    person_id: str,
    feature_names: tuple[str, ...],
) -> PersonModelBundle:
    if not feature_names:
        raise ValueError("feature_names must not be empty")
    ledger = EventLedger.verify(ledger.records, authority)
    if validation_ledger is not None:
        validation_ledger = EventLedger.verify(
            validation_ledger.records,
            authority,
        )
    if applicability_ledger is not None:
        applicability_ledger = EventLedger.verify(
            applicability_ledger.records,
            authority,
        )
    if validation_ledger is not None and applicability_ledger is None:
        raise ValueError(
            "independent applicability calibration ledger is required "
            "for validated models"
        )
    person_records = ledger.records_for_person(person_id)
    if not person_records:
        raise ValueError(f"no events found for person {person_id}")
    labelled_ledgers = [("training", ledger)]
    if applicability_ledger is not None:
        _require_target_only(
            applicability_ledger,
            person_id,
            "applicability calibration",
        )
        labelled_ledgers.append(
            ("applicability calibration", applicability_ledger)
        )
    if validation_ledger is not None:
        _require_target_only(validation_ledger, person_id, "validation")
        labelled_ledgers.append(("validation", validation_ledger))
    _ensure_ledgers_disjoint(labelled_ledgers)
    observations = ledger.observations()
    prior_estimator = PopulationPriorEstimator(feature_names)
    prior = prior_estimator.fit(observations)
    encoder = MapPersonEncoder()
    representation = encoder.fit(person_id, observations, prior)
    adapter = IdentityAdapterGenerator().generate(representation, prior)
    validation = _validate_person_model(
        person_id=person_id,
        training_ledger=ledger,
        validation_ledger=validation_ledger,
        authority=authority,
        population_model=prior,
        representation=representation,
        adapter=adapter,
    )
    reference_ledger = applicability_ledger or EventLedger(
        records=person_records
    )
    applicability_profile = fit_applicability_profile(
        reference_ledger.observations(),
        feature_names,
        valid_through=_latest_observed_at(
            [
                ledger,
                *(
                    [applicability_ledger]
                    if applicability_ledger is not None
                    else []
                ),
                *(
                    [validation_ledger]
                    if validation_ledger is not None
                    else []
                ),
            ]
        ),
    )
    training_config = tuple(
        sorted(
            {
                "population_l2_precision": str(
                    prior_estimator.l2_precision
                ),
                "initial_person_variance": str(
                    prior_estimator.initial_person_variance
                ),
                "min_person_variance": str(
                    prior_estimator.min_person_variance
                ),
                "max_person_variance": str(
                    prior_estimator.max_person_variance
                ),
                "eb_iterations": str(prior_estimator.eb_iterations),
                "eb_damping": str(prior_estimator.eb_damping),
                "eb_tolerance": str(prior_estimator.eb_tolerance),
                "prior_precision_scale": str(
                    encoder.prior_precision_scale
                ),
            }.items()
        )
    )
    manifest = _build_manifest(
        person_id=person_id,
        person_records=person_records,
        applicability_records=reference_ledger.records,
        lineage_records=tuple(
            record
            for _, lineage_ledger in labelled_ledgers
            for record in lineage_ledger.records
        ),
        population_event_ids=tuple(
            sorted(record.event_id for record in ledger.records)
        ),
        population_data_hash=EventLedger.snapshot_hash(ledger.records),
        feature_names=feature_names,
        parent_model_id=None,
        experiment_plan_ids=(),
        training_config=training_config,
        validation=validation,
        applicability_profile=applicability_profile,
        population_model=prior,
        representation=representation,
        adapter=adapter,
    )
    return PersonModelBundle(
        population_model=prior,
        representation=representation,
        adapter=adapter,
        manifest=manifest,
    )


def predict_with_bundle(
    bundle: PersonModelBundle,
    scenario: Scenario,
    *,
    prediction_at: str | None = None,
    validation_override: bool = False,
    applicability_override: bool = False,
) -> Prediction:
    validation_status = bundle.manifest.validation.status
    assessment = bundle.manifest.applicability_profile.assess(
        scenario,
        prediction_at=prediction_at,
    )
    blocking_reasons = []
    overrides = []
    if validation_status != "passed":
        reason = f"model_validation_{validation_status}"
        if validation_override:
            overrides.append(reason)
        else:
            blocking_reasons.append(reason)
    if assessment.reasons:
        if applicability_override:
            overrides.extend(assessment.reasons)
        else:
            blocking_reasons.extend(assessment.reasons)
    if blocking_reasons:
        raise PredictionRefusedError(blocking_reasons, assessment)
    prediction = DecisionIntegrator().predict(
        scenario,
        bundle.population_model,
        bundle.adapter,
        parameter_covariance=bundle.representation.covariance,
    )
    extrapolative = bool(assessment.warnings)
    applicability_overridden = bool(
        assessment.reasons and applicability_override
    )
    any_override = bool(overrides)
    return replace(
        prediction,
        active_modules=prediction.active_modules
        + ("applicability_guard",),
        applicability_status=(
            "overridden"
            if applicability_overridden
            else assessment.status
        ),
        applicability_warnings=assessment.warnings,
        ood_score=assessment.ood_score,
        ood_threshold=assessment.ood_threshold,
        local_ood_score=assessment.local_ood_score,
        local_ood_threshold=assessment.local_ood_threshold,
        probability_lower_95=(
            None
            if extrapolative or any_override
            else prediction.probability_lower_95
        ),
        probability_upper_95=(
            None
            if extrapolative or any_override
            else prediction.probability_upper_95
        ),
        model_form_uncertainty_status=(
            "unquantified_override"
            if any_override
            else (
                "unquantified_extrapolation"
                if extrapolative
                else "within_supported_metadata"
            )
        ),
        validation_status=validation_status,
        gate_overrides=tuple(overrides),
    )


@dataclass(frozen=True)
class WorkflowUpdate:
    bundle: PersonModelBundle
    ledger: EventLedger


def _validate_person_model(
    *,
    person_id: str,
    training_ledger: EventLedger,
    validation_ledger: EventLedger | None,
    authority: VerificationAuthority,
    population_model: PopulationModel,
    representation: PersonRepresentation,
    adapter: PersonalAdapter,
) -> ModelValidation:
    if validation_ledger is None:
        return ModelValidation.unvalidated()
    validation_ledger = EventLedger.verify(
        validation_ledger.records,
        authority,
    )
    validation_records = validation_ledger.records_for_person(person_id)
    if len(validation_records) != len(validation_ledger.records):
        raise ValueError(
            "validation ledger must contain only the target person"
        )
    training_event_ids = {
        record.event_id for record in training_ledger.records
    }
    validation_event_ids = {
        record.event_id for record in validation_records
    }
    if training_event_ids & validation_event_ids:
        raise ValueError("training and validation event ids overlap")
    training_trials = {
        (
            record.observation.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in training_ledger.records
    }
    validation_trials = {
        (
            record.observation.person_id,
            record.observation.scenario.scenario_id,
        )
        for record in validation_records
    }
    if training_trials & validation_trials:
        raise ValueError("training and validation scenarios overlap")

    integrator = DecisionIntegrator()
    observations = tuple(
        record.observation for record in validation_records
    )
    personal_probabilities = tuple(
        integrator.predict(
            observation.scenario,
            population_model,
            adapter,
            parameter_covariance=representation.covariance,
        ).probability_option_1
        for observation in observations
    )
    population_probabilities = tuple(
        integrator.predict_population(
            observation.scenario,
            population_model,
        ).probability_option_1
        for observation in observations
    )
    diagnostics = assess_person_validation(
        observations,
        personal_probabilities,
        population_probabilities,
        population_model.feature_names,
    )
    temporal = assess_temporal_stability(
        validation_records,
        personal_probabilities,
        population_model.feature_names,
    )
    reasons = list(
        diagnostics["personalization_reasons"]
        + diagnostics["mechanism_reasons"]
    )
    if temporal.drift_detected:
        reasons.append("temporal_behavior_drift_suspected")
    if temporal.status == "not_assessed":
        reasons.append("temporal_stability_not_assessed")
    status = (
        "passed"
        if (
            diagnostics["passed"]
            and temporal.status == "stable"
            and not temporal.drift_detected
        )
        else "failed"
    )
    personal_report = diagnostics["personal_report"]
    population_report = diagnostics["population_report"]
    ordered_records = tuple(
        sorted(validation_records, key=lambda record: record.event_id)
    )
    return ModelValidation(
        status=status,
        validation_event_ids=tuple(
            record.event_id for record in ordered_records
        ),
        validation_data_hash=EventLedger.snapshot_hash(ordered_records),
        sample_count=len(ordered_records),
        personal_nll=personal_report.negative_log_likelihood,
        population_nll=population_report.negative_log_likelihood,
        nll_uplift=float(diagnostics["nll_uplift"]),
        nll_uplift_ci_lower=float(
            diagnostics["nll_uplift_ci_lower"]
        ),
        nll_uplift_ci_upper=float(
            diagnostics["nll_uplift_ci_upper"]
        ),
        calibration_error=personal_report.expected_calibration_error,
        personalization_passed=bool(
            diagnostics["personalization_passed"]
        ),
        mechanism_probe_nll_uplift=float(
            diagnostics["mechanism_probe_nll_uplift"]
        ),
        mechanism_adequacy_passed=bool(
            diagnostics["mechanism_adequacy_passed"]
        ),
        temporal_stability_status=temporal.status,
        temporal_drift_score=temporal.maximum_score_z,
        temporal_critical_score_z=temporal.critical_score_z,
        temporal_score_effect=temporal.maximum_score_effect,
        temporal_early_nll=temporal.early_nll,
        temporal_late_nll=temporal.late_nll,
        temporal_early_sample_count=temporal.early_sample_count,
        temporal_late_sample_count=temporal.late_sample_count,
        temporal_drift_detected=temporal.drift_detected,
        reasons=tuple(reasons),
    )


def _build_manifest(
    *,
    person_id: str,
    person_records: Sequence[EventRecord],
    applicability_records: Sequence[EventRecord],
    lineage_records: Sequence[EventRecord],
    population_event_ids: tuple[str, ...],
    population_data_hash: str,
    feature_names: Sequence[str],
    parent_model_id: str | None,
    experiment_plan_ids: tuple[str, ...],
    training_config: tuple[tuple[str, str], ...],
    validation: ModelValidation,
    applicability_profile: ApplicabilityProfile,
    population_model: PopulationModel,
    representation: PersonRepresentation,
    adapter: PersonalAdapter,
) -> ModelManifest:
    ordered_records = tuple(
        sorted(person_records, key=lambda record: record.event_id)
    )
    person_data_hash = EventLedger.snapshot_hash(ordered_records)
    person_event_ids = tuple(record.event_id for record in ordered_records)
    ordered_applicability_records = tuple(
        sorted(
            applicability_records,
            key=lambda record: record.event_id,
        )
    )
    applicability_event_ids = tuple(
        record.event_id for record in ordered_applicability_records
    )
    applicability_data_hash = EventLedger.snapshot_hash(
        ordered_applicability_records
    )
    lineage_trial_hashes = tuple(
        sorted(
            {
                trial_key_hash(
                    record.observation.person_id,
                    record.observation.scenario.scenario_id,
                )
                for record in lineage_records
            }
        )
    )
    lineage_design_hashes = tuple(
        sorted(
            {
                scenario_design_hash(
                    record.observation.person_id,
                    record.observation.scenario,
                )
                for record in lineage_records
            }
        )
    )
    latest_record = max(
        ordered_records,
        key=lambda record: datetime.fromisoformat(
            record.observed_at.replace("Z", "+00:00")
        ),
    )
    training_cutoff = latest_record.observed_at
    schema_digest = feature_schema_hash(tuple(feature_names))
    content_digest = model_content_hash(
        population_model,
        representation,
        adapter,
    )
    model_id = manifest_model_id(
        parent_model_id=parent_model_id,
        person_id=person_id,
        person_data_hash=person_data_hash,
        population_data_hash=population_data_hash,
        feature_schema_digest=schema_digest,
        model_content_digest=content_digest,
        validation_digest=validation.digest(),
        applicability_digest=applicability_profile.digest(),
        applicability_event_ids=applicability_event_ids,
        applicability_data_hash=applicability_data_hash,
        lineage_trial_hashes=lineage_trial_hashes,
        lineage_design_hashes=lineage_design_hashes,
        experiment_plan_ids=experiment_plan_ids,
        training_config=training_config,
        code_version="pcfm-mvp-0.8.0",
    )
    return ModelManifest(
        model_id=model_id,
        parent_model_id=parent_model_id,
        person_id=person_id,
        person_event_ids=person_event_ids,
        population_event_ids=population_event_ids,
        person_data_hash=person_data_hash,
        population_data_hash=population_data_hash,
        feature_schema_hash=schema_digest,
        model_content_hash=content_digest,
        applicability_event_ids=applicability_event_ids,
        applicability_data_hash=applicability_data_hash,
        lineage_trial_hashes=lineage_trial_hashes,
        lineage_design_hashes=lineage_design_hashes,
        experiment_plan_ids=experiment_plan_ids,
        training_cutoff=training_cutoff,
        training_config=training_config,
        validation=validation,
        applicability_profile=applicability_profile,
    )


def update_person_model(
    bundle: PersonModelBundle,
    ledger: EventLedger,
    outcome: EventRecord,
    authority: VerificationAuthority,
    applicability_ledger: EventLedger | None = None,
    validation_ledger: EventLedger | None = None,
    experiment_plan_id: str | None = None,
) -> WorkflowUpdate:
    ledger = EventLedger.verify(ledger.records, authority)
    person_id = bundle.representation.person_id
    current_records = ledger.records_for_person(person_id)
    current_ids = tuple(
        record.event_id
        for record in sorted(current_records, key=lambda item: item.event_id)
    )
    if current_ids != bundle.manifest.person_event_ids:
        raise ValueError("event history does not match the model manifest")
    if EventLedger.snapshot_hash(current_records) != bundle.manifest.person_data_hash:
        raise ValueError("event history content does not match the model manifest")
    if outcome.observation.person_id != person_id:
        raise ValueError("outcome person_id does not match the model")
    experiment_plan_ids = bundle.manifest.experiment_plan_ids
    if experiment_plan_id is not None:
        if len(experiment_plan_id) != 64:
            raise ValueError(
                "experiment_plan_id must be a SHA-256 digest"
            )
        try:
            bytes.fromhex(experiment_plan_id)
        except ValueError as error:
            raise ValueError(
                "experiment_plan_id must be a SHA-256 digest"
            ) from error
        if experiment_plan_id not in experiment_plan_ids:
            experiment_plan_ids = tuple(
                sorted(experiment_plan_ids + (experiment_plan_id,))
            )
    updated_ledger = ledger.append(outcome, authority)
    if applicability_ledger is not None:
        applicability_ledger = EventLedger.verify(
            applicability_ledger.records,
            authority,
        )
    if validation_ledger is not None:
        validation_ledger = EventLedger.verify(
            validation_ledger.records,
            authority,
        )
    if validation_ledger is not None and applicability_ledger is None:
        raise ValueError(
            "independent applicability calibration ledger is required "
            "for validated models"
        )
    labelled_ledgers = [("training", updated_ledger)]
    if applicability_ledger is not None:
        _require_target_only(
            applicability_ledger,
            person_id,
            "applicability calibration",
        )
        labelled_ledgers.append(
            ("applicability calibration", applicability_ledger)
        )
    if validation_ledger is not None:
        _require_target_only(validation_ledger, person_id, "validation")
        labelled_ledgers.append(("validation", validation_ledger))
    _ensure_ledgers_disjoint(labelled_ledgers)
    config = dict(bundle.manifest.training_config)
    required_config = {
        "population_l2_precision",
        "initial_person_variance",
        "min_person_variance",
        "max_person_variance",
        "eb_iterations",
        "eb_damping",
        "eb_tolerance",
        "prior_precision_scale",
    }
    if set(config) != required_config:
        raise ValueError("model training configuration is incomplete")
    updater = ModelUpdater(
        encoder=MapPersonEncoder(
            prior_precision_scale=float(config["prior_precision_scale"])
        ),
        adapter_generator=IdentityAdapterGenerator(),
    )
    updated = updater.update(
        person_id,
        tuple(record.observation for record in current_records),
        outcome,
        authority,
        bundle.population_model,
    )
    updated_records = updated_ledger.records_for_person(person_id)
    validation = _validate_person_model(
        person_id=person_id,
        training_ledger=updated_ledger,
        validation_ledger=validation_ledger,
        authority=authority,
        population_model=bundle.population_model,
        representation=updated.representation,
        adapter=updated.adapter,
    )
    reference_ledger = applicability_ledger or EventLedger(
        records=updated_records
    )
    applicability_profile = fit_applicability_profile(
        reference_ledger.observations(),
        bundle.population_model.feature_names,
        valid_through=_latest_observed_at(
            [
                updated_ledger,
                *(
                    [applicability_ledger]
                    if applicability_ledger is not None
                    else []
                ),
                *(
                    [validation_ledger]
                    if validation_ledger is not None
                    else []
                ),
            ]
        ),
    )
    updated_bundle = PersonModelBundle(
        population_model=bundle.population_model,
        representation=updated.representation,
        adapter=updated.adapter,
        manifest=_build_manifest(
            person_id=person_id,
            person_records=updated_records,
            applicability_records=reference_ledger.records,
            lineage_records=tuple(
                record
                for _, lineage_ledger in labelled_ledgers
                for record in lineage_ledger.records
            ),
            population_event_ids=bundle.manifest.population_event_ids,
            population_data_hash=bundle.manifest.population_data_hash,
            feature_names=bundle.population_model.feature_names,
            parent_model_id=bundle.manifest.model_id,
            experiment_plan_ids=experiment_plan_ids,
            training_config=bundle.manifest.training_config,
            validation=validation,
            applicability_profile=applicability_profile,
            population_model=bundle.population_model,
            representation=updated.representation,
            adapter=updated.adapter,
        ),
    )
    return WorkflowUpdate(bundle=updated_bundle, ledger=updated_ledger)
