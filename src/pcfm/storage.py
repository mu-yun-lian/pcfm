from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .applicability import ApplicabilityProfile
from .contracts import PersonalAdapter, PersonRepresentation, Scenario
from .interfaces import PopulationModel


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from error


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def feature_schema_hash(feature_names: tuple[str, ...]) -> str:
    return _digest_json({"feature_names": list(feature_names), "version": 1})


def trial_key_hash(person_id: str, scenario_id: str) -> str:
    if not person_id or not scenario_id:
        raise ValueError("trial key identity is required")
    return _digest_json(
        {
            "person_id": person_id,
            "scenario_id": scenario_id,
            "version": 1,
        }
    )


def scenario_design_hash(person_id: str, scenario: Scenario) -> str:
    if not person_id:
        raise ValueError("scenario design person_id is required")
    features = sorted(
        zip(
            scenario.feature_names,
            scenario.features,
            strict=True,
        )
    )
    context = sorted(
        (str(name), str(value))
        for name, value in scenario.context.items()
        if name != "prediction_at"
    )
    return _digest_json(
        {
            "person_id": person_id,
            "features": [
                [str(name), float(value)]
                for name, value in features
            ],
            "options": list(scenario.options),
            "domain": scenario.domain,
            "context": [list(item) for item in context],
            "version": 1,
        }
    )


def model_content_hash(
    population_model: PopulationModel,
    representation: PersonRepresentation,
    adapter: PersonalAdapter,
) -> str:
    return _digest_json(
        {
            "population_model": {
                "weights": list(population_model.weights),
                "covariance": [
                    list(row) for row in population_model.covariance
                ],
                "feature_names": list(population_model.feature_names),
                "model_version": population_model.model_version,
            },
            "representation": {
                "person_id": representation.person_id,
                "latent_mean": list(representation.latent_mean),
                "covariance": [
                    list(row) for row in representation.covariance
                ],
                "representation_version": representation.representation_version,
                "observation_count": representation.observation_count,
                "feature_names": list(representation.feature_names),
            },
            "adapter": {
                "person_id": adapter.person_id,
                "delta_weights": list(adapter.delta_weights),
                "adapter_version": adapter.adapter_version,
                "representation_version": adapter.representation_version,
            },
        }
    )


def manifest_model_id(
    *,
    parent_model_id: str | None,
    person_id: str,
    person_data_hash: str,
    population_data_hash: str,
    feature_schema_digest: str,
    model_content_digest: str,
    validation_digest: str,
    applicability_digest: str,
    applicability_event_ids: tuple[str, ...],
    applicability_data_hash: str,
    lineage_trial_hashes: tuple[str, ...],
    lineage_design_hashes: tuple[str, ...],
    experiment_plan_ids: tuple[str, ...],
    training_config: tuple[tuple[str, str], ...],
    code_version: str,
) -> str:
    return _digest_json(
        {
            "parent_model_id": parent_model_id,
            "person_id": person_id,
            "person_data_hash": person_data_hash,
            "population_data_hash": population_data_hash,
            "feature_schema_hash": feature_schema_digest,
            "model_content_hash": model_content_digest,
            "validation_digest": validation_digest,
            "applicability_digest": applicability_digest,
            "applicability_event_ids": list(
                applicability_event_ids
            ),
            "applicability_data_hash": applicability_data_hash,
            "lineage_trial_hashes": list(lineage_trial_hashes),
            "lineage_design_hashes": list(lineage_design_hashes),
            "experiment_plan_ids": list(experiment_plan_ids),
            "training_config": [list(item) for item in training_config],
            "code_version": code_version,
        }
    )


@dataclass(frozen=True)
class ModelValidation:
    status: str
    validation_event_ids: tuple[str, ...]
    validation_data_hash: str | None
    sample_count: int
    personal_nll: float | None
    population_nll: float | None
    nll_uplift: float | None
    nll_uplift_ci_lower: float | None
    nll_uplift_ci_upper: float | None
    calibration_error: float | None
    personalization_passed: bool
    mechanism_probe_nll_uplift: float | None
    mechanism_adequacy_passed: bool | None
    temporal_stability_status: str
    temporal_drift_score: float | None
    temporal_critical_score_z: float | None
    temporal_score_effect: float | None
    temporal_early_nll: float | None
    temporal_late_nll: float | None
    temporal_early_sample_count: int
    temporal_late_sample_count: int
    temporal_drift_detected: bool
    reasons: tuple[str, ...]

    @classmethod
    def unvalidated(cls) -> ModelValidation:
        return cls(
            status="unvalidated",
            validation_event_ids=(),
            validation_data_hash=None,
            sample_count=0,
            personal_nll=None,
            population_nll=None,
            nll_uplift=None,
            nll_uplift_ci_lower=None,
            nll_uplift_ci_upper=None,
            calibration_error=None,
            personalization_passed=False,
            mechanism_probe_nll_uplift=None,
            mechanism_adequacy_passed=None,
            temporal_stability_status="not_assessed",
            temporal_drift_score=None,
            temporal_critical_score_z=None,
            temporal_score_effect=None,
            temporal_early_nll=None,
            temporal_late_nll=None,
            temporal_early_sample_count=0,
            temporal_late_sample_count=0,
            temporal_drift_detected=False,
            reasons=("independent_validation_required",),
        )

    def __post_init__(self) -> None:
        if self.status not in {"passed", "failed", "unvalidated"}:
            raise ValueError("unsupported model validation status")
        if len(set(self.validation_event_ids)) != len(
            self.validation_event_ids
        ):
            raise ValueError("validation event ids must be unique")
        if self.status == "unvalidated":
            if (
                self.validation_event_ids
                or self.validation_data_hash is not None
                or self.sample_count != 0
                or any(
                    value is not None
                    for value in (
                        self.personal_nll,
                        self.population_nll,
                        self.nll_uplift,
                        self.nll_uplift_ci_lower,
                        self.nll_uplift_ci_upper,
                        self.calibration_error,
                        self.mechanism_probe_nll_uplift,
                        self.temporal_drift_score,
                        self.temporal_critical_score_z,
                        self.temporal_score_effect,
                        self.temporal_early_nll,
                        self.temporal_late_nll,
                    )
                )
                or self.personalization_passed
                or self.mechanism_adequacy_passed
                or self.temporal_stability_status != "not_assessed"
                or self.temporal_early_sample_count != 0
                or self.temporal_late_sample_count != 0
                or self.temporal_drift_detected
            ):
                raise ValueError("unvalidated model cannot claim validation data")
            return
        if not self.validation_event_ids or self.validation_data_hash is None:
            raise ValueError("validated model requires validation data lineage")
        _require_digest(self.validation_data_hash, "validation_data_hash")
        if self.sample_count != len(self.validation_event_ids):
            raise ValueError("validation sample count does not match event ids")
        numeric_values = (
            self.personal_nll,
            self.population_nll,
            self.nll_uplift,
            self.nll_uplift_ci_lower,
            self.nll_uplift_ci_upper,
            self.calibration_error,
        )
        if any(
            value is None or not math.isfinite(value)
            for value in numeric_values
        ):
            raise ValueError("validated model requires finite diagnostics")
        if self.mechanism_adequacy_passed is not None and (
            self.mechanism_probe_nll_uplift is None
            or not math.isfinite(self.mechanism_probe_nll_uplift)
        ):
            raise ValueError(
                "assessed mechanism adequacy requires a finite probe uplift"
            )
        if self.personal_nll < 0 or self.population_nll < 0:
            raise ValueError("validation log loss cannot be negative")
        if not 0 <= self.calibration_error <= 1:
            raise ValueError("validation calibration error must be in [0, 1]")
        if self.nll_uplift_ci_lower > self.nll_uplift_ci_upper:
            raise ValueError("validation confidence interval is invalid")
        if self.temporal_stability_status not in {
            "stable",
            "unstable",
            "not_assessed",
        }:
            raise ValueError("unsupported temporal stability status")
        if self.temporal_drift_detected != (
            self.temporal_stability_status == "unstable"
        ):
            raise ValueError("temporal drift diagnostics are contradictory")
        if (
            self.temporal_stability_status == "not_assessed"
            and (
                self.temporal_drift_score is not None
                or self.temporal_critical_score_z is not None
                or self.temporal_score_effect is not None
                or self.temporal_early_nll is not None
                or self.temporal_late_nll is not None
                or self.temporal_early_sample_count != 0
                or self.temporal_late_sample_count != 0
            )
        ):
            raise ValueError(
                "unassessed temporal stability cannot claim a drift score"
            )
        if (
            self.temporal_stability_status != "not_assessed"
            and (
                self.temporal_drift_score is None
                or not math.isfinite(self.temporal_drift_score)
                or self.temporal_drift_score < 0
                or self.temporal_critical_score_z is None
                or not math.isfinite(self.temporal_critical_score_z)
                or self.temporal_critical_score_z <= 0
                or self.temporal_score_effect is None
                or not math.isfinite(self.temporal_score_effect)
                or self.temporal_score_effect < 0
                or self.temporal_early_nll is None
                or not math.isfinite(self.temporal_early_nll)
                or self.temporal_early_nll < 0
                or self.temporal_late_nll is None
                or not math.isfinite(self.temporal_late_nll)
                or self.temporal_late_nll < 0
                or self.temporal_early_sample_count <= 0
                or self.temporal_late_sample_count <= 0
            )
        ):
            raise ValueError(
                "assessed temporal stability requires a finite drift score"
            )
        if self.status == "passed" and (
            not self.personalization_passed
            or self.mechanism_adequacy_passed is False
            or self.temporal_drift_detected
            or self.temporal_stability_status != "stable"
            or self.reasons
        ):
            raise ValueError("passed validation has contradictory diagnostics")
        if self.status == "failed" and (
            not self.reasons
            or (
                self.personalization_passed
                and self.mechanism_adequacy_passed
                and not self.temporal_drift_detected
                and self.temporal_stability_status == "stable"
            )
        ):
            raise ValueError("failed validation has contradictory diagnostics")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "validation_event_ids": list(self.validation_event_ids),
            "validation_data_hash": self.validation_data_hash,
            "sample_count": self.sample_count,
            "personal_nll": self.personal_nll,
            "population_nll": self.population_nll,
            "nll_uplift": self.nll_uplift,
            "nll_uplift_ci_lower": self.nll_uplift_ci_lower,
            "nll_uplift_ci_upper": self.nll_uplift_ci_upper,
            "calibration_error": self.calibration_error,
            "personalization_passed": self.personalization_passed,
            "mechanism_probe_nll_uplift": (
                self.mechanism_probe_nll_uplift
            ),
            "mechanism_adequacy_passed": self.mechanism_adequacy_passed,
            "temporal_stability_status": self.temporal_stability_status,
            "temporal_drift_score": self.temporal_drift_score,
            "temporal_critical_score_z": self.temporal_critical_score_z,
            "temporal_score_effect": self.temporal_score_effect,
            "temporal_early_nll": self.temporal_early_nll,
            "temporal_late_nll": self.temporal_late_nll,
            "temporal_early_sample_count": (
                self.temporal_early_sample_count
            ),
            "temporal_late_sample_count": (
                self.temporal_late_sample_count
            ),
            "temporal_drift_detected": self.temporal_drift_detected,
            "reasons": list(self.reasons),
        }

    def digest(self) -> str:
        return _digest_json(self.to_dict())


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    parent_model_id: str | None
    person_id: str
    person_event_ids: tuple[str, ...]
    population_event_ids: tuple[str, ...]
    person_data_hash: str
    population_data_hash: str
    feature_schema_hash: str
    model_content_hash: str
    applicability_event_ids: tuple[str, ...]
    applicability_data_hash: str
    lineage_trial_hashes: tuple[str, ...]
    lineage_design_hashes: tuple[str, ...]
    experiment_plan_ids: tuple[str, ...]
    training_cutoff: str
    training_config: tuple[tuple[str, str], ...]
    validation: ModelValidation
    applicability_profile: ApplicabilityProfile
    code_version: str = "pcfm-mvp-0.8.0"

    def __post_init__(self) -> None:
        _require_digest(self.model_id, "model_id")
        if self.parent_model_id is not None:
            _require_digest(self.parent_model_id, "parent_model_id")
        for value, label in (
            (self.person_data_hash, "person_data_hash"),
            (self.population_data_hash, "population_data_hash"),
            (self.feature_schema_hash, "feature_schema_hash"),
            (self.model_content_hash, "model_content_hash"),
            (
                self.applicability_data_hash,
                "applicability_data_hash",
            ),
        ):
            _require_digest(value, label)
        if (
            not self.person_id
            or not self.person_event_ids
            or not self.population_event_ids
        ):
            raise ValueError("manifest person and event ids are required")
        if len(set(self.person_event_ids)) != len(self.person_event_ids):
            raise ValueError("manifest person_event_ids must be unique")
        if len(set(self.population_event_ids)) != len(
            self.population_event_ids
        ):
            raise ValueError("manifest population_event_ids must be unique")
        if (
            not self.applicability_event_ids
            or len(set(self.applicability_event_ids))
            != len(self.applicability_event_ids)
        ):
            raise ValueError(
                "manifest applicability_event_ids must be non-empty "
                "and unique"
            )
        if (
            not self.lineage_trial_hashes
            or len(set(self.lineage_trial_hashes))
            != len(self.lineage_trial_hashes)
        ):
            raise ValueError(
                "manifest lineage trial hashes must be non-empty "
                "and unique"
            )
        for value in self.lineage_trial_hashes:
            _require_digest(value, "lineage_trial_hash")
        if (
            not self.lineage_design_hashes
            or len(set(self.lineage_design_hashes))
            != len(self.lineage_design_hashes)
        ):
            raise ValueError(
                "manifest lineage design hashes must be non-empty "
                "and unique"
            )
        for value in self.lineage_design_hashes:
            _require_digest(value, "lineage_design_hash")
        if len(set(self.experiment_plan_ids)) != len(
            self.experiment_plan_ids
        ):
            raise ValueError(
                "manifest experiment plan ids must be unique"
            )
        for value in self.experiment_plan_ids:
            _require_digest(value, "experiment_plan_id")
        config_names = [name for name, _ in self.training_config]
        if not config_names or len(set(config_names)) != len(config_names):
            raise ValueError("manifest training_config must be non-empty and unique")
        if not self.training_cutoff or not self.code_version:
            raise ValueError("manifest versions and cutoff are required")


@dataclass(frozen=True)
class PersonModelBundle:
    population_model: PopulationModel
    representation: PersonRepresentation
    adapter: PersonalAdapter
    manifest: ModelManifest
    bundle_version: str = "pcfm-bundle-v7"

    def __post_init__(self) -> None:
        person_ids = {
            self.representation.person_id,
            self.adapter.person_id,
            self.manifest.person_id,
        }
        if len(person_ids) != 1:
            raise ValueError("bundle components refer to different person ids")
        if (
            self.population_model.feature_names
            != self.representation.feature_names
        ):
            raise ValueError("bundle components use different feature schemas")
        if (
            self.population_model.feature_names
            != self.manifest.applicability_profile.feature_names
        ):
            raise ValueError(
                "bundle applicability profile uses a different feature schema"
            )
        if len(self.adapter.delta_weights) != len(self.population_model.weights):
            raise ValueError("bundle adapter dimensions do not match")
        if (
            self.adapter.representation_version
            != self.representation.representation_version
        ):
            raise ValueError("bundle representation versions do not match")
        if self.representation.observation_count != len(
            self.manifest.person_event_ids
        ):
            raise ValueError("bundle observation count does not match manifest")
        expected_schema_hash = feature_schema_hash(
            self.population_model.feature_names
        )
        if self.manifest.feature_schema_hash != expected_schema_hash:
            raise ValueError("bundle feature schema hash does not match")
        expected_content_hash = model_content_hash(
            self.population_model,
            self.representation,
            self.adapter,
        )
        if self.manifest.model_content_hash != expected_content_hash:
            raise ValueError("bundle model content hash does not match")
        expected_model_id = manifest_model_id(
            parent_model_id=self.manifest.parent_model_id,
            person_id=self.manifest.person_id,
            person_data_hash=self.manifest.person_data_hash,
            population_data_hash=self.manifest.population_data_hash,
            feature_schema_digest=self.manifest.feature_schema_hash,
            model_content_digest=self.manifest.model_content_hash,
            validation_digest=self.manifest.validation.digest(),
            applicability_digest=(
                self.manifest.applicability_profile.digest()
            ),
            applicability_event_ids=(
                self.manifest.applicability_event_ids
            ),
            applicability_data_hash=(
                self.manifest.applicability_data_hash
            ),
            lineage_trial_hashes=self.manifest.lineage_trial_hashes,
            lineage_design_hashes=self.manifest.lineage_design_hashes,
            experiment_plan_ids=self.manifest.experiment_plan_ids,
            training_config=self.manifest.training_config,
            code_version=self.manifest.code_version,
        )
        if self.manifest.model_id != expected_model_id:
            raise ValueError("bundle model_id does not match its manifest")


def bundle_to_dict(bundle: PersonModelBundle) -> dict[str, object]:
    return {
        "bundle_version": bundle.bundle_version,
        "manifest": {
            "model_id": bundle.manifest.model_id,
            "parent_model_id": bundle.manifest.parent_model_id,
            "person_id": bundle.manifest.person_id,
            "person_event_ids": list(bundle.manifest.person_event_ids),
            "population_event_ids": list(
                bundle.manifest.population_event_ids
            ),
            "person_data_hash": bundle.manifest.person_data_hash,
            "population_data_hash": bundle.manifest.population_data_hash,
            "feature_schema_hash": bundle.manifest.feature_schema_hash,
            "model_content_hash": bundle.manifest.model_content_hash,
            "applicability_event_ids": list(
                bundle.manifest.applicability_event_ids
            ),
            "applicability_data_hash": (
                bundle.manifest.applicability_data_hash
            ),
            "lineage_trial_hashes": list(
                bundle.manifest.lineage_trial_hashes
            ),
            "lineage_design_hashes": list(
                bundle.manifest.lineage_design_hashes
            ),
            "experiment_plan_ids": list(
                bundle.manifest.experiment_plan_ids
            ),
            "training_cutoff": bundle.manifest.training_cutoff,
            "training_config": {
                name: value
                for name, value in bundle.manifest.training_config
            },
            "validation": bundle.manifest.validation.to_dict(),
            "applicability_profile": (
                bundle.manifest.applicability_profile.to_dict()
            ),
            "code_version": bundle.manifest.code_version,
        },
        "population_model": {
            "weights": list(bundle.population_model.weights),
            "covariance": [
                list(row) for row in bundle.population_model.covariance
            ],
            "feature_names": list(bundle.population_model.feature_names),
            "model_version": bundle.population_model.model_version,
        },
        "representation": {
            "person_id": bundle.representation.person_id,
            "latent_mean": list(bundle.representation.latent_mean),
            "covariance": [
                list(row) for row in bundle.representation.covariance
            ],
            "representation_version": (
                bundle.representation.representation_version
            ),
            "observation_count": bundle.representation.observation_count,
            "feature_names": list(bundle.representation.feature_names),
        },
        "adapter": {
            "person_id": bundle.adapter.person_id,
            "delta_weights": list(bundle.adapter.delta_weights),
            "adapter_version": bundle.adapter.adapter_version,
            "representation_version": (
                bundle.adapter.representation_version
            ),
        },
    }


def bundle_from_dict(data: dict[str, object]) -> PersonModelBundle:
    if data.get("bundle_version") != "pcfm-bundle-v7":
        raise ValueError("unsupported person model bundle version")
    manifest = dict(data["manifest"])
    population = dict(data["population_model"])
    representation = dict(data["representation"])
    adapter = dict(data["adapter"])
    validation = dict(manifest["validation"])
    applicability = dict(manifest["applicability_profile"])
    return PersonModelBundle(
        population_model=PopulationModel(
            weights=tuple(float(value) for value in population["weights"]),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in population["covariance"]
            ),
            feature_names=tuple(population["feature_names"]),
            model_version=str(population["model_version"]),
        ),
        representation=PersonRepresentation(
            person_id=str(representation["person_id"]),
            latent_mean=tuple(
                float(value) for value in representation["latent_mean"]
            ),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in representation["covariance"]
            ),
            representation_version=str(
                representation["representation_version"]
            ),
            observation_count=int(representation["observation_count"]),
            feature_names=tuple(representation["feature_names"]),
        ),
        adapter=PersonalAdapter(
            person_id=str(adapter["person_id"]),
            delta_weights=tuple(
                float(value) for value in adapter["delta_weights"]
            ),
            adapter_version=str(adapter["adapter_version"]),
            representation_version=str(adapter["representation_version"]),
        ),
        manifest=ModelManifest(
            model_id=str(manifest["model_id"]),
            parent_model_id=(
                str(manifest["parent_model_id"])
                if manifest.get("parent_model_id") is not None
                else None
            ),
            person_id=str(manifest["person_id"]),
            person_event_ids=tuple(
                str(value) for value in manifest["person_event_ids"]
            ),
            population_event_ids=tuple(
                str(value) for value in manifest["population_event_ids"]
            ),
            person_data_hash=str(manifest["person_data_hash"]),
            population_data_hash=str(manifest["population_data_hash"]),
            feature_schema_hash=str(manifest["feature_schema_hash"]),
            model_content_hash=str(manifest["model_content_hash"]),
            applicability_event_ids=tuple(
                str(value)
                for value in manifest["applicability_event_ids"]
            ),
            applicability_data_hash=str(
                manifest["applicability_data_hash"]
            ),
            lineage_trial_hashes=tuple(
                str(value)
                for value in manifest["lineage_trial_hashes"]
            ),
            lineage_design_hashes=tuple(
                str(value)
                for value in manifest["lineage_design_hashes"]
            ),
            experiment_plan_ids=tuple(
                str(value)
                for value in manifest["experiment_plan_ids"]
            ),
            training_cutoff=str(manifest["training_cutoff"]),
            training_config=tuple(
                sorted(
                    (str(name), str(value))
                    for name, value in dict(
                        manifest["training_config"]
                    ).items()
                )
            ),
            validation=ModelValidation(
                status=str(validation["status"]),
                validation_event_ids=tuple(
                    str(value)
                    for value in validation["validation_event_ids"]
                ),
                validation_data_hash=(
                    str(validation["validation_data_hash"])
                    if validation.get("validation_data_hash") is not None
                    else None
                ),
                sample_count=int(validation["sample_count"]),
                personal_nll=(
                    float(validation["personal_nll"])
                    if validation.get("personal_nll") is not None
                    else None
                ),
                population_nll=(
                    float(validation["population_nll"])
                    if validation.get("population_nll") is not None
                    else None
                ),
                nll_uplift=(
                    float(validation["nll_uplift"])
                    if validation.get("nll_uplift") is not None
                    else None
                ),
                nll_uplift_ci_lower=(
                    float(validation["nll_uplift_ci_lower"])
                    if validation.get("nll_uplift_ci_lower") is not None
                    else None
                ),
                nll_uplift_ci_upper=(
                    float(validation["nll_uplift_ci_upper"])
                    if validation.get("nll_uplift_ci_upper") is not None
                    else None
                ),
                calibration_error=(
                    float(validation["calibration_error"])
                    if validation.get("calibration_error") is not None
                    else None
                ),
                personalization_passed=bool(
                    validation["personalization_passed"]
                ),
                mechanism_probe_nll_uplift=(
                    float(validation["mechanism_probe_nll_uplift"])
                    if validation.get("mechanism_probe_nll_uplift")
                    is not None
                    else None
                ),
                mechanism_adequacy_passed=(
                    bool(validation["mechanism_adequacy_passed"])
                    if validation.get("mechanism_adequacy_passed") is not None
                    else None
                ),
                temporal_stability_status=str(
                    validation["temporal_stability_status"]
                ),
                temporal_drift_score=(
                    float(validation["temporal_drift_score"])
                    if validation.get("temporal_drift_score") is not None
                    else None
                ),
                temporal_critical_score_z=(
                    float(validation["temporal_critical_score_z"])
                    if validation.get("temporal_critical_score_z")
                    is not None
                    else None
                ),
                temporal_score_effect=(
                    float(validation["temporal_score_effect"])
                    if validation.get("temporal_score_effect") is not None
                    else None
                ),
                temporal_early_nll=(
                    float(validation["temporal_early_nll"])
                    if validation.get("temporal_early_nll") is not None
                    else None
                ),
                temporal_late_nll=(
                    float(validation["temporal_late_nll"])
                    if validation.get("temporal_late_nll") is not None
                    else None
                ),
                temporal_early_sample_count=int(
                    validation["temporal_early_sample_count"]
                ),
                temporal_late_sample_count=int(
                    validation["temporal_late_sample_count"]
                ),
                temporal_drift_detected=bool(
                    validation["temporal_drift_detected"]
                ),
                reasons=tuple(
                    str(value) for value in validation["reasons"]
                ),
            ),
            applicability_profile=ApplicabilityProfile(
                feature_names=tuple(
                    str(value) for value in applicability["feature_names"]
                ),
                center=tuple(
                    float(value) for value in applicability["center"]
                ),
                inverse_covariance=tuple(
                    tuple(float(value) for value in row)
                    for row in applicability["inverse_covariance"]
                ),
                squared_distance_threshold=float(
                    applicability["squared_distance_threshold"]
                ),
                reference_features=tuple(
                    tuple(float(value) for value in row)
                    for row in applicability["reference_features"]
                ),
                local_neighbor_count=int(
                    applicability["local_neighbor_count"]
                ),
                local_squared_distance_threshold=float(
                    applicability["local_squared_distance_threshold"]
                ),
                reference_sample_count=int(
                    applicability["reference_sample_count"]
                ),
                calibration_sample_count=int(
                    applicability["calibration_sample_count"]
                ),
                supported_domains=tuple(
                    str(value)
                    for value in applicability["supported_domains"]
                ),
                supported_option_signatures=tuple(
                    str(value)
                    for value in applicability[
                        "supported_option_signatures"
                    ]
                ),
                supported_context_signatures=tuple(
                    str(value)
                    for value in applicability[
                        "supported_context_signatures"
                    ]
                ),
                valid_through=(
                    str(applicability["valid_through"])
                    if applicability.get("valid_through") is not None
                    else None
                ),
                maximum_age_days=float(
                    applicability["maximum_age_days"]
                ),
                calibration_safety_factor=float(
                    applicability["calibration_safety_factor"]
                ),
                model_version=str(applicability["model_version"]),
            ),
            code_version=str(manifest["code_version"]),
        ),
        bundle_version=str(data["bundle_version"]),
    )


def save_bundle(path: Path, bundle: PersonModelBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle_to_dict(bundle), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_bundle(path: Path) -> PersonModelBundle:
    return bundle_from_dict(json.loads(path.read_text(encoding="utf-8")))
