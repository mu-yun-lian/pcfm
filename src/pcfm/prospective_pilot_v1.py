from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import EvaluationReport, Observation, Scenario
from .evaluation import evaluate_probability_array
from .ledger import EventLedger, VerificationAuthority


REQUIRED_METHOD_KINDS = frozenset(
    {
        "pcfm_person_model",
        "population_model",
        "constant_history",
        "profile_llm",
    }
)
PRIMARY_METHOD_KIND = "pcfm_person_model"
MINIMUM_SCENARIO_COUNT = 100
CONFIDENCE_Z_95 = 1.96
NEAR_DUPLICATE_FEATURE_DISTANCE = 1e-6


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


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
        "features": [
            [name, float(value)]
            for name, value in zip(
                scenario.feature_names,
                scenario.features,
                strict=True,
            )
        ],
        "options": list(scenario.options),
        "domain": scenario.domain,
        "context": dict(sorted(scenario.context.items())),
    }


def _scenario_design_digest(scenario: Scenario) -> str:
    payload = _scenario_payload(scenario)
    payload.pop("scenario_id")
    return _digest_json(payload)


def _scenario_from_dict(data: Mapping[str, object]) -> Scenario:
    raw_features = data["features"]
    if isinstance(raw_features, Mapping):
        feature_names = tuple(str(name) for name in raw_features)
        features = tuple(float(raw_features[name]) for name in raw_features)
    else:
        pairs = tuple(raw_features)
        if pairs and isinstance(pairs[0], (list, tuple)):
            feature_names = tuple(str(item[0]) for item in pairs)
            features = tuple(float(item[1]) for item in pairs)
        else:
            feature_names = tuple(str(name) for name in data["feature_names"])
            features = tuple(float(value) for value in pairs)
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


def _binary_log_loss(
    probabilities: Sequence[float],
    choices: Sequence[int],
) -> np.ndarray:
    probability_array = np.clip(
        np.asarray(probabilities, dtype=np.float64),
        1e-9,
        1.0 - 1e-9,
    )
    choice_array = np.asarray(choices, dtype=np.float64)
    return -(
        choice_array * np.log(probability_array)
        + (1.0 - choice_array) * np.log(1.0 - probability_array)
    )


class ProspectivePilotRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__(
            "prospective pilot refused: " + ", ".join(self.reasons)
        )


@dataclass(frozen=True)
class PilotConfig:
    minimum_nll_uplift: float = 0.01
    maximum_primary_nll: float = 0.65
    maximum_calibration_error: float = 0.15
    artifact_version: str = "prospective-pilot-config-v1"

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_nll_uplift)
            or self.minimum_nll_uplift < 0.01
        ):
            raise ValueError("minimum_nll_uplift must be at least 0.01")
        if (
            not isfinite(self.maximum_primary_nll)
            or self.maximum_primary_nll <= 0
            or self.maximum_primary_nll > 0.65
        ):
            raise ValueError("maximum_primary_nll cannot exceed 0.65")
        if (
            not isfinite(self.maximum_calibration_error)
            or self.maximum_calibration_error <= 0
            or self.maximum_calibration_error > 0.15
        ):
            raise ValueError(
                "maximum_calibration_error cannot exceed 0.15"
            )
        if self.artifact_version != "prospective-pilot-config-v1":
            raise ValueError("unsupported pilot config version")

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_nll_uplift": self.minimum_nll_uplift,
            "maximum_primary_nll": self.maximum_primary_nll,
            "maximum_calibration_error": (
                self.maximum_calibration_error
            ),
            "artifact_version": self.artifact_version,
        }


def pilot_config_from_dict(data: Mapping[str, object]) -> PilotConfig:
    return PilotConfig(
        minimum_nll_uplift=float(data["minimum_nll_uplift"]),
        maximum_primary_nll=float(data["maximum_primary_nll"]),
        maximum_calibration_error=float(
            data["maximum_calibration_error"]
        ),
        artifact_version=str(data["artifact_version"]),
    )


@dataclass(frozen=True)
class PilotForecast:
    method_kind: str
    model_reference: str
    training_cutoff: str
    probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.method_kind not in REQUIRED_METHOD_KINDS:
            raise ValueError("unsupported pilot forecast method")
        if not self.model_reference:
            raise ValueError("forecast model_reference is required")
        _require_digest(self.model_reference, "model_reference")
        _parse_timestamp(self.training_cutoff, "training_cutoff")
        converted = tuple(
            sorted(
                (
                    (str(scenario_id), float(probability))
                    for scenario_id, probability in self.probabilities
                ),
                key=lambda item: item[0],
            )
        )
        if not converted:
            raise ValueError("forecast probabilities must not be empty")
        scenario_ids = tuple(item[0] for item in converted)
        if (
            any(not scenario_id for scenario_id in scenario_ids)
            or len(set(scenario_ids)) != len(scenario_ids)
        ):
            raise ValueError(
                "forecast scenario identities must be non-empty and unique"
            )
        if any(
            not isfinite(probability)
            or not 0.0 < probability < 1.0
            for _, probability in converted
        ):
            raise ValueError(
                "forecast probabilities must be finite and strictly between zero and one"
            )
        object.__setattr__(self, "probabilities", converted)

    def to_dict(self) -> dict[str, object]:
        return {
            "method_kind": self.method_kind,
            "model_reference": self.model_reference,
            "training_cutoff": self.training_cutoff,
            "probabilities": [
                [scenario_id, probability]
                for scenario_id, probability in self.probabilities
            ],
        }


def pilot_forecast_from_dict(
    data: Mapping[str, object],
) -> PilotForecast:
    raw_probabilities = data["probabilities"]
    if isinstance(raw_probabilities, Mapping):
        probabilities = tuple(
            (str(scenario_id), float(probability))
            for scenario_id, probability in raw_probabilities.items()
        )
    else:
        probabilities = tuple(
            (str(item[0]), float(item[1]))
            for item in raw_probabilities
        )
    return PilotForecast(
        method_kind=str(data["method_kind"]),
        model_reference=str(data["model_reference"]),
        training_cutoff=str(data["training_cutoff"]),
        probabilities=probabilities,
    )


@dataclass(frozen=True)
class ProspectivePilotPlan:
    person_id: str
    created_at: str
    collection_end: str
    scenarios: tuple[Scenario, ...]
    forecasts: tuple[PilotForecast, ...]
    config: PilotConfig
    verifier_id: str
    artifact_version: str = "prospective-single-person-pilot-v1"
    plan_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        if not self.person_id or not self.verifier_id:
            raise ValueError("pilot person and verifier identities are required")
        created_at = _parse_timestamp(self.created_at, "created_at")
        collection_end = _parse_timestamp(
            self.collection_end,
            "collection_end",
        )
        if collection_end <= created_at:
            raise ValueError("collection_end must follow created_at")
        if self.artifact_version != "prospective-single-person-pilot-v1":
            raise ValueError("unsupported pilot plan version")
        if self.signature_method != "hmac-sha256":
            raise ValueError("unsupported pilot plan signature method")

        scenarios = tuple(
            sorted(self.scenarios, key=lambda item: item.scenario_id)
        )
        if len(scenarios) < MINIMUM_SCENARIO_COUNT:
            raise ValueError("pilot requires at least 100 scenarios")
        scenario_ids = tuple(item.scenario_id for item in scenarios)
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("pilot scenario ids must be unique")
        design_hashes = tuple(_scenario_design_digest(item) for item in scenarios)
        if len(set(design_hashes)) != len(design_hashes):
            raise ValueError("pilot scenario designs must be unique")
        domains = {item.domain for item in scenarios}
        option_sets = {item.options for item in scenarios}
        feature_schemas = {item.feature_names for item in scenarios}
        if len(domains) != 1:
            raise ValueError("pilot scenarios must share one domain")
        if len(option_sets) != 1:
            raise ValueError("pilot scenarios must share one option set")
        if len(feature_schemas) != 1:
            raise ValueError("pilot scenarios must share one feature schema")
        if any(
            not str(item.context.get("question_text", "")).strip()
            for item in scenarios
        ):
            raise ValueError(
                "every pilot scenario must bind non-empty question_text"
            )
        for left_index, left in enumerate(scenarios):
            left_features = np.asarray(left.features, dtype=np.float64)
            for right in scenarios[left_index + 1 :]:
                if (
                    left.domain == right.domain
                    and left.options == right.options
                    and dict(left.context) == dict(right.context)
                    and float(
                        np.max(
                            np.abs(
                                left_features
                                - np.asarray(
                                    right.features,
                                    dtype=np.float64,
                                )
                            )
                        )
                    )
                    <= NEAR_DUPLICATE_FEATURE_DISTANCE
                ):
                    raise ValueError(
                        "pilot scenario designs contain a near-duplicate"
                    )
        object.__setattr__(self, "scenarios", scenarios)

        forecasts = tuple(
            sorted(self.forecasts, key=lambda item: item.method_kind)
        )
        method_kinds = {item.method_kind for item in forecasts}
        if (
            len(forecasts) != len(REQUIRED_METHOD_KINDS)
            or method_kinds != REQUIRED_METHOD_KINDS
        ):
            raise ValueError(
                "required forecast methods are pcfm_person_model, "
                "population_model, constant_history, and profile_llm"
            )
        expected_ids = set(scenario_ids)
        for forecast in forecasts:
            if {
                scenario_id
                for scenario_id, _ in forecast.probabilities
            } != expected_ids:
                raise ValueError(
                    "forecast probabilities must cover the exact scenario set"
                )
            if (
                _parse_timestamp(
                    forecast.training_cutoff,
                    "training_cutoff",
                )
                > created_at
            ):
                raise ValueError(
                    "forecast training_cutoff cannot follow plan creation"
                )
        object.__setattr__(self, "forecasts", forecasts)

        expected_id = self.digest()
        if self.plan_id:
            _require_digest(self.plan_id, "plan_id")
            if self.plan_id != expected_id:
                raise ValueError("pilot plan_id does not match content")
        else:
            object.__setattr__(self, "plan_id", expected_id)
        if self.signature:
            _require_digest(self.signature, "signature")

    @property
    def domain(self) -> str:
        return self.scenarios[0].domain

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.scenarios[0].feature_names

    @property
    def options(self) -> tuple[str, str]:
        return self.scenarios[0].options

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "person_id": self.person_id,
            "created_at": self.created_at,
            "collection_end": self.collection_end,
            "scenarios": [
                _scenario_payload(scenario) for scenario in self.scenarios
            ],
            "forecasts": [
                forecast.to_dict() for forecast in self.forecasts
            ],
            "config": self.config.to_dict(),
            "verifier_id": self.verifier_id,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "plan_id": self.plan_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("pilot plan is unsigned")
        authority.verify_payload(
            self.signed_payload(),
            self.verifier_id,
            self.signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.signed_payload(), "signature": self.signature}


def create_pilot_plan(
    *,
    person_id: str,
    scenarios: Sequence[Scenario],
    forecasts: Sequence[PilotForecast],
    authority: VerificationAuthority,
    verifier_id: str,
    created_at: str,
    collection_end: str,
    config: PilotConfig | None = None,
) -> ProspectivePilotPlan:
    if not scenarios:
        raise ValueError("pilot scenarios must not be empty")
    first_names = tuple(scenarios[0].feature_names)
    normalized = tuple(
        Scenario(
            scenario_id=scenario.scenario_id,
            features=scenario.ordered_features(first_names),
            feature_names=first_names,
            options=scenario.options,
            domain=scenario.domain,
            context=dict(scenario.context),
        )
        for scenario in scenarios
    )
    unsigned = ProspectivePilotPlan(
        person_id=person_id,
        created_at=created_at,
        collection_end=collection_end,
        scenarios=normalized,
        forecasts=tuple(forecasts),
        config=config or PilotConfig(),
        verifier_id=verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        verifier_id,
    )
    return replace(unsigned, signature=signature)


def pilot_plan_from_dict(
    data: Mapping[str, object],
) -> ProspectivePilotPlan:
    return ProspectivePilotPlan(
        person_id=str(data["person_id"]),
        created_at=str(data["created_at"]),
        collection_end=str(data["collection_end"]),
        scenarios=tuple(
            _scenario_from_dict(dict(item))
            for item in data["scenarios"]
        ),
        forecasts=tuple(
            pilot_forecast_from_dict(dict(item))
            for item in data["forecasts"]
        ),
        config=pilot_config_from_dict(dict(data["config"])),
        verifier_id=str(data["verifier_id"]),
        artifact_version=str(data["artifact_version"]),
        plan_id=str(data["plan_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )


@dataclass(frozen=True)
class PilotRegistryReceipt:
    plan_id: str
    plan_verifier_id: str
    registered_at: str
    registry_verifier_id: str
    artifact_version: str = "prospective-pilot-registry-receipt-v1"
    receipt_id: str = ""
    signature: str = ""
    signature_method: str = "hmac-sha256"

    def __post_init__(self) -> None:
        _require_digest(self.plan_id, "plan_id")
        _parse_timestamp(self.registered_at, "registered_at")
        if (
            not self.plan_verifier_id
            or not self.registry_verifier_id
            or self.plan_verifier_id == self.registry_verifier_id
        ):
            raise ValueError("registry role must differ from plan role")
        if (
            self.artifact_version
            != "prospective-pilot-registry-receipt-v1"
        ):
            raise ValueError("unsupported pilot receipt version")
        if self.signature_method != "hmac-sha256":
            raise ValueError("unsupported pilot receipt signature method")
        expected_id = self.digest()
        if self.receipt_id:
            _require_digest(self.receipt_id, "receipt_id")
            if self.receipt_id != expected_id:
                raise ValueError("pilot receipt_id does not match content")
        else:
            object.__setattr__(self, "receipt_id", expected_id)
        if self.signature:
            _require_digest(self.signature, "signature")

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "plan_id": self.plan_id,
            "plan_verifier_id": self.plan_verifier_id,
            "registered_at": self.registered_at,
            "registry_verifier_id": self.registry_verifier_id,
            "signature_method": self.signature_method,
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def signed_payload(self) -> dict[str, object]:
        return {**self._content_dict(), "receipt_id": self.receipt_id}

    def verify(self, authority: VerificationAuthority) -> None:
        if not self.signature:
            raise ValueError("pilot registry receipt is unsigned")
        authority.verify_payload(
            self.signed_payload(),
            self.registry_verifier_id,
            self.signature,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.signed_payload(), "signature": self.signature}


def register_pilot_plan(
    plan: ProspectivePilotPlan,
    authority: VerificationAuthority,
    *,
    registry_verifier_id: str,
    registered_at: str,
) -> PilotRegistryReceipt:
    try:
        plan.verify(authority)
    except ValueError as error:
        raise ProspectivePilotRefusedError(
            ("pilot_plan_signature_invalid",)
        ) from error
    if registry_verifier_id == plan.verifier_id:
        raise ProspectivePilotRefusedError(
            ("registry_role_not_independent",)
        )
    registered = _parse_timestamp(registered_at, "registered_at")
    if registered < _parse_timestamp(plan.created_at, "created_at"):
        raise ProspectivePilotRefusedError(
            ("registration_precedes_plan",)
        )
    if registered >= _parse_timestamp(
        plan.collection_end,
        "collection_end",
    ):
        raise ProspectivePilotRefusedError(
            ("registration_not_before_collection_end",)
        )
    unsigned = PilotRegistryReceipt(
        plan_id=plan.plan_id,
        plan_verifier_id=plan.verifier_id,
        registered_at=registered_at,
        registry_verifier_id=registry_verifier_id,
    )
    signature = authority.sign_payload(
        unsigned.signed_payload(),
        registry_verifier_id,
    )
    return replace(unsigned, signature=signature)


def pilot_receipt_from_dict(
    data: Mapping[str, object],
) -> PilotRegistryReceipt:
    return PilotRegistryReceipt(
        plan_id=str(data["plan_id"]),
        plan_verifier_id=str(data["plan_verifier_id"]),
        registered_at=str(data["registered_at"]),
        registry_verifier_id=str(data["registry_verifier_id"]),
        artifact_version=str(data["artifact_version"]),
        receipt_id=str(data["receipt_id"]),
        signature=str(data["signature"]),
        signature_method=str(data["signature_method"]),
    )


@dataclass(frozen=True)
class PilotMethodMetrics:
    method_kind: str
    sample_count: int
    negative_log_likelihood: float
    brier_score: float
    accuracy: float
    expected_calibration_error: float

    @classmethod
    def from_evaluation(
        cls,
        method_kind: str,
        report: EvaluationReport,
    ) -> PilotMethodMetrics:
        return cls(
            method_kind=method_kind,
            sample_count=report.sample_count,
            negative_log_likelihood=report.negative_log_likelihood,
            brier_score=report.brier_score,
            accuracy=report.accuracy,
            expected_calibration_error=(
                report.expected_calibration_error
            ),
        )

    def __post_init__(self) -> None:
        if self.method_kind not in REQUIRED_METHOD_KINDS:
            raise ValueError("unsupported pilot metric method")
        if self.sample_count < MINIMUM_SCENARIO_COUNT:
            raise ValueError("pilot metric sample count is below hard floor")
        metrics = (
            self.negative_log_likelihood,
            self.brier_score,
            self.accuracy,
            self.expected_calibration_error,
        )
        if not all(isfinite(value) for value in metrics):
            raise ValueError("pilot metrics must be finite")
        if (
            self.negative_log_likelihood < 0
            or not 0 <= self.brier_score <= 1
            or not 0 <= self.accuracy <= 1
            or not 0 <= self.expected_calibration_error <= 1
        ):
            raise ValueError("pilot metrics are outside valid bounds")

    def to_dict(self) -> dict[str, object]:
        return {
            "method_kind": self.method_kind,
            "sample_count": self.sample_count,
            "negative_log_likelihood": self.negative_log_likelihood,
            "brier_score": self.brier_score,
            "accuracy": self.accuracy,
            "expected_calibration_error": (
                self.expected_calibration_error
            ),
        }


def _method_metrics_from_dict(
    data: Mapping[str, object],
) -> PilotMethodMetrics:
    return PilotMethodMetrics(
        method_kind=str(data["method_kind"]),
        sample_count=int(data["sample_count"]),
        negative_log_likelihood=float(data["negative_log_likelihood"]),
        brier_score=float(data["brier_score"]),
        accuracy=float(data["accuracy"]),
        expected_calibration_error=float(
            data["expected_calibration_error"]
        ),
    )


@dataclass(frozen=True)
class PilotComparison:
    baseline_kind: str
    mean_nll_gain: float
    standard_error: float
    ci_lower_95: float
    ci_upper_95: float
    minimum_required_gain: float
    dependence_method: str
    hac_lag: int
    passed: bool

    def __post_init__(self) -> None:
        if (
            self.baseline_kind not in REQUIRED_METHOD_KINDS
            or self.baseline_kind == PRIMARY_METHOD_KIND
        ):
            raise ValueError("invalid pilot comparison baseline")
        values = (
            self.mean_nll_gain,
            self.standard_error,
            self.ci_lower_95,
            self.ci_upper_95,
            self.minimum_required_gain,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("pilot comparison values must be finite")
        if self.standard_error < 0 or self.minimum_required_gain < 0.01:
            raise ValueError("pilot comparison gates are invalid")
        if (
            self.dependence_method != "newey-west-bartlett"
            or self.hac_lag <= 0
        ):
            raise ValueError(
                "pilot comparison dependence correction is invalid"
            )
        if (
            not np.isclose(
                self.ci_lower_95,
                self.mean_nll_gain
                - CONFIDENCE_Z_95 * self.standard_error,
                rtol=1e-12,
                atol=1e-12,
            )
            or not np.isclose(
                self.ci_upper_95,
                self.mean_nll_gain
                + CONFIDENCE_Z_95 * self.standard_error,
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError("pilot comparison interval is inconsistent")
        expected_passed = (
            self.mean_nll_gain >= self.minimum_required_gain
            and self.ci_lower_95 > 0
        )
        if self.passed != expected_passed:
            raise ValueError("pilot comparison status is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_kind": self.baseline_kind,
            "mean_nll_gain": self.mean_nll_gain,
            "standard_error": self.standard_error,
            "ci_lower_95": self.ci_lower_95,
            "ci_upper_95": self.ci_upper_95,
            "minimum_required_gain": self.minimum_required_gain,
            "dependence_method": self.dependence_method,
            "hac_lag": self.hac_lag,
            "passed": self.passed,
        }


def _comparison_from_dict(
    data: Mapping[str, object],
) -> PilotComparison:
    return PilotComparison(
        baseline_kind=str(data["baseline_kind"]),
        mean_nll_gain=float(data["mean_nll_gain"]),
        standard_error=float(data["standard_error"]),
        ci_lower_95=float(data["ci_lower_95"]),
        ci_upper_95=float(data["ci_upper_95"]),
        minimum_required_gain=float(data["minimum_required_gain"]),
        dependence_method=str(data["dependence_method"]),
        hac_lag=int(data["hac_lag"]),
        passed=bool(data["passed"]),
    )


@dataclass(frozen=True)
class ProspectivePilotReport:
    status: str
    plan_id: str
    receipt_id: str
    outcome_event_ids: tuple[str, ...]
    outcome_data_hash: str
    primary_metrics: PilotMethodMetrics
    method_metrics: tuple[PilotMethodMetrics, ...]
    comparisons: tuple[PilotComparison, ...]
    reasons: tuple[str, ...]
    artifact_version: str = "prospective-pilot-report-v1"
    report_id: str = ""

    def __post_init__(self) -> None:
        if self.status not in {
            "passed_prospective_pilot",
            "completed_no_support",
        }:
            raise ValueError("unsupported prospective pilot status")
        _require_digest(self.plan_id, "plan_id")
        _require_digest(self.receipt_id, "receipt_id")
        _require_digest(self.outcome_data_hash, "outcome_data_hash")
        if (
            len(self.outcome_event_ids) < MINIMUM_SCENARIO_COUNT
            or len(set(self.outcome_event_ids))
            != len(self.outcome_event_ids)
        ):
            raise ValueError("pilot outcome event identities are invalid")
        if self.primary_metrics.method_kind != PRIMARY_METHOD_KIND:
            raise ValueError("pilot primary metrics use the wrong method")
        metrics = tuple(
            sorted(self.method_metrics, key=lambda item: item.method_kind)
        )
        if {item.method_kind for item in metrics} != REQUIRED_METHOD_KINDS:
            raise ValueError("pilot report method set is incomplete")
        if any(
            item.sample_count != len(self.outcome_event_ids)
            for item in metrics
        ):
            raise ValueError(
                "pilot report metric sample count is inconsistent"
            )
        if self.primary_metrics != next(
            item
            for item in metrics
            if item.method_kind == PRIMARY_METHOD_KIND
        ):
            raise ValueError("pilot primary metrics are inconsistent")
        object.__setattr__(self, "method_metrics", metrics)
        comparisons = tuple(
            sorted(self.comparisons, key=lambda item: item.baseline_kind)
        )
        if {
            item.baseline_kind for item in comparisons
        } != REQUIRED_METHOD_KINDS - {PRIMARY_METHOD_KIND}:
            raise ValueError("pilot comparison set is incomplete")
        object.__setattr__(self, "comparisons", comparisons)
        reasons = tuple(sorted(set(self.reasons)))
        object.__setattr__(self, "reasons", reasons)
        should_pass = not reasons and all(
            item.passed for item in comparisons
        )
        if (self.status == "passed_prospective_pilot") != should_pass:
            raise ValueError("pilot report status is inconsistent")
        if self.artifact_version != "prospective-pilot-report-v1":
            raise ValueError("unsupported pilot report version")
        expected_id = self.digest()
        if self.report_id:
            _require_digest(self.report_id, "report_id")
            if self.report_id != expected_id:
                raise ValueError("pilot report_id does not match content")
        else:
            object.__setattr__(self, "report_id", expected_id)

    @property
    def sample_count(self) -> int:
        return len(self.outcome_event_ids)

    @property
    def outcome_count(self) -> int:
        return len(self.outcome_event_ids)

    def _content_dict(self) -> dict[str, object]:
        return {
            "artifact_version": self.artifact_version,
            "status": self.status,
            "plan_id": self.plan_id,
            "receipt_id": self.receipt_id,
            "outcome_event_ids": list(self.outcome_event_ids),
            "outcome_count": self.outcome_count,
            "outcome_data_hash": self.outcome_data_hash,
            "primary_metrics": self.primary_metrics.to_dict(),
            "method_metrics": [
                item.to_dict() for item in self.method_metrics
            ],
            "comparisons": [
                item.to_dict() for item in self.comparisons
            ],
            "reasons": list(self.reasons),
        }

    def digest(self) -> str:
        return _digest_json(self._content_dict())

    def to_dict(self) -> dict[str, object]:
        return {**self._content_dict(), "report_id": self.report_id}


def _verify_inputs(
    plan: ProspectivePilotPlan,
    receipt: PilotRegistryReceipt,
    outcomes: EventLedger,
    authority: VerificationAuthority,
) -> tuple[tuple[Observation, ...], tuple[str, ...], str]:
    reasons = []
    try:
        plan.verify(authority)
    except ValueError:
        reasons.append("pilot_plan_signature_invalid")
    try:
        receipt.verify(authority)
    except ValueError:
        reasons.append("registry_receipt_signature_invalid")
    try:
        verified = EventLedger.verify(outcomes.records, authority)
    except ValueError as error:
        raise ProspectivePilotRefusedError(
            ("outcome_ledger_signature_invalid",)
        ) from error
    if receipt.plan_id != plan.plan_id:
        reasons.append("registry_receipt_plan_mismatch")
    if receipt.plan_verifier_id != plan.verifier_id:
        reasons.append("registry_receipt_role_mismatch")
    if receipt.registry_verifier_id == plan.verifier_id:
        reasons.append("registry_role_not_independent")
    if len(verified.records) != len(plan.scenarios):
        reasons.append("outcome_count_mismatch")
    if any(
        record.observation.person_id != plan.person_id
        for record in verified.records
    ):
        reasons.append("outcome_person_mismatch")
    outcome_verifiers = {record.verifier_id for record in verified.records}
    if (
        len(outcome_verifiers) != 1
        or plan.verifier_id in outcome_verifiers
        or receipt.registry_verifier_id in outcome_verifiers
    ):
        reasons.append("outcome_role_not_independent")
    if any(
        record.observation.provenance != "human_record"
        for record in verified.records
    ):
        reasons.append("outcome_provenance_not_human")

    registered_at = _parse_timestamp(
        receipt.registered_at,
        "registered_at",
    )
    collection_end = _parse_timestamp(
        plan.collection_end,
        "collection_end",
    )
    if any(
        _parse_timestamp(record.observed_at, "observed_at")
        <= registered_at
        for record in verified.records
    ):
        reasons.append("outcome_precedes_registration")
    if any(
        _parse_timestamp(record.observed_at, "observed_at")
        > collection_end
        for record in verified.records
    ):
        reasons.append("outcome_after_collection_end")
    observed_timestamps = tuple(
        record.observed_at for record in verified.records
    )
    if len(set(observed_timestamps)) != len(observed_timestamps):
        reasons.append("outcome_timestamp_sequence_ambiguous")

    planned = {
        scenario.scenario_id: scenario for scenario in plan.scenarios
    }
    actual_ids = {
        record.observation.scenario.scenario_id
        for record in verified.records
    }
    if actual_ids != set(planned):
        reasons.append("outcome_scenario_set_mismatch")
    by_scenario = {}
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
            reasons.append("outcome_scenario_content_mismatch")
            continue
        if _scenario_payload(normalized) != _scenario_payload(expected):
            reasons.append("outcome_scenario_content_mismatch")
        by_scenario[observed.scenario_id] = record
    if reasons:
        raise ProspectivePilotRefusedError(reasons)

    ordered_records = tuple(
        sorted(
            by_scenario.values(),
            key=lambda record: _parse_timestamp(
                record.observed_at,
                "observed_at",
            ),
        )
    )
    return (
        tuple(record.observation for record in ordered_records),
        tuple(record.event_id for record in ordered_records),
        EventLedger.snapshot_hash(verified.records),
    )


def _newey_west_standard_error(
    values: np.ndarray,
) -> tuple[float, int]:
    count = len(values)
    lag_count = max(1, int(np.floor(count ** (1.0 / 3.0))))
    centered = values - float(np.mean(values))
    long_run_variance = float(centered @ centered) / count
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        autocovariance = float(
            centered[lag:] @ centered[:-lag]
        ) / count
        long_run_variance += 2.0 * weight * autocovariance
    return (
        float(np.sqrt(max(long_run_variance, 0.0) / count)),
        lag_count,
    )


def score_prospective_pilot(
    plan: ProspectivePilotPlan,
    receipt: PilotRegistryReceipt,
    outcomes: EventLedger,
    authority: VerificationAuthority,
) -> ProspectivePilotReport:
    observations, event_ids, outcome_hash = _verify_inputs(
        plan,
        receipt,
        outcomes,
        authority,
    )
    ordered_ids = tuple(
        observation.scenario.scenario_id for observation in observations
    )
    choices = tuple(
        observation.actual_choice for observation in observations
    )
    probabilities_by_method = {}
    metrics = []
    for forecast in plan.forecasts:
        by_id = dict(forecast.probabilities)
        probabilities = tuple(by_id[scenario_id] for scenario_id in ordered_ids)
        probabilities_by_method[forecast.method_kind] = probabilities
        metrics.append(
            PilotMethodMetrics.from_evaluation(
                forecast.method_kind,
                evaluate_probability_array(
                    observations,
                    probabilities,
                ),
            )
        )
    primary_metrics = next(
        item
        for item in metrics
        if item.method_kind == PRIMARY_METHOD_KIND
    )
    primary_losses = _binary_log_loss(
        probabilities_by_method[PRIMARY_METHOD_KIND],
        choices,
    )
    comparisons = []
    for baseline_kind in sorted(
        REQUIRED_METHOD_KINDS - {PRIMARY_METHOD_KIND}
    ):
        baseline_losses = _binary_log_loss(
            probabilities_by_method[baseline_kind],
            choices,
        )
        paired_gain = baseline_losses - primary_losses
        mean_gain = float(np.mean(paired_gain))
        standard_error, hac_lag = _newey_west_standard_error(
            paired_gain
        )
        ci_lower = mean_gain - CONFIDENCE_Z_95 * standard_error
        ci_upper = mean_gain + CONFIDENCE_Z_95 * standard_error
        comparisons.append(
            PilotComparison(
                baseline_kind=baseline_kind,
                mean_nll_gain=mean_gain,
                standard_error=standard_error,
                ci_lower_95=ci_lower,
                ci_upper_95=ci_upper,
                minimum_required_gain=(
                    plan.config.minimum_nll_uplift
                ),
                dependence_method="newey-west-bartlett",
                hac_lag=hac_lag,
                passed=(
                    mean_gain >= plan.config.minimum_nll_uplift
                    and ci_lower > 0
                ),
            )
        )
    reasons = []
    if (
        primary_metrics.negative_log_likelihood
        > plan.config.maximum_primary_nll
    ):
        reasons.append("primary_nll_above_limit")
    if (
        primary_metrics.expected_calibration_error
        > plan.config.maximum_calibration_error
    ):
        reasons.append("primary_calibration_error_above_limit")
    for comparison in comparisons:
        if (
            comparison.mean_nll_gain
            < comparison.minimum_required_gain
        ):
            reasons.append(
                "insufficient_nll_uplift:"
                + comparison.baseline_kind
            )
        if comparison.ci_lower_95 <= 0:
            reasons.append(
                "nll_uplift_not_significant:"
                + comparison.baseline_kind
            )
    status = (
        "passed_prospective_pilot"
        if not reasons
        else "completed_no_support"
    )
    return ProspectivePilotReport(
        status=status,
        plan_id=plan.plan_id,
        receipt_id=receipt.receipt_id,
        outcome_event_ids=event_ids,
        outcome_data_hash=outcome_hash,
        primary_metrics=primary_metrics,
        method_metrics=tuple(metrics),
        comparisons=tuple(comparisons),
        reasons=tuple(reasons),
    )


def pilot_report_from_dict(
    data: Mapping[str, object],
) -> ProspectivePilotReport:
    if int(data["outcome_count"]) != len(data["outcome_event_ids"]):
        raise ValueError("pilot report outcome count is inconsistent")
    return ProspectivePilotReport(
        status=str(data["status"]),
        plan_id=str(data["plan_id"]),
        receipt_id=str(data["receipt_id"]),
        outcome_event_ids=tuple(
            str(value) for value in data["outcome_event_ids"]
        ),
        outcome_data_hash=str(data["outcome_data_hash"]),
        primary_metrics=_method_metrics_from_dict(
            dict(data["primary_metrics"])
        ),
        method_metrics=tuple(
            _method_metrics_from_dict(dict(item))
            for item in data["method_metrics"]
        ),
        comparisons=tuple(
            _comparison_from_dict(dict(item))
            for item in data["comparisons"]
        ),
        reasons=tuple(str(value) for value in data["reasons"]),
        artifact_version=str(data["artifact_version"]),
        report_id=str(data["report_id"]),
    )


def verify_pilot_report(
    plan: ProspectivePilotPlan,
    receipt: PilotRegistryReceipt,
    outcomes: EventLedger,
    authority: VerificationAuthority,
    report: ProspectivePilotReport,
) -> None:
    recomputed = score_prospective_pilot(
        plan,
        receipt,
        outcomes,
        authority,
    )
    if recomputed != report:
        raise ProspectivePilotRefusedError(
            ("pilot_report_recomputation_mismatch",)
        )


def _save_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def save_pilot_plan(path: Path, plan: ProspectivePilotPlan) -> None:
    _save_json(path, plan.to_dict())


def load_pilot_plan(path: Path) -> ProspectivePilotPlan:
    return pilot_plan_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def save_pilot_receipt(
    path: Path,
    receipt: PilotRegistryReceipt,
) -> None:
    _save_json(path, receipt.to_dict())


def load_pilot_receipt(path: Path) -> PilotRegistryReceipt:
    return pilot_receipt_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def save_pilot_report(
    path: Path,
    report: ProspectivePilotReport,
) -> None:
    _save_json(path, report.to_dict())


def load_pilot_report(path: Path) -> ProspectivePilotReport:
    return pilot_report_from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
