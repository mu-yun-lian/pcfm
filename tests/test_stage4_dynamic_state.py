from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pcfm.dynamic_state import (
    DynamicStateConfig,
    DynamicStateRefusedError,
    create_dynamic_state_plan,
    dynamic_state_plan_from_dict,
    dynamic_state_report_from_dict,
    infer_dynamic_state,
    load_dynamic_state_report,
    predict_with_dynamic_state,
    save_dynamic_state_report,
)
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.math_utils import sigmoid
from pcfm.storage import bundle_to_dict, save_bundle
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import fit_person_model, save_event_ledger_jsonl


class Stage4DynamicStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        people, source, target = generate_population_dataset(
            seed=401,
            person_count=24,
            source_trials=140,
            target_trials=900,
            heterogeneity_scale=1.5,
        )
        cls.person_id = people[0].person_id
        cls.target = target[cls.person_id]
        cls.authority = VerificationAuthority(
            {"state-verifier": b"stage-four-state-secret"}
        )
        training = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        cls.training_ledger = cls._ledger(
            training,
            "training",
            "2026-07-01T00:00:00Z",
        )
        cls.applicability_ledger = cls._ledger(
            cls.target[:220],
            "applicability",
            "2026-07-15T00:00:00Z",
        )
        cls.validation_ledger = cls._validation_ledger(
            cls.target[220:440]
        )
        cls.bundle = fit_person_model(
            cls.training_ledger,
            cls.authority,
            applicability_ledger=cls.applicability_ledger,
            validation_ledger=cls.validation_ledger,
            person_id=cls.person_id,
            feature_names=FEATURE_NAMES,
        )
        if cls.bundle.manifest.validation.status != "passed":
            raise AssertionError("stage-four fixture requires a validated base")
        cls.config = DynamicStateConfig(
            half_life_days=14.0,
            stationary_variance=0.5,
            initial_variance=0.25,
            minimum_samples=80,
            minimum_effect=0.25,
            minimum_consecutive_detections=4,
            minimum_nll_uplift=0.005,
            maximum_prediction_gap_days=14.0,
        )

    @classmethod
    def _ledger(
        cls,
        observations,
        prefix: str,
        observed_at: str,
    ) -> EventLedger:
        records = []
        for index, observation in enumerate(observations):
            event_id = f"{prefix}-{index:05d}"
            records.append(
                cls.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="state-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), cls.authority)

    @classmethod
    def _validation_ledger(cls, observations) -> EventLedger:
        records = []
        midpoint = len(observations) // 2
        for index, observation in enumerate(observations):
            observed_at = (
                "2026-08-01T00:00:00Z"
                if index < midpoint
                else "2026-09-01T00:00:00Z"
            )
            event_id = f"validation-{index:05d}"
            records.append(
                cls.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="state-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), cls.authority)

    def _monitoring_ledger(
        self,
        shifts: tuple[float, ...],
        *,
        seed: int,
        start: datetime | None = None,
        prefix: str = "monitor",
    ) -> EventLedger:
        start = start or datetime(
            2026,
            9,
            2,
            tzinfo=timezone.utc,
        )
        stable_weights = np.asarray(
            self.bundle.population_model.weights
        ) + np.asarray(self.bundle.adapter.delta_weights)
        rng = np.random.default_rng(seed)
        records = []
        candidates = iter(self.target[440:])
        selected = []
        while len(selected) < len(shifts):
            try:
                candidate = next(candidates)
            except StopIteration as error:
                raise AssertionError(
                    "not enough applicable monitoring scenarios"
                ) from error
            assessment = (
                self.bundle.manifest.applicability_profile.assess(
                    candidate.scenario,
                    prediction_at="2026-09-02T00:00:00Z",
                )
            )
            if assessment.status == "in_distribution":
                selected.append(candidate)
        for index, (source, shift) in enumerate(
            zip(selected, shifts, strict=True)
        ):
            base_logit = float(
                np.asarray(source.scenario.features) @ stable_weights
            )
            choice = int(rng.random() < sigmoid(base_logit + shift))
            scenario = replace(
                source.scenario,
                scenario_id=f"{prefix}-scenario-{index:05d}",
            )
            observation = replace(
                source,
                scenario=scenario,
                actual_choice=choice,
            )
            observed = start + timedelta(hours=3 * index)
            observed_at = observed.isoformat().replace("+00:00", "Z")
            event_id = f"{prefix}-event-{index:05d}"
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="state-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), self.authority)

    def _plan_for(
        self,
        ledger: EventLedger,
        *,
        config: DynamicStateConfig | None = None,
    ):
        first = datetime.fromisoformat(
            ledger.records[0].observed_at.replace("Z", "+00:00")
        )
        registered_at = (
            first - timedelta(days=1)
        ).isoformat().replace("+00:00", "Z")
        return create_dynamic_state_plan(
            self.bundle,
            self.authority,
            verifier_id="state-verifier",
            registered_at=registered_at,
            monitoring_start_at=ledger.records[0].observed_at,
            monitoring_end_at=ledger.records[-1].observed_at,
            expected_event_count=len(ledger.records),
            config=config or self.config,
        )

    def _infer(
        self,
        ledger: EventLedger,
        *,
        config: DynamicStateConfig | None = None,
    ):
        plan = self._plan_for(ledger, config=config)
        report = infer_dynamic_state(
            self.bundle,
            ledger,
            self.authority,
            plan,
        )
        return report, plan

    def test_static_person_does_not_gain_a_validated_state(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 180,
            seed=1001,
            prefix="static",
        )
        report, _ = self._infer(ledger)
        self.assertEqual(
            report.status,
            "no_prequential_residual_signal",
        )
        self.assertLessEqual(
            report.maximum_detection_run,
            self.config.minimum_consecutive_detections,
        )
        self.assertEqual(
            report.interpretation_status,
            "no_causal_interpretation",
        )

    def test_temporary_shift_is_detected_then_recovers(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.2,) * 100 + (0.0,) * 50,
            seed=1002,
            prefix="temporary",
        )
        before = copy.deepcopy(bundle_to_dict(self.bundle))
        report, _ = self._infer(ledger)
        self.assertEqual(
            report.status,
            "prequential_residual_signal",
        )
        self.assertGreater(report.nll_uplift, 0.005)
        self.assertGreater(report.nll_uplift_ci_lower, 0.0)
        self.assertGreaterEqual(
            report.maximum_log_e_value,
            -math.log(self.config.sequential_alpha),
        )
        self.assertGreaterEqual(
            report.maximum_detection_run,
            self.config.minimum_consecutive_detections,
        )
        self.assertEqual(
            report.interpretation_status,
            "unidentified_latent_shift",
        )
        self.assertEqual(
            report.points[-1].evidence_status,
            "no_detectable_shift",
        )
        self.assertEqual(bundle_to_dict(self.bundle), before)

    def test_insufficient_monitoring_is_not_assessed(self) -> None:
        ledger = self._monitoring_ledger(
            (1.8,) * 30,
            seed=1003,
            prefix="short",
        )
        report, _ = self._infer(ledger)
        self.assertEqual(report.status, "not_assessed")
        self.assertIn(
            "insufficient_dynamic_state_samples",
            report.reasons,
        )

    def test_monitoring_must_follow_base_evidence(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1004,
            start=datetime(2026, 8, 15, tzinfo=timezone.utc),
            prefix="past",
        )
        plan = self._plan_for(ledger)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                ledger,
                self.authority,
                plan,
            )
        self.assertIn(
            "state_evidence_precedes_base_reference",
            raised.exception.reasons,
        )

    def test_monitoring_event_cannot_reuse_model_lineage(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1005,
            prefix="overlap",
        )
        first = replace(
            ledger.records[0],
            event_id=self.bundle.manifest.person_event_ids[0],
        )
        resigned = self.authority.sign(
            event_id=first.event_id,
            observation=first.observation,
            observed_at=first.observed_at,
            evidence_hash=first.evidence_hash,
            verifier_id=first.verifier_id,
            verified_at=first.verified_at,
        )
        overlapping = EventLedger(
            records=(resigned,) + ledger.records[1:]
        )
        plan = self._plan_for(overlapping)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                overlapping,
                self.authority,
                plan,
            )
        self.assertIn(
            "state_evidence_reuses_model_event",
            raised.exception.reasons,
        )

    def test_monitoring_timestamps_must_be_strictly_ordered(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1010,
            prefix="same-time",
        )
        first = ledger.records[0]
        second = ledger.records[1]
        resigned_second = self.authority.sign(
            event_id=second.event_id,
            observation=second.observation,
            observed_at=first.observed_at,
            evidence_hash=second.evidence_hash,
            verifier_id=second.verifier_id,
            verified_at=first.observed_at,
        )
        same_time = EventLedger.verify(
            (first, resigned_second) + ledger.records[2:],
            self.authority,
        )
        plan = self._plan_for(same_time)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                same_time,
                self.authority,
                plan,
            )
        self.assertIn(
            "state_evidence_timestamps_not_strictly_increasing",
            raised.exception.reasons,
        )

    def test_unvalidated_base_model_is_refused(self) -> None:
        unvalidated = fit_person_model(
            self.training_ledger,
            self.authority,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1006,
            prefix="unvalidated",
        )
        plan = self._plan_for(ledger)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                unvalidated,
                ledger,
                self.authority,
                plan,
            )
        self.assertIn(
            "base_model_validation_unvalidated",
            raised.exception.reasons,
        )

    def test_out_of_distribution_monitoring_is_refused(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1007,
            prefix="remote",
        )
        changed_records = []
        for record in ledger.records:
            scenario = replace(
                record.observation.scenario,
                features=tuple(
                    value + 30.0
                    for value in record.observation.scenario.features
                ),
            )
            observation = replace(
                record.observation,
                scenario=scenario,
            )
            changed_records.append(
                self.authority.sign(
                    event_id=record.event_id,
                    observation=observation,
                    observed_at=record.observed_at,
                    evidence_hash=record.evidence_hash,
                    verifier_id=record.verifier_id,
                    verified_at=record.verified_at,
                )
            )
        remote = EventLedger.verify(
            tuple(changed_records),
            self.authority,
        )
        plan = self._plan_for(remote)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                remote,
                self.authority,
                plan,
            )
        self.assertIn(
            "state_evidence_outside_applicability",
            raised.exception.reasons,
        )

    def test_dynamic_prediction_uses_only_validated_state(self) -> None:
        sustained = self._monitoring_ledger(
            (0.0,) * 30 + (2.0,) * 150,
            seed=1008,
            prefix="sustained",
        )
        report, plan = self._infer(sustained)
        self.assertEqual(
            report.status,
            "prequential_residual_signal",
        )
        scenario = replace(
            self.target[700].scenario,
            scenario_id="dynamic-future",
        )
        prediction_time = (
            datetime.fromisoformat(
                report.last_observed_at.replace("Z", "+00:00")
            )
            + timedelta(hours=3)
        ).isoformat().replace("+00:00", "Z")
        prediction = predict_with_dynamic_state(
            self.bundle,
            report,
            scenario,
            self.authority,
            plan,
            sustained,
            prediction_at=prediction_time,
        )
        self.assertEqual(
            prediction.dynamic_state_status,
            "prequential_residual_signal",
        )
        self.assertIn("dynamic_state", prediction.active_modules)
        self.assertEqual(
            prediction.dynamic_state_artifact_id,
            report.artifact_id,
        )
        self.assertGreater(prediction.dynamic_state_mean, 0.25)
        self.assertEqual(
            prediction.dynamic_state_current_evidence_status,
            "latent_shift_detected",
        )

        static = self._monitoring_ledger(
            (0.0,) * 180,
            seed=1009,
            prefix="prediction-static",
        )
        no_signal, static_plan = self._infer(static)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            predict_with_dynamic_state(
                self.bundle,
                no_signal,
                scenario,
                self.authority,
                static_plan,
                static,
                prediction_at=prediction_time,
            )
        self.assertIn(
            "dynamic_state_no_prequential_residual_signal",
            raised.exception.reasons,
        )
        overridden = predict_with_dynamic_state(
            self.bundle,
            no_signal,
            scenario,
            self.authority,
            static_plan,
            static,
            prediction_at=prediction_time,
            state_override=True,
        )
        self.assertEqual(overridden.dynamic_state_status, "overridden")
        self.assertIn(
            "dynamic_state_no_prequential_residual_signal",
            overridden.gate_overrides,
        )
        self.assertIsNone(overridden.probability_lower_95)
        self.assertEqual(
            overridden.model_form_uncertainty_status,
            "unquantified_override",
        )

    def test_state_artifact_round_trip_and_tamper_detection(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.0,) * 150,
            seed=1011,
            prefix="artifact",
        )
        report, _ = self._infer(ledger)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state-report.json"
            save_dynamic_state_report(path, report)
            restored = load_dynamic_state_report(
                path,
                self.authority,
            )
        self.assertEqual(restored, report)
        tampered = copy.deepcopy(report.to_dict())
        tampered["points"][-1]["posterior_state_mean"] += 1.0
        with self.assertRaises(ValueError):
            dynamic_state_report_from_dict(
                tampered,
                self.authority,
            )

    def test_dynamic_prediction_rejects_old_or_reversed_state(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.0,) * 150,
            seed=1012,
            prefix="prediction-time",
        )
        report, plan = self._infer(ledger)
        scenario = replace(
            self.target[750].scenario,
            scenario_id="dynamic-time-boundary",
        )
        with self.assertRaises(DynamicStateRefusedError) as reversed_time:
            predict_with_dynamic_state(
                self.bundle,
                report,
                scenario,
                self.authority,
                plan,
                ledger,
                prediction_at=report.last_observed_at,
            )
        self.assertIn(
            "prediction_not_after_state_evidence",
            reversed_time.exception.reasons,
        )
        last = datetime.fromisoformat(
            report.last_observed_at.replace("Z", "+00:00")
        )
        stale_time = (
            last
            + timedelta(
                days=self.config.maximum_prediction_gap_days + 1
            )
        ).isoformat().replace("+00:00", "Z")
        with self.assertRaises(DynamicStateRefusedError) as stale:
            predict_with_dynamic_state(
                self.bundle,
                report,
                scenario,
                self.authority,
                plan,
                ledger,
                prediction_at=stale_time,
            )
        self.assertIn("stale_dynamic_state", stale.exception.reasons)

    def test_signed_custom_configuration_is_preregistered(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.5,) * 150,
            seed=1013,
            prefix="custom-config",
        )
        custom = replace(
            self.config,
            half_life_days=self.config.half_life_days + 1.0,
        )
        report, plan = self._infer(
            ledger,
            config=custom,
        )
        self.assertEqual(
            report.status,
            "prequential_residual_signal",
        )
        self.assertEqual(
            report.config_status,
            "signed_preregistered",
        )
        scenario = replace(
            self.target[760].scenario,
            scenario_id="custom-config-future",
        )
        prediction_time = (
            datetime.fromisoformat(
                report.last_observed_at.replace("Z", "+00:00")
            )
            + timedelta(hours=3)
        ).isoformat().replace("+00:00", "Z")
        prediction = predict_with_dynamic_state(
            self.bundle,
            report,
            scenario,
            self.authority,
            plan,
            ledger,
            prediction_at=prediction_time,
        )
        self.assertEqual(
            prediction.dynamic_state_status,
            "prequential_residual_signal",
        )

    def test_state_cli_round_trip(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.5,) * 150,
            seed=1014,
            prefix="cli-state",
        )
        last_time = datetime.fromisoformat(
            ledger.records[-1].observed_at.replace("Z", "+00:00")
        )
        prediction_time = (
            last_time + timedelta(hours=3)
        ).isoformat().replace("+00:00", "Z")
        scenario = replace(
            ledger.records[-1].observation.scenario,
            scenario_id="cli-state-future",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "person-model.json"
            ledger_path = root / "monitoring.jsonl"
            keys_path = root / "keys.json"
            plan_path = root / "state-plan.json"
            state_path = root / "state.json"
            scenario_path = root / "scenario.json"
            save_bundle(model_path, self.bundle)
            save_event_ledger_jsonl(ledger_path, ledger)
            keys_path.write_text(
                json.dumps(
                    {
                        "state-verifier": (
                            "stage-four-state-secret"
                        )
                    }
                ),
                encoding="utf-8",
            )
            scenario_path.write_text(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "features": {
                            name: value
                            for name, value in zip(
                                scenario.feature_names,
                                scenario.features,
                                strict=True,
                            )
                        },
                        "options": list(scenario.options),
                        "domain": scenario.domain,
                        "context": dict(scenario.context),
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            registered_at = (
                datetime.fromisoformat(
                    ledger.records[0].observed_at.replace(
                        "Z",
                        "+00:00",
                    )
                )
                - timedelta(days=1)
            ).isoformat().replace("+00:00", "Z")
            planned = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "plan-state",
                    "--model",
                    str(model_path),
                    "--verification-keys",
                    str(keys_path),
                    "--verifier-id",
                    "state-verifier",
                    "--registered-at",
                    registered_at,
                    "--monitoring-start-at",
                    ledger.records[0].observed_at,
                    "--monitoring-end-at",
                    ledger.records[-1].observed_at,
                    "--expected-event-count",
                    str(len(ledger.records)),
                    "--output",
                    str(plan_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                planned.returncode,
                0,
                msg=planned.stderr + planned.stdout,
            )
            inferred = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "infer-state",
                    "--model",
                    str(model_path),
                    "--input",
                    str(ledger_path),
                    "--verification-keys",
                    str(keys_path),
                    "--plan",
                    str(plan_path),
                    "--output",
                    str(state_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                inferred.returncode,
                0,
                msg=inferred.stderr + inferred.stdout,
            )
            inferred_payload = json.loads(inferred.stdout)
            self.assertEqual(
                inferred_payload["status"],
                "prequential_residual_signal",
            )
            predicted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "predict-state",
                    "--model",
                    str(model_path),
                    "--state",
                    str(state_path),
                    "--plan",
                    str(plan_path),
                    "--state-ledger",
                    str(ledger_path),
                    "--verification-keys",
                    str(keys_path),
                    "--scenario",
                    str(scenario_path),
                    "--prediction-at",
                    prediction_time,
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                predicted.returncode,
                0,
                msg=predicted.stderr + predicted.stdout,
            )
            predicted_payload = json.loads(predicted.stdout)
            self.assertEqual(
                predicted_payload["dynamic_state_status"],
                "prequential_residual_signal",
            )

    def test_recomputed_hash_cannot_forge_signed_report(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 180,
            seed=1015,
            prefix="signed-tamper",
        )
        report, _ = self._infer(ledger)
        tampered = copy.deepcopy(report.to_dict())
        tampered["points"][-1]["prior_state_mean"] += 5.0
        content = {
            key: value
            for key, value in tampered.items()
            if key not in {"artifact_id", "signature"}
        }
        tampered["artifact_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "signature"):
            dynamic_state_report_from_dict(
                tampered,
                self.authority,
            )

    def test_prediction_recomputes_signed_report_from_ledger(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.5,) * 150,
            seed=1016,
            prefix="recompute",
        )
        report, plan = self._infer(ledger)
        tampered = copy.deepcopy(report.to_dict())
        tampered["points"][-1]["prior_state_mean"] += 0.5
        content = {
            key: value
            for key, value in tampered.items()
            if key not in {"artifact_id", "signature"}
        }
        tampered["artifact_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        signed_payload = {
            key: value
            for key, value in tampered.items()
            if key != "signature"
        }
        tampered["signature"] = self.authority.sign_payload(
            signed_payload,
            "state-verifier",
        )
        forged = dynamic_state_report_from_dict(
            tampered,
            self.authority,
        )
        scenario = replace(
            ledger.records[-1].observation.scenario,
            scenario_id="recompute-future",
        )
        prediction_time = (
            datetime.fromisoformat(
                report.last_observed_at.replace("Z", "+00:00")
            )
            + timedelta(hours=3)
        ).isoformat().replace("+00:00", "Z")
        with self.assertRaises(DynamicStateRefusedError) as raised:
            predict_with_dynamic_state(
                self.bundle,
                forged,
                scenario,
                self.authority,
                plan,
                ledger,
                prediction_at=prediction_time,
            )
        self.assertIn(
            "dynamic_state_derivation_mismatch",
            raised.exception.reasons,
        )

    def test_replayed_model_scenario_is_refused_with_new_id(self) -> None:
        start = datetime(2026, 9, 2, tzinfo=timezone.utc)
        records = []
        for old in self.applicability_ledger.records:
            observed_at = (
                start + timedelta(hours=3 * len(records))
            ).isoformat().replace("+00:00", "Z")
            assessment = (
                self.bundle.manifest.applicability_profile.assess(
                    old.observation.scenario,
                    prediction_at=observed_at,
                )
            )
            if assessment.status != "in_distribution":
                continue
            event_id = f"replayed-new-id-{len(records):04d}"
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=old.observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="state-verifier",
                    verified_at=observed_at,
                )
            )
            if len(records) == 80:
                break
        replayed = EventLedger.verify(
            tuple(records),
            self.authority,
        )
        plan = self._plan_for(replayed)
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                replayed,
                self.authority,
                plan,
            )
        self.assertIn(
            "state_evidence_reuses_model_scenario",
            raised.exception.reasons,
        )

    def test_dynamic_state_does_not_transfer_cross_domain(self) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.5,) * 150,
            seed=1017,
            prefix="cross-domain",
        )
        report, plan = self._infer(ledger)
        original = ledger.records[-1].observation.scenario
        changed = replace(
            original,
            scenario_id="cross-domain-future",
            domain="unvalidated-domain",
            options=("X", "Y"),
            context={"condition": "unvalidated"},
        )
        prediction_time = (
            datetime.fromisoformat(
                report.last_observed_at.replace("Z", "+00:00")
            )
            + timedelta(hours=3)
        ).isoformat().replace("+00:00", "Z")
        with self.assertRaises(DynamicStateRefusedError) as raised:
            predict_with_dynamic_state(
                self.bundle,
                report,
                changed,
                self.authority,
                plan,
                ledger,
                prediction_at=prediction_time,
            )
        self.assertIn(
            "dynamic_state_transfer_unvalidated",
            raised.exception.reasons,
        )
        overridden = predict_with_dynamic_state(
            self.bundle,
            report,
            changed,
            self.authority,
            plan,
            ledger,
            prediction_at=prediction_time,
            state_override=True,
        )
        self.assertIn(
            "dynamic_state_transfer_unvalidated",
            overridden.gate_overrides,
        )
        self.assertIsNone(overridden.probability_lower_95)

    def test_first_prior_uses_gap_from_base_reference(self) -> None:
        early = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1018,
            prefix="gap-early",
        )
        late = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1018,
            start=datetime(2026, 12, 10, tzinfo=timezone.utc),
            prefix="gap-late",
        )
        early_report, _ = self._infer(early)
        late_report, _ = self._infer(late)
        self.assertGreater(
            late_report.points[0].prior_state_variance,
            early_report.points[0].prior_state_variance,
        )

    def test_modified_preregistration_plan_signature_is_rejected(
        self,
    ) -> None:
        ledger = self._monitoring_ledger(
            (0.0,) * 80,
            seed=1019,
            prefix="plan-tamper",
        )
        plan = self._plan_for(ledger)
        tampered = copy.deepcopy(plan.to_dict())
        tampered["expected_event_count"] += 1
        content = {
            key: value
            for key, value in tampered.items()
            if key not in {"plan_id", "signature"}
        }
        tampered["plan_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "signature"):
            dynamic_state_plan_from_dict(
                tampered,
                self.authority,
            )

    def test_preregistered_window_cannot_be_cherry_picked(self) -> None:
        full_ledger = self._monitoring_ledger(
            (0.0,) * 30 + (2.5,) * 150,
            seed=1020,
            prefix="window",
        )
        plan = self._plan_for(full_ledger)
        selected = EventLedger.verify(
            full_ledger.records[30:130],
            self.authority,
        )
        with self.assertRaises(DynamicStateRefusedError) as raised:
            infer_dynamic_state(
                self.bundle,
                selected,
                self.authority,
                plan,
            )
        self.assertTrue(
            {
                "state_evidence_count_differs_from_plan",
                "state_evidence_window_differs_from_plan",
            }.intersection(raised.exception.reasons)
        )


if __name__ == "__main__":
    unittest.main()
