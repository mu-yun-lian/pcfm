from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.active_experiment import (
    ActiveExperimentRefusedError,
    apply_active_experiment_results,
)
from pcfm.composite import (
    CompositeModelRefusedError,
    composite_model_from_dict,
    create_composite_active_experiment_plan,
    create_composite_model,
    load_composite_model,
    predict_with_composite_model,
    save_composite_model,
    verify_composite_active_experiment_plan,
    verify_composite_model,
)
from pcfm.ledger import EventLedger
from pcfm.mechanism import (
    compare_mechanisms,
    predict_with_mechanism,
    save_mechanism_comparison_plan,
    save_mechanism_comparison_report,
)
from pcfm.storage import save_bundle
from pcfm.registry import ModuleSlot
from pcfm.workflow import save_event_ledger_jsonl
import tests.test_stage6_mechanism as stage6


class Stage65CompositeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = stage6.Stage6MechanismTests
        fixture.setUpClass()
        cls.fixture = fixture
        cls.helper = fixture(
            "test_nonlinear_structure_is_selected_and_confirmed"
        )
        cls.bundle = fixture.bundle
        cls.authority = fixture.authority
        cls.plan = cls.helper._plan()
        cls.evidence = cls.helper._evidence()
        cls.report = compare_mechanisms(
            cls.bundle,
            cls.plan,
            *cls.evidence,
            cls.authority,
        )
        cls.created_at = "2026-10-10T23:59:59Z"
        cls.composite = create_composite_model(
            cls.bundle,
            cls.plan,
            cls.report,
            *cls.evidence,
            cls.authority,
            verifier_id="mechanism-verifier",
            created_at=cls.created_at,
        )

    def _predict(self, scenario=None, *, prediction_at=None):
        return predict_with_composite_model(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            scenario or self.fixture.target[990].scenario,
            prediction_at=prediction_at or "2026-10-15T00:00:00Z",
        )

    def test_composite_prediction_matches_confirmed_mechanism(
        self,
    ) -> None:
        scenario = self.fixture.target[990].scenario
        expected = predict_with_mechanism(
            self.bundle,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            scenario,
            prediction_at="2026-10-15T00:00:00Z",
        )
        prediction = self._predict(scenario)
        self.assertEqual(
            prediction.probability_option_1,
            expected.probability_option_1,
        )
        self.assertEqual(
            prediction.logit_standard_deviation,
            expected.logit_standard_deviation,
        )
        self.assertEqual(
            prediction.predictive_model_id,
            self.composite.composite_model_id,
        )
        self.assertEqual(
            prediction.active_components,
            ("stable_person_model", "confirmed_mechanism"),
        )

    def test_unsupported_mechanism_cannot_form_composite(self) -> None:
        null_evidence = self.helper._evidence(nonlinear=False)
        null_report = compare_mechanisms(
            self.bundle,
            self.plan,
            *null_evidence,
            self.authority,
        )
        with self.assertRaises(CompositeModelRefusedError) as refused:
            create_composite_model(
                self.bundle,
                self.plan,
                null_report,
                *null_evidence,
                self.authority,
                verifier_id="mechanism-verifier",
                created_at=self.created_at,
            )
        self.assertIn(
            "mechanism_candidate_not_supported",
            refused.exception.reasons,
        )

    def test_mechanism_only_failed_base_is_accepted_through_composite(
        self,
    ) -> None:
        validation = replace(
            self.bundle.manifest.validation,
            status="failed",
            mechanism_adequacy_passed=False,
            reasons=("mechanism_misspecification_suspected",),
        )
        repairable = self.helper._bundle_with_validation(validation)
        plan = self.helper._plan_for_bundle(repairable)
        report = compare_mechanisms(
            repairable,
            plan,
            *self.evidence,
            self.authority,
        )
        composite = create_composite_model(
            repairable,
            plan,
            report,
            *self.evidence,
            self.authority,
            verifier_id="mechanism-verifier",
            created_at=self.created_at,
        )
        prediction = predict_with_composite_model(
            repairable,
            composite,
            plan,
            report,
            *self.evidence,
            self.authority,
            self.fixture.target[990].scenario,
            prediction_at="2026-10-15T00:00:00Z",
        )
        self.assertEqual(
            prediction.validation_status,
            "mechanism_only_failure_repaired",
        )

    def test_composite_recomputes_component_from_raw_ledgers(
        self,
    ) -> None:
        discovery, selection, confirmation = self.evidence
        attacked_confirmation = EventLedger.verify(
            tuple(
                self.fixture._signed_record(
                    replace(
                        record.observation,
                        actual_choice=(
                            1 - record.observation.actual_choice
                            if index == 0
                            else record.observation.actual_choice
                        ),
                    ),
                    (
                        "composite-attacked-confirmation"
                        if index == 0
                        else record.event_id
                    ),
                    record.observed_at,
                )
                if index == 0
                else record
                for index, record in enumerate(confirmation.records)
            ),
            self.authority,
        )
        with self.assertRaises(CompositeModelRefusedError) as refused:
            predict_with_composite_model(
                self.bundle,
                self.composite,
                self.plan,
                self.report,
                discovery,
                selection,
                attacked_confirmation,
                self.authority,
                self.fixture.target[990].scenario,
                prediction_at="2026-10-15T00:00:00Z",
            )
        self.assertIn(
            "composite_component_derivation_mismatch",
            refused.exception.reasons,
        )

    def test_composite_identity_is_input_order_invariant(self) -> None:
        reversed_ledgers = tuple(
            EventLedger(tuple(reversed(ledger.records)))
            for ledger in self.evidence
        )
        repeated = create_composite_model(
            self.bundle,
            self.plan,
            self.report,
            *reversed_ledgers,
            self.authority,
            verifier_id="mechanism-verifier",
            created_at=self.created_at,
        )
        self.assertEqual(repeated, self.composite)

    def test_unbound_dynamic_state_is_refused(self) -> None:
        with self.assertRaises(CompositeModelRefusedError) as refused:
            create_composite_model(
                self.bundle,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                verifier_id="mechanism-verifier",
                created_at=self.created_at,
                dynamic_state_artifact_id="0" * 64,
            )
        self.assertIn(
            "dynamic_state_not_reinferred_for_composite",
            refused.exception.reasons,
        )

    def test_composite_preserves_mechanism_scope_and_expiry(
        self,
    ) -> None:
        transferred = replace(
            self.fixture.target[990].scenario,
            scenario_id="composite-transfer",
            domain="unvalidated-composite-domain",
        )
        with self.assertRaises(CompositeModelRefusedError) as transfer:
            self._predict(transferred)
        self.assertIn(
            "mechanism_transfer_unvalidated",
            transfer.exception.reasons,
        )
        with self.assertRaises(CompositeModelRefusedError) as expired:
            self._predict(prediction_at="2027-05-01T00:00:00Z")
        self.assertTrue(
            {
                "composite_model_expired",
                "mechanism_report_expired",
            }
            & set(expired.exception.reasons)
        )

    def test_composite_round_trip_signature_and_tamper_detection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "composite.json"
            save_composite_model(path, self.composite)
            restored = load_composite_model(path, self.authority)
        self.assertEqual(restored, self.composite)
        verify_composite_model(
            self.bundle,
            restored,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
        )

        tampered = copy.deepcopy(self.composite.to_dict())
        tampered["selected_hypothesis_id"] = "linear-residual"
        with self.assertRaises(ValueError):
            composite_model_from_dict(tampered, self.authority)

        resigned = copy.deepcopy(self.composite.to_dict())
        resigned["selected_hypothesis_id"] = "linear-residual"
        content = {
            key: value
            for key, value in resigned.items()
            if key not in {"composite_model_id", "signature"}
        }
        resigned["composite_model_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = {
            key: value
            for key, value in resigned.items()
            if key != "signature"
        }
        resigned["signature"] = self.authority.sign_payload(
            payload,
            "mechanism-verifier",
        )
        altered = composite_model_from_dict(
            resigned,
            self.authority,
        )
        with self.assertRaises(CompositeModelRefusedError) as refused:
            verify_composite_model(
                self.bundle,
                altered,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
            )
        self.assertIn(
            "composite_artifact_derivation_mismatch",
            refused.exception.reasons,
        )

    def test_active_planner_uses_composite_probability_and_identity(
        self,
    ) -> None:
        candidates = tuple(
            observation.scenario
            for observation in self.fixture.target[980:1000]
        )
        active = create_composite_active_experiment_plan(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            candidates,
            verifier_id="mechanism-verifier",
            created_at="2026-10-15T00:00:00Z",
            selection_count=1,
        )
        self.assertEqual(
            active.predictive_model_id,
            self.composite.composite_model_id,
        )
        selected = active.selections[0]
        prediction = self._predict(
            selected.scenario,
            prediction_at="2026-10-15T00:00:00Z",
        )
        self.assertAlmostEqual(
            selected.expected_choice_probability,
            prediction.probability_option_1,
            places=12,
        )

    def test_composite_active_plan_recomputes_and_rejects_forgery(
        self,
    ) -> None:
        candidates = tuple(
            observation.scenario
            for observation in self.fixture.target[980:1000]
        )
        active = create_composite_active_experiment_plan(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            candidates,
            verifier_id="mechanism-verifier",
            created_at="2026-10-15T00:00:00Z",
            selection_count=1,
        )
        verify_composite_active_experiment_plan(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            candidates,
            active,
        )
        changed_selection = replace(
            active.selections[0],
            expected_choice_probability=(
                active.selections[0].expected_choice_probability
                + 0.001
            ),
        )
        unsigned = replace(
            active,
            selections=(changed_selection,),
            plan_id="",
            signature="",
        )
        forged = replace(
            unsigned,
            signature=self.authority.sign_payload(
                unsigned.signed_payload(),
                unsigned.verifier_id,
            ),
        )
        with self.assertRaises(CompositeModelRefusedError) as refused:
            verify_composite_active_experiment_plan(
                self.bundle,
                self.composite,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                candidates,
                forged,
            )
        self.assertIn(
            "composite_active_experiment_derivation_mismatch",
            refused.exception.reasons,
        )

    def test_base_updater_refuses_composite_active_plan(self) -> None:
        candidates = tuple(
            observation.scenario
            for observation in self.fixture.target[980:1000]
        )
        active = create_composite_active_experiment_plan(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            candidates,
            verifier_id="mechanism-verifier",
            created_at="2026-10-15T00:00:00Z",
            selection_count=1,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as refused:
            apply_active_experiment_results(
                self.bundle,
                self.fixture.training_ledger,
                self.fixture.applicability_ledger,
                self.evidence[0],
                candidates,
                active,
                self.evidence[1],
                self.authority,
            )
        self.assertIn(
            "composite_active_plan_requires_composite_update",
            refused.exception.reasons,
        )

    def test_composite_module_slot_is_explicit(self) -> None:
        self.assertEqual(
            ModuleSlot.COMPOSITE_MODEL.value,
            "composite_model",
        )

    def test_composite_cli_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            mechanism_plan_path = root / "mechanism-plan.json"
            mechanism_report_path = root / "mechanism-report.json"
            composite_path = root / "composite.json"
            keys_path = root / "keys.json"
            scenario_path = root / "scenario.json"
            ledger_paths = tuple(
                root / f"{label}.jsonl"
                for label in ("discovery", "selection", "confirmation")
            )
            save_bundle(model_path, self.bundle)
            save_mechanism_comparison_plan(
                mechanism_plan_path,
                self.plan,
            )
            save_mechanism_comparison_report(
                mechanism_report_path,
                self.report,
            )
            for path, ledger in zip(
                ledger_paths,
                self.evidence,
                strict=True,
            ):
                save_event_ledger_jsonl(path, ledger)
            keys_path.write_text(
                json.dumps(
                    {
                        "mechanism-verifier": (
                            "stage-six-mechanism-secret"
                        )
                    }
                ),
                encoding="utf-8",
            )
            scenario = self.fixture.target[990].scenario
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
            common = [
                "--model",
                str(model_path),
                "--mechanism-plan",
                str(mechanism_plan_path),
                "--mechanism-report",
                str(mechanism_report_path),
                "--discovery-ledger",
                str(ledger_paths[0]),
                "--selection-ledger",
                str(ledger_paths[1]),
                "--confirmation-ledger",
                str(ledger_paths[2]),
                "--verification-keys",
                str(keys_path),
            ]
            created = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "create-composite",
                    *common,
                    "--verifier-id",
                    "mechanism-verifier",
                    "--created-at",
                    self.created_at,
                    "--output",
                    str(composite_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                created.returncode,
                0,
                msg=created.stderr + created.stdout,
            )
            predicted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "predict-composite",
                    *common,
                    "--composite",
                    str(composite_path),
                    "--scenario",
                    str(scenario_path),
                    "--prediction-at",
                    "2026-10-15T00:00:00Z",
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
            payload = json.loads(predicted.stdout)
            self.assertEqual(
                payload["predictive_model_id"],
                self.composite.composite_model_id,
            )

    def test_stage65_full_regression_command_is_documented(self) -> None:
        gate = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "MODULE_GATE_STAGE6_5.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn(
            "python -m unittest discover -s tests -v",
            gate["verification"]["commands"],
        )


if __name__ == "__main__":
    unittest.main()
