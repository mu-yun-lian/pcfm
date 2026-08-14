from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pcfm.composite as composite_module
import pcfm.mechanism as mechanism_module
from pcfm.composite import (
    CompositeModelRefusedError,
    create_composite_active_experiment_plan,
    create_composite_model,
    predict_with_composite_model,
)
from pcfm.active_experiment import (
    ActiveExperimentConfig,
    load_active_experiment_plan,
)
from pcfm.ledger import EventLedger
from pcfm.mechanism import (
    compare_mechanisms,
    save_mechanism_comparison_plan,
    save_mechanism_comparison_report,
)
from pcfm.storage import save_bundle
from pcfm.synthetic import generate_population_dataset
from pcfm.workflow import save_event_ledger_jsonl
import tests.test_stage6_mechanism as stage6


class Stage65RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = stage6.Stage6MechanismTests
        fixture.setUpClass()
        helper = fixture(
            "test_nonlinear_structure_is_selected_and_confirmed"
        )
        cls.fixture = fixture
        cls.bundle = fixture.bundle
        cls.authority = fixture.authority
        cls.plan = helper._plan()
        cls.evidence = helper._evidence()
        cls.report = compare_mechanisms(
            cls.bundle,
            cls.plan,
            *cls.evidence,
            cls.authority,
        )
        cls.composite = create_composite_model(
            cls.bundle,
            cls.plan,
            cls.report,
            *cls.evidence,
            cls.authority,
            verifier_id="mechanism-verifier",
            created_at="2026-10-10T23:59:59Z",
        )

    def _plan(self, candidates, *, selection_count=1):
        return create_composite_active_experiment_plan(
            self.bundle,
            self.composite,
            self.plan,
            self.report,
            *self.evidence,
            self.authority,
            candidates,
            verifier_id="mechanism-verifier",
            created_at="2026-10-15T00:00:00Z",
            selection_count=selection_count,
        )

    def test_mechanism_evidence_trial_and_relabelled_design_are_refused(
        self,
    ) -> None:
        seen = self.evidence[0].records[0].observation.scenario
        with self.assertRaises(CompositeModelRefusedError) as direct:
            self._plan((seen,))
        self.assertIn(
            "candidate_reuses_mechanism_scenario",
            direct.exception.reasons,
        )
        relabelled = replace(
            seen,
            scenario_id="relabelled-mechanism-evidence",
        )
        with self.assertRaises(CompositeModelRefusedError) as content:
            self._plan((relabelled,))
        self.assertIn(
            "candidate_reuses_mechanism_design",
            content.exception.reasons,
        )

    def test_prediction_and_planning_recompute_mechanism_once(
        self,
    ) -> None:
        original = mechanism_module.compare_mechanisms
        with patch(
            "pcfm.mechanism.compare_mechanisms",
            wraps=original,
        ) as tracked_prediction:
            predict_with_composite_model(
                self.bundle,
                self.composite,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                self.fixture.target[990].scenario,
                prediction_at="2026-10-15T00:00:00Z",
            )
        self.assertEqual(tracked_prediction.call_count, 1)

        candidates = tuple(
            item.scenario for item in self.fixture.target[980:1000]
        )
        with patch(
            "pcfm.mechanism.compare_mechanisms",
            wraps=original,
        ) as tracked_planning:
            self._plan(candidates)
        self.assertEqual(tracked_planning.call_count, 1)

    def test_active_information_floor_cannot_be_disabled(self) -> None:
        with self.assertRaises(ValueError):
            ActiveExperimentConfig(minimum_information_gain=0.0)

    def test_composite_results_update_base_and_invalidate_composite(
        self,
    ) -> None:
        candidates = tuple(
            item.scenario for item in self.fixture.target[980:1000]
        )
        active = self._plan(candidates)
        source_by_id = {
            item.scenario.scenario_id: item
            for item in self.fixture.target
        }
        selected = active.selections[0].scenario
        result_observation = replace(
            source_by_id[selected.scenario_id],
            scenario=selected,
        )
        result = self.authority.sign(
            event_id="composite-active-result-0000",
            observation=result_observation,
            observed_at="2026-10-16T00:00:00Z",
            evidence_hash=hashlib.sha256(
                b"composite-active-result-0000"
            ).hexdigest(),
            verifier_id="mechanism-verifier",
            verified_at="2026-10-16T00:00:00Z",
        )
        result_ledger = EventLedger.verify(
            (result,),
            self.authority,
        )

        _, _, extended_target = generate_population_dataset(
            seed=501,
            person_count=24,
            source_trials=140,
            target_trials=1220,
            heterogeneity_scale=1.5,
        )
        future_records = []
        for index, observation in enumerate(
            extended_target[self.fixture.person_id][1000:1220]
        ):
            observed_at = (
                "2026-11-01T00:00:00Z"
                if index < 110
                else "2026-12-01T00:00:00Z"
            )
            event_id = f"composite-future-{index:04d}"
            future_records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="mechanism-verifier",
                    verified_at=observed_at,
                )
            )
        future_validation = EventLedger.verify(
            tuple(future_records),
            self.authority,
        )

        update = (
            composite_module.apply_composite_active_experiment_results(
                self.bundle,
                self.composite,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                self.fixture.training_ledger,
                self.fixture.applicability_ledger,
                future_validation,
                candidates,
                active,
                result_ledger,
            )
        )
        self.assertEqual(
            update.status,
            "base_updated_composite_invalidated",
        )
        self.assertEqual(
            update.invalidated_composite_model_id,
            self.composite.composite_model_id,
        )
        self.assertIn(
            active.plan_id,
            update.base_update.bundle.manifest.experiment_plan_ids,
        )
        self.assertNotEqual(
            update.base_update.bundle.manifest.model_id,
            self.bundle.manifest.model_id,
        )
        with self.assertRaises(CompositeModelRefusedError):
            predict_with_composite_model(
                update.base_update.bundle,
                self.composite,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                selected,
                prediction_at="2026-12-02T00:00:00Z",
            )

    def test_future_validation_cannot_reuse_mechanism_evidence(
        self,
    ) -> None:
        candidates = tuple(
            item.scenario for item in self.fixture.target[980:1000]
        )
        active = self._plan(candidates)
        selected = active.selections[0].scenario
        source_by_id = {
            item.scenario.scenario_id: item
            for item in self.fixture.target
        }
        result_observation = replace(
            source_by_id[selected.scenario_id],
            scenario=selected,
        )
        result = self.authority.sign(
            event_id="validation-replay-result",
            observation=result_observation,
            observed_at="2026-10-16T00:00:00Z",
            evidence_hash=hashlib.sha256(
                b"validation-replay-result"
            ).hexdigest(),
            verifier_id="mechanism-verifier",
            verified_at="2026-10-16T00:00:00Z",
        )
        replayed = []
        for index, record in enumerate(
            self.evidence[2].records
        ):
            event_id = f"replayed-future-{index:04d}"
            replayed.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=record.observation,
                    observed_at="2026-11-01T00:00:00Z",
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="mechanism-verifier",
                    verified_at="2026-11-01T00:00:00Z",
                )
            )
        with self.assertRaises(CompositeModelRefusedError) as refused:
            composite_module.apply_composite_active_experiment_results(
                self.bundle,
                self.composite,
                self.plan,
                self.report,
                *self.evidence,
                self.authority,
                self.fixture.training_ledger,
                self.fixture.applicability_ledger,
                EventLedger.verify(
                    tuple(replayed),
                    self.authority,
                ),
                candidates,
                active,
                EventLedger.verify((result,), self.authority),
            )
        self.assertIn(
            "future_validation_reuses_mechanism_scenario",
            refused.exception.reasons,
        )

    def test_composite_plan_and_update_are_available_through_cli(
        self,
    ) -> None:
        candidates = tuple(
            item.scenario for item in self.fixture.target[980:1000]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            composite_path = root / "composite.json"
            mechanism_plan_path = root / "mechanism-plan.json"
            mechanism_report_path = root / "mechanism-report.json"
            keys_path = root / "keys.json"
            candidates_path = root / "candidates.json"
            active_path = root / "active.json"
            result_path = root / "results.jsonl"
            training_path = root / "training.jsonl"
            applicability_path = root / "applicability.jsonl"
            future_path = root / "future.jsonl"
            output_model = root / "updated-model.json"
            output_ledger = root / "updated-training.jsonl"
            evidence_paths = tuple(
                root / f"{name}.jsonl"
                for name in ("discovery", "selection", "confirmation")
            )
            save_bundle(model_path, self.bundle)
            composite_module.save_composite_model(
                composite_path,
                self.composite,
            )
            save_mechanism_comparison_plan(
                mechanism_plan_path,
                self.plan,
            )
            save_mechanism_comparison_report(
                mechanism_report_path,
                self.report,
            )
            for path, ledger in zip(
                evidence_paths,
                self.evidence,
                strict=True,
            ):
                save_event_ledger_jsonl(path, ledger)
            save_event_ledger_jsonl(
                training_path,
                self.fixture.training_ledger,
            )
            save_event_ledger_jsonl(
                applicability_path,
                self.fixture.applicability_ledger,
            )
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
            candidates_path.write_text(
                json.dumps(
                    [
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
                        for scenario in candidates
                    ]
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
                "--composite",
                str(composite_path),
                "--mechanism-plan",
                str(mechanism_plan_path),
                "--mechanism-report",
                str(mechanism_report_path),
                "--discovery-ledger",
                str(evidence_paths[0]),
                "--selection-ledger",
                str(evidence_paths[1]),
                "--confirmation-ledger",
                str(evidence_paths[2]),
                "--verification-keys",
                str(keys_path),
                "--candidates",
                str(candidates_path),
            ]
            planned = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "plan-composite-experiment",
                    *common,
                    "--verifier-id",
                    "mechanism-verifier",
                    "--created-at",
                    "2026-10-15T00:00:00Z",
                    "--selection-count",
                    "1",
                    "--output",
                    str(active_path),
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
            active = load_active_experiment_plan(
                active_path,
                self.authority,
            )
            selected = active.selections[0].scenario
            source_by_id = {
                item.scenario.scenario_id: item
                for item in self.fixture.target
            }
            result_observation = replace(
                source_by_id[selected.scenario_id],
                scenario=selected,
            )
            result = self.authority.sign(
                event_id="composite-cli-result-0000",
                observation=result_observation,
                observed_at="2026-10-16T00:00:00Z",
                evidence_hash=hashlib.sha256(
                    b"composite-cli-result-0000"
                ).hexdigest(),
                verifier_id="mechanism-verifier",
                verified_at="2026-10-16T00:00:00Z",
            )
            save_event_ledger_jsonl(
                result_path,
                EventLedger.verify((result,), self.authority),
            )
            _, _, extended_target = generate_population_dataset(
                seed=501,
                person_count=24,
                source_trials=140,
                target_trials=1220,
                heterogeneity_scale=1.5,
            )
            future = self.fixture._ledger(
                extended_target[self.fixture.person_id][1000:1220],
                "composite-cli-future",
                "2026-11-01T00:00:00Z",
            )
            save_event_ledger_jsonl(future_path, future)
            applied = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "apply-composite-experiment",
                    *common,
                    "--active-plan",
                    str(active_path),
                    "--input",
                    str(result_path),
                    "--ledger",
                    str(training_path),
                    "--applicability-ledger",
                    str(applicability_path),
                    "--future-validation-ledger",
                    str(future_path),
                    "--output",
                    str(output_model),
                    "--output-ledger",
                    str(output_ledger),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                applied.returncode,
                0,
                msg=applied.stderr + applied.stdout,
            )
            payload = json.loads(applied.stdout)
            self.assertEqual(
                payload["status"],
                "base_updated_composite_invalidated",
            )
            self.assertTrue(output_model.is_file())
            self.assertTrue(output_ledger.is_file())


if __name__ == "__main__":
    unittest.main()
