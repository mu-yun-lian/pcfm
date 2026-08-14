from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.contracts import Observation, Scenario
from pcfm.evaluation import evaluate_probability_array
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.prospective_pilot_v1 import (
    PilotConfig,
    PilotForecast,
    ProspectivePilotRefusedError,
    create_pilot_plan,
    load_pilot_plan,
    load_pilot_receipt,
    load_pilot_report,
    pilot_plan_from_dict,
    pilot_report_from_dict,
    register_pilot_plan,
    save_pilot_plan,
    save_pilot_receipt,
    save_pilot_report,
    score_prospective_pilot,
    verify_pilot_report,
)
from pcfm.workflow import save_event_ledger_jsonl


class ProspectivePilotV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.person_id = "person-real-pilot"
        cls.feature_names = ("cost", "benefit", "risk")
        cls.created_at = "2026-01-01T00:00:00Z"
        cls.registered_at = "2026-01-02T00:00:00Z"
        cls.collection_end = "2026-02-01T00:00:00Z"
        cls.authority = VerificationAuthority(
            {
                "study-author": b"study-secret",
                "external-registry": b"registry-secret",
                "outcome-verifier": b"outcome-secret",
            }
        )
        scenarios = []
        choices = []
        for index in range(100):
            choice = int((index % 5) in {1, 2, 4})
            choices.append(choice)
            scenarios.append(
                Scenario(
                    scenario_id=f"future-{index:03d}",
                    features=(
                        float((index % 7) - 3),
                        float((index % 11) - 5),
                        float((index % 3) - 1),
                    ),
                    feature_names=cls.feature_names,
                    options=("accept", "reject"),
                    domain="bounded-policy-choice",
                    context={
                        "population": "pilot-v1",
                        "language": "zh-CN",
                        "question_text": f"测试人物是否接受方案 {index:03d}？",
                    },
                )
            )
        cls.scenarios = tuple(scenarios)
        cls.choices = tuple(choices)

    def _probabilities(self, correct: float) -> tuple[float, ...]:
        return tuple(
            correct if choice else 1.0 - correct
            for choice in self.choices
        )

    def _forecasts(
        self,
        *,
        primary_correct: float = 0.86,
        population_correct: float = 0.58,
        constant_correct: float = 0.5,
        llm_correct: float = 0.64,
    ) -> tuple[PilotForecast, ...]:
        created_before = "2025-12-31T00:00:00Z"
        return (
            PilotForecast(
                method_kind="pcfm_person_model",
                model_reference=hashlib.sha256(
                    b"pcfm-model-primary"
                ).hexdigest(),
                training_cutoff=created_before,
                probabilities=tuple(
                    zip(
                        (item.scenario_id for item in self.scenarios),
                        self._probabilities(primary_correct),
                        strict=True,
                    )
                ),
            ),
            PilotForecast(
                method_kind="population_model",
                model_reference=hashlib.sha256(
                    b"pcfm-model-population"
                ).hexdigest(),
                training_cutoff=created_before,
                probabilities=tuple(
                    zip(
                        (item.scenario_id for item in self.scenarios),
                        self._probabilities(population_correct),
                        strict=True,
                    )
                ),
            ),
            PilotForecast(
                method_kind="constant_history",
                model_reference=hashlib.sha256(
                    b"history-snapshot-constant"
                ).hexdigest(),
                training_cutoff=created_before,
                probabilities=tuple(
                    zip(
                        (item.scenario_id for item in self.scenarios),
                        self._probabilities(constant_correct),
                        strict=True,
                    )
                ),
            ),
            PilotForecast(
                method_kind="profile_llm",
                model_reference=hashlib.sha256(
                    b"external-llm-model-prompt-profile"
                ).hexdigest(),
                training_cutoff=created_before,
                probabilities=tuple(
                    zip(
                        (item.scenario_id for item in self.scenarios),
                        self._probabilities(llm_correct),
                        strict=True,
                    )
                ),
            ),
        )

    def _plan(
        self,
        *,
        forecasts=None,
        scenarios=None,
        config: PilotConfig | None = None,
    ):
        return create_pilot_plan(
            person_id=self.person_id,
            scenarios=scenarios or self.scenarios,
            forecasts=forecasts or self._forecasts(),
            authority=self.authority,
            verifier_id="study-author",
            created_at=self.created_at,
            collection_end=self.collection_end,
            config=config or PilotConfig(),
        )

    def _receipt(self, plan=None):
        return register_pilot_plan(
            plan or self._plan(),
            self.authority,
            registry_verifier_id="external-registry",
            registered_at=self.registered_at,
        )

    def _ledger(
        self,
        *,
        scenarios=None,
        choices=None,
        person_id: str | None = None,
        verifier_id: str = "outcome-verifier",
        observed_start: str = "2026-01-03T00:00:00Z",
        event_prefix: str = "outcome",
    ) -> EventLedger:
        selected = tuple(scenarios or self.scenarios)
        selected_choices = tuple(choices or self.choices)
        start = datetime.fromisoformat(
            observed_start.replace("Z", "+00:00")
        )
        records = []
        for index, (scenario, choice) in enumerate(
            zip(selected, selected_choices, strict=True)
        ):
            observed_at = (
                start + timedelta(minutes=index)
            ).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            event_id = f"{event_prefix}-{index:03d}"
            observation = Observation(
                person_id=person_id or self.person_id,
                scenario=scenario,
                actual_choice=choice,
                provenance="human_record",
            )
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        f"evidence:{event_id}".encode()
                    ).hexdigest(),
                    verifier_id=verifier_id,
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), self.authority)

    def _score(self, *, plan=None, receipt=None, ledger=None):
        selected_plan = plan or self._plan()
        selected_receipt = receipt or self._receipt(selected_plan)
        return score_prospective_pilot(
            selected_plan,
            selected_receipt,
            ledger or self._ledger(),
            self.authority,
        )

    def test_clear_primary_gain_passes(self) -> None:
        report = self._score()
        self.assertEqual(report.status, "passed_prospective_pilot")
        self.assertEqual(report.sample_count, 100)
        self.assertTrue(all(item.passed for item in report.comparisons))
        self.assertLess(report.primary_metrics.negative_log_likelihood, 0.65)

    def test_null_and_worse_primary_do_not_pass(self) -> None:
        null_report = self._score(
            plan=self._plan(
                forecasts=self._forecasts(
                    primary_correct=0.5,
                    population_correct=0.5,
                    constant_correct=0.5,
                    llm_correct=0.5,
                )
            )
        )
        self.assertEqual(null_report.status, "completed_no_support")
        worse_report = self._score(
            plan=self._plan(
                forecasts=self._forecasts(
                    primary_correct=0.55,
                    population_correct=0.7,
                    constant_correct=0.5,
                    llm_correct=0.65,
                )
            )
        )
        self.assertEqual(worse_report.status, "completed_no_support")

    def test_confident_wrong_primary_fails_absolute_gate(self) -> None:
        forecasts = list(self._forecasts())
        wrong = tuple(
            (scenario.scenario_id, 0.95 if not choice else 0.05)
            for scenario, choice in zip(
                self.scenarios, self.choices, strict=True
            )
        )
        forecasts[0] = replace(forecasts[0], probabilities=wrong)
        report = self._score(plan=self._plan(forecasts=tuple(forecasts)))
        self.assertEqual(report.status, "completed_no_support")
        self.assertIn("primary_nll_above_limit", report.reasons)

    def test_outcomes_must_follow_independent_registration(self) -> None:
        plan = self._plan()
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "registry_role_not_independent",
        ):
            register_pilot_plan(
                plan,
                self.authority,
                registry_verifier_id="study-author",
                registered_at=self.registered_at,
            )
        receipt = self._receipt(plan)
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_precedes_registration",
        ):
            self._score(
                plan=plan,
                receipt=receipt,
                ledger=self._ledger(
                    observed_start="2026-01-01T12:00:00Z"
                ),
            )
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_role_not_independent",
        ):
            self._score(
                plan=plan,
                receipt=receipt,
                ledger=self._ledger(verifier_id="study-author"),
            )
        same_time_records = tuple(
            self.authority.sign(
                event_id=f"same-time-{index:03d}",
                observation=Observation(
                    person_id=self.person_id,
                    scenario=scenario,
                    actual_choice=choice,
                    provenance="human_record",
                ),
                observed_at="2026-01-03T00:00:00Z",
                evidence_hash=hashlib.sha256(
                    f"same-time:{index}".encode()
                ).hexdigest(),
                verifier_id="outcome-verifier",
                verified_at="2026-01-03T00:00:00Z",
            )
            for index, (scenario, choice) in enumerate(
                zip(self.scenarios, self.choices, strict=True)
            )
        )
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_timestamp_sequence_ambiguous",
        ):
            self._score(
                plan=plan,
                receipt=receipt,
                ledger=EventLedger.verify(
                    same_time_records,
                    self.authority,
                ),
            )

    def test_changed_or_missing_scenario_is_refused(self) -> None:
        plan = self._plan()
        changed = list(self.scenarios)
        changed[0] = replace(
            changed[0],
            features=(99.0,) + changed[0].features[1:],
        )
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_scenario_content_mismatch",
        ):
            self._score(plan=plan, ledger=self._ledger(scenarios=changed))
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_count_mismatch",
        ):
            self._score(
                plan=plan,
                ledger=self._ledger(
                    scenarios=self.scenarios[:-1],
                    choices=self.choices[:-1],
                ),
            )

    def test_input_order_and_irrelevant_event_ids_do_not_change_scores(
        self,
    ) -> None:
        plan = self._plan(
            scenarios=tuple(reversed(self.scenarios)),
            forecasts=tuple(reversed(self._forecasts())),
        )
        original = self._score(plan=plan)
        reversed_ledger = self._ledger(
            scenarios=tuple(reversed(self.scenarios)),
            choices=tuple(reversed(self.choices)),
            event_prefix="renamed",
        )
        transformed = self._score(plan=plan, ledger=reversed_ledger)
        self.assertEqual(original.primary_metrics, transformed.primary_metrics)
        self.assertEqual(original.comparisons, transformed.comparisons)
        self.assertNotEqual(original.report_id, transformed.report_id)

    def test_hard_floors_and_required_methods_cannot_be_disabled(self) -> None:
        with self.assertRaises(ValueError):
            PilotConfig(minimum_nll_uplift=0.0)
        with self.assertRaises(ValueError):
            PilotConfig(maximum_primary_nll=0.9)
        with self.assertRaises(ValueError):
            PilotConfig(maximum_calibration_error=0.9)
        with self.assertRaisesRegex(ValueError, "at least 100"):
            self._plan(scenarios=self.scenarios[:99])
        with self.assertRaisesRegex(ValueError, "required forecast methods"):
            self._plan(forecasts=self._forecasts()[:-1])

    def test_near_duplicate_questions_cannot_inflate_sample_count(
        self,
    ) -> None:
        scenarios = list(self.scenarios)
        scenarios[-1] = replace(
            scenarios[0],
            scenario_id="renamed-near-duplicate",
            features=(
                scenarios[0].features[0] + 1e-8,
                *scenarios[0].features[1:],
            ),
        )
        with self.assertRaisesRegex(ValueError, "near-duplicate"):
            self._plan(scenarios=tuple(scenarios))

    def test_question_text_and_model_artifact_hash_are_required(
        self,
    ) -> None:
        missing_text = list(self.scenarios)
        missing_text[0] = replace(
            missing_text[0],
            context={"language": "zh-CN"},
        )
        with self.assertRaisesRegex(ValueError, "question_text"):
            self._plan(scenarios=tuple(missing_text))
        forecasts = list(self._forecasts())
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            forecasts[0] = replace(
                forecasts[0],
                model_reference="unbound-model-label",
            )

    def test_metrics_equal_direct_probability_scoring(self) -> None:
        plan = self._plan()
        report = self._score(plan=plan)
        primary = next(
            item
            for item in plan.forecasts
            if item.method_kind == "pcfm_person_model"
        )
        by_id = dict(primary.probabilities)
        ordered_observations = tuple(
            Observation(
                person_id=self.person_id,
                scenario=scenario,
                actual_choice=choice,
            )
            for scenario, choice in zip(
                self.scenarios, self.choices, strict=True
            )
        )
        expected = evaluate_probability_array(
            ordered_observations,
            tuple(by_id[item.scenario.scenario_id] for item in ordered_observations),
        )
        self.assertAlmostEqual(
            report.primary_metrics.negative_log_likelihood,
            expected.negative_log_likelihood,
        )
        self.assertAlmostEqual(
            report.primary_metrics.brier_score,
            expected.brier_score,
        )
        self.assertTrue(
            all(
                item.dependence_method == "newey-west-bartlett"
                and item.hac_lag == 4
                for item in report.comparisons
            )
        )

    def test_cross_person_domain_options_and_time_are_refused(self) -> None:
        plan = self._plan()
        for expected_reason, changed_scenario in (
            (
                "outcome_scenario_content_mismatch",
                replace(self.scenarios[0], domain="other-domain"),
            ),
            (
                "outcome_scenario_content_mismatch",
                replace(self.scenarios[0], options=("yes", "no")),
            ),
        ):
            changed = (changed_scenario,) + self.scenarios[1:]
            with self.assertRaisesRegex(
                ProspectivePilotRefusedError,
                expected_reason,
            ):
                self._score(
                    plan=plan,
                    ledger=self._ledger(scenarios=changed),
                )
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_person_mismatch",
        ):
            self._score(
                plan=plan,
                ledger=self._ledger(person_id="other-person"),
            )
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "outcome_after_collection_end",
        ):
            self._score(
                plan=plan,
                ledger=self._ledger(
                    observed_start="2026-03-01T00:00:00Z"
                ),
            )

    def test_round_trip_tamper_and_raw_recomputation(self) -> None:
        plan = self._plan()
        receipt = self._receipt(plan)
        ledger = self._ledger()
        report = self._score(plan=plan, receipt=receipt, ledger=ledger)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "plan.json"
            receipt_path = root / "receipt.json"
            report_path = root / "report.json"
            save_pilot_plan(plan_path, plan)
            save_pilot_receipt(receipt_path, receipt)
            save_pilot_report(report_path, report)
            self.assertEqual(load_pilot_plan(plan_path), plan)
            self.assertEqual(load_pilot_receipt(receipt_path), receipt)
            self.assertEqual(load_pilot_report(report_path), report)
        verify_pilot_report(plan, receipt, ledger, self.authority, report)
        tampered_plan = copy.deepcopy(plan.to_dict())
        tampered_plan["forecasts"][0]["probabilities"][0][1] = 0.01
        with self.assertRaises(ValueError):
            pilot_plan_from_dict(tampered_plan)
        tampered_report = copy.deepcopy(report.to_dict())
        tampered_report["primary_metrics"]["negative_log_likelihood"] = 0.0
        with self.assertRaises(ValueError):
            pilot_report_from_dict(tampered_report)
        rehashed_report = copy.deepcopy(report.to_dict())
        rehashed_report["outcome_data_hash"] = hashlib.sha256(
            b"forged-outcome-ledger"
        ).hexdigest()
        rehashed_report["report_id"] = ""
        altered = pilot_report_from_dict(rehashed_report)
        with self.assertRaisesRegex(
            ProspectivePilotRefusedError,
            "pilot_report_recomputation_mismatch",
        ):
            verify_pilot_report(
                plan,
                receipt,
                ledger,
                self.authority,
                altered,
            )

    def test_report_binds_plan_receipt_and_outcome_ledger(self) -> None:
        plan = self._plan()
        receipt = self._receipt(plan)
        report = self._score(plan=plan, receipt=receipt)
        self.assertEqual(report.plan_id, plan.plan_id)
        self.assertEqual(report.receipt_id, receipt.receipt_id)
        self.assertEqual(report.outcome_count, 100)
        self.assertEqual(len(report.outcome_event_ids), 100)
        self.assertEqual(len(report.outcome_data_hash), 64)

    def test_cli_create_register_and_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys_path = root / "keys.json"
            scenarios_path = root / "scenarios.json"
            forecasts_path = root / "forecasts.json"
            plan_path = root / "plan.json"
            receipt_path = root / "receipt.json"
            outcomes_path = root / "outcomes.jsonl"
            report_path = root / "report.json"
            keys_path.write_text(
                json.dumps(
                    {
                        "study-author": "study-secret",
                        "external-registry": "registry-secret",
                        "outcome-verifier": "outcome-secret",
                    }
                ),
                encoding="utf-8",
            )
            scenarios_path.write_text(
                json.dumps(
                    [
                        {
                            "scenario_id": item.scenario_id,
                            "features": dict(
                                zip(
                                    item.feature_names,
                                    item.features,
                                    strict=True,
                                )
                            ),
                            "options": list(item.options),
                            "domain": item.domain,
                            "context": dict(item.context),
                        }
                        for item in self.scenarios
                    ]
                ),
                encoding="utf-8",
            )
            forecasts_path.write_text(
                json.dumps(
                    [
                        {
                            "method_kind": item.method_kind,
                            "model_reference": item.model_reference,
                            "training_cutoff": item.training_cutoff,
                            "probabilities": dict(item.probabilities),
                        }
                        for item in self._forecasts()
                    ]
                ),
                encoding="utf-8",
            )
            save_event_ledger_jsonl(outcomes_path, self._ledger())
            env = os.environ.copy()
            env["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )

            def run(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-m", "pcfm", *arguments],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )

            created = run(
                "pilot-create",
                "--person-id",
                self.person_id,
                "--scenarios",
                str(scenarios_path),
                "--forecasts",
                str(forecasts_path),
                "--keys",
                str(keys_path),
                "--verifier-id",
                "study-author",
                "--created-at",
                self.created_at,
                "--collection-end",
                self.collection_end,
                "--output",
                str(plan_path),
            )
            self.assertEqual(created.returncode, 0, created.stderr)
            registered = run(
                "pilot-register",
                "--plan",
                str(plan_path),
                "--keys",
                str(keys_path),
                "--registry-verifier-id",
                "external-registry",
                "--registered-at",
                self.registered_at,
                "--output",
                str(receipt_path),
            )
            self.assertEqual(registered.returncode, 0, registered.stderr)
            scored = run(
                "pilot-score",
                "--plan",
                str(plan_path),
                "--receipt",
                str(receipt_path),
                "--outcomes",
                str(outcomes_path),
                "--keys",
                str(keys_path),
                "--output",
                str(report_path),
            )
            self.assertEqual(scored.returncode, 0, scored.stderr)
            self.assertEqual(
                json.loads(scored.stdout)["status"],
                "passed_prospective_pilot",
            )
            self.assertEqual(
                load_pilot_report(report_path).status,
                "passed_prospective_pilot",
            )

    def test_full_regression_command_is_documented(self) -> None:
        readme = (
            Path(__file__).resolve().parents[1] / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "python -m unittest discover -s tests -v",
            readme,
        )


if __name__ == "__main__":
    unittest.main()
