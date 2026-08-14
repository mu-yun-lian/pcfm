from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from pcfm.hypernetwork_v1 import (
    HyperNetworkConfig,
    HyperNetworkInvalidError,
    fit_support_set_hypernetwork,
    hypernetwork_artifact_from_dict,
    run_hypernetwork_benchmark,
    run_hypernetwork_seed_audit,
    verify_support_set_hypernetwork_artifact,
)
from pcfm.person_choice_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    generate_benchmark_dataset,
)


class HyperNetworkV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = HyperNetworkConfig()
        cls.dataset = generate_benchmark_dataset(
            BenchmarkConfig.smoke(seed=7301)
        )
        cls.artifact = fit_support_set_hypernetwork(
            cls.dataset,
            cls.config,
        )
        cls.report = run_hypernetwork_benchmark(
            cls.dataset,
            cls.config,
            artifact=cls.artifact,
            train_embedding=True,
        )

    def test_zero_support_is_exactly_the_population_head(self) -> None:
        generated = self.artifact.generated_model(())
        self.assertEqual(
            generated.weights,
            self.artifact.population_weights,
        )

    def test_support_generates_a_bounded_nonzero_head_delta(
        self,
    ) -> None:
        person_id = self.dataset.test_person_ids[0]
        support = self.dataset.records_for(
            person_id,
            "support",
        )
        generated = self.artifact.generated_model(support)
        delta = sum(
            (
                generated_value - population_value
            )
            ** 2
            for generated_value, population_value in zip(
                generated.weights,
                self.artifact.population_weights,
            )
        ) ** 0.5
        self.assertGreater(delta, 1e-6)
        self.assertLessEqual(
            delta,
            self.config.maximum_head_delta_norm,
        )

    def test_hypernetwork_recovers_some_person_specific_signal(
        self,
    ) -> None:
        generated = self.report.metric(
            "support_set_hypernetwork",
            "scenario_test",
            64,
        )
        population = self.report.metric(
            "population_logistic",
            "scenario_test",
            64,
        )
        self.assertLess(
            generated.negative_log_likelihood,
            population.negative_log_likelihood - 0.01,
        )

    def test_single_seed_decision_uses_preregistered_gates(
        self,
    ) -> None:
        checks = dict(self.report.acceptance_checks)
        generated_scenario = self.report.metric(
            "support_set_hypernetwork",
            "scenario_test",
            64,
        )
        personal_scenario = self.report.metric(
            "personal_map_logistic",
            "scenario_test",
            64,
        )
        embedding_scenario = self.report.metric(
            "person_embedding_mlp",
            "scenario_test",
            64,
        )
        generated_temporal = self.report.metric(
            "support_set_hypernetwork",
            "temporal_test",
            64,
        )
        personal_temporal = self.report.metric(
            "personal_map_logistic",
            "temporal_test",
            64,
        )
        expected = {
            "beats_personal_map": (
                personal_scenario.negative_log_likelihood
                - generated_scenario.negative_log_likelihood
                >= self.config.minimum_primary_nll_gain
            ),
            "beats_person_embedding": (
                embedding_scenario.negative_log_likelihood
                - generated_scenario.negative_log_likelihood
                >= self.config.minimum_primary_nll_gain
            ),
            "absolute_primary_adequacy": (
                generated_scenario.negative_log_likelihood
                <= self.config.maximum_primary_nll
            ),
            "temporal_not_materially_worse": (
                generated_temporal.negative_log_likelihood
                - personal_temporal.negative_log_likelihood
                <= self.config.maximum_temporal_nll_excess
            ),
        }
        self.assertEqual(checks, expected)
        self.assertEqual(
            self.report.single_seed_status,
            (
                "single_seed_pass"
                if all(expected.values())
                else "single_seed_fail"
            ),
        )

    def test_heldout_answers_cannot_change_fitted_artifact(
        self,
    ) -> None:
        heldout_roles = {
            "scenario_test",
            "temporal_test",
            "ood_test",
        }
        changed_records = tuple(
            replace(
                record,
                observation=replace(
                    record.observation,
                    actual_choice=(
                        1 - record.observation.actual_choice
                    ),
                ),
            )
            if record.role in heldout_roles
            else record
            for record in self.dataset.records
        )
        changed = BenchmarkDataset(
            config=self.dataset.config,
            records=changed_records,
            meta_train_person_ids=(
                self.dataset.meta_train_person_ids
            ),
            validation_person_ids=(
                self.dataset.validation_person_ids
            ),
            test_person_ids=self.dataset.test_person_ids,
        )
        repeated = fit_support_set_hypernetwork(
            changed,
            self.config,
        )
        self.assertEqual(
            repeated.to_dict(),
            self.artifact.to_dict(),
        )

    def test_support_order_and_identifier_renaming_do_not_matter(
        self,
    ) -> None:
        person_id = self.dataset.test_person_ids[0]
        support = self.dataset.records_for(
            person_id,
            "support",
        )[:32]
        renamed = tuple(
            replace(
                record,
                record_id=f"renamed-{index:04d}",
                observation=replace(
                    record.observation,
                    scenario=replace(
                        record.observation.scenario,
                        scenario_id=f"renamed-scenario-{index:04d}",
                    ),
                ),
            )
            for index, record in enumerate(reversed(support))
        )
        original_model = self.artifact.generated_model(support)
        renamed_model = self.artifact.generated_model(renamed)
        self.assertEqual(
            original_model.weights,
            renamed_model.weights,
        )

    def test_artifact_round_trip_tamper_and_recomputation(
        self,
    ) -> None:
        payload = self.artifact.to_dict()
        restored = hypernetwork_artifact_from_dict(payload)
        self.assertEqual(
            restored.to_dict(),
            self.artifact.to_dict(),
        )
        self.assertTrue(
            verify_support_set_hypernetwork_artifact(
                restored,
                self.dataset,
                self.config,
            )
        )
        tampered = json.loads(json.dumps(payload))
        tampered["left_matrix"][0][0] += 0.01
        with self.assertRaises(HyperNetworkInvalidError):
            hypernetwork_artifact_from_dict(tampered)

    def test_configuration_gates_cannot_be_weakened(self) -> None:
        with self.assertRaises(ValueError):
            replace(self.config, rank=2)
        with self.assertRaises(ValueError):
            replace(
                self.config,
                minimum_primary_nll_gain=0.0,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                maximum_temporal_nll_excess=0.10,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                maximum_trainable_scalars=1000,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                learning_rate=1.0,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                null_control_seed=9999,
            )

    def test_cross_domain_support_is_refused(self) -> None:
        person_id = self.dataset.test_person_ids[0]
        support = list(
            self.dataset.records_for(
                person_id,
                "support",
            )[:16]
        )
        support[0] = replace(
            support[0],
            observation=replace(
                support[0].observation,
                scenario=replace(
                    support[0].observation.scenario,
                    domain="different_domain",
                ),
            ),
        )
        with self.assertRaises(HyperNetworkInvalidError):
            self.artifact.generated_model(support)

    def test_zero_heterogeneity_does_not_manufacture_gain(
        self,
    ) -> None:
        dataset = generate_benchmark_dataset(
            replace(
                BenchmarkConfig.smoke(seed=7310),
                heterogeneity_scale=0.0,
            )
        )
        report = run_hypernetwork_benchmark(
            dataset,
            self.config,
            train_embedding=False,
        )
        generated = report.metric(
            "support_set_hypernetwork",
            "scenario_test",
            64,
        )
        population = report.metric(
            "population_logistic",
            "scenario_test",
            64,
        )
        self.assertLessEqual(
            abs(
                generated.negative_log_likelihood
                - population.negative_log_likelihood
            ),
            self.config.maximum_null_nll_difference,
        )

    def test_report_metrics_are_independently_recalculated(
        self,
    ) -> None:
        self.assertLess(
            self.report.maximum_metric_recalculation_error,
            1e-12,
        )

    def test_ood_outcomes_do_not_enter_acceptance_decision(
        self,
    ) -> None:
        changed_records = tuple(
            replace(
                record,
                observation=replace(
                    record.observation,
                    actual_choice=(
                        1 - record.observation.actual_choice
                    ),
                ),
            )
            if record.role == "ood_test"
            else record
            for record in self.dataset.records
        )
        changed = BenchmarkDataset(
            config=self.dataset.config,
            records=changed_records,
            meta_train_person_ids=(
                self.dataset.meta_train_person_ids
            ),
            validation_person_ids=(
                self.dataset.validation_person_ids
            ),
            test_person_ids=self.dataset.test_person_ids,
        )
        repeated = run_hypernetwork_benchmark(
            changed,
            self.config,
            artifact=self.artifact,
            train_embedding=True,
        )
        self.assertEqual(
            repeated.acceptance_checks,
            self.report.acceptance_checks,
        )
        self.assertEqual(
            repeated.single_seed_status,
            self.report.single_seed_status,
        )

    def test_five_seed_audit_computes_final_candidate_status(
        self,
    ) -> None:
        audit = run_hypernetwork_seed_audit(self.config)
        passed = sum(
            report.single_seed_status == "single_seed_pass"
            for report in audit.seed_reports
        )
        expected = (
            "accepted_candidate"
            if (
                passed >= self.config.minimum_passing_seeds
                and audit.null_control_passed
            )
            else "rejected_candidate"
        )
        self.assertEqual(audit.passing_seed_count, passed)
        self.assertEqual(audit.candidate_status, expected)
        self.assertEqual(
            tuple(report.seed for report in audit.seed_reports),
            self.config.audit_seeds,
        )

    def test_cli_runs_the_fixed_seed_audit(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1] / "src"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pcfm",
                "hypernetwork-v1",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr + completed.stdout,
        )
        payload = json.loads(completed.stdout)
        self.assertIn(
            payload["candidate_status"],
            {"accepted_candidate", "rejected_candidate"},
        )
        self.assertLessEqual(
            payload["resource_usage"][
                "hypernetwork_trainable_scalars"
            ],
            self.config.maximum_trainable_scalars,
        )
        self.assertEqual(payload["device"], "cpu")

    def test_full_regression_is_in_the_gate_manifest(self) -> None:
        gate = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "MODULE_GATE_HYPERNETWORK_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn(
            "python -m unittest discover -s tests -v",
            gate["verification"]["commands"],
        )


if __name__ == "__main__":
    unittest.main()
