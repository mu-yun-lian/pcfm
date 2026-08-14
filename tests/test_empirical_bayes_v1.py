from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from pcfm.empirical_bayes_v1 import (
    EmpiricalBayesConfig,
    EmpiricalBayesInvalidError,
    empirical_bayes_artifact_from_dict,
    fit_anisotropic_prior,
    generate_stable_misspecified_dataset,
    run_empirical_bayes_benchmark,
    run_empirical_bayes_seed_audit,
    verify_anisotropic_prior_artifact,
)
from pcfm.person_choice_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    generate_benchmark_dataset,
)


class EmpiricalBayesV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = EmpiricalBayesConfig()
        cls.dataset = generate_benchmark_dataset(
            BenchmarkConfig.smoke(seed=8101)
        )
        cls.artifact = fit_anisotropic_prior(
            cls.dataset,
            cls.config,
        )
        cls.report = run_empirical_bayes_benchmark(
            cls.dataset,
            cls.config,
            artifact=cls.artifact,
        )

    def test_prior_is_finite_positive_definite_and_not_isotropic(
        self,
    ) -> None:
        covariance = np.asarray(
            self.artifact.prior_covariance,
            dtype=np.float64,
        )
        precision = np.asarray(
            self.artifact.prior_precision,
            dtype=np.float64,
        )
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))
        self.assertLess(
            np.max(
                np.abs(
                    precision @ covariance
                    - np.eye(covariance.shape[0])
                )
            ),
            1e-10,
        )
        diagonal = np.diag(covariance)
        self.assertGreater(
            float(np.max(diagonal) - np.min(diagonal)),
            0.01,
        )

    def test_zero_support_is_exact_population_model(self) -> None:
        model = self.artifact.adapted_model(())
        self.assertEqual(
            model.weights,
            self.artifact.population_weights,
        )

    def test_adapter_recovers_person_specific_signal(self) -> None:
        adapted = self.report.metric(
            "anisotropic_empirical_bayes_map",
            "scenario_test",
            64,
        )
        population = self.report.metric(
            "population_logistic",
            "scenario_test",
            64,
        )
        self.assertLess(
            adapted.negative_log_likelihood,
            population.negative_log_likelihood - 0.01,
        )

    def test_single_seed_decision_matches_fixed_gates(self) -> None:
        checks = dict(self.report.acceptance_checks)
        adapted_scenario = self.report.metric(
            "anisotropic_empirical_bayes_map",
            "scenario_test",
            64,
        )
        isotropic_scenario = self.report.metric(
            "personal_map_logistic",
            "scenario_test",
            64,
        )
        wrong_scenario = self.report.metric(
            "wrong_person_anisotropic_empirical_bayes_map",
            "scenario_test",
            64,
        )
        adapted_temporal = self.report.metric(
            "anisotropic_empirical_bayes_map",
            "temporal_test",
            64,
        )
        isotropic_temporal = self.report.metric(
            "personal_map_logistic",
            "temporal_test",
            64,
        )
        expected = {
            "beats_isotropic_map": (
                isotropic_scenario.negative_log_likelihood
                - adapted_scenario.negative_log_likelihood
                >= self.config.minimum_primary_nll_gain
            ),
            "beats_wrong_person_support": (
                wrong_scenario.negative_log_likelihood
                - adapted_scenario.negative_log_likelihood
                >= self.config.minimum_wrong_person_nll_gain
            ),
            "absolute_primary_adequacy": (
                adapted_scenario.negative_log_likelihood
                <= self.config.maximum_primary_nll
            ),
            "temporal_not_materially_worse": (
                adapted_temporal.negative_log_likelihood
                - isotropic_temporal.negative_log_likelihood
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

    def test_heldout_answers_cannot_change_prior_artifact(
        self,
    ) -> None:
        heldout_roles = {
            "support",
            "scenario_test",
            "temporal_test",
            "ood_test",
        }
        records = tuple(
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
            records=records,
            meta_train_person_ids=(
                self.dataset.meta_train_person_ids
            ),
            validation_person_ids=(
                self.dataset.validation_person_ids
            ),
            test_person_ids=self.dataset.test_person_ids,
        )
        repeated = fit_anisotropic_prior(
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
                record_id=f"renamed-eb-{index:04d}",
                observation=replace(
                    record.observation,
                    scenario=replace(
                        record.observation.scenario,
                        scenario_id=(
                            f"renamed-eb-scenario-{index:04d}"
                        ),
                    ),
                ),
            )
            for index, record in enumerate(reversed(support))
        )
        self.assertEqual(
            self.artifact.adapted_model(support).weights,
            self.artifact.adapted_model(renamed).weights,
        )

    def test_near_duplicate_support_cannot_be_relabelled(
        self,
    ) -> None:
        person_id = self.dataset.test_person_ids[0]
        support = list(
            self.dataset.records_for(
                person_id,
                "support",
            )[:16]
        )
        original = support[0]
        features = list(
            original.observation.scenario.features
        )
        features[0] += 1e-8
        support[-1] = replace(
            original,
            record_id="near-duplicate-record",
            observation=replace(
                original.observation,
                scenario=replace(
                    original.observation.scenario,
                    scenario_id="near-duplicate-scenario",
                    features=tuple(features),
                ),
            ),
        )
        with self.assertRaises(EmpiricalBayesInvalidError):
            self.artifact.adapted_model(support)

    def test_artifact_round_trip_tamper_and_recomputation(
        self,
    ) -> None:
        payload = self.artifact.to_dict()
        restored = empirical_bayes_artifact_from_dict(payload)
        self.assertEqual(
            restored.to_dict(),
            self.artifact.to_dict(),
        )
        self.assertTrue(
            verify_anisotropic_prior_artifact(
                restored,
                self.dataset,
                self.config,
            )
        )
        tampered = json.loads(json.dumps(payload))
        tampered["prior_covariance"][0][0] += 0.01
        with self.assertRaises(EmpiricalBayesInvalidError):
            empirical_bayes_artifact_from_dict(tampered)

    def test_configuration_gates_cannot_be_weakened(self) -> None:
        with self.assertRaises(ValueError):
            replace(
                self.config,
                audit_seeds=(7301, 7302, 7303, 7304, 7305),
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                minimum_primary_nll_gain=0.0,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                eigenvalue_floor=0.0,
            )
        with self.assertRaises(ValueError):
            replace(
                self.config,
                maximum_temporal_nll_excess=0.1,
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
        with self.assertRaises(EmpiricalBayesInvalidError):
            self.artifact.adapted_model(support)

    def test_zero_heterogeneity_does_not_manufacture_gain(
        self,
    ) -> None:
        dataset = generate_benchmark_dataset(
            replace(
                BenchmarkConfig.smoke(
                    seed=self.config.null_control_seed
                ),
                heterogeneity_scale=0.0,
            )
        )
        report = run_empirical_bayes_benchmark(
            dataset,
            self.config,
        )
        adapted = report.metric(
            "anisotropic_empirical_bayes_map",
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
                adapted.negative_log_likelihood
                - population.negative_log_likelihood
            ),
            self.config.maximum_null_nll_difference,
        )

    def test_stable_misspecification_is_not_confirmed(self) -> None:
        dataset = generate_stable_misspecified_dataset(
            self.config.misspecification_seed
        )
        report = run_empirical_bayes_benchmark(
            dataset,
            self.config,
        )
        self.assertEqual(
            report.single_seed_status,
            "single_seed_fail",
        )
        self.assertFalse(
            dict(report.acceptance_checks)[
                "absolute_primary_adequacy"
            ]
        )

    def test_ood_outcomes_do_not_change_acceptance(self) -> None:
        records = tuple(
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
            records=records,
            meta_train_person_ids=(
                self.dataset.meta_train_person_ids
            ),
            validation_person_ids=(
                self.dataset.validation_person_ids
            ),
            test_person_ids=self.dataset.test_person_ids,
        )
        repeated = run_empirical_bayes_benchmark(
            changed,
            self.config,
            artifact=self.artifact,
        )
        self.assertEqual(
            repeated.acceptance_checks,
            self.report.acceptance_checks,
        )
        self.assertEqual(
            repeated.single_seed_status,
            self.report.single_seed_status,
        )

    def test_report_metrics_are_recalculated(self) -> None:
        self.assertLess(
            self.report.maximum_metric_recalculation_error,
            1e-12,
        )

    def test_five_seed_audit_uses_all_final_gates(self) -> None:
        audit = run_empirical_bayes_seed_audit(self.config)
        passed = sum(
            report.single_seed_status == "single_seed_pass"
            for report in audit.seed_reports
        )
        gains = []
        for report in audit.seed_reports:
            adapted = report.metric(
                "anisotropic_empirical_bayes_map",
                "scenario_test",
                64,
            )
            isotropic = report.metric(
                "personal_map_logistic",
                "scenario_test",
                64,
            )
            gains.append(
                isotropic.negative_log_likelihood
                - adapted.negative_log_likelihood
            )
        expected = (
            "accepted_candidate"
            if (
                passed >= self.config.minimum_passing_seeds
                and sum(gains) / len(gains)
                >= self.config.minimum_mean_nll_gain
                and audit.null_control_passed
                and audit.misspecification_rejected
            )
            else "rejected_candidate"
        )
        self.assertEqual(audit.passing_seed_count, passed)
        self.assertAlmostEqual(
            audit.mean_primary_nll_gain,
            sum(gains) / len(gains),
            places=15,
        )
        self.assertEqual(audit.candidate_status, expected)
        self.assertEqual(
            tuple(report.seed for report in audit.seed_reports),
            self.config.audit_seeds,
        )

    def test_cli_runs_fixed_unseen_seed_audit(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1] / "src"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pcfm",
                "empirical-bayes-v1",
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
        self.assertEqual(
            payload["audit_seeds"],
            list(self.config.audit_seeds),
        )
        self.assertEqual(payload["device"], "cpu")

    def test_full_regression_is_documented(self) -> None:
        gate = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "MODULE_GATE_EMPIRICAL_BAYES_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn(
            "python -m unittest discover -s tests -v",
            gate["verification"]["commands"],
        )


if __name__ == "__main__":
    unittest.main()
