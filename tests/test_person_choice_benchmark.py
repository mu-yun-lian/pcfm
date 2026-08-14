from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from pcfm.person_choice_benchmark import (
    BenchmarkConfig,
    BenchmarkDataset,
    BenchmarkInvalidError,
    generate_benchmark_dataset,
    run_person_choice_benchmark,
)


class PersonChoiceBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = BenchmarkConfig.smoke(seed=7301)
        cls.dataset = generate_benchmark_dataset(cls.config)
        cls.report = run_person_choice_benchmark(
            cls.dataset,
            train_neural=True,
        )

    def test_personal_model_beats_population_and_wrong_person(
        self,
    ) -> None:
        personal = self.report.metric(
            "personal_map_logistic",
            "scenario_test",
            64,
        )
        population = self.report.metric(
            "population_logistic",
            "scenario_test",
            64,
        )
        wrong = self.report.metric(
            "wrong_person_logistic",
            "scenario_test",
            64,
        )
        self.assertLess(
            personal.negative_log_likelihood,
            population.negative_log_likelihood - 0.01,
        )
        self.assertLess(
            personal.negative_log_likelihood,
            wrong.negative_log_likelihood - 0.01,
        )

    def test_zero_heterogeneity_does_not_create_personal_gain(
        self,
    ) -> None:
        dataset = generate_benchmark_dataset(
            replace(
                BenchmarkConfig.smoke(seed=7302),
                heterogeneity_scale=0.0,
            )
        )
        report = run_person_choice_benchmark(
            dataset,
            train_neural=False,
        )
        personal = report.metric(
            "personal_map_logistic",
            "scenario_test",
            64,
        )
        population = report.metric(
            "population_logistic",
            "scenario_test",
            64,
        )
        self.assertLess(
            abs(
                personal.negative_log_likelihood
                - population.negative_log_likelihood
            ),
            0.04,
        )
        self.assertEqual(
            report.interpretation,
            "synthetic_benchmark_only",
        )

    def test_personal_gain_is_not_a_single_seed_artifact(
        self,
    ) -> None:
        gains = []
        for seed in (7301, 7302, 7303, 7304, 7305):
            dataset = generate_benchmark_dataset(
                BenchmarkConfig.smoke(seed=seed)
            )
            report = run_person_choice_benchmark(
                dataset,
                train_neural=False,
            )
            population = report.metric(
                "population_logistic",
                "scenario_test",
                64,
            )
            personal = report.metric(
                "personal_map_logistic",
                "scenario_test",
                64,
            )
            gains.append(
                population.negative_log_likelihood
                - personal.negative_log_likelihood
            )
        self.assertTrue(
            all(gain > 0.05 for gain in gains),
            msg=f"per-seed NLL gains: {gains}",
        )
        self.assertGreater(
            sum(gains) / len(gains),
            0.10,
        )

    def test_neural_baseline_runs_without_claiming_semantic_recovery(
        self,
    ) -> None:
        neural = self.report.metric(
            "person_embedding_mlp",
            "scenario_test",
            64,
        )
        self.assertGreater(neural.sample_count, 0)
        self.assertGreaterEqual(
            neural.negative_log_likelihood,
            0.0,
        )
        self.assertIn(
            "neural_representation_has_no_psychological_semantics",
            self.report.non_claims,
        )

    def test_split_replay_and_relabelled_design_are_rejected(
        self,
    ) -> None:
        support = next(
            record
            for record in self.dataset.records
            if record.role == "support"
        )
        scenario = next(
            record
            for record in self.dataset.records
            if (
                record.person_id == support.person_id
                and record.role == "scenario_test"
            )
        )
        replayed = replace(
            scenario,
            observation=replace(
                scenario.observation,
                scenario=replace(
                    support.observation.scenario,
                    scenario_id="relabelled-test-design",
                ),
            ),
        )
        records = tuple(
            replayed if item.record_id == scenario.record_id else item
            for item in self.dataset.records
        )
        with self.assertRaises(BenchmarkInvalidError):
            BenchmarkDataset(
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

    def test_record_order_does_not_change_logistic_results(self) -> None:
        reversed_dataset = BenchmarkDataset(
            config=self.dataset.config,
            records=tuple(reversed(self.dataset.records)),
            meta_train_person_ids=self.dataset.meta_train_person_ids,
            validation_person_ids=self.dataset.validation_person_ids,
            test_person_ids=self.dataset.test_person_ids,
        )
        repeated = run_person_choice_benchmark(
            reversed_dataset,
            train_neural=False,
        )
        for baseline in (
            "population_logistic",
            "personal_map_logistic",
            "wrong_person_logistic",
        ):
            self.assertEqual(
                repeated.metric(
                    baseline,
                    "scenario_test",
                    64,
                ),
                self.report.metric(
                    baseline,
                    "scenario_test",
                    64,
                ),
            )

    def test_resource_and_support_floors_cannot_be_disabled(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            replace(
                BenchmarkConfig.smoke(),
                test_person_count=1,
            )
        with self.assertRaises(ValueError):
            replace(
                BenchmarkConfig.smoke(),
                maximum_records=100,
            )
        with self.assertRaises(ValueError):
            replace(
                BenchmarkConfig.smoke(),
                support_sizes=(0, 1),
            )

    def test_report_probabilities_match_fitted_models(self) -> None:
        self.assertLess(
            self.report.maximum_metric_recalculation_error,
            1e-12,
        )

    def test_temporal_and_ood_roles_are_explicit(self) -> None:
        for person_id in self.dataset.test_person_ids:
            support = self.dataset.records_for(
                person_id,
                "support",
            )
            temporal = self.dataset.records_for(
                person_id,
                "temporal_test",
            )
            ood = self.dataset.records_for(
                person_id,
                "ood_test",
            )
            self.assertTrue(support)
            self.assertTrue(temporal)
            self.assertTrue(ood)
            self.assertLess(
                max(item.observed_at for item in support),
                min(item.observed_at for item in temporal),
            )

    def test_dataset_identity_is_deterministic(self) -> None:
        repeated = generate_benchmark_dataset(self.config)
        self.assertEqual(repeated.dataset_id, self.dataset.dataset_id)
        self.assertEqual(repeated, self.dataset)

    def test_cli_smoke_run(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(
            Path(__file__).resolve().parents[1] / "src"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pcfm",
                "benchmark-v1",
                "--seed",
                "7301",
                "--skip-neural",
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
        self.assertEqual(
            payload["interpretation"],
            "synthetic_benchmark_only",
        )
        self.assertLessEqual(
            payload["resource_usage"]["record_count"],
            self.config.maximum_records,
        )

    def test_report_contains_hypernetwork_comparison_contract(
        self,
    ) -> None:
        self.assertEqual(
            self.report.required_next_baseline,
            "support_set_hypernetwork_low_rank_adapter",
        )
        self.assertIn(
            "person_embedding_mlp",
            self.report.hypernetwork_must_beat,
        )
        self.assertIn(
            "personal_map_logistic",
            self.report.hypernetwork_must_beat,
        )

    def test_full_regression_command_is_documented(self) -> None:
        gate = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "MODULE_GATE_BENCHMARK_V1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertIn(
            "python -m unittest discover -s tests -v",
            gate["verification"]["commands"],
        )


if __name__ == "__main__":
    unittest.main()
