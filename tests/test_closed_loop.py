from __future__ import annotations

import unittest

from pcfm.demo import run_demo


class ClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_demo(
            seed=42,
            person_count=24,
            source_trials=140,
            target_trials=220,
        )

    def test_correct_person_beats_population_on_log_loss(self) -> None:
        self.assertGreater(
            self.report["uplift"]["nll_vs_population"],
            0.015,
        )

    def test_correct_person_beats_wrong_person_on_log_loss(self) -> None:
        self.assertGreater(
            self.report["uplift"]["nll_vs_wrong_person"],
            0.02,
        )

    def test_parameter_recovery_is_bounded(self) -> None:
        self.assertLess(
            self.report["parameter_recovery"]["mean_absolute_error"],
            0.55,
        )

    def test_demo_is_reproducible(self) -> None:
        repeated = run_demo(
            seed=42,
            person_count=24,
            source_trials=140,
            target_trials=220,
        )
        self.assertEqual(self.report, repeated)

    def test_person_uplift_is_not_single_seed_artifact(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                report = run_demo(
                    seed=seed,
                    person_count=18,
                    source_trials=90,
                    target_trials=120,
                )
                self.assertGreater(
                    report["uplift"]["nll_vs_population"],
                    0.03,
                )
                self.assertGreater(
                    report["uplift"]["nll_vs_wrong_person"],
                    0.06,
                )


if __name__ == "__main__":
    unittest.main()
