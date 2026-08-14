from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("person_issue_experiment", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load person-issue experiment")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class PersonIssueExperimentContractTests(unittest.TestCase):
    def test_bound_raw_and_cohort_manifests(self) -> None:
        manifest, cohort, manifest_digest = experiment.load_bound_inputs()
        self.assertEqual(len(manifest["cohort_person_ids"]), 6)
        self.assertEqual(manifest["feature_map"]["hash_dimension"], 128)
        self.assertEqual(manifest["feature_map"]["svd_rank"], 8)
        self.assertEqual(len(cohort["feature_names"]), 20)
        self.assertEqual(len(manifest_digest), 64)

    def test_hard_gates_and_roles_are_frozen(self) -> None:
        manifest, _cohort, _digest = experiment.load_bound_inputs()
        gates = manifest["hard_gates"]
        self.assertEqual(gates["required_final_events_per_person"], 300)
        self.assertEqual(gates["minimum_positive_people_over_party_dynamic"], 4)
        self.assertEqual(gates["minimum_equal_person_mean_nll_uplift_over_party_dynamic"], 0.01)
        self.assertEqual(gates["minimum_mean_nll_uplift_over_wrong_person"], 0.005)
        self.assertEqual(gates["minimum_mean_nll_uplift_over_shuffled_history"], 0.005)
        self.assertEqual(gates["maximum_applicability_age_days"], 365)
        self.assertEqual(manifest["applicability"]["reference_events_per_person"], 80)
        self.assertEqual(len(manifest["applicability"]["continuous_features"]), 8)
        self.assertFalse(gates["overrides_allowed"])
        self.assertIn("dynamic_warmup", manifest["roles"])
        self.assertIn("final_comparison", manifest["roles"])

    def test_final_windows_and_age_are_valid(self) -> None:
        audit = experiment.audit_role_windows()
        self.assertEqual(len(audit), 6)
        for row in audit:
            self.assertEqual(row["warmup_count"], 450)
            self.assertEqual(row["applicability_count"], 80)
            self.assertEqual(row["final_count"], 300)
            self.assertGreater(row["applicability_first_roll"], row["warmup_last_roll"])
            self.assertGreater(row["final_first_roll"], row["applicability_last_roll"])
            self.assertLessEqual(
                (date(2026, 8, 1) - date.fromisoformat(row["final_end_date"])).days,
                180,
            )

    def test_work_only_and_production_unchanged(self) -> None:
        manifest, _cohort, _digest = experiment.load_bound_inputs()
        self.assertFalse(manifest["production_changes_authorized"])
        self.assertFalse(manifest["semantic_module_resume_authorized"])
        self.assertIn("work", str(Path(experiment.__file__).resolve()).lower())

    def test_within_congress_shuffle_is_deterministic_and_count_preserving(self) -> None:
        records = []
        for person_id in ("a", "b"):
            for congress in (116, 117):
                for rollnumber, choice in enumerate((0, 0, 1, 1, 1), start=1):
                    records.append(
                        {
                            "person_id": person_id,
                            "congress": congress,
                            "rollnumber": rollnumber,
                            "choice": choice,
                        }
                    )
        first = experiment._shuffled_choices(records, 91023)
        second = experiment._shuffled_choices(records, 91023)
        self.assertTrue(np.array_equal(first, second))
        original = np.asarray([row["choice"] for row in records])
        for person_id in ("a", "b"):
            for congress in (116, 117):
                mask = np.asarray(
                    [
                        row["person_id"] == person_id and row["congress"] == congress
                        for row in records
                    ]
                )
                self.assertEqual(int(np.sum(original[mask])), int(np.sum(first[mask])))
        self.assertFalse(np.array_equal(original, first))


if __name__ == "__main__":
    unittest.main()
