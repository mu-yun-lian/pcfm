from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("run_experiment.py")
SPEC = importlib.util.spec_from_file_location("joint_core_experiment", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load joint-core experiment")
experiment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment
SPEC.loader.exec_module(experiment)


class JointCoreExperimentContractTests(unittest.TestCase):
    def test_bound_manifest_and_cohort_load(self) -> None:
        manifest, cohort, manifest_digest = experiment.load_bound_inputs()
        self.assertEqual(len(manifest["cohort_person_ids"]), 6)
        self.assertEqual(len(cohort["feature_names"]), 20)
        self.assertEqual(len(manifest_digest), 64)

    def test_environment_mapping_uses_only_prechoice_values(self) -> None:
        manifest, _cohort, _digest = experiment.load_bound_inputs()
        rollcall = {
            "date": "2025-05-01",
            "session": 1,
            "rollnumber": 300,
            "vote_result": "Passed",
            "yea_count": 99,
        }
        first = experiment.environment_features(119, rollcall, 200, manifest)
        attacked = dict(rollcall)
        attacked["vote_result"] = "Rejected"
        attacked["yea_count"] = 0
        second = experiment.environment_features(119, attacked, 200, manifest)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 6)

    def test_final_role_is_inside_frozen_age_limit(self) -> None:
        manifest, _cohort, _digest = experiment.load_bound_inputs()
        self.assertEqual(
            manifest["applicability"]["maximum_age_days"],
            180.0,
        )
        self.assertLessEqual(
            (date(2025, 8, 1) - date(2025, 3, 26)).days,
            180,
        )


if __name__ == "__main__":
    unittest.main()
