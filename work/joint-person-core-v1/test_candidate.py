from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("candidate.py")
SPEC = importlib.util.spec_from_file_location("joint_person_core_candidate", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load joint person core candidate")
candidate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = candidate
SPEC.loader.exec_module(candidate)


def synthetic_people(*, heterogeneous: bool, seed: int = 17):
    rng = np.random.default_rng(seed)
    person_ids = ("a", "b", "c")
    scenario = []
    environment = []
    choices = []
    people = []
    true_person = {
        "a": np.asarray([1.1, -0.8]),
        "b": np.asarray([-1.0, 0.9]),
        "c": np.asarray([0.4, 0.5]),
    }
    for person_id in person_ids:
        for _ in range(450):
            x = np.asarray([1.0, rng.normal()])
            e = np.asarray([rng.choice((-1.0, 1.0))])
            delta = true_person[person_id] if heterogeneous else np.zeros(2)
            logit = x @ (np.asarray([-0.2, 0.7]) + delta) + 0.8 * e[0]
            probability = 1.0 / (1.0 + np.exp(-logit))
            scenario.append(x)
            environment.append(e)
            choices.append(int(rng.random() < probability))
            people.append(person_id)
    return (
        np.asarray(scenario),
        np.asarray(environment),
        np.asarray(choices),
        tuple(people),
    )


class JointPersonCoreCandidateTests(unittest.TestCase):
    def test_correct_person_fit_beats_wrong_person_on_heterogeneity(self) -> None:
        x, e, y, people = synthetic_people(heterogeneous=True)
        model = candidate.fit_joint_core(
            x,
            e,
            y,
            people,
            scenario_feature_names=("intercept", "signal"),
            environment_feature_names=("regime",),
            stable_person_precision=4.0,
            coordinate_passes=6,
        )
        mask = np.asarray([person == "a" for person in people])
        correct = model.probabilities("a", x[mask], e[mask])
        wrong = model.probabilities("b", x[mask], e[mask])
        self.assertLess(
            candidate.negative_log_likelihood(y[mask], correct),
            candidate.negative_log_likelihood(y[mask], wrong),
        )

    def test_zero_heterogeneity_is_shrunk(self) -> None:
        x, e, y, people = synthetic_people(heterogeneous=False, seed=29)
        model = candidate.fit_joint_core(
            x,
            e,
            y,
            people,
            scenario_feature_names=("intercept", "signal"),
            environment_feature_names=("regime",),
            stable_person_precision=16.0,
            coordinate_passes=6,
        )
        norms = [np.linalg.norm(model.person_weights[key]) for key in people[::450]]
        self.assertLess(max(norms), 0.35)

    def test_person_id_renaming_preserves_corresponding_predictions(self) -> None:
        x, e, y, people = synthetic_people(heterogeneous=True, seed=31)
        first = candidate.fit_joint_core(
            x,
            e,
            y,
            people,
            scenario_feature_names=("intercept", "signal"),
            environment_feature_names=("regime",),
            stable_person_precision=4.0,
            coordinate_passes=5,
        )
        mapping = {"a": "z9", "b": "x2", "c": "q7"}
        renamed = tuple(mapping[value] for value in people)
        second = candidate.fit_joint_core(
            x,
            e,
            y,
            renamed,
            scenario_feature_names=("intercept", "signal"),
            environment_feature_names=("regime",),
            stable_person_precision=4.0,
            coordinate_passes=5,
        )
        probe_x = x[:80]
        probe_e = e[:80]
        self.assertTrue(
            np.allclose(
                first.probabilities("a", probe_x, probe_e),
                second.probabilities("z9", probe_x, probe_e),
                atol=1e-10,
            )
        )

    def test_prequential_probability_cannot_read_current_outcome(self) -> None:
        timestamps = tuple(
            (
                datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=index)
            ).isoformat()
            for index in range(40)
        )
        logits = np.linspace(-0.5, 0.5, 40)
        variances = np.full(40, 0.04)
        choices = np.asarray(([0, 1] * 20), dtype=np.int64)
        flipped = choices.copy()
        flipped[-1] = 1 - flipped[-1]
        config = candidate.StateConfig(half_life_days=30.0, stationary_variance=0.5)
        original = candidate.run_prequential_state(
            logits, variances, choices, timestamps, config
        )
        attacked = candidate.run_prequential_state(
            logits, variances, flipped, timestamps, config
        )
        self.assertTrue(np.allclose(original.probabilities, attacked.probabilities))
        self.assertNotEqual(original.final_state_mean, attacked.final_state_mean)

    def test_model_round_trip_preserves_identity_and_probability(self) -> None:
        x, e, y, people = synthetic_people(heterogeneous=True, seed=43)
        model = candidate.fit_joint_core(
            x,
            e,
            y,
            people,
            scenario_feature_names=("intercept", "signal"),
            environment_feature_names=("regime",),
            stable_person_precision=4.0,
            coordinate_passes=4,
        )
        restored = candidate.JointCoreModel.from_dict(model.to_dict())
        self.assertEqual(model.model_id, restored.model_id)
        self.assertTrue(
            np.allclose(
                model.probabilities("a", x[:100], e[:100]),
                restored.probabilities("a", x[:100], e[:100]),
                atol=1e-12,
            )
        )


if __name__ == "__main__":
    unittest.main()
