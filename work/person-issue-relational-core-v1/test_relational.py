from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("relational.py")
SPEC = importlib.util.spec_from_file_location("person_issue_relational", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load person-issue relational candidate")
relational = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = relational
SPEC.loader.exec_module(relational)


def rollcall(topic: str, *, congress: int = 118, result: str = "ignored") -> dict:
    return {
        "congress": congress,
        "rollnumber": 1,
        "date": "2023-02-01",
        "session": 1,
        "bill_number": "S42" if topic == "alpha" else "SJRES7",
        "vote_desc": f"A measure concerning {topic} and public administration",
        "vote_question": "On the Amendment" if topic == "alpha" else "On Cloture",
        "dtl_desc": None,
        "crs_policy_area": "Health" if topic == "alpha" else "Education",
        "crs_subjects": [f"{topic} policy", "public administration"],
        "cast_code": 1,
        "prob": 99.0,
        "yea_count": 99,
        "nay_count": 1,
        "vote_result": result,
        "nominate_mid_1": 0.9,
        "nominate_mid_2": -0.9,
        "nominate_spread_1": 1.2,
        "nominate_spread_2": -1.2,
        "nominate_log_likelihood": -0.01,
    }


def synthetic_relation(*, person_signal: bool, regime_only: bool = False, seed: int = 73):
    rng = np.random.default_rng(seed)
    people = ("d1", "d2", "r1", "r2")
    party = {"d1": 100, "d2": 100, "r1": 200, "r2": 200}
    rows = []
    choices = []
    person_ids = []
    party_codes = []
    congresses = []
    for person_id in people:
        for index in range(360):
            topic = "alpha" if rng.random() < 0.5 else "beta"
            congress = 116 if index < 180 else 117
            row = rollcall(topic, congress=congress)
            topic_sign = 1.0 if topic == "alpha" else -1.0
            party_sign = 1.0 if party[person_id] == 200 else -1.0
            person_delta = {
                "d1": 1.0,
                "d2": -1.0,
                "r1": 0.9,
                "r2": -0.9,
            }[person_id]
            logit = -0.15 + 0.55 * party_sign * topic_sign
            if person_signal:
                logit += person_delta * topic_sign
            if regime_only:
                logit += 1.1 if congress == 117 else -1.1
            probability = 1.0 / (1.0 + np.exp(-logit))
            rows.append(row)
            choices.append(int(rng.random() < probability))
            person_ids.append(person_id)
            party_codes.append(party[person_id])
            congresses.append(congress)
    return rows, np.asarray(choices), tuple(person_ids), tuple(party_codes), tuple(congresses)


class PersonIssueRelationalTests(unittest.TestCase):
    def test_recovers_party_relative_profile(self) -> None:
        rows, y, people, parties, congresses = synthetic_relation(person_signal=True)
        feature_map = relational.RelationalFeatureMap.fit(rows, hash_dimension=32, svd_rank=2)
        train = np.asarray([(index % 360) < 260 for index in range(len(rows))])
        test = ~train
        artifact = relational.fit_relational_artifact(
            feature_map,
            [rows[index] for index in np.flatnonzero(train)],
            y[train],
            [people[index] for index in np.flatnonzero(train)],
            [parties[index] for index in np.flatnonzero(train)],
            [congresses[index] for index in np.flatnonzero(train)],
            stable_person_precision=4.0,
        )
        correct = []
        wrong = []
        actual = []
        wrong_map = {"d1": "d2", "d2": "d1", "r1": "r2", "r2": "r1"}
        for person_id in ("d1", "d2", "r1", "r2"):
            indices = np.asarray(
                [index for index in np.flatnonzero(test) if people[index] == person_id]
            )
            target_rows = [rows[index] for index in indices]
            correct.extend(
                artifact.probabilities(
                    person_id,
                    target_rows,
                    parties[indices[0]],
                    congresses[indices[0]],
                )
            )
            wrong.extend(
                artifact.probabilities(
                    person_id,
                    target_rows,
                    parties[indices[0]],
                    congresses[indices[0]],
                    profile_person_id=wrong_map[person_id],
                )
            )
            actual.extend(y[indices])
        self.assertLess(
            relational.negative_log_likelihood(actual, correct),
            relational.negative_log_likelihood(actual, wrong) - 0.02,
        )

    def test_zero_person_signal_does_not_create_profile_gain(self) -> None:
        rows, y, people, parties, congresses = synthetic_relation(person_signal=False, seed=91)
        feature_map = relational.RelationalFeatureMap.fit(rows, hash_dimension=32, svd_rank=2)
        artifact = relational.fit_relational_artifact(
            feature_map,
            rows,
            y,
            people,
            parties,
            congresses,
            stable_person_precision=64.0,
        )
        norms = [
            np.linalg.norm(artifact.joint_model.person_weights[person_id])
            for person_id in ("d1", "d2", "r1", "r2")
        ]
        self.assertLess(max(norms), 0.3)

    def test_shared_regime_shift_is_not_person_identity(self) -> None:
        rows, y, people, parties, congresses = synthetic_relation(
            person_signal=False, regime_only=True, seed=101
        )
        feature_map = relational.RelationalFeatureMap.fit(rows, hash_dimension=32, svd_rank=2)
        artifact = relational.fit_relational_artifact(
            feature_map,
            rows,
            y,
            people,
            parties,
            congresses,
            stable_person_precision=64.0,
        )
        norms = [
            np.linalg.norm(artifact.joint_model.person_weights[person_id])
            for person_id in ("d1", "d2", "r1", "r2")
        ]
        self.assertLess(max(norms), 0.35)

    def test_current_outcome_fields_cannot_change_features(self) -> None:
        basis = [rollcall("alpha"), rollcall("beta")]
        feature_map = relational.RelationalFeatureMap.fit(
            basis * 6, hash_dimension=32, svd_rank=2
        )
        original = rollcall("alpha", result="Agreed")
        attacked = dict(original)
        attacked.update(
            {
                "cast_code": 6,
                "prob": 0.0,
                "yea_count": 0,
                "nay_count": 100,
                "vote_result": "Rejected",
                "nominate_mid_1": -100.0,
                "nominate_log_likelihood": -999.0,
            }
        )
        self.assertTrue(
            np.array_equal(feature_map.transform(original), feature_map.transform(attacked))
        )

    def test_text_and_person_identifier_invariance(self) -> None:
        first = rollcall("alpha")
        second = dict(first)
        second["crs_subjects"] = list(reversed(first["crs_subjects"]))
        feature_map = relational.RelationalFeatureMap.fit(
            [first, rollcall("beta")] * 6, hash_dimension=32, svd_rank=2
        )
        self.assertTrue(np.array_equal(feature_map.transform(first), feature_map.transform(second)))

        rows, y, people, parties, congresses = synthetic_relation(person_signal=True, seed=113)
        renamed = {"d1": "x9", "d2": "x2", "r1": "q8", "r2": "q1"}
        a = relational.fit_relational_artifact(
            feature_map, rows, y, people, parties, congresses, stable_person_precision=16.0
        )
        b = relational.fit_relational_artifact(
            feature_map,
            rows,
            y,
            [renamed[value] for value in people],
            parties,
            congresses,
            stable_person_precision=16.0,
        )
        probe = [rollcall("alpha"), rollcall("beta")]
        self.assertTrue(
            np.allclose(
                a.probabilities("d1", probe, 100, 117),
                b.probabilities("x9", probe, 100, 117),
                atol=1e-10,
            )
        )

    def test_reload_and_prequential_probability_equality(self) -> None:
        rows, y, people, parties, congresses = synthetic_relation(person_signal=True, seed=127)
        feature_map = relational.RelationalFeatureMap.fit(rows, hash_dimension=32, svd_rank=2)
        artifact = relational.fit_relational_artifact(
            feature_map, rows, y, people, parties, congresses, stable_person_precision=16.0
        )
        restored = relational.RelationalCoreArtifact.from_dict(artifact.to_dict())
        probe = [rollcall("alpha", congress=119), rollcall("beta", congress=119)] * 20
        self.assertEqual(artifact.artifact_id, restored.artifact_id)
        self.assertTrue(
            np.allclose(
                artifact.probabilities("d1", probe, 100, 119),
                restored.probabilities("d1", probe, 100, 119),
                atol=1e-12,
            )
        )
        timestamps = tuple(
            (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
            for i in range(len(probe))
        )
        choices = np.asarray(([0, 1] * 20), dtype=np.int64)
        flipped = choices.copy()
        flipped[-1] = 1 - flipped[-1]
        config = relational.StateConfig(half_life_days=30.0, stationary_variance=0.5)
        original = artifact.run_prequential("d1", probe, 100, 119, choices, timestamps, config)
        attacked = restored.run_prequential("d1", probe, 100, 119, flipped, timestamps, config)
        self.assertTrue(np.allclose(original.probabilities, attacked.probabilities))
        self.assertNotEqual(original.final_state_mean, attacked.final_state_mean)

    def test_feature_map_round_trip(self) -> None:
        rows = [rollcall("alpha"), rollcall("beta")] * 8
        feature_map = relational.RelationalFeatureMap.fit(rows, hash_dimension=32, svd_rank=2)
        restored = relational.RelationalFeatureMap.from_dict(feature_map.to_dict())
        self.assertEqual(feature_map.map_id, restored.map_id)
        self.assertTrue(
            np.allclose(feature_map.transform(rows[0]), restored.transform(rows[0]), atol=0.0)
        )


if __name__ == "__main__":
    unittest.main()
