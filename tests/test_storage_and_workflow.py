from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.storage import load_bundle, save_bundle
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import (
    fit_person_model,
    load_observations_jsonl,
    predict_with_bundle,
    update_person_model,
)


class StorageAndWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        people, source, target = generate_population_dataset(
            seed=21,
            person_count=5,
            source_trials=60,
            target_trials=5,
        )
        self.person_id = people[0].person_id
        self.all_source = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        self.person_source = source[self.person_id]
        self.target = target[self.person_id]
        self.authority = VerificationAuthority(
            {"test-verifier": b"workflow-test-secret"}
        )
        self.ledger = EventLedger.verify(
            tuple(
                self.authority.sign(
                    event_id=f"source-{index:04d}",
                    observation=observation,
                    observed_at="2026-07-30T00:00:00Z",
                    evidence_hash=hashlib.sha256(
                        observation.scenario.scenario_id.encode()
                    ).hexdigest(),
                    verifier_id="test-verifier",
                    verified_at="2026-07-30T00:01:00Z",
                )
                for index, observation in enumerate(self.all_source)
            ),
            self.authority,
        )

    def test_bundle_round_trip_preserves_prediction(self) -> None:
        bundle = fit_person_model(
            self.ledger,
            self.authority,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        expected = predict_with_bundle(
            bundle,
            self.target[0].scenario,
            prediction_at="2026-07-31T00:00:00Z",
            validation_override=True,
            applicability_override=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person-model.json"
            save_bundle(path, bundle)
            restored = load_bundle(path)
        actual = predict_with_bundle(
            restored,
            self.target[0].scenario,
            prediction_at="2026-07-31T00:00:00Z",
            validation_override=True,
            applicability_override=True,
        )
        self.assertEqual(expected, actual)

    def test_workflow_update_uses_verified_outcome(self) -> None:
        bundle = fit_person_model(
            self.ledger,
            self.authority,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        trial = self.target[0]
        outcome = self.authority.sign(
            event_id="target-0000",
            observation=trial,
            observed_at="2026-07-31T00:00:00Z",
            evidence_hash=hashlib.sha256(b"target-0000").hexdigest(),
            verifier_id="test-verifier",
            verified_at="2026-07-31T00:01:00Z",
        )
        updated = update_person_model(
            bundle,
            self.ledger,
            outcome,
            self.authority,
        )
        self.assertEqual(
            updated.bundle.representation.observation_count,
            len(self.person_source) + 1,
        )

    def test_jsonl_loader_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"person_id": "x"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_observations_jsonl(path)


if __name__ == "__main__":
    unittest.main()
