from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from pcfm.core import (
    IdentityAdapterGenerator,
    MapPersonEncoder,
    ModelUpdater,
    PopulationPriorEstimator,
)
from pcfm.ledger import VerificationAuthority
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset


class UpdateSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        people, source, target = generate_population_dataset(
            seed=7,
            person_count=4,
            source_trials=50,
            target_trials=10,
        )
        self.person = people[0]
        self.source = source
        self.target = target
        all_source = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        self.prior = PopulationPriorEstimator(FEATURE_NAMES).fit(all_source)
        self.updater = ModelUpdater(
            encoder=MapPersonEncoder(),
            adapter_generator=IdentityAdapterGenerator(),
        )
        self.authority = VerificationAuthority(
            {"test-verifier": b"update-test-secret"}
        )

    def _signed_outcome(self):
        trial = self.target[self.person.person_id][0]
        return self.authority.sign(
            event_id="new-event",
            observation=trial,
            observed_at="2026-07-30T00:00:00Z",
            evidence_hash=hashlib.sha256(b"test-evidence").hexdigest(),
            verifier_id="test-verifier",
            verified_at="2026-07-30T00:01:00Z",
        )

    def test_invalid_signature_is_rejected(self) -> None:
        outcome = replace(self._signed_outcome(), signature="0" * 64)
        with self.assertRaisesRegex(ValueError, "signature"):
            self.updater.update(
                self.person.person_id,
                self.source[self.person.person_id],
                outcome,
                self.authority,
                self.prior,
            )

    def test_model_generated_outcome_is_rejected(self) -> None:
        trial = self.target[self.person.person_id][0]
        generated = replace(trial, provenance="model_prediction")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            self.authority.sign(
                event_id="generated-event",
                observation=generated,
                observed_at="2026-07-30T00:00:00Z",
                evidence_hash=hashlib.sha256(b"generated").hexdigest(),
                verifier_id="test-verifier",
                verified_at="2026-07-30T00:01:00Z",
            )

    def test_verified_outcome_updates_observation_count(self) -> None:
        outcome = self._signed_outcome()
        updated = self.updater.update(
            self.person.person_id,
            self.source[self.person.person_id],
            outcome,
            self.authority,
            self.prior,
        )
        self.assertEqual(
            updated.representation.observation_count,
            len(self.source[self.person.person_id]) + 1,
        )
        self.assertEqual(len(updated.observations), 51)

    def test_update_rejects_mixed_person_history(self) -> None:
        other_person_id = next(
            person_id
            for person_id in self.source
            if person_id != self.person.person_id
        )
        mixed = (
            self.source[self.person.person_id][0],
            self.source[other_person_id][0],
        )
        outcome = self._signed_outcome()
        with self.assertRaisesRegex(ValueError, "only the target person"):
            self.updater.update(
                self.person.person_id,
                mixed,
                outcome,
                self.authority,
                self.prior,
            )


if __name__ == "__main__":
    unittest.main()
