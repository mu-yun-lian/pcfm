from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from pcfm.contracts import Observation, Scenario
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.storage import bundle_from_dict, bundle_to_dict
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import (
    fit_person_model,
    predict_with_bundle,
    update_person_model,
)


class IntegrityV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        people, source, target = generate_population_dataset(
            seed=31,
            person_count=3,
            source_trials=60,
            target_trials=2,
        )
        self.person_id = people[0].person_id
        self.target = target[self.person_id]
        self.authority = VerificationAuthority(
            {"test-verifier": b"integrity-test-secret"}
        )
        all_source = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        self.ledger = self._signed_ledger(all_source)
        self.bundle = fit_person_model(
            self.ledger,
            self.authority,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )

    def _signed_ledger(
        self,
        observations: tuple[Observation, ...],
    ) -> EventLedger:
        records = []
        for index, observation in enumerate(observations):
            evidence = (
                f"{observation.person_id}:{observation.scenario.scenario_id}"
            ).encode()
            records.append(
                self.authority.sign(
                    event_id=f"event-{index:04d}",
                    observation=observation,
                    observed_at="2026-07-30T00:00:00Z",
                    evidence_hash=hashlib.sha256(evidence).hexdigest(),
                    verifier_id="test-verifier",
                    verified_at="2026-07-30T00:01:00Z",
                )
            )
        return EventLedger.verify(records, self.authority)

    def _signed_outcome(self, event_id: str = "new-event"):
        observation = self.target[0]
        return self.authority.sign(
            event_id=event_id,
            observation=observation,
            observed_at="2026-07-31T00:00:00Z",
            evidence_hash=hashlib.sha256(event_id.encode()).hexdigest(),
            verifier_id="test-verifier",
            verified_at="2026-07-31T00:01:00Z",
        )

    def test_update_rejects_truncated_history(self) -> None:
        person_records = self.ledger.records_for_person(self.person_id)
        truncated = EventLedger.verify(person_records[:1], self.authority)
        with self.assertRaisesRegex(ValueError, "history"):
            update_person_model(
                self.bundle,
                truncated,
                self._signed_outcome(),
                self.authority,
            )

    def test_duplicate_event_id_is_rejected(self) -> None:
        record = self.ledger.records[0]
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            EventLedger.verify((record, record), self.authority)

    def test_update_rejects_existing_event(self) -> None:
        existing = self.ledger.records_for_person(self.person_id)[0]
        with self.assertRaisesRegex(ValueError, "already exists"):
            update_person_model(
                self.bundle,
                self.ledger,
                existing,
                self.authority,
            )

    def test_update_rejects_same_trial_with_new_event_id(self) -> None:
        existing = self.ledger.records_for_person(self.person_id)[0]
        duplicate_trial = self.authority.sign(
            event_id="renamed-duplicate-event",
            observation=existing.observation,
            observed_at="2026-07-31T00:00:00Z",
            evidence_hash=hashlib.sha256(b"duplicate-trial").hexdigest(),
            verifier_id="test-verifier",
            verified_at="2026-07-31T00:01:00Z",
        )
        with self.assertRaisesRegex(ValueError, "duplicate person/scenario"):
            update_person_model(
                self.bundle,
                self.ledger,
                duplicate_trial,
                self.authority,
            )

    def test_update_rejects_resigned_history_change(self) -> None:
        original = self.ledger.records_for_person(self.person_id)[0]
        changed_observation = replace(
            original.observation,
            actual_choice=1 - original.observation.actual_choice,
        )
        changed = self.authority.sign(
            event_id=original.event_id,
            observation=changed_observation,
            observed_at=original.observed_at,
            evidence_hash=hashlib.sha256(b"changed-evidence").hexdigest(),
            verifier_id=original.verifier_id,
            verified_at="2026-07-30T00:02:00Z",
        )
        changed_records = tuple(
            changed if record.event_id == original.event_id else record
            for record in self.ledger.records
        )
        changed_ledger = EventLedger.verify(changed_records, self.authority)
        with self.assertRaisesRegex(ValueError, "content"):
            update_person_model(
                self.bundle,
                changed_ledger,
                self._signed_outcome(),
                self.authority,
            )

    def test_invalid_signature_is_rejected(self) -> None:
        signed = self._signed_outcome()
        forged = replace(signed, signature="0" * 64)
        with self.assertRaisesRegex(ValueError, "signature"):
            EventLedger.verify((forged,), self.authority)

    def test_fit_rechecks_programmatically_constructed_ledger(self) -> None:
        signed = self.ledger.records[0]
        forged = replace(signed, signature="0" * 64)
        bypass_attempt = EventLedger(records=(forged,))
        with self.assertRaisesRegex(ValueError, "signature"):
            fit_person_model(
                bypass_attempt,
                self.authority,
                person_id=forged.observation.person_id,
                feature_names=FEATURE_NAMES,
            )

    def test_named_feature_reordering_preserves_prediction(self) -> None:
        scenario = self.target[0].scenario
        reordered = Scenario(
            scenario_id="reordered",
            features=tuple(reversed(scenario.features)),
            feature_names=tuple(reversed(scenario.feature_names)),
            options=scenario.options,
            domain=scenario.domain,
        )
        expected = predict_with_bundle(
            self.bundle,
            scenario,
            prediction_at="2026-07-31T00:00:00Z",
            validation_override=True,
            applicability_override=True,
        )
        actual = predict_with_bundle(
            self.bundle,
            reordered,
            prediction_at="2026-07-31T00:00:00Z",
            validation_override=True,
            applicability_override=True,
        )
        self.assertAlmostEqual(
            expected.probability_option_1,
            actual.probability_option_1,
        )

    def test_missing_named_feature_is_rejected(self) -> None:
        scenario = self.target[0].scenario
        incomplete = Scenario(
            scenario_id="incomplete",
            features=scenario.features[:-1],
            feature_names=scenario.feature_names[:-1],
        )
        with self.assertRaisesRegex(ValueError, "feature schema"):
            predict_with_bundle(
                self.bundle,
                incomplete,
                prediction_at="2026-07-31T00:00:00Z",
                validation_override=True,
                applicability_override=True,
            )

    def test_bundle_rejects_cross_person_adapter(self) -> None:
        raw = bundle_to_dict(self.bundle)
        raw["adapter"]["person_id"] = "different-person"
        with self.assertRaisesRegex(ValueError, "person"):
            bundle_from_dict(raw)

    def test_bundle_rejects_parameter_tampering(self) -> None:
        raw = bundle_to_dict(self.bundle)
        raw["adapter"]["delta_weights"][0] += 0.5
        with self.assertRaisesRegex(ValueError, "content hash"):
            bundle_from_dict(raw)

    def test_update_records_parent_model_and_new_event(self) -> None:
        outcome = self._signed_outcome()
        updated = update_person_model(
            self.bundle,
            self.ledger,
            outcome,
            self.authority,
        )
        self.assertEqual(
            updated.bundle.manifest.parent_model_id,
            self.bundle.manifest.model_id,
        )
        self.assertIn(
            outcome.event_id,
            updated.bundle.manifest.person_event_ids,
        )
        self.assertEqual(
            updated.bundle.representation.observation_count,
            self.bundle.representation.observation_count + 1,
        )


if __name__ == "__main__":
    unittest.main()
