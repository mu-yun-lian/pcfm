from __future__ import annotations

import hashlib
import copy
import unittest

import numpy as np

from pcfm.core import PopulationPriorEstimator
from pcfm.demo import run_demo, run_misspecification_demo
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.storage import bundle_from_dict, bundle_to_dict
from pcfm.workflow import fit_person_model, predict_with_bundle


class Stage2GateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        people, source, target = generate_population_dataset(
            seed=301,
            person_count=24,
            source_trials=140,
            target_trials=440,
            heterogeneity_scale=1.5,
        )
        self.person_id = people[0].person_id
        target_observations = target[self.person_id]
        self.target_scenario = target_observations[-1].scenario
        self.authority = VerificationAuthority(
            {"gate-verifier": b"gate-integration-secret"}
        )
        source_observations = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        self.training_ledger = self._ledger(source_observations, "train")
        self.applicability_ledger = self._ledger(
            target_observations[:220],
            "applicability",
        )
        self.validation_ledger = self._ledger(
            target_observations[220:],
            "validation",
        )

    def _ledger(self, observations, prefix: str) -> EventLedger:
        records = []
        for index, observation in enumerate(observations):
            event_id = f"{prefix}-{index:05d}"
            if prefix == "train":
                observed_at = "2026-07-01T00:00:00Z"
            elif prefix == "applicability":
                observed_at = "2026-07-15T00:00:00Z"
            else:
                observed_at = (
                    "2026-08-01T00:00:00Z"
                    if index < len(observations) // 2
                    else "2026-09-01T00:00:00Z"
                )
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(event_id.encode()).hexdigest(),
                    verifier_id="gate-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(records, self.authority)

    def test_validation_is_persisted_and_controls_prediction(self) -> None:
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            applicability_ledger=self.applicability_ledger,
            validation_ledger=self.validation_ledger,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(bundle.manifest.validation.status, "passed")
        self.assertTrue(bundle.manifest.validation.personalization_passed)
        prediction = predict_with_bundle(
            bundle,
            self.target_scenario,
            prediction_at="2026-09-02T00:00:00Z",
        )
        self.assertEqual(prediction.person_id, self.person_id)
        restored = bundle_from_dict(bundle_to_dict(bundle))
        self.assertEqual(restored.manifest.validation.status, "passed")
        self.assertEqual(
            restored.manifest.validation.validation_data_hash,
            bundle.manifest.validation.validation_data_hash,
        )

    def test_validation_diagnostics_are_covered_by_model_identity(self) -> None:
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            applicability_ledger=self.applicability_ledger,
            validation_ledger=self.validation_ledger,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        tampered = copy.deepcopy(bundle_to_dict(bundle))
        tampered["manifest"]["validation"]["nll_uplift"] += 0.1
        with self.assertRaisesRegex(ValueError, "model_id"):
            bundle_from_dict(tampered)

    def test_training_data_cannot_be_reused_as_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "only the target person|overlap"):
            fit_person_model(
                self.training_ledger,
                self.authority,
                applicability_ledger=self.applicability_ledger,
                validation_ledger=self.training_ledger,
                person_id=self.person_id,
                feature_names=FEATURE_NAMES,
            )

    def test_validated_model_requires_separate_applicability_data(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "applicability calibration ledger is required",
        ):
            fit_person_model(
                self.training_ledger,
                self.authority,
                validation_ledger=self.validation_ledger,
                person_id=self.person_id,
                feature_names=FEATURE_NAMES,
            )

    def test_applicability_and_validation_scenarios_cannot_overlap(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_person_model(
                self.training_ledger,
                self.authority,
                applicability_ledger=self.validation_ledger,
                validation_ledger=self.validation_ledger,
                person_id=self.person_id,
                feature_names=FEATURE_NAMES,
            )

    def test_unvalidated_model_is_blocked_without_explicit_validation_override(self) -> None:
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            validation_ledger=None,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(bundle.manifest.validation.status, "unvalidated")
        with self.assertRaisesRegex(ValueError, "validation"):
            predict_with_bundle(
                bundle,
                self.target_scenario,
                prediction_at="2026-09-02T00:00:00Z",
            )
        prediction = predict_with_bundle(
            bundle,
            self.target_scenario,
            prediction_at="2026-09-02T00:00:00Z",
            validation_override=True,
            applicability_override=True,
        )
        self.assertEqual(prediction.person_id, self.person_id)
        self.assertEqual(
            prediction.model_form_uncertainty_status,
            "unquantified_override",
        )
        self.assertTrue(prediction.gate_overrides)

    def test_misspecification_and_no_personalization_are_separate(self) -> None:
        misspecified = run_misspecification_demo(
            seed=302,
            person_count=20,
            source_trials=120,
            target_trials=180,
        )
        no_individuality = run_demo(
            seed=303,
            person_count=20,
            source_trials=120,
            target_trials=180,
            heterogeneity_scale=0.0,
        )
        self.assertFalse(
            misspecified["validity_gate"]["mechanism_adequacy"]["passed"]
        )
        self.assertTrue(
            no_individuality["validity_gate"]["mechanism_adequacy"]["passed"]
        )
        self.assertFalse(
            no_individuality["validity_gate"]["personalization"]["passed"]
        )

    def test_laplace_eb_is_stable_across_initial_variance(self) -> None:
        people, source, _ = generate_population_dataset(
            seed=304,
            person_count=24,
            source_trials=120,
            target_trials=5,
        )
        observations = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        low = PopulationPriorEstimator(
            FEATURE_NAMES,
            initial_person_variance=0.05,
        ).fit(observations)
        high = PopulationPriorEstimator(
            FEATURE_NAMES,
            initial_person_variance=4.0,
        ).fit(observations)
        low_variance = np.diag(low.covariance)
        high_variance = np.diag(high.covariance)
        relative_difference = np.max(
            np.abs(low_variance - high_variance)
            / np.maximum(low_variance, high_variance)
        )
        self.assertLess(relative_difference, 0.08)

    def test_wrong_person_baseline_uses_all_other_people(self) -> None:
        report = run_demo(
            seed=305,
            person_count=8,
            source_trials=80,
            target_trials=60,
        )
        self.assertEqual(
            report["models"]["wrong_person"]["sample_count"],
            8 * 7 * 60,
        )

    def test_misspecified_metadata_reports_actual_heterogeneity(self) -> None:
        report = run_misspecification_demo(
            seed=306,
            person_count=8,
            source_trials=80,
            target_trials=60,
        )
        self.assertEqual(report["experiment"]["heterogeneity_scale"], 0.0)
        self.assertFalse(
            report["experiment"]["heterogeneity_scale_applies"]
        )


if __name__ == "__main__":
    unittest.main()
