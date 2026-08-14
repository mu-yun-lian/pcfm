from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import math
import unittest

from pcfm.applicability import (
    PredictionRefusedError,
    assess_temporal_stability,
    fit_applicability_profile,
)
from pcfm.contracts import Observation, Scenario
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.storage import bundle_from_dict, bundle_to_dict
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import fit_person_model, predict_with_bundle


class Stage3ApplicabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        people, source, target = generate_population_dataset(
            seed=401,
            person_count=24,
            source_trials=140,
            target_trials=660,
            heterogeneity_scale=1.5,
        )
        cls.person_id = people[0].person_id
        target_observations = target[cls.person_id]
        cls.calibration = target_observations[:220]
        cls.validation = target_observations[220:440]
        cls.target = target_observations[440:]
        cls.authority = VerificationAuthority(
            {"boundary-verifier": b"stage-three-boundary-secret"}
        )
        training = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        cls.training_ledger = cls._ledger(training, "training")
        cls.applicability_ledger = cls._ledger(
            cls.calibration,
            "applicability",
        )
        cls.validation_ledger = cls._ledger(
            cls.validation,
            "validation",
        )
        cls.bundle = fit_person_model(
            cls.training_ledger,
            cls.authority,
            applicability_ledger=cls.applicability_ledger,
            validation_ledger=cls.validation_ledger,
            person_id=cls.person_id,
            feature_names=FEATURE_NAMES,
        )

    @classmethod
    def _ledger(cls, observations, prefix: str) -> EventLedger:
        if prefix == "training":
            observed_times = (
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:00Z",
            )
        elif prefix == "applicability":
            observed_times = (
                "2026-07-15T00:00:00Z",
                "2026-07-15T00:00:00Z",
            )
        else:
            observed_times = (
                "2026-08-01T00:00:00Z",
                "2026-09-01T00:00:00Z",
            )
        records = tuple(
            cls.authority.sign(
                event_id=f"{prefix}-{index:05d}",
                observation=observation,
                observed_at=(
                    observed_times[0]
                    if index < len(observations) // 2
                    else observed_times[1]
                ),
                evidence_hash=hashlib.sha256(
                    f"{prefix}-{index:05d}".encode()
                ).hexdigest(),
                verifier_id="boundary-verifier",
                verified_at=(
                    observed_times[0]
                    if index < len(observations) // 2
                    else observed_times[1]
                ),
            )
            for index, observation in enumerate(observations)
        )
        return EventLedger.verify(records, cls.authority)

    def test_in_distribution_scenario_is_assessed_and_predicted(self) -> None:
        prediction = predict_with_bundle(
            self.bundle,
            self.target[0].scenario,
            prediction_at="2026-09-02T00:00:00Z",
        )
        self.assertEqual(prediction.applicability_status, "in_distribution")
        self.assertIsNotNone(prediction.ood_score)
        self.assertLessEqual(
            prediction.ood_score,
            prediction.ood_threshold,
        )
        self.assertIn("applicability_guard", prediction.active_modules)

    def test_remote_scenario_is_refused_despite_matching_schema(self) -> None:
        original = self.target[0].scenario
        remote = Scenario(
            scenario_id="remote-but-schema-compatible",
            features=tuple(value + 25.0 for value in original.features),
            feature_names=original.feature_names,
            domain=original.domain,
        )
        with self.assertRaises(PredictionRefusedError) as raised:
            predict_with_bundle(
                self.bundle,
                remote,
                prediction_at="2026-09-02T00:00:00Z",
            )
        self.assertIn("feature_distribution_shift", raised.exception.reasons)
        self.assertGreater(
            raised.exception.ood_score,
            raised.exception.ood_threshold,
        )
        diagnostic = predict_with_bundle(
            self.bundle,
            remote,
            prediction_at="2026-09-02T00:00:00Z",
            applicability_override=True,
        )
        self.assertEqual(diagnostic.applicability_status, "overridden")
        self.assertIn(
            "feature_distribution_shift",
            diagnostic.gate_overrides,
        )
        self.assertEqual(
            diagnostic.model_form_uncertainty_status,
            "unquantified_override",
        )

    def test_fresh_scenarios_from_supported_distribution_are_accepted(
        self,
    ) -> None:
        people, _, target = generate_population_dataset(
            seed=402,
            person_count=2,
            source_trials=5,
            target_trials=500,
        )
        scenarios = tuple(
            observation.scenario
            for observation in target[people[0].person_id]
        )
        assessments = tuple(
            self.bundle.manifest.applicability_profile.assess(
                scenario,
                prediction_at="2026-09-02T00:00:00Z",
            )
            for scenario in scenarios
        )
        accepted_rate = sum(
            not assessment.reasons for assessment in assessments
        ) / len(assessments)
        self.assertGreaterEqual(accepted_rate, 0.98)

    def test_unknown_domain_is_flagged_but_not_automatically_refused(
        self,
    ) -> None:
        original = self.target[0].scenario
        unknown = replace(
            original,
            scenario_id="unknown-domain",
            domain="unseen_real_world_domain",
        )
        prediction = predict_with_bundle(
            self.bundle,
            unknown,
            prediction_at="2026-09-02T00:00:00Z",
        )
        self.assertEqual(
            prediction.applicability_status,
            "cross_domain_extrapolation",
        )
        self.assertIn(
            "unvalidated_domain_label",
            prediction.applicability_warnings,
        )
        self.assertEqual(
            prediction.model_form_uncertainty_status,
            "unquantified_extrapolation",
        )
        self.assertIsNone(prediction.probability_lower_95)
        self.assertIsNone(prediction.probability_upper_95)

    def test_multimodal_support_hole_is_refused_locally(self) -> None:
        observations = []
        for index in range(240):
            angle = 2.0 * math.pi * index / 240
            scenario = Scenario(
                scenario_id=f"ring-{index:04d}",
                features=(
                    5.0 * math.cos(angle),
                    5.0 * math.sin(angle),
                ),
                feature_names=("x", "y"),
                domain="ring",
            )
            observations.append(
                Observation(
                    person_id="ring-person",
                    scenario=scenario,
                    actual_choice=index % 2,
                )
            )
        profile = fit_applicability_profile(
            tuple(observations),
            ("x", "y"),
        )
        assessment = profile.assess(
            Scenario(
                scenario_id="unsupported-center",
                features=(0.0, 0.0),
                feature_names=("x", "y"),
                domain="ring",
            )
        )
        self.assertIn("local_support_gap", assessment.reasons)
        self.assertGreater(
            assessment.local_ood_score,
            assessment.local_ood_threshold,
        )

    def test_applicability_profile_round_trips_and_is_hash_protected(self) -> None:
        restored = bundle_from_dict(bundle_to_dict(self.bundle))
        self.assertEqual(
            restored.manifest.applicability_profile,
            self.bundle.manifest.applicability_profile,
        )
        tampered = copy.deepcopy(bundle_to_dict(self.bundle))
        tampered["manifest"]["applicability_profile"][
            "squared_distance_threshold"
        ] *= 100.0
        with self.assertRaisesRegex(ValueError, "model_id"):
            bundle_from_dict(tampered)

    def test_final_validation_features_do_not_enter_boundary_profile(
        self,
    ) -> None:
        profile = self.bundle.manifest.applicability_profile
        self.assertEqual(
            (
                profile.reference_sample_count
                + profile.calibration_sample_count
            ),
            len(self.applicability_ledger.records),
        )
        self.assertEqual(
            profile.reference_sample_count,
            len(profile.reference_features),
        )

    def test_applicability_fit_is_invariant_to_input_order(self) -> None:
        forward = fit_applicability_profile(
            self.calibration,
            FEATURE_NAMES,
            valid_through="2026-07-15T00:00:00Z",
        )
        reversed_profile = fit_applicability_profile(
            tuple(reversed(self.calibration)),
            FEATURE_NAMES,
            valid_through="2026-07-15T00:00:00Z",
        )
        self.assertEqual(forward.digest(), reversed_profile.digest())

    def test_unseen_options_and_context_are_extrapolation_warnings(
        self,
    ) -> None:
        original = self.target[0].scenario
        changed = replace(
            original,
            scenario_id="new-options-and-context",
            options=("accept", "reject"),
            context={"condition": "target", "stake": "irreversible"},
        )
        prediction = predict_with_bundle(
            self.bundle,
            changed,
            prediction_at="2026-09-02T00:00:00Z",
        )
        self.assertIn(
            "unvalidated_option_pair",
            prediction.applicability_warnings,
        )
        self.assertIn(
            "unvalidated_context",
            prediction.applicability_warnings,
        )
        self.assertEqual(
            prediction.model_form_uncertainty_status,
            "unquantified_extrapolation",
        )

    def test_temporal_residual_shift_is_detected(self) -> None:
        observations = []
        probabilities = []
        records = []
        for index in range(120):
            source = self.target[index]
            early = index < 60
            observation = replace(
                source,
                actual_choice=1 if early else 0,
            )
            observed_at = (
                "2026-08-01T00:00:00Z"
                if early
                else "2026-10-01T00:00:00Z"
            )
            event_id = f"drift-{index:04d}"
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="boundary-verifier",
                    verified_at=observed_at,
                )
            )
            observations.append(observation)
            probabilities.append(0.8)
        stability = assess_temporal_stability(
            tuple(records),
            tuple(probabilities),
            FEATURE_NAMES,
        )
        self.assertEqual(stability.status, "unstable")
        self.assertTrue(stability.drift_detected)
        self.assertGreater(stability.maximum_score_z, 3.3)

    def test_unchanged_systematic_bias_is_not_temporal_drift(self) -> None:
        records = []
        for index in range(120):
            observed_at = (
                "2026-08-01T00:00:00Z"
                if index < 60
                else "2026-10-01T00:00:00Z"
            )
            scenario = Scenario(
                scenario_id=f"stable-bias-{index:04d}",
                features=(float(index % 2),),
                feature_names=("x",),
                domain="stable-bias",
            )
            observation = Observation(
                person_id="stable-person",
                scenario=scenario,
                actual_choice=1,
            )
            event_id = f"stable-bias-{index:04d}"
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="boundary-verifier",
                    verified_at=observed_at,
                )
            )
        stability = assess_temporal_stability(
            tuple(records),
            (0.8,) * len(records),
            ("x",),
        )
        self.assertEqual(stability.status, "stable")
        self.assertFalse(stability.drift_detected)
        self.assertAlmostEqual(
            stability.early_nll,
            stability.late_nll,
        )

    def test_stale_model_is_refused(self) -> None:
        with self.assertRaises(PredictionRefusedError) as raised:
            predict_with_bundle(
                self.bundle,
                self.target[0].scenario,
                prediction_at="2027-08-01T00:00:00Z",
            )
        self.assertIn("stale_model", raised.exception.reasons)

    def test_prediction_time_is_required(self) -> None:
        with self.assertRaises(PredictionRefusedError) as raised:
            predict_with_bundle(
                self.bundle,
                self.target[0].scenario,
            )
        self.assertIn(
            "prediction_time_required",
            raised.exception.reasons,
        )

    def test_validation_override_is_explicit_in_prediction(self) -> None:
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            validation_ledger=None,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        prediction = predict_with_bundle(
            bundle,
            self.training_ledger.records_for_person(
                self.person_id
            )[0].observation.scenario,
            validation_override=True,
            prediction_at="2026-09-02T00:00:00Z",
        )
        self.assertIn(
            "model_validation_unvalidated",
            prediction.gate_overrides,
        )
        self.assertEqual(
            prediction.applicability_status,
            "in_distribution",
        )
        self.assertEqual(
            prediction.model_form_uncertainty_status,
            "unquantified_override",
        )

    def test_validation_override_does_not_bypass_applicability(
        self,
    ) -> None:
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            validation_ledger=None,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        original = self.training_ledger.records_for_person(
            self.person_id
        )[0].observation.scenario
        remote = replace(
            original,
            scenario_id="unvalidated-remote",
            features=tuple(value + 30.0 for value in original.features),
        )
        with self.assertRaises(PredictionRefusedError) as raised:
            predict_with_bundle(
                bundle,
                remote,
                prediction_at="2026-09-02T00:00:00Z",
                validation_override=True,
            )
        self.assertIn(
            "feature_distribution_shift",
            raised.exception.reasons,
        )
    def test_stable_temporal_windows_are_not_flagged(self) -> None:
        records = []
        probabilities = []
        for index, record in enumerate(self.validation_ledger.records):
            observed_at = (
                "2026-08-01T00:00:00Z"
                if index < len(self.validation_ledger.records) // 2
                else "2026-10-01T00:00:00Z"
            )
            records.append(
                self.authority.sign(
                    event_id=record.event_id,
                    observation=record.observation,
                    observed_at=observed_at,
                    evidence_hash=record.evidence_hash,
                    verifier_id="boundary-verifier",
                    verified_at=observed_at,
                )
            )
            probabilities.append(
                predict_with_bundle(
                    self.bundle,
                    record.observation.scenario,
                    prediction_at="2026-09-02T00:00:00Z",
                    applicability_override=True,
                ).probability_option_1
            )
        stability = assess_temporal_stability(
            tuple(records),
            tuple(probabilities),
            FEATURE_NAMES,
        )
        self.assertEqual(stability.status, "stable")
        self.assertFalse(stability.drift_detected)

    def test_temporal_stability_is_not_claimed_without_two_windows(
        self,
    ) -> None:
        records = self.validation_ledger.records[:80]
        probabilities = tuple(
            predict_with_bundle(
                self.bundle,
                record.observation.scenario,
                prediction_at="2026-09-02T00:00:00Z",
                applicability_override=True,
            ).probability_option_1
            for record in records
        )
        stability = assess_temporal_stability(
            records,
            probabilities,
            FEATURE_NAMES,
        )
        self.assertEqual(stability.status, "not_assessed")
        self.assertFalse(stability.drift_detected)
        self.assertIsNone(stability.maximum_score_z)

    def test_temporal_drift_is_included_in_workflow_validation(self) -> None:
        drifted_records = []
        for index, record in enumerate(self.validation_ledger.records):
            late = index >= len(self.validation_ledger.records) // 2
            observation = (
                replace(
                    record.observation,
                    actual_choice=1 - record.observation.actual_choice,
                )
                if late
                else record.observation
            )
            observed_at = (
                "2026-08-01T00:00:00Z"
                if not late
                else "2026-10-01T00:00:00Z"
            )
            drifted_records.append(
                self.authority.sign(
                    event_id=record.event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=record.evidence_hash,
                    verifier_id="boundary-verifier",
                    verified_at=observed_at,
                )
            )
        drifted = EventLedger.verify(drifted_records, self.authority)
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            applicability_ledger=self.applicability_ledger,
            validation_ledger=drifted,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(bundle.manifest.validation.status, "failed")
        self.assertEqual(
            bundle.manifest.validation.temporal_stability_status,
            "unstable",
        )
        self.assertEqual(
            bundle.manifest.validation.temporal_early_sample_count,
            110,
        )
        self.assertEqual(
            bundle.manifest.validation.temporal_late_sample_count,
            110,
        )
        self.assertIn(
            "temporal_behavior_drift_suspected",
            bundle.manifest.validation.reasons,
        )

    def test_unassessed_temporal_stability_cannot_fully_pass(
        self,
    ) -> None:
        records = []
        observed_at = "2026-08-15T00:00:00Z"
        for record in self.validation_ledger.records:
            records.append(
                self.authority.sign(
                    event_id=record.event_id,
                    observation=record.observation,
                    observed_at=observed_at,
                    evidence_hash=record.evidence_hash,
                    verifier_id="boundary-verifier",
                    verified_at=observed_at,
                )
            )
        same_time = EventLedger.verify(records, self.authority)
        bundle = fit_person_model(
            self.training_ledger,
            self.authority,
            applicability_ledger=self.applicability_ledger,
            validation_ledger=same_time,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(bundle.manifest.validation.status, "failed")
        self.assertIn(
            "temporal_stability_not_assessed",
            bundle.manifest.validation.reasons,
        )


if __name__ == "__main__":
    unittest.main()
