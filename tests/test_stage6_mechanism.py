from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.math_utils import sigmoid
from pcfm.mechanism import (
    EvidenceWindow,
    MechanismComparisonConfig,
    MechanismHypothesis,
    MechanismRefusedError,
    MechanismTerm,
    compare_mechanisms,
    create_mechanism_comparison_plan,
    load_mechanism_comparison_plan,
    load_mechanism_comparison_report,
    mechanism_plan_from_dict,
    mechanism_report_from_dict,
    predict_with_mechanism,
    save_mechanism_comparison_plan,
    save_mechanism_comparison_report,
    verify_mechanism_report,
)
from pcfm.registry import ModuleSlot
from pcfm.storage import (
    PersonModelBundle,
    manifest_model_id,
    save_bundle,
)
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import (
    fit_person_model,
    save_event_ledger_jsonl,
)


class Stage6MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        people, source, target = generate_population_dataset(
            seed=501,
            person_count=24,
            source_trials=140,
            target_trials=1000,
            heterogeneity_scale=1.5,
        )
        cls.person = people[0]
        cls.person_id = cls.person.person_id
        cls.target = target[cls.person_id]
        cls.authority = VerificationAuthority(
            {"mechanism-verifier": b"stage-six-mechanism-secret"}
        )
        training = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        cls.training_ledger = cls._ledger(
            training,
            "training",
            "2026-07-01T00:00:00Z",
        )
        cls.applicability_ledger = cls._ledger(
            cls.target[:200],
            "applicability",
            "2026-07-15T00:00:00Z",
        )
        validation_records = []
        for index, observation in enumerate(cls.target[200:420]):
            observed_at = (
                "2026-08-01T00:00:00Z"
                if index < 110
                else "2026-09-01T00:00:00Z"
            )
            validation_records.append(
                cls._signed_record(
                    observation,
                    f"validation-{index:05d}",
                    observed_at,
                )
            )
        cls.validation_ledger = EventLedger.verify(
            tuple(validation_records),
            cls.authority,
        )
        cls.bundle = fit_person_model(
            cls.training_ledger,
            cls.authority,
            applicability_ledger=cls.applicability_ledger,
            validation_ledger=cls.validation_ledger,
            person_id=cls.person_id,
            feature_names=FEATURE_NAMES,
        )
        if cls.bundle.manifest.validation.status != "passed":
            raise AssertionError("stage-six fixture requires validation")
        rng = np.random.default_rng(602)
        nonlinear = []
        for observation in cls.target[420:980]:
            features = np.asarray(
                observation.scenario.ordered_features(FEATURE_NAMES),
                dtype=np.float64,
            )
            base_logit = float(
                features @ np.asarray(cls.person.true_weights)
            )
            correction = (
                2.8 * features[0] * features[3]
                - 2.4 * features[1] * features[4]
                + 1.8 * (abs(features[2]) - 0.7)
            )
            choice = int(rng.random() < sigmoid(base_logit + correction))
            nonlinear.append(
                replace(observation, actual_choice=choice)
            )
        cls.nonlinear = tuple(nonlinear)
        cls.hypotheses = (
            MechanismHypothesis(
                hypothesis_id="linear-residual",
                terms=tuple(
                    MechanismTerm(
                        term_id=f"linear-{name}",
                        kind="linear",
                        feature_names=(name,),
                    )
                    for name in FEATURE_NAMES
                ),
            ),
            MechanismHypothesis(
                hypothesis_id="nonlinear-interactions",
                terms=(
                    MechanismTerm("intercept", "intercept", ()),
                    MechanismTerm(
                        "reward-control",
                        "interaction",
                        ("reward_gain", "control"),
                    ),
                    MechanismTerm(
                        "risk-fairness",
                        "interaction",
                        ("loss_risk", "fairness"),
                    ),
                    MechanismTerm(
                        "absolute-delay",
                        "absolute",
                        ("delay",),
                    ),
                ),
            ),
            MechanismHypothesis(
                hypothesis_id="quadratic-residual",
                terms=tuple(
                    MechanismTerm(
                        term_id=f"square-{name}",
                        kind="quadratic",
                        feature_names=(name,),
                    )
                    for name in FEATURE_NAMES
                ),
            ),
        )
        cls.discovery_window = EvidenceWindow(
            "2026-09-02T00:00:00Z",
            "2026-09-10T23:59:59Z",
            180,
        )
        cls.selection_window = EvidenceWindow(
            "2026-09-11T00:00:00Z",
            "2026-09-20T23:59:59Z",
            160,
        )
        cls.confirmation_window = EvidenceWindow(
            "2026-09-21T00:00:00Z",
            "2026-10-10T23:59:59Z",
            220,
        )

    @classmethod
    def _signed_record(cls, observation, event_id, observed_at):
        return cls.authority.sign(
            event_id=event_id,
            observation=observation,
            observed_at=observed_at,
            evidence_hash=hashlib.sha256(event_id.encode()).hexdigest(),
            verifier_id="mechanism-verifier",
            verified_at=observed_at,
        )

    @classmethod
    def _ledger(cls, observations, prefix, observed_at):
        return EventLedger.verify(
            tuple(
                cls._signed_record(
                    observation,
                    f"{prefix}-{index:05d}",
                    observed_at,
                )
                for index, observation in enumerate(observations)
            ),
            cls.authority,
        )

    def _evidence(self, *, nonlinear: bool = True):
        observations = self.nonlinear if nonlinear else self.target[420:980]
        return (
            self._ledger(
                observations[:180],
                "mechanism-discovery",
                "2026-09-05T00:00:00Z",
            ),
            self._ledger(
                observations[180:340],
                "mechanism-selection",
                "2026-09-15T00:00:00Z",
            ),
            self._ledger(
                observations[340:560],
                "mechanism-confirmation",
                "2026-10-01T00:00:00Z",
            ),
        )

    def _plan(self):
        return create_mechanism_comparison_plan(
            self.bundle,
            self.hypotheses,
            self.authority,
            verifier_id="mechanism-verifier",
            registered_at="2026-09-01T01:00:00Z",
            discovery_window=self.discovery_window,
            selection_window=self.selection_window,
            confirmation_window=self.confirmation_window,
        )

    def _report(self, *, nonlinear: bool = True):
        ledgers = self._evidence(nonlinear=nonlinear)
        report = compare_mechanisms(
            self.bundle,
            self._plan(),
            *ledgers,
            self.authority,
        )
        return report, ledgers

    def _bundle_with_validation(self, validation):
        manifest = replace(
            self.bundle.manifest,
            validation=validation,
            model_id=manifest_model_id(
                parent_model_id=self.bundle.manifest.parent_model_id,
                person_id=self.bundle.manifest.person_id,
                person_data_hash=self.bundle.manifest.person_data_hash,
                population_data_hash=(
                    self.bundle.manifest.population_data_hash
                ),
                feature_schema_digest=(
                    self.bundle.manifest.feature_schema_hash
                ),
                model_content_digest=(
                    self.bundle.manifest.model_content_hash
                ),
                validation_digest=validation.digest(),
                applicability_digest=(
                    self.bundle.manifest.applicability_profile.digest()
                ),
                applicability_event_ids=(
                    self.bundle.manifest.applicability_event_ids
                ),
                applicability_data_hash=(
                    self.bundle.manifest.applicability_data_hash
                ),
                lineage_trial_hashes=(
                    self.bundle.manifest.lineage_trial_hashes
                ),
                lineage_design_hashes=(
                    self.bundle.manifest.lineage_design_hashes
                ),
                experiment_plan_ids=(
                    self.bundle.manifest.experiment_plan_ids
                ),
                training_config=self.bundle.manifest.training_config,
                code_version=self.bundle.manifest.code_version,
            ),
        )
        return PersonModelBundle(
            population_model=self.bundle.population_model,
            representation=self.bundle.representation,
            adapter=self.bundle.adapter,
            manifest=manifest,
        )

    def _plan_for_bundle(self, bundle, *, config=None):
        return create_mechanism_comparison_plan(
            bundle,
            self.hypotheses,
            self.authority,
            verifier_id="mechanism-verifier",
            registered_at="2026-09-01T01:00:00Z",
            discovery_window=self.discovery_window,
            selection_window=self.selection_window,
            confirmation_window=self.confirmation_window,
            config=config,
        )

    def test_nonlinear_structure_is_selected_and_confirmed(self) -> None:
        plan = self._plan()
        report, ledgers = self._report()
        self.assertEqual(
            report.selected_hypothesis_id,
            "nonlinear-interactions",
        )
        self.assertEqual(report.status, "supported_candidate")
        self.assertGreater(report.confirmation_nll_uplift, 0.05)
        self.assertGreater(report.confirmation_nll_uplift_ci_lower, 0.0)
        self.assertEqual(
            report.interpretation,
            "predictive_structure_only",
        )
        verify_mechanism_report(
            self.bundle,
            plan,
            *ledgers,
            self.authority,
            report,
        )

    def test_null_data_does_not_support_a_candidate(self) -> None:
        report, _ledgers = self._report(nonlinear=False)
        self.assertEqual(report.status, "no_supported_candidate")
        self.assertIn(
            "confirmation_support_not_established",
            report.reasons,
        )

    def test_confirmation_configuration_has_hard_statistical_floors(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            MechanismComparisonConfig(confidence_z=0.001)
        with self.assertRaises(ValueError):
            MechanismComparisonConfig(
                minimum_confirmation_nll_uplift=0.0
            )
        with self.assertRaises(ValueError):
            MechanismComparisonConfig(minimum_samples_per_split=99)
        with self.assertRaises(ValueError):
            MechanismComparisonConfig(
                maximum_confirmation_calibration_error=1.0
            )
        with self.assertRaises(ValueError):
            MechanismComparisonConfig(maximum_report_age_days=181)

    def test_only_mechanism_only_validation_failure_is_repairable(
        self,
    ) -> None:
        base_validation = self.bundle.manifest.validation
        mechanism_failed = replace(
            base_validation,
            status="failed",
            mechanism_adequacy_passed=False,
            reasons=("mechanism_misspecification_suspected",),
        )
        repairable_bundle = self._bundle_with_validation(
            mechanism_failed
        )
        plan = self._plan_for_bundle(repairable_bundle)
        report = compare_mechanisms(
            repairable_bundle,
            plan,
            *self._evidence(),
            self.authority,
        )
        self.assertEqual(report.status, "supported_candidate")

        personalization_failed = replace(
            base_validation,
            status="failed",
            personalization_passed=False,
            reasons=("insufficient_personalization_uplift",),
        )
        blocked_bundle = self._bundle_with_validation(
            personalization_failed
        )
        with self.assertRaises(MechanismRefusedError) as blocked:
            self._plan_for_bundle(blocked_bundle)
        self.assertIn(
            "base_model_personalization_not_validated",
            blocked.exception.reasons,
        )

    def test_stable_misspecified_model_is_recovered_end_to_end(
        self,
    ) -> None:
        rng = np.random.default_rng(911)

        def nonlinear_choice(observation):
            features = np.asarray(
                observation.scenario.ordered_features(FEATURE_NAMES),
                dtype=np.float64,
            )
            base_logit = float(
                features @ np.asarray(self.person.true_weights)
            )
            correction = (
                2.8 * features[0] * features[3]
                - 2.4 * features[1] * features[4]
                + 1.8 * (abs(features[2]) - 0.7)
            )
            return replace(
                observation,
                actual_choice=int(
                    rng.random() < sigmoid(base_logit + correction)
                ),
            )

        training_records = []
        for index, record in enumerate(self.training_ledger.records):
            observation = record.observation
            if observation.person_id == self.person_id:
                observation = nonlinear_choice(observation)
            training_records.append(
                self._signed_record(
                    observation,
                    f"stable-misspecified-training-{index:05d}",
                    "2026-07-01T00:00:00Z",
                )
            )
        target = tuple(
            nonlinear_choice(observation)
            for observation in self.target
        )
        applicability = self._ledger(
            target[:200],
            "stable-misspecified-applicability",
            "2026-07-15T00:00:00Z",
        )
        validation = EventLedger.verify(
            tuple(
                self._signed_record(
                    observation,
                    f"stable-misspecified-validation-{index:05d}",
                    (
                        "2026-08-01T00:00:00Z"
                        if index < 110
                        else "2026-09-01T00:00:00Z"
                    ),
                )
                for index, observation in enumerate(target[200:420])
            ),
            self.authority,
        )
        bundle = fit_person_model(
            EventLedger.verify(
                tuple(training_records),
                self.authority,
            ),
            self.authority,
            applicability_ledger=applicability,
            validation_ledger=validation,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        self.assertEqual(bundle.manifest.validation.status, "failed")
        self.assertTrue(
            bundle.manifest.validation.personalization_passed
        )
        self.assertFalse(
            bundle.manifest.validation.mechanism_adequacy_passed
        )
        self.assertEqual(
            bundle.manifest.validation.reasons,
            ("mechanism_misspecification_suspected",),
        )

        evidence = (
            self._ledger(
                target[420:600],
                "stable-misspecified-discovery",
                "2026-09-05T00:00:00Z",
            ),
            self._ledger(
                target[600:760],
                "stable-misspecified-selection",
                "2026-09-15T00:00:00Z",
            ),
            self._ledger(
                target[760:980],
                "stable-misspecified-confirmation",
                "2026-10-01T00:00:00Z",
            ),
        )
        plan = self._plan_for_bundle(bundle)
        report = compare_mechanisms(
            bundle,
            plan,
            *evidence,
            self.authority,
        )
        self.assertEqual(
            report.selected_hypothesis_id,
            "nonlinear-interactions",
        )
        self.assertEqual(report.status, "supported_candidate")
        prediction = predict_with_mechanism(
            bundle,
            plan,
            report,
            *evidence,
            self.authority,
            target[990].scenario,
            prediction_at="2026-10-15T00:00:00Z",
        )
        self.assertEqual(
            prediction.selected_hypothesis_id,
            "nonlinear-interactions",
        )

    def test_confirmation_scores_the_deployed_predictive_kernel(
        self,
    ) -> None:
        report, (_discovery, _selection, confirmation) = self._report()
        fit = next(
            candidate
            for candidate in report.candidate_fits
            if candidate.hypothesis_id
            == report.selected_hypothesis_id
        )
        hypothesis = next(
            candidate
            for candidate in self.hypotheses
            if candidate.hypothesis_id
            == report.selected_hypothesis_id
        )
        records = tuple(
            sorted(
                confirmation.records,
                key=lambda record: (
                    record.observed_at,
                    record.event_id,
                ),
            )
        )
        base_features = np.asarray(
            [
                record.observation.scenario.ordered_features(
                    FEATURE_NAMES
                )
                for record in records
            ],
            dtype=np.float64,
        )
        raw_terms = np.asarray(
            [
                [
                    term.evaluate(
                        record.observation.scenario,
                        FEATURE_NAMES,
                    )
                    for term in hypothesis.terms
                ]
                for record in records
            ],
            dtype=np.float64,
        )
        terms = (
            raw_terms - np.asarray(fit.centers)
        ) / np.asarray(fit.scales)
        base_mean = base_features @ np.asarray(
            self.bundle.representation.latent_mean
        )
        base_covariance = np.asarray(
            self.bundle.representation.covariance
        )
        base_variance = np.einsum(
            "ij,jk,ik->i",
            base_features,
            base_covariance,
            base_features,
        )
        correction_covariance = np.asarray(fit.covariance)
        candidate_mean = (
            base_mean + terms @ np.asarray(fit.coefficients)
        )
        candidate_variance = base_variance + np.einsum(
            "ij,jk,ik->i",
            terms,
            correction_covariance,
            terms,
        )
        base_probability = sigmoid(
            base_mean / np.sqrt(1.0 + np.pi * base_variance / 8.0)
        )
        candidate_probability = sigmoid(
            candidate_mean
            / np.sqrt(1.0 + np.pi * candidate_variance / 8.0)
        )
        choices = np.asarray(
            [
                record.observation.actual_choice
                for record in records
            ],
            dtype=np.float64,
        )

        def nll(probability):
            clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
            return float(
                np.mean(
                    -choices * np.log(clipped)
                    - (1.0 - choices) * np.log(1.0 - clipped)
                )
            )

        self.assertAlmostEqual(
            report.confirmation_base_nll,
            nll(base_probability),
            places=12,
        )
        self.assertAlmostEqual(
            report.confirmation_nll,
            nll(candidate_probability),
            places=12,
        )

    def test_tied_timestamp_uncertainty_is_event_id_invariant(
        self,
    ) -> None:
        discovery, selection, confirmation = self._evidence()
        original = compare_mechanisms(
            self.bundle,
            self._plan(),
            discovery,
            selection,
            confirmation,
            self.authority,
        )
        relabelled = EventLedger.verify(
            tuple(
                self._signed_record(
                    record.observation,
                    f"permuted-confirmation-{len(confirmation.records)-index:05d}",
                    record.observed_at,
                )
                for index, record in enumerate(confirmation.records)
            ),
            self.authority,
        )
        changed = compare_mechanisms(
            self.bundle,
            self._plan(),
            discovery,
            selection,
            relabelled,
            self.authority,
        )
        self.assertEqual(
            original.confirmation_standard_error_method,
            "iid",
        )
        self.assertAlmostEqual(
            original.confirmation_nll_uplift_standard_error,
            changed.confirmation_nll_uplift_standard_error,
            places=12,
        )
        self.assertEqual(original.status, changed.status)

    def test_plan_and_report_round_trip_with_signatures(self) -> None:
        plan = self._plan()
        report, _ledgers = self._report()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "mechanism-plan.json"
            report_path = root / "mechanism-report.json"
            save_mechanism_comparison_plan(plan_path, plan)
            save_mechanism_comparison_report(report_path, report)
            restored_plan = load_mechanism_comparison_plan(
                plan_path,
                self.authority,
            )
            restored_report = load_mechanism_comparison_report(
                report_path,
                self.authority,
            )
        self.assertEqual(restored_plan, plan)
        self.assertEqual(restored_report, report)

        tampered = copy.deepcopy(plan.to_dict())
        tampered["hypotheses"][0]["terms"][0]["kind"] = "quadratic"
        with self.assertRaises(ValueError):
            mechanism_plan_from_dict(
                tampered,
                self.authority,
            )

        resigned = copy.deepcopy(report.to_dict())
        resigned["candidate_fits"][0]["coefficients"][0] += 0.1
        resigned_content = {
            key: value
            for key, value in resigned.items()
            if key not in {"report_id", "signature"}
        }
        resigned["report_id"] = hashlib.sha256(
            json.dumps(
                resigned_content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        resigned_payload = {
            key: value
            for key, value in resigned.items()
            if key != "signature"
        }
        resigned["signature"] = self.authority.sign_payload(
            resigned_payload,
            "mechanism-verifier",
        )
        altered_report = mechanism_report_from_dict(
            resigned,
            self.authority,
        )
        with self.assertRaises(MechanismRefusedError) as derivation:
            verify_mechanism_report(
                self.bundle,
                plan,
                *self._evidence(),
                self.authority,
                altered_report,
            )
        self.assertIn(
            "mechanism_report_derivation_mismatch",
            derivation.exception.reasons,
        )

    def test_evidence_must_be_preregistered_disjoint_and_new(self) -> None:
        plan = self._plan()
        discovery, selection, confirmation = self._evidence()
        with self.assertRaises(MechanismRefusedError) as overlap:
            compare_mechanisms(
                self.bundle,
                plan,
                discovery,
                discovery,
                confirmation,
                self.authority,
            )
        self.assertTrue(
            {
                "selection_event_count_mismatch",
                "mechanism_evidence_overlap",
                "mechanism_evidence_time_order_invalid",
            }
            & set(overlap.exception.reasons)
        )

        old = self.training_ledger.records_for_person(self.person_id)[0]
        replayed_observation = replace(
            old.observation,
            scenario=replace(
                old.observation.scenario,
                scenario_id="relabelled-old-mechanism-design",
            ),
        )
        replayed = self._signed_record(
            replayed_observation,
            "relabelled-old-mechanism-event",
            "2026-09-05T00:00:00Z",
        )
        attacked = EventLedger.verify(
            (replayed,) + discovery.records[1:],
            self.authority,
        )
        with self.assertRaises(MechanismRefusedError) as reused:
            compare_mechanisms(
                self.bundle,
                plan,
                attacked,
                selection,
                confirmation,
                self.authority,
            )
        self.assertIn(
            "mechanism_evidence_reuses_base_design",
            reused.exception.reasons,
        )

    def test_confirmation_outcomes_cannot_change_selection(self) -> None:
        plan = self._plan()
        discovery, selection, confirmation = self._evidence()
        original = compare_mechanisms(
            self.bundle,
            plan,
            discovery,
            selection,
            confirmation,
            self.authority,
        )
        flipped = EventLedger.verify(
            tuple(
                self._signed_record(
                    replace(
                        record.observation,
                        actual_choice=1 - record.observation.actual_choice,
                    ),
                    f"flipped-{index:05d}",
                    record.observed_at,
                )
                for index, record in enumerate(confirmation.records)
            ),
            self.authority,
        )
        changed = compare_mechanisms(
            self.bundle,
            plan,
            discovery,
            selection,
            flipped,
            self.authority,
        )
        self.assertEqual(
            changed.selected_hypothesis_id,
            original.selected_hypothesis_id,
        )
        self.assertNotEqual(
            changed.confirmation_nll,
            original.confirmation_nll,
        )

    def test_evidence_order_does_not_change_report(self) -> None:
        plan = self._plan()
        discovery, selection, confirmation = self._evidence()
        original = compare_mechanisms(
            self.bundle,
            plan,
            discovery,
            selection,
            confirmation,
            self.authority,
        )
        reordered = compare_mechanisms(
            self.bundle,
            plan,
            EventLedger(tuple(reversed(discovery.records))),
            EventLedger(tuple(reversed(selection.records))),
            EventLedger(tuple(reversed(confirmation.records))),
            self.authority,
        )
        self.assertEqual(reordered, original)

    def test_ood_evidence_and_unknown_terms_are_refused(self) -> None:
        unknown = MechanismHypothesis(
            hypothesis_id="unknown-feature",
            terms=(
                MechanismTerm(
                    "missing",
                    "linear",
                    ("not_in_model",),
                ),
            ),
        )
        with self.assertRaises(MechanismRefusedError) as bad_plan:
            create_mechanism_comparison_plan(
                self.bundle,
                (unknown, self.hypotheses[1]),
                self.authority,
                verifier_id="mechanism-verifier",
                registered_at="2026-09-01T01:00:00Z",
                discovery_window=self.discovery_window,
                selection_window=self.selection_window,
                confirmation_window=self.confirmation_window,
            )
        self.assertIn(
            "mechanism_term_feature_unknown",
            bad_plan.exception.reasons,
        )

        discovery, selection, confirmation = self._evidence()
        first = discovery.records[0]
        remote_scenario = replace(
            first.observation.scenario,
            scenario_id="remote-mechanism-evidence",
            features=tuple(
                value + 30.0
                for value in first.observation.scenario.features
            ),
        )
        remote_record = self._signed_record(
            replace(first.observation, scenario=remote_scenario),
            "remote-mechanism-event",
            first.observed_at,
        )
        attacked = EventLedger.verify(
            (remote_record,) + discovery.records[1:],
            self.authority,
        )
        with self.assertRaises(MechanismRefusedError) as ood:
            compare_mechanisms(
                self.bundle,
                self._plan(),
                attacked,
                selection,
                confirmation,
                self.authority,
            )
        self.assertIn(
            "mechanism_evidence_outside_applicability",
            ood.exception.reasons,
        )

    def test_prediction_recomputes_report_and_improves_true_probability(
        self,
    ) -> None:
        plan = self._plan()
        report, ledgers = self._report()
        scenario = self.target[990].scenario
        prediction = predict_with_mechanism(
            self.bundle,
            plan,
            report,
            *ledgers,
            self.authority,
            scenario,
            prediction_at="2026-10-15T00:00:00Z",
        )
        features = np.asarray(
            scenario.ordered_features(FEATURE_NAMES),
            dtype=np.float64,
        )
        true_probability = float(
            sigmoid(
                features @ np.asarray(self.person.true_weights)
                + 2.8 * features[0] * features[3]
                - 2.4 * features[1] * features[4]
                + 1.8 * (abs(features[2]) - 0.7)
            )
        )
        self.assertEqual(
            prediction.selected_hypothesis_id,
            "nonlinear-interactions",
        )
        self.assertLess(
            abs(prediction.probability_option_1 - true_probability),
            abs(
                prediction.base_probability_option_1
                - true_probability
            ),
        )
        self.assertEqual(
            prediction.model_form_uncertainty_status,
            "candidate_selection_not_quantified",
        )

        null_report, null_ledgers = self._report(nonlinear=False)
        with self.assertRaises(MechanismRefusedError) as refused:
            predict_with_mechanism(
                self.bundle,
                plan,
                null_report,
                *null_ledgers,
                self.authority,
                scenario,
                prediction_at="2026-10-15T00:00:00Z",
            )
        self.assertIn(
            "mechanism_candidate_not_supported",
            refused.exception.reasons,
        )

    def test_prediction_refuses_unvalidated_mechanism_metadata_scope(
        self,
    ) -> None:
        plan = self._plan()
        report, ledgers = self._report()
        transferred = replace(
            self.target[990].scenario,
            scenario_id="mechanism-cross-domain-transfer",
            domain="unobserved-mechanism-domain",
        )
        with self.assertRaises(MechanismRefusedError) as refused:
            predict_with_mechanism(
                self.bundle,
                plan,
                report,
                *ledgers,
                self.authority,
                transferred,
                prediction_at="2026-10-15T00:00:00Z",
            )
        self.assertIn(
            "mechanism_transfer_unvalidated",
            refused.exception.reasons,
        )

    def test_prediction_refuses_expired_mechanism_report(self) -> None:
        plan = self._plan()
        report, ledgers = self._report()
        with self.assertRaises(MechanismRefusedError) as refused:
            predict_with_mechanism(
                self.bundle,
                plan,
                report,
                *ledgers,
                self.authority,
                self.target[990].scenario,
                prediction_at="2027-05-01T00:00:00Z",
            )
        self.assertIn(
            "mechanism_report_expired",
            refused.exception.reasons,
        )

    def test_registry_exposes_mechanism_distiller(self) -> None:
        from pcfm.demo import run_demo

        report = run_demo(
            seed=603,
            person_count=12,
            source_trials=80,
            target_trials=80,
        )
        mechanism = next(
            item
            for item in report["module_slots"]
            if item["slot"] == ModuleSlot.MECHANISM_DISTILLER.value
        )
        self.assertEqual(mechanism["status"], "implemented")
        self.assertEqual(
            mechanism["module_version"],
            "preregistered-mechanism-comparison-v2",
        )

    def test_cli_plans_compares_and_predicts(self) -> None:
        discovery, selection, confirmation = self._evidence()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.json"
            hypotheses_path = root / "hypotheses.json"
            keys_path = root / "keys.json"
            plan_path = root / "mechanism-plan.json"
            report_path = root / "mechanism-report.json"
            discovery_path = root / "discovery.jsonl"
            selection_path = root / "selection.jsonl"
            confirmation_path = root / "confirmation.jsonl"
            scenario_path = root / "scenario.json"
            save_bundle(model_path, self.bundle)
            hypotheses_path.write_text(
                json.dumps(
                    [
                        hypothesis.to_dict()
                        for hypothesis in self.hypotheses
                    ]
                ),
                encoding="utf-8",
            )
            keys_path.write_text(
                json.dumps(
                    {
                        "mechanism-verifier": (
                            "stage-six-mechanism-secret"
                        )
                    }
                ),
                encoding="utf-8",
            )
            save_event_ledger_jsonl(discovery_path, discovery)
            save_event_ledger_jsonl(selection_path, selection)
            save_event_ledger_jsonl(
                confirmation_path,
                confirmation,
            )
            scenario = self.target[990].scenario
            scenario_path.write_text(
                json.dumps(
                    {
                        "scenario_id": scenario.scenario_id,
                        "features": {
                            name: value
                            for name, value in zip(
                                scenario.feature_names,
                                scenario.features,
                                strict=True,
                            )
                        },
                        "options": list(scenario.options),
                        "domain": scenario.domain,
                        "context": dict(scenario.context),
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            planned = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "plan-mechanisms",
                    "--model",
                    str(model_path),
                    "--hypotheses",
                    str(hypotheses_path),
                    "--verification-keys",
                    str(keys_path),
                    "--verifier-id",
                    "mechanism-verifier",
                    "--registered-at",
                    "2026-09-01T01:00:00Z",
                    "--discovery-start-at",
                    self.discovery_window.start_at,
                    "--discovery-end-at",
                    self.discovery_window.end_at,
                    "--discovery-event-count",
                    "180",
                    "--selection-start-at",
                    self.selection_window.start_at,
                    "--selection-end-at",
                    self.selection_window.end_at,
                    "--selection-event-count",
                    "160",
                    "--confirmation-start-at",
                    self.confirmation_window.start_at,
                    "--confirmation-end-at",
                    self.confirmation_window.end_at,
                    "--confirmation-event-count",
                    "220",
                    "--output",
                    str(plan_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                planned.returncode,
                0,
                msg=planned.stderr + planned.stdout,
            )
            compared = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "compare-mechanisms",
                    "--model",
                    str(model_path),
                    "--plan",
                    str(plan_path),
                    "--discovery-ledger",
                    str(discovery_path),
                    "--selection-ledger",
                    str(selection_path),
                    "--confirmation-ledger",
                    str(confirmation_path),
                    "--verification-keys",
                    str(keys_path),
                    "--output",
                    str(report_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                compared.returncode,
                0,
                msg=compared.stderr + compared.stdout,
            )
            comparison_payload = json.loads(compared.stdout)
            self.assertEqual(
                comparison_payload["status"],
                "supported_candidate",
            )
            predicted = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "predict-mechanism",
                    "--model",
                    str(model_path),
                    "--plan",
                    str(plan_path),
                    "--report",
                    str(report_path),
                    "--discovery-ledger",
                    str(discovery_path),
                    "--selection-ledger",
                    str(selection_path),
                    "--confirmation-ledger",
                    str(confirmation_path),
                    "--verification-keys",
                    str(keys_path),
                    "--scenario",
                    str(scenario_path),
                    "--prediction-at",
                    "2026-10-15T00:00:00Z",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                predicted.returncode,
                0,
                msg=predicted.stderr + predicted.stdout,
            )
            prediction_payload = json.loads(predicted.stdout)
            self.assertEqual(
                prediction_payload["selected_hypothesis_id"],
                "nonlinear-interactions",
            )
            self.assertEqual(
                prediction_payload["interpretation"],
                "predictive_structure_only",
            )


if __name__ == "__main__":
    unittest.main()
