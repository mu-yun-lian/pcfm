from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
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

from pcfm.active_experiment import (
    ActiveExperimentConfig,
    ActiveExperimentRefusedError,
    active_experiment_plan_from_dict,
    apply_active_experiment_results,
    create_active_experiment_plan,
    create_next_active_experiment_plan,
    gaussian_binary_information,
    load_active_experiment_plan,
    save_active_experiment_plan,
    verify_active_experiment_plan,
    verify_active_experiment_results,
)
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.math_utils import fit_map_logistic, sigmoid
from pcfm.registry import ModuleSlot
from pcfm.storage import load_bundle, save_bundle
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset
from pcfm.workflow import fit_person_model, save_event_ledger_jsonl


class Stage5ActiveExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        people, source, target = generate_population_dataset(
            seed=501,
            person_count=24,
            source_trials=140,
            target_trials=900,
            heterogeneity_scale=1.5,
        )
        cls.person_id = people[0].person_id
        cls.target = target[cls.person_id]
        cls.authority = VerificationAuthority(
            {"experiment-verifier": b"stage-five-experiment-secret"}
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
            cls.target[:220],
            "applicability",
            "2026-07-15T00:00:00Z",
        )
        cls.validation_ledger = cls._validation_ledger(
            cls.target[220:440]
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
            raise AssertionError("stage-five fixture requires validation")
        cls.created_at = "2026-09-02T00:00:00Z"
        cls.candidates = cls._applicable_candidates(120)

    @classmethod
    def _ledger(
        cls,
        observations,
        prefix: str,
        observed_at: str,
    ) -> EventLedger:
        records = []
        for index, observation in enumerate(observations):
            event_id = f"{prefix}-{index:05d}"
            records.append(
                cls.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="experiment-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), cls.authority)

    @classmethod
    def _validation_ledger(cls, observations) -> EventLedger:
        records = []
        midpoint = len(observations) // 2
        for index, observation in enumerate(observations):
            observed_at = (
                "2026-08-01T00:00:00Z"
                if index < midpoint
                else "2026-09-01T00:00:00Z"
            )
            event_id = f"validation-{index:05d}"
            records.append(
                cls.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="experiment-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), cls.authority)

    @classmethod
    def _applicable_candidates(cls, count: int):
        candidates = []
        for observation in cls.target[440:]:
            assessment = cls.bundle.manifest.applicability_profile.assess(
                observation.scenario,
                prediction_at=cls.created_at,
            )
            if assessment.status == "in_distribution":
                candidates.append(observation.scenario)
            if len(candidates) == count:
                return tuple(candidates)
        raise AssertionError("not enough applicable active candidates")

    @classmethod
    def _future_validation(cls) -> EventLedger:
        records = []
        observations = cls.target[700:900]
        midpoint = len(observations) // 2
        for index, observation in enumerate(observations):
            observed_at = (
                "2026-10-01T00:00:00Z"
                if index < midpoint
                else "2026-11-01T00:00:00Z"
            )
            event_id = f"future-validation-{index:05d}"
            records.append(
                cls.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="experiment-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), cls.authority)

    def _plan(
        self,
        candidates=None,
        *,
        selection_count: int = 12,
        config: ActiveExperimentConfig | None = None,
    ):
        return create_active_experiment_plan(
            self.bundle,
            candidates or self.candidates,
            self.authority,
            verifier_id="experiment-verifier",
            created_at=self.created_at,
            selection_count=selection_count,
            config=config or ActiveExperimentConfig(),
        )

    def _result_ledger(self, plan) -> EventLedger:
        by_scenario = {
            observation.scenario.scenario_id: observation
            for observation in self.target
        }
        start = datetime.fromisoformat(
            plan.created_at.replace("Z", "+00:00")
        ) + timedelta(hours=1)
        records = []
        for index, selection in enumerate(plan.selections):
            source = by_scenario[selection.scenario.scenario_id]
            observation = replace(
                source,
                scenario=selection.scenario,
            )
            observed_at = (
                start + timedelta(hours=index)
            ).isoformat().replace("+00:00", "Z")
            event_id = f"active-result-{index:04d}"
            records.append(
                self.authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=hashlib.sha256(
                        event_id.encode()
                    ).hexdigest(),
                    verifier_id="experiment-verifier",
                    verified_at=observed_at,
                )
            )
        return EventLedger.verify(tuple(records), self.authority)

    def test_selection_is_deterministic_and_order_invariant(self) -> None:
        first = self._plan()
        second = self._plan(tuple(reversed(self.candidates)))
        self.assertEqual(first, second)
        self.assertEqual(len(first.selections), 12)
        self.assertEqual(
            tuple(item.rank for item in first.selections),
            tuple(range(1, 13)),
        )
        self.assertTrue(
            all(
                item.expected_information_gain > 0
                for item in first.selections
            )
        )
        self.assertGreater(first.total_expected_information_gain, 0)
        self.assertEqual(
            first.selection_mode,
            "outcome_blind_batch_approximation",
        )

    def test_optimizer_exceeds_random_on_declared_objective(self) -> None:
        active = self._plan(selection_count=10)
        rng = np.random.default_rng(502)
        random_gains = []
        for _ in range(40):
            indices = rng.choice(
                len(self.candidates),
                size=10,
                replace=False,
            )
            subset = tuple(self.candidates[int(index)] for index in indices)
            random_gains.append(
                self._plan(
                    subset,
                    selection_count=len(subset),
                ).total_expected_information_gain
            )
        self.assertGreater(
            active.total_expected_information_gain,
            float(np.mean(random_gains)),
        )

    def test_active_selection_improves_external_heldout_nll(self) -> None:
        active = self._plan(selection_count=10)
        prior_mean = np.asarray(
            self.bundle.representation.latent_mean,
            dtype=np.float64,
        )
        prior_covariance = np.asarray(
            self.bundle.representation.covariance,
            dtype=np.float64,
        )
        prior_precision = np.linalg.inv(prior_covariance)
        observations = {
            observation.scenario.scenario_id: observation
            for observation in self.target
        }
        holdout = self.target[700:900]
        holdout_features = np.asarray(
            [
                observation.scenario.ordered_features(
                    self.bundle.representation.feature_names
                )
                for observation in holdout
            ],
            dtype=np.float64,
        )
        holdout_choices = np.asarray(
            [observation.actual_choice for observation in holdout],
            dtype=np.float64,
        )

        def heldout_nll(scenarios) -> float:
            features = np.asarray(
                [
                    scenario.ordered_features(
                        self.bundle.representation.feature_names
                    )
                    for scenario in scenarios
                ],
                dtype=np.float64,
            )
            choices = np.asarray(
                [
                    observations[
                        scenario.scenario_id
                    ].actual_choice
                    for scenario in scenarios
                ],
                dtype=np.float64,
            )
            weights, _covariance = fit_map_logistic(
                features,
                choices,
                prior_mean,
                prior_precision,
            )
            probabilities = np.clip(
                sigmoid(holdout_features @ weights),
                1e-15,
                1.0 - 1e-15,
            )
            return float(
                np.mean(
                    -(
                        holdout_choices * np.log(probabilities)
                        + (1.0 - holdout_choices)
                        * np.log(1.0 - probabilities)
                    )
                )
            )

        active_nll = heldout_nll(
            tuple(item.scenario for item in active.selections)
        )
        rng = np.random.default_rng(502)
        random_nlls = []
        for _ in range(100):
            indices = rng.choice(
                len(self.candidates),
                size=10,
                replace=False,
            )
            random_nlls.append(
                heldout_nll(
                    tuple(
                        self.candidates[int(index)]
                        for index in indices
                    )
                )
            )
        self.assertLess(active_nll, float(np.mean(random_nlls)))
        self.assertGreater(
            float(np.mean(active_nll < np.asarray(random_nlls))),
            0.8,
        )

    def test_single_step_plan_is_explicitly_adaptive(self) -> None:
        first = create_next_active_experiment_plan(
            self.bundle,
            self.candidates,
            self.authority,
            verifier_id="experiment-verifier",
            created_at=self.created_at,
        )
        self.assertEqual(first.selection_count, 1)
        self.assertEqual(first.selection_mode, "adaptive_single_step")
        update = apply_active_experiment_results(
            self.bundle,
            self.training_ledger,
            self.applicability_ledger,
            self._future_validation(),
            self.candidates,
            first,
            self._result_ledger(first),
            self.authority,
        )
        selected_id = first.selections[0].scenario.scenario_id
        remaining = tuple(
            scenario
            for scenario in self.candidates
            if scenario.scenario_id != selected_id
        )
        second = create_next_active_experiment_plan(
            update.bundle,
            remaining,
            self.authority,
            verifier_id="experiment-verifier",
            created_at="2026-12-01T00:00:00Z",
        )
        self.assertEqual(
            second.base_model_id,
            update.bundle.manifest.model_id,
        )
        self.assertNotEqual(
            second.selections[0].scenario.scenario_id,
            selected_id,
        )

    def test_mutual_information_handles_a_broad_posterior(self) -> None:
        narrow = gaussian_binary_information(
            logit_mean=0.8,
            logit_variance=1.0,
            quadrature_points=48,
        )
        broad = gaussian_binary_information(
            logit_mean=8.0,
            logit_variance=100.0,
            quadrature_points=48,
        )
        self.assertGreater(
            broad.mutual_information,
            narrow.mutual_information,
        )
        self.assertAlmostEqual(
            broad.mutual_information,
            0.427,
            delta=0.015,
        )

    def test_plan_round_trip_signature_and_derivation(self) -> None:
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment-plan.json"
            save_active_experiment_plan(path, plan)
            restored = load_active_experiment_plan(
                path,
                self.authority,
            )
        self.assertEqual(restored, plan)
        verify_active_experiment_plan(
            self.bundle,
            self.candidates,
            self.authority,
            restored,
        )

    def test_result_ledger_must_exactly_execute_signed_plan(self) -> None:
        plan = self._plan(selection_count=8)
        ledger = self._result_ledger(plan)
        verified = verify_active_experiment_results(
            plan,
            ledger,
            self.authority,
        )
        self.assertEqual(verified, ledger)

        first = ledger.records[0]
        changed_scenario = replace(
            first.observation.scenario,
            features=tuple(
                value + 0.01
                for value in first.observation.scenario.features
            ),
        )
        changed_observation = replace(
            first.observation,
            scenario=changed_scenario,
        )
        changed_record = self.authority.sign(
            event_id=first.event_id,
            observation=changed_observation,
            observed_at=first.observed_at,
            evidence_hash=first.evidence_hash,
            verifier_id=first.verifier_id,
            verified_at=first.verified_at,
        )
        substituted = EventLedger.verify(
            (changed_record,) + ledger.records[1:],
            self.authority,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as changed:
            verify_active_experiment_results(
                plan,
                substituted,
                self.authority,
            )
        self.assertIn(
            "experiment_result_scenario_mismatch",
            changed.exception.reasons,
        )

        with self.assertRaises(ActiveExperimentRefusedError) as missing:
            verify_active_experiment_results(
                plan,
                EventLedger.verify(
                    ledger.records[:-1],
                    self.authority,
                ),
                self.authority,
            )
        self.assertIn(
            "experiment_result_count_mismatch",
            missing.exception.reasons,
        )

    def test_results_must_be_collected_after_plan(self) -> None:
        plan = self._plan(selection_count=4)
        ledger = self._result_ledger(plan)
        first = ledger.records[0]
        earlier = "2026-09-01T23:00:00Z"
        backdated = self.authority.sign(
            event_id=first.event_id,
            observation=first.observation,
            observed_at=earlier,
            evidence_hash=first.evidence_hash,
            verifier_id=first.verifier_id,
            verified_at=earlier,
        )
        changed = EventLedger.verify(
            (backdated,) + ledger.records[1:],
            self.authority,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as raised:
            verify_active_experiment_results(
                plan,
                changed,
                self.authority,
            )
        self.assertIn(
            "experiment_result_precedes_plan",
            raised.exception.reasons,
        )

    def test_recomputed_hash_cannot_forge_plan(self) -> None:
        plan = self._plan()
        tampered = copy.deepcopy(plan.to_dict())
        tampered["candidate_count"] += 1
        content = {
            key: value
            for key, value in tampered.items()
            if key not in {"plan_id", "signature"}
        }
        tampered["plan_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "signature"):
            active_experiment_plan_from_dict(
                tampered,
                self.authority,
            )

    def test_resigned_altered_plan_fails_candidate_recomputation(
        self,
    ) -> None:
        plan = self._plan()
        tampered = copy.deepcopy(plan.to_dict())
        tampered["candidate_pool_hash"] = "f" * 64
        content = {
            key: value
            for key, value in tampered.items()
            if key not in {"plan_id", "signature"}
        }
        tampered["plan_id"] = hashlib.sha256(
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        signed_payload = {
            key: value
            for key, value in tampered.items()
            if key != "signature"
        }
        tampered["signature"] = self.authority.sign_payload(
            signed_payload,
            "experiment-verifier",
        )
        forged = active_experiment_plan_from_dict(
            tampered,
            self.authority,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as raised:
            verify_active_experiment_plan(
                self.bundle,
                self.candidates,
                self.authority,
                forged,
            )
        self.assertIn(
            "active_experiment_derivation_mismatch",
            raised.exception.reasons,
        )

    def test_used_model_scenario_is_refused(self) -> None:
        used = self.applicability_ledger.records[0].observation.scenario
        with self.assertRaises(ActiveExperimentRefusedError) as raised:
            self._plan((used,) + self.candidates[:20])
        self.assertIn(
            "candidate_reuses_model_scenario",
            raised.exception.reasons,
        )
        relabelled = replace(
            used,
            scenario_id="relabelled-used-scenario",
        )
        with self.assertRaises(ActiveExperimentRefusedError) as content:
            self._plan((relabelled,) + self.candidates[:20])
        self.assertIn(
            "candidate_reuses_model_design",
            content.exception.reasons,
        )

    def test_ood_and_cross_domain_candidates_are_refused(self) -> None:
        original = self.candidates[0]
        remote = replace(
            original,
            scenario_id="active-remote",
            features=tuple(value + 30 for value in original.features),
        )
        with self.assertRaises(ActiveExperimentRefusedError) as ood:
            self._plan((remote,) + self.candidates[:20])
        self.assertIn("candidate_outside_applicability", ood.exception.reasons)

        cross_domain = replace(
            original,
            scenario_id="active-cross-domain",
            domain="unknown-domain",
        )
        with self.assertRaises(ActiveExperimentRefusedError) as cross:
            self._plan((cross_domain,) + self.candidates[:20])
        self.assertIn(
            "candidate_cross_domain_unvalidated",
            cross.exception.reasons,
        )

    def test_unvalidated_base_and_non_scenario_input_are_refused(
        self,
    ) -> None:
        unvalidated = fit_person_model(
            self.training_ledger,
            self.authority,
            person_id=self.person_id,
            feature_names=FEATURE_NAMES,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as invalid:
            create_active_experiment_plan(
                unvalidated,
                self.candidates,
                self.authority,
                verifier_id="experiment-verifier",
                created_at=self.created_at,
                selection_count=10,
            )
        self.assertIn(
            "base_model_validation_unvalidated",
            invalid.exception.reasons,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as wrong_type:
            self._plan((self.target[500],))
        self.assertIn(
            "candidate_not_scenario",
            wrong_type.exception.reasons,
        )

    def test_selection_count_and_candidate_ids_are_validated(self) -> None:
        with self.assertRaises(ActiveExperimentRefusedError) as too_many:
            self._plan(
                self.candidates[:5],
                selection_count=6,
            )
        self.assertIn(
            "selection_count_exceeds_candidates",
            too_many.exception.reasons,
        )
        duplicate = replace(
            self.candidates[1],
            scenario_id=self.candidates[0].scenario_id,
        )
        with self.assertRaises(ActiveExperimentRefusedError) as repeated:
            self._plan((self.candidates[0], duplicate))
        self.assertIn(
            "candidate_scenario_ids_not_unique",
            repeated.exception.reasons,
        )
        relabelled = replace(
            self.candidates[0],
            scenario_id="same-design-new-id",
        )
        with self.assertRaises(ActiveExperimentRefusedError) as same_design:
            self._plan((self.candidates[0], relabelled))
        self.assertIn(
            "candidate_designs_not_unique",
            same_design.exception.reasons,
        )
        first = replace(
            self.candidates[0],
            scenario_id="same-design-metadata-a",
            context={
                **dict(self.candidates[0].context),
                "prediction_at": "2026-09-03T00:00:00Z",
            },
        )
        second = replace(
            self.candidates[0],
            scenario_id="same-design-metadata-b",
            context={
                **dict(self.candidates[0].context),
                "prediction_at": "2026-09-04T00:00:00Z",
            },
        )
        with self.assertRaises(
            ActiveExperimentRefusedError
        ) as metadata_only:
            self._plan((first, second))
        self.assertIn(
            "candidate_designs_not_unique",
            metadata_only.exception.reasons,
        )

    def test_programmatic_context_round_trips_canonically(self) -> None:
        scenario = replace(
            self.candidates[0],
            scenario_id="canonical-context",
            context={
                **dict(self.candidates[0].context),
                "prediction_at": 1,
            },
        )
        plan = self._plan((scenario,), selection_count=1)
        restored = active_experiment_plan_from_dict(
            plan.to_dict(),
            self.authority,
        )
        self.assertEqual(restored, plan)
        self.assertEqual(
            restored.selections[0].scenario.context["prediction_at"],
            "1",
        )

    def test_results_update_model_and_bind_plan_lineage(self) -> None:
        plan = self._plan(selection_count=8)
        result_ledger = self._result_ledger(plan)
        update = apply_active_experiment_results(
            self.bundle,
            self.training_ledger,
            self.applicability_ledger,
            self._future_validation(),
            self.candidates,
            plan,
            result_ledger,
            self.authority,
        )
        self.assertIn(
            plan.plan_id,
            update.bundle.manifest.experiment_plan_ids,
        )
        self.assertEqual(
            update.bundle.representation.observation_count,
            self.bundle.representation.observation_count + 8,
        )
        self.assertGreater(
            update.realized_covariance_entropy_reduction,
            0.0,
        )
        self.assertEqual(
            update.result_data_hash,
            EventLedger.snapshot_hash(result_ledger.records),
        )
        self.assertEqual(
            update.bundle.manifest.validation.status,
            "passed",
        )

    def test_update_requires_future_sealed_validation(self) -> None:
        plan = self._plan(selection_count=4)
        result_ledger = self._result_ledger(plan)
        with self.assertRaises(ActiveExperimentRefusedError) as raised:
            apply_active_experiment_results(
                self.bundle,
                self.training_ledger,
                self.applicability_ledger,
                self.validation_ledger,
                self.candidates,
                plan,
                result_ledger,
                self.authority,
            )
        self.assertIn(
            "validation_not_after_experiment_results",
            raised.exception.reasons,
        )

    def test_registry_and_cli_expose_active_experiment(self) -> None:
        from pcfm.demo import run_demo

        report = run_demo(
            seed=503,
            person_count=12,
            source_trials=80,
            target_trials=80,
        )
        active = next(
            item
            for item in report["module_slots"]
            if item["slot"] == ModuleSlot.ACTIVE_EXPERIMENT.value
        )
        self.assertEqual(active["status"], "implemented")
        self.assertEqual(
            active["module_version"],
            "gaussian-mutual-information-v2",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "person-model.json"
            candidates_path = root / "candidates.json"
            keys_path = root / "keys.json"
            output_path = root / "experiment-plan.json"
            results_path = root / "experiment-results.jsonl"
            training_path = root / "training.jsonl"
            applicability_path = root / "applicability.jsonl"
            future_validation_path = root / "future-validation.jsonl"
            updated_model_path = root / "updated-model.json"
            updated_ledger_path = root / "updated-training.jsonl"
            save_bundle(model_path, self.bundle)
            save_event_ledger_jsonl(
                training_path,
                self.training_ledger,
            )
            save_event_ledger_jsonl(
                applicability_path,
                self.applicability_ledger,
            )
            save_event_ledger_jsonl(
                future_validation_path,
                self._future_validation(),
            )
            candidates_path.write_text(
                json.dumps(
                    [
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
                        for scenario in self.candidates[:30]
                    ]
                ),
                encoding="utf-8",
            )
            keys_path.write_text(
                json.dumps(
                    {
                        "experiment-verifier": (
                            "stage-five-experiment-secret"
                        )
                    }
                ),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "plan-experiment",
                    "--model",
                    str(model_path),
                    "--candidates",
                    str(candidates_path),
                    "--verification-keys",
                    str(keys_path),
                    "--verifier-id",
                    "experiment-verifier",
                    "--created-at",
                    self.created_at,
                    "--selection-count",
                    "8",
                    "--output",
                    str(output_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr + completed.stdout,
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["selection_count"], 8)
            restored = load_active_experiment_plan(
                output_path,
                self.authority,
            )
            verify_active_experiment_plan(
                self.bundle,
                self.candidates[:30],
                self.authority,
                restored,
            )
            result_ledger = self._result_ledger(restored)
            save_event_ledger_jsonl(results_path, result_ledger)
            verified = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "verify-experiment",
                    "--model",
                    str(model_path),
                    "--candidates",
                    str(candidates_path),
                    "--plan",
                    str(output_path),
                    "--input",
                    str(results_path),
                    "--verification-keys",
                    str(keys_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                verified.returncode,
                0,
                msg=verified.stderr + verified.stdout,
            )
            verification_payload = json.loads(verified.stdout)
            self.assertEqual(
                verification_payload["status"],
                "verified",
            )
            self.assertEqual(
                verification_payload["result_event_count"],
                8,
            )
            applied = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "apply-experiment",
                    "--model",
                    str(model_path),
                    "--ledger",
                    str(training_path),
                    "--applicability-ledger",
                    str(applicability_path),
                    "--future-validation-ledger",
                    str(future_validation_path),
                    "--candidates",
                    str(candidates_path),
                    "--plan",
                    str(output_path),
                    "--input",
                    str(results_path),
                    "--verification-keys",
                    str(keys_path),
                    "--output",
                    str(updated_model_path),
                    "--output-ledger",
                    str(updated_ledger_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                applied.returncode,
                0,
                msg=applied.stderr + applied.stdout,
            )
            application_payload = json.loads(applied.stdout)
            self.assertEqual(
                application_payload["status"],
                "updated",
            )
            updated_bundle = load_bundle(updated_model_path)
            self.assertIn(
                restored.plan_id,
                updated_bundle.manifest.experiment_plan_ids,
            )
            self.assertTrue(updated_ledger_path.exists())


if __name__ == "__main__":
    unittest.main()
