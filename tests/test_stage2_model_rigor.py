from __future__ import annotations

import unittest

import numpy as np

from pcfm.core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    PopulationPriorEstimator,
)
from pcfm.demo import run_demo, run_misspecification_demo
from pcfm.math_utils import sigmoid
from pcfm.synthetic import FEATURE_NAMES, generate_population_dataset


class Stage2ModelRigorTests(unittest.TestCase):
    def test_estimated_person_variance_responds_to_population(self) -> None:
        low_people, low_source, _ = generate_population_dataset(
            seed=90,
            person_count=32,
            source_trials=120,
            target_trials=10,
            heterogeneity_scale=0.0,
        )
        high_people, high_source, _ = generate_population_dataset(
            seed=90,
            person_count=32,
            source_trials=120,
            target_trials=10,
            heterogeneity_scale=1.5,
        )
        low_observations = tuple(
            observation
            for person in low_people
            for observation in low_source[person.person_id]
        )
        high_observations = tuple(
            observation
            for person in high_people
            for observation in high_source[person.person_id]
        )
        low_prior = PopulationPriorEstimator(FEATURE_NAMES).fit(
            low_observations
        )
        high_prior = PopulationPriorEstimator(FEATURE_NAMES).fit(
            high_observations
        )
        low_variance = float(np.mean(np.diag(low_prior.covariance)))
        high_variance = float(np.mean(np.diag(high_prior.covariance)))
        self.assertLess(low_variance, 0.15)
        self.assertGreater(high_variance, low_variance + 0.2)

    def test_parameter_uncertainty_shrinks_with_more_person_data(self) -> None:
        people, source, target = generate_population_dataset(
            seed=91,
            person_count=20,
            source_trials=160,
            target_trials=5,
        )
        all_source = tuple(
            observation
            for person in people
            for observation in source[person.person_id]
        )
        person_id = people[0].person_id
        prior = PopulationPriorEstimator(FEATURE_NAMES).fit(all_source)
        encoder = MapPersonEncoder()
        generator = IdentityAdapterGenerator()
        few = encoder.fit(person_id, source[person_id][:15], prior)
        many = encoder.fit(person_id, source[person_id], prior)
        scenario = target[person_id][0].scenario
        integrator = DecisionIntegrator()
        few_prediction = integrator.predict(
            scenario,
            prior,
            generator.generate(few, prior),
            parameter_covariance=few.covariance,
        )
        many_prediction = integrator.predict(
            scenario,
            prior,
            generator.generate(many, prior),
            parameter_covariance=many.covariance,
        )
        few_width = (
            few_prediction.probability_upper_95
            - few_prediction.probability_lower_95
        )
        many_width = (
            many_prediction.probability_upper_95
            - many_prediction.probability_lower_95
        )
        self.assertGreater(few_prediction.logit_standard_deviation, 0.0)
        self.assertLess(many_width, few_width)

    def test_approximate_interval_has_reasonable_parameter_coverage(self) -> None:
        covered = []
        for seed in (200, 201, 202):
            people, source, target = generate_population_dataset(
                seed=seed,
                person_count=16,
                source_trials=100,
                target_trials=50,
            )
            all_source = tuple(
                observation
                for person in people
                for observation in source[person.person_id]
            )
            prior = PopulationPriorEstimator(FEATURE_NAMES).fit(all_source)
            encoder = MapPersonEncoder()
            generator = IdentityAdapterGenerator()
            integrator = DecisionIntegrator()
            for person in people:
                representation = encoder.fit(
                    person.person_id,
                    all_source,
                    prior,
                )
                adapter = generator.generate(representation, prior)
                for observation in target[person.person_id]:
                    prediction = integrator.predict(
                        observation.scenario,
                        prior,
                        adapter,
                        parameter_covariance=representation.covariance,
                    )
                    features = np.asarray(
                        observation.scenario.ordered_features(FEATURE_NAMES)
                    )
                    true_probability = float(
                        sigmoid(features @ np.asarray(person.true_weights))
                    )
                    covered.append(
                        prediction.probability_lower_95
                        <= true_probability
                        <= prediction.probability_upper_95
                    )
        coverage = sum(covered) / len(covered)
        self.assertGreater(coverage, 0.88)
        self.assertLess(coverage, 0.99)

    def test_in_family_personalization_passes_validity_gate(self) -> None:
        report = run_demo(
            seed=92,
            person_count=24,
            source_trials=120,
            target_trials=180,
        )
        self.assertTrue(report["validity_gate"]["passed"])

    def test_no_individuality_fails_personalization_gate(self) -> None:
        report = run_demo(
            seed=94,
            person_count=24,
            source_trials=120,
            target_trials=180,
            heterogeneity_scale=0.0,
        )
        self.assertFalse(report["validity_gate"]["passed"])
        self.assertIn(
            "insufficient_personalization_uplift",
            report["validity_gate"]["reasons"],
        )

    def test_misspecified_linear_model_fails_validity_gate(self) -> None:
        report = run_misspecification_demo(
            seed=93,
            person_count=24,
            source_trials=120,
            target_trials=180,
        )
        self.assertFalse(report["validity_gate"]["passed"])
        self.assertIn(
            "insufficient_personalization_uplift",
            report["validity_gate"]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
