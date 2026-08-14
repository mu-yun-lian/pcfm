from __future__ import annotations

from .core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    PopulationPriorEstimator,
)
from .evaluation import (
    aggregate_reports,
    assess_person_validation,
    evaluate_predictions,
    evaluate_probability_array,
    report_to_dict,
)
from .registry import ModuleRegistry, ModuleSlot
from .synthetic import (
    FEATURE_NAMES,
    generate_misspecified_dataset,
    generate_population_dataset,
)


def run_demo(
    *,
    seed: int = 42,
    person_count: int = 24,
    source_trials: int = 120,
    target_trials: int = 180,
    dataset_kind: str = "in_family",
    heterogeneity_scale: float = 1.0,
) -> dict[str, object]:
    if dataset_kind == "in_family":
        people, source, target = generate_population_dataset(
            seed=seed,
            person_count=person_count,
            source_trials=source_trials,
            target_trials=target_trials,
            heterogeneity_scale=heterogeneity_scale,
        )
    elif dataset_kind == "misspecified":
        people, source, target = generate_misspecified_dataset(
            seed=seed,
            person_count=person_count,
            source_trials=source_trials,
            target_trials=target_trials,
        )
    else:
        raise ValueError(f"unknown dataset_kind: {dataset_kind}")
    all_source = tuple(
        observation
        for person in people
        for observation in source[person.person_id]
    )
    prior = PopulationPriorEstimator(FEATURE_NAMES).fit(all_source)
    encoder = MapPersonEncoder()
    generator = IdentityAdapterGenerator()
    integrator = DecisionIntegrator()

    representations = {
        person.person_id: encoder.fit(person.person_id, all_source, prior)
        for person in people
    }
    adapters = {
        person_id: generator.generate(representation, prior)
        for person_id, representation in representations.items()
    }

    personal_reports = []
    population_reports = []
    swapped_reports = []
    person_validations = []
    for person in people:
        person_id = person.person_id
        observations = target[person_id]
        correct_adapter = adapters[person_id]
        correct_covariance = representations[person_id].covariance
        personal_probabilities = tuple(
            integrator.predict(
                observation.scenario,
                prior,
                correct_adapter,
                parameter_covariance=correct_covariance,
            ).probability_option_1
            for observation in observations
        )
        population_probabilities = tuple(
            integrator.predict_population(
                observation.scenario,
                prior,
            ).probability_option_1
            for observation in observations
        )
        personal_reports.append(
            evaluate_probability_array(
                observations,
                personal_probabilities,
            )
        )
        population_reports.append(
            evaluate_probability_array(
                observations,
                population_probabilities,
            )
        )
        person_validations.append(
            assess_person_validation(
                observations,
                personal_probabilities,
                population_probabilities,
                FEATURE_NAMES,
            )
        )
        for wrong_person in people:
            wrong_person_id = wrong_person.person_id
            if wrong_person_id == person_id:
                continue
            wrong_adapter = adapters[wrong_person_id]
            wrong_covariance = representations[wrong_person_id].covariance
            swapped_reports.append(
                evaluate_predictions(
                    observations,
                    lambda observation,
                    adapter=wrong_adapter,
                    covariance=wrong_covariance: integrator.predict(
                        observation.scenario,
                        prior,
                        adapter,
                        parameter_covariance=covariance,
                    ).probability_option_1,
                )
            )

    personal = aggregate_reports(personal_reports)
    population = aggregate_reports(population_reports)
    swapped = aggregate_reports(swapped_reports)
    minimum_personalization_pass_rate = 0.60
    minimum_mechanism_pass_rate = 0.80
    personalization_pass_count = sum(
        bool(item["personalization_passed"])
        for item in person_validations
    )
    mechanism_pass_count = sum(
        bool(item["mechanism_adequacy_passed"])
        for item in person_validations
    )
    personalization_pass_rate = (
        personalization_pass_count / len(person_validations)
    )
    mechanism_pass_rate = mechanism_pass_count / len(person_validations)
    personalization_passed = (
        personalization_pass_rate >= minimum_personalization_pass_rate
    )
    mechanism_passed = (
        mechanism_pass_rate >= minimum_mechanism_pass_rate
    )
    reasons = []
    if not personalization_passed:
        reasons.append("insufficient_personalization_uplift")
    if not mechanism_passed:
        reasons.append("mechanism_misspecification_suspected")
    validity_gate = {
        "passed": personalization_passed and mechanism_passed,
        "reasons": reasons,
        "personalization": {
            "passed": personalization_passed,
            "passed_person_count": personalization_pass_count,
            "total_person_count": len(person_validations),
            "pass_rate": personalization_pass_rate,
            "minimum_pass_rate": minimum_personalization_pass_rate,
            "person_results": [
                {
                    "person_id": person.person_id,
                    "passed": bool(result["personalization_passed"]),
                    "reasons": list(result["personalization_reasons"]),
                    "nll_uplift": float(result["nll_uplift"]),
                    "nll_uplift_ci_lower": float(
                        result["nll_uplift_ci_lower"]
                    ),
                    "nll_uplift_ci_upper": float(
                        result["nll_uplift_ci_upper"]
                    ),
                }
                for person, result in zip(
                    people,
                    person_validations,
                    strict=True,
                )
            ],
        },
        "mechanism_adequacy": {
            "passed": mechanism_passed,
            "passed_person_count": mechanism_pass_count,
            "total_person_count": len(person_validations),
            "pass_rate": mechanism_pass_rate,
            "minimum_pass_rate": minimum_mechanism_pass_rate,
            "person_results": [
                {
                    "person_id": person.person_id,
                    "passed": bool(
                        result["mechanism_adequacy_passed"]
                    ),
                    "reasons": list(result["mechanism_reasons"]),
                    "probe_nll_uplift": float(
                        result["mechanism_probe_nll_uplift"]
                    ),
                }
                for person, result in zip(
                    people,
                    person_validations,
                    strict=True,
                )
            ],
        },
    }
    registry = ModuleRegistry()
    registry.register_implementation(
        ModuleSlot.POPULATION_PRIOR,
        "hierarchical-logit-population-prior",
        prior.model_version,
    )
    registry.register_implementation(
        ModuleSlot.PERSON_ENCODER,
        "map-person-encoder",
        encoder.representation_version,
    )
    registry.register_implementation(
        ModuleSlot.ADAPTER_GENERATOR,
        "identity-adapter-generator",
        generator.adapter_version,
    )
    registry.register_implementation(
        ModuleSlot.DECISION_INTEGRATOR,
        "central-decision-integrator",
        integrator.model_version,
    )
    registry.register_implementation(
        ModuleSlot.EVALUATOR,
        "behavioral-evaluator",
        "behavioral-evaluator-v1",
    )
    registry.register_implementation(
        ModuleSlot.UPDATER,
        "signed-event-updater",
        "signed-event-updater-v2",
    )
    registry.register_implementation(
        ModuleSlot.UNCERTAINTY,
        "laplace-logit-uncertainty",
        "laplace-logit-uncertainty-v1",
    )
    registry.register_implementation(
        ModuleSlot.APPLICABILITY_GUARD,
        "hybrid-applicability-guard",
        "hybrid-applicability-v2",
    )
    registry.register_implementation(
        ModuleSlot.TEMPORAL_DRIFT,
        "two-sample-temporal-score-monitor",
        "two-sample-temporal-score-v2",
    )
    registry.register_implementation(
        ModuleSlot.ACTIVE_EXPERIMENT,
        "bayesian-active-experiment-planner",
        "gaussian-mutual-information-v2",
    )
    registry.register_implementation(
        ModuleSlot.DYNAMIC_STATE,
        "continuous-time-logit-state-tracker",
        "continuous-time-logit-state-v2",
    )
    registry.register_implementation(
        ModuleSlot.MECHANISM_DISTILLER,
        "preregistered-mechanism-distiller",
        "preregistered-mechanism-comparison-v2",
    )
    registry.register_implementation(
        ModuleSlot.COMPOSITE_MODEL,
        "validated-composite-predictive-view",
        "composite-predictive-view-v1",
    )

    parameter_recovery: dict[str, object]
    if dataset_kind == "in_family":
        parameter_recovery = {
            "mean_absolute_error": sum(
                abs(estimate - truth)
                for person in people
                for estimate, truth in zip(
                    representations[person.person_id].latent_mean,
                    person.true_weights,
                    strict=True,
                )
            )
            / (len(people) * len(FEATURE_NAMES))
        }
    else:
        parameter_recovery = {
            "status": "not_applicable",
            "reason": "generator is outside the fitted linear model family",
        }

    return {
        "experiment": {
            "seed": seed,
            "person_count": person_count,
            "source_trials_per_person": source_trials,
            "target_trials_per_person": target_trials,
            "feature_names": list(FEATURE_NAMES),
            "dataset_kind": dataset_kind,
            "heterogeneity_scale": (
                heterogeneity_scale
                if dataset_kind == "in_family"
                else 0.0
            ),
            "heterogeneity_scale_applies": dataset_kind == "in_family",
        },
        "models": {
            "population": report_to_dict(population),
            "correct_person": report_to_dict(personal),
            "wrong_person": report_to_dict(swapped),
        },
        "uplift": {
            "nll_vs_population": (
                population.negative_log_likelihood
                - personal.negative_log_likelihood
            ),
            "nll_vs_wrong_person": (
                swapped.negative_log_likelihood
                - personal.negative_log_likelihood
            ),
            "accuracy_vs_population": personal.accuracy - population.accuracy,
            "accuracy_vs_wrong_person": personal.accuracy - swapped.accuracy,
        },
        "parameter_recovery": parameter_recovery,
        "validity_gate": validity_gate,
        "module_slots": [
            {
                "slot": item.slot.value,
                "status": item.status,
                "module_id": item.module_id,
                "module_version": item.module_version,
            }
            for item in registry.manifest()
        ],
        "claim_boundary": (
            "Synthetic structured-choice closed-loop result only; "
            "not evidence of open-domain human cognitive simulation."
        ),
    }


def run_misspecification_demo(
    *,
    seed: int = 42,
    person_count: int = 24,
    source_trials: int = 120,
    target_trials: int = 180,
) -> dict[str, object]:
    return run_demo(
        seed=seed,
        person_count=person_count,
        source_trials=source_trials,
        target_trials=target_trials,
        dataset_kind="misspecified",
    )
