from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
BRIDGE_WORK = REPO_ROOT / "work" / "reality-bridge-v1"
for path in (str(SRC), str(BRIDGE_WORK), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import candidate
import run_bridge as bridge
from pcfm.applicability import fit_applicability_profile
from pcfm.contracts import Observation, Scenario
from pcfm.core import (
    DecisionIntegrator,
    IdentityAdapterGenerator,
    MapPersonEncoder,
    PopulationPriorEstimator,
)
from pcfm.evaluation import evaluate_probability_array, report_to_dict


MANIFEST_PATH = HERE / "experiment-manifest.json"
COHORT_PLAN_PATH = REPO_ROOT / "artifacts" / "reality_bridge_v1" / "cohort-plan.json"
RAW = REPO_ROOT / "artifacts" / "voteview_real_audit" / "raw"
OUTPUT = REPO_ROOT / "artifacts" / "joint_person_core_v1"
FINAL_CUTOFF = date(2026, 7, 31)
ENVIRONMENT_NAMES = (
    "env:majority_party_republican",
    "env:person_matches_majority",
    "env:person_matches_president",
    "env:congress_offset",
    "env:session_two",
    "env:elapsed_years",
)


def digest(value: object) -> str:
    return hashlib.sha256(bridge.canonical_json(value)).hexdigest()


def load_bound_inputs() -> tuple[dict[str, object], dict[str, object], str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_digest = digest(manifest)
    cohort_plan = json.loads(COHORT_PLAN_PATH.read_text(encoding="utf-8"))
    recorded_cohort_digest = str(cohort_plan.pop("cohort_plan_digest"))
    if digest(cohort_plan) != recorded_cohort_digest:
        raise ValueError("cohort plan self-digest mismatch")
    cohort_plan["cohort_plan_digest"] = recorded_cohort_digest
    if manifest["cohort_plan_digest"] != recorded_cohort_digest:
        raise ValueError("experiment manifest cohort digest mismatch")
    _, raw_digest = bridge.verify_raw_manifest()
    if manifest["raw_manifest_digest"] != raw_digest:
        raise ValueError("experiment manifest raw digest mismatch")
    if tuple(manifest["cohort_person_ids"]) != tuple(
        str(row["person_id"]) for row in cohort_plan["cohort"]
    ):
        raise ValueError("experiment manifest cohort identity mismatch")
    if tuple(manifest["environment_features"]) != ENVIRONMENT_NAMES:
        raise ValueError("experiment manifest environment schema mismatch")
    return manifest, cohort_plan, manifest_digest


def environment_features(
    congress: int,
    rollcall: dict[str, object],
    party_code: int,
    manifest: dict[str, object],
) -> np.ndarray:
    mapping = manifest["environment_mapping"]
    majority = int(mapping["majority_party_code_by_congress"][str(congress)])
    president = int(mapping["president_party_code_by_congress"][str(congress)])
    event_date = date.fromisoformat(str(rollcall["date"]))
    elapsed_years = (((event_date - date(2019, 1, 1)).days / 365.25) - 3.75) / 2.5
    return np.asarray(
        (
            float(majority == 200),
            float(party_code == majority),
            float(party_code == president),
            (congress - 117.5) / 1.5,
            float(int(rollcall.get("session") or 1) == 2),
            elapsed_years,
        ),
        dtype=np.float64,
    )


def expected_calibration_error(choices: np.ndarray, probabilities: np.ndarray) -> float:
    observations = tuple(
        Observation(
            person_id="pooled",
            scenario=Scenario(
                scenario_id=f"pooled-{index}",
                features=(1.0,),
                feature_names=("intercept",),
                options=("Nay", "Yea"),
                domain="joint_core_metric",
                context={"task": "metric"},
            ),
            actual_choice=int(choice),
            provenance="human_record",
        )
        for index, choice in enumerate(choices)
    )
    return float(
        report_to_dict(evaluate_probability_array(observations, probabilities))[
            "expected_calibration_error"
        ]
    )


def metric_dict(observations: tuple[Observation, ...], probabilities: np.ndarray):
    return report_to_dict(evaluate_probability_array(observations, probabilities))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, cohort_plan, manifest_digest = load_bound_inputs()
    cohort_ids = tuple(str(value) for value in manifest["cohort_person_ids"])
    cohort_by_id = {
        str(row["person_id"]): row for row in cohort_plan["cohort"]
    }
    training_ids = set(cohort_ids) | set(cohort_plan["population_reference_ids"])
    feature_names = tuple(str(value) for value in cohort_plan["feature_names"])
    top_policies = tuple(str(value) for value in cohort_plan["top_policies"])
    text_tokens = tuple(str(value) for value in cohort_plan["text_tokens"])

    members: dict[int, dict[str, dict[str, str]]] = {}
    vote_rows: dict[int, dict[str, list[dict[str, str]]]] = {}
    rollcalls: dict[tuple[int, int], dict[str, object]] = {}
    for congress in range(116, 120):
        members[congress] = {
            row["icpsr"]: row
            for row in bridge.read_csv(RAW / f"S{congress}_members.csv")
            if row["chamber"] == "Senate"
        }
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in bridge.read_csv(RAW / f"S{congress}_votes.csv"):
            if row["chamber"] == "Senate" and bridge.valid_choice(row["cast_code"]) is not None:
                grouped[row["icpsr"]].append(row)
        vote_rows[congress] = {
            person_id: sorted(rows, key=lambda item: int(item["rollnumber"]))
            for person_id, rows in grouped.items()
        }
        for row in bridge.read_json(RAW / f"S{congress}_rollcalls.json"):
            rollcalls[(congress, int(row["rollnumber"]))] = row

    scenario_cache: dict[tuple[int, int], Scenario] = {}

    def scenario_for(congress: int, rollnumber: int) -> Scenario:
        key = (congress, rollnumber)
        if key in scenario_cache:
            return scenario_cache[key]
        row = rollcalls[key]
        policy = str(row.get("crs_policy_area") or "__missing__")
        row_tokens = bridge.tokens_for(row)
        current_motion = bridge.motion_type(row.get("vote_question"))
        values = [1.0]
        values.extend(float(current_motion == value) for value in bridge.MOTION_TYPES)
        values.append(float(bool(str(row.get("bill_number") or "").strip())))
        values.append(float(int(row.get("session") or 1) == 2))
        values.extend(float(policy == value) for value in top_policies)
        values.append(float(policy not in top_policies))
        values.extend(float(token in row_tokens) for token in text_tokens)
        scenario = Scenario(
            scenario_id=f"voteview:S:{congress}:{rollnumber}",
            features=tuple(values),
            feature_names=feature_names,
            options=("Nay", "Yea"),
            domain="us_senate_roll_call",
            context=bridge.TASK_CONTEXT,
        )
        if len(values) != len(feature_names):
            raise ValueError("scenario feature dimension mismatch")
        scenario_cache[key] = scenario
        return scenario

    party_code = {
        person_id: int(members[119][person_id]["party_code"])
        for person_id in training_ids
    }

    def record(person_id: str, congress: int, row: dict[str, str]):
        rollnumber = int(row["rollnumber"])
        rollcall = rollcalls[(congress, rollnumber)]
        scenario = scenario_for(congress, rollnumber)
        environment = environment_features(
            congress, rollcall, party_code[person_id], manifest
        )
        observation = Observation(
            person_id=person_id,
            scenario=scenario,
            actual_choice=int(bridge.valid_choice(row["cast_code"])),
            provenance="human_record",
        )
        return {
            "person_id": person_id,
            "congress": congress,
            "rollnumber": rollnumber,
            "scenario": scenario,
            "scenario_features": np.asarray(scenario.features, dtype=np.float64),
            "environment_features": environment,
            "choice": observation.actual_choice,
            "timestamp": bridge.ordered_timestamp(rollcall),
            "observation": observation,
        }

    training_records = []
    for person_id in sorted(training_ids, key=int):
        training_records.extend(
            record(person_id, 116, row) for row in vote_rows[116][person_id]
        )
        training_records.extend(
            record(person_id, 117, row)
            for row in vote_rows[117][person_id]
            if int(row["rollnumber"]) <= 475
        )
    training_records.sort(
        key=lambda item: (int(item["person_id"]), item["congress"], item["rollnumber"])
    )
    train_x = np.asarray([item["scenario_features"] for item in training_records])
    train_e = np.asarray([item["environment_features"] for item in training_records])
    train_y = np.asarray([item["choice"] for item in training_records])
    train_people = tuple(str(item["person_id"]) for item in training_records)
    train_observations = tuple(item["observation"] for item in training_records)

    prior = PopulationPriorEstimator(feature_names).fit(train_observations)
    encoder = MapPersonEncoder()
    generator = IdentityAdapterGenerator()
    integrator = DecisionIntegrator()
    baseline_representations = {
        person_id: encoder.fit(person_id, train_observations, prior)
        for person_id in cohort_ids
    }
    baseline_adapters = {
        person_id: generator.generate(baseline_representations[person_id], prior)
        for person_id in cohort_ids
    }

    stable_grid = tuple(float(value) for value in manifest["fitting"]["stable_person_precision_grid"])
    models = {
        precision: candidate.fit_joint_core(
            train_x,
            train_e,
            train_y,
            train_people,
            scenario_feature_names=feature_names,
            environment_feature_names=ENVIRONMENT_NAMES,
            stable_person_precision=precision,
            global_l2_precision=float(manifest["fitting"]["global_l2_precision"]),
            coordinate_passes=int(manifest["fitting"]["coordinate_passes"]),
            maximum_iterations=int(manifest["fitting"]["newton_maximum_iterations"]),
            tolerance=float(manifest["fitting"]["newton_tolerance"]),
        )
        for precision in stable_grid
    }

    selection_records = {
        person_id: tuple(
            record(person_id, 118, row) for row in vote_rows[118][person_id]
        )
        for person_id in cohort_ids
    }
    selection_rows = []
    selection_states = {}
    for precision in stable_grid:
        model = models[precision]
        for half_life in manifest["dynamic_state_grid"]["half_life_days"]:
            for stationary_variance in manifest["dynamic_state_grid"]["stationary_variance"]:
                config = candidate.StateConfig(
                    half_life_days=float(half_life),
                    stationary_variance=float(stationary_variance),
                    update_maximum_iterations=int(
                        manifest["dynamic_state_grid"]["update_maximum_iterations"]
                    ),
                    update_tolerance=float(
                        manifest["dynamic_state_grid"]["update_tolerance"]
                    ),
                )
                person_nll = []
                states = {}
                for person_id in cohort_ids:
                    rows = selection_records[person_id]
                    x = np.asarray([item["scenario_features"] for item in rows])
                    e = np.asarray([item["environment_features"] for item in rows])
                    y = np.asarray([item["choice"] for item in rows])
                    timestamps = tuple(str(item["timestamp"]) for item in rows)
                    logits, variances = model.logits_and_variances(person_id, x, e)
                    result = candidate.run_prequential_state(
                        logits, variances, y, timestamps, config
                    )
                    person_nll.append(
                        candidate.negative_log_likelihood(y, result.probabilities)
                    )
                    states[person_id] = result
                row = {
                    "stable_person_precision": precision,
                    "half_life_days": float(half_life),
                    "stationary_variance": float(stationary_variance),
                    "equal_person_mean_nll": float(np.mean(person_nll)),
                    "person_nll": person_nll,
                }
                selection_rows.append(row)
                selection_states[(precision, float(half_life), float(stationary_variance))] = states
    selected = min(
        selection_rows,
        key=lambda item: (
            item["equal_person_mean_nll"],
            item["stable_person_precision"],
            item["half_life_days"],
            item["stationary_variance"],
        ),
    )
    selected_key = (
        selected["stable_person_precision"],
        selected["half_life_days"],
        selected["stationary_variance"],
    )
    selected_model = models[selected["stable_person_precision"]]
    selected_config = candidate.StateConfig(
        half_life_days=selected["half_life_days"],
        stationary_variance=selected["stationary_variance"],
        update_maximum_iterations=int(
            manifest["dynamic_state_grid"]["update_maximum_iterations"]
        ),
        update_tolerance=float(manifest["dynamic_state_grid"]["update_tolerance"]),
    )
    selected_target_states = selection_states[selected_key]

    training_rate = {
        person_id: float(
            np.mean(
                [item["choice"] for item in training_records if item["person_id"] == person_id]
            )
        )
        for person_id in cohort_ids
    }
    wrong_person = {}
    for person_id in cohort_ids:
        same_party = [
            other
            for other in cohort_ids
            if other != person_id and party_code[other] == party_code[person_id]
        ]
        wrong_person[person_id] = min(
            same_party,
            key=lambda other: (
                abs(training_rate[other] - training_rate[person_id]),
                int(other),
            ),
        )

    per_person = []
    pooled_choices = []
    pooled_full_probabilities = []
    for person_id in cohort_ids:
        rows119 = vote_rows[119][person_id]
        applicability_source = rows119[:150]
        target_final_source = rows119[150:450]
        wrong_id = wrong_person[person_id]
        wrong_by_roll = {
            int(row["rollnumber"]): row for row in vote_rows[119][wrong_id]
        }
        target_final_source = tuple(
            row for row in target_final_source if int(row["rollnumber"]) in wrong_by_roll
        )
        applicability_rows = tuple(
            record(person_id, 119, row) for row in applicability_source
        )
        final_rows = tuple(record(person_id, 119, row) for row in target_final_source)
        wrong_final_rows = tuple(
            record(wrong_id, 119, wrong_by_roll[int(row["rollnumber"])])
            for row in target_final_source
        )
        x = np.asarray([item["scenario_features"] for item in final_rows])
        e = np.asarray([item["environment_features"] for item in final_rows])
        y = np.asarray([item["choice"] for item in final_rows])
        timestamps = tuple(str(item["timestamp"]) for item in final_rows)
        observations = tuple(item["observation"] for item in final_rows)

        combined_names = feature_names + ENVIRONMENT_NAMES
        applicability_observations = tuple(
            Observation(
                person_id=person_id,
                scenario=Scenario(
                    scenario_id=item["scenario"].scenario_id,
                    features=tuple(
                        np.concatenate(
                            (item["scenario_features"], item["environment_features"])
                        )
                    ),
                    feature_names=combined_names,
                    options=("Nay", "Yea"),
                    domain="us_senate_joint_core",
                    context=bridge.TASK_CONTEXT,
                ),
                actual_choice=int(item["choice"]),
                provenance="human_record",
            )
            for item in applicability_rows
        )
        profile = fit_applicability_profile(
            applicability_observations,
            combined_names,
            valid_through=str(applicability_rows[-1]["timestamp"]),
        )
        applicability_refusals = []
        for item in final_rows:
            scenario = Scenario(
                scenario_id=item["scenario"].scenario_id,
                features=tuple(
                    np.concatenate(
                        (item["scenario_features"], item["environment_features"])
                    )
                ),
                feature_names=combined_names,
                options=("Nay", "Yea"),
                domain="us_senate_joint_core",
                context=bridge.TASK_CONTEXT,
            )
            assessment = profile.assess(scenario, prediction_at=str(item["timestamp"]))
            if assessment.reasons or assessment.warnings:
                applicability_refusals.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "reasons": list(assessment.reasons),
                        "warnings": list(assessment.warnings),
                    }
                )

        population_probabilities = np.asarray(
            [
                integrator.predict_population(item["scenario"], prior).probability_option_1
                for item in final_rows
            ]
        )
        static_probabilities = np.asarray(
            [
                integrator.predict(
                    item["scenario"],
                    prior,
                    baseline_adapters[person_id],
                    parameter_covariance=baseline_representations[person_id].covariance,
                ).probability_option_1
                for item in final_rows
            ]
        )
        environment_population = selected_model.probabilities(
            person_id, x, e, include_person=False, include_environment=True
        )

        full_logits, full_variances = selected_model.logits_and_variances(
            person_id, x, e
        )
        target_selection_state = selected_target_states[person_id]
        full = candidate.run_prequential_state(
            full_logits,
            full_variances,
            y,
            timestamps,
            selected_config,
            initial_state_mean=target_selection_state.final_state_mean,
            initial_state_variance=target_selection_state.final_state_variance,
            previous_timestamp=target_selection_state.final_timestamp,
        )

        dynamic_logits, dynamic_variances = selected_model.logits_and_variances(
            person_id, x, e, include_person=False, include_environment=True
        )
        selection_target = selection_records[person_id]
        selection_x = np.asarray(
            [item["scenario_features"] for item in selection_target]
        )
        selection_e = np.asarray(
            [item["environment_features"] for item in selection_target]
        )
        selection_y = np.asarray([item["choice"] for item in selection_target])
        selection_timestamps = tuple(str(item["timestamp"]) for item in selection_target)
        dynamic_selection_logits, dynamic_selection_variances = (
            selected_model.logits_and_variances(
                person_id,
                selection_x,
                selection_e,
                include_person=False,
                include_environment=True,
            )
        )
        dynamic_selection_state = candidate.run_prequential_state(
            dynamic_selection_logits,
            dynamic_selection_variances,
            selection_y,
            selection_timestamps,
            selected_config,
        )
        dynamic_population = candidate.run_prequential_state(
            dynamic_logits,
            dynamic_variances,
            y,
            timestamps,
            selected_config,
            initial_state_mean=dynamic_selection_state.final_state_mean,
            initial_state_variance=dynamic_selection_state.final_state_variance,
            previous_timestamp=dynamic_selection_state.final_timestamp,
        )

        wrong_selection = selection_records[wrong_id]
        wrong_selection_x = np.asarray(
            [item["scenario_features"] for item in wrong_selection]
        )
        wrong_selection_e = np.asarray(
            [item["environment_features"] for item in wrong_selection]
        )
        wrong_selection_y = np.asarray([item["choice"] for item in wrong_selection])
        wrong_selection_times = tuple(str(item["timestamp"]) for item in wrong_selection)
        wrong_selection_logits, wrong_selection_variances = (
            selected_model.logits_and_variances(
                wrong_id, wrong_selection_x, wrong_selection_e
            )
        )
        wrong_selection_state = candidate.run_prequential_state(
            wrong_selection_logits,
            wrong_selection_variances,
            wrong_selection_y,
            wrong_selection_times,
            selected_config,
        )
        wrong_x = np.asarray([item["scenario_features"] for item in wrong_final_rows])
        wrong_e = np.asarray([item["environment_features"] for item in wrong_final_rows])
        wrong_y = np.asarray([item["choice"] for item in wrong_final_rows])
        wrong_logits, wrong_variances = selected_model.logits_and_variances(
            wrong_id, wrong_x, wrong_e
        )
        wrong_state = candidate.run_prequential_state(
            wrong_logits,
            wrong_variances,
            wrong_y,
            timestamps,
            selected_config,
            initial_state_mean=wrong_selection_state.final_state_mean,
            initial_state_variance=wrong_selection_state.final_state_variance,
            previous_timestamp=wrong_selection_state.final_timestamp,
        )

        permutation = sorted(
            range(len(selection_target)),
            key=lambda index: hashlib.sha256(
                (
                    person_id
                    + "\0"
                    + selection_target[index]["scenario"].scenario_id
                ).encode("utf-8")
            ).digest(),
        )
        shuffled_selection_state = candidate.run_prequential_state(
            *selected_model.logits_and_variances(
                person_id, selection_x, selection_e
            ),
            selection_y[permutation],
            selection_timestamps,
            selected_config,
        )
        shuffled = candidate.run_prequential_state(
            full_logits,
            full_variances,
            y,
            timestamps,
            selected_config,
            initial_state_mean=shuffled_selection_state.final_state_mean,
            initial_state_variance=shuffled_selection_state.final_state_variance,
            previous_timestamp=shuffled_selection_state.final_timestamp,
        )

        metrics = {
            "population_scenario": metric_dict(observations, population_probabilities),
            "static_correct_person": metric_dict(observations, static_probabilities),
            "environment_population": metric_dict(observations, environment_population),
            "dynamic_population": metric_dict(
                observations, dynamic_population.probabilities
            ),
            "wrong_person_joint": metric_dict(observations, wrong_state.probabilities),
            "history_shuffled_joint": metric_dict(
                observations, shuffled.probabilities
            ),
            "full_joint": metric_dict(observations, full.probabilities),
        }
        non_person_nll = min(
            metrics[name]["negative_log_likelihood"]
            for name in (
                "population_scenario",
                "environment_population",
                "dynamic_population",
            )
        )
        full_nll = metrics["full_joint"]["negative_log_likelihood"]
        per_person.append(
            {
                "person_id": person_id,
                "name": cohort_by_id[person_id]["name"],
                "wrong_person_id": wrong_id,
                "sample_count": len(y),
                "applicability_refusal_count": len(applicability_refusals),
                "applicability_refusal_examples": applicability_refusals[:10],
                "metrics": metrics,
                "nll_uplift_over_best_non_person": non_person_nll - full_nll,
                "nll_uplift_over_wrong_person": (
                    metrics["wrong_person_joint"]["negative_log_likelihood"]
                    - full_nll
                ),
                "nll_uplift_over_history_shuffled": (
                    metrics["history_shuffled_joint"]["negative_log_likelihood"]
                    - full_nll
                ),
            }
        )
        pooled_choices.extend(int(value) for value in y)
        pooled_full_probabilities.extend(float(value) for value in full.probabilities)

    uplifts = np.asarray(
        [row["nll_uplift_over_best_non_person"] for row in per_person]
    )
    wrong_uplifts = np.asarray(
        [row["nll_uplift_over_wrong_person"] for row in per_person]
    )
    shuffled_uplifts = np.asarray(
        [row["nll_uplift_over_history_shuffled"] for row in per_person]
    )
    uncertainty = manifest["uncertainty_and_aggregation"]
    rng = np.random.default_rng(int(uncertainty["bootstrap_seed"]))
    bootstrap = np.mean(
        uplifts[
            rng.integers(
                0,
                len(uplifts),
                size=(int(uncertainty["bootstrap_draws"]), len(uplifts)),
            )
        ],
        axis=1,
    )
    lower = float(np.quantile(bootstrap, float(uncertainty["lower_quantile"])))
    mean_uplift = float(np.mean(uplifts))
    pooled_ece = expected_calibration_error(
        np.asarray(pooled_choices), np.asarray(pooled_full_probabilities)
    )
    positive_people = int(np.sum(uplifts > 0.0))
    applicability_refusals = sum(
        int(row["applicability_refusal_count"]) for row in per_person
    )
    gates = {
        "all_six_people_reported": len(per_person) == 6,
        "at_least_four_positive_people": positive_people >= 4,
        "mean_nll_uplift_at_least_0_01": mean_uplift >= 0.01,
        "paired_lower_bound_positive": lower > 0.0,
        "pooled_ece_at_most_0_15": pooled_ece <= 0.15,
        "beats_wrong_person": float(np.mean(wrong_uplifts)) > 0.0,
        "beats_history_shuffled": float(np.mean(shuffled_uplifts)) > 0.0,
        "no_applicability_refusal": applicability_refusals == 0,
    }
    status = (
        manifest["status_on_pass"]
        if all(gates.values())
        else manifest["status_on_fail"]
    )

    model_path = OUTPUT / "selected-model.json"
    model_path.write_text(
        json.dumps(selected_model.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    restored_model = candidate.JointCoreModel.from_dict(
        json.loads(model_path.read_text(encoding="utf-8"))
    )
    if restored_model.model_id != selected_model.model_id:
        raise ValueError("selected model round trip mismatch")

    result = {
        "schema_version": "pcfm-joint-person-core-report-v1",
        "status": status,
        "evidence_status": manifest["evidence_status"],
        "semantic_module_resume_authorized": False,
        "experiment_manifest_digest": manifest_digest,
        "selected_configuration": selected,
        "selected_model_id": selected_model.model_id,
        "selection_grid": selection_rows,
        "per_person": per_person,
        "aggregate": {
            "positive_person_count": positive_people,
            "equal_person_mean_nll_uplift": mean_uplift,
            "cluster_bootstrap_lower": lower,
            "equal_person_mean_wrong_person_uplift": float(np.mean(wrong_uplifts)),
            "equal_person_mean_history_shuffled_uplift": float(
                np.mean(shuffled_uplifts)
            ),
            "pooled_full_joint_ece": pooled_ece,
            "applicability_refusal_count": applicability_refusals,
        },
        "gates": gates,
        "non_claims": [
            "prospective confirmation",
            "belief identification",
            "value identification",
            "causal mechanism recovery",
            "general person simulation",
        ],
    }
    result["report_digest"] = digest(result)
    (OUTPUT / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
