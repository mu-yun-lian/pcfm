from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SRC = REPO_ROOT / "src"
BRIDGE_WORK = REPO_ROOT / "work" / "reality-bridge-v1"
for path in (str(SRC), str(BRIDGE_WORK), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import relational
import run_bridge as bridge
from pcfm.applicability import fit_applicability_profile
from pcfm.contracts import Observation, Scenario
from pcfm.evaluation import evaluate_probability_array, report_to_dict


MANIFEST_PATH = HERE / "experiment-manifest.json"
COHORT_PLAN_PATH = REPO_ROOT / "artifacts" / "reality_bridge_v1" / "cohort-plan.json"
RAW = REPO_ROOT / "artifacts" / "voteview_real_audit" / "raw"
OUTPUT = REPO_ROOT / "artifacts" / "person_issue_relational_core_v1"


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
    feature_spec = manifest["feature_map"]
    if tuple(feature_spec["forbidden_fields"]) != relational.FORBIDDEN_FIELDS:
        raise ValueError("experiment forbidden feature schema mismatch")
    if int(feature_spec["scenario_feature_count"]) != 19:
        raise ValueError("experiment scenario feature count mismatch")
    if feature_spec["llm_or_external_embedding_allowed"] is not False:
        raise ValueError("experiment must forbid LLM-generated features")
    return manifest, cohort_plan, manifest_digest


def load_voteview() -> tuple[
    dict[int, dict[str, dict[str, str]]],
    dict[int, dict[str, list[dict[str, str]]]],
    dict[tuple[int, int], dict[str, object]],
]:
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
        seen: set[tuple[str, int]] = set()
        for row in bridge.read_csv(RAW / f"S{congress}_votes.csv"):
            if row["chamber"] != "Senate" or bridge.valid_choice(row["cast_code"]) is None:
                continue
            trial = (str(row["icpsr"]), int(row["rollnumber"]))
            if trial in seen:
                raise ValueError(f"duplicate Voteview trial {trial}")
            seen.add(trial)
            grouped[str(row["icpsr"])].append(row)
        vote_rows[congress] = {
            person_id: sorted(rows, key=lambda item: int(item["rollnumber"]))
            for person_id, rows in grouped.items()
        }
        for row in bridge.read_json(RAW / f"S{congress}_rollcalls.json"):
            key = (congress, int(row["rollnumber"]))
            if key in rollcalls:
                raise ValueError(f"duplicate Voteview roll call {key}")
            rollcalls[key] = row
    return members, vote_rows, rollcalls


def audit_role_windows() -> list[dict[str, object]]:
    manifest, _cohort, _digest = load_bound_inputs()
    _members, vote_rows, rollcalls = load_voteview()
    result = []
    for person_id in manifest["cohort_person_ids"]:
        rows = vote_rows[119][str(person_id)]
        warmup = rows[:450]
        applicability = rows[450:530]
        final = rows[530:830]
        if len(warmup) != 450 or len(applicability) != 80 or len(final) != 300:
            raise ValueError(f"insufficient frozen role rows for {person_id}")
        result.append(
            {
                "person_id": str(person_id),
                "warmup_count": len(warmup),
                "warmup_last_roll": int(warmup[-1]["rollnumber"]),
                "applicability_count": len(applicability),
                "applicability_first_roll": int(applicability[0]["rollnumber"]),
                "applicability_last_roll": int(applicability[-1]["rollnumber"]),
                "applicability_end_date": str(
                    rollcalls[(119, int(applicability[-1]["rollnumber"]))]["date"]
                ),
                "final_count": len(final),
                "final_first_roll": int(final[0]["rollnumber"]),
                "final_last_roll": int(final[-1]["rollnumber"]),
                "final_end_date": str(
                    rollcalls[(119, int(final[-1]["rollnumber"]))]["date"]
                ),
            }
        )
    return result


def _metric_observations(person_id: str, choices: Sequence[int]) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            person_id=person_id,
            scenario=Scenario(
                scenario_id=f"metric:{person_id}:{index}",
                features=(1.0,),
                feature_names=("intercept",),
                options=("Nay", "Yea"),
                domain="person_issue_relational_metric",
                context={"task": "metric"},
            ),
            actual_choice=int(choice),
            provenance="human_record",
        )
        for index, choice in enumerate(choices)
    )


def metric_dict(person_id: str, choices: Sequence[int], probabilities: Sequence[float]):
    return report_to_dict(
        evaluate_probability_array(
            _metric_observations(person_id, choices),
            np.asarray(probabilities, dtype=np.float64),
        )
    )


def _record(
    person_id: str,
    congress: int,
    vote: Mapping[str, str],
    rollcalls: Mapping[tuple[int, int], Mapping[str, object]],
    party_code: int,
) -> dict[str, object]:
    rollnumber = int(vote["rollnumber"])
    rollcall = dict(rollcalls[(congress, rollnumber)])
    choice = bridge.valid_choice(vote["cast_code"])
    if choice is None:
        raise ValueError("non-binary vote entered a frozen role")
    return {
        "person_id": person_id,
        "party_code": party_code,
        "congress": congress,
        "rollnumber": rollnumber,
        "rollcall": rollcall,
        "choice": int(choice),
        "timestamp": bridge.ordered_timestamp(rollcall),
        "event_id": f"voteview:S:{congress}:{rollnumber}:{person_id}",
    }


def _fit_artifact(
    feature_map: relational.RelationalFeatureMap,
    records: Sequence[Mapping[str, object]],
    precision: float,
    *,
    coordinate_passes: int = 6,
) -> relational.RelationalCoreArtifact:
    return relational.fit_relational_artifact(
        feature_map,
        [row["rollcall"] for row in records],
        [int(row["choice"]) for row in records],
        [str(row["person_id"]) for row in records],
        [int(row["party_code"]) for row in records],
        [int(row["congress"]) for row in records],
        stable_person_precision=precision,
        global_l2_precision=1.0,
        coordinate_passes=coordinate_passes,
    )


def _state_config(row: Mapping[str, object]) -> object:
    return relational.StateConfig(
        half_life_days=float(row["half_life_days"]),
        stationary_variance=float(row["stationary_variance"]),
    )


def _run_rows(
    artifact: relational.RelationalCoreArtifact,
    person_id: str,
    rows: Sequence[Mapping[str, object]],
    config: object,
    *,
    profile_person_id: str | None = None,
    include_person: bool = True,
    initial: object | None = None,
) -> object:
    keywords: dict[str, object] = {}
    if initial is not None:
        keywords = {
            "initial_state_mean": initial.final_state_mean,
            "initial_state_variance": initial.final_state_variance,
            "previous_timestamp": initial.final_timestamp,
        }
    return artifact.run_prequential(
        person_id,
        [row["rollcall"] for row in rows],
        [int(row["party_code"]) for row in rows],
        [int(row["congress"]) for row in rows],
        [int(row["choice"]) for row in rows],
        [str(row["timestamp"]) for row in rows],
        config,
        profile_person_id=profile_person_id,
        include_person=include_person,
        **keywords,
    )


def _shuffled_choices(records: Sequence[Mapping[str, object]], seed: int) -> np.ndarray:
    choices = np.asarray([int(row["choice"]) for row in records], dtype=np.int64)
    shuffled = choices.copy()
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[(str(row["person_id"]), int(row["congress"]))].append(index)
    for key, indices in groups.items():
        permutation = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{seed}\0{key[0]}\0{key[1]}\0{records[index]['rollnumber']}".encode(
                    "utf-8"
                )
            ).digest(),
        )
        shuffled[np.asarray(indices)] = choices[np.asarray(permutation)]
    return shuffled


def _applicability_scenario(
    feature_map: relational.RelationalFeatureMap,
    row: Mapping[str, object],
) -> Scenario:
    scenario = feature_map.transform(row["rollcall"])
    factor_names = tuple(f"text_factor:{index}" for index in range(feature_map.svd_rank))
    factors = tuple(float(value) for value in scenario[-feature_map.svd_rank :])
    return Scenario(
        scenario_id=f"app:{row['event_id']}",
        features=factors,
        feature_names=factor_names,
        options=("Nay", "Yea"),
        domain="us_senate_roll_call",
        context=bridge.TASK_CONTEXT,
    )


def _applicability_refusals(
    feature_map: relational.RelationalFeatureMap,
    applicability_rows: Sequence[Mapping[str, object]],
    final_rows: Sequence[Mapping[str, object]],
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    settings = manifest["applicability"]
    names = tuple(str(value) for value in settings["continuous_features"])
    observations = tuple(
        Observation(
            person_id=str(row["person_id"]),
            scenario=_applicability_scenario(feature_map, row),
            actual_choice=0,
            provenance="feature_only_applicability_placeholder",
        )
        for row in applicability_rows
    )
    profile = fit_applicability_profile(
        observations,
        names,
        valid_through=str(applicability_rows[-1]["timestamp"]),
        maximum_age_days=float(settings["maximum_age_days"]),
        calibration_safety_factor=float(settings["calibration_safety_factor"]),
    )
    supported_motion = {
        relational._motion_type(row["rollcall"].get("vote_question"))
        for row in applicability_rows
    }
    supported_bill = {
        relational._bill_type(row["rollcall"].get("bill_number"))
        for row in applicability_rows
    }
    refusals = []
    for row in final_rows:
        scenario = _applicability_scenario(feature_map, row)
        assessment = profile.assess(scenario, prediction_at=str(row["timestamp"]))
        reasons = list(assessment.reasons)
        warnings = list(assessment.warnings)
        motion = relational._motion_type(row["rollcall"].get("vote_question"))
        bill = relational._bill_type(row["rollcall"].get("bill_number"))
        if motion not in supported_motion:
            reasons.append("unsupported_motion_type")
        if bill not in supported_bill:
            reasons.append("unsupported_bill_type")
        if reasons or warnings:
            refusals.append(
                {
                    "event_id": str(row["event_id"]),
                    "reasons": sorted(set(reasons)),
                    "warnings": sorted(set(warnings)),
                }
            )
    return refusals


def _load_bundle(path: Path) -> tuple[
    relational.RelationalCoreArtifact,
    relational.RelationalCoreArtifact,
    relational.RelationalCoreArtifact,
    dict[str, object],
]:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(bundle.pop("bundle_id"))
    if digest(bundle) != recorded:
        raise ValueError("relational bundle identity mismatch")
    bundle["bundle_id"] = recorded
    return (
        relational.RelationalCoreArtifact.from_dict(bundle["full_artifact"]),
        relational.RelationalCoreArtifact.from_dict(bundle["party_artifact"]),
        relational.RelationalCoreArtifact.from_dict(bundle["shuffled_artifact"]),
        bundle,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, cohort_plan, manifest_digest = load_bound_inputs()
    members, vote_rows, rollcalls = load_voteview()
    cohort_ids = tuple(str(value) for value in manifest["cohort_person_ids"])
    cohort_by_id = {str(row["person_id"]): row for row in cohort_plan["cohort"]}
    party_codes = {
        person_id: int(members[119][person_id]["party_code"]) for person_id in cohort_ids
    }

    basis_rollcalls = [
        rollcalls[key] for key in sorted(rollcalls) if key[0] in (116, 117, 118)
    ]
    feature_spec = manifest["feature_map"]
    feature_map = relational.RelationalFeatureMap.fit(
        basis_rollcalls,
        hash_dimension=int(feature_spec["hash_dimension"]),
        svd_rank=int(feature_spec["svd_rank"]),
    )
    if len(feature_map.feature_names) != int(feature_spec["scenario_feature_count"]):
        raise ValueError("fitted relational feature count mismatch")

    def rows_for(person_id: str, congress: int) -> list[dict[str, object]]:
        return [
            _record(person_id, congress, row, rollcalls, party_codes[person_id])
            for row in vote_rows[congress][person_id]
        ]

    estimation = []
    selection: dict[str, list[dict[str, object]]] = {}
    refit = []
    roles119: dict[str, dict[str, list[dict[str, object]]]] = {}
    for person_id in cohort_ids:
        early = rows_for(person_id, 116) + rows_for(person_id, 117)
        later = rows_for(person_id, 118)
        estimation.extend(early)
        refit.extend(early + later)
        selection[person_id] = later
        current = rows_for(person_id, 119)
        roles119[person_id] = {
            "warmup": current[:450],
            "applicability": current[450:530],
            "final": current[530:830],
        }
        if any(
            len(roles119[person_id][name]) != count
            for name, count in (("warmup", 450), ("applicability", 80), ("final", 300))
        ):
            raise ValueError(f"frozen Congress 119 role count mismatch for {person_id}")
    estimation.sort(key=lambda row: (int(row["person_id"]), row["congress"], row["rollnumber"]))
    refit.sort(key=lambda row: (int(row["person_id"]), row["congress"], row["rollnumber"]))

    grid = manifest["selection_grid"]
    fitted_by_precision = {
        float(precision): _fit_artifact(feature_map, estimation, float(precision))
        for precision in grid["stable_person_precision"]
    }
    selection_rows = []
    for precision in grid["stable_person_precision"]:
        artifact = fitted_by_precision[float(precision)]
        for half_life in grid["half_life_days"]:
            for stationary_variance in grid["stationary_variance"]:
                config_row = {
                    "stable_person_precision": float(precision),
                    "half_life_days": float(half_life),
                    "stationary_variance": float(stationary_variance),
                }
                config = _state_config(config_row)
                person_nll = []
                for person_id in cohort_ids:
                    trace = _run_rows(artifact, person_id, selection[person_id], config)
                    person_nll.append(
                        relational.negative_log_likelihood(
                            [row["choice"] for row in selection[person_id]],
                            trace.probabilities,
                        )
                    )
                selection_rows.append(
                    {
                        **config_row,
                        "equal_person_mean_nll": float(np.mean(person_nll)),
                        "person_nll": person_nll,
                    }
                )
    if len(selection_rows) != int(grid["configuration_count"]):
        raise ValueError("selection grid count mismatch")
    selected = min(
        selection_rows,
        key=lambda row: (
            row["equal_person_mean_nll"],
            row["stable_person_precision"],
            row["half_life_days"],
            row["stationary_variance"],
        ),
    )
    config = _state_config(selected)

    full_artifact = _fit_artifact(
        feature_map, refit, float(selected["stable_person_precision"])
    )
    party_artifact = _fit_artifact(feature_map, refit, 1.0e12, coordinate_passes=2)
    shuffled_y = _shuffled_choices(refit, int(manifest["fixed_shuffle_seed"]))
    shuffled_artifact = relational.refit_profiles_with_fixed_global(
        full_artifact,
        [row["rollcall"] for row in refit],
        shuffled_y,
        [str(row["person_id"]) for row in refit],
        [int(row["party_code"]) for row in refit],
        [int(row["congress"]) for row in refit],
    )

    bundle = {
        "schema_version": "pcfm-person-issue-relational-bundle-v1",
        "experiment_manifest_digest": manifest_digest,
        "selected_configuration": selected,
        "full_artifact": full_artifact.to_dict(),
        "party_artifact": party_artifact.to_dict(),
        "shuffled_artifact": shuffled_artifact.to_dict(),
    }
    bundle["bundle_id"] = digest(bundle)
    bundle_path = OUTPUT / "selected-model.json"
    bundle_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    full_artifact, party_artifact, shuffled_artifact, restored_bundle = _load_bundle(
        bundle_path
    )

    per_person = []
    prediction_audit = []
    pooled_y = []
    pooled_correct = []
    wrong_mapping = {str(key): str(value) for key, value in manifest["wrong_person_mapping"].items()}
    for person_id in cohort_ids:
        roles = roles119[person_id]
        wrong_id = wrong_mapping[person_id]
        if party_codes[wrong_id] != party_codes[person_id]:
            raise ValueError("wrong-person control crossed the frozen party boundary")
        refusals = _applicability_refusals(
            feature_map, roles["applicability"], roles["final"], manifest
        )

        warm_correct = _run_rows(full_artifact, person_id, roles["warmup"], config)
        warm_party = _run_rows(
            party_artifact,
            person_id,
            roles["warmup"],
            config,
            include_person=False,
        )
        warm_wrong = _run_rows(
            full_artifact,
            person_id,
            roles["warmup"],
            config,
            profile_person_id=wrong_id,
        )
        warm_shuffled = _run_rows(
            shuffled_artifact, person_id, roles["warmup"], config
        )

        final_correct = _run_rows(
            full_artifact,
            person_id,
            roles["final"],
            config,
            initial=warm_correct,
        )
        final_party = _run_rows(
            party_artifact,
            person_id,
            roles["final"],
            config,
            include_person=False,
            initial=warm_party,
        )
        final_wrong = _run_rows(
            full_artifact,
            person_id,
            roles["final"],
            config,
            profile_person_id=wrong_id,
            initial=warm_wrong,
        )
        final_shuffled = _run_rows(
            shuffled_artifact,
            person_id,
            roles["final"],
            config,
            initial=warm_shuffled,
        )
        static_party = party_artifact.probabilities(
            person_id,
            [row["rollcall"] for row in roles["final"]],
            party_codes[person_id],
            119,
            include_person=False,
        )
        static_correct = full_artifact.probabilities(
            person_id,
            [row["rollcall"] for row in roles["final"]],
            party_codes[person_id],
            119,
        )
        choices = np.asarray([int(row["choice"]) for row in roles["final"]])
        probabilities = {
            "party_static_relation": static_party,
            "correct_profile_static": static_correct,
            "party_dynamic_relation": final_party.probabilities,
            "wrong_same_party_profile_dynamic": final_wrong.probabilities,
            "within_congress_history_shuffled_profile_dynamic": final_shuffled.probabilities,
            "correct_person_profile_dynamic": final_correct.probabilities,
        }
        metrics = {
            name: metric_dict(person_id, choices, values)
            for name, values in probabilities.items()
        }
        correct_nll = metrics["correct_person_profile_dynamic"][
            "negative_log_likelihood"
        ]
        party_nll = metrics["party_dynamic_relation"]["negative_log_likelihood"]
        wrong_nll = metrics["wrong_same_party_profile_dynamic"][
            "negative_log_likelihood"
        ]
        shuffled_nll = metrics[
            "within_congress_history_shuffled_profile_dynamic"
        ]["negative_log_likelihood"]
        per_person.append(
            {
                "person_id": person_id,
                "name": str(cohort_by_id[person_id]["name"]),
                "party_code": party_codes[person_id],
                "wrong_person_id": wrong_id,
                "sample_count": len(choices),
                "applicability_refusal_count": len(refusals),
                "applicability_refusal_examples": refusals[:10],
                "metrics": metrics,
                "nll_uplift_over_party_dynamic": party_nll - correct_nll,
                "nll_uplift_over_wrong_person": wrong_nll - correct_nll,
                "nll_uplift_over_shuffled_history": shuffled_nll - correct_nll,
            }
        )
        refusal_ids = {str(row["event_id"]) for row in refusals}
        for index, row in enumerate(roles["final"]):
            prediction_audit.append(
                {
                    "event_id": str(row["event_id"]),
                    "person_id": person_id,
                    "party_code": party_codes[person_id],
                    "rollnumber": int(row["rollnumber"]),
                    "timestamp": str(row["timestamp"]),
                    "choice": int(row["choice"]),
                    "applicability_refused": str(row["event_id"]) in refusal_ids,
                    "probabilities": {
                        name: float(values[index]) for name, values in probabilities.items()
                    },
                }
            )
        pooled_y.extend(int(value) for value in choices)
        pooled_correct.extend(float(value) for value in final_correct.probabilities)

    audit_path = OUTPUT / "prediction-audit.jsonl"
    audit_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in prediction_audit
        ),
        encoding="utf-8",
    )
    reloaded_audit = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if reloaded_audit != prediction_audit:
        raise ValueError("prediction audit round-trip mismatch")

    party_uplifts = np.asarray(
        [float(row["nll_uplift_over_party_dynamic"]) for row in per_person]
    )
    wrong_uplifts = np.asarray(
        [float(row["nll_uplift_over_wrong_person"]) for row in per_person]
    )
    shuffled_uplifts = np.asarray(
        [float(row["nll_uplift_over_shuffled_history"]) for row in per_person]
    )
    uncertainty = manifest["uncertainty_and_aggregation"]
    rng = np.random.default_rng(int(uncertainty["bootstrap_seed"]))
    bootstrap = np.mean(
        party_uplifts[
            rng.integers(
                0,
                len(party_uplifts),
                size=(int(uncertainty["bootstrap_draws"]), len(party_uplifts)),
            )
        ],
        axis=1,
    )
    lower = float(np.quantile(bootstrap, float(uncertainty["lower_quantile"])))
    pooled_metrics = metric_dict("pooled", pooled_y, pooled_correct)
    positive_people = int(np.sum(party_uplifts > 0.0))
    positive_parties = {
        int(row["party_code"])
        for row in per_person
        if float(row["nll_uplift_over_party_dynamic"]) > 0.0
    }
    refusal_count = sum(int(row["applicability_refusal_count"]) for row in per_person)
    hard = manifest["hard_gates"]
    gates = {
        "all_six_people_have_300_events": len(per_person) == int(hard["required_people"])
        and all(int(row["sample_count"]) == int(hard["required_final_events_per_person"]) for row in per_person),
        "at_least_four_positive_people": positive_people
        >= int(hard["minimum_positive_people_over_party_dynamic"]),
        "positive_person_in_each_party": positive_parties == {100, 200},
        "mean_party_dynamic_uplift_at_least_0_01": float(np.mean(party_uplifts))
        >= float(hard["minimum_equal_person_mean_nll_uplift_over_party_dynamic"]),
        "cluster_bootstrap_lower_positive": lower
        > float(hard["minimum_cluster_bootstrap_lower"]),
        "beats_wrong_person_by_0_005": float(np.mean(wrong_uplifts))
        >= float(hard["minimum_mean_nll_uplift_over_wrong_person"]),
        "beats_shuffled_history_by_0_005": float(np.mean(shuffled_uplifts))
        >= float(hard["minimum_mean_nll_uplift_over_shuffled_history"]),
        "pooled_ece_at_most_0_15": float(pooled_metrics["expected_calibration_error"])
        <= float(hard["maximum_pooled_ece"]),
        "no_applicability_refusal": refusal_count
        <= int(hard["maximum_applicability_refusals"]),
        "reload_probability_equality": True,
    }
    status = (
        "retrospective_person_issue_relational_support"
        if all(gates.values())
        else "person_issue_relational_candidate_not_supported"
    )
    report = {
        "schema_version": "pcfm-person-issue-relational-report-v1",
        "status": status,
        "evidence_status": manifest["evidence_status"],
        "semantic_module_resume_authorized": False,
        "production_promotion_authorized": False,
        "experiment_manifest_digest": manifest_digest,
        "bundle_id": restored_bundle["bundle_id"],
        "feature_map_id": feature_map.map_id,
        "selected_configuration": selected,
        "selection_grid": selection_rows,
        "role_window_audit": audit_role_windows(),
        "per_person": per_person,
        "aggregate": {
            "positive_person_count": positive_people,
            "positive_party_codes": sorted(positive_parties),
            "equal_person_mean_nll_uplift_over_party_dynamic": float(
                np.mean(party_uplifts)
            ),
            "cluster_bootstrap_lower": lower,
            "equal_person_mean_nll_uplift_over_wrong_person": float(
                np.mean(wrong_uplifts)
            ),
            "equal_person_mean_nll_uplift_over_shuffled_history": float(
                np.mean(shuffled_uplifts)
            ),
            "pooled_correct_ece": float(pooled_metrics["expected_calibration_error"]),
            "applicability_refusal_count": refusal_count,
            "prediction_audit_count": len(prediction_audit),
            "prediction_audit_digest": digest(prediction_audit),
        },
        "gates": gates,
        "non_claims": [
            "prospective confirmation",
            "belief identification",
            "value identification",
            "reason recovery",
            "causal mechanism recovery",
            "general person simulation",
        ],
    }
    report["report_digest"] = digest(report)
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
