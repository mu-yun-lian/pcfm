from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcfm.contracts import Observation, Scenario
from pcfm.core import DecisionIntegrator
from pcfm.evaluation import evaluate_probability_array, report_to_dict
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.storage import load_bundle, save_bundle
from pcfm.workflow import (
    fit_person_model,
    load_event_ledger_jsonl,
    load_verification_authority,
    save_event_ledger_jsonl,
)


RAW = REPO_ROOT / "artifacts" / "voteview_real_audit" / "raw"
OUTPUT = REPO_ROOT / "artifacts" / "voteview_real_audit"
VERIFIED_AT = "2026-08-01T12:00:00Z"
VALIDATION_CUTOFF = date(2026, 7, 31)
VERIFIER_ID = "voteview-retrospective-snapshot-v1"
STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "bill",
        "by", "for", "from", "has", "have", "in", "is", "it", "its",
        "motion", "of", "on", "or", "other", "resolution", "senate",
        "that", "the", "this", "to", "united", "was", "were", "with",
    }
)
MOTION_TYPES = (
    "amendment",
    "cloture",
    "nomination",
    "other",
    "other_motion",
    "passage_or_resolution",
)


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a JSON array")
    return [dict(item) for item in value]


def valid_choice(cast_code: str) -> int | None:
    code = int(cast_code)
    if 1 <= code <= 3:
        return 1
    if 4 <= code <= 6:
        return 0
    return None


def motion_type(question: object) -> str:
    text = str(question or "")
    if re.search(r"nomination", text, re.IGNORECASE):
        return "nomination"
    if re.search(r"cloture", text, re.IGNORECASE):
        return "cloture"
    if re.search(r"amendment", text, re.IGNORECASE):
        return "amendment"
    if re.search(
        r"passage|joint resolution|concurrent resolution|resolution",
        text,
        re.IGNORECASE,
    ):
        return "passage_or_resolution"
    if re.search(
        r"motion|proceed|recommit|table",
        text,
        re.IGNORECASE,
    ):
        return "other_motion"
    return "other"


def tokens_for(rollcall: dict[str, object]) -> set[str]:
    text = " ".join(
        str(rollcall.get(name) or "")
        for name in (
            "vote_desc",
            "vote_question",
            "dtl_desc",
            "crs_policy_area",
        )
    ).lower()
    return {
        token
        for token in re.findall(r"[a-z][a-z-]{2,}", text)
        if token not in STOPWORDS
    }


def metric_dict(report) -> dict[str, float | int]:
    return report_to_dict(report)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    required = []
    for congress in range(116, 120):
        required.extend(
            [
                RAW / f"S{congress}_votes.csv",
                RAW / f"S{congress}_members.csv",
                RAW / f"S{congress}_rollcalls.json",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("missing raw Voteview files: " + ", ".join(missing))

    manifest_files = []
    for path in sorted(required):
        manifest_files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_bytes(path.read_bytes()),
                "size": path.stat().st_size,
                "source_url": (
                    "https://voteview.com/static/data/out/"
                    + (
                        "votes/"
                        if "_votes." in path.name
                        else "members/"
                        if "_members." in path.name
                        else "rollcalls/"
                    )
                    + path.name
                ),
            }
        )
    manifest = {
        "schema_version": "voteview-reality-test-raw-manifest-v1",
        "retrieved_at": VERIFIED_AT,
        "files": manifest_files,
    }
    manifest_digest = sha256_bytes(canonical_json(manifest))
    manifest["manifest_digest"] = manifest_digest
    (OUTPUT / "raw-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    votes_by_congress: dict[int, list[dict[str, str]]] = {}
    members_by_congress: dict[int, dict[str, dict[str, str]]] = {}
    rollcalls: dict[tuple[int, int], dict[str, object]] = {}
    for congress in range(116, 120):
        votes_by_congress[congress] = read_csv(
            RAW / f"S{congress}_votes.csv"
        )
        members_by_congress[congress] = {
            row["icpsr"]: row
            for row in read_csv(RAW / f"S{congress}_members.csv")
            if row["chamber"] == "Senate"
        }
        for row in read_json(RAW / f"S{congress}_rollcalls.json"):
            key = (int(row["congress"]), int(row["rollnumber"]))
            if key in rollcalls:
                raise ValueError(f"duplicate roll call {key}")
            rollcalls[key] = row

    vote_rows: dict[int, dict[str, list[dict[str, str]]]] = {}
    for congress, rows in votes_by_congress.items():
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen_trials: set[tuple[str, int]] = set()
        for row in rows:
            choice = valid_choice(row["cast_code"])
            if row["chamber"] != "Senate" or choice is None:
                continue
            trial = (row["icpsr"], int(row["rollnumber"]))
            if trial in seen_trials:
                raise ValueError(f"duplicate member roll-call trial {trial}")
            seen_trials.add(trial)
            grouped[row["icpsr"]].append(row)
        vote_rows[congress] = dict(grouped)

    current_members = members_by_congress[119]
    candidate_rows = []
    for person_id, member in current_members.items():
        fitting = vote_rows[116].get(person_id, []) + vote_rows[117].get(
            person_id, []
        )
        applicability = vote_rows[118].get(person_id, [])
        validation = [
            row
            for row in vote_rows[119].get(person_id, [])
            if date.fromisoformat(
                str(rollcalls[(119, int(row["rollnumber"]))]["date"])
            )
            <= VALIDATION_CUTOFF
        ]
        fitting_choices = [valid_choice(row["cast_code"]) for row in fitting]
        fitting_yea = sum(choice == 1 for choice in fitting_choices)
        fitting_nay = sum(choice == 0 for choice in fitting_choices)
        eligible = (
            len(fitting) >= 500
            and fitting_yea >= 50
            and fitting_nay >= 50
            and len(applicability) >= 100
            and len(validation) >= 100
        )
        candidate_rows.append(
            {
                "person_id": person_id,
                "name": member["bioname"],
                "party_code": int(member["party_code"]),
                "fitting_count": len(fitting),
                "fitting_yea": fitting_yea,
                "fitting_nay": fitting_nay,
                "fitting_minor_count": min(fitting_yea, fitting_nay),
                "applicability_count": len(applicability),
                "validation_count": len(validation),
                "eligible": eligible,
            }
        )
    eligible = [row for row in candidate_rows if row["eligible"]]
    if not eligible:
        raise ValueError("no candidate satisfies the frozen eligibility gate")
    target = sorted(
        eligible,
        key=lambda row: (
            -int(row["fitting_minor_count"]),
            -int(row["fitting_count"]),
            int(row["person_id"]),
        ),
    )[0]
    target_id = str(target["person_id"])

    fitting_counts = {
        row["person_id"]: int(row["fitting_count"])
        for row in candidate_rows
    }
    references = []
    for party_code in (100, 200):
        party_members = [
            row
            for row in candidate_rows
            if row["person_id"] != target_id
            and row["party_code"] == party_code
            and row["fitting_count"] >= 500
        ]
        references.extend(
            sorted(
                party_members,
                key=lambda row: (
                    -int(row["fitting_count"]),
                    int(row["person_id"]),
                ),
            )[:10]
        )
    if len(references) != 20:
        raise ValueError("could not select the frozen 20-person population")
    reference_ids = {str(row["person_id"]) for row in references}
    training_person_ids = reference_ids | {target_id}

    target_rate = int(target["fitting_yea"]) / int(target["fitting_count"])
    wrong_candidates = [
        row
        for row in eligible
        if row["person_id"] != target_id
        and row["party_code"] == target["party_code"]
    ]
    wrong = min(
        wrong_candidates,
        key=lambda row: (
            abs(
                int(row["fitting_yea"]) / int(row["fitting_count"])
                - target_rate
            ),
            int(row["person_id"]),
        ),
    )
    wrong_id = str(wrong["person_id"])

    training_rollcalls = [
        row for key, row in rollcalls.items() if key[0] in (116, 117)
    ]
    policy_areas = tuple(
        sorted(
            {
                str(row.get("crs_policy_area") or "__missing__")
                for row in training_rollcalls
            }
        )
    )
    token_counts = Counter()
    for row in training_rollcalls:
        token_counts.update(tokens_for(row))
    text_tokens = tuple(
        token
        for token, _ in sorted(
            token_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:32]
    )
    feature_names = (
        "intercept",
        *(f"motion:{value}" for value in MOTION_TYPES),
        "has_bill_number",
        "session_two",
        *(f"policy:{value}" for value in policy_areas),
        *(f"token:{value}" for value in text_tokens),
    )

    forbidden = {
        "cast_code",
        "prob",
        "yea_count",
        "nay_count",
        "vote_result",
        "nominate_mid_1",
        "nominate_mid_2",
        "nominate_spread_1",
        "nominate_spread_2",
        "nominate_log_likelihood",
    }
    if any(name in forbidden for name in feature_names):
        raise ValueError("outcome leakage entered the feature schema")

    scenario_cache: dict[tuple[int, int], Scenario] = {}

    def scenario_for(congress: int, rollnumber: int) -> Scenario:
        key = (congress, rollnumber)
        if key in scenario_cache:
            return scenario_cache[key]
        row = rollcalls[key]
        policy = str(row.get("crs_policy_area") or "__missing__")
        tokens = tokens_for(row)
        current_motion = motion_type(row.get("vote_question"))
        values = [1.0]
        values.extend(float(current_motion == value) for value in MOTION_TYPES)
        values.append(float(bool(str(row.get("bill_number") or "").strip())))
        values.append(float(int(row.get("session") or 1) == 2))
        values.extend(float(policy == value) for value in policy_areas)
        values.extend(float(token in tokens) for token in text_tokens)
        scenario = Scenario(
            scenario_id=f"voteview:S:{congress}:{rollnumber}",
            features=tuple(values),
            feature_names=tuple(feature_names),
            options=("Nay", "Yea"),
            domain="us_senate_roll_call",
            context={
                "bill_number": str(row.get("bill_number") or ""),
                "crs_policy_area": policy,
                "date": str(row["date"]),
                "vote_desc": str(row.get("vote_desc") or ""),
                "vote_question": str(row.get("vote_question") or ""),
            },
        )
        if len(scenario.features) != len(feature_names):
            raise ValueError("scenario feature dimension mismatch")
        scenario_cache[key] = scenario
        return scenario

    authority_secret = hashlib.sha256(
        ("voteview-reality-test-v1:" + manifest_digest).encode("utf-8")
    ).hexdigest()
    authority_key = authority_secret.encode("utf-8")
    authority = VerificationAuthority({VERIFIER_ID: authority_key})

    def sign_rows(
        rows: list[tuple[str, dict[str, str]]],
    ) -> EventLedger:
        records = []
        for person_id, vote in rows:
            congress = int(vote["congress"])
            rollnumber = int(vote["rollnumber"])
            rollcall = rollcalls[(congress, rollnumber)]
            observed_at = f"{rollcall['date']}T12:00:00Z"
            observation = Observation(
                person_id=person_id,
                scenario=scenario_for(congress, rollnumber),
                actual_choice=int(valid_choice(vote["cast_code"])),
                provenance="human_record",
            )
            evidence = {
                "vote": vote,
                "rollcall": rollcall,
                "raw_manifest_digest": manifest_digest,
            }
            records.append(
                authority.sign(
                    event_id=(
                        f"voteview:{congress}:S:{rollnumber}:{person_id}"
                    ),
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=sha256_bytes(canonical_json(evidence)),
                    verifier_id=VERIFIER_ID,
                    verified_at=VERIFIED_AT,
                )
            )
        return EventLedger.verify(records, authority)

    training_pairs = []
    for congress in (116, 117):
        for person_id in sorted(training_person_ids, key=int):
            training_pairs.extend(
                (person_id, row)
                for row in vote_rows[congress].get(person_id, [])
            )
    applicability_pairs = [
        (target_id, row) for row in vote_rows[118].get(target_id, [])
    ]
    validation_pairs = [
        (target_id, row)
        for row in vote_rows[119].get(target_id, [])
        if date.fromisoformat(
            str(rollcalls[(119, int(row["rollnumber"]))]["date"])
        )
        <= VALIDATION_CUTOFF
    ]
    training_ledger = sign_rows(training_pairs)
    applicability_ledger = sign_rows(applicability_pairs)
    validation_ledger = sign_rows(validation_pairs)

    bundle = fit_person_model(
        training_ledger,
        authority,
        applicability_ledger=applicability_ledger,
        validation_ledger=validation_ledger,
        person_id=target_id,
        feature_names=tuple(feature_names),
    )

    wrong_bundle = fit_person_model(
        training_ledger,
        authority,
        person_id=wrong_id,
        feature_names=tuple(feature_names),
    )
    integrator = DecisionIntegrator()
    observations = validation_ledger.observations()
    personal_probabilities = [
        integrator.predict(
            observation.scenario,
            bundle.population_model,
            bundle.adapter,
            parameter_covariance=bundle.representation.covariance,
        ).probability_option_1
        for observation in observations
    ]
    population_probabilities = [
        integrator.predict_population(
            observation.scenario,
            bundle.population_model,
        ).probability_option_1
        for observation in observations
    ]
    wrong_probabilities = [
        integrator.predict(
            observation.scenario,
            wrong_bundle.population_model,
            wrong_bundle.adapter,
            parameter_covariance=wrong_bundle.representation.covariance,
        ).probability_option_1
        for observation in observations
    ]
    target_training = [
        record.observation
        for record in training_ledger.records_for_person(target_id)
    ]
    frequency = sum(item.actual_choice for item in target_training) / len(
        target_training
    )
    frequency_probabilities = [frequency] * len(observations)

    def metrics_for(split_observations) -> dict[str, dict[str, float | int]]:
        split_personal = [
            integrator.predict(
                observation.scenario,
                bundle.population_model,
                bundle.adapter,
                parameter_covariance=bundle.representation.covariance,
            ).probability_option_1
            for observation in split_observations
        ]
        split_population = [
            integrator.predict_population(
                observation.scenario,
                bundle.population_model,
            ).probability_option_1
            for observation in split_observations
        ]
        split_wrong = [
            integrator.predict(
                observation.scenario,
                wrong_bundle.population_model,
                wrong_bundle.adapter,
                parameter_covariance=wrong_bundle.representation.covariance,
            ).probability_option_1
            for observation in split_observations
        ]
        split_frequency = [frequency] * len(split_observations)
        return {
            "personal_map_logistic": metric_dict(
                evaluate_probability_array(
                    split_observations, split_personal
                )
            ),
            "population_logistic": metric_dict(
                evaluate_probability_array(
                    split_observations, split_population
                )
            ),
            "wrong_person_logistic": metric_dict(
                evaluate_probability_array(split_observations, split_wrong)
            ),
            "training_frequency": metric_dict(
                evaluate_probability_array(
                    split_observations, split_frequency
                )
            ),
        }

    metrics = metrics_for(observations)
    split_metrics = {
        "fitting_target_in_sample": metrics_for(tuple(target_training)),
        "applicability_congress_118": metrics_for(
            applicability_ledger.observations()
        ),
        "validation_congress_119": metrics,
    }
    result = {
        "schema_version": "voteview-reality-test-report-v1",
        "status": (
            "retrospective_real_choice_support"
            if bundle.manifest.validation.status == "passed"
            else "retrospective_no_support"
        ),
        "interpretation": "structured_roll_call_choice_prediction_only",
        "raw_manifest_digest": manifest_digest,
        "target": target,
        "wrong_person": wrong,
        "population_reference_ids": sorted(reference_ids, key=int),
        "eligible_candidate_count": len(eligible),
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "policy_areas": list(policy_areas),
        "text_tokens": list(text_tokens),
        "role_counts": {
            "training": len(training_ledger.records),
            "target_training": len(target_training),
            "applicability": len(applicability_ledger.records),
            "validation": len(validation_ledger.records),
        },
        "metrics": metrics,
        "split_metrics": split_metrics,
        "existing_model_validation": {
            "status": bundle.manifest.validation.status,
            "reasons": list(bundle.manifest.validation.reasons),
            "sample_count": bundle.manifest.validation.sample_count,
            "personal_nll": bundle.manifest.validation.personal_nll,
            "population_nll": bundle.manifest.validation.population_nll,
            "nll_uplift": bundle.manifest.validation.nll_uplift,
            "nll_uplift_ci_lower": (
                bundle.manifest.validation.nll_uplift_ci_lower
            ),
            "nll_uplift_ci_upper": (
                bundle.manifest.validation.nll_uplift_ci_upper
            ),
            "calibration_error": (
                bundle.manifest.validation.calibration_error
            ),
            "mechanism_probe_nll_uplift": (
                bundle.manifest.validation.mechanism_probe_nll_uplift
            ),
            "temporal_stability_status": (
                bundle.manifest.validation.temporal_stability_status
            ),
            "temporal_drift_detected": (
                bundle.manifest.validation.temporal_drift_detected
            ),
        },
        "non_claims": [
            "belief identification",
            "reasoning-process recovery",
            "general cognitive simulation",
            "Tyler Cowen validity",
            "untouched prospective confirmation",
        ],
    }
    candidates_path = OUTPUT / "candidate-eligibility.csv"
    with candidates_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(candidate_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            sorted(
                candidate_rows,
                key=lambda row: (
                    not bool(row["eligible"]),
                    -int(row["fitting_minor_count"]),
                    int(row["person_id"]),
                ),
            )
        )
    save_event_ledger_jsonl(OUTPUT / "training-ledger.jsonl", training_ledger)
    save_event_ledger_jsonl(
        OUTPUT / "applicability-ledger.jsonl", applicability_ledger
    )
    save_event_ledger_jsonl(
        OUTPUT / "validation-ledger.jsonl", validation_ledger
    )
    save_bundle(OUTPUT / "person-model.json", bundle)
    (OUTPUT / "verification-keys.json").write_text(
        json.dumps(
            {VERIFIER_ID: authority_secret},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    reloaded_authority = load_verification_authority(
        OUTPUT / "verification-keys.json"
    )
    reloaded_training = load_event_ledger_jsonl(
        OUTPUT / "training-ledger.jsonl", reloaded_authority
    )
    reloaded_applicability = load_event_ledger_jsonl(
        OUTPUT / "applicability-ledger.jsonl", reloaded_authority
    )
    reloaded_validation = load_event_ledger_jsonl(
        OUTPUT / "validation-ledger.jsonl", reloaded_authority
    )
    reloaded_bundle = load_bundle(OUTPUT / "person-model.json")
    round_trip_mismatches = []
    if EventLedger.snapshot_hash(
        reloaded_training.records
    ) != EventLedger.snapshot_hash(training_ledger.records):
        round_trip_mismatches.append("training_ledger")
    if EventLedger.snapshot_hash(
        reloaded_applicability.records
    ) != EventLedger.snapshot_hash(applicability_ledger.records):
        round_trip_mismatches.append("applicability_ledger")
    if EventLedger.snapshot_hash(
        reloaded_validation.records
    ) != EventLedger.snapshot_hash(validation_ledger.records):
        round_trip_mismatches.append("validation_ledger")
    sample_scenario = observations[0].scenario
    original_probability = integrator.predict(
        sample_scenario,
        bundle.population_model,
        bundle.adapter,
        parameter_covariance=bundle.representation.covariance,
    ).probability_option_1
    reloaded_probability = integrator.predict(
        sample_scenario,
        reloaded_bundle.population_model,
        reloaded_bundle.adapter,
        parameter_covariance=reloaded_bundle.representation.covariance,
    ).probability_option_1
    if (
        reloaded_bundle.manifest.model_id != bundle.manifest.model_id
        or abs(reloaded_probability - original_probability) > 1e-12
    ):
        round_trip_mismatches.append("person_model_behavior")
    if round_trip_mismatches:
        raise ValueError(
            "saved experiment artifact round trip mismatch: "
            + ", ".join(round_trip_mismatches)
        )
    result["artifact_round_trip_verified"] = True
    result["report_digest"] = sha256_bytes(canonical_json(result))
    (OUTPUT / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
