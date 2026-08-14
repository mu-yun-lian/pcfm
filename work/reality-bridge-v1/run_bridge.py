from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcfm.composite import (
    CompositeModelRefusedError,
    create_composite_model,
    load_composite_model,
    save_composite_model,
)
from pcfm.contracts import Observation, Scenario
from pcfm.core import DecisionIntegrator
from pcfm.dynamic_state import (
    DynamicStateRefusedError,
    create_dynamic_state_plan,
    infer_dynamic_state,
    save_dynamic_state_report,
)
from pcfm.evaluation import evaluate_probability_array, report_to_dict
from pcfm.ledger import EventLedger, VerificationAuthority
from pcfm.mechanism import (
    EvidenceWindow,
    MechanismHypothesis,
    MechanismRefusedError,
    MechanismTerm,
    compare_mechanisms,
    create_mechanism_comparison_plan,
    save_mechanism_comparison_plan,
    save_mechanism_comparison_report,
)
from pcfm.storage import load_bundle, save_bundle
from pcfm.workflow import fit_person_model


RAW = REPO_ROOT / "artifacts" / "voteview_real_audit" / "raw"
RAW_MANIFEST = (
    REPO_ROOT / "artifacts" / "voteview_real_audit" / "raw-manifest.json"
)
OUTPUT = REPO_ROOT / "artifacts" / "reality_bridge_v1"
VERIFIER_ID = "voteview-counterfactual-replay-v1"
VERIFIED_AT = "2026-08-01T15:00:00Z"
TASK_CONTEXT = {
    "institution": "us_senate",
    "task": "roll_call_vote",
}
ROLE_COUNTS = {
    "applicability": 100,
    "dynamic": 80,
    "mechanism_discovery": 100,
    "mechanism_selection": 100,
    "mechanism_confirmation": 100,
}
MOTION_TYPES = (
    "amendment",
    "cloture",
    "nomination",
    "other",
    "other_motion",
    "passage_or_resolution",
)
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "bill",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "motion",
        "of",
        "on",
        "or",
        "other",
        "resolution",
        "senate",
        "that",
        "the",
        "this",
        "to",
        "united",
        "was",
        "were",
        "with",
    }
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
    if re.search(r"motion|proceed|recommit|table", text, re.IGNORECASE):
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


def ordered_timestamp(rollcall: dict[str, object]) -> str:
    observed = datetime.combine(
        date.fromisoformat(str(rollcall["date"])),
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(hours=12, seconds=10 * int(rollcall["rollnumber"]))
    return observed.isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def verify_raw_manifest() -> tuple[dict[str, object], str]:
    manifest = json.loads(RAW_MANIFEST.read_text(encoding="utf-8"))
    recorded_digest = str(manifest.pop("manifest_digest"))
    actual_digest = sha256_bytes(canonical_json(manifest))
    if actual_digest != recorded_digest:
        raise ValueError("raw manifest self-digest mismatch")
    for item in manifest["files"]:
        path = REPO_ROOT / str(item["path"])
        if not path.exists():
            raise ValueError(f"missing raw file {path}")
        if sha256_bytes(path.read_bytes()) != item["sha256"]:
            raise ValueError(f"raw digest drift for {path.name}")
        if path.stat().st_size != int(item["size"]):
            raise ValueError(f"raw size drift for {path.name}")
    manifest["manifest_digest"] = recorded_digest
    return manifest, recorded_digest


def select_cohort(
    eligible: Sequence[dict[str, object]],
    manifest_digest: str,
) -> tuple[dict[str, object], ...]:
    selected = []
    for party_code in (100, 200):
        party = [row for row in eligible if row["party_code"] == party_code]
        ordered = sorted(
            party,
            key=lambda row: (
                hashlib.sha256(
                    (
                        "reality-bridge-v1:"
                        + manifest_digest
                        + ":"
                        + str(row["person_id"])
                    ).encode("utf-8")
                ).digest(),
                int(str(row["person_id"])),
            ),
        )
        if len(ordered) < 3:
            raise ValueError(f"party {party_code} has fewer than 3 candidates")
        selected.extend(ordered[:3])
    return tuple(selected)


def allocate_roles(
    rows: Sequence[dict[str, str]],
) -> dict[str, tuple[dict[str, str], ...]]:
    ordered = tuple(sorted(rows, key=lambda row: int(row["rollnumber"])))
    result: dict[str, tuple[dict[str, str], ...]] = {}
    cursor = 0
    for role, count in ROLE_COUNTS.items():
        result[role] = ordered[cursor : cursor + count]
        cursor += count
    if any(len(result[role]) != count for role, count in ROLE_COUNTS.items()):
        raise ValueError("insufficient Congress 119 role records")
    allocated = [
        (int(row["congress"]), int(row["rollnumber"]))
        for role_rows in result.values()
        for row in role_rows
    ]
    if len(allocated) != len(set(allocated)):
        raise ValueError("Congress 119 evidence roles overlap")
    return result


def metric_dict(report: object) -> dict[str, float | int]:
    return report_to_dict(report)


def build_hypotheses(
    feature_names: tuple[str, ...],
    policy_names: tuple[str, ...],
    token_names: tuple[str, ...],
) -> tuple[MechanismHypothesis, ...]:
    nonconstant = tuple(name for name in feature_names if name != "intercept")
    return (
        MechanismHypothesis(
            "intercept-shift",
            (MechanismTerm("intercept", "intercept", ()),),
        ),
        MechanismHypothesis(
            "limited-linear-residual",
            tuple(
                MechanismTerm(f"linear-{index}", "linear", (name,))
                for index, name in enumerate(nonconstant[:15])
            ),
        ),
        MechanismHypothesis(
            "bill-by-motion",
            (
                MechanismTerm("intercept", "intercept", ()),
                *(
                    MechanismTerm(
                        f"bill-motion-{index}",
                        "interaction",
                        ("has_bill_number", motion),
                    )
                    for index, motion in enumerate(
                        name
                        for name in feature_names
                        if name.startswith("motion:")
                    )
                ),
            ),
        ),
        MechanismHypothesis(
            "policy-by-token",
            (
                MechanismTerm("intercept", "intercept", ()),
                *(
                    MechanismTerm(
                        f"policy-token-{index}",
                        "interaction",
                        pair,
                    )
                    for index, pair in enumerate(
                        zip(policy_names, token_names, strict=False)
                    )
                ),
            ),
        ),
    )


def refusal(error: Exception) -> dict[str, object]:
    return {
        "status": "refused",
        "reasons": list(getattr(error, "reasons", (str(error),))),
    }


def main(*, prepare_only: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest, manifest_digest = verify_raw_manifest()

    members: dict[int, dict[str, dict[str, str]]] = {}
    vote_rows: dict[int, dict[str, list[dict[str, str]]]] = {}
    rollcalls: dict[tuple[int, int], dict[str, object]] = {}
    for congress in range(116, 120):
        members[congress] = {
            row["icpsr"]: row
            for row in read_csv(RAW / f"S{congress}_members.csv")
            if row["chamber"] == "Senate"
        }
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        seen: set[tuple[str, int]] = set()
        for row in read_csv(RAW / f"S{congress}_votes.csv"):
            if row["chamber"] != "Senate" or valid_choice(row["cast_code"]) is None:
                continue
            trial = (row["icpsr"], int(row["rollnumber"]))
            if trial in seen:
                raise ValueError(f"duplicate member roll-call trial {trial}")
            seen.add(trial)
            grouped[row["icpsr"]].append(row)
        vote_rows[congress] = dict(grouped)
        for row in read_json(RAW / f"S{congress}_rollcalls.json"):
            key = (congress, int(row["rollnumber"]))
            if key in rollcalls:
                raise ValueError(f"duplicate roll call {key}")
            rollcalls[key] = row

    common_ids = set(members[119])
    for congress in range(116, 119):
        common_ids &= set(members[congress])
    candidates = []
    total_role_count = sum(ROLE_COUNTS.values())
    for person_id in sorted(common_ids, key=int):
        party_code = int(members[119][person_id]["party_code"])
        fitting = vote_rows[116].get(person_id, [])
        validation = vote_rows[117].get(person_id, [])
        future = sorted(
            vote_rows[119].get(person_id, []),
            key=lambda row: int(row["rollnumber"]),
        )
        fitting_choices = [valid_choice(row["cast_code"]) for row in fitting]
        fitting_yea = sum(choice == 1 for choice in fitting_choices)
        fitting_nay = sum(choice == 0 for choice in fitting_choices)
        enough_future = len(future) >= total_role_count
        within_age = False
        if enough_future:
            roles = allocate_roles(future)
            app_end = parse_timestamp(
                ordered_timestamp(
                    rollcalls[(119, int(roles["applicability"][-1]["rollnumber"]))]
                )
            )
            evidence_end = parse_timestamp(
                ordered_timestamp(
                    rollcalls[
                        (
                            119,
                            int(roles["mechanism_confirmation"][-1]["rollnumber"]),
                        )
                    ]
                )
            )
            within_age = (evidence_end - app_end).total_seconds() <= 180 * 86400
        eligible = (
            party_code in (100, 200)
            and len(fitting) >= 500
            and fitting_yea >= 50
            and fitting_nay >= 50
            and len(validation) >= 500
            and enough_future
            and within_age
        )
        candidates.append(
            {
                "person_id": person_id,
                "name": members[119][person_id]["bioname"],
                "party_code": party_code,
                "fitting_count": len(fitting),
                "fitting_yea": fitting_yea,
                "fitting_nay": fitting_nay,
                "validation_count": len(validation),
                "future_count": len(future),
                "future_roles_within_180_days": within_age,
                "eligible": eligible,
            }
        )
    eligible = tuple(row for row in candidates if row["eligible"])
    cohort = select_cohort(eligible, manifest_digest)
    cohort_ids = {str(row["person_id"]) for row in cohort}

    references = []
    for party_code in (100, 200):
        references.extend(
            sorted(
                (row for row in eligible if row["party_code"] == party_code),
                key=lambda row: (-int(row["fitting_count"]), int(row["person_id"])),
            )[:10]
        )
    reference_ids = {str(row["person_id"]) for row in references}
    training_ids = cohort_ids | reference_ids

    training_rollcalls = [
        row for (congress, _), row in rollcalls.items() if congress == 116
    ]
    policy_counts = Counter(
        str(row.get("crs_policy_area") or "__missing__")
        for row in training_rollcalls
    )
    top_policies = tuple(
        name
        for name, _ in sorted(
            policy_counts.items(), key=lambda item: (-item[1], item[0])
        )[:4]
    )
    token_counts = Counter()
    for row in training_rollcalls:
        token_counts.update(tokens_for(row))
    text_tokens = tuple(
        token
        for token, _ in sorted(
            token_counts.items(), key=lambda item: (-item[1], item[0])
        )[:6]
    )
    policy_features = tuple(f"policy:{value}" for value in top_policies) + (
        "policy:__other__",
    )
    token_features = tuple(f"token:{value}" for value in text_tokens)
    feature_names = (
        "intercept",
        *(f"motion:{value}" for value in MOTION_TYPES),
        "has_bill_number",
        "session_two",
        *policy_features,
        *token_features,
    )
    if len(feature_names) != 20:
        raise ValueError("frozen feature budget must equal 20")
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
    if forbidden & set(feature_names):
        raise ValueError("outcome leakage entered the feature schema")

    cohort_plan = {
        "schema_version": "pcfm-reality-bridge-cohort-v1",
        "evidence_status": "counterfactual_historical_replay",
        "raw_manifest_digest": manifest_digest,
        "eligible_count": len(eligible),
        "cohort": list(cohort),
        "population_reference_ids": sorted(reference_ids, key=int),
        "feature_names": list(feature_names),
        "top_policies": list(top_policies),
        "text_tokens": list(text_tokens),
        "role_counts": ROLE_COUNTS,
    }
    cohort_plan["cohort_plan_digest"] = sha256_bytes(canonical_json(cohort_plan))
    (OUTPUT / "cohort-plan.json").write_text(
        json.dumps(cohort_plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT / "candidate-eligibility.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    if prepare_only:
        print(json.dumps(cohort_plan, ensure_ascii=False, indent=2, sort_keys=True))
        return

    scenario_cache: dict[tuple[int, int], Scenario] = {}

    def scenario_for(congress: int, rollnumber: int) -> Scenario:
        key = (congress, rollnumber)
        if key in scenario_cache:
            return scenario_cache[key]
        row = rollcalls[key]
        policy = str(row.get("crs_policy_area") or "__missing__")
        current_motion = motion_type(row.get("vote_question"))
        row_tokens = tokens_for(row)
        values = [1.0]
        values.extend(float(current_motion == value) for value in MOTION_TYPES)
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
            context=TASK_CONTEXT,
        )
        scenario_cache[key] = scenario
        return scenario

    authority_secret = hashlib.sha256(
        ("reality-bridge-v1:" + manifest_digest).encode("utf-8")
    ).hexdigest()
    authority = VerificationAuthority({VERIFIER_ID: authority_secret.encode("utf-8")})

    def sign_rows(
        pairs: Sequence[tuple[str, dict[str, str]]],
    ) -> EventLedger:
        records = []
        for person_id, vote in pairs:
            congress = int(vote["congress"])
            rollnumber = int(vote["rollnumber"])
            rollcall = rollcalls[(congress, rollnumber)]
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
                "timestamp_semantics": "date_plus_rollnumber_order_encoding",
            }
            records.append(
                authority.sign(
                    event_id=f"bridge:{congress}:S:{rollnumber}:{person_id}",
                    observation=observation,
                    observed_at=ordered_timestamp(rollcall),
                    evidence_hash=sha256_bytes(canonical_json(evidence)),
                    verifier_id=VERIFIER_ID,
                    verified_at=VERIFIED_AT,
                )
            )
        return EventLedger.verify(tuple(records), authority)

    training_pairs = [
        (person_id, row)
        for person_id in sorted(training_ids, key=int)
        for row in sorted(
            vote_rows[116].get(person_id, []),
            key=lambda item: int(item["rollnumber"]),
        )
    ]
    training_ledger = sign_rows(training_pairs)
    models_dir = OUTPUT / "models"
    models_dir.mkdir(exist_ok=True)
    plans_dir = OUTPUT / "plans"
    plans_dir.mkdir(exist_ok=True)
    reports_dir = OUTPUT / "reports"
    reports_dir.mkdir(exist_ok=True)

    roles_by_person: dict[str, dict[str, tuple[dict[str, str], ...]]] = {}
    bundles = {}
    target_ledgers = {}
    base_rows = []
    for target in cohort:
        person_id = str(target["person_id"])
        roles = allocate_roles(vote_rows[119][person_id])
        roles_by_person[person_id] = roles
        applicability_ledger = sign_rows(
            [(person_id, row) for row in roles["applicability"]]
        )
        validation_ledger = sign_rows(
            [
                (person_id, row)
                for row in sorted(
                    vote_rows[117][person_id],
                    key=lambda item: int(item["rollnumber"]),
                )
            ]
        )
        bundle = fit_person_model(
            training_ledger,
            authority,
            applicability_ledger=applicability_ledger,
            validation_ledger=validation_ledger,
            person_id=person_id,
            feature_names=feature_names,
        )
        bundles[person_id] = bundle
        target_ledgers[person_id] = {
            "applicability": applicability_ledger,
            "validation": validation_ledger,
        }
        model_path = models_dir / f"{person_id}.json"
        save_bundle(model_path, bundle)
        reloaded = load_bundle(model_path)
        if reloaded.manifest.model_id != bundle.manifest.model_id:
            raise ValueError(f"model round trip failed for {person_id}")
        base_rows.append(
            {
                "person_id": person_id,
                "name": target["name"],
                "party_code": target["party_code"],
                "validation_status": bundle.manifest.validation.status,
                "validation_reasons": list(bundle.manifest.validation.reasons),
                "personal_nll": bundle.manifest.validation.personal_nll,
                "population_nll": bundle.manifest.validation.population_nll,
                "nll_uplift": bundle.manifest.validation.nll_uplift,
                "nll_uplift_ci_lower": (
                    bundle.manifest.validation.nll_uplift_ci_lower
                ),
                "calibration_error": bundle.manifest.validation.calibration_error,
                "temporal_status": (
                    bundle.manifest.validation.temporal_stability_status
                ),
            }
        )

    cohort_by_id = {str(row["person_id"]): row for row in cohort}
    integrator = DecisionIntegrator()
    for row in base_rows:
        person_id = str(row["person_id"])
        same_party = [
            str(candidate["person_id"])
            for candidate in cohort
            if candidate["party_code"] == row["party_code"]
            and str(candidate["person_id"]) != person_id
        ]
        target_rate = int(cohort_by_id[person_id]["fitting_yea"]) / int(
            cohort_by_id[person_id]["fitting_count"]
        )
        wrong_id = min(
            same_party,
            key=lambda candidate_id: (
                abs(
                    int(cohort_by_id[candidate_id]["fitting_yea"])
                    / int(cohort_by_id[candidate_id]["fitting_count"])
                    - target_rate
                ),
                int(candidate_id),
            ),
        )
        bundle = bundles[person_id]
        wrong_bundle = bundles[wrong_id]
        observations = target_ledgers[person_id]["validation"].observations()
        personal = [
            integrator.predict(
                observation.scenario,
                bundle.population_model,
                bundle.adapter,
                parameter_covariance=bundle.representation.covariance,
            ).probability_option_1
            for observation in observations
        ]
        population = [
            integrator.predict_population(
                observation.scenario, bundle.population_model
            ).probability_option_1
            for observation in observations
        ]
        wrong = [
            integrator.predict(
                observation.scenario,
                wrong_bundle.population_model,
                wrong_bundle.adapter,
                parameter_covariance=wrong_bundle.representation.covariance,
            ).probability_option_1
            for observation in observations
        ]
        target_training = training_ledger.records_for_person(person_id)
        frequency = sum(
            record.observation.actual_choice for record in target_training
        ) / len(target_training)
        constant = [frequency] * len(observations)
        row["wrong_person_id"] = wrong_id
        row["metrics"] = {
            "personal": metric_dict(evaluate_probability_array(observations, personal)),
            "population": metric_dict(
                evaluate_probability_array(observations, population)
            ),
            "wrong_person": metric_dict(
                evaluate_probability_array(observations, wrong)
            ),
            "training_frequency": metric_dict(
                evaluate_probability_array(observations, constant)
            ),
        }

    hypotheses = build_hypotheses(feature_names, policy_features, token_features)
    downstream = []
    for target in cohort:
        person_id = str(target["person_id"])
        bundle = bundles[person_id]
        roles = roles_by_person[person_id]
        app_ledger = target_ledgers[person_id]["applicability"]
        app_end = parse_timestamp(app_ledger.records[-1].observed_at)
        counterfactual_registered = (app_end + timedelta(seconds=1)).isoformat().replace(
            "+00:00", "Z"
        )
        person_result: dict[str, object] = {
            "person_id": person_id,
            "evidence_status": "counterfactual_historical_replay",
        }

        dynamic_ledger = sign_rows(
            [(person_id, row) for row in roles["dynamic"]]
        )
        if bundle.manifest.validation.status == "passed":
            try:
                dynamic_plan = create_dynamic_state_plan(
                    bundle,
                    authority,
                    verifier_id=VERIFIER_ID,
                    registered_at=counterfactual_registered,
                    monitoring_start_at=dynamic_ledger.records[0].observed_at,
                    monitoring_end_at=dynamic_ledger.records[-1].observed_at,
                    expected_event_count=len(dynamic_ledger.records),
                )
                dynamic_report = infer_dynamic_state(
                    bundle, dynamic_ledger, authority, dynamic_plan
                )
                (plans_dir / f"{person_id}-dynamic-plan.json").write_text(
                    json.dumps(
                        dynamic_plan.to_dict(),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                save_dynamic_state_report(
                    reports_dir / f"{person_id}-dynamic-report.json",
                    dynamic_report,
                )
                person_result["dynamic"] = {
                    "status": dynamic_report.status,
                    "reasons": list(dynamic_report.reasons),
                    "static_nll": dynamic_report.static_nll,
                    "dynamic_prequential_nll": (
                        dynamic_report.dynamic_prequential_nll
                    ),
                    "nll_uplift": dynamic_report.mean_nll_uplift,
                    "maximum_log_e_value": dynamic_report.maximum_log_e_value,
                }
            except DynamicStateRefusedError as error:
                person_result["dynamic"] = refusal(error)
        else:
            person_result["dynamic"] = {
                "status": "blocked",
                "reasons": [
                    f"base_model_validation_{bundle.manifest.validation.status}"
                ],
            }

        discovery = sign_rows(
            [(person_id, row) for row in roles["mechanism_discovery"]]
        )
        selection = sign_rows(
            [(person_id, row) for row in roles["mechanism_selection"]]
        )
        confirmation = sign_rows(
            [(person_id, row) for row in roles["mechanism_confirmation"]]
        )
        try:
            mechanism_plan = create_mechanism_comparison_plan(
                bundle,
                hypotheses,
                authority,
                verifier_id=VERIFIER_ID,
                registered_at=counterfactual_registered,
                discovery_window=EvidenceWindow(
                    discovery.records[0].observed_at,
                    discovery.records[-1].observed_at,
                    len(discovery.records),
                ),
                selection_window=EvidenceWindow(
                    selection.records[0].observed_at,
                    selection.records[-1].observed_at,
                    len(selection.records),
                ),
                confirmation_window=EvidenceWindow(
                    confirmation.records[0].observed_at,
                    confirmation.records[-1].observed_at,
                    len(confirmation.records),
                ),
            )
            mechanism_report = compare_mechanisms(
                bundle,
                mechanism_plan,
                discovery,
                selection,
                confirmation,
                authority,
            )
            save_mechanism_comparison_plan(
                plans_dir / f"{person_id}-mechanism-plan.json", mechanism_plan
            )
            save_mechanism_comparison_report(
                reports_dir / f"{person_id}-mechanism-report.json",
                mechanism_report,
            )
            person_result["mechanism"] = {
                "status": mechanism_report.status,
                "reasons": list(mechanism_report.reasons),
                "selected_hypothesis_id": (
                    mechanism_report.selected_hypothesis_id
                ),
                "base_confirmation_nll": mechanism_report.base_confirmation_nll,
                "candidate_confirmation_nll": (
                    mechanism_report.candidate_confirmation_nll
                ),
                "confirmation_nll_uplift": (
                    mechanism_report.confirmation_nll_uplift
                ),
                "confirmation_nll_uplift_ci_lower": (
                    mechanism_report.confirmation_nll_uplift_ci_lower
                ),
                "confirmation_calibration_error": (
                    mechanism_report.confirmation_calibration_error
                ),
            }
            if mechanism_report.status == "supported_candidate":
                composite = create_composite_model(
                    bundle,
                    mechanism_plan,
                    mechanism_report,
                    discovery,
                    selection,
                    confirmation,
                    authority,
                    verifier_id=VERIFIER_ID,
                    created_at=confirmation.records[-1].observed_at,
                )
                composite_path = models_dir / f"{person_id}-composite.json"
                save_composite_model(composite_path, composite)
                reloaded_composite = load_composite_model(
                    composite_path, authority
                )
                if (
                    reloaded_composite.composite_model_id
                    != composite.composite_model_id
                ):
                    raise ValueError(
                        f"composite round trip failed for {person_id}"
                    )
                person_result["composite"] = {
                    "status": "created",
                    "composite_model_id": composite.composite_model_id,
                }
            else:
                person_result["composite"] = {
                    "status": "not_created",
                    "reasons": ["mechanism_candidate_not_supported"],
                }
        except (MechanismRefusedError, CompositeModelRefusedError) as error:
            person_result["mechanism"] = refusal(error)
            person_result["composite"] = {
                "status": "blocked",
                "reasons": list(getattr(error, "reasons", (str(error),))),
            }
        downstream.append(person_result)

    passed_count = sum(row["validation_status"] == "passed" for row in base_rows)
    bridge_status = (
        "core_entry_not_established"
        if passed_count == 0
        else (
            "limited_structured_domain_entry"
            if passed_count <= 3
            else "structured_domain_entry"
        )
    )
    result = {
        "schema_version": "pcfm-reality-bridge-report-v1",
        "status": bridge_status,
        "evidence_status": "counterfactual_historical_replay",
        "semantic_module_resume_authorized": False,
        "raw_manifest_digest": manifest_digest,
        "cohort_plan_digest": cohort_plan["cohort_plan_digest"],
        "feature_names": list(feature_names),
        "cohort_size": len(cohort),
        "base_validation_passed_count": passed_count,
        "base_results": base_rows,
        "downstream_results": downstream,
        "known_integration_gap": (
            "dynamic_state_and_composite_model_not_jointly_deployable"
        ),
        "non_claims": [
            "prospective confirmation",
            "belief identification",
            "value identification",
            "causal mechanism recovery",
            "general person simulation",
        ],
    }
    result["report_digest"] = sha256_bytes(canonical_json(result))
    (OUTPUT / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(prepare_only=args.prepare_only)
