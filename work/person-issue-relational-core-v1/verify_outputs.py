from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_experiment as experiment


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def verify() -> dict[str, object]:
    manifest, cohort, manifest_digest = experiment.load_bound_inputs()
    members, vote_rows, rollcalls = experiment.load_voteview()
    report_path = experiment.OUTPUT / "report.json"
    bundle_path = experiment.OUTPUT / "selected-model.json"
    audit_path = experiment.OUTPUT / "prediction-audit.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_unsigned = dict(report)
    recorded_report_digest = str(report_unsigned.pop("report_digest"))
    if experiment.digest(report_unsigned) != recorded_report_digest:
        raise ValueError("report identity mismatch")
    if report["experiment_manifest_digest"] != manifest_digest:
        raise ValueError("report manifest lineage mismatch")
    full, party, shuffled, bundle = experiment._load_bundle(bundle_path)
    if report["bundle_id"] != bundle["bundle_id"]:
        raise ValueError("report bundle lineage mismatch")
    audit = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(audit) != 1800 or experiment.digest(audit) != report["aggregate"][
        "prediction_audit_digest"
    ]:
        raise ValueError("prediction audit identity mismatch")
    by_person = {
        person_id: [row for row in audit if row["person_id"] == person_id]
        for person_id in manifest["cohort_person_ids"]
    }
    party_codes = {
        person_id: int(members[119][person_id]["party_code"])
        for person_id in manifest["cohort_person_ids"]
    }
    config = experiment._state_config(bundle["selected_configuration"])
    report_people = {row["person_id"]: row for row in report["per_person"]}
    pooled_choices = []
    pooled_correct = []
    for person_id in manifest["cohort_person_ids"]:
        source = vote_rows[119][person_id]
        warmup = [
            experiment._record(person_id, 119, row, rollcalls, party_codes[person_id])
            for row in source[:450]
        ]
        applicability = [
            experiment._record(person_id, 119, row, rollcalls, party_codes[person_id])
            for row in source[450:530]
        ]
        final = [
            experiment._record(person_id, 119, row, rollcalls, party_codes[person_id])
            for row in source[530:830]
        ]
        wrong_id = str(manifest["wrong_person_mapping"][person_id])
        warm_correct = experiment._run_rows(full, person_id, warmup, config)
        warm_party = experiment._run_rows(
            party, person_id, warmup, config, include_person=False
        )
        warm_wrong = experiment._run_rows(
            full, person_id, warmup, config, profile_person_id=wrong_id
        )
        warm_shuffled = experiment._run_rows(shuffled, person_id, warmup, config)
        correct = experiment._run_rows(
            full, person_id, final, config, initial=warm_correct
        ).probabilities
        party_dynamic = experiment._run_rows(
            party,
            person_id,
            final,
            config,
            include_person=False,
            initial=warm_party,
        ).probabilities
        wrong = experiment._run_rows(
            full,
            person_id,
            final,
            config,
            profile_person_id=wrong_id,
            initial=warm_wrong,
        ).probabilities
        shuffled_probabilities = experiment._run_rows(
            shuffled, person_id, final, config, initial=warm_shuffled
        ).probabilities
        static_party = party.probabilities(
            person_id,
            [row["rollcall"] for row in final],
            party_codes[person_id],
            119,
            include_person=False,
        )
        static_correct = full.probabilities(
            person_id,
            [row["rollcall"] for row in final],
            party_codes[person_id],
            119,
        )
        recomputed = {
            "party_static_relation": static_party,
            "correct_profile_static": static_correct,
            "party_dynamic_relation": party_dynamic,
            "wrong_same_party_profile_dynamic": wrong,
            "within_congress_history_shuffled_profile_dynamic": shuffled_probabilities,
            "correct_person_profile_dynamic": correct,
        }
        person_audit = by_person[person_id]
        if [row["event_id"] for row in person_audit] != [row["event_id"] for row in final]:
            raise ValueError(f"prediction audit event order mismatch for {person_id}")
        for name, values in recomputed.items():
            recorded = np.asarray(
                [row["probabilities"][name] for row in person_audit], dtype=np.float64
            )
            if not np.allclose(values, recorded, rtol=0.0, atol=1e-12):
                raise ValueError(f"reloaded probability mismatch for {person_id} {name}")
            metrics = experiment.metric_dict(
                person_id, [row["choice"] for row in final], values
            )
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not _close(
                    value, report_people[person_id]["metrics"][name][key]
                ):
                    raise ValueError(f"metric mismatch for {person_id} {name} {key}")
        refusals = experiment._applicability_refusals(
            full.feature_map, applicability, final, manifest
        )
        if len(refusals) != int(report_people[person_id]["applicability_refusal_count"]):
            raise ValueError(f"applicability refusal mismatch for {person_id}")
        pooled_choices.extend(int(row["choice"]) for row in final)
        pooled_correct.extend(float(value) for value in correct)
    pooled = experiment.metric_dict("pooled", pooled_choices, pooled_correct)
    if not _close(pooled["expected_calibration_error"], report["aggregate"]["pooled_correct_ece"]):
        raise ValueError("pooled calibration mismatch")
    if report["feature_map_id"] != full.feature_map.map_id:
        raise ValueError("feature-map lineage mismatch")
    if cohort["cohort_plan_digest"] != manifest["cohort_plan_digest"]:
        raise ValueError("cohort lineage mismatch")
    return {
        "status": "verified",
        "event_count": len(audit),
        "person_count": len(by_person),
        "report_digest": recorded_report_digest,
        "bundle_id": bundle["bundle_id"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2, sort_keys=True))
