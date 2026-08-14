from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .active_experiment import (
    ActiveExperimentRefusedError,
    apply_active_experiment_results,
    create_active_experiment_plan,
    load_active_experiment_plan,
    save_active_experiment_plan,
    verify_active_experiment_plan,
    verify_active_experiment_results,
)
from .applicability import PredictionRefusedError
from .composite import (
    CompositeModelRefusedError,
    apply_composite_active_experiment_results,
    create_composite_active_experiment_plan,
    create_composite_model,
    load_composite_model,
    predict_with_composite_model,
    save_composite_model,
)
from .contracts import Scenario
from .decision_evidence_v1 import (
    create_decision_evidence_bundle,
    decision_evidence_config_from_dict,
    decision_evidence_record_from_dict,
    load_decision_evidence_bundle,
    save_decision_evidence_bundle,
    source_snapshot_from_dict,
    validate_decision_evidence_bundle,
)
from .demo import run_demo
from .dynamic_state import (
    DynamicStateConfig,
    DynamicStateRefusedError,
    create_dynamic_state_plan,
    dynamic_state_config_from_dict,
    infer_dynamic_state,
    load_dynamic_state_plan,
    load_dynamic_state_report,
    predict_with_dynamic_state,
    save_dynamic_state_report,
    save_dynamic_state_plan,
)
from .empirical_bayes_v1 import (
    run_empirical_bayes_seed_audit,
)
from .ledger import EventLedger, observation_payload
from .hypernetwork_v1 import run_hypernetwork_seed_audit
from .mechanism import (
    EvidenceWindow,
    MechanismHypothesis,
    MechanismRefusedError,
    MechanismTerm,
    compare_mechanisms,
    create_mechanism_comparison_plan,
    load_mechanism_comparison_plan,
    load_mechanism_comparison_report,
    predict_with_mechanism,
    save_mechanism_comparison_plan,
    save_mechanism_comparison_report,
)
from .person_choice_benchmark import (
    BenchmarkConfig,
    generate_benchmark_dataset,
    run_person_choice_benchmark,
)
from .prospective_pilot_v1 import (
    create_pilot_plan,
    load_pilot_plan,
    load_pilot_receipt,
    pilot_forecast_from_dict,
    register_pilot_plan,
    save_pilot_plan,
    save_pilot_receipt,
    save_pilot_report,
    score_prospective_pilot,
)
from .storage import load_bundle, save_bundle
from .tyler_source_v1 import (
    extract_tyler_source_page,
    extract_tyler_source_rss,
    save_tyler_source_artifact,
)
from .tyler_corpus_v1 import (
    create_tyler_corpus,
    load_corpus_source_manifest,
    save_tyler_corpus,
)
from .tyler_annotation_v1 import (
    create_annotation_packets,
    save_annotation_packet,
    save_annotation_submission_template,
)
from .workflow import (
    fit_person_model,
    event_record_from_dict,
    load_event_ledger_jsonl,
    load_observations_jsonl,
    load_verification_authority,
    observation_from_dict,
    predict_with_bundle,
    save_event_ledger_jsonl,
    update_person_model,
)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scenario_from_dict(data: dict[str, object]) -> Scenario:
    raw_features = data["features"]
    if isinstance(raw_features, dict):
        feature_names = tuple(str(name) for name in raw_features)
        features = tuple(float(raw_features[name]) for name in raw_features)
    else:
        feature_names = tuple(str(name) for name in data["feature_names"])
        features = tuple(float(value) for value in raw_features)
    return Scenario(
        scenario_id=str(data["scenario_id"]),
        features=features,
        feature_names=feature_names,
        options=tuple(data.get("options", ("A", "B"))),
        domain=str(data.get("domain", "structured_choice")),
        context=dict(data.get("context", {})),
    )


def _candidate_scenarios(path: Path) -> tuple[Scenario, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            "candidate scenario file must contain a JSON array"
        )
    return tuple(
        _scenario_from_dict(dict(item))
        for item in raw
    )


def _mechanism_hypotheses(
    path: Path,
) -> tuple[MechanismHypothesis, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            "mechanism hypothesis file must contain a JSON array"
        )
    return tuple(
        MechanismHypothesis(
            hypothesis_id=str(item["hypothesis_id"]),
            terms=tuple(
                MechanismTerm(
                    term_id=str(term["term_id"]),
                    kind=str(term["kind"]),
                    feature_names=tuple(
                        str(value)
                        for value in term.get("feature_names", ())
                    ),
                )
                for raw_term in item["terms"]
                for term in (dict(raw_term),)
            ),
        )
        for raw_item in raw
        for item in (dict(raw_item),)
    )


def _dynamic_state_config(path: Path | None) -> DynamicStateConfig:
    defaults = DynamicStateConfig().to_dict()
    if path is None:
        return DynamicStateConfig()
    supplied = _read_json(path)
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(
            "unknown dynamic state config fields: "
            + ", ".join(sorted(unknown))
        )
    defaults.update(supplied)
    return dynamic_state_config_from_dict(defaults)


def _prediction_to_dict(prediction) -> dict[str, object]:
    return {
        "scenario_id": prediction.scenario_id,
        "person_id": prediction.person_id,
        "probability_option_1": prediction.probability_option_1,
        "predicted_choice": prediction.predicted_choice,
        "active_modules": list(prediction.active_modules),
        "model_version": prediction.model_version,
        "probability_lower_95": prediction.probability_lower_95,
        "probability_upper_95": prediction.probability_upper_95,
        "logit_standard_deviation": (
            prediction.logit_standard_deviation
        ),
        "applicability_status": prediction.applicability_status,
        "applicability_warnings": list(
            prediction.applicability_warnings
        ),
        "ood_score": prediction.ood_score,
        "ood_threshold": prediction.ood_threshold,
        "local_ood_score": prediction.local_ood_score,
        "local_ood_threshold": prediction.local_ood_threshold,
        "model_form_uncertainty_status": (
            prediction.model_form_uncertainty_status
        ),
        "validation_status": prediction.validation_status,
        "gate_overrides": list(prediction.gate_overrides),
        "dynamic_state_status": prediction.dynamic_state_status,
        "dynamic_state_mean": prediction.dynamic_state_mean,
        "dynamic_state_standard_deviation": (
            prediction.dynamic_state_standard_deviation
        ),
        "dynamic_state_reference_time": (
            prediction.dynamic_state_reference_time
        ),
        "dynamic_state_artifact_id": (
            prediction.dynamic_state_artifact_id
        ),
        "dynamic_state_current_evidence_status": (
            prediction.dynamic_state_current_evidence_status
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="pcfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the synthetic closed-loop benchmark.")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--persons", type=int, default=24)
    demo.add_argument("--source-trials", type=int, default=120)
    demo.add_argument("--target-trials", type=int, default=180)
    demo.add_argument(
        "--dataset-kind",
        choices=("in_family", "misspecified"),
        default="in_family",
    )
    demo.add_argument("--heterogeneity-scale", type=float, default=1.0)
    demo.add_argument("--output", type=Path)

    benchmark_v1 = subparsers.add_parser(
        "benchmark-v1",
        help=(
            "Run the bounded synthetic person-choice benchmark "
            "with population and personalized baselines."
        ),
    )
    benchmark_v1.add_argument("--seed", type=int, default=7301)
    benchmark_v1.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip the small NumPy person-embedding MLP baseline.",
    )

    subparsers.add_parser(
        "hypernetwork-v1",
        help=(
            "Run the fixed five-seed audit for the bounded "
            "support-set HyperNetwork candidate."
        ),
    )

    subparsers.add_parser(
        "empirical-bayes-v1",
        help=(
            "Run the fixed unseen-seed audit for the "
            "anisotropic empirical-Bayes adapter."
        ),
    )

    pilot_create = subparsers.add_parser(
        "pilot-create",
        help=(
            "Sign exact questions and four outcome-blind probability "
            "forecasts for a single-person prospective pilot."
        ),
    )
    pilot_create.add_argument("--person-id", required=True)
    pilot_create.add_argument("--scenarios", type=Path, required=True)
    pilot_create.add_argument("--forecasts", type=Path, required=True)
    pilot_create.add_argument("--keys", type=Path, required=True)
    pilot_create.add_argument("--verifier-id", required=True)
    pilot_create.add_argument("--created-at", required=True)
    pilot_create.add_argument("--collection-end", required=True)
    pilot_create.add_argument("--output", type=Path, required=True)

    pilot_register = subparsers.add_parser(
        "pilot-register",
        help=(
            "Create an independently signed registry receipt for a "
            "prospective pilot plan."
        ),
    )
    pilot_register.add_argument("--plan", type=Path, required=True)
    pilot_register.add_argument("--keys", type=Path, required=True)
    pilot_register.add_argument(
        "--registry-verifier-id",
        required=True,
    )
    pilot_register.add_argument("--registered-at", required=True)
    pilot_register.add_argument("--output", type=Path, required=True)

    pilot_score = subparsers.add_parser(
        "pilot-score",
        help=(
            "Blind-score registered forecasts against a later signed "
            "human outcome ledger."
        ),
    )
    pilot_score.add_argument("--plan", type=Path, required=True)
    pilot_score.add_argument("--receipt", type=Path, required=True)
    pilot_score.add_argument("--outcomes", type=Path, required=True)
    pilot_score.add_argument("--keys", type=Path, required=True)
    pilot_score.add_argument("--output", type=Path, required=True)

    decision_evidence_build = subparsers.add_parser(
        "decision-evidence-build-v1",
        help=(
            "Build a signed decision/context/rationale evidence bundle "
            "without authorizing model training."
        ),
    )
    decision_evidence_build.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    decision_evidence_build.add_argument(
        "--keys",
        type=Path,
        required=True,
    )
    decision_evidence_build.add_argument("--verifier-id", required=True)
    decision_evidence_build.add_argument("--created-at", required=True)
    decision_evidence_build.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    decision_evidence_verify = subparsers.add_parser(
        "decision-evidence-verify-v1",
        help=(
            "Recompute and verify a signed decision evidence bundle."
        ),
    )
    decision_evidence_verify.add_argument(
        "--bundle",
        type=Path,
        required=True,
    )
    decision_evidence_verify.add_argument(
        "--keys",
        type=Path,
        required=True,
    )

    tyler_source = subparsers.add_parser(
        "tyler-source-v1",
        help=(
            "Extract Tyler Cowen stance candidates from a locally "
            "saved official author-archive HTML page."
        ),
    )
    tyler_input = tyler_source.add_mutually_exclusive_group(
        required=True
    )
    tyler_input.add_argument("--html", type=Path)
    tyler_input.add_argument("--rss", type=Path)
    tyler_source.add_argument("--source-url", required=True)
    tyler_source.add_argument("--collected-at", required=True)
    tyler_source.add_argument("--output", type=Path, required=True)

    tyler_corpus = subparsers.add_parser(
        "tyler-corpus-v1",
        help=(
            "Replay verified local Tyler source snapshots into a "
            "temporally separated corpus."
        ),
    )
    tyler_corpus.add_argument("--manifest", type=Path, required=True)
    tyler_corpus.add_argument("--created-at", required=True)
    tyler_corpus.add_argument("--output", type=Path, required=True)

    tyler_annotation = subparsers.add_parser(
        "tyler-annotation-pack-v1",
        help=(
            "Create two blind Tyler Cowen annotation packets from a "
            "verified source artifact."
        ),
    )
    tyler_annotation.add_argument("--source", type=Path, required=True)
    tyler_annotation.add_argument("--created-at", required=True)
    tyler_annotation.add_argument("--output-a", type=Path, required=True)
    tyler_annotation.add_argument("--output-b", type=Path, required=True)
    tyler_annotation.add_argument("--template-a", type=Path)
    tyler_annotation.add_argument("--template-b", type=Path)

    fit = subparsers.add_parser(
        "fit",
        help="Fit a person model from a signed event ledger.",
    )
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument(
        "--validation-ledger",
        type=Path,
        required=True,
        help="Independent signed holdout ledger for the target person.",
    )
    fit.add_argument(
        "--applicability-ledger",
        type=Path,
        required=True,
        help="Independent signed ledger for applicability calibration.",
    )
    fit.add_argument("--verification-keys", type=Path, required=True)
    fit.add_argument("--person-id", required=True)
    fit.add_argument("--feature-names", required=True)
    fit.add_argument("--output", type=Path, required=True)

    predict = subparsers.add_parser(
        "predict",
        help="Predict a structured scenario with a saved person model.",
    )
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--scenario", type=Path, required=True)
    predict.add_argument("--prediction-at", required=True)
    predict.add_argument(
        "--validation-override",
        action="store_true",
        help="Override model validation for diagnostics only.",
    )
    predict.add_argument(
        "--applicability-override",
        action="store_true",
        help="Override applicability refusal for diagnostics only.",
    )

    plan_experiment = subparsers.add_parser(
        "plan-experiment",
        help=(
            "Select and sign one adaptive next experiment or an "
            "outcome-blind approximate batch by predictive mutual "
            "information."
        ),
    )
    plan_experiment.add_argument("--model", type=Path, required=True)
    plan_experiment.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="JSON array of candidate scenarios without outcomes.",
    )
    plan_experiment.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    plan_experiment.add_argument("--verifier-id", required=True)
    plan_experiment.add_argument("--created-at", required=True)
    plan_experiment.add_argument(
        "--selection-count",
        type=int,
        required=True,
    )
    plan_experiment.add_argument("--output", type=Path, required=True)

    verify_experiment = subparsers.add_parser(
        "verify-experiment",
        help=(
            "Recompute an active experiment plan and verify its signed "
            "result ledger."
        ),
    )
    verify_experiment.add_argument("--model", type=Path, required=True)
    verify_experiment.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    verify_experiment.add_argument("--plan", type=Path, required=True)
    verify_experiment.add_argument("--input", type=Path, required=True)
    verify_experiment.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )

    apply_experiment = subparsers.add_parser(
        "apply-experiment",
        help=(
            "Verify experiment results, update the model, and revalidate "
            "against a later sealed holdout ledger."
        ),
    )
    apply_experiment.add_argument("--model", type=Path, required=True)
    apply_experiment.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Training ledger from which the base model was derived.",
    )
    apply_experiment.add_argument(
        "--applicability-ledger",
        type=Path,
        required=True,
    )
    apply_experiment.add_argument(
        "--future-validation-ledger",
        type=Path,
        required=True,
        help="Sealed holdout ledger collected after all experiment results.",
    )
    apply_experiment.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    apply_experiment.add_argument("--plan", type=Path, required=True)
    apply_experiment.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Signed experiment-result ledger.",
    )
    apply_experiment.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    apply_experiment.add_argument("--output", type=Path, required=True)
    apply_experiment.add_argument(
        "--output-ledger",
        type=Path,
        required=True,
    )

    plan_mechanisms = subparsers.add_parser(
        "plan-mechanisms",
        help=(
            "Preregister explicit predictive-structure hypotheses and "
            "three ordered evidence windows."
        ),
    )
    plan_mechanisms.add_argument("--model", type=Path, required=True)
    plan_mechanisms.add_argument(
        "--hypotheses",
        type=Path,
        required=True,
        help="JSON array of explicit mechanism hypotheses.",
    )
    plan_mechanisms.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    plan_mechanisms.add_argument("--verifier-id", required=True)
    plan_mechanisms.add_argument("--registered-at", required=True)
    for split in ("discovery", "selection", "confirmation"):
        plan_mechanisms.add_argument(
            f"--{split}-start-at",
            required=True,
        )
        plan_mechanisms.add_argument(
            f"--{split}-end-at",
            required=True,
        )
        plan_mechanisms.add_argument(
            f"--{split}-event-count",
            type=int,
            required=True,
        )
    plan_mechanisms.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    compare_mechanism_parser = subparsers.add_parser(
        "compare-mechanisms",
        help=(
            "Fit on discovery evidence, select on a separate ledger, "
            "and confirm once on a sealed final ledger."
        ),
    )
    compare_mechanism_parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    compare_mechanism_parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )
    for split in ("discovery", "selection", "confirmation"):
        compare_mechanism_parser.add_argument(
            f"--{split}-ledger",
            type=Path,
            required=True,
        )
    compare_mechanism_parser.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    compare_mechanism_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    predict_mechanism_parser = subparsers.add_parser(
        "predict-mechanism",
        help=(
            "Predict with a supported mechanism candidate after "
            "recomputing its report from all signed evidence."
        ),
    )
    predict_mechanism_parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    predict_mechanism_parser.add_argument(
        "--plan",
        type=Path,
        required=True,
    )
    predict_mechanism_parser.add_argument(
        "--report",
        type=Path,
        required=True,
    )
    for split in ("discovery", "selection", "confirmation"):
        predict_mechanism_parser.add_argument(
            f"--{split}-ledger",
            type=Path,
            required=True,
        )
    predict_mechanism_parser.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    predict_mechanism_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
    )
    predict_mechanism_parser.add_argument(
        "--prediction-at",
        required=True,
    )

    create_composite_parser = subparsers.add_parser(
        "create-composite",
        help=(
            "Bind a stable person model and independently confirmed "
            "mechanism into one signed predictive identity."
        ),
    )
    predict_composite_parser = subparsers.add_parser(
        "predict-composite",
        help=(
            "Predict through a signed composite after recomputing its "
            "confirmed mechanism from all signed evidence."
        ),
    )
    plan_composite_experiment_parser = subparsers.add_parser(
        "plan-composite-experiment",
        help=(
            "Select experiments from the verified composite predictive "
            "view while excluding all prior mechanism evidence."
        ),
    )
    apply_composite_experiment_parser = subparsers.add_parser(
        "apply-composite-experiment",
        help=(
            "Verify composite experiment results, update the base model, "
            "and explicitly invalidate the old composite."
        ),
    )
    for composite_parser in (
        create_composite_parser,
        predict_composite_parser,
        plan_composite_experiment_parser,
        apply_composite_experiment_parser,
    ):
        composite_parser.add_argument(
            "--model",
            type=Path,
            required=True,
        )
        composite_parser.add_argument(
            "--mechanism-plan",
            type=Path,
            required=True,
        )
        composite_parser.add_argument(
            "--mechanism-report",
            type=Path,
            required=True,
        )
        for split in ("discovery", "selection", "confirmation"):
            composite_parser.add_argument(
                f"--{split}-ledger",
                type=Path,
                required=True,
            )
        composite_parser.add_argument(
            "--verification-keys",
            type=Path,
            required=True,
        )
    create_composite_parser.add_argument(
        "--verifier-id",
        required=True,
    )
    create_composite_parser.add_argument(
        "--created-at",
        required=True,
    )
    create_composite_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    predict_composite_parser.add_argument(
        "--composite",
        type=Path,
        required=True,
    )
    predict_composite_parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
    )
    predict_composite_parser.add_argument(
        "--prediction-at",
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--composite",
        type=Path,
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--verifier-id",
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--created-at",
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--selection-count",
        type=int,
        required=True,
    )
    plan_composite_experiment_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--composite",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--active-plan",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Signed composite experiment result ledger.",
    )
    apply_composite_experiment_parser.add_argument(
        "--ledger",
        type=Path,
        required=True,
        help="Signed base training ledger.",
    )
    apply_composite_experiment_parser.add_argument(
        "--applicability-ledger",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--future-validation-ledger",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    apply_composite_experiment_parser.add_argument(
        "--output-ledger",
        type=Path,
        required=True,
    )

    infer_state = subparsers.add_parser(
        "infer-state",
        help=(
            "Infer a dynamic residual state from a signed future ledger."
        ),
    )
    infer_state.add_argument("--model", type=Path, required=True)
    infer_state.add_argument("--input", type=Path, required=True)
    infer_state.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    infer_state.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Signed pre-registered dynamic state plan.",
    )
    infer_state.add_argument("--output", type=Path, required=True)

    predict_state = subparsers.add_parser(
        "predict-state",
        help=(
            "Predict with a verified prequential residual-state artifact."
        ),
    )
    predict_state.add_argument("--model", type=Path, required=True)
    predict_state.add_argument("--state", type=Path, required=True)
    predict_state.add_argument("--plan", type=Path, required=True)
    predict_state.add_argument(
        "--state-ledger",
        type=Path,
        required=True,
        help="Original signed ledger used to derive the state.",
    )
    predict_state.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    predict_state.add_argument("--scenario", type=Path, required=True)
    predict_state.add_argument("--prediction-at", required=True)
    predict_state.add_argument(
        "--validation-override",
        action="store_true",
        help="Override model validation for diagnostics only.",
    )
    predict_state.add_argument(
        "--applicability-override",
        action="store_true",
        help="Override applicability refusal for diagnostics only.",
    )
    predict_state.add_argument(
        "--state-override",
        action="store_true",
        help="Override dynamic state validation for diagnostics only.",
    )

    plan_state = subparsers.add_parser(
        "plan-state",
        help="Create a signed preregistered dynamic state plan.",
    )
    plan_state.add_argument("--model", type=Path, required=True)
    plan_state.add_argument(
        "--verification-keys",
        type=Path,
        required=True,
    )
    plan_state.add_argument("--verifier-id", required=True)
    plan_state.add_argument("--registered-at", required=True)
    plan_state.add_argument("--monitoring-start-at", required=True)
    plan_state.add_argument("--monitoring-end-at", required=True)
    plan_state.add_argument(
        "--expected-event-count",
        type=int,
        required=True,
    )
    plan_state.add_argument(
        "--config",
        type=Path,
        help="Optional dynamic state JSON config fixed by the plan.",
    )
    plan_state.add_argument("--output", type=Path, required=True)

    update = subparsers.add_parser(
        "update",
        help="Update a saved model with a signed outcome event.",
    )
    update.add_argument("--model", type=Path, required=True)
    update.add_argument("--ledger", type=Path, required=True)
    update.add_argument("--outcome", type=Path, required=True)
    update.add_argument("--verification-keys", type=Path, required=True)
    update.add_argument(
        "--validation-ledger",
        type=Path,
        required=True,
        help="Independent signed holdout ledger used to revalidate the update.",
    )
    update.add_argument(
        "--applicability-ledger",
        type=Path,
        required=True,
        help="Independent ledger used to recalibrate applicability.",
    )
    update.add_argument("--output", type=Path, required=True)
    update.add_argument("--output-ledger", type=Path, required=True)

    sign_event = subparsers.add_parser(
        "sign-event",
        help="Create a signed event record from a raw observation.",
    )
    sign_event.add_argument("--observation", type=Path, required=True)
    sign_event.add_argument("--event-id", required=True)
    sign_event.add_argument("--observed-at", required=True)
    sign_event.add_argument("--evidence", type=Path, required=True)
    sign_event.add_argument("--verifier-id", required=True)
    sign_event.add_argument("--verified-at", required=True)
    sign_event.add_argument("--verification-keys", type=Path, required=True)
    sign_event.add_argument("--output", type=Path, required=True)

    sign_ledger = subparsers.add_parser(
        "sign-ledger",
        help="Convert raw observation JSONL into a signed event ledger.",
    )
    sign_ledger.add_argument("--input", type=Path, required=True)
    sign_ledger.add_argument("--observed-at", required=True)
    sign_ledger.add_argument("--verifier-id", required=True)
    sign_ledger.add_argument("--verified-at", required=True)
    sign_ledger.add_argument("--verification-keys", type=Path, required=True)
    sign_ledger.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "tyler-annotation-pack-v1":
        from .tyler_source_v1 import load_tyler_source_artifact

        packet_a, packet_b = create_annotation_packets(
            load_tyler_source_artifact(args.source),
            created_at=args.created_at,
        )
        save_annotation_packet(packet_a, args.output_a)
        save_annotation_packet(packet_b, args.output_b)
        if (args.template_a is None) != (args.template_b is None):
            raise ValueError(
                "--template-a and --template-b must be supplied together"
            )
        if args.template_a is not None:
            save_annotation_submission_template(
                packet_a,
                args.template_a,
            )
            save_annotation_submission_template(
                packet_b,
                args.template_b,
            )
        print(
            json.dumps(
                {
                    "candidate_count": len(packet_a.items),
                    "candidate_set_digest": (
                        packet_a.candidate_set_digest
                    ),
                    "codebook_digest": packet_a.codebook_digest,
                    "packet_a_digest": packet_a.artifact_digest,
                    "packet_b_digest": packet_b.artifact_digest,
                    "training_eligible_before_annotation": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "tyler-source-v1":
        raw_source = (args.html or args.rss).read_text(
            encoding="utf-8"
        )
        artifact = (
            extract_tyler_source_page(
                raw_source,
                source_url=args.source_url,
                collected_at=args.collected_at,
            )
            if args.html is not None
            else extract_tyler_source_rss(
                raw_source,
                source_url=args.source_url,
                collected_at=args.collected_at,
            )
        )
        save_tyler_source_artifact(artifact, args.output)
        print(
            json.dumps(
                {
                    "artifact_digest": artifact.artifact_digest,
                    "candidate_counts": artifact.candidate_counts(),
                    "extraction_digest": artifact.extraction_digest,
                    "post_count": len(artifact.posts),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "tyler-corpus-v1":
        source_inputs = load_corpus_source_manifest(args.manifest)
        corpus = create_tyler_corpus(
            source_inputs,
            expected_source_urls=tuple(
                item.artifact.source_url for item in source_inputs
            ),
            created_at=args.created_at,
        )
        save_tyler_corpus(corpus, args.output)
        print(
            json.dumps(
                {
                    "candidate_counts": corpus.candidate_counts,
                    "corpus_digest": corpus.corpus_digest,
                    "post_count": len(corpus.records),
                    "role_counts": corpus.role_counts,
                    "source_count": len(corpus.sources),
                    "training_eligible_before_annotation": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    elif args.command == "pilot-create":
        authority = load_verification_authority(args.keys)
        raw_forecasts = json.loads(
            args.forecasts.read_text(encoding="utf-8")
        )
        if not isinstance(raw_forecasts, list):
            raise ValueError(
                "pilot forecast file must contain a JSON array"
            )
        plan = create_pilot_plan(
            person_id=args.person_id,
            scenarios=_candidate_scenarios(args.scenarios),
            forecasts=tuple(
                pilot_forecast_from_dict(dict(item))
                for item in raw_forecasts
            ),
            authority=authority,
            verifier_id=args.verifier_id,
            created_at=args.created_at,
            collection_end=args.collection_end,
        )
        save_pilot_plan(args.output, plan)
        print(
            json.dumps(
                plan.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "pilot-register":
        authority = load_verification_authority(args.keys)
        receipt = register_pilot_plan(
            load_pilot_plan(args.plan),
            authority,
            registry_verifier_id=args.registry_verifier_id,
            registered_at=args.registered_at,
        )
        save_pilot_receipt(args.output, receipt)
        print(
            json.dumps(
                receipt.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "pilot-score":
        authority = load_verification_authority(args.keys)
        report = score_prospective_pilot(
            load_pilot_plan(args.plan),
            load_pilot_receipt(args.receipt),
            load_event_ledger_jsonl(args.outcomes, authority),
            authority,
        )
        save_pilot_report(args.output, report)
        print(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "decision-evidence-build-v1":
        authority = load_verification_authority(args.keys)
        raw = _read_json(args.input)
        if set(raw) != {"config", "sources", "records"}:
            raise ValueError(
                "decision evidence input fields must be config, sources, "
                "and records"
            )
        bundle = create_decision_evidence_bundle(
            config=decision_evidence_config_from_dict(
                dict(raw["config"])
            ),
            sources=tuple(
                source_snapshot_from_dict(dict(item))
                for item in raw["sources"]
            ),
            records=tuple(
                decision_evidence_record_from_dict(dict(item))
                for item in raw["records"]
            ),
            created_at=args.created_at,
            authority=authority,
            verifier_id=args.verifier_id,
        )
        save_decision_evidence_bundle(args.output, bundle)
        summary = validate_decision_evidence_bundle(bundle, authority)
        print(
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "decision-evidence-verify-v1":
        authority = load_verification_authority(args.keys)
        bundle = load_decision_evidence_bundle(
            args.bundle,
            authority,
        )
        summary = validate_decision_evidence_bundle(bundle, authority)
        print(
            json.dumps(
                summary.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "empirical-bayes-v1":
        audit = run_empirical_bayes_seed_audit()
        print(
            json.dumps(
                audit.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "hypernetwork-v1":
        audit = run_hypernetwork_seed_audit()
        print(
            json.dumps(
                audit.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "benchmark-v1":
        config = BenchmarkConfig.smoke(seed=args.seed)
        dataset = generate_benchmark_dataset(config)
        report = run_person_choice_benchmark(
            dataset,
            train_neural=not args.skip_neural,
        )
        print(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "demo":
        report = run_demo(
            seed=args.seed,
            person_count=args.persons,
            source_trials=args.source_trials,
            target_trials=args.target_trials,
            dataset_kind=args.dataset_kind,
            heterogeneity_scale=args.heterogeneity_scale,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    elif args.command == "fit":
        authority = load_verification_authority(args.verification_keys)
        ledger = load_event_ledger_jsonl(args.input, authority)
        validation_ledger = load_event_ledger_jsonl(
            args.validation_ledger,
            authority,
        )
        applicability_ledger = load_event_ledger_jsonl(
            args.applicability_ledger,
            authority,
        )
        feature_names = tuple(
            name.strip()
            for name in args.feature_names.split(",")
            if name.strip()
        )
        bundle = fit_person_model(
            ledger,
            authority,
            applicability_ledger=applicability_ledger,
            validation_ledger=validation_ledger,
            person_id=args.person_id,
            feature_names=feature_names,
        )
        save_bundle(args.output, bundle)
        print(
            json.dumps(
                {
                    "saved_model": str(args.output),
                    "person_id": bundle.representation.person_id,
                    "observation_count": (
                        bundle.representation.observation_count
                    ),
                    "bundle_version": bundle.bundle_version,
                    "model_id": bundle.manifest.model_id,
                    "validation_status": (
                        bundle.manifest.validation.status
                    ),
                    "validation_reasons": list(
                        bundle.manifest.validation.reasons
                    ),
                    "temporal_stability_status": (
                        bundle.manifest.validation
                        .temporal_stability_status
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "predict":
        bundle = load_bundle(args.model)
        scenario = _scenario_from_dict(_read_json(args.scenario))
        try:
            prediction = predict_with_bundle(
                bundle,
                scenario,
                prediction_at=args.prediction_at,
                validation_override=args.validation_override,
                applicability_override=args.applicability_override,
            )
        except PredictionRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                        "ood_score": error.ood_score,
                        "ood_threshold": error.ood_threshold,
                        "local_ood_score": error.local_ood_score,
                        "local_ood_threshold": error.local_ood_threshold,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        print(
            json.dumps(
                _prediction_to_dict(prediction),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "plan-experiment":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        candidates = _candidate_scenarios(args.candidates)
        try:
            plan = create_active_experiment_plan(
                bundle,
                candidates,
                authority,
                verifier_id=args.verifier_id,
                created_at=args.created_at,
                selection_count=args.selection_count,
            )
        except ActiveExperimentRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        save_active_experiment_plan(args.output, plan)
        print(
            json.dumps(
                {
                    "saved_plan": str(args.output),
                    "plan_id": plan.plan_id,
                    "base_model_id": plan.base_model_id,
                    "person_id": plan.person_id,
                    "candidate_count": plan.candidate_count,
                    "selection_count": plan.selection_count,
                    "selection_mode": plan.selection_mode,
                    "total_expected_information_gain": (
                        plan.total_expected_information_gain
                    ),
                    "expected_covariance_entropy_reduction": (
                        plan.expected_covariance_entropy_reduction
                    ),
                    "selected_scenario_ids": [
                        item.scenario.scenario_id
                        for item in plan.selections
                    ],
                    "verifier_id": plan.verifier_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "verify-experiment":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        candidates = _candidate_scenarios(args.candidates)
        plan = load_active_experiment_plan(args.plan, authority)
        result_ledger = load_event_ledger_jsonl(
            args.input,
            authority,
        )
        try:
            verify_active_experiment_plan(
                bundle,
                candidates,
                authority,
                plan,
            )
            verified = verify_active_experiment_results(
                plan,
                result_ledger,
                authority,
            )
        except ActiveExperimentRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        print(
            json.dumps(
                {
                    "status": "verified",
                    "plan_id": plan.plan_id,
                    "person_id": plan.person_id,
                    "result_event_count": len(verified.records),
                    "result_event_ids": [
                        record.event_id
                        for record in verified.records
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "apply-experiment":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        training_ledger = load_event_ledger_jsonl(
            args.ledger,
            authority,
        )
        applicability_ledger = load_event_ledger_jsonl(
            args.applicability_ledger,
            authority,
        )
        future_validation_ledger = load_event_ledger_jsonl(
            args.future_validation_ledger,
            authority,
        )
        candidates = _candidate_scenarios(args.candidates)
        plan = load_active_experiment_plan(args.plan, authority)
        result_ledger = load_event_ledger_jsonl(
            args.input,
            authority,
        )
        try:
            update = apply_active_experiment_results(
                bundle,
                training_ledger,
                applicability_ledger,
                future_validation_ledger,
                candidates,
                plan,
                result_ledger,
                authority,
            )
        except ActiveExperimentRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        save_bundle(args.output, update.bundle)
        save_event_ledger_jsonl(
            args.output_ledger,
            update.ledger,
        )
        print(
            json.dumps(
                {
                    "status": "updated",
                    "saved_model": str(args.output),
                    "saved_ledger": str(args.output_ledger),
                    "plan_id": update.plan_id,
                    "model_id": update.bundle.manifest.model_id,
                    "observation_count": (
                        update.bundle.representation.observation_count
                    ),
                    "result_event_ids": list(
                        update.result_event_ids
                    ),
                    "result_data_hash": update.result_data_hash,
                    "realized_covariance_entropy_reduction": (
                        update.realized_covariance_entropy_reduction
                    ),
                    "validation_status": (
                        update.bundle.manifest.validation.status
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "plan-mechanisms":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        hypotheses = _mechanism_hypotheses(args.hypotheses)
        try:
            plan = create_mechanism_comparison_plan(
                bundle,
                hypotheses,
                authority,
                verifier_id=args.verifier_id,
                registered_at=args.registered_at,
                discovery_window=EvidenceWindow(
                    args.discovery_start_at,
                    args.discovery_end_at,
                    args.discovery_event_count,
                ),
                selection_window=EvidenceWindow(
                    args.selection_start_at,
                    args.selection_end_at,
                    args.selection_event_count,
                ),
                confirmation_window=EvidenceWindow(
                    args.confirmation_start_at,
                    args.confirmation_end_at,
                    args.confirmation_event_count,
                ),
            )
        except MechanismRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        save_mechanism_comparison_plan(args.output, plan)
        print(
            json.dumps(
                {
                    "status": "planned",
                    "saved_plan": str(args.output),
                    "plan_id": plan.plan_id,
                    "base_model_id": plan.base_model_id,
                    "person_id": plan.person_id,
                    "hypothesis_ids": [
                        hypothesis.hypothesis_id
                        for hypothesis in plan.hypotheses
                    ],
                    "interpretation": "predictive_structure_only",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "compare-mechanisms":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        plan = load_mechanism_comparison_plan(
            args.plan,
            authority,
        )
        discovery_ledger = load_event_ledger_jsonl(
            args.discovery_ledger,
            authority,
        )
        selection_ledger = load_event_ledger_jsonl(
            args.selection_ledger,
            authority,
        )
        confirmation_ledger = load_event_ledger_jsonl(
            args.confirmation_ledger,
            authority,
        )
        try:
            report = compare_mechanisms(
                bundle,
                plan,
                discovery_ledger,
                selection_ledger,
                confirmation_ledger,
                authority,
            )
        except MechanismRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        save_mechanism_comparison_report(args.output, report)
        print(
            json.dumps(
                {
                    "status": report.status,
                    "saved_report": str(args.output),
                    "report_id": report.report_id,
                    "plan_id": report.plan_id,
                    "selected_hypothesis_id": (
                        report.selected_hypothesis_id
                    ),
                    "confirmation_nll_uplift": (
                        report.confirmation_nll_uplift
                    ),
                    "confirmation_nll_uplift_ci_lower": (
                        report.confirmation_nll_uplift_ci_lower
                    ),
                    "confirmation_nll_uplift_ci_upper": (
                        report.confirmation_nll_uplift_ci_upper
                    ),
                    "interpretation": report.interpretation,
                    "reasons": list(report.reasons),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "predict-mechanism":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        plan = load_mechanism_comparison_plan(
            args.plan,
            authority,
        )
        report = load_mechanism_comparison_report(
            args.report,
            authority,
        )
        discovery_ledger = load_event_ledger_jsonl(
            args.discovery_ledger,
            authority,
        )
        selection_ledger = load_event_ledger_jsonl(
            args.selection_ledger,
            authority,
        )
        confirmation_ledger = load_event_ledger_jsonl(
            args.confirmation_ledger,
            authority,
        )
        scenario = _scenario_from_dict(_read_json(args.scenario))
        try:
            prediction = predict_with_mechanism(
                bundle,
                plan,
                report,
                discovery_ledger,
                selection_ledger,
                confirmation_ledger,
                authority,
                scenario,
                prediction_at=args.prediction_at,
            )
        except MechanismRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        print(
            json.dumps(
                {
                    "scenario_id": prediction.scenario_id,
                    "person_id": prediction.person_id,
                    "probability_option_1": (
                        prediction.probability_option_1
                    ),
                    "predicted_choice": prediction.predicted_choice,
                    "base_probability_option_1": (
                        prediction.base_probability_option_1
                    ),
                    "probability_lower_95": (
                        prediction.probability_lower_95
                    ),
                    "probability_upper_95": (
                        prediction.probability_upper_95
                    ),
                    "logit_standard_deviation": (
                        prediction.logit_standard_deviation
                    ),
                    "selected_hypothesis_id": (
                        prediction.selected_hypothesis_id
                    ),
                    "mechanism_plan_id": (
                        prediction.mechanism_plan_id
                    ),
                    "mechanism_report_id": (
                        prediction.mechanism_report_id
                    ),
                    "applicability_status": (
                        prediction.applicability_status
                    ),
                    "model_form_uncertainty_status": (
                        prediction.model_form_uncertainty_status
                    ),
                    "uncertainty_scope": (
                        prediction.uncertainty_scope
                    ),
                    "interpretation": prediction.interpretation,
                    "model_version": prediction.model_version,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command in {
        "create-composite",
        "predict-composite",
        "plan-composite-experiment",
        "apply-composite-experiment",
    }:
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        mechanism_plan = load_mechanism_comparison_plan(
            args.mechanism_plan,
            authority,
        )
        mechanism_report = load_mechanism_comparison_report(
            args.mechanism_report,
            authority,
        )
        evidence = tuple(
            load_event_ledger_jsonl(
                getattr(args, f"{split}_ledger"),
                authority,
            )
            for split in ("discovery", "selection", "confirmation")
        )
        try:
            if args.command == "create-composite":
                composite = create_composite_model(
                    bundle,
                    mechanism_plan,
                    mechanism_report,
                    *evidence,
                    authority,
                    verifier_id=args.verifier_id,
                    created_at=args.created_at,
                )
                save_composite_model(args.output, composite)
                payload = {
                    "status": "created",
                    "saved_composite": str(args.output),
                    "composite_model_id": (
                        composite.composite_model_id
                    ),
                    "base_model_id": composite.base_model_id,
                    "mechanism_report_id": (
                        composite.mechanism_report_id
                    ),
                    "valid_through": composite.valid_through,
                    "interpretation": composite.interpretation,
                    "uncertainty_scope": (
                        composite.uncertainty_scope
                    ),
                }
            elif args.command == "predict-composite":
                composite = load_composite_model(
                    args.composite,
                    authority,
                )
                scenario = _scenario_from_dict(
                    _read_json(args.scenario)
                )
                prediction = predict_with_composite_model(
                    bundle,
                    composite,
                    mechanism_plan,
                    mechanism_report,
                    *evidence,
                    authority,
                    scenario,
                    prediction_at=args.prediction_at,
                )
                payload = {
                    "scenario_id": prediction.scenario_id,
                    "person_id": prediction.person_id,
                    "probability_option_1": (
                        prediction.probability_option_1
                    ),
                    "predicted_choice": (
                        prediction.predicted_choice
                    ),
                    "probability_lower_95": (
                        prediction.probability_lower_95
                    ),
                    "probability_upper_95": (
                        prediction.probability_upper_95
                    ),
                    "logit_standard_deviation": (
                        prediction.logit_standard_deviation
                    ),
                    "predictive_model_id": (
                        prediction.predictive_model_id
                    ),
                    "base_model_id": prediction.base_model_id,
                    "mechanism_plan_id": (
                        prediction.mechanism_plan_id
                    ),
                    "mechanism_report_id": (
                        prediction.mechanism_report_id
                    ),
                    "selected_hypothesis_id": (
                        prediction.selected_hypothesis_id
                    ),
                    "active_components": list(
                        prediction.active_components
                    ),
                    "applicability_status": (
                        prediction.applicability_status
                    ),
                    "validation_status": (
                        prediction.validation_status
                    ),
                    "uncertainty_scope": (
                        prediction.uncertainty_scope
                    ),
                    "interpretation": (
                        prediction.interpretation
                    ),
                    "model_version": prediction.model_version,
                }
            elif args.command == "plan-composite-experiment":
                composite = load_composite_model(
                    args.composite,
                    authority,
                )
                candidates = _candidate_scenarios(args.candidates)
                active_plan = (
                    create_composite_active_experiment_plan(
                        bundle,
                        composite,
                        mechanism_plan,
                        mechanism_report,
                        *evidence,
                        authority,
                        candidates,
                        verifier_id=args.verifier_id,
                        created_at=args.created_at,
                        selection_count=args.selection_count,
                    )
                )
                save_active_experiment_plan(
                    args.output,
                    active_plan,
                )
                payload = {
                    "status": "planned",
                    "saved_plan": str(args.output),
                    "plan_id": active_plan.plan_id,
                    "base_model_id": active_plan.base_model_id,
                    "predictive_model_id": (
                        active_plan.predictive_model_id
                    ),
                    "predictive_model_version": (
                        active_plan.predictive_model_version
                    ),
                    "selection_mode": (
                        active_plan.selection_mode
                    ),
                    "selected_scenario_ids": [
                        item.scenario.scenario_id
                        for item in active_plan.selections
                    ],
                }
            else:
                composite = load_composite_model(
                    args.composite,
                    authority,
                )
                candidates = _candidate_scenarios(args.candidates)
                active_plan = load_active_experiment_plan(
                    args.active_plan,
                    authority,
                )
                training_ledger = load_event_ledger_jsonl(
                    args.ledger,
                    authority,
                )
                applicability_ledger = load_event_ledger_jsonl(
                    args.applicability_ledger,
                    authority,
                )
                future_validation_ledger = (
                    load_event_ledger_jsonl(
                        args.future_validation_ledger,
                        authority,
                    )
                )
                result_ledger = load_event_ledger_jsonl(
                    args.input,
                    authority,
                )
                update = (
                    apply_composite_active_experiment_results(
                        bundle,
                        composite,
                        mechanism_plan,
                        mechanism_report,
                        *evidence,
                        authority,
                        training_ledger,
                        applicability_ledger,
                        future_validation_ledger,
                        candidates,
                        active_plan,
                        result_ledger,
                    )
                )
                save_bundle(
                    args.output,
                    update.base_update.bundle,
                )
                save_event_ledger_jsonl(
                    args.output_ledger,
                    update.base_update.ledger,
                )
                payload = {
                    "status": update.status,
                    "saved_model": str(args.output),
                    "saved_ledger": str(args.output_ledger),
                    "plan_id": update.base_update.plan_id,
                    "new_base_model_id": (
                        update.base_update.bundle.manifest.model_id
                    ),
                    "invalidated_composite_model_id": (
                        update.invalidated_composite_model_id
                    ),
                    "invalidated_mechanism_report_id": (
                        update.invalidated_mechanism_report_id
                    ),
                    "required_next_action": (
                        update.required_next_action
                    ),
                    "result_event_ids": list(
                        update.base_update.result_event_ids
                    ),
                    "result_data_hash": (
                        update.base_update.result_data_hash
                    ),
                    "validation_status": (
                        update.base_update.bundle.manifest.validation.status
                    ),
                }
        except CompositeModelRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        except ActiveExperimentRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        except ValueError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": [
                            "composite_input_or_signature_invalid"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from error
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "plan-state":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        plan = create_dynamic_state_plan(
            bundle,
            authority,
            verifier_id=args.verifier_id,
            registered_at=args.registered_at,
            monitoring_start_at=args.monitoring_start_at,
            monitoring_end_at=args.monitoring_end_at,
            expected_event_count=args.expected_event_count,
            config=_dynamic_state_config(args.config),
        )
        save_dynamic_state_plan(args.output, plan)
        print(
            json.dumps(
                {
                    "saved_plan": str(args.output),
                    "plan_id": plan.plan_id,
                    "base_model_id": plan.base_model_id,
                    "person_id": plan.person_id,
                    "registered_at": plan.registered_at,
                    "monitoring_start_at": (
                        plan.monitoring_start_at
                    ),
                    "monitoring_end_at": plan.monitoring_end_at,
                    "expected_event_count": (
                        plan.expected_event_count
                    ),
                    "verifier_id": plan.verifier_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "infer-state":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        ledger = load_event_ledger_jsonl(args.input, authority)
        plan = load_dynamic_state_plan(args.plan, authority)
        try:
            report = infer_dynamic_state(
                bundle,
                ledger,
                authority,
                plan,
            )
        except DynamicStateRefusedError as error:
            print(
                json.dumps(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        save_dynamic_state_report(args.output, report)
        print(
            json.dumps(
                {
                    "saved_state": str(args.output),
                    "artifact_id": report.artifact_id,
                    "base_model_id": report.base_model_id,
                    "person_id": report.person_id,
                    "status": report.status,
                    "reasons": list(report.reasons),
                    "sample_count": len(report.points),
                    "static_nll": report.static_nll,
                    "dynamic_prequential_nll": (
                        report.dynamic_prequential_nll
                    ),
                    "nll_uplift": report.nll_uplift,
                    "nll_uplift_ci_lower": (
                        report.nll_uplift_ci_lower
                    ),
                    "nll_uplift_ci_upper": (
                        report.nll_uplift_ci_upper
                    ),
                    "maximum_detection_run": (
                        report.maximum_detection_run
                    ),
                    "final_log_e_value": report.final_log_e_value,
                    "maximum_log_e_value": (
                        report.maximum_log_e_value
                    ),
                    "config_status": report.config_status,
                    "interpretation_status": (
                        report.interpretation_status
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "predict-state":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(
            args.verification_keys
        )
        plan = load_dynamic_state_plan(args.plan, authority)
        report = load_dynamic_state_report(
            args.state,
            authority,
        )
        evidence_ledger = load_event_ledger_jsonl(
            args.state_ledger,
            authority,
        )
        scenario = _scenario_from_dict(_read_json(args.scenario))
        try:
            prediction = predict_with_dynamic_state(
                bundle,
                report,
                scenario,
                authority,
                plan,
                evidence_ledger,
                prediction_at=args.prediction_at,
                validation_override=args.validation_override,
                applicability_override=args.applicability_override,
                state_override=args.state_override,
            )
        except (
            DynamicStateRefusedError,
            PredictionRefusedError,
        ) as error:
            result = {
                "status": "refused",
                "reasons": list(error.reasons),
            }
            if isinstance(error, PredictionRefusedError):
                result.update(
                    {
                        "ood_score": error.ood_score,
                        "ood_threshold": error.ood_threshold,
                        "local_ood_score": error.local_ood_score,
                        "local_ood_threshold": (
                            error.local_ood_threshold
                        ),
                    }
                )
            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from None
        print(
            json.dumps(
                _prediction_to_dict(prediction),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "update":
        bundle = load_bundle(args.model)
        authority = load_verification_authority(args.verification_keys)
        ledger = load_event_ledger_jsonl(args.ledger, authority)
        validation_ledger = load_event_ledger_jsonl(
            args.validation_ledger,
            authority,
        )
        applicability_ledger = load_event_ledger_jsonl(
            args.applicability_ledger,
            authority,
        )
        outcome = event_record_from_dict(_read_json(args.outcome))
        updated = update_person_model(
            bundle,
            ledger,
            outcome,
            authority,
            applicability_ledger=applicability_ledger,
            validation_ledger=validation_ledger,
        )
        save_bundle(args.output, updated.bundle)
        save_event_ledger_jsonl(args.output_ledger, updated.ledger)
        print(
            json.dumps(
                {
                    "saved_model": str(args.output),
                    "person_id": updated.bundle.representation.person_id,
                    "observation_count": (
                        updated.bundle.representation.observation_count
                    ),
                    "bundle_version": updated.bundle.bundle_version,
                    "model_id": updated.bundle.manifest.model_id,
                    "parent_model_id": (
                        updated.bundle.manifest.parent_model_id
                    ),
                    "saved_ledger": str(args.output_ledger),
                    "validation_status": (
                        updated.bundle.manifest.validation.status
                    ),
                    "validation_reasons": list(
                        updated.bundle.manifest.validation.reasons
                    ),
                    "temporal_stability_status": (
                        updated.bundle.manifest.validation
                        .temporal_stability_status
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "sign-event":
        authority = load_verification_authority(args.verification_keys)
        observation = observation_from_dict(_read_json(args.observation))
        record = authority.sign(
            event_id=args.event_id,
            observation=observation,
            observed_at=args.observed_at,
            evidence_hash=hashlib.sha256(
                args.evidence.read_bytes()
            ).hexdigest(),
            verifier_id=args.verifier_id,
            verified_at=args.verified_at,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                record.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "saved_event": str(args.output),
                    "event_id": record.event_id,
                    "verifier_id": record.verifier_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "sign-ledger":
        authority = load_verification_authority(args.verification_keys)
        observations = load_observations_jsonl(args.input)
        records = []
        for observation in observations:
            event_id = (
                f"{observation.person_id}:"
                f"{observation.scenario.scenario_id}"
            )
            evidence = json.dumps(
                observation_payload(observation),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            records.append(
                authority.sign(
                    event_id=event_id,
                    observation=observation,
                    observed_at=args.observed_at,
                    evidence_hash=hashlib.sha256(evidence).hexdigest(),
                    verifier_id=args.verifier_id,
                    verified_at=args.verified_at,
                )
            )
        ledger = EventLedger.verify(records, authority)
        save_event_ledger_jsonl(args.output, ledger)
        print(
            json.dumps(
                {
                    "saved_ledger": str(args.output),
                    "event_count": len(ledger.records),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
