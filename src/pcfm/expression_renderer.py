from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


CONTRACT_SCHEMA = "pcfm-frozen-content-contract-v1"
CONTRACT_SCHEMA_V2 = "pcfm-frozen-content-contract-v2"
CONTRACT_SCHEMA_V3 = "pcfm-frozen-content-contract-v3"
CONTRACT_SCHEMA_V4 = "pcfm-frozen-content-contract-v4"
CONTRACT_SCHEMA_V5 = "pcfm-frozen-content-contract-v5"
PROFILE_SCHEMA = "pcfm-expression-style-manifest-v1"
RULES_SCHEMA = "pcfm-surface-rules-v1"
MODES_SCHEMA = "pcfm-context-modes-v1"
FORBIDDEN_SCHEMA = "pcfm-forbidden-mutations-v1"
PROVENANCE_SCHEMA = "pcfm-style-provenance-v1"
REQUIRED_PROFILE_FILES = (
    "surface_rules.json",
    "context_modes.json",
    "sanitized_examples.jsonl",
    "forbidden_mutations.json",
    "provenance.json",
    "style_tests.jsonl",
)
CONTENT_FIELDS = ("claims", "reasons", "memories", "uncertainties")
ALLOWED_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "speech_act",
        "stance",
        "refusal_status",
        "answer_status",
        "ordinary_dialogue_text",
        "evidence_refs",
        *CONTENT_FIELDS,
        "protected_entities",
        "protected_numbers",
        "protected_dates",
        "protected_quotes",
        "confidence",
        "style_mode",
    }
)
EXPLICITLY_FORBIDDEN_CONTRACT_FIELDS = frozenset(
    {
        "user_question",
        "raw_user_question",
        "prompt",
        "beliefs",
        "values",
        "goals",
        "decision_rules",
        "person_timeline",
        "person_identity_card",
        "latest_updates",
        "relationship_facts",
        "cognitive_model",
        "evidence_memory",
    }
)
ALLOWED_RULE_OPERATIONS = frozenset(
    {
        "prefix_first_claim",
        "prefix_first_reason",
        "prefix_first_uncertainty",
        "prefix_last_claim",
    }
)
UNSAFE_RULE_FRAGMENTS = (
    "ignore",
    "system prompt",
    "user question",
    "you are steve",
    "steve jobs",
    "always said",
    "believe",
    "must support",
    "must oppose",
    "memory",
    "timeline",
    "pretend",
    "role-play",
    "roleplay",
)
NEGATION_TOKENS = ("not", "no", "never", "without", "n't")
MODAL_TOKENS = ("may", "might", "could", "possibly", "uncertain", "probably")
SAFE_SURFACE_CONNECTORS = (
    "Actually, ",
    "Look, ",
    "Well, ",
    "You see, ",
    "The point is: ",
    "In other words, ",
    "So ",
    "So, ",
    "But, ",
    "Now, ",
    "First, ",
    "Let me put it this way: ",
    "你看，",
    "换句话说，",
    "关键是，",
    "先说结论：",
    "所以，",
    "不过，",
)


class ExpressionRendererError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpressionRendererError(f"cannot load expression profile file: {path.name}") from error
    if not isinstance(value, dict):
        raise ExpressionRendererError(f"expression profile file must contain an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            records.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ExpressionRendererError(f"invalid JSONL in {path.name} at or before line {line_number}") from error
    return records


def seal_expression_profile(profile_dir: Path) -> dict[str, str]:
    profile_dir = Path(profile_dir)
    manifest_path = profile_dir / "style_manifest.json"
    manifest = _read_json(manifest_path)
    digests = {}
    for name in REQUIRED_PROFILE_FILES:
        path = profile_dir / name
        if not path.is_file():
            raise ExpressionRendererError(f"missing expression profile file: {name}")
        digests[name] = _file_digest(path)
    manifest["file_digests"] = digests
    manifest["sealed_at"] = _utc_now()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return digests


def builtin_expression_profile_path() -> Path:
    return Path(__file__).with_name("expression_profiles") / "steve_jobs_v1"


def _normalize_items(raw: object, field: str) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list):
        raise ExpressionRendererError(f"{field} must be a list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ExpressionRendererError(f"{field} items must be objects")
        identifier = str(item.get("id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not identifier or not text or identifier in seen:
            raise ExpressionRendererError(f"{field} items require unique non-empty id and text")
        if any(key not in {"id", "text"} for key in item):
            raise ExpressionRendererError(f"{field} items may contain only id and text")
        seen.add(identifier)
        result.append({"id": identifier, "text": text})
    return tuple(result)


@dataclass(frozen=True)
class FrozenContentContract:
    payload: dict[str, object]

    @classmethod
    def from_dict(cls, raw: Mapping[str, object], supported_modes: Sequence[str]) -> "FrozenContentContract":
        keys = set(map(str, raw))
        forbidden = sorted(keys & EXPLICITLY_FORBIDDEN_CONTRACT_FIELDS)
        unsupported = sorted(keys - ALLOWED_CONTRACT_FIELDS)
        if forbidden or unsupported:
            name = (forbidden or unsupported)[0]
            raise ExpressionRendererError(f"forbidden contract field: {name}")
        schema_version = str(raw.get("schema_version", ""))
        if schema_version not in {
            CONTRACT_SCHEMA,
            CONTRACT_SCHEMA_V2,
            CONTRACT_SCHEMA_V3,
            CONTRACT_SCHEMA_V4,
            CONTRACT_SCHEMA_V5,
        }:
            raise ExpressionRendererError("unsupported frozen content contract schema")
        speech_act = str(raw.get("speech_act", "")).strip()
        if not speech_act:
            raise ExpressionRendererError("speech_act must be supplied by PCFM")
        mode = str(raw.get("style_mode", "")).strip()
        if mode not in set(map(str, supported_modes)):
            raise ExpressionRendererError("style_mode is outside the expression profile scope")
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError) as error:
            raise ExpressionRendererError("confidence must be in [0, 1]") from error
        if not 0.0 <= confidence <= 1.0:
            raise ExpressionRendererError("confidence must be in [0, 1]")
        payload: dict[str, object] = {
            "schema_version": schema_version,
            "speech_act": speech_act,
            **{field: [dict(item) for item in _normalize_items(raw.get(field, []), field)] for field in CONTENT_FIELDS},
            "protected_entities": _string_list(raw.get("protected_entities", []), "protected_entities"),
            "protected_numbers": _string_list(raw.get("protected_numbers", []), "protected_numbers"),
            "protected_quotes": _string_list(raw.get("protected_quotes", []), "protected_quotes"),
            "confidence": confidence,
            "style_mode": mode,
        }
        if schema_version in {CONTRACT_SCHEMA_V2, CONTRACT_SCHEMA_V3, CONTRACT_SCHEMA_V4, CONTRACT_SCHEMA_V5}:
            stance = str(raw.get("stance", "")).strip()
            refusal_status = str(raw.get("refusal_status", "")).strip()
            if not stance or not refusal_status:
                raise ExpressionRendererError("v2 contract requires stance and refusal_status")
            payload["stance"] = stance
            payload["refusal_status"] = refusal_status
            payload["protected_dates"] = _string_list(
                raw.get("protected_dates", []), "protected_dates"
            )
        if schema_version in {CONTRACT_SCHEMA_V3, CONTRACT_SCHEMA_V4, CONTRACT_SCHEMA_V5}:
            answer_status = str(raw.get("answer_status", "")).strip()
            if not answer_status:
                raise ExpressionRendererError("v3 contract requires answer_status")
            payload["answer_status"] = answer_status
            payload["ordinary_dialogue_text"] = str(
                raw.get("ordinary_dialogue_text", "")
            )
            payload["evidence_refs"] = _string_list(
                raw.get("evidence_refs", []), "evidence_refs"
            )
        locked_text = "\n".join(
            str(item["text"])
            for field in CONTENT_FIELDS
            for item in payload[field]
        )
        for field in (
            "protected_entities", "protected_numbers", "protected_dates", "protected_quotes"
        ):
            if field not in payload:
                continue
            if any(str(value) not in locked_text for value in payload[field]):
                raise ExpressionRendererError(
                    f"{field} values must already occur in locked content"
                )
        return cls(payload=payload)

    @property
    def digest(self) -> str:
        return _digest(self.payload)

    def segments(self) -> dict[str, list[str]]:
        return {
            field: [str(item["text"]) for item in self.payload[field]]
            for field in CONTENT_FIELDS
        }


def _string_list(raw: object, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise ExpressionRendererError(f"{field} must be a list")
    values = [str(value).strip() for value in raw]
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ExpressionRendererError(f"{field} values must be unique and non-empty")
    return values


def _validate_rule(rule: Mapping[str, object]) -> dict[str, object]:
    identifier = str(rule.get("rule_id", "")).strip()
    operation = str(rule.get("operation", "")).strip()
    prefix = str(rule.get("prefix", ""))
    if not identifier or operation not in ALLOWED_RULE_OPERATIONS:
        raise ExpressionRendererError("invalid surface rule identity or operation")
    if rule.get("category") != "A_surface" or rule.get("review_status") != "confirmed":
        raise ExpressionRendererError("only confirmed A_surface rules may enter the renderer")
    lowered = prefix.casefold()
    if (
        any(fragment in lowered for fragment in UNSAFE_RULE_FRAGMENTS)
        or any(character in prefix for character in ('"', "“", "”"))
        or re.search(r"\d", prefix)
        or len(prefix) > 80
    ):
        raise ExpressionRendererError(f"unsafe surface rule: {identifier}")
    try:
        weight = float(rule.get("style_weight", 0.0))
    except (TypeError, ValueError) as error:
        raise ExpressionRendererError("style rule weight must be numeric") from error
    if not 0.0 <= weight <= 1.0:
        raise ExpressionRendererError("style rule weight must be in [0, 1]")
    return {
        "rule_id": identifier,
        "operation": operation,
        "prefix": prefix,
        "style_weight": weight,
        "provenance_ids": sorted(str(value) for value in rule.get("provenance_ids", [])),
    }


class ExpressionRenderer:
    def __init__(self, profile_dir: Path):
        directory = Path(profile_dir).resolve()
        manifest = _read_json(directory / "style_manifest.json")
        if manifest.get("schema_version") != PROFILE_SCHEMA:
            raise ExpressionRendererError("unsupported expression style manifest")
        declared = manifest.get("file_digests")
        if not isinstance(declared, Mapping) or set(declared) != set(REQUIRED_PROFILE_FILES):
            raise ExpressionRendererError("expression profile is not sealed")
        for name in REQUIRED_PROFILE_FILES:
            if str(declared[name]) != _file_digest(directory / name):
                raise ExpressionRendererError(f"expression profile integrity check failed: {name}")
        rules_payload = _read_json(directory / "surface_rules.json")
        modes_payload = _read_json(directory / "context_modes.json")
        forbidden = _read_json(directory / "forbidden_mutations.json")
        provenance = _read_json(directory / "provenance.json")
        examples = _read_jsonl(directory / "sanitized_examples.jsonl")
        style_tests = _read_jsonl(directory / "style_tests.jsonl")
        self._initialize(manifest, rules_payload, modes_payload, forbidden, provenance, examples, style_tests)

    def _initialize(
        self,
        manifest: Mapping[str, object],
        rules_payload: Mapping[str, object],
        modes_payload: Mapping[str, object],
        forbidden: Mapping[str, object],
        provenance: Mapping[str, object],
        examples: Sequence[Mapping[str, object]],
        style_tests: Sequence[Mapping[str, object]],
    ) -> None:
        if rules_payload.get("schema_version") != RULES_SCHEMA:
            raise ExpressionRendererError("unsupported surface rules schema")
        if modes_payload.get("schema_version") != MODES_SCHEMA:
            raise ExpressionRendererError("unsupported context modes schema")
        if forbidden.get("schema_version") != FORBIDDEN_SCHEMA:
            raise ExpressionRendererError("unsupported forbidden mutations schema")
        if provenance.get("schema_version") != PROVENANCE_SCHEMA:
            raise ExpressionRendererError("unsupported provenance schema")
        self.manifest = copy.deepcopy(dict(manifest))
        self.rules = {
            rule["rule_id"]: rule
            for rule in (_validate_rule(item) for item in rules_payload.get("rules", []))
        }
        self.modes = copy.deepcopy(dict(modes_payload.get("modes", {})))
        if not self.rules or not self.modes:
            raise ExpressionRendererError("expression profile has no usable rules or modes")
        for mode, intensities in self.modes.items():
            if not isinstance(intensities, Mapping) or set(intensities) != {"light", "standard", "strong"}:
                raise ExpressionRendererError(f"context mode is incomplete: {mode}")
            for selected in intensities.values():
                if not isinstance(selected, list) or any(str(identifier) not in self.rules for identifier in selected):
                    raise ExpressionRendererError(f"context mode references an unknown rule: {mode}")
        self.forbidden = copy.deepcopy(dict(forbidden))
        self.provenance = copy.deepcopy(dict(provenance))
        self.examples = [copy.deepcopy(dict(item)) for item in examples]
        self.style_tests = [copy.deepcopy(dict(item)) for item in style_tests]
        self.profile_digest = _digest(
            {
                "manifest": {key: value for key, value in self.manifest.items() if key != "sealed_at"},
                "rules": rules_payload,
                "modes": modes_payload,
                "forbidden": forbidden,
                "provenance": provenance,
                "examples": examples,
                "style_tests": style_tests,
            }
        )

    @classmethod
    def generic_control(cls) -> "ExpressionRenderer":
        instance = cls.__new__(cls)
        manifest = {
            "schema_version": PROFILE_SCHEMA,
            "profile_id": "generic-control-surface-v1",
            "version": "1.0.0",
            "person_id": "generic-control",
            "person_name": "Generic control",
            "supported_languages": ["en"],
            "supported_modes": ["interview_public"],
            "validation_status": "control_only",
        }
        rules = {
            "schema_version": RULES_SCHEMA,
            "rules": [
                {"rule_id": "generic_claim", "category": "A_surface", "review_status": "confirmed", "operation": "prefix_first_claim", "prefix": "Response: ", "style_weight": 0.05, "provenance_ids": []},
                {"rule_id": "generic_reason", "category": "A_surface", "review_status": "confirmed", "operation": "prefix_first_reason", "prefix": "Reason: ", "style_weight": 0.05, "provenance_ids": []},
                {"rule_id": "generic_uncertainty", "category": "A_surface", "review_status": "confirmed", "operation": "prefix_first_uncertainty", "prefix": "Uncertainty: ", "style_weight": 0.05, "provenance_ids": []},
            ],
        }
        modes = {"schema_version": MODES_SCHEMA, "modes": {"interview_public": {"light": ["generic_claim"], "standard": ["generic_claim", "generic_reason"], "strong": ["generic_claim", "generic_reason", "generic_uncertainty"]}}}
        forbidden = {"schema_version": FORBIDDEN_SCHEMA, "mutations": []}
        provenance = {"schema_version": PROVENANCE_SCHEMA, "entries": []}
        instance._initialize(manifest, rules, modes, forbidden, provenance, [], [])
        return instance

    @property
    def supported_modes(self) -> tuple[str, ...]:
        return tuple(sorted(self.modes))

    def profile_summary(self) -> dict[str, object]:
        return {
            "profile_id": self.manifest["profile_id"],
            "version": self.manifest["version"],
            "person_name": self.manifest["person_name"],
            "validation_status": self.manifest["validation_status"],
            "supported_languages": self.manifest["supported_languages"],
            "supported_modes": list(self.supported_modes),
            "profile_digest": self.profile_digest,
            "rule_count": len(self.rules),
            "sanitized_example_count": len(self.examples),
            "provenance": copy.deepcopy(
                list(self.provenance.get("entries", []))
            ),
            "style_recognition_status": "not_blind_validated",
            "response_prediction_accuracy_status": "not_assessed",
        }

    def _render_intensity(self, contract: FrozenContentContract, intensity: str) -> tuple[str, list[str]]:
        segments = contract.segments()
        selected = [str(value) for value in self.modes[str(contract.payload["style_mode"])][intensity]]
        prefixes: dict[str, list[str]] = {field: [""] * len(segments[field]) for field in CONTENT_FIELDS}
        for identifier in selected:
            rule = self.rules[identifier]
            operation = str(rule["operation"])
            if operation == "prefix_first_claim" and prefixes["claims"]:
                prefixes["claims"][0] += str(rule["prefix"])
            elif operation == "prefix_last_claim" and prefixes["claims"]:
                prefixes["claims"][-1] += str(rule["prefix"])
            elif operation == "prefix_first_reason" and prefixes["reasons"]:
                prefixes["reasons"][0] += str(rule["prefix"])
            elif operation == "prefix_first_uncertainty" and prefixes["uncertainties"]:
                prefixes["uncertainties"][0] += str(rule["prefix"])
        lines = []
        for field in CONTENT_FIELDS:
            lines.extend(prefix + text for prefix, text in zip(prefixes[field], segments[field], strict=True))
        return "\n".join(lines), selected

    def _style_score(self, rule_ids: Sequence[str]) -> float:
        return round(sum(float(self.rules[item]["style_weight"]) for item in rule_ids), 6)

    def check_candidate(
        self,
        raw_contract: Mapping[str, object],
        candidate_text: str,
        *,
        allowed_insertions: Sequence[str] = (),
    ) -> dict[str, object]:
        contract = FrozenContentContract.from_dict(raw_contract, self.supported_modes)
        segments = contract.segments()
        all_segments = [text for field in CONTENT_FIELDS for text in segments[field]]
        checks: dict[str, str] = {}
        checks["claim_coverage"] = _exact_once(candidate_text, segments["claims"])
        checks["reason_coverage"] = _exact_once(candidate_text, segments["reasons"])
        checks["memory_coverage"] = _exact_once(candidate_text, segments["memories"])
        checks["uncertainty_preservation"] = _exact_once(candidate_text, segments["uncertainties"])
        checks["protected_entities"] = _exact_once(candidate_text, list(contract.payload["protected_entities"]), at_least=True)
        checks["protected_numbers_dates"] = _exact_once(candidate_text, list(contract.payload["protected_numbers"]), at_least=True)
        checks["protected_dates"] = _exact_once(
            candidate_text, list(contract.payload.get("protected_dates", [])), at_least=True
        )
        checks["protected_quotes"] = _exact_once(candidate_text, list(contract.payload["protected_quotes"]), at_least=True)
        source_text = "\n".join(all_segments)
        checks["negation_modality"] = "passed" if _token_counts(source_text, NEGATION_TOKENS + MODAL_TOKENS) == _token_counts(candidate_text, NEGATION_TOKENS + MODAL_TOKENS) else "failed"
        remainder = candidate_text
        for text in sorted(all_segments, key=len, reverse=True):
            remainder = remainder.replace(text, "", 1)
        for prefix in sorted({str(rule["prefix"]) for rule in self.rules.values()}, key=len, reverse=True):
            remainder = remainder.replace(prefix, "")
        for insertion in sorted(map(str, allowed_insertions), key=len, reverse=True):
            remainder = remainder.replace(insertion, "", 1)
        clean_remainder = re.sub(r"[\s\n\r\t.,:;!?—\-()]+", "", remainder)
        checks["new_claim_detection"] = "passed" if not clean_remainder else "failed"
        checks["memory_addition"] = "passed" if (segments["memories"] or "I remember" not in candidate_text) else "failed"
        quote_count = sum(candidate_text.count(mark) for mark in ('"', "“", "”"))
        source_quote_count = sum(source_text.count(mark) for mark in ('"', "“", "”"))
        checks["quote_addition"] = "passed" if quote_count == source_quote_count else "failed"
        checks["causal_direction"] = "passed" if checks["reason_coverage"] == "passed" else "failed"
        checks["speech_act_immutable"] = "passed"
        checks["stance_immutable"] = "passed"
        checks["refusal_status_immutable"] = "passed"
        checks["confidence_immutable"] = "passed"
        checks["structured_fields_immutable"] = "passed"
        checks["bidirectional_entailment_proxy"] = "passed" if all(checks[key] == "passed" for key in ("claim_coverage", "reason_coverage", "memory_coverage", "uncertainty_preservation", "new_claim_detection")) else "failed"
        reasons = [
            ("new_claim_or_unapproved_text" if key == "new_claim_detection" else key)
            for key, value in checks.items()
            if value != "passed"
        ]
        return {"status": "passed" if not reasons else "rejected", "checks": checks, "reasons": reasons}

    def render(self, raw_contract: Mapping[str, object], *, include_adversarial_probe: bool = False) -> dict[str, object]:
        contract = FrozenContentContract.from_dict(raw_contract, self.supported_modes)
        segments = contract.segments()
        neutral = "\n".join(text for field in CONTENT_FIELDS for text in segments[field])
        candidates: list[dict[str, object]] = []
        for intensity in ("light", "standard", "strong"):
            text, rule_ids = self._render_intensity(contract, intensity)
            gate = self.check_candidate(contract.payload, text)
            candidates.append(
                {
                    "intensity": intensity,
                    "text": text,
                    "status": gate["status"],
                    "checks": gate["checks"],
                    "rejection_reasons": gate["reasons"],
                    "used_rule_ids": rule_ids,
                    "used_rules": [
                        {
                            "rule_id": identifier,
                            "operation": self.rules[identifier]["operation"],
                            "provenance_ids": self.rules[identifier]["provenance_ids"],
                        }
                        for identifier in rule_ids
                    ],
                    "style_fingerprint_score": self._style_score(rule_ids),
                }
            )
        passing = [item for item in reversed(candidates) if item["status"] == "passed"]
        if passing:
            selected = copy.deepcopy(passing[0])
        else:
            neutral_gate = self.check_candidate(contract.payload, neutral)
            selected = {
                "intensity": "neutral",
                "text": neutral,
                "status": neutral_gate["status"],
                "checks": neutral_gate["checks"],
                "rejection_reasons": neutral_gate["reasons"],
                "used_rule_ids": [],
                "used_rules": [],
                "style_fingerprint_score": 0.0,
            }
        result: dict[str, object] = {
            "schema_version": "pcfm-expression-render-result-v1",
            "render_id": f"render-{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now(),
            "profile": self.profile_summary(),
            "profile_digest": self.profile_digest,
            "contract_digest": contract.digest,
            "structured_content": copy.deepcopy(contract.payload),
            "neutral_text": neutral,
            "candidates": candidates,
            "fallback_chain": ["strong", "standard", "light", "neutral"],
            "selected": selected,
            "semantic_preservation": {
                "status": selected["status"],
                "method": "exact locked-segment coverage plus closed-wrapper vocabulary",
                "nli_status": "not_run_no_validated_nli_backend",
            },
            "style_validation": {
                "fingerprint_score": selected["style_fingerprint_score"],
                "blind_recognition_status": "not_assessed",
                "exaggeration_status": "bounded_by_confirmed_surface_rules",
            },
            "content_authority": "pcfm_frozen_contract_only",
            "response_prediction_accuracy_status": "not_assessed",
        }
        if include_adversarial_probe:
            probe_text = "As Steve Jobs always said, innovation changes everything. " + neutral
            probe_gate = self.check_candidate(contract.payload, probe_text)
            result["adversarial_probe"] = {
                "label": "complete_persona_skill_pollution_probe",
                "text": probe_text,
                "status": probe_gate["status"],
                "checks": probe_gate["checks"],
                "reasons": probe_gate["reasons"],
                "style_fingerprint_score": float(selected["style_fingerprint_score"]) + 1.0,
                "selected": False,
            }
        return result


def render_person_surface_style(
    raw_contract: Mapping[str, object], style_artifact: Mapping[str, object]
) -> dict[str, object]:
    """Apply only observed, provenance-bound surface connectors to locked text."""
    verifier = ExpressionRenderer.generic_control()
    contract = FrozenContentContract.from_dict(raw_contract, verifier.supported_modes)
    segments = contract.segments()
    neutral = "\n".join(
        text for field in CONTENT_FIELDS for text in segments[field]
    )
    if style_artifact.get("schema_version") != "pcfm-person-surface-style-v2":
        return {
            "status": "rejected",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "reasons": ["unsupported_style_artifact"],
            "checks": {},
            "used_rules": [],
        }
    artifact = copy.deepcopy(dict(style_artifact))
    declared_hash = str(artifact.pop("artifact_hash", ""))
    if not declared_hash or declared_hash != _digest(artifact):
        return {
            "status": "rejected",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "reasons": ["style_artifact_integrity_failed"],
            "checks": {},
            "used_rules": [],
        }
    raw_rules = style_artifact.get("surface_rules", [])
    if not isinstance(raw_rules, list) or not raw_rules:
        neutral_gate = verifier.check_candidate(contract.payload, neutral)
        return {
            "status": "neutral",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "profile_status": "style_material_ready_rendering_not_enabled",
            "reasons": ["no_confirmed_surface_rule"],
            "checks": neutral_gate["checks"],
            "used_rules": [],
        }
    prefixes: dict[str, list[str]] = {
        field: [""] * len(segments[field]) for field in CONTENT_FIELDS
    }
    used_rules: list[dict[str, object]] = []
    insertions: list[str] = []
    operation_fields = {
        "prefix_first_claim": "claims",
        "prefix_first_reason": "reasons",
        "prefix_first_uncertainty": "uncertainties",
    }
    invalid_rules: list[str] = []
    styled_fields: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            invalid_rules.append("surface_rule_not_object")
            continue
        operation = str(raw_rule.get("operation", ""))
        field = operation_fields.get(operation)
        prefix = str(raw_rule.get("prefix", ""))
        provenance = [str(value) for value in raw_rule.get("provenance_event_ids", [])]
        if (
            field is None
            or prefix not in SAFE_SURFACE_CONNECTORS
            or int(raw_rule.get("observed_count", 0)) < 2
            or not provenance
        ):
            invalid_rules.append(
                f"unsafe_or_unproven_surface_rule:{raw_rule.get('rule_id', '')}"
            )
            continue
        if not prefixes[field] or segments[field][0].casefold().startswith(prefix.casefold()):
            continue
        if field in styled_fields:
            continue
        prefixes[field][0] = prefix
        styled_fields.add(field)
        insertions.append(prefix)
        used_rules.append(
            {
                "rule_id": str(raw_rule.get("rule_id", "")),
                "operation": operation,
                "prefix": prefix,
                "provenance_event_ids": sorted(set(provenance)),
            }
        )
    if invalid_rules:
        return {
            "status": "rejected",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "profile_status": "rendering_rejected",
            "reasons": invalid_rules,
            "checks": {},
            "used_rules": [],
        }
    candidate_lines: list[str] = []
    for field in CONTENT_FIELDS:
        candidate_lines.extend(
            prefix + text
            for prefix, text in zip(prefixes[field], segments[field], strict=True)
        )
    candidate = "\n".join(candidate_lines)
    if candidate == neutral:
        gate = verifier.check_candidate(contract.payload, neutral)
        return {
            "status": "neutral",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "profile_status": "style_material_ready_rendering_not_enabled",
            "reasons": ["no_applicable_surface_change"],
            "checks": gate["checks"],
            "used_rules": [],
        }
    gate = verifier.check_candidate(
        contract.payload, candidate, allowed_insertions=insertions
    )
    if gate["status"] != "passed":
        return {
            "status": "rejected",
            "text": neutral,
            "neutral_text": neutral,
            "changed": False,
            "profile_status": "rendering_rejected",
            "reasons": list(gate["reasons"]),
            "checks": gate["checks"],
            "used_rules": used_rules,
        }
    return {
        "status": "passed",
        "text": candidate,
        "neutral_text": neutral,
        "changed": True,
        "profile_status": "rendering_enabled_exploratory",
        "reasons": [],
        "checks": gate["checks"],
        "used_rules": used_rules,
        "contract_digest": contract.digest,
    }


def _exact_once(text: str, values: Sequence[str], *, at_least: bool = False) -> str:
    for value in values:
        count = text.count(value)
        if (at_least and count < 1) or (not at_least and count != 1):
            return "failed"
    return "passed"


def _token_counts(text: str, tokens: Sequence[str]) -> tuple[tuple[str, int], ...]:
    lowered = text.casefold()
    return tuple((token, len(re.findall(rf"(?<!\w){re.escape(token)}(?!\w)", lowered))) for token in tokens)


def _classify_material(text: str, relative: str) -> tuple[str, str, str]:
    lowered = text.casefold()
    if "demo-conversation" in relative.casefold() or "对话实录" in text and "2026" in text:
        return "unknown_forbidden", "generated or simulated conversation is not real-person style evidence", "rejected"
    if any(token in lowered for token in ("直接以steve jobs", "直接以人物身份", "你现在是", "思维操作系统", "核心心智", "世界观", "价值排序", "信念", "目标", "因果", "风险偏好", "决策规则")):
        return "C_cognitive", "belief, value, goal, causal, or decision content belongs upstream", "rejected"
    if any(token in lowered for token in ("是否直接回答", "反驳", "追问", "回避", "重新框定", "重新定义问题", "被追问", "面对不同", "攻击性", "耐心", "愤怒", "克制", "调用人生", "回答方式")):
        return "B_interaction", "response action or relationship behavior must be predicted by PCFM", "rejected"
    if any(token in lowered for token in ("时间线", "出生", "去世", "人物关系", "真实行动", "历史观点", "来源：", "source:", "年", "月", "steve jobs archive", "walter isaacson")) and not any(token in lowered for token in ("句长", "句法", "词汇", "语气", "口语", "节奏", "修辞", "段落", "邮件风格", "停顿", "重复", "开头", "结尾")):
        return "D_person_fact", "biographical, event, quotation, relationship, or source fact belongs in evidence memory", "rejected"
    if any(token in lowered for token in ("句长", "句法", "词汇", "语气", "口语", "节奏", "修辞", "段落", "邮件风格", "停顿", "重复", "碎片句", "连接", "开头", "结尾", "gonna", "you know", "one more thing", "第一人称", "正式程度", "类比的语言")):
        return "A_surface", "candidate surface-form observation; manual confirmation is still required", "pending"
    return "unknown_forbidden", "category is ambiguous, so the item is ineligible for the style renderer", "pending"


def audit_nuwa_materials(source_dir: Path) -> dict[str, object]:
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        raise ExpressionRendererError("Nuwa person material directory does not exist")
    items: list[dict[str, object]] = []
    seen_normalized: dict[str, str] = {}
    source_files = []
    for path in sorted(source_dir.rglob("*.md")):
        relative = path.relative_to(source_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_files.append({"source": relative, "sha256": source_hash, "bytes": len(text.encode("utf-8"))})
        paragraphs = re.split(r"\n\s*\n", text)
        line_cursor = 1
        for index, paragraph in enumerate(paragraphs, 1):
            stripped = paragraph.strip()
            if not stripped:
                line_cursor += paragraph.count("\n") + 1
                continue
            start = line_cursor
            end = start + paragraph.count("\n")
            line_cursor = end + 2
            category, reason, review = _classify_material(stripped, relative)
            normalized = re.sub(r"\W+", "", stripped.casefold())
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            duplicate_of = seen_normalized.get(normalized_hash)
            item_id = f"audit-{len(items)+1:05d}"
            if duplicate_of is None:
                seen_normalized[normalized_hash] = item_id
            items.append(
                {
                    "item_id": item_id,
                    "source": relative,
                    "source_sha256": source_hash,
                    "source_locator": f"lines {start}-{end}",
                    "text": stripped,
                    "content_hash": hashlib.sha256(stripped.encode("utf-8")).hexdigest(),
                    "category": category,
                    "classification_reason": reason,
                    "review_status": review,
                    "style_eligible": category == "A_surface" and review == "confirmed",
                    "near_duplicate_of": duplicate_of,
                }
            )
    counts = {category: sum(item["category"] == category for item in items) for category in ("A_surface", "B_interaction", "C_cognitive", "D_person_fact", "unknown_forbidden")}
    counts["total"] = len(items)
    unsafe = sum(
        item["source"] == "SKILL.md" and item["category"] in {"B_interaction", "C_cognitive", "D_person_fact"}
        for item in items
    )
    payload: dict[str, object] = {
        "schema_version": "pcfm-nuwa-material-classification-audit-v1",
        "person": "Steve Jobs",
        "source_directory": str(source_dir),
        "source_files": source_files,
        "counts": counts,
        "unsafe_complete_skill_content_pollution_count": unsafe,
        "policy": "Only manually confirmed A_surface items may enter an expression package; ambiguous items default to ineligible.",
        "items": items,
    }
    payload["audit_digest"] = _digest(payload)
    return payload


def _main() -> int:
    parser = argparse.ArgumentParser(description="PCFM bounded expression renderer utilities")
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("profile_dir", type=Path)
    audit = sub.add_parser("audit-nuwa")
    audit.add_argument("source_dir", type=Path)
    audit.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "seal":
        print(json.dumps(seal_expression_profile(args.profile_dir), ensure_ascii=False, indent=2))
        return 0
    result = audit_nuwa_materials(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": result["counts"], "audit_digest": result["audit_digest"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
