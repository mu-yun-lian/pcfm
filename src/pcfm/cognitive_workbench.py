from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "pcfm-cognitive-workbench-v1"
CARD_SCHEMA_VERSION = "pcfm-cognitive-card-v1"
SCENARIO_SCHEMA_VERSION = "pcfm-cognitive-scenario-v1"
PREDICTION_SCHEMA_VERSION = "pcfm-cognitive-prediction-v1"
EVIDENCE_ROLES = {
    "direct_observation",
    "person_self_report",
    "external_fact",
    "model_inference",
}
MODEL_CATEGORIES = (
    "beliefs",
    "values",
    "causal_assumptions",
    "decision_rules",
    "risk_preferences",
    "dynamic_state",
    "contradictions",
    "unknowns",
)
MODEL_STATUSES = {"observed", "inferred", "contested", "unknown"}
MIN_CONFIRMED_EVIDENCE = 4
MIN_DISTINCT_EVIDENCE_DATES = 3
MAX_EVIDENCE_AGE_DAYS = 1460


class CognitiveWorkbenchError(ValueError):
    """A user-facing cognitive-workbench contract error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _artifact_digest(value: Mapping[str, object]) -> str:
    return _digest({key: item for key, item in value.items() if key != "artifact_digest"})


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise CognitiveWorkbenchError("日期时间格式无效。") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CognitiveWorkbenchError("日期时间必须包含时区。")
    return parsed


def _parse_date(value: str) -> str:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
    except ValueError as error:
        raise CognitiveWorkbenchError("证据日期必须使用 YYYY-MM-DD。") from error


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value).lower()))


def _status_multiplier(status: str) -> float:
    return {
        "observed": 1.0,
        "inferred": 0.62,
        "contested": 0.25,
        "unknown": 0.0,
    }[status]


def _sigmoid(value: float) -> float:
    clipped = max(-20.0, min(20.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def score_cognitive_scenario(
    card: Mapping[str, object], scenario: Mapping[str, object]
) -> dict[str, object]:
    """The single deterministic deployed cognitive scoring kernel."""
    scope = dict(card["scope"])
    reasons: list[str] = []
    if scenario.get("review_status") != "confirmed":
        reasons.append("scenario_not_confirmed")
    if scenario.get("domain") != scope.get("domain"):
        reasons.append("domain_out_of_scope")
    if scenario.get("decision_type") != scope.get("decision_type"):
        reasons.append("decision_type_out_of_scope")
    if reasons:
        return {
            "status": "refused",
            "reasons": reasons,
            "applicability": {"status": "out_of_scope", "reasons": reasons},
        }

    prediction_at = _parse_time(str(scenario["prediction_at"]))
    latest_evidence = _parse_time(str(card["latest_evidence_date"]) + "T23:59:59Z")
    if prediction_at < latest_evidence:
        reasons.append("prediction_precedes_model_evidence")
    age_days = (prediction_at - latest_evidence).total_seconds() / 86400.0
    if age_days > MAX_EVIDENCE_AGE_DAYS:
        reasons.append("cognitive_evidence_stale")

    factors = {str(key): float(value) for key, value in dict(scenario["factor_values"]).items()}
    definitions = {str(item["name"]): dict(item) for item in card["factor_definitions"]}
    if set(factors) != set(definitions):
        reasons.append("factor_schema_mismatch")
    if any(not 0.0 <= value <= 1.0 for value in factors.values()):
        reasons.append("factor_value_out_of_range")
    supported = {
        name
        for item in card["all_items"]
        for name, weight in dict(item.get("factor_weights", {})).items()
        if abs(float(weight)) > 1e-12 and item.get("status") != "unknown"
    }
    unsupported = sorted(
        name for name, value in factors.items() if value >= 0.75 and name not in supported
    )
    if unsupported:
        reasons.append("unsupported_high_impact_factor")
    if reasons:
        return {
            "status": "refused",
            "reasons": reasons,
            "applicability": {
                "status": "out_of_scope",
                "reasons": reasons,
                "unsupported_factors": unsupported,
                "evidence_age_days": age_days,
            },
        }

    evidence_lookup = {str(item["evidence_id"]): dict(item) for item in card["evidence_snapshot"]}
    contributions: list[dict[str, object]] = []
    factor_coefficients = {name: 0.0 for name in factors}
    raw_score = 0.0
    for raw_item in sorted(card["all_items"], key=lambda item: str(item["item_id"])):
        item = dict(raw_item)
        status = str(item["status"])
        multiplier = _status_multiplier(status)
        weights = {str(key): float(value) for key, value in dict(item.get("factor_weights", {})).items()}
        normalization = sum(abs(value) for value in weights.values()) or 1.0
        activation = sum(weights.get(name, 0.0) * factors[name] for name in factors) / normalization
        confidence = float(item["confidence"])
        direction = float(item.get("decision_weight", 0.0))
        contribution = activation * confidence * multiplier * direction
        raw_score += contribution
        for name, weight in weights.items():
            if name in factor_coefficients:
                factor_coefficients[name] += (
                    weight / normalization * confidence * multiplier * direction
                )
        if abs(contribution) > 1e-12:
            evidence = [
                evidence_lookup[evidence_id]
                for evidence_id in item["evidence_ids"]
                if evidence_id in evidence_lookup
            ]
            contributions.append(
                {
                    "item_id": item["item_id"],
                    "category": item["category"],
                    "statement": item["statement"],
                    "status": status,
                    "confidence": confidence,
                    "activation": activation,
                    "contribution": contribution,
                    "evidence": evidence,
                }
            )

    confirmed_count = len(card["evidence_snapshot"])
    evidence_coverage = min(1.0, confirmed_count / 6.0)
    unresolved_factor_count = sum(abs(value - 0.5) < 0.08 for value in factors.values())
    uncertainty_multiplier = max(0.45, 1.0 - 0.08 * unresolved_factor_count)
    deployed_score = raw_score * evidence_coverage * uncertainty_multiplier / 1.5
    probability_b = _sigmoid(deployed_score)
    contributions.sort(key=lambda item: (-abs(float(item["contribution"])), str(item["item_id"])))

    flip_conditions: list[dict[str, object]] = []
    for name in sorted(factor_coefficients):
        coefficient = factor_coefficients[name]
        if abs(coefficient) < 1e-9:
            continue
        target = factors[name] - raw_score / coefficient
        if 0.0 <= target <= 1.0 and abs(target - factors[name]) >= 0.05:
            flip_conditions.append(
                {
                    "factor": name,
                    "label": definitions[name]["label"],
                    "current": factors[name],
                    "threshold": target,
                    "direction": "increase" if target > factors[name] else "decrease",
                }
            )
    flip_conditions.sort(key=lambda item: abs(float(item["threshold"]) - float(item["current"])))

    kernel_basis = {
        "card_digest": card["artifact_digest"],
        "factor_values": {name: factors[name] for name in sorted(factors)},
        "raw_score": raw_score,
        "deployed_score": deployed_score,
        "probability_b": probability_b,
    }
    return {
        "status": "predicted",
        "probability_a": 1.0 - probability_b,
        "probability_b": probability_b,
        "predicted_choice": 1 if probability_b >= 0.5 else 0,
        "raw_score": raw_score,
        "deployed_score": deployed_score,
        "probability_kind": "uncalibrated_evidence_constrained_score",
        "drivers": contributions[:6],
        "flip_conditions": flip_conditions[:6],
        "unknowns": list(card["unknowns"]),
        "applicability": {
            "status": "within_scope",
            "domain": scope["domain"],
            "decision_type": scope["decision_type"],
            "evidence_age_days": age_days,
            "unresolved_factor_count": unresolved_factor_count,
            "evidence_coverage": evidence_coverage,
        },
        "kernel_check": _digest(kernel_basis),
    }


class CognitiveWorkbench:
    def __init__(self, people_dir: Path) -> None:
        self.people_dir = Path(people_dir).resolve()

    def _person_dir(self, person_id: str) -> Path:
        path = (self.people_dir / str(person_id)).resolve()
        if path.parent != self.people_dir:
            raise CognitiveWorkbenchError("人物编号无效。")
        return path

    def _cognitive_dir(self, person_id: str) -> Path:
        return self._person_dir(person_id) / "cognitive"

    def _path(self, person_id: str, name: str) -> Path:
        return self._cognitive_dir(person_id) / name

    def _list(self, person_id: str, name: str) -> list[dict[str, object]]:
        raw = _read_json(self._path(person_id, name), [])
        if not isinstance(raw, list):
            raise CognitiveWorkbenchError(f"认知工作台文件损坏：{name}")
        return [dict(item) for item in raw]

    def _config(self, person_id: str) -> dict[str, object]:
        raw = _read_json(self._path(person_id, "config.json"), {})
        if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
            raise CognitiveWorkbenchError("这个人物还没有可用的认知建模范围配置。")
        return dict(raw)

    def has_config(self, person_id: str) -> bool:
        return self._path(person_id, "config.json").exists()

    def initialize_person(self, person_id: str, config: Mapping[str, object]) -> None:
        directory = self._cognitive_dir(person_id)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": SCHEMA_VERSION, **dict(config)}
        _write_json(directory / "config.json", payload)
        for name in (
            "evidence.json",
            "card_versions.json",
            "scenarios.json",
            "cognitive_predictions.json",
            "cognitive_outcomes.json",
        ):
            path = directory / name
            if not path.exists():
                _write_json(path, [])

    def seed_case(self, person_id: str, case: Mapping[str, object]) -> None:
        person = dict(case["person"])
        self.initialize_person(
            person_id,
            {
                "domain": person["domain"],
                "decision_type": person["decision_type"],
                "option_a": person["option_a"],
                "option_b": person["option_b"],
                "factor_definitions": case["factor_definitions"],
                "unknown_templates": case.get("unknown_templates", []),
                "evidence_window": person["evidence_window"],
                "suggested_scenario_text": person["suggested_scenario_text"],
                "suggested_prediction_at": person.get("suggested_prediction_at"),
                "holdout_notice": person["holdout_notice"],
            },
        )
        if not self._list(person_id, "evidence.json"):
            evidence = [self._normalize_evidence(item) for item in case["evidence"]]
            _write_json(self._path(person_id, "evidence.json"), evidence)

    def _normalize_evidence(self, raw: Mapping[str, object]) -> dict[str, object]:
        required = (
            "text",
            "source",
            "date",
            "context",
            "position",
            "explicit_rationale",
            "role",
            "domain",
            "source_locator",
        )
        missing = [name for name in required if not str(raw.get(name, "")).strip()]
        if missing:
            raise CognitiveWorkbenchError(f"证据缺少字段：{', '.join(missing)}")
        role = str(raw["role"])
        if role not in EVIDENCE_ROLES:
            raise CognitiveWorkbenchError("证据角色无效。")
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError) as error:
            raise CognitiveWorkbenchError("证据可信度必须是 0 到 1 的数值。") from error
        if not 0.0 <= confidence <= 1.0:
            raise CognitiveWorkbenchError("证据可信度必须位于 0 到 1。")
        candidate_items = []
        for candidate in raw.get("candidate_model_items", []):
            item = dict(candidate)
            category = str(item.get("category", ""))
            status = str(item.get("status", "inferred"))
            if category not in MODEL_CATEGORIES or status not in MODEL_STATUSES:
                raise CognitiveWorkbenchError("候选模型项的类别或状态无效。")
            candidate_items.append(
                {
                    "item_id": str(item.get("item_id", "")).strip()
                    or f"item-{_digest(item)[:12]}",
                    "category": category,
                    "statement": str(item.get("statement", "")).strip(),
                    "status": status,
                    "decision_weight": float(item.get("decision_weight", 0.0)),
                    "factor_weights": {
                        str(key): float(value)
                        for key, value in dict(item.get("factor_weights", {})).items()
                    },
                }
            )
        evidence = {
            "schema_version": SCHEMA_VERSION,
            "evidence_id": str(raw.get("evidence_id", "")).strip()
            or f"evidence-{uuid.uuid4().hex[:12]}",
            "text": str(raw["text"]).strip(),
            "source": str(raw["source"]).strip(),
            "source_title": str(raw.get("source_title", "")).strip(),
            "date": _parse_date(str(raw["date"])),
            "context": str(raw["context"]).strip(),
            "position": str(raw["position"]).strip(),
            "explicit_rationale": str(raw["explicit_rationale"]).strip(),
            "role": role,
            "domain": str(raw["domain"]).strip(),
            "is_conflicting": bool(raw.get("is_conflicting", False)),
            "conflict_with": sorted(str(value) for value in raw.get("conflict_with", [])),
            "confidence": confidence,
            "source_locator": str(raw["source_locator"]).strip(),
            "extraction_method": str(raw.get("extraction_method", "manual_candidate")),
            "candidate_model_items": candidate_items,
            "review_status": "pending",
            "reviewed_at": None,
            "created_at": _utc_now(),
        }
        evidence["content_hash"] = _digest(
            {
                key: evidence[key]
                for key in (
                    "text",
                    "source",
                    "date",
                    "context",
                    "position",
                    "explicit_rationale",
                    "role",
                    "domain",
                    "source_locator",
                    "candidate_model_items",
                )
            }
        )
        return evidence

    def evidence(self, person_id: str) -> list[dict[str, object]]:
        return sorted(
            self._list(person_id, "evidence.json"),
            key=lambda item: (str(item["date"]), str(item["evidence_id"])),
        )

    def add_evidence(self, person_id: str, raw: Mapping[str, object]) -> dict[str, object]:
        self._config(person_id)
        evidence = self.evidence(person_id)
        item = self._normalize_evidence(raw)
        if any(existing["content_hash"] == item["content_hash"] for existing in evidence):
            raise CognitiveWorkbenchError("这条证据与已有材料重复。")
        if any(existing["evidence_id"] == item["evidence_id"] for existing in evidence):
            raise CognitiveWorkbenchError("证据编号重复。")
        evidence.append(item)
        _write_json(self._path(person_id, "evidence.json"), evidence)
        return item

    def review_evidence(self, person_id: str, evidence_id: str, decision: str) -> dict[str, object]:
        if decision not in {"confirmed", "rejected"}:
            raise CognitiveWorkbenchError("审核结果只能是 confirmed 或 rejected。")
        evidence = self.evidence(person_id)
        try:
            item = next(value for value in evidence if value["evidence_id"] == evidence_id)
        except StopIteration as error:
            raise CognitiveWorkbenchError("找不到这条证据。") from error
        item["review_status"] = decision
        item["reviewed_at"] = _utc_now()
        _write_json(self._path(person_id, "evidence.json"), evidence)
        return dict(item)

    def _verified_cards(self, person_id: str) -> list[dict[str, object]]:
        cards = self._list(person_id, "card_versions.json")
        for card in cards:
            if card.get("schema_version") != CARD_SCHEMA_VERSION:
                raise CognitiveWorkbenchError("认知模型卡版本不受支持。")
            if card.get("artifact_digest") != _artifact_digest(card):
                raise CognitiveWorkbenchError("认知模型卡完整性校验失败。")
        return cards

    def latest_card(self, person_id: str) -> dict[str, object] | None:
        cards = self._verified_cards(person_id)
        return dict(cards[-1]) if cards else None

    def _build_card_items(
        self, confirmed: Sequence[Mapping[str, object]], config: Mapping[str, object]
    ) -> list[dict[str, object]]:
        grouped: dict[str, dict[str, object]] = {}
        evidence_by_id = {str(item["evidence_id"]): item for item in confirmed}
        for evidence in sorted(confirmed, key=lambda item: str(item["evidence_id"])):
            for raw in evidence.get("candidate_model_items", []):
                candidate = dict(raw)
                identifier = str(candidate["item_id"])
                scoring = {
                    "decision_weight": float(candidate["decision_weight"]),
                    "factor_weights": {
                        str(key): float(value)
                        for key, value in dict(candidate["factor_weights"]).items()
                    },
                }
                if identifier not in grouped:
                    grouped[identifier] = {
                        "item_id": identifier,
                        "category": str(candidate["category"]),
                        "statement": str(candidate["statement"]),
                        "status": str(candidate["status"]),
                        "evidence_ids": [],
                        "confidences": [],
                        **scoring,
                    }
                existing = grouped[identifier]
                if (
                    existing["category"] != candidate["category"]
                    or existing["statement"] != candidate["statement"]
                    or existing["decision_weight"] != scoring["decision_weight"]
                    or existing["factor_weights"] != scoring["factor_weights"]
                ):
                    raise CognitiveWorkbenchError("同编号候选模型项存在不一致定义。")
                existing["evidence_ids"].append(str(evidence["evidence_id"]))
                existing["confidences"].append(float(evidence["confidence"]))
                if evidence.get("is_conflicting"):
                    existing["status"] = "contested"

        result = []
        scope = {
            "domain": config["domain"],
            "decision_type": config["decision_type"],
        }
        for identifier in sorted(grouped):
            item = grouped[identifier]
            dates = [str(evidence_by_id[value]["date"]) for value in item["evidence_ids"]]
            result.append(
                {
                    "item_id": item["item_id"],
                    "category": item["category"],
                    "statement": item["statement"],
                    "status": item["status"],
                    "evidence_ids": sorted(set(item["evidence_ids"])),
                    "valid_from": min(dates),
                    "valid_to": None,
                    "scope": scope,
                    "confidence": sum(item["confidences"]) / len(item["confidences"]),
                    "decision_weight": item["decision_weight"],
                    "factor_weights": item["factor_weights"],
                }
            )
        for raw in config.get("unknown_templates", []):
            item = dict(raw)
            evidence_ids = [
                str(value) for value in item.get("evidence_ids", []) if str(value) in evidence_by_id
            ]
            if not evidence_ids:
                continue
            dates = [str(evidence_by_id[value]["date"]) for value in evidence_ids]
            result.append(
                {
                    "item_id": str(item["item_id"]),
                    "category": "unknowns",
                    "statement": str(item["statement"]),
                    "status": "unknown",
                    "evidence_ids": sorted(evidence_ids),
                    "valid_from": min(dates),
                    "valid_to": None,
                    "scope": scope,
                    "confidence": float(item.get("confidence", 0.25)),
                    "decision_weight": 0.0,
                    "factor_weights": {},
                }
            )
        return sorted(result, key=lambda item: str(item["item_id"]))

    def generate_card(
        self,
        person_id: str,
        *,
        source: str = "confirmed_evidence_rebuild",
    ) -> dict[str, object]:
        config = self._config(person_id)
        evidence = self.evidence(person_id)
        confirmed = [item for item in evidence if item["review_status"] == "confirmed"]
        direct = [
            item
            for item in confirmed
            if item["role"] in {"direct_observation", "person_self_report"}
        ]
        if len(confirmed) < MIN_CONFIRMED_EVIDENCE:
            raise CognitiveWorkbenchError(
                f"证据不足：至少需要 {MIN_CONFIRMED_EVIDENCE} 条经用户确认的材料。"
            )
        if len({str(item["date"]) for item in confirmed}) < MIN_DISTINCT_EVIDENCE_DATES:
            raise CognitiveWorkbenchError("证据不足：需要至少三个不同日期的材料。")
        if len(direct) < 3:
            raise CognitiveWorkbenchError("证据不足：至少需要三条直接观察或人物自述。")
        if any(str(item["domain"]) != str(config["domain"]) for item in confirmed):
            raise CognitiveWorkbenchError("已确认材料包含范围外领域，不能生成当前模型卡。")
        items = self._build_card_items(confirmed, config)
        cards = self._verified_cards(person_id)
        previous = cards[-1] if cards else None
        evidence_snapshot = [
            {
                "evidence_id": item["evidence_id"],
                "content_hash": item["content_hash"],
                "date": item["date"],
                "source": item["source"],
                "source_title": item["source_title"],
                "source_locator": item["source_locator"],
                "text": item["text"],
                "explicit_rationale": item["explicit_rationale"],
                "position": item["position"],
                "role": item["role"],
                "confidence": item["confidence"],
            }
            for item in sorted(confirmed, key=lambda value: str(value["evidence_id"]))
        ]
        card: dict[str, object] = {
            "schema_version": CARD_SCHEMA_VERSION,
            "version": len(cards) + 1,
            "parent_digest": previous["artifact_digest"] if previous else None,
            "created_at": _utc_now(),
            "source": source,
            "scope": {
                "domain": config["domain"],
                "decision_type": config["decision_type"],
                "option_a": config["option_a"],
                "option_b": config["option_b"],
                "evidence_window": config["evidence_window"],
            },
            "factor_definitions": config["factor_definitions"],
            "evidence_snapshot": evidence_snapshot,
            "evidence_snapshot_digest": _digest(evidence_snapshot),
            "latest_evidence_date": max(str(item["date"]) for item in confirmed),
            "all_items": items,
            "validation_status": "exploratory_unvalidated",
            "claim_status": "product_loop_complete_only",
            "non_claims": [
                "private thoughts recovered",
                "psychological mechanism identified",
                "calibrated probability",
                "person-specific superiority established",
            ],
        }
        for category in MODEL_CATEGORIES:
            card[category] = [item for item in items if item["category"] == category]
        previous_ids = {str(item["item_id"]) for item in previous["all_items"]} if previous else set()
        current_ids = {str(item["item_id"]) for item in items}
        card["version_change"] = {
            "modified": sorted(current_ids - previous_ids)
            if previous
            else ["initial card from confirmed evidence"],
            "unchanged": sorted(current_ids & previous_ids),
            "removed": sorted(previous_ids - current_ids),
        }
        card["artifact_digest"] = _artifact_digest(card)
        cards.append(card)
        _write_json(self._path(person_id, "card_versions.json"), cards)
        return dict(card)

    def draft_scenario(self, person_id: str, text: str) -> dict[str, object]:
        clean = str(text).strip()
        if not clean:
            raise CognitiveWorkbenchError("请先输入需要推演的新情境。")
        config = self._config(person_id)
        lowered = clean.lower()
        values: dict[str, float] = {}
        uncertainties = []
        for raw in config["factor_definitions"]:
            definition = dict(raw)
            positive = any(str(keyword).lower() in lowered for keyword in definition.get("positive_keywords", []))
            negative = any(str(keyword).lower() in lowered for keyword in definition.get("negative_keywords", []))
            if positive and not negative:
                value = 1.0
            elif negative and not positive:
                value = 0.0
            else:
                value = float(definition.get("default", 0.5))
                if definition.get("high_impact"):
                    uncertainties.append(
                        {
                            "factor": definition["name"],
                            "label": definition["label"],
                            "question": f"请确认：{definition['description']}",
                            "current_assumption": value,
                        }
                    )
            values[str(definition["name"])] = value
        scenario = {
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "scenario_id": f"cog-scenario-{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now(),
            "original_text": clean,
            "domain": config["domain"],
            "decision_type": config["decision_type"],
            "option_a": config["option_a"],
            "option_b": config["option_b"],
            "factor_values": values,
            "high_impact_uncertainties": uncertainties,
            "review_status": "pending",
            "confirmed_at": None,
            "prediction_at": None,
            "structuring_method": "deterministic_domain_parser_v1",
            "structuring_notice": "这是待确认草案；用户确认前不会进入预测。",
        }
        scenarios = self._list(person_id, "scenarios.json")
        scenarios.append(scenario)
        _write_json(self._path(person_id, "scenarios.json"), scenarios)
        return dict(scenario)

    def get_scenario(self, person_id: str, scenario_id: str) -> dict[str, object]:
        scenarios = self._list(person_id, "scenarios.json")
        try:
            return dict(next(item for item in scenarios if item["scenario_id"] == scenario_id))
        except StopIteration as error:
            raise CognitiveWorkbenchError("找不到这个结构化情境。") from error

    def confirm_scenario(
        self, person_id: str, scenario_id: str, changes: Mapping[str, object]
    ) -> dict[str, object]:
        config = self._config(person_id)
        scenarios = self._list(person_id, "scenarios.json")
        try:
            scenario = next(item for item in scenarios if item["scenario_id"] == scenario_id)
        except StopIteration as error:
            raise CognitiveWorkbenchError("找不到这个结构化情境。") from error
        expected = {str(item["name"]) for item in config["factor_definitions"]}
        raw_values = changes.get("factor_values", scenario["factor_values"])
        if not isinstance(raw_values, Mapping) or set(map(str, raw_values)) != expected:
            raise CognitiveWorkbenchError("请确认全部结构化因素，不能缺项或增加未定义因素。")
        values = {str(key): float(value) for key, value in raw_values.items()}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise CognitiveWorkbenchError("结构化因素必须是 0 到 1 的有限数值。")
        prediction_at = str(changes.get("prediction_at", "")).strip()
        _parse_time(prediction_at)
        scenario.update(
            {
                "domain": str(changes.get("domain", scenario["domain"])),
                "decision_type": str(changes.get("decision_type", scenario["decision_type"])),
                "factor_values": values,
                "prediction_at": prediction_at,
                "review_status": "confirmed",
                "confirmed_at": _utc_now(),
                "high_impact_uncertainties": list(changes.get("high_impact_uncertainties", [])),
            }
        )
        _write_json(self._path(person_id, "scenarios.json"), scenarios)
        return dict(scenario)

    def score_for_test(
        self, card: Mapping[str, object], scenario: Mapping[str, object]
    ) -> dict[str, object]:
        return score_cognitive_scenario(card, scenario)

    def predict(
        self,
        person_id: str,
        scenario_id: str,
        *,
        behavior_baseline: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        card = self.latest_card(person_id)
        if card is None:
            raise CognitiveWorkbenchError("还没有认知模型卡，请先审核证据并生成模型卡。")
        scenario = self.get_scenario(person_id, scenario_id)
        if scenario.get("review_status") != "confirmed":
            raise CognitiveWorkbenchError("结构化情境尚未由用户确认，不能进入预测。")
        result = score_cognitive_scenario(card, scenario)
        record = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "prediction_id": f"cog-prediction-{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now(),
            "person_id": person_id,
            "card_version": card["version"],
            "card_digest": card["artifact_digest"],
            "scenario": scenario,
            "claim_status": "product_loop_complete_only",
            "behavior_baseline": dict(behavior_baseline or {"status": "insufficient_evidence"}),
            "model_agreement": "unknown",
            "actual_choice": None,
            "actual_rationale": None,
            "updated_card_version": None,
            **result,
        }
        if result["status"] == "predicted":
            option = scenario["option_b"] if result["predicted_choice"] == 1 else scenario["option_a"]
            top = result["drivers"][:3]
            record["most_likely_option"] = option
            record["explanation"] = (
                "该说明只复述计算结果：当前结构化因素主要激活了 "
                + "、".join(str(item["statement"]) for item in top)
                + "。语言层没有修改概率或最可能选项。"
            )
            baseline_choice = record["behavior_baseline"].get("predicted_choice")
            if record["behavior_baseline"].get("status") == "predicted":
                record["model_agreement"] = (
                    "agree" if int(baseline_choice) == int(result["predicted_choice"]) else "disagree"
                )
        predictions = self._list(person_id, "cognitive_predictions.json")
        predictions.append(record)
        _write_json(self._path(person_id, "cognitive_predictions.json"), predictions)
        return dict(record)

    def _append_confirmed_outcome_evidence(
        self,
        person_id: str,
        prediction: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        scenario = dict(prediction["scenario"])
        actual_choice = int(payload["actual_choice"])
        option = scenario["option_b"] if actual_choice == 1 else scenario["option_a"]
        factor_weights = {
            str(name): float(value)
            for name, value in dict(scenario["factor_values"]).items()
            if float(value) >= 0.6
        }
        raw = {
            "evidence_id": f"outcome-{prediction['prediction_id']}",
            "text": f"Observed outcome: {option}. {str(payload['actual_rationale']).strip()}",
            "source": str(payload["source"]),
            "source_title": "Post-prediction observed result and stated rationale",
            "date": _parse_time(str(payload["observed_at"])).date().isoformat(),
            "context": str(scenario["original_text"]),
            "position": str(option),
            "explicit_rationale": str(payload["actual_rationale"]),
            "role": "person_self_report",
            "domain": scenario["domain"],
            "is_conflicting": int(prediction.get("predicted_choice", -1)) != actual_choice,
            "confidence": 0.9,
            "source_locator": str(payload["source_locator"]),
            "extraction_method": "user_confirmed_external_outcome",
            "candidate_model_items": [
                {
                    "item_id": f"dynamic-{prediction['prediction_id']}",
                    "category": "dynamic_state",
                    "statement": (
                        f"At {payload['observed_at']}, the person chose {option} and gave this reason: "
                        f"{str(payload['actual_rationale']).strip()}"
                    ),
                    "status": "observed",
                    "decision_weight": 1.0 if actual_choice == 1 else -1.0,
                    "factor_weights": factor_weights,
                }
            ],
        }
        item = self._normalize_evidence(raw)
        item["review_status"] = "confirmed"
        item["reviewed_at"] = _utc_now()
        evidence = self.evidence(person_id)
        if any(existing["content_hash"] == item["content_hash"] for existing in evidence):
            raise CognitiveWorkbenchError("这个真实结果已经回填过。")
        evidence.append(item)
        _write_json(self._path(person_id, "evidence.json"), evidence)
        return item

    def record_outcome(
        self, person_id: str, prediction_id: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        if payload.get("confirm_real_external_result") is not True:
            raise CognitiveWorkbenchError("必须确认这是真实外部结果，不是模型生成文本。")
        rationale = str(payload.get("actual_rationale", "")).strip()
        source = str(payload.get("source", "")).strip()
        locator = str(payload.get("source_locator", "")).strip()
        if not rationale or not source or not locator:
            raise CognitiveWorkbenchError("真实理由、来源和原始位置都不能为空。")
        predictions = self._list(person_id, "cognitive_predictions.json")
        try:
            prediction = next(item for item in predictions if item["prediction_id"] == prediction_id)
        except StopIteration as error:
            raise CognitiveWorkbenchError("找不到这条认知预测。") from error
        if prediction.get("status") != "predicted":
            raise CognitiveWorkbenchError("被拒绝的推演不能回填为有效预测结果。")
        if prediction.get("actual_choice") is not None:
            raise CognitiveWorkbenchError("这条预测已经回填过真实结果。")
        try:
            actual_choice = int(payload.get("actual_choice"))
        except (TypeError, ValueError) as error:
            raise CognitiveWorkbenchError("真实选择必须是 0 或 1。") from error
        if actual_choice not in {0, 1}:
            raise CognitiveWorkbenchError("真实选择必须是 0 或 1。")
        observed_at = _parse_time(str(payload.get("observed_at", "")))
        prediction_at = _parse_time(str(prediction["scenario"]["prediction_at"]))
        if observed_at < prediction_at:
            raise CognitiveWorkbenchError("真实结果时间不能早于预测时间。")
        if rationale == str(prediction.get("explanation", "")).strip():
            raise CognitiveWorkbenchError("不能把模型自己的解释当作人物真实理由。")

        previous = self.latest_card(person_id)
        assert previous is not None
        evidence = self._append_confirmed_outcome_evidence(person_id, prediction, payload)
        updated_card = self.generate_card(person_id, source="confirmed_real_outcome")
        choice_match = int(prediction["predicted_choice"]) == actual_choice
        potentially_wrong = (
            []
            if choice_match
            else [str(item["item_id"]) for item in prediction["drivers"][:3]]
        )
        prediction.update(
            {
                "actual_choice": actual_choice,
                "actual_rationale": rationale,
                "outcome_observed_at": str(payload["observed_at"]),
                "outcome_source": source,
                "outcome_source_locator": locator,
                "choice_match": choice_match,
                "potentially_wrong_items": potentially_wrong,
                "updated_card_version": updated_card["version"],
            }
        )
        _write_json(self._path(person_id, "cognitive_predictions.json"), predictions)
        outcome = {
            "outcome_id": f"cog-outcome-{uuid.uuid4().hex[:12]}",
            "prediction_id": prediction_id,
            "actual_choice": actual_choice,
            "actual_rationale": rationale,
            "choice_match": choice_match,
            "potentially_wrong_items": potentially_wrong,
            "evidence_id": evidence["evidence_id"],
            "updated_card_version": updated_card["version"],
            "version_change": updated_card["version_change"],
            "created_at": _utc_now(),
        }
        outcomes = self._list(person_id, "cognitive_outcomes.json")
        outcomes.append(outcome)
        _write_json(self._path(person_id, "cognitive_outcomes.json"), outcomes)
        return outcome

    def summary(self, person_id: str) -> dict[str, object]:
        if not self.has_config(person_id):
            return {
                "status": "not_configured",
                "evidence": [],
                "latest_card": None,
                "scenarios": [],
                "predictions": [],
                "outcome_count": 0,
            }
        config = self._config(person_id)
        evidence = self.evidence(person_id)
        cards = self._verified_cards(person_id)
        scenarios = self._list(person_id, "scenarios.json")
        predictions = self._list(person_id, "cognitive_predictions.json")
        outcomes = self._list(person_id, "cognitive_outcomes.json")
        confirmed = sum(item["review_status"] == "confirmed" for item in evidence)
        if confirmed < MIN_CONFIRMED_EVIDENCE:
            status = "evidence_review_required"
        elif not cards:
            status = "ready_for_card"
        else:
            status = "ready_for_scenario"
        return {
            "status": status,
            "scope": {
                "domain": config["domain"],
                "decision_type": config["decision_type"],
                "option_a": config["option_a"],
                "option_b": config["option_b"],
            },
            "factor_definitions": config["factor_definitions"],
            "suggested_scenario_text": config.get("suggested_scenario_text"),
            "suggested_prediction_at": config.get("suggested_prediction_at"),
            "holdout_notice": config.get("holdout_notice"),
            "evidence": evidence,
            "evidence_counts": {
                "total": len(evidence),
                "confirmed": confirmed,
                "pending": sum(item["review_status"] == "pending" for item in evidence),
                "rejected": sum(item["review_status"] == "rejected" for item in evidence),
            },
            "latest_card": dict(cards[-1]) if cards else None,
            "card_versions": [
                {
                    "version": card["version"],
                    "created_at": card["created_at"],
                    "source": card["source"],
                    "validation_status": card["validation_status"],
                    "artifact_digest": card["artifact_digest"],
                    "version_change": card["version_change"],
                }
                for card in cards
            ],
            "card_version_count": len(cards),
            "scenarios": scenarios[-10:],
            "predictions": predictions[-20:],
            "outcome_count": len(outcomes),
            "completion_label": "product_loop_complete_only" if cards else "not_complete",
            "person_specific_evidence_status": "not_assessed",
        }


def load_builtin_hawley_case() -> dict[str, object]:
    path = Path(__file__).with_name("cases") / "hawley_section230_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))
