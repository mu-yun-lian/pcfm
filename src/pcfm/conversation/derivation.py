"""推导视图工具：把领域倾向画像转成 LLM 可读的抽象视图（剥逐字原话、中文本地化）。"""
from __future__ import annotations

import json
from typing import Mapping

from ..simulation_v4 import INTERESTS


OBJECT_CATEGORY_ZH = {
    "organization": "组织",
    "individual": "个人",
    "product": "产品",
    "institution": "机构",
    "market": "市场",
    "technology": "技术",
    "group": "群体",
    "behavior": "行为",
    "abstract_concept": "抽象概念",
}
STANCE_ZH = {
    "support": "支持",
    "oppose": "反对",
    "neutral": "中立",
    "conditional_support": "有条件支持",
    "mixed": "混合",
    "insufficient_evidence": "证据不足",
}
TENDENCY_TYPE_ZH = {
    "object_evaluation": "对象评价",
    "principle_priority": "原则优先",
    "conditional_policy_preference": "条件性政策偏好",
    "means_ends": "手段目的",
    "responsibility_attribution": "责任归属",
    "risk_tolerance": "风险容忍",
    "rule_procedure_tradeoff": "规则程序权衡",
    "behavior_evaluation": "行为评价",
}
EVENT_STRUCTURE_ZH = {
    "conflict_management": "冲突处理",
    "resource_allocation": "资源分配",
    "risk_decision": "风险决策",
    "personnel_choice": "人员选择",
    "moral_evaluation": "道德评价",
    "policy_stance": "政策立场",
    "means_ends": "手段目的",
    "responsibility_attribution": "责任归属",
}
DOMAIN_ZH = {
    "health": "健康",
    "technology": "技术",
    "product": "产品",
    "governance": "治理",
    "economics": "经济",
    "social": "社会",
    "space": "航天",
    "aviation": "航空",
    "environment": "环境",
    "education": "教育",
    "personal": "个人",
}


def _json_mapping(value: object) -> dict[str, object]:
    clean = str(value).strip()
    if clean.startswith("\`\`\`") and clean.endswith("\`\`\`"):
        lines = clean.splitlines()
        if len(lines) >= 3 and lines[0].strip().casefold() in {"\`\`\`", "\`\`\`json"}:
            clean = "\n".join(lines[1:-1]).strip()
    parsed = json.loads(clean)
    if not isinstance(parsed, Mapping):
        raise json.JSONDecodeError("expected a JSON object", clean, 0)
    return dict(parsed)


def _derivation_view(index: object) -> dict[str, object]:
    result: dict[str, object] = {}
    for domain, cells in (index or {}).items():
        if not isinstance(cells, Mapping):
            continue
        domain_view: dict[str, object] = {}
        for structure, cell in cells.items():
            if not isinstance(cell, Mapping):
                continue
            atoms = []
            for atom in cell.get("atoms", []):
                if not isinstance(atom, Mapping):
                    continue
                atoms.append({key: value for key, value in atom.items() if key != "reason"})
            domain_view[str(structure)] = {
                "dominant_value": cell.get("dominant_value", ""),
                "opposes": cell.get("opposes", []),
                "supports": cell.get("supports", []),
                "atoms": atoms,
            }
        result[str(domain)] = domain_view
    return result


def _localize_view(view: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for domain, cells in view.items():
        domain_view: dict[str, object] = {}
        for structure, cell in cells.items():
            if not isinstance(cell, Mapping):
                continue
            atoms = []
            for atom in cell.get("atoms", []):
                if not isinstance(atom, Mapping):
                    continue
                item = dict(atom)
                direction = str(item.get("direction", ""))
                item["direction"] = STANCE_ZH.get(direction, direction)
                target = str(item.get("target", ""))
                item["target"] = OBJECT_CATEGORY_ZH.get(target, target)
                for field in ("protected_interest_id", "accepted_cost_id"):
                    value = str(item.get(field, ""))
                    if value in INTERESTS:
                        item[field] = str(INTERESTS[value].get("label_zh", value))
                tendency = str(item.get("tendency_type", ""))
                item["tendency_type"] = TENDENCY_TYPE_ZH.get(tendency, tendency)
                atoms.append(item)
            dominant = str(cell.get("dominant_value", ""))
            domain_view[EVENT_STRUCTURE_ZH.get(str(structure), str(structure))] = {
                "dominant_value": (
                    str(INTERESTS[dominant].get("label_zh", dominant))
                    if dominant in INTERESTS
                    else dominant
                ),
                "opposes": [
                    OBJECT_CATEGORY_ZH.get(str(value), str(value))
                    for value in cell.get("opposes", [])
                ],
                "supports": [
                    OBJECT_CATEGORY_ZH.get(str(value), str(value))
                    for value in cell.get("supports", [])
                ],
                "atoms": atoms,
            }
        result[DOMAIN_ZH.get(str(domain), str(domain))] = domain_view
    return result
