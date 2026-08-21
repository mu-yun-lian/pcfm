"""领域整体画像 DomainProfile：同领域伴随原子按维度聚合（可回溯）。

对齐 docs/事件原子提取与领域价值画像设计方案.md §3.3/§5.1：
同领域所有伴随原子综合成该领域的一套多维度整体画像，每一条
`item -> atom_ids -> event_ids -> evidence_span` 全程可回溯。
本模块是纯函数聚合，不调用 LLM，不新增立场，只归类已存在的原子。
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

DOMAIN_PROFILE_SCHEMA = "pcfm-domain-profile-v1"

# 倾向类型 → 画像维度映射（对齐设计方案 §3.2）。
# 8 类公开反应倾向只覆盖「价值概念 / 想法 / 策略」三个维度；
# 「认知」来自知识原子，「条件与例外」来自各原子的 conditions 并集。
VALUE_TENDENCY_TYPES = frozenset(
    {"principle_priority", "risk_tolerance", "rule_procedure_tradeoff"}
)
IDEA_TENDENCY_TYPES = frozenset(
    {
        "object_evaluation",
        "conditional_policy_preference",
        "behavior_evaluation",
        "responsibility_attribution",
    }
)
STRATEGY_TENDENCY_TYPES = frozenset({"means_ends"})


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _domain_match(item: Mapping[str, object], domain_id: str) -> bool:
    return domain_id in {str(value) for value in item.get("domain_tags", [])}


def _atom_id(item: Mapping[str, object]) -> str:
    return str(
        item.get("preference_atom_id")
        or item.get("statement_atom_id")
        or item.get("knowledge_claim_id")
        or ""
    )


def _trace(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "atom_ids": sorted({_atom_id(item) for item in items if _atom_id(item)}),
        "event_ids": sorted(
            {str(item.get("event_frame_id", "")) for item in items if str(item.get("event_frame_id", ""))}
        ),
        "evidence_spans": sorted(
            {
                str(item.get("evidence_span", "") or item.get("statement", ""))
                for item in items
                if str(item.get("evidence_span", "") or item.get("statement", ""))
            }
        ),
    }


def build_domain_profile(
    domain_id: str,
    preference_atoms: Sequence[Mapping[str, object]],
    statement_atoms: Sequence[Mapping[str, object]],
    knowledge_claims: Sequence[Mapping[str, object]],
    inferred_value_atoms: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """聚合同领域伴随原子为一个 DomainProfile 结构化画像。

    每个 item 都挂 atom_ids / event_ids / evidence_spans，可回溯到事件原子。
    """
    pref = [dict(a) for a in preference_atoms if _domain_match(a, domain_id)]
    stmts = [dict(a) for a in statement_atoms if _domain_match(a, domain_id)]
    know = [dict(k) for k in knowledge_claims if _domain_match(k, domain_id)]
    inferred = [dict(a) for a in inferred_value_atoms if _domain_match(a, domain_id)]

    # ① 认知：知识原子（事实/概念/因果等公开主张，不宣称恢复内心知识）
    cognition = []
    for item in know:
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        cognition.append(
            {
                "item_id": f"cog-{_hash([domain_id, statement])[:16]}",
                "statement": statement,
                "knowledge_type": str(item.get("knowledge_type", "fact")),
                "knowledge_boundary": str(
                    item.get(
                        "status",
                        "exact_publicly_demonstrated_claim_not_complete_person_knowledge",
                    )
                ),
                **_trace([item]),
            }
        )

    # ② 价值概念：取舍类倾向原子，按「优先侧/牺牲侧」聚合。
    # 明写取舍（explicit）与 LLM 推断取舍（inferred）分标 origin；推断不参与确定预测。
    values: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = {}
    for item in pref:
        if item.get("tendency_type") not in VALUE_TENDENCY_TYPES:
            continue
        protected = str(item.get("protected_interest_id", "")).strip()
        cost = str(item.get("accepted_cost_id", "")).strip()
        if not protected:
            continue
        values.setdefault((protected, cost), []).append(("explicit", item))
    for item in inferred:
        protected = str(item.get("protected_interest_id", "")).strip()
        cost = str(item.get("accepted_cost_id", "")).strip()
        if not protected:
            continue
        values.setdefault((protected, cost), []).append(("inferred", item))
    value_items = []
    for (protected, cost), entries in sorted(values.items()):
        origins = sorted({origin for origin, _ in entries})
        items = [item for _, item in entries]
        value_items.append(
            {
                "item_id": f"val-{_hash([domain_id, protected, cost])[:16]}",
                "preferred_side": protected,
                "sacrificed_side": cost,
                "direction": "support",
                "tendency_types": sorted({str(it.get("tendency_type", "")) for it in items if it.get("tendency_type")}),
                "status": str(items[0].get("status", "")),
                "origin": (
                    "explicit"
                    if origins == ["explicit"]
                    else "inferred"
                    if origins == ["inferred"]
                    else "mixed"
                ),
                **_trace(items),
            }
        )

    # ③ 想法：评价/偏好/责任归属类倾向原子（带方向+对象类别）
    ideas: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for item in pref:
        if item.get("tendency_type") not in IDEA_TENDENCY_TYPES:
            continue
        key = (
            str(item.get("tendency_type", "")),
            str(item.get("direction", "")),
            str(item.get("target", "")),
        )
        ideas.setdefault(key, []).append(item)
    idea_items = []
    for (tendency_type, direction, target), items in sorted(ideas.items()):
        idea_items.append(
            {
                "item_id": f"idea-{_hash([domain_id, tendency_type, direction, target])[:16]}",
                "tendency_type": tendency_type,
                "direction": direction,
                "target": target,
                **_trace(items),
            }
        )
    # 公开表态（无取舍/无结构化方向）也归入「想法」，明确标为方向未结构化
    for item in stmts:
        statement = str(item.get("statement", "")).strip()
        if not statement:
            continue
        idea_items.append(
            {
                "item_id": f"idea-{_hash([domain_id, 'statement', statement])[:16]}",
                "tendency_type": "",
                "direction": "",
                "target": "",
                "public_statement": statement,
                "statement_status": str(
                    item.get("status", "reviewed_public_statement_without_tradeoff")
                ),
                **_trace([item]),
            }
        )

    # ④ 策略：手段—目的类倾向原子
    strategy_items = []
    for item in pref:
        if item.get("tendency_type") not in STRATEGY_TENDENCY_TYPES:
            continue
        evidence = str(item.get("evidence_span", "")).strip()
        if not evidence:
            continue
        strategy_items.append(
            {
                "item_id": f"strategy-{_hash([domain_id, evidence])[:16]}",
                "statement": evidence,
                **_trace([item]),
            }
        )

    # ⑤ 条件与例外：所有原子 conditions 的并集
    conditions: dict[str, list[dict[str, object]]] = {}
    for item in [*pref, *stmts]:
        for condition in item.get("conditions", []):
            text = str(condition).strip()
            if text:
                conditions.setdefault(text, []).append(item)
    condition_items = []
    for text, items in sorted(conditions.items()):
        condition_items.append(
            {
                "item_id": f"cond-{_hash([domain_id, text])[:16]}",
                "condition": text,
                **_trace(items),
            }
        )

    return {
        "schema_version": DOMAIN_PROFILE_SCHEMA,
        "domain_id": domain_id,
        "cognition": cognition,
        "values": value_items,
        "ideas": idea_items,
        "strategies": strategy_items,
        "conditions": condition_items,
        "counts": {
            "cognition": len(cognition),
            "values": len(value_items),
            "ideas": len(idea_items),
            "strategies": len(strategy_items),
            "conditions": len(condition_items),
            "source_event_count": len(
                {eid for item in [*pref, *stmts, *know] for eid in [str(item.get("event_frame_id", ""))] if eid}
            ),
        },
    }


def build_domain_profiles(
    preference_atoms: Sequence[Mapping[str, object]],
    statement_atoms: Sequence[Mapping[str, object]],
    knowledge_claims: Sequence[Mapping[str, object]],
    inferred_value_atoms: Sequence[Mapping[str, object]] = (),
) -> dict[str, dict[str, object]]:
    """按领域聚合所有伴随原子，返回 {domain_id: DomainProfile}。"""
    domain_ids = sorted(
        {
            str(domain)
            for item in [
                *preference_atoms,
                *statement_atoms,
                *knowledge_claims,
                *inferred_value_atoms,
            ]
            for domain in item.get("domain_tags", [])
        }
    )
    return {
        domain_id: build_domain_profile(
            domain_id,
            preference_atoms,
            statement_atoms,
            knowledge_claims,
            inferred_value_atoms,
        )
        for domain_id in domain_ids
    }
