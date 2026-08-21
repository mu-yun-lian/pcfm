from __future__ import annotations

import re
from typing import Mapping, Sequence

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _canonical_hash,
    _decode_web_bytes,
    _derivation_view,
    _extract_html,
    _extract_pdf,
    _extract_qa,
    _is_short_reference,
    _json_mapping,
    _localize_view,
    _read_json,
    _segments,
    _similarity,
    _structured_rows_text,
    _text_hash,
    _tokens,
    _utc_now,
    _write_json,
)

KNOWLEDGE_TYPE_ZH = {
    "fact": "事实判断",
    "procedural": "程序",
    "professional": "专业",
    "autobiographical": "自传",
    "social": "社会",
    "causal": "因果",
    "conceptual_framework": "概念框架",
}


def _abstract_domain_profiles(
    profiles: Mapping[str, object], domains: Sequence[str], is_chinese: bool
) -> list[dict[str, object]]:
    """结构化画像 → 抽象视图（§5.2 输入）。

    剥掉逐字 evidence_span 与具体 target_span（DomainProfile 已剥 target_span），
    只给抽象字段 + item_id，供 LLM 忠实转述；防止照抄原文、防专名脑补。
    """
    view: list[dict[str, object]] = []
    for domain_id in sorted(set(map(str, domains))):
        profile = profiles.get(domain_id)
        if not isinstance(profile, Mapping):
            continue
        entry: dict[str, object] = {"domain_id": domain_id}
        if is_chinese:
            entry["domain_label"] = DOMAIN_ZH.get(domain_id, domain_id)

        def _loc(value: str, table: Mapping[str, str]) -> str:
            return str(table.get(value, value)) if is_chinese else str(value)

        cognition = []
        for item in profile.get("cognition", []):
            if not isinstance(item, Mapping):
                continue
            cognition.append(
                {
                    "item_id": str(item.get("item_id", "")),
                    "knowledge_type": _loc(str(item.get("knowledge_type", "fact")), KNOWLEDGE_TYPE_ZH),
                    "statement": str(item.get("statement", "")),
                }
            )
        entry["cognition"] = cognition

        values = []
        for item in profile.get("values", []):
            if not isinstance(item, Mapping):
                continue
            protected = str(item.get("preferred_side", ""))
            cost = str(item.get("sacrificed_side", ""))
            if is_chinese:
                protected = str(INTERESTS.get(protected, {}).get("label_zh", protected))
                cost = str(INTERESTS.get(cost, {}).get("label_zh", cost)) if cost else ""
            else:
                protected = protected.replace("_", " ")
                cost = cost.replace("_", " ") if cost else ""
            values.append(
                {
                    "item_id": str(item.get("item_id", "")),
                    "preferred_side": protected,
                    "sacrificed_side": cost,
                    "tendency_types": [str(v) for v in item.get("tendency_types", [])],
                    "status": str(item.get("status", "")),
                }
            )
        entry["values"] = values

        ideas = []
        for item in profile.get("ideas", []):
            if not isinstance(item, Mapping):
                continue
            idea = {
                "item_id": str(item.get("item_id", "")),
                "tendency_type": _loc(str(item.get("tendency_type", "")), TENDENCY_TYPE_ZH),
                "direction": _loc(str(item.get("direction", "")), STANCE_ZH),
                "target": _loc(str(item.get("target", "")), OBJECT_CATEGORY_ZH),
            }
            if item.get("public_statement"):
                idea["public_statement"] = str(item.get("public_statement", ""))
            ideas.append(idea)
        entry["ideas"] = ideas

        strategies = []
        for item in profile.get("strategies", []):
            if not isinstance(item, Mapping):
                continue
            strategies.append(
                {
                    "item_id": str(item.get("item_id", "")),
                    "statement": str(item.get("statement", "")),
                }
            )
        entry["strategies"] = strategies

        conditions = []
        for item in profile.get("conditions", []):
            if not isinstance(item, Mapping):
                continue
            conditions.append(
                {
                    "item_id": str(item.get("item_id", "")),
                    "condition": str(item.get("condition", "")),
                }
            )
        entry["conditions"] = conditions
        view.append(entry)
    return view


def _view_item_ids(view: Sequence[Mapping[str, object]]) -> set[str]:
    ids: set[str] = set()
    for entry in view:
        for field in ("cognition", "values", "ideas", "strategies", "conditions"):
            for item in entry.get(field, []):
                if isinstance(item, Mapping) and item.get("item_id"):
                    ids.add(str(item["item_id"]))
    return ids


def _closed_label_vocabulary() -> set[str]:
    """全部封闭词表标签（英文 ID + 中文 label_zh），供转述门禁扫描。

    不含利益「别名」（如「公开」既是透明度别名又是普通词「公开地说」），
    避免把普通词误判成价值概念；宁可用 label_zh 这种无歧义的价值短语。
    """
    labels: set[str] = set()
    for interest_id, definition in INTERESTS.items():
        labels.add(str(interest_id))
        labels.add(str(definition.get("label_zh", "")))
    labels.update(OBJECT_CATEGORIES)
    labels.update(OBJECT_CATEGORY_ZH.values())
    labels.update(STANCES)
    labels.update(STANCE_ZH.values())
    # 倾向类型（如「风险容忍」「手段-目的」）是分析维度词，不是人物声称的立场/价值；
    # 中文转述用它们做合理概括（如「愿牺牲成本控制」→「风险容忍」）不构成观点编造，
    # 故不作为「画像外概念」判定源，避免过度严格的标签门禁误拒合法转述。
    # 立场(STANCES)/对象类别(OBJECT_CATEGORY)/价值(INTERESTS)仍受封闭门禁保护。
    labels.update(DOMAIN_ALIASES.keys())
    labels.update(DOMAIN_ZH.values())
    labels.update(KNOWLEDGE_TYPE_ZH.values())
    labels.discard("")
    return labels


def _label_present(label: str, text: str) -> bool:
    if re.search(r"[\u4e00-\u9fff]", label):
        return label in text
    return bool(re.search(r"\b" + re.escape(label) + r"\b", text, re.I))


def _collect_closed_labels_from_text(text: str) -> set[str]:
    source = str(text)
    return {label for label in _closed_label_vocabulary() if _label_present(label, source)}


class ProfileNarrationMixin:
    def _narrate_domain_profile(
        self,
        *,
        person_id: str,
        artifact: Mapping[str, object],
        domains: Sequence[str],
        is_chinese: bool,
        model_ref: str,
    ) -> dict[str, object]:
        """把结构化 DomainProfile 忠实转述成自然语言画像（§5.2）。

        LLM 只做表达组织，不新增立场；输出必须标 referenced_item_ids 且 ⊆ 输入画像。
        无模型 / 调用失败 / 门禁失败均返回非 ok，由调用方回退硬拼接。
        """
        profiles = dict(artifact.get("domain_profiles") or {})
        view = _abstract_domain_profiles(profiles, domains, is_chinese)
        if not view:
            return {"status": "no_matching_profile", "narration": "", "model_calls": 0, "referenced_item_ids": []}
        valid_ids = _view_item_ids(view)
        if not valid_ids:
            return {"status": "empty_profile", "narration": "", "model_calls": 0, "referenced_item_ids": []}
        if not model_ref or self._model_services is None:
            return {"status": "no_model", "narration": "", "model_calls": 0, "referenced_item_ids": []}
        try:
            service, model_id = self._model_services.resolve_model_ref(model_ref)
        except ModelServiceError:
            return {"status": "model_unavailable", "narration": "", "model_calls": 0, "referenced_item_ids": []}
        language = "Chinese" if is_chinese else "English"
        system = (
            "You are narrating a structured domain profile of a real person's "
            "PUBLIC, observable tendencies (not their private values or beliefs). "
            "domain_profiles is the ONLY source of truth. Rewrite it into ONE "
            "coherent natural-language paragraph organized as 认知/价值/想法/策略/条件. "
            "HARD RULES: (1) use ONLY the given items — do not add any cognition, "
            "value ranking, opinion, or strategy that is not in the profile; "
            "(2) do NOT assert what the person 'believes' or 'is' beyond these "
            "public statements — cognition items are the person's public claims, "
            "not verified facts; (3) reference every item you used by its item_id "
            "in referenced_item_ids; (4) write in " + language + " — if Chinese, "
            "the WHOLE narration must be Chinese; (5) do not introduce numbers, "
            "dates, or proper names that are absent from the input; (6) keep it "
            "under 800 characters. Return JSON with exactly narration and "
            "referenced_item_ids."
        )
        payload = {
            "domain_profiles": view,
            "language": language,
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        temperature = self._generation_temperature(person_id)
        try:
            response = self._model_services.invoke(
                str(service["service_id"]),
                model_id,
                messages,
                structured=True,
                temperature=temperature,
            )
        except ModelServiceError:
            return {"status": "invoke_failed", "narration": "", "model_calls": 1, "referenced_item_ids": []}
        try:
            candidate = _json_mapping(response["text"])
        except json.JSONDecodeError:
            candidate = None
        if not isinstance(candidate, Mapping):
            return {"status": "unparseable", "narration": "", "model_calls": 1, "referenced_item_ids": []}
        narration = str(candidate.get("narration", "")).strip()
        referenced = [str(value) for value in candidate.get("referenced_item_ids", [])]
        ok, reason = self._gate_narration(narration, referenced, view, valid_ids, is_chinese)
        if not ok:
            return {
                "status": "gate_failed",
                "reason": reason,
                "narration": "",
                "model_calls": 1,
                "referenced_item_ids": referenced,
            }
        return {
            "status": "ok",
            "narration": narration,
            "model_calls": 1,
            "model_ref": model_ref,
            "snapshot_id": dict(response["snapshot"])["snapshot_id"],
            "referenced_item_ids": referenced,
        }

    @staticmethod
    def _gate_narration(
        narration: str,
        referenced: Sequence[str],
        view: Sequence[Mapping[str, object]],
        valid_ids: set[str],
        is_chinese: bool,
    ) -> tuple[bool, str]:
        """确定性门禁：转述只做表达组织，不得引入画像之外的内容。"""
        if not narration or len(narration) > 1200:
            return False, "narration_empty_or_too_long"
        # 引用完整性：转述标出的 item_id 必须 ⊆ 输入画像。
        if not set(referenced) <= valid_ids:
            return False, "referenced_item_id_not_in_profile"
        # 语言跟随：中文问题必须中文转述。
        if is_chinese and not re.search(r"[\u4e00-\u9fff]", narration):
            return False, "language_mismatch"
        # 封闭词表封闭性：转述里出现的每个封闭词表标签必须已在输入画像里出现，
        # 防止 LLM 引入画像之外的价值判断/立场/对象类别。
        allowed_labels = _collect_closed_labels_from_text(
            json.dumps(view, ensure_ascii=False)
        )
        used_labels = _collect_closed_labels_from_text(narration)
        if not used_labels <= allowed_labels:
            return False, "narration_introduced_out_of_profile_concept"
        # 不新增数字：允许数字只从画像的文本字段收集（statement/condition/public_statement），
        # 不含 item_id（如 cog-1 会贡献 "1"，形同虚设）。
        allowed_numbers: set[str] = set()
        for entry in view:
            for field in ("cognition", "values", "ideas", "strategies", "conditions"):
                for item in entry.get(field, []):
                    if not isinstance(item, Mapping):
                        continue
                    for text_field in ("statement", "public_statement", "condition"):
                        allowed_numbers.update(
                            re.findall(r"\d+(?:[.,]\d+)*", str(item.get(text_field, "")))
                        )
        for number in re.findall(r"\d+(?:[.,]\d+)*", narration):
            if number not in allowed_numbers:
                return False, "narration_introduced_new_number"
        return True, ""
