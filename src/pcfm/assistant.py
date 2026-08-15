# -*- coding: utf-8 -*-
"""AI 助手：操作员流程引擎（引导式，非被模拟人物）。

用户说意图 -> 助手列步骤 -> 逐步收集 -> 展示汇总 + 确认 -> 执行 ProductService 方法。
边界：不可逆操作二次确认；搜索结果只进候选(reference_only)；助手不改人物内核。
"""
from __future__ import annotations

import json
from pathlib import Path


class AssistantError(ValueError):
    pass


_INTENTS = [
    ("permanent_delete", ("永久删除", "彻底删除")),
    ("restore", ("恢复", "恢复归档")),
    ("archive", ("归档", "移入归档")),
    ("search", ("搜索", "上网搜", "联网搜", "搜一下", "搜人物")),
    ("add_source", ("加材料", "添加材料", "添加资料", "加资料", "加文本", "加文件", "加网页")),
    ("create_person", ("建人物", "新建人物", "创建人物", "建一个人物", "创建一个人")),
    ("list_people", ("列出人物", "有哪些人物", "人物列表", "看看人物")),
    ("list_archived", ("归档列表", "已归档")),
    ("help", ("帮助", "能做什么", "怎么用", "命令")),
]

_FLOWS = {
    "create_person": {
        "label": "建人物",
        "steps": [
            ("name", "① 人物姓名是什么？", True),
            ("aliases", "② 别名（逗号分隔，可空，直接回车跳过）", False),
            ("language", "③ 语言（默认中文，直接回车用默认）", False),
            ("identity_note", "④ 身份说明（一句话，推荐，可空）", False),
            ("focus_domain", "⑤ 专注领域（可空）", False),
        ],
        "execute": "_do_create_person",
    },
    "add_source": {
        "label": "添加材料",
        "steps": [
            ("person", "先选要给谁加材料：", True),
            ("content", "把文本材料粘贴进来（一段或整篇）", True),
        ],
        "execute": "_do_add_source",
    },
    "search": {
        "label": "联网搜索",
        "steps": [
            ("person", "搜谁的材料？选一个人物", True),
        ],
        "execute": "_do_search",
    },
    "archive": {
        "label": "归档人物",
        "steps": [
            ("person", "要归档哪个人物？", True),
        ],
        "execute": "_do_archive",
    },
    "restore": {
        "label": "恢复归档",
        "steps": [
            ("person", "要恢复哪个归档人物？", True),
        ],
        "execute": "_do_restore",
    },
    "permanent_delete": {
        "label": "永久删除",
        "steps": [
            ("person", "要永久删除哪个归档人物？（⚠️不可恢复）", True),
        ],
        "execute": "_do_permanent_delete",
    },
}

_CONFIRM = ("确认", "是", "好", "好的", "执行", "可以", "对", "yes", "ok", "y")
_CANCEL = ("取消", "重来", "不要", "算了", "no", "n", "换一个")


def _norm(text):
    return str(text).strip().casefold()


def _people_summary(people):
    return "\n".join("%d. %s" % (i, p.get("name", "")) for i, p in enumerate(people, 1))


class AssistantEngine:
    def __init__(self, service, state_path):
        self.service = service
        self.state_path = Path(state_path)

    def _load_state(self):
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, state):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def reset(self):
        state = {"flow": None, "step": 0, "fields": {}, "pending_action": None}
        self._save_state(state)
        return state

    def handle(self, text):
        state = self._load_state() or self.reset()
        clean = _norm(text)

        # 任意步骤都可以「取消」中止流程
        if clean in _CANCEL and (state.get("flow") or state.get("pending_action")):
            self.reset()
            return {
                "reply": "已取消。\n\n我可以：建人物 / 加材料 / 搜索 / 归档 / 恢复 / 永久删除 / 列出人物。",
                "state": self._load_state(),
            }

        if state.get("pending_action"):
            if clean in _CONFIRM:
                result = self._execute(state["pending_action"])
                self.reset()
                return {"reply": result, "state": self._load_state()}
            if clean in _CANCEL:
                self.reset()
                return {
                    "reply": "已取消。\n\n我可以：建人物 / 加材料 / 搜索 / 归档 / 恢复 / 永久删除 / 列出人物。",
                    "state": self._load_state(),
                }
            return {
                "reply": "请确认或取消（说「确认」执行 / 「取消」重来）",
                "state": state,
            }

        if state.get("flow"):
            return self._collect(state, clean)

        for flow_id, keywords in _INTENTS:
            if any(kw in clean for kw in keywords):
                if flow_id == "help":
                    return {"reply": self._help(), "state": state}
                if flow_id == "list_people":
                    people = self.service.list_people()
                    reply = (
                        "现有人物：\n" + _people_summary(people)
                        if people
                        else "还没有人物。说「建人物」开始。"
                    )
                    return {"reply": reply, "state": state}
                if flow_id == "list_archived":
                    people = self.service.list_archived_people()
                    reply = (
                        "已归档人物：\n" + _people_summary(people)
                        if people
                        else "归档里没有人物。"
                    )
                    return {"reply": reply, "state": state}
                return self._start_flow(flow_id)

        return {"reply": self._help(), "state": state}

    def _help(self):
        return (
            "我能帮你操作这个系统，说个意图我就列步骤：\n"
            "· 建人物 —— 逐步填姓名/别名/语言/身份说明\n"
            "· 加材料 —— 给某个人物加文本资料\n"
            "· 搜索 —— 联网搜人物公开资料（结果只进候选）\n"
            "· 归档 / 恢复 / 永久删除\n"
            "· 列出人物 / 归档列表\n"
            "现在说你想做什么？"
        )

    def _start_flow(self, flow_id):
        flow = _FLOWS[flow_id]
        steps = flow["steps"]
        state = {"flow": flow_id, "step": 0, "fields": {}, "pending_action": None}
        self._save_state(state)
        lines = ["开始「%s」，共 %d 步：" % (flow["label"], len(steps))]
        for _field, label, _req in steps:
            lines.append("  " + label)
        if steps[0][0] == "person":
            people = self.service.list_people()
            if people:
                lines.append("")
                lines.append("现有人物：")
                lines.append(_people_summary(people))
        lines.append("")
        lines.append("先回答第一步：")
        lines.append(steps[0][1])
        return {"reply": "\n".join(lines), "state": state}

    def _collect(self, state, clean):
        flow = _FLOWS[state["flow"]]
        steps = flow["steps"]
        step_index = int(state["step"])
        field, label, required = steps[step_index]

        if field == "person":
            person_id = self._pick_person_id(clean)
            if person_id is None:
                return {"reply": self._people_list_prompt(), "state": state}
            state["fields"]["person_id"] = person_id
        else:
            value = str(clean)
            if not value and not required:
                value = ""
            if not value and required:
                return {"reply": "这一步必须填：\n" + label, "state": state}
            state["fields"][field] = value

        next_index = step_index + 1
        if next_index < len(steps):
            state["step"] = next_index
            self._save_state(state)
            return {"reply": "好。\n\n" + steps[next_index][1], "state": state}

        summary = self._summary(state)
        state["pending_action"] = {"flow": state["flow"], "summary": summary}
        state["flow"] = None
        self._save_state(state)
        return {
            "reply": "都齐了，请核对：\n" + summary + "\n\n确认执行吗？（说「确认」执行，或「取消」重来）",
            "state": state,
        }

    def _pick_person_id(self, text):
        people = self.service.list_people()
        if not people:
            return None
        clean = _norm(text)
        for index, person in enumerate(people, 1):
            if clean == str(index):
                return str(person["person_id"])
        matched = [
            p for p in people
            if _norm(str(p.get("name", ""))) in clean or clean in _norm(str(p.get("name", "")))
        ]
        if len(matched) == 1:
            return str(matched[0]["person_id"])
        return None

    def _people_list_prompt(self):
        people = self.service.list_people()
        if not people:
            return "现在还没有人物。先走「建人物」流程。"
        return "没匹配到，请用序号或全名。现有人物：\n" + _people_summary(people)

    def _summary(self, state):
        flow = _FLOWS[state["flow"]]
        fields = state["fields"]
        lines = ["流程：" + flow["label"]]
        for field, label, _req in flow["steps"]:
            if field == "person":
                value = self._person_name(fields.get("person_id", ""))
            else:
                value = fields.get(field, "")
            lines.append("  " + label.rstrip("：") + ": " + (value or "（空）"))
        return "\n".join(lines)

    def _person_name(self, person_id):
        if not person_id:
            return ""
        try:
            return str(self.service.get_person(person_id).get("name", person_id))
        except Exception:
            return str(person_id)

    def _execute(self, action):
        method = getattr(self, _FLOWS[action["flow"]]["execute"])
        return method()

    def _do_create_person(self):
        state = self._load_state()
        fields = state.get("fields", {})
        aliases = [
            a.strip()
            for a in str(fields.get("aliases", "")).replace("，", ",").split(",")
            if a.strip()
        ]
        language = str(fields.get("language", "") or "zh").strip()
        identity = str(fields.get("identity_note", ""))
        person = self.service.create_conversation_person(
            name=str(fields["name"]),
            aliases=aliases,
            language=language,
            description=identity,
            source_mode="user_provided",
            identity_note=identity,
            focus_domain=str(fields.get("focus_domain", "")),
        )
        return "已创建人物「%s」。下一步可以「加材料」贴原始资料，或「搜索」联网找资料。" % person["name"]

    def _do_add_source(self):
        state = self._load_state()
        fields = state.get("fields", {})
        person_id = str(fields["person_id"])
        content = str(fields["content"])
        self.service.add_conversation_text_source(
            person_id,
            title="助手添加的文本材料",
            text=content,
            speaker="",
            source_date="",
            dataset_role="model_source",
        )
        return "已添加文本材料（%d 字）。现在只是候选，可到该人物「人物资料」里审核确认。" % len(content)

    def _do_search(self):
        state = self._load_state()
        fields = state.get("fields", {})
        person_id = str(fields["person_id"])
        if self.service.public_search is None:
            return "联网搜索服务未配置，系统不会伪装成已经搜索。"
        result = self.service.collect_public_sources(person_id)
        collection = dict(result.get("collection", result))
        if collection.get("status") in {"temporarily_unavailable", "search_service_not_configured"}:
            return str(collection.get("message", "搜索失败"))
        return "已搜索并保存为候选（reference_only，未核实）。去该人物「人物资料」查看待审核候选。"

    def _do_archive(self):
        state = self._load_state()
        person_id = str(state["fields"]["person_id"])
        name = self._person_name(person_id)
        self.service.delete_person(person_id)
        return "已归档「%s」（可恢复）。要找回时说「恢复」。" % name

    def _do_restore(self):
        state = self._load_state()
        person_id = str(state["fields"]["person_id"])
        person = self.service.restore_person(person_id)
        return "已恢复「%s」。" % person["name"]

    def _do_permanent_delete(self):
        state = self._load_state()
        person_id = str(state["fields"]["person_id"])
        name = self._person_name(person_id)
        self.service.permanently_delete_archived_person(person_id)
        return "已永久删除「%s」。" % name
