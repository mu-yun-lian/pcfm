# -*- coding: utf-8 -*-
"""AI 助手：LLM 工具调用代理（引导式操作员）。

用户说自然语言 -> 助手用大模型理解意图 -> 需要时调用网页功能(工具) -> 逐步收集/执行。
边界：不可逆操作先确认；搜索结果只进候选；助手只调用工具、不编造结果。
"""
from __future__ import annotations

import json
from pathlib import Path

from .model_services import ModelServiceError

MAX_TOOL_ITER = 4

_BASE_SYSTEM = (
    "你是 PCFM 对话式人物模拟系统的操作助手。你帮用户操作系统，可以调用工具执行操作。\n"
    "规则：\n"
    "1. 引导式：用户说明意图后，你逐步收集必要信息；信息不足就先问，别乱猜。\n"
    "2. 不可逆操作（永久删除）执行前必须向用户确认一次。\n"
    "3. 联网搜索的结果只是候选材料，不是已核实真相，别声称是真相。\n"
    "4. 你只能调用工具，不能编造执行结果；执行结果以工具返回为准。\n"
    "5. 你的回复必须是 JSON 对象：\n"
    '   - 只要说话：{"reply": "..."}\n'
    '   - 要调用工具：{"reply": "先说一句正在做什么", "tool_calls": [{"tool": "工具名", "args": {...}}]}\n'
    "下面是你可调用的工具（含参数说明）：\n"
    "{tools}"
)

_TOOLS = [
    ("list_people", "列出所有已创建的人物（返回名称和ID）", {}),
    ("list_archived_people", "列出所有已归档的人物（返回名称和ID）", {}),
    (
        "create_person",
        "新建一个人物",
        {
            "name": "人物姓名(必填)",
            "aliases": "别名列表(可空,数组)",
            "language": "语言(默认 zh)",
            "identity_note": "身份说明(一句话,可空)",
            "focus_domain": "专注领域(可空)",
        },
    ),
    (
        "add_text_source",
        "给某个人物添加文本原始材料",
        {
            "person_id": "人物ID(必填,先用 list_people 获取)",
            "text": "文本材料内容(必填)",
        },
    ),
    (
        "search_person",
        "联网搜索某个人物的公开资料(结果只进候选)",
        {"person_id": "人物ID(必填)"},
    ),
    ("archive_person", "把某个人物移入归档(可恢复)", {"person_id": "人物ID(必填)"}),
    ("restore_person", "恢复某个已归档的人物", {"person_id": "人物ID(必填)"}),
    (
        "permanent_delete_person",
        "永久删除某个已归档人物(不可恢复,执行前须向用户确认)",
        {"person_id": "人物ID(必填)"},
    ),
    (
        "process_materials",
        "一键处理某个人物的全部原始材料(确认来源→提取候选→确认候选,自动生成模型版本),之后即可对话",
        {"person_id": "人物ID(必填)"},
    ),
    (
        "web_search",
        "联网搜索某个人物/话题的公开资料并抓取原文(结果只进候选,未核实说话人)",
        {"person_id": "人物ID(必填)", "query": "搜索关键词(必填,如人物姓名)"},
    ),
]


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
        state = {"history": [], "model_ref": ""}
        self._save_state(state)
        return state

    def set_model(self, model_ref):
        state = self._load_state() or self.reset()
        state["model_ref"] = str(model_ref).strip()
        self._save_state(state)
        return state

    def conversation(self):
        """把助手历史导出成人物对话同构结构，供前端共用聊天界面。"""
        state = self._load_state() or self.reset()
        history = state.get("history", [])
        messages = []
        for index, item in enumerate(history):
            messages.append(
                {
                    "message_id": "assistant-msg-%d" % index,
                    "role": str(item.get("role", "user")),
                    "text": str(item.get("content", "")),
                    "status": "answered",
                    "answer_status": "assistant",
                    "person_prediction_status": "not_available",
                    "knowledge_source": "assistant",
                }
            )
        return {
            "active_version": None,
            "source_counts": {"confirmed": 0},
            "profile": {"collection": {"status": "assistant", "message": "AI 助手"}},
            "public_response_model": {},
            "dialogue_model_ref": self._model_ref(state),
            "status": "assistant",
            "status_text": "AI 助手（操作员）",
            "messages": messages,
        }

    def _model_ref(self, state):
        configured = str(state.get("model_ref", "")).strip()
        if configured:
            return configured
        roles = self.service.model_services.roles() if self.service.model_services else {}
        return str(roles.get("default_dialogue", ""))

    def _tools_text(self):
        lines = []
        for name, description, parameters in _TOOLS:
            params = ", ".join(
                "%s=%s" % (k, v) for k, v in parameters.items()
            ) if parameters else "无参数"
            lines.append("- %s: %s（参数：%s）" % (name, description, params))
        return "\n".join(lines)

    def _execute_tool(self, name, args):
        active = self.service.list_people()
        archived = self.service.list_archived_people()
        # 名称→ID：活跃 + 归档都搜，restore/permanent_delete 需要归档里的人物
        by_name = {
            str(p["name"]).casefold(): str(p["person_id"])
            for p in active + archived
        }

        if name == "list_people":
            people = self.service.list_people()
            return "还没有人物。" if not people else "\n".join(
                "%d. %s (ID: %s)" % (i, p["name"], p["person_id"])
                for i, p in enumerate(people, 1)
            )
        if name == "list_archived_people":
            return "归档里没有人物。" if not archived else "\n".join(
                "%d. %s (ID: %s)" % (i, p["name"], p["person_id"])
                for i, p in enumerate(archived, 1)
            )
        if name == "create_person":
            aliases = args.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [a.strip() for a in aliases.replace("，", ",").split(",") if a.strip()]
            person = self.service.create_conversation_person(
                name=str(args["name"]),
                aliases=[str(a) for a in aliases],
                language=str(args.get("language") or "zh"),
                description=str(args.get("identity_note") or ""),
                source_mode="user_provided",
                identity_note=str(args.get("identity_note") or ""),
                focus_domain=str(args.get("focus_domain") or ""),
            )
            return "已创建人物「%s」，ID=%s" % (person["name"], person["person_id"])
        if name == "add_text_source":
            pid = self._resolve_person(args.get("person_id"), by_name)
            self.service.add_conversation_text_source(
                pid,
                title="助手添加的文本材料",
                text=str(args["text"]),
                speaker="",
                source_date="",
                dataset_role="model_source",
            )
            return "已添加文本材料。现在只是候选，需在人物资料里审核确认。"
        if name == "search_person":
            pid = self._resolve_person(args.get("person_id"), by_name)
            if self.service.public_search is None:
                return "联网搜索服务未配置，系统不会伪装成已经搜索。"
            result = self.service.collect_public_sources(pid)
            collection = dict(result.get("collection", result))
            if collection.get("status") in {"temporarily_unavailable", "search_service_not_configured"}:
                return str(collection.get("message", "搜索失败"))
            return "已搜索并保存为候选（未核实）。去该人物「人物资料」查看待审核候选。"
        if name == "archive_person":
            pid = self._resolve_person(args.get("person_id"), by_name)
            self.service.delete_person(pid)
            return "已归档。"
        if name == "restore_person":
            pid = self._resolve_person(args.get("person_id"), by_name)
            person = self.service.restore_person(pid)
            return "已恢复「%s」。" % person["name"]
        if name == "permanent_delete_person":
            pid = self._resolve_person(args.get("person_id"), by_name)
            self.service.permanently_delete_archived_person(pid)
            return "已永久删除。"
        if name == "process_materials":
            return self._process_materials(self._resolve_person(args.get("person_id"), by_name))
        if name == "web_search":
            return self._web_search(self._resolve_person(args.get("person_id"), by_name), args)
        return "未知工具：" + name

    def _web_search(self, pid, args):
        query = str(args.get("query", "")).strip()
        if not query:
            return "请提供搜索关键词。"
        if self.service.public_search is None:
            return "搜索服务未配置。"
        try:
            person = self.service.get_person(pid)
        except Exception:
            person = {}
        results = self.service.public_search.search(
            person_name=query,
            identity_note=str(person.get("identity_note", "")),
            language="zh",
            limit=10,
        )
        if not results:
            return "没搜到结果。"
        lines = []
        added = 0
        for r in results:
            title = str(r.get("title", "")).strip() or str(r.get("url", ""))
            url = str(r.get("url", "")).strip()
            snippet = str(r.get("snippet", "")).strip()
            if not url:
                continue
            fetched = False
            try:
                self.service.add_conversation_url_source(
                    pid,
                    url=url,
                    speaker="",
                    dataset_role="reference_only",
                    content_authenticity="unverified_material",
                    source_locator="web_search",
                    source_context="联网搜索抓取原文；尚未确认说话人。",
                )
                fetched = True
            except Exception:
                fetched = False
            if not fetched and snippet:
                try:
                    self.service.add_conversation_text_source(
                        pid,
                        title=title,
                        text=snippet,
                        speaker="",
                        source_date=str(r.get("published_at", "")),
                        dataset_role="reference_only",
                        content_authenticity="search_snippet",
                        source_locator="web_search",
                        source_url=url,
                        source_context="联网搜索摘要；尚未确认说话人。",
                    )
                    added += 1
                    lines.append("· %s（摘要）" % title)
                    continue
                except Exception:
                    continue
            if fetched:
                added += 1
                lines.append("· %s（原文）" % title)
        if not added:
            return "搜索到 %d 条，但都没能抓取。" % len(results)
        return "搜索到 %d 条，已加入 %d 篇候选：\n%s" % (len(results), added, "\n".join(lines))

    def _process_materials(self, pid):
        """确认来源 → 提取候选 → 确认候选；每确认一条候选自动生成模型版本。"""
        conv = self.service.conversation
        confirmed_sources = 0
        extracted = 0
        confirmed_candidates = 0
        # 第一遍：确认所有待审核来源
        for s in conv.sources(pid):
            if str(s.get("review_status")) == "pending":
                try:
                    self.service.review_conversation_source(pid, str(s["source_id"]), "confirmed")
                    confirmed_sources += 1
                except Exception:
                    pass
        # 第二遍：给没有候选的已确认来源提取候选
        for s in conv.sources(pid):
            if not s.get("llm_response_event_candidates"):
                try:
                    self.service.extract_conversation_response_candidates(pid, str(s["source_id"]))
                    extracted += 1
                except Exception:
                    pass
        # 第三遍：确认所有待审核候选
        for s in conv.sources(pid):
            for c in s.get("llm_response_event_candidates") or []:
                if str(c.get("review_status")) == "pending":
                    try:
                        self.service.review_conversation_response_candidate(
                            pid, str(s["source_id"]), str(c["candidate_id"]), "confirmed"
                        )
                        confirmed_candidates += 1
                    except Exception:
                        pass
        if not confirmed_sources and not extracted and not confirmed_candidates:
            return "这个人物没有待处理的材料（来源都已确认、候选也都处理完了）。"
        return (
            "处理完成：确认来源 %d 份、提取候选 %d 份、确认候选 %d 条。"
            "每确认一条候选会生成一个模型版本，现在可以直接对话了。"
        ) % (confirmed_sources, extracted, confirmed_candidates)

    def _resolve_person(self, ref, by_name):
        ref = str(ref or "").strip()
        if ref.casefold() in by_name:
            return by_name[ref.casefold()]
        for person in self.service.list_people():
            if str(person["person_id"]) == ref:
                return ref
        raise ValueError("找不到这个人，请先用 list_people 查人物ID。")

    def _call_model(self, model_ref, messages):
        service, model_id = self.service.model_services.resolve_model_ref(model_ref)
        return self.service.model_services.invoke(
            str(service["service_id"]),
            model_id,
            messages,
            structured=True,
            temperature=0.2,
        )

    def handle(self, text):
        state = self._load_state() or self.reset()
        model_ref = self._model_ref(state)
        if not model_ref:
            return {
                "reply": "助手还没配置大模型。请先在「模型服务」里配置并设为默认对话模型。",
                "state": state,
            }
        system = _BASE_SYSTEM.replace("{tools}", self._tools_text())
        history = state.get("history", [])[-12:]
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": str(text)})

        reply = "（助手没有返回内容）"
        for _ in range(MAX_TOOL_ITER):
            try:
                response = self._call_model(model_ref, messages)
                parsed = json.loads(response["text"])
            except (ModelServiceError, json.JSONDecodeError) as error:
                reply = "助手调用大模型失败：" + str(error)
                break
            if not isinstance(parsed, dict):
                reply = "助手返回了无法读取的内容。"
                break
            reply = str(parsed.get("reply", "")).strip() or reply
            tool_calls = parsed.get("tool_calls") or []
            if not tool_calls:
                break
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            results = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    results.append({"tool": "", "ok": False, "error": "无效工具调用"})
                    continue
                name = str(tc.get("tool", ""))
                args = tc.get("args") or {}
                try:
                    result = self._execute_tool(name, args)
                    results.append({"tool": name, "ok": True, "result": result})
                except Exception as error:
                    results.append({"tool": name, "ok": False, "error": str(error)})
            messages.append(
                {"role": "user", "content": "工具执行结果：" + json.dumps(results, ensure_ascii=False)}
            )

        history.append({"role": "user", "content": str(text)})
        history.append({"role": "assistant", "content": reply})
        state["history"] = history[-24:]
        self._save_state(state)
        return {"reply": reply, "state": state}
