# 会话一等实体重构 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把「会话」从"一个活跃文件 + 一个只进不出的归档目录"重构为"每会话一个文件 + 人物级 active_session_id 指针"，支持列/新建/切换/重命名/删除，修复"新对话后切不回旧对话"。

**Architecture:** 在 `ConversationWorkbench` 内新增会话文件存储层与 CRUD 方法；`summary()`/`send_message()` 等读写路径改为经 `active_session_id` 定位活跃会话文件；`ProductService` 与 `webapp.py` 透传 5 个新 API；前端加顶部会话条 + 历史会话抽屉。人物级状态（`active_version`、`dialogue_model_ref`、`rollback_history`）留在 `conversation_state.json` 不变。

**Tech Stack:** Python 3（标准库 `http.server`、`unittest`）、无第三方后端依赖；前端原生 JS（无框架）。

## Global Constraints

- 会话文件目录：`<person_dir>/conversation_sessions/{session_id}.json`。
- 会话文件 `schema_version` = `pcfm-conversation-mvp-v1`（复用现有 `SCHEMA_VERSION`）。
- 会话标题：首条 `role=="user"` 消息 `text.strip()[:24]`；无用户消息时为 `"新对话"`。
- 删除会话后**始终至少保留一个会话**；删到最后一个时自动新建空会话。
- 切换会话 = 改 `conversation_state.json` 的 `active_session_id`（O(1)），不搬消息、不改 `active_version`。
- 迁移幂等：`conversation_state.json` 已含 `active_session_id` 即视为已迁移，跳过。
- 旧 `conversation_archives/` 与 `conversation_messages.json` 迁移后保留不删。
- 测试用 `unittest`，运行方式：`$env:PYTHONPATH='src'; python -m unittest tests.<module>`（PowerShell）。
- 代码库用 `from __future__ import annotations`；`_write_json`/`_read_json`/`_utc_now`/`ConversationError` 均已在 `conversation_mvp.py` 顶部可用。

---

### Task 1: 会话文件存储层 + 幂等迁移

**Files:**
- Modify: `src/pcfm/conversation_mvp.py`（在 `ConversationWorkbench._path` 之后新增一组 helper 方法）
- Test: `tests/test_conversation_sessions.py`（新建）

**Interfaces:**
- Produces（后续任务依赖）:
  - `_sessions_dir(person_id) -> Path`
  - `_session_path(person_id, session_id) -> Path`
  - `_new_session_id() -> str`
  - `_title_from_messages(messages) -> str`
  - `_read_session(person_id, session_id) -> dict`
  - `_write_session(person_id, session) -> None`
  - `_session_meta(session, active_session_id) -> dict`
  - `_list_sessions(person_id) -> list[dict]`（完整会话文件，按 updated_at 倒序）
  - `_active_session_id(person_id) -> str`（确保存在，缺则迁移）
  - `_migrate_sessions(person_id) -> str`

- [ ] **Step 1: 写失败测试**

在 `tests/test_conversation_sessions.py` 新建：

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.product_service import ProductService


class SessionMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = ProductService(self.storage, seed_example=False)
        self.person = self.service.create_conversation_person(name="Alice Example")
        self.person_id = str(self.person["person_id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_new_person_gets_one_empty_active_session(self) -> None:
        sessions = self.service.conversation.list_sessions(self.person_id)
        self.assertEqual(1, len(sessions))
        self.assertEqual("新对话", sessions[0]["title"])
        self.assertTrue(sessions[0]["active"])
        self.assertEqual(0, sessions[0]["message_count"])

    def test_legacy_archive_migrates_into_session_with_title(self) -> None:
        person_dir = self.storage / "people" / self.person_id
        archive_dir = person_dir / "conversation_archives"
        archive_dir.mkdir(parents=True)
        archive = {
            "schema_version": "pcfm-conversation-mvp-v1",
            "archive_id": "conversation-abc123def456",
            "person_id": self.person_id,
            "archived_at": "2026-08-15T00:00:00Z",
            "active_version": None,
            "messages": [
                {"message_id": "m1", "person_id": self.person_id, "role": "user",
                 "text": "你如何看待设计上的极简主义？", "created_at": "2026-08-15T00:00:00Z"},
            ],
        }
        (archive_dir / "conversation-abc123def456.json").write_text(
            json.dumps(archive, ensure_ascii=False), encoding="utf-8"
        )
        sid = self.service.conversation._active_session_id(self.person_id)
        sessions = self.service.conversation.list_sessions(self.person_id)
        # 当前活跃为空但有归档：归档成为唯一活跃会话（spec 第 7 节）
        self.assertEqual(1, len(sessions))
        migrated = sessions[0]
        self.assertEqual("你如何看待设计上的极简主义？", migrated["title"])
        self.assertEqual(1, migrated["message_count"])
        self.assertTrue(migrated["active"])
        self.assertEqual(sid, migrated["session_id"])

    def test_migration_is_idempotent(self) -> None:
        first = self.service.conversation._active_session_id(self.person_id)
        second = self.service.conversation._active_session_id(self.person_id)
        self.assertEqual(first, second)
        self.assertEqual(1, len(self.service.conversation.list_sessions(self.person_id)))
```

- [ ] **Step 2: 运行测试确认失败**

```
$env:PYTHONUTF8='1'; $env:PYTHONPATH='src'; python -m unittest tests.test_conversation_sessions -v
```
预期：FAIL，`AttributeError: 'ConversationWorkbench' object has no attribute 'list_sessions'`。

- [ ] **Step 3: 实现存储层与迁移**

在 `ConversationWorkbench._path` 方法之后插入：

```python
    def _sessions_dir(self, person_id: str) -> Path:
        return self._person_dir(person_id) / "conversation_sessions"

    def _session_path(self, person_id: str, session_id: str) -> Path:
        return self._sessions_dir(person_id) / f"{str(session_id)}.json"

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _title_from_messages(messages: Sequence[Mapping[str, object]]) -> str:
        for message in messages:
            if str(message.get("role", "")) == "user" and str(message.get("text", "")).strip():
                return str(message["text"]).strip()[:24]
        return "新对话"

    def _read_session(self, person_id: str, session_id: str) -> dict[str, object]:
        raw = _read_json(self._session_path(person_id, session_id), {})
        if not isinstance(raw, dict) or str(raw.get("session_id", "")) != str(session_id):
            raise ConversationError("会话不存在。")
        return dict(raw)

    def _write_session(self, person_id: str, session: Mapping[str, object]) -> None:
        self._sessions_dir(person_id).mkdir(parents=True, exist_ok=True)
        _write_json(self._session_path(person_id, str(session["session_id"])), dict(session))

    @staticmethod
    def _session_meta(session: Mapping[str, object], active_session_id: str) -> dict[str, object]:
        return {
            "session_id": str(session.get("session_id", "")),
            "title": str(session.get("title", "")),
            "created_at": str(session.get("created_at", "")),
            "updated_at": str(session.get("updated_at", "")),
            "message_count": int(session.get("message_count", 0)),
            "active": str(session.get("session_id", "")) == active_session_id,
        }

    def _list_sessions(self, person_id: str) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        directory = self._sessions_dir(person_id)
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict) and raw.get("session_id"):
                    sessions.append(dict(raw))
        sessions.sort(key=lambda value: str(value.get("updated_at", "")), reverse=True)
        return sessions

    def _active_session_id(self, person_id: str) -> str:
        state = self._state(person_id)
        active = state.get("active_session_id")
        if active and self._session_path(person_id, str(active)).exists():
            return str(active)
        return self._migrate_sessions(person_id)

    def _migrate_sessions(self, person_id: str) -> str:
        state = self._state(person_id)
        now = _utc_now()
        self._sessions_dir(person_id).mkdir(parents=True, exist_ok=True)
        person_dialogue_state = dict(state.get("dialogue_state") or {})
        created_ids: list[str] = []

        def build_session(messages, *, created_at):
            sid = self._new_session_id()
            session = {
                "schema_version": SCHEMA_VERSION,
                "session_id": sid,
                "person_id": person_id,
                "title": self._title_from_messages(messages),
                "created_at": created_at,
                "updated_at": created_at,
                "message_count": len(messages),
                "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
                "messages": [dict(message) for message in messages],
            }
            self._write_session(person_id, session)
            created_ids.append(sid)
            return session

        archive_dir = self._person_dir(person_id) / "conversation_archives"
        if archive_dir.exists():
            for path in sorted(archive_dir.glob("*.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict) and raw.get("messages"):
                    build_session([dict(m) for m in raw["messages"]], created_at=str(raw.get("archived_at") or now))

        active_messages = self._list(person_id, "conversation_messages.json")
        active_session_id = ""
        if active_messages:
            active = build_session(active_messages, created_at=now)
            active["dialogue_state"] = person_dialogue_state or active["dialogue_state"]
            self._write_session(person_id, active)
            active_session_id = str(active["session_id"])

        if not active_session_id and created_ids:
            active_session_id = created_ids[-1]
        if not active_session_id:
            sid = self._new_session_id()
            session = {
                "schema_version": SCHEMA_VERSION,
                "session_id": sid,
                "person_id": person_id,
                "title": "新对话",
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
                "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
                "messages": [],
            }
            self._write_session(person_id, session)
            active_session_id = sid

        state["active_session_id"] = active_session_id
        state.pop("dialogue_state", None)
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return active_session_id

    def list_sessions(self, person_id: str) -> list[dict[str, object]]:
        active = self._active_session_id(person_id)
        return [self._session_meta(item, active) for item in self._list_sessions(person_id)]
```

- [ ] **Step 4: 运行测试确认通过**

```
$env:PYTHONUTF8='1'; $env:PYTHONPATH='src'; python -m unittest tests.test_conversation_sessions -v
```
预期：PASS（3 个测试）。

- [ ] **Step 5: 提交**

```
git add src/pcfm/conversation_mvp.py tests/test_conversation_sessions.py
git commit -m "feat: 会话文件存储层与幂等迁移(旧归档+当前消息→会话目录)"
```

---

### Task 2: 会话 CRUD 公共方法

**Files:**
- Modify: `src/pcfm/conversation_mvp.py`（`list_sessions` 之后新增 create/switch/rename/delete，`start_new_conversation` 改为委托 create_session）
- Test: `tests/test_conversation_sessions.py`（追加）

**Interfaces:**
- Consumes: Task 1 全部 helper。
- Produces:
  - `create_session(person_id) -> dict`（返回 meta）
  - `switch_session(person_id, session_id) -> dict`（返回 meta）
  - `rename_session(person_id, session_id, title) -> dict`（返回 meta）
  - `delete_session(person_id, session_id) -> dict`（返回 `{"active_session_id": str, "sessions": list}`）
  - `start_new_conversation(person_id) -> dict`（委托 create_session）

- [ ] **Step 1: 写失败测试**

在 `tests/test_conversation_sessions.py` 追加：

```python
class SessionCrudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = ProductService(self.storage, seed_example=False)
        self.person = self.service.create_conversation_person(name="Alice Example")
        self.person_id = str(self.person["person_id"])
        self.cv = self.service.conversation

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create_switch_rename_delete_roundtrip(self) -> None:
        a = self.cv.create_session(self.person_id)
        b = self.cv.create_session(self.person_id)
        self.assertNotEqual(a["session_id"], b["session_id"])
        self.assertEqual(b["session_id"], self.cv._active_session_id(self.person_id))
        switched = self.cv.switch_session(self.person_id, a["session_id"])
        self.assertEqual(a["session_id"], switched["session_id"])
        self.assertEqual(a["session_id"], self.cv._active_session_id(self.person_id))
        renamed = self.cv.rename_session(self.person_id, a["session_id"], "产品讨论")
        self.assertEqual("产品讨论", renamed["title"])
        result = self.cv.delete_session(self.person_id, b["session_id"])
        ids = {s["session_id"] for s in result["sessions"]}
        self.assertNotIn(b["session_id"], ids)
        self.assertEqual(a["session_id"], self.cv._active_session_id(self.person_id))

    def test_delete_active_switches_to_most_recent(self) -> None:
        a = self.cv.create_session(self.person_id)
        self.cv.create_session(self.person_id)
        result = self.cv.delete_session(self.person_id, a["session_id"])
        remaining = result["sessions"]
        self.assertEqual(1, len(remaining))
        self.assertEqual(result["active_session_id"], remaining[0]["session_id"])

    def test_delete_last_session_creates_empty_one(self) -> None:
        only = self.cv.list_sessions(self.person_id)[0]
        result = self.cv.delete_session(self.person_id, only["session_id"])
        self.assertEqual(1, len(result["sessions"]))
        self.assertNotEqual(only["session_id"], result["sessions"][0]["session_id"])
        self.assertEqual("新对话", result["sessions"][0]["title"])

    def test_rename_blank_title_rejected(self) -> None:
        sid = self.cv.create_session(self.person_id)["session_id"]
        with self.assertRaises(Exception):
            self.cv.rename_session(self.person_id, sid, "   ")
```

- [ ] **Step 2: 运行测试确认失败**
预期：FAIL，`AttributeError: 'ConversationWorkbench' object has no attribute 'create_session'`。

- [ ] **Step 3: 实现 CRUD**

在 `list_sessions` 之后追加：

```python
    def create_session(self, person_id: str) -> dict[str, object]:
        now = _utc_now()
        sid = self._new_session_id()
        session = {
            "schema_version": SCHEMA_VERSION,
            "session_id": sid,
            "person_id": person_id,
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
            "messages": [],
        }
        self._write_session(person_id, session)
        state = self._state(person_id)
        state["active_session_id"] = sid
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return self._session_meta(session, sid)

    def switch_session(self, person_id: str, session_id: str) -> dict[str, object]:
        self._read_session(person_id, session_id)
        state = self._state(person_id)
        state["active_session_id"] = str(session_id)
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return self._session_meta(self._read_session(person_id, session_id), str(session_id))

    def rename_session(self, person_id: str, session_id: str, title: str) -> dict[str, object]:
        session = self._read_session(person_id, session_id)
        clean = str(title).strip()
        if not clean:
            raise ConversationError("标题不能为空。")
        session["title"] = clean[:100]
        self._write_session(person_id, session)
        active = str(self._state(person_id).get("active_session_id", ""))
        return self._session_meta(session, active)

    def delete_session(self, person_id: str, session_id: str) -> dict[str, object]:
        sessions = self._list_sessions(person_id)
        remaining = [item for item in sessions if str(item["session_id"]) != str(session_id)]
        path = self._session_path(person_id, session_id)
        if path.exists():
            path.unlink()
        state = self._state(person_id)
        if not remaining:
            meta = self.create_session(person_id)
            return {"active_session_id": meta["session_id"], "sessions": [meta]}
        if str(state.get("active_session_id", "")) == str(session_id):
            state["active_session_id"] = str(remaining[0]["session_id"])
            _write_json(self._path(person_id, "conversation_state.json"), state)
        active = str(self._state(person_id).get("active_session_id", ""))
        return {"active_session_id": active, "sessions": [self._session_meta(item, active) for item in remaining]}
```

把 `start_new_conversation` 原实现整个替换为：

```python
    def start_new_conversation(self, person_id: str) -> dict[str, object]:
        """新建一个空会话并设为活跃；旧的归档到 conversation_archives 已由会话文件取代。"""
        self.profile(person_id)
        return self.create_session(person_id)
```

- [ ] **Step 4: 运行测试确认通过**
预期：PASS（CRUD 4 个 + Task 1 的 3 个）。

- [ ] **Step 5: 提交**

```
git add src/pcfm/conversation_mvp.py tests/test_conversation_sessions.py
git commit -m "feat: 会话CRUD(create/switch/rename/delete)并替换旧归档实现"
```

---

### Task 3: 改写读写路径指向活跃会话

**Files:**
- Modify: `src/pcfm/conversation_mvp.py`（`send_message`、`summary`、`feedback`、`_find_message`、`find_conversation_reality_answer` 内所有 `conversation_messages.json` 读写点）
- Test: 回归 `tests/test_conversation_mvp.py`、`tests/test_webapp.py`

**Interfaces:**
- Consumes: Task 1 `_active_session_id`/`_read_session`/`_write_session`。
- Produces:
  - `_active_messages(person_id) -> list[dict]`
  - `_save_active_messages(person_id, messages, dialogue_state=None) -> None`
  - `summary()` 新增返回字段 `session_id`、`session_title`、`active_session_id`

- [ ] **Step 1: 新增两个读写 helper**

在 `_list_sessions` 之后追加：

```python
    def _active_messages(self, person_id: str) -> list[dict[str, object]]:
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        return [dict(item) for item in session.get("messages", [])]

    def _save_active_messages(self, person_id, messages, dialogue_state=None) -> None:
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        session["messages"] = [dict(item) for item in messages]
        session["message_count"] = len(messages)
        session["updated_at"] = _utc_now()
        if dialogue_state is not None:
            session["dialogue_state"] = dict(dialogue_state)
        if session.get("title") in ("", "新对话"):
            first_user = next((item for item in messages if str(item.get("role", "")) == "user" and str(item.get("text", "")).strip()), None)
            if first_user is not None:
                session["title"] = str(first_user["text"]).strip()[:24]
        self._write_session(person_id, session)
```

- [ ] **Step 2: 改写 send_message**

将 `send_message` 中 `messages = self._list(person_id, "conversation_messages.json")` 改为 `messages = self._active_messages(person_id)`。

将两处（原写消息后）的：

```python
        _write_json(self._path(person_id, "conversation_messages.json"), messages)
        state = self._state(person_id)
        state["dialogue_state"] = self._conversation_context(profile, messages, "")
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return copy.deepcopy(base)
```
改为：

```python
        self._save_active_messages(person_id, messages, dialogue_state=self._conversation_context(profile, messages, ""))
        return copy.deepcopy(base)
```

- [ ] **Step 3: 改写 summary**

将 `messages = self._list(person_id, "conversation_messages.json")` 改为：

```python
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        messages = [dict(item) for item in session.get("messages", [])]
```

将 `"dialogue_state": copy.deepcopy(dict(state.get("dialogue_state") or {})),` 改为：

```python
        "dialogue_state": copy.deepcopy(dict(session.get("dialogue_state") or {})),
        "session_id": session_id,
        "session_title": str(session.get("title", "")),
        "active_session_id": session_id,
```

- [ ] **Step 4: 改写 feedback 与 _find_message**

`_find_message` 的 `messages = self._list(person_id, "conversation_messages.json")` 改为 `messages = self._active_messages(person_id)`。
`feedback` 的 `_write_json(self._path(person_id, "conversation_messages.json"), messages)` 改为 `self._save_active_messages(person_id, messages)`。

- [ ] **Step 5: 改写 find_conversation_reality_answer**

将两处 `_write_json(self._path(person_id, "conversation_messages.json"), messages)` 改为 `self._save_active_messages(person_id, messages)`。

- [ ] **Step 6: 运行回归测试**

```
$env:PYTHONUTF8='1'; $env:PYTHONPATH='src'; python -m unittest tests.test_conversation_sessions tests.test_conversation_mvp
```
预期：全部通过。用 `grep -n "conversation_messages.json" src/pcfm/conversation_mvp.py` 确认只剩 create 初始化与迁移中的兼容读取。

- [ ] **Step 7: 提交**

```
git add src/pcfm/conversation_mvp.py
git commit -m "feat: send/summary/feedback/reality读写活跃会话文件,summary返回会话字段"
```

---

### Task 4: ProductService 透传 + webapp 路由 + 旧测试更新

**Files:**
- Modify: `src/pcfm/product_service.py`（新增 5 个透传方法）
- Modify: `src/pcfm/webapp.py`（GET/POST/DELETE 分支各加路由）
- Test: `tests/test_webapp.py`（更新旧断言 + 新增会话路由测试）

**Interfaces:**
- Consumes: Task 2 的 `list_sessions/create_session/switch_session/rename_session/delete_session`。
- Produces: `ProductService` 同名 5 方法；路由 GET `/api/people/{id}/conversation/sessions`、POST `.../sessions/{sid}/switch`、`.../sessions/{sid}/rename`、DELETE `.../sessions/{sid}`。

- [ ] **Step 1: 新增 ProductService 透传方法**

在 `product_service.py` 的 `start_new_conversation` 之后追加：

```python
    def list_sessions(self, person_id: str) -> list[dict[str, object]]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.list_sessions, person_id)

    def create_session(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.create_session, person_id)

    def switch_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.switch_session, person_id, session_id)

    def rename_session(self, person_id: str, session_id: str, title: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.rename_session, person_id, session_id, title)

    def delete_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(self.conversation.delete_session, person_id, session_id)
```

- [ ] **Step 2: 加 webapp 路由**

GET 分支（`conversation_summary` 那段之后）新增：

```python
                if (
                    len(parts) == 5
                    and parts[:2] == ["api", "people"]
                    and parts[3:5] == ["conversation", "sessions"]
                ):
                    self._send_json({"ok": True, "sessions": service.list_sessions(parts[2])})
                    return
```

POST 分支（`action == ["conversation", "new"]` 之后）新增：

```python
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sessions"]
                        and action[3] == "switch"
                    ):
                        result = service.switch_session(person_id, action[2])
                        self._send_json({"ok": True, "session": result})
                        return
                    if (
                        len(action) == 4
                        and action[:2] == ["conversation", "sessions"]
                        and action[3] == "rename"
                    ):
                        result = service.rename_session(person_id, action[2], str(body.get("title", "")))
                        self._send_json({"ok": True, "session": result})
                        return
```

DELETE 分支（`delete_person` 之后）新增：

```python
                if (
                    len(parts) == 6
                    and parts[:2] == ["api", "people"]
                    and parts[3:5] == ["conversation", "sessions"]
                ):
                    result = service.delete_session(parts[2], parts[5])
                    self._send_json({"ok": True, **result})
                    return
```

- [ ] **Step 3: 更新旧测试并新增路由测试**

把 `test_new_conversation_archives_old_context_instead_of_deleting_it` 重写为：

```python
    def test_new_conversation_creates_switchable_session(self) -> None:
        status, created = self.json_request("/api/conversation/people", "POST", {"name": "Archive Chat"})
        self.assertEqual(status, 201)
        person_id = created["person"]["person_id"]
        status, _reply = self.json_request(f"/api/people/{person_id}/conversation/messages", "POST", {"text": "你好"})
        self.assertEqual(status, 201)
        status, new_session = self.json_request(f"/api/people/{person_id}/conversation/new", "POST", {})
        self.assertEqual(status, 200)
        first_id = new_session["conversation"]["session_id"]
        status, listed = self.json_request(f"/api/people/{person_id}/conversation/sessions")
        self.assertEqual(status, 200)
        self.assertEqual(2, len(listed["sessions"]))
        active = next(s for s in listed["sessions"] if s["active"])
        self.assertEqual(first_id, active["session_id"])
        old_id = next(s for s in listed["sessions"] if not s["active"])["session_id"]
        status, switched = self.json_request(f"/api/people/{person_id}/conversation/sessions/{old_id}/switch", "POST", {})
        self.assertEqual(status, 200)
        self.assertEqual(old_id, switched["session"]["session_id"])
        status, summary = self.json_request(f"/api/people/{person_id}/conversation")
        self.assertEqual(status, 200)
        self.assertEqual(1, len(summary["conversation"]["messages"]))
```

- [ ] **Step 4: 运行测试**

```
$env:PYTHONUTF8='1'; $env:PYTHONPATH='src'; python -m unittest tests.test_webapp tests.test_conversation_sessions tests.test_conversation_mvp
```
预期：全部通过。

- [ ] **Step 5: 提交**

```
git add src/pcfm/product_service.py src/pcfm/webapp.py tests/test_webapp.py
git commit -m "feat: 会话API透传与路由(list/switch/rename/delete)"
```

---

### Task 5: 前端会话条 + 历史会话抽屉

**Files:**
- Modify: `src/pcfm/web_static/index.html`（chat-header 加会话条、新增 sessions-dialog 抽屉）
- Modify: `src/pcfm/web_static/app.js`（渲染会话条、列/切/重命名/删除会话）

**Interfaces:**
- Consumes: Task 4 的 5 个路由；后端 `summary()` 的 `session_id/session_title` 字段。
- Produces: `renderSessionBar()`、`loadSessions()`、会话切换/重命名/删除事件处理。

- [ ] **Step 1: index.html 加会话条与抽屉**

在 `.person-heading` 之后、`<div class="header-actions">` 之前插入：

```html
        <div class="session-bar">
          <span class="session-title" id="session-title">新对话</span>
          <span class="session-count" id="session-count">0 条消息</span>
        </div>
```

在 `header-actions` 里 `new-conversation` 按钮后加：

```html
            <button class="button quiet" id="open-sessions">历史会话</button>
```

在 `</main>` 之后、`comparison-drawer` 之前新增抽屉：

```html
    <dialog id="sessions-dialog" class="wide-dialog">
      <div class="dialog-card">
        <div class="dialog-head"><div><p class="kicker">会话管理</p><h2>历史会话</h2></div><button type="button" class="close-button" data-close="sessions-dialog">关闭</button></div>
        <div id="sessions-list" class="source-list"></div>
      </div>
    </dialog>
```

- [ ] **Step 2: app.js 渲染与事件**

在 `renderWorkspace()` 末尾（`renderMessages(); renderSources(); ...` 之后）加 `renderSessionBar();`。

在 app.js 追加：

```javascript
function renderSessionBar() {
  $("#session-title").textContent = state.conversation?.session_title || "新对话";
  $("#session-count").textContent = (state.conversation?.messages?.length || 0) + " 条消息";
}

async function loadSessions() {
  const data = await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions');
  const sessions = data.sessions;
  $("#sessions-list").innerHTML = sessions.length ? sessions.map(s => {
    return '<article class="session-item ' + (s.active ? 'active' : '') + '" data-session-id="' + escapeHtml(s.session_id) + '">' +
      '<button class="session-select" data-session-id="' + escapeHtml(s.session_id) + '"><strong>' + escapeHtml(s.title || '新对话') + '</strong>' +
      '<small>' + s.message_count + ' 条消息 · ' + escapeHtml(shortTime(s.updated_at)) + '</small></button>' +
      '<div class="session-actions"><button class="mini-button" data-rename-session="' + escapeHtml(s.session_id) + '">重命名</button>' +
      '<button class="mini-button" data-delete-session="' + escapeHtml(s.session_id) + '">删除</button></div></article>';
  }).join("") : '<p class="people-empty">暂无会话。</p>';
  $$(".session-select").forEach(button => button.onclick = async () => {
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.sessionId) + '/switch', {method:'POST', body:'{}'});
    $("#sessions-dialog").close();
    await refreshConversation();
  });
  $$("[data-rename-session]").forEach(button => button.onclick = async () => {
    const title = prompt('新标题：', state.conversation?.session_title || '');
    if (title === null) return;
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.renameSession) + '/rename', {method:'POST', body:JSON.stringify({title})});
    await loadSessions(); await refreshConversation();
  });
  $$("[data-delete-session]").forEach(button => button.onclick = async () => {
    if (!confirm('删除这个会话？消息将无法恢复。')) return;
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.deleteSession), {method:'DELETE', body:'{}'});
    await loadSessions(); await refreshConversation();
  });
}
```

在初始化事件绑定区（`$("#new-conversation").onclick=...` 附近）加：

```javascript
  $("#open-sessions").onclick = async () => { await loadSessions(); $("#sessions-dialog").showModal(); };
```

把 `startNewConversation` 改为：

```javascript
async function startNewConversation() {
  await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/new', {method:'POST', body:'{}'});
  $("#new-conversation-dialog").close();
  await refreshConversation();
  toast('新对话已开始。');
}
```

- [ ] **Step 3: 手动验证（Playwright）**

预期：标题显示当前会话标题，抽屉打开后有 ≥1 个会话项，无 JS 报错。

- [ ] **Step 4: 提交**

```
git add src/pcfm/web_static/index.html src/pcfm/web_static/app.js
git commit -m "feat: 前端会话条与历史会话抽屉"
```

---

### Task 6: 端到端 + 真实数据迁移验证

**Files:** 无代码改动（验证）。

- [ ] **Step 1: 清 pycache 并重启服务**
- [ ] **Step 2: 验证真实数据迁移**（Jobs 24 条、Obama 90 条旧归档都作为会话出现且有标题；旧目录保留）
- [ ] **Step 3: UI 端到端切换验证**（新建会话 → 发消息 → 切回旧会话 → 旧消息回来不串）
- [ ] **Step 4: 全量回归**（test_conversation_sessions / test_conversation_mvp / test_product_service / test_webapp）
- [ ] **Step 5: 提交**（如有运行时数据变更单独说明）
