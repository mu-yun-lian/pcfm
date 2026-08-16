from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm import conversation_mvp as cm
from pcfm.services import PcfmService


class SessionMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = PcfmService(self.storage, seed_example=False)
        self.person = self.service.create_conversation_person(name="Alice Example")
        self.person_id = str(self.person["person_id"])
        self._reset_sessions()

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _reset_sessions(self) -> None:
        import shutil
        person_dir = self.storage / "people" / self.person_id
        sessions_dir = person_dir / "conversation_sessions"
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir)
        state_path = person_dir / "conversation_state.json"
        state = cm._read_json(state_path, {})
        state.pop("active_session_id", None)
        state.pop("dialogue_state", None)
        cm._write_json(state_path, state)

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

    def test_legacy_messages_migrate_into_active_session_preserving_dialogue_state(self) -> None:
        person_dir = self.storage / "people" / self.person_id
        messages = [
            {"message_id": "m1", "person_id": self.person_id, "role": "user",
             "text": "你好，我们来聊聊设计", "created_at": "2026-08-15T00:00:00Z"},
            {"message_id": "m2", "person_id": self.person_id, "role": "assistant",
             "text": "好，我们从简洁开始", "created_at": "2026-08-15T00:00:01Z"},
        ]
        (person_dir / "conversation_messages.json").write_text(
            json.dumps(messages, ensure_ascii=False), encoding="utf-8"
        )
        state = {
            "schema_version": cm.SCHEMA_VERSION,
            "active_version": None,
            "rollback_history": [],
            "dialogue_state": {
                "status": "active",
                "topic_threads": [],
                "active_topic_id": "t1",
                "active_topic_message_ids": ["m1"],
            },
        }
        (person_dir / "conversation_state.json").write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )
        sid = self.service.conversation._active_session_id(self.person_id)
        sessions = self.service.conversation.list_sessions(self.person_id)
        self.assertEqual(1, len(sessions))
        active = sessions[0]
        self.assertEqual("你好，我们来聊聊设计", active["title"])
        self.assertEqual(2, active["message_count"])
        self.assertTrue(active["active"])
        self.assertEqual(sid, active["session_id"])
        # dialogue_state 从 conversation_state.json 保留，而不是被重置为空
        full = self.service.conversation._read_session(self.person_id, sid)
        self.assertEqual("t1", full["dialogue_state"]["active_topic_id"])

    def test_start_new_conversation_migrates_legacy_first(self) -> None:
        person_dir = self.storage / "people" / self.person_id
        (person_dir / "conversation_messages.json").write_text(
            json.dumps([{"message_id": "m1", "person_id": self.person_id, "role": "user",
                        "text": "你好，我们来聊聊设计", "created_at": "2026-08-15T00:00:00Z"}], ensure_ascii=False),
            encoding="utf-8",
        )
        self.service.conversation.start_new_conversation(self.person_id)
        sessions = self.service.conversation.list_sessions(self.person_id)
        self.assertEqual(2, len(sessions))  # legacy + 新空会话
        titles = {s["title"] for s in sessions}
        self.assertIn("你好，我们来聊聊设计", titles)
        self.assertIn("新对话", titles)

    def test_delete_active_session_file_does_not_duplicate(self) -> None:
        sid = self.service.conversation._active_session_id(self.person_id)
        self.assertEqual(1, len(self.service.conversation.list_sessions(self.person_id)))
        (self.storage / "people" / self.person_id / "conversation_sessions" / (sid + ".json")).unlink()
        new_sid = self.service.conversation._active_session_id(self.person_id)
        sessions = self.service.conversation.list_sessions(self.person_id)
        self.assertEqual(1, len(sessions))  # 无重复
        self.assertEqual(new_sid, sessions[0]["session_id"])


class SessionCrudTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = PcfmService(self.storage, seed_example=False)
        self.person = self.service.create_conversation_person(name="Alice Example")
        self.person_id = str(self.person["person_id"])
        self.cv = self.service.conversation
        self._reset_sessions()

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _reset_sessions(self) -> None:
        import shutil
        person_dir = self.storage / "people" / self.person_id
        sessions_dir = person_dir / "conversation_sessions"
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir)
        state_path = person_dir / "conversation_state.json"
        state = cm._read_json(state_path, {})
        state.pop("active_session_id", None)
        state.pop("dialogue_state", None)
        cm._write_json(state_path, state)

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

    def test_delete_active_switches_to_remaining_session(self) -> None:
        a = self.cv.create_session(self.person_id)
        b = self.cv.create_session(self.person_id)
        c = self.cv.create_session(self.person_id)  # active 是 c
        result = self.cv.delete_session(self.person_id, c["session_id"])
        remaining_ids = {s["session_id"] for s in result["sessions"]}
        self.assertEqual({a["session_id"], b["session_id"]}, remaining_ids)
        self.assertIn(result["active_session_id"], {a["session_id"], b["session_id"]})
        self.assertNotEqual(c["session_id"], result["active_session_id"])

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

    def test_delete_unknown_session_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.cv.delete_session(self.person_id, "session-does-not-exist")

    def test_summary_returns_session_fields(self) -> None:
        sid = self.cv.create_session(self.person_id)["session_id"]
        summary = self.service.conversation_summary(self.person_id)
        self.assertEqual(sid, summary["session_id"])
        self.assertEqual(sid, summary["active_session_id"])
        self.assertEqual("新对话", summary["session_title"])

    def test_send_message_persists_to_active_session_file(self) -> None:
        reply = self.service.send_conversation_message(self.person_id, "你好")
        self.assertIn("message_id", reply)
        session = self.cv._read_session(self.person_id, self.cv._active_session_id(self.person_id))
        self.assertGreaterEqual(session["message_count"], 2)  # user + assistant
        self.assertEqual("你好", session["title"])
        legacy = self.cv._list(self.person_id, "conversation_messages.json")
        self.assertEqual([], legacy)  # 旧文件不再承载消息

