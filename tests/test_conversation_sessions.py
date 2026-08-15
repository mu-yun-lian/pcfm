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
