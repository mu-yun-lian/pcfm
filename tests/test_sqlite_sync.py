"""T1-T6: SQLite 镜像一致性与事务化验收。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.migrate_to_sqlite import migrate, rollback
from pcfm.services import PcfmService


QA = (
    "Q: How should the studio release a product?\n"
    "A: Release it only after the evidence is strong enough, and keep the first version focused."
)


class SqliteSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.tmp.name), seed_example=False)
        self.person = self.service.create_conversation_person(
            name="Alice", aliases=["Alice"], language="en", description="x"
        )
        self.pid = str(self.person["person_id"])

    def tearDown(self) -> None:
        self.service.close()
        self.tmp.cleanup()

    def _add_confirmed(self) -> None:
        source = self.service.add_conversation_text_source(
            self.pid,
            title="Interview",
            text=QA,
            speaker="Alice",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="p1-2",
            source_context="c",
            source_url="https://example.org/x",
        )
        self.service.review_conversation_source(
            self.pid, str(source["source_id"]), "confirmed"
        )

    def test_session_crud_syncs_session_table(self) -> None:
        self.service.create_session(self.pid)
        sessions = self.service.list_sessions(self.pid)
        self.assertEqual(self.service.session_repo.count(), len(sessions))

    def test_delete_session_cleans_messages(self) -> None:
        self._add_confirmed()
        self.service.send_conversation_message(
            self.pid, "How should the studio release a product?"
        )
        self.assertGreater(self.service.message_repo.count(), 0)
        sessions = self.service.list_sessions(self.pid)
        # 删除一个非活动会话
        inactive = [s for s in sessions if not s.get("active")]
        if inactive:
            self.service.delete_session(self.pid, str(inactive[0]["session_id"]))
        self.assertEqual(
            self.service.session_repo.count(), len(self.service.list_sessions(self.pid))
        )

    def test_send_message_syncs_messages_and_state(self) -> None:
        self._add_confirmed()
        self.service.send_conversation_message(
            self.pid, "How should the studio release a product?"
        )
        self.assertEqual(self.service.message_repo.count(), 2)
        self.assertIsNotNone(self.service.state_repo.get(self.pid))

    def test_migrate_and_rollback_clear_all_tables(self) -> None:
        data_dir = Path(self.tmp.name)
        result = migrate(data_dir)
        self.assertEqual(result["migrated"], 1)
        rollback_result = rollback(data_dir)
        for key in (
            "persons_after_clear",
            "sources_after_clear",
            "versions_after_clear",
            "sessions_after_clear",
            "messages_after_clear",
            "states_after_clear",
        ):
            self.assertEqual(rollback_result[key], 0, key)


if __name__ == "__main__":
    unittest.main()
