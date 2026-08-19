"""主线第5步: SQLite 读路径(影子/灰度/回退/自愈)回归测试。"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pcfm.services import PcfmService


QA = (
    "Q: How should the studio release a product?\n"
    "A: Release it only after the evidence is strong enough, and keep the first version focused."
)


class ReadPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.tmp.name), seed_example=False)
        self.person = self.service.create_conversation_person(
            name="Alice", aliases=["Alice"], language="en", description="x"
        )
        self.pid = str(self.person["person_id"])
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

    def tearDown(self) -> None:
        self.service.close()
        self.tmp.cleanup()

    def _json_versions(self):
        return self.service.conversation._list(self.pid, "conversation_versions.json")

    def _corrupt_version_data(self) -> None:
        self.service.db.conn.execute(
            "UPDATE version SET data='{}' WHERE person_id=?", (self.pid,)
        )
        self.service.db.conn.commit()

    def test_shadow_read_returns_json_and_logs_mismatch(self) -> None:
        self._corrupt_version_data()
        with mock.patch.dict(os.environ, {"PCFM_SQLITE_READ_PRIMARY": "0"}):
            with self.assertLogs("pcfm", level="WARNING") as captured:
                result = self.service.conversation._read_versions(self.pid)
        self.assertEqual(result, self._json_versions())
        self.assertTrue(any("mismatch" in line for line in captured.output))

    def test_sqlite_primary_returns_sqlite_when_consistent(self) -> None:
        with mock.patch.dict(os.environ, {"PCFM_SQLITE_READ_PRIMARY": "1"}):
            result = self.service.conversation._read_versions(self.pid)
        self.assertEqual(result, self._json_versions())

    def test_sqlite_primary_falls_back_when_empty(self) -> None:
        self.service.db.conn.execute("DELETE FROM version WHERE person_id=?", (self.pid,))
        self.service.db.conn.commit()
        with mock.patch.dict(os.environ, {"PCFM_SQLITE_READ_PRIMARY": "1"}):
            result = self.service.conversation._read_versions(self.pid)
        self.assertEqual(result, self._json_versions())
        # 自愈后 SQLite 重新有版本行
        self.assertEqual(len(self.service.version_repo.list_by_person(self.pid)), len(self._json_versions()))

    def test_sqlite_primary_falls_back_and_heals_on_mismatch(self) -> None:
        self._corrupt_version_data()
        with mock.patch.dict(os.environ, {"PCFM_SQLITE_READ_PRIMARY": "1"}):
            result = self.service.conversation._read_versions(self.pid)
        self.assertEqual(result, self._json_versions())
        # 自愈后 SQLite data 恢复与 JSON 一致
        healed = self.service.version_repo.list_full_by_person(self.pid)
        self.assertEqual(healed, self._json_versions())

    def test_sqlite_primary_state_and_session_read_return_json(self) -> None:
        with mock.patch.dict(os.environ, {"PCFM_SQLITE_READ_PRIMARY": "1"}):
            state = self.service.conversation._read_state(self.pid)
            sessions = self.service.conversation._read_sessions(self.pid)
        self.assertEqual(state, self.service.conversation._state(self.pid))
        self.assertEqual(sessions, self.service.conversation._list_sessions(self.pid))

    def test_message_read_still_json(self) -> None:
        self.service.send_conversation_message(self.pid, "How should the studio release a product?")
        messages = self.service.conversation._active_messages(self.pid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
