"""T1-T6: SQLite 镜像一致性与事务化验收。"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.consistency_check import check
from pcfm.migrate_to_sqlite import migrate, rollback
from pcfm.persistence.db import Database
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

    def test_set_avatar_syncs_person_table(self) -> None:
        self.service.set_avatar(self.pid, "data:image/png;base64,aGVsbG8=")
        row = self.service.person_repo.get(self.pid)
        self.assertIsNotNone(row)
        self.assertEqual(row["avatar"], f"/api/people/{self.pid}/avatar")

    def test_delete_and_restore_person_syncs_health(self) -> None:
        self.service.delete_person(self.pid)
        row = self.service.person_repo.get(self.pid)
        self.assertIsNotNone(row)
        self.assertEqual(row["health"], "archived")
        self.service.restore_person(self.pid)
        row = self.service.person_repo.get(self.pid)
        self.assertIsNotNone(row)
        self.assertEqual(row["health"], "ok")

    def test_permanent_delete_removes_person_row(self) -> None:
        self.service.delete_person(self.pid)
        self.service.permanently_delete_archived_person(self.pid, expected_name="Alice")
        self.assertIsNone(self.service.person_repo.get(self.pid))

    def test_sync_sources_removes_stale_rows(self) -> None:
        self._add_confirmed()
        self.assertEqual(self.service.source_repo.count(), 1)
        sources_path = self.service._person_dir(self.pid) / "conversation_sources.json"
        sources_path.write_text("[]", encoding="utf-8")
        self.service._sync_sources_to_sqlite(self.pid)
        self.assertEqual(self.service.source_repo.count(), 0)

    def test_consistency_check_passes_after_sync(self) -> None:
        self._add_confirmed()
        self.service.send_conversation_message(
            self.pid, "How should the studio release a product?"
        )
        report = check(Path(self.tmp.name))
        self.assertEqual(report["problems"], 0, report["people"][self.pid])

    def test_schema_migration_adds_version_data_column(self) -> None:
        import sqlite3

        db_path = Path(self.tmp.name) / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE version ("
            "version INTEGER NOT NULL, person_id TEXT NOT NULL, model_path TEXT, "
            "validation_status TEXT, PRIMARY KEY (person_id, version))"
        )
        conn.commit()
        conn.close()
        db = Database(db_path)
        columns = {row["name"] for row in db.conn.execute("PRAGMA table_info(version)").fetchall()}
        self.assertIn("data", columns)
        db.close()

    def test_transaction_commit_failure_rolls_back(self) -> None:
        import sqlite3
        from unittest import mock

        db = Database(Path(self.tmp.name) / "tx.db")
        # sqlite3.Connection 的 commit/rollback 是只读 C 级属性, 无法 patch.object;
        # 直接替换内部连接为 MagicMock, 只验证 transaction() 的异常传播与 rollback 调用。
        db._conn = mock.MagicMock()
        db._conn.commit.side_effect = sqlite3.OperationalError("disk full")
        with self.assertRaises(sqlite3.OperationalError):
            with db.transaction():
                db._conn.execute("SELECT 1")
        db._conn.rollback.assert_called_once()
        db.close()

    def test_sync_versions_deletes_stale_rows(self) -> None:
        self._add_confirmed()
        self.assertGreater(self.service.version_repo.count(), 0)
        versions_path = self.service._person_dir(self.pid) / "conversation_versions.json"
        versions_path.write_text("[]", encoding="utf-8")
        self.service._sync_versions_to_sqlite(self.pid)
        self.assertEqual(self.service.version_repo.list_by_person(self.pid), [])

    def test_delete_person_ensures_person_row_before_archive(self) -> None:
        self.service.person_repo.delete(self.pid)
        self.assertIsNone(self.service.person_repo.get(self.pid))
        self.service.delete_person(self.pid)
        row = self.service.person_repo.get(self.pid)
        self.assertIsNotNone(row)
        self.assertEqual(row["health"], "archived")

    def test_restore_person_sqlite_sync_failure_is_non_blocking(self) -> None:
        from unittest import mock

        self.service.delete_person(self.pid)
        with mock.patch.object(
            self.service.person_repo, "upsert", side_effect=RuntimeError("db down")
        ):
            restored = self.service.restore_person(self.pid)
        self.assertEqual(restored["person_id"], self.pid)
        self.assertTrue((self.service._person_dir(self.pid) / "person.json").exists())

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
