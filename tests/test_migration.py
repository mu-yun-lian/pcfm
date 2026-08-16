# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.persistence.db import Database
from pcfm.migrate_to_sqlite import migrate, rollback
from pcfm.persistence.repositories import PersonRepository


class MigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        people_dir = self.data_dir / "people"
        for name, pid in [("Alice", "person-alice"), ("Bob", "person-bob")]:
            d = people_dir / pid
            d.mkdir(parents=True)
            (d / "person.json").write_text(
                json.dumps({
                    "person_id": pid, "name": name, "description": "", "avatar": "",
                    "identity_note": "", "focus_domain": "", "feature_names": ["x"],
                    "created_at": "2025-01-01T00:00:00Z", "updated_at": "2025-01-01T00:00:00Z",
                }),
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_migrate_and_rollback(self) -> None:
        result = migrate(self.data_dir)
        self.assertEqual(result["migrated"], 2)
        self.assertEqual(result["total_in_db"], 2)

        db = Database(self.data_dir / "pcfm.db")
        repo = PersonRepository(db)
        names = sorted(p["name"] for p in repo.list_index())
        self.assertEqual(names, ["Alice", "Bob"])
        db.close()

        result2 = rollback(self.data_dir)
        self.assertEqual(result2["persons_after_clear"], 0)


if __name__ == "__main__":
    unittest.main()
