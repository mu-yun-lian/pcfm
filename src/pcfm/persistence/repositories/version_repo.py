"""Version 表 Repository：版本元数据写透。"""
from __future__ import annotations

import json


class VersionRepository:
    def __init__(self, db) -> None:
        self.db = db

    def _execute(self, version: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO version "
            "(version, person_id, model_path, simulation_model_path, style_artifact_path, validation_status, source_ids, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                int(version.get("version", 0)),
                str(version.get("person_id", "")),
                str(version.get("response_model_path", "")),
                str(version.get("simulation_model_path", "")),
                str(version.get("style_artifact_path", "")),
                str(version.get("validation_status", "")),
                json.dumps(version.get("source_ids", [])),
                str(version.get("created_at", "")),
            ),
        )

    def upsert(self, version: dict) -> None:
        self._execute(version)
        self.db.conn.commit()

    def upsert_no_commit(self, version: dict) -> None:
        self._execute(version)

    def list_by_person(self, person_id: str) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM version WHERE person_id=? ORDER BY version", (person_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM version").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM version")
        self.db.conn.commit()
