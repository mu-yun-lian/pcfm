"""Version 表 Repository：版本元数据写透。"""
from __future__ import annotations

import json


class VersionRepository:
    def __init__(self, db) -> None:
        self.db = db

    def _execute(self, version: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO version "
            "(version, person_id, model_path, simulation_model_path, style_artifact_path, validation_status, source_ids, data, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                int(version.get("version", 0)),
                str(version.get("person_id", "")),
                str(version.get("response_model_path", "")),
                str(version.get("simulation_model_path", "")),
                str(version.get("style_artifact_path", "")),
                str(version.get("validation_status", "")),
                json.dumps(version.get("source_ids", [])),
                json.dumps(version, ensure_ascii=False),
                str(version.get("created_at", "")),
            ),
        )

    def upsert(self, version: dict) -> None:
        self._execute(version)
        self.db.conn.commit()

    def upsert_no_commit(self, version: dict) -> None:
        self._execute(version)

    def delete_by_person_no_commit(self, person_id: str) -> None:
        self.db.conn.execute("DELETE FROM version WHERE person_id=?", (person_id,))

    def list_by_person(self, person_id: str) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM version WHERE person_id=? ORDER BY version", (person_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def list_full_by_person(self, person_id: str) -> list[dict]:
        """从 data 列读取完整版本字典(供 SQLite 优先读路径的影子对比 / 灰度读取)。"""
        rows = self.db.conn.execute(
            "SELECT data FROM version WHERE person_id=? ORDER BY version",
            (person_id,),
        ).fetchall()
        result: list[dict] = []
        for row in rows:
            try:
                item = json.loads(row["data"])
                if isinstance(item, dict):
                    # data 列在同步时注入了 person_id, 而 JSON 版本不含; 剥离以对齐 JSON 真相源
                    item.pop("person_id", None)
                    result.append(item)
            except (ValueError, TypeError):
                continue
        return result

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM version").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM version")
        self.db.conn.commit()
