"""Session 表 Repository：会话元数据写透。"""
from __future__ import annotations

import json


class SessionRepository:
    def __init__(self, db) -> None:
        self.db = db

    def _execute(self, session: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO session "
            "(session_id, person_id, title, active, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?)",
            (
                str(session.get("session_id", "")),
                str(session.get("person_id", "")),
                str(session.get("title", "新对话")),
                int(bool(session.get("active"))),
                str(session.get("created_at", "")),
                str(session.get("updated_at", "")),
            ),
        )

    def upsert(self, session: dict) -> None:
        self._execute(session)
        self.db.conn.commit()

    def upsert_no_commit(self, session: dict) -> None:
        self._execute(session)

    def delete_by_person_no_commit(self, person_id: str) -> None:
        self.db.conn.execute("DELETE FROM session WHERE person_id=?", (person_id,))

    def get(self, session_id: str) -> dict | None:
        row = self.db.conn.execute("SELECT * FROM session WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def delete(self, session_id: str) -> None:
        self.db.conn.execute("DELETE FROM session WHERE session_id=?", (session_id,))
        self.db.conn.commit()

    def list_by_person(self, person_id: str) -> list[dict]:
        rows = self.db.conn.execute("SELECT * FROM session WHERE person_id=?", (person_id,)).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM session").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM session")
        self.db.conn.commit()
