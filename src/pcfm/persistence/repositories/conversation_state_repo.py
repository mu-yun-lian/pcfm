"""conversation_state 表 Repository：人物对话状态(active_version 等)写透。"""
from __future__ import annotations


class ConversationStateRepository:
    def __init__(self, db) -> None:
        self.db = db

    def _execute(self, person_id: str, state: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO conversation_state "
            "(person_id, active_version, active_session_id, dialogue_model_ref, updated_at) "
            "VALUES (?,?,?,?,?)",
            (
                person_id,
                state.get("active_version"),
                str(state.get("active_session_id", "")) or None,
                str(state.get("dialogue_model_ref", "")),
                str(state.get("updated_at", "")),
            ),
        )

    def upsert(self, person_id: str, state: dict) -> None:
        self._execute(person_id, state)
        self.db.conn.commit()

    def upsert_no_commit(self, person_id: str, state: dict) -> None:
        self._execute(person_id, state)

    def get(self, person_id: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM conversation_state WHERE person_id=?", (person_id,)
        ).fetchone()
        return dict(row) if row else None

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM conversation_state").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM conversation_state")
        self.db.conn.commit()
