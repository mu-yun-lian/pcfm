"""Message 表 Repository：消息元数据写透。"""
from __future__ import annotations

import json


class MessageRepository:
    def __init__(self, db) -> None:
        self.db = db

    @staticmethod
    def _json(value) -> str | None:
        return json.dumps(value, ensure_ascii=False) if value else None

    def _execute(self, message: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO message "
            "(message_id, session_id, role, text, status, answer_status, structured_prediction, prediction_trace, model_usage, style_status, comparison, feedback, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(message.get("message_id", "")),
                str(message.get("session_id", "")),
                str(message.get("role", "user")),
                str(message.get("text", "")),
                str(message.get("status", "pending")),
                str(message.get("answer_status", "")) or None,
                self._json(message.get("structured_prediction")),
                self._json(message.get("prediction_trace")),
                self._json(message.get("model_usage")),
                str(message.get("style_status", "")) or None,
                self._json(message.get("comparison")),
                self._json(message.get("feedback")),
                str(message.get("created_at", "")),
            ),
        )

    def upsert(self, message: dict) -> None:
        self._execute(message)
        self.db.conn.commit()

    def upsert_no_commit(self, message: dict) -> None:
        self._execute(message)

    def delete_by_session(self, session_id: str) -> None:
        self.delete_by_session_no_commit(session_id)
        self.db.conn.commit()

    def delete_by_session_no_commit(self, session_id: str) -> None:
        self.db.conn.execute("DELETE FROM message WHERE session_id=?", (session_id,))

    def delete_by_id(self, message_id: str) -> None:
        self.db.conn.execute("DELETE FROM message WHERE message_id=?", (message_id,))
        self.db.conn.commit()

    def list_by_session(self, session_id: str) -> list[dict]:
        rows = self.db.conn.execute("SELECT * FROM message WHERE session_id=?", (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM message").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM message")
        self.db.conn.commit()
