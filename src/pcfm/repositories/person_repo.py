"""Person 表 Repository。"""
from __future__ import annotations

import json


class PersonRepository:
    def __init__(self, db) -> None:
        self.db = db

    def upsert(self, person: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO person "
            "(person_id, name, description, avatar, identity_note, focus_domain, feature_names, health, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                str(person.get("person_id", "")),
                str(person.get("name", "")),
                str(person.get("description", "")),
                str(person.get("avatar", "")),
                str(person.get("identity_note", "")),
                str(person.get("focus_domain", "")),
                json.dumps(person.get("feature_names", []), ensure_ascii=False),
                str(person.get("health", "ok")),
                str(person.get("created_at", "")),
                str(person.get("updated_at", "")),
            ),
        )
        self.db.conn.commit()

    def get(self, person_id: str) -> dict | None:
        row = self.db.conn.execute("SELECT * FROM person WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            return None
        return dict(row)

    def list_index(self) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT person_id, name, description, avatar, identity_note, focus_domain, health FROM person"
        ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM person").fetchone()
        return int(row["n"])

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM person")
        self.db.conn.commit()
