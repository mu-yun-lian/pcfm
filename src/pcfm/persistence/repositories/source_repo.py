"""Source 表 Repository：资料元数据写透。"""
from __future__ import annotations

import json


class SourceRepository:
    def __init__(self, db) -> None:
        self.db = db

    def _execute(self, source: dict) -> None:
        self.db.conn.execute(
            "INSERT OR REPLACE INTO source "
            "(source_id, person_id, title, source_type, format, source_url, filename, speaker, dataset_role, content_authenticity, review_status, content_hash, text_path, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(source.get("source_id", "")),
                str(source.get("person_id", "")),
                str(source.get("title", "")),
                str(source.get("source_type", "text")),
                str(source.get("format", "text")),
                str(source.get("source_url", "")),
                str(source.get("filename", "")),
                str(source.get("speaker", "")),
                str(source.get("dataset_role", "model_source")),
                str(source.get("content_authenticity", "unverified_material")),
                str(source.get("review_status", "pending")),
                str(source.get("content_hash", "")),
                str(source.get("text_path", "")) or None,
                str(source.get("created_at", "")),
                str(source.get("reviewed_at") or source.get("updated_at") or source.get("created_at", "")),
            ),
        )

    def upsert(self, source: dict) -> None:
        self._execute(source)
        self.db.conn.commit()

    def upsert_no_commit(self, source: dict) -> None:
        self._execute(source)

    def count(self) -> int:
        row = self.db.conn.execute("SELECT COUNT(*) AS n FROM source").fetchone()
        return int(row["n"])

    def source_counts(self, person_id: str) -> dict[str, int]:
        """按人物的资料聚合计数(避免 light 列表全量读 JSON)。"""
        row = self.db.conn.execute(
            "SELECT "
            "  COUNT(*) AS total, "
            "  SUM(CASE WHEN review_status='confirmed' THEN 1 ELSE 0 END) AS confirmed, "
            "  SUM(CASE WHEN review_status='pending' THEN 1 ELSE 0 END) AS pending, "
            "  SUM(CASE WHEN review_status='confirmed' AND dataset_role='model_source' THEN 1 ELSE 0 END) AS model_source, "
            "  SUM(CASE WHEN review_status='confirmed' AND dataset_role='final_holdout' THEN 1 ELSE 0 END) AS final_holdout "
            "FROM source WHERE person_id=?",
            (person_id,),
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "confirmed": int(row["confirmed"] or 0),
            "pending": int(row["pending"] or 0),
            "model_source": int(row["model_source"] or 0),
            "final_holdout": int(row["final_holdout"] or 0),
        }

    def clear(self) -> None:
        self.db.conn.execute("DELETE FROM source")
        self.db.conn.commit()
