"""SQLite 持久化基础层：连接、schema、事务。

P2 用 SQLite 管理可变元数据，大对象仍存文件系统（text_path / content_hash 关联）。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS person (
    person_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    avatar TEXT NOT NULL DEFAULT '',
    identity_note TEXT NOT NULL DEFAULT '',
    focus_domain TEXT NOT NULL DEFAULT '',
    feature_names TEXT NOT NULL DEFAULT '[]',
    health TEXT NOT NULL DEFAULT 'ok',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session (
    session_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES person(person_id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '新对话',
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS message (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES session(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    answer_status TEXT,
    structured_prediction TEXT,
    prediction_trace TEXT,
    model_usage TEXT,
    style_status TEXT,
    comparison TEXT,
    feedback TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source (
    source_id TEXT PRIMARY KEY,
    person_id TEXT NOT NULL REFERENCES person(person_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    format TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL DEFAULT '',
    speaker TEXT NOT NULL DEFAULT '',
    dataset_role TEXT NOT NULL DEFAULT 'model_source',
    content_authenticity TEXT NOT NULL DEFAULT 'unverified_material',
    review_status TEXT NOT NULL DEFAULT 'pending',
    content_hash TEXT NOT NULL,
    text_path TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS version (
    version INTEGER NOT NULL,
    person_id TEXT NOT NULL,
    model_path TEXT,
    simulation_model_path TEXT,
    style_artifact_path TEXT,
    validation_status TEXT,
    source_ids TEXT,
    data TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (person_id, version)
);

CREATE TABLE IF NOT EXISTS job (
    job_id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    person_id TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    error_code TEXT,
    error_message TEXT,
    result TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_state (
    person_id TEXT PRIMARY KEY REFERENCES person(person_id) ON DELETE CASCADE,
    active_version INTEGER,
    active_session_id TEXT,
    dialogue_model_ref TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""


class Database:
    """SQLite 数据库封装：连接、schema、事务。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
        return self._conn

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            try:
                self.conn.commit()
            except Exception:
                pass
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            raise

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
