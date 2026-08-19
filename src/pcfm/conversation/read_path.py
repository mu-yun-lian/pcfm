"""SQLite 优先读路径: 影子对比 + 灰度开关 + 不一致自愈。

设计: docs/主线第5步SQLite读路径流程设计方案.md
- 影子阶段(默认): 读 JSON + 对比 SQLite, 不一致只告警, 返回值始终 JSON。
- 灰度阶段(PCFM_SQLITE_READ_PRIMARY=1): 实际可切换范围只有 version(表有完整 data 列);
  state/session 表只有关键字段(缺 rollback_history/message_count 等), 故做
  "SQLite 对比 + 不一致自愈", 仍返回 JSON。
"""
from __future__ import annotations

import logging
import os


def _sqlite_read_primary() -> bool:
    # 5.3 默认开启 SQLite 元数据读; 设 PCFM_SQLITE_READ_PRIMARY=0 可回退影子模式
    return os.environ.get("PCFM_SQLITE_READ_PRIMARY", "1") == "1"


def _norm(value) -> str:
    return "" if value is None else str(value)


def _log_fallback(person_id: str, table: str, reason: str, exc_info: bool = False, level: int = logging.WARNING) -> None:
    logging.getLogger("pcfm").log(
        level,
        "sqlite read fallback json person_id=%s table=%s reason=%s",
        person_id, table, reason, exc_info=exc_info,
    )


def _repo_open(repo) -> bool:
    """镜像库连接是否已打开(即是否发生过同步)。影子模式在库未打开时跳过对比, 避免无谓打开连接。"""
    db = getattr(repo, "db", None)
    return db is not None and db.is_open


class ReadPathMixin:
    def _heal(self, table: str, person_id: str) -> None:
        """触发服务层 _sync_* 自愈; 失败只告警, 不阻断读。"""
        sync = getattr(self, "_sync_callback", None)
        if not sync:
            return
        fn = sync.get(table)
        if fn is None:
            return
        try:
            fn(person_id)
            logging.getLogger("pcfm").info(
                "sqlite self-heal triggered person_id=%s table=%s", person_id, table
            )
        except Exception:
            logging.getLogger("pcfm").warning(
                "sqlite self-heal failed person_id=%s table=%s", person_id, table, exc_info=True
            )

    def _read_versions(self, person_id: str) -> list[dict]:
        json_versions = self._list(person_id, "conversation_versions.json")
        repo = getattr(self, "_version_repo", None)
        if repo is None:
            return json_versions
        if not _repo_open(repo) and not _sqlite_read_primary():
            return json_versions
        try:
            sqlite_versions = repo.list_full_by_person(person_id)
        except Exception:
            if _sqlite_read_primary():
                _log_fallback(person_id, "version", "read_error", exc_info=True)
            return json_versions
        if not sqlite_versions:
            if _sqlite_read_primary() and json_versions:
                _log_fallback(person_id, "version", "empty", level=logging.INFO)
                self._heal("versions", person_id)
            return json_versions
        if sqlite_versions != json_versions:
            _log_fallback(person_id, "version", "mismatch")
            if _sqlite_read_primary():
                self._heal("versions", person_id)
            return json_versions
        if _sqlite_read_primary():
            return sqlite_versions
        return json_versions

    def _read_state(self, person_id: str) -> dict:
        json_state = self._state(person_id)
        repo = getattr(self, "_state_repo", None)
        if repo is None:
            return json_state
        if not _repo_open(repo) and not _sqlite_read_primary():
            return json_state
        try:
            sqlite_row = repo.get(person_id)
        except Exception:
            if _sqlite_read_primary():
                _log_fallback(person_id, "state", "read_error", exc_info=True)
            return json_state
        if sqlite_row is None:
            if (
                _sqlite_read_primary()
                and (
                    json_state.get("active_version") is not None
                    or json_state.get("active_session_id")
                    or json_state.get("dialogue_model_ref")
                )
            ):
                _log_fallback(person_id, "state", "empty", level=logging.INFO)
                self._heal("state", person_id)
            return json_state
        for key in ("active_version", "active_session_id", "dialogue_model_ref"):
            if _norm(sqlite_row.get(key)) != _norm(json_state.get(key)):
                _log_fallback(person_id, "state", "mismatch")
                if _sqlite_read_primary():
                    self._heal("state", person_id)
                return json_state
        # state 表仅存关键字段, 不能完整表示 JSON state(含 rollback_history 等), 故始终返回 JSON。
        return json_state

    def _read_sessions(self, person_id: str) -> list[dict]:
        json_sessions = self._list_sessions(person_id)
        repo = getattr(self, "_session_repo", None)
        if repo is None:
            return json_sessions
        if not _repo_open(repo) and not _sqlite_read_primary():
            return json_sessions
        try:
            sqlite_rows = repo.list_by_person(person_id)
        except Exception:
            if _sqlite_read_primary():
                _log_fallback(person_id, "session", "read_error", exc_info=True)
            return json_sessions
        if not sqlite_rows:
            if _sqlite_read_primary() and json_sessions:
                _log_fallback(person_id, "session", "empty", level=logging.INFO)
                self._heal("sessions", person_id)
            return json_sessions
        json_by_id = {str(item.get("session_id", "")): item for item in json_sessions}
        sqlite_by_id = {str(row["session_id"]): row for row in sqlite_rows}
        if set(json_by_id) != set(sqlite_by_id):
            _log_fallback(person_id, "session", "mismatch")
            if _sqlite_read_primary():
                self._heal("sessions", person_id)
            return json_sessions
        active_id = str(self._active_session_id(person_id))
        for sid, item in json_by_id.items():
            row = sqlite_by_id[sid]
            if _norm(row.get("title")) != _norm(item.get("title")) or \
                    _norm(row.get("updated_at")) != _norm(item.get("updated_at")) or \
                    bool(row.get("active")) != (str(sid) == active_id):
                _log_fallback(person_id, "session", "mismatch")
                if _sqlite_read_primary():
                    self._heal("sessions", person_id)
                return json_sessions
        # session 表缺 message_count/dialogue_state/messages, 故始终返回 JSON。
        return json_sessions
