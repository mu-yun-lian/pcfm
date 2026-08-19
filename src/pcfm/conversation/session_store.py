from __future__ import annotations

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _canonical_hash,
    _decode_web_bytes,
    _derivation_view,
    _extract_html,
    _extract_pdf,
    _extract_qa,
    _is_short_reference,
    _json_mapping,
    _localize_view,
    _read_json,
    _segments,
    _similarity,
    _structured_rows_text,
    _text_hash,
    _tokens,
    _utc_now,
    _write_json,
)



class SessionStoreMixin:
    def _sessions_dir(self, person_id: str) -> Path:
        return self._person_dir(person_id) / "conversation_sessions"

    def _session_path(self, person_id: str, session_id: str) -> Path:
        return self._sessions_dir(person_id) / f"{str(session_id)}.json"

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _title_from_messages(messages: Sequence[Mapping[str, object]]) -> str:
        for message in messages:
            if str(message.get("role", "")) == "user" and str(message.get("text", "")).strip():
                return str(message["text"]).strip()[:24]
        return "新对话"

    def _read_session(self, person_id: str, session_id: str) -> dict[str, object]:
        raw = _read_json(self._session_path(person_id, session_id), {})
        if not isinstance(raw, dict) or str(raw.get("session_id", "")) != str(session_id):
            raise ConversationError("会话不存在。")
        return dict(raw)

    def _write_session(self, person_id: str, session: Mapping[str, object]) -> None:
        self._sessions_dir(person_id).mkdir(parents=True, exist_ok=True)
        _write_json(self._session_path(person_id, str(session["session_id"])), dict(session))

    @staticmethod
    def _session_meta(session: Mapping[str, object], active_session_id: str) -> dict[str, object]:
        return {
            "session_id": str(session.get("session_id", "")),
            "title": str(session.get("title", "")),
            "created_at": str(session.get("created_at", "")),
            "updated_at": str(session.get("updated_at", "")),
            "message_count": int(session.get("message_count", 0)),
            "active": str(session.get("session_id", "")) == active_session_id,
        }

    def _list_sessions(self, person_id: str) -> list[dict[str, object]]:
        sessions: list[dict[str, object]] = []
        directory = self._sessions_dir(person_id)
        if directory.exists():
            for path in sorted(directory.glob("*.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict) and raw.get("session_id"):
                    sessions.append(dict(raw))
        sessions.sort(key=lambda value: str(value.get("updated_at", "")), reverse=True)
        return sessions

    def _active_messages(self, person_id: str) -> list[dict[str, object]]:
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        return [dict(item) for item in session.get("messages", [])]

    def _save_active_messages(self, person_id, messages, dialogue_state=None) -> None:
        session_id = self._active_session_id(person_id)
        session = self._read_session(person_id, session_id)
        session["messages"] = [dict(item) for item in messages]
        session["message_count"] = len(messages)
        session["updated_at"] = _utc_now()
        if dialogue_state is not None:
            session["dialogue_state"] = dict(dialogue_state)
        if session.get("title") in ("", "新对话"):
            first_user = next((item for item in messages if str(item.get("role", "")) == "user" and str(item.get("text", "")).strip()), None)
            if first_user is not None:
                session["title"] = str(first_user["text"]).strip()[:24]
        self._write_session(person_id, session)

    def _active_session_id(self, person_id: str) -> str:
        state = self._state(person_id)
        active = state.get("active_session_id")
        if active and self._session_path(person_id, str(active)).exists():
            return str(active)
        if not active:
            migrated = self._migrate_sessions(person_id)
            if migrated:
                return str(migrated)
        else:
            surviving = self._list_sessions(person_id)
            if surviving:
                state["active_session_id"] = str(surviving[0]["session_id"])
                _write_json(self._path(person_id, "conversation_state.json"), state)
                return str(surviving[0]["session_id"])
        now = _utc_now()
        sid = self._new_session_id()
        self._write_session(person_id, {
            "schema_version": SCHEMA_VERSION,
            "session_id": sid,
            "person_id": person_id,
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
            "messages": [],
        })
        state["active_session_id"] = sid
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return sid

    def _migrate_sessions(self, person_id: str) -> str:
        state = self._state(person_id)
        existing = state.get("active_session_id")
        if existing:
            return str(existing)
        now = _utc_now()
        self._sessions_dir(person_id).mkdir(parents=True, exist_ok=True)
        person_dialogue_state = dict(state.get("dialogue_state") or {})
        created_ids: list[str] = []

        def build_session(messages, *, created_at):
            sid = self._new_session_id()
            session = {
                "schema_version": SCHEMA_VERSION,
                "session_id": sid,
                "person_id": person_id,
                "title": self._title_from_messages(messages),
                "created_at": created_at,
                "updated_at": created_at,
                "message_count": len(messages),
                "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
                "messages": [dict(message) for message in messages],
            }
            self._write_session(person_id, session)
            created_ids.append(sid)
            return session

        archive_dir = self._person_dir(person_id) / "conversation_archives"
        if archive_dir.exists():
            for path in sorted(archive_dir.glob("*.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict) and raw.get("messages"):
                    build_session([dict(m) for m in raw["messages"]], created_at=str(raw.get("archived_at") or now))

        active_messages = self._list(person_id, "conversation_messages.json")
        active_session_id = ""
        if active_messages:
            active = build_session(active_messages, created_at=now)
            active["dialogue_state"] = person_dialogue_state or active["dialogue_state"]
            self._write_session(person_id, active)
            active_session_id = str(active["session_id"])

        if not active_session_id and created_ids:
            active_session_id = created_ids[-1]
        if not active_session_id:
            return ""

        state["active_session_id"] = active_session_id
        state.pop("dialogue_state", None)
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return active_session_id

    def list_sessions(self, person_id: str) -> list[dict[str, object]]:
        active = self._active_session_id(person_id)
        return [self._session_meta(item, active) for item in self._read_sessions(person_id)]

    def create_session(self, person_id: str) -> dict[str, object]:
        self._migrate_sessions(person_id)
        now = _utc_now()
        sid = self._new_session_id()
        session = {
            "schema_version": SCHEMA_VERSION,
            "session_id": sid,
            "person_id": person_id,
            "title": "新对话",
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "dialogue_state": {"status": "empty", "topic_threads": [], "active_topic_id": "", "active_topic_message_ids": []},
            "messages": [],
        }
        self._write_session(person_id, session)
        state = self._state(person_id)
        state["active_session_id"] = sid
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return self._session_meta(session, sid)

    def switch_session(self, person_id: str, session_id: str) -> dict[str, object]:
        self._read_session(person_id, session_id)
        state = self._state(person_id)
        state["active_session_id"] = str(session_id)
        _write_json(self._path(person_id, "conversation_state.json"), state)
        return self._session_meta(self._read_session(person_id, session_id), str(session_id))

    def rename_session(self, person_id: str, session_id: str, title: str) -> dict[str, object]:
        session = self._read_session(person_id, session_id)
        clean = str(title).strip()
        if not clean:
            raise ConversationError("标题不能为空。")
        session["title"] = clean[:100]
        self._write_session(person_id, session)
        active = str(self._state(person_id).get("active_session_id", ""))
        return self._session_meta(session, active)

    def delete_session(self, person_id: str, session_id: str) -> dict[str, object]:
        self._read_session(person_id, session_id)
        sessions = self._list_sessions(person_id)
        remaining = [item for item in sessions if str(item["session_id"]) != str(session_id)]
        path = self._session_path(person_id, session_id)
        if path.exists():
            path.unlink()
        state = self._state(person_id)
        if not remaining:
            meta = self.create_session(person_id)
            return {"active_session_id": meta["session_id"], "sessions": [meta]}
        if str(state.get("active_session_id", "")) == str(session_id):
            state["active_session_id"] = str(remaining[0]["session_id"])
            _write_json(self._path(person_id, "conversation_state.json"), state)
        active = str(self._state(person_id).get("active_session_id", ""))
        return {"active_session_id": active, "sessions": [self._session_meta(item, active) for item in remaining]}
