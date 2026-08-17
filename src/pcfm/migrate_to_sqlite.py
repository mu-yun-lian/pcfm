"""JSON → SQLite 迁移脚本（P2 阶段）。

用法:
  python -m pcfm.migrate_to_sqlite --data-dir <data_dir>
  python -m pcfm.migrate_to_sqlite --data-dir <data_dir> --rollback
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_errors import safe_read_json
from .persistence.db import Database
from .persistence.repositories import (
    ConversationStateRepository,
    MessageRepository,
    PersonRepository,
    SessionRepository,
    SourceRepository,
    VersionRepository,
)


def migrate(data_dir: Path) -> dict:
    people_dir = Path(data_dir) / "people"
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    source_repo = SourceRepository(db)
    version_repo = VersionRepository(db)
    session_repo = SessionRepository(db)
    message_repo = MessageRepository(db)
    state_repo = ConversationStateRepository(db)
    migrated = 0
    sources_migrated = 0
    versions_migrated = 0
    sessions_migrated = 0
    messages_migrated = 0
    errors = []
    for path in sorted(people_dir.glob("*/person.json")):
        person_id = path.parent.name
        try:
            person = safe_read_json(path, {})
            if isinstance(person, dict) and person.get("person_id"):
                repo.upsert(person)
                migrated += 1
        except Exception as error:  # 单人物损坏不阻断迁移
            errors.append(f"{person_id}: {error}")
        try:
            sources = safe_read_json(path.parent / "conversation_sources.json", [])
            if isinstance(sources, list):
                for item in sources:
                    record = dict(item)
                    record["person_id"] = person_id
                    source_repo.upsert(record)
                    sources_migrated += 1
        except Exception:
            pass
        try:
            versions = safe_read_json(path.parent / "conversation_versions.json", [])
            if isinstance(versions, list):
                for item in versions:
                    record = dict(item)
                    record["person_id"] = person_id
                    version_repo.upsert(record)
                    versions_migrated += 1
        except Exception:
            pass
        try:
            state = safe_read_json(path.parent / "conversation_state.json", {})
            active_id = str(state.get("active_session_id", "")) if isinstance(state, dict) else ""
            if isinstance(state, dict):
                state_repo.upsert(person_id, {
                    "active_version": state.get("active_version"),
                    "active_session_id": state.get("active_session_id", ""),
                    "dialogue_model_ref": state.get("dialogue_model_ref", ""),
                    "updated_at": str(state.get("updated_at", "")),
                })
            sessions_dir = path.parent / "conversation_sessions"
            if sessions_dir.exists():
                for session_path in sorted(sessions_dir.glob("*.json")):
                    session = safe_read_json(session_path, {})
                    if not isinstance(session, dict) or not session.get("session_id"):
                        continue
                    sid = str(session["session_id"])
                    session_record = dict(session)
                    session_record["person_id"] = person_id
                    session_record["active"] = sid == active_id
                    session_repo.upsert(session_record)
                    sessions_migrated += 1
                    for item in session.get("messages", []):
                        if isinstance(item, dict):
                            msg_record = dict(item)
                            msg_record["session_id"] = sid
                            message_repo.upsert(msg_record)
                            messages_migrated += 1
        except Exception:
            pass
    result = {
        "migrated": migrated,
        "total_in_db": repo.count(),
        "sources": sources_migrated,
        "versions": versions_migrated,
        "sessions": sessions_migrated,
        "messages": messages_migrated,
        "errors": errors,
    }
    db.close()
    return result


def rollback(data_dir: Path) -> dict:
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    source_repo = SourceRepository(db)
    version_repo = VersionRepository(db)
    session_repo = SessionRepository(db)
    message_repo = MessageRepository(db)
    state_repo = ConversationStateRepository(db)
    repo.clear()
    source_repo.clear()
    version_repo.clear()
    session_repo.clear()
    message_repo.clear()
    state_repo.clear()
    result = {
        "persons_after_clear": repo.count(),
        "sources_after_clear": source_repo.count(),
        "versions_after_clear": version_repo.count(),
        "sessions_after_clear": session_repo.count(),
        "messages_after_clear": message_repo.count(),
        "states_after_clear": state_repo.count(),
    }
    db.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="JSON → SQLite 迁移")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        print(rollback(args.data_dir))
    else:
        print(migrate(args.data_dir))


if __name__ == "__main__":
    main()
