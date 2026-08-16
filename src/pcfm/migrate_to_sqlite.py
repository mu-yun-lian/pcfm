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
from .persistence.repositories import PersonRepository, SourceRepository, VersionRepository


def migrate(data_dir: Path) -> dict:
    people_dir = Path(data_dir) / "people"
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    source_repo = SourceRepository(db)
    version_repo = VersionRepository(db)
    migrated = 0
    sources_migrated = 0
    versions_migrated = 0
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
    result = {
        "migrated": migrated,
        "total_in_db": repo.count(),
        "sources": sources_migrated,
        "versions": versions_migrated,
        "errors": errors,
    }
    db.close()
    return result


def rollback(data_dir: Path) -> dict:
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    source_repo = SourceRepository(db)
    version_repo = VersionRepository(db)
    repo.clear()
    source_repo.clear()
    version_repo.clear()
    result = {
        "persons_after_clear": repo.count(),
        "sources_after_clear": source_repo.count(),
        "versions_after_clear": version_repo.count(),
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
