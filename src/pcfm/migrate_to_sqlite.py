"""JSON → SQLite 迁移脚本（P2 阶段）。

用法:
  python -m pcfm.migrate_to_sqlite --data-dir <data_dir>
  python -m pcfm.migrate_to_sqlite --data-dir <data_dir> --rollback
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_errors import safe_read_json
from .db import Database
from .repositories import PersonRepository


def migrate(data_dir: Path) -> dict:
    people_dir = Path(data_dir) / "people"
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    migrated = 0
    errors = []
    for path in sorted(people_dir.glob("*/person.json")):
        try:
            person = safe_read_json(path, {})
            if isinstance(person, dict) and person.get("person_id"):
                repo.upsert(person)
                migrated += 1
        except Exception as error:  # 单人物损坏不阻断迁移
            errors.append(f"{path.parent.name}: {error}")
    result = {"migrated": migrated, "total_in_db": repo.count(), "errors": errors}
    db.close()
    return result


def rollback(data_dir: Path) -> dict:
    db = Database(Path(data_dir) / "pcfm.db")
    repo = PersonRepository(db)
    repo.clear()
    result = {"persons_after_clear": repo.count()}
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
