"""一致性校验: 逐人物比较 SQLite 镜像与 JSON 真相源。

用法:
    python -m pcfm.consistency-check --data-dir <data_dir>

退出码: 0 = 全部一致; 1 = 存在不一致或数据损坏。

比较范围(6 张镜像表 ↔ 6 类 JSON 文件):
    person / source / version / conversation_state / session / message(活动会话)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pcfm.persistence.db import Database
from pcfm.persistence.repositories import (
    ConversationStateRepository,
    MessageRepository,
    PersonRepository,
    SessionRepository,
    SourceRepository,
    VersionRepository,
)


def _read_json(path: Path, default):
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, (dict, list)) else default
    except Exception:
        return default


def _norm(value) -> str:
    return "" if value is None else str(value)


def _check_person(db, repos, person_dir: Path, person_id: str, problems: list[str]) -> None:
    person_json = _read_json(person_dir / "person.json", {})
    row = repos["person"].get(person_id)
    if row is None:
        problems.append("person 表缺行")
    else:
        for key in ("name", "description", "avatar", "identity_note", "focus_domain"):
            if _norm(row.get(key)) != _norm(person_json.get(key)):
                problems.append(f"person.{key}: sqlite={row.get(key)!r} json={person_json.get(key)!r}")

    sources = _read_json(person_dir / "conversation_sources.json", [])
    srows = db.conn.execute(
        "SELECT source_id, review_status, dataset_role FROM source WHERE person_id=?",
        (person_id,),
    ).fetchall()
    by_id = {str(r["source_id"]): r for r in srows}
    if len(sources) != len(srows):
        problems.append(f"source 数量不一致: json={len(sources)} sqlite={len(srows)}")
    else:
        for item in sources:
            sid = str(item.get("source_id", ""))
            r = by_id.get(sid)
            if r is None:
                problems.append(f"source {sid} 表缺行")
                continue
            for key in ("review_status", "dataset_role"):
                if _norm(r[key]) != _norm(item.get(key)):
                    problems.append(f"source {sid}.{key}: sqlite={r[key]!r} json={item.get(key)!r}")

    versions = _read_json(person_dir / "conversation_versions.json", [])
    vrows = repos["version"].list_by_person(person_id)
    vmap = {int(r["version"]): r for r in vrows}
    if len(versions) != len(vrows):
        problems.append(f"version 数量不一致: json={len(versions)} sqlite={len(vrows)}")
    else:
        # 全量 data 列对比(与读路径 list_full_by_person 同源), 校验完整版本字典一致
        sqlite_full = repos["version"].list_full_by_person(person_id)
        json_sorted = sorted(versions, key=lambda v: int(v.get("version", 0)))
        if sqlite_full != json_sorted:
            problems.append("version.data 全量与 JSON 不一致")
        for item in versions:
            vn = int(item.get("version", 0))
            r = vmap.get(vn)
            if r is None:
                problems.append(f"version {vn} 表缺行")
                continue
            if _norm(r.get("validation_status")) != _norm(item.get("validation_status")):
                problems.append(
                    f"version {vn}.validation_status: sqlite={r.get('validation_status')!r} "
                    f"json={item.get('validation_status')!r}"
                )
            sqlite_ids = json.loads(r.get("source_ids") or "[]")
            if sorted(map(str, sqlite_ids)) != sorted(map(str, item.get("source_ids", []))):
                problems.append(f"version {vn}.source_ids 不一致")

    state = _read_json(person_dir / "conversation_state.json", {})
    srow = repos["state"].get(person_id)
    if srow is None:
        if state.get("active_version") is not None or state.get("dialogue_model_ref"):
            problems.append("state 表缺行")
    else:
        for key in ("active_version", "active_session_id", "dialogue_model_ref"):
            if _norm(srow.get(key)) != _norm(state.get(key)):
                problems.append(f"state.{key}: sqlite={srow.get(key)!r} json={state.get(key)!r}")

    sessions_dir = person_dir / "conversation_sessions"
    json_sessions = sorted(p.stem for p in sessions_dir.glob("*.json")) if sessions_dir.exists() else []
    sqlite_sessions = sorted(str(r["session_id"]) for r in repos["session"].list_by_person(person_id))
    if json_sessions != sqlite_sessions:
        problems.append(f"session 不一致: json={json_sessions} sqlite={sqlite_sessions}")

    active_sid = state.get("active_session_id")
    if active_sid:
        active_session = _read_json(sessions_dir / f"{active_sid}.json", {})
        json_messages = active_session.get("messages", [])
        mrows = repos["message"].list_by_session(str(active_sid))
        if len(json_messages) != len(mrows):
            problems.append(f"message 数量不一致: json={len(json_messages)} sqlite={len(mrows)}")
        else:
            mids_json = [str(m.get("message_id", "")) for m in json_messages]
            mids_sqlite = [str(r["message_id"]) for r in mrows]
            if mids_json != mids_sqlite:
                problems.append("message message_id 序列不一致")


def check(data_dir: Path) -> dict:
    db = Database(data_dir / "pcfm.db")
    repos = {
        "person": PersonRepository(db),
        "source": SourceRepository(db),
        "version": VersionRepository(db),
        "session": SessionRepository(db),
        "message": MessageRepository(db),
        "state": ConversationStateRepository(db),
    }
    people_dir = data_dir / "people"
    report: dict[str, object] = {"people": {}, "problems": 0}
    try:
        person_dirs = sorted(people_dir.glob("*/person.json"))
        for path in person_dirs:
            person_id = path.parent.name
            problems: list[str] = []
            try:
                _check_person(db, repos, path.parent, person_id, problems)
            except Exception as error:
                problems.append(f"校验异常: {error}")
            report["people"][person_id] = problems
            if problems:
                report["problems"] = int(report["problems"]) + len(problems)
    finally:
        db.close()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="比较 SQLite 镜像与 JSON 真相源")
    parser.add_argument("--data-dir", required=True, help="数据目录(含 people/ 与 pcfm.db)")
    args = parser.parse_args(argv)
    report = check(Path(args.data_dir))
    people = report["people"]
    if not people:
        print("没有找到任何人物。")
        return 0
    for person_id, problems in people.items():
        if problems:
            print(f"[{person_id}] 不一致 {len(problems)} 处:")
            for p in problems:
                print(f"  - {p}")
        else:
            print(f"[{person_id}] OK")
    print(f"共 {len(people)} 人, 不一致 {report['problems']} 处。")
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
