from __future__ import annotations

import base64
import logging
import shutil

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _as_choice,
    _canonical_hash,
    _parse_time,
    _read_json,
    _reason_text,
    _slug,
    _utc_now,
    _write_json,
)

# 导出包中按二进制处理的文件后缀（头像等），base64 编码存储。
_BINARY_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".tiff",
    ".tif",
    ".pdf",
    ".mp3",
    ".mp4",
    ".wav",
    ".ogg",
    ".bin",
    ".zip",
}



class ArchiveServiceMixin:
    def list_archived_people(self) -> list[dict[str, object]]:
        with self._lock:
            result = []
            for path in sorted(self.archive_dir.glob("*/person.json")):
                raw = _read_json(path, {})
                if isinstance(raw, dict):
                    conversation_sources = _read_json(path.parent / "conversation_sources.json", [])
                    conversation_messages = _read_json(path.parent / "conversation_messages.json", [])
                    conversation_versions = _read_json(path.parent / "conversation_versions.json", [])
                    result.append(
                        {
                            "person_id": raw.get("person_id"),
                            "name": raw.get("name"),
                            "archived_at": raw.get("archived_at"),
                            "avatar": raw.get("avatar", ""),
                            "source_count": len(conversation_sources) if isinstance(conversation_sources, list) else 0,
                            "message_count": len(conversation_messages) if isinstance(conversation_messages, list) else 0,
                            "version_count": len(conversation_versions) if isinstance(conversation_versions, list) else 0,
                        }
                    )
            return result

    def restore_person(self, person_id: str) -> dict[str, object]:
        with self._lock:
            source = self._archived_person_dir(person_id)
            target = self._person_dir(person_id)
            if not (source / "person.json").exists():
                raise ProductError("归档人物不存在。")
            if target.exists():
                raise ProductError("人物库中已经存在同编号人物。")
            person = dict(_read_json(source / "person.json", {}))
            person.pop("archived_at", None)
            _write_json(source / "person.json", person)
            # Windows 下 os.rename 跨目录可能因文件句柄/占用抛 PermissionError；
            # shutil.move 在 Windows 上会先复制再删除，规避源目录被占用导致的失败。
            shutil.move(str(source), str(target))
            try:
                self.person_repo.upsert(person)
            except Exception:
                logging.getLogger("pcfm").warning(
                    "restore_person sqlite sync failed person_id=%s", person_id, exc_info=True
                )
            return self.get_person(person_id)

    def permanently_delete_archived_person(
        self, person_id: str, *, expected_name: str
    ) -> None:
        with self._lock:
            if (self._person_dir(person_id) / "person.json").exists():
                raise ProductError("只能永久删除已经在归档中的人物。")
            target = self._archived_person_dir(person_id)
            if not (target / "person.json").exists():
                raise ProductError("归档人物不存在。")
            archived = dict(_read_json(target / "person.json", {}))
            if str(expected_name).strip() != str(archived.get("name", "")):
                raise ProductError("永久删除前必须输入完全一致的人物名称。")
            shutil.rmtree(target)
            self.person_repo.delete(person_id)

    # ---------- ordinary and advanced data import ----------

    def export_person(self, person_id: str) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            directory = self._person_dir(person_id)
            files: dict[str, str] = {}
            binary_files: list[str] = []
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(directory).as_posix()
                    if path.suffix.lower() in _BINARY_SUFFIXES:
                        files[relative] = base64.b64encode(path.read_bytes()).decode("ascii")
                        binary_files.append(relative)
                    else:
                        files[relative] = path.read_text(encoding="utf-8")
            return {
                "format": PRODUCT_FORMAT,
                "exported_at": _utc_now(),
                "person_id": person_id,
                "files": files,
                "binary_files": binary_files,
                "notice": "导出包包含此人物的本地完整性密钥，请像备份文件一样保管。",
            }

    def import_product_export(self, payload: Mapping[str, object], *, replace: bool = False) -> dict[str, object]:
        with self._lock:
            if payload.get("format") != PRODUCT_FORMAT or not isinstance(payload.get("files"), Mapping):
                raise ProductError("不是受支持的 PCFM 人物导出文件。")
            person_id = str(payload.get("person_id", ""))
            directory = self._person_dir(person_id)
            backup = self.people_dir / f"_restore-{uuid.uuid4().hex}"
            if directory.exists():
                if not replace:
                    raise ProductError("同编号人物已经存在；确认覆盖后才能加载。")
                directory.rename(backup)
            directory.mkdir(parents=True)
            binary_files = {str(value) for value in payload.get("binary_files", []) or []}
            try:
                for relative, text in dict(payload["files"]).items():
                    target = (directory / str(relative)).resolve()
                    if directory not in target.parents:
                        raise ProductError("导出文件包含不安全的路径。")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if str(relative) in binary_files:
                        target.write_bytes(base64.b64decode(str(text)))
                    else:
                        target.write_text(str(text), encoding="utf-8")
                person = self._require_person(person_id)
                if person.get("format") != PRODUCT_FORMAT:
                    raise ProductError("人物文件版本不受支持。")
                authority = self._authority(person)
                for version in self._versions(person_id):
                    def local_path(value: object) -> Path:
                        resolved = (directory / str(value)).resolve()
                        if directory not in resolved.parents:
                            raise ProductError("备份中的模型路径不安全。")
                        return resolved

                    bundle = load_bundle(local_path(version["model_path"]))
                    ledger = load_event_ledger_jsonl(
                        local_path(version["training_ledger_path"]),
                        authority,
                    )
                    target_records = ledger.records_for_person(person_id)
                    target_ids = tuple(
                        record.event_id
                        for record in sorted(
                            target_records,
                            key=lambda record: record.event_id,
                        )
                    )
                    if (
                        target_ids != bundle.manifest.person_event_ids
                        or EventLedger.snapshot_hash(target_records)
                        != bundle.manifest.person_data_hash
                    ):
                        raise ProductError("备份中的模型与训练历史不一致。")
                    for ledger_key in (
                        "applicability_ledger_path",
                        "validation_ledger_path",
                    ):
                        if version.get(ledger_key):
                            load_event_ledger_jsonl(
                                local_path(version[ledger_key]),
                                authority,
                            )
                result = self.get_person(person_id)
                if backup.exists():
                    shutil.rmtree(backup)
                return result
            except Exception:
                if directory.exists():
                    shutil.rmtree(directory)
                if backup.exists():
                    backup.rename(directory)
                raise
