from __future__ import annotations

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



class PersonServiceMixin:
    def _person_dir(self, person_id: str) -> Path:
        if not person_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in person_id):
            raise ProductError("人物编号无效。")
        path = (self.people_dir / person_id).resolve()
        if path.parent != self.people_dir:
            raise ProductError("人物编号无效。")
        return path

    def _person_path(self, person_id: str) -> Path:
        return self._person_dir(person_id) / "person.json"

    def _archived_person_dir(self, person_id: str) -> Path:
        if not person_id or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in person_id):
            raise ProductError("人物编号无效。")
        path = (self.archive_dir / person_id).resolve()
        if path.parent != self.archive_dir:
            raise ProductError("人物编号无效。")
        return path

    def _require_person(self, person_id: str) -> dict[str, object]:
        raw = _read_json(self._person_path(person_id))
        if not isinstance(raw, dict):
            raise ProductError("人物文件损坏。")
        return dict(raw)

    def _person_lock(self, person_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._person_locks.setdefault(person_id, threading.RLock())

    def _history(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "history.json", [])
        if not isinstance(raw, list):
            raise ProductError("历史数据文件损坏。")
        return [dict(item) for item in raw]

    def _predictions(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "predictions.json", [])
        if not isinstance(raw, list):
            raise ProductError("预测记录文件损坏。")
        return [dict(item) for item in raw]

    def _versions(self, person_id: str) -> list[dict[str, object]]:
        raw = _read_json(self._person_dir(person_id) / "versions.json", [])
        if not isinstance(raw, list):
            raise ProductError("模型版本文件损坏。")
        return [dict(item) for item in raw]

    def _authority(self, person: Mapping[str, object]) -> VerificationAuthority:
        try:
            secret = bytes.fromhex(str(person["local_verifier_secret"]))
        except (KeyError, ValueError) as error:
            raise ProductError("人物的本地完整性密钥损坏。") from error
        return VerificationAuthority({VERIFIER_ID: secret})

    def list_people(self) -> list[dict[str, object]]:
        with self._lock:
            result = []
            for path in sorted(self.people_dir.glob("*/person.json")):
                dir_id = path.parent.name
                try:
                    person = dict(_read_json(path))
                    person_id = str(person["person_id"])
                    history = self._history(person_id)
                    versions = self._versions(person_id)
                    conversation = self._conversation_call(
                        self.conversation.summary, person_id, light=True
                    )
                    messages = list(conversation["messages"])
                    result.append(
                        {
                            "person_id": person["person_id"],
                            "name": person["name"],
                            "description": person.get("description", ""),
                            "avatar": person.get("avatar", ""),
                            "identity_note": person.get("identity_note", ""),
                            "focus_domain": person.get("focus_domain", ""),
                            "collection": person.get(
                                "collection", conversation["profile"].get("collection", {})
                            ),
                            "feature_names": person["feature_names"],
                            "is_example": bool(person.get("is_example")),
                            "is_demo": bool(person.get("is_demo")),
                            "sample_count": len(history),
                            "model_version_count": len(versions),
                            "model_status": versions[-1]["validation_status"] if versions else "not_trained",
                            "model_kind": "behavior_baseline_logistic",
                            "cognitive_status": self.cognitive.summary(str(person["person_id"]))["status"],
                            "conversation_status": conversation["status"],
                            "conversation_status_text": conversation["status_text"],
                            "conversation_version": conversation["active_version"],
                            "source_count": conversation["source_counts"]["confirmed"],
                            "message_count": len(messages),
                            "last_message": messages[-1]["text"] if messages else "",
                            "language": conversation["profile"]["language"],
                            "style_profile_id": conversation["profile"]["style_profile_id"],
                        }
                    )
                except (ProductError, ConversationError, PcfmDataError, CognitiveWorkbenchError):
                    result.append({
                        "person_id": dir_id,
                        "name": "（数据损坏）",
                        "description": "",
                        "avatar": "",
                        "identity_note": "",
                        "focus_domain": "",
                        "collection": {"mode": "user_provided", "status": "corrupted", "message": "人物数据文件损坏，已隔离；其他人物不受影响。"},
                        "feature_names": [],
                        "is_example": False,
                        "is_demo": False,
                        "sample_count": 0,
                        "model_version_count": 0,
                        "model_status": "not_trained",
                        "model_kind": "behavior_baseline_logistic",
                        "cognitive_status": "not_configured",
                        "conversation_status": "corrupted",
                        "conversation_status_text": "数据损坏，已隔离",
                        "conversation_version": None,
                        "source_count": 0,
                        "message_count": 0,
                        "last_message": "",
                        "language": "zh",
                        "style_profile_id": "neutral_v1",
                        "health": "corrupted",
                    })
            return result

    def create_person(
        self,
        *,
        name: str,
        description: str,
        feature_names: Sequence[str],
        person_id: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            clean_name = str(name).strip()
            names = tuple(str(item).strip() for item in feature_names if str(item).strip())
            if not clean_name:
                raise ProductError("请填写人物名称。")
            if not names or len(set(names)) != len(names):
                raise ProductError("至少需要一个不重复的数值影响项名称。")
            identifier = _slug(person_id or f"person-{uuid.uuid4().hex[:12]}")
            directory = self._person_dir(identifier)
            if directory.exists():
                raise ProductError("这个人物编号已经存在。")
            directory.mkdir(parents=True)
            person = {
                "format": PRODUCT_FORMAT,
                "person_id": identifier,
                "name": clean_name,
                "description": str(description).strip(),
                "feature_names": list(names),
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "is_example": False,
                "local_verifier_secret": secrets.token_hex(32),
            }
            _write_json(directory / "person.json", person)
            _write_json(directory / "history.json", [])
            _write_json(directory / "predictions.json", [])
            _write_json(directory / "versions.json", [])
            return self.get_person(identifier)

    def update_person(self, person_id: str, changes: Mapping[str, object]) -> dict[str, object]:
        with self._person_lock(person_id):
            person = self._require_person(person_id)
            if "name" in changes:
                name = str(changes["name"]).strip()
                if not name:
                    raise ProductError("人物名称不能为空。")
                person["name"] = name
            if "description" in changes:
                person["description"] = str(changes["description"]).strip()
            for key in ("identity_note", "focus_domain", "avatar", "notes"):
                if key in changes:
                    person[key] = str(changes[key]).strip()
            if "source_mode" in changes:
                source_mode = str(changes["source_mode"]).strip()
                if source_mode not in {"user_provided", "system_search"}:
                    raise ProductError("请选择系统自动搜索或用户自行提供资料。")
                if source_mode == "system_search" and self.public_search is None:
                    raise ProductError("系统自动搜索公开资料暂未配置，请选择自行提供原始资料。")
                person["collection"] = {
                    "mode": source_mode,
                    "status": "search_ready" if source_mode == "system_search" else "awaiting_user_materials",
                    "message": (
                        "搜索服务已配置；结果只会进入待审核候选资料。"
                        if source_mode == "system_search"
                        else "等待用户提供原始资料。"
                    ),
                }
            if "feature_names" in changes:
                if self._history(person_id) or self._versions(person_id):
                    raise ProductError("已有数据或模型时不能修改数值影响项；请新建人物。")
                names = tuple(str(item).strip() for item in changes["feature_names"] if str(item).strip())
                if not names or len(set(names)) != len(names):
                    raise ProductError("数值影响项不能为空或重复。")
                person["feature_names"] = list(names)
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            self.person_repo.upsert(person)
            if any(
                key in changes
                for key in (
                    "aliases", "language", "time_start", "time_end",
                    "source_mode", "identity_note", "focus_domain",
                    "generation_params",
                )
            ):
                profile = self._conversation_call(
                    self.conversation.profile, person_id
                )
                self._conversation_call(
                    self.conversation.configure,
                    person_id,
                    aliases=changes.get("aliases", profile.get("aliases", [])),
                    language=str(changes.get("language", profile.get("language", "zh"))),
                    time_start=str(
                        changes.get(
                            "time_start", profile.get("time_scope", {}).get("start", "")
                        )
                    ),
                    time_end=str(
                        changes.get(
                            "time_end", profile.get("time_scope", {}).get("end", "")
                        )
                    ),
                    style_profile_id=str(profile.get("style_profile_id", "neutral_v1")),
                    source_mode=str(
                        changes.get(
                            "source_mode",
                            profile.get("collection", {}).get("mode", "user_provided"),
                        )
                    ),
                    identity_note=str(
                        changes.get("identity_note", profile.get("identity_note", ""))
                    ),
                    focus_domain=str(
                        changes.get("focus_domain", profile.get("focus_domain", ""))
                    ),
                    generation_params=dict(
                        changes.get(
                            "generation_params",
                            profile.get("generation_params", {}),
                        )
                    ),
                )
            return self.get_person(person_id)

    def delete_person(self, person_id: str) -> None:
        with self._person_lock(person_id):
            directory = self._person_dir(person_id)
            if not (directory / "person.json").exists():
                raise ProductError("人物不存在。")
            target = self._archived_person_dir(person_id)
            if target.exists():
                raise ProductError("归档中已经存在同编号人物。")
            person = self._require_person(person_id)
            person["archived_at"] = _utc_now()
            _write_json(directory / "person.json", person)
            self._ensure_person_in_sqlite(person_id)
            self.person_repo.mark_archived(person_id, person["archived_at"])
            directory.rename(target)

    def set_avatar(self, person_id: str, data_url: str) -> dict[str, object]:
        """保存人物头像（本地文件），data_url 形如 data:image/png;base64,...；传空则移除。"""
        with self._person_lock(person_id):
            person = self._require_person(person_id)
            if not str(data_url).strip():
                for old_ext in ("png", "jpg", "jpeg", "webp", "gif"):
                    old = self._person_dir(person_id) / f"avatar.{old_ext}"
                    if old.exists():
                        old.unlink(missing_ok=True)
                person["avatar"] = ""
                person["updated_at"] = _utc_now()
                _write_json(self._person_path(person_id), person)
                self.person_repo.upsert(person)
                return self.get_person(person_id)
            header = "image/png"
            encoded = str(data_url).strip()
            if "," in encoded:
                header, encoded = encoded.split(",", 1)
                header = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
            try:
                raw = base64.b64decode(encoded)
            except Exception as error:
                raise ProductError("头像数据无效。") from error
            if len(raw) > 2 * 1024 * 1024:
                raise ProductError("头像图片不能超过 2 MB。")
            ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(header, "png")
            for old_ext in ("png", "jpg", "jpeg", "webp", "gif"):
                old = self._person_dir(person_id) / f"avatar.{old_ext}"
                if old.exists() and old_ext != ext:
                    old.unlink(missing_ok=True)
            path = self._person_dir(person_id) / f"avatar.{ext}"
            path.write_bytes(raw)
            person["avatar"] = f"/api/people/{person_id}/avatar"
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            self.person_repo.upsert(person)
            return self.get_person(person_id)

    def get_avatar(self, person_id: str) -> tuple[bytes, str]:
        """返回头像字节与 MIME；无头像时抛 ProductError。

        只读头像文件，不修改人物状态，故不加 _person_lock（与 REQ-02 的写锁语义区分）。
        """
        person = self._require_person(person_id)
        for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg"), ("jpeg", "image/jpeg"), ("webp", "image/webp"), ("gif", "image/gif")):
            path = self._person_dir(person_id) / f"avatar.{ext}"
            if path.exists():
                return path.read_bytes(), mime
        raise ProductError("该人物还没有头像。")

    def get_person(self, person_id: str) -> dict[str, object]:
        with self._person_lock(person_id):
            person = self._require_person(person_id)
            history = self._history(person_id)
            training, applicability, validation = self._partition_history(history)
            versions = self._versions(person_id)
            predictions = self._predictions(person_id)
            public_person = {key: value for key, value in person.items() if key != "local_verifier_secret"}
            public_person.update(
                {
                    "sample_count": len(history),
                    "partition_counts": {
                        "training": len(training),
                        "applicability": len(applicability),
                        "validation": len(validation),
                    },
                    "data_sufficiency": self._data_sufficiency(len(training), len(applicability), len(validation)),
                    "predictions": predictions,
                    "prediction_metrics": self._prediction_metrics(person_id),
                    "versions": [self._version_public(item) for item in versions],
                    "latest_model": self._version_public(versions[-1]) if versions else None,
                    "history_preview": history[-20:],
                    "advanced_evidence_count": len(list((self._person_dir(person_id) / "advanced_evidence").glob("*.summary.json"))) if (self._person_dir(person_id) / "advanced_evidence").exists() else 0,
                    "behavior_baseline": {
                        "display_name": "行为基线模型（Logistic）",
                        "status": self._version_public(versions[-1])["validation_status"] if versions else "not_trained",
                        "notice": "它只拟合结构化历史选择，不等同于人物认知模型。",
                    },
                    "cognitive": self.cognitive.summary(person_id),
                    "conversation": self._conversation_call(
                        self.conversation.summary, person_id
                    ),
                }
            )
            return public_person

    # ---------- evidence-constrained cognitive workbench ----------

    def csv_template(self, person_id: str) -> str:
        person = self._require_person(person_id)
        fields = ["observed_at", "scenario_id", "question", "option_a", "option_b", "choice", "domain", *person["feature_names"]]
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerow([_utc_now(), "example-001", "示例情境", "选项A", "选项B", "A", "structured_choice", *([0.0] * len(person["feature_names"]))])
        return stream.getvalue()

    # ---------- built-in demonstrator ----------
