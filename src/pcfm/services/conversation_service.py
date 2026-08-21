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



class ConversationServiceMixin:
    def create_conversation_person(
        self,
        *,
        name: str,
        aliases: Sequence[str] = (),
        language: str = "zh",
        description: str = "",
        time_start: str = "",
        time_end: str = "",
        source_mode: str = "user_provided",
        identity_note: str = "",
        focus_domain: str = "",
        avatar: str = "",
        notes: str = "",
    ) -> dict[str, object]:
        with self._lock:
            if source_mode not in {"user_provided", "system_search"}:
                raise ProductError("请选择系统自动搜索或用户自行提供资料。")
            if source_mode == "system_search" and self.public_search is None:
                raise ProductError("系统自动搜索公开资料暂未配置，请选择自行提供原始资料。")
            person = self.create_person(
                name=name,
                description=description,
                feature_names=("evidence_overlap", "intercept"),
            )
            person_id = str(person["person_id"])
            raw_person = self._require_person(person_id)
            raw_person.update(
                {
                    "identity_note": str(identity_note).strip(),
                    "focus_domain": str(focus_domain).strip(),
                    "avatar": str(avatar).strip(),
                    "notes": str(notes).strip(),
                    "collection": {
                        "mode": source_mode,
                        "status": "search_ready" if source_mode == "system_search" else "awaiting_user_materials",
                        "message": (
                            "搜索服务已配置；结果只会进入待审核候选资料。"
                            if source_mode == "system_search"
                            else "等待用户提供原始资料。"
                        ),
                    },
                }
            )
            _write_json(self._person_path(person_id), raw_person)
            self.person_repo.upsert(raw_person)
            self._conversation_call(
                self.conversation.configure,
                person_id,
                aliases=aliases,
                language=language,
                time_start=time_start,
                time_end=time_end,
                source_mode=source_mode,
                identity_note=identity_note,
                focus_domain=focus_domain,
            )
            # 公开搜索不再在创建人物请求内同步执行; 由前端随后触发搜索任务
            return self.get_person(person_id)

    def conversation_summary(self, person_id: str, *, light: bool = False, full_messages: bool = False) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            return self._conversation_call(self.conversation.summary, person_id, light=light, full_messages=full_messages)

    def start_new_conversation(self, person_id: str) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.start_new_conversation, person_id
            )
            self._sync_sessions_to_sqlite(person_id)
            return result

    def list_sessions(self, person_id: str) -> list[dict[str, object]]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            return self._conversation_call(self.conversation.list_sessions, person_id)

    def create_session(self, person_id: str) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(self.conversation.create_session, person_id)
            self._sync_sessions_to_sqlite(person_id)
            return result

    def switch_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(self.conversation.switch_session, person_id, session_id)
            self._sync_sessions_to_sqlite(person_id)
            return result

    def rename_session(self, person_id: str, session_id: str, title: str) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(self.conversation.rename_session, person_id, session_id, title)
            self._sync_sessions_to_sqlite(person_id)
            return result

    def delete_session(self, person_id: str, session_id: str) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(self.conversation.delete_session, person_id, session_id)
            self.message_repo.delete_by_session(session_id)
            self._sync_sessions_to_sqlite(person_id)
            return result

    def send_conversation_message(
        self,
        person_id: str,
        text: str,
        *,
        reality_lookup_requested: bool = False,
        dialogue_model_ref: str = "",
        _cancel_event: object = None,
        _progress: object = None,
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.send_message,
                person_id,
                text,
                reality_lookup_requested=reality_lookup_requested,
                dialogue_model_ref=dialogue_model_ref,
                cancel_event=_cancel_event,
                progress=_progress,
            )
            self._sync_messages_to_sqlite(person_id)
            return result

    def find_conversation_reality_answer(
        self, person_id: str, message_id: str,
        _progress: object = None,
        _cancel_event: object = None,
    ) -> dict[str, object]:
        if _cancel_event is not None and _cancel_event.is_set():
            raise JobCancelled()
        if _progress:
            _progress(0.3, "matching", "正在本地匹配现实回答…")
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.find_reality_answer, person_id, message_id
            )
            if _progress:
                _progress(1.0, "done", "完成")
            self._sync_messages_to_sqlite(person_id)
            return result

    def create_optimization_candidate(
        self,
        person_id: str,
        message_id: str,
        *,
        allow_retry: bool = False,
        comparison_candidate_id: str = "",
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.create_optimization_candidate,
                person_id,
                message_id,
                allow_retry=allow_retry,
                comparison_candidate_id=comparison_candidate_id,
            )

    def review_optimization_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.review_optimization_candidate,
                person_id,
                candidate_id,
                decision,
            )
            self._sync_versions_to_sqlite(person_id)
            return result

    def review_optimization_style_candidate(
        self, person_id: str, candidate_id: str, decision: str
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.review_optimization_style_candidate,
                person_id,
                candidate_id,
                decision,
            )
            self._sync_versions_to_sqlite(person_id)
            return result

    def record_conversation_feedback(
        self, person_id: str, message_id: str, value: str
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.feedback, person_id, message_id, value
            )
            self._sync_messages_to_sqlite(person_id)
            return result

    # ---------- person and local persistence ----------
