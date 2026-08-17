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



class SourceServiceMixin:
    def collect_public_sources(self, person_id: str) -> dict[str, object]:
        """Collect public source candidates. Discovery never creates training truth."""
        with self._person_lock(person_id):
            person = self._require_person(person_id)
            profile = self._conversation_call(self.conversation.profile, person_id)
            provider_id = getattr(self.public_search, "provider_id", "configured-provider")
            if self.public_search is None:
                collection = {
                    "mode": "system_search",
                    "status": "search_service_not_configured",
                    "message": "公开资料搜索服务未配置，当前不会伪装成已经搜索。",
                }
            else:
                try:
                    results = self.public_search.search(
                        person_name=str(person["name"]),
                        identity_note=str(person.get("identity_note", "")),
                        language=str(profile.get("language", "zh")),
                        limit=8,
                    )
                except Exception as error:
                    collection = {
                        "mode": "system_search",
                        "status": "temporarily_unavailable",
                        "message": f"公开资料搜索暂时失败：{error}",
                        "provider": provider_id,
                    }
                    results = []
                else:
                    for result in results:
                        text = "\n\n".join(
                            value
                            for value in (
                                str(result.get("title", "")),
                                str(result.get("snippet", "")),
                            )
                            if value
                        )
                        try:
                            self.conversation.add_text_source(
                                person_id,
                                title=str(result.get("title", "公开资料候选")),
                                text=text or str(result.get("url", "")),
                                speaker="",
                                source_date=str(result.get("published_at", "")),
                                dataset_role="reference_only",
                                source_type="system_search_result",
                                source_url=str(result.get("url", "")),
                                source_format="search_result",
                                content_authenticity="search_snippet",
                                source_locator=f"search result rank {result.get('provider_rank', '')}",
                                source_context="自动搜索候选；尚未访问原文或确认说话人。",
                            )
                        except ConversationError:
                            continue
                    collection = {
                        "mode": "system_search",
                        "status": "candidates_found" if results else "no_candidates",
                        "message": (
                            f"找到 {len(results)} 条公开资料候选，均需核验原文后才能训练。"
                            if results
                            else "未找到可用的公开资料候选。"
                        ),
                        "provider": provider_id,
                        "candidate_count": len(results),
                    }
            person["collection"] = collection
            _write_json(self._person_path(person_id), person)
            profile["collection"] = collection
            _write_json(
                self._person_dir(person_id) / "conversation_profile.json", profile
            )
            return collection

    def add_conversation_text_source(
        self,
        person_id: str,
        *,
        title: str,
        text: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        source_url: str = "",
        original_language: str = "",
        translation_of: str = "",
        speaker_scope: str = "single_speaker_entire_document",
        entity_aliases: Sequence[str] = (),
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.add_text_source,
                person_id,
                title=title,
                text=text,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                source_url=source_url,
                original_language=original_language,
                translation_of=translation_of,
                speaker_scope=speaker_scope,
                entity_aliases=entity_aliases,
            )
            self._sync_sources_to_sqlite(person_id)
            return result

    def add_conversation_file_source(
        self,
        person_id: str,
        *,
        filename: str,
        content_base64: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        speaker_scope: str = "single_speaker_entire_document",
        _progress: object = None,
        _cancel_event: object = None,
    ) -> dict[str, object]:
        if _cancel_event is not None and _cancel_event.is_set():
            raise JobCancelled()
        if _progress:
            _progress(0.2, "parsing", "正在解析文件…")
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.add_file_source,
                person_id,
                filename=filename,
                content_base64=content_base64,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                speaker_scope=speaker_scope,
            )
            if _cancel_event is not None and _cancel_event.is_set():
                raise JobCancelled()
            self._sync_sources_to_sqlite(person_id)
            if _progress:
                _progress(1.0, "done", "完成")
            return result

    def add_conversation_url_source(
        self,
        person_id: str,
        *,
        url: str,
        speaker: str,
        source_date: str = "",
        dataset_role: str = "model_source",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        speaker_scope: str = "single_speaker_entire_document",
        _progress: object = None,
        _cancel_event: object = None,
    ) -> dict[str, object]:
        if _cancel_event is not None and _cancel_event.is_set():
            raise JobCancelled()
        if _progress:
            _progress(0.2, "fetching", "正在抓取网页…")
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.add_url_source,
                person_id,
                url=url,
                speaker=speaker,
                source_date=source_date,
                dataset_role=dataset_role,
                content_authenticity=content_authenticity,
                source_locator=source_locator,
                source_context=source_context,
                speaker_scope=speaker_scope,
            )
            if _cancel_event is not None and _cancel_event.is_set():
                raise JobCancelled()
            self._sync_sources_to_sqlite(person_id)
            if _progress:
                _progress(1.0, "done", "完成")
            return result

    def review_conversation_source(
        self, person_id: str, source_id: str, decision: str
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.review_source, person_id, source_id, decision
            )
            self._sync_sources_to_sqlite(person_id)
            return result
