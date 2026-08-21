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
from ..response_prediction import TRAINABLE_AUTHENTICITY



class SourcesMixin:
    def add_text_source(
        self,
        person_id: str,
        *,
        title: str,
        text: str,
        speaker: str,
        source_date: str,
        dataset_role: str,
        source_type: str = "pasted_text",
        source_url: str = "",
        filename: str = "",
        source_format: str = "text",
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        original_language: str = "",
        translation_of: str = "",
        speaker_scope: str = "single_speaker_entire_document",
        entity_aliases: Sequence[str] = (),
    ) -> dict[str, object]:
        self.profile(person_id)
        clean = str(text).strip()
        if not clean:
            raise ConversationError("资料没有可用文字。")
        if len(clean.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ConversationError("单份资料不能超过 25 MB。")
        if dataset_role not in SOURCE_ROLES:
            raise ConversationError("资料角色无效。")
        if speaker_scope not in {
            "single_speaker_entire_document",
            "mixed_speakers",
            "candidate_span_confirmed",
        }:
            raise ConversationError("材料说话人范围无效。")
        sources = self._list(person_id, "conversation_sources.json")
        digest = _text_hash(clean)
        if any(item.get("content_hash") == digest for item in sources):
            raise ConversationError("这份资料与已有资料重复。")
        near_duplicate = None
        for existing in sources:
            if _similarity(clean[:3000], str(existing.get("text", ""))[:3000]) >= 0.96:
                near_duplicate = existing.get("source_id")
                break
        item = {
            "schema_version": SCHEMA_VERSION,
            "source_id": f"source-{uuid.uuid4().hex[:12]}",
            "person_id": person_id,
            "title": str(title).strip() or filename or "未命名资料",
            "source_type": source_type,
            "format": source_format,
            "source_url": source_url,
            "filename": filename,
            "speaker": str(speaker).strip(),
            "speaker_scope": speaker_scope,
            "source_date": str(source_date).strip(),
            "dataset_role": dataset_role,
            "content_authenticity": str(content_authenticity).strip() or "unverified_material",
            "source_locator": str(source_locator).strip(),
            "source_context": str(source_context).strip(),
            "original_language": str(original_language).strip(),
            "translation_of": str(translation_of).strip(),
            "entity_aliases": sorted(
                {str(value).strip() for value in entity_aliases if str(value).strip()}
            ),
            "role_history": [],
            "review_status": "pending",
            "text": clean,
            "text_preview": clean[:240],
            "content_hash": digest,
            "near_duplicate_of": near_duplicate,
            "qas": _extract_qa(clean),
            "segments": _segments(clean),
            "created_at": _utc_now(),
            "reviewed_at": None,
        }
        item["response_events"] = response_events_from_source(item)
        sources.append(item)
        _write_json(self._path(person_id, "conversation_sources.json"), sources)
        return self._source_public(item)

    def add_file_source(
        self,
        person_id: str,
        *,
        filename: str,
        content_base64: str,
        speaker: str,
        source_date: str,
        dataset_role: str,
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        original_language: str = "",
        translation_of: str = "",
        speaker_scope: str = "single_speaker_entire_document",
    ) -> dict[str, object]:
        try:
            raw = base64.b64decode(content_base64, validate=True)
        except Exception as error:
            raise ConversationError("文件内容不是有效的 Base64。") from error
        if len(raw) > MAX_SOURCE_BYTES:
            raise ConversationError("单个文件不能超过 25 MB。")
        extension = Path(filename).suffix.casefold()
        if extension in {".txt", ".md", ".markdown", ".srt", ".vtt"}:
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError as error:
                raise ConversationError("文本文件必须使用 UTF-8 编码。") from error
            source_format = (
                "markdown"
                if extension in {".md", ".markdown"}
                else ("subtitle" if extension in {".srt", ".vtt"} else "txt")
            )
        elif extension == ".json":
            try:
                text = _structured_rows_text(json.loads(raw.decode("utf-8-sig")))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ConversationError("JSON 文件必须是 UTF-8 且结构有效。") from error
            source_format = "json"
        elif extension == ".csv":
            try:
                decoded = raw.decode("utf-8-sig")
                rows = list(csv.DictReader(StringIO(decoded)))
            except (UnicodeDecodeError, csv.Error) as error:
                raise ConversationError("CSV 文件必须是 UTF-8 且结构有效。") from error
            if not rows:
                raise ConversationError("CSV 文件没有可用数据行。")
            text = _structured_rows_text(rows)
            source_format = "csv"
        elif extension in {".html", ".htm"}:
            try:
                text, _title = _extract_html(raw.decode("utf-8-sig"))
            except UnicodeDecodeError as error:
                raise ConversationError("HTML 文件必须使用 UTF-8 编码。") from error
            source_format = "html"
        elif extension == ".pdf":
            text = _extract_pdf(raw)
            source_format = "pdf"
        else:
            raise ConversationError("当前支持 TXT、Markdown、HTML、字幕、JSON、CSV 和带文字层的 PDF；音视频转写尚未配置。")
        return self.add_text_source(
            person_id,
            title=filename,
            text=text,
            speaker=speaker,
            source_date=source_date,
            dataset_role=dataset_role,
            source_type="uploaded_file",
            filename=filename,
            source_format=source_format,
            content_authenticity=content_authenticity,
            source_locator=source_locator or filename,
            source_context=source_context,
            original_language=original_language,
            translation_of=translation_of,
            speaker_scope=speaker_scope,
        )

    def add_url_source(
        self,
        person_id: str,
        *,
        url: str,
        speaker: str,
        source_date: str,
        dataset_role: str,
        content_authenticity: str = "unverified_material",
        source_locator: str = "",
        source_context: str = "",
        original_language: str = "",
        translation_of: str = "",
        speaker_scope: str = "single_speaker_entire_document",
    ) -> dict[str, object]:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConversationError("网页地址必须是有效的 HTTP 或 HTTPS 地址。")
        host = parsed.hostname.casefold()
        if host in {"localhost", "localhost.localdomain"}:
            raise ConversationError("不能采集本机内部地址。")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address and (address.is_private or address.is_loopback or address.is_link_local):
            raise ConversationError("不能采集局域网或本机内部地址。")
        request = Request(
            parsed.geturl(),
            headers={"User-Agent": "PCFM-Local-MVP/1.0 (+evidence collection)"},
        )
        try:
            with urlopen(request, timeout=15) as response:
                raw = response.read(MAX_SOURCE_BYTES + 1)
                content_type = str(response.headers.get("Content-Type", ""))
        except Exception as error:
            raise ConversationError("网页无法访问或读取超时。") from error
        if len(raw) > MAX_SOURCE_BYTES:
            raise ConversationError("网页内容超过 25 MB。")
        if "pdf" in content_type.casefold() or parsed.path.casefold().endswith(".pdf"):
            text = _extract_pdf(raw)
            title = Path(parsed.path).name or parsed.netloc
            source_format = "pdf"
        else:
            decoded = _decode_web_bytes(raw, content_type)
            text, extracted_title = _extract_html(decoded)
            title = extracted_title or parsed.netloc
            source_format = "webpage"
        if len(text) > MAX_WEBPAGE_TEXT_CHARS:
            text = text[:MAX_WEBPAGE_TEXT_CHARS]
            title = str(title) + "（正文已截断到前 %d 字）" % MAX_WEBPAGE_TEXT_CHARS
        return self.add_text_source(
            person_id,
            title=title,
            text=text,
            speaker=speaker,
            source_date=source_date,
            dataset_role=dataset_role,
            source_type="web_url",
            source_url=parsed.geturl(),
            source_format=source_format,
            content_authenticity=content_authenticity,
            source_locator=source_locator,
            source_context=source_context,
            original_language=original_language,
            translation_of=translation_of,
            speaker_scope=speaker_scope,
        )

    def migrate_evidence_contract(self, person_id: str) -> dict[str, object]:
        """Downgrade legacy/self-disclaimed summaries and invalidate dependent models."""
        sources = self._list(person_id, "conversation_sources.json")
        versions = self._list(person_id, "conversation_versions.json")
        state = self._state(person_id)
        changed = False
        invalid_source_ids: set[str] = set()
        disclaimer_markers = (
            "editorial summary",
            "not the person's verbatim response",
            "not steve jobs' verbatim",
            "不是 steve jobs 的中文原话",
            "准确中文摘要",
        )
        for source in sources:
            text = str(source.get("text", "")).casefold()
            authenticity = str(source.get("content_authenticity", ""))
            legacy_invalid = not authenticity or any(marker in text for marker in disclaimer_markers)
            if not legacy_invalid:
                continue
            source_id = str(source.get("source_id", ""))
            invalid_source_ids.add(source_id)
            previous_role = str(source.get("dataset_role", "reference_only"))
            if previous_role != "reference_only":
                source.setdefault("role_history", []).append(
                    {
                        "from": previous_role,
                        "to": "reference_only",
                        "reason": "evidence_contract_v2_migration",
                        "changed_at": _utc_now(),
                    }
                )
            source["dataset_role"] = "reference_only"
            source["content_authenticity"] = "editorial_summary" if legacy_invalid else authenticity
            source["evidence_contract_status"] = "unverified_candidate"
            source["migration_notice"] = "摘要、生成内容或缺少逐字来源位置的材料不能作为本人回答训练标签。"
            source["response_events"] = response_events_from_source(source)
            if source.get("review_status") == "confirmed":
                source["response_events"] = review_response_events(
                    source,
                    str(self._person(person_id)["name"]),
                    [str(value) for value in self.profile(person_id).get("aliases", [])],
                )
            changed = True
        invalid_versions: set[int] = set()
        if invalid_source_ids:
            for version in versions:
                if invalid_source_ids.intersection(map(str, version.get("source_ids", []))):
                    version["validation_status"] = "invalidated_evidence_contract"
                    version["invalidation_reason"] = "source_not_verbatim_or_traceable"
                    invalid_versions.add(int(version["version"]))
                    changed = True
            if state.get("active_version") in invalid_versions:
                state["active_version"] = None
                state["evidence_contract_migration"] = {
                    "at": _utc_now(),
                    "invalidated_versions": sorted(invalid_versions),
                }
        if changed:
            _write_json(self._path(person_id, "conversation_sources.json"), sources)
            _write_json(self._path(person_id, "conversation_versions.json"), versions)
            _write_json(self._path(person_id, "conversation_state.json"), state)
        return {"changed": changed, "invalidated_versions": sorted(invalid_versions)}

    @staticmethod
    def _source_public(item: Mapping[str, object]) -> dict[str, object]:
        return {
            key: copy.deepcopy(value)
            for key, value in item.items()
            if key not in {"text", "segments"}
        }

    def sources(self, person_id: str) -> list[dict[str, object]]:
        return [self._source_public(item) for item in self._list(person_id, "conversation_sources.json")]

    def merge_source_entity_aliases(
        self, person_id: str, source_id: str, aliases: Sequence[str]
    ) -> bool:
        """Add reviewed retrieval aliases and version the affected person model."""
        sources = self._list(person_id, "conversation_sources.json")
        try:
            source = next(item for item in sources if item["source_id"] == source_id)
        except StopIteration as error:
            raise ConversationError("Source does not exist.") from error
        merged = sorted(
            {
                *(str(value).strip() for value in source.get("entity_aliases", [])),
                *(str(value).strip() for value in aliases),
            }
            - {""}
        )
        if merged == list(source.get("entity_aliases", [])):
            return False
        source["entity_aliases"] = merged
        _write_json(self._path(person_id, "conversation_sources.json"), sources)
        state = self._state(person_id)
        active = state.get("active_version")
        active_sources = self._version_source_ids(person_id)
        if active is not None and source_id in active_sources:
            self._create_version(
                person_id,
                source_ids=active_sources,
                reason="reviewed entity aliases added",
                validation_status=(
                    "exploratory_source_integrity_passed_accuracy_not_assessed"
                ),
                parent_version=int(active),
                update_style=False,
            )
        return True

    def review_source(self, person_id: str, source_id: str, decision: str) -> dict[str, object]:
        if decision not in {"confirmed", "rejected"}:
            raise ConversationError("审核结果必须是 confirmed 或 rejected。")
        sources = self._list(person_id, "conversation_sources.json")
        try:
            item = next(value for value in sources if value["source_id"] == source_id)
        except StopIteration as error:
            raise ConversationError("资料不存在。") from error
        if decision == "confirmed" and item.get("near_duplicate_of"):
            raise ConversationError("近重复资料需要先处理重复关系，不能直接确认。")
        item["review_status"] = decision
        item["reviewed_at"] = _utc_now()
        if decision == "confirmed":
            person = self._person(person_id)
            profile = self.profile(person_id)
            item["response_events"] = review_response_events(
                item,
                str(person["name"]),
                [str(value) for value in profile.get("aliases", [])],
            )
            has_confirmed_event = any(
                event.get("label_status") == "confirmed_response_weak_semantic_labels"
                for event in item["response_events"]
            )
            # 训练资格改为「形式资格」（逐字/可溯源/说话人），而非「确定性推导是否已产出回应事件」：
            # 本人逐字且可溯源的材料即使暂时没被自动切出问答，也应保留 model_source，
            # 等待「一件事」提取；只有形式不合格才降级为参考（材料三路分流 §7）。
            allowed_speakers = {
                str(person["name"]).casefold(),
                *(str(value).casefold() for value in profile.get("aliases", [])),
            }
            form_eligible = (
                str(item.get("content_authenticity", "")) in TRAINABLE_AUTHENTICITY
                and bool(
                    str(item.get("source_url", "")).strip()
                    or str(item.get("filename", "")).strip()
                )
                and bool(str(item.get("source_locator", "")).strip())
                and str(item.get("speaker", "")).casefold() in allowed_speakers
            )
            if (
                not has_confirmed_event
                and not form_eligible
                and item.get("speaker_scope") != "mixed_speakers"
            ):
                previous_role = str(item.get("dataset_role", "reference_only"))
                if previous_role != "reference_only":
                    item.setdefault("role_history", []).append(
                        {
                            "from": previous_role,
                            "to": "reference_only",
                            "reason": "no_verified_direct_response_events",
                            "changed_at": _utc_now(),
                        }
                    )
                item["dataset_role"] = "reference_only"
                for event in item["response_events"]:
                    event["data_role"] = "external_reality_comparison"
        _write_json(self._path(person_id, "conversation_sources.json"), sources)
        if decision == "confirmed" and item["dataset_role"] == "model_source":
            active = self._version_source_ids(person_id)
            proposed = [*active, source_id]
            has_episode_boundary = bool(item.get("qas")) or any(
                frame.get("review_status") == "confirmed"
                for frame in item.get("reviewed_event_frames_v4", [])
                if isinstance(frame, Mapping)
            )
            if (
                source_id not in active
                and item.get("speaker_scope") != "mixed_speakers"
                and item.get("content_authenticity")
                in {"verbatim_transcript", "verified_quote", "first_party_public_statement"}
                and has_episode_boundary
            ):
                state = self._state(person_id)
                self._create_version(
                    person_id,
                    source_ids=proposed,
                    reason="confirmed source added",
                    validation_status="exploratory_source_integrity_passed_accuracy_not_assessed",
                    parent_version=state.get("active_version"),
                )
            elif not has_episode_boundary:
                item["model_ingestion_status"] = (
                    "awaiting_verbatim_response_episode_extraction"
                )
                _write_json(
                    self._path(person_id, "conversation_sources.json"), sources
                )
        return self._source_public(item)

    def _source_records(self, person_id: str, source_ids: Sequence[str]) -> list[dict[str, object]]:
        allowed = set(map(str, source_ids))
        sources = self._list(person_id, "conversation_sources.json")
        changed = False
        person = self._person(person_id)
        profile = self.profile(person_id)
        for source in sources:
            if "response_events" not in source or any(
                not isinstance(event.get("event_atom"), Mapping)
                for event in source.get("response_events", [])
            ):
                source["response_events"] = response_events_from_source(source)
                if source.get("review_status") == "confirmed":
                    source["response_events"] = review_response_events(
                        source,
                        str(person["name"]),
                        [str(value) for value in profile.get("aliases", [])],
                    )
                changed = True
        if changed:
            _write_json(self._path(person_id, "conversation_sources.json"), sources)
        return [
            item
            for item in sources
            if item.get("review_status") == "confirmed" and item.get("source_id") in allowed
        ]
