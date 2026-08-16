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



class RealityLookupMixin:
    def find_reality_answer(self, person_id: str, message_id: str) -> dict[str, object]:
        messages, message = self._find_message(person_id, message_id)
        if message.get("role") != "assistant":
            raise ConversationError("只能为人物回答查找现实对照。")
        index = messages.index(message)
        if index == 0 or messages[index - 1].get("role") != "user":
            raise ConversationError("这条回答缺少对应的用户问题。")
        question = str(messages[index - 1]["text"])
        telemetry = self._telemetry(person_id)
        telemetry["reality_lookup_requests"] += 1
        telemetry["reality_local_search_calls"] += 1
        self._save_telemetry(person_id, telemetry)
        active_ids = set(self._version_source_ids(person_id))
        source_candidates = [
            item
            for item in self._list(person_id, "conversation_sources.json")
            if item.get("review_status") == "confirmed"
            and item.get("source_id") not in active_ids
            and item.get("dataset_role") in {"reference_only", "applicability_reference"}
        ]
        profile = self.profile(person_id)
        person = self._person(person_id)
        allowed_speakers = {
            str(person["name"]).casefold(),
            *(str(value).casefold() for value in profile.get("aliases", [])),
        }
        source_candidates = [
            item for item in source_candidates if str(item.get("speaker", "")).casefold() in allowed_speakers
        ]
        sources = self._source_records(
            person_id, [str(item["source_id"]) for item in source_candidates]
        )
        reality_candidates = self._reality_support_candidates(question, sources)
        if not reality_candidates:
            result = {
                "status": "not_found",
                "message_id": message_id,
                "notice": "未找到可核验的现实回答。",
                "online_search_status": "not_configured",
            }
            message["comparison"] = result
            message["reality_lookup_status"] = "not_found"
            self._save_active_messages(person_id, messages)
            return result
        best = reality_candidates[0]
        predicted = str(message.get("neutral_content") or message.get("text", ""))
        actual = str(best["answer"])
        overlap = sorted(_tokens(predicted) & _tokens(actual))[:12]
        comparison = {
            "schema_version": SCHEMA_VERSION,
            "comparison_id": f"comparison-{uuid.uuid4().hex[:12]}",
            "status": "candidate_found",
            "message_id": message_id,
            "person_id": person_id,
            "predicted_answer": predicted,
            "reality_answer": actual,
            "reality_question": best["question"],
            "source_id": best["source_id"],
            "source_title": best["source_title"],
            "source_url": best["source_url"],
            "source_date": best["source_date"],
            "speaker": best["speaker"],
            "source_locator": best["locator"],
            "question_similarity": best["score"],
            "similarity_label": "same_question" if float(best["score"]) >= 0.8 else "highly_similar",
            "context_consistency": "metadata_consistent_not_independently_verified",
            "agreements": overlap or ["没有足够稳定的词汇重合，需人工阅读"],
            "differences": ["现实回答与预测回答应由用户结合完整上下文判断"],
            "notice": "现实回答尚未自动进入人物模型。",
            "reality_candidates": reality_candidates,
            "selected_candidate_id": (
                reality_candidates[0]["comparison_candidate_id"]
                if len(reality_candidates) == 1
                else None
            ),
            "created_at": _utc_now(),
        }
        message["comparison"] = comparison
        message["reality_lookup_status"] = "candidate_found"
        self._save_active_messages(person_id, messages)
        return copy.deepcopy(comparison)
