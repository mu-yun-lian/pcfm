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



class ExtractionServiceMixin:
    def extract_conversation_response_candidates(
        self, person_id: str, source_id: str
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.extract_response_event_candidates,
                person_id,
                source_id,
            )

    def review_conversation_response_candidate(
        self,
        person_id: str,
        source_id: str,
        candidate_id: str,
        decision: str,
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._conversation_call(
                self.conversation.review_response_event_candidate,
                person_id,
                source_id,
                candidate_id,
                decision,
            )
