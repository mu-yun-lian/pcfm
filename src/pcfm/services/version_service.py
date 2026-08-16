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



class VersionServiceMixin:
    def rollback_conversation_version(
        self, person_id: str, version_number: int
    ) -> dict[str, object]:
        with self._person_lock(person_id):
            self._require_person(person_id)
            result = self._conversation_call(
                self.conversation.rollback_version, person_id, version_number
            )
            self._sync_versions_to_sqlite(person_id)
            return result
