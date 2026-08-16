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



class JobServiceMixin:
    def get_job(self, job_id: str) -> dict[str, object] | None:
        job = self.job_runner.get(job_id)
        return asdict(job) if job else None

    def cancel_job(self, job_id: str) -> bool:
        return self.job_runner.cancel(job_id)
