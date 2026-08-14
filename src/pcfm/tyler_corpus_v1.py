from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from .tyler_source_v1 import (
    AuthoredProseUnit,
    DECISION_AXES,
    PERSON_ID,
    QuotedUnit,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    TylerSourceArtifact,
    TylerSourcePost,
    load_tyler_source_artifact,
    verify_tyler_source_artifact,
)


CORPUS_SCHEMA_VERSION = "tyler-corpus-v1"
MANIFEST_SCHEMA_VERSION = "tyler-corpus-input-manifest-v1"
PROTOCOL_VALIDATION_START = datetime(
    2025, 1, 1, tzinfo=timezone.utc
)
RETROSPECTIVE_DIAGNOSTIC_START = datetime(
    2026, 1, 1, tzinfo=timezone.utc
)
UNREGISTERED_FUTURE_START = datetime(
    2026, 8, 1, tzinfo=timezone.utc
)
ALLOWED_ROLES = (
    "protocol_validation",
    "retrospective_diagnostic",
    "training_discovery",
)


class TylerCorpusRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(
            "Tyler corpus refused: " + ", ".join(self.reasons)
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a SHA-256 digest"
        ) from error


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"invalid {label}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _temporal_role(published_at: str) -> str:
    published = _parse_timestamp(published_at, "published_at")
    if published < PROTOCOL_VALIDATION_START:
        return "training_discovery"
    if published < RETROSPECTIVE_DIAGNOSTIC_START:
        return "protocol_validation"
    if published < UNREGISTERED_FUTURE_START:
        return "retrospective_diagnostic"
    raise TylerCorpusRefusedError(("unregistered_future_evidence",))


@dataclass(frozen=True)
class CorpusSourceInput:
    artifact: TylerSourceArtifact
    raw_snapshot: str


@dataclass(frozen=True)
class CorpusSourceEntry:
    source_url: str
    collected_at: str
    raw_snapshot_sha256: str
    extraction_digest: str
    artifact_digest: str
    post_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_url": self.source_url,
            "collected_at": self.collected_at,
            "raw_snapshot_sha256": self.raw_snapshot_sha256,
            "extraction_digest": self.extraction_digest,
            "artifact_digest": self.artifact_digest,
            "post_count": self.post_count,
        }


@dataclass(frozen=True)
class CorpusRecord:
    source_artifact_digest: str
    evidence_role: str
    post: TylerSourcePost

    def to_dict(self) -> dict[str, object]:
        return {
            "source_artifact_digest": self.source_artifact_digest,
            "evidence_role": self.evidence_role,
            "post": self.post.to_dict(),
        }


@dataclass(frozen=True)
class TylerCorpusArtifact:
    schema_version: str
    person_id: str
    created_at: str
    expected_source_urls: tuple[str, ...]
    sources: tuple[CorpusSourceEntry, ...]
    records: tuple[CorpusRecord, ...]
    role_counts: dict[str, int]
    candidate_counts: dict[str, int]
    corpus_digest: str

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "person_id": self.person_id,
            "created_at": self.created_at,
            "expected_source_urls": list(self.expected_source_urls),
            "sources": [source.to_dict() for source in self.sources],
            "records": [record.to_dict() for record in self.records],
            "role_counts": dict(sorted(self.role_counts.items())),
            "candidate_counts": dict(
                sorted(self.candidate_counts.items())
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "corpus_digest": self.corpus_digest,
        }


def _source_entry(artifact: TylerSourceArtifact) -> CorpusSourceEntry:
    return CorpusSourceEntry(
        source_url=artifact.source_url,
        collected_at=artifact.collected_at,
        raw_snapshot_sha256=artifact.raw_snapshot_sha256,
        extraction_digest=artifact.extraction_digest,
        artifact_digest=artifact.artifact_digest,
        post_count=len(artifact.posts),
    )


def create_tyler_corpus(
    source_inputs: Sequence[CorpusSourceInput],
    *,
    expected_source_urls: Sequence[str],
    created_at: str,
) -> TylerCorpusArtifact:
    created = _parse_timestamp(created_at, "created_at")
    if not source_inputs or not expected_source_urls:
        raise TylerCorpusRefusedError(("empty_source_plan",))

    source_urls = [item.artifact.source_url for item in source_inputs]
    if len(set(source_urls)) != len(source_urls):
        raise TylerCorpusRefusedError(("duplicate_source_url",))
    expected = tuple(sorted(str(value) for value in expected_source_urls))
    if len(set(expected)) != len(expected):
        raise TylerCorpusRefusedError(("duplicate_expected_source_url",))
    if tuple(sorted(source_urls)) != expected:
        raise TylerCorpusRefusedError(("planned_source_set_mismatch",))

    sources: list[CorpusSourceEntry] = []
    records: list[CorpusRecord] = []
    seen_urls: set[str] = set()
    seen_post_ids: set[str] = set()
    seen_prose: set[str] = set()
    for item in source_inputs:
        try:
            verify_tyler_source_artifact(
                item.artifact,
                raw_snapshot=item.raw_snapshot,
            )
        except (ValueError, TylerCorpusRefusedError) as error:
            raise TylerCorpusRefusedError(
                ("source_replay_invalid",)
            ) from error
        if _parse_timestamp(
            item.artifact.collected_at, "collected_at"
        ) > created:
            raise TylerCorpusRefusedError(
                ("source_collected_after_corpus_creation",)
            )
        sources.append(_source_entry(item.artifact))
        for post in item.artifact.posts:
            if _parse_timestamp(post.published_at, "published_at") > created:
                raise TylerCorpusRefusedError(
                    ("publication_after_corpus_creation",)
                )
            if post.canonical_url in seen_urls:
                raise TylerCorpusRefusedError(("duplicate_post_url",))
            if post.post_id in seen_post_ids:
                raise TylerCorpusRefusedError(("duplicate_post_id",))
            if (
                post.normalized_prose_sha256 is not None
                and post.normalized_prose_sha256 in seen_prose
            ):
                raise TylerCorpusRefusedError(
                    ("duplicate_normalized_prose",)
                )
            seen_urls.add(post.canonical_url)
            seen_post_ids.add(post.post_id)
            if post.normalized_prose_sha256 is not None:
                seen_prose.add(post.normalized_prose_sha256)
            records.append(
                CorpusRecord(
                    source_artifact_digest=(
                        item.artifact.artifact_digest
                    ),
                    evidence_role=_temporal_role(post.published_at),
                    post=post,
                )
            )

    canonical_sources = tuple(
        sorted(sources, key=lambda source: source.source_url)
    )
    canonical_records = tuple(
        sorted(records, key=lambda record: record.post.canonical_url)
    )
    role_counts = {
        role: sum(
            record.evidence_role == role for record in canonical_records
        )
        for role in ALLOWED_ROLES
        if any(record.evidence_role == role for record in canonical_records)
    }
    statuses = sorted(
        {record.post.candidate_status for record in canonical_records}
    )
    candidate_counts = {
        status: sum(
            record.post.candidate_status == status
            for record in canonical_records
        )
        for status in statuses
    }
    canonical_created_at = created.isoformat().replace("+00:00", "Z")
    provisional = TylerCorpusArtifact(
        schema_version=CORPUS_SCHEMA_VERSION,
        person_id=PERSON_ID,
        created_at=canonical_created_at,
        expected_source_urls=expected,
        sources=canonical_sources,
        records=canonical_records,
        role_counts=role_counts,
        candidate_counts=candidate_counts,
        corpus_digest="",
    )
    corpus = TylerCorpusArtifact(
        **{
            **provisional.__dict__,
            "corpus_digest": _digest_json(provisional._unsigned_dict()),
        }
    )
    verify_tyler_corpus(corpus)
    return corpus


def verify_tyler_corpus(
    corpus: TylerCorpusArtifact,
    *,
    source_inputs: Sequence[CorpusSourceInput] | None = None,
) -> bool:
    if corpus.schema_version != CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CORPUS_SCHEMA_VERSION}"
        )
    if corpus.person_id != PERSON_ID:
        raise ValueError(f"person_id must be {PERSON_ID}")
    created = _parse_timestamp(corpus.created_at, "created_at")
    _require_digest(corpus.corpus_digest, "corpus_digest")
    if corpus.expected_source_urls != tuple(
        sorted(corpus.expected_source_urls)
    ):
        raise ValueError("expected source URLs are not canonical")
    if len(set(corpus.expected_source_urls)) != len(
        corpus.expected_source_urls
    ):
        raise ValueError("duplicate expected source URL")
    if tuple(source.source_url for source in corpus.sources) != (
        corpus.expected_source_urls
    ):
        raise ValueError("source set does not match expected URLs")
    if corpus.records != tuple(
        sorted(
            corpus.records,
            key=lambda record: record.post.canonical_url,
        )
    ):
        raise ValueError("records are not canonically ordered")

    source_by_digest: dict[str, CorpusSourceEntry] = {}
    for source in corpus.sources:
        _require_digest(source.raw_snapshot_sha256, "raw snapshot")
        _require_digest(source.extraction_digest, "extraction")
        _require_digest(source.artifact_digest, "source artifact")
        if source.artifact_digest in source_by_digest:
            raise ValueError("duplicate source artifact digest")
        if _parse_timestamp(source.collected_at, "collected_at") > created:
            raise ValueError("source collected after corpus creation")
        if source.post_count <= 0:
            raise ValueError("source post count must be positive")
        source_by_digest[source.artifact_digest] = source

    grouped: dict[str, list[TylerSourcePost]] = {
        digest: [] for digest in source_by_digest
    }
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    seen_prose: set[str] = set()
    for record in corpus.records:
        if record.source_artifact_digest not in grouped:
            raise ValueError("record source lineage is unknown")
        if record.evidence_role not in ALLOWED_ROLES:
            raise ValueError("invalid evidence role")
        if record.evidence_role != _temporal_role(
            record.post.published_at
        ):
            raise ValueError("evidence role mismatch")
        if _parse_timestamp(
            record.post.published_at, "published_at"
        ) > created:
            raise ValueError("publication after corpus creation")
        if record.post.canonical_url in seen_urls:
            raise ValueError("duplicate post URL")
        if record.post.post_id in seen_ids:
            raise ValueError("duplicate post ID")
        prose = record.post.normalized_prose_sha256
        if prose is not None and prose in seen_prose:
            raise ValueError("duplicate normalized prose")
        seen_urls.add(record.post.canonical_url)
        seen_ids.add(record.post.post_id)
        if prose is not None:
            seen_prose.add(prose)
        grouped[record.source_artifact_digest].append(record.post)

    for digest, posts in grouped.items():
        source = source_by_digest[digest]
        if len(posts) != source.post_count:
            raise ValueError("source post count mismatch")
        reconstructed = TylerSourceArtifact(
            schema_version=SOURCE_SCHEMA_VERSION,
            person_id=PERSON_ID,
            source_url=source.source_url,
            collected_at=source.collected_at,
            raw_snapshot_sha256=source.raw_snapshot_sha256,
            extraction_digest=source.extraction_digest,
            decision_axes=DECISION_AXES,
            posts=tuple(sorted(posts, key=lambda post: post.canonical_url)),
            artifact_digest=source.artifact_digest,
        )
        verify_tyler_source_artifact(reconstructed)

    expected_role_counts = {
        role: sum(
            record.evidence_role == role for record in corpus.records
        )
        for role in ALLOWED_ROLES
        if any(record.evidence_role == role for record in corpus.records)
    }
    if corpus.role_counts != expected_role_counts:
        raise ValueError("role counts mismatch")
    statuses = sorted(
        {record.post.candidate_status for record in corpus.records}
    )
    expected_candidate_counts = {
        status: sum(
            record.post.candidate_status == status
            for record in corpus.records
        )
        for status in statuses
    }
    if corpus.candidate_counts != expected_candidate_counts:
        raise ValueError("candidate counts mismatch")
    if corpus.corpus_digest != _digest_json(corpus._unsigned_dict()):
        raise ValueError("corpus_digest mismatch")

    if source_inputs is not None:
        recomputed = create_tyler_corpus(
            source_inputs,
            expected_source_urls=corpus.expected_source_urls,
            created_at=corpus.created_at,
        )
        if recomputed != corpus:
            raise ValueError("raw source recomputation mismatch")
    return True


def _post_from_dict(data: Mapping[str, object]) -> TylerSourcePost:
    return TylerSourcePost(
        post_id=str(data["post_id"]),
        canonical_url=str(data["canonical_url"]),
        title=str(data["title"]),
        author=str(data["author"]),
        published_at=str(data["published_at"]),
        categories=tuple(str(value) for value in data["categories"]),
        authored_prose=tuple(
            AuthoredProseUnit(
                text=str(unit["text"]),
                sha256=str(unit["sha256"]),
                plain_text_char_count=int(
                    unit["plain_text_char_count"]
                ),
                link_text_char_count=int(unit["link_text_char_count"]),
                question_only=bool(unit["question_only"]),
            )
            for unit in data["authored_prose"]
        ),
        quoted_units=tuple(
            QuotedUnit(
                sha256=str(unit["sha256"]),
                char_count=int(unit["char_count"]),
            )
            for unit in data["quoted_units"]
        ),
        candidate_status=str(data["candidate_status"]),
        normalized_prose_sha256=(
            None
            if data.get("normalized_prose_sha256") is None
            else str(data["normalized_prose_sha256"])
        ),
    )


def tyler_corpus_from_dict(
    data: Mapping[str, object],
) -> TylerCorpusArtifact:
    if data.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {CORPUS_SCHEMA_VERSION}"
        )
    corpus = TylerCorpusArtifact(
        schema_version=str(data["schema_version"]),
        person_id=str(data["person_id"]),
        created_at=str(data["created_at"]),
        expected_source_urls=tuple(
            str(value) for value in data["expected_source_urls"]
        ),
        sources=tuple(
            CorpusSourceEntry(
                source_url=str(value["source_url"]),
                collected_at=str(value["collected_at"]),
                raw_snapshot_sha256=str(
                    value["raw_snapshot_sha256"]
                ),
                extraction_digest=str(value["extraction_digest"]),
                artifact_digest=str(value["artifact_digest"]),
                post_count=int(value["post_count"]),
            )
            for value in data["sources"]
        ),
        records=tuple(
            CorpusRecord(
                source_artifact_digest=str(
                    value["source_artifact_digest"]
                ),
                evidence_role=str(value["evidence_role"]),
                post=_post_from_dict(value["post"]),
            )
            for value in data["records"]
        ),
        role_counts={
            str(key): int(value)
            for key, value in dict(data["role_counts"]).items()
        },
        candidate_counts={
            str(key): int(value)
            for key, value in dict(data["candidate_counts"]).items()
        },
        corpus_digest=str(data["corpus_digest"]),
    )
    verify_tyler_corpus(corpus)
    return corpus


def save_tyler_corpus(
    corpus: TylerCorpusArtifact,
    path: Path,
) -> None:
    verify_tyler_corpus(corpus)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            corpus.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_tyler_corpus(path: Path) -> TylerCorpusArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Tyler corpus artifact must be a JSON object")
    return tyler_corpus_from_dict(raw)


def load_corpus_source_manifest(
    path: Path,
) -> tuple[CorpusSourceInput, ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corpus input manifest must be a JSON object")
    if raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported corpus input manifest schema")
    inputs = raw.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("corpus input manifest requires inputs")
    result: list[CorpusSourceInput] = []
    for value in inputs:
        if not isinstance(value, dict):
            raise ValueError("corpus manifest input must be an object")
        artifact_path = path.parent / str(value["artifact"])
        raw_path = path.parent / str(value["raw_snapshot"])
        artifact = load_tyler_source_artifact(artifact_path)
        if str(value["source_url"]) != artifact.source_url:
            raise ValueError("manifest source URL mismatch")
        result.append(
            CorpusSourceInput(
                artifact=artifact,
                raw_snapshot=raw_path.read_text(encoding="utf-8"),
            )
        )
    return tuple(result)
