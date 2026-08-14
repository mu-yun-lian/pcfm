from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Mapping, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree


SCHEMA_VERSION = "tyler-source-v1"
PERSON_ID = "tyler-cowen"
EXPECTED_AUTHOR = "Tyler Cowen"
OFFICIAL_HOST = "marginalrevolution.com"
AUTHOR_PATH_PATTERN = re.compile(
    r"^/marginalrevolution/author/tyler-cowen"
    r"(?:/page/[1-9][0-9]*)?/?$"
)
OFFICIAL_FEED_URLS = frozenset(
    {
        "https://feeds.feedblitz.com/marginalrevolution",
        "https://marginalrevolution.com/feed",
    }
)
CHALLENGE_MARKERS = (
    "cf-chl",
    "just a moment...",
    "enable javascript and cookies to continue",
    "cf-mitigated",
)
DECISION_AXES = (
    "ai_acceleration_vs_risk_regulation",
    "market_mechanisms_vs_government_intervention",
    "technological_progress_vs_employment_displacement",
    "state_capacity_vs_individual_liberty",
    "short_term_social_cost_vs_long_term_growth",
)
ALLOWED_STATUSES = frozenset(
    {
        "needs_human_annotation",
        "ambiguous_not_trainable",
        "no_stance_candidate",
    }
)
_METADATA_CLASS_TOKENS = frozenset(
    {
        "author",
        "byline",
        "entry-meta",
        "meta",
        "post-meta",
        "postmetadata",
    }
)


class TylerSourceRefusedError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__(
            "Tyler source extraction refused: "
            + ", ".join(self.reasons)
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: object) -> str:
    return _digest_bytes(_canonical_json(value))


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a SHA-256 hex digest"
        ) from error


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"{label} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _normalized_prose(value: str) -> str:
    return "".join(
        character.casefold()
        for character in _normalize_space(value)
        if character.isalnum()
    )


def _canonical_url(value: str, base_url: str) -> str:
    parsed = urlsplit(urljoin(base_url, value))
    if parsed.scheme not in {"http", "https"}:
        raise TylerSourceRefusedError(("invalid_post_url",))
    if (parsed.hostname or "").casefold() != OFFICIAL_HOST:
        raise TylerSourceRefusedError(("external_post_url",))
    path = re.sub(r"/{2,}", "/", parsed.path) or "/"
    return urlunsplit(("https", OFFICIAL_HOST, path, "", ""))


def _validate_archive_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != OFFICIAL_HOST
        or not AUTHOR_PATH_PATTERN.fullmatch(parsed.path)
        or parsed.query
        or parsed.fragment
    ):
        raise TylerSourceRefusedError(("unofficial_source",))
    path = parsed.path.rstrip("/")
    return f"https://{OFFICIAL_HOST}{path}"


def _validate_rss_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if parsed.query or parsed.fragment:
        raise TylerSourceRefusedError(("unofficial_source",))
    canonical = urlunsplit(
        (
            parsed.scheme,
            (parsed.hostname or "").casefold(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )
    if canonical not in OFFICIAL_FEED_URLS:
        raise TylerSourceRefusedError(("unofficial_source",))
    return canonical


def _validate_source_url(source_url: str) -> str:
    parsed = urlsplit(source_url)
    if (
        parsed.scheme == "https"
        and (parsed.hostname or "").casefold() == OFFICIAL_HOST
        and AUTHOR_PATH_PATTERN.fullmatch(parsed.path)
        and not parsed.query
        and not parsed.fragment
    ):
        return _validate_archive_source_url(source_url)
    return _validate_rss_source_url(source_url)


def _is_question_only(value: str) -> bool:
    text = _normalize_space(value)
    return bool(text) and text.rstrip("\"'”’)]} ").endswith(("?", "？"))


@dataclass(frozen=True)
class AuthoredProseUnit:
    text: str
    sha256: str
    plain_text_char_count: int
    link_text_char_count: int
    question_only: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "sha256": self.sha256,
            "plain_text_char_count": self.plain_text_char_count,
            "link_text_char_count": self.link_text_char_count,
            "question_only": self.question_only,
        }


@dataclass(frozen=True)
class QuotedUnit:
    sha256: str
    char_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sha256": self.sha256,
            "char_count": self.char_count,
        }


@dataclass(frozen=True)
class TylerSourcePost:
    post_id: str
    canonical_url: str
    title: str
    author: str
    published_at: str
    categories: tuple[str, ...]
    authored_prose: tuple[AuthoredProseUnit, ...]
    quoted_units: tuple[QuotedUnit, ...]
    candidate_status: str
    normalized_prose_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "post_id": self.post_id,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "author": self.author,
            "published_at": self.published_at,
            "categories": list(self.categories),
            "authored_prose": [
                unit.to_dict() for unit in self.authored_prose
            ],
            "quoted_units": [
                unit.to_dict() for unit in self.quoted_units
            ],
            "candidate_status": self.candidate_status,
            "normalized_prose_sha256": self.normalized_prose_sha256,
        }


@dataclass(frozen=True)
class TylerSourceArtifact:
    schema_version: str
    person_id: str
    source_url: str
    collected_at: str
    raw_snapshot_sha256: str
    extraction_digest: str
    decision_axes: tuple[str, ...]
    posts: tuple[TylerSourcePost, ...]
    artifact_digest: str

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "person_id": self.person_id,
            "source_url": self.source_url,
            "collected_at": self.collected_at,
            "raw_snapshot_sha256": self.raw_snapshot_sha256,
            "extraction_digest": self.extraction_digest,
            "decision_axes": list(self.decision_axes),
            "posts": [post.to_dict() for post in self.posts],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._unsigned_dict(),
            "artifact_digest": self.artifact_digest,
        }

    def candidate_counts(self) -> dict[str, int]:
        return {
            status: sum(
                post.candidate_status == status
                for post in self.posts
            )
            for status in sorted(ALLOWED_STATUSES)
        }


@dataclass
class _Paragraph:
    in_blockquote: bool
    excluded: bool
    text_parts: list[str]
    plain_parts: list[str]
    link_parts: list[str]


@dataclass
class _Anchor:
    href: str
    rel: frozenset[str]
    text_parts: list[str]


@dataclass
class _Article:
    title_parts: list[str]
    title_url: str
    author_names: list[str]
    author_slugs: list[str]
    published_at: str
    categories: list[str]
    authored_paragraphs: list[tuple[str, int, int]]
    quoted_paragraphs: list[str]


class _ArchiveParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.articles: list[_Article] = []
        self.current: _Article | None = None
        self.blockquote_depth = 0
        self.title_depth = 0
        self.paragraph: _Paragraph | None = None
        self.anchors: list[_Anchor] = []

    @staticmethod
    def _attributes(
        attributes: list[tuple[str, str | None]],
    ) -> dict[str, str]:
        return {
            name.casefold(): value or ""
            for name, value in attributes
        }

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        tag = tag.casefold()
        attrs = self._attributes(attributes)
        if tag == "article":
            if self.current is not None:
                raise TylerSourceRefusedError(("nested_article",))
            self.current = _Article(
                title_parts=[],
                title_url="",
                author_names=[],
                author_slugs=[],
                published_at="",
                categories=[],
                authored_paragraphs=[],
                quoted_paragraphs=[],
            )
            return
        if self.current is None:
            return
        if tag == "blockquote":
            self.blockquote_depth += 1
        elif tag in {"h1", "h2"}:
            if (
                not self.current.title_parts
                and not self.current.title_url
            ):
                self.title_depth = 1
        elif tag == "p":
            class_tokens = frozenset(
                attrs.get("class", "").casefold().split()
            )
            self.paragraph = _Paragraph(
                in_blockquote=self.blockquote_depth > 0,
                excluded=bool(
                    class_tokens.intersection(_METADATA_CLASS_TOKENS)
                ),
                text_parts=[],
                plain_parts=[],
                link_parts=[],
            )
        elif tag == "a":
            anchor = _Anchor(
                href=attrs.get("href", ""),
                rel=frozenset(attrs.get("rel", "").casefold().split()),
                text_parts=[],
            )
            self.anchors.append(anchor)
            if self.title_depth and anchor.href:
                self.current.title_url = anchor.href
        elif tag == "time" and attrs.get("datetime"):
            self.current.published_at = attrs["datetime"].strip()
        elif tag == "br" and self.paragraph is not None:
            self.paragraph.text_parts.append(" ")
            if self.anchors:
                self.paragraph.link_parts.append(" ")
            else:
                self.paragraph.plain_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self.current is None:
            return
        if self.title_depth:
            self.current.title_parts.append(data)
        if self.anchors:
            self.anchors[-1].text_parts.append(data)
        if self.paragraph is not None:
            self.paragraph.text_parts.append(data)
            if self.anchors:
                self.paragraph.link_parts.append(data)
            else:
                self.paragraph.plain_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if self.current is None:
            return
        if tag == "a" and self.anchors:
            anchor = self.anchors.pop()
            text = _normalize_space("".join(anchor.text_parts))
            parsed = urlsplit(anchor.href)
            path = parsed.path.rstrip("/")
            if "author" in anchor.rel or "/author/" in path:
                if text:
                    self.current.author_names.append(text)
                slug = path.rsplit("/", 1)[-1]
                if slug:
                    self.current.author_slugs.append(slug.casefold())
            if (
                "category" in anchor.rel
                or "tag" in anchor.rel
                or "/category/" in path
            ) and text:
                self.current.categories.append(text)
        elif tag == "p" and self.paragraph is not None:
            paragraph = self.paragraph
            self.paragraph = None
            text = _normalize_space("".join(paragraph.text_parts))
            plain = _normalize_space("".join(paragraph.plain_parts))
            link = _normalize_space("".join(paragraph.link_parts))
            if text and not paragraph.excluded:
                if paragraph.in_blockquote:
                    self.current.quoted_paragraphs.append(text)
                elif plain:
                    self.current.authored_paragraphs.append(
                        (text, len(plain), len(link))
                    )
        elif tag in {"h1", "h2"} and self.title_depth:
            self.title_depth -= 1
        elif tag == "blockquote" and self.blockquote_depth:
            self.blockquote_depth -= 1
        elif tag == "article":
            self.articles.append(self.current)
            self.current = None
            self.blockquote_depth = 0
            self.title_depth = 0
            self.paragraph = None
            self.anchors.clear()

    def handle_startendtag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attributes)
        self.handle_endtag(tag)


def _unit_from_paragraph(
    paragraph: tuple[str, int, int],
) -> AuthoredProseUnit:
    text, plain_count, link_count = paragraph
    return AuthoredProseUnit(
        text=text,
        sha256=_digest_bytes(text.encode("utf-8")),
        plain_text_char_count=plain_count,
        link_text_char_count=link_count,
        question_only=_is_question_only(text),
    )


def _quote_unit(text: str) -> QuotedUnit:
    return QuotedUnit(
        sha256=_digest_bytes(text.encode("utf-8")),
        char_count=len(text),
    )


def _is_feed_footer(text: str, title: str) -> bool:
    normalized = _normalize_space(text).casefold().rstrip(".")
    expected = (
        f"the post {title} appeared first on marginal revolution"
    ).casefold()
    return normalized == expected


def _candidate_status(
    title: str,
    authored_prose: Sequence[AuthoredProseUnit],
) -> str:
    if "assorted links" in title.casefold() or not authored_prose:
        return "no_stance_candidate"
    if all(unit.question_only for unit in authored_prose):
        return "ambiguous_not_trainable"
    return "needs_human_annotation"


def _post_unsigned_dict(
    *,
    canonical_url: str,
    title: str,
    author: str,
    published_at: str,
    categories: tuple[str, ...],
    authored_prose: tuple[AuthoredProseUnit, ...],
    quoted_units: tuple[QuotedUnit, ...],
    candidate_status: str,
    normalized_prose_sha256: str | None,
) -> dict[str, object]:
    return {
        "canonical_url": canonical_url,
        "title": title,
        "author": author,
        "published_at": published_at,
        "categories": list(categories),
        "authored_prose": [
            unit.to_dict() for unit in authored_prose
        ],
        "quoted_units": [
            unit.to_dict() for unit in quoted_units
        ],
        "candidate_status": candidate_status,
        "normalized_prose_sha256": normalized_prose_sha256,
    }


def _convert_article(
    article: _Article,
    source_url: str,
) -> TylerSourcePost:
    title = _normalize_space("".join(article.title_parts))
    if not title or not article.title_url:
        raise TylerSourceRefusedError(("missing_post_identity",))
    canonical_url = _canonical_url(article.title_url, source_url)
    author_names = tuple(dict.fromkeys(article.author_names))
    author_slugs = tuple(dict.fromkeys(article.author_slugs))
    if not author_names and not author_slugs:
        raise TylerSourceRefusedError(("missing_author",))
    if (
        author_names != (EXPECTED_AUTHOR,)
        or author_slugs != (PERSON_ID,)
    ):
        raise TylerSourceRefusedError(("unexpected_author",))
    if not article.published_at:
        raise TylerSourceRefusedError(("missing_publication",))
    try:
        _parse_timestamp(article.published_at, "published_at")
    except ValueError as error:
        raise TylerSourceRefusedError(
            ("invalid_publication",)
        ) from error
    authored_prose = tuple(
        _unit_from_paragraph(paragraph)
        for paragraph in article.authored_paragraphs
        if not _is_feed_footer(paragraph[0], title)
    )
    quoted_units = tuple(
        _quote_unit(text) for text in article.quoted_paragraphs
    )
    candidate_status = _candidate_status(title, authored_prose)
    normalized = _normalized_prose(
        " ".join(unit.text for unit in authored_prose)
    )
    normalized_digest = (
        _digest_bytes(normalized.encode("utf-8"))
        if normalized
        else None
    )
    categories = tuple(
        sorted(dict.fromkeys(article.categories), key=str.casefold)
    )
    unsigned = _post_unsigned_dict(
        canonical_url=canonical_url,
        title=title,
        author=EXPECTED_AUTHOR,
        published_at=article.published_at,
        categories=categories,
        authored_prose=authored_prose,
        quoted_units=quoted_units,
        candidate_status=candidate_status,
        normalized_prose_sha256=normalized_digest,
    )
    return TylerSourcePost(
        post_id=_digest_json(unsigned),
        canonical_url=canonical_url,
        title=title,
        author=EXPECTED_AUTHOR,
        published_at=article.published_at,
        categories=categories,
        authored_prose=authored_prose,
        quoted_units=quoted_units,
        candidate_status=candidate_status,
        normalized_prose_sha256=normalized_digest,
    )


def _extraction_payload(
    source_url: str,
    posts: Sequence[TylerSourcePost],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "person_id": PERSON_ID,
        "source_url": source_url,
        "decision_axes": list(DECISION_AXES),
        "posts": [post.to_dict() for post in posts],
    }


def _create_artifact(
    raw_snapshot: str,
    *,
    source_url: str,
    collected_at: str,
    posts: Sequence[TylerSourcePost],
) -> TylerSourceArtifact:
    canonical_source_url = _validate_source_url(source_url)
    _parse_timestamp(collected_at, "collected_at")
    canonical_posts = tuple(
        sorted(posts, key=lambda post: post.canonical_url)
    )
    urls: set[str] = set()
    prose_digests: set[str] = set()
    for post in canonical_posts:
        if post.canonical_url in urls:
            raise TylerSourceRefusedError(("duplicate_post_url",))
        urls.add(post.canonical_url)
        digest = post.normalized_prose_sha256
        if digest is not None and digest in prose_digests:
            raise TylerSourceRefusedError(
                ("duplicate_normalized_prose",)
            )
        if digest is not None:
            prose_digests.add(digest)
    extraction_digest = _digest_json(
        _extraction_payload(canonical_source_url, canonical_posts)
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "person_id": PERSON_ID,
        "source_url": canonical_source_url,
        "collected_at": collected_at,
        "raw_snapshot_sha256": _digest_bytes(
            raw_snapshot.encode("utf-8")
        ),
        "extraction_digest": extraction_digest,
        "decision_axes": DECISION_AXES,
        "posts": canonical_posts,
    }
    artifact = TylerSourceArtifact(
        **unsigned,
        artifact_digest=_digest_json(
            {
                "schema_version": unsigned["schema_version"],
                "person_id": unsigned["person_id"],
                "source_url": unsigned["source_url"],
                "collected_at": unsigned["collected_at"],
                "raw_snapshot_sha256": unsigned[
                    "raw_snapshot_sha256"
                ],
                "extraction_digest": unsigned["extraction_digest"],
                "decision_axes": list(DECISION_AXES),
                "posts": [
                    post.to_dict() for post in canonical_posts
                ],
            }
        ),
    )
    verify_tyler_source_artifact(artifact)
    return artifact


def extract_tyler_source_page(
    html: str,
    *,
    source_url: str,
    collected_at: str,
) -> TylerSourceArtifact:
    canonical_source_url = _validate_archive_source_url(source_url)
    _parse_timestamp(collected_at, "collected_at")
    lowered = html.casefold()
    if any(marker in lowered for marker in CHALLENGE_MARKERS):
        raise TylerSourceRefusedError(("challenge_page",))
    parser = _ArchiveParser()
    parser.feed(html)
    parser.close()
    if not parser.articles:
        raise TylerSourceRefusedError(("no_articles",))
    posts = tuple(
        _convert_article(article, canonical_source_url)
        for article in parser.articles
    )
    return _create_artifact(
        html,
        source_url=canonical_source_url,
        collected_at=collected_at,
        posts=posts,
    )


def _rss_text(item: ElementTree.Element, tag: str) -> str:
    element = item.find(tag)
    return "" if element is None else _normalize_space(element.text or "")


def extract_tyler_source_rss(
    xml: str,
    *,
    source_url: str,
    collected_at: str,
) -> TylerSourceArtifact:
    canonical_source_url = _validate_rss_source_url(source_url)
    _parse_timestamp(collected_at, "collected_at")
    lowered = xml.casefold()
    if any(marker in lowered for marker in CHALLENGE_MARKERS):
        raise TylerSourceRefusedError(("challenge_page",))
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise TylerSourceRefusedError(("invalid_rss",)) from error
    if root.tag != "rss":
        raise TylerSourceRefusedError(("invalid_rss",))
    channel = root.find("channel")
    if channel is None:
        raise TylerSourceRefusedError(("invalid_rss",))
    items = tuple(channel.findall("item"))
    if not items:
        raise TylerSourceRefusedError(("no_rss_items",))
    posts = []
    dc_creator = "{http://purl.org/dc/elements/1.1/}creator"
    encoded = "{http://purl.org/rss/1.0/modules/content/}encoded"
    original_link = (
        "{http://rssnamespace.org/feedburner/ext/1.0}origLink"
    )
    for item in items:
        creator = _rss_text(item, dc_creator)
        if not creator:
            raise TylerSourceRefusedError(("missing_rss_creator",))
        if creator != EXPECTED_AUTHOR:
            continue
        title = _rss_text(item, "title")
        origin = _rss_text(item, original_link)
        if not origin:
            candidate_link = _rss_text(item, "link")
            if (
                urlsplit(candidate_link).hostname or ""
            ).casefold() == OFFICIAL_HOST:
                origin = candidate_link
        publication = _rss_text(item, "pubDate")
        body_element = item.find(encoded)
        body = "" if body_element is None else body_element.text or ""
        if not title or not origin:
            raise TylerSourceRefusedError(("missing_post_identity",))
        if not publication:
            raise TylerSourceRefusedError(("missing_rss_publication",))
        if not body:
            raise TylerSourceRefusedError(("missing_rss_content",))
        try:
            published_at = parsedate_to_datetime(
                publication
            ).isoformat()
        except (TypeError, ValueError) as error:
            raise TylerSourceRefusedError(
                ("invalid_rss_publication",)
            ) from error
        categories = tuple(
            _normalize_space(element.text or "")
            for element in item.findall("category")
            if _normalize_space(element.text or "")
        )
        synthetic = (
            "<article><h2><a href=\""
            + escape(origin, quote=True)
            + "\">"
            + escape(title)
            + "</a></h2><p class=\"byline\"><a rel=\"author\" "
            + "href=\"https://marginalrevolution.com/"
            + "marginalrevolution/author/tyler-cowen\">"
            + escape(creator)
            + "</a></p><time datetime=\""
            + escape(published_at, quote=True)
            + "\"></time>"
            + "".join(
                "<a rel=\"category tag\" href=\"https://"
                + "marginalrevolution.com/marginalrevolution/"
                + "category/rss\">"
                + escape(category)
                + "</a>"
                for category in categories
            )
            + "<div class=\"entry-content\">"
            + body
            + "</div></article>"
        )
        parser = _ArchiveParser()
        parser.feed(synthetic)
        parser.close()
        if len(parser.articles) != 1:
            raise TylerSourceRefusedError(("invalid_rss_content",))
        posts.append(
            _convert_article(
                parser.articles[0],
                canonical_source_url,
            )
        )
    if not posts:
        raise TylerSourceRefusedError(("no_tyler_items",))
    return _create_artifact(
        xml,
        source_url=canonical_source_url,
        collected_at=collected_at,
        posts=posts,
    )


def _authored_unit_from_dict(
    data: Mapping[str, object],
) -> AuthoredProseUnit:
    return AuthoredProseUnit(
        text=str(data["text"]),
        sha256=str(data["sha256"]),
        plain_text_char_count=int(data["plain_text_char_count"]),
        link_text_char_count=int(data["link_text_char_count"]),
        question_only=bool(data["question_only"]),
    )


def _quoted_unit_from_dict(data: Mapping[str, object]) -> QuotedUnit:
    return QuotedUnit(
        sha256=str(data["sha256"]),
        char_count=int(data["char_count"]),
    )


def _post_from_dict(data: Mapping[str, object]) -> TylerSourcePost:
    return TylerSourcePost(
        post_id=str(data["post_id"]),
        canonical_url=str(data["canonical_url"]),
        title=str(data["title"]),
        author=str(data["author"]),
        published_at=str(data["published_at"]),
        categories=tuple(str(value) for value in data["categories"]),
        authored_prose=tuple(
            _authored_unit_from_dict(dict(value))
            for value in data["authored_prose"]
        ),
        quoted_units=tuple(
            _quoted_unit_from_dict(dict(value))
            for value in data["quoted_units"]
        ),
        candidate_status=str(data["candidate_status"]),
        normalized_prose_sha256=(
            None
            if data.get("normalized_prose_sha256") is None
            else str(data["normalized_prose_sha256"])
        ),
    )


def tyler_source_artifact_from_dict(
    data: Mapping[str, object],
) -> TylerSourceArtifact:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    artifact = TylerSourceArtifact(
        schema_version=str(data["schema_version"]),
        person_id=str(data["person_id"]),
        source_url=str(data["source_url"]),
        collected_at=str(data["collected_at"]),
        raw_snapshot_sha256=str(data["raw_snapshot_sha256"]),
        extraction_digest=str(data["extraction_digest"]),
        decision_axes=tuple(
            str(value) for value in data["decision_axes"]
        ),
        posts=tuple(
            _post_from_dict(dict(value))
            for value in data["posts"]
        ),
        artifact_digest=str(data["artifact_digest"]),
    )
    expected_artifact_digest = _digest_json(artifact._unsigned_dict())
    if artifact.artifact_digest != expected_artifact_digest:
        raise ValueError("artifact_digest mismatch")
    verify_tyler_source_artifact(artifact)
    return artifact


def verify_tyler_source_artifact(
    artifact: TylerSourceArtifact,
    *,
    raw_snapshot: str | None = None,
) -> bool:
    if artifact.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if artifact.person_id != PERSON_ID:
        raise ValueError(f"person_id must be {PERSON_ID}")
    _validate_source_url(artifact.source_url)
    _parse_timestamp(artifact.collected_at, "collected_at")
    _require_digest(
        artifact.raw_snapshot_sha256,
        "raw_snapshot_sha256",
    )
    _require_digest(
        artifact.extraction_digest,
        "extraction_digest",
    )
    _require_digest(artifact.artifact_digest, "artifact_digest")
    if artifact.decision_axes != DECISION_AXES:
        raise ValueError("decision_axes mismatch")
    if not artifact.posts:
        raise ValueError("posts cannot be empty")
    if tuple(
        sorted(artifact.posts, key=lambda post: post.canonical_url)
    ) != artifact.posts:
        raise ValueError("posts must be canonically ordered")
    urls: set[str] = set()
    prose_digests: set[str] = set()
    for post in artifact.posts:
        if post.author != EXPECTED_AUTHOR:
            raise ValueError("unexpected post author")
        if post.candidate_status not in ALLOWED_STATUSES:
            raise ValueError("invalid candidate_status")
        if post.canonical_url in urls:
            raise ValueError("duplicate post URL")
        urls.add(post.canonical_url)
        for unit in post.authored_prose:
            if unit.sha256 != _digest_bytes(unit.text.encode("utf-8")):
                raise ValueError("authored prose digest mismatch")
            if (
                unit.plain_text_char_count <= 0
                or unit.link_text_char_count < 0
            ):
                raise ValueError("invalid authored prose character count")
            if unit.question_only != _is_question_only(unit.text):
                raise ValueError("question_only mismatch")
        if post.candidate_status != _candidate_status(
            post.title,
            post.authored_prose,
        ):
            raise ValueError("candidate_status mismatch")
        for unit in post.quoted_units:
            _require_digest(unit.sha256, "quoted unit sha256")
            if unit.char_count <= 0:
                raise ValueError("invalid quoted unit character count")
        normalized = _normalized_prose(
            " ".join(unit.text for unit in post.authored_prose)
        )
        expected_normalized_digest = (
            _digest_bytes(normalized.encode("utf-8"))
            if normalized
            else None
        )
        if post.normalized_prose_sha256 != expected_normalized_digest:
            raise ValueError("normalized prose digest mismatch")
        if (
            expected_normalized_digest is not None
            and expected_normalized_digest in prose_digests
        ):
            raise ValueError("duplicate normalized prose")
        if expected_normalized_digest is not None:
            prose_digests.add(expected_normalized_digest)
        unsigned_post = _post_unsigned_dict(
            canonical_url=post.canonical_url,
            title=post.title,
            author=post.author,
            published_at=post.published_at,
            categories=post.categories,
            authored_prose=post.authored_prose,
            quoted_units=post.quoted_units,
            candidate_status=post.candidate_status,
            normalized_prose_sha256=post.normalized_prose_sha256,
        )
        if post.post_id != _digest_json(unsigned_post):
            raise ValueError("post_id mismatch")
    expected_extraction_digest = _digest_json(
        _extraction_payload(artifact.source_url, artifact.posts)
    )
    if artifact.extraction_digest != expected_extraction_digest:
        raise ValueError("extraction_digest mismatch")
    if artifact.artifact_digest != _digest_json(
        artifact._unsigned_dict()
    ):
        raise ValueError("artifact_digest mismatch")
    if raw_snapshot is not None:
        if _digest_bytes(
            raw_snapshot.encode("utf-8")
        ) != artifact.raw_snapshot_sha256:
            raise ValueError("raw snapshot digest mismatch")
        expected = (
            extract_tyler_source_rss(
                raw_snapshot,
                source_url=artifact.source_url,
                collected_at=artifact.collected_at,
            )
            if artifact.source_url in OFFICIAL_FEED_URLS
            else extract_tyler_source_page(
                raw_snapshot,
                source_url=artifact.source_url,
                collected_at=artifact.collected_at,
            )
        )
        if expected != artifact:
            raise ValueError("raw snapshot replay mismatch")
    return True


def save_tyler_source_artifact(
    artifact: TylerSourceArtifact,
    path: Path,
) -> None:
    verify_tyler_source_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            artifact.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_tyler_source_artifact(path: Path) -> TylerSourceArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Tyler source artifact must be a JSON object")
    return tyler_source_artifact_from_dict(raw)
