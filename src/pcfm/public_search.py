from __future__ import annotations

import re
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree


class PublicSearchError(RuntimeError):
    pass


class PublicSearchProvider(Protocol):
    provider_id: str

    def search(
        self,
        *,
        person_name: str,
        identity_note: str,
        language: str,
        limit: int,
    ) -> list[dict[str, object]]: ...


class BingRssPublicSearch:
    """Small keyless public-search adapter. Results are candidates, never truth."""

    provider_id = "bing-rss-public-web"
    endpoint = "https://www.bing.com/search?format=rss&q="

    def search(
        self,
        *,
        person_name: str,
        identity_note: str,
        language: str,
        limit: int,
    ) -> list[dict[str, object]]:
        clean_name = str(person_name).strip()
        if not clean_name:
            raise PublicSearchError("人物姓名不能为空。")
        disambiguation = re.sub(r"\s+", " ", str(identity_note).strip())
        query = f'"{clean_name}" interview speech transcript Q&A {disambiguation}'.strip()
        request = Request(
            self.endpoint + quote_plus(query),
            headers={
                "User-Agent": "PCFM/0.4 local evidence collector",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.5",
                "Accept-Language": str(language).strip() or "en",
            },
        )
        try:
            with urlopen(request, timeout=15) as response:
                payload = response.read(2 * 1024 * 1024)
        except Exception as error:
            raise PublicSearchError(f"公开搜索服务暂时不可用：{error}") from error
        try:
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as error:
            raise PublicSearchError("公开搜索服务返回了无法解析的数据。") from error
        results: list[dict[str, object]] = []
        seen: set[str] = set()
        name_tokens = {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+|[\u3400-\u9fff]+", clean_name)
            if token
        }
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            url = (item.findtext("link") or "").strip()
            snippet = re.sub(r"\s+", " ", item.findtext("description") or "").strip()
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            canonical = url.split("#", 1)[0]
            if canonical in seen:
                continue
            combined = f"{title} {snippet}".casefold()
            identity_hits = sum(token in combined for token in name_tokens)
            if name_tokens and identity_hits == 0:
                continue
            published = (item.findtext("pubDate") or "").strip()
            if published:
                try:
                    published = parsedate_to_datetime(published).date().isoformat()
                except (TypeError, ValueError, OverflowError):
                    published = ""
            seen.add(canonical)
            results.append(
                {
                    "title": title or parsed.hostname,
                    "url": canonical,
                    "snippet": snippet,
                    "published_at": published,
                    "provider_rank": len(results) + 1,
                    "identity_match": "name_token_match",
                }
            )
            if len(results) >= max(1, int(limit)):
                break
        return results
