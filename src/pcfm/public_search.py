from __future__ import annotations

import json
import re
from email.utils import parsedate_to_datetime
from typing import Protocol
from urllib.parse import quote, quote_plus, urlparse
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


class WikipediaCollector:
    """免费维基收集器：搜词条 → 抓全文 → 提取外部链接（一手来源）。结果均为候选。"""

    provider_id = "wikipedia-public-web"

    def __init__(self, language: str = "zh"):
        self.language = language
        self.base = f"https://{language}.wikipedia.org/w"
        self.headers = {"User-Agent": "PCFM/0.5 local evidence collector"}

    def _get_json(self, url: str) -> dict[str, object]:
        request = Request(url, headers={**self.headers, "Accept": "application/json"})
        with urlopen(request, timeout=20) as response:
            payload = response.read(8 * 1024 * 1024)
        return json.loads(payload.decode("utf-8"))

    def search_titles(self, query: str, limit: int = 3) -> list[str]:
        url = (
            f"{self.base}/api.php?action=query&list=search&srsearch={quote_plus(query)}"
            f"&format=json&srlimit={max(1, int(limit))}"
        )
        data = self._get_json(url)
        return [str(item["title"]) for item in data.get("query", {}).get("search", [])]

    def article_text(self, title: str) -> str:
        url = f"{self.base}/api/rest_v1/page/plain/{quote(title)}"
        request = Request(url, headers=self.headers)
        with urlopen(request, timeout=25) as response:
            return response.read(8 * 1024 * 1024).decode("utf-8")

    def external_links(self, title: str, limit: int = 60) -> list[str]:
        url = (
            f"{self.base}/api.php?action=query&prop=extlinks&ellimit=max"
            f"&titles={quote_plus(title)}&format=json&redirects=1"
        )
        data = self._get_json(url)
        links: list[str] = []
        for page in data.get("query", {}).get("pages", {}).values():
            for item in page.get("extlinks", []) or []:
                link = str(item.get("url", "")).strip()
                if link and link not in links:
                    links.append(link)
        return links[: max(1, int(limit))]

