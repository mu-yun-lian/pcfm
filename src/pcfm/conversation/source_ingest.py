"""资料摄取工具：文本清洗、HTML/PDF 解析、编码检测、分段与 Q&A 抽取。"""
from __future__ import annotations

import hashlib
import html
import re
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Mapping


def _text_hash(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _tokens(value: str) -> set[str]:
    lowered = value.casefold()
    latin = set(re.findall(r"[a-z0-9][a-z0-9'-]*", lowered))
    chinese = "".join(re.findall(r"[\u3400-\u9fff]", lowered))
    grams = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}
    if len(chinese) == 1:
        grams.add(chinese)
    return {item for item in latin | grams if len(item) > 1 or item.isdigit()}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(
        None,
        re.sub(r"\s+", " ", left.casefold()).strip(),
        re.sub(r"\s+", " ", right.casefold()).strip(),
    ).ratio()
    return round(max(jaccard, sequence * 0.8), 6)


_BOILERPLATE_TAGS = frozenset({
    "script", "style", "noscript", "nav", "footer", "aside",
    "form", "button", "select", "option", "textarea", "input", "iframe", "svg",
})


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._in_title = False
        self._blocked = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _BOILERPLATE_TAGS:
            self._blocked += 1
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "article", "section", "li", "br", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _BOILERPLATE_TAGS and self._blocked:
            self._blocked -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._blocked:
            return
        clean = html.unescape(data).strip()
        if clean:
            if self._in_title:
                self.title = clean
            self.parts.append(clean)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", " ".join(self.parts)).strip()


def _extract_html(value: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(value)
    return parser.text(), parser.title


def _charset_from_header(content_type: str) -> str:
    match = re.search(r"charset=[\"']?\s*([A-Za-z0-9._-]+)", content_type, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _charset_from_meta(raw: bytes) -> str:
    head = raw[:4096].decode("latin-1", errors="ignore")
    match = re.search(r"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9._-]+)", head, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _decode_web_bytes(raw: bytes, content_type: str) -> str:
    charset = _charset_from_header(content_type) or _charset_from_meta(raw)
    if charset:
        try:
            return raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            pass
    return raw.decode("utf-8", errors="replace")


def _extract_qa(text: str) -> list[dict[str, object]]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Q(?:uestion)?|问题|问)\s*[:：]\s*(?P<question>.+?)"
        r"\n\s*(?:A(?:nswer)?|回答|答)\s*[:：]\s*(?P<answer>.+?)"
        r"(?=(?:\n\s*(?:Q(?:uestion)?|问题|问)\s*[:：])|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    records: list[dict[str, object]] = []
    for index, match in enumerate(pattern.finditer(normalized), start=1):
        question = re.sub(r"\s+", " ", match.group("question")).strip()
        answer = re.sub(r"\s+", " ", match.group("answer")).strip()
        if question and answer:
            records.append(
                {
                    "qa_id": f"qa-{index:04d}",
                    "question": question,
                    "answer": answer,
                    "locator": f"extracted Q&A {index}",
                    "content_hash": _text_hash(question + "\n" + answer),
                }
            )
    inline = re.compile(
        r"(?:Question|问题|问)\s*[:：]\s*(?P<question>.+?)\s+"
        r"(?:Answer|回答|答)\s*[:：]\s*(?P<answer>.+?)(?=(?:Question|问题|问)\s*[:：]|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in inline.finditer(normalized):
        question = re.sub(r"\s+", " ", match.group("question")).strip()
        answer = re.sub(r"\s+", " ", match.group("answer")).strip()
        digest = _text_hash(question + "\n" + answer)
        if question and answer and all(item["content_hash"] != digest for item in records):
            records.append(
                {
                    "qa_id": f"qa-{len(records) + 1:04d}",
                    "question": question,
                    "answer": answer,
                    "locator": f"extracted inline Q&A {len(records) + 1}",
                    "content_hash": digest,
                }
            )
    return records


def _segments(text: str) -> list[dict[str, object]]:
    raw = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"\n\s*\n", text)]
    chunks = [item for item in raw if len(item) >= 18]
    if not chunks and text.strip():
        chunks = [re.sub(r"\s+", " ", text).strip()]
    result = []
    for index, item in enumerate(chunks, start=1):
        for offset in range(0, len(item), 1200):
            value = item[offset : offset + 1200].strip()
            if value:
                result.append(
                    {
                        "segment_id": f"segment-{len(result) + 1:04d}",
                        "text": value,
                        "locator": f"text segment {index}",
                        "content_hash": _text_hash(value),
                    }
                )
    return result


def _structured_rows_text(value: object) -> str:
    rows = value if isinstance(value, list) else [value]
    rendered: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            question = row.get("question") or row.get("问题") or row.get("prompt")
            answer = row.get("answer") or row.get("回答") or row.get("response")
            if question is not None and answer is not None:
                rendered.append(f"Q: {question}\nA: {answer}")
            else:
                rendered.append(
                    "\n".join(f"{key}: {item}" for key, item in row.items())
                )
        else:
            rendered.append(str(row))
    return "\n\n".join(rendered)
