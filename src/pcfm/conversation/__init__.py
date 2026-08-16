"""对话核心拆分：资料摄取、提取、建模等模块。"""
from .source_ingest import (  # noqa: F401
    _decode_web_bytes,
    _extract_html,
    _extract_qa,
    _segments,
    _similarity,
    _structured_rows_text,
    _text_hash,
)

__all__ = [
    "_decode_web_bytes",
    "_extract_html",
    "_extract_qa",
    "_segments",
    "_similarity",
    "_structured_rows_text",
    "_text_hash",
]
