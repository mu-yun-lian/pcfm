from __future__ import annotations

import base64
import csv
import copy
import hashlib
import html
import ipaddress
import json
import os
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from html.parser import HTMLParser
from io import BytesIO, StringIO
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..atomic import atomic_write_json
from ..expression_renderer import (
    ExpressionRenderer,
    ExpressionRendererError,
    SAFE_SURFACE_CONNECTORS,
    builtin_expression_profile_path,
    render_person_surface_style,
)
from ..response_prediction import (
    EVALUATION_TENDENCY_TYPES,
    EVENT_STRUCTURE_TYPES,
    STANCES,
    TENDENCY_TYPES,
    TRADEOFF_TENDENCY_TYPES,
    ResponsePredictionError,
    ResponsePredictionKernel,
    canonical_hash as response_canonical_hash,
    response_events_from_source,
    review_response_events,
)
from ..model_services import ModelServiceError, ModelServiceManager
from ..response_prediction_v2 import (
    KERNEL_ID_V2,
    MODEL_SCHEMA_V2,
    ResponsePredictionKernelV2,
)
from ..simulation_v4 import (
    DOMAIN_ALIASES,
    INTERESTS,
    OBJECT_CATEGORIES,
    REVIEWED_EVENT_SCHEMA_V4,
)
from ..simulation_v5 import (
    MODEL_BUILD_V5,
    MODEL_SCHEMA_V5,
    SimulationKernelV5,
    SimulationV5Error,
    _is_short_reference,
)
from .source_ingest import (
    _decode_web_bytes,
    _extract_html,
    _extract_qa,
    _segments,
    _similarity,
    _structured_rows_text,
    _text_hash,
    _tokens,
)
from .derivation import (
    DOMAIN_ZH,
    EVENT_STRUCTURE_ZH,
    OBJECT_CATEGORY_ZH,
    STANCE_ZH,
    TENDENCY_TYPE_ZH,
    _derivation_view,
    _json_mapping,
    _localize_view,
)


SCHEMA_VERSION = "pcfm-conversation-mvp-v1"
SOURCE_ROLES = {
    "model_source",
    "applicability_reference",
    "feature_discovery",
    "candidate_selection",
    "final_holdout",
    "post_deployment_monitoring",
    "reference_only",
}
MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_WEBPAGE_TEXT_CHARS = 50000
MAX_EXTRACTION_CHUNKS = 24

# 生成参数：默认与按人物（表达包）的采样温度。温度只影响最终回答的
# 语言组织与措辞，不参与价值推导；建模/提取/守门路径保持确定性 0.0。
DEFAULT_GENERATION_TEMPERATURE = 0.7
CHARACTER_GENERATION_TEMPERATURE = {
    "steve_jobs_v1": 0.65,
}

class ConversationError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return copy.deepcopy(default)
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_pdf(value: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - dependency declaration guards this
        raise ConversationError("当前环境缺少 PDF 文本提取组件 pypdf。") from error
    try:
        reader = PdfReader(BytesIO(value))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except Exception as error:
        raise ConversationError("PDF 无法解析或文件已经损坏。") from error
    text = text.strip()
    if not text:
        raise ConversationError("这个 PDF 没有可提取文字；当前 MVP 尚未实现 OCR。")
    return text
