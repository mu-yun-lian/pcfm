from __future__ import annotations

import base64
import csv
import copy
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..applicability import PredictionRefusedError
from ..persistence.atomic import atomic_write_json
from ..assistant import AssistantEngine
from ..data_errors import PcfmDataError, safe_read_json
from ..persistence.db import Database
from ..jobs import JobRunner, JobStore
from ..persistence.repositories import PersonRepository
from ..contracts import Observation, Scenario
from ..cognitive_workbench import (
    CognitiveWorkbench,
    CognitiveWorkbenchError,
    load_builtin_hawley_case,
)
from ..conversation_mvp import ConversationError, ConversationWorkbench
from ..decision_evidence_v1 import decision_evidence_bundle_from_dict
from ..demo_people import DEMO_PEOPLE, DEMO_SEED_VERSION
from ..evaluation import evaluate_probability_array, report_to_dict
from ..expression_renderer import (
    ExpressionRenderer,
    ExpressionRendererError,
    builtin_expression_profile_path,
)
from ..ledger import EventLedger, VerificationAuthority, observation_payload
from ..model_services import ModelServiceError, ModelServiceManager
from ..public_search import BingRssPublicSearch, PublicSearchError, WikipediaCollector
from ..storage import load_bundle, save_bundle
from ..synthetic import FEATURE_NAMES, generate_population_dataset
from ..workflow import (
    fit_person_model,
    load_event_ledger_jsonl,
    predict_with_bundle,
    save_event_ledger_jsonl,
    update_person_model,
)


PRODUCT_FORMAT = "pcfm-local-product-v1"
VERIFIER_ID = "pcfm-local-product"
MINIMUM_PROFILE_SAMPLES = 50
MINIMUM_VALIDATION_SAMPLES = 100


class ProductError(ValueError):
    """An error that is safe to show to an ordinary product user."""


REASON_TEXT = {
    "independent_validation_required": "还没有独立验证数据。",
    "insufficient_person_validation_samples": "独立验证样本不足 100 条。",
    "insufficient_personalization_uplift": "人物模型没有稳定优于群体基线。",
    "personalization_uplift_not_significant": "人物化提升的统计证据还不稳定。",
    "calibration_error_too_high": "概率校准误差过高。",
    "mechanism_misspecification_suspected": "现有线性模型可能遗漏了重要数值结构。",
    "temporal_behavior_drift_suspected": "不同时段的选择规律可能发生变化。",
    "temporal_stability_not_assessed": "时间稳定性数据不足。",
    "feature_distribution_shift": "新情境的数值特征超出了历史数据范围。",
    "local_support_gap": "历史数据中缺少与这个新情境足够接近的样本。",
    "prediction_time_required": "需要提供预测时间。",
    "prediction_precedes_reference_data": "预测时间早于模型参考数据。",
    "stale_model": "模型距离最近参考数据已经过久，需要更新。",
    "unvalidated_domain_label": "这个情境类型没有在适用域数据中验证。",
    "unvalidated_option_pair": "这一组选项文字没有在适用域数据中验证。",
    "unvalidated_context": "这一类情境说明没有在适用域数据中验证。",
    "model_validation_unvalidated": "模型尚未通过独立验证。",
    "model_validation_failed": "模型没有通过独立验证。",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ProductError("日期时间格式无效。") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProductError("日期时间必须包含时区。")
    return parsed


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _read_json(path: Path, default: object | None = None) -> object:
    try:
        return safe_read_json(path, default)
    except PcfmDataError as error:
        raise ProductError(str(error)) from error


def _slug(value: str) -> str:
    normalized = "-".join(str(value).strip().lower().split())
    safe = "".join(
        char
        for char in normalized
        if char.isascii() and (char.isalnum() or char in "-_")
    )
    return safe[:40] or uuid.uuid4().hex[:12]


def _as_choice(value: object, option_a: str, option_b: str) -> int:
    text = str(value).strip()
    if text in {"0", "A", "a", option_a}:
        return 0
    if text in {"1", "B", "b", option_b}:
        return 1
    raise ProductError("真实选择必须是 A、B、0、1 或对应的选项文字。")


def _reason_text(reason: str) -> str:
    return REASON_TEXT.get(reason, f"模型门禁原因：{reason}")
