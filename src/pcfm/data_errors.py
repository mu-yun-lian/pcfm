"""本地数据文件的统一错误类型与安全读取。"""
from __future__ import annotations

import copy
import json
from pathlib import Path


class PcfmDataError(ValueError):
    """本地数据文件缺失、损坏或版本不兼容。"""


class PersonDataCorrupted(PcfmDataError):
    """单个人物的数据文件损坏。"""


def safe_read_json(path: Path, default=None):
    if not path.exists():
        if default is not None:
            return copy.deepcopy(default)
        raise PcfmDataError(f"本地文件不存在：{path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise PcfmDataError(f"本地数据文件损坏：{path.name}") from error
