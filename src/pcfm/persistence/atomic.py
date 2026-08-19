"""原子写工具：临时文件 + fsync + 原子替换，避免断电留下半写文件。"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


def _replace_with_retry(tmp: Path, path: Path, attempts: int = 3) -> None:
    """os.replace 到同一目标在 Windows 上偶发瞬时文件锁(PermissionError/FileNotFoundError), 重试吸收。"""
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError):
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path: Path, value: object) -> None:
    """唯一化临时文件名, 避免并发写同一文件时共用 .tmp 产生半写/替换失败。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
