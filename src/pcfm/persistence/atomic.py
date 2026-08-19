"""原子写工具：临时文件 + fsync + 原子替换，避免断电留下半写文件。"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


def atomic_write_json(path: Path, value: object) -> None:
    """唯一化临时文件名, 避免并发写同一文件时共用 .tmp 产生半写/替换失败。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise
