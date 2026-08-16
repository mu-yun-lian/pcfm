"""P5-4 性能验收: 100 人列表 / 切换人物延迟 / 500 消息 JSON 读取。"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from pcfm.services import PcfmService


def timed(label, fn, runs=5):
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    print(f"{label}: {best*1000:.1f} ms (best of {runs})")
    return best


def main() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    service = PcfmService(root, seed_example=False)

    # 100 人
    created = []
    for i in range(100):
        person = service.create_conversation_person(
            name=f"Perf 人物 {i}",
            aliases=[f"P{i}"],
            language="zh",
            description="performance fixture",
            focus_domain="test",
        )
        created.append(str(person["person_id"]))

    timed("list_people(100人)", lambda: service.list_people(), runs=5)
    timed("get_person(切换人物)", lambda: service.get_person(created[42]), runs=10)
    timed("conversation_summary(单人物)", lambda: service.conversation_summary(created[42]), runs=5)

    # 500 消息 JSON 读取/写出(虚拟滚动前的数据层成本)
    messages = [{"message_id": f"m{i}", "role": "user", "text": "你好" * 10} for i in range(500)]
    state_path = root / "people" / created[0] / "conversation_active.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"messages": messages}, ensure_ascii=False), encoding="utf-8")
    timed("读取500消息JSON", lambda: json.loads(state_path.read_text(encoding="utf-8"))["messages"], runs=5)

    service.close()
    tmp.cleanup()
    print("PERF_OK")


if __name__ == "__main__":
    main()
