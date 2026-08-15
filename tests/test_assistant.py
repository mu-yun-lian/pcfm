# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pcfm.product_service import ProductService


class AssistantFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = ProductService(Path(self.temporary.name), seed_example=False)
        self.assistant = self.service.assistant

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, *messages):
        replies = []
        for message in messages:
            replies.append(self.assistant.handle(message)["reply"])
        return replies

    def test_create_person_flow_creates_person(self) -> None:
        self._run("建人物", "张三", "小张", "", "技术专家", "")
        result = self.assistant.handle("确认")["reply"]
        self.assertIn("已创建人物", result)
        names = [p["name"] for p in self.service.list_people()]
        self.assertIn("张三", names)

    def test_list_people_intent(self) -> None:
        self.service.create_conversation_person(name="Alice", aliases=[], language="zh")
        reply = self.assistant.handle("列出人物")["reply"]
        self.assertIn("Alice", reply)

    def test_cancel_flow_resets(self) -> None:
        self._run("建人物", "张三")
        reply = self.assistant.handle("取消")["reply"]
        self.assertIn("已取消", reply)


if __name__ == "__main__":
    unittest.main()
