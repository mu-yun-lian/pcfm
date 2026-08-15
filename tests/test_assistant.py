# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pcfm.product_service import ProductService


class AssistantToolTests(unittest.TestCase):
    """直接测助手的工具执行（不含 LLM 调用，LLM 由真机验证）。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = ProductService(Path(self.temporary.name), seed_example=False)
        self.assistant = self.service.assistant

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_create_person_tool(self) -> None:
        result = self.assistant._execute_tool(
            "create_person", {"name": "张三", "identity_note": "测试"}
        )
        self.assertIn("已创建", result)
        self.assertIn("张三", [p["name"] for p in self.service.list_people()])

    def test_list_people_tool(self) -> None:
        self.service.create_conversation_person(name="Alice", aliases=[], language="zh")
        result = self.assistant._execute_tool("list_people", {})
        self.assertIn("Alice", result)

    def test_archive_and_restore_by_name(self) -> None:
        person = self.service.create_conversation_person(name="Bob", aliases=[], language="zh")
        pid = str(person["person_id"])
        # 归档
        self.assistant._execute_tool("archive_person", {"person_id": "Bob"})
        self.assertNotIn(pid, [p["person_id"] for p in self.service.list_people()])
        # 恢复（名称在归档里也能解析）
        result = self.assistant._execute_tool("restore_person", {"person_id": "Bob"})
        self.assertIn("已恢复", result)
        self.assertIn(pid, [p["person_id"] for p in self.service.list_people()])


if __name__ == "__main__":
    unittest.main()
