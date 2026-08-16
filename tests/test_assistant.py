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
        self.service.close()
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

    def test_add_url_source_tool(self) -> None:
        from unittest.mock import patch

        person = self.service.create_conversation_person(name="乔布斯", aliases=[], language="zh")
        pid = str(person["person_id"])
        with patch.object(
            self.service, "add_conversation_url_source", return_value={"source_id": "s1"}
        ) as mocked:
            result = self.assistant._execute_tool(
                "add_url_source",
                {"person_id": pid, "url": "https://example.com/jobs-speech"},
            )
        self.assertIn("已抓取", result)
        mocked.assert_called_once()
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["url"], "https://example.com/jobs-speech")
        self.assertEqual(kwargs["dataset_role"], "reference_only")
        self.assertEqual(kwargs["content_authenticity"], "unverified_material")
        self.assertEqual(kwargs["source_locator"], "user_provided_url")

    def test_parse_json_prefix_tolerates_trailing_data(self) -> None:
        from pcfm.assistant import _parse_json_prefix

        # 模型返回合法 JSON 后又跟了额外内容(Extra data) → 只取第一个 JSON 对象
        parsed = _parse_json_prefix('{"reply": "ok", "tool_calls": []} trailing junk')
        self.assertEqual(parsed["reply"], "ok")
        # markdown 代码围栏
        fenced = _parse_json_prefix('```json\n{"reply": "hi"}\n```')
        self.assertEqual(fenced["reply"], "hi")

    def test_create_person_dedupes_existing_name(self) -> None:
        self.assistant._execute_tool("create_person", {"name": "特朗普"})
        result = self.assistant._execute_tool("create_person", {"name": "特朗普"})
        self.assertIn("已经存在", result)
        names = [p["name"] for p in self.service.list_people()]
        self.assertEqual(names.count("特朗普"), 1)


if __name__ == "__main__":
    unittest.main()
