"""P5-2 集成测试: 并发发送消息到不同人物, 校验无竞争、无串话。"""
from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pcfm.services import PcfmService


MODEL_QA = """Q: How should the studio release a product?
A: Release it only after the evidence is strong enough, and keep the first version focused.
"""

BOB_QA = """Q: How should the studio release a product?
A: Bob prefers a broad launch immediately.
"""


class ConcurrencyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = PcfmService(self.storage, seed_example=False)
        self.alice = self.service.create_conversation_person(
            name="Alice Example", aliases=["Alice"], language="en",
            description="Product studio interviews",
        )
        self.bob = self.service.create_conversation_person(
            name="Bob Example", aliases=["Bob"], language="en",
            description="Independent test person",
        )
        self._confirm(str(self.alice["person_id"]), MODEL_QA)
        self._confirm(str(self.bob["person_id"]), BOB_QA)

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _confirm(self, person_id: str, text: str) -> None:
        source = self.service.add_conversation_text_source(
            person_id,
            title="Interview",
            text=text,
            speaker=self.service.get_person(person_id)["name"],
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="Interview paragraphs 1-2",
            source_context="Recorded public interview transcript",
            source_url="https://example.org/interview",
        )
        self.service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )

    def test_concurrent_sends_to_different_people_are_isolated(self) -> None:
        alice_id = str(self.alice["person_id"])
        bob_id = str(self.bob["person_id"])
        question = "How should the studio release a product?"

        def ask(person_id: str) -> dict[str, object]:
            return self.service.send_conversation_message(person_id, question)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ask, alice_id if i % 2 == 0 else bob_id) for i in range(6)]
            replies = [future.result() for future in futures]

        alice_replies = [r for r in replies if r["person_id"] == alice_id]
        bob_replies = [r for r in replies if r["person_id"] == bob_id]
        self.assertEqual(3, len(alice_replies))
        self.assertEqual(3, len(bob_replies))
        for reply in alice_replies:
            self.assertIn("focused", reply["text"])
            self.assertNotIn("broad launch", reply["text"])
        for reply in bob_replies:
            self.assertIn("broad launch", reply["text"])

        alice_summary = self.service.conversation_summary(alice_id)
        bob_summary = self.service.conversation_summary(bob_id)
        self.assertTrue(all(item["person_id"] == alice_id for item in alice_summary["messages"]))
        self.assertTrue(all(item["person_id"] == bob_id for item in bob_summary["messages"]))


if __name__ == "__main__":
    unittest.main()
