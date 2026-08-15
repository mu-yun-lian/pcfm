from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pcfm.product_service import ProductError, ProductService


MODEL_QA = """Q: How should the studio release a product?
A: Release it only after the evidence is strong enough, and keep the first version focused.
"""

REFERENCE_QA = """Q: How should the studio release a product?
A: Ship a focused first version after careful testing, then learn from real use.
"""

HOLDOUT_QA = """Q: How should a small studio release its product?
A: Ship a focused first version after careful testing, then learn from real use.
"""


class MaterialExtractionModel:
    def roles(self):
        return {"default_dialogue": "", "material_processing": "fake:model", "validation": ""}

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {"snapshot_id": "fixture-material", "provider": "fixture", "model_id": "model"}

    def invoke(self, service_id, model_id, messages, **kwargs):
        payload = json.loads(messages[-1]["content"])
        material = str(payload["material"])
        known = "I would test the hardest assumption before expanding."
        response_text = known if known in material else material
        return {
            "text": json.dumps(
                {
                    "events": [
                        {
                            "trigger": "How should a team test an early product?",
                            "context": "Public product talk",
                            "response": response_text,
                            "occasion": "Public product talk",
                            "interlocutor": "audience",
                            "speaker": "Alice Example",
                            "locator": "paragraph 1",
                            "speech_act": "direct_answer",
                            "stance": "support",
                            "claims": ["Test the hardest assumption first."],
                            "reasons": [],
                            "memories": [],
                            "uncertainties": [],
                        }
                    ]
                }
            ),
            "snapshot": self.snapshot(""),
        }


class ConversationMVPTests(unittest.TestCase):
    def test_chinese_short_question_answer_markers_are_extracted(self) -> None:
        person_id = str(self.alice["person_id"])
        source = self.service.add_conversation_text_source(
            person_id,
            title="中文访谈",
            text="问：工作室应该怎样发布新产品？\n答：证据足够时再发布，第一版保持聚焦。",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
        )
        self.assertEqual(1, len(source["qas"]))
        self.assertEqual("工作室应该怎样发布新产品？", source["qas"][0]["question"])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = ProductService(self.storage, seed_example=False)
        self.alice = self.service.create_conversation_person(
            name="Alice Example",
            aliases=["Alice"],
            language="en",
            description="Product studio interviews",
        )
        self.bob = self.service.create_conversation_person(
            name="Bob Example",
            aliases=["Bob"],
            language="en",
            description="Independent test person",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _add_confirmed(
        self,
        person_id: str,
        text: str,
        *,
        role: str = "model_source",
        title: str = "Interview",
        speaker: str | None = None,
        source_date: str = "2025-01-01",
    ) -> dict[str, object]:
        source = self.service.add_conversation_text_source(
            person_id,
            title=title,
            text=text,
            speaker=speaker or self.service.get_person(person_id)["name"],
            source_date=source_date,
            dataset_role=role,
            content_authenticity="verbatim_transcript",
            source_locator=f"{title} paragraphs 1-2",
            source_context="Recorded public interview transcript",
            source_url=f"https://example.org/{person_id}/{title.replace(' ', '-').lower()}",
        )
        return self.service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )

    def test_people_sources_messages_and_reload_are_isolated(self) -> None:
        alice_id = str(self.alice["person_id"])
        bob_id = str(self.bob["person_id"])
        self._add_confirmed(alice_id, MODEL_QA)
        self._add_confirmed(
            bob_id,
            "Q: How should the studio release a product?\nA: Bob prefers a broad launch immediately.",
        )

        alice_reply = self.service.send_conversation_message(
            alice_id, "How should the studio release a product?"
        )
        bob_reply = self.service.send_conversation_message(
            bob_id, "How should the studio release a product?"
        )
        self.assertIn("focused", alice_reply["text"])
        self.assertIn("broad launch", bob_reply["text"])
        self.assertNotEqual(alice_reply["text"], bob_reply["text"])

        reloaded = ProductService(self.storage, seed_example=False)
        alice_summary = reloaded.conversation_summary(alice_id)
        bob_summary = reloaded.conversation_summary(bob_id)
        self.assertEqual(len(alice_summary["messages"]), 2)
        self.assertEqual(len(bob_summary["messages"]), 2)
        self.assertTrue(all(item["person_id"] == alice_id for item in alice_summary["messages"]))
        self.assertTrue(all(item["person_id"] == bob_id for item in bob_summary["messages"]))

    def test_disabled_reality_lookup_has_zero_extra_calls(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        reply = self.service.send_conversation_message(
            person_id,
            "How should the studio release a product?",
            reality_lookup_requested=False,
        )
        telemetry = self.service.conversation_summary(person_id)["telemetry"]
        self.assertEqual(telemetry["reality_lookup_requests"], 0)
        self.assertEqual(telemetry["reality_local_search_calls"], 0)
        self.assertEqual(telemetry["reality_online_search_calls"], 0)
        self.assertEqual(reply["reality_lookup_status"], "not_requested")

    def test_local_reality_comparison_and_pending_candidate_do_not_mutate_model(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        self._add_confirmed(person_id, REFERENCE_QA, role="reference_only", title="Verified transcript")
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        version_before = self.service.conversation_summary(person_id)["active_version"]
        comparison = self.service.find_conversation_reality_answer(
            person_id, str(reply["message_id"])
        )
        self.assertEqual(comparison["status"], "candidate_found")
        self.assertEqual(comparison["speaker"], "Alice Example")
        candidate = self.service.create_optimization_candidate(
            person_id, str(reply["message_id"])
        )
        self.assertEqual(candidate["status"], "pending")
        summary = self.service.conversation_summary(person_id)
        self.assertEqual(summary["active_version"], version_before)
        self.assertEqual(summary["optimization_candidates"][-1]["status"], "pending")

    def test_optimization_requires_holdout_then_creates_rollbackable_version(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        self._add_confirmed(
            person_id,
            REFERENCE_QA,
            role="reference_only",
            title="Reality answer",
            source_date="2025-02-01",
        )
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        self.service.find_conversation_reality_answer(person_id, str(reply["message_id"]))
        candidate = self.service.create_optimization_candidate(person_id, str(reply["message_id"]))
        failed = self.service.review_optimization_candidate(
            person_id, str(candidate["candidate_id"]), "confirmed"
        )
        self.assertEqual(failed["status"], "failed_validation")
        self.assertIn("independent_holdout_required", failed["validation_reasons"])

        self._add_confirmed(
            person_id,
            HOLDOUT_QA,
            role="final_holdout",
            title="Sealed holdout",
            source_date="2026-01-01",
        )
        candidate2 = self.service.create_optimization_candidate(
            person_id, str(reply["message_id"]), allow_retry=True
        )
        accepted = self.service.review_optimization_candidate(
            person_id, str(candidate2["candidate_id"]), "confirmed"
        )
        self.assertEqual(accepted["status"], "accepted_exploratory")
        self.assertEqual(
            "not_assessed_full_conversation_holdout_required",
            accepted["simulation_v5_holdout_after"]["status"],
        )
        summary = self.service.conversation_summary(person_id)
        self.assertEqual(summary["active_version"], 2)
        first_version, content_version = summary["versions"]
        self.assertEqual(content_version["style_update_status"], "unchanged_separate_review_required")
        self.assertEqual(
            "selected_event_integrity_passed_v5_accuracy_not_assessed",
            content_version["validation_status"],
        )
        self.assertEqual(content_version["style_artifact_hash"], first_version["style_artifact_hash"])
        self.assertEqual(content_version["style_revision"], first_version["style_revision"])
        self.assertEqual(accepted["surface_extraction"]["status"], "pending_separate_style_review")

        style_accepted = self.service.review_optimization_style_candidate(
            person_id, str(candidate2["candidate_id"]), "confirmed"
        )
        self.assertEqual(style_accepted["surface_extraction"]["status"], "accepted_exploratory")
        summary = self.service.conversation_summary(person_id)
        self.assertEqual(summary["active_version"], 3)
        style_version = summary["versions"][-1]
        self.assertEqual(style_version["content_update_status"], "unchanged")
        self.assertEqual(style_version["response_model_hash"], content_version["response_model_hash"])
        self.assertEqual(style_version["content_revision"], content_version["content_revision"])
        self.assertGreater(style_version["style_revision"], content_version["style_revision"])

        rolled_back = self.service.rollback_conversation_version(person_id, 1)
        self.assertEqual(rolled_back["active_version"], 1)
        self.assertEqual(self.service.conversation_summary(person_id)["active_version"], 1)

    def test_optimization_promotes_only_the_selected_reality_event(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        reference = self._add_confirmed(
            person_id,
            REFERENCE_QA
            + "\nQ: What private password was used?\nA: A private password was used for the archive.\n",
            role="reference_only",
            title="Multi-event reality source",
            source_date="2025-02-01",
        )
        self._add_confirmed(
            person_id,
            HOLDOUT_QA,
            role="final_holdout",
            title="Sealed holdout",
            source_date="2026-01-01",
        )
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        self.service.find_conversation_reality_answer(person_id, str(reply["message_id"]))
        candidate = self.service.create_optimization_candidate(
            person_id, str(reply["message_id"])
        )
        accepted = self.service.review_optimization_candidate(
            person_id, str(candidate["candidate_id"]), "confirmed"
        )
        self.assertEqual("accepted_exploratory", accepted["status"])
        artifact = self.service.conversation._simulation_model(
            person_id, accepted["new_version"]
        )
        promoted = [
            frame
            for frame in artifact["event_frames"]
            if frame["source_id"] == reference["source_id"]
        ]
        self.assertEqual(1, len(promoted))
        self.assertNotIn("password", promoted[0]["observed_response"]["verbatim"])

    def test_optimization_recomputes_selected_event_from_raw_source(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        reference = self._add_confirmed(
            person_id,
            REFERENCE_QA,
            role="reference_only",
            title="Reality answer",
            source_date="2025-02-01",
        )
        self._add_confirmed(
            person_id,
            HOLDOUT_QA,
            role="final_holdout",
            title="Sealed holdout",
            source_date="2026-01-01",
        )
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        self.service.find_conversation_reality_answer(person_id, str(reply["message_id"]))
        candidate = self.service.create_optimization_candidate(
            person_id, str(reply["message_id"])
        )
        source_path = self.storage / "people" / person_id / "conversation_sources.json"
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        stored = next(item for item in sources if item["source_id"] == reference["source_id"])
        selected = next(
            event
            for event in stored["response_events"]
            if event["event_id"] == candidate["source_event_id"]
        )
        selected["actual_response"] = "Forged derived response not present in the source."
        source_path.write_text(json.dumps(sources), encoding="utf-8")
        reviewed = self.service.review_optimization_candidate(
            person_id, str(candidate["candidate_id"]), "confirmed"
        )
        self.assertEqual("failed_validation", reviewed["status"])
        self.assertIn("selected_event_recompute_mismatch", reviewed["validation_reasons"])

    def test_unstructured_public_material_builds_event_model_without_qa_format(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = MaterialExtractionModel()
        source = self._add_confirmed(
            person_id,
            (
                "Small teams should release a focused first version. "
                "Because early evidence is incomplete, they should test the hardest "
                "assumption before expanding the product."
            ),
            title="Public product talk",
        )
        self.assertFalse(source["qas"])
        self.assertEqual(
            "awaiting_verbatim_response_episode_extraction",
            source["model_ingestion_status"],
        )
        self.assertIsNone(self.service.conversation_summary(person_id)["active_version"])
        extracted = self.service.extract_conversation_response_candidates(
            person_id, str(source["source_id"])
        )
        candidate_id = str(
            extracted["llm_response_event_candidates"][0]["candidate_id"]
        )
        source = self.service.review_conversation_response_candidate(
            person_id, str(source["source_id"]), candidate_id, "confirmed"
        )
        self.assertTrue(
            any(
                event.get("label_status")
                == "confirmed_response_weak_semantic_labels"
                and event.get("event_atom", {}).get("event_type")
                for event in source["response_events"]
            )
        )
        summary = self.service.conversation_summary(person_id)
        self.assertEqual(summary["active_version"], 1)
        model = self.service.conversation._response_model(person_id, 1)
        self.assertTrue(model["episode_bundles"])
        self.assertTrue(model["conditional_tendencies"])
        self.assertTrue(model["demonstrated_knowledge"])

    def test_mixed_speaker_material_is_not_auto_attributed_to_the_person(self) -> None:
        person_id = str(self.alice["person_id"])
        source = self.service.add_conversation_text_source(
            person_id,
            title="Panel transcript",
            text="Alice and Bob debate whether a focused launch is better than a broad launch.",
            speaker="Alice Example",
            speaker_scope="mixed_speakers",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="Multi-person panel",
            source_url="https://example.org/panel",
        )
        reviewed = self.service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )
        self.assertEqual("model_source", reviewed["dataset_role"])
        self.assertFalse(
            any(
                event.get("label_status")
                == "confirmed_response_weak_semantic_labels"
                for event in reviewed["response_events"]
            )
        )
        self.assertIsNone(self.service.conversation_summary(person_id)["active_version"])

    def test_non_qa_reality_event_can_be_compared_after_dialogue(self) -> None:
        person_id = str(self.alice["person_id"])
        self._add_confirmed(person_id, MODEL_QA)
        source = self.service.add_conversation_text_source(
            person_id,
            title="Later public statement",
            text="Ship a focused first version after careful testing, then learn from real use.",
            speaker="Alice Example",
            source_date="2025-02-01",
            dataset_role="reference_only",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="How should the studio release a product?",
            source_url="https://example.org/later-statement",
        )
        self.service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        comparison = self.service.find_conversation_reality_answer(
            person_id, str(reply["message_id"])
        )
        self.assertEqual("candidate_found", comparison["status"])
        self.assertTrue(comparison["reality_candidates"][0]["event_id"])

    def test_llm_material_candidate_needs_verbatim_and_explicit_review_before_promotion(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = MaterialExtractionModel()
        source = self.service.add_conversation_text_source(
            person_id,
            title="Public product talk",
            text="I would test the hardest assumption before expanding.",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="Public product talk",
            source_url="https://example.org/talk",
        )
        extracted = self.service.extract_conversation_response_candidates(
            person_id, str(source["source_id"])
        )
        candidate = extracted["llm_response_event_candidates"][0]
        self.assertEqual("pending", candidate["review_status"])
        self.assertIsNone(self.service.conversation_summary(person_id)["active_version"])

        self.service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )
        before = self.service.conversation_summary(person_id)["active_version"]
        promoted = self.service.review_conversation_response_candidate(
            person_id,
            str(source["source_id"]),
            str(candidate["candidate_id"]),
            "confirmed",
        )
        promoted_candidate = promoted["llm_response_event_candidates"][0]
        self.assertEqual("confirmed_promoted", promoted_candidate["review_status"])
        self.assertEqual(1, len(promoted["reviewed_event_frames_v4"]))
        self.assertIsNone(before)
        self.assertEqual(1, self.service.conversation_summary(person_id)["active_version"])

    def test_no_source_and_unmatched_question_do_not_invent_person_stance(self) -> None:
        bob_id = str(self.bob["person_id"])
        no_source = self.service.send_conversation_message(bob_id, "What do you think?")
        self.assertEqual(no_source["status"], "needs_model")
        self.assertEqual(no_source["answer_status"], "general_assisted")
        self.assertEqual(no_source["person_prediction_status"], "not_available")

        alice_id = str(self.alice["person_id"])
        self._add_confirmed(alice_id, MODEL_QA)
        out_of_scope = self.service.send_conversation_message(
            alice_id, "Explain orbital mechanics around Neptune."
        )
        self.assertEqual(out_of_scope["status"], "needs_model")
        self.assertEqual(out_of_scope["answer_status"], "general_assisted")
        self.assertEqual(out_of_scope["person_prediction_status"], "not_available")
        self.assertEqual("simulation-v5", out_of_scope["prediction_trace"]["kernel"])
        self.assertFalse(out_of_scope["structured_prediction"]["claims"])

    def test_txt_markdown_pdf_and_url_sources_are_real_inputs(self) -> None:
        person_id = str(self.alice["person_id"])
        txt = self.service.add_conversation_file_source(
            person_id,
            filename="notes.txt",
            content_base64=base64.b64encode(MODEL_QA.encode()).decode(),
            speaker="Alice Example",
            dataset_role="model_source",
        )
        md = self.service.add_conversation_file_source(
            person_id,
            filename="notes.md",
            content_base64=base64.b64encode(b"# Notes\nA markdown statement.").decode(),
            speaker="Alice Example",
            dataset_role="reference_only",
        )
        pdf_bytes = b"""%PDF-1.4
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>endobj
4 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj
5 0 obj<< /Length 61 >>stream
BT /F1 12 Tf 72 720 Td (Question: What matters? Answer: Evidence matters.) Tj ET
endstream endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000241 00000 n 
0000000311 00000 n 
trailer<< /Size 6 /Root 1 0 R >>
startxref
421
%%EOF"""
        pdf = self.service.add_conversation_file_source(
            person_id,
            filename="interview.pdf",
            content_base64=base64.b64encode(pdf_bytes).decode(),
            speaker="Alice Example",
            dataset_role="reference_only",
        )

        class Response:
            headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit: int):
                return b"<html><title>Interview page</title><body><p>Public source statement.</p></body></html>"

        with patch("pcfm.conversation_mvp.urlopen", return_value=Response()):
            url = self.service.add_conversation_url_source(
                person_id,
                url="https://example.org/interview",
                speaker="Alice Example",
                dataset_role="reference_only",
            )
        self.assertEqual(txt["format"], "txt")
        self.assertEqual(md["format"], "markdown")
        self.assertEqual(pdf["format"], "pdf")
        self.assertIn("Evidence matters", pdf["text_preview"])
        self.assertEqual(url["format"], "webpage")

    def test_verbatim_answer_bypasses_mutating_style_renderer(self) -> None:
        person = self.service.create_conversation_person(
            name="Steve Jobs",
            aliases=[],
            language="en",
            description="Style fallback test",
        )
        person_id = str(person["person_id"])
        self._add_confirmed(person_id, MODEL_QA, speaker="Steve Jobs")

        class FailingRenderer:
            def render(self, contract, **_kwargs):
                neutral = " ".join(item["text"] for item in contract["claims"])
                return {
                    "neutral_text": neutral,
                    "selected": {"status": "rejected", "text": "mutated"},
                    "semantic_preservation": {"status": "rejected"},
                }

        self.service.conversation._renderers["steve_jobs_v1"] = FailingRenderer()
        reply = self.service.send_conversation_message(
            person_id, "How should the studio release a product?"
        )
        self.assertEqual(
            reply["style_status"], "source_verbatim_person_style"
        )
        self.assertEqual(reply["text"], reply["neutral_content"])
        self.assertNotIn("mutated", reply["text"])

    def test_generation_temperature_follows_character_and_override(self) -> None:
        jobs = self.service.create_conversation_person(
            name="Steve Jobs",
            aliases=[],
            language="en",
            description="Apple co-founder",
        )
        jobs_id = str(jobs["person_id"])
        # steve_jobs_v1 表达包 → 人物默认温度 0.65（非全局 0.7）
        self.assertAlmostEqual(
            0.65, self.service.conversation._generation_temperature(jobs_id), places=6
        )
        # 中性人物 → 全局默认 0.7
        self.assertAlmostEqual(
            0.7,
            self.service.conversation._generation_temperature(str(self.alice["person_id"])),
            places=6,
        )
        # 显式覆盖后按覆盖值生效
        self.service.update_person(jobs_id, {"generation_params": {"temperature": 1.1}})
        self.assertAlmostEqual(
            1.1, self.service.conversation._generation_temperature(jobs_id), places=6
        )
        # 越界被夹回 [0, 2]
        self.service.update_person(jobs_id, {"generation_params": {"temperature": 99}})
        self.assertEqual(2.0, self.service.conversation._generation_temperature(jobs_id))
        self.service.update_person(jobs_id, {"generation_params": {"temperature": -3}})
        self.assertEqual(0.0, self.service.conversation._generation_temperature(jobs_id))


if __name__ == "__main__":
    unittest.main()
