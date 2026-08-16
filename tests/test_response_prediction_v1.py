from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

from pcfm.services import ProductError, PcfmService


INTERVIEW = """Q: Should a studio release a broad first version?
A: No. A focused first version is easier to test. Because real use reveals what matters, the team should iterate after launch.

Q: What should the studio do when the evidence is weak?
A: It should wait and say what remains uncertain. Better evidence should come before a confident launch.

Q: Should the team ignore user testing?
A: No. User testing can overturn an attractive internal assumption.
"""


class ResponsePredictionV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = PcfmService(self.storage, seed_example=False)
        self.person = self.service.create_conversation_person(
            name="Alice Example",
            aliases=["Alice"],
            language="en",
            description="Product interview evidence",
            source_mode="user_provided",
            identity_note="Fictional test identity",
            focus_domain="product development",
        )
        self.person_id = str(self.person["person_id"])

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _confirm_interview(self) -> dict[str, object]:
        source = self.service.add_conversation_text_source(
            self.person_id,
            title="Verified interview",
            text=INTERVIEW,
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="transcript paragraphs 1-6",
            source_context="Recorded public interview",
            source_url="https://example.org/alice-interview",
        )
        return self.service.review_conversation_source(
            self.person_id, str(source["source_id"]), "confirmed"
        )

    def test_chat_uses_independent_simulation_v4_before_rendering(self) -> None:
        self._confirm_interview()
        reply = self.service.send_conversation_message(
            self.person_id, "Should the studio release a broad first version?"
        )

        self.assertEqual(reply["model_kind"], "pcfm_conversation_conditioned_response_simulation_v5")
        self.assertEqual(reply["prediction_trace"]["kernel"], "simulation-v5")
        self.assertEqual(reply["prediction_trace"]["prediction_path"], "direct_event")
        self.assertIn("speech_act", reply["structured_prediction"])
        self.assertIn("stance", reply["structured_prediction"])
        self.assertIn("claims", reply["structured_prediction"])
        self.assertIn("reasons", reply["structured_prediction"])
        self.assertEqual(
            reply["frozen_contract_hash"],
            reply["structured_prediction"]["renderer_contract_digest"],
        )
        self.assertEqual(
            reply["structured_prediction_hash"],
            reply["structured_prediction"]["content_digest"],
        )
        self.assertNotEqual(reply["frozen_contract_hash"], reply["structured_prediction_hash"])
        adopted = " ".join(
            item["text"]
            for field in ("claims", "reasons")
            for item in reply["structured_prediction"][field]
        )
        self.assertNotIn("User testing can overturn", adopted)
        self.assertNotEqual(reply["model_kind"], "pcfm_unified_response_predictor_v2")

    def test_multiturn_history_changes_context_and_generated_reply_is_not_evidence(self) -> None:
        self._confirm_interview()
        first = self.service.send_conversation_message(
            self.person_id, "What should the studio do when the evidence is weak?"
        )
        second = self.service.send_conversation_message(
            self.person_id, "Why?"
        )

        self.assertEqual(first["context_role"], "model_generated_context")
        used = second["prediction_trace"]["context_used"]
        self.assertIn(first["message_id"], used["message_ids"])
        self.assertGreaterEqual(used["turn_count"], 2)
        self.assertNotEqual(
            first["prediction_trace"]["context_digest"],
            second["prediction_trace"]["context_digest"],
        )
        second_content = " ".join(
            item["text"]
            for field in ("claims", "reasons")
            for item in second["structured_prediction"][field]
        )
        self.assertIn("remains uncertain", second_content)
        self.assertNotIn("focused first version", second_content)
        summary = self.service.conversation_summary(self.person_id)
        self.assertEqual(summary["telemetry"]["content_generation_llm_calls"], 0)
        self.assertTrue(
            all(
                event["origin"] != "model_generated_context"
                for source in summary["sources"]
                for event in source.get("response_events", [])
            )
        )

    def test_raw_non_qa_material_becomes_unverified_evidence_not_a_training_label(self) -> None:
        source = self.service.add_conversation_text_source(
            self.person_id,
            title="Article fragment",
            text="The studio released a product after a long internal review.",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
        )
        reviewed = self.service.review_conversation_source(
            self.person_id, str(source["source_id"]), "confirmed"
        )
        self.assertTrue(reviewed["response_events"])
        self.assertTrue(
            all(item["label_status"] == "unverified_candidate" for item in reviewed["response_events"])
        )
        summary = self.service.conversation_summary(self.person_id)
        self.assertIsNone(summary["active_version"])
        self.assertEqual(summary["status"], "insufficient_evidence")

    def test_system_search_mode_is_rejected_when_provider_is_unconfigured(self) -> None:
        service = PcfmService(self.storage / "unconfigured-search", seed_example=False, public_search=False)
        with self.assertRaisesRegex(ProductError, "未配置"):
            service.create_conversation_person(
                name="Public Person",
                aliases=[],
                language="en",
                description="",
                source_mode="system_search",
                identity_note="Public speaker",
                focus_domain="public interviews",
            )

    def test_json_csv_and_subtitle_files_are_processed_as_raw_materials(self) -> None:
        fixtures = {
            "interview.json": b'[{"question":"What matters?","answer":"Evidence matters."}]',
            "interview.csv": b"question,answer\nWhat changes?,Real use changes assumptions.\n",
            "interview.srt": b"1\n00:00:01,000 --> 00:00:03,000\nQ: What is first?\nA: Test the focused version first.\n",
        }
        formats = []
        for filename, raw in fixtures.items():
            source = self.service.add_conversation_file_source(
                self.person_id,
                filename=filename,
                content_base64=base64.b64encode(raw).decode(),
                speaker="Alice Example",
                dataset_role="feature_discovery",
            )
            formats.append(source["format"])
            self.assertTrue(source["response_events"])
        self.assertEqual(formats, ["json", "csv", "subtitle"])

    def test_delete_archives_restore_recovers_and_only_archive_can_be_destroyed(self) -> None:
        self._confirm_interview()
        self.service.send_conversation_message(
            self.person_id, "Should the team ignore user testing?"
        )
        self.service.delete_person(self.person_id)
        self.assertEqual(self.service.list_people(), [])
        archived = self.service.list_archived_people()
        self.assertEqual([item["person_id"] for item in archived], [self.person_id])

        restored = self.service.restore_person(self.person_id)
        self.assertEqual(restored["person_id"], self.person_id)
        self.assertEqual(
            len(self.service.conversation_summary(self.person_id)["messages"]), 2
        )
        with self.assertRaises(ProductError):
            self.service.permanently_delete_archived_person(
                self.person_id, expected_name="Alice Example"
            )

        self.service.delete_person(self.person_id)
        self.service.permanently_delete_archived_person(
            self.person_id, expected_name="Alice Example"
        )
        self.assertEqual(self.service.list_archived_people(), [])

    def test_inactive_research_modules_are_visible_but_cannot_enter_prediction(self) -> None:
        self._confirm_interview()
        reply = self.service.send_conversation_message(
            self.person_id, "Should the team ignore user testing?"
        )
        components = {
            item["component_id"]: item for item in reply["structured_prediction"]["components"]
        }
        self.assertEqual(components["reviewed_response_episodes_v5"]["status"], "active")
        self.assertEqual(components["conversation_state_v1"]["status"], "active")
        self.assertEqual(components["simulation_v4"]["status"], "frozen_evidence_submodel")
        self.assertNotIn("simulation_v3", reply["structured_prediction"]["active_components"])

    def test_model_artifact_tampering_is_refused_before_chat_prediction(self) -> None:
        self._confirm_interview()
        model_path = (
            self.storage
            / "people"
            / self.person_id
            / "simulation_models"
            / "simulation-model-v1.json"
        )
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        artifact["event_frames"][0]["decision_frame"]["preferred_interest"] = "tampered"
        model_path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ProductError, "integrity"):
            self.service.send_conversation_message(
                self.person_id, "Should the team ignore user testing?"
            )

    def test_reordering_sources_does_not_change_deployed_prediction(self) -> None:
        first = self._confirm_interview()
        second = self.service.add_conversation_text_source(
            self.person_id,
            title="Second verified interview",
            text="Q: Should evidence be ignored?\nA: No. Evidence should guide the next test.",
            speaker="Alice Example",
            source_date="2025-02-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="transcript paragraphs 7-8",
            source_context="Recorded public interview",
            source_url="https://example.org/alice-interview-2",
        )
        self.service.review_conversation_source(
            self.person_id, str(second["source_id"]), "confirmed"
        )
        kernel = self.service.conversation._predictor
        events = self.service.conversation._trainable_events(
            self.person_id, [str(first["source_id"]), str(second["source_id"])]
        )
        left = kernel.fit(person_id=self.person_id, version=90, events=events)
        right = kernel.fit(person_id=self.person_id, version=91, events=list(reversed(events)))
        left_prediction = kernel.predict(left, text="Should evidence be ignored?", history=[])
        right_prediction = kernel.predict(right, text="Should evidence be ignored?", history=[])
        self.assertEqual(
            left_prediction["structured_prediction"]["stance_distribution"],
            right_prediction["structured_prediction"]["stance_distribution"],
        )
        self.assertEqual(
            left_prediction["prediction_trace"]["candidate_event_ids"],
            right_prediction["prediction_trace"]["candidate_event_ids"],
        )

    def test_same_kernel_exposes_correct_population_retrieval_and_frequency_baselines(self) -> None:
        self._confirm_interview()
        holdout = self.service.add_conversation_text_source(
            self.person_id,
            title="Later sealed interview",
            text="Q: Should a team launch when evidence is still weak?\nA: No. It should wait for better evidence.",
            speaker="Alice Example",
            source_date="2026-01-01",
            dataset_role="final_holdout",
            content_authenticity="verbatim_transcript",
            source_locator="later transcript paragraphs 1-2",
            source_context="Later recorded public interview",
            source_url="https://example.org/alice-holdout",
        )
        self.service.review_conversation_source(
            self.person_id, str(holdout["source_id"]), "confirmed"
        )
        report = self.service.conversation_summary(self.person_id)["baseline_report"]
        self.assertEqual(report["status"], "exploratory_not_confirmatory")
        self.assertEqual(report["sample_count"], 1)
        for baseline in (
            "correct_person",
            "population",
            "retrieval",
            "person_history_frequency",
            "recent_dynamic_population",
        ):
            self.assertIn(baseline, report)
        self.assertEqual(report["release_gate"], "not_passed")


if __name__ == "__main__":
    unittest.main()
