from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.services import PcfmService


class VerifiedDemoPeopleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = PcfmService(
            self.storage, seed_example=False, seed_demos=True
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def test_seed_is_idempotent_and_creates_two_isolated_demo_people(self) -> None:
        people = {item["person_id"]: item for item in self.service.list_people()}
        self.assertEqual(
            {"demo-sally-ride", "demo-barack-obama"}, set(people)
        )
        self.assertTrue(all(item["is_demo"] for item in people.values()))
        expected_model_sources = {
            "demo-sally-ride": 1,
            "demo-barack-obama": 2,
        }
        for person_id in people:
            detail = self.service.get_person(person_id)
            summary = detail["conversation"]
            self.assertEqual(
                expected_model_sources[person_id],
                summary["source_counts"]["model_source"],
            )
            self.assertEqual(1, summary["source_counts"]["final_holdout"])
            self.assertIsNotNone(summary["active_version"])
            self.assertEqual(3, len(detail["recommended_questions"]))
            self.assertEqual(
                {"direct", "nearby", "out_of_scope"},
                {item["kind"] for item in detail["recommended_questions"]},
            )
        reloaded = PcfmService(
            self.storage, seed_example=False, seed_demos=True
        )
        self.assertEqual(2, len(reloaded.list_people()))
        obama = reloaded.conversation_summary("demo-barack-obama")
        self.assertEqual(2, obama["source_counts"]["model_source"])
        self.assertEqual(
            1,
            sum(
                source.get("source_url", "").endswith(
                    "/remarks-president-obama-and-prime-minister-lee-singapore-joint-press"
                )
                for source in obama["sources"]
            ),
        )
        reloaded.close()

    def test_each_demo_has_five_training_events_and_two_sealed_holdouts(self) -> None:
        for person_id in ("demo-sally-ride", "demo-barack-obama"):
            summary = self.service.conversation_summary(person_id)
            training_events = [
                event
                for source in summary["sources"]
                if source["dataset_role"] == "model_source"
                for event in source["response_events"]
                if event["data_role"] == "parameter_training"
            ]
            holdout_events = [
                event
                for source in summary["sources"]
                if source["dataset_role"] == "final_holdout"
                for event in source["response_events"]
                if event["data_role"] == "sealed_final_validation"
            ]
            self.assertGreaterEqual(len(training_events), 5)
            self.assertGreaterEqual(len(holdout_events), 2)
            version = summary["versions"][-1]
            style_path = (
                self.storage
                / "people"
                / person_id
                / str(version["style_artifact_path"])
            )
            style = json.loads(style_path.read_text(encoding="utf-8"))
            self.assertTrue(style["surface_rules"])
            self.assertTrue(
                set(style["source_event_ids"]).isdisjoint(
                    {event["event_id"] for event in holdout_events}
                )
            )

    def test_same_frozen_content_is_rendered_differently_without_semantic_change(self) -> None:
        contract = {
            "schema_version": "pcfm-frozen-content-contract-v2",
            "speech_act": "direct_answer",
            "stance": "conditional_support",
            "refusal_status": "not_refused",
            "claims": [{"id": "C1", "text": "The proposal can proceed."}],
            "reasons": [{"id": "R1", "text": "The evidence supports a limited trial."}],
            "memories": [],
            "uncertainties": [{"id": "U1", "text": "The long-term result remains uncertain."}],
            "protected_entities": [],
            "protected_numbers": [],
            "protected_dates": [],
            "protected_quotes": [],
            "confidence": 0.63,
            "style_mode": "interview_public",
        }
        rendered: dict[str, tuple[str, str, dict[str, object]]] = {}
        for person_id in ("demo-sally-ride", "demo-barack-obama"):
            rendered[person_id] = self.service.conversation._render_reply(
                person_id, contract
            )
            text, status, gate = rendered[person_id]
            self.assertEqual("person_style_applied", status)
            self.assertEqual("passed", gate["status"])
            for segment in (
                "The proposal can proceed.",
                "The evidence supports a limited trial.",
                "The long-term result remains uncertain.",
            ):
                self.assertEqual(1, text.count(segment))
        self.assertNotEqual(rendered["demo-sally-ride"][0], rendered["demo-barack-obama"][0])

    def test_direct_question_styles_answer_and_out_of_scope_does_not_invent_stance(self) -> None:
        for person_id in ("demo-sally-ride", "demo-barack-obama"):
            detail = self.service.get_person(person_id)
            direct_question = next(
                item["text"]
                for item in detail["recommended_questions"]
                if item["kind"] == "direct"
            )
            direct = self.service.send_conversation_message(
                person_id,
                direct_question,
                reality_lookup_requested=False,
            )
            self.assertEqual("answered", direct["status"])
            self.assertEqual(
                "source_verbatim_person_style", direct["style_status"]
            )
            self.assertTrue(direct["evidence"])
        inferred = self.service.send_conversation_message(
            "demo-sally-ride",
            "What investment strategy would you use for cryptocurrency in 2026?",
            reality_lookup_requested=False,
        )
        self.assertEqual("needs_model", inferred["status"])
        self.assertEqual("general_assisted", inferred["answer_status"])
        self.assertEqual("not_available", inferred["person_prediction_status"])
        self.assertEqual("simulation-v5", inferred["prediction_trace"]["kernel"])
        self.assertFalse(inferred["structured_prediction"]["claims"])
        telemetry = self.service.conversation_summary("demo-sally-ride")["telemetry"]
        self.assertEqual(0, telemetry["reality_lookup_requests"])
        self.assertEqual(0, telemetry["reality_local_search_calls"])
        self.assertEqual(0, telemetry["reality_online_search_calls"])

    def test_demo_archive_restore_preserves_sources_messages_and_versions(self) -> None:
        person_id = "demo-barack-obama"
        self.service.send_conversation_message(
            person_id,
            "Why did the agreement focus on Iran's nuclear program?",
        )
        before = self.service.conversation_summary(person_id)
        self.service.delete_person(person_id)
        self.assertIn(
            person_id,
            {item["person_id"] for item in self.service.list_archived_people()},
        )
        self.service.restore_person(person_id)
        after = self.service.conversation_summary(person_id)
        self.assertEqual(before["source_counts"], after["source_counts"])
        self.assertEqual(len(before["messages"]), len(after["messages"]))
        self.assertEqual(len(before["versions"]), len(after["versions"]))


if __name__ == "__main__":
    unittest.main()
