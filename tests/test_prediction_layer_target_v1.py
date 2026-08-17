from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from pcfm.services import PcfmService
from pcfm.response_prediction import (
    classify_event_type,
    classify_event_types,
    response_events_from_source,
    review_response_events,
    tokens,
)
from pcfm.response_prediction_v2 import ResponsePredictionKernelV2


def source(*, source_date: str = "2024-03-02") -> dict[str, object]:
    return {
        "person_id": "person-a",
        "source_id": "source-a",
        "speaker": "Person A",
        "source_date": source_date,
        "title": "Public policy interview",
        "source_context": "Recorded interview with a public audience",
        "source_url": "https://example.test/interview",
        "source_locator": "transcript paragraph 4",
        "filename": "",
        "dataset_role": "model_source",
        "content_authenticity": "verbatim_transcript",
        "original_language": "en",
        "translation_of": "",
        "speaker_scope": "single_speaker",
        "qas": [
            {
                "question": "Should government regulate artificial intelligence used in hospitals?",
                "answer": "Yes, when patient safety evidence shows serious risk, because safety must come before speed.",
                "locator": "qa:1",
            }
        ],
    }


def reviewed_events(raw: dict[str, object]) -> list[dict[str, object]]:
    raw["response_events"] = response_events_from_source(raw)
    return review_response_events(raw, "Person A", [])


class PredictionLayerTargetTests(unittest.TestCase):
    def test_classifier_uses_word_boundaries_multilabels_and_chinese_topics(self) -> None:
        self.assertEqual("general", classify_event_type("He said we should wait."))
        labels = classify_event_types(
            "Should government regulate artificial intelligence used in hospitals?"
        )
        self.assertIn("technology_science", labels)
        self.assertIn("governance_policy", labels)
        self.assertTrue(any(len(token) > 1 for token in tokens("人工智能监管")))

    def test_event_atom_declares_conditions_time_and_permitted_uses(self) -> None:
        event = response_events_from_source(source())[0]
        atom = event["event_atom"]
        self.assertEqual("known", atom["completeness"]["temporal_status"])
        self.assertIn("conditional_tendency", atom["completeness"]["allowed_uses"])
        self.assertIn("target", atom)
        self.assertIn("response_time", atom["temporal_context"])
        tendency = event["tendency_atoms"][0]
        for field in (
            "target",
            "direction",
            "conditions",
            "tradeoffs",
            "exceptions",
            "domain",
            "temporal_scope",
            "counterevidence_event_ids",
        ):
            self.assertIn(field, tendency)
        self.assertEqual("single_event_candidate", tendency["status"])

    def test_missing_time_is_explicit_and_cannot_train_or_form_tendency(self) -> None:
        raw = source(source_date="")
        event = response_events_from_source(raw)[0]
        atom = event["event_atom"]
        self.assertEqual("unknown", atom["completeness"]["temporal_status"])
        self.assertNotIn("conditional_tendency", atom["completeness"]["allowed_uses"])
        self.assertIn("temporal_prediction", atom["completeness"]["prohibited_uses"])
        raw["response_events"] = [event]
        reviewed = review_response_events(raw, "Person A", [])[0]
        self.assertEqual("feature_discovery", reviewed["data_role"])
        self.assertIn("response_time_missing", reviewed["training_rejection_reasons"])

    def test_model_aggregates_conditions_source_independence_and_knowledge(self) -> None:
        raw = source()
        event = reviewed_events(raw)[0]
        model = ResponsePredictionKernelV2().fit(
            person_id="person-a", version=1, events=[event]
        )
        tendency = model["conditional_tendencies"][0]
        self.assertEqual(1, tendency["independent_source_count"])
        self.assertTrue(tendency["conditions"])
        self.assertTrue(tendency["targets"])
        self.assertIn("counterevidence_event_ids", tendency)
        knowledge = model["demonstrated_knowledge"][0]
        self.assertEqual("person_demonstrated_claim_not_verified_fact", knowledge["knowledge_kind"])
        self.assertIn("temporal_status", knowledge)

    def test_retrieval_requires_event_scope_not_accidental_token_overlap(self) -> None:
        raw = source()
        event = reviewed_events(raw)[0]
        kernel = ResponsePredictionKernelV2()
        model = kernel.fit(person_id="person-a", version=1, events=[event])
        unrelated = kernel.recall(
            model,
            text="What did you say while waiting for a train?",
            history=[],
        )
        self.assertFalse(any(item["eligible_same_event"] for item in unrelated["candidates"]))
        related = kernel.recall(
            model,
            text="How should government regulate AI risks?",
            history=[],
        )
        self.assertTrue(any(item["eligible_same_event"] for item in related["candidates"]))

    def test_duplicate_source_does_not_inflate_and_conflict_is_not_averaged(self) -> None:
        same_source = source()
        same_source["qas"] = [
            *same_source["qas"],
            {
                "question": "Should government regulate medical software?",
                "answer": "Yes, when safety evidence shows serious risk.",
                "locator": "qa:2",
            },
        ]
        first_events = reviewed_events(same_source)

        opposing_source = source(source_date="2024-04-03")
        opposing_source["source_id"] = "source-b"
        opposing_source["source_url"] = "https://example.test/second-interview"
        opposing_source["qas"] = [
            {
                "question": "Should government regulate artificial intelligence?",
                "answer": "I oppose regulation because premature rules can block useful research.",
                "locator": "qa:1",
            }
        ]
        opposing_event = reviewed_events(opposing_source)[0]
        model = ResponsePredictionKernelV2().fit(
            person_id="person-a",
            version=1,
            events=[*first_events, opposing_event],
        )
        support = next(
            item
            for item in model["conditional_tendencies"]
            if item["stance"] == "conditional_support"
        )
        oppose = next(
            item
            for item in model["conditional_tendencies"]
            if item["stance"] == "oppose"
        )
        self.assertEqual(1, support["independent_source_count"])
        self.assertAlmostEqual(1.0 / 3.0, support["evidence_strength"], places=5)
        self.assertTrue(support["counterevidence_event_ids"])
        self.assertTrue(oppose["counterevidence_event_ids"])
        self.assertIn("contradicted", support["status"])

    def test_public_runtime_consumes_v4_frames_without_inventing_person_knowledge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = PcfmService(
                Path(directory), seed_example=False, seed_demos=True
            )
            reply = service.send_conversation_message(
                "demo-sally-ride",
                "Tell us what prompted you to write that note and describe the events that followed.",
            )
            service.close()
        self.assertEqual("simulation-v5", reply["prediction_trace"]["kernel"])
        self.assertTrue(reply["prediction_trace"]["selected_event_ids"])
        basis = reply["structured_prediction"]["response_basis"]
        self.assertEqual([], basis["selected_demonstrated_knowledge"])
        self.assertEqual(
            "exact_publicly_demonstrated_claims_only_not_complete_person_knowledge",
            basis["knowledge_boundary"],
        )

    def test_real_people_holdouts_report_baselines_without_accuracy_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = PcfmService(
                Path(directory), seed_example=False, seed_demos=True
            )
            reports = [
                service.conversation_summary(person_id)["baseline_report"]
                for person_id in ("demo-sally-ride", "demo-barack-obama")
            ]
            service.close()
        for report in reports:
            self.assertEqual("exploratory_not_confirmatory", report["status"])
            self.assertGreater(report["sample_count"], 0)
            self.assertIn("correct_person", report)
            self.assertIn("population", report)
            self.assertIn("retrieval", report)
            self.assertIn("person_history_frequency", report)
            self.assertTrue(report["wrong_people"])
            self.assertEqual("not_passed", report["release_gate"])


if __name__ == "__main__":
    unittest.main()
