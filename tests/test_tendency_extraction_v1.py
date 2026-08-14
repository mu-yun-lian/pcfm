"""Tendency-type contract: public-response tendency atoms (tradeoffs) must carry a
tendency_type from the closed 8-class taxonomy, validated by code."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.product_service import ProductService
from pcfm.response_prediction import TENDENCY_TYPES


class TradeoffExtractionModel:
    """Deterministic LLM stub returning one event with one tradeoff."""

    def __init__(self, tendency_type: str) -> None:
        self.tendency_type = tendency_type

    def roles(self):
        return {"default_dialogue": "", "material_processing": "fake:model", "validation": ""}

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {"snapshot_id": "fixture", "provider": "fixture", "model_id": "model"}

    def invoke(self, service_id, model_id, messages, **kwargs):
        return {
            "text": json.dumps(
                {
                    "events": [
                        {
                            "trigger": "Should we ship fast or safe?",
                            "context": "Public product talk",
                            "response": "We should prioritize safety over speed.",
                            "occasion": "Public product talk",
                            "interlocutor": "audience",
                            "speaker": "Alice Example",
                            "locator": "paragraph 1",
                            "speech_act": "direct_answer",
                            "stance": "support",
                            "claims": ["We should prioritize safety over speed."],
                            "reasons": [],
                            "memories": [],
                            "uncertainties": [],
                            "tradeoffs": [
                                {
                                    "tendency_type": self.tendency_type,
                                    "protected_interest_id": "safety",
                                    "accepted_cost_id": "speed",
                                    "protected_interest_span": "safety",
                                    "accepted_cost_span": "speed",
                                    "evidence_span": "prioritize safety over speed",
                                }
                            ],
                        }
                    ]
                }
            ),
            "snapshot": self.snapshot(""),
        }


class TendencyExtractionV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = ProductService(Path(self.temporary.name), seed_example=False)
        self.alice = self.service.create_conversation_person(
            name="Alice Example", aliases=["Alice"], language="en"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _add_source(self) -> str:
        person_id = str(self.alice["person_id"])
        source = self.service.add_conversation_text_source(
            person_id,
            title="Public product talk",
            text="We should prioritize safety over speed. Test the hardest assumption before expanding.",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="Public product talk",
            source_url="https://example.org/talk",
        )
        self.service.review_conversation_source(person_id, str(source["source_id"]), "confirmed")
        return str(source["source_id"])

    def test_tendency_types_vocabulary_has_eight_classes(self) -> None:
        self.assertEqual(8, len(TENDENCY_TYPES))
        self.assertEqual(8, len(set(TENDENCY_TYPES)))

    def test_valid_tendency_type_is_promoted_and_carried(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = TradeoffExtractionModel("principle_priority")
        source_id = self._add_source()
        extracted = self.service.extract_conversation_response_candidates(person_id, source_id)
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        reviewed = self.service.review_conversation_response_candidate(
            person_id, source_id, candidate_id, "confirmed"
        )
        frame = reviewed["reviewed_event_frames_v4"][-1]
        self.assertEqual("principle_priority", frame["tradeoffs"][0]["tendency_type"])
        self.assertEqual("safety", frame["tradeoffs"][0]["protected_interest_id"])

    def test_invalid_tendency_type_is_rejected(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = TradeoffExtractionModel("not_a_tendency_type")
        source_id = self._add_source()
        extracted = self.service.extract_conversation_response_candidates(person_id, source_id)
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        with self.assertRaises(Exception):
            self.service.review_conversation_response_candidate(
                person_id, source_id, candidate_id, "confirmed"
            )

    def test_tendency_type_flows_into_model_artifact(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = TradeoffExtractionModel("principle_priority")
        source_id = self._add_source()
        extracted = self.service.extract_conversation_response_candidates(person_id, source_id)
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            person_id, source_id, candidate_id, "confirmed"
        )
        summary = self.service.conversation_summary(person_id)
        version = int(summary["active_version"])
        model = self.service.conversation._simulation_model(person_id, version)
        atoms = model["reviewed_public_model"]["preference_atoms"]
        self.assertTrue(atoms)
        self.assertEqual("principle_priority", atoms[0]["tendency_type"])
        structures = model["reviewed_public_model"]["preference_structures"]
        self.assertTrue(structures)
        self.assertIn("principle_priority", structures[0]["tendency_types"])


if __name__ == "__main__":
    unittest.main()
