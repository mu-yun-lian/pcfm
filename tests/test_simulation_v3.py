from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pcfm.simulation_v3 import (
    MODEL_SCHEMA_V3,
    SimulationKernelV3,
    SimulationV3Error,
)
from pcfm.product_service import ProductService


def reviewed_source(
    source_id: str,
    *,
    question: str,
    answer: str,
    source_date: str = "2024-01-01",
    title: str = "Public interview",
    context: str = "Recorded public interview",
    role: str = "model_source",
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "person_id": "person-a",
        "review_status": "confirmed",
        "dataset_role": role,
        "content_authenticity": "verbatim_transcript",
        "speaker": "Person A",
        "source_date": source_date,
        "title": title,
        "source_context": context,
        "source_url": f"https://example.test/{source_id}",
        "source_locator": "transcript paragraph 1",
        "near_duplicate_of": None,
        "qas": [
            {"question": question, "answer": answer, "locator": "qa:1"}
        ],
        "segments": [],
        # Deliberately hostile V2 content: V3 must ignore it.
        "response_events": [
            {
                "stance": "oppose",
                "event_atom": {"event_type": "wrong-v2-type"},
                "tendency_atoms": [{"stance": "oppose"}],
            }
        ],
    }


class SimulationV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = SimulationKernelV3()
        self.first = reviewed_source(
            "source-a",
            question="Should hospitals deploy the system quickly?",
            answer=(
                "We should protect safety before speed because failures are irreversible."
            ),
            title="Hospital technology interview",
            context="Public hospital technology hearing",
        )
        self.second = reviewed_source(
            "source-b",
            question="How should a team launch a new product?",
            answer=(
                "Safety must come before speed because a harmful launch is costly."
            ),
            source_date="2024-02-01",
            title="Product launch interview",
            context="Public product strategy interview",
        )

    def fit(self, sources=None):
        return self.kernel.fit(
            person_id="person-a",
            version=1,
            reviewed_sources=sources or [self.first, self.second],
            scope={"language": "en", "time_scope": {"start": "2024-01-01"}},
        )

    def test_v3_builds_from_raw_spans_and_ignores_v2_predictions(self) -> None:
        artifact = self.fit()
        self.assertEqual(MODEL_SCHEMA_V3, artifact["schema_version"])
        self.assertEqual("reviewed_raw_sources_v1", artifact["input_contract"])
        self.assertNotIn("wrong-v2-type", str(artifact))
        frame = artifact["event_frames"][0]
        self.assertEqual("exact_source_span", frame["evidence"]["span_status"])
        self.assertIn(frame["observed_response"]["verbatim"], self.first["qas"][0]["answer"])
        self.assertEqual("safety", frame["decision_frame"]["preferred_interest"])
        self.assertEqual("speed", frame["decision_frame"]["accepted_cost"])

    def test_missing_time_is_rejected_from_fitting_without_guessing(self) -> None:
        undated = reviewed_source(
            "undated",
            question="What matters?",
            answer="Safety should come before speed.",
            source_date="",
        )
        artifact = self.fit([self.first, undated])
        self.assertEqual(["undated"], artifact["rejected_sources"]["missing_time"])
        self.assertTrue(
            all(frame["source_id"] != "undated" for frame in artifact["event_frames"])
        )

    def test_preference_structure_uses_tradeoff_not_stance_frequency(self) -> None:
        artifact = self.fit()
        structure = artifact["preference_structures"][0]
        self.assertEqual("safety", structure["protected_interest"])
        self.assertEqual("speed", structure["accepted_cost"])
        self.assertEqual(2, structure["independent_source_count"])
        self.assertEqual(2, structure["domain_count"])
        self.assertEqual("cross_domain_public_preference", structure["status"])
        self.assertNotIn("stance", structure)

    def test_same_source_repetition_does_not_create_cross_domain_structure(self) -> None:
        duplicate = copy.deepcopy(self.second)
        duplicate["source_id"] = "source-b-copy"
        duplicate["near_duplicate_of"] = "source-a"
        artifact = self.fit([self.first, duplicate])
        structure = artifact["preference_structures"][0]
        self.assertEqual(1, structure["independent_source_count"])
        self.assertNotEqual("cross_domain_public_preference", structure["status"])

    def test_reversed_tradeoff_is_counterevidence_not_average(self) -> None:
        counter = reviewed_source(
            "source-c",
            question="What matters in a reversible prototype?",
            answer=(
                "Speed should come before safety because cheap experiments can be reversed."
            ),
            source_date="2024-03-01",
            title="Prototype experiment interview",
            context="Public product experiment discussion",
        )
        artifact = self.fit([self.first, self.second, counter])
        safety = next(
            item
            for item in artifact["preference_structures"]
            if item["protected_interest"] == "safety"
        )
        self.assertTrue(safety["counterevidence_event_ids"])
        self.assertEqual("context_split_required", safety["conflict_status"])

    def test_direct_response_and_cross_domain_projection_use_v3_only(self) -> None:
        artifact = self.fit()
        direct = self.kernel.predict(
            artifact,
            text="Should hospitals deploy the system quickly?",
            history=[],
        )
        self.assertEqual("direct_answer", direct["answer_status"])
        self.assertEqual("simulation-v3", direct["prediction_trace"]["kernel"])

        projected = self.kernel.predict(
            artifact,
            text="Should an aviation team prioritize speed or safety?",
            history=[],
        )
        self.assertEqual("preference_structure_answer", projected["answer_status"])
        basis = projected["structured_prediction"]["response_basis"]
        self.assertEqual("value_conflict_projection", basis["path"])
        self.assertEqual("safety", basis["protected_interest"])
        self.assertEqual("speed", basis["accepted_cost"])
        self.assertIn("prioritize safety over speed", basis["prediction_statement"])

    def test_unmatched_question_has_no_person_stance_and_uses_general_path(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="How does a black hole form?",
            history=[],
        )
        self.assertEqual("general_assisted", result["answer_status"])
        structured = result["structured_prediction"]
        self.assertEqual("not_available", structured["response_basis"]["person_prediction_status"])
        self.assertEqual([], structured["claims"])
        self.assertEqual([], structured["reasons"])

    def test_order_invariance_integrity_and_old_schema_refusal(self) -> None:
        first = self.fit([self.first, self.second])
        second = self.fit([self.second, self.first])
        self.assertEqual(
            first["semantic_model_digest"], second["semantic_model_digest"]
        )
        tampered = copy.deepcopy(first)
        tampered["preference_structures"][0]["protected_interest"] = "profit"
        with self.assertRaisesRegex(SimulationV3Error, "integrity"):
            self.kernel.verify(tampered)
        old = copy.deepcopy(first)
        old["schema_version"] = "pcfm-unified-response-model-v2"
        with self.assertRaisesRegex(SimulationV3Error, "schema"):
            self.kernel.verify(old)

    def test_one_event_with_two_tradeoffs_is_not_duplicated(self) -> None:
        source = reviewed_source(
            "two-tradeoffs",
            question="How should the team decide?",
            answer=(
                "Safety should come before speed. "
                "Transparency should come before secrecy."
            ),
        )
        artifact = self.fit([source])
        self.assertEqual(1, len(artifact["event_frames"]))
        self.assertEqual(2, len(artifact["preference_atoms"]))
        self.assertEqual(
            2,
            artifact["event_frames"][0]["evidence_support"][
                "explicit_tradeoff_count"
            ],
        )

    def test_later_query_is_flagged_as_temporal_extrapolation(self) -> None:
        projected = self.kernel.predict(
            self.fit(),
            text="In 2030, should an aviation team prioritize speed or safety?",
            history=[],
        )
        basis = projected["structured_prediction"]["response_basis"]
        self.assertEqual(
            "later_than_evidence_window",
            basis["temporal_applicability"]["status"],
        )
        self.assertIn(
            "temporal stability is unknown",
            projected["structured_prediction"]["uncertainties"][0]["text"],
        )

    def test_accuracy_report_uses_only_later_explicit_holdout_tradeoffs(self) -> None:
        holdout = reviewed_source(
            "holdout",
            question="Should an aviation team prioritize speed or safety?",
            answer="Safety should come before speed because avoidable harm is costly.",
            source_date="2025-01-01",
            role="final_holdout",
        )
        report = self.kernel.evaluate(self.fit(), [holdout])
        self.assertEqual("assessed_exploratory", report["status"])
        self.assertEqual(1, report["sample_count"])
        self.assertEqual(1.0, report["coverage"])
        self.assertEqual(1.0, report["covered_direction_accuracy"])
        self.assertEqual([], report["holdout_leakage_source_ids"])


class SimulationV3ProductIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)
        self.service = ProductService(self.storage, seed_example=False)
        person = self.service.create_conversation_person(
            name="Person A",
            aliases=[],
            language="en",
            description="Synthetic public evidence",
            source_mode="user_provided",
            identity_note="Synthetic test person",
            focus_domain="public decisions",
        )
        self.person_id = str(person["person_id"])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _add(self, source_id: str, question: str, answer: str, date: str) -> None:
        source = self.service.add_conversation_text_source(
            self.person_id,
            title=source_id,
            text=f"Q: {question}\nA: {answer}",
            speaker="Person A",
            source_date=date,
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="qa:1",
            source_context=source_id,
            source_url=f"https://example.test/{source_id}",
        )
        self.service.review_conversation_source(
            self.person_id, str(source["source_id"]), "confirmed"
        )

    def test_product_runtime_does_not_promote_raw_v3_regex_tradeoffs(self) -> None:
        self._add(
            "hospital",
            "Should hospitals deploy the system quickly?",
            "Safety should come before speed because failures are irreversible.",
            "2024-01-01",
        )
        self._add(
            "product",
            "How should a product team launch?",
            "Safety should come before speed because a harmful launch is costly.",
            "2024-02-01",
        )
        reply = self.service.send_conversation_message(
            self.person_id,
            "Should an aviation team prioritize speed or safety?",
        )
        self.assertEqual("similar_event_evidence_answer", reply["answer_status"])
        self.assertEqual(
            "analogical_evidence_not_new_stance",
            reply["person_prediction_status"],
        )
        self.assertEqual("simulation-v5", reply["prediction_trace"]["kernel"])
        self.assertEqual(
            "pcfm_conversation_conditioned_response_simulation_v5", reply["model_kind"]
        )
        summary = self.service.conversation_summary(self.person_id)
        active = next(
            item
            for item in summary["versions"]
            if item["version"] == summary["active_version"]
        )
        self.assertTrue(active["simulation_model_path"])
        self.assertEqual("frozen_baseline_only", active["v2_response_model_role"])
        self.assertEqual("frozen_baseline_only", active["simulation_v3_role"])
        self.assertNotIn(
            "response_kernel_v2",
            reply["structured_prediction"]["active_components"],
        )
        self.assertNotIn(
            "simulation_v3",
            reply["structured_prediction"]["active_components"],
        )


if __name__ == "__main__":
    unittest.main()
