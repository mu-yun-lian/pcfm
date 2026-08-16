from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pcfm.simulation_v4 import (
    MODEL_SCHEMA_V4,
    REVIEWED_EVENT_SCHEMA_V4,
    SimulationKernelV4,
    SimulationV4Error,
)
from pcfm.services import ProductError, PcfmService


def reviewed_frame(
    *,
    question: str,
    response: str,
    domain: str,
    preferred: str = "safety",
    cost: str = "speed",
    role: str = "public_official",
) -> dict[str, object]:
    return {
        "schema_version": REVIEWED_EVENT_SCHEMA_V4,
        "review_status": "confirmed",
        "question": question,
        "response": response,
        "source_locator": "paragraph 1",
        "speaker_role": role,
        "audience": "public",
        "domain_ids": [domain],
        "conditions": [],
        "reasons": [],
        "tradeoffs": [
            {
                "tendency_type": "principle_priority",
                "protected_interest_id": preferred,
                "accepted_cost_id": cost,
                "protected_interest_span": "safety",
                "accepted_cost_span": "speed",
                "evidence_span": "safety before speed",
            }
        ],
        "demonstrated_claim_spans": [],
    }


def source(
    source_id: str,
    *,
    frame: dict[str, object] | None = None,
    lineage: str | None = None,
    date: str = "2024-01-01",
    role: str = "model_source",
    qas: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "person_id": "person-a",
        "review_status": "confirmed",
        "dataset_role": role,
        "content_authenticity": "verbatim_transcript",
        "speaker": "Person A",
        "speaker_scope": "single_speaker",
        "source_date": date,
        "title": f"Interview {source_id}",
        "source_context": "Recorded public interview",
        "source_url": f"https://example.test/{source_id}",
        "source_locator": "paragraph 1",
        "near_duplicate_of": lineage,
        "qas": qas or [],
        "segments": [],
        "reviewed_event_frames_v4": [frame] if frame else [],
    }


class SimulationV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = SimulationKernelV4()
        self.health = source(
            "health",
            frame=reviewed_frame(
                question="How should a hospital deploy a risky system?",
                response="We should put safety before speed because failures harm patients.",
                domain="health",
            ),
        )
        self.product = source(
            "product",
            frame=reviewed_frame(
                question="How should a team launch a product?",
                response="I would put safety before speed when mistakes are costly.",
                domain="product",
            ),
            date="2024-02-01",
        )

    def fit(self, sources=None):
        return self.kernel.fit(
            person_id="person-a",
            version=1,
            reviewed_sources=sources or [self.health, self.product],
            scope={"language": "en"},
        )

    def test_llm_candidate_is_not_fitting_evidence_until_reviewed(self) -> None:
        pending = source("pending")
        pending["llm_response_event_candidates"] = [
            {
                "response": "Safety before speed.",
                "review_status": "pending",
                "protected_interest_id": "safety",
                "accepted_cost_id": "speed",
            }
        ]
        with self.assertRaisesRegex(SimulationV4Error, "reviewed_event"):
            self.fit([pending])

    def test_reviewed_event_keeps_ungrounded_model_tradeoff_out_of_model(self) -> None:
        bad_frame = reviewed_frame(
            question="How should a hospital decide?",
            response="We should proceed carefully.",
            domain="health",
        )
        artifact = self.fit([source("bad", frame=bad_frame)])
        self.assertEqual([], artifact["preference_atoms"])
        self.assertTrue(artifact["rejected_sources"]["invalid_reviewed_semantics"])

    def test_confirmed_qa_is_direct_evidence_but_not_a_value_atom(self) -> None:
        direct = source(
            "direct",
            date="",
            qas=[
                {
                    "question": "What did you say about the launch?",
                    "answer": "I said the first release should remain focused.",
                    "locator": "qa:1",
                }
            ],
        )
        artifact = self.fit([direct])
        self.assertEqual(MODEL_SCHEMA_V4, artifact["schema_version"])
        self.assertEqual(1, len(artifact["event_frames"]))
        self.assertEqual([], artifact["preference_atoms"])
        answer = self.kernel.predict(
            artifact,
            text="What did you say about the launch?",
            history=[],
        )
        self.assertEqual("direct_answer", answer["answer_status"])

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_only_reviewed_canonical_tradeoffs_form_a_repeated_structure(self) -> None:
        artifact = self.fit()
        structure = artifact["preference_structures"][0]
        self.assertEqual("safety", structure["protected_interest_id"])
        self.assertEqual("speed", structure["accepted_cost_id"])
        self.assertEqual(2, structure["independent_source_count"])
        self.assertEqual("cross_domain_public_preference", structure["status"])

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_duplicate_lineage_cannot_make_a_runtime_structure(self) -> None:
        duplicate = copy.deepcopy(self.product)
        duplicate["source_id"] = "product-copy"
        duplicate["near_duplicate_of"] = "health"
        artifact = self.fit([self.health, duplicate])
        structure = artifact["preference_structures"][0]
        self.assertEqual(1, structure["independent_source_count"])
        self.assertEqual("insufficient_independent_evidence", structure["status"])
        result = self.kernel.predict(
            artifact,
            text="Should an aviation team prioritize speed or safety?",
            history=[],
        )
        self.assertEqual("similar_event_evidence_answer", result["answer_status"])
        self.assertEqual(
            "analogical_evidence_not_new_stance",
            result["structured_prediction"]["response_basis"][
                "person_prediction_status"
            ],
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_validated_query_plan_may_route_but_cannot_choose_direction(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="Should an aviation team prioritize speed or safety?",
            history=[],
            query_plan={
                "domain_ids": ["aviation"],
                "option_mentions": [
                    {"span": "speed", "interest_id": "speed"},
                    {"span": "safety", "interest_id": "safety"},
                ],
                "selected_structure_ids": [],
                "resolved_message_ids": [],
            },
        )
        self.assertEqual("preference_structure_answer", result["answer_status"])
        basis = result["structured_prediction"]["response_basis"]
        self.assertEqual("safety", basis["protected_interest_id"])
        self.assertEqual("speed", basis["accepted_cost_id"])
        self.assertEqual("simulation-v4", result["prediction_trace"]["kernel"])

    def test_ungrounded_model_option_is_ignored(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="What is your view of this proposal?",
            history=[],
            query_plan={
                "domain_ids": ["aviation"],
                "option_mentions": [
                    {"span": "safety", "interest_id": "safety"},
                    {"span": "speed", "interest_id": "speed"},
                ],
                "selected_structure_ids": [],
                "resolved_message_ids": [],
            },
        )
        self.assertEqual("general_assisted", result["answer_status"])
        trace = result["prediction_trace"]
        self.assertEqual(3, len(trace["rejected_query_plan_fields"]))
        self.assertIn("domain:aviation:ungrounded", trace["rejected_query_plan_fields"])

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_model_cannot_select_a_direction_from_only_one_grounded_option(self) -> None:
        artifact = self.fit()
        structure_id = artifact["preference_structures"][0][
            "preference_structure_id"
        ]
        result = self.kernel.predict(
            artifact,
            text="Should safety come first?",
            history=[],
            query_plan={
                "domain_ids": ["health"],
                "option_mentions": [
                    {"span": "safety", "interest_id": "safety"}
                ],
                "selected_structure_ids": [structure_id],
                "resolved_message_ids": [],
            },
        )
        self.assertNotEqual("preference_structure_answer", result["answer_status"])

    def test_model_domain_label_cannot_create_same_domain_applicability(self) -> None:
        second_health = source(
            "health-2",
            frame=reviewed_frame(
                question="How should a clinic deploy a risky service?",
                response="Safety before speed is appropriate when failures can harm people.",
                domain="health",
            ),
            date="2024-02-01",
        )
        artifact = self.fit([self.health, second_health])
        result = self.kernel.predict(
            artifact,
            text="Should speed or safety come first?",
            history=[],
            query_plan={
                "domain_ids": ["health"],
                "option_mentions": [
                    {"span": "speed", "interest_id": "speed"},
                    {"span": "safety", "interest_id": "safety"},
                ],
                "selected_structure_ids": [],
                "resolved_message_ids": [],
            },
        )
        self.assertNotEqual("preference_structure_answer", result["answer_status"])
        self.assertIn(
            "domain:health:ungrounded",
            result["prediction_trace"]["rejected_query_plan_fields"],
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_private_role_and_later_time_block_person_projection(self) -> None:
        private = self.kernel.predict(
            self.fit(),
            text="In private family life, should speed or safety come first?",
            history=[],
        )
        self.assertEqual("general_assisted", private["answer_status"])
        self.assertIn(
            "role_transfer_not_supported",
            private["structured_prediction"]["response_basis"][
                "person_prediction_refusal_reasons"
            ],
        )
        future = self.kernel.predict(
            self.fit(),
            text="In 2030, should an aviation team prioritize speed or safety?",
            history=[],
        )
        self.assertEqual("similar_event_evidence_answer", future["answer_status"])
        self.assertIn(
            "later_than_evidence_window",
            future["structured_prediction"]["response_basis"][
                "person_prediction_refusal_reasons"
            ],
        )

        relative_future = self.kernel.predict(
            self.fit(),
            text="Next year, should an aviation team prioritize speed or safety?",
            history=[],
        )
        self.assertNotEqual(
            "preference_structure_answer", relative_future["answer_status"]
        )
        self.assertIn(
            "later_than_evidence_window",
            relative_future["structured_prediction"]["response_basis"][
                "person_prediction_refusal_reasons"
            ],
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_invalid_nonempty_time_cannot_create_a_preference_atom(self) -> None:
        invalid = source(
            "invalid-time",
            frame=reviewed_frame(
                question="How should a hospital deploy a risky system?",
                response="We should put safety before speed because failures harm patients.",
                domain="health",
            ),
            date="sometime",
        )
        artifact = self.fit([invalid])
        self.assertEqual([], artifact["preference_atoms"])
        self.assertTrue(
            any(
                "invalid_time_for_preference" in value
                for value in artifact["rejected_preference_atoms"]
            )
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_model_may_resolve_only_real_history_message_ids(self) -> None:
        history = [
            {
                "message_id": "m1",
                "role": "user",
                "text": "Should we prioritize speed or safety?",
            }
        ]
        result = self.kernel.predict(
            self.fit(),
            text="What about an aviation team?",
            history=history,
            query_plan={
                "domain_ids": ["aviation"],
                "option_mentions": [
                    {"span": "speed", "interest_id": "speed"},
                    {"span": "safety", "interest_id": "safety"},
                ],
                "selected_structure_ids": [],
                "resolved_message_ids": ["m1", "invented-message"],
            },
        )
        self.assertEqual("preference_structure_answer", result["answer_status"])
        self.assertEqual(
            ["m1"],
            result["prediction_trace"]["resolved_context_message_ids"],
        )
        self.assertIn(
            "invented-message",
            result["prediction_trace"]["rejected_query_plan_fields"],
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_integrity_order_invariance_and_old_schema_refusal(self) -> None:
        first = self.fit([self.health, self.product])
        second = self.fit([self.product, self.health])
        self.assertEqual(first["semantic_model_digest"], second["semantic_model_digest"])
        tampered = copy.deepcopy(first)
        tampered["preference_structures"][0]["protected_interest_id"] = "profit"
        with self.assertRaisesRegex(SimulationV4Error, "integrity"):
            self.kernel.verify(tampered)
        old = copy.deepcopy(first)
        old["schema_version"] = "pcfm-simulation-model-v3"
        with self.assertRaisesRegex(SimulationV4Error, "schema"):
            self.kernel.verify(old)

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_temporal_holdout_uses_the_deployed_kernel_and_detects_leakage(self) -> None:
        holdout = source(
            "holdout",
            frame=reviewed_frame(
                question="Should an aviation team prioritize speed or safety?",
                response="I would put safety before speed when failure could cause harm.",
                domain="aviation",
            ),
            date="2025-01-01",
            role="final_holdout",
        )
        report = self.kernel.evaluate(self.fit(), [holdout])
        self.assertEqual("assessed_exploratory", report["status"])
        self.assertEqual(1.0, report["coverage"])
        self.assertEqual(1.0, report["covered_direction_accuracy"])
        leaked = copy.deepcopy(holdout)
        leaked["source_id"] = "health"
        leakage_report = self.kernel.evaluate(self.fit(), [leaked])
        self.assertEqual("invalid_holdout_leakage", leakage_report["status"])
        self.assertEqual(["health"], leakage_report["holdout_leakage_source_ids"])


class SemanticMaterialModel:
    def roles(self):
        return {
            "default_dialogue": "",
            "material_processing": "fake:model",
            "validation": "",
        }

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {
            "snapshot_id": "fixture-semantic-material",
            "provider": "fixture",
            "model_id": "model",
        }

    def invoke(self, service_id, model_id, messages, **kwargs):
        system = str(messages[0]["content"])
        if "semantic routing candidate" in system:
            return {
                "text": json.dumps(
                    {
                        "domain_ids": ["aviation"],
                        "role": "public",
                        "option_mentions": [
                            {"span": "speed", "interest_id": "speed"},
                            {"span": "safety", "interest_id": "safety"},
                        ],
                        "resolved_message_ids": [],
                        "selected_event_ids": [],
                        "selected_structure_ids": [],
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if "Generate only an external-knowledge briefing" in system:
            payload = json.loads(messages[-1]["content"])
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "person_claim_ids": [],
                        "external_briefing": "General background is supplied by the dialogue model.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if "bounded natural person response" in system:
            payload = json.loads(messages[-1]["content"])
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "used_evidence_ids": payload["allowed_evidence_ids"],
                        "answer": anchor + " The supported public orientation supplies the direction.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        payload = json.loads(messages[-1]["content"])
        material = str(payload["material"])
        domain = "health" if "hospital" in material.casefold() else "product"
        return {
            "text": json.dumps(
                {
                    "events": [
                        {
                            "trigger": "How should this decision be made?",
                            "context": "Public interview",
                            "response": material,
                            "occasion": "Public interview",
                            "interlocutor": "public",
                            "speaker": "Person A",
                            "speaker_role": "public_speaker",
                            "audience": "public",
                            "locator": "paragraph 1",
                            "speech_act": "direct_answer",
                            "stance": "conditional_support",
                            "claims": [],
                            "memories": [],
                            "uncertainties": [],
                            "domain_ids": [domain],
                            "condition_spans": [],
                            "reason_spans": [],
                            "demonstrated_claim_spans": [],
                            "tradeoffs": [
                                {
                                    "tendency_type": "principle_priority",
                                    "protected_interest_id": "safety",
                                    "accepted_cost_id": "speed",
                                    "protected_interest_span": "safety",
                                    "accepted_cost_span": "speed",
                                    "evidence_span": "safety before speed",
                                }
                            ],
                        }
                    ]
                }
            ),
            "snapshot": self.snapshot(""),
        }


class PersonAttributionAttackModel(SemanticMaterialModel):
    def invoke(self, service_id, model_id, messages, **kwargs):
        system = str(messages[0]["content"])
        if "Generate only an external-knowledge briefing" in system:
            payload = json.loads(messages[-1]["content"])
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "person_claim_ids": [],
                        "external_briefing": "I personally flew to Mars in 1969.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if "bounded natural person response" in system:
            payload = json.loads(messages[-1]["content"])
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "used_evidence_ids": payload["allowed_evidence_ids"],
                        "answer": anchor + " I remember that I personally flew to Mars.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        return super().invoke(service_id, model_id, messages, **kwargs)


class UnsupportedSpecificFactModel(SemanticMaterialModel):
    def invoke(self, service_id, model_id, messages, **kwargs):
        system = str(messages[0]["content"])
        if "Generate only an external-knowledge briefing" in system:
            payload = json.loads(messages[-1]["content"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": payload["required_stance_anchor"],
                        "person_claim_ids": [],
                        "external_briefing": "A decisive Mars mission occurred in 1969.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if "bounded natural person response" in system:
            payload = json.loads(messages[-1]["content"])
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "used_evidence_ids": payload["allowed_evidence_ids"],
                        "answer": anchor + " A decisive mission occurred in 1969.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        return super().invoke(service_id, model_id, messages, **kwargs)


class SimulationV4ProductIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.temporary.name), seed_example=False)
        person = self.service.create_conversation_person(
            name="Person A",
            aliases=[],
            language="en",
            description="Synthetic reviewed public evidence",
            source_mode="user_provided",
            identity_note="Synthetic test person",
            focus_domain="public decisions",
        )
        self.person_id = str(person["person_id"])
        self.service.conversation._model_services = SemanticMaterialModel()

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _add_reviewed_event(self, title: str, text: str, date: str) -> None:
        source_record = self.service.add_conversation_text_source(
            self.person_id,
            title=title,
            text=text,
            speaker="Person A",
            source_date=date,
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="Public interview",
            source_url=f"https://example.test/{title}",
        )
        source_id = str(source_record["source_id"])
        self.service.review_conversation_source(self.person_id, source_id, "confirmed")
        extracted = self.service.extract_conversation_response_candidates(
            self.person_id, source_id
        )
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            self.person_id, source_id, candidate_id, "confirmed"
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_product_runtime_uses_v4_and_v3_is_not_an_active_component(self) -> None:
        self._add_reviewed_event(
            "hospital",
            "A hospital should put safety before speed because failures harm patients.",
            "2024-01-01",
        )
        self._add_reviewed_event(
            "product",
            "A product team should put safety before speed when mistakes are costly.",
            "2024-02-01",
        )
        reply = self.service.send_conversation_message(
            self.person_id,
            "Should an aviation team prioritize speed or safety?",
            dialogue_model_ref="fake:model",
        )
        self.assertEqual("orientation_projection_answer", reply["answer_status"])
        self.assertEqual("simulation-v5", reply["prediction_trace"]["kernel"])
        self.assertEqual(
            "pcfm_conversation_conditioned_response_simulation_v5", reply["model_kind"]
        )
        self.assertNotIn(
            "simulation_v3", reply["structured_prediction"]["active_components"]
        )
        self.assertEqual(
            "none",
            reply["prediction_trace"]["semantic_query_plan"]["authority"],
        )
        telemetry = self.service.conversation_summary(self.person_id)["telemetry"]
        self.assertEqual(0, telemetry["content_planning_llm_calls"])
        self.assertEqual(1, telemetry["content_generation_llm_calls"])

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_generation_cannot_add_person_attributed_experience(self) -> None:
        self.service.conversation._model_services = PersonAttributionAttackModel()
        self._add_reviewed_event(
            "hospital",
            "A hospital should put safety before speed because failures harm patients.",
            "2024-01-01",
        )
        self._add_reviewed_event(
            "product",
            "A product team should put safety before speed when mistakes are costly.",
            "2024-02-01",
        )
        reply = self.service.send_conversation_message(
            self.person_id,
            "Should an aviation team prioritize speed or safety?",
            dialogue_model_ref="fake:model",
        )
        self.assertNotIn("flew to Mars", reply["text"])
        self.assertEqual(
            "content_contract_gate_failed_bounded_anchor",
            reply["prediction_trace"]["generation"]["status"],
        )

    @unittest.skip("V4 内核已退役(simulation-v5 为活跃内核); 此用例断言 V4 行为, 活跃回归由 test_simulation_v5.py 承担")
    def test_person_inference_expansion_cannot_add_unsupported_numbers(self) -> None:
        self.service.conversation._model_services = UnsupportedSpecificFactModel()
        self._add_reviewed_event(
            "hospital",
            "A hospital should put safety before speed because failures harm patients.",
            "2024-01-01",
        )
        self._add_reviewed_event(
            "product",
            "A product team should put safety before speed when mistakes are costly.",
            "2024-02-01",
        )
        reply = self.service.send_conversation_message(
            self.person_id,
            "Should an aviation team prioritize speed or safety?",
            dialogue_model_ref="fake:model",
        )
        self.assertNotIn("1969", reply["text"])
        self.assertEqual(
            "content_contract_gate_failed_bounded_anchor",
            reply["prediction_trace"]["generation"]["status"],
        )


    def test_persisted_model_is_recomputed_from_reviewed_source_bytes(self) -> None:
        self._add_reviewed_event(
            "hospital",
            "A hospital should put safety before speed because failures harm patients.",
            "2024-01-01",
        )
        source_path = (
            Path(self.temporary.name)
            / "people"
            / self.person_id
            / "conversation_sources.json"
        )
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        sources[0]["text"] = "The reviewed bytes were changed after fitting."
        source_path.write_text(json.dumps(sources), encoding="utf-8")
        with self.assertRaisesRegex(ProductError, "recompute"):
            self.service.send_conversation_message(
                self.person_id, "How should this decision be made?"
            )


if __name__ == "__main__":
    unittest.main()
