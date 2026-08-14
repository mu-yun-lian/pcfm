import json
import tempfile
import unittest
from pathlib import Path

from pcfm.model_services import ModelServiceError
from pcfm.product_service import ProductService
from pcfm.simulation_v5 import (
    SimulationKernelV5,
    SimulationV5Error,
)


def reviewed_frame(question: str, response: str, domain: str) -> dict[str, object]:
    return {
        "schema_version": "pcfm-reviewed-public-response-event-v4",
        "review_status": "confirmed",
        "question": question,
        "response": response,
        "source_locator": "paragraph 1",
        "speaker_role": "public_speaker",
        "audience": "public",
        "domain_ids": [domain],
        "conditions": ["when failure can harm people"],
        "reasons": ["failures can harm people"],
        "demonstrated_claim_spans": [],
        "tradeoffs": [
            {
                "protected_interest_id": "safety",
                "accepted_cost_id": "speed",
                "protected_interest_span": "safety",
                "accepted_cost_span": "speed",
                "evidence_span": "safety before speed",
            }
        ],
    }


def source(source_id: str, *, domain: str, question: str, response: str) -> dict[str, object]:
    frame = reviewed_frame(question, response, domain)
    return {
        "source_id": source_id,
        "person_id": "person-a",
        "review_status": "confirmed",
        "dataset_role": "model_source",
        "content_authenticity": "verbatim_transcript",
        "source_date": "2024-01-01" if source_id == "health" else "2024-02-01",
        "speaker": "Person A",
        "speaker_scope": "candidate_span_confirmed",
        "source_context": "Public interview",
        "source_url": f"https://example.test/{source_id}",
        "text": response,
        "qas": [],
        "segments": [{"text": response, "locator": "paragraph 1"}],
        "reviewed_event_frames_v4": [frame],
    }


class SimulationV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.kernel = SimulationKernelV5()
        self.sources = [
            source(
                "health",
                domain="health",
                question="How should a hospital deploy a risky service?",
                response="A hospital should put safety before speed because failures can harm people.",
            ),
            source(
                "product",
                domain="product",
                question="How should a product team release a risky service?",
                response="A product team should put safety before speed when mistakes can harm people.",
            ),
        ]

    def fit(self):
        return self.kernel.fit(
            person_id="person-a",
            version=1,
            reviewed_sources=self.sources,
            scope={"language": "en"},
        )

    def test_current_message_is_a_delta_to_selected_prior_topic(self) -> None:
        history = [
            {"message_id": "m1", "role": "user", "text": "Should safety come before speed?"},
            {
                "message_id": "m2",
                "role": "assistant",
                "text": "I would put safety first.",
                "context_role": "model_generated_context",
            },
            {"message_id": "m3", "role": "user", "text": "What did you have for breakfast?"},
            {
                "message_id": "m4",
                "role": "assistant",
                "text": "I do not have evidence for a personal breakfast.",
                "context_role": "model_generated_context",
            },
        ]
        for index in range(5, 25, 2):
            history.extend(
                [
                    {
                        "message_id": f"m{index}",
                        "role": "user",
                        "text": f"Unrelated breakfast detail {index}.",
                    },
                    {
                        "message_id": f"m{index + 1}",
                        "role": "assistant",
                        "text": "That remains unrelated context.",
                        "context_role": "model_generated_context",
                    },
                ]
            )
        result = self.kernel.predict(
            self.fit(),
            text="Back to that issue: what about an aviation team?",
            history=history,
            conversation_context={"active_topic_id": "topic-m1"},
            query_plan={
                "resolved_message_ids": ["m1"],
                "domain_ids": ["aviation"],
                "scenario_effects": [
                    {"interest_id": "safety", "effect": "advances", "scenario_span": "safety"},
                    {"interest_id": "speed", "effect": "constrains", "scenario_span": "speed"},
                ],
                "selected_event_ids": [],
                "selected_structure_ids": [],
            },
        )
        self.assertEqual("orientation_projection_answer", result["answer_status"])
        query = result["structured_prediction"]["response_basis"]["query_frame"]
        self.assertEqual(["m1"], query["resolved_message_ids"])
        self.assertIn("Should safety come before speed?", query["combined_query"])
        self.assertNotIn("breakfast", query["combined_query"])
        self.assertFalse(
            result["prediction_trace"]["context_used"]["generated_context_is_fitting_evidence"]
        )

    def test_model_may_map_scenario_but_cannot_supply_person_stance(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="Would you support a safety rule even if it slows deployment?",
            history=[],
            query_plan={
                "predicted_stance": "oppose",
                "scenario_effects": [
                    {"interest_id": "safety", "effect": "advances", "scenario_span": "safety rule"},
                    {"interest_id": "speed", "effect": "constrains", "scenario_span": "slows deployment"},
                ],
            },
        )
        self.assertEqual("orientation_projection_answer", result["answer_status"])
        self.assertEqual(
            "support",
            result["structured_prediction"]["stance"]["label"],
        )
        self.assertIn(
            "predicted_stance",
            result["prediction_trace"]["rejected_query_plan_fields"],
        )

    def test_ambiguous_short_followup_requires_resolution(self) -> None:
        history = [
            {"message_id": "m1", "role": "user", "text": "Tell me about product safety."},
            {"message_id": "m2", "role": "assistant", "text": "Safety matters."},
            {"message_id": "m3", "role": "user", "text": "Tell me about public budgets."},
            {"message_id": "m4", "role": "assistant", "text": "Budgets impose constraints."},
        ]
        result = self.kernel.predict(
            self.fit(), text="那呢？", history=history, query_plan={}
        )
        self.assertEqual("clarification_needed", result["answer_status"])

    def test_model_cannot_attach_unseen_value_spans_to_an_unrelated_question(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="What did you have for breakfast today?",
            history=[],
            query_plan={
                "scenario_effects": [
                    {
                        "interest_id": "safety",
                        "effect": "advances",
                        "scenario_span": "safety rule",
                    },
                    {
                        "interest_id": "speed",
                        "effect": "constrains",
                        "scenario_span": "slow deployment",
                    },
                ]
            },
        )
        self.assertEqual("general_assisted", result["answer_status"])
        self.assertEqual([], result["structured_prediction"]["claims"])
        self.assertTrue(
            all(
                value.startswith("scenario_effect:")
                for value in result["prediction_trace"]["rejected_query_plan_fields"]
            )
        )

    def test_fixed_width_navigation_chunks_are_not_response_episodes(self) -> None:
        raw = dict(self.sources[0])
        raw["reviewed_event_frames_v4"] = []
        raw["qas"] = []
        raw["segments"] = [{"text": "x" * 1200, "locator": "text segment 1"}]
        raw["text"] = "x" * 1200
        with self.assertRaisesRegex(SimulationV5Error, "no_eligible_reviewed_episode"):
            self.kernel.fit(
                person_id="person-a",
                version=1,
                reviewed_sources=[raw],
            )

    def test_missing_time_stays_unknown_and_blocks_temporal_aggregation_only(self) -> None:
        raw = {
            "source_id": "qa-undated",
            "person_id": "person-a",
            "review_status": "confirmed",
            "dataset_role": "model_source",
            "content_authenticity": "verbatim_transcript",
            "source_date": "",
            "speaker": "Person A",
            "speaker_scope": "single_speaker",
            "source_context": "Public interview",
            "text": "Q: What matters?\nA: Evidence matters.",
            "qas": [
                {
                    "question": "What matters?",
                    "answer": "Evidence matters.",
                    "locator": "qa:1",
                }
            ],
            "segments": [],
            "reviewed_event_frames_v4": [],
        }
        artifact = self.kernel.fit(
            person_id="person-a", version=1, reviewed_sources=[raw]
        )
        context = artifact["event_frames"][0]["episode_context"]
        self.assertEqual("unknown", context["response_time"])
        self.assertIn("response_time", context["missing_fields"])
        self.assertFalse(context["temporal_aggregation_eligible"])
        result = self.kernel.predict(
            artifact, text="What matters?", history=[], query_plan={}
        )
        self.assertEqual("direct_answer", result["answer_status"])

    def test_person_opinion_request_never_degrades_to_general_encyclopedia(self) -> None:
        result = self.kernel.predict(
            self.fit(),
            text="What do you think of Donald Trump?",
            history=[],
            query_plan={},
        )
        self.assertEqual("refused", result["answer_status"])
        self.assertIn(
            "person_opinion_evidence_required",
            result["structured_prediction"]["refusal_reasons"],
        )

    def test_single_event_creates_low_confidence_public_orientation(self) -> None:
        raw = {
            "source_id": "trump-assessment",
            "person_id": "person-a",
            "review_status": "confirmed",
            "dataset_role": "model_source",
            "content_authenticity": "verified_quote",
            "source_date": "2016-08-02",
            "speaker": "Person A",
            "speaker_scope": "single_speaker",
            "source_context": "Official press conference",
            "source_url": "https://example.test/trump-assessment",
            "text": (
                "Q: Does it make you question his fitness to be President?\n"
                "A: Yes, I think the Republican nominee is unfit to serve as President."
            ),
            "qas": [
                {
                    "question": "Does it make you question his fitness to be President?",
                    "answer": "Yes, I think the Republican nominee is unfit to serve as President.",
                    "locator": "official transcript",
                }
            ],
            "segments": [],
            "reviewed_event_frames_v4": [],
        }
        artifact = self.kernel.fit(
            person_id="person-a", version=1, reviewed_sources=[raw]
        )
        self.assertTrue(artifact["value_atoms"])
        orientation = next(
            item
            for item in artifact["value_orientation_index"]
            if item["interest_id"] == "competence"
        )
        self.assertEqual("single_source_public_salience", orientation["status"])
        self.assertLess(orientation["support"], 0.6)
        projected = self.kernel.predict(
            artifact,
            text="What do you think of this candidate?",
            history=[],
            query_plan={
                "domain_ids": ["governance"],
                "scenario_effects": [
                    {
                        "interest_id": "competence",
                        "effect": "threatens",
                        "scenario_span": "this candidate",
                    }
                ],
            },
        )
        self.assertEqual("orientation_projection_answer", projected["answer_status"])
        self.assertEqual("oppose", projected["structured_prediction"]["stance"]["label"])

    def test_cross_language_entity_alias_retrieves_one_reviewed_person_event(self) -> None:
        trump_source = source(
            "trump",
            domain="governance",
            question="Does it make you question his fitness to be President?",
            response=(
                "Yes, I think the Republican nominee is unfit to serve as President."
            ),
        )
        trump_source["entity_aliases"] = [
            "Donald Trump",
            "Trump",
            "特朗普",
        ]
        artifact = self.kernel.fit(
            person_id="person-a",
            version=1,
            reviewed_sources=[trump_source],
            scope={"language": "en"},
        )
        result = self.kernel.predict(
            artifact,
            text="你认为特朗普怎么样",
            history=[],
            conversation_context={},
            query_plan={},
        )
        self.assertEqual("similar_event_evidence_answer", result["answer_status"])
        self.assertEqual(
            "similar_event_evidence", result["prediction_trace"]["prediction_path"]
        )
        self.assertEqual(
            1, len(result["structured_prediction"]["evidence_event_ids"])
        )


class V5ModelFixture:
    def roles(self):
        return {
            "default_dialogue": "fake:model",
            "material_processing": "fake:model",
            "validation": "",
        }

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {
            "snapshot_id": "fixture-v5",
            "provider": "fixture",
            "model_id": "model",
        }

    def invoke(self, service_id, model_id, messages, **kwargs):
        system = str(messages[0]["content"])
        payload = json.loads(str(messages[-1]["content"]))
        if "semantic routing candidate" in system:
            return {
                "text": json.dumps(
                    {
                        "resolved_message_ids": [],
                        "domain_ids": ["aviation"],
                        "scenario_effects": [
                            {"interest_id": "safety", "effect": "advances", "scenario_span": "safety rule"},
                            {"interest_id": "speed", "effect": "constrains", "scenario_span": "slows deployment"},
                        ],
                        "selected_event_ids": [],
                        "selected_structure_ids": [],
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if "bounded natural person response" in system:
            anchor = str(payload["required_stance_anchor"])
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "used_evidence_ids": payload["allowed_evidence_ids"],
                        "answer": anchor + " The avoidable harm matters more than speed here.",
                    }
                ),
                "snapshot": self.snapshot(""),
            }
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


class RetryingOpinionFixture(V5ModelFixture):
    def invoke(self, service_id, model_id, messages, **kwargs):
        system = str(messages[0]["content"])
        if "semantic routing candidate" not in system:
            return super().invoke(service_id, model_id, messages, **kwargs)
        if kwargs.get("structured"):
            raise ModelServiceError("structured response was empty")
        payload = json.loads(str(messages[-1]["content"]))
        event_id = str(payload["event_candidates"][0]["event_frame_id"])
        return {
            "text": json.dumps(
                {
                    "resolved_message_ids": [],
                    "domain_ids": ["governance"],
                    "scenario_effects": [],
                    "selected_event_ids": [event_id],
                    "selected_structure_ids": [],
                }
            ),
            "snapshot": self.snapshot(""),
        }


class SimulationV5ProductIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = ProductService(Path(self.temporary.name), seed_example=False)
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
        self.service.conversation._model_services = V5ModelFixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def add_event(self, title: str, text: str, date: str) -> None:
        record = self.service.add_conversation_text_source(
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
        source_id = str(record["source_id"])
        self.service.review_conversation_source(self.person_id, source_id, "confirmed")
        extracted = self.service.extract_conversation_response_candidates(
            self.person_id, source_id
        )
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            self.person_id, source_id, candidate_id, "confirmed"
        )

    def test_product_uses_semantic_model_before_general_fallback(self) -> None:
        self.add_event(
            "hospital",
            "A hospital should put safety before speed because failures can harm people.",
            "2024-01-01",
        )
        self.add_event(
            "product",
            "A product team should put safety before speed when mistakes can harm people.",
            "2024-02-01",
        )
        reply = self.service.send_conversation_message(
            self.person_id,
            "Would you support a new aviation safety rule if it slows deployment?",
            dialogue_model_ref="fake:model",
        )
        self.assertEqual("orientation_projection_answer", reply["answer_status"])
        self.assertEqual("simulation-v5", reply["prediction_trace"]["kernel"])
        self.assertEqual(1, reply["model_usage"]["planning_calls"])
        self.assertEqual(1, reply["model_usage"]["generation_calls"])
        self.assertTrue(reply["text"].startswith("I would support"))
        self.assertEqual(
            "pcfm_conversation_conditioned_response_simulation_v5",
            reply["model_kind"],
        )

    def test_cross_language_opinion_uses_person_event_after_same_model_json_retry(self) -> None:
        record = self.service.add_conversation_text_source(
            self.person_id,
            title="official assessment",
            text=(
                "Q: Does it make you question his fitness to be President?\n"
                "A: Yes, I think the Republican nominee is unfit to serve as President."
            ),
            speaker="Person A",
            source_date="2016-08-02",
            dataset_role="model_source",
            content_authenticity="verified_quote",
            source_locator="official transcript",
            source_context="Official press conference",
            source_url="https://example.test/official-assessment",
        )
        self.service.review_conversation_source(
            self.person_id, str(record["source_id"]), "confirmed"
        )
        self.service.conversation._model_services = RetryingOpinionFixture()
        reply = self.service.send_conversation_message(
            self.person_id,
            "你认为特朗普怎么样",
            dialogue_model_ref="fake:model",
        )
        self.assertNotEqual("general_assisted", reply["answer_status"])
        self.assertEqual("similar_event_evidence_answer", reply["answer_status"])
        self.assertEqual("simulation-v5", reply["prediction_trace"]["kernel"])
        self.assertEqual(2, reply["model_usage"]["planning_calls"])
        self.assertNotEqual("not_run_no_person_prediction", reply["style_status"])


if __name__ == "__main__":
    unittest.main()
