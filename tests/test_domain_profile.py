"""领域整体画像 DomainProfile：聚合 + 无取舍「公开表态」原子的可回溯性。

对齐 docs/事件原子提取与领域价值画像设计方案.md §3.3/§5.1/§10：
- 无问答/无取舍的「一件事」仍作为「公开表态」落原子（验收 #4）；
- 同领域伴随原子聚出 DomainProfile，含认知/价值/想法/策略/条件五类，每项可回溯（验收 #3）。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.services import PcfmService
from pcfm.conversation.domain_profile import (
    build_domain_profiles,
    DOMAIN_PROFILE_SCHEMA,
)
from pcfm.conversation.extraction import _int_offset, _merge_candidates


class MergeCandidatesTests(unittest.TestCase):
    def test_int_offset_handles_non_numeric(self) -> None:
        self.assertEqual(0, _int_offset(None))
        self.assertEqual(0, _int_offset("abc"))
        self.assertEqual(1500, _int_offset("1500"))

    def test_merge_dedupes_same_verbatim_response(self) -> None:
        candidates = [
            {"candidate_id": "a", "actual_response": "能用程序写的尽量用程序写", "offset_start": 0, "offset_end": 20},
            {"candidate_id": "b", "actual_response": "能用程序写的尽量用程序写", "offset_start": 700, "offset_end": 720},
            {"candidate_id": "c", "actual_response": "另一件事", "offset_start": 100, "offset_end": 120},
        ]
        merged = _merge_candidates(candidates)
        self.assertEqual(2, len(merged))
        responses = {c["actual_response"] for c in merged}
        self.assertEqual({"能用程序写的尽量用程序写", "另一件事"}, responses)


class BuildDomainProfilesTests(unittest.TestCase):
    def test_aggregates_values_ideas_strategies_and_traceability(self) -> None:
        preference_atoms = [
            {
                "preference_atom_id": "pa-1",
                "tendency_type": "principle_priority",
                "direction": "support",
                "target": "",
                "event_frame_id": "e1",
                "protected_interest_id": "safety",
                "accepted_cost_id": "speed",
                "domain_tags": ["product"],
                "conditions": ["when scale is small"],
                "evidence_span": "prioritize safety over speed",
            },
            {
                "preference_atom_id": "pa-2",
                "tendency_type": "behavior_evaluation",
                "direction": "oppose",
                "target": "individual",
                "event_frame_id": "e2",
                "protected_interest_id": "competence",
                "accepted_cost_id": "",
                "domain_tags": ["product"],
                "conditions": [],
                "evidence_span": "unfit to lead",
            },
            {
                "preference_atom_id": "pa-3",
                "tendency_type": "means_ends",
                "direction": "support",
                "target": "",
                "event_frame_id": "e3",
                "protected_interest_id": "quality",
                "accepted_cost_id": "speed",
                "domain_tags": ["product"],
                "conditions": [],
                "evidence_span": "write a program for every repeatable step",
            },
        ]
        statement_atoms = [
            {
                "statement_atom_id": "sa-1",
                "event_frame_id": "e4",
                "statement": "能用程序写的尽量用程序写。",
                "domain_tags": ["product"],
                "conditions": ["在流程稳定的前提下"],
                "evidence_span": "能用程序写的尽量用程序写。",
                "status": "reviewed_public_statement_without_tradeoff",
            }
        ]
        knowledge_claims = [
            {
                "knowledge_claim_id": "kc-1",
                "statement": "现有 SOP 不准。",
                "event_frame_id": "e1",
                "domain_tags": ["product"],
            }
        ]
        profiles = build_domain_profiles(preference_atoms, statement_atoms, knowledge_claims)

        self.assertIn("product", profiles)
        profile = profiles["product"]
        self.assertEqual(DOMAIN_PROFILE_SCHEMA, profile["schema_version"])

        # ① 认知
        self.assertEqual(["现有 SOP 不准。"], [c["statement"] for c in profile["cognition"]])
        # ② 价值概念
        values = profile["values"]
        self.assertEqual(1, len(values))
        self.assertEqual("safety", values[0]["preferred_side"])
        self.assertEqual("speed", values[0]["sacrificed_side"])
        self.assertIn("pa-1", values[0]["atom_ids"])
        self.assertIn("e1", values[0]["event_ids"])
        # ③ 想法（评价类 + 公开表态）
        ideas = profile["ideas"]
        self.assertEqual(2, len(ideas))
        eval_idea = next(i for i in ideas if i["tendency_type"] == "behavior_evaluation")
        self.assertEqual("oppose", eval_idea["direction"])
        self.assertEqual("individual", eval_idea["target"])
        statement_idea = next(i for i in ideas if i.get("public_statement"))
        self.assertEqual("能用程序写的尽量用程序写。", statement_idea["public_statement"])
        self.assertIn("sa-1", statement_idea["atom_ids"])
        self.assertIn("e4", statement_idea["event_ids"])
        # ④ 策略
        self.assertEqual(1, len(profile["strategies"]))
        self.assertEqual("write a program for every repeatable step", profile["strategies"][0]["statement"])
        # ⑤ 条件与例外
        self.assertEqual(["when scale is small", "在流程稳定的前提下"], [c["condition"] for c in profile["conditions"]])


class StatementExtractionModel:
    """确定性 LLM stub：返回一个无取舍的「一件事」（独白式表态）。"""

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
                            "trigger": "团队方法论讨论",
                            "context": "内部会议",
                            "response": "能用程序写的尽量用程序写。",
                            "occasion": "内部会议",
                            "interlocutor": "team",
                            "speaker": "Alice Example",
                            "locator": "paragraph 1",
                            "speech_act": "statement",
                            "stance": "support",
                            "claims": ["能用程序写的尽量用程序写。"],
                            "reasons": [],
                            "memories": [],
                            "uncertainties": [],
                            "domain_ids": ["technology"],
                            "condition_spans": ["在流程稳定的前提下"],
                            "reason_spans": [],
                            "demonstrated_claim_spans": ["能用程序写的尽量用程序写"],
                            "event_structure_type": "means_ends",
                            "tradeoffs": [],
                        }
                    ]
                }
            ),
            "snapshot": self.snapshot(""),
        }


class StatementAtomIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.temporary.name), seed_example=False)
        self.alice = self.service.create_conversation_person(
            name="Alice Example", aliases=["Alice"], language="en"
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def test_no_tradeoff_statement_becomes_public_statement_atom_and_profile(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = StatementExtractionModel()
        source = self.service.add_conversation_text_source(
            person_id,
            title="内部会议",
            text="能用程序写的尽量用程序写。在流程稳定的前提下。",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="内部会议",
            source_url="https://example.org/talk",
        )
        self.service.review_conversation_source(person_id, str(source["source_id"]), "confirmed")
        extracted = self.service.extract_conversation_response_candidates(
            person_id, str(source["source_id"])
        )
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            person_id, str(source["source_id"]), candidate_id, "confirmed"
        )
        summary = self.service.conversation_summary(person_id)
        self.assertIsNotNone(summary["active_version"], "无取舍材料也应形成版本")
        model = self.service.conversation._simulation_model(person_id, int(summary["active_version"]))

        # 无取舍 → preference_atoms 空，但 statement_atoms 有公开表态原子
        self.assertEqual([], model["reviewed_public_model"]["preference_atoms"])
        statement_atoms = model["statement_atoms"]
        self.assertEqual(1, len(statement_atoms))
        self.assertEqual("能用程序写的尽量用程序写。", statement_atoms[0]["statement"])
        self.assertEqual("reviewed_public_statement_without_tradeoff", statement_atoms[0]["status"])

        # 领域画像里想法维度含该公开表态，且可回溯到事件原子
        profiles = model["domain_profiles"]
        self.assertIn("technology", profiles)
        ideas = profiles["technology"]["ideas"]
        statement_idea = next(i for i in ideas if i.get("public_statement"))
        self.assertEqual("能用程序写的尽量用程序写。", statement_idea["public_statement"])
        self.assertTrue(statement_idea["atom_ids"])
        self.assertTrue(statement_idea["event_ids"])

    def test_domain_profile_drives_query_answer(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = StatementExtractionModel()
        source = self.service.add_conversation_text_source(
            person_id,
            title="内部会议",
            text="能用程序写的尽量用程序写。在流程稳定的前提下。",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="内部会议",
            source_url="https://example.org/talk",
        )
        self.service.review_conversation_source(person_id, str(source["source_id"]), "confirmed")
        extracted = self.service.extract_conversation_response_candidates(
            person_id, str(source["source_id"])
        )
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            person_id, str(source["source_id"]), candidate_id, "confirmed"
        )
        reply = self.service.send_conversation_message(
            person_id, "在软件开发上，怎么推进重复工作？"
        )
        self.assertEqual("domain_profile_answer", reply["answer_status"])
        self.assertIn("能用程序写的尽量用程序写", reply["text"])
        self.assertTrue(reply["structured_prediction"]["evidence_event_ids"])


class InferredExtractionModel:
    """确定性 LLM stub：返回无明写取舍、但带 inferred_tendencies 的事件。"""

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
                            "trigger": "团队方法论讨论",
                            "context": "内部会议",
                            "response": "能用程序写的尽量用程序写。",
                            "occasion": "内部会议",
                            "interlocutor": "team",
                            "speaker": "Alice Example",
                            "locator": "paragraph 1",
                            "speech_act": "statement",
                            "stance": "support",
                            "claims": ["能用程序写的尽量用程序写。"],
                            "reasons": [],
                            "memories": [],
                            "uncertainties": [],
                            "domain_ids": ["technology"],
                            "condition_spans": [],
                            "reason_spans": [],
                            "demonstrated_claim_spans": [],
                            "event_structure_type": "means_ends",
                            "tradeoffs": [],
                            "inferred_tendencies": [
                                {
                                    "protected_interest_id": "evidence_quality",
                                    "accepted_cost_id": "speed",
                                    "evidence_span": "能用程序写的尽量用程序写",
                                }
                            ],
                        }
                    ]
                }
            ),
            "snapshot": self.snapshot(""),
        }


class InferredTendencyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.temporary.name), seed_example=False)
        self.alice = self.service.create_conversation_person(
            name="Alice Example", aliases=["Alice"], language="en"
        )

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def test_inferred_tendency_becomes_inferred_value_atom_with_origin(self) -> None:
        person_id = str(self.alice["person_id"])
        self.service.conversation._model_services = InferredExtractionModel()
        source = self.service.add_conversation_text_source(
            person_id,
            title="内部会议",
            text="能用程序写的尽量用程序写。在流程稳定的前提下。",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="paragraph 1",
            source_context="内部会议",
            source_url="https://example.org/talk",
        )
        self.service.review_conversation_source(person_id, str(source["source_id"]), "confirmed")
        extracted = self.service.extract_conversation_response_candidates(
            person_id, str(source["source_id"])
        )
        candidate_id = str(extracted["llm_response_event_candidates"][0]["candidate_id"])
        self.service.review_conversation_response_candidate(
            person_id, str(source["source_id"]), candidate_id, "confirmed"
        )
        summary = self.service.conversation_summary(person_id)
        model = self.service.conversation._simulation_model(person_id, int(summary["active_version"]))

        inferred = model["inferred_value_atoms"]
        self.assertEqual(1, len(inferred))
        self.assertEqual("evidence_quality", inferred[0]["protected_interest_id"])
        self.assertEqual("speed", inferred[0]["accepted_cost_id"])
        self.assertEqual("inferred", inferred[0]["origin"])

        values = model["domain_profiles"]["technology"]["values"]
        self.assertTrue(values)
        self.assertEqual("inferred", values[0]["origin"])
        self.assertEqual("evidence_quality", values[0]["preferred_side"])
        self.assertEqual("speed", values[0]["sacrificed_side"])


if __name__ == "__main__":
    unittest.main()
