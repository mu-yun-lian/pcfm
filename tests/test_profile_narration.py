"""领域画像 LLM 忠实转述（§5.2）：抽象视图 + 门禁 + 降级回退。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pcfm.services import PcfmService
from pcfm.conversation.profile_narration import (
    ProfileNarrationMixin,
    _abstract_domain_profiles,
    _view_item_ids,
)


def _profiles() -> dict:
    return {
        "product": {
            "schema_version": "pcfm-domain-profile-v1",
            "domain_id": "product",
            "cognition": [
                {
                    "item_id": "cog-1",
                    "statement": "现有 SOP 不准",
                    "knowledge_type": "fact",
                    "atom_ids": ["kc-1"],
                    "event_ids": ["e1"],
                    "evidence_spans": ["现有 SOP 不准"],
                }
            ],
            "values": [
                {
                    "item_id": "val-1",
                    "preferred_side": "quality",
                    "sacrificed_side": "speed",
                    "tendency_types": ["principle_priority"],
                    "status": "reviewed_observable_public_tradeoff",
                    "atom_ids": ["pa-1"],
                    "event_ids": ["e1"],
                    "evidence_spans": ["prioritize quality over speed"],
                }
            ],
            "ideas": [
                {
                    "item_id": "idea-1",
                    "tendency_type": "behavior_evaluation",
                    "direction": "oppose",
                    "target": "individual",
                    "atom_ids": ["pa-2"],
                    "event_ids": ["e2"],
                    "evidence_spans": ["unfit to lead"],
                }
            ],
            "strategies": [
                {
                    "item_id": "strategy-1",
                    "statement": "能程序化的步骤尽量程序化",
                    "atom_ids": ["pa-3"],
                    "event_ids": ["e3"],
                    "evidence_spans": ["能用程序写的尽量用程序写"],
                }
            ],
            "conditions": [
                {
                    "item_id": "cond-1",
                    "condition": "在流程稳定的前提下",
                    "atom_ids": ["sa-1"],
                    "event_ids": ["e4"],
                    "evidence_spans": ["在流程稳定的前提下"],
                }
            ],
            "counts": {},
        }
    }


class AbstractViewTests(unittest.TestCase):
    def test_abstract_view_strips_verbatim_and_localizes(self) -> None:
        view = _abstract_domain_profiles(_profiles(), ["product"], True)
        self.assertEqual(1, len(view))
        entry = view[0]
        # 剥掉逐字原话与回溯 span，只留抽象字段 + item_id
        cognition = entry["cognition"][0]
        self.assertNotIn("evidence_spans", cognition)
        self.assertNotIn("atom_ids", cognition)
        self.assertNotIn("event_ids", cognition)
        self.assertEqual("事实判断", cognition["knowledge_type"])
        # 封闭词表本地化
        self.assertEqual("质量", entry["values"][0]["preferred_side"])
        self.assertEqual("反对", entry["ideas"][0]["direction"])
        self.assertEqual("个人", entry["ideas"][0]["target"])
        self.assertEqual("产品", entry["domain_label"])

    def test_view_item_ids_collects_all_dimensions(self) -> None:
        view = _abstract_domain_profiles(_profiles(), ["product"], False)
        ids = _view_item_ids(view)
        self.assertEqual({"cog-1", "val-1", "idea-1", "strategy-1", "cond-1"}, ids)


class NarrationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.view = _abstract_domain_profiles(_profiles(), ["product"], True)
        self.valid_ids = _view_item_ids(self.view)

    def _gate(self, narration, referenced):
        return ProfileNarrationMixin._gate_narration(
            narration, referenced, self.view, self.valid_ids, True
        )

    def test_valid_narration_passes(self) -> None:
        ok, reason = self._gate(
            "在知识库构建领域，此人公开主张现有 SOP 不够准确，更看重质量而不是速度。",
            ["cog-1", "val-1"],
        )
        self.assertTrue(ok, reason)

    def test_referenced_item_not_in_profile_is_rejected(self) -> None:
        ok, reason = self._gate(
            "在知识库构建领域，此人公开主张现有 SOP 不够准确。",
            ["cog-1", "not-in-profile"],
        )
        self.assertFalse(ok)
        self.assertEqual("referenced_item_id_not_in_profile", reason)

    def test_new_number_is_rejected(self) -> None:
        ok, reason = self._gate(
            "此人更看重质量而不是速度，准确率 99%。",
            ["val-1"],
        )
        self.assertFalse(ok)
        self.assertEqual("narration_introduced_new_number", reason)

    def test_out_of_profile_concept_is_rejected(self) -> None:
        # 画像只有 quality>speed，没有 safety；转述引入「安全与避免伤害」→ 拒绝。
        ok, reason = self._gate(
            "此人更看重安全与避免伤害而不是速度与效率。",
            ["val-1"],
        )
        self.assertFalse(ok)
        self.assertEqual("narration_introduced_out_of_profile_concept", reason)

    def test_in_profile_concept_passes(self) -> None:
        ok, reason = self._gate(
            "此人更看重质量而不是速度与效率。",
            ["val-1"],
        )
        self.assertTrue(ok, reason)


class NarrationModel:
    def roles(self):
        return {"material_processing": "", "default_dialogue": "fake:model"}

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {"snapshot_id": "fixture", "provider": "fixture", "model_id": "model"}

    def invoke(self, service_id, model_id, messages, **kwargs):
        return {
            "text": json.dumps(
                {
                    "narration": "在知识库构建领域，此人公开主张现有 SOP 不够准确，更看重质量而不是速度。",
                    "referenced_item_ids": ["cog-1", "val-1"],
                }
            ),
            "snapshot": self.snapshot(""),
        }


class InvalidRefsNarrationModel(NarrationModel):
    def invoke(self, service_id, model_id, messages, **kwargs):
        return {
            "text": json.dumps(
                {
                    "narration": "此人有一些看法。",
                    "referenced_item_ids": ["not-in-profile"],
                }
            ),
            "snapshot": self.snapshot(""),
        }


class NarrationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.temporary.name), seed_example=False)
        self.person = self.service.create_conversation_person(name="Alice Example", language="en")
        self.workbench = self.service.conversation

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def _narrate(self, model_services, model_ref):
        self.workbench._model_services = model_services
        return self.workbench._narrate_domain_profile(
            person_id=str(self.person["person_id"]),
            artifact={"domain_profiles": _profiles()},
            domains=["product"],
            is_chinese=True,
            model_ref=model_ref,
        )

    def test_no_model_falls_back(self) -> None:
        result = self._narrate(None, "")
        self.assertEqual("no_model", result["status"])
        self.assertEqual("", result["narration"])

    def test_narration_success(self) -> None:
        result = self._narrate(NarrationModel(), "fake:model")
        self.assertEqual("ok", result["status"])
        self.assertIn("质量", result["narration"])
        self.assertEqual(["cog-1", "val-1"], result["referenced_item_ids"])

    def test_invalid_refs_gate_fails(self) -> None:
        result = self._narrate(InvalidRefsNarrationModel(), "fake:model")
        self.assertEqual("gate_failed", result["status"])
        self.assertEqual("referenced_item_id_not_in_profile", result["reason"])


if __name__ == "__main__":
    unittest.main()
