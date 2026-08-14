from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pcfm.model_services import ModelServiceError, ModelServiceManager, SUPPORTED_PROTOCOLS
from pcfm.product_service import ProductError, ProductService
from pcfm.response_prediction_v2 import KERNEL_ID_V2, ResponsePredictionKernelV2


class MaliciousDialogueModel:
    def __init__(self, mutation: str) -> None:
        self.mutation = mutation
        self.calls = 0

    def roles(self):
        return {"default_dialogue": "", "material_processing": "", "validation": ""}

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {"snapshot_id": "fixture-snapshot", "provider": "fixture", "model_id": "model"}

    def invoke(self, service_id, model_id, messages, **kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            return {"text": '{"plans": []}', "snapshot": self.snapshot("")}
        contract = json.loads(messages[-1]["content"])["frozen_contract"]
        neutral = "\n".join(item["text"] for field in ("claims", "reasons", "memories", "uncertainties") for item in contract[field])
        if self.mutation == "new_fact":
            text = neutral + "\nThis adds an unsupported 1969 Mars mission."
        else:
            text = "I oppose this."  # removes the frozen stance-bearing content
        return {"text": text, "snapshot": self.snapshot("")}


class AssistedDialogueModel:
    def __init__(
        self,
        *,
        preserve_anchor: bool = True,
        briefing: str = "This is a general explanation without person attribution.",
    ) -> None:
        self.preserve_anchor = preserve_anchor
        self.briefing = briefing
        self.last_payload = {}
        self.calls = 0

    def roles(self):
        return {"default_dialogue": "", "material_processing": "", "validation": ""}

    def resolve_model_ref(self, model_ref, require_available=True):
        return ({"service_id": "fake", "provider": "fixture"}, "model")

    def snapshot(self, model_ref):
        return {"snapshot_id": "fixture-snapshot", "provider": "fixture", "model_id": "model"}

    def invoke(self, service_id, model_id, messages, **kwargs):
        self.calls += 1
        system = str(messages[0]["content"])
        if "Generate only an external-knowledge briefing" in system:
            payload = json.loads(messages[-1]["content"])
            self.last_payload = payload
            anchor = payload.get("required_stance_anchor", "")
            briefing = self.briefing
            if anchor and not self.preserve_anchor:
                anchor = "invented stance"
            return {
                "text": json.dumps(
                    {
                        "required_stance_anchor": anchor,
                        "person_claim_ids": [],
                        "external_briefing": briefing,
                    }
                ),
                "snapshot": self.snapshot(""),
            }
        if kwargs.get("structured"):
            return {"text": '{"plans": []}', "snapshot": self.snapshot("")}
        payload = json.loads(messages[-1]["content"])
        anchor = payload.get("required_stance_anchor", "")
        text = "This is a general explanation without person attribution."
        if anchor:
            text = (
                f"{anchor}\nGeneral background can inform the explanation."
                if self.preserve_anchor
                else "I definitely support this for a new invented reason."
            )
        return {"text": text, "snapshot": self.snapshot("")}


class ResponseKernelV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.service = ProductService(self.root, seed_example=False, seed_demos=True)
        self.person_id = "demo-sally-ride"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def send(self, text: str, model_ref: str = "") -> dict[str, object]:
        return self.service.send_conversation_message(
            self.person_id, text, dialogue_model_ref=model_ref
        )

    def test_greeting_thanks_and_continue_are_safe_ordinary_dialogue(self) -> None:
        for text in ("你好", "谢谢", "接着说"):
            reply = self.send(text)
            self.assertEqual("ordinary_dialogue", reply["answer_status"])
            self.assertEqual([], reply["structured_prediction"]["claims"])
            self.assertEqual([], reply["structured_prediction"]["memories"])

    def test_short_followups_resolve_previous_turn_without_becoming_evidence(self) -> None:
        first = self.send("Tell us what prompted you to write that note and describe the events that followed.")
        second = self.send("Why?")
        third = self.send("What about that?")
        self.assertEqual("direct_answer", first["answer_status"])
        self.assertNotEqual("out_of_domain", second["applicability"])
        self.assertGreater(second["prediction_trace"]["resolved_context_turns"], 0)
        self.assertGreater(third["prediction_trace"]["resolved_context_turns"], 0)

    def test_direct_match_uses_the_verified_response_event(self) -> None:
        reply = self.send("Tell us what prompted you to write that note and describe the events that followed.")
        self.assertEqual("direct_answer", reply["answer_status"])
        self.assertEqual(1, len(reply["prediction_trace"]["selected_event_ids"]))
        self.assertTrue(reply["evidence"])

    def test_artifact_contains_event_relations_and_separate_overall_tendencies(self) -> None:
        summary = self.service.conversation_summary(self.person_id)
        artifact = self.service.conversation._response_model(
            self.person_id, summary["active_version"]
        )
        self.assertTrue(artifact["episode_bundles"])
        self.assertTrue(artifact["conditional_tendencies"])
        self.assertTrue(artifact["overall_tendencies"])
        self.assertTrue(artifact["event_relations"])
        self.assertTrue(artifact["demonstrated_knowledge"])
        self.assertTrue(
            all(
                value["confidence_kind"]
                == "recurring_evidence_strength_not_prediction_accuracy"
                for value in artifact["overall_tendencies"]
            )
        )

    def test_compound_question_returns_multiple_related_historical_events(self) -> None:
        reply = self.send(
            "How did NASA's selection process and working culture shape your early experience as an astronaut?"
        )
        self.assertEqual("similar_event_evidence_answer", reply["answer_status"])
        self.assertGreaterEqual(len(reply["evidence"]), 2)
        self.assertEqual(
            "analogical_evidence_not_new_stance",
            reply["person_prediction_status"],
        )

    def test_unrelated_question_does_not_use_overall_tendency(self) -> None:
        reply = self.send("What would you do with cryptocurrency in 2026?")
        self.assertEqual("general_assisted", reply["answer_status"])
        self.assertFalse(reply["structured_prediction"]["claims"])
        self.assertEqual("not_available", reply["person_prediction_status"])

    def test_no_person_evidence_can_use_model_as_disclosed_general_answer(self) -> None:
        person = self.service.create_conversation_person(name="Empty Person", language="en")
        person_id = str(person["person_id"])
        fake = AssistedDialogueModel()
        self.service.conversation._model_services = fake
        reply = self.service.conversation.send_message(
            person_id, "Explain this topic.", dialogue_model_ref="fake:model"
        )
        self.assertEqual("general_assisted", reply["answer_status"])
        self.assertEqual("not_available", reply["person_prediction_status"])
        self.assertEqual("external_model_briefing", reply["knowledge_source"])
        self.assertIn("without person attribution", reply["text"])
        self.assertEqual("match_current_question", fake.last_payload["response_language"])

    def test_general_assisted_generation_cannot_invent_a_person_stance(self) -> None:
        fake = AssistedDialogueModel(preserve_anchor=False)
        self.service.conversation._model_services = fake
        reply = self.service.conversation.send_message(
            self.person_id,
            "What would you do with cryptocurrency in 2026?",
            dialogue_model_ref="fake:model",
        )
        self.assertEqual("general_assisted", reply["answer_status"])
        self.assertEqual("not_available", reply["person_prediction_status"])
        self.assertIn("我会先保留判断", reply["text"])
        self.assertNotIn("invented", reply["text"])

    def test_person_opinion_does_not_degrade_to_third_party_briefing(self) -> None:
        fake = AssistedDialogueModel(
            briefing="Trump is a polarizing political figure. He attracts both strong support and opposition."
        )
        self.service.conversation._model_services = fake
        reply = self.service.conversation.send_message(
            self.person_id,
            "What do you think of Trump?",
            dialogue_model_ref="fake:model",
        )
        # 宽评价问题不得降级到第三方简报；走对象评价投影
        self.assertEqual("object_evaluation_projection_answer", reply["answer_status"])
        self.assertNotIn("He attracts", reply["text"])
        self.assertNotIn("polarizing", reply["text"])

    def test_partial_answer_and_clarification_are_distinct_from_refusal(self) -> None:
        partial = self.send("Tell us what prompted you to write that note and describe the events that followed.")
        clarification = self.send("Why?")
        self.assertEqual("direct_answer", partial["answer_status"])
        self.assertIn(clarification["answer_status"], {"clarification_needed", "direct_answer"})

    def test_generated_messages_never_enter_trainable_events(self) -> None:
        before = self.service.conversation_summary(self.person_id)["versions"][0]["source_set_digest"]
        self.send("你好")
        after = self.service.conversation_summary(self.person_id)["versions"][0]["source_set_digest"]
        self.assertEqual(before, after)

    def test_frozen_evidence_answer_does_not_call_selected_model_for_style(self) -> None:
        fake = MaliciousDialogueModel("new_fact")
        self.service.conversation._model_services = fake
        reply = self.service.conversation.send_message(
            self.person_id,
            "Tell us what prompted you to write that note and describe the events that followed.",
            dialogue_model_ref="fake:model",
        )
        self.assertNotIn("Mars", reply["text"])
        self.assertEqual(0, fake.calls)
        self.assertEqual("selected_but_not_needed", reply["model_usage"]["status"])

    def test_frozen_evidence_answer_cannot_have_stance_changed_by_model(self) -> None:
        fake = MaliciousDialogueModel("stance")
        self.service.conversation._model_services = fake
        reply = self.service.conversation.send_message(
            self.person_id,
            "Tell us what prompted you to write that note and describe the events that followed.",
            dialogue_model_ref="fake:model",
        )
        self.assertNotEqual("I oppose this.", reply["text"])
        self.assertEqual(0, fake.calls)

    def test_kernel_v2_is_order_id_and_alias_invariant(self) -> None:
        summary = self.service.conversation_summary(self.person_id)
        artifact = self.service.conversation._response_model(self.person_id, summary["active_version"])
        kernel = ResponsePredictionKernelV2()
        first = kernel.predict(artifact, text="Tell us what prompted you to write that note and describe the events that followed.", history=[])
        renamed = json.loads(json.dumps(artifact))
        renamed["events"] = list(reversed(renamed["events"]))
        for index, event in enumerate(renamed["events"]):
            event["event_id"] = f"irrelevant-{index}"
        renamed = kernel.reseal_for_test(renamed)
        second = kernel.predict(renamed, text="Tell us what prompted you to write that note and describe the events that followed.", history=[])
        self.assertEqual(first["structured_prediction"]["stance"], second["structured_prediction"]["stance"])
        first_units = [(item["id"], item["text"], item["probability"]) for item in first["structured_prediction"]["claims"]]
        second_units = [(item["id"], item["text"], item["probability"]) for item in second["structured_prediction"]["claims"]]
        self.assertEqual(first_units, second_units)


class ModelServiceSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = ModelServiceManager(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_service_public_state_never_contains_api_key(self) -> None:
        created = self.manager.save_service(
            {
                "display_name": "Compatible",
                "provider": "custom",
                "protocol": "openai_compatible",
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "top-secret-value",
                "enabled": True,
                "models": ["model-a"],
            }
        )
        rendered = json.dumps(created) + json.dumps(self.manager.public_state())
        self.assertNotIn("top-secret-value", rendered)
        self.assertTrue(created["api_key_configured"])
        self.assertNotIn("api_key", created)

    def test_required_provider_protocols_are_declared_without_fixed_model_catalogue(self) -> None:
        self.assertEqual(
            {
                "openai_native", "openai_compatible", "anthropic", "gemini",
                "ollama", "custom_compatible",
            },
            set(SUPPORTED_PROTOCOLS),
        )

    def test_person_export_and_browser_state_do_not_contain_secret(self) -> None:
        product = ProductService(self.root / "product", seed_example=False, seed_demos=True)
        product.model_services.save_service(
            {"display_name": "X", "provider": "custom", "protocol": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "api_key": "never-export-me", "enabled": True, "models": ["x"]}
        )
        self.assertNotIn("never-export-me", json.dumps(product.export_person("demo-sally-ride")))
        self.assertNotIn("never-export-me", json.dumps(product.model_services.public_state()))

    def test_unavailable_model_does_not_silently_fallback(self) -> None:
        service = self.manager.save_service(
            {"display_name": "Down", "provider": "custom", "protocol": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "enabled": True, "models": ["down-model"]}
        )
        with self.assertRaises(ModelServiceError):
            self.manager.invoke(service["service_id"], "down-model", [{"role": "user", "content": "test"}])

    def test_model_is_ready_only_after_real_structured_chat_probe(self) -> None:
        service = self.manager.save_service(
            {"display_name": "Probe", "provider": "custom", "protocol": "openai_compatible", "base_url": "https://example.test/v1", "enabled": True, "models": ["model-a"]}
        )
        self.assertEqual("needs_test", service["call_readiness"])

        def fake_request(_item, path, **kwargs):
            if path == "/models":
                return {"data": [{"id": "model-a"}]}
            self.assertEqual("/chat/completions", path)
            self.assertEqual(128, kwargs["body"]["max_tokens"])
            return {"choices": [{"message": {"content": '{"pcfm_probe":true}'}}]}

        with mock.patch.object(self.manager, "_json_request", side_effect=fake_request):
            result = self.manager.test_connection(service["service_id"], "model-a")
        self.assertEqual("connected", result["status"])
        public = self.manager.public_state()["services"][0]
        self.assertEqual("ready", public["call_readiness"])
        self.assertEqual("model-a", public["last_probe_model"])

    def test_model_selection_only_changes_next_message_and_not_person_version(self) -> None:
        product = ProductService(self.root / "selection", seed_example=False, seed_demos=True)
        manager = product.model_services
        first = manager.save_service({"display_name": "A", "provider": "custom", "protocol": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "enabled": True, "models": ["a"]})
        second = manager.save_service({"display_name": "B", "provider": "custom", "protocol": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "enabled": True, "models": ["b"]})
        person_id = "demo-sally-ride"
        version = product.conversation_summary(person_id)["active_version"]
        with mock.patch.object(manager, "resolve_model_ref", return_value=({}, "model")):
            product.select_dialogue_model(person_id, f"{first['service_id']}:a")
            product.select_dialogue_model(person_id, f"{second['service_id']}:b")
        self.assertEqual(version, product.conversation_summary(person_id)["active_version"])
        self.assertEqual(f"{second['service_id']}:b", product.conversation_summary(person_id)["dialogue_model_ref"])
        product.select_dialogue_model(person_id, "")
        self.assertEqual("", product.conversation_summary(person_id)["dialogue_model_ref"])

    def test_person_model_selections_are_isolated_and_persistent(self) -> None:
        product = ProductService(self.root / "isolation", seed_example=False, seed_demos=True)
        cfg = product.model_services.save_service({"display_name": "A", "provider": "custom", "protocol": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "enabled": True, "models": ["a", "b"]})
        with mock.patch.object(product.model_services, "resolve_model_ref", return_value=({}, "model")):
            product.select_dialogue_model("demo-sally-ride", f"{cfg['service_id']}:a")
            product.select_dialogue_model("demo-barack-obama", f"{cfg['service_id']}:b")
        reloaded = ProductService(self.root / "isolation", seed_example=False)
        self.assertTrue(reloaded.conversation_summary("demo-sally-ride")["dialogue_model_ref"].endswith(":a"))
        self.assertTrue(reloaded.conversation_summary("demo-barack-obama")["dialogue_model_ref"].endswith(":b"))


if __name__ == "__main__":
    unittest.main()
