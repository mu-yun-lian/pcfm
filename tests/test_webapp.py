from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pcfm.webapp import APP_VERSION, build_server
from tests.test_product_service import diagnostic_records


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.server = build_server(
            "127.0.0.1",
            0,
            Path(self.temporary.name),
            seed_example=True,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.server.service.close()
        self.temporary.cleanup()

    def request(self, path: str, method: str = "GET", payload: object | None = None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base + path,
            method=method,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=20) as response:
                return response.status, response.headers, response.read()
        except HTTPError as error:
            try:
                return error.code, error.headers, error.read()
            finally:
                error.close()

    def json_request(self, path: str, method: str = "GET", payload: object | None = None):
        status, _headers, raw = self.request(path, method, payload)
        return status, json.loads(raw.decode("utf-8"))

    def test_static_page_health_and_full_http_closed_loop(self) -> None:
        status, _headers, page = self.request("/")
        self.assertEqual(status, 200)
        page_text = page.decode("utf-8")
        self.assertIn("PCFM 对话式人物模拟", page_text)
        # Vue3(Vite) 构建产物优先; 未构建时回退旧 vanilla 前端
        if '<div id="app">' in page_text:
            self.assertIn("/assets/", page_text)
        else:
            self.assertIn(f'/app.js?v={APP_VERSION}', page_text)
            status, _headers, app = self.request("/app.js")
            self.assertEqual(status, 200)
            self.assertIn(f'const APP_VERSION = "{APP_VERSION}";', app.decode("utf-8"))
        status, health = self.json_request("/api/health")
        self.assertEqual((status, health["ok"]), (200, True))

    def test_root_static_assets_served_with_correct_mime(self) -> None:
        cases = [
            ("/default-person-avatar.png", "image/png"),
            ("/demo-barack-obama.svg", "image/svg+xml"),
            ("/demo-sally-ride.svg", "image/svg+xml"),
        ]
        for path, expected_mime in cases:
            status, headers, body = self.request(path)
            self.assertEqual(status, 200, path)
            content_type = headers.get("Content-Type", "")
            self.assertTrue(content_type.startswith(expected_mime), f"{path}: {content_type}")
            self.assertGreater(len(body), 0, path)

    def test_new_conversation_creates_switchable_session(self) -> None:
        status, created = self.json_request("/api/conversation/people", "POST", {"name": "Archive Chat"})
        self.assertEqual(status, 201)
        person_id = created["person"]["person_id"]
        status, sent = self.json_request(f"/api/people/{person_id}/conversation/messages", "POST", {"text": "你好"})
        self.assertEqual(status, 201)
        for _ in range(200):
            _, job_state = self.json_request(f"/api/jobs/{sent['job_id']}")
            if job_state["job"]["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        status, new_session = self.json_request(f"/api/people/{person_id}/conversation/new", "POST", {})
        self.assertEqual(status, 200)
        first_id = new_session["conversation"]["session_id"]
        status, listed = self.json_request(f"/api/people/{person_id}/conversation/sessions")
        self.assertEqual(status, 200)
        self.assertEqual(2, len(listed["sessions"]))
        active = next(s for s in listed["sessions"] if s["active"])
        self.assertEqual(first_id, active["session_id"])
        old_id = next(s for s in listed["sessions"] if not s["active"])["session_id"]
        status, switched = self.json_request(f"/api/people/{person_id}/conversation/sessions/{old_id}/switch", "POST", {})
        self.assertEqual(status, 200)
        self.assertEqual(old_id, switched["session"]["session_id"])
        status, summary = self.json_request(f"/api/people/{person_id}/conversation")
        self.assertEqual(status, 200)
        self.assertEqual(2, len(summary["conversation"]["messages"]))

    def test_model_service_api_and_person_selection_never_expose_secret(self) -> None:
        status, created = self.json_request(
            "/api/model-services",
            "POST",
            {
                "display_name": "本地兼容服务",
                "provider": "LM Studio",
                "protocol": "openai_compatible",
                "base_url": "http://127.0.0.1:9/v1",
                "api_key": "browser-secret-must-not-return",
                "models": ["local-model-a", "local-model-b"],
                "enabled_models": ["local-model-a", "local-model-b"],
                "enabled": True,
            },
        )
        self.assertEqual(status, 201)
        self.assertNotIn("browser-secret-must-not-return", json.dumps(created))
        self.assertTrue(created["service"]["api_key_configured"])
        status, public = self.json_request("/api/model-services")
        self.assertEqual(status, 200)
        self.assertNotIn("browser-secret-must-not-return", json.dumps(public))
        service_id = created["service"]["service_id"]
        status, person = self.json_request(
            "/api/conversation/people",
            "POST",
            {"name": "模型选择测试人物", "source_mode": "user_provided"},
        )
        self.assertEqual(status, 201)
        person_id = person["person"]["person_id"]
        status, selected = self.json_request(
            f"/api/people/{person_id}/conversation/model",
            "POST",
            {"model_ref": f"{service_id}:local-model-b"},
        )
        self.assertEqual(status, 400)
        self.assertIn("真实对话调用验证", selected["message"])
        self.assertNotIn("browser-secret-must-not-return", json.dumps(selected))
        status, exported_headers, exported = self.request(
            f"/api/people/{person_id}/export"
        )
        self.assertEqual(status, 200)
        self.assertNotIn(b"browser-secret-must-not-return", exported)

        status, created = self.json_request(
            "/api/people",
            "POST",
            {
                "name": "网页测试人物",
                "description": "测试闭环",
                "feature_names": ["条件得分", "常数项"],
            },
        )
        self.assertEqual(status, 201)
        person_id = created["person"]["person_id"]
        status, reference = self.json_request(
            "/api/people",
            "POST",
            {
                "name": "网页真实参照人物",
                "description": "行为基线参照",
                "feature_names": ["条件得分", "常数项"],
            },
        )
        self.assertEqual(status, 201)
        status, reference_import = self.json_request(
            f"/api/people/{reference['person']['person_id']}/history/import",
            "POST",
            {"format": "json", "payload": diagnostic_records()},
        )
        self.assertEqual((status, reference_import["sample_count"]), (200, 50))
        status, imported = self.json_request(
            f"/api/people/{person_id}/history/import",
            "POST",
            {"format": "json", "payload": diagnostic_records()},
        )
        self.assertEqual((status, imported["sample_count"]), (200, 50))
        status, trained = self.json_request(
            f"/api/people/{person_id}/train",
            "POST",
            {},
        )
        self.assertEqual(trained["model"]["validation_status"], "unvalidated")
        scenario = {
            "scenario_id": "http-new-choice",
            "question": "是否执行固定类型方案？",
            "option_a": "执行",
            "option_b": "不执行",
            "domain": "方案选择",
            "features": {"条件得分": 0.0, "常数项": 1.0},
        }
        status, predicted = self.json_request(
            f"/api/people/{person_id}/predict",
            "POST",
            {"scenario": scenario, "diagnostic_override": True},
        )
        prediction = predicted["prediction"]
        self.assertEqual((status, prediction["status"]), (200, "predicted"))
        status, updated = self.json_request(
            f"/api/people/{person_id}/predictions/{prediction['prediction_id']}/outcome",
            "POST",
            {"actual_choice": 1},
        )
        self.assertEqual(updated["prediction"]["updated_model_version"], 2)
        status, detail = self.json_request(f"/api/people/{person_id}")
        self.assertEqual(len(detail["person"]["versions"]), 2)
        self.assertEqual(detail["person"]["prediction_metrics"]["sample_count"], 1)

    def test_cognitive_http_vertical_loop(self) -> None:
        person_id = "josh-hawley-section230"
        status, detail = self.json_request(f"/api/people/{person_id}")
        self.assertEqual(status, 200)
        evidence = detail["person"]["cognitive"]["evidence"]
        self.assertEqual(len(evidence), 5)
        for item in evidence:
            status, reviewed = self.json_request(
                f"/api/people/{person_id}/cognitive/evidence/{item['evidence_id']}/review",
                "POST",
                {"decision": "confirmed"},
            )
            self.assertEqual((status, reviewed["evidence"]["review_status"]), (200, "confirmed"))
        status, generated = self.json_request(
            f"/api/people/{person_id}/cognitive/card/generate", "POST", {}
        )
        self.assertEqual((status, generated["card"]["version"]), (200, 1))
        status, drafted = self.json_request(
            f"/api/people/{person_id}/cognitive/scenarios/draft",
            "POST",
            {
                "text": (
                    "Should Congress remove Section 230 immunity for claims arising "
                    "from generative AI output by large technology companies so parents can sue?"
                )
            },
        )
        self.assertEqual(status, 201)
        scenario = drafted["scenario"]
        status, confirmed = self.json_request(
            f"/api/people/{person_id}/cognitive/scenarios/{scenario['scenario_id']}/confirm",
            "POST",
            {
                "factor_values": scenario["factor_values"],
                "prediction_at": "2023-06-20T12:00:00Z",
            },
        )
        self.assertEqual((status, confirmed["scenario"]["review_status"]), (200, "confirmed"))
        status, predicted = self.json_request(
            f"/api/people/{person_id}/cognitive/scenarios/{scenario['scenario_id']}/predict",
            "POST",
            {},
        )
        prediction = predicted["prediction"]
        self.assertEqual((status, prediction["status"]), (200, "predicted"))
        self.assertEqual(prediction["behavior_baseline"]["status"], "insufficient_evidence")
        status, outcome = self.json_request(
            f"/api/people/{person_id}/cognitive/predictions/{prediction['prediction_id']}/outcome",
            "POST",
            {
                "actual_choice": 1,
                "actual_rationale": "AI companies should face ordinary liability and parents should be able to make their case.",
                "observed_at": "2023-12-13T12:00:00Z",
                "source": "https://www.hawley.senate.gov/hawleys-bipartisan-ai-bill-empower-parents-hold-big-tech-accountable-blocked-senate-floor/",
                "source_locator": "press release paragraphs 1-4",
                "confirm_real_external_result": True,
            },
        )
        self.assertEqual((status, outcome["outcome"]["updated_card_version"]), (200, 2))

    def test_plain_chinese_error_is_returned(self) -> None:
        status, body = self.json_request(
            "/api/people",
            "POST",
            {"name": "", "feature_names": []},
        )
        self.assertEqual(status, 400)
        self.assertIn("人物名称", body["message"])


    def test_expression_renderer_http_loop(self) -> None:
        status, profile = self.json_request("/api/expression/profile")
        self.assertEqual(status, 200)
        self.assertEqual(profile["profile"]["profile_id"], "steve-jobs-surface-en-v1")
        self.assertEqual(
            profile["profile"]["validation_status"],
            "semantic_gate_passed_style_recognition_not_assessed",
        )

        contract = {
            "schema_version": "pcfm-frozen-content-contract-v1",
            "speech_act": "answer",
            "claims": [{"id": "C1", "text": "Orchid Team should not ship 17 units on 2027-04-03."}],
            "reasons": [{"id": "R1", "text": "It can be completed with the evidence currently available."}],
            "memories": [],
            "uncertainties": [{"id": "U1", "text": "The result may change after user testing."}],
            "protected_entities": ["Orchid Team"],
            "protected_numbers": ["17", "2027-04-03"],
            "protected_quotes": [],
            "confidence": 0.37,
            "style_mode": "interview_public",
        }
        status, rendered = self.json_request(
            "/api/expression/render",
            "POST",
            {"contract": contract, "include_adversarial_probe": True},
        )
        self.assertEqual(status, 200)
        rendered = rendered["render"]
        self.assertEqual(rendered["structured_content"], contract)
        self.assertEqual(len(rendered["candidates"]), 3)
        self.assertTrue(all(item["status"] == "passed" for item in rendered["candidates"]))
        self.assertEqual(rendered["selected"]["intensity"], "strong")
        self.assertEqual(rendered["adversarial_probe"]["status"], "rejected")

        status, records = self.json_request("/api/expression/renders")
        self.assertEqual(status, 200)
        self.assertEqual(len(records["renders"]), 1)


    def test_conversation_http_people_sources_chat_and_comparison(self) -> None:
        status, created = self.json_request(
            "/api/conversation/people",
            "POST",
            {
                "name": "HTTP Alice",
                "aliases": ["Alice"],
                "language": "en",
                "description": "Conversation HTTP test",
            },
        )
        self.assertEqual(status, 201)
        person_id = created["person"]["person_id"]
        status, source = self.json_request(
            f"/api/people/{person_id}/conversation/sources/text",
            "POST",
            {
                "title": "Model interview",
                "text": "Q: What matters most?\nA: Evidence and careful iteration matter most.",
                "speaker": "HTTP Alice",
                "source_date": "2025-01-01",
                "dataset_role": "model_source",
                "content_authenticity": "verbatim_transcript",
                "source_locator": "transcript paragraphs 1-2",
                "source_context": "Recorded public interview",
                "source_url": "https://example.org/http-alice-transcript",
            },
        )
        self.assertEqual(status, 201)
        source_id = source["source"]["source_id"]
        status, reviewed = self.json_request(
            f"/api/people/{person_id}/conversation/sources/{source_id}/review",
            "POST",
            {"decision": "confirmed"},
        )
        self.assertEqual((status, reviewed["source"]["review_status"]), (200, "confirmed"))
        status, sent = self.json_request(
            f"/api/people/{person_id}/conversation/messages",
            "POST",
            {"text": "What matters most?", "reality_lookup_requested": False},
        )
        self.assertEqual(status, 201)
        job_id = sent["job_id"]
        message = None
        for _ in range(200):
            _, job_state = self.json_request(f"/api/jobs/{job_id}")
            if job_state["job"]["status"] in {"succeeded", "failed"}:
                message = job_state["job"].get("result")
                break
            time.sleep(0.05)
        self.assertIsNotNone(message, "send_message 任务未完成")
        self.assertEqual(message["status"], "answered")
        self.assertEqual(message["prediction_trace"]["kernel"], "simulation-v5")
        self.assertEqual(message["prediction_trace"]["generative_content_calls"], 0)
        status, summary = self.json_request(
            f"/api/people/{person_id}/conversation"
        )
        self.assertEqual(status, 200)
        self.assertEqual(summary["conversation"]["telemetry"]["reality_local_search_calls"], 0)
        self.assertEqual(len(summary["conversation"]["messages"]), 2)

        status, _archived = self.json_request(
            f"/api/people/{person_id}", "DELETE"
        )
        self.assertEqual(status, 200)
        status, archived = self.json_request("/api/archived-people")
        self.assertEqual(status, 200)
        self.assertIn(person_id, [item["person_id"] for item in archived["people"]])
        status, restored = self.json_request(
            f"/api/archived-people/{person_id}/restore", "POST", {}
        )
        self.assertEqual((status, restored["person"]["person_id"]), (200, person_id))
        status, restored_summary = self.json_request(
            f"/api/people/{person_id}/conversation"
        )
        self.assertEqual(len(restored_summary["conversation"]["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
