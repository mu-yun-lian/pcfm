from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from pcfm.product_service import ProductError, ProductService
from pcfm.webapp import APP_VERSION, build_server


QA = """Q: How should a small team launch a product?
A: It should launch a focused version after the evidence is strong enough.
"""

LOOK_STYLE_QA = """Q: How should a small team launch a product?
A: It should launch a focused version after the evidence is strong enough.

Q: What matters in an early prototype?
A: Look, the prototype should expose the hardest assumption.

Q: When should a team simplify?
A: Look, simplify when complexity hides whether the product works.

Q: How should design feedback be used?
A: Look, feedback matters when it reveals a real use problem.

Q: What should a review meeting produce?
A: Look, the review should produce one clear next test.
"""

WELL_STYLE_QA = LOOK_STYLE_QA.replace("Look,", "Well,")


class FakePublicSearch:
    provider_id = "fake-public-search"

    def search(self, *, person_name: str, identity_note: str, language: str, limit: int):
        return [
            {
                "title": f"{person_name} interview transcript",
                "url": "https://example.org/verified-interview",
                "snippet": "Interview transcript candidate returned by the configured provider.",
                "published_at": "2025-01-01",
                "provider_rank": 1,
            }
        ][:limit]


class AlignmentV04Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _service(self, **kwargs) -> ProductService:
        return ProductService(self.storage, seed_example=False, **kwargs)

    def _person(self, service: ProductService, name: str = "Alice Example") -> dict[str, object]:
        return service.create_conversation_person(
            name=name,
            aliases=["Alice"] if name == "Alice Example" else [],
            language="en",
            source_mode="user_provided",
            focus_domain="product interviews",
        )

    def _verified_source(self, service: ProductService, person_id: str) -> dict[str, object]:
        source = service.add_conversation_text_source(
            person_id,
            title="Verified transcript",
            text=QA,
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="transcript paragraphs 10-11",
            source_context="Public recorded interview",
            source_url="https://example.org/alice-transcript",
        )
        return service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )

    def _verified_style_source(
        self, service: ProductService, person_id: str, *, person_name: str, text: str
    ) -> dict[str, object]:
        source = service.add_conversation_text_source(
            person_id,
            title="Verified multi-question transcript",
            text=text,
            speaker=person_name,
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="verbatim_transcript",
            source_locator="transcript questions 1-5",
            source_context="Public recorded interview",
            source_url="https://example.org/verified-style-transcript",
        )
        return service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )

    def test_summary_or_translation_without_original_never_becomes_training_truth(self) -> None:
        service = self._service()
        person = self._person(service)
        person_id = str(person["person_id"])
        source = service.add_conversation_text_source(
            person_id,
            title="Chinese editorial summary",
            text="Q: 应该怎样发布产品？\nA: 应该先聚焦，再根据反馈迭代。",
            speaker="Alice Example",
            source_date="2025-01-01",
            dataset_role="model_source",
            content_authenticity="editorial_summary",
            source_locator="editor notes paragraph 2",
            source_context="Researcher-created Chinese summary",
            source_url="https://example.org/original",
        )
        reviewed = service.review_conversation_source(
            person_id, str(source["source_id"]), "confirmed"
        )
        self.assertEqual(reviewed["dataset_role"], "reference_only")
        self.assertTrue(
            all(event["label_status"] == "unverified_candidate" for event in reviewed["response_events"])
        )
        self.assertIsNone(service.conversation_summary(person_id)["active_version"])

    def test_legacy_self_disclaimed_summary_is_downgraded_and_version_invalidated(self) -> None:
        service = self._service()
        person = self._person(service)
        person_id = str(person["person_id"])
        self._verified_source(service, person_id)
        source_path = self.storage / "people" / person_id / "conversation_sources.json"
        sources = json.loads(source_path.read_text(encoding="utf-8"))
        sources[0]["text"] += "\nThis is an editorial summary and is not the person's verbatim response."
        sources[0].pop("content_authenticity", None)
        source_path.write_text(json.dumps(sources), encoding="utf-8")

        migrated = self._service()
        summary = migrated.conversation_summary(person_id)
        self.assertIsNone(summary["active_version"])
        self.assertEqual(summary["sources"][0]["dataset_role"], "reference_only")
        self.assertEqual(summary["versions"][0]["validation_status"], "invalidated_evidence_contract")
        with self.assertRaisesRegex(ProductError, "证据契约"):
            migrated.rollback_conversation_version(person_id, 1)

    def test_configured_name_search_creates_real_pending_candidates_without_training(self) -> None:
        service = self._service(public_search=FakePublicSearch())
        self.assertTrue(service.capabilities()["public_search"]["available"])
        person = service.create_conversation_person(
            name="Public Person",
            language="en",
            source_mode="system_search",
            identity_note="technology executive",
            focus_domain="public interviews",
        )
        detail = service.get_person(str(person["person_id"]))
        summary = service.conversation_summary(str(person["person_id"]))
        self.assertEqual(detail["collection"]["status"], "candidates_found")
        self.assertEqual(detail["collection"]["provider"], "fake-public-search")
        self.assertEqual(len(summary["sources"]), 1)
        self.assertEqual(summary["sources"][0]["source_type"], "system_search_result")
        self.assertEqual(summary["sources"][0]["dataset_role"], "reference_only")
        self.assertIsNone(summary["active_version"])

    def test_unconfigured_search_is_disabled_and_rejected_before_person_creation(self) -> None:
        service = self._service(public_search=False)
        capability = service.capabilities()["public_search"]
        self.assertFalse(capability["available"])
        self.assertEqual(capability["status"], "not_configured")
        with self.assertRaisesRegex(ProductError, "未配置"):
            service.create_conversation_person(
                name="Unavailable Search Person",
                language="en",
                source_mode="system_search",
            )
        self.assertEqual(service.list_people(), [])

    def test_creation_form_defaults_to_user_materials_and_disables_unconfigured_search(self) -> None:
        static_dir = Path(__file__).parents[1] / "src" / "pcfm" / "web_static"
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        script = (static_dir / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            '<option value="system_search" disabled>系统自动搜索公开资料（暂未配置）</option>',
            html,
        )
        self.assertIn('<option value="user_provided" selected>我自行提供原始资料</option>', html)
        self.assertIn("configureSearchCapability", script)

    def test_verified_source_uses_shared_pcfm_map_core_and_creates_person_style_artifact(self) -> None:
        service = self._service()
        person = self._person(service)
        person_id = str(person["person_id"])
        self._verified_source(service, person_id)
        summary = service.conversation_summary(person_id)
        model = service.conversation._response_model(person_id, 1)
        self.assertIn("pcfm_core_map_person_encoder", model["active_components"])
        version = summary["versions"][0]
        self.assertEqual(version["style_update_status"], "style_material_ready_rendering_not_enabled")
        style_path = self.storage / "people" / person_id / str(version["style_artifact_path"])
        style = json.loads(style_path.read_text(encoding="utf-8"))
        self.assertEqual(style["person_id"], person_id)
        self.assertNotIn("beliefs", style)
        self.assertNotIn("values", style)
        self.assertTrue(style["surface_statistics"])

        reply = service.send_conversation_message(
            person_id, "How should a small team launch a product?"
        )
        context = reply["prediction_trace"]["conversation_context"]
        for field in (
            "current_topic", "recent_context", "relationship", "occasion",
            "time_stage", "prior_claims", "dynamic_state",
        ):
            self.assertIn(field, context)
        self.assertEqual(
            reply["style_status"], "source_verbatim_person_style"
        )
        self.assertFalse(reply["style_gate"]["changed"])

    def test_person_style_changes_surface_only_when_observed_rules_exist(self) -> None:
        service = self._service()
        first = self._person(service, name="Look Speaker")
        second = self._person(service, name="Well Speaker")
        first_id, second_id = str(first["person_id"]), str(second["person_id"])
        self._verified_style_source(
            service, first_id, person_name="Look Speaker", text=LOOK_STYLE_QA
        )
        self._verified_style_source(
            service, second_id, person_name="Well Speaker", text=WELL_STYLE_QA
        )
        contract = {
            "schema_version": "pcfm-frozen-content-contract-v2",
            "speech_act": "answer",
            "stance": "conditional_support",
            "refusal_status": "not_refused",
            "claims": [{"id": "C1", "text": "The team should ship 17 units on 2027-04-03."}],
            "reasons": [{"id": "R1", "text": "The evidence may still change after testing."}],
            "memories": [],
            "uncertainties": [{"id": "U1", "text": "The result may remain incomplete."}],
            "protected_entities": ["The team"],
            "protected_numbers": ["17"],
            "protected_dates": ["2027-04-03"],
            "protected_quotes": [],
            "confidence": 0.61,
            "style_mode": "interview_public",
        }
        look_text, look_status, look_gate = service.conversation._render_reply(first_id, contract)
        well_text, well_status, well_gate = service.conversation._render_reply(second_id, contract)
        neutral = "\n".join(
            item["text"] for field in ("claims", "reasons", "memories", "uncertainties") for item in contract[field]
        )
        self.assertEqual(look_status, "person_style_applied")
        self.assertEqual(well_status, "person_style_applied")
        self.assertNotEqual(look_text, neutral)
        self.assertNotEqual(well_text, neutral)
        self.assertNotEqual(look_text, well_text)
        self.assertTrue(look_gate["changed"])
        self.assertTrue(well_gate["changed"])
        self.assertEqual(look_gate["status"], "passed")
        self.assertEqual(well_gate["status"], "passed")
        for locked in ("The team should ship 17 units on 2027-04-03.", "The evidence may still change after testing.", "The result may remain incomplete."):
            self.assertEqual(look_text.count(locked), 1)
            self.assertEqual(well_text.count(locked), 1)

    def test_final_holdout_never_enters_person_style_artifact(self) -> None:
        service = self._service()
        person = self._person(service, name="Holdout Style Speaker")
        person_id = str(person["person_id"])
        self._verified_style_source(
            service,
            person_id,
            person_name="Holdout Style Speaker",
            text=LOOK_STYLE_QA,
        )
        holdout = service.add_conversation_text_source(
            person_id,
            title="Sealed later transcript",
            text="Q: What should remain sealed?\nA: Well, this later answer remains independent.",
            speaker="Holdout Style Speaker",
            source_date="2025-02-01",
            dataset_role="final_holdout",
            content_authenticity="verbatim_transcript",
            source_locator="transcript question 9",
            source_context="Later recorded interview",
            source_url="https://example.org/sealed-style-holdout",
        )
        reviewed = service.review_conversation_source(
            person_id, str(holdout["source_id"]), "confirmed"
        )
        version = service.conversation_summary(person_id)["versions"][0]
        artifact = json.loads(
            (self.storage / "people" / person_id / version["style_artifact_path"]).read_text(encoding="utf-8")
        )
        self.assertTrue(set(artifact["source_event_ids"]).isdisjoint(
            {event["event_id"] for event in reviewed["response_events"]}
        ))

    def test_archive_metadata_and_name_confirmed_permanent_delete(self) -> None:
        service = self._service()
        person = self._person(service, name="Disposable Archive Person")
        person_id = str(person["person_id"])
        service.delete_person(person_id)
        archived = service.list_archived_people()[0]
        self.assertEqual(archived["name"], "Disposable Archive Person")
        self.assertIn("source_count", archived)
        self.assertIn("message_count", archived)
        self.assertIn("avatar", archived)
        with self.assertRaisesRegex(ProductError, "人物名称"):
            service.permanently_delete_archived_person(person_id, expected_name="wrong")
        service.permanently_delete_archived_person(
            person_id, expected_name="Disposable Archive Person"
        )
        self.assertEqual(service.list_archived_people(), [])

    def test_health_exposes_frontend_version_contract(self) -> None:
        service = self._service()
        server = build_server("127.0.0.1", 0, self.storage, seed_example=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/health"
            )
            with urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                self.assertEqual(payload["app_version"], APP_VERSION)
                self.assertEqual(response.headers["X-PCFM-Version"], APP_VERSION)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_people_cards_can_request_archive_by_menu_or_drag_drop(self) -> None:
        static_dir = Path(__file__).parents[1] / "src" / "pcfm" / "web_static"
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        script = (static_dir / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="archive-drop-zone"', html)
        self.assertIn("也可以把人物卡片拖到这里", html)
        self.assertIn('draggable="true"', script)
        self.assertIn(">更多</button>", script)
        self.assertIn(">拖动</span>", script)
        self.assertIn('addEventListener("dragstart"', script)
        self.assertIn('addEventListener("drop"', script)
        self.assertIn('addEventListener("pointerdown"', script)
        self.assertIn('document.addEventListener("pointerup"', script)
        self.assertIn('document.addEventListener("mouseup"', script)
        self.assertIn("requestArchive(personId)", script)
        self.assertIn('$("#archive-confirm-dialog").showModal()', script)

    def test_frontend_has_one_formal_render_path_and_no_legacy_confirm_delete(self) -> None:
        script = (
            Path(__file__).parents[1] / "src" / "pcfm" / "web_static" / "app.js"
        ).read_text(encoding="utf-8")
        for function_name in ("renderPeople", "renderSources", "renderVersions", "loadArchive"):
            self.assertEqual(script.count(f"function {function_name}("), 1, function_name)
        self.assertNotIn("Aligned", script)
        self.assertNotIn("renderPeople =", script)
        self.assertNotIn("renderWorkspaceBase", script)
        self.assertNotIn("confirm(", script)

    def test_windows_launcher_uses_only_supported_webapp_arguments(self) -> None:
        launcher = (
            Path(__file__).parents[1] / "start_pcfm_simulator.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("python -m pcfm.webapp", launcher)
        self.assertIn("--seed-demos", launcher)
        self.assertNotIn("--no-seed", launcher)


if __name__ == "__main__":
    unittest.main()
