from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pcfm.services import ProductError, PcfmService


CHALLENGE = (
    "Should Congress remove Section 230 immunity for claims arising from "
    "generative AI output produced by large technology companies, so harmed "
    "users and parents can sue?"
)


class CognitiveWorkbenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = PcfmService(Path(self.temporary.name), seed_example=True)
        self.person_id = "josh-hawley-section230"

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def confirm_all_seed_evidence(self) -> list[dict[str, object]]:
        detail = self.service.get_person(self.person_id)
        evidence = detail["cognitive"]["evidence"]
        for item in evidence:
            self.service.review_cognitive_evidence(
                self.person_id, str(item["evidence_id"]), "confirmed"
            )
        return evidence

    def test_complete_evidence_card_scenario_prediction_outcome_loop(self) -> None:
        detail = self.service.get_person(self.person_id)
        cognitive = detail["cognitive"]
        self.assertEqual(cognitive["status"], "evidence_review_required")
        self.assertEqual(len(cognitive["evidence"]), 5)
        self.assertTrue(all(item["review_status"] == "pending" for item in cognitive["evidence"]))
        with self.assertRaises(ProductError):
            self.service.generate_cognitive_card(self.person_id)

        self.confirm_all_seed_evidence()
        card = self.service.generate_cognitive_card(self.person_id)
        self.assertEqual(card["version"], 1)
        self.assertEqual(card["validation_status"], "exploratory_unvalidated")
        for category in (
            "beliefs",
            "values",
            "causal_assumptions",
            "decision_rules",
            "risk_preferences",
            "dynamic_state",
            "contradictions",
            "unknowns",
        ):
            self.assertIn(category, card)
        for item in card["all_items"]:
            self.assertIn(item["status"], {"observed", "inferred", "contested", "unknown"})
            self.assertIn("evidence_ids", item)
            self.assertIn("scope", item)
            self.assertIn("confidence", item)
            self.assertIn("valid_from", item)

        draft = self.service.draft_cognitive_scenario(self.person_id, CHALLENGE)
        self.assertEqual(draft["review_status"], "pending")
        self.assertTrue(draft["high_impact_uncertainties"])
        with self.assertRaises(ProductError):
            self.service.predict_cognitive_scenario(self.person_id, str(draft["scenario_id"]))
        confirmed = self.service.confirm_cognitive_scenario(
            self.person_id,
            str(draft["scenario_id"]),
            {"factor_values": draft["factor_values"], "prediction_at": "2023-06-20T12:00:00Z"},
        )
        self.assertEqual(confirmed["review_status"], "confirmed")

        prediction = self.service.predict_cognitive_scenario(
            self.person_id, str(draft["scenario_id"])
        )
        self.assertEqual(prediction["status"], "predicted")
        self.assertAlmostEqual(
            prediction["probability_a"] + prediction["probability_b"], 1.0
        )
        self.assertTrue(prediction["drivers"])
        self.assertTrue(all(driver["evidence"] for driver in prediction["drivers"]))
        self.assertIn("flip_conditions", prediction)
        self.assertIn("unknowns", prediction)
        self.assertEqual(prediction["applicability"]["status"], "within_scope")
        self.assertEqual(prediction["behavior_baseline"]["status"], "insufficient_evidence")
        self.assertEqual(prediction["claim_status"], "product_loop_complete_only")

        with self.assertRaisesRegex(ProductError, "模型自己的解释"):
            self.service.record_cognitive_outcome(
                self.person_id,
                str(prediction["prediction_id"]),
                {
                    "actual_choice": 1,
                    "actual_rationale": prediction["explanation"],
                    "observed_at": "2023-12-13T12:00:00Z",
                    "source": "https://example.invalid/model-output",
                    "source_locator": "model output",
                    "confirm_real_external_result": True,
                },
            )

        updated = self.service.record_cognitive_outcome(
            self.person_id,
            str(prediction["prediction_id"]),
            {
                "actual_choice": 1,
                "actual_rationale": (
                    "Generative AI companies should face ordinary liability and harmed "
                    "parents and users should be able to make their case in court."
                ),
                "observed_at": "2023-12-13T12:00:00Z",
                "source": "https://www.hawley.senate.gov/hawleys-bipartisan-ai-bill-empower-parents-hold-big-tech-accountable-blocked-senate-floor/",
                "source_locator": "press release paragraphs 1-4 and Senate floor remarks",
                "confirm_real_external_result": True,
            },
        )
        self.assertEqual(updated["updated_card_version"], 2)
        self.assertTrue(updated["version_change"]["modified"])
        self.assertTrue(updated["version_change"]["unchanged"])
        self.assertNotEqual(updated["actual_rationale"], prediction["explanation"])

        exported = self.service.export_person(self.person_id)
        other = tempfile.TemporaryDirectory()
        try:
            restored = PcfmService(Path(other.name), seed_example=False)
            try:
                restored.import_product_export(exported)
                restored_detail = restored.get_person(self.person_id)
                self.assertEqual(restored_detail["cognitive"]["latest_card"]["version"], 2)
                self.assertEqual(restored_detail["cognitive"]["outcome_count"], 1)
            finally:
                restored.close()
        finally:
            other.cleanup()

    def test_unconfirmed_llm_candidate_never_enters_card(self) -> None:
        self.confirm_all_seed_evidence()
        candidate = self.service.add_cognitive_evidence(
            self.person_id,
            {
                "text": "Model-proposed summary that has not been checked.",
                "source": "https://example.invalid/unconfirmed",
                "date": "2022-01-01",
                "context": "candidate only",
                "position": "unknown",
                "explicit_rationale": "candidate only",
                "role": "model_inference",
                "domain": "section_230_platform_liability",
                "confidence": 0.4,
                "source_locator": "not verified",
                "extraction_method": "llm_candidate",
                "candidate_model_items": [
                    {
                        "category": "beliefs",
                        "statement": "Unverified model claim",
                        "status": "inferred",
                        "decision_weight": 1.0,
                        "factor_weights": {"large_platform_power": 1.0},
                    }
                ],
            },
        )
        self.assertEqual(candidate["review_status"], "pending")
        card = self.service.generate_cognitive_card(self.person_id)
        statements = {item["statement"] for item in card["all_items"]}
        self.assertNotIn("Unverified model claim", statements)

    def test_duplicate_evidence_is_rejected(self) -> None:
        evidence = self.service.get_person(self.person_id)["cognitive"]["evidence"][0]
        with self.assertRaisesRegex(ProductError, "重复"):
            self.service.add_cognitive_evidence(self.person_id, evidence)

    def test_scope_and_order_invariance(self) -> None:
        self.confirm_all_seed_evidence()
        card = self.service.generate_cognitive_card(self.person_id)
        draft = self.service.draft_cognitive_scenario(self.person_id, CHALLENGE)
        self.service.confirm_cognitive_scenario(
            self.person_id,
            str(draft["scenario_id"]),
            {"factor_values": draft["factor_values"], "prediction_at": "2023-06-20T12:00:00Z"},
        )
        first = self.service.predict_cognitive_scenario(self.person_id, str(draft["scenario_id"]))
        shuffled = copy.deepcopy(card)
        shuffled["all_items"] = list(reversed(shuffled["all_items"]))
        self.assertEqual(
            first["kernel_check"],
            self.service.cognitive.score_for_test(shuffled, self.service.cognitive.get_scenario(self.person_id, str(draft["scenario_id"]))) ["kernel_check"],
        )

        outside = self.service.draft_cognitive_scenario(
            self.person_id, "Should the senator support a farm subsidy?"
        )
        outside["domain"] = "agriculture_subsidy"
        confirmed = self.service.confirm_cognitive_scenario(
            self.person_id,
            str(outside["scenario_id"]),
            {
                "domain": "agriculture_subsidy",
                "factor_values": outside["factor_values"],
                "prediction_at": "2023-06-20T12:00:00Z",
            },
        )
        self.assertEqual(confirmed["review_status"], "confirmed")
        refused = self.service.predict_cognitive_scenario(
            self.person_id, str(outside["scenario_id"])
        )
        self.assertEqual(refused["status"], "refused")
        self.assertIn("domain_out_of_scope", refused["reasons"])

    def test_real_behavior_baseline_does_not_inject_synthetic_people(self) -> None:
        person = self.service.create_person(
            name="真实人物行为基线测试",
            description="只有一个真实人物",
            feature_names=("signal",),
        )
        records = [
            {
                "scenario_id": f"r-{index}",
                "observed_at": f"2024-01-{(index % 28) + 1:02d}T12:00:00Z",
                "question": "test",
                "option_a": "A",
                "option_b": "B",
                "choice": index % 2,
                "domain": "test",
                "features": {"signal": float(index % 3)},
            }
            for index in range(60)
        ]
        self.service.import_history(str(person["person_id"]), records, input_format="json")
        with self.assertRaisesRegex(ProductError, "真实参照人物"):
            self.service.train(str(person["person_id"]))


if __name__ == "__main__":
    unittest.main()
