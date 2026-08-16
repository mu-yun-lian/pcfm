from __future__ import annotations

import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pcfm.product_service import ProductError, ProductService


def diagnostic_records(count: int = 50) -> list[dict[str, object]]:
    start = datetime.now(timezone.utc) - timedelta(days=20)
    records = []
    for index in range(count):
        x = math.sin(index / 5.0)
        records.append(
            {
                "scenario_id": f"history-{index:03d}",
                "observed_at": (start + timedelta(hours=index)).isoformat(),
                "question": "是否执行固定类型方案？",
                "option_a": "执行",
                "option_b": "不执行",
                "choice": "B" if x > 0 else "A",
                "domain": "方案选择",
                "features": {"条件得分": x, "常数项": 1.0},
            }
        )
    return records


class ProductServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.service = ProductService(Path(self.temporary.name), seed_example=False)
        self.person = self.service.create_person(
            name="测试人物",
            description="本地产品测试",
            feature_names=("条件得分", "常数项"),
        )
        self.person_id = str(self.person["person_id"])

    def tearDown(self) -> None:
        self.service.close()
        self.temporary.cleanup()

    def test_ordinary_json_train_predict_update_export_and_reload(self) -> None:
        reference = self.service.create_person(
            name="真实参照人物",
            description="相同字段的真实参照",
            feature_names=("条件得分", "常数项"),
        )
        self.service.import_history(
            str(reference["person_id"]),
            diagnostic_records(),
            input_format="json",
        )
        imported = self.service.import_history(
            self.person_id,
            diagnostic_records(),
            input_format="json",
        )
        self.assertEqual(imported["sample_count"], 50)
        model = self.service.train(self.person_id)
        self.assertEqual(model["validation_status"], "unvalidated")
        self.assertEqual(model["reference_mode"], "real_multi_person")
        self.assertEqual(model["model_kind"], "behavior_baseline_logistic")
        scenario = {
            "scenario_id": "new-choice",
            "question": "是否执行固定类型方案？",
            "option_a": "执行",
            "option_b": "不执行",
            "domain": "方案选择",
            "features": {"条件得分": 0.1, "常数项": 1.0},
        }
        refused = self.service.predict(self.person_id, scenario)
        self.assertEqual(refused["status"], "refused")
        self.assertIn("model_validation_unvalidated", refused["reasons"])
        prediction = self.service.predict(
            self.person_id,
            scenario,
            diagnostic_override=True,
        )
        self.assertEqual(prediction["status"], "predicted")
        self.assertAlmostEqual(
            prediction["probability_a"] + prediction["probability_b"],
            1.0,
        )
        self.assertIn("不代表真实信念", prediction["influence_notice"])
        outcome = self.service.record_outcome(
            self.person_id,
            str(prediction["prediction_id"]),
            1,
        )
        self.assertEqual(outcome["updated_model_version"], 2)
        detail = self.service.get_person(self.person_id)
        self.assertEqual(len(detail["versions"]), 2)
        self.assertEqual(detail["prediction_metrics"]["sample_count"], 1)

        exported = self.service.export_person(self.person_id)
        other = tempfile.TemporaryDirectory()
        try:
            restored_service = ProductService(Path(other.name), seed_example=False)
            restored = restored_service.import_product_export(exported)
            self.assertEqual(restored["person_id"], self.person_id)
            self.assertEqual(len(restored["versions"]), 2)
            restored_prediction = restored_service.predict(
                self.person_id,
                {**scenario, "scenario_id": "restored-choice"},
                diagnostic_override=True,
            )
            self.assertEqual(restored_prediction["status"], "predicted")
        finally:
            other.cleanup()

    def test_csv_import_edit_and_delete(self) -> None:
        csv_text = self.service.csv_template(self.person_id)
        result = self.service.import_history(
            self.person_id,
            csv_text,
            input_format="csv",
        )
        self.assertEqual(result["imported_count"], 1)
        updated = self.service.update_person(
            self.person_id,
            {"name": "已编辑人物", "description": "修改完成"},
        )
        self.assertEqual(updated["name"], "已编辑人物")
        with self.assertRaises(ProductError):
            self.service.update_person(
                self.person_id,
                {"feature_names": ["新字段"]},
            )
        self.service.delete_person(self.person_id)
        self.assertEqual(self.service.list_people(), [])

    def test_invalid_replacement_restores_existing_person(self) -> None:
        malformed = {
            "format": "pcfm-local-product-v1",
            "person_id": self.person_id,
            "files": {"person.json": "{}"},
        }
        with self.assertRaises(ProductError):
            self.service.import_product_export(malformed, replace=True)
        self.assertEqual(self.service.get_person(self.person_id)["name"], "测试人物")

    def test_built_in_example_passes_and_completes_closed_loop(self) -> None:
        other = tempfile.TemporaryDirectory()
        try:
            service = ProductService(Path(other.name), seed_example=True)
            detail = service.get_person("example-person")
            self.assertEqual(detail["sample_count"], 580)
            model = service.train("example-person")
            self.assertEqual(model["validation_status"], "passed")
            prediction = service.predict(
                "example-person",
                detail["suggested_scenario"],
            )
            self.assertEqual(prediction["status"], "predicted")
            outcome = service.record_outcome(
                "example-person",
                str(prediction["prediction_id"]),
                detail["suggested_actual_choice"],
            )
            self.assertEqual(outcome["updated_model_version"], 2)
        finally:
            other.cleanup()


if __name__ == "__main__":
    unittest.main()
