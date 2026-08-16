from __future__ import annotations

from ._shared import *  # noqa: F401, F403
from ._shared import (  # noqa: F401
    _as_choice,
    _canonical_hash,
    _parse_time,
    _read_json,
    _reason_text,
    _slug,
    _utc_now,
    _write_json,
)



class CognitiveServiceMixin:
    def add_cognitive_evidence(
        self, person_id: str, raw: Mapping[str, object]
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.add_evidence, person_id, raw)

    def review_cognitive_evidence(
        self, person_id: str, evidence_id: str, decision: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.review_evidence,
                person_id,
                evidence_id,
                decision,
            )

    def generate_cognitive_card(self, person_id: str) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.generate_card, person_id)

    def draft_cognitive_scenario(
        self, person_id: str, text: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(self.cognitive.draft_scenario, person_id, text)

    def confirm_cognitive_scenario(
        self,
        person_id: str,
        scenario_id: str,
        changes: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.confirm_scenario,
                person_id,
                scenario_id,
                changes,
            )

    def _behavior_baseline_for_cognitive(
        self, person_id: str, scenario: Mapping[str, object]
    ) -> dict[str, object]:
        versions = self._versions(person_id)
        if not versions:
            return {
                "status": "insufficient_evidence",
                "reason": "没有可比较的行为基线版本。",
                "model_kind": "behavior_baseline_logistic",
            }
        latest = versions[-1]
        if latest.get("reference_mode") != "real_multi_person":
            return {
                "status": "insufficient_evidence",
                "reason": "现有行为基线没有真实多人物参照，不能用于普通人物推演。",
                "model_kind": "behavior_baseline_logistic",
            }
        person = self._require_person(person_id)
        factor_values = dict(scenario.get("factor_values", {}))
        if set(map(str, person["feature_names"])) != set(map(str, factor_values)):
            return {
                "status": "incompatible_structure",
                "reason": "行为基线数值字段与本次认知情境因素不同。",
                "model_kind": "behavior_baseline_logistic",
            }
        raw = {
            "scenario_id": f"behavior-{scenario['scenario_id']}",
            "question": scenario["original_text"],
            "option_a": scenario["option_a"],
            "option_b": scenario["option_b"],
            "features": factor_values,
            "domain": scenario["domain"],
            "context": {"decision_type": scenario["decision_type"]},
            "prediction_at": scenario["prediction_at"],
        }
        baseline = self.predict(person_id, raw, diagnostic_override=False)
        return {
            "status": baseline["status"],
            "predicted_choice": baseline.get("predicted_choice"),
            "probability_a": baseline.get("probability_a"),
            "probability_b": baseline.get("probability_b"),
            "reasons": baseline.get("reasons", []),
            "model_kind": "behavior_baseline_logistic",
        }

    def predict_cognitive_scenario(
        self, person_id: str, scenario_id: str
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            scenario = self._cognitive_call(
                self.cognitive.get_scenario, person_id, scenario_id
            )
            baseline = self._behavior_baseline_for_cognitive(person_id, scenario)
            return self._cognitive_call(
                self.cognitive.predict,
                person_id,
                scenario_id,
                behavior_baseline=baseline,
            )

    def record_cognitive_outcome(
        self,
        person_id: str,
        prediction_id: str,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            return self._cognitive_call(
                self.cognitive.record_outcome,
                person_id,
                prediction_id,
                payload,
            )

    def _data_sufficiency(self, training: int, applicability: int, validation: int) -> dict[str, object]:
        if training < MINIMUM_PROFILE_SAMPLES:
            message = f"还差 {MINIMUM_PROFILE_SAMPLES - training} 条训练记录，才能建立当前模型的适用范围。"
            level = "insufficient"
        elif applicability < MINIMUM_PROFILE_SAMPLES:
            message = "可以训练诊断模型，但还没有独立的适用域校准数据。"
            level = "diagnostic"
        elif validation < MINIMUM_VALIDATION_SAMPLES:
            message = f"可以训练，但还差 {MINIMUM_VALIDATION_SAMPLES - validation} 条独立验证记录，不能声称模型已验证。"
            level = "diagnostic"
        else:
            message = "样本数量达到训练、适用域校准和独立验证的最低数量门槛；是否通过仍取决于 NLL、校准和时间稳定性。"
            level = "validation_ready"
        return {"level": level, "message": message}
