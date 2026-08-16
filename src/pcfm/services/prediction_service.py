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



class PredictionServiceMixin:
    def _scenario_from_input(
        self, person: Mapping[str, object], raw: Mapping[str, object]
    ) -> Scenario:
        features_raw = raw.get("features", {})
        if not isinstance(features_raw, Mapping):
            raise ProductError("数值影响项格式无效。")
        names = tuple(str(name) for name in person["feature_names"])
        try:
            values = tuple(float(features_raw[name]) for name in names)
        except (KeyError, TypeError, ValueError) as error:
            raise ProductError("请填写全部数值影响项，并确保都是数字。") from error
        option_a = str(raw.get("option_a", "")).strip()
        option_b = str(raw.get("option_b", "")).strip()
        if not option_a or not option_b:
            raise ProductError("请填写两个候选选项。")
        context = raw.get("context", {})
        if not isinstance(context, Mapping):
            context = {}
        return Scenario(
            scenario_id=str(raw.get("scenario_id", "")).strip() or f"prediction-{uuid.uuid4().hex}",
            features=values,
            feature_names=names,
            options=(option_a, option_b),
            domain=str(raw.get("domain", "structured_choice")).strip() or "structured_choice",
            context={"question": str(raw.get("question", "")).strip(), **{str(key): str(value) for key, value in context.items()}},
        )

    def predict(
        self,
        person_id: str,
        raw: Mapping[str, object],
        *,
        diagnostic_override: bool = False,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            version, bundle = self._latest_bundle(person_id)
            scenario = self._scenario_from_input(person, raw)
            prediction_at = str(raw.get("prediction_at", "")).strip() or _utc_now()
            record: dict[str, object] = {
                "prediction_id": uuid.uuid4().hex,
                "created_at": _utc_now(),
                "prediction_at": prediction_at,
                "person_id": person_id,
                "model_version": version["version"],
                "model_id": version["model_id"],
                "scenario": {
                    "scenario_id": scenario.scenario_id,
                    "question": scenario.context.get("question", ""),
                    "option_a": scenario.options[0],
                    "option_b": scenario.options[1],
                    "features": dict(zip(scenario.feature_names, scenario.features, strict=True)),
                    "domain": scenario.domain,
                    "context": dict(scenario.context),
                },
                "actual_choice": None,
                "updated_model_version": None,
            }
            try:
                prediction = predict_with_bundle(
                    bundle,
                    scenario,
                    prediction_at=prediction_at,
                    validation_override=diagnostic_override,
                    applicability_override=False,
                )
                effective_weights = np.asarray(bundle.population_model.weights) + np.asarray(bundle.adapter.delta_weights)
                contributions = [
                    {
                        "name": name,
                        "feature_value": float(value),
                        "effective_weight": float(weight),
                        "logit_contribution": float(value * weight),
                    }
                    for name, value, weight in zip(
                        bundle.population_model.feature_names,
                        scenario.ordered_features(bundle.population_model.feature_names),
                        effective_weights,
                        strict=True,
                    )
                ]
                contributions.sort(key=lambda item: abs(float(item["logit_contribution"])), reverse=True)
                p_b = prediction.probability_option_1
                record.update(
                    {
                        "status": "predicted",
                        "probability_a": 1.0 - p_b,
                        "probability_b": p_b,
                        "predicted_choice": prediction.predicted_choice,
                        "probability_lower_95_b": prediction.probability_lower_95,
                        "probability_upper_95_b": prediction.probability_upper_95,
                        "applicability_status": prediction.applicability_status,
                        "applicability_warnings": list(prediction.applicability_warnings),
                        "applicability_warning_text": [_reason_text(item) for item in prediction.applicability_warnings],
                        "ood_score": prediction.ood_score,
                        "ood_threshold": prediction.ood_threshold,
                        "local_ood_score": prediction.local_ood_score,
                        "local_ood_threshold": prediction.local_ood_threshold,
                        "validation_status": prediction.validation_status,
                        "gate_overrides": list(prediction.gate_overrides),
                        "diagnostic_override": bool(prediction.gate_overrides),
                        "influences": contributions,
                        "influence_notice": "这些只是模型中的数值贡献，不代表真实信念、价值观或心理机制。",
                    }
                )
            except PredictionRefusedError as error:
                record.update(
                    {
                        "status": "refused",
                        "reasons": list(error.reasons),
                        "reason_text": [_reason_text(item) for item in error.reasons],
                        "ood_score": error.ood_score,
                        "ood_threshold": error.ood_threshold,
                        "local_ood_score": error.local_ood_score,
                        "local_ood_threshold": error.local_ood_threshold,
                    }
                )
            predictions = self._predictions(person_id)
            predictions.append(record)
            _write_json(self._person_dir(person_id) / "predictions.json", predictions)
            return record

    def record_outcome(
        self,
        person_id: str,
        prediction_id: str,
        actual_choice: object,
        observed_at: str | None = None,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            predictions = self._predictions(person_id)
            try:
                record = next(item for item in predictions if item["prediction_id"] == prediction_id)
            except StopIteration as error:
                raise ProductError("预测记录不存在。") from error
            if record.get("actual_choice") is not None:
                raise ProductError("这条预测已经录入真实结果。")
            scenario_raw = dict(record["scenario"])
            choice = _as_choice(actual_choice, str(scenario_raw["option_a"]), str(scenario_raw["option_b"]))
            record["actual_choice"] = choice
            record["outcome_recorded_at"] = _utc_now()
            if record.get("status") != "predicted":
                record["comparison"] = {"status": "prediction_was_refused"}
                _write_json(self._person_dir(person_id) / "predictions.json", predictions)
                return record
            versions = self._versions(person_id)
            latest = versions[-1]
            if int(record["model_version"]) != int(latest["version"]):
                raise ProductError("这条预测来自旧模型。为避免分叉，请在最新模型上重新预测。")
            directory = self._person_dir(person_id)
            authority = self._authority(person)
            bundle = load_bundle(directory / str(latest["model_path"]))
            training_ledger = load_event_ledger_jsonl(directory / str(latest["training_ledger_path"]), authority)
            app_path = latest.get("applicability_ledger_path")
            val_path = latest.get("validation_ledger_path")
            applicability_ledger = load_event_ledger_jsonl(directory / str(app_path), authority) if app_path else None
            validation_ledger = load_event_ledger_jsonl(directory / str(val_path), authority) if val_path else None
            outcome_time = observed_at or _utc_now()
            _parse_time(outcome_time)
            history_record = self._normalize_history_record(
                person,
                {
                    **scenario_raw,
                    "observed_at": outcome_time,
                    "actual_choice": choice,
                    "record_id": f"prediction-{prediction_id}",
                },
                role="update",
            )
            observation = self._observation(person_id, history_record)
            verified_at = max(_parse_time(outcome_time), datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            outcome = authority.sign(
                event_id=f"outcome:{prediction_id}",
                observation=observation,
                observed_at=outcome_time,
                evidence_hash=_canonical_hash(observation_payload(observation)),
                verifier_id=VERIFIER_ID,
                verified_at=verified_at,
            )
            updated = update_person_model(
                bundle,
                training_ledger,
                outcome,
                authority,
                applicability_ledger=applicability_ledger,
                validation_ledger=validation_ledger,
            )
            new_version = self._save_version(
                person_id,
                updated.bundle,
                updated.ledger,
                applicability_ledger,
                validation_ledger,
                source="prediction_outcome_update",
                reference_mode=str(latest["reference_mode"]),
            )
            probability = float(record["probability_b"])
            selected_probability = probability if choice == 1 else 1.0 - probability
            record["comparison"] = {
                "correct": int(record["predicted_choice"]) == choice,
                "selected_probability": selected_probability,
                "nll": -math.log(max(selected_probability, 1e-9)),
            }
            record["updated_model_version"] = new_version["version"]
            history = self._history(person_id)
            history.append(history_record)
            _write_json(directory / "history.json", history)
            _write_json(directory / "predictions.json", predictions)
            return record

    # ---------- views, metrics and portability ----------

    def _version_public(self, version: Mapping[str, object], bundle=None) -> dict[str, object]:
        reasons = list(version.get("validation_reasons", []))
        result = {key: value for key, value in version.items() if not str(key).endswith("_path")}
        result["validation_reason_text"] = [_reason_text(str(item)) for item in reasons]
        result["model_kind"] = "behavior_baseline_logistic"
        result["display_name"] = "行为基线模型（Logistic）"
        if version.get("reference_mode") == "synthetic_compatibility_only":
            result["model_explanation"] = "只有一个真实人物可用于当前特征结构，因此群体参照是合成兼容数据；模型只能作诊断使用，不能声称人物化已验证。"
        elif version.get("reference_mode") == "synthetic_demo_population":
            result["model_explanation"] = "这是内置合成演示的验证结果，只证明软件流程可运行，不代表真实人物效果。"
        else:
            result["model_explanation"] = "模型使用了本地多个同特征结构人物作为群体参照。"
        return result

    def _prediction_metrics(self, person_id: str) -> dict[str, object] | None:
        completed = [item for item in self._predictions(person_id) if item.get("status") == "predicted" and item.get("actual_choice") is not None]
        if not completed:
            return None
        observations = []
        probabilities = []
        for item in completed:
            raw = dict(item["scenario"])
            person = self._require_person(person_id)
            scenario = self._scenario_from_input(person, raw)
            observations.append(Observation(person_id=person_id, scenario=scenario, actual_choice=int(item["actual_choice"])))
            probabilities.append(float(item["probability_b"]))
        return report_to_dict(evaluate_probability_array(observations, probabilities))
