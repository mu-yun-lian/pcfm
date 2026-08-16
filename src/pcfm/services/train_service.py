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



class TrainServiceMixin:
    def _normalize_history_record(
        self,
        person: Mapping[str, object],
        raw: Mapping[str, object],
        *,
        provenance: str = "human_record",
        role: str = "pending",
    ) -> dict[str, object]:
        option_a = str(raw.get("option_a", raw.get("option_0", ""))).strip()
        option_b = str(raw.get("option_b", raw.get("option_1", ""))).strip()
        if not option_a or not option_b or option_a == option_b:
            raise ProductError("每条记录需要两个不同且非空的选项。")
        observed_at = str(raw.get("observed_at", raw.get("date", ""))).strip()
        _parse_time(observed_at)
        scenario_id = str(raw.get("scenario_id", "")).strip() or uuid.uuid4().hex
        features_raw = raw.get("features")
        if isinstance(features_raw, str):
            try:
                features_raw = json.loads(features_raw)
            except json.JSONDecodeError as error:
                raise ProductError("features 列必须是 JSON 对象。") from error
        if not isinstance(features_raw, Mapping):
            features_raw = {name: raw.get(name) for name in person["feature_names"]}
        features: dict[str, float] = {}
        for name in person["feature_names"]:
            try:
                value = float(features_raw[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ProductError(f"数值影响项“{name}”缺失或不是数字。") from error
            if not math.isfinite(value):
                raise ProductError(f"数值影响项“{name}”必须是有限数字。")
            features[str(name)] = value
        context_raw = raw.get("context", {})
        if isinstance(context_raw, str):
            try:
                context_raw = json.loads(context_raw) if context_raw.strip() else {}
            except json.JSONDecodeError as error:
                raise ProductError("context 列必须是 JSON 对象。") from error
        if not isinstance(context_raw, Mapping):
            raise ProductError("情境补充信息必须是键值对象。")
        actual_raw = raw.get("actual_choice", raw.get("choice"))
        if actual_raw is None:
            raise ProductError("每条历史记录都需要真实选择。")
        return {
            "record_id": str(raw.get("record_id", "")).strip() or uuid.uuid4().hex,
            "scenario_id": scenario_id,
            "observed_at": observed_at,
            "question": str(raw.get("question", raw.get("situation", ""))).strip(),
            "option_a": option_a,
            "option_b": option_b,
            "actual_choice": _as_choice(actual_raw, option_a, option_b),
            "features": features,
            "domain": str(raw.get("domain", "structured_choice")).strip() or "structured_choice",
            "context": {str(key): str(value) for key, value in context_raw.items()},
            "confidence": float(raw["confidence"]) if raw.get("confidence") not in {None, ""} else None,
            "reaction_time_ms": float(raw["reaction_time_ms"]) if raw.get("reaction_time_ms") not in {None, ""} else None,
            "provenance": provenance,
            "role": role,
        }

    def import_history(
        self,
        person_id: str,
        payload: object,
        *,
        input_format: str,
    ) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            if input_format == "csv":
                if not isinstance(payload, str):
                    raise ProductError("CSV 内容无效。")
                rows = list(csv.DictReader(io.StringIO(payload.lstrip("\ufeff"))))
            elif input_format == "json":
                parsed = json.loads(payload) if isinstance(payload, str) else payload
                if isinstance(parsed, Mapping) and "records" in parsed:
                    parsed = parsed["records"]
                if not isinstance(parsed, list):
                    raise ProductError("JSON 必须是记录数组，或包含 records 数组。")
                rows = parsed
            elif input_format == "form":
                rows = [payload]
            else:
                raise ProductError("不支持的数据格式。")
            if not rows:
                raise ProductError("没有可导入的记录。")
            normalized = [self._normalize_history_record(person, dict(row)) for row in rows]
            history = self._history(person_id)
            scenario_ids = {str(item["scenario_id"]) for item in history}
            for record in normalized:
                if record["scenario_id"] in scenario_ids:
                    raise ProductError(f"情境编号重复：{record['scenario_id']}")
                scenario_ids.add(str(record["scenario_id"]))
            history.extend(normalized)
            _write_json(self._person_dir(person_id) / "history.json", history)
            person["updated_at"] = _utc_now()
            _write_json(self._person_path(person_id), person)
            return {"imported_count": len(normalized), "sample_count": len(history)}

    def import_decision_evidence(
        self,
        person_id: str,
        bundle_data: Mapping[str, object],
        verification_keys: Mapping[str, str],
    ) -> dict[str, object]:
        with self._lock:
            self._require_person(person_id)
            authority = VerificationAuthority(
                {str(key): str(value).encode("utf-8") for key, value in verification_keys.items()}
            )
            bundle = decision_evidence_bundle_from_dict(bundle_data, authority)
            summary = {
                "artifact_digest": bundle.artifact_digest,
                "record_count": len(bundle.records),
                "source_count": len(bundle.sources),
                "training_authorized": bundle.training_authorized,
                "message": "高级证据包已验证并保存，但不会自动进入普通训练。",
            }
            directory = self._person_dir(person_id) / "advanced_evidence"
            _write_json(directory / f"{bundle.artifact_digest}.json", bundle.to_dict())
            _write_json(directory / f"{bundle.artifact_digest}.summary.json", summary)
            return summary

    # ---------- conversion, partitioning and ledgers ----------

    def _observation(
        self,
        person_id: str,
        record: Mapping[str, object],
    ) -> Observation:
        features = dict(record["features"])
        return Observation(
            person_id=person_id,
            scenario=Scenario(
                scenario_id=str(record["scenario_id"]),
                features=tuple(float(features[name]) for name in features),
                feature_names=tuple(str(name) for name in features),
                options=(str(record["option_a"]), str(record["option_b"])),
                domain=str(record.get("domain", "structured_choice")),
                context={
                    "question": str(record.get("question", "")),
                    **{str(key): str(value) for key, value in dict(record.get("context", {})).items()},
                },
            ),
            actual_choice=int(record["actual_choice"]),
            confidence=(float(record["confidence"]) if record.get("confidence") is not None else None),
            reaction_time_ms=(float(record["reaction_time_ms"]) if record.get("reaction_time_ms") is not None else None),
            provenance=str(record.get("provenance", "human_record")),
        )

    def _partition_history(
        self, history: Sequence[Mapping[str, object]]
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        explicit = all(str(item.get("role", "pending")) in {"training", "update", "applicability", "validation"} for item in history)
        if explicit:
            training = [dict(item) for item in history if item["role"] in {"training", "update"}]
            applicability = [dict(item) for item in history if item["role"] == "applicability"]
            validation = [dict(item) for item in history if item["role"] == "validation"]
            return training, applicability, validation
        ordered = sorted((dict(item) for item in history), key=lambda item: (_parse_time(str(item["observed_at"])), str(item["scenario_id"])))
        count = len(ordered)
        if count >= 200:
            return ordered[:-150], ordered[-150:-100], ordered[-100:]
        if count >= 100:
            return ordered[:-50], ordered[-50:], []
        return ordered, [], []

    def _sign_records(
        self,
        observations: Sequence[tuple[Observation, Mapping[str, object]]],
        authority: VerificationAuthority,
        prefix: str,
    ) -> EventLedger:
        records = []
        for index, (observation, raw) in enumerate(observations):
            evidence = observation_payload(observation)
            observed_at = str(raw["observed_at"])
            verified_at = max(_parse_time(observed_at), datetime.now(timezone.utc)).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            records.append(
                authority.sign(
                    event_id=f"{prefix}:{index:06d}:{observation.person_id}:{observation.scenario.scenario_id}",
                    observation=observation,
                    observed_at=observed_at,
                    evidence_hash=_canonical_hash(evidence),
                    verifier_id=VERIFIER_ID,
                    verified_at=verified_at,
                )
            )
        return EventLedger.verify(records, authority)

    def _example_population(
        self,
        person_id: str,
    ) -> tuple[list[tuple[Observation, dict[str, object]]], list[dict[str, object]]]:
        people, source, _target = generate_population_dataset(
            seed=401,
            person_count=24,
            source_trials=140,
            target_trials=1,
            heterogeneity_scale=1.5,
        )
        history = self._history(person_id)
        training, _app, _validation = self._partition_history(history)
        target_items = [(self._observation(person_id, item), item) for item in training]
        reference_time = str(training[0]["observed_at"])
        for person in people[1:]:
            for item in source[person.person_id]:
                transformed = Observation(
                    person_id=f"example-reference-{person.person_id}",
                    scenario=Scenario(
                        scenario_id=f"reference-{item.scenario.scenario_id}",
                        features=item.scenario.features,
                        feature_names=FEATURE_NAMES,
                        options=("执行方案", "暂不执行"),
                        domain="日常方案选择",
                        context={"question": "合成群体参考题"},
                    ),
                    actual_choice=item.actual_choice,
                    confidence=item.confidence,
                    reaction_time_ms=item.reaction_time_ms,
                    provenance="synthetic_ground_truth",
                )
                target_items.append((transformed, {"observed_at": reference_time}))
        return target_items, training

    def _real_population(
        self,
        target_person: Mapping[str, object],
        target_training: list[dict[str, object]],
    ) -> tuple[list[tuple[Observation, dict[str, object]]], bool]:
        target_id = str(target_person["person_id"])
        items = [(self._observation(target_id, record), record) for record in target_training]
        people_used = 1
        for other in self.list_people():
            other_id = str(other["person_id"])
            if other_id == target_id or tuple(other["feature_names"]) != tuple(target_person["feature_names"]):
                continue
            other_training, _app, _validation = self._partition_history(self._history(other_id))
            if len(other_training) < MINIMUM_PROFILE_SAMPLES:
                continue
            for record in other_training:
                items.append((self._observation(other_id, record), record))
            people_used += 1
        return items, people_used >= 2

    # ---------- model training, prediction and update ----------

    def train(self, person_id: str) -> dict[str, object]:
        with self._lock:
            person = self._require_person(person_id)
            history = self._history(person_id)
            training, applicability, validation = self._partition_history(history)
            if len(training) < MINIMUM_PROFILE_SAMPLES:
                raise ProductError(
                    f"当前只有 {len(history)} 条可用历史记录；现有模型至少需要 {MINIMUM_PROFILE_SAMPLES} 条训练记录。"
                )
            assigned_roles = {
                str(item["record_id"]): role
                for role, records in (
                    ("training", training),
                    ("applicability", applicability),
                    ("validation", validation),
                )
                for item in records
            }
            for item in history:
                record_id = str(item["record_id"])
                if record_id in assigned_roles:
                    item["role"] = assigned_roles[record_id]
            _write_json(self._person_dir(person_id) / "history.json", history)
            authority = self._authority(person)
            reference_mode = "real_multi_person"
            if person.get("is_example"):
                population_items, training = self._example_population(person_id)
                real_population = True
                reference_mode = "synthetic_demo_population"
            else:
                population_items, real_population = self._real_population(person, training)
                if not real_population:
                    raise ProductError(
                        "行为基线证据不足：至少需要另一个具有相同数值字段、且样本充足的真实参照人物。系统不会再自动注入合成人群。"
                    )
            training_ledger = self._sign_records(population_items, authority, "training")
            applicability_ledger = (
                self._sign_records(
                    [(self._observation(person_id, item), item) for item in applicability],
                    authority,
                    "applicability",
                )
                if applicability
                else None
            )
            validation_ledger = (
                self._sign_records(
                    [(self._observation(person_id, item), item) for item in validation],
                    authority,
                    "validation",
                )
                if validation and real_population
                else None
            )
            bundle = fit_person_model(
                training_ledger,
                authority,
                applicability_ledger=applicability_ledger,
                validation_ledger=validation_ledger,
                person_id=person_id,
                feature_names=tuple(str(name) for name in person["feature_names"]),
            )
            version = self._save_version(
                person_id,
                bundle,
                training_ledger,
                applicability_ledger,
                validation_ledger,
                source="train",
                reference_mode=reference_mode,
            )
            return self._version_public(version, bundle)

    def _save_version(
        self,
        person_id: str,
        bundle,
        training_ledger: EventLedger,
        applicability_ledger: EventLedger | None,
        validation_ledger: EventLedger | None,
        *,
        source: str,
        reference_mode: str,
    ) -> dict[str, object]:
        versions = self._versions(person_id)
        number = len(versions) + 1
        directory = self._person_dir(person_id)
        model_rel = f"models/model-{number:04d}.json"
        training_rel = f"ledgers/training-{number:04d}.jsonl"
        app_rel = f"ledgers/applicability-{number:04d}.jsonl" if applicability_ledger else None
        val_rel = f"ledgers/validation-{number:04d}.jsonl" if validation_ledger else None
        save_bundle(directory / model_rel, bundle)
        save_event_ledger_jsonl(directory / training_rel, training_ledger)
        if applicability_ledger:
            save_event_ledger_jsonl(directory / str(app_rel), applicability_ledger)
        if validation_ledger:
            save_event_ledger_jsonl(directory / str(val_rel), validation_ledger)
        validation = bundle.manifest.validation
        version = {
            "version": number,
            "created_at": _utc_now(),
            "source": source,
            "model_id": bundle.manifest.model_id,
            "parent_model_id": bundle.manifest.parent_model_id,
            "model_path": model_rel,
            "training_ledger_path": training_rel,
            "applicability_ledger_path": app_rel,
            "validation_ledger_path": val_rel,
            "training_sample_count": bundle.representation.observation_count,
            "validation_sample_count": validation.sample_count,
            "validation_status": validation.status,
            "personal_nll": validation.personal_nll,
            "calibration_error": validation.calibration_error,
            "validation_reasons": list(validation.reasons),
            "reference_mode": reference_mode,
        }
        versions.append(version)
        _write_json(directory / "versions.json", versions)
        return version

    def _latest_bundle(self, person_id: str):
        versions = self._versions(person_id)
        if not versions:
            raise ProductError("这个人物还没有训练模型。")
        latest = versions[-1]
        bundle = load_bundle(self._person_dir(person_id) / str(latest["model_path"]))
        return latest, bundle
