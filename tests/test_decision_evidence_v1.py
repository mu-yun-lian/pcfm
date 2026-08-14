from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.decision_evidence_v1 import (
    DecisionEvidenceConfig,
    DecisionEvidenceRefusedError,
    DecisionEvidenceRecord,
    DecisionOption,
    EvidenceCitation,
    SourceSnapshot,
    create_decision_evidence_bundle,
    decision_evidence_bundle_from_dict,
    decision_evidence_record_from_dict,
    load_decision_evidence_bundle,
    save_decision_evidence_bundle,
    source_snapshot_from_dict,
    validate_decision_evidence_bundle,
)
from pcfm.ledger import VerificationAuthority
from pcfm.registry import ModuleSlot


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class DecisionEvidenceV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = VerificationAuthority(
            {"evidence-verifier": b"decision-evidence-secret"}
        )
        cls.created_at = "2026-02-01T00:00:00Z"

    def _source(
        self,
        source_id: str,
        *,
        event_id: str,
        content: str,
        published_at: str,
        provenance: str,
        author_person_id: str | None = None,
        captured_at: str | None = None,
    ) -> SourceSnapshot:
        return SourceSnapshot(
            source_id=source_id,
            source_locator=f"https://evidence.example/{source_id}",
            provenance=provenance,
            published_at=published_at,
            captured_at=captured_at or published_at,
            linked_event_id=event_id,
            content=content,
            content_digest=_digest(content),
            author_person_id=author_person_id,
        )

    def _fixture(self):
        person_id = "person-alpha"
        fit_time = datetime(2026, 1, 10, 12, tzinfo=timezone.utc)
        confirm_time = datetime(2026, 1, 20, 9, tzinfo=timezone.utc)

        fit_context = self._source(
            "fit-context",
            event_id="fit-event",
            content=(
                "是否批准城市交通方案？选项：批准方案；拒绝方案。"
            ),
            published_at=_timestamp(fit_time - timedelta(days=1)),
            provenance="official_primary",
        )
        fit_choice = self._source(
            "fit-choice",
            event_id="fit-event",
            content="公开记录：person-alpha 选择批准方案。",
            published_at=_timestamp(fit_time + timedelta(minutes=5)),
            provenance="official_primary",
        )
        fit_rationale = self._source(
            "fit-rationale",
            event_id="fit-event",
            content="person-alpha 表示：公共收益超过短期财政成本。",
            published_at=_timestamp(fit_time + timedelta(hours=1)),
            provenance="human_primary",
            author_person_id=person_id,
        )
        fit_record = DecisionEvidenceRecord(
            event_id="fit-event",
            person_id=person_id,
            domain="municipal-policy",
            task="public-binary-decision",
            decision_at=_timestamp(fit_time),
            question_text="是否批准城市交通方案？",
            options=(
                DecisionOption("approve", "批准方案"),
                DecisionOption("reject", "拒绝方案"),
            ),
            chosen_option_id="approve",
            evidence_role="parameter_fitting",
            role_assigned_at="2026-01-31T00:00:00Z",
            role_assignment_reference=_digest("fit-role-plan"),
            context_citations=(
                EvidenceCitation(
                    "fit-context",
                    "是否批准城市交通方案？",
                ),
            ),
            choice_citations=(
                EvidenceCitation(
                    "fit-choice",
                    "person-alpha 选择批准方案",
                ),
            ),
            rationale_citations=(
                EvidenceCitation(
                    "fit-rationale",
                    "公共收益超过短期财政成本",
                ),
            ),
        )

        confirm_context = self._source(
            "confirm-context",
            event_id="confirm-event",
            content=(
                "是否采用河道修复预算？选项：采用预算；维持现状。"
            ),
            published_at=_timestamp(confirm_time - timedelta(days=2)),
            provenance="archival_primary_snapshot",
        )
        confirm_choice = self._source(
            "confirm-choice",
            event_id="confirm-event",
            content="公开记录：person-alpha 选择采用预算。",
            published_at=_timestamp(confirm_time + timedelta(minutes=10)),
            provenance="official_primary",
        )
        confirm_consequence = self._source(
            "confirm-consequence",
            event_id="confirm-event",
            content="该次选择触发了已登记的河道修复预算执行。",
            published_at=_timestamp(confirm_time + timedelta(days=1)),
            provenance="verified_external_consequence",
        )
        confirm_record = DecisionEvidenceRecord(
            event_id="confirm-event",
            person_id=person_id,
            domain="municipal-policy",
            task="public-binary-decision",
            decision_at=_timestamp(confirm_time),
            question_text="是否采用河道修复预算？",
            options=(
                DecisionOption("adopt", "采用预算"),
                DecisionOption("status-quo", "维持现状"),
            ),
            chosen_option_id="adopt",
            evidence_role="sealed_confirmation",
            role_assigned_at="2026-01-19T00:00:00Z",
            role_assignment_reference=_digest("confirmation-registry"),
            context_citations=(
                EvidenceCitation(
                    "confirm-context",
                    "是否采用河道修复预算？",
                ),
            ),
            choice_citations=(
                EvidenceCitation(
                    "confirm-choice",
                    "person-alpha 选择采用预算",
                ),
            ),
            consequence_citations=(
                EvidenceCitation(
                    "confirm-consequence",
                    "触发了已登记的河道修复预算执行",
                ),
            ),
        )
        sources = (
            fit_context,
            fit_choice,
            fit_rationale,
            confirm_context,
            confirm_choice,
            confirm_consequence,
        )
        return sources, (fit_record, confirm_record)

    def _bundle(self, *, sources=None, records=None, config=None):
        default_sources, default_records = self._fixture()
        return create_decision_evidence_bundle(
            sources=tuple(sources or default_sources),
            records=tuple(records or default_records),
            created_at=self.created_at,
            authority=self.authority,
            verifier_id="evidence-verifier",
            config=config or DecisionEvidenceConfig(),
        )

    def test_valid_rationale_and_consequence_records_pass(self) -> None:
        bundle = self._bundle()
        summary = validate_decision_evidence_bundle(
            bundle,
            self.authority,
        )
        self.assertEqual(summary.status, "admission_contract_satisfied")
        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.rationale_record_count, 1)
        self.assertEqual(summary.consequence_record_count, 1)
        self.assertEqual(
            dict(summary.role_counts),
            {"parameter_fitting": 1, "sealed_confirmation": 1},
        )
        self.assertFalse(summary.training_authorized)
        self.assertEqual(summary.semantic_claims_authorized, ())

    def test_missing_explanation_or_choice_is_refused(self) -> None:
        sources, records = self._fixture()
        missing_explanation = replace(
            records[0],
            rationale_citations=(),
        )
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "explanation_evidence_required",
        ):
            self._bundle(
                sources=sources,
                records=(missing_explanation, records[1]),
            )

        missing_choice = replace(records[0], choice_citations=())
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "choice_evidence_required",
        ):
            self._bundle(
                sources=sources,
                records=(missing_choice, records[1]),
            )

    def test_contract_does_not_promote_semantic_or_training_status(self) -> None:
        bundle = self._bundle()
        self.assertEqual(bundle.module_status, "implemented_unintegrated")
        self.assertFalse(bundle.training_authorized)
        self.assertEqual(bundle.semantic_claims_authorized, ())
        self.assertNotIn(
            "decision_evidence",
            {slot.value for slot in ModuleSlot},
        )

    def test_cross_role_replay_and_near_duplicate_are_refused(self) -> None:
        sources, records = self._fixture()
        repeated = replace(
            records[0],
            event_id="renamed-event",
            evidence_role="sealed_confirmation",
        )
        renamed_sources = tuple(
            replace(source, linked_event_id="renamed-event")
            if source.linked_event_id == "fit-event"
            else source
            for source in sources
        )
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "decision_design_replay",
        ):
            self._bundle(
                sources=renamed_sources,
                records=(records[0], repeated),
            )

        context = next(
            item for item in sources if item.source_id == "confirm-context"
        )
        near_content = (
            "是否采用河道修复预算？选项：采用预算；维持现状！"
        )
        near_source = replace(
            context,
            source_id="near-context",
            linked_event_id="near-event",
            content=near_content,
            content_digest=_digest(near_content),
        )
        near_choice_content = "公开记录：person-alpha 选择采用预算。"
        near_choice = self._source(
            "near-choice",
            event_id="near-event",
            content=near_choice_content,
            published_at="2026-01-25T09:05:00Z",
            provenance="official_primary",
        )
        near_consequence = self._source(
            "near-consequence",
            event_id="near-event",
            content="该次选择再次触发了河道预算执行。",
            published_at="2026-01-26T09:00:00Z",
            provenance="verified_external_consequence",
        )
        near_record = replace(
            records[1],
            event_id="near-event",
            decision_at="2026-01-25T09:00:00Z",
            question_text="是否采用河道修复预算？",
            context_citations=(
                EvidenceCitation(
                    "near-context",
                    "是否采用河道修复预算？",
                ),
            ),
            choice_citations=(
                EvidenceCitation(
                    "near-choice",
                    "person-alpha 选择采用预算",
                ),
            ),
            consequence_citations=(
                EvidenceCitation(
                    "near-consequence",
                    "再次触发了河道预算执行",
                ),
            ),
        )
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "near_duplicate_decision_content",
        ):
            self._bundle(
                sources=(*sources, near_source, near_choice, near_consequence),
                records=(*records, near_record),
            )

    def test_input_order_and_identifier_renaming_are_invariant(self) -> None:
        sources, records = self._fixture()
        forward = self._bundle(sources=sources, records=records)
        reverse = self._bundle(
            sources=tuple(reversed(sources)),
            records=tuple(reversed(records)),
        )
        self.assertEqual(forward, reverse)

        event_map = {
            "fit-event": "renamed-fit",
            "confirm-event": "renamed-confirm",
        }
        source_map = {
            source.source_id: f"renamed-{index}"
            for index, source in enumerate(sources)
        }
        renamed_sources = tuple(
            replace(
                source,
                source_id=source_map[source.source_id],
                linked_event_id=event_map[source.linked_event_id],
            )
            for source in sources
        )

        def rename_citations(citations):
            return tuple(
                replace(item, source_id=source_map[item.source_id])
                for item in citations
            )

        renamed_records = tuple(
            replace(
                record,
                event_id=event_map[record.event_id],
                context_citations=rename_citations(
                    record.context_citations
                ),
                choice_citations=rename_citations(
                    record.choice_citations
                ),
                rationale_citations=rename_citations(
                    record.rationale_citations
                ),
                consequence_citations=rename_citations(
                    record.consequence_citations
                ),
            )
            for record in records
        )
        renamed = self._bundle(
            sources=renamed_sources,
            records=renamed_records,
        )
        original_summary = validate_decision_evidence_bundle(
            forward,
            self.authority,
        )
        renamed_summary = validate_decision_evidence_bundle(
            renamed,
            self.authority,
        )
        self.assertEqual(
            original_summary.role_counts,
            renamed_summary.role_counts,
        )
        self.assertEqual(
            original_summary.content_fingerprint,
            renamed_summary.content_fingerprint,
        )

    def test_timing_authorship_provenance_and_hard_ceiling_attacks(self) -> None:
        sources, records = self._fixture()
        rationale_index = next(
            index
            for index, source in enumerate(sources)
            if source.source_id == "fit-rationale"
        )
        context_index = next(
            index
            for index, source in enumerate(sources)
            if source.source_id == "fit-context"
        )

        attacks = []
        late = list(sources)
        late[rationale_index] = replace(
            late[rationale_index],
            published_at="2026-01-18T13:00:00Z",
            captured_at="2026-01-18T13:00:00Z",
        )
        attacks.append((late, "rationale_outside_time_window"))

        wrong_author = list(sources)
        wrong_author[rationale_index] = replace(
            wrong_author[rationale_index],
            author_person_id="person-beta",
        )
        attacks.append((wrong_author, "rationale_wrong_person"))

        generated = list(sources)
        generated[rationale_index] = replace(
            generated[rationale_index],
            provenance="model_generated",
        )
        attacks.append((generated, "source_provenance_not_allowed"))

        late_context = list(sources)
        late_context[context_index] = replace(
            late_context[context_index],
            published_at="2026-01-10T12:01:00Z",
            captured_at="2026-01-10T12:01:00Z",
        )
        attacks.append((late_context, "context_not_available_prechoice"))

        for attacked_sources, reason in attacks:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(
                    DecisionEvidenceRefusedError,
                    reason,
                ):
                    self._bundle(
                        sources=attacked_sources,
                        records=records,
                    )

        with self.assertRaisesRegex(ValueError, "hard ceiling"):
            DecisionEvidenceConfig(max_rationale_lag_hours=169.0)
        for invalid in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                DecisionEvidenceConfig(
                    max_rationale_lag_hours=invalid
                )

        late_assignment = replace(
            records[1],
            role_assigned_at="2026-01-20T09:01:00Z",
        )
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "sealed_role_assigned_after_decision",
        ):
            self._bundle(
                sources=sources,
                records=(records[0], late_assignment),
            )

    def test_creation_load_and_verify_share_one_admission_result(self) -> None:
        bundle = self._bundle()
        created = validate_decision_evidence_bundle(
            bundle,
            self.authority,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            save_decision_evidence_bundle(path, bundle)
            loaded = load_decision_evidence_bundle(
                path,
                self.authority,
            )
        loaded_summary = validate_decision_evidence_bundle(
            loaded,
            self.authority,
        )
        self.assertEqual(loaded, bundle)
        self.assertEqual(loaded_summary, created)

    def test_person_domain_task_option_and_time_scope_are_bound(self) -> None:
        sources, records = self._fixture()
        baseline = self._bundle(sources=sources, records=records)
        for field, value in (
            ("person_id", "person-other"),
            ("domain", "another-domain"),
            ("task", "another-task"),
        ):
            changed_records = (
                replace(records[0], **{field: value}),
                records[1],
            )
            if field == "person_id":
                changed_sources = tuple(
                    replace(source, author_person_id="person-other")
                    if source.source_id == "fit-rationale"
                    else source
                    for source in sources
                )
            else:
                changed_sources = sources
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    DecisionEvidenceRefusedError,
                    "scope_role_coverage_incomplete",
                ):
                    self._bundle(
                        sources=changed_sources,
                        records=changed_records,
                    )

        changed_time = replace(
            records[0],
            decision_at="2026-01-10T11:30:00Z",
        )
        time_bound = self._bundle(
            sources=sources,
            records=(changed_time, records[1]),
        )
        self.assertNotEqual(
            time_bound.artifact_digest,
            baseline.artifact_digest,
        )

        changed_options = replace(
            records[0],
            options=(
                DecisionOption("approve", "批准修订方案"),
                DecisionOption("reject", "拒绝方案"),
            ),
            choice_citations=(
                EvidenceCitation(
                    "fit-choice",
                    "person-alpha 选择批准修订方案",
                ),
            ),
        )
        changed_context_content = (
            "是否批准城市交通方案？选项：批准修订方案；拒绝方案。"
        )
        changed_choice_content = (
            "公开记录：person-alpha 选择批准修订方案。"
        )
        changed_sources = tuple(
            replace(
                source,
                content=changed_context_content,
                content_digest=_digest(changed_context_content),
            )
            if source.source_id == "fit-context"
            else replace(
                source,
                content=changed_choice_content,
                content_digest=_digest(changed_choice_content),
            )
            if source.source_id == "fit-choice"
            else source
            for source in sources
        )
        changed = self._bundle(
            sources=changed_sources,
            records=(changed_options, records[1]),
        )
        self.assertNotEqual(changed.artifact_digest, baseline.artifact_digest)

        context_source = next(
            source for source in sources if source.source_id == "fit-context"
        )
        incomplete_content = "是否批准城市交通方案？"
        incomplete_context = replace(
            context_source,
            content=incomplete_content,
            content_digest=_digest(incomplete_content),
        )
        incomplete_sources = tuple(
            incomplete_context
            if source.source_id == "fit-context"
            else source
            for source in sources
        )
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "option_text_not_in_context_source",
        ):
            self._bundle(sources=incomplete_sources, records=records)

    def test_round_trip_tamper_and_resign_recomputation(self) -> None:
        bundle = self._bundle()
        data = bundle.to_dict()
        data["sources"][0]["content"] += "被篡改"
        with self.assertRaisesRegex(ValueError, "content_digest"):
            source_snapshot_from_dict(data["sources"][0])

        data = bundle.to_dict()
        data["role_counts"] = {"parameter_fitting": 2}
        base = {
            key: value
            for key, value in data.items()
            if key not in {"artifact_digest", "signature"}
        }
        data["artifact_digest"] = hashlib.sha256(
            json.dumps(
                base,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload = {
            key: value for key, value in data.items() if key != "signature"
        }
        data["signature"] = self.authority.sign_payload(
            payload,
            "evidence-verifier",
        )
        with self.assertRaisesRegex(
            (ValueError, DecisionEvidenceRefusedError),
            "role_counts",
        ):
            decision_evidence_bundle_from_dict(data, self.authority)

        old = bundle.to_dict()
        old["schema_version"] = "decision-evidence-v0"
        with self.assertRaisesRegex(ValueError, "bundle version"):
            decision_evidence_bundle_from_dict(old, self.authority)

        forged = bundle.to_dict()
        forged["signature"] = "0" * 64
        with self.assertRaisesRegex(
            DecisionEvidenceRefusedError,
            "bundle_signature_invalid",
        ):
            decision_evidence_bundle_from_dict(forged, self.authority)

    def test_cli_build_and_verify(self) -> None:
        sources, records = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.json"
            keys_path = root / "keys.json"
            output_path = root / "bundle.json"
            input_path.write_text(
                json.dumps(
                    {
                        "config": DecisionEvidenceConfig().to_dict(),
                        "sources": [item.to_dict() for item in sources],
                        "records": [item.to_dict() for item in records],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            keys_path.write_text(
                json.dumps(
                    {"evidence-verifier": "decision-evidence-secret"}
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            project_root = Path(__file__).resolve().parents[1]
            env["PYTHONPATH"] = str(project_root / "src")

            def run(*arguments: str):
                return subprocess.run(
                    [sys.executable, "-m", "pcfm", *arguments],
                    capture_output=True,
                    text=True,
                    env=env,
                    check=False,
                )

            built = run(
                "decision-evidence-build-v1",
                "--input",
                str(input_path),
                "--keys",
                str(keys_path),
                "--verifier-id",
                "evidence-verifier",
                "--created-at",
                self.created_at,
                "--output",
                str(output_path),
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(
                json.loads(built.stdout)["status"],
                "admission_contract_satisfied",
            )
            verified = run(
                "decision-evidence-verify-v1",
                "--bundle",
                str(output_path),
                "--keys",
                str(keys_path),
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(
                json.loads(verified.stdout)["record_count"],
                2,
            )
            self.assertEqual(
                load_decision_evidence_bundle(
                    output_path,
                    self.authority,
                ).artifact_digest,
                json.loads(verified.stdout)["artifact_digest"],
            )

    def test_existing_model_registry_and_fit_paths_are_unchanged(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        registry_text = (
            project_root / "src" / "pcfm" / "registry.py"
        ).read_text(encoding="utf-8")
        main_text = (
            project_root / "src" / "pcfm" / "__main__.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("DECISION_EVIDENCE", registry_text)
        self.assertIn('"fit"', main_text)
        sources, records = self._fixture()
        parsed = decision_evidence_record_from_dict(records[0].to_dict())
        self.assertEqual(parsed, records[0])
        self.assertNotEqual(type(parsed).__name__, "Observation")
        self.assertEqual(len(sources), 6)


if __name__ == "__main__":
    unittest.main()
