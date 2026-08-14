from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.ledger import VerificationAuthority
from pcfm.tyler_annotation_v1 import (
    AdjudicationResolution,
    AnnotationLabel,
    AnnotationRefusedError,
    create_adjudication_packet,
    create_annotation_packets,
    create_annotation_submission,
    finalize_adjudicated_dataset,
    load_annotation_packet,
    load_annotation_submission,
    load_adjudication_packet,
    load_adjudicated_dataset,
    save_annotation_packet,
    save_annotation_submission,
    save_adjudication_packet,
    save_adjudicated_dataset,
    verify_adjudicated_dataset,
    verify_adjudication_packet,
    verify_annotation_packet,
    verify_annotation_submission,
)
from pcfm.tyler_source_v1 import (
    TylerSourceArtifact,
    extract_tyler_source_page,
)


SOURCE_URL = (
    "https://marginalrevolution.com/"
    "marginalrevolution/author/tyler-cowen"
)
COLLECTED_AT = "2026-07-31T12:00:00Z"
PACKET_AT = "2026-07-31T13:00:00Z"
SUBMITTED_AT = "2026-07-31T14:00:00Z"
ADJUDICATED_AT = "2026-07-31T15:00:00Z"
AXIS = "market_mechanisms_vs_government_intervention"


def _article(index: int, published_at: str) -> str:
    return f"""
    <article>
      <h2><a href="https://marginalrevolution.com/post-{index}">
        Post {index}
      </a></h2>
      <p class="byline">by <a rel="author"
        href="/marginalrevolution/author/tyler-cowen">
        Tyler Cowen</a></p>
      <time datetime="{published_at}"></time>
      <p>I prefer policy option {index} under these conditions.</p>
    </article>
    """


def _source_artifact(
    dates: tuple[str, ...] = (
        "2024-01-01T00:00:00Z",
        "2024-02-01T00:00:00Z",
        "2025-01-01T00:00:00Z",
        "2025-02-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
    ),
) -> TylerSourceArtifact:
    html = (
        "<html><body><main>"
        + "".join(
            _article(index, date)
            for index, date in enumerate(dates)
        )
        + """
        <article>
          <h2><a href="https://marginalrevolution.com/links">
            Thursday assorted links
          </a></h2>
          <p class="byline">by <a rel="author"
            href="/marginalrevolution/author/tyler-cowen">
            Tyler Cowen</a></p>
          <time datetime="2026-03-01T00:00:00Z"></time>
          <p>1. <a href="/one">One</a></p>
        </article>
        </main></body></html>
        """
    )
    return extract_tyler_source_page(
        html,
        source_url=SOURCE_URL,
        collected_at=COLLECTED_AT,
    )


class TylerAnnotationV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.authority = VerificationAuthority(
            {
                "annotator-alpha": b"alpha-secret",
                "annotator-beta": b"beta-secret",
                "adjudicator": b"adjudicator-secret",
            }
        )

    def _packets(self, source: TylerSourceArtifact | None = None):
        return create_annotation_packets(
            source or _source_artifact(),
            created_at=PACKET_AT,
        )

    @staticmethod
    def _label(
        item,
        disposition: str,
        *,
        direction: str = "toward_pole_1",
    ) -> AnnotationLabel:
        axes = (
            ((AXIS, direction),)
            if disposition == "clear_in_scope_stance"
            else ()
        )
        return AnnotationLabel(
            candidate_id=item.candidate_id,
            disposition=disposition,
            axis_directions=axes,
            evidence_unit_ids=(item.units[0].unit_id,),
            counterevidence_unit_ids=(),
            annotator_confidence="medium",
            rationale=f"Test rationale for {disposition}.",
        )

    def _labels(self, packet, dispositions):
        by_id = {item.candidate_id: item for item in packet.items}
        ordered_ids = sorted(by_id)
        return tuple(
            self._label(by_id[candidate_id], disposition)
            for candidate_id, disposition in zip(
                ordered_ids,
                dispositions,
                strict=True,
            )
        )

    def _submission(
        self,
        packet,
        annotator_id,
        dispositions,
    ):
        return create_annotation_submission(
            packet,
            labels=self._labels(packet, dispositions),
            annotator_id=annotator_id,
            completed_at=SUBMITTED_AT,
            authority=self.authority,
        )

    def test_packets_are_blind_complete_and_source_recomputable(self) -> None:
        source = _source_artifact()
        packet_a, packet_b = self._packets(source)

        self.assertEqual(packet_a.slot, "A")
        self.assertEqual(packet_b.slot, "B")
        self.assertEqual(packet_a.candidate_set_digest, packet_b.candidate_set_digest)
        self.assertEqual(
            {item.candidate_id for item in packet_a.items},
            {
                post.post_id
                for post in source.posts
                if post.candidate_status == "needs_human_annotation"
            },
        )
        self.assertEqual(len(packet_a.items), 6)
        self.assertNotEqual(
            tuple(item.candidate_id for item in packet_a.items),
            tuple(item.candidate_id for item in packet_b.items),
        )
        for packet in (packet_a, packet_b):
            serialized = json.dumps(packet.to_dict())
            self.assertNotIn('"labels"', serialized)
            self.assertNotIn("prediction", serialized.casefold())
            self.assertTrue(verify_annotation_packet(packet, source))

    def test_packet_refuses_too_few_candidates_and_unregistered_future(self) -> None:
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "insufficient_candidates",
        ):
            self._packets(
                _source_artifact(
                    (
                        "2024-01-01T00:00:00Z",
                        "2024-02-01T00:00:00Z",
                        "2025-01-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                    )
                )
            )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "future_registration_required",
        ):
            self._packets(
                _source_artifact(
                    (
                        "2024-01-01T00:00:00Z",
                        "2024-02-01T00:00:00Z",
                        "2025-01-01T00:00:00Z",
                        "2025-02-01T00:00:00Z",
                        "2026-01-01T00:00:00Z",
                        "2026-08-01T00:00:00Z",
                    )
                )
            )

    def test_packet_round_trip_tamper_old_schema_and_wrong_source(self) -> None:
        source = _source_artifact()
        packet, _ = self._packets(source)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packet.json"
            save_annotation_packet(packet, path)
            loaded = load_annotation_packet(path)
        self.assertEqual(loaded, packet)
        self.assertTrue(verify_annotation_packet(loaded, source))

        tampered = copy.deepcopy(packet.to_dict())
        tampered["items"][0]["title"] = "Tampered"
        path_data = json.dumps(tampered)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            path.write_text(path_data, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact_digest"):
                load_annotation_packet(path)

        old = copy.deepcopy(packet.to_dict())
        old["schema_version"] = "tyler-annotation-packet-v0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_annotation_packet(path)

        other_source = replace(
            source,
            artifact_digest="0" * 64,
        )
        with self.assertRaisesRegex(ValueError, "source_artifact_digest"):
            verify_annotation_packet(packet, other_source)

    def test_submission_is_signed_complete_and_order_invariant(self) -> None:
        packet, _ = self._packets()
        dispositions = (
            "clear_in_scope_stance",
            "clear_in_scope_stance",
            "not_a_stance",
            "not_a_stance",
            "clear_out_of_scope_stance",
            "clear_out_of_scope_stance",
        )
        labels = self._labels(packet, dispositions)
        first = create_annotation_submission(
            packet,
            labels=labels,
            annotator_id="annotator-alpha",
            completed_at=SUBMITTED_AT,
            authority=self.authority,
        )
        second = create_annotation_submission(
            packet,
            labels=tuple(reversed(labels)),
            annotator_id="annotator-alpha",
            completed_at=SUBMITTED_AT,
            authority=self.authority,
        )
        self.assertEqual(first, second)
        self.assertTrue(
            verify_annotation_submission(
                first,
                packet,
                self.authority,
            )
        )

    def test_submission_refuses_bad_labels_evidence_and_coverage(self) -> None:
        packet, _ = self._packets()
        dispositions = ("not_a_stance",) * 6
        labels = list(self._labels(packet, dispositions))

        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "incomplete_coverage",
        ):
            create_annotation_submission(
                packet,
                labels=labels[:-1],
                annotator_id="annotator-alpha",
                completed_at=SUBMITTED_AT,
                authority=self.authority,
            )

        invalid_evidence = replace(
            labels[0],
            evidence_unit_ids=("not-a-real-unit",),
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "invalid_evidence_unit",
        ):
            create_annotation_submission(
                packet,
                labels=(invalid_evidence, *labels[1:]),
                annotator_id="annotator-alpha",
                completed_at=SUBMITTED_AT,
                authority=self.authority,
            )

        invalid_axes = replace(
            labels[0],
            axis_directions=((AXIS, "toward_pole_1"),),
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "axis_not_allowed",
        ):
            create_annotation_submission(
                packet,
                labels=(invalid_axes, *labels[1:]),
                annotator_id="annotator-alpha",
                completed_at=SUBMITTED_AT,
                authority=self.authority,
            )

    def test_pair_refuses_same_annotator_cross_packet_and_bad_signature(self) -> None:
        packet_a, packet_b = self._packets()
        dispositions = ("not_a_stance",) * 6
        alpha_a = self._submission(
            packet_a,
            "annotator-alpha",
            dispositions,
        )
        alpha_b = self._submission(
            packet_b,
            "annotator-alpha",
            dispositions,
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "annotators_must_be_distinct",
        ):
            create_adjudication_packet(
                packet_a,
                packet_b,
                alpha_a,
                alpha_b,
                source=_source_artifact(),
                authority=self.authority,
            )

        beta_b = self._submission(
            packet_b,
            "annotator-beta",
            dispositions,
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "packet_submission_mismatch",
        ):
            create_adjudication_packet(
                packet_a,
                packet_b,
                beta_b,
                alpha_a,
                source=_source_artifact(),
                authority=self.authority,
            )

        bad = replace(beta_b, signature="0" * 64)
        with self.assertRaisesRegex(ValueError, "signature"):
            verify_annotation_submission(
                bad,
                packet_b,
                self.authority,
            )

    def test_agreement_metrics_and_disagreement_packet(self) -> None:
        packet_a, packet_b = self._packets()
        a = (
            "clear_in_scope_stance",
            "clear_in_scope_stance",
            "not_a_stance",
            "not_a_stance",
            "clear_out_of_scope_stance",
            "clear_out_of_scope_stance",
        )
        b = (
            "clear_in_scope_stance",
            "clear_in_scope_stance",
            "not_a_stance",
            "not_a_stance",
            "clear_out_of_scope_stance",
            "not_a_stance",
        )
        submission_a = self._submission(
            packet_a,
            "annotator-alpha",
            a,
        )
        submission_b = self._submission(
            packet_b,
            "annotator-beta",
            b,
        )
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            submission_a,
            submission_b,
            source=_source_artifact(),
            authority=self.authority,
        )
        self.assertAlmostEqual(adjudication.exact_agreement, 5 / 6)
        self.assertAlmostEqual(adjudication.disposition_kappa, 0.75)
        self.assertTrue(adjudication.reliability_passed)
        self.assertEqual(len(adjudication.disagreements), 1)
        self.assertEqual(len(adjudication.agreed_labels), 5)
        self.assertTrue(
            verify_adjudication_packet(
                adjudication,
                packet_a,
                packet_b,
                submission_a,
                submission_b,
                source=_source_artifact(),
                authority=self.authority,
            )
        )

    def test_low_agreement_cannot_be_adjudicated_into_passing(self) -> None:
        packet_a, packet_b = self._packets()
        submission_a = self._submission(
            packet_a,
            "annotator-alpha",
            ("clear_in_scope_stance",) * 6,
        )
        submission_b = self._submission(
            packet_b,
            "annotator-beta",
            ("not_a_stance",) * 6,
        )
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            submission_a,
            submission_b,
            source=_source_artifact(),
            authority=self.authority,
        )
        self.assertFalse(adjudication.reliability_passed)
        resolutions = tuple(
            AdjudicationResolution(
                candidate_id=item.candidate_id,
                final_label=item.label_a,
                rationale="Resolve all disagreements.",
            )
            for item in adjudication.disagreements
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "reliability_floor_failed",
        ):
            finalize_adjudicated_dataset(
                adjudication,
                resolutions=resolutions,
                adjudicator_id="adjudicator",
                adjudicated_at=ADJUDICATED_AT,
                authority=self.authority,
            )

    def test_finalization_requires_exact_disagreement_resolution(self) -> None:
        packet_a, packet_b = self._packets()
        roles = {
            item.candidate_id: item.evidence_role
            for item in packet_a.items
        }
        ordered_ids = sorted(roles)
        a = tuple(
            (
                "clear_in_scope_stance"
                if roles[candidate_id] == "training_discovery"
                else "not_a_stance"
                if roles[candidate_id] == "protocol_validation"
                else "clear_out_of_scope_stance"
            )
            for candidate_id in ordered_ids
        )
        b_list = list(a)
        disagreement_index = next(
            index
            for index, candidate_id in enumerate(ordered_ids)
            if roles[candidate_id] == "retrospective_diagnostic"
        )
        b_list[disagreement_index] = "not_a_stance"
        b = tuple(b_list)
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            self._submission(packet_a, "annotator-alpha", a),
            self._submission(packet_b, "annotator-beta", b),
            source=_source_artifact(),
            authority=self.authority,
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "unresolved_disagreements",
        ):
            finalize_adjudicated_dataset(
                adjudication,
                resolutions=(),
                adjudicator_id="adjudicator",
                adjudicated_at=ADJUDICATED_AT,
                authority=self.authority,
            )

        disagreement = adjudication.disagreements[0]
        resolution = AdjudicationResolution(
            candidate_id=disagreement.candidate_id,
            final_label=disagreement.label_a,
            rationale="The cited unit supports A.",
        )
        dataset = finalize_adjudicated_dataset(
            adjudication,
            resolutions=(resolution,),
            adjudicator_id="adjudicator",
            adjudicated_at=ADJUDICATED_AT,
            authority=self.authority,
        )
        self.assertTrue(
            verify_adjudicated_dataset(
                dataset,
                adjudication,
                self.authority,
            )
        )
        self.assertEqual(len(dataset.records), 6)
        self.assertEqual(dataset.training_eligible_count, 2)

        extra = AdjudicationResolution(
            candidate_id=adjudication.agreed_labels[0].candidate_id,
            final_label=adjudication.agreed_labels[0],
            rationale="Attempt to rewrite agreement.",
        )
        with self.assertRaisesRegex(
            AnnotationRefusedError,
            "invalid_resolution_set",
        ):
            finalize_adjudicated_dataset(
                adjudication,
                resolutions=(resolution, extra),
                adjudicator_id="adjudicator",
                adjudicated_at=ADJUDICATED_AT,
                authority=self.authority,
            )

    def test_real_retrospective_packet_can_never_be_training_data(self) -> None:
        project_root = Path(__file__).parents[1]
        source_path = (
            project_root
            / "artifacts"
            / "tyler_source_v1"
            / "tyler-cowen-rss-2026-07-31.json"
        )
        from pcfm.tyler_source_v1 import load_tyler_source_artifact

        source = load_tyler_source_artifact(source_path)
        packet_a, packet_b = create_annotation_packets(
            source,
            created_at="2026-07-31T14:00:00+08:00",
        )
        dispositions = ("clear_in_scope_stance",) * len(packet_a.items)
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            self._submission(
                packet_a,
                "annotator-alpha",
                dispositions,
            ),
            self._submission(
                packet_b,
                "annotator-beta",
                dispositions,
            ),
            source=source,
            authority=self.authority,
        )
        dataset = finalize_adjudicated_dataset(
            adjudication,
            resolutions=(),
            adjudicator_id="adjudicator",
            adjudicated_at=ADJUDICATED_AT,
            authority=self.authority,
        )
        self.assertEqual(dataset.training_eligible_count, 0)
        self.assertEqual(
            {record.evidence_role for record in dataset.records},
            {"retrospective_diagnostic"},
        )

    def test_submission_and_dataset_round_trip(self) -> None:
        packet_a, packet_b = self._packets()
        dispositions = (
            "clear_in_scope_stance",
            "clear_in_scope_stance",
            "not_a_stance",
            "not_a_stance",
            "clear_out_of_scope_stance",
            "clear_out_of_scope_stance",
        )
        submission_a = self._submission(
            packet_a,
            "annotator-alpha",
            dispositions,
        )
        submission_b = self._submission(
            packet_b,
            "annotator-beta",
            dispositions,
        )
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            submission_a,
            submission_b,
            source=_source_artifact(),
            authority=self.authority,
        )
        dataset = finalize_adjudicated_dataset(
            adjudication,
            resolutions=(),
            adjudicator_id="adjudicator",
            adjudicated_at=ADJUDICATED_AT,
            authority=self.authority,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            submission_path = root / "submission.json"
            adjudication_path = root / "adjudication.json"
            dataset_path = root / "dataset.json"
            save_annotation_submission(submission_a, submission_path)
            save_adjudication_packet(
                adjudication,
                adjudication_path,
            )
            save_adjudicated_dataset(dataset, dataset_path)
            loaded_submission = load_annotation_submission(submission_path)
            loaded_adjudication = load_adjudication_packet(
                adjudication_path
            )
            loaded_dataset = load_adjudicated_dataset(dataset_path)
        self.assertEqual(loaded_submission, submission_a)
        self.assertEqual(loaded_adjudication, adjudication)
        self.assertEqual(loaded_dataset, dataset)
        self.assertTrue(
            verify_adjudicated_dataset(
                loaded_dataset,
                adjudication,
                self.authority,
            )
        )

    def test_resigned_dataset_cannot_rewrite_an_agreed_label(self) -> None:
        packet_a, packet_b = self._packets()
        dispositions = (
            "clear_in_scope_stance",
            "clear_in_scope_stance",
            "not_a_stance",
            "not_a_stance",
            "clear_out_of_scope_stance",
            "clear_out_of_scope_stance",
        )
        adjudication = create_adjudication_packet(
            packet_a,
            packet_b,
            self._submission(
                packet_a,
                "annotator-alpha",
                dispositions,
            ),
            self._submission(
                packet_b,
                "annotator-beta",
                dispositions,
            ),
            source=_source_artifact(),
            authority=self.authority,
        )
        dataset = finalize_adjudicated_dataset(
            adjudication,
            resolutions=(),
            adjudicator_id="adjudicator",
            adjudicated_at=ADJUDICATED_AT,
            authority=self.authority,
        )
        agreed = adjudication.agreed_labels[0]
        record_index = next(
            index
            for index, record in enumerate(dataset.records)
            if record.candidate_id == agreed.candidate_id
        )
        record = dataset.records[record_index]
        changed_label = replace(
            record.final_label,
            disposition="not_a_stance",
            axis_directions=(),
            rationale="Re-signed changed label.",
        )
        changed_record = replace(
            record,
            final_label=changed_label,
            training_eligible=False,
        )
        changed_records = list(dataset.records)
        changed_records[record_index] = changed_record
        changed = replace(
            dataset,
            records=tuple(changed_records),
            training_eligible_count=sum(
                item.training_eligible for item in changed_records
            ),
        )
        base = changed._base_dict()
        digest = hashlib.sha256(
            json.dumps(
                base,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        changed = replace(changed, artifact_digest=digest)
        changed = replace(
            changed,
            signature=self.authority.sign_payload(
                changed.signed_payload(),
                "adjudicator",
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "agreed label changed",
        ):
            verify_adjudicated_dataset(
                changed,
                adjudication,
                self.authority,
            )

    def test_cli_generates_two_blind_packets(self) -> None:
        project_root = Path(__file__).parents[1]
        source_path = (
            project_root
            / "artifacts"
            / "tyler_source_v1"
            / "tyler-cowen-rss-2026-07-31.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_a = root / "a.json"
            output_b = root / "b.json"
            template_a = root / "template-a.json"
            template_b = root / "template-b.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(project_root / "src")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "tyler-annotation-pack-v1",
                    "--source",
                    str(source_path),
                    "--created-at",
                    "2026-07-31T14:00:00+08:00",
                    "--output-a",
                    str(output_a),
                    "--output-b",
                    str(output_b),
                    "--template-a",
                    str(template_a),
                    "--template-b",
                    str(template_b),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            packet_a = load_annotation_packet(output_a)
            packet_b = load_annotation_packet(output_b)
            self.assertEqual(len(packet_a.items), 9)
            self.assertEqual(packet_a.slot, "A")
            self.assertEqual(packet_b.slot, "B")
            template = json.loads(
                template_a.read_text(encoding="utf-8")
            )
            self.assertEqual(template["slot"], "A")
            self.assertEqual(len(template["labels"]), 9)
            self.assertTrue(
                all(
                    label["disposition"] == ""
                    and label["evidence_unit_ids"] == []
                    for label in template["labels"]
                )
            )


if __name__ == "__main__":
    unittest.main()
