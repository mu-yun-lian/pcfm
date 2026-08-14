from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.tyler_corpus_v1 import (
    CorpusSourceInput,
    TylerCorpusRefusedError,
    create_tyler_corpus,
    load_tyler_corpus,
    save_tyler_corpus,
    tyler_corpus_from_dict,
    verify_tyler_corpus,
)
from pcfm.tyler_source_v1 import (
    extract_tyler_source_page,
    save_tyler_source_artifact,
)


def _article(*, slug: str, published_at: str, prose: str) -> str:
    return f"""
    <article id="post-{slug}">
      <h2 class="entry-title">
        <a href="https://marginalrevolution.com/{slug}">{slug}</a>
      </h2>
      <p class="byline">by <a rel="author"
        href="https://marginalrevolution.com/marginalrevolution/author/tyler-cowen">
        Tyler Cowen</a></p>
      <time datetime="{published_at}">{published_at}</time>
      <div class="entry-content"><p>{prose}</p></div>
    </article>
    """


def _page(*articles: str) -> str:
    return (
        "<!doctype html><html><head><title>Tyler Cowen, Author at "
        "Marginal REVOLUTION</title></head><body><main>"
        + "".join(articles)
        + "</main></body></html>"
    )


def _source(page_number: int, *articles: str) -> CorpusSourceInput:
    raw = _page(*articles)
    source_url = (
        "https://marginalrevolution.com/marginalrevolution/author/"
        f"tyler-cowen/page/{page_number}"
    )
    artifact = extract_tyler_source_page(
        raw,
        source_url=source_url,
        collected_at="2026-08-01T00:00:00Z",
    )
    return CorpusSourceInput(artifact=artifact, raw_snapshot=raw)


class TylerCorpusV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = _source(
            100,
            _article(
                slug="training-post",
                published_at="2024-12-31T12:00:00Z",
                prose="I favor this market test for the stated reason.",
            ),
            _article(
                slug="validation-post",
                published_at="2025-06-01T12:00:00Z",
                prose="I would compare both approaches before proceeding.",
            ),
        )
        self.second = _source(
            101,
            _article(
                slug="diagnostic-post",
                published_at="2026-07-31T12:00:00Z",
                prose="I expect this institutional change to persist.",
            ),
        )
        self.expected_urls = (
            self.first.artifact.source_url,
            self.second.artifact.source_url,
        )

    def _corpus(self):
        return create_tyler_corpus(
            (self.first, self.second),
            expected_source_urls=self.expected_urls,
            created_at="2026-08-01T01:00:00Z",
        )

    def test_combines_all_posts_and_assigns_fixed_temporal_roles(self) -> None:
        corpus = self._corpus()
        self.assertEqual(len(corpus.records), 3)
        self.assertEqual(
            corpus.role_counts,
            {
                "protocol_validation": 1,
                "retrospective_diagnostic": 1,
                "training_discovery": 1,
            },
        )
        self.assertTrue(
            verify_tyler_corpus(
                corpus,
                source_inputs=(self.first, self.second),
            )
        )

    def test_input_order_is_irrelevant_but_plan_coverage_is_exact(self) -> None:
        forward = self._corpus()
        reversed_corpus = create_tyler_corpus(
            (self.second, self.first),
            expected_source_urls=tuple(reversed(self.expected_urls)),
            created_at="2026-08-01T01:00:00Z",
        )
        self.assertEqual(forward, reversed_corpus)
        timezone_equivalent = create_tyler_corpus(
            (self.first, self.second),
            expected_source_urls=self.expected_urls,
            created_at="2026-08-01T09:00:00+08:00",
        )
        self.assertEqual(forward, timezone_equivalent)
        with self.assertRaisesRegex(
            TylerCorpusRefusedError,
            "planned_source_set_mismatch",
        ):
            create_tyler_corpus(
                (self.first,),
                expected_source_urls=self.expected_urls,
                created_at="2026-08-01T01:00:00Z",
            )

    def test_cross_source_replay_and_identical_prose_are_refused(self) -> None:
        replay = CorpusSourceInput(
            artifact=self.first.artifact,
            raw_snapshot=self.first.raw_snapshot,
        )
        with self.assertRaisesRegex(
            TylerCorpusRefusedError,
            "duplicate_source_url",
        ):
            create_tyler_corpus(
                (self.first, replay),
                expected_source_urls=(self.first.artifact.source_url,),
                created_at="2026-08-01T01:00:00Z",
            )
        duplicate = _source(
            102,
            _article(
                slug="renamed-copy",
                published_at="2024-01-01T12:00:00Z",
                prose="I favor this market test for the stated reason.",
            ),
        )
        with self.assertRaisesRegex(
            TylerCorpusRefusedError,
            "duplicate_normalized_prose",
        ):
            create_tyler_corpus(
                (self.first, duplicate),
                expected_source_urls=(
                    self.first.artifact.source_url,
                    duplicate.artifact.source_url,
                ),
                created_at="2026-08-01T01:00:00Z",
            )

    def test_raw_tamper_and_unregistered_future_publication_are_refused(self) -> None:
        tampered = CorpusSourceInput(
            artifact=self.first.artifact,
            raw_snapshot=self.first.raw_snapshot + "<!-- changed -->",
        )
        with self.assertRaisesRegex(
            TylerCorpusRefusedError,
            "source_replay_invalid",
        ):
            create_tyler_corpus(
                (tampered,),
                expected_source_urls=(self.first.artifact.source_url,),
                created_at="2026-08-01T01:00:00Z",
            )
        future = _source(
            1,
            _article(
                slug="future-post",
                published_at="2026-08-01T00:30:00Z",
                prose="This future statement has no registered role.",
            ),
        )
        with self.assertRaisesRegex(
            TylerCorpusRefusedError,
            "unregistered_future_evidence",
        ):
            create_tyler_corpus(
                (future,),
                expected_source_urls=(future.artifact.source_url,),
                created_at="2026-08-01T01:00:00Z",
            )

    def test_temporal_boundaries_have_no_subsecond_gap(self) -> None:
        boundary = _source(
            103,
            _article(
                slug="last-training-instant",
                published_at="2024-12-31T23:59:59.999999Z",
                prose="This remains in the discovery interval.",
            ),
            _article(
                slug="first-validation-instant",
                published_at="2025-01-01T00:00:00Z",
                prose="This begins the validation interval.",
            ),
        )
        corpus = create_tyler_corpus(
            (boundary,),
            expected_source_urls=(boundary.artifact.source_url,),
            created_at="2026-08-01T01:00:00Z",
        )
        roles = {
            record.post.canonical_url: record.evidence_role
            for record in corpus.records
        }
        self.assertEqual(
            roles["https://marginalrevolution.com/last-training-instant"],
            "training_discovery",
        )
        self.assertEqual(
            roles[
                "https://marginalrevolution.com/first-validation-instant"
            ],
            "protocol_validation",
        )

    def test_round_trip_tamper_and_cli_manifest(self) -> None:
        corpus = self._corpus()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.json"
            save_tyler_corpus(corpus, corpus_path)
            self.assertEqual(load_tyler_corpus(corpus_path), corpus)

            tampered = copy.deepcopy(corpus.to_dict())
            tampered["records"][0]["evidence_role"] = "not_a_role"
            with self.assertRaises(ValueError):
                tyler_corpus_from_dict(tampered)

            inputs = []
            for index, source in enumerate((self.first, self.second)):
                artifact_path = root / f"source-{index}.json"
                raw_path = root / f"source-{index}.html"
                save_tyler_source_artifact(source.artifact, artifact_path)
                raw_path.write_text(source.raw_snapshot, encoding="utf-8")
                inputs.append(
                    {
                        "source_url": source.artifact.source_url,
                        "artifact": artifact_path.name,
                        "raw_snapshot": raw_path.name,
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "tyler-corpus-input-manifest-v1"
                        ),
                        "inputs": inputs,
                    }
                ),
                encoding="utf-8",
            )
            output_path = root / "cli-corpus.json"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(
                Path(__file__).resolve().parents[1] / "src"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "tyler-corpus-v1",
                    "--manifest",
                    str(manifest_path),
                    "--created-at",
                    "2026-08-01T01:00:00Z",
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(load_tyler_corpus(output_path), corpus)


if __name__ == "__main__":
    unittest.main()
