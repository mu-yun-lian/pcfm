from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pcfm.tyler_source_v1 import (
    TylerSourceRefusedError,
    extract_tyler_source_page,
    extract_tyler_source_rss,
    load_tyler_source_artifact,
    save_tyler_source_artifact,
    tyler_source_artifact_from_dict,
    verify_tyler_source_artifact,
)


SOURCE_URL = (
    "https://marginalrevolution.com/"
    "marginalrevolution/author/tyler-cowen/page/2"
)
COLLECTED_AT = "2026-07-31T12:00:00+08:00"
RSS_URL = "https://feeds.feedblitz.com/marginalrevolution"


def _article(
    *,
    title: str,
    slug: str,
    body: str,
    author: str = "Tyler Cowen",
    author_slug: str = "tyler-cowen",
    extra_attributes: str = "",
) -> str:
    return f"""
    <article id="post-{slug}" {extra_attributes}>
      <h2 class="entry-title">
        <a href="https://marginalrevolution.com/{slug}">{title}</a>
      </h2>
      <p class="byline">
        by <a rel="author"
          href="https://marginalrevolution.com/marginalrevolution/author/{author_slug}">
          {author}
        </a>
      </p>
      <time datetime="2025-06-01T10:00:00-04:00">June 1, 2025</time>
      <a rel="category tag"
        href="https://marginalrevolution.com/marginalrevolution/category/economics">
        Economics
      </a>
      <div class="entry-content">{body}</div>
    </article>
    """


def _page(*articles: str, html_attrs: str = "") -> str:
    return (
        f"<!doctype html><html {html_attrs}><head>"
        "<title>Tyler Cowen, Author at Marginal REVOLUTION</title>"
        "</head><body><main>"
        + "".join(articles)
        + "</main></body></html>"
    )


class TylerSourceV1Tests(unittest.TestCase):
    @staticmethod
    def _rss_item(
        *,
        title: str,
        slug: str,
        creator: str | None,
        body: str,
    ) -> str:
        creator_xml = (
            ""
            if creator is None
            else f"<dc:creator>{creator}</dc:creator>"
        )
        return f"""
        <item>
          <feedburner:origLink>
            https://marginalrevolution.com/marginalrevolution/2025/06/{slug}.html
          </feedburner:origLink>
          <title>{title}</title>
          <link>https://feeds.feedblitz.com/tracking/{slug}</link>
          {creator_xml}
          <pubDate>Sun, 01 Jun 2025 14:00:00 +0000</pubDate>
          <category>Economics</category>
          <content:encoded><![CDATA[{body}]]></content:encoded>
        </item>
        """

    @classmethod
    def _rss(cls, *items: str) -> str:
        return f"""
        <rss version="2.0"
          xmlns:dc="http://purl.org/dc/elements/1.1/"
          xmlns:content="http://purl.org/rss/1.0/modules/content/"
          xmlns:feedburner="http://rssnamespace.org/feedburner/ext/1.0">
          <channel><title>Marginal Revolution</title>
          {''.join(items)}
          </channel>
        </rss>
        """

    def test_extracts_official_tyler_post_and_separates_quote(self) -> None:
        html = _page(
            _article(
                title="A policy proposal",
                slug="policy-proposal",
                body=(
                    "<blockquote><p>Someone else's market claim.</p></blockquote>"
                    "<p>I favor testing this proposal in several cities.</p>"
                ),
            )
        )
        artifact = extract_tyler_source_page(
            html,
            source_url=SOURCE_URL,
            collected_at=COLLECTED_AT,
        )

        self.assertTrue(verify_tyler_source_artifact(artifact))
        self.assertTrue(
            verify_tyler_source_artifact(artifact, raw_snapshot=html)
        )
        self.assertEqual(len(artifact.posts), 1)
        post = artifact.posts[0]
        self.assertEqual(post.author, "Tyler Cowen")
        self.assertEqual(post.candidate_status, "needs_human_annotation")
        self.assertEqual(
            tuple(unit.text for unit in post.authored_prose),
            ("I favor testing this proposal in several cities.",),
        )
        self.assertEqual(len(post.quoted_units), 1)
        self.assertNotIn(
            "Someone else's market claim.",
            json.dumps(artifact.to_dict()),
        )

    def test_refuses_unofficial_source_url(self) -> None:
        html = _page(
            _article(
                title="Post",
                slug="post",
                body="<p>My own statement.</p>",
            )
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "unofficial_source",
        ):
            extract_tyler_source_page(
                html,
                source_url="https://example.com/author/tyler-cowen",
                collected_at=COLLECTED_AT,
            )

    def test_rss_filters_explicit_author_and_uses_official_origin(self) -> None:
        rss = self._rss(
            self._rss_item(
                title="Tyler item",
                slug="tyler-item",
                creator="Tyler Cowen",
                body=(
                    "<blockquote><p>A paper's conclusion.</p></blockquote>"
                    "<p>I am skeptical of this conclusion.</p>"
                    "<p>The post Tyler item appeared first on "
                    "<a href='https://marginalrevolution.com'>"
                    "Marginal REVOLUTION</a>.</p>"
                ),
            ),
            self._rss_item(
                title="Alex item",
                slug="alex-item",
                creator="Alex Tabarrok",
                body="<p>Alex's statement.</p>",
            ),
        )
        artifact = extract_tyler_source_rss(
            rss,
            source_url=RSS_URL,
            collected_at=COLLECTED_AT,
        )
        self.assertEqual(len(artifact.posts), 1)
        post = artifact.posts[0]
        self.assertEqual(post.title, "Tyler item")
        self.assertEqual(
            post.canonical_url,
            "https://marginalrevolution.com/"
            "marginalrevolution/2025/06/tyler-item.html",
        )
        self.assertEqual(
            tuple(unit.text for unit in post.authored_prose),
            ("I am skeptical of this conclusion.",),
        )
        self.assertEqual(len(post.quoted_units), 1)

    def test_rss_refuses_missing_creator_and_unofficial_feed(self) -> None:
        rss = self._rss(
            self._rss_item(
                title="Unknown author",
                slug="unknown",
                creator=None,
                body="<p>A statement.</p>",
            )
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "missing_rss_creator",
        ):
            extract_tyler_source_rss(
                rss,
                source_url=RSS_URL,
                collected_at=COLLECTED_AT,
            )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "unofficial_source",
        ):
            extract_tyler_source_rss(
                rss,
                source_url="https://example.com/feed",
                collected_at=COLLECTED_AT,
            )

    def test_refuses_mixed_or_missing_author(self) -> None:
        mixed = _page(
            _article(
                title="Tyler",
                slug="tyler",
                body="<p>Tyler text.</p>",
            ),
            _article(
                title="Alex",
                slug="alex",
                body="<p>Alex text.</p>",
                author="Alex Tabarrok",
                author_slug="alex-tabarrok",
            ),
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "unexpected_author",
        ):
            extract_tyler_source_page(
                mixed,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

        missing = _page(
            """
            <article>
              <h2><a href="https://marginalrevolution.com/missing">
                Missing author
              </a></h2>
              <time datetime="2025-01-01T00:00:00Z"></time>
              <p>A claim without an attributable author.</p>
            </article>
            """
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "missing_author",
        ):
            extract_tyler_source_page(
                missing,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

        missing_publication = _page(
            """
            <article>
              <h2><a href="https://marginalrevolution.com/no-date">
                Missing date
              </a></h2>
              <p class="byline">by <a rel="author"
                href="/marginalrevolution/author/tyler-cowen">
                Tyler Cowen</a></p>
              <p>A dated claim without a date.</p>
            </article>
            """
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "missing_publication",
        ):
            extract_tyler_source_page(
                missing_publication,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

    def test_link_only_assorted_links_and_questions_are_not_trainable(self) -> None:
        artifact = extract_tyler_source_page(
            _page(
                _article(
                    title="Thursday assorted links",
                    slug="assorted",
                    body="<p><a href='/one'>One</a> <a href='/two'>Two</a></p>",
                ),
                _article(
                    title="Questions about growth",
                    slug="questions",
                    body="<p>Would the policy increase long-run growth?</p>",
                ),
            ),
            source_url=SOURCE_URL,
            collected_at=COLLECTED_AT,
        )
        statuses = {
            post.title: post.candidate_status
            for post in artifact.posts
        }
        self.assertEqual(
            statuses["Thursday assorted links"],
            "no_stance_candidate",
        )
        self.assertEqual(
            statuses["Questions about growth"],
            "ambiguous_not_trainable",
        )

    def test_refuses_duplicate_url_and_normalized_duplicate_prose(self) -> None:
        duplicate_url = _page(
            _article(
                title="First",
                slug="same",
                body="<p>First statement.</p>",
            ),
            _article(
                title="Second",
                slug="same",
                body="<p>Second statement.</p>",
            ),
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "duplicate_post_url",
        ):
            extract_tyler_source_page(
                duplicate_url,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

        duplicate_prose = _page(
            _article(
                title="First",
                slug="first",
                body="<p>Markets work.</p>",
            ),
            _article(
                title="Second",
                slug="second",
                body="<p>MARKETS   WORK!</p>",
            ),
        )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "duplicate_normalized_prose",
        ):
            extract_tyler_source_page(
                duplicate_prose,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

    def test_refuses_cloudflare_challenge_and_empty_page(self) -> None:
        challenge = """
        <html><title>Just a moment...</title>
        <body><div id="cf-chl-widget">Enable JavaScript and cookies to continue</div>
        </body></html>
        """
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "challenge_page",
        ):
            extract_tyler_source_page(
                challenge,
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )
        with self.assertRaisesRegex(
            TylerSourceRefusedError,
            "no_articles",
        ):
            extract_tyler_source_page(
                "<html><body>No posts</body></html>",
                source_url=SOURCE_URL,
                collected_at=COLLECTED_AT,
            )

    def test_canonical_records_ignore_attributes_and_article_order(self) -> None:
        first = _article(
            title="First",
            slug="first",
            body="<p>I prefer the first policy.</p>",
        )
        second = _article(
            title="Second",
            slug="second",
            body="<p>I reject the second policy.</p>",
        )
        base = extract_tyler_source_page(
            _page(first, second),
            source_url=SOURCE_URL,
            collected_at=COLLECTED_AT,
        )
        changed = extract_tyler_source_page(
            _page(
                second.replace("<article ", "<article data-noise='x' "),
                first.replace("<article ", "<article aria-label='noise' "),
                html_attrs="data-render-id='999'",
            ),
            source_url=SOURCE_URL,
            collected_at=COLLECTED_AT,
        )
        self.assertNotEqual(
            base.raw_snapshot_sha256,
            changed.raw_snapshot_sha256,
        )
        self.assertEqual(base.extraction_digest, changed.extraction_digest)
        self.assertEqual(
            tuple(post.post_id for post in base.posts),
            tuple(post.post_id for post in changed.posts),
        )

    def test_round_trip_tamper_and_old_schema(self) -> None:
        artifact = extract_tyler_source_page(
            _page(
                _article(
                    title="Post",
                    slug="round-trip",
                    body="<p>I support this experiment.</p>",
                )
            ),
            source_url=SOURCE_URL,
            collected_at=COLLECTED_AT,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            save_tyler_source_artifact(artifact, path)
            loaded = load_tyler_source_artifact(path)
        self.assertEqual(loaded, artifact)

        tampered = copy.deepcopy(artifact.to_dict())
        tampered["posts"][0]["title"] = "Tampered"
        with self.assertRaisesRegex(ValueError, "artifact_digest"):
            tyler_source_artifact_from_dict(tampered)

        old = copy.deepcopy(artifact.to_dict())
        old["schema_version"] = "tyler-source-v0"
        with self.assertRaisesRegex(ValueError, "schema_version"):
            tyler_source_artifact_from_dict(old)
        with self.assertRaisesRegex(
            ValueError,
            "raw snapshot digest",
        ):
            verify_tyler_source_artifact(
                artifact,
                raw_snapshot="<html>different</html>",
            )

    def test_cli_extracts_local_snapshot(self) -> None:
        html = _page(
            _article(
                title="CLI post",
                slug="cli",
                body="<p>I would run the trial.</p>",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "page.html"
            output_path = root / "artifact.json"
            input_path.write_text(html, encoding="utf-8")
            environment = dict(os.environ)
            source_root = Path(__file__).parents[1] / "src"
            environment["PYTHONPATH"] = str(source_root)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "tyler-source-v1",
                    "--html",
                    str(input_path),
                    "--source-url",
                    SOURCE_URL,
                    "--collected-at",
                    COLLECTED_AT,
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            artifact = load_tyler_source_artifact(output_path)
            self.assertTrue(verify_tyler_source_artifact(artifact))
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["post_count"], 1)
            self.assertEqual(
                summary["candidate_counts"]["needs_human_annotation"],
                1,
            )

    def test_cli_extracts_local_rss_snapshot(self) -> None:
        rss = self._rss(
            self._rss_item(
                title="RSS CLI",
                slug="rss-cli",
                creator="Tyler Cowen",
                body="<p>I would test the RSS route.</p>",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "feed.xml"
            output_path = root / "artifact.json"
            input_path.write_text(rss, encoding="utf-8")
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(
                Path(__file__).parents[1] / "src"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pcfm",
                    "tyler-source-v1",
                    "--rss",
                    str(input_path),
                    "--source-url",
                    RSS_URL,
                    "--collected-at",
                    COLLECTED_AT,
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(
                verify_tyler_source_artifact(
                    load_tyler_source_artifact(output_path)
                )
            )


if __name__ == "__main__":
    unittest.main()
