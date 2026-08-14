# Tyler Cowen Official-Source v1 — Implementation Report

## Result

The source layer is implemented as a deterministic, standard-library-only
module. It accepts local official author-archive HTML or official RSS
snapshots, enforces source and author identity, separates HTML block quotes,
removes the deterministic FeedBlitz footer, gates structural candidates, and
produces replay-verifiable JSON artifacts.

This is a source/annotation-routing module. It is not a belief model and does
not create stance labels.

## Implemented files

- `src/pcfm/tyler_source_v1.py`
- `tests/test_tyler_source_v1.py`
- `TYLER_COWEN_SOURCE_V1.md`
- `MODULE_GATE_TYLER_SOURCE_V1.json`
- `artifacts/tyler_source_v1/feedblitz-2026-07-31.xml`
- `artifacts/tyler_source_v1/tyler-cowen-rss-2026-07-31.json`

## Real official-RSS audit

Snapshot time: `2026-07-31T13:26:59+08:00`

- official RSS items: 15
- explicit `Tyler Cowen` creators: 13
- explicit `Alex Tabarrok` creators: 2, excluded
- Tyler posts in artifact: 13
- `needs_human_annotation`: 9
- `ambiguous_not_trainable`: 1
- `no_stance_candidate`: 3
- block-quote units represented by hash/count: 29
- authored-prose units after filtering: 38
- leaked FeedBlitz footer units: 0
- raw-snapshot replay verification: passed
- extraction digest:
  `796ae1cfed6eba51d667c6b6b1ce9c5bb024a2856370087127a1708849e80497`
- artifact digest:
  `8c9e84df73c8a863bb394bacb104f0915020df810240e657ea97a62c5ec148b4`

The first audit exposed a real fixture gap: FeedBlitz appends a platform footer
to every item. A regression test was added before the saved artifact was
accepted.

## Verification

- focused Tyler source tests: 12/12 passed
- full PCFM regression: 210/210 passed
- full regression duration: 191.203 seconds
- regression log: `artifacts/full_regression_tyler_source_v1.err.log`

## Functional guarantees exercised

- non-official sources fail closed;
- Cloudflare challenge HTML fails closed;
- author-archive pages containing Alex or missing an author fail closed;
- mixed RSS is allowed only with explicit creators and exact Tyler filtering;
- missing publication time fails closed;
- FeedBlitz tracking URLs cannot replace the official original link;
- block-quote text is absent from authored-prose fields;
- link-only and assorted-link posts are not trainable candidates;
- question-only authored prose is ambiguous by default;
- duplicate URL and normalized duplicate prose are refused;
- irrelevant HTML attributes and article ordering preserve canonical post IDs;
- schema downgrade, internal tampering, raw mismatch, and replay mismatch are
  refused;
- HTML and RSS local-file CLI paths are covered.

## Remaining functional limits

1. Only publisher-marked `<blockquote>` text is structurally separable. Inline
   quotations and unmarked paraphrases require human attribution review.
2. “Needs human annotation” means only “contains non-link-only prose”; it does
   not mean the post contains a stance.
3. Semantic near duplicates are not detected. Current deduplication covers
   canonical URLs and normalized identical authored prose.
4. The saved RSS snapshot covers only the 15 items exposed by the feed. The
   2,297-page archive has not been bulk acquired.
5. Irony, reporting, strategic rhetoric, changing views, and public/private
   divergence are not identifiable from this module.
6. No dual annotation, adjudication, inter-rater reliability study, model
   fitting, or prospective prediction has yet occurred.

## Next gate

Freeze the annotation codebook and build a dual-annotator packet from the nine
eligible real candidates. The packet must retain unit hashes, require exact
evidence-unit citations, and permit `not_a_stance`, `uncertain_attribution`,
`multi_axis`, and `out_of_scope`. No model training should begin before
agreement/adjudication and temporal split checks pass.
