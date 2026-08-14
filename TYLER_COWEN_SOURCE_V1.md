# Tyler Cowen Official-Source Contract v1

## Narrow operational claim

This module deterministically extracts auditable post metadata and
**stance candidates** from locally saved Marginal Revolution author-archive
HTML pages or official RSS snapshots for Tyler Cowen. It separates text inside
block quotes from Tyler-authored prose outside block quotes.

It does **not** claim that:

- every extracted candidate contains a position;
- a link, quotation, title, category, or question is Tyler Cowen's belief;
- a candidate's topic, stance, confidence, causal model, or decision rule has
  been inferred;
- the archive is complete;
- a model trained on these records reproduces Tyler Cowen's cognition.

No LLM is used in source validation, author validation, quote separation,
deduplication, candidate gating, hashing, serialization, or verification.

## Allowed input and acquisition boundary

Input is either:

- a local HTML snapshot saved from
  `https://marginalrevolution.com/marginalrevolution/author/tyler-cowen[/page/N]`;
- a local RSS snapshot saved from
  `https://feeds.feedblitz.com/marginalrevolution` or
  `https://marginalrevolution.com/feed`.

The mixed-author RSS feed requires an explicit non-empty creator on every
item. Only exact `Tyler Cowen` items are retained. Other explicitly named
authors remain in the raw snapshot but are excluded from the Tyler artifact.
The official `feedburner:origLink`, not the FeedBlitz tracking URL, supplies
canonical post identity.

Version 1 deliberately does not bypass Cloudflare, execute CAPTCHA workflows,
or silently fall back to search-engine copies. A Cloudflare challenge,
non-official host, wrong author path, missing article set, missing per-post
author, missing RSS creator, or unexpected author on an author-archive page
causes a refusal.

The raw snapshot SHA-256 is retained for source identity. A second canonical
extraction digest identifies the semantically extracted records; irrelevant
HTML attributes may change the raw snapshot hash but must not change the
canonical post IDs or extraction digest.

## Evidence-role separation

1. **Source discovery** verifies the official author archive and records its
   URL; it cannot label positions.
2. **Raw snapshot** is immutable source evidence and has a SHA-256 digest.
3. **Structural extractor** identifies posts, authors, dates, URLs, ordinary
   paragraphs, links, and block quotes; it cannot label beliefs.
4. **Annotator A and annotator B** independently label eligible candidates
   using a frozen codebook. They cannot see model predictions.
5. **Adjudicator** resolves disagreements and records both original labels.
6. **Model fitting** consumes only adjudicated training records.
7. **Evaluation** reads only frozen validation or holdout outcomes and
   recomputes metrics independently.

No role may overwrite an earlier artifact. Every transformation gets a new
artifact and a digest of its inputs.

## Structural candidate statuses

- `needs_human_annotation`: at least one non-link-only, non-metadata paragraph
  exists outside block quotes and the prose is not structurally question-only.
- `ambiguous_not_trainable`: the only authored prose outside quotes consists
  of questions. This can be reviewed later but is excluded by default.
- `no_stance_candidate`: no authored prose exists outside quotes, the post is
  link-only, or the title is an “assorted links” collection.

These are routing statuses, not semantic stance labels.

Quoted text is represented by hashes and character counts, never merged into
the authored-prose field. Link-only anchor text does not count as authored
prose. A sentence outside a block quote remains merely a candidate until human
annotation.

FeedBlitz's deterministic “The post … appeared first on Marginal REVOLUTION”
footer is excluded before candidate routing. Inline quotations or paraphrases
that the publisher did not mark with `<blockquote>` cannot be reliably
separated by structure alone; their containing paragraph remains an annotation
candidate and cannot become a training label without human attribution review.

## Pre-registered decision axes

The first bounded model may represent only these five axes:

1. AI acceleration vs. risk regulation
2. market mechanisms vs. government intervention
3. technological progress vs. employment displacement
4. state capacity vs. individual liberty
5. short-term social cost vs. long-term growth

The extractor does not assign an axis. Annotators may select one axis,
`multi_axis`, or `out_of_scope`, and must cite the exact authored-prose unit
supporting the decision.

## Time partitions

- discovery/training candidate pool: publication time through 2024-12-31;
- annotation-protocol validation: 2025-01-01 through 2025-12-31;
- retrospective diagnostic only: 2026-01-01 through 2026-07-31;
- genuine prospective holdout: records published after a separately signed
  study registration created after this contract.

The 2026 retrospective interval is already observable and can never be
reported as prospective confirmation.

## Deduplication and leakage rules

- duplicate canonical post URLs are refused;
- identical normalized authored prose is refused when non-empty;
- semantic near-duplicate detection is not claimed in v1 and must be a later,
  separately evaluated module;
- quoted text, titles, categories, and linked source text cannot become target
  labels;
- the temporal holdout cannot influence codebook revision, feature selection,
  prompt construction, thresholds, model selection, or stopping decisions.

## Acceptance criteria

The module passes only when tests show:

- official-host and exact-author-path enforcement;
- official-feed allow-list and exact RSS-creator filtering;
- mixed-author and missing-author refusal;
- missing-date refusal;
- quote text cannot enter authored-prose units;
- link-only and assorted-link posts are not training candidates;
- question-only prose is gated as ambiguous;
- duplicate URLs and normalized duplicate prose are refused;
- Cloudflare challenge HTML is refused;
- canonical extraction is reproducible under irrelevant attributes and article
  order;
- artifact round trips preserve all fields;
- artifact plus raw snapshot recomputes to an identical extraction;
- tampering, schema downgrade, and digest mismatch are refused;
- the local-file CLI produces a verifiable artifact;
- all pre-existing PCFM tests continue to pass.

## Known non-solutions

This module does not solve irony, strategic rhetoric, changed beliefs,
unpublished preferences, social desirability, topic-dependent identity,
context omitted by a link, or whether a public statement reflects a private
decision rule. Those require annotation studies, controlled choice tasks,
temporal state models, contradiction probes, and prospective evaluation. They
must not be hidden inside an LLM prompt.
