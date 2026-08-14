# Tyler Cowen Historical Corpus Contract v1

## Operational claim

Given an explicitly enumerated set of local `tyler-source-v1` artifacts and
their raw official snapshots, this module verifies every source by replay,
combines the posts in an order-invariant corpus, rejects cross-snapshot replay
or exact normalized-prose duplication, and assigns each post exactly one
predeclared temporal evidence role.

Passing supports only that the supplied local snapshots were completely
included, internally consistent, and separated by time role. It does not prove
that the Marginal Revolution archive was completely collected.

## Inputs and outputs

Input:

- an exact list of expected official source URLs;
- one `tyler-source-v1` artifact for every expected URL;
- the raw HTML or RSS snapshot used to create every artifact;
- a corpus creation timestamp.

Output:

- a versioned corpus artifact containing every verified post;
- source and raw-snapshot digests;
- per-post source lineage and temporal role;
- deterministic role and candidate-status counts;
- one corpus digest covering all deployed fields.

## Evidence roles

- publication through 2024-12-31: `training_discovery`;
- 2025-01-01 through 2025-12-31: `protocol_validation`;
- 2026-01-01 through 2026-07-31: `retrospective_diagnostic`;
- later publication: refused until a separately registered prospective study
  defines its role.

These roles cannot be configured or weakened. Corpus assembly does not make a
record training eligible; later dual annotation and adjudication are still
required.

## Null and alternatives

The null is that supplied snapshots cannot be replayed into the claimed
records. Alternative explanations for an apparently large corpus include
overlapping archive pages, duplicated or renamed posts, missing planned pages,
page drift during collection, and a collector who omitted unplanned pages.

The module detects exact planned-input omission, extra inputs, URL replay,
post-ID replay, and identical normalized authored prose. It cannot detect an
unregistered hidden page, semantic paraphrase, archive mutation between
requests, or missing posts that were absent from every supplied snapshot.

## Refusal states

The module refuses unsupported schemas, unofficial or unreplayable sources,
raw-snapshot mismatch, missing or extra expected URLs, duplicate source URLs,
duplicate post URLs or IDs, duplicate normalized prose, publication after the
retrospective cutoff, publication after collection, invalid time roles,
noncanonical ordering, and digest tampering.

## Downstream boundary

The only intended downstream consumer is the annotation-packet builder. Model
fitting must consume only records that later pass dual annotation,
adjudication, temporal eligibility, and task-specific target construction.
Public-writing stance labels must never be silently converted into choice
outcomes for the PCFM prediction kernel.

