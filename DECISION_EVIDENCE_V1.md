# Decision-Context-Rationale Evidence Contract v1

## Purpose

This stage defines the smallest admissible real-person evidence unit for a
future PCFM model. It does not fit a cognitive model. It binds one observed
decision to the information available before that decision and to either a
contemporaneous first-person explanation or a verifiable consequence.

The operational output is a signed, recomputable evidence bundle. A valid
bundle supports only this claim:

> The supplied records satisfy the declared decision/context/explanation
> linkage, timing, provenance, role-separation, and integrity contract.

It does not establish that an explanation is sincere, complete, causal,
stable, or sufficient to predict another decision. It does not identify a
belief, value, goal, preference, personality trait, or mental state.

## Observable record

Every record binds all of the following:

- one `event_id`, `person_id`, domain, task, and timezone-aware decision time;
- at least two distinct option identifiers and exactly one chosen option;
- the full decision prompt and option text available before the decision;
- one or more immutable source snapshots containing the supplied evidence;
- a source-backed observed choice;
- either a first-person rationale attributable to the same person or a
  verifiable consequence linked to the event;
- exactly one declared evidence role;
- the time at which that role was assigned and the SHA-256 digest of its
  external assignment or registry reference;
- source timestamps, locators, content digests, and explicit evidence links.

The v1 contract accepts these evidence roles only:

1. `candidate_discovery`
2. `parameter_fitting`
3. `applicability_calibration`
4. `candidate_selection`
5. `sealed_confirmation`
6. `post_deployment_monitoring`
7. `external_utility_evaluation`

The same event, decision design, or near-duplicate decision content may not be
assigned to multiple roles.

## Source contract

Each source snapshot contains its text inside the bundle so its SHA-256 digest
can be recomputed without network access. Accepted provenance classes are:

- `official_primary`
- `human_primary`
- `archival_primary_snapshot`
- `verified_external_consequence`

`model_generated`, summaries without the underlying snapshot, and unknown
provenance are refused. Signatures prove only that supplied bytes did not
change. They do not prove that an upstream collector disclosed every source,
that a remote publication timestamp is truthful, or that a statement is
sincere.

## Timing and attribution gates

- Every context source must have been published no later than the decision.
- A `sealed_confirmation` role must be assigned before the decision occurs.
  Its assignment reference is content-bound, but its external timestamp still
  requires an independent registry to be credible.
- Every snapshot must be captured no earlier than publication.
- A first-person rationale must name the target person as author and be
  published no earlier than 24 hours before, and no later than 168 hours after,
  the decision. The 168-hour ceiling is immutable; callers may choose a
  stricter ceiling.
- A consequence must be published after the decision and explicitly linked to
  the same event.
- The observed-choice source must be published no earlier than the decision.
- All timestamps require explicit UTC offsets.

These gates establish temporal linkage only. They do not establish causal
direction or rule out strategic, post-hoc, performative, or socially
coordinated explanations.

## Refusal states

Bundle creation and verification refuse at least:

- incomplete decision, context, choice, or explanation evidence;
- context that was not available before the choice;
- explanation outside the allowed temporal window;
- rationale attributed to another or unknown person;
- model-generated or unsupported provenance;
- invalid or non-recomputable content, record, bundle, or signature digests;
- duplicate event identifiers, source identifiers, decision designs, or
  near-duplicate content across roles;
- evidence links to absent sources or sources that do not contain the cited
  text span;
- attempts to weaken the immutable 168-hour rationale ceiling;
- old or unknown schema versions.

The public refusal result is `decision_evidence_refused` with stable machine
reason codes. No partially valid bundle is silently emitted.

## Alternatives that remain live

Even if a later model benefits from these records, at least these competing
explanations remain:

- the explanation is a post-hoc public justification rather than the cause of
  the decision;
- shared party, institution, incentive, or recent-state information explains
  the gain;
- topic or phrase repetition, rather than person-specific reasoning, explains
  the gain;
- the selected source corpus omits contradictory or unpublished evidence;
- the useful signal is domain-specific behavior and does not transfer to a
  general person model.

## Downstream boundary

The v1 bundle is an admission artifact, not training authorization. The
existing PCFM fit, update, prediction, mechanism, dynamic-state, and semantic
module paths must not consume it automatically. A later frozen experiment must
declare which fields each evidence role may expose, use a single predictive
kernel, and beat shared-dynamic, wrong-person, history-only, and content-only
controls on a sealed later role before any person-model claim changes.
