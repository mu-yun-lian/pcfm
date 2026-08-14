# PCFM Reality Bridge v1

## Purpose

Evaluate whether the existing PCFM base, dynamic-state, and mechanism paths can
operate on a small, deterministically selected cohort of real people in one
structured domain. This is an experiment harness, not a new cognitive module.
No production estimator, gate, prediction kernel, or module registry entry may
be changed by this experiment.

The supported claim is deliberately narrow: the current PCFM stack may exhibit
person-specific predictive utility for repeated United States Senate Yea/Nay
choices under a fixed feature mapping. It does not identify beliefs, values,
goals, reasons, private cognition, or a general person model.

## Evidence status

All Voteview outcomes already existed before this contract was written and some
Congress 119 aggregate outcomes were inspected during the previous v1 audit.
Therefore every result is retrospective diagnostic evidence.

The existing dynamic and mechanism APIs require a signed plan whose declared
registration time precedes its evidence. For historical replay, the harness may
construct an explicitly counterfactual plan time immediately after the frozen
applicability records. Such plans MUST use verifier ID
`voteview-counterfactual-replay-v1`, MUST be reported as
`counterfactual_historical_replay`, and MUST NOT be called preregistered,
prospective, sealed, or confirmatory. This tests computation and gate
compatibility only. It also demonstrates why an external trusted registry is
required for real temporal claims.

## Frozen population and cohort

Use the official Voteview Senate files for Congresses 116, 117, and 119 from
the existing `voteview_real_audit/raw` snapshot and bind their recorded
digests.

A person is cohort-eligible only when all of the following are true:

- present in the Senate member files for Congresses 116, 117, 118, and 119;
- current party code is 100 or 200;
- at least 500 valid Yea/Nay votes in Congress 116;
- at least 50 Yea and 50 Nay votes in Congress 116;
- at least 500 valid Yea/Nay votes in Congress 117;
- enough Congress 119 valid votes to allocate, in order, 100 applicability,
  80 dynamic, 100 mechanism-discovery, 100 mechanism-selection, and 100
  mechanism-confirmation records;
- the last allocated Congress 119 record is no more than 180 days after the
  last applicability record.

Within each of party codes 100 and 200, select the three eligible people with
the lowest SHA-256 digest of
`reality-bridge-v1:<raw-manifest-digest>:<person-id>`. No Congress 117 or 119
vote direction may affect cohort selection or ordering.

The population fitting ledger contains the selected cohort plus the ten
eligible people from each party with the most Congress 116 valid votes, with
numeric person ID as tie-breaker. Each selected target is fitted from the same
population ledger. Its wrong-person control is the other selected same-party
person with the closest Congress 116 Yea rate, then lowest numeric person ID.

## Frozen evidence roles

- Congress 116: population and person fitting.
- Congress 117: independent base-model validation.
- Congress 119 first 100 valid target votes: applicability calibration.
- Next 80 valid target votes: dynamic-state counterfactual replay.
- Next 100: mechanism discovery.
- Next 100: mechanism selection.
- Next 100: mechanism confirmation.

Every person/roll-call trial has exactly one role within a target audit. The
Congress 118 outcomes are not used by v1.

## Frozen scenario mapping

Only cast codes 1–3 (Yea) and 4–6 (Nay) are outcomes. Outcome aggregates,
Voteview `prob`, NOMINATE outputs, vote result, and cast code are forbidden as
features.

Features are learned from Congress 116 roll-call metadata without target vote
directions:

- intercept;
- six fixed motion types;
- bill-number-present flag;
- session-two flag;
- one-hot top four CRS policy areas plus `other`;
- six most frequent non-stopword description tokens.

This yields exactly 20 features, so the existing applicability hard floor of
five calibration records per feature is met by the frozen 100-record role.

The scenario context is the stable task context
`{"institution":"us_senate","task":"roll_call_vote"}`. Full per-roll-call
metadata remains bound inside signed evidence instead of being misused as an
applicability context identity.

Within-day timestamps are deterministic order encodings derived from the roll
call number because Voteview supplies a date and roll-call sequence but not an
exact vote time. They preserve order but are not claimed as exact clock times.

## Existing-module execution

For every cohort person:

1. fit with existing `fit_person_model` and unchanged validation gates;
2. score correct-person, population, training-frequency, and frozen
   wrong-person controls on Congress 117;
3. if and only if base validation passes, run existing dynamic-state inference
   on the 80-record stream with its default hard floors;
4. if and only if mechanism eligibility passes, compare four training-only
   declared candidate families: intercept shift, limited linear residual,
   bill-by-motion interactions, and limited policy-by-token interactions;
5. create the existing composite artifact only when the mechanism report is
   `supported_candidate`;
6. record every public-interface refusal as a result, not as a harness crash.

The dynamic and mechanism branches use disjoint evidence and are not combined;
the current composite/dynamic integration gap remains visible.

## Decision labels

Let `B` be the number of six cohort members whose unchanged base validation
passes.

- `core_entry_not_established`: `B = 0`;
- `limited_structured_domain_entry`: `1 <= B <= 3`;
- `structured_domain_entry`: `B >= 4`.

These labels concern only entry into downstream PCFM paths in this one
retrospective structured domain. None authorizes belief/value/goal modules or a
general-person claim. Resuming semantic modules still requires a second domain
or genuinely prospective evidence.

## Refusal and reproducibility

Refuse raw digest drift, role overlap, outcome leakage, cohort mismatch,
insufficient counts, timestamp-order ambiguity, signature failure, saved
artifact reload mismatch, or any attempt to silently weaken an existing gate.
Report cohort-wide failures and successes; never select a successful person
after observing model performance.
