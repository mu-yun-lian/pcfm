# Person-Issue Relational Core v1

## Why this candidate exists

Joint Person Core Candidate v1 was rejected because the shared dynamic control
beat the full stable-person model for all six frozen people. The result did not
show that history was useless. It showed that the prior 20-feature
representation did not yield stable person-specific utility beyond a shared
dynamic structure.

This candidate changes only the observable relation representation. It reuses
the same joint Logistic-normal prediction kernel and remains under `work/`.
It may not enter `src/pcfm` unless it passes this diagnostic and later external
confirmation.

## Operational claim

For repeated Senate Yea/Nay choices, a stable profile of how a named person
deviates from the same-party baseline across deterministic pre-choice text,
motion, and bill-type factors may improve later prequential probabilities over
a same-party dynamic baseline.

Success supports only
`party_relative_person_issue_predictive_utility` for the frozen people,
features, chamber, and time window.

It does not identify beliefs, values, goals, reasons, ideology, personality,
private mental state, causal mechanism, consciousness, or a general simulation
of the person.

## Competing explanations

1. Null: no stable person-specific signal remains after party and recent
   history are known.
2. Party-dynamic alternative: apparent individuality is entirely a shared
   party relation plus an online residual state.
3. Lexical-proxy alternative: apparent gain comes from repeated wording or
   procedural templates, not a transferable person-item relation.
4. Wrong-profile alternative: any same-party profile works equally well.
5. Omitted-structure alternative: the deterministic representation is still
   missing the conditions that distinguish the person's choices.

## Observable representation

Only fields declared available before the choice are accepted:

- `vote_desc`;
- `vote_question`;
- `dtl_desc`;
- `crs_policy_area`;
- `crs_subjects`;
- the structural prefix of `bill_number`.

Text is lower-cased, tokenized by a fixed ASCII alphanumeric rule, and mapped
by signed SHA-256 hashing into 128 bins. An outcome-blind truncated SVD fitted
only on unique Congress 116-118 roll-call descriptions reduces it to eight
frozen factors. No LLM, embedding API, outcome-based vocabulary selection, or
future-label feature is permitted.

The deployed scenario vector contains:

- one intercept;
- five exclusive motion types;
- five exclusive bill types;
- eight frozen text factors.

The global same-party baseline receives the scenario vector, its interaction
with a pre-choice Republican indicator, and two regime indicators: party
matches Senate majority and party matches President. Only the 19-dimensional
scenario vector receives a stable person-specific deviation.

The following are forbidden because they contain the current outcome or are
computed from it: `cast_code`, `prob`, `yea_count`, `nay_count`, `vote_result`,
all `nominate_mid_*`, all `nominate_spread_*`, and
`nominate_log_likelihood`.

## One deployed kernel

For person `p` and event `t`:

```text
eta[p,t] = global_party_regime(phi[t])
         + phi[t] dot person_profile[p]
         + dynamic_state[p,t]

P(y[p,t] = 1 before y[p,t]) = logistic_normal(eta[p,t], variance[p,t])
```

The probability is always emitted before the current outcome may update the
dynamic state. Selection, final scoring, wrong-person control, shuffled-history
control, saved artifacts, and reload verification must call this same kernel.

## Frozen evidence roles

- representation basis: unique Congress 116-118 roll-call descriptions,
  without any outcomes;
- estimation: Congress 116-117 target-person outcomes;
- candidate selection: Congress 118 target-person outcomes;
- final refit: Congress 116-118 after one configuration is selected;
- dynamic warm-up: each target's first 450 valid Congress 119 outcomes;
- applicability calibration: the next 80 valid Congress 119 events per target,
  with outcomes excluded from every parameter and state update;
- final comparison: the following 300 valid Congress 119 events per target.

The last two roles do not overlap Joint Person Core v1's first 450 Congress 119
events. The raw files were already locally available, so the evidence remains
`retrospective_diagnostic`, not confirmatory.

## Frozen controls

All controls score the same final event for the same target:

1. same-party static relation model;
2. same-party dynamic relation model without a person profile;
3. correct full person profile plus dynamic state;
4. wrong same-party profile plus its own causally updated residual state;
5. profile fitted after deterministically shuffling each person's historical
   outcomes within Congress, plus its own causally updated residual state.

The correct model is person-specific only if it beats controls 2, 4, and 5.

## Frozen selection grid

- stable person precision: `4`, `16`, `64`;
- dynamic half-life in days: `7`, `30`, `90`;
- dynamic stationary variance: `0.5`, `1.5`;
- observation variance: fixed at `1.0`;
- global ridge precision: fixed at `1.0`;
- hash dimension: fixed at `128`;
- SVD rank: fixed at `8`.

Exactly one of 18 configurations is selected by six-person equal-weighted
Congress 118 NLL. No feature, rank, grid point, threshold, or control may be
changed after final results are read.

## Hard diagnostic gates

Every item must pass:

- all six people have exactly 300 final scored events;
- at least four of six improve over the same-party dynamic control, including
  at least one person from each party;
- equal-person mean NLL improvement over that control is at least `0.01`;
- six-person cluster-bootstrap 95% lower bound is above zero;
- equal-person mean NLL improvement over both wrong-person and shuffled-history
  controls is at least `0.005`;
- pooled correct-model ECE is at most `0.15`;
- zero applicability, leakage, replay, sequence, identity, or reload refusal;
- every saved probability exactly matches a reloaded public candidate call
  made before the current outcome update.

Applicability uses the eight frozen continuous text factors for global and
local support, plus exact observed support for motion type, bill type, domain,
options, and context. Each person's 80-event feature-only calibration remains
valid for at most 365 days. This duration is frozen because the 300-event final
window spans about nine months; a 180-day duration would mechanically refuse
the end of the preregistered window before evaluating the candidate.

Failure returns `person_issue_relational_candidate_not_supported`. Passing
returns only `retrospective_person_issue_relational_support` and authorizes a
prospective or second-domain test, not semantic modules or production use.

## Identifiability and completeness boundary

Hashes bind the local Voteview files and the experiment design. They cannot
prove that Voteview published every vote, that the source timestamps are true,
or that every allowed text field was actually available before voting began.
Those facts require an external archive or preregistered future collection.

Even a passing predictive comparison cannot determine whether the fitted
profile represents reasons, strategic voting, constituency pressure, party
coordination, stable preferences, or another observationally equivalent cause.

## Promotion rule

The candidate remains work-only unless it passes all frozen diagnostic gates.
Even if it passes, production promotion requires the same frozen feature map
and model identity to pass a prospective or second-domain comparison and then
requires redesign of every production consumer around the unified core.
