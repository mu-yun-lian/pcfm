# Joint Person Core Candidate v1

## Why this candidate exists

Reality Bridge v1 returned `core_entry_not_established`: all six frozen people
failed static personalization and temporal stability, so the existing dynamic
and mechanism modules could not legally start. The next candidate must test a
joint predictive core without weakening or modifying the deployed PCFM gates.

This document freezes an exploratory candidate. Implementation belongs under
`work/` until it passes its own comparison. It is not a new registry module and
must not replace the current model merely because it is more complex.

## Operational claim

For repeated binary choices with named pre-choice scenario and environment
features, a joint model containing population effects, stable person-specific
effects, and an online time-varying residual may predict a frozen real-person
cohort better than the strongest of the current population and static-person
baselines on later events.

Success supports only `joint_person_specific_predictive_utility` in the tested
domain, cohort, features, and time window.

It does not identify personality, belief, value, goal, emotion, private mental
state, causal mechanism, consciousness, or general person simulation.

## Competing explanations

- Null: repeated behavior contains no usable person-specific signal beyond the
  population/environment model.
- Environment-only alternative: apparent person change is explained by
  observable regime or task context, not a changing person state.
- Dynamic-population alternative: recent outcomes improve prediction, but the
  history is not person-identifying.
- Static-person alternative: stable person parameters suffice and the dynamic
  component only overfits noise.
- Omitted-structure alternative: all proposed components remain misspecified.

The evaluation must retain controls for all five explanations.

## One candidate kernel

For person `p`, event `t`, scenario features `x`, and pre-choice observable
environment features `e`:

```text
eta[p,t] = x[t] · beta
         + x[t] · u[p]
         + e[p,t] · gamma
         + z[p,t]

P(y[p,t] = 1 before observing y[p,t]) = logistic_normal(eta[p,t], V[p,t])
```

- `beta`: population scenario weights;
- `u[p]`: stable person-specific deviation with hierarchical shrinkage;
- `gamma`: shared coefficients for declared pre-choice environment features;
- `z[p,t]`: scalar continuous-time residual state;
- `V[p,t]`: declared parameter/state uncertainty used by the existing shared
  Logistic-normal probability kernel.

The stable terms are fitted only from fitting roles. During evaluation they are
frozen. For each later event, the probability is emitted first; only then may
the observed outcome update `z[p,t+1]`. No event may score itself after update.

The joint candidate is validated as one kernel. A failed static-only component
does not block evaluation of the joint candidate. Deployment remains forbidden
until the whole joint kernel passes independent gates.

## Environment interface

The core accepts named numeric variables known before the choice. A domain
adapter defines them and binds provenance. It may not pass outcome summaries,
future information, model predictions, or identifiers that merely memorize an
event.

For the frozen Voteview diagnostic adapter, the initial allowed environment
variables are:

- whether the person's recorded party matches the Senate majority party;
- whether it matches the sitting President's party;
- Congress/session indicator known before the vote;
- normalized elapsed calendar time known before the vote.

These are domain-adapter variables, not permanent political assumptions inside
the joint core. Their values and historical sources must be frozen before the
candidate is run.

## Evidence roles

The first diagnostic reuses the already frozen six-person Reality Bridge cohort
but creates a new, non-overlapping temporal evaluation design:

- fitting: early historical window, stable and environment parameters only;
- applicability calibration: separate scenarios without parameter fitting;
- sequential validation: later events scored in order, with only past
  sequential-validation outcomes allowed to update the dynamic state;
- untouched comparison: a final later block never used for feature, threshold,
  decay, or candidate selection.

Exact cutoffs, counts, feature names, decay candidates, and stopping rules must
be written to a separate executable experiment manifest before reading the new
candidate's results. Previously inspected Voteview outcomes make this first run
retrospective diagnostic, not confirmatory.

## Frozen controls

Score the same events with the same probability and metric code for:

1. current population Logistic;
2. current static correct-person MAP Logistic;
3. observable-environment population model without `u[p]` or `z[p,t]`;
4. dynamic-population model without stable person deviation;
5. wrong-person joint model using another frozen same-party history;
6. history-shuffled joint model preserving marginal outcome counts;
7. full joint candidate.

The full candidate is not person-specific unless it beats controls 3, 4, 5,
and 6, not merely the old static model.

## Candidate selection and hard decision

Only a small, preregistered grid of state half-lives and shrinkage strengths may
be fitted on the candidate-selection role. Exactly one configuration is then
frozen before final comparison. No neural architecture or LLM-generated feature
may be added after results are seen.

Minimum diagnostic support requires all of the following:

- all six frozen people are reported;
- at least four of six have positive final-block NLL improvement over the best
  non-person-specific control;
- equal-person-weighted mean NLL improvement is at least `0.01`;
- paired uncertainty lower bound for the pooled improvement is above zero;
- final-block ECE is at most `0.15`;
- correct-person joint prediction beats both wrong-person and shuffled-history
  controls;
- no applicability, time, leakage, replay, or artifact-identity override;
- every reported probability is reproduced by the same public candidate
  kernel before its outcome update.

Failure returns `joint_core_candidate_not_supported`. Passing returns only
`retrospective_joint_core_support`; it authorizes a genuinely prospective or
second-domain test, not semantic cognitive modules.

## Required adversarial tests

- zero person heterogeneity;
- stable omitted environment structure;
- abrupt and gradual regime changes;
- shuffled person histories;
- wrong-person history substitution;
- outcome leakage through environment features;
- update-before-score attack;
- timestamp permutation and equal-time ambiguity;
- feature and person-ID renaming invariance;
- applicability holes and stale state;
- serialization/reload equality;
- report probabilities equal deployed candidate probabilities.

## Promotion rule

The current production model and module registry remain unchanged until:

1. the candidate passes the frozen retrospective comparison;
2. the same candidate, without architecture changes, passes a second domain or
   prospective collection;
3. dynamic, mechanism, active planning, update, storage, CLI, and composite
   consumers are redesigned around one joint model identity and pass full
   regression.

If the candidate fails, do not add semantic modules. Reassess the person/data
representation or narrow the project to domains where person-specific
prediction is empirically identifiable.
