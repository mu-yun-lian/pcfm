# Voteview Reality Test v1

## Purpose

Run the existing PCFM stable person-choice model, unchanged, on real recorded
human decisions. This is an experiment harness and dataset mapping, not a new
cognitive module.

The operational claim is narrow: in United States Senate roll-call votes, a
person-specific MAP logistic adapter may predict one selected senator's later
Yea/Nay votes more accurately than the existing population model.

This test does not identify beliefs, reasons, ideology, private cognition, or
general decision ability. It does not validate Tyler Cowen. Success would show
only that the existing binary-choice machinery has some real-person predictive
utility in one structured domain.

## Frozen data and roles

Official Voteview Senate files are used for Congresses 116 through 119.

- Congresses 116–117: model fitting and target selection;
- Congress 118: applicability calibration;
- Congress 119 through 2026-07-31: retrospective diagnostic validation.

Only cast codes 1–3 (Yea) and 4–6 (Nay) are outcomes. Non-membership,
abstention, present, and not-voting codes are excluded.

Congress 119 is not confirmatory: its outcome distribution was inspected
during feasibility work. It must not be described as an untouched holdout.

## Frozen target-selection rule

Candidates must be present in the Congress 119 member file, have at least 500
valid fitting votes, at least 50 fitting Yea and 50 fitting Nay votes, and at
least 100 valid recorded votes in both Congress 118 and Congress 119. Among
eligible candidates, select the person maximizing the smaller of fitting Yea
and fitting Nay counts, then total fitting count, then lowest numeric ICPSR.

Validation vote direction is not used by the executable target-selection rule.

## Frozen population and wrong-person controls

The population ledger contains the target plus the ten current Democratic and
ten current Republican senators with the most valid fitting votes, excluding
the target before reference selection. The wrong-person control is the
eligible current senator from the target's party with the closest fitting Yea
rate, with ICPSR as the final tie-breaker.

## Frozen scenario representation

Every roll call is one scenario with options `Nay` and `Yea`. Features are
available without reading the target's vote:

- constant intercept;
- six fixed vote-question types;
- whether a bill number is present;
- whether the vote occurs in session two;
- one-hot CRS policy areas observed in Congresses 116–117, including missing;
- 32 binary description-token features selected by frequency from Congresses
  116–117 only.

Text token selection is unsupervised and does not use vote direction. The
following fields are forbidden as features because they contain or summarize
outcomes: `cast_code`, Voteview `prob`, Yea/Nay counts, `vote_result`, NOMINATE
midpoints, spreads, and log likelihoods.

## Metrics and existing gates

Report NLL, Brier score, accuracy, and ECE for:

- existing personal MAP logistic;
- existing population logistic;
- training-frequency constant;
- wrong-person MAP logistic.

The existing PCFM validation gate remains unchanged: at least 100 validation
records, mean NLL uplift of at least 0.01 over population, positive 95% paired
uplift lower bound, ECE at most 0.15, no excessive nonlinear mechanism-probe
uplift, and no detected temporal instability.

## Refusal and interpretation

Refuse missing raw files, digest changes after manifest creation, duplicate
person/roll-call trials, feature leakage, insufficient role counts, target
selection mismatch, schema mismatch, or ledger signature failure.

Passing is `retrospective_real_choice_support`. Failing is evidence against the
adequacy of the current representation/model for this domain, not evidence
that person-specific prediction is impossible in general.

