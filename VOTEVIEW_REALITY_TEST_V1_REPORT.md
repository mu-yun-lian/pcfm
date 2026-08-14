# Voteview Reality Test v1 — Result

## Verdict

**Status: `retrospective_no_support`.**

The unchanged PCFM personal MAP logistic model failed this real-person,
later-period Senate roll-call test. It learned a person-specific pattern in the
fitting period, but that advantage did not persist. In Congress 119 the
population model was substantially better, and the wrong-person control was
slightly better than the correct-person model.

This result rejects the narrow v1 claim for this dataset and representation. It
does not establish that person-specific prediction is impossible, and it says
nothing direct about beliefs, reasons, private cognition, general intelligence,
or Tyler Cowen.

## Frozen experiment

- Official Voteview Senate roll-call and member data, Congresses 116–119.
- Congresses 116–117: fitting and target selection.
- Congress 118: applicability/transfer diagnostic.
- Congress 119 through 2026-07-31: retrospective diagnostic validation.
- Existing PCFM model, prediction path, and validation gates were not changed.
- The executable target rule selected **Joshua David Hawley** (ICPSR 41901)
  without using Congress 119 vote direction.
- The same-party wrong-person control was **Mike Lee** (ICPSR 41110).
- 34,743 population fitting records, including 1,649 target records; 640 target
  applicability records; 836 target validation records; 59 frozen features.
- All generated ledgers passed signature verification. Saved ledgers, keys,
  model identity, and a sample prediction survived artifact reload.

Congress 119 is not an untouched confirmatory holdout because aggregate outcome
distribution was inspected during feasibility work. The correct label is
therefore retrospective diagnostic evidence.

## Main validation result

| Predictor | Accuracy | NLL (lower is better) | Brier | ECE |
|---|---:|---:|---:|---:|
| Correct-person MAP logistic | 0.467 | 0.828 | 0.297 | 0.315 |
| Population logistic | **0.764** | **0.530** | **0.173** | **0.166** |
| Training-frequency constant | 0.748 | 0.667 | 0.237 | 0.220 |
| Wrong-person MAP logistic | 0.537 | 0.820 | 0.296 | 0.295 |

Personalization NLL uplift over population was **-0.298**, with paired 95%
bootstrap interval **[-0.338, -0.258]**. The entire interval is below zero: this
is not a close or noisy miss. Under the frozen metric, personalization made the
predictions worse.

## Period split

| Period | Records | Correct-person NLL | Population NLL | Wrong-person NLL | Frequency NLL |
|---|---:|---:|---:|---:|---:|
| Fitting, Congresses 116–117 (in sample) | 1,649 | **0.606** | 0.760 | 0.631 | 0.692 |
| Applicability, Congress 118 | 640 | **0.717** | 0.957 | 0.755 | 0.722 |
| Validation, Congress 119 | 836 | 0.828 | **0.530** | 0.820 | 0.667 |

The fitting-period result shows that the implementation can fit a
person-specific signal. Congress 118 shows that this signal had some transfer
value by NLL, although its absolute accuracy was only 0.509 and it barely beat
the frequency baseline by NLL. Congress 119 shows a strong reversal.

These observations support a **temporal-transfer failure**, but do not by
themselves identify its cause. Plausible competing explanations include an
unrepresented political regime/state change, inadequate scenario features,
unstable person parameters, interactions missing from the linear form, or a
mixture of these. A controlled follow-up is required to distinguish them.

## Existing validation gates

The existing model returned `failed` with all five recorded reasons:

1. `insufficient_personalization_uplift`
2. `personalization_uplift_not_significant`
3. `calibration_error_too_high`
4. `mechanism_misspecification_suspected`
5. `temporal_behavior_drift_suspected`

Supporting diagnostics were ECE 0.315, nonlinear mechanism-probe NLL uplift
0.088, and temporal status `unstable` with drift detected.

## What this changes

No production module was added and no PCFM model code was modified. The result
does, however, change what may honestly be claimed: the current stable
person-choice kernel is **not yet empirically adequate for later-period real
roll-call prediction under this representation**.

The next experiment should remain below the module layer. Freeze this failed v1
as evidence, then run an ablation matrix that changes one factor at a time:

1. Add only pre-vote, non-outcome regime/context variables.
2. Test rolling or state-dependent parameter adaptation while retaining the
   same prediction task and controls.
3. Test interaction/nonlinear capacity only after the context ablation.
4. Require a new later time block or prospectively collected votes before any
   confirmatory claim.

Do not add a new cognitive module merely to hide this failure. A new component
is justified only if these controlled tests isolate a missing mechanism that
the existing contracts cannot express.

## Reproducibility artifacts

- Frozen contract: `VOTEVIEW_REALITY_TEST_V1.md`
- Executable harness: `work/voteview-real-audit/run_experiment.py`
- Machine-readable report: `artifacts/voteview_real_audit/report.json`
- Raw-file digest manifest: `artifacts/voteview_real_audit/raw-manifest.json`
- Eligibility audit: `artifacts/voteview_real_audit/candidate-eligibility.csv`
- Signed ledgers, model, and verification keys:
  `artifacts/voteview_real_audit/`
- Report digest:
  `b87ca68df9b4f8e6cab15d50619457b0dd0b543c88c17101ed378f453cfee80d`
