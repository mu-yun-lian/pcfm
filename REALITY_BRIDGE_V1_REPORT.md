# PCFM Reality Bridge v1 — Result

## Decision

**Status: `core_entry_not_established`.**

Zero of the six deterministically selected real people passed the unchanged
PCFM base-model validation gate. The dynamic-state path was consequently
blocked for all six, and the mechanism path refused all six because neither
personalization nor temporal stability had been validated.

This result does not show that the downstream algorithms fail after entering
their eligible state. It shows that the current real-data path cannot reach
that state in this cohort and domain. Under the frozen bridge decision rule,
development of belief, value, goal, memory, social, and self-model modules is
not authorized by this evidence.

## Frozen cohort

The cohort was selected from 49 eligible current senators by a party-balanced,
raw-manifest-bound hash rule. Congress 117 and Congress 119 vote directions
were not used for cohort selection.

| Person | Party code | Congress 116 fitting votes | Congress 117 validation votes |
|---|---:|---:|---:|
| Christopher Van Hollen | 100 | 718 | 900 |
| Richard Durbin | 100 | 708 | 946 |
| Jeanne Shaheen | 100 | 712 | 902 |
| John Thune | 200 | 717 | 940 |
| James Risch | 200 | 702 | 882 |
| Charles Grassley | 200 | 710 | 949 |

All six also had the frozen 480 valid Congress 119 records needed for
applicability, dynamic, discovery, selection, and confirmation roles inside the
existing 180-day model-age limit.

## Base validation

| Person | Correct-person NLL | Population NLL | Uplift | Wrong-person NLL | ECE | Temporal status |
|---|---:|---:|---:|---:|---:|---|
| Van Hollen | 0.946 | **0.425** | -0.521 | 0.760 | 0.401 | unstable |
| Durbin | 0.761 | **0.414** | -0.347 | 0.679 | 0.284 | unstable |
| Shaheen | 0.673 | **0.420** | -0.253 | 0.754 | 0.229 | unstable |
| Thune | 1.943 | **1.129** | -0.814 | 1.993 | 0.538 | unstable |
| Risch | 1.886 | **1.104** | -0.782 | 1.874 | 0.468 | unstable |
| Grassley | 1.532 | **0.925** | -0.607 | 1.511 | 0.346 | unstable |

NLL uplift is population NLL minus correct-person NLL, so every negative value
means personalization made prediction worse. Every paired 95% lower bound was
also negative. This is a cohort-wide failure, not a single-person edge case.

Every person triggered the same five existing reasons:

1. `insufficient_personalization_uplift`
2. `personalization_uplift_not_significant`
3. `calibration_error_too_high`
4. `mechanism_misspecification_suspected`
5. `temporal_behavior_drift_suspected`

## Downstream execution

The existing dynamic module requires a base model with validation status
`passed`. It therefore returned `blocked` for all six without consuming its
80-record role.

The existing mechanism module may repair only an otherwise personalized and
temporally stable base with a mechanism-only failure. It therefore refused all
six with:

- `base_model_personalization_not_validated`;
- `base_model_temporal_stability_not_validated`.

No mechanism report was produced and no composite artifact was created. The
gates behaved as implemented; the experiment did not bypass them merely to
obtain a downstream score.

## What was learned

1. The earlier single-Hawley failure generalizes to this frozen six-person
   cohort under a different, smaller feature mapping.
2. The current static person adapter is not a reliable real-data entry point in
   this domain. Correct-person identity was not recovered: for five of six
   people the wrong-person model was also better than the correct-person model.
3. Existing downstream modules cannot currently test whether state or mechanism
   components rescue this failure, because their eligibility contracts require
   the failed properties first.
4. The project therefore has an architectural sequencing problem, not merely a
   missing semantic module: reality can require state-dependent or structural
   modeling before a static person core earns full validation.
5. The population model itself performed very differently by party, so the
   current scenario mapping also omits important environment/regime structure.
   This is a competing explanation, not yet an identified cause.

## Claim boundary

This was a retrospective counterfactual replay. Historical plan timestamps
cannot prove genuine preregistration, and no dynamic or mechanism evidence was
actually eligible after the base failures. The result does not prove that
dynamic-state or mechanism algorithms are ineffective, nor that a real-person
model is impossible.

It does justify one project decision: do not resume reserved semantic modules
on top of the present entry dependency. The next design task must change the
testable core architecture so that stable trait, observable environment, and
time-varying state can be evaluated jointly without allowing one component to
explain its own validation data.

## Reproducibility

- Frozen contract: `REALITY_BRIDGE_V1.md`
- Harness: `work/reality-bridge-v1/run_bridge.py`
- Harness tests: `work/reality-bridge-v1/test_run_bridge.py`
- Frozen cohort and feature plan:
  `artifacts/reality_bridge_v1/cohort-plan.json`
- Machine-readable result: `artifacts/reality_bridge_v1/report.json`
- Saved and reloaded person bundles: `artifacts/reality_bridge_v1/models/`
- Report digest:
  `030723b4efaf230b1fdc668696f0776ad3c2cf2d113242bf9e6b62a4f16aa354`
