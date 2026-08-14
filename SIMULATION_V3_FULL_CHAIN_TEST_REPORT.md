# Simulation V3 full-chain test report

Date: 2026-08-14  
Decision: `NO_GO_REBUILD_CORE`  
Software regression: `349 passed`  
Real-person prediction accuracy: `not_assessed`

## Scope

The test used the frozen `SimulationKernelV3` and the pre-execution protocol in
`SIMULATION_V3_FULL_CHAIN_TEST_PROTOCOL.md`. No product extraction rule,
threshold, query mapper, or predictor was changed after observing real-source
results. One incorrect predeclared archive URL was corrected to the verified
official page alias before the final run.

The audit separates:

1. official-source availability;
2. exact attributable-turn preparation;
3. event and preference construction;
4. query and dialogue behavior;
5. strictly later confirmation;
6. ordinary software compatibility.

## Corpus result

Fourteen official archived Obama White House press-conference pages were
downloaded successfully. The archive spans 2009 through 2016. Raw HTML and
normalized visible text are stored separately; all 14 raw and normalized hashes
recomputed successfully against the source manifest.

Frozen roles:

| Role | Period | Label-parsed turn candidates |
|---|---|---:|
| parameter fitting | 2009-2014 | 137 |
| applicability calibration | 2015 | 34 |
| sealed confirmation | 2016 | 63 |
| total | 2009-2016 | 234 |

The source-availability gate passes. The confirmed-model-corpus gate does not.
The deterministic label parser flagged 45 turns for review, and zero turns have
independent human confirmation. The official archive itself contains occasional
speaker-label anomalies; for example, a long Obama answer in the 2015-07-15
transcript is labelled `Q`. Therefore 234 is a candidate count, not a gold-label
count.

## Chain results

### Honest raw-document ingestion

An entire press conference is a mixed-speaker document. When the metadata is
truthfully set to `mixed_speakers`, V3 rejects all fitting sources:

- accepted event frames: 0;
- error: `no_eligible_reviewed_raw_source_frames`.

This is a valid safety refusal, but the product currently has no connected path
that promotes confirmed speaker spans into the V3 frame input.

### Unsafe whole-document override

Pretending that each complete press conference is one-person material allows
the source to enter V3, but produces:

- 289 mechanical text-chunk frames;
- 0 frames exactly equal to a label-parsed Obama answer;
- 2 preference atoms;
- 2 single-event structures.

One extracted atom is a truncated/nonsensical phrase around “semblance of
credit ... rather than Social Security recipients”; this path is not usable.

### Label-parsed attributable-turn audit view

Supplying 137 fitting-period Q/A candidates directly produces:

- 137 event frames;
- 1 preference atom;
- 1 single-event preference structure;
- 0 runtime-eligible repeated preference structures;
- 0 preference relations;
- 1,838 sentence-level “knowledge claims”.

The only extracted preference atom treats the metaphor “taken a scalpel to the
discretionary budget rather than a machete” as:

- protected interest: `so ve done ve taken scalpel discretionary budget`;
- accepted cost: `machete`.

This is not a valid operational value atom. The knowledge-claim count is also a
sentence inventory, not evidence of 1,838 verified items of person knowledge.

### Confirmed event-candidate integration

Adding a confirmed/promoted `response_events` candidate changes neither the V3
semantic digest nor its frames. Current material review writes the candidate to
the legacy structure, while V3 reads only raw `qas` or segments.

Result: `response_events_changed_v3 = false`.

## Strictly later confirmation

The 2016 sealed set contains 63 label-parsed turn candidates. The current
explicit English tradeoff regex identifies only 2 as scorable labels and rejects
61 for no explicit tradeoff label.

- scorable sample count: 2;
- V3 person-prediction coverage: 0.0;
- covered-direction accuracy: `not_assessed`.

The evaluator reports `assessed_exploratory` because two target labels exist,
but no prediction is covered. It does not establish accuracy.

## Adversarial results

### Chinese extraction

Two dated, attributable Chinese safety-versus-speed examples produce:

- 2 event frames;
- 0 preference atoms;
- 0 preference structures;
- domain: `general` only.

### Semantic paraphrase

Two semantically consistent English examples, one expressed as “safety before
speed” and one as “preventing harm matters more than moving quickly”, produce
only one atom. The expressions are not normalized into repeated support.

### Same-lineage pseudoreplication

Two duplicate events sharing one lineage create a runtime
`preference_structure_answer` with reported confidence 0.6. The existing unit
test checks only that the duplicate does not become a *cross-domain* structure;
it does not prevent same-domain runtime prediction from one lineage.

### Role and time transfer

A structure supported in public hospital/product contexts still predicts for a
private-parent role. A question using “Today” receives no temporal warning. An
explicit 2026 question is marked later than the 2021 evidence window but still
returns the person prediction instead of refusing or expiring it.

### Retrieval and dialogue

The unrelated compound question “How can phone calls and private life improve a
crypto product?” retrieves two Sally Ride events as similar evidence. A fixed
short follow-up (`Why?`) resolves context; the natural follow-up “Given that,
would you have reacted differently ...?” does not.

## Gate and regression results

The cognitive module completion checker exits 1. Failures include:

- module status is `implemented_exploratory`, not confirmatory;
- `future_holdout` is false;
- test references are not existing file paths in the required schema;
- verification commands/results do not have aligned structured exit records.

Independent software checks:

- Python compilation: exit 0;
- focused Simulation V3 tests: 12 passed;
- full regression: 349 passed in 286.955 seconds, exit 0.

The regression result establishes software compatibility only. It cannot
override the failed real-person and completion gates.

## Final judgment

| Prerequisite | Result |
|---|---|
| A real person with abundant official raw material | pass |
| Recomputable collected raw-source manifest | pass |
| Human-confirmed attributable event corpus | not yet complete |
| Honest raw material to V3 event-frame path | fail |
| Useful real-person preference extraction | fail |
| Independent repeated preference construction | fail |
| Chinese support | fail |
| Context/role/time applicability | fail |
| Strictly later covered accuracy | not assessed |
| Software regression | pass |
| Confirmatory module gate | fail |

The current V3 core must not be promoted as a usable or accurate person
simulator. The evidence-rich Obama source base is adequate for the rebuild, but
the attributable turns still require exact-span human review. The next
authorized implementation should replace the V3 event-ingestion, semantic
normalization, independent-support, applicability, dialogue-state, and
confirmation chain rather than tune the current regex thresholds.

## Reproducible artifacts

- `SIMULATION_V3_FULL_CHAIN_TEST_PROTOCOL.md`
- `work/run_simulation_v3_full_chain_audit.py`
- `artifacts/simulation_v3_full_chain_audit/source_manifest.json`
- `artifacts/simulation_v3_full_chain_audit/oracle_turns.json`
- `artifacts/simulation_v3_full_chain_audit/raw/`
- `artifacts/simulation_v3_full_chain_audit/full_chain_results.json`
- `artifacts/simulation_v3_full_chain_audit/audit_console.log`
- `artifacts/simulation_v3_full_chain_audit/module_gate_check.log`
- `artifacts/simulation_v3_full_chain_audit/focused_simulation_v3_tests.log`
- `artifacts/simulation_v3_full_chain_audit/full_regression.log`
- `artifacts/simulation_v3_full_chain_audit/full_regression_result.json`
