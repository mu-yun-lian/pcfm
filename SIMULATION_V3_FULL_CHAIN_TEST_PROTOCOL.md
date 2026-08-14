# Simulation V3 full-chain test protocol

Status: `frozen_before_real-source_execution`

This protocol audits the currently deployed `SimulationKernelV3` without
changing its extraction rules, thresholds, query mapping, or prediction logic.
Observed failures may not be used to tune the frozen run. Any later repair must
use a new schema/version and a new sealed confirmation set.

## Operational question

Can the current product take attributable, dated public material for one real
person and produce source-bound event frames, repeated public-preference
structures, context-appropriate dialogue predictions, and content-preserving
style output that perform usefully on strictly later public responses?

Success supports only an executable model of observable public response
patterns in the tested presidential role and time range. It does not identify
private beliefs, sincere motives, complete knowledge, causal psychology, or
future behavior outside the tested scope.

Null and competing explanations:

1. direct lexical retrieval explains the apparent success;
2. generic language-model knowledge explains fluent output;
3. repeated reporting or one press event explains apparent preference support;
4. scripted office communication rather than a personal pattern explains the
   observed regularity;
5. time, role, audience, or policy regime changes explain later disagreement.

The refusal result is `no_person_specific_prediction` whenever no independently
supported, locally applicable public-preference structure exists.

## Frozen evidence roles

| Time | Role | Permitted use |
|---|---|---|
| 2009-01-20 through 2014-12-31 | parameter fitting | event and preference construction only |
| 2015-01-01 through 2015-12-31 | applicability calibration | diagnostics only; no final scoring |
| 2016-01-01 through 2017-01-20 | sealed confirmation | final scoring only; never threshold tuning |

The source set is fixed in the audit runner. Pages from the same press event or
near-duplicate transcript share one lineage. Collection order and generated
identifiers must not affect fitted artifacts.

## Two ingestion views

1. **Honest raw-document view.** A complete press conference is declared
   `mixed_speakers`. Passing it by pretending that the whole document is Barack
   Obama's speech is a failure, not a workaround.
2. **Independent attributable-turn oracle.** A deterministic audit parser uses
   explicit `Q` and `THE PRESIDENT`/`PRESIDENT OBAMA` labels to create exact
   question-answer spans. This is an audit reference, not production input and
   not evidence that the current product can perform the extraction.

## Predeclared checks

### Corpus readiness

- at least 10 official archived pages fetched successfully;
- at least 60 attributable Obama question-answer turns;
- at least five calendar years and multiple policy domains;
- exact URL, response date, speaker label, content hash, and collection result;
- no search snippets, summaries, or model-generated answers as fitting truth.

### Source and event integrity

- honest mixed-speaker sources must not silently enter fitting;
- at least 95% of accepted event responses must be exact attributable answer
  spans with no reporter or other speaker text;
- missing time stays unknown or is rejected explicitly, never guessed;
- one source repeated or split into many spans must not create independent
  support.

### Preference extraction and aggregation

- every preference direction, accepted cost, condition, exception, and reason
  must point to an exact source span;
- paraphrases may not be missed solely because strings differ;
- opposing directions must remain contextual counterevidence;
- no runtime-eligible repeated structure may be created from one lineage;
- Chinese and English formulations of the same tested tradeoff must map
  consistently or return an explicit unsupported status.

### Query, dialogue, and applicability

- query mapping must preserve options, domain, role, audience, time, and changed
  conditions;
- natural elliptical follow-ups must resolve the referenced event or request
  clarification;
- unrelated compound questions must not retrieve events due to a low lexical
  threshold;
- role, audience, domain, and stale-time metadata must affect applicability;
- generic model knowledge may explain an issue but may not create a person
  stance or become person fitting evidence.

### Confirmation and baselines

- final responses must be strictly later than all fitting records;
- compare the deployed V3 path with direct retrieval, wrong-person/person-name
  removal, person-history frequency, frozen V2, and generic-answer baselines
  where the target label is identified;
- report coverage, covered-direction accuracy, abstention behavior, false
  retrieval rate, and calibration separately;
- fewer than 30 independent covered confirmation cases is pipeline evidence
  only, not an accuracy claim; a confirmatory claim additionally requires a
  preregistered power/precision calculation.

### Integration and software

- material review, version creation, prediction API, web dialogue, selected
  model use, style rendering, reality optimization, persistence, and reports
  must consume the same V3 artifact view;
- focused tests and the complete existing regression suite must pass;
- the cognitive module gate checker must pass before any confirmatory label.

## Stop rules

The result is `NO_GO_REBUILD_CORE` if any of the following occurs:

- confirmed event candidates do not alter the V3 event frames actually used;
- the honest raw-material path cannot create attributable event frames;
- real fitting material creates no runtime-eligible preference structure;
- sealed confirmation accuracy remains unassessed;
- the deployed query path has reproducible context or false-retrieval failures;
- the completion gate does not pass.

Software regression success cannot override a `NO_GO_REBUILD_CORE` result.
