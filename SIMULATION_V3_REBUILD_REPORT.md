# Simulation V3 rebuild report

Status: `implemented_exploratory`

## What was rebuilt

The deployed person-simulation layer is now
`pcfm.simulation_v3.SimulationKernelV3`. It is fitted from reviewed raw source
records and refuses V2 model artifacts. Response Kernel V2 is retained only as
a frozen comparison baseline.

The V3 data path is:

1. reviewed attributable source spans;
2. one event frame per public response;
3. zero or more explicit public tradeoff atoms attached to that event;
4. source-lineage-aware preference structures, counterevidence and relations;
5. a query conflict/event frame;
6. direct historical response, related historical evidence, eligible public
   preference projection, or no person-specific prediction;
7. a frozen content contract;
8. the independent surface-style renderer.

## Evidence and inference boundaries

- Missing time, mixed speakers and unverified authenticity block V3 fitting.
- Multiple tradeoffs in one response do not duplicate the event count.
- Repeated reports from one source lineage do not create cross-domain support.
- Reversed tradeoffs are counterevidence and require contextual separation.
- Support weights describe context completeness and independent evidence, not
  prediction probability.
- Publicly used claims form a bounded demonstrated-knowledge inventory; they
  are not treated as verified facts or a complete knowledge model.
- A dialogue model may provide disclosed general knowledge and wording, but it
  cannot select or create the person's position.
- Unmatched questions use `general_assisted` with
  `person_prediction_status=not_available`; the old overall-tendency fallback
  is not deployed.

## Accuracy status

The evaluator accepts only confirmed `final_holdout` sources strictly later
than the model sources and only scores responses containing an explicit,
reviewable tradeoff direction. It reports prediction coverage separately from
direction accuracy. If no eligible sample exists, the result is
`not_assessed`; software tests cannot upgrade this state.

The current Sally Ride and Barack Obama demo holdouts contain no eligible
explicit tradeoff labels. Both therefore remain `not_assessed`. Their V3
artifacts contain six event frames each and zero explicit preference atoms.
They demonstrate direct evidence, related historical evidence, style rendering
and safe no-stance degradation, but they do not demonstrate new-situation
preference accuracy.

## Verification

- Focused product and V3 suites: 84 tests passed.
- Final full regression: 349 tests passed in 188.765 seconds.
- Module artifact integrity, source-order invariance, old-schema refusal,
  duplicate-lineage attacks, temporal extrapolation and product-entry isolation
  are covered in `tests/test_simulation_v3.py` and the integration suites.

## Remaining work before an accuracy claim

Collect or curate multiple independent, attributable public events with
reviewable tradeoffs plus strictly later sealed events of the same decision
families. Compare V3 against direct retrieval, V2, person-history frequency,
population and wrong-person baselines. A larger language model or more software
tests cannot substitute for that evidence.
