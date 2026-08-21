# Simulation V4 rebuild report

## Result

The production conversation path now uses `SimulationKernelV4`. V3 and Response Kernel V2 remain frozen baselines and are not active prediction components. The stage status is `implemented_exploratory`: the software path is integrated and tested, but real-person accuracy is still not assessed.

## Rebuilt path

1. Confirmed raw Q&A and single-speaker spans enter V4 only as direct historical evidence.
2. The material-processing model proposes exact event, condition, reason, knowledge and tradeoff spans using closed interest/domain IDs.
3. A candidate remains outside fitting until source, speaker and exact response are confirmed. Ungrounded tradeoff spans are rejected.
4. Only reviewed semantic events with valid ISO dates create preference atoms. Missing or invalid time remains direct evidence only.
5. A runtime preference structure requires at least two independent source lineages. Domain, role, explicit or relative later-time and counterevidence gates are enforced by code.
6. A selected dialogue model may propose query domains, exact option spans, history message IDs and supplied event/structure IDs. Both sides of a tradeoff and any model-added domain require grounded message spans; selecting a structure cannot choose the direction.
7. V4 freezes the person direction and evidence support before the dialogue model runs. The model returns structured external briefing and allow-listed person-knowledge IDs, while code assembles the response and rejects added person attribution or unsupported numbers.
8. Reality optimization recomputes the selected event from raw source material, promotes only that event, and records V4 holdout status. Derived-event tampering, whole-source promotion and assessed V4 holdout regression are rejected.
9. Persisted V4 artifacts are verified and their semantic payload is recomputed from the version's reviewed source bytes before conversation prediction.

## Preserved product behavior

- ordinary greetings and short follow-ups;
- exact historical answers;
- related historical evidence without a new stance;
- normal general-knowledge answers when no person prediction is supported;
- per-person model selection with transparent model failure;
- separate content and style processing;
- opt-in reality comparison, optimization, versioning, rollback, archive and restore.

## Verification

- Python compilation: passed with `python -m compileall -q src tests`.
- JavaScript syntax: passed with `node --check src/pcfm/web_static/app.js`.
- Focused product/integration regression: 104 tests passed in 72.907 seconds.
- Full regression: 369 tests passed in 286.455 seconds.
- Cognitive completion gate: intentionally not passed. Its only remaining failures are `module.status` not being `implemented_confirmatory` and `evidence.future_holdout` being false.

## Costs and limits

- Recomputing the semantic payload from reviewed sources increased the full-suite runtime. The current implementation favors integrity; source-hash caching should be added only if measured conversation latency requires it.
- The closed interest taxonomy is intentionally limited. Unsupported meanings remain unknown rather than being silently merged.
- Human confirmation bounds fabricated spans but cannot prove that a semantic label is correct.
- General-model background remains unverified external knowledge. The hard gates prevent new person attribution and unsupported numeric specificity in person projections, but cannot prove every ordinary factual sentence true.
- No adequate strictly later real-person holdout currently establishes coverage, direction accuracy or calibration. Evidence support shown by the product is not accuracy probability.
