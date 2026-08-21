# Simulation V5 rebuild report

## Result

The product now uses one conversation-conditioned Simulation V5 entry point for person-response planning. The implementation is usable as an exploratory MVP, but it is not a confirmatory accuracy result.

Status: `implemented_exploratory_accuracy_not_assessed`.

## Repaired failures

- The current message is treated as a delta to the persisted full conversation state rather than an isolated query.
- Topic resolution may select real earlier user-message IDs beyond the previous 12-message window. Generated assistant prose is context only and never fitting evidence.
- Fixed-width navigation chunks from raw material cannot become response episodes. Confirmed question-answer pairs and explicitly reviewed, source-grounded event frames remain eligible.
- Each episode records trigger, response, time, interlocutor, occasion, available information, missing fields, and grounding status. Missing time remains unknown and blocks temporal aggregation only.
- A language model may resolve semantics and provide general knowledge, but it cannot set the person's stance or attach unseen value evidence. Code selects person-specific evidence and public-orientation direction.
- Ordinary unmatched questions reach the selected and verified dialogue model before the general fallback. Unavailable services fail explicitly; there is no silent provider fallback.
- Natural-response composition is bounded by a code-selected stance anchor and allow-listed evidence IDs. Unsupported numbers, dates, memories, biography, and person claims are rejected.
- Reality-answer comparison remains opt-in. Optimization promotes only the selected, source-grounded reality event, recomputes the version, evaluates with the same V5 kernel, and remains rollbackable.
- Person archive, restore, permanent-delete confirmation, model selection, and conversation isolation remain in the unified product path.
- The Barack Obama demo's direct recommended question was corrected to an actual source question after browser acceptance exposed that the former paraphrase did not match without a dialogue model.

## Verification

- `PYTHONPATH=src python -m unittest discover -s tests -v`: 379 tests passed in 303.648 seconds, exit code 0.
- `PYTHONPATH=src python -m compileall -q src tests`: passed, exit code 0.
- `node --check src/pcfm/web_static/app.js`: passed, exit code 0.
- Real local-page acceptance: app version `0.10.0-simulation-v5`, two demo people loaded, corrected direct recommendation returned `historical_direct_evidence`, and the page error log was empty.
- Module gate: all implementation and test-record requirements pass; the completion gate intentionally fails only `implemented_confirmatory` and `future_holdout`.

## Accuracy boundary

The 379 tests validate software behavior, invariants, leakage controls, and product integration. They do not establish accurate prediction of a real person's future response. That claim requires an independent, time-separated, full-conversation holdout corpus for a material-rich person and comparison against current-message-only, retrieval-only, generic-model, wrong-person, and shuffled-history baselines.

Until that evaluation exists, evidence support scores must not be described as prediction accuracy or probability that the person would really say the answer.

The built-in demo people currently have one independent model source each. They therefore demonstrate direct historical evidence, archive/conversation flows, and bounded refusal, but intentionally show zero repeated public-orientation structures. Their nearby-question label now states that more orientation material is required. A live call to an external dialogue-model provider was not verified in browser acceptance because no provider was configured; provider adapters and authority gates were exercised with controlled integration tests.
