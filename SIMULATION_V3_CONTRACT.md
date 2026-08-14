# PCFM Simulation V3 contract

Status: `implemented_exploratory`. The independent public conversation path and
its failure paths are now connected. V2 remains a frozen comparison baseline
and is not an input to V3 fitting or prediction. This status is not a claim of
real-person prediction accuracy.

## Observable input

- reviewed raw sources, including exact Q/A turns or attributable narrative
  spans, source lineage, speaker, role, audience, occasion and time metadata;
- current user message plus bounded prior dialogue context;
- an optional external-model query/event proposal whose every populated field
  cites an exact source span and which has no authority to confirm evidence;
- a selected dialogue model used only for disclosed general knowledge and
  surface wording after V3 freezes the person-specific content.

## Deployed output

One versioned result containing:

- the parsed decision/conflict frame of the current question;
- the selected direct event, similar event frames or public preference
  structure;
- predicted public priority, accepted cost, conditions, exceptions and
  counterevidence;
- demonstrated-knowledge references separated from external general knowledge;
- applicability, evidence support, uncalibrated status and refusal/degradation
  reasons;
- a frozen content contract consumed by the independent style renderer.

## Operational claim

Given fixed reviewed sources and dialogue context, V3 deterministically maps a
new question to an observable decision/conflict frame and selects either exact
historical response evidence, a matching public preference structure, or no
person-specific prediction. A preference structure is a repeated observable
relation of the form: under recorded conditions, the person publicly
prioritized one interest/action over another and accepted a recorded cost.

## Explicit non-claims

- The structure is not the person's private, sincere or complete value system.
- A public statement does not prove factual truth, motive or later action.
- Text similarity, evidence strength or fluent rendering is not prediction
  accuracy.
- Missing actors, options, conditions, time or costs are `unknown`; neither V3
  nor a language model may fill them as facts.
- Single events are candidates, not stable or cross-domain preferences.
- Software tests and synthetic cases do not establish real-person validity.

## Alternatives and null

- Null: V3 does not outperform direct retrieval, V2, person-history frequency
  or population baselines on strictly later real responses.
- Alternative 1: apparent value transfer is only topic overlap.
- Alternative 2: apparent person specificity is a generic language-model prior.
- Alternative 3: repeated events are duplicated reporting from one source.

## Evidence roles and gates

- `model_source`: construct event frames and exploratory preference structures.
- `applicability_reference`: calibrate matching only, never fit preferences.
- `final_holdout`: strictly later sealed evaluation only.
- `reference_only`: reality comparison only until explicit user selection and
  validation creates a new version.
- Missing time blocks fitting, change claims, holdout and optimization.
- Cross-domain preference use requires at least two domains and two independent
  source lineages. Same-domain repeated use requires at least two event frames;
  otherwise only direct historical evidence is allowed.

## Independent deployed computation

`pcfm.simulation_v3.SimulationKernelV3.predict` is the only V3 selection and
prediction kernel. It consumes only a `pcfm-simulation-model-v3` artifact built
by `SimulationKernelV3.fit` from reviewed raw sources. V2 artifacts are rejected.
