# PCFM Response Kernel v2 contract

Status: `implemented_exploratory` only after the public conversation path and
failure path pass. This contract does not claim that a real person's response
has been predicted accurately.

## Observable input

- current user message and the last bounded conversation turns;
- resolved topic, references, relationship, occasion and time scope;
- one version-bound set of reviewed direct-person response events;
- event atoms, event-type links, same-domain conditional tendencies, aggregated
  overall public tendencies, and publicly demonstrated knowledge claims;
- PCFM response-head state and active component manifest;
- zero or more model-proposed plans whose every content item cites an allowed
  event and content-unit identifier;
- the selected dialogue model reference and immutable configuration snapshot.

## Deployed output

One `pcfm-structured-response-v2` result containing response act and stance
distributions, selected claim/reason/memory/uncertainty IDs, answer status,
scope, evidence IDs, confidence status, refusal reasons, the frozen rendering
contract, active PCFM components, person model version, and model-call trace.

Answer status is exactly one of: `ordinary_dialogue`, `direct_answer`,
`composite_answer`, `partial_answer`, `tendency_answer`, `general_assisted`,
`clarification_needed`, or `refused`.

## Operational claim

Given a fixed person artifact, fixed reviewed evidence, fixed dialogue context,
and fixed candidate plans, the kernel deterministically chooses one of four
content paths: similar reviewed events, same-domain public tendency, aggregated
overall public tendency, or no person prediction. An LLM may supply disclosed
general knowledge and wording after that choice, but cannot choose or change the
person stance. Input order, irrelevant event IDs, and model aliases do not
change the numerical PCFM result.

## Non-claims

- A fluent answer is not evidence of real-person accuracy.
- A selected claim is not identified as a belief, value, motive or mechanism.
- A recurring tendency is an observed public-response pattern, not private
  psychology or proof that the person sincerely holds an inner value.
- Evidence strength is not calibrated prediction accuracy.
- A model-proposed plan is not truth and has no authority to add content.
- Retrieval support is not proof the person would answer the new question.
- Software tests do not establish calibration or correct-person uplift.

## Null and alternatives

- Null: person-specific evidence does not improve over population/history
  baselines.
- Alternative 1: apparent performance is nearest-neighbour copying.
- Alternative 2: apparent performance is topic overlap rather than person
  specificity.
- Alternative 3: language-model prior knowledge, rather than supplied evidence,
  produced the answer.

The last alternative is controlled by an evidence-ID allow-list and a frozen
content gate. The first three require independent time-held-out evaluation and
remain unresolved in this MVP.

## Scope and graceful degradation

The kernel asks for clarification when a short follow-up cannot be resolved. If
there is no similar event, it uses a same-domain tendency; if none exists, it
uses an aggregated overall tendency at lower confidence. If no person artifact
exists, an explicitly selected LLM may answer with general knowledge while the
result is marked `not_available` as a person prediction. Ordinary greetings,
thanks and conversational continuations use content-free dialogue templates.
Model failure is explicit and never silently routed to another provider.

## Material and optimization boundary

Raw articles, transcripts, subtitles and other text do not need to be supplied
as Q/A pairs. Single-speaker sources may be segmented into response-event
candidates; mixed-speaker sources require per-candidate speaker and verbatim
span confirmation. Reality comparison is opt-in and searches confirmed event
records outside the active version. Only the user-selected event, never every
event in its source, can enter an optimization candidate, and a sealed holdout
cannot be used for training.
