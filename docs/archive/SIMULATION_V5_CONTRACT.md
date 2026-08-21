# Simulation V5 conversation-conditioned response contract

Status: `implemented_exploratory_accuracy_not_assessed`. The public conversation
path, failure path, artifact recomputation, and synthetic/adversarial conversation
tests pass. An independent time-separated real-person full-conversation holdout
is still missing, so software completion does not establish prediction accuracy.

## Observable input

- one immutable, version-bound set of reviewed public response episodes;
- each episode's exact trigger window, response span, source lineage, time,
  interlocutor, occasion, publicly observed response act, claims, reasons,
  uncertainty, demonstrated knowledge, and reviewed trade-off candidates;
- the current user message as a delta to a conversation state containing topic
  threads, referenced message IDs, prior commitments, unresolved questions,
  relationship, occasion, and time scope;
- an optional model-proposed semantic query plan restricted to real message,
  event, structure, domain, and interest IDs;
- an optional external-knowledge brief that has no authority over the person's
  response direction.

## Deployed output

One `pcfm-conversation-conditioned-prediction-v5` result containing:

- response act and stance;
- a frozen content plan with claims, reasons, uncertainty and allowed memories;
- the exact conversation messages and reviewed events used;
- the applicability route and refusal reasons;
- evidence-strength components, explicitly not an accuracy probability;
- a generation contract that permits natural first-person wording but forbids
  new biography, experiences, attributed facts, numbers, dates, or positions;
- the model/service snapshots and every rejected semantic-plan field.

## Operational claim

Given fixed reviewed episodes, a fixed conversation history, and a fixed
semantic query candidate, the V5 kernel deterministically selects one of:

1. exact historical response;
2. context-matched reviewed episode evidence;
3. repeated public orientation projected onto model-proposed scenario effects;
4. clarification because the conversation reference is ambiguous;
5. no person-specific projection.

The language model may resolve references, map a scenario to allow-listed
domains/interests, retrieve only supplied IDs, provide disclosed external
background, and realize a frozen response plan. It may not choose the person's
orientation, create evidence, update the person model, or silently replace a
failed model call.

## Non-claims

- Public response patterns do not identify private beliefs, sincerity, motives,
  or a complete mental model.
- Evidence strength and software confidence are not calibrated response
  accuracy.
- A semantic query mapping is not a fact merely because a model produced it.
- A fluent answer is not evidence that the real person would say it.
- Generated dialogue is conversation context only and never fitting evidence.

## Null and alternatives

- Null: full conversation state and person assets do not improve next-response
  prediction over the current message and a generic model.
- Alternative 1: apparent gains come from copying a nearby historical answer.
- Alternative 2: apparent gains come from topic overlap rather than the correct
  person.
- Alternative 3: apparent gains come from the dialogue model's prior knowledge.
- Alternative 4: a mechanically cut source span is mistaken for a response
  episode.

These alternatives require time- and lineage-separated full-conversation
holdouts plus current-message-only, wrong-person, history-shuffled, retrieval-
only, and generic-model baselines.

## Conversation state

The current message is never the whole scenario. It is applied as a delta to a
recomputable state with:

- `topic_threads`: anchored by real user message IDs, with last-touched order;
- `active_topic_id` and `resolved_message_ids`;
- `participant_roles`, `relationship`, `occasion`, and `time_scope`;
- `assistant_commitments`: prior generated claims used only for conversational
  consistency;
- `unresolved_questions` and explicit correction/supersession links;
- `context_digest` over the exact messages and state fields.

The raw history remains the authority. Summaries are caches and must be
recomputable. Model-supplied message IDs that are absent from the conversation
are rejected. An ambiguous reference returns clarification rather than using
the latest unrelated topic.

## Episode and source boundary

- Atomic means independently traceable, not context-free.
- A reviewed episode contains a trigger window and response span; arbitrary
  fixed-width text chunks are source-navigation aids, not response episodes.
- Confirmed Q/A may support an exact historical response. Unstructured or mixed-
  speaker material needs a verbatim, speaker-confirmed episode candidate before
  entering the person model.
- Missing time, interlocutor, occasion, or available-information fields remain
  explicit unknowns. Missing time blocks temporal aggregation but not source
  reference or an exact historical answer.
- Candidate discovery, parameter fitting, applicability calibration, temporal
  holdout, reality comparison, and post-deployment monitoring are disjoint roles.

## One deployed computation

`predict_v5(person_artifact, conversation_state, current_delta, query_plan)` is
the only selector used by conversation serving, evaluation, comparison, and
future optimization acceptance. Fit objectives and language generation are not
alternate prediction paths.

Projection requires repeated independent public orientation evidence. A query
model may propose scenario effects over allow-listed interests; code combines
those effects with the person's reviewed orientation and counterevidence. The
model cannot provide the final stance field. Context, role, time, domain, local
support, counterevidence, and source-lineage gates run before generation.

## Failure and degradation

- Ambiguous conversation reference: `clarification_needed`.
- No reviewed person structure: normal answer may still be produced by an
  explicitly selected model, but it is marked `not_available` as a person
  prediction.
- Model unavailable or invalid structured output: explicit error or a bounded
  deterministic person anchor; never switch providers silently.
- Content or style candidate changes the frozen stance, adds unsupported person
  facts, or adds protected numbers/dates: discard it and return the neutral
  frozen plan.

## Acceptance

Engineering acceptance requires focused and full regression, raw-evidence
recomputation, version/schema refusal, topic-switch/return and pronoun tests,
model-call transparency, and proof that generated dialogue never enters fitting.

Model acceptance remains separate. It requires held-out whole conversations
split by time and source lineage, with response-act, stance, claim/reason
support, context resolution, contradiction, calibration/coverage, correct-
person uplift, and semantic-preservation metrics. Until then the status is at
most `implemented_exploratory_accuracy_not_assessed`.
