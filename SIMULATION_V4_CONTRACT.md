# Simulation V4 contract

## Operational claim

Given reviewed public response spans for one person, V4 can return one of four auditable results:

1. a direct historical response;
2. related historical evidence without a new stance;
3. a conditional direction supported by the same reviewed tradeoff in at least two independent source lineages;
4. no person-specific prediction, while permitting a disclosed general-knowledge answer.

The result concerns observable public response patterns. It does not identify private values, sincere beliefs, causal mental mechanisms, complete knowledge, or calibrated prediction accuracy.

## Authority boundary

- A material model may propose event boundaries, closed domain and interest IDs, exact condition/reason/knowledge spans, and tradeoffs. Nothing enters fitting until the source, speaker, response span and semantic candidate are confirmed.
- A dialogue model may propose closed query fields and select supplied event or structure IDs. Domain and option semantics require exact message spans; a structure cannot produce a direction unless both sides of its tradeoff are grounded in the query. V4 rejects invented or ungrounded message IDs, event IDs, structure IDs, domains, interest IDs and spans.
- Code alone enforces source roles, lineage deduplication, repeated-evidence floors, counterevidence, role transfer, time direction, artifact identity and final direction.
- For a person projection, a dialogue model returns only structured external briefing plus allow-listed demonstrated-knowledge IDs. Code assembles the reply, rejects person attribution and unsupported numbers, and labels the briefing as model-generated rather than person evidence. Style rendering occurs only after this content contract is frozen.

## Hard gates

- Raw Q&A is direct evidence only; it cannot create a preference atom.
- Pending or ungrounded model candidates cannot create a preference atom.
- Preference projection requires at least two independent source lineages.
- Same-domain transfer requires matching domain support; cross-domain use requires a cross-domain structure.
- Private-role transfer and explicit questions later than the evidence window cannot produce a person projection.
- Missing or invalid ISO response dates cannot create preference atoms. Relative future questions such as "next year" are treated as later than the evidence window.
- Unresolved counterevidence blocks the projection.
- Final holdout events must be confirmed, strictly later than training, and source-ID disjoint.
- Reality optimization recomputes the selected event from raw source material and promotes only that event. Derived-event tampering, whole-source promotion and V4 holdout regression are rejected.

## Compatibility

`SimulationKernelV4` is the sole production conversation kernel. V3 and Response Kernel V2 remain frozen baselines and are not active components. Existing stored V3 artifacts are refit from reviewed version sources; V4 refuses to reinterpret an old artifact as V4.

## Known limits

The closed interest taxonomy is intentionally small and reviewable. It should be extended only from real rejected candidates. Semantic review can bound fabricated spans but cannot prove that a human or language model assigned the correct meaning. General-model background can still contain ordinary non-numeric factual errors and is therefore disclosed as unverified external briefing. Real-person accuracy remains `not_assessed` until an adequate independent temporal holdout exists.
