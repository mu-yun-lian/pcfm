# Prospective Single-Person Pilot v1

## 1. Operational claim

For one declared person, one binary-choice domain, one feature schema, and one
prospective collection window, compare four probability forecasts that are
committed before outcomes exist:

- `pcfm_person_model` (the primary candidate);
- `population_model`;
- `constant_history`;
- `profile_llm`.

If the primary forecast passes every fixed gate, the experiment supports only:

> On this registered question set and time window, the committed PCFM
> probabilities predict this person's recorded choices better than each
> committed baseline under negative log-likelihood.

The observable input is a signed forecast plan, an independently signed
registry receipt, and a signed human-outcome ledger. The output is a
recomputable score report with `passed_prospective_pilot`,
`completed_no_support`, or a refusal error.

## 2. Non-claims

This experiment does not identify or establish:

- the person's thoughts, beliefs, values, consciousness, or causal mechanism;
- open-domain agreement with the person;
- transfer to another person, domain, option set, context, or time window;
- factual completeness of an upstream study registry;
- independence of organizations merely because different signing keys exist;
- superiority to an LLM configuration that was not committed in the plan.

## 3. Alternatives

The design must remain capable of supporting these alternatives:

1. the population model is sufficient;
2. the person's historical base rate is sufficient;
3. a profile-conditioned LLM is at least as predictive as PCFM;
4. any apparent gain is chance variation;
5. predictions or questions were changed after outcomes were known;
6. the question set is outside the fitted domain or feature schema;
7. a temporary state change makes the committed person model stale.

## 4. Evidence roles

Each datum has exactly one role:

| Role | Artifact | May affect |
|---|---|---|
| prior fitting | model artifacts referenced by forecasts | forecast probabilities only |
| question design | scenarios in the signed plan | registered test set only |
| forecast commitment | four probability vectors in the signed plan | no later tuning |
| external registration | independent receipt over the plan ID | chronology gate only |
| sealed confirmation | future human outcome ledger | final scores only |

Confirmation outcomes may not change questions, forecasts, thresholds,
candidate methods, stopping, or sample count. The plan binds exact scenario
content, order-independent identities, all probability vectors, model
references, collection bounds, and analysis gates.

## 5. Fixed hard floors

- exactly one target person and one domain;
- exactly four method kinds:
  `pcfm_person_model`, `population_model`, `constant_history`, `profile_llm`;
- exactly one forecast for each method kind;
- at least 100 registered scenarios and exactly one future outcome per scenario;
- binary options and one shared named feature schema;
- every scenario binds its exact non-empty `question_text`;
- every forecast binds a SHA-256 digest of the model, prompt, profile, or
  history artifact used to produce it;
- plan signer, registry signer, and outcome signer are three distinct roles;
- registry time is not earlier than plan creation;
- every outcome time is strictly later than registry time;
- every outcome has `human_record` provenance;
- primary NLL no greater than `0.65`;
- primary expected calibration error no greater than `0.15`;
- against every baseline, mean paired NLL gain at least `0.01`;
- against every baseline, a two-sided 95% Newey–West interval lower bound
  strictly greater than zero;
- scenario content with the same metadata and maximum absolute feature
  distance no greater than `1e-6` is treated as a relabelled near duplicate;
- outcome timestamps must be unique so the dependence correction has a
  defined temporal order.

Configuration may make the NLL, calibration, or uplift gates stricter, never
weaker. The minimum sample count, method set, confidence level, chronology, and
role separation are not configurable.

## 6. Deployed/scored computation

The plan contains the probabilities that are later scored. The scoring path
uses `evaluate_probability_array` for NLL, Brier score, accuracy, and ECE. The
paired comparison uses per-question binary log loss:

`baseline_loss - primary_loss`

and reports its mean and a fixed Newey–West/Bartlett standard error with
`floor(n^(1/3))` lags in outcome-time order, followed by
`mean ± 1.96 × standard_error`. No model is fitted, selected, or updated during
scoring.

The report identity binds the verified plan, registry receipt, exact outcome
ledger, all metrics, every gate result, and final status. Verification
recomputes the report from those raw artifacts.

## 7. Scope and refusal

The scorer refuses rather than scores when signatures, identities, roles,
chronology, scenario content, feature schema, domain, options, provenance, or
outcome completeness do not match. A valid experiment whose metrics fail is
`completed_no_support`; this is not a software error.

The code cannot establish that all trials were disclosed or that the three
signing roles belong to independent real-world parties. A real confirmatory
study therefore needs an external append-only registry, independently held
keys, and trusted timestamps. Without those external controls, a passing local
run remains `implemented_exploratory`.
