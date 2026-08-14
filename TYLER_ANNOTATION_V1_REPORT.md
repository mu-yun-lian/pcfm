# Tyler Cowen Dual-Annotation v1 — Implementation Report

## Outcome

The dual-annotation workflow is implemented and tested as
`implemented_exploratory`.

It creates two blind packets from the verified Tyler source artifact, requires
complete separately signed submissions, validates evidence-unit citations,
calculates exact full-label agreement and Cohen disposition kappa, exposes
disagreements to a separate adjudicator, and refuses finalization below fixed
reliability floors.

It is not `implemented_confirmatory` because no real independent human
annotations have been collected.

## Real packet audit

Source:
`artifacts/tyler_source_v1/tyler-cowen-rss-2026-07-31.json`

- selected candidates: 9
- included authored-prose evidence units: 15
- A/B candidate-set equality: passed
- A/B item order differs: yes
- labels or model predictions in either packet: none
- evidence roles: 9 `retrospective_diagnostic`
- training-eligible records before annotation: 0
- training-eligible records even under unanimous in-scope test labels: 0
- candidate-set digest:
  `5293b0407eebee4b97f3fa68075b50a26c0fec165229a29281898e3e3b70f235`
- codebook digest:
  `696186478ddfdc284a77409a9a4eadf8ba5ed5a1c6b8165d52e14e3757734af8`
- packet A digest:
  `0a6ab564e4fae917e47de51abc8c07b3e8c5a8a454ceef8ca7340ce4a452884c`
- packet B digest:
  `dce1fea7c676e7a41c149f5dd6e75bec7262ce3aedc4eef8fea33bf00c452e32`

## Fixed gates

- at least 5 candidates;
- two distinct HMAC verifier IDs;
- exact packet and slot lineage;
- 100% candidate coverage;
- at least one valid evidence-unit citation per label;
- axis directions only for `clear_in_scope_stance`;
- exact full-label agreement at least 2/3;
- Cohen disposition kappa at least 0.40;
- every disagreement resolved;
- adjudicator identity distinct from both annotators;
- agreed labels cannot be changed by adjudication or by re-signing;
- post-2026-07-31 records refused without a separate prospective
  registration.

There is no configuration or CLI argument that weakens these floors.

## Hostile audit repairs

Two integration risks were found after the ordinary tests first passed:

1. A re-signed dataset could have changed a previously agreed label unless the
   verifier compared final records with the adjudication packet. A regression
   test now proves this is refused.
2. Two mutually consistent A/B packets could both have been fabricated unless
   the adjudication entry point recomputed them from the Tyler source artifact.
   The source artifact is now a mandatory adjudication input.

## Verification

- focused annotation tests: 13/13 passed;
- Python compilation: passed;
- full PCFM regression: 223/223 passed;
- full regression duration: 191.025 seconds;
- full regression log:
  `artifacts/full_regression_tyler_annotation_v1.err.log`.

The cognitive-module completion checker correctly returns exit 1 with only:

`completion gate requires module.status implemented_confirmatory`

This is expected. Changing the status to confirmatory before real independent
annotation would be claim inflation.

## Remaining limitations

1. The software cannot prove the two humans were physically isolated or did
   not communicate.
2. Agreement can reflect shared bias, shared context loss, or label
   prevalence.
3. Inline quotations and paraphrases remain a human attribution problem.
4. The five axes may omit the relevant stance.
5. Public writing may be reporting, rhetoric, or role behavior rather than a
   private decision rule.
6. HMAC keys authenticate supplied identities but do not prove that one person
   did not control multiple identities.
7. The current packets are retrospective diagnostic material and cannot train
   the person model.

## Required next external step

Assign packet A and packet B to two independent human annotators. Give each
annotator only their own packet and blank submission template, issue distinct
verification keys, and prevent access to the other submission and all model
outputs. Do not revise the codebook after viewing these nine labels.
