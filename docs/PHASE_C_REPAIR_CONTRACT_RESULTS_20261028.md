# Returned repair-contract confirmation: 28 October 2026

Both ACES jobs completed. All three episodes per model size found at least one
candidate eligible under the frozen runtime gate. This is progress in generating
alternatives, not a finding of scientific correctness or numerical accuracy.

## Scope and provenance

- Source: `4b97729`; profile `repair-contract`; frozen T2-easy parent `P00-r1`;
  seeds 0, 1, 2; five initial alternatives and one shallow follow-up per selected
  failed initial branch, at most fifteen calls per arm.
- 20B job `2264236`; uploaded `inspection (1).tar.gz`, SHA-256
  `c4a7abcfdc1be20191c5e38456aa97be8d7cd9a7716b85fca756344fd0c382f3`.
- 120B job `2264237`; uploaded `inspection (2).tar.gz`, SHA-256
  `a56c5ee9b6ab2e87f0c1d468bf68eb1823fe0e1011325aa33c028f46f61bd9ea`.
- All 24 provider calls stopped normally with recorded usage and valid outer
  response schemas. No output truncation or provider failure was found.
- Independent operational checks passed: 175/175 for 20B, 224/224 for 120B.
  Replaying all 48 candidate records reproduced every recorded outcome exactly.
- Analysis made no LLM calls, optimizer calls, solver rollouts or test-data reads.
  Historical inputs, decisions, prompts and production gates are unchanged.

## Outcomes

| Run | Eligible episodes | Initial eligible | Follow-up eligible | Repeated failed follow-ups | Calls | Eligible complete-model hashes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous 20B action feedback | 2/3 | 2/15 | 1/12 | 0/12 | 15 | 2 |
| New 20B repair contract | 3/3 | 8/15 | 0/7 | 1/7 | 10 | 8 |
| Previous 120B focused | 2/3 | 4/15 | 2/6 | 2/6 | 9 | 6 |
| New 120B repair contract | 3/3 | 4/15 | 5/11 | 0/11 | 14 | 8 |

20B used 190,875 tokens; its episode median was 58,341 with unscaled MAD 5,632.
120B used 259,475 tokens; its episode median was 92,476 with MAD 481. The nine
eligible 120B records contain eight normalized complete-model/domain identities.
These hashes do not count distinct scientific mechanisms. Every eligible record
retains deferred training-consistency checks.

20B improved primarily in its initial alternatives. 120B obtained five eligible
follow-up repairs, including fixed-exponent Hill laws and restoration of a
required public input path. Neither observation establishes superiority: there
is only one parent, subsequent repair tasks differ, and the initial schema and
system instructions changed from the previous experiments.

## Equation findings

Fifteen of seventeen eligible records still reduce to `Gp' = -U` at zero meal.
If disposal `U` stays nonnegative throughout such an interval, glucose cannot
increase. This is a scientific balance concern. Positive sampled values alone
do not prove the between-sample integral or supply an unconditional rejection.

Two 20B initial options add a meal reservoir supplying glucose after meal input
stops. For example, seed 0 option 2 has

```text
A' = k_abs_rate * meal - k_abs * A
Gp' = meal - U + k_abs * A
```

This is a genuine additional source path. The retained direct meal path and
free nonnegative reservoir initial still need scientific and numerical review.
No fitted improvement is claimed.

### Three missed exact-reproduction conflicts

All three 120B initial option-0 proposals replace the meal law by
`V * meal / (K + meal)`. This is nonlinear in the supplied input, but leaves
the state dynamics affine for common input. The current checker conservatively
defers because the declared parameter domain does not exclude `K = 0`.

An independent mathematical audit separates two cases:

1. `K = 0`: the literal rational expression is undefined at zero meal. There is
   no declared continuation that supplies a value for `0/0`.
2. `K != 0`: the law vanishes on the certified zero-meal interval, leaving a
   closed shared affine system whose trajectory identity contradicts the data.

The witness uses `train_000`, `train_013`, and `train_014` over `[0,25]`, where
both supplied inputs are zero at every complete interpolation knot. Exact
rational weights close to `(-1, 1/2, 1/2)` sum to zero and cancel all initial
coordinates `Gp`, `I`, `X`, and `Q`. Here `Q(0)=Uii(0)` and exact reproduction
of `U(0)=Q(0)+X(0)` forces `X(0)=U(0)-Uii(0)`; freely fitted latent initials
are not simply assumed equal.

The model therefore requires weighted `I(25)` to be zero, but the observed
combination is `0.034823740686775274`, above the frozen numerical comparison
guard `5.134230958376486e-6`. Two independent audits checked the source values,
weights, full input interval, declared equations and domain cases.

This proves an exact-reproduction obstruction under the stated candidate/data
contract. It does not bound approximate NMSE or show fitting is futile. The
historical gate statuses remain unchanged; the finding identifies a coverage
gap to address offline before another GPU run.

## What the feedback actually tested

- All thirty initial options use the required decision and option envelope;
  the previous routing failure did not recur.
- Two 20B follow-ups receive typed new-definition placement choices. Both add
  a new-process declaration, but only one performs the explicit cleanup;
  neither becomes eligible.
- The new existing-component movement examples and exact duplicate-action
  normalization were not exercised by the selected live branches.
- Two 120B repairs both add and drop the same `remove_states` action. Rejection
  is appropriate because intent is ambiguous. Feedback should show complete,
  mutually exclusive keep-state and remove-state alternatives.
- One 20B failed repair repeats. Its prose claims a zero initial value while
  its structured initializer remains trajectory-specific and nonnegative.
  The structured model, not the prose, determines the equations and checks.
- Recorded omitted actions are explicit drops or replacements; this audit
  found no demonstrated accidental loss of earlier corrections by the runtime.

## Recommended next milestone

Extend the qualified common-forcing proof with explicit domain case handling;
replay saved candidates before live testing. Make pending-action deletion versus
model removal feedback explicit, including initializer ownership and disagreement
between prose and typed edits. Retain all alternatives with their equation
audits for later fitting; do not equate a deferred check with a scientifically
adequate model or select by unqualified NMSE.

The local report is
`output/construction-repair-contract-results-20261028/REPORT.html`. It contains
plots, all 48 candidate equations/patches, all 24 complete requests/replies,
independent arm audits, exact replay evidence, archive hashes, and the complete
`meal-saturation-witness.json`. Generated artifacts remain uncommitted.

Verification: 118 focused replay/search/feedback tests passed; all 48 saved
candidates replayed exactly and all 328 report links resolved. An additional
full-suite run was interrupted after 1,205 passes and three skips; it was not
completed. `ruff check .` reports 37 existing findings under `analysis/claude/`;
this documentation-only change does not alter those files.
