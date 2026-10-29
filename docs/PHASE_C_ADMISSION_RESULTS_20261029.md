# Returned admission-guard results: 29 October 2026

Both ACES jobs completed operationally. The frozen runtime admitted five 20B
records and six 120B records, all with deferred checks. Independent equation
review holds one additional 120B model back. After removing that record and
merging one cross-arm duplicate, nine new complete-model/domain identities
remain for fitting qualification. Admission does not establish scientific
correctness or numerical accuracy.

## Scope and provenance

- Source `034fa79`, profile `admission-guard`; frozen T2-easy parent `P00-r1`,
  seeds 0/1/2, five initial alternatives and at most four shallow follow-ups.
- 20B: job `2266860`, archive SHA-256
  `f738a71a6bbdd967d223f6d0e7cae3f8f38f84101f9db6e5328203e924e4742c`.
- 120B: job `2266861`, archive SHA-256
  `c160ea172cf28909fb7f56983d182c8ca867e254ebf4b842cf73c5c92df0abec`.
- All 27 responses stopped normally with known usage. No output truncation,
  provider failure or outer-schema parsing failure occurred. Both workers exited
  zero; 39 seals and the source/entrypoint identities verified.
- The 20B scheduler acknowledgment timed out, but its actual completed job and
  saved results verify. Do not resubmit solely because its acknowledgment is absent.
- Exact replay reproduces all 27 calls and 51 candidate outcomes. Input files,
  frozen decisions, benchmark data and public prompts remain unchanged.
- Analysis performed no generation, optimization, rollouts or test-data reads.

## Results under a common admission check

The previous rows below are independent reassessments of actual v6 saved
patches under the current gate. Historical generation, routing and costs remain
as recorded; these rows are not simulated reruns.

| Arm | Eligible seeds | Initial eligible | Follow-up eligible | Calls | Tokens/episode: median [unscaled MAD] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous 20B, corrected gate | 3/3 | 8/15 | 0/7 | 10 | 58,341 [5,632] |
| New 20B | 2/3 | 2/15 | 3/12 | 15 | 98,794 [1,050] |
| Previous 120B, corrected gate | 3/3 | 1/15 | 5/11 | 14 | 92,476 [481] |
| New 120B | 3/3 | 6/15 | 0/9 | 12 | 74,211 [270] |

20B used 287,770 observed tokens; 120B used 222,378. The 120B arm had three
failed initial branches per seed, each repaired once, so four calls per seed
were sufficient to exhaust its shallow policy. It did not time out or lose a call.

20B now obtains three eligible models through repair, but has lower initial
coverage. 120B has better initial coverage and no eligible follow-up. Three
seeds on one parent do not establish model-size superiority or general
construction reliability. Every admitted record still has deferred checks.

Initial requests differ from v6 by exactly two fields: the general initialization
reminder and the new proof-policy label, together 99 input tokens. The system,
schemas, public task, model view and sampling settings otherwise match. The
reminder's causal effect is not isolated from stochastic generation and changed
feedback/gate behavior. The new initializer-owner and conflicting-removal cards
were not exercised by selected 20B branches.

## What the new guard actually prevented

Reassessing all 51 actual saved patches with only `admission_guard=False`
changes exactly two outcomes: 20B seed2 initial option2 and its call4 repair
would become eligible. The enabled guard correctly rejects both.

Both retain `V * meal / (K + meal)` with nonnegative `K`. On the certified
zero-meal interval, `K=0` makes the literal quotient undefined and `K>0` leaves
the shared affine dynamics with the demonstrated trajectory contradiction. The
call4 addition `X_U` is not connected to an existing equation, so it cannot
remove that conflict. The remaining 49 statuses/codes do not change.

The new 120B input-only meal saturations use strictly positive denominator parameters;
none exercises this additional domain split. This is a successful but bounded
admission correction, not universal model-data compatibility coverage.

## Equation audit

Three admitted records add a genuine source path into glucose at zero current
meal: 20B seed1 initial option2 and 120B seed0 option4 / seed2 option1. For example:

```text
R' = meal - k R
Gp' = C meal - U + k R
R(0): trajectory-specific nonnegative initial value
```

This can supply glucose while current input is zero. It does not prove a correct
physiological decomposition; inherited direct meal pathways and parameter sharing
remain in the literal model.

Eight of eleven recorded eligible models instead retain `Gp'=-U` at zero meal.
If `U` stays nonnegative throughout an interval, glucose cannot increase there.
Positive sampled values alone do not establish the between-sample integral, so
this audit retains a conditional balance concern rather than inventing an
unconditional rejection or an NMSE lower bound.

### One new missed exact-reproduction conflict

120B seed0 initial option1 removes the old insulin memory and declares:

```text
U = Q + k Z
Z' = (I - Z) / tau
Q' = c
Q(0) = recorded Uii(0), k >= 0, tau > 0
Z(0): free trajectory-specific nonnegative value
```

For `k=0`, `U(0)=Uii(0)` already contradicts the recorded initial values. For
`k>0`, let `W=k Z`. Then `W'=(k I-W)/tau` and exact reproduction imposes
`W(0)=U(0)-Uii(0)`. The system in `(Gp,I,Q,W)` remains shared and affine.

The exact-rational initial relation among `train_000`, `train_013`, and
`train_014` must persist over the common zero-input interval `[0,25]`. Observed
`I(25)` violates it by `0.034823740686775274`, exceeding the frozen comparison
guard `5.134230958376486e-6`. The new output gain does not provide an extra
observable initial degree of freedom. The complete train-only certificate is
retained in `audit-120b-scaled-memory-witness.json` in the generated report.

This is an exact-reproduction obstruction, not a bound on approximate fitting
error. The historical runtime status stays unchanged; the analysis intake holds
this record separately. None of the earlier 14 retained candidates has this
scaled-replacement structure: their relevant differences introduce an unpinned
initial direction or true state nonlinearity. This specific comparison does
not certify the earlier portfolio scientifically.

## Repair behavior

- 20B seed1 calls1/3 add missing new-state coefficients and become eligible.
- 20B seed2 call3 connects a previously isolated `X_U` to `U` and becomes eligible.
- 20B seed2 call1 omits an invalid edit whose component is `new_parameters`, but
  returns no explicit drop. Delta semantics retain the old edit. Its repeated
  failure is a cleanup mismatch, not merely a verbatim repeated answer.
- 120B Hill-form repairs fix `^` to `**`, then encounter the integer-exponent
  restriction. The earlier feedback did not disclose both grammar issues together.
- 120B seed2 repeats an edit-and-remove conflict despite specific feedback.
- Prose claims of nonlinearity or removed initializers are sometimes inconsistent
  with the actual patch; the typed equations determine the assessment.

## Fitting intake and next milestone

The new run contributes ten retained records / nine normalized identities;
one insulin saturation model is duplicated across arms. Together with the
previous 14, there are 23 saved intake identities and no exact hash overlap.
These are candidates for fitting-interface qualification, not completed fits
or distinct scientific mechanisms.

The generated `FITTING_HANDOFF.md` and `fitting-intake.json` preserve full models,
initializer rules, parameter domains, unresolved checks and provenance. Numerical
failures must remain separate from equation inadequacy. No optimization or
NMSE-based selection occurred in this analysis.

Recommended next: an offline, benchmark-independent scaled-readout proof
extension with saved replay, plus compact explicit cleanup/exponent feedback.
Then qualify this finite portfolio with fitting and assess multi-benchmark
coverage, instead of interpreting another prompt-only batch on one parent as
construction readiness. The current run does not warrant an immediate new GPU
submission.

Local report:
`output/construction-admission-guard-results-20261029/REPORT.html`.
It includes plots, all 51 candidate audits, all 27 exact requests/responses,
the guard-only comparison, proof certificate, fitting intake and source hashes.

Verification: 230 focused tests passed; 51 outcomes replayed exactly; all 362
report links resolved. Generated report/packet checks passed. Repository-wide
`ruff check .` still reports 37 existing findings in `analysis/claude/`.
No production code or prompt behavior changed in this analysis milestone.
