# Model-15 meal continuation: implementation handoff

3 November 2026. Implements the next milestone recommended in
`2026-11-03-construction-portfolio-targeted-results.md`.
Run commands and retention rules are in
`docs/PHASE_C_CONSTRUCTION_MEAL_CONTINUATION.md`.

## What changes

Root-profile optimization no longer treats a raw Powell success flag as a
qualified terminal point. It retains a bounded ledger of evaluated shared
coordinates and their scalar objective receipts. The terminal point must be
finite, domain-valid and paired exactly with a completed feasible evaluation.
Its reported objective must equal that receipt. The check does not perform an
extra fit or extend a spent allowance. A later rejection at the same point
supersedes its earlier evaluation for termination qualification.

`optimizer_raw_success` preserves the optimizer's own flag;
`optimizer_qualified_success` in the new campaign includes terminal qualification;
`optimizer_feasible_termination` reports terminal feasibility independently of raw
success. A feasible but unsuccessful optimizer return is therefore represented
without conflating these facts. Missing terminal evidence stays unknown.
Neither qualified success nor exact conditional root fitting establishes outer
stationarity, uniqueness, parameter confidence, or scientific adequacy.

Callbacks without a valid paired evaluation are logged in a bounded audit and
excluded from accepted numerical histories. Nonfinite numbers become explicit
missing diagnostic values rather than nonstandard JSON infinities. Regardless
of the optimizer's return, the best finite checkpoint survives for independent
original-equation verification.

## Frozen campaign

The `construction-meal-continuation-1` campaign authenticates the returned
targeted campaign and selects only its reviewed
`candidate-15-saved-meal_flux` endpoint, NMSE **0.4266471542730798**. The source
plan and compact projected record are pinned by content hashes. It does not
reselect among models by new outcomes or copy the entire historical input chain.

Six tasks start from that endpoint:

1. Free positive meal-flux continuation.
2. Fixed rate 0.001/T.
3. Fixed rate 0.01/T.
4. Fixed rate 0.1/T.
5. Fixed rate 1/T.
6. Exact original zero-rate control.

T is the maximum public training duration (300). Positive starts preserve
g=k*q and every B0=k*A0 while changing k. Fixed positive k is removed from the
outer search, while the effective gain remains free. Rates are diagnostic
conditions, not added physical parameter bounds. The ODE and the original-root
profiler remain unchanged; every lift must recover finite original parameters.

The zero branch has its own algebraic certificate. The recognized exclusive-use
reservoir has A'=q*r-k*A and its sole outward path is +k*A. At k=0, q and A0
cannot affect observed outputs, so q is held at zero and A0 is initialized at
zero as a representative. This preserves outputs under declared schedules, not
the hidden A trajectory. The inner root profiler still contains the flat meal
initial columns; they do not identify the hidden initial state. The certificate
also rejects direct use of the generated initializer root in the output balance.
Nonzero meal flux is never permitted at exact zero through a singular inverse.

Each task has 1,800 numerical seconds and 2,000 objective attempts, with separate
setup, qualification and verification reservations. Workers request one CPU,
8 GB and a four-hour ceiling, at most two concurrently; the dependent report
has thirty minutes. An interrupted operation cannot receive a renewed allowance.
Every fixed-rate trial score remains distinct from the best retained score.
Failed or worse trials cannot erase the inherited verified endpoint.

## Boundaries and remaining work

This is a training-reconstruction campaign with training-estimated initial roots.
It performs no model revision, LLM call, validation/test evaluation or benchmark
change. All global optimum, stationarity and parameter-confidence conclusions
remain unclaimed. Larger fixed-rate errors are only upper bounds on profile
minima until nuisance fitting is adequate.

The new run does not include models 10/14 or model 12. Their unresolved searches
remain separate follow-ups. Checkpoint persistence policy is unchanged: routine
progress is throttled and every strict incumbent improvement is forced durable.
Its overhead is still reported; reducing that overhead without losing useful
checkpoints is deferred.

No cluster submission is performed locally. The delivered source-only archive
is built from the committed tree, with `SOURCE_COMMIT` and per-file checksums.
Historical result roots and existing submitted code directories stay unchanged.

## Saved-endpoint preflight

All six actual model-15 controls passed compilation, qualification and independent
original-equation verification in a separate bounded local preflight. It used at
most three objective attempts per task, 20 numerical search seconds, 90 seconds
for qualification and 120 seconds for each verification. All searches stopped
at their evaluation cap and every terminal reconstructed exactly. The largest
independent solver prediction discrepancy was 1.165e-9. Source identity still
matched the completed implementation on re-verification.

| Control | Three-attempt trial NMSE | Retained NMSE |
| --- | ---: | ---: |
| Warm flux | 0.426647012240 | 0.426647012240 |
| 0.001/T | 0.426625857245 | 0.426625857245 |
| 0.01/T | 0.426458041693 | 0.426458041693 |
| 0.1/T | 0.426691865475 | 0.426647154273 |
| 1/T | 0.612254707232 | 0.426647154273 |
| Exact zero | 28.031344769664 | 0.426647154273 |

These numbers verify execution and incumbent protection, not completed nuisance
optimization, coefficient accuracy or generalization. In particular, the zero
control starts from the new best endpoint's other shared parameters and has only
three attempts; its error must not be treated as an optimized zero-rate minimum
or compared as such with the prior longer zero-rate search. Delta starts from
the pinned historical endpoint, not these local smoke endpoints.

Evidence: `output/meal-continuation-preflight-_w_hus6k/smoke.json` and the sealed
operation receipts beside it. No historical receipts were altered.

## Verification

The final combined regression suite passes **928 tests**, covering portfolio
fitting and continuations, reductions/charts, root profiling, residual searches,
checkpoint retention, receipt reconstruction and scheduler launchers. An earlier
run encountered two source-identity guards while parallel implementation edits
were still finishing; the complete suite was rerun successfully after freezing
the source. The guards were preserved, not bypassed.

Changed Python files pass Ruff; repository-wide `ruff check .` reports the same
37 unrelated pre-existing findings under `analysis/claude`. Bash syntax, CLI
help, all six saved-endpoint preflights, source/runtime verification, and exact
receipt-derived reporting pass. Tests also explicitly cover false optimizer
success, invalid callbacks, failed source qualification, interruption without
budget renewal, separate trial/incumbent reporting, and output-equivalence
certificate rejection. No production benchmark data or prompts were modified.
