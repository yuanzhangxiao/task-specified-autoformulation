# Returned 23-model fitting campaign: numerical recovery before model revision

30 October 2026. This is Orion's review of the corrected Delta campaign. All
23 tasks returned terminal records, and rebuilding the report from authenticated
operation receipts reproduces the supplied summary exactly. Operational
completion does not establish optimization convergence. No new fitting, remote
job, validation evaluation, test access, or LLM call was performed in this review.

## Evidence and scope

- Source commit: `653a9a58fb173a77489b3e29f4b0a950607a63df`.
- Delta root: `/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2`.
- Slurm run array: `23347968`; report: `23347972`.
- Returned archive SHA256:
  `3cb76c175db86900a8f9133c6b4011b1c1abafe0ac7d98c5427cfc34745133cc`.
- Plan identity:
  `8498d0d08631ed81543ef1f4a96eb3471b1017f295a06ccb72da7bbd02961725`.
- Local generated evidence: `output/construction-portfolio-review-20261030/`;
  `audit-replay.json` records exact report reconstruction.

These are 23 model/domain alternatives for one T2-easy parent, not 23 benchmark
cells. The data contain 16 training trajectories. The prefix is eight time units
(nine samples); scoring follows the cutoff. Shared parameters were fitted using
the full training trajectories, so even fresh-prefix forecasts here are training
diagnostics, not independent validation. Matching validation data were not bundled.

## What worked

The launcher correction worked. All 15 certified-affine models entered shared
parameter fitting and retained independently verified original-equation endpoints.
The largest endpoint prediction discrepancy against Radau is `1.87e-9` in the
normalized residual coordinates. Their reported reconstruction errors are
supported by the saved independent replay receipts.

| Candidate | Initial objective | Retained training NMSE | Complete prefix forecast NMSE |
| --- | ---: | ---: | ---: |
| 12 | 63.95982 | 0.564509 | unavailable |
| 15 | 58.98569 | 0.564608 | unavailable |
| 21 | 156.3189 | 1.552123 | 2.824981 |
| 08 | 36.88289 | 1.587021 | 5.698031 |
| 06 | 65.62889 | 1.640191 | unavailable |

Candidate 21's initial objective is from its affine derivative qualification;
the initial independent Radau replay timed out. Its final retained endpoint did
pass independent replay. Candidates 12 and 15 both include an additional meal
absorption state. This is a promising association, not a controlled attribution
of their improvement to that state or a scientific-validity certificate.

Eleven affine searches exhausted their allowance; four (12, 15, 19, 21) returned
the optimizer's step/gradient stopping condition. All 15 retain
`optimization_unresolved`. No model is certified as sufficiently optimized to
attribute its residual error to its equation family.

## Why the best models lack forecasts

Training uses certified affine propagation. The subsequent prefix inference and
forecast functions still use the generic ODE route, including explicit DOP853.
This route inconsistency wastes the advantage of the affine certificate.

There are 256 attempted prefix inferences for the 16 retained endpoints:

- 180 infer and independently verify roots successfully.
- 64 fail on the first sensitivity evaluation at the 30-second allowance:
  all 16 trajectories for each of candidates 05, 12, 15, and 20.
- 12 fail the independent-integrator agreement check: candidate 00 has seven,
  11 has four, and 22 has one. These are numerical discrepancies, not proof that
  no initial condition fits the prefix.

Of the 180 successful inferences, 32 subsequent forecasts time out (all rows of
06 and 09), and ten fail integrator agreement (00: six; 11: three; 16: one).
Thus 138 individual forecasts are available, but only six models have all 16
rows needed for a complete-model aggregate. Candidate 10 is one of these six,
but its shared parameters were never optimized.

Static evaluation of the saved affine matrices confirms extreme stiffness in
12 and 15. Their fastest decay rates are approximately `2.08e6` and `9.15e7`
inverse time units, alongside a slow absorption rate near `0.0033`. This was a
small matrix calculation, not another trajectory integration. The generated
`static-affine-timescales.json` records the matrices and eigenvalues. Such rate
separation explains why the explicit prefix solver can stall despite successful
affine training and successful implicit original-equation verification.

Large cancelling coefficients are also present. In candidate 12 two coefficients
of the same meal input are `-629312.731962` and `+629312.683294`, with net sum
about `-0.04866845`. The earlier product/unused-parameter reductions are valid,
but do not remove every additive redundancy. Any further reduction needs an
exact equation, feasible-domain, and lift certificate; it must preserve the
construction model and all affected readouts/initializers.

## Why eight nonlinear models did not enter fitting

| Candidates | Established cause at the supplied start |
| --- | --- |
| 01, 02, 04, 13, 17 | First DOP853 residual integration reports required step size smaller than floating-point spacing. No Jacobian or shared search was reached. |
| 03 | First residual integration times out at 180 seconds, both in verification and qualification. |
| 10 | Initial replay verifies; all five completed derivative directions pass, but qualification times out before covering all 29 coordinates. |
| 14 | State-only/sensitivity residual discrepancy `1.395612e-7` and independent-solver discrepancy `1.395599e-7` exceed the `1e-7` policy. |

The saved `methods` list describes planned methods, not successful execution.
For the first six blocked models, the first DOP853 failure prevented the second
Radau call. These results therefore do not establish that an implicit solver
would also fail, or that the whole parameterized family is unusable.

Some failed models contain state-dependent denominators, but candidate 17 has
polynomial feedback instead. A pole, runaway dynamics, and stiffness must be
distinguished through diagnostics. The integration layer already constructs
trajectory/time/state failure evidence; the portfolio's common invalidation
handler currently keeps only exception type and message, losing that detail.

Candidate 10 retains its supplied start, with reconstruction NMSE `128915.84`.
It must not be described as a completed poor fit. Its recorded prefix optimizer
calls are not shared-parameter search. Candidates 10 and 14 have the same
equations but different parameter domains and guesses, so their outcomes are
not a matched solver comparison.

## Diagnostic bookkeeping issue

The four normally stopped searches omit `budget_exhausted`. The diagnosis wrapper
uses `get("budget_exhausted", True)`, adding
`search_incomplete_or_budget_status_unknown` even though their search receipts
record normal termination. A successful search must explicitly report its budget
state; missing information must remain unknown elsewhere.

Removing this spurious reason alone would not certify convergence:

- Candidate 12: projected gradient `2.87e-6`; uncertainty upper bound `1.99e-4`.
- Candidate 15: nominal gradient `1.49e-9`, but uncertainty upper bound `7.82e-4`;
  its retained best point differs slightly from the last accepted point, so that
  point's history cannot be borrowed without an endpoint-specific check.
- Candidate 19 has unresolved derivative evidence; candidate 21 retains gradient
  and history concerns.

The strict stationarity tolerance is `1e-8`. Step-based optimizer termination,
accurate integration, parameter uniqueness, and scientific correctness remain
separate claims. Do not weaken precision checks to force admission.

## Recommended next milestone

First repair numerical route consistency and diagnostic accounting, using the
saved endpoints rather than restarting all 23 shared fits:

1. Extend certified affine propagation into fresh prefix-root inference and
   suffix forecasting. Preserve composite readouts, the prefix-only information
   boundary, frozen shared coefficients, declared root domains, and independent
   original-equation verification. Where the prefix residual is certified affine
   in roots, bounded least squares can replace general nonlinear root search.
2. Record explicit budget termination and preserve structured rollout failures.
   Qualify representative best endpoints and failed prefixes before a broad run.
3. Recover nonlinear qualification in a bounded, separately journaled diagnostic:
   alternate stiff-solver replay for failed starts, tighter precision for 14,
   and resumable coordinate coverage for 10. Do not relax thresholds, modify
   equations, or infer a structural failure from one invalid start.

After those checks, consider additional exact additive-coordinate reductions,
profiling and matched-budget continuation of the useful endpoints. Any new
allowance is a new documented continuation; old spent allowances stay spent.
Function-form revision feedback is still premature from these fits alone.

Recorded operation costs total about 5.68 CPU-hours and 7.26 summed task-wall
hours. These include failed numerical work and synchronous checkpoints, but
exclude queue wait, source/runtime verification, reporting and adapter rebuilds
on partial resume. Summed task-wall time is not campaign elapsed time.

## Review verification

The authenticated report reconstructed exactly. All 91 focused intake, adapter,
campaign, and launcher tests passed, including the small synthetic numerical
smoke. Repository-wide Ruff still reports the same 37 unrelated findings under
`analysis/`; this review changes documentation only. No full-suite pass is
claimed. Historical result files, numerical implementation, and campaign
allowances remain unchanged.
