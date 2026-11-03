# Construction portfolio residual campaign: results review

2 November 2026. Review of `construction-portfolio-residual-inspection.tar.gz`
from `~/Downloads`. This review changes no fitter, benchmark, sealed receipt or
campaign budget, and runs no new fitting or held-out evaluation.

Archive SHA256:
`7ab47c72a61c2ee0c32304ec7f4502776861cd9285b86b2b54dbc1737959cc68`.
Extracted evidence is in ignored
`output/construction-portfolio-residual-review-20261102`.
Campaign plan SHA256:
`299fd4f5a0861fe4966a95b71f8501651bc872a21c75c24aecd41e5d0d2d5673`.
Implementation commit: `8ddc1b32d2c9d324c732c16e533753c6dab82e99`.

## Verified numerical results

All four terminal tasks completed operationally. All four sealed terminals
reconstruct exactly from their operation receipts, including scientific fields.
All retained models pass independent original-equation replay. Normalized
prediction discrepancies between the two evaluations range from 1.95e-10 to
1.26e-9. No optimizer reported convergence, and no stationarity, parameter
confidence, global optimum or equation-family inadequacy is certified.

| Model | Previous training NMSE | Retained training NMSE | Relative reduction | Search objective attempts |
| --- | ---: | ---: | ---: | ---: |
| 10 | 1.749666619 | 1.712941252 | 2.099% | 198 |
| 12 | 0.449981937 | 0.436484876 | 2.999% | 858 |
| 14 | 1.689072972 | 1.688725401 | 0.0206% | 213 |
| 15 | 0.439421177 | 0.438860884 | 0.1275% | 845 |

Models 10/14 now enter derivative-free optimization after state-only residual
qualification; the sensitivity computation no longer blocks these controls.
They optimize 24 coordinates, including 16 trajectory-specific initial roots.
Both complete the initial coordinate sweep, but neither completes a full Powell
iteration before interruption. The best evaluated trial is still retained after
independent verification; it need not be an accepted whole Powell iteration.

Models 12/15 profile 32 initial roots while searching over 9/8 shared
coordinates. Model 12 becomes the lowest training-error member of this group.
Model 15 improves only slightly, but this is not evidence of convergence.

Improvements are in the aggregate objective, not every target:

| Model | Gp before / after | I before / after | U before / after |
| --- | --- | --- | --- |
| 10 | .929247 / .660994 | 1.010528 / 1.007989 | 3.309225 / 3.469841 |
| 12 | .293832 / .269622 | .534800 / .540479 | .521314 / .499354 |
| 14 | .609022 / .624068 | .999849 / 1.000707 | 3.458347 / 3.441402 |
| 15 | .247471 / .244456 | .549925 / .550397 | .520868 / .521730 |

These are training reconstructions using training-estimated initial roots.
There are no new causal-prefix forecasts, validation scores or test scores.

## Main new limitation: checkpoint overhead

The report's `numerical_budget_exhausted` label is inaccurate. Every detailed
search receipt says `timing.timeout_reason = overall_wall_limit`, at approximately
1,920 seconds. None reaches its 2,000-evaluation cap or 1,800-second
numerical-active allowance.

| Model | Search wall seconds | Numerical-active seconds | Checkpoint-callback seconds | Callback count |
| --- | ---: | ---: | ---: | ---: |
| 10 | 1920.000 | 1627.840 | 292.161 | 421 |
| 12 | 1920.441 | 747.663 | 1172.778 | 1722 |
| 14 | 1920.000 | 1650.927 | 269.074 | 451 |
| 15 | 1920.000 | 741.056 | 1178.944 | 1696 |

For 12/15, roughly 61% of search wall time is synchronous checkpoint work.
Those callbacks consume only 11–12 CPU seconds, so the dominant observed cost
is waiting during persistence rather than arithmetic. The receipts do not
establish a current cluster-wide outage.

Code inspection explains the excessive write frequency: root profiling saves
before and after each objective evaluation. Each callback deep-copies the full
result, including immutable certificates, the initial profile, best profile and
per-trajectory root-solve diagnostics. Approximately 63–65 KB are written through
the operation journal using flush, fsync and atomic replacement on shared
storage. Residual search also saves before and after evaluations, but its
numerical work is slower, so persistence consumes a smaller fraction of time.

`construction_portfolio_residual._termination` currently maps any
`budget_exhausted` flag to `numerical_budget_exhausted`. The finer timing reason
must be retained in the next reporting schema. Preserve these historical sealed
terminals; do not rewrite them to conceal the original label.

## Boundary controls and rejected trials

The exact-zero coefficient control passes for both profiled models. After
independent replay it improves model 12 by 7.2604e-8 and model 15 by 1.0103e-7
NMSE, so the terminal legitimately selects that boundary endpoint. These tiny
gains exceed the declared numerical-tie threshold but are not scientifically
substantial improvements. They do not license deleting a required mechanism.
The boundary control's internal `candidate_selected=false` means the control
does not select itself; the enclosing campaign selects only after verification.

There are 11 and 7 rejected root-profile trials, respectively. The saved
examples have nearly dependent basis columns and fail the conditional KKT
check. They are excluded from selection. Their existence is not a reason to
relax the gate. Neither residual-search arm records an invalid completed trial;
its final in-progress integration is interrupted by the overall wall ceiling.

## Recommended next milestone

1. Distinguish evaluation caps, numerical-active limits and overall-wall limits
   in reports. Keep operation completion separate from optimization convergence.
2. Compact progress snapshots and throttle counter-only/unchanged progress
   writes. Persist immutable certificates once. Preserve prompt durable writes
   of improved incumbents, accepted checkpoints and terminal outcomes, with
   explicit interruption/resume tests and unchanged nonrenewable reservations.
3. Test a deterministic sequence of objectives against a deliberately slow
   checkpoint sink. Verify unchanged search/selection semantics, durable best
   recovery and a large reduction in callback overhead. Measure actual write
   counts and time rather than assuming smaller files solve filesystem latency.
4. Continue the four verified endpoints in a separately identified campaign,
   preserving the same equations, domains, solver precision and declared
   numerical policy. Check whether the searches can now use their numerical
   allowances and complete outer iterations. Do not simply enlarge the wall
   ceiling and call the current models fully optimized.

After this reliability issue is addressed, assess remaining optimization
progress and causal-prefix forecasts before attributing residual patterns to
equation forms. The evidence already supports numerical improvements, but does
not yet support a claim that an error near .44 is the best these families can do.

## Review verification

Reauthenticated the frozen source and independently reconstructed all four
terminals without changing the archive. The 208 targeted campaign, profiling,
reduction, launcher and report-accounting tests pass (14.61 seconds).
Repository-wide Ruff retains the same 37 pre-existing findings under unrelated
`analysis/claude` files. Only this communication document is changed.
