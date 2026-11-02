# Fitting follow-up: reduced nonlinear models and profiled initial values

The authorized milestone is implemented as a separate seven-task CPU campaign.
See `docs/PHASE_C_CONSTRUCTION_PORTFOLIO_PROFILE.md` for the interface and Delta
commands. It reads the completed `construction-portfolio-search-v1` campaign and
writes a fresh `construction-portfolio-profile-v1` root. Source records, model
equations, domains, initialization contracts and benchmark data remain unchanged.

## Implemented work

- **03:** import the exact tighter baseline/Jacobian and successful perturbation
  receipts. Retry only missing/incomplete primitives for directions 10/22 in
  explicitly new bounded reservations. Direction 19 stays passed; direction 8
  remains inconclusive. Partial evidence never authorizes a full search.
- **10/14:** certify an additional product and three additive combinations.
  Optimizer dimension falls from 29 to 24. Compose the constant certified lift
  into each trajectory's symbolic system before differentiation, reducing local
  sensitivity columns from 21 to nine. Verify supplied-to-canonical equivalence
  and qualify derivatives before continuation. Both default methods are BDF;
  original Radau remains the independent endpoint check.
- **12/15:** certify affine dependence on 32 trajectory initial roots and solve
  the conditional bounded least-squares problem with feasibility/KKT and actual
  rollout checks. Compare the incumbent and saved action-timescale-10 seed with
  identical outer-search budgets. All shared coordinates, including both
  timescales, remain free: nine/eight outer coordinates, rather than 41/40.
  Positive timescales use logarithmic relative coordinates with no artificial
  positive floor. A balanced initial sweep probes every shared coordinate before
  Powell, avoiding the previous early-coordinate starvation.
- **Preservation:** independently verify and retain the conditional root-fit
  checkpoint even before outer search, so an early search interruption cannot
  lose it. Exact best/last-accepted endpoint equality permits one diagnosis;
  different endpoints keep separate histories and checks.
- **Precision/accounting:** target smaller local steps for the dominant meal-rate
  derivative uncertainty in 12/15. Record evaluation requests per oracle view,
  including exceptions, and record profiling scopes separately. Nested views
  must not be summed as physical solves. Historical missing counters receive
  explicit additive corrections; historical terminals are not rewritten.

## Saved-endpoint checks

No long search or remote job was executed locally. Bounded implementation checks
used the supplied training data only.

| Model | Slower-seed training NMSE | Conditional root fit | Independent original rollout |
| --- | ---: | ---: | --- |
| 12 | .556565972 | .534137382 | Passed; residual discrepancy 7.22e-10 |
| 15 | .558997699 | .537017157 | Passed; residual discrepancy 7.36e-10 |

The corresponding Gp/I/U NMSEs are .406824/.619876/.575712 and
.421085/.613346/.576620. At the incumbent shared parameters, the existing roots
were already conditionally optimal, so projection leaves .564509/.564608
unchanged. These are training reconstruction results, not held-out forecasts.

For models 10/14, reduced state-only loss matches the saved loss within about
1e-13. Full Jacobian tests remain incomplete within the deliberately short
20-second local cap. Three-way equivalence checks completed the two BDF solves
but not the independent Radau solve within that cap. No sensitivity-speedup or
convergence claim follows: Delta must complete its separately reserved checks,
or preserve the old models and report the blocker.

Evidence remains in ignored local output directories:
`output/construction-portfolio-reduction-smoke-20261101/`,
`output/root-profile-saved-smoke-20261101.json`, and
`output/root-profile-slower-seed-original-replay-20261101.json`.

## Research boundary

This remains a fitting experiment. Conditional least-squares optimality does
not imply that the shared-parameter problem is solved, that parameters are
identified, or that a scientific mechanism is valid. No construction revision,
production promotion, validation/test access, GPU use or LLM call is enabled.
Fresh causal-prefix evaluation is still required before claiming forecasting
performance for a new endpoint.

## Verification and resource limits

Across the affected targeted suites, 172 tests pass after the fixes, including
synthetic rollout, chart/gradient agreement, original-domain refusals, inner
least-squares feasibility, interruption, immutable resume and launcher tests.
Changed Python files pass Ruff. Repository-wide Ruff retains the same 37
unrelated findings in `analysis/claude/`; those files were not changed.

The production source projection recreates all five pinned source records and
the seven-task roster. The maximum inclusive per-task reservation is 6,060
seconds; the scheduler requests four hours, one CPU and 8 GB per worker, with
two concurrent tasks. Search allowances are matched at 1,800 numerical seconds
and at most 1,000 profiled objective evaluations; nonlinear searches may stop
earlier on their numerical budget or qualification failure. The report receives
30 minutes. Reporting and inspection are receipt-only operations.
