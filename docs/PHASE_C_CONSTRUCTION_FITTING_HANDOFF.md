# Orion handoff: fitting the returned construction models

## Scope and evidence

The user wants confirmation that construction fitting used Orion's current
algorithm, independent verification of the saved parameters, and concrete
fitting improvements. No new search or model change is authorized by this
verification campaign. Proposed fitting experiments below are follow-up work,
not changes to historical results.

Source archive SHA-256:
`1ec8f1deb8f25a698c6c3e4a948874398fdd89b446e81dbc5f0c3c7a7e2a6265`.
Source plan:
`7c017ab9d29700375335f0c8c6619f254b3c27002effc01639def7c4d8e25de1`.
Local preserved source: `output/construction-fit-analysis-20261021/inspection`.
Read-only equation/result audit: the parent directory's `REPORT.html` and
`analysis.json`. These are generated artifacts, not committed datasets.

All eight models are T2-easy: topology seed 1 / proposal seeds 1,2,3 and topology
seed 2 / proposal seeds 0,1,2,3,4. There are 16 training and four validation
trajectories, each with 301 samples and three targets. The two initializer
families are three analytic-map models and five trajectory-root models. All
eight remain in scope; no selection on error.

## Current source, specific execution route

The run used the current scientific tree
`5747e11830f6fa3c4672f74d0039d3b045a4a52fcd851f0fb4ec97f44b22fde0`:
scoped initializer fitting through `prefix_execution`/`prefix_initialization`,
analytic forward sensitivities, guarded relative scaling, bounded trust-region
least squares, DOP853/Radau verification, and measured checkpoint accounting.
It was one direct-rollout start with a fixed eight-minute validation prefix.

It did not route through the separate public fitter's certified affine profiling,
terminal-output profiling, restart portfolio, or collocation recovery. M47's
known-equation duration/noise controls are not the same problem: the construction
fits have many more coordinates and an unqualified model skeleton. Keep the
eight-minute construction rule fixed in any like-for-like comparison.

## What the original receipts show

| Topology / proposal | Completed search evaluations | Saved train NMSE | Original verification |
|---|---:|---:|---|
| 1 / 1 | 10 | 8.434942497 | 60-second timeout |
| 1 / 2 | 8 | 3.831008922 | 60-second timeout |
| 1 / 3 | 7 | 7.738445841 | 60-second timeout |
| 2 / 0 | 6 | 1.812508749 | 60-second timeout |
| 2 / 1 | 6 | 1.501872701 | 60-second timeout |
| 2 / 2 | 14 | 1.802079673 | 60-second timeout |
| 2 / 3 | 33 | 1.052356465 | DOP853/Radau agreed |
| 2 / 4 | 14 | 1.663064289 | 60-second timeout |

All searches stopped at their 600 numerical-active second budget. Seven
verification passes failed before the first full DOP853 sensitivity result;
their Radau passes never started. The saved search values are provisional until
independently checked; they are not convergence certificates.

For 2/3, training NMSE is 1.052356465 and validation suffix NMSE is 1.100672634.
Validation per-target NMSE is Gp 0.336630729, I 2.005066988, U 0.960320185.
Rank is 27/41. Its approximately 5,328 total wall seconds are almost all numerical
work (~5,318.5 seconds); measured checkpoint callbacks are only ~7.55 seconds.
Storage latency does not explain this particular candidate's cost.

The new local supplement completed eight operations and verified seven exact
endpoints. Their training NMSE median is **1.802080**, unscaled MAD **0.749723**,
with 7/8 available. All seven agree with saved search errors; insulin NMSE remains
1.79–1.85. Candidate 2/0 timed out in state-only DOP853 after 11 complete training
trajectories; the twelfth started and Radau never started. Its first 11 completed
trajectories alone used 3,660,981 RHS evaluations. The 300-second allowance was
not renewed. This strengthens the solver-cost diagnosis but does not establish
that the saved endpoint would disagree with an independent integrator.

The local supplement used 1,102.801 CPU seconds, zero optimization/LLM calls,
and no new validation inference. The source snapshot, executed implementation,
per-row diagnostics and exact hashes are retained under
`output/saved-parameter-verification-20261021/`. ACES replay remains a separate
execution with its own runtime identity and costs.

## Concrete work items for fitting

1. **Separate output verification from sensitivities/rank.** The supplement checks
   the frozen endpoint with two state/output-only integrators. Preserve expensive
   derivative diagnostics as a separate status/budget so they do not erase an
   otherwise usable numerical prediction. Do not turn a timeout into a pass.
   Compare methods and per-trajectory cost on the same host before asserting a
   speedup. Completion and accurate fitting remain different outcomes.

2. **Profile stiffness and repeated segmented integration.** The completed local
   saved-endpoint results already show a strong method dependence: for topology
   seed 1 / proposals 1,2,3, DOP853 takes about 158.5/86.5/174.4 seconds and
   2,292,199/1,249,147/2,515,027 RHS evaluations, versus Radau
   10.2/6.0/7.4 seconds and 115,403/65,442/84,956 evaluations. The methods
   agree at the frozen tolerance. These are paired state-only checks on the same
   host, not a comparison with the original ACES fitting times. This is evidence
   to qualify an implicit sensitivity-search route and, separately, an independent
   stiff-system verification pair (or certified affine reference). It does not
   justify silently replacing the frozen verification protocol. The new receipts
   include per-method/per-trajectory wall/CPU and solver counters when available.
   The search needed only 6–33 completed evaluations to exhaust ten minutes.
   Inspect RHS calls, segments and scales; test an appropriate implicit-search
   route if stiffness is confirmed. An affine-system propagator may be useful
   for these linear models, but requires certified detection and exact treatment
   of the declared input interpolation and all initializers.

3. **Remove proven coordinate redundancy without changing the model class.** In
   candidate 2/3, 25 shared parameters collapse to 11 effective RHS coefficients;
   16 training roots give 27 effective coordinates, matching rank 27/41. Consumer
   gain × law amplitude products and overlapping linear terms are candidates
   for exact canonicalization or profiling. Preserve sign/domain constraints,
   shared-consumer coupling and a reversible map to original parameters. Do not
   merge effects merely because their drivers match. Qualify affine/terminal
   profiling for scoped roots rather than assuming the older router transfers.

4. **Compare search strategies only after controlling cost and coordinates.**
   Use these same eight frozen equations/initializers/data, keep original
   endpoints as controls, record all starts and budgets, and compare current
   direct rollout with justified scaling/reduction, collocation-informed starts
   or a small frozen restart set. Report training convergence and held-out prefix
   forecasts separately. No current evidence establishes the best attainable
   fit or that increasing the budget alone will solve it.

5. **Return actionable fit diagnostics to construction.** Distinguish a solver
   timeout, flat/redundant parameter direction, and a trajectory contradiction.
   Preserve target-wise residuals and input conditions. A bad endpoint is not
   proof that every parameterization of its skeleton is invalid.

## Equation evidence to retain

For candidate 2/3, the fitted insulin subsystem reduces approximately to

```text
I' = 3.313354*insulin_input - 5.599086*meal - 13.311635*I + 2.324277*I_mem
I_mem' = 0.673595*I - 4.906202*I_mem
```

The proposed "constant secretion" process actually depends on insulin input;
there is no additive constant in this fitted subsystem. A zero-meal,
zero-infusion training trajectory has constant observed insulin 25.58781346,
whereas this saved endpoint strongly decays; several validation rollouts become
negative. This is concrete endpoint evidence, not proof that every allowed
coefficient choice is infeasible: an alternative zero eigenvalue could preserve
a nonzero equilibrium. Better fitting and better scientific construction are
both still live questions.

## Supplement boundaries

The new campaign uses exact saved search parameters and training roots, with
unchanged `rtol=1e-9`, `atol=1e-11`, agreement tolerance `1e-6`. Its separately
declared budget is 300 numerical-active / 360 inclusive wall seconds per model.
No optimizer, sensitivity/rank calculation, validation-root inference, test data,
LLM, selection, or promotion is run. Source and new costs remain separate.
Seven missing validation scores remain missing. See
`docs/PHASE_C_CONSTRUCTION_SAVED_VERIFICATION.md` for reproducible execution.
