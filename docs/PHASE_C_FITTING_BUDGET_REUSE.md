# Phase C fitting milestone 4: budget and fixed-mesh reuse

Agreed 2026-10-01. This opt-in CPU study follows the
[M3 archive review](PHASE_C_FITTING_M3_OPTIMIZATION_ANATOMY.md). Correct equation
structures, training/validation separation and public initialization contracts
remain fixed. No construction changes, LLM calls, test access or production
default promotion are included.

## Questions and interpretation

Rollout-only fitting evaluates the deployed continuous trajectory directly,
within numerical integration tolerance. This avoids optimizing a discretized
trajectory that may predict differently when integrated. It can help accuracy,
but M3 does not isolate this factor: solver, dimension, mesh and time allocation
also differ. Multiple shooting already integrates accurately within windows,
yet underperformed in M3. Accurate integration is not a sufficient explanation.
Converged and adequately refined collocation can also approximate the same ODE
closely. Independent rollout remains the final check for every strategy.

The observation objective is quadratic in observed nodal states in these cases,
but the nonlinear dynamics constrain those states. Minimizing the observation
loss alone would simply fit observations and leave latent states unconstrained.
Fixing coefficients does not in general make the trajectory subproblem convex.

A promising later alternative is conditional coefficient estimation. At fixed
trajectories X and nonlinear shape parameters eta, a defect vector may have form
`D(X,eta,w) = A(X,eta) w - b(X)`. For CSTR, activation is nonlinear and the six
other dynamic coefficients are affine in the RHS once states and activation
are fixed. Under a *soft-defect* objective, the bounded w subproblem is a convex
quadratic least-squares problem. It could alternate with the nonlinear
trajectory/shape block. Under the current hard equalities, arbitrary fixed
trajectories need not admit any feasible w. A penalty/continuation formulation
would therefore be a separate method, with its own defect schedule and accuracy
checks. It is recorded for the next milestone, not silently inserted here.

## Frozen comparison

Protocol: `phase-c-fitting-budget-reuse-1`.
Configuration: `configs/phase_c_fitting_strategies_v2.json`.

Three cases (saturating-shape identifiable control, CSTR easy and CSTR hard),
three original generic starts, five arms: **45 fits**. The four original methods
remain available and use the same algorithms as M3, now with **600 seconds** and
at most **600 training rollout evaluations** each. A fifth `collocation_reuse`
arm tests the revised collocation schedule under that same ceiling.

This is a fresh matched-budget comparison from the same generic physical starts,
not a warm continuation of whichever M3 seed looked best. Previous M3 results
are retained for the 180-versus-600-second comparison. Wall-clock effects can
vary across nodes, so this is a strategy comparison rather than a bitwise replay
or a clean attribution of every improvement to one numerical ingredient.

All data and generic starts are imported exactly from the sealed M3 plan and
inputs, without regenerating synthetic trajectories or reading fitted results.
The portable `matched-m3-inputs` directory contains just those two sealed files.
The original CSTR source is the sealed M1 Phase C input artifact. Original
observations, equation skeletons, input knots, coordinate transforms, initial
node guesses, coefficient bounds and output weights are retained. Identical
initial vectors and nodes are shared across arms within each case/start.
Every result is reported; no best-of-three selection is made.

| Arm | Work inside the common ceiling |
|---|---|
| Rollout only | Existing centered/scaled joint TRF with forward sensitivities and training stopping criteria. |
| Collocation then rollout | Existing hybrid: one third of the allowance for collocation, remainder for rollout refinement. |
| Direct collocation | Original three-pass mesh/tolerance schedule with larger total allowance. |
| Adaptive multiple shooting | Original CVODES/IPOPT three-pass schedule with larger total allowance. |
| Collocation with reuse | Solve the current mesh before refinement; retain the same graph and warm-start repeated native solves. |

Training accuracy stops remain `1e-10` overall and `1e-9` on each trajectory;
existing optimizer/stagnation stops remain active. An allowance is a ceiling,
not a required duration. The new native variant retains a 400-iteration maximum
per mesh, split into at most eight 50-iteration calls. Exceeding iterations is
different from exhausting wall time and remains visible in its records.

## What graph reuse does and does not do

The new collocation arm allocates 75% of remaining fitting time to the current
native mesh solve, reserving the remainder for full-training screening. It
constructs one CasADi Opti problem and configures IPOPT once. If a call reaches
its iteration limit, it reuses that instance and explicitly transfers the
primal values and constraint multipliers. Native failure other than an iteration
limit does not trigger an unconditional restart. The external process deadline
still bounds startup, graph construction, solver work and checkpoint writing.

Only native convergence permits refinement according to the existing off-node
defect indicator. An unfinished coarse solve retains checkpoints for rollout
screening and reports `coarse_solve_incomplete_pending_replay`; it does not
automatically spend another stage on a larger problem. Changed meshes need new
graphs, and old values are interpolated as starts. Screening alone selects
the retained parameters, never the low nodal objective of an infeasible iterate.

Reuse is **within one process and one mesh**. No cross-process compiled-solver
cache or portable IPOPT internal-state resume is claimed. Primal/dual warm
starts do not preserve all internal state such as quasi-Newton history, and
can themselves have costs; the original collocation arm is kept as a control.
`layout.json` records graph construction time, dimensions and build count;
`solve_chunks.json` records actual calls, iterations and reuse. On process death,
already flushed checkpoints/records survive; missing final counts stay unknown.

## Evaluation, resume and exit criteria

Worker payloads contain only training data, public equation restrictions,
generic starts and numerical settings. Qualification and endpoint evaluation
may access evaluator truth; fitting, checkpoint selection and stopping may not.
Independent Radau/DOP853 endpoint replay has the same separate 240-second
allowance as M3. Validation never fits coefficients or initial conditions.

Report prediction NMSE for each split, solver agreement, per-coefficient
absolute/relative errors and maximum relative coefficient error. Report hidden
initial errors separately in physical units; relative temperature percentages
depend on the arbitrary zero and are avoided. Undefined relative errors at zero
reference values remain null. A coefficient pass requires independent replay
agreement and every dynamic coefficient within 1%; output accuracy has its
existing separate criterion. CSTR global identifiability is still unproven.

Groups give median and worst coefficient errors, median prediction errors,
available-vector/score counts, and pass counts out of all planned starts.
Unavailable and interrupted outcomes are not dropped from denominators. Plans,
source/runtime identities, starts and result/backend seals remain immutable.
Completed tasks reuse their artifacts; an interrupted native budget is not
silently restarted. No historical result is overwritten.

Exit: reconcile all 45 outcomes; compare the four longer-budget strategies;
compare the two collocation policies at matched budget; inspect whether hard
CSTR coefficients continue recovering and whether low discretized loss still
differs from free rollout. Then decide whether to pursue conditional affine
blocks, a soft-defect continuation, additional robustness controls or a pipeline
integration. No such next-stage promotion is automatic.

## Delta commands

Upload `phase-c-fitting-m4.tar.gz` to the Delta home directory. It contains the
frozen source, configuration, relevant tests, sealed M3 plan/inputs, commit and checksums.
Existing Python/CasADi environments are reused. One CPU and 8 GiB per fit, two
concurrent tasks by default; no GPU or ACES allocation is needed. The sum of
optimizer ceilings is 7.5 CPU-hours, plus qualification, replay and overhead.
Many runs may stop earlier. Scheduler fit allocations are 20 minutes each.

```bash
bash <<'BASH'
set -euo pipefail
AF_ARCHIVE="$HOME/phase-c-fitting-m4.tar.gz"
AF_REV=$(tar -xOf "$AF_ARCHIVE" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-m4-${AF_REV:0:7}"
mkdir -p "$AF_CODE"
tar -xzf "$AF_ARCHIVE" -C "$AF_CODE"
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-budget-reuse-v1
export AF_ACCOUNT=bibo-delta-cpu AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_budget_delta.sh"
BASH
```

Expected: preparation, a 45-task fitting array, and report job IDs. Preparation
runs focused tests and the reference/sensitivity gates. Saved uncertain scheduler
replies stop for reconciliation instead of duplicate submission. Repeating this
command reuses confirmed submissions; it does not grant extra fitting time.

After completion:

```bash
bash <<'BASH'
set -euo pipefail
AF_REV=$(tar -xOf "$HOME/phase-c-fitting-m4.tar.gz" SOURCE_COMMIT)
AF_CODE="/projects/bibo/yxiao2/repos/phase-c-fitting-m4-${AF_REV:0:7}"
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-budget-reuse-v1
bash "$AF_CODE/scripts/hpc/inspect_phase_c_fitting_strategies_delta.sh"
BASH
```

The inspector refreshes the summary, prints scheduler state/group comparisons,
and writes the review archive for download. It performs no new fitting.

Numerical reference: [CasADi Opti and warm starts](https://web.casadi.org/docs/#opti-stack).

## Local verification

- Full repository pytest: **3,824 passed, 8 skipped, 1 warning** in 1,608.01
  seconds. Optional dependency skips remain explicit in the test report.
- All 31 focused tests pass in both the project and portable checkout. They
  cover real repeated IPOPT calls with one graph/solver construction, explicit
  primal/dual warm starts, unchanged-mesh handling after failure, coefficient
  versus initial-value errors, missing outcomes, source-bound resume and import
  of exact prior generic starts without regenerating data.
- The five-arm linear smoke passes output/parameter/latent recovery and exact
  completed-result resume. Both split NMSEs are below `4.5e-8` for every arm.
  This qualifies the plumbing; it does not establish CSTR improvement.
- All three full-case reference replay and local sensitivity-rank gates pass.
  The nine imported common records (starts, nodes and coordinates) and all
  three case payloads match the supplied M3 archive exactly. Import avoids the
  roughly `1e-14` cross-platform rounding differences observed in regeneration.
- Changed Python files pass Ruff, launchers pass `bash -n`, and patch whitespace
  checks pass. Repository-wide Ruff retains 37 existing findings in unrelated
  `analysis/claude` files.
