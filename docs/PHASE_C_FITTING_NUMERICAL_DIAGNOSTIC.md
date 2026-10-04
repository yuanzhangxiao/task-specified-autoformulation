# Phase C M7: fitting size, native cost and rollout stagnation

Implemented 2026-10-03. Protocol `phase-c-fitting-numerical-diagnostic-1`.
This follows the [M6 results](PHASE_C_FITTING_M6_RESULTS_2026-10-03.md).
It is a bounded development diagnostic, not a production fitter promotion.

## Why 115,218 variables, and how to reduce them

The alien problem fits only **18 global quantities**: 13 dynamic factors and
five shared hidden initial values. The rest are states at discretization nodes:

```
18 + 16 training trajectories * 600 intervals * 2 Radau nodes * 6 states
= 115,218 variables.
```

All six states receive nodal variables, including the observed state. Measurements
contribute an observation loss; they are not imposed as exact state resets.
Dynamical equalities connect neighboring nodes, and global parameters connect
trajectories. This is a sparse constrained problem, not a dense optimization
with 115,218 unrelated physical parameters. Dimension alone does not establish
infeasibility, but the current implementation and budget did not solve it reliably.

The main Phase B collocation path in `matching_probe.latent_start` also created
Radau variables at each supplied interval. Its size depended on the particular
number of trajectories, samples, and candidate states. There was no general
small-variable guarantee. The collocation-plus-sensitivity pipeline subsequently
ran full-trajectory parameter refinement and evaluated actual rollouts. A useful
final fit or a `complete` result does not establish successful collocation.
M6 demonstrates this explicitly: the successful hybrid runs used the original
parameter guesses after collocation failed to provide better ones.

A separate experimental `collocation_mesh.plan_meshes` already decouples the
state mesh from observation sampling. M7 reuses it in this comparison:

- Keep every original observation at its original weight. Evaluate the quadratic
  Radau state polynomial at observations inside an element.
- Preserve supplied piecewise-linear forcing exactly up to the existing numerical
  interpolation tolerance. Retain forcing corners and check reconstruction over
  every supplied input sample; refuse a mesh that skips a pulse.
- Enforce a maximum element length of the trajectory duration divided by 24,
  where sample resolution permits. Allocate a soft global target of 6,000
  variables by deterministic bisection. Input fidelity and resolution take priority.

On the frozen M6 inputs this gives **33,906 variables**, a 70.6% reduction from
115,218. Four densely varying input schedules alone require 586, 593, 597 and
600 intervals under this conservative interpolation rule. The remaining schedules
use 32--42 intervals. Thus 6,000 is not claimed as an achieved hard limit.
Basin retains its existing 1,442-variable mesh and serves as a regression control.

Further options remain open: higher-order elements, elimination/condensing of
local state variables, multiple shooting with fewer free boundaries, and
conditional coefficient fitting. Fixing observed trajectories to interpolated
measurements would alter the fitting problem, especially with noise, and is not
part of this milestone. Eliminating a handful of coefficients alone would barely
reduce a node-dominated problem. Shooting already has only 2,304 variables on
alien; M6's poor shooting performance therefore also requires a cost diagnosis.

## Frozen comparison

Two unchanged M6 cases (`basin_coupled`, `alien_hard`), three original generic
starts and six arms: **36 CPU fits**. All methods remain available in earlier
protocols; hybrid and reuse arms are not rerun in M7.

| Pair | First arm | Second arm | Question |
|---|---|---|---|
| Collocation | `fixed_dense_collocation` | `fixed_reduced_collocation` | Does reducing free state nodes help under equal total ceilings? |
| Shooting | `fixed_shooting_exact` | `fixed_shooting_lbfgs` | Does avoiding integrated second derivatives improve progress? |
| Rollout | `rollout_continue` | `rollout_restart` | Is training-stagnation-triggered restart useful compared with continuing the incumbent? |

The equations, fixed internal couplings/shapes, bounds, scaling, original starts,
training data and evaluator assistance remain those of M6. In particular, this
is not identification of all coefficients in the alien generator. The supplied
fixed blocks anchor latent coordinates. Full local sensitivity rank does not
prove global or practical identifiability.

### Fixed-mesh native diagnostics

Both shooting arms use the same initial shooting mesh and CVODES tolerances
`rtol=1e-7`, `atol=1e-9`. Every original input/sample interval is integrated
within a shooting window; observations cannot reset hidden or observed states.
Only IPOPT's Hessian approximation differs. Both collocation arms use exact
Hessians. Native limits are 400 iterations and approximately 75% of the remaining
900-second fit allowance, reserving the rest for training rollout screens.

There is **one mesh and one native solve** per fit. Timeout, iteration exhaustion
or native failure does not trigger refinement. Nodal consistency and actual
rollout quality are still different quantities. Native checkpoint candidates
must pass complete training rollouts to replace the ordinary starting incumbent.

Artifacts record:

- `fit/mesh_audit.json`: reduced mesh, input fidelity, all-observation retention
  and whether mandatory resolution/input corners exceed the soft variable target.
- `fit/mesh-0/layout.json`: variable/constraint/residual counts, explicit graph
  construction time, Hessian option and observation interpolation policy.
- `fit/mesh-0/derivative_profile.json`: generation and evaluation of objective,
  objective gradient, constraint Jacobian and, for exact-Hessian arms, the
  Lagrangian Hessian at the original point (constraint multipliers set to one).
  Sparse nonzeros are counted without materializing dense matrices.
- `fit/mesh-0/solver_progress.json`: last observed solver phase/iteration and
  checkpoint time; `solve_chunks.json` contains returned native call/time totals.

Derivative profiling is a **single-point diagnostic** charged to the native
allowance. It may populate caches, and does not measure total derivative cost
across the optimizer. Limited-memory arms deliberately do not construct the exact
Hessian. If a process is killed during generation/evaluation, its last phase is
retained; the missing final duration is unknown, not zero. Solver-entry-to-iteration
work includes lazy solver setup and numerical work. This instrumentation does not
separately measure every linear algebra operation, and wall time depends on hardware.

### Rollout continuation versus restart

Both arms first run the same scaled joint TRF policy, from the same generic start,
for at most **41 residual calls including screening**, or **600 seconds**,
whichever limit is reached first. Successful training accuracy stops immediately.
The two phases together share the 900-call and 900-second ceilings; initialization,
screens and failed attempts consume budget. The second phase gets only the remainder.

At the phase boundary, inspect completed training evaluations. At least 20 are
required. A restart is triggered when the best complete training NMSE improved by
no more than 0.5% over the last ten completed evaluations. The rule was selected
as a development diagnostic after observing M6, not as a preregistered independent
test of a universal optimizer policy. It does not prove a local minimum.

- Continuation starts from the best complete first-phase point.
- The restart arm uses one deterministic generic draw if the rule triggers;
  otherwise it also continues the best point. Dynamic values are multiplicatively
  perturbed around the original generic guess, and shared initials are drawn in
  scaled coordinates, respecting declared bounds. No reference values are used.
- Both arms retain the best complete training rollout across both phases. A worse
  restart cannot erase the previous fitted incumbent.

Continuation transfers parameters/initials; SciPy's internal trust-region state
is **not** resumed. Both arms incur the same phase boundary. If wall limits end
first-phase solves at different evaluations, inspect those histories before making
paired causal claims. `fit/refinement-*/training_history.json` records completed
training evaluations; `fit/diagnostic.json` records the decision, completed
phase starts and consumed work. The frozen policy and seed define the restart draw
and limits; active-phase records also contain the proposed next start. An outer interruption keeps verified
checkpoints without silently restarting consumed budgets; incomplete call totals
are reported as unknown.

## Evaluation and exit criteria

Qualification again checks reference replay and local training sensitivity rank.
The worker receives training arrays only; validation/reference are used only by
qualification and post-selection evaluation. No LLM, GPU, test access, new data
or changed prompts. All starts are reported, without best-of-three selection.

Independent Radau/DOP853 endpoint replay has a separate 300-second allowance.
Report prediction, dynamic-coefficient recovery and shared-initial recovery as
separate outcomes using the unchanged M6 thresholds. `complete` is not an accuracy
or optimizer-convergence claim. Read all 36 endpoints, including unavailable ones.

The milestone succeeds as a diagnostic when we can distinguish node count,
derivative/native cost, and stalled rollout search using saved evidence. Better
fitting is an experimental outcome, not an assumed exit criterion. A follow-up
can then choose mesh accuracy, derivative strategy or restart design based on
training diagnostics and post-selection development results.

## Run on Delta

Upload `phase-c-fitting-m7.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
The portable bundle contains committed source and the exact sealed M6 input
artifact (content SHA256
`97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`).
No M6 result/checkpoint download or GPU allocation is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
AF_CODE="$AF_BASE/code/fitting-m7"
mkdir -p "$AF_CODE"
tar -xzf "$AF_BASE/phase-c-fitting-m7.tar.gz" -C "$AF_CODE"
unset AF_MATCHED_SOURCE AF_FITTING_INPUTS
export AF_CAMPAIGN="$AF_BASE/fitting-numerical-diagnostic-v1"
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_numerical_delta.sh"
BASH
```

Defaults: existing Delta Python/CasADi environments, `bibo-delta-cpu`, one CPU,
16 GB and 25 minutes per fit array element, concurrency two. Fitting is capped
at 15 minutes plus at most five minutes for independent replay; remaining Slurm
time allows startup/reporting. Prepare has 45 minutes for regressions and reference
qualification. Account/environment/concurrency overrides follow the shared launcher.

Rerunning submission reuses confirmed receipts. An uncertain scheduler reply
requires reconciliation; do not delete intents to bypass it. Existing completed
fit records are reused exactly; interrupted fit budgets are not restarted.

After completion:

```bash
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-numerical-diagnostic-v1
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m7/scripts/hpc/inspect_phase_c_fitting_numerical_delta.sh
```

This prints scheduler/report status and creates a review archive containing plans,
sealed inputs/results, qualification, checkpoints, diagnostic profiles and logs.
Download the printed archive path for review.

## Local verification

The 67 focused strategy, reuse, challenging-case and M7 tests pass. The six-arm
native smoke passes prediction and coefficient recovery under independent replay;
completed-result resume returns identical records without rerunning fitting.
The smoke exercises native derivatives and both Hessian modes on a small control;
it is not evidence of success on alien-device. The full M7 plan prepares 36 tasks
from unchanged input bytes and equation requests. Comparing its local generic
starts against M6's Delta export gives maximum cross-platform floating-point
roundoff of 4.44e-16; original physical node guesses match exactly. Within each
M7 comparison, all arms receive the same frozen common record.

Changed implementation: the transcription campaign/worker/solver, optional
training-history recording in joint refinement, and two small diagnostic modules.
New deliverables: M7 config, tests, native smoke, Delta submission/inspection
wrappers and this runbook. No benchmark files or production fitting profiles change.
Repository-wide lint retains 37 pre-existing findings in unrelated
`analysis/claude/` scripts; modified/new Python files pass lint.
