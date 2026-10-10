# M23: preserve rollout recovery and diagnose conditional fitting

Protocol: `phase-c-nonlinear-allocation-1`.
Configuration: `configs/phase_c_fitting_allocation_v1.json`.
This follows the completed [M22 comparison](PHASE_C_NONLINEAR_COMPARISON.md).
It is an opt-in fitting experiment, not a change to the production fitter,
benchmark assets, public prompts or construction stages.

## Scientific question

M22 recovered the known nonlinear alien-device model from one of three generic
starts using our strongest applicable rollout fitter. Adding conditional node
work recovered none. However, two conditional warm starts were much worse than
their originals; ten of twelve conditional stages could not finish screening;
and 27 ultimate-precision verification retries consumed 54 minutes without
finishing. M23 separates these allocation defects from coefficient-solve,
discretization and auxiliary-trajectory errors.

The frozen inputs are unchanged:
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
All three original starts are retained, with the same anchored six-state model,
16 training trajectories, four validation trajectories, thirteen dynamic
parameters and five shared hidden initial values. Fitting outcomes receive
reference/validation scoring only after fitting decisions are sealed. The
separate evaluator diagnostic below uses reference values before fitting but
does not supply them to the fitter. No test data, LLMs or GPUs are used.

## First: a separate evaluator diagnostic

The portable package includes the six saved M22 warm checkpoints: two meshes
for each of the three starts. Export verifies the reviewed plan, backend digests,
checkpoint metadata and numeric-array hashes. Source plan:
`89d3225646b010783bc8baa14b19b40279c7cb120da5f9bd239f3298fdb5d0db`.
The sealed checkpoint export digest is
`c15dd333ba8323455b634b46203a833bd94f5fbe7b255fe573cd469a46c4726a`.
It is included in the campaign identity and is never a starting-point pool.
Preparation rejects changes to the frozen mesh/anchor/penalty settings rather
than interpreting saved arrays on a different grid with the same dimensions.

The evaluator integrates the original equations at reference parameters and
initials, preserving every input knot and sampling each actual Radau stage.
Reference derivatives are evaluated analytically from that supplied skeleton;
they are not estimated from noisy data or supplied to the fitter.

For the twelve conditionally affine coefficients `beta`, fixed states and fixed
nonlinear shape give `f(x,u; beta, eta) = b(x,u; eta) + A(x,u; eta) beta`.
The diagnostic compares:

| Case | States | Nonlinear shape and hidden initials | Purpose |
|---|---|---|---|
| Exact RHS derivatives | Reference | Reference | Verify bounded LS and conditional rank without mesh truncation |
| Radau residuals | Reference at actual stage nodes | Reference | Measure finite-mesh coefficient bias |
| Radau residuals | Saved estimated nodes | Reference | Isolate estimated-trajectory/boundary mismatch |
| Radau residuals | Saved estimated nodes | Saved estimates | Reproduce the actual conditional subproblem |

The derivative design columns are normalized before bounded LS and numerical
rank is reported. Relative residual norm below `1e-8` is required; maximum
relative coefficient error below `1e-5` is required only when the design has full
column rank. Rank-deficient solutions do not certify coefficient recovery.
This is a numerical implementation check, not a global-identifiability claim.
The three Radau cases report continuous errors, ranks, observation loss and
mean/maximum scaled increment defects. They have no accuracy gate: poor results
are precisely what this diagnostic is meant to measure.

Each checkpoint diagnostic has 180 seconds, independently journaled and charged.
Interrupted work is not restarted with a fresh budget. The preparation job must
finish all six diagnostics and pass the exact-derivative correctness checks
before fitting starts. Failure leaves evidence available for inspection and
does not consume a six-task fitting array. Reference arrays and fitted diagnostic
coefficients never enter the `TrainingProblem` allowlist or the optimizer.

## Then: six matched fitting tasks

Three generic starts times two methods:

1. `best_rollout`: the strongest applicable profiled rollout.
2. `conditional_then_best_rollout`: exactly the same original-start rollout,
   with conditional assistance available during subsequent restart trials.

Both retain exact profiling of four terminal-output gains, analytical
sensitivities, scaled bounded TRF, numerical polishing, deterministic diverse
starts and independent original-equation verification. The fourteen outer
unknowns include all five shared hidden initials. Nothing in the nonlinear
optimization kernel, parameter domains, data, or restart-vector generator changes.

Both arms first receive 600 seconds/600 evaluations from the original start.
There is **no conditional work in that warm stage**. If needed, each receives
three restart trials of 200 seconds/200 evaluations, followed by 600 seconds/600
evaluations continuing the best verified trial. A promising distinct trial may
receive continuation even when its short trial is worse than the incumbent;
retention still requires independent improvement. Total search ceiling remains
1,800 seconds per task. Diagnostics and verification costs are additional and
reported separately. Successful tasks stop early.

### Measured screening allocation

Ordinary verification records the duration of each complete DOP853 and Radau
training batch. The conditional arm uses the maximum available ordinary DOP853
batch duration, augmented by later completed conditional screening durations,
as its screening estimate. Only training computations determine this estimate.

At most 30% of a restart trial (60 seconds) is available for conditional work.
It reserves twice the measured screening cost, with a one-second floor, and
requires at least twenty seconds left for node optimization. Otherwise it
records a `budget_skip` and returns that unspent allowance to rollout search.
Without a completed cost measurement it also skips rather than guessing.

The existing two meshes, penalties and alternating solves remain unchanged.
Only mesh levels that can receive a meaningful node allowance are attempted.
Graph and node costs consume the conditional budget. Screening tries the finest
available endpoint first, with the whole remaining allowance; it never divides
time among three candidates that cannot individually finish. It does not rerun
the raw control merely to choose a restart. A later endpoint is screened only
if at least a full measured batch still fits. The factor-of-two reservation
provides headroom for cooperative node-solver overrun; small overhead does not
invalidate the whole screening window.

Costs are estimates, not guarantees: a new parameter vector can integrate more
slowly. Timeouts remain explicit, and library calls can slightly overrun their
cooperative deadline. No failed or incomplete rollout receives a finite score.
All conditional time, including unsuccessful work, is charged. An interrupted
conditional operation consumes its full allowance and cannot restart for free.
Fallback uses the original trial vector and the remaining rollout allowance.

### Verification proportional to the decision

Every candidate considered for retention still receives ordinary DOP853/Radau
checks. The historical retention rule, based on solver agreement and improvement
of the two-loss interval, remains in force.

The new policy requests stricter integration only if ordinary outputs are
unreliable, mean and worst-trajectory losses are within ten times their targets,
or two distinct candidates' loss intervals overlap and precision could decide
retention. In the overlap case the incumbent is also checked if necessary.
Each parameter vector gets at most one stricter attempt; repeated appearances
reuse its result, including a timeout. A failed stricter check preserves a useful
ordinary-verified incumbent.

Final success still requires mean training NMSE at most `1e-12`, worst trajectory
at most `1e-11`, prediction disagreement at most `1e-7` and loss disagreement at
most `1e-13`. A reliably poor fit routes to further search even if ultimate
precision is unresolved; its uncertainty label is not silently upgraded.
This is numerical triage, not a confidence probability or a coefficient-accuracy
certificate. No profile-likelihood grid is added.

## Evidence to inspect

Compare all six outcomes, not only the best start: post-warm/final prediction,
retrospective coefficient and initial recovery, actual calls, elapsed/CPU time,
budget skips, completed screens, screening failures and numerical retry cost.
The experiment tests whether corrected allocation makes conditional proposals
useful. It does not promise recovery of starts 1 and 2. M22 remains an immutable
historical experiment; its pinned code is required to resume its own jobs.

Backend artifacts are sealed before validation/reference scoring. Conditional
primal checkpoints remain saved separately from retained physical models. All
operations and scheduler submissions are journaled for deterministic completed
resume; partial or interrupted accounting is explicit.

## Delta commands

Upload `phase-c-fitting-m23.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c`, then run:

```bash
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m23"
tar -xzf "$AF_BASE/phase-c-fitting-m23.tar.gz" -C "$AF_BASE/code/fitting-m23"
bash "$AF_BASE/code/fitting-m23/scripts/hpc/submit_phase_c_fitting_allocation_delta.sh"
```

The launcher checks package checksums and uses existing Python/CasADi dependencies.
The preparation job runs tests and the separate diagnostics, followed by six
fitting tasks with at most three concurrent tasks, then a report job. Each uses
one CPU and 16 GB, no GPU. Preparation requests 30 minutes; fitting requests
90 minutes per task, including verification and retrospective evaluation. These
are scheduler limits, not fixed optimization durations.

Default results: `/work/hdd/bibo/yxiao2/phase_c/fitting-allocation-v1`.
Overrides: `AF_FITTING_ALLOCATION_ROOT`, `AF_ACCOUNT`, `AF_CONCURRENCY`,
`AF_PYTHON`, `AF_CASADI_ROOT` and `AF_FITTING_ALLOCATION_INPUTS`.
An alternative input location must have `conditional-diagnostic-inputs.json`
beside the unchanged generic inputs.

After completion (or to inspect a partial run):

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m23/scripts/hpc/inspect_phase_c_fitting_allocation_delta.sh
```

It prints compact diagnostics, fitting scores, screening/retry evidence and the
exact review archive path. Keep the plan, source commit, inputs, all checkpoints
and logs. No cleanup or cancellation of other campaigns is performed.

## Local verification

The independent nonlinear smoke gives identical original-start results in both
arms: training NMSE `5.08e-13`, maximum absolute parameter error `4.04e-6`, and
identical completed resume. A separate real conditional restart screens two
finite candidates without repeating the control. This establishes mechanics on
a small example, not alien-device recovery. The hard benchmark and evaluator
diagnostics run on Delta; full fitting is not run on the laptop.

Focused tests cover protected warm fitting, measured screening, deadline
headroom, failed-candidate retention, near-target/tie precision, cached failed
retries, interrupted budgets, conditional rank deficiency, reference separation,
six-task preparation and submission resume. Verification on 2026-10-09:

- Focused and portable-bundle qualification: 34 tests passed in each environment.
- Full repository suite: 4,369 passed and eight Torch-dependent tests skipped.
  Three source-identity checks failed while the final source edit was made during
  the run. With source stable, all 65 tests across their three affected modules
  and the five fitting qualification modules passed; no persistent test failure
  remains from this run.
- Changed Python files pass Ruff, shell launchers pass syntax checks, and
  `git diff --check` passes. Repository-wide `ruff check .` still reports 37
  pre-existing findings in unrelated `analysis/claude` files, left untouched.
- The portable package prepares the real six-task plan and reports an incomplete
  campaign correctly. Its inspection command handles pending diagnostics and
  missing task records without a jq failure.

## Completed Delta review: `review-20261010-013958`

Source commit: `55285b532628c61b833f8d958911a311101a2fbf`.
Plan: `3f9c8d1286131dc3ae57c6eda8c0d0e3d61f9ae4a17e02edd8b26b95845a250c`.
Root: `/work/hdd/bibo/yxiao2/phase_c/fitting-allocation-v1`.
All six fitting tasks and all six evaluator diagnostics completed, with complete
cost accounting. The downloaded plan/input/diagnostic-input seals and six
backend/result identities agree. Regenerating the report reproduces the supplied
summary exactly. No new benchmark fitting, LLM calls or test-data access was
performed in this review. Reference diagnostics remain evaluator-only.
Review verification: 21 relevant tests and the two-arm numerical smoke passed;
`git diff --check` passed. Repository-wide Ruff retains the same 37 unrelated
analysis findings. This review changes documentation only.

### Final retained models

Errors below are retrospective, after training-only selection. Coefficient
errors are percentages; initial errors are absolute physical-coordinate errors.
Minutes include search, verification, sensitivity and setup, but exclude the
separate evaluator diagnostic and retrospective validation/reference scoring.

| Start | Arm | Train NMSE | Validation NMSE | Maximum coefficient error | Maximum initial error | Minutes |
|---|---|---:|---:|---:|---:|---:|
| 0 | Rollout | 5.28929e-5 | 7.97332e-5 | 13.276% | 0.080945 | 39.33 |
| 0 | Conditional-assisted | 3.89791e-10 | 1.54143e-9 | 0.027687% | 0.00024575 | 39.79 |
| 1 | Rollout | 0.406018 | 0.610700 | 243.02% | 0.99659 | 38.13 |
| 1 | Conditional-assisted | 0.391615 | 0.546782 | 243.02% | 0.99659 | 38.95 |
| 2 | Rollout | 0.325708 | 0.539544 | 1407.8% | 1.2006 | 38.48 |
| 2 | Conditional-assisted | 0.736700 | 0.581581 | 1018.8% | 1.3591 | 38.88 |

The summary's `accuracy_passed` uses the inherited retrospective NMSE gate of
0.01 for both splits. It is true for start 0 in both arms. Coefficient/initial
recovery gates are respectively 1% relative and 0.01 absolute, passed only by
the conditional-labelled start-0 task. None reaches the fitter's much stricter
mean/worst training targets of `1e-12`/`1e-11`. These are different questions;
`complete` is protocol completion, not convergence or coefficient recovery.

The excellent conditional-labelled start-0 endpoint is from `warm/check`,
before any conditional operation. All three conditional-labelled tasks retain
their original warm result. No conditional restart or its continuation improves
any final incumbent. Thus the table is not evidence that conditional profiling
caused start-0 recovery. Rollout-only start 2 does improve through a restart and
continuation, from train/validation 0.736818/0.581649 to 0.325708/0.539544, but
still fails the accuracy gate.

### The coefficient diagnostic isolates the main difficulty

All six exact-derivative checks have conditional rank 12/12. Their maximum
relative coefficient errors are `2.42e-15` to `9.66e-15`. The bounded coefficient
solve therefore recovers this block with accurate trajectories and derivatives.
This establishes conditional numerical correctness for this model, not global
identifiability with unobserved states.

| Supplied information | Maximum relative coefficient error |
|---|---:|
| Reference states and exact RHS derivatives | below 1e-14 |
| Reference states at coarse Radau nodes | 0.581913% |
| Reference states at finer Radau nodes | 0.0163293% |
| Saved estimated nodes, reference nonlinear shape/initials | 687.16%–4203.52% |
| Saved estimated nodes, saved nonlinear shape/initials | 1053.72%–4203.52% |

Refining the reference mesh reduces discretization bias by about 36 times.
However, supplying correct nonlinear shape and initial values while retaining
the estimated trajectories still gives very wrong coefficients. The dominant
problem in these six saved M22 checkpoints is their latent-trajectory/boundary
mismatch, not a defective bounded least-squares solver. For five of the six
saved-node cases, another coefficient solve with the saved shape leaves the
objective unchanged to roundoff: the coefficient block is already essentially
optimal for those incorrect trajectories. This does not imply that the full
problem has converged or is unidentifiable.

There are three original starts on one problem, and two meshes per start;
the six diagnostic records are not six independent benchmark cases. All six
diagnostics together cost 157.19 seconds wall time and 152.94 CPU seconds.

### Allocation fixes worked, but the conditional proposals remained poor

- Nine of nine conditional stages produced a complete finite rollout screen;
  none failed screening. M22 had complete screening in only two of twelve stages.
- All nine M23 proposals were coarse-mesh endpoints, with rollout training NMSE
  from 1.2465 to 134.6007. No finer mesh fit within the meaningful solve allowance.
- The 27 node solves comprise 18 iteration-limit exits and nine deadline callback
  stops, with no converged node solve. Callback traces in the error logs are
  recorded bounded-stage stops; they did not abort the campaign.
- Conditional work cost 419.87 seconds total: 294.78 in node work/setup and
  125.09 in screening. It was charged against rollout search allowances.
- No ultimate-precision retry was attempted, versus 27 failed 120-second retries
  in M22. Ordinary independent DOP853/Radau checks remained active throughout;
  they cost 3,094.41 seconds across the six tasks.
- Fitting/verification/setup totals were 115.94 minutes for rollout and 117.63
  minutes for conditional-assisted, a 1.46% difference. Retrospective scoring
  added 13.39 and 12.59 minutes respectively. These are sums of task times, not
  queue-to-completion latency. Changed search outcomes prevent attributing every
  cross-M22 runtime difference solely to the allocation policy.

### A continuation rule now wastes a strong incumbent

Start 0 uses identical original parameters and the same profiled-rollout kernel
and 600-second allowance in both arms. Rollout-only starts 33 evaluations and
the other arm starts 35; both stop inside an evaluation at the deadline. Counts
include that incomplete final attempt. Their saved warm losses differ sharply:
`5.29e-5` and `3.90e-10`. The archive supports a wall-budget cutoff explanation,
not a beneficial conditional effect; it does not record enough machine context
to assign the speed difference to a particular hardware cause. M22's unchanged
rollout kernel finished start 0 in 35 calls and 590.54 seconds, reaching
`1.26e-16`. The identical algorithm can therefore finish on different sides of
the last useful evaluation under a wall-clock ceiling.

After its warm stage, each M23 task spends another 1,200 seconds on three short
restarts and continuation of the best restart. In start 0, the continued restart
has training NMSE 0.48893 in rollout-only and 1.02714 in conditional-assisted,
far worse than the retained warm solution. Neither warm incumbent receives
additional fitting. This preserves the result correctly but allocates effort
poorly. Restarts remain useful for genuinely stalled cases, as start 2 shows.

### Recommended next milestone, not yet implemented

Prioritize training-only incumbent continuation before another broad conditional
comparison. Give a budget-limited incumbent a bounded polishing opportunity,
particularly when its saved progress shows substantial recent improvement;
retain a separate restart allowance for stalled solutions. Record progress per
completed evaluation, active bounds and stop reasons, so slow computation is
not confused with optimizer convergence. Preserve incumbent protection and the
existing independent numerical checks.

For a clean comparison, generate one sealed warm checkpoint per original start
and branch competing continuation policies from that identical checkpoint.
Compare completed evaluations as well as wall/CPU budgets, with a generous hard
wall guard; report shared warm cost separately. Choose priorities using training
evidence only, never retrospective coefficient or validation accuracy. Include
all three starts, rather than only the successful one.

Keep conditional profiling experimental. A subsequent isolated test could
initialize its nodes from a feasible rollout and verify any coefficient update
again through a free rollout. More mesh density alone is unlikely to cure the
estimated-trajectory error documented here, although this does not rule out
better trajectory estimation or better-budgeted collocation.
