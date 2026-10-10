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

## M24 implementation: shared-warm incumbent continuation

The user approved incumbent continuation on 2026-10-09. The new opt-in protocol
is `phase-c-incumbent-continuation-1`. It tests allocation of further rollout
search; it does not change the production fitter or the conditional-trajectory
algorithm.

### Frozen handoff and comparison

Export the **warm/check** checkpoint of the `best_rollout` arm for each of the
three original M23 starts. This predeclared source arm is used for all seeds;
neither validation nor reference coefficient accuracy chooses a parent. In
particular, start 0 uses the less accurate `5.289e-5` training warm checkpoint,
not the other arm's already excellent warm checkpoint. Starts 1 and 2 retain
their difficult warm points (`0.4060` and `0.7368`).

The exporter verifies source plan
`3f9c8d1286131dc3ae57c6eda8c0d0e3d61f9ae4a17e02edd8b26b95845a250c`,
original input identity, result/backend hashes, original training-problem
identity, the warm search's starting vector, and agreement between its best
vector and the saved independent check. All three sources are mandatory.
The sealed portable handoff has content digest
`796c2f2ac5a12ddeaf12d54e2a67541fca67d24c5060c9e1b54c5fa805aa3144`.

Each pair receives exactly the same warm parameter vector and the same three
training/domain-only restart vectors, generated using the original problem
identity. Both independently recheck their shared warm vector before work:

1. **`incumbent_first`:** continue the verified incumbent, then try the three
   distinct starts if accuracy remains insufficient.
2. **`restart_first`:** try those same three starts, then continue the best
   independently verified restart. This preserves the M23 allocation rule even
   when that restart is worse than the retained incumbent; it never authorizes
   replacing the incumbent with a worse result.

Continuation starts a fresh bounded TRF optimizer at saved parameters. It does
not restore an internal trust-region radius, Hessian approximation or optimizer
iteration state. Both arms preserve exact terminal-output coefficient profiling
where applicable, bounded joint fallback on numerical profiling failure,
independent DOP853/Radau verification, selective tighter checks and conservative
incumbent retention. There is no conditional-collocation work in M24.

### Allowances and stopping

Each arm has one continuation block of at most **60 started residual/Jacobian
evaluations / 1,200 seconds**, and three restart blocks of at most **15 started
evaluations / 400 seconds each**. Thus both have the same potential 105 attempts
and 2,400 search seconds. A failed profiling stage's bounded joint fallback
consumes that stage's remaining attempts/time, not a new allocation.

These wall guards are more generous than M23's 600/200-second blocks. M24 is a
within-experiment comparison from shared checkpoints, not a matched-budget
claim against M23. Actual work can differ because a solver converges, fails,
hits its guard, or achieves the early-stop certificate. Independent checks have
180 seconds per attempt; final sensitivity has 180 seconds; retrospective
scoring has its own recorded budget. Slurm reserves 90 minutes per one-CPU task
including overhead, rather than forcing search to last 90 minutes.

Search stops early only when independent numerical checks support both the
`1e-12` mean training target and `1e-11` worst-trajectory target. Ordinary TRF
step/gradient termination also ends a block. A small solver step by itself is
not an accuracy or identifiability certificate. If the incumbent remains poor,
the restart allowance stays available. This initial experiment deliberately
tests one bounded continuation opportunity; it does not yet infer an optimal
adaptive allocation rule from noisy progress curves.

### Progress, uncertainty and accounting

For every **completed** residual/Jacobian evaluation, save training NMSE,
best-so-far training NMSE, elapsed time, parameter-vector hash, active outer
bounds, and the projected gradient of the unamplified mean squared residual
in scaled outer optimizer coordinates. Bound activity uses a relative `1e-8`
proximity tolerance. These records include rejected optimizer trial points;
they must not be described as accepted iterations. Profiled inner coefficients
are excluded from the outer gradient/bound diagnostic.

Started calls, completed calls, stop reasons and incomplete accounting are
distinct. A timeout inside integration can increase the first count without
increasing the second. The summary includes first/best/last evaluation details
and the final five best-so-far losses for each search stage. These are numerical
diagnostics, not calibrated confidence levels.
If an interruption prevents complete accounting, aggregate call counts are
`null` with an explicit incomplete flag, rather than zero. If the imported warm
vector cannot be independently reverified on the current worker, the task keeps
that vector and reports `retained_unverified` without spending its search budget
or promoting a restart against an unavailable comparison.

The existing fitting assessment and full joint sensitivity, including shared
hidden initials, remain outputs. Warm computation is reported **once per
original seed**, separately from new per-arm search, verification, sensitivity
and retrospective scoring. Interrupted stage allowances are charged without
reset; completed operations and final tasks resume by exact identity. An
unverified resumed operation cannot erase a verified incumbent. Validation and
ground-truth coefficient errors enter only after the fitting backend is sealed.

### Delta execution

The portable archive is `transfers/phase-c-fitting-m24.tar.gz`. It includes the
sealed handoff, configuration and source snapshot, with `SOURCE_COMMIT` and
`SHA256SUMS`. Generated inputs, results and source API artifacts are not committed.
No prior M23 directory is required on the worker after this archive is unpacked.

After uploading that archive to `/work/hdd/bibo/yxiao2/phase_c`:

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m24.tar.gz
bash phase-c-fitting-m24/scripts/hpc/submit_phase_c_incumbent_continuation_delta.sh
```

This creates `/work/hdd/bibo/yxiao2/phase_c/fitting-incumbent-v1`, with a CPU
preflight, six CPU fitting tasks (three concurrent by default), and a dependent
report. It uses existing Delta Python/CasADi environments, no GPU or LLM calls.
`AF_CONCURRENCY` overrides the array throttle; it does not change account limits.

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m24/scripts/hpc/inspect_phase_c_incumbent_continuation_delta.sh
```

Inspection reports completion and produces a review archive with source
provenance, results, progress and logs. A repeated submission reuses confirmed
job IDs; an uncertain scheduler reply requires reconciliation, not resubmission.

Local qualification covers shared vector identity, identical restart vectors,
early stopping, incumbent protection, interruption charging, completed-call
telemetry, frozen-source rejection, evaluator separation and idempotent CPU
submission. A small real nonlinear smoke exercises profiling, independent
rollouts and exact finished-task resume. Full alien-device fitting remains a
Delta experiment; no empirical improvement on that benchmark is claimed yet.

Qualification on 2026-10-09: the full repository suite passed 4,394 tests with
eight Torch-dependent skips. After final verification-failure/count-accounting
guards, all 42 affected tests passed again. The final nonlinear smoke reached
training NMSE `5.083e-13` with incumbent-first (27 completed calls) and
`1.363e-11` with restart-first (127 completed of 128 started calls); both resumed
identically. The smoke uses small test-only budgets, not the benchmark budgets
above, and is a wiring check rather than comparative research evidence.
Changed-file Ruff, shell syntax and whitespace checks passed. Whole-repository
Ruff retains 37 pre-existing findings confined to `analysis/claude`; those
unrelated files were left untouched.

## Completed M24 review: `review-20261010-174649`

All six tasks completed on Delta using commit
`7fbbec1f72d228fc0702833ed974e62623033639` and plan
`562879ac15f0614cc69a7110536fe5dbe2d369f75687165cf581e7b2bd525391`.
The downloaded report reproduces exactly from the sealed inputs/results. All
159 sealed JSON records verify; each policy pair has the same warm vector and
restart roster. The two difficult seeds' corresponding short-restart evaluation
traces match exactly in parameters, loss and gradient despite differing runtime.
No interrupted search calls, domain-failure events or joint fallback stages are
recorded. No test observations or live LLM calls were used.

### Outcome: continue a useful incumbent, but preserve restart opportunities

Retrospective scores below use independent original-equation rollouts. Dynamic
coefficient error is the maximum relative error, expressed as a **percentage**;
hidden-initial error is maximum absolute error in physical coordinates. New
minutes include fitting, setup, verification and sensitivity, excluding the
shared historical warm cost and retrospective evaluation.

| Original start | Policy | Train NMSE | Validation NMSE | Max coefficient error | Max initial error | Completed search evaluations | New minutes |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | incumbent first | 1.6935e-14 | 2.2565e-14 | 0.0000550% | 2.002e-7 | 25 | 11.89 |
| 0 | restart first | 5.2893e-5 | 7.9733e-5 | 13.2756% | 0.08095 | 105 | 28.83 |
| 1 | incumbent first | 0.35383 | 0.55813 | 264.955% | 0.99659 | 105 | 30.77 |
| 1 | restart first | 0.40602 | 0.61070 | 243.018% | 0.99659 | 105 | 27.90 |
| 2 | incumbent first | 0.49699 | 0.34627 | 1516.725% | 1.91241 | 105 | 31.15 |
| 2 | restart first | 0.39292 | 0.35737 | 1190.765% | 1.05643 | 105 | 27.91 |

Start 0 is the decisive positive result. Its incumbent training error falls
from `5.2893e-5` to `1.6935e-14` in 25 additional evaluations and 466.97 search
seconds. Original-equation DOP853/Radau checks independently satisfy both the
mean training target and the worst-trajectory target (maximum trajectory NMSE
`4.2653e-14`). Restarts are therefore skipped. All thirteen dynamic coefficients
and five shared hidden initials recover accurately in the retrospective check.
Restart-first instead spends its 105 evaluations continuing a much worse new
start, ending at 0.36736; conservative retention correctly keeps the original
`5.2893e-5` warm incumbent. Incumbent-first uses 58.8% less new fitting/check time
for this pair and achieves much better accuracy. This supports the allocation
fix, not a new profiling formula.

Start 1 improves from 0.40602 to 0.35383 with incumbent continuation, but remains
inaccurate and its maximum coefficient error actually increases. Start 2's
incumbent continuation barely helps (`0.73682 -> 0.73493`); a subsequent short
restart supplies its retained 0.49699 result. Restart-first gives that same
restart a long continuation and reaches 0.39292. Its validation score happens
to be slightly worse than incumbent-first's, but validation does not choose
the policy or retained model. Neither start-2 endpoint is accurate. Thus an
unconditional "always continue the incumbent" rule is not supported.

Recovery of both coefficients and hidden initials is **1/3 original starts**
for incumbent-first and **0/3** for restart-first. The six-state nonlinear
problem still has thirteen dynamic coefficients and five shared hidden initials;
exact terminal-output profiling eliminates four gains, leaving fourteen outer
unknowns. The skeleton and data are unchanged. These are three starts on one
anchored, noiseless development block, not independent benchmark families or
evidence that generic initialization is solved.

### Failure diagnosis and fitting assessment

Every poor search stage reaches its evaluation ceiling: 15 evaluations per
short restart or 60 for continuation. All started evaluations finish; none is
cut off by the wall guard, unlike the M23 warm-stage comparison. They do not
terminate by gradient/step convergence. Incumbent-continuation final projected
gradient infinity norms are 0.713 for start 1 and 1.499 for start 2, with no active
outer bounds at those points. Start 2's profiled inner gains do include active
lower bounds. Across the last five evaluations the incumbent losses change
very little despite non-small recorded gradients. This warrants an optimizer
and derivative diagnostic; it does not establish a derivative bug, a local
minimum, or nonidentifiability. Saved evaluation traces include rejected trial
points and do not reconstruct accepted-step or trust-region history.

Independent solvers agree on the poor endpoints, so their large losses are not
an artifact of coarse collocation or failed verification. No ultimate-precision
retry is needed. Joint sensitivity includes all eighteen unknowns and has local
rank 18 at all retained endpoints; condition numbers range from approximately
779 to 41,495. Full local rank does not certify global uniqueness or optimizer
success. Only start 0/incumbent-first receives
`numerical_target_reached / strictly_verified / retain_with_local_evidence`.
The others remain `above_numerical_target` and recommend further search.
Likelihood profiles were not minimized in M24, so no calibrated confidence or
coefficient-accuracy guarantee is reported to the fitter.

### Costs and next proposed diagnostic

Summed new fitting/check wall time is 73.81 minutes for incumbent-first and
84.63 for restart-first (235 versus 315 completed search evaluations). This is
12.8% less new time overall, driven by start 0's early stop; starts 1 and 2 cost
more under incumbent-first. Verification costs 16.16/17.92 minutes and sensitivity
0.91/0.87 minutes respectively, already included in those totals. Retrospective
scoring adds 12.59/11.84 minutes. Historical warm costs are 13.76, 13.26 and 13.06
minutes, charged once per seed rather than once per arm. These are sums of task
times, not queue latency, and do not imply identical evaluation speed.

Keep incumbent continuation and early stopping. Before a larger restart sweep,
use the saved difficult endpoints for a bounded **stagnation diagnostic**:

1. Compare the supplied profiled residual Jacobian against finite-difference
   directional derivatives at several step sizes, recording inner active sets
   so nonsmooth transitions are distinguished from incorrect derivatives.
2. If derivatives check out, compare additional continuation with the current
   scaling against a predeclared alternative parameter scaling, from identical
   saved points and with matched evaluation limits. Record accepted-step progress,
   step norms and predicted/actual reductions where the solver exposes them.
3. Preserve independently verified incumbents and a distinct restart allowance.
   A later adaptive allocator can use recent training improvement per evaluation
   to divide work between useful continuation and genuine exploration, instead
   of exhausting a fixed long block on the almost unchanged start-2 incumbent.

These are recommendations, not implemented changes. No new remote fitting was
launched during this review; production defaults, benchmarks and prompts remain
unchanged.

Review verification: exact report regeneration, sealed-record and matched-start
checks, and 37 focused continuation/numerics/profiling tests passed. Whitespace
checks passed. Whole-repository Ruff still reports the same 37 unrelated
`analysis/claude` findings. This review changes documentation only.
