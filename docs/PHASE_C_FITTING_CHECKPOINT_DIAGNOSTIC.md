# Phase C M8: cheaper checkpoints and explicit evaluation failures

Implemented 2026-10-04 (Pacific/Honolulu). Protocol
`phase-c-fitting-checkpoint-diagnostic-1`. This follows the
[M7 results](PHASE_C_FITTING_M7_RESULTS_2026-10-04.md).

## M6 and M7 did not use different difficulty levels

Both studies use exactly the same sealed basin and alien-device development
arrays, correct equation templates, fixed internal blocks and three generic
starts. Their input artifact SHA256 is
`97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`.
M7 changes numerical policies; it is not a new or harder benchmark.

- M6 compared rollout-only, collocation-plus-rollout, adaptive collocation,
  adaptive shooting, and two cached-primal policies.
- M7 isolated dense versus reduced fixed collocation, exact versus limited-memory
  shooting Hessians, and rollout continuation versus a triggered generic restart.
  It removed adaptive enlargement after an unfinished native solve and charged
  new single-point derivative profiles to the native budget.
- Basin passes 18/18 fits in each study. Alien rollout-only in M6 and both
  rollout arms in M7 recover two of three starts. Those successful parameter
  vectors are identical. M6's two hybrid successes also use rollout from the
  original generic guesses; collocation did not supply the successful starts.
- M6's other alien strategies did not recover the model. M7 improves measured
  numerical progress in some native methods but still does not recover it.
  There is no successful-to-unsuccessful transition on an identical strategy
  caused by a more difficult dataset.

The earlier all-success results on smaller controls/CSTR are a different
comparison. The alien study uses six states, five hidden initials and 13 free
dynamic coefficients fitted jointly across 16 trajectories, with only one output
observed. Its poor starting region remains difficult despite local full rank at
the reference vector.

The alien arrays come from the actual Phase C hard alien-device cell. However,
this fitting qualification is deliberately assisted: the correct six-state
structure and some internal couplings/nonlinear shapes are supplied and fixed.
The full discovery task does not know these in advance; its constructed models
may have wrong structure, redundant latent coordinates or weakly identifiable
coefficients. Success here is necessary evidence about the fitter, not proof of
successful end-to-end hard-benchmark discovery. Basin is an easier regression
control with two free coefficients and supplied initial conditions.

## What changes

### Compact checkpoints, opt-in for fixed meshes

The M7 legacy callback repeatedly extracts all node trajectories, computes
off-node diagnostics and serializes the retained history. It consumed 55--59%
of recorded native-worker time on dense alien collocation.

The compact callback records the parameter/shared-initial vector, discretized
observation loss, maximum equality defect, iteration and time on each iteration.
It keeps the existing bounded pool: six recent distinct parameter vectors plus
the least-defective and best feasible candidates (at most eight records).
Only complete training rollouts select the fitted incumbent, using the same
screening rule in both arms.

Detailed node trajectories and off-node indicators are computed once on ordinary
native exit and saved separately as `fit/mesh-0/final_checkpoint_diagnostics.json`.
If the native process is killed before that write, detailed final diagnostics
may be absent; previously written compact parameter candidates survive. Compact
records do not serialize IPOPT internals or support a latent-node warm restart.
Consumed optimization budgets are never silently restarted.

This policy is restricted to fixed single solves. Adaptive refinement and cached
warm-start protocols retain their previous node-rich checkpoints. No benchmark,
equation, observation weight, domain, tolerance, mesh or production default changes.

### Replay failures and partial metadata

Independent replay catches numerical exceptions and records which solver/row
failed. Complete rows and individual solver scores remain available. A split
with incomplete coverage has null aggregate NMSE; partial data cannot receive
an accuracy pass. A replay timeout is distinct from the fitting budget status.

M8 journals replay identity and progress before/after integration. A completed
journal is reused. An interrupted journal is finalized as unavailable without
further integration, preserving already completed rows and preventing a fresh
evaluation allowance on resume. Its identity binds parameters, equations, both
data splits and the replay allowance. Numerical failure handling also benefits
the shared replay function; existing saved results remain untouched.

Interrupted rollout reports retain the restart decision, completed first-phase
metadata, and counts of completed training-history entries by phase. Total
attempted calls remain unknown if work was in flight. Native partial reports
recover available layout/profile/progress files even if screening was interrupted
before the normal stage-summary write.

## Matched experiment

**36 CPU fits**: two unchanged cases, three original starts, three native methods,
and two checkpoint policies per method:

| Method | Policies |
|---|---|
| Fixed dense collocation | legacy / compact |
| Fixed reduced collocation | legacy / compact |
| Fixed shooting with limited-memory Hessian | legacy / compact |

Within each pair, fitter payloads differ only in checkpoint mode. The whole-fit
ceiling remains 900 seconds, independent replay 300 seconds, native iteration
limit 400, and native allocation approximately 75% of remaining fit time. The
same M7 meshes, tolerances, derivative profiles and training-screening policy
are used. One CPU/16 GB per job, concurrency two; no GPU or LLM requests.

Compact mode may save more intermediate candidates because its callback runs
on every iteration. Thus this compares checkpoint policies, including their
retained pools, rather than timing an identical sequence of saved points.
Wall-limited native iteration counts and screening counts may also differ.
Report all starts; do not select a winning seed using validation/reference error.

Inspect callback time, iteration count, native convergence, screening cost,
training/validation rollout accuracy and coefficient/initial recovery separately.
M7's rollout/restart and exact-Hessian methods remain available but are not
repeated here. Better recovery is an experimental outcome, not an assumed result.

## Delta commands

Upload `phase-c-fitting-m8.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`, then:

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
AF_CODE="$AF_BASE/code/fitting-m8"
mkdir -p "$AF_CODE"
tar -xzf "$AF_BASE/phase-c-fitting-m8.tar.gz" -C "$AF_CODE"
unset AF_MATCHED_SOURCE AF_FITTING_INPUTS
export AF_CAMPAIGN="$AF_BASE/fitting-checkpoint-diagnostic-v1"
export AF_CONCURRENCY=2
bash "$AF_CODE/scripts/hpc/submit_phase_c_fitting_checkpoint_delta.sh"
BASH
```

The existing Delta fitting Python and CasADi environments are reused. The portable
bundle contains committed source, verification tests and the unchanged sealed
M6/M7 inputs. No old result files are needed for this new experiment.

After the jobs finish:

```bash
export AF_CAMPAIGN=/work/hdd/bibo/yxiao2/phase_c/fitting-checkpoint-diagnostic-v1
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m8/scripts/hpc/inspect_phase_c_fitting_checkpoint_delta.sh
```

This prints status/results and creates a review archive. Resubmission reuses
confirmed scheduler receipts; uncertain replies require reconciliation. It does
not restart consumed fitting budgets.

## Close out the one M7 timeout without refitting

Optional accounting on the existing M7 directory, using the new bundle:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m8
export PYTHONPATH="$AF_CODE/src:$AF_CODE:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
export PYTHONDONTWRITEBYTECODE=1
/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  "$AF_CODE/scripts/closeout_fitting_replay_timeout.py" \
  --root /work/hdd/bibo/yxiao2/phase_c/fitting-numerical-diagnostic-v1 --index 31
BASH
```

This verifies the old plan, backend identity, scheduler identity and saved timeout
log. It writes a sealed `evaluation-closeout/task-31.json` and a separate
`summary-closeout.json`: 35 complete evaluations plus one unavailable evaluation.
Original `summary.json`, fitted parameters and scores are preserved. Published
group metrics remain unchanged; the missing backend's cost and saved training
screen are recorded separately. No fitting or integration runs. This is a post-hoc
accounting correction, not an independent replay result or accuracy estimate.

## Exit criteria and limitations

Local verification: the full repository run completed with **3,924 passed,
eight skipped**. All 12 final M8 regression tests pass separately. The six-arm
small-control smoke passes prediction and coefficient recovery and exact resume;
worst validation NMSE is 5.225e-8. Preflight verifies all 18 paired payloads differ
only in checkpoint mode and use the same input artifact as M7. The timeout closeout
was exercised twice on the downloaded M7 copy, preserving the original summary and
backend bytes. Changed Python files and shell syntax pass lint/checks; repository
Ruff reports 37 existing findings in unrelated `analysis/claude/` files.

The local six-arm control must pass independent replay and exact completed-result
resume. Regression tests cover compact interruption persistence, pair isolation,
replay timeout and solver failures, partial-row preservation, journal identity,
no duplicate fit/replay budget, partial restart metadata and historical closeout.

The remote comparison must account for every planned endpoint, including numerical
failures. It should establish whether checkpoint overhead falls and whether the
freed time improves optimization. It does not yet introduce better coefficient
parameterization, a new restart distribution, adaptive mesh certification, or
faster shooting sensitivities. Those remain follow-ups guided by the results.
