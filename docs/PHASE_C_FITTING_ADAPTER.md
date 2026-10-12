# M28: common rollout fitting interface and bounded numerical verification

Implemented October 11, 2026. The [completed M28 review](PHASE_C_FITTING_M28_RESULTS_2026-10-12.md)
records accurate recovery in all 30 fits and successful fixed-parameter replay.
The fixed-scaling alternative never activates, so its empirical benefit remains
untested. This follows
the [M27 results](PHASE_C_FITTING_M27_RESULTS_2026-10-11.md). Guarded scaling
reached the training target on all fifteen tested case/start combinations,
including nonlinear CSTR fitting without output profiling. One accurate basin
endpoint exposed an independent-evaluator tolerance discrepancy.

## Scope and interface

`autoformalism.fitting.rollout_adapter.run(problem, starts, policy, folder)`
accepts a typed training-only problem, four frozen domain-only start vectors,
a typed policy and a durable checkpoint directory. It returns `FitResult` with:

- Retained parameters, including shared initializer coefficients, and origin.
- Symbolic kernel routing/certificates and reasons a profiling route did not apply.
- Both independent training NMSEs, worst-trajectory error, normalized solver
  disagreement, whether tighter verification was used, and target attainment.
- Separate fit quality, numerical status, search status and local joint
  parameter/initial sensitivity evidence via `FittingAssessment`.
- Actual search calls, block count, measured time, charged budget and explicitly
  incomplete accounting after interrupted operations.

There are no validation, test, reference coefficients or hidden-state trajectories
in this input schema. A distinct supplied incumbent is independently checked;
unsuccessful search branches cannot overwrite it. Initializer coefficients are
globally shared and remain fixed on validation. Completed reuse verifies source,
runtime, input, policy and backend identity. Interrupted operations retain their
charged allowances and are never silently rerun from zero.

This is an opt-in adapter, not a change to the public fitting dispatcher or the
constructor Astra is developing. Current limitations include explicit shared
parameterized initializers, supported restricted analytic expressions, and caller-
supplied numerical coordinates/start portfolios. Piecewise expressions require
`PiecewiseTrainingProblem`; derivatives at hinges are branch conventions, not
global smoothness certificates. Unsupported structures raise validation errors.
The qualification target remains for noiseless data; noisy fitting needs an
appropriate stopping policy. No calibrated confidence probability, full profile
likelihood interval or global identifiability guarantee is returned.

## Kernel routing and one shared search budget

The existing symbolic router tries exact affine-system profiling, then exact
terminal-output profiling. If neither certificate applies, it uses joint rollout
fitting of the parameters and shared hidden initials. It never approximates a
nonlinear coupled coefficient as an exact linear rollout gain. Numerical profile
failure can fall back to joint fitting within the same block's remaining call
and time allowance.

The new `guarded_portfolio` strategy starts each frozen initial vector with
guarded relative scaling. The base physical coordinates, normalized objective,
derivatives and independent retention rules remain those qualified in M27.

1. Continue guarded scaling when the latest block is making sufficient progress.
2. After a plateau, native stop, or insufficient progress evidence, try fixed
   scaling from that branch's verified point if a continuation slot is available.
3. Reserve one block for every unvisited restart. A fixed alternative consumes
   a continuation block; it does not add an allowance.
4. Leave an unverified branch and move to a genuinely different start. Keep the
   independently verified incumbent throughout.
5. Stop early only after both-solvers' aggregate and worst-trajectory checks pass.

Each task has six blocks maximum, forty charged evaluations or 800 seconds per
block, and at most two consecutive blocks per start. The original plus three
restart candidates are frozen before either arm runs. These are the M27 search
ceilings (240 calls, 80 minutes), not a new full budget for each scaling policy.
Actual work can stop earlier. The control `guarded_only` uses the same adapter,
router, input vectors, ceilings and checks, but never switches to fixed scaling.
Legacy M26/M27 methods and their serialized policies remain unchanged.

## Independent evaluator retry

`verified_replay.run` is outside the optimizer interface and receives the final
sealed vector. It first replays all train/validation trajectories with Radau and
DOP853 at rtol=1e-9, atol=1e-11, with a 300-second ceiling. If unavailable or
disagreeing, it gets exactly one reserved 300-second attempt at rtol=1e-12,
atol=1e-14. The normalized maximum prediction-discrepancy limit stays 1e-4.
This evaluator tolerance is distinct from the fitter's tighter training checks.

Both complete trajectories and finite metrics are required for agreement. Poor
fit with agreeing solvers does not trigger a numerical retry. A persistently
unavailable/disagreeing replay remains unverified. Both attempts, original errors,
fixed-vector identity, elapsed time and charges remain in the artifacts. There
is no parameter update, validation-specific initialization or fitting feedback.
An interrupted initial attempt consumes its full allowance; only the separately
reserved retry may run. Interrupted retries are not replaced with fresh retries.

Reports separate `coefficient_error_within_threshold` (retrospective arithmetic)
from `solver_agreement_passed` (numerical verification) and prediction accuracy.
An inaccurate coefficient vector cannot gain an accuracy claim simply because
its rollout is finite, and an integrator disagreement is not mislabeled as a
coefficient-estimation error. Original M27 records are never rewritten.

## Frozen qualification

Thirty new fits pair `guarded_only` and `guarded_portfolio` on the same three
known structures and five original generic starts per structure as M27:

| Case | States | Dynamic coefficients | Shared hidden initials | Expected kernel |
|---|---:|---:|---:|---|
| Alien hard | 6 | 13 | 5 | Terminal-output profiling; four gains eliminated |
| Nonlinear CSTR hard | 3 | 7 | 2 | Joint rollout |
| Coupled basin | 2 | 2 | 0 | Joint rollout with explicit piecewise opt-in |

An additional task replays the exact flagged M27 basin endpoint at fixed
parameters. Its result is reported separately; it is not a thirty-first fit and
cannot seed either fitting arm. The exporter copies the sealed M27 observations,
start lists and this endpoint. It does not regenerate data or change prompts.
The known reference is evaluator-only. There are no LLM calls, GPUs or test data.

Frozen exported input SHA256:
`eb1d681faa7d98af0d0130e010f7099513b71e041f41c0ae574632770ae142a2`.

Exit criteria: complete accounting; all outcomes retained in denominators;
fitter/evaluator separation and resume checks pass; equal maximum search budgets;
the combined policy is assessed against guarded-only for recovery, coefficient
and initial error, and total cost including verification. Any lost recovery is
investigated before changing a production default. The saved basin discrepancy
may not recur on every runtime; tests also inject persistent disagreement and
interruption to exercise the failure paths without claiming empirical recovery.

## Delta launch and inspection

Upload `transfers/phase-c-fitting-m28.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. It contains committed source, the frozen export,
`SOURCE_COMMIT`, `SHA256SUMS`, tests and launchers. On Delta:

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m28.tar.gz
bash phase-c-fitting-m28/scripts/hpc/submit_phase_c_rollout_adapter_delta.sh
```

The default root is `/work/hdd/bibo/yxiao2/phase_c/fitting-rollout-adapter-v1`.
The 31-task CPU array uses one CPU and 16 GB per task, a 2.5-hour scheduler
ceiling, and three concurrent tasks. `AF_CONCURRENCY` controls this array, not
the cluster limit. It reuses the existing Delta fitting environment; no new
installation is required. The launcher checks archive hashes before preparation.
Use the same launch command to resume confirmed submissions. An uncertain
scheduler reply blocks resubmission until the existing intent is reconciled.

Inspect and produce the review archive after completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m28/scripts/hpc/inspect_phase_c_rollout_adapter_delta.sh
```

Download the exact `review-*.tar.gz` path printed. The inspector can also report
partial results. Missing/unverified fits remain visible; Slurm completion alone
does not certify fitting accuracy.

## Verification

Tests cover shared fallback budgets, reserved restarts, incumbent preservation,
training-only schemas, worst-trajectory stopping, legacy behavior, parameter-
fixed retry, persistent disagreement, interrupted allowances, frozen tolerances,
sealed post-fit evaluation, diagnostic isolation and scheduler idempotence.
Local smoke checks use small synthetic equations and fixed-parameter replay;
the full benchmark fitting experiment is left for the user's Delta submission.

The local regression selection passes 105 tests across the adapter, independent
replay, campaign, prior recovery/scaling paths, sensitivity checks and both
profiling kernels. All changed Python files pass Ruff; shell syntax and Git
whitespace checks pass. Repository-wide `ruff check .` still reports 37 existing
issues in four unrelated `analysis/claude` files, which were left unchanged.

Six actual small synthetic fits (both strategies across affine profiling,
terminal-output profiling and joint rollout) reach their training targets and
resume exactly. Their NMSEs range from approximately 1.2e-20 to 1.5e-14. These
checks establish interface operation, not performance on the full M28 cases.

A fixed-parameter basin smoke deliberately limited the initial evaluator to one
second, then allowed 120 seconds for the reserved tighter replay. The incomplete
first attempt remained recorded; the retry completed with train/validation NMSE
1.56e-16/2.65e-16 and maximum normalized solver discrepancy 2.71e-9. Both attempts
cost 39.34 seconds combined, with complete accounting and exact terminal resume.
This tests retry execution under an artificially short initial allowance; it
does not replace the original M27 discrepancy or the 300+300-second Delta test.
