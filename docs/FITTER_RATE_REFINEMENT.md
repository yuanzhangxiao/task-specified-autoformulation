# Saved-start rate refinement (Delta CPU diagnostic)

This milestone tests the v4 collocation-to-refinement handoff. It does not change
collocation, proposer behavior, or production fitting defaults.

In six alternating runs, a near-zero decay coefficient became a time constant
near the largest finite float. Refinement in physical time constants then stopped
after two calls with large optimality values. Differentiating through the
reciprocal loses the rate direction numerically. This experiment rewrites simple
declared time-constant divisors into direct products with named rates before
symbolic differentiation. For example, `-p/tau_p` becomes `-p*rate_tau_p`.

## Frozen comparison

The default imports all 39 saved v4 refinement starts (C, J, A, including the
stress start). Each array task runs a physical-coordinate sensitivity fit and a
rate-coordinate sensitivity fit from the same saved point. There are 78 fits;
no initializer is rerun. Each arm receives `600 - saved_initializer_seconds`
seconds and the original 150-evaluation limit. Original initialization time is
reported separately; new fitting and verification times are not mixed.

The import requires the v4 source hash, launcher hash, normalized plan and task
matrix shipped at commit `3f3fdf43744ee4490e672106e52ddd97863b003d`. The Python
version of that historical experiment need not equal the new environment.
Imported candidate files, numerical guards, reference arrays, initializer arrays,
source results, and successful replay arrays are copied and hashed. A new
freeze records the current Python/package/source/launcher identity. Subsequent
execution requires that identity and every imported byte to stay unchanged.
The original directory can remain untouched or unavailable after preparation.

Both arms retain the same noisy training observations, fixed zero initial states,
ODE method/tolerances, observation residual objective, and native stopping rules.
A training-only check verifies equivalence at the saved point and compares direct
rate sensitivities with forward differences. Failed guards block both arms.

Bounds are mapped from the existing physical domains. A small positive rate floor
represents the largest finite float time constant; no truth-derived bounds or
exact-zero-rate model extension is introduced. The existing `1e-12` denominator
guard is retained in the effective-rate mapping. Signed offsets and nonnegative
gains retain their existing roles. Unsupported time-constant uses fail closed.

SciPy may move a starting point slightly inside its bounds. Results report the
nominal saved point, nominal training cost, and first evaluated point/cost. This
makes the adjustment visible; neither arm is claimed to start at a different
scientific model by design.

Rates are authoritative in the corrected arm, including independent replay.
The informational physical view reports `null` for a time constant whose
`rate * training_horizon` is below float64 relative resolution, with the actual
rate and explanation attached. This is a reporting label, not an optimizer
constraint or a declaration of scientific unidentifiability. No giant reciprocal
is passed back into refinement.

## Verification retry

Every source result marked `replay_unverified` gets a separate retry task. For the
reported v4 run this is one additional task (40 array tasks in total). It uses the
saved final physical parameters, identical Radau/BDF methods and tolerances, and
90 seconds per split instead of 30. Successful trajectories are copied with
explicit provenance. Only missing/failed trajectories are integrated again.
Neither fitting nor parameter selection occurs in the retry. The old incomplete
result is preserved.

New fits also receive the same 90-second-per-split verification allowance.
Clean train/validation scores are computed after fitting, never supplied to the
optimizer. `complete` means independent numerical verification; native optimizer
success and output recovery are separate fields. Gradient and optimality values
use different coordinate units and bound scaling, so their magnitudes are not
directly comparable between arms. The existing `1e-4` clean NMSE
threshold is descriptive, not a selection criterion. No benchmark/test data or
LLM calls are used.

## Execution and resume

Use a clean isolated checkout and the existing Delta Python/CasADi environment:

```bash
bash scripts/hpc/submit_fitter_rate_refinement_delta.sh
```

Defaults:

- Source: `/work/hdd/bibo/yxiao2/phase_b/fitter-methods-v4`
- Output: `/work/hdd/bibo/yxiao2/phase_b/fitter-rate-refinement-v1`
- Python: `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`
- CasADi: `/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps`
- One CPU, 8 GB, no GPU; at most two array tasks concurrently; 45 minutes/task.

The launcher freezes/imports before submission, rejects partial or duplicate
submissions, and schedules a summary with `afterany`. Worker process groups have
hard deadlines shorter than the Slurm limit. No dependency installation occurs.
`AF_SOURCE_ROOT`, `AF_OUTPUT_ROOT`, and `AF_ARRAY_CONCURRENCY=1` can override paths
and concurrency. `source_tasks` in an alternate config is an explicit subset for
local smoke testing; the default requires all 39 historical results.

Completed fits and replay trajectories are checkpointed and reused. Task locks
prevent concurrent mutation. Native optimizer state is not serializable. An
interrupted fit is reported as `interrupted_fit`, rather than silently restarting
with a fresh budget. Completed task results, including failed results, are stable
on resume; retain them and use a separately named follow-up run if another attempt
is needed. The importer and retry never erase or overwrite historical outputs.

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-rate-refinement-v1/summary.md
```

`summary.json` includes per-arm parameters, native stop reason, optimality,
initial/first evaluated points, clean scores, verification checks, and full retry
provenance. It is the detailed artifact for diagnosing failures.

## Interpretation and remaining limits

This isolates the coordinate handoff. It cannot establish that alternating
collocation finds as good an initialization as joint collocation. It also cannot
resolve the separate J solution with an effectively suppressed p pathway. Rates
can repair a numerical obstruction while leaving local optima, poor parameter
scaling, and finite-budget limitations. A successful synthetic correction does
not automatically promote any method into production.

## Local verification before the Delta run

Two full-size saved local initializers (moderate A and separated stress J) passed
all independent replay checks with both coordinate systems. Both arms recovered
clean outputs in each case. Rate refinement used 25 versus 29 calls for moderate,
and 13 versus 19 for the stress start.

A separate local regression used the exact moderate/noiseless extreme-time-
constant initializer pasted by the user. Both arms had a 120-second refinement
allowance. Physical refinement reproduced the two-call `xtol` stop (clean
validation NMSE 11.26); direct rates took 33 calls and reached approximately
`1.86e-20` validation NMSE. Independent replay passed in both cases. This supports
the coordinate-failure diagnosis, but is not a substitute for the complete
Delta paired comparison or evidence that alternating initialization is generally
competitive.

Repository verification: `pytest -q` passed 1,203 tests; three optional PyTorch
tests were skipped because PyTorch is not installed. `ruff check .`, shell syntax,
checkpoint/retry tests, and a scheduler-stub submission/resubmission smoke passed.
