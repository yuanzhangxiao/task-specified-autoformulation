# M35: fixed-scale Huber measurement fitting

Implemented 14 October 2026 as an opt-in continuation of
[M34 fixed-noise fitting](PHASE_C_FITTING_MEASUREMENT.md). The 96-attempt Delta
qualification completed on Delta in the October 15 review. M34 fixtures, campaigns,
benchmark data, finalized prompts and production defaults are unchanged.

## Objective and interpretation

The public measurement problem retains its raw readings, acquisition identities,
known positive sigma, fixed units/calibration, masks and preparation contract.
For standardized residual `r = (prediction - observation) / sigma`, the explicit
Gaussian arm minimizes `sum(r**2 / 2)`. The Huber arm minimizes

```
rho_delta(r) = r**2 / 2                       if abs(r) <= delta
               delta * (abs(r) - delta / 2)   otherwise
L_H = sum(rho_delta(r)),  delta = 1.345.
```

The threshold is frozen before fitting and is never tuned against clean truth or
corruption labels. Huber is an estimation objective; its value is not a Gaussian
negative log likelihood or a probability of sensor failure. Standardized quadratic
loss, acquisition-unit and analysis-unit residuals, individual robust contributions
and working weights remain available. The descriptive Gaussian quadratic is not used to
select a Huber endpoint. The reported working weights limit influence; they do
not identify which readings are faulty.

Explicit `Loss(kind="gaussian" | "huber", delta=1.345)` selects the new execution
identity and reporting contract. Omitting the loss retains M34 behavior and
identity. Objective, data, units, preparation, coordinates, starts, policy,
source and runtime are bound to checkpoint identities. A changed loss or delta
cannot reuse an earlier fit.

## Numerical procedure

Direct fitting uses a residual transformation whose sum of squared values equals
twice the declared objective, with its exact chain-rule derivative. The existing
bounded optimizer therefore minimizes the actual Huber objective. Incumbent and
candidate checks use original-equation DOP853 and Radau rollouts, including exact
or shared initial conditions. Selection compares the same declared objective
under both independent solvers, preserving a verified incumbent on failure or
ties. Raw residual agreement remains a separate numerical check.

Eligible profiles preserve the affine output certificate and fixed whitening.
The bounded Huber inner problem is solved as a convex objective using L-BFGS-B
and bounded Newton polishing. A least-squares solution supplies only a starting
point. Acceptance requires an explicit bounded KKT check to its recorded numerical
tolerance. The outer derivative
uses the inlier Hessian and the full Huber score, including the design-derivative
term. Rank loss, threshold proximity and unstable active constraints can make
the profile derivative unavailable. Those cases fall back to joint fitting
within the remaining allowance. A single reweighting step is never represented
as an exact profile. Inner evaluations, iterations, time, convergence, active
bounds and fallback reasons remain in the audit.

All four arms share the same supplied start, incumbent, deterministic domain-only
restart and policy: 30 seconds and 100 search calls, one restart, and 15 seconds
of independent verification. Inner work consumes the search wall allowance and
is reported separately, without charging its time twice. Interrupted work retains
unknown usage and its original budget charge; resume does not grant it a new
allowance. The campaign seals the fitted endpoint before opening its evaluator,
and journals evaluation before execution so interrupted evaluation is not silently
repeated.

## Frozen qualification

Eight cases, three acquisition seeds, two losses and two strategies produce
96 attempts. Every case uses isolated synthetic acquisitions with known sigma;
these are not modified benchmark observations.

| Cases | Purpose |
| --- | --- |
| `clean_exact`, `sparse_outliers` | Matched Gaussian readings with exact independent initial preparation, then four prespecified target perturbations |
| `shared_clean`, `shared_outliers` | The same comparison with one unknown initial value shared by both trajectories |
| `nonlinear_clean`, `nonlinear_outliers` | The same comparison with a fitted nonlinear decay coefficient and a profiled gain |
| `wrong_gain_bound` | Uncorrupted readings with an upper gain bound excluding the generating gain |
| `wrong_dynamics` | Uncorrupted readings with fixed candidate decay 0.25 instead of generating decay 0.65 |

“Clean” means uncorrupted Gaussian observations, not noiseless measurements.
“Exact” refers to independent preparation. Each contaminated case reuses the
same underlying clean acquisition and changes four of fifty target readings by
`+18 sigma`. Public sigma, calibrations, initial rules, candidate, starts and
budgets stay paired. Original raw artifacts and actual transformed input bytes
are retained and hashed. Corruption locations, original values, clean predictions
and generating parameters live only in the separate evaluator records. The
public candidate identity contains no clean/corruption labels.

Preparation generates inputs once. Later prepare/verify/run/report calls read the
same sealed `inputs.json`; they do not regenerate it on resume or another host.
Importing a sealed input export preserves its exact values. Plan and per-task
identities reject changed data, loss, policy, source/runtime, copied-arm results
and evaluations from another endpoint. Scheduler intents and receipts prevent
duplicate resubmission when an earlier scheduler response was uncertain.

The independent post-fit evaluator uses closed-form candidate predictions.
For affine cases it solves each loss's own bounded convex reference, using a
scalar derivative root or a separate bounded optimization with a projected KKT
check. Nonlinear cases have no asserted global optimum; they retain analytic
endpoint prediction checks and matched direct/profiled comparisons. Clean RMSE
and coefficient errors are evaluator-only. Neither selects a fit or tunes delta.

Reports compare direct/profiled objective values only within one loss. They show
clean/contaminated prediction and coefficient shifts for each loss, preserve all
missing/failed denominators, and aggregate known cost and unavailable counters
by loss and strategy. Wrong-bound and wrong-dynamics controls retain raw residual
and descriptive observed NMSE evidence even when Huber downweights many readings.
There is no automatic scientific mismatch or data-fault diagnosis. Better Huber
performance on this sparse-contamination construction is not a universal claim.

## Local verification and limits

The local integration smoke ran all eight seed-zero cases across all four arms:
32 actual fits passed independent rollout verification and objective replay;
all 24 applicable affine optimum checks passed. Nonlinear direct/profiled
objectives agreed. All 84 campaign/fixture tests passed, covering immutable raw
inputs, evaluator isolation, loss identities, failed denominators, terminal and
interrupted resume, copied backend/evaluator rejection, scheduler receipts and
shell/CLI entry points. An actual shared-initial contaminated Huber profile also
passed the full prepare/verify/run/identical completed-resume/report workflow.
Remote numerical results are recorded below.

## Delta qualification results

The review `review-20261015-001905.tar.gz` binds commit
`03e8b2b2b697d7512e5d7afe52b2b5f60d403ea1` and plan
`e40ea0c2543386e15b1be63b681781e2d35a13ada6834fee3685bc4f90be63ee`.
All 96 attempts completed and passed independent rollout and analytic prediction
checks. All 72 applicable affine own-loss optimum checks passed. All 48
direct/profiled pairs agreed, with maximum objective difference 2.93e-11 and
parameter difference 1.91e-8. Recomputed saved-residual objectives differed by at
most 2.84e-14. One Gaussian nonlinear restart exhausted its allowance; the
verified first-start endpoint was retained. No profile fallback occurred, so
these remote cases do not exercise the fallback tests.

Median noise-free prediction RMSE across three seeds, using the profiled arms:

| Contaminated case | Gaussian | Huber |
| --- | ---: | ---: |
| Exact preparation | 0.05202 | 0.001008 |
| Shared unknown initial | 0.05819 | 0.008593 |
| Unknown nonlinear decay | 0.05401 | 0.009162 |

Noise-free trajectories were evaluator-only. All nine contaminated case/seed
combinations improved prediction RMSE with Huber; clean noisy cases did not
uniformly favor either loss. Individual coefficients need not improve: nonlinear
decay error increased from 1.46% to 5.08% at seed 0 and from 9.42% to 13.27% at
seed 1. The wrong-bound and wrong-dynamics controls retained substantial mismatch.
Huber downweighted 46–48/50 readings under the incorrect bound without any added
outliers; downweighting cannot establish data faults.

Across 24 Huber fits, profiling reduced search calls from 930 to 256, recorded
CPU time from 42.64 to 28.87 seconds, and summed stage wall time from 148.99 to
81.30 seconds. Robust inner solves consumed 0.86 seconds, included in those
totals. These are small controls, not full benchmark or calibrated-confidence
results. The next bounded experiment is
[M36 shared sensor drift](PHASE_C_FITTING_SENSOR_DRIFT.md).

Final targeted verification: 360 measurement, robust-fitting, profiling and legacy
adapter tests passed. Changed-file Ruff, shell syntax and whitespace checks pass.
Repository-wide Ruff reports 37 existing errors in unrelated analysis files.
The full 5,249-test suite was attempted with a five-minute local limit; it did
not finish (1,147 passing outcomes, five skips and three failures before the
limit). All three failed tests passed on a separate rerun after implementation
edits stopped. This is not a claim that the complete repository suite passed.

This milestone does not estimate sigma or contamination probabilities, calibrate
confidence intervals, relax arbitrary initial conditions, infer sensor faults,
or automatically revise scientific specifications. The controls are small and
their outlier mechanism is deliberately limited. The nonlinear outer problems
have no global-optimum certificate. Huber's influence bound cannot distinguish
measurement error from an incorrect model.

## Delta launch and inspection

Use the pinned `phase-c-fitting-m35.tar.gz` bundle. From Delta, after uploading
it to `$HOME`:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/robust-measurement-fitting-m35.XXXXXX)
tar -xzf "$HOME/phase-c-fitting-m35.tar.gz" -C "$AF_CODE"
cd "$AF_CODE"
sha256sum -c SHA256SUMS --quiet
bash scripts/hpc/submit_phase_c_robust_measurement_fitting_delta.sh
BASH
```

The default output root is
`/work/hdd/bibo/yxiao2/phase_c/robust-measurement-fitting-v1`. The existing Delta
CPU environment and dependency overlay are reused, with one CPU per attempt and
two concurrent array tasks. Preparation runs the M35 numerical and campaign
tests on a compute node before the array. No GPU, API key, provider call or test
dataset is needed. Resume from the same extracted code and output root, preserving
the submission journal.

From that extracted directory, inspect and package the results:

```bash
bash scripts/hpc/inspect_phase_c_robust_measurement_fitting_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/robust-measurement-fitting-v1
```

The inspector prints scheduler status, numerical comparisons, paired contamination
effects and costs, then creates a review archive containing the plan, exact inputs,
summary, fit/evaluator journals, scheduler receipts and logs.

For local or manual execution, use the notebook-independent CLI:

```bash
python scripts/phase_c_robust_measurement_fitting.py prepare --root /tmp/m35-study
python scripts/phase_c_robust_measurement_fitting.py verify --root /tmp/m35-study
python scripts/phase_c_robust_measurement_fitting.py run --root /tmp/m35-study --index 0
python scripts/phase_c_robust_measurement_fitting.py report --root /tmp/m35-study
```
