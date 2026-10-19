# M44: choosing an observation-prefix duration

Implemented 19 October 2026 as an opt-in fitting experiment. The shared interface
is [the initialization contract](PHASE_C_OBSERVATION_PREFIX_INITIALIZATION.md).
M43 established that shared constants, dependent initial maps and new-trajectory
root inference work on a small known-equation control. M44 asks how much observed
history is needed before forecasting. It changes no benchmark, finalized prompt,
historical campaign or production default, and uses no LLM or test data.

## Decision rule

Prefix duration is a fitting hyperparameter. The proposer declares the root scope
and initializer dependencies, but does not choose an evaluation window. On each
training acquisition we fit the shared dynamics, shared initializer coefficients
and independent training roots once. Those shared quantities and the normalization
remain fixed for every subsequent duration and validation inference.

The duration grid is 0, 0.1, 0.2, 0.5, 1 and 2 physical time units. Each prefix is
fitted independently from the same two predeclared root guesses, 0.5 and 1.1,
with equal numerical budgets. There is no warm start from a longer prefix or a
full-trajectory fitted root. The winner minimizes **prefix loss only**, with
predeclared start order breaking ties. Both starts are forecast to expose solver
sensitivity; neither is chosen by its validation suffix score.

Let p_j(t; tau, s) be the prediction from start s, using only observations through
tau. Let s_hat minimize its prefix objective and let S be the frozen training
standard deviation. All durations use the same scoring set H={t_k > 2}:

    L_j(tau) = mean_{k in H} [(p_j(t_k; tau, s_hat) - y_j(t_k))/S]^2
    L(tau)   = mean_j L_j(tau)
    B(tau)   = max_j sqrt(mean_{k in H}
                           [(p_j(t_k; tau, 0.5)-p_j(t_k; tau, 1.1))/S]^2).

The shortest adequate duration satisfies L <= 0.01 and B <= 0.01. These are
predeclared operational tolerances, not statistical confidence thresholds. All
four validation rows and both starts must have verified results for every
duration; a missing or failed attempt produces `unavailable`, not a reduced
denominator. A complete sweep with no adequate duration produces the separate
outcome `no_adequate_window`. Root uncertainty is reported alongside the choice,
not used to veto equally predictive parameterizations automatically.

Each task reports its conditional choice. A conservative recommendation is also
reported per noise regime: the shortest duration meeting both tolerances in
**every** acquisition and training start in that regime. The report never picks
the most favorable training start or noise draw. Missing task results prevent a
regime recommendation.

The choice concerns the common **late horizon t>2**. It does not certify forecast
accuracy immediately after a shorter chosen prefix. Scores over each duration's
own suffix are reported descriptively, but do not enter selection. This prevents
longer windows from winning simply because their evaluated horizon is shorter.
Validation is being used for tuning; its selected score is not an independent
generalization estimate. A later test must freeze the complete policy first.

## Six tasks and three acquisitions

M44 retains M43's known three-state equations:

    x' = -theta*x + z + r,   z' = -z,   r' = -2*r,   y=x,
    x_j(0)=y_j(0),           z_j(0)=phi*r_j(0).

Theta and phi are shared unknowns; each trajectory has one independent root
r_j(0). Theta is restricted to [0.1,0.8] and phi/roots are nonnegative under the
existing coefficient-domain policy. The grid is 0..4 in steps of 0.1.

The acquisition cases are:

- One clean dataset.
- Two separate acquisitions with independent Gaussian measurement noise,
  declared standard deviation 0.01 in measured-output units.

Each acquisition is fitted from the same two predetermined training starts,
giving six tasks, not six independent noise replicates. Each task has three
training trajectories and four validation trajectories. Their first observed
value is identically 0.4; their hidden roots differ. The noisy cases keep the
first observation exact, consistent with the current boundary contract. Noisy
time-zero estimation remains a separate extension.

Generating values are used only to create the synthetic measurements and in a
post-selection evaluator. Training roots are (0.4,1.2,1.6), validation roots are
(0.65,0.95,1.45,1.8), theta=0.35 and phi=0.7. The last validation root is outside
the training root range. Actual measured arrays are frozen once and reused across
training starts and prefix durations; resume never redraws noise. Clean signals
and root values are not passed to fitting or duration selection.

## Prefix information and verification

At each chosen prefix endpoint, a separate diagnostic computes the Jacobian J of
the training-normalized prefix residuals with respect to the physical root. It
reports its singular value, local numerical rank, and across-start root spread.
For positive known measurement noise sigma, the one-root noise amplification is

    delta_r_noise = (sigma/S) / ||J_{t>0}||_2.

This describes the root displacement corresponding locally to one noise unit in
the residual norm. Exact t=0 readings are excluded from this noise calculation.
Zero sensitivity is `uninformative`, not zero uncertainty. With sigma=0 the
noise-unit scale is not reported; the sensitivity and rank remain available.
The quantities are conditional on the learned shared coefficients. They do not
include uncertainty in theta/phi, prove global identification, or define a
calibrated confidence interval. A diagnostic failure remains visible and does
not silently become evidence that the root is constrained.

Independent integration checks verify every numerical fit and forecast. After
the duration decision is sealed, the evaluator records root error, clean-signal
suffix RMSE and agreement with the analytic trajectory at fitted values. These
are retrospective qualification metrics; none select the start or window.

## Execution, accounting and failure behavior

The implementation is in `fitting/prefix_window_selection.py` and
`fitting/prefix_window_diagnostics.py`. The existing prefix kernel adds an optional
roots-only start override; native domains are checked and shared-parameter edits
are rejected. The historical default inference result remains unchanged.

Training has at most 100 search calls / 30 seconds, plus 10 seconds for numerical
verification. Each prefix start has at most 100 search calls / 10 seconds, plus
5 seconds verification. Each forecast and sensitivity diagnostic has a separate
5-second allowance. Search, verification and diagnostic calls have separate
counters. Bounded-operation CPU and wall costs include setup and progress writes;
final receipt sealing, outer task/report writes and scheduler waiting are excluded.
Oracle calls may contain multiple integrations and are not RHS evaluation counts.

Every bounded operation has durable start/progress/result records. Completed
operations are reused exactly. An interrupted operation remains charged without
automatic retry or a budget reset; native optimizer-state continuation is not
claimed. Reporting verifies identities, start choices, forecast receipts, scores,
metrics and selection against saved arrays. New code/policy/runtime requires a
fresh experiment root. Scheduler submission intents prevent blind duplicate
submissions after uncertain replies.

The report's `status: complete` means all expected terminal records exist. Check
`status_counts` and `selections` for numerical success and an adequate window;
recording a failure does not qualify its result.

Delta runs six CPU array tasks, one CPU and 16 GB each, concurrency two by default,
with a 30-minute scheduler limit per task. The limit covers all bounded stages;
it is not a target runtime. No GPU is requested. This campaign is self-contained
and does not need a prior archive, remote repository access or new dependencies.

## Run on Delta

Upload `phase-c-fitting-m44.tar.gz` to the path assigned to AF_BUNDLE, then run:

```bash
AF_BUNDLE=/work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m44.tar.gz
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/prefix-window-m44.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_CODE"
cd "$AF_CODE"
export AF_PREFIX_WINDOW_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-window-selection-v1
export AF_ACCOUNT=bibo-delta-cpu
export AF_CONCURRENCY=2
bash scripts/hpc/submit_phase_c_prefix_window_selection_delta.sh
```

The launcher uses the existing `autoformalism-v21` Python and
`fitter-methods-v1-deps` packages. After the report job finishes:

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/prefix-window-selection-v1
jq '{status, expected, recorded, status_counts, selections, cost}' "$AF_ROOT/summary.json"
AF_REVIEW="$AF_ROOT/review-$(date +%Y%m%d-%H%M%S).tar.gz"
tar -czf "$AF_REVIEW" -C "$AF_ROOT" plan.json inputs.json summary.json results
printf '%s\n' "$AF_REVIEW"
```

This measures duration choice on a deliberately small known-equation control.
General nonlinear/multiple-root initializers, uncertain first readings, weak
shared parameters, prospective control before the prefix ends, and a real
construction-export-to-fitting benchmark remain separate qualifications.

## Local smoke results

All six local tasks completed under the frozen default policy. Both training
starts selected duration 0.1 on the clean acquisition; all four noisy runs selected
0.2. The conservative recommendations agree. At duration 0, root rank was zero
and the two starts produced different forecasts. Clean nonzero prefixes recovered
the root accurately. For noisy acquisitions, the common-horizon mean NMSE was
0.0148--0.0165 at duration 0.1 (above the declared 0.01 limit), and 0.00242--0.00321
at duration 0.2. These are validation tuning results, not independent test scores.

In noisy runs the conditional one-noise-unit root scale fell from 0.0648--0.0676
at duration 0.1 to 0.0312--0.0326 at 0.2, and to 0.00353--0.00371 at 2. All these
nonzero prefixes have rank one: the continuous information diagnostic adds detail
that a full-rank flag alone cannot provide. Agreement between optimization starts
does not remove measurement uncertainty.

The complete local ledger records 732 operations, 1,110 search calls, 1,164
verification oracle calls and 144 additional sensitivity calls. Summed operation
CPU was 121.40 seconds and wall time 124.41 seconds. All costs were observed;
there were no interrupted stages. Six terminal replays preserved the bytes and
timestamps of all 2,205 JSON artifacts. The smoke artifacts remain uncommitted in
`output/phase-c-fitting-m44-smoke/`. The returned Delta results follow below.

## Returned Delta review, 19 October 2026

The replacement archive `review-20261019-131033 (1).tar.gz` is complete, with
SHA256 `117bec9cd138116369ab3d16b79662434174c0c15df4a54e2c97f8f9b86d4aec`.
The first download was truncated and is not the review source. The full report
rebuild reproduces the saved summary exactly, including operation receipts,
scores, start choices and selections. Its source fingerprint matches the local
qualification. Regenerated inputs differ from the local control only by floating
point differences up to 2.22e-16 and their derived hashes. Each campaign uses its
own sealed measurements; datasets were not replaced during review.

All six terminal records exist, but only four tasks are numerically complete:
the four noisy tasks pass; both clean tasks have incomplete windows. Consequently
the official clean recommendation is `unavailable`, while all four noisy tasks
and the aggregate noisy recommendation select duration 0.2.

| Prefix duration | Clean common-horizon mean NMSE | Noisy common-horizon mean NMSE |
|---|---:|---:|
| 0 | 1.74464 | 1.74978--1.82312 |
| 0.1 | 6.80e-21 | 0.01475--0.01646 |
| 0.2 | One complete task: 4.64e-21 | 0.002415--0.003210 |
| 0.5 | 1.40e-19 | 0.000907--0.001534 |
| 1 | One complete task: 3.80e-21 | 0.000858--0.001077 |
| 2 | One complete task: 1.40e-21 | 0.000784--0.000986 |

These remain means over the same four validation rows at t>2. The two training
starts share each measured acquisition, so the four noisy tasks are two noise
replicates, not four. Both clean tasks fully verify the 0.1 window, but the frozen
all-windows-complete rule prevents selecting it retrospectively when other
windows are incomplete. No accuracy threshold or missing-result rule was relaxed.

An independent saved-array audit reproduces all 285 successful forecast scores
to within 8.9e-16 and verifies prefix-loss start selection. Forecasts agree with
the analytic model at fitted parameters within 8.4e-10. Analytic root-sensitivity
norms agree with the saved diagnostics within 2.0e-10 relative error. No returned
candidate was refitted, and no LLM calls or test-data access were used.

### Three verification timeouts

| Task and operation | Wall seconds | CPU seconds | Outcome |
|---|---:|---:|---|
| clean start 0, duration 0.2, val_0, inference start 1.1 | 20.47 | 0.093 | Search allowance exhausted; first verifier timed out |
| clean start 0, duration 2, val_3, forecast start 0.5 | 5.06 | 0.493 | Second verifier timed out |
| clean start 1, duration 1, val_3, inference start 1.1 | 6.52 | 0.080 | Search converged; second verifier timed out |

The last search already found root 1.79999999973 with prefix NMSE 7.34e-19.
All six shared training fits passed verification. The timing failures therefore
must not be interpreted as evidence that a longer prefix or different equations
are needed. `_solve` and `forecast` start the five-second verification deadline
before synchronous checkpoint writes, and those writes call `fsync`. Waiting
for storage can consume the solver allowance. The large wall/CPU gaps support
a waiting-overhead diagnosis, but current receipts do not isolate filesystem
latency from descheduling, so the exact cause is not proved.

The ledger contains 730 operations, 1,119 search calls, 1,159 verification calls
started / 1,156 completed, and 144 information calls. Two dependent forecasts
were not attempted after failed inferences. Total recorded operation CPU is
189.80 seconds versus 1,602.98 seconds of summed operation wall time; the latter
is not campaign makespan. Accounting is complete and the failures remain visible.

### Interpretation and next gate

At duration 0.2 the noisy runs' across-start prediction spread is at most 3.2e-9,
while the conditional one-noise-unit root scale is 0.0312--0.0326. Agreement
between starts does not remove measurement uncertainty. In the first noisy
acquisition the shared initializer coefficient is about 0.6296 instead of its
generating value 0.7; the maximum root error is 0.1251 at duration 0.2 and 0.0947
at duration 2. The latter exceeds its conditional noise scale of about 0.00371
because that diagnostic holds shared parameters fixed. It is not a bound on total
root error. Predictive adequacy does not certify accurate latent coordinates.

The immediate follow-up is to measure checkpoint time separately from numerical
work and make verification robust to storage waits while retaining a finite
overall deadline and complete cost accounting. Recover the affected clean cases
under an explicitly new supplementary policy; preserve the original failed
ledger and do not relabel this run as a six-task pass. The four completed noisy
tasks need no numerical rerun to interpret this experiment. After execution is
qualified, move to multiple independent roots and a construction-exported model,
and assess a frozen duration policy on new trajectories. These results do not
establish a universal duration or immediate post-prefix forecast accuracy.

Review verification: all 154 relevant tests pass in 17.44 seconds, report
reconstruction and `git diff --check` pass, and Ruff passes outside the unchanged
37 existing findings in unrelated `analysis/claude/` files. This results update
changes documentation only; it does not implement the proposed timing recovery.

## Verification

The 154 targeted tests cover the prefix kernel, diagnostics, campaign, malformed
and incomplete receipts, and CLI/scheduler behavior. They pass. The six-task smoke
and terminal replay check also pass, as do both shell syntax checks, Python 3.11
syntax compatibility and `git diff --check`.

`ruff check . --exclude analysis` passes. Full `ruff check .` reports 37 existing
findings in unrelated, untracked `analysis/claude/` files; those were not changed.
Repository-wide `.venv/bin/pytest -q -n 4 --dist=loadfile` passes: 7,240 passed,
8 skipped because local Torch is unavailable, in 948.82 seconds. The full log is
uncommitted at `output/phase-c-fitting-m44-pytest.log`.
