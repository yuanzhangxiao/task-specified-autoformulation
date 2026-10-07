# M15: training-only polishing of remaining coefficient errors

Protocol: `phase-c-fitting-polishing-1`.
Configuration: `configs/phase_c_fitting_polishing_v1.json`.

## Question and unchanged comparison

[M14](PHASE_C_FITTING_M14_RESULTS_2026-10-06.md) recovered predictions from all
three generic starts using profiled output gains. Two runs stopped with maximum
coefficient errors of 1.34% and 1.25%, just outside the 1% assessment gate. Their
training errors were still decreasing. M15 tests whether a tighter training-only
stopping target improves coefficient recovery, and measures its extra cost.
Reference coefficients cannot trigger continuation, selection or stopping.

Use exactly the same sealed development inputs, with SHA-256
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
All three original generic starts are rerun under both `rollout_only` and
`profiled_rollout`; no historical fitted start is imported. This is six CPU tasks
on the correct six-state alien-device hard structure with anchored coordinates,
16 training trajectories and four validation trajectories. Both methods estimate
13 dynamic coefficients and five shared latent initials; profiling eliminates
four output gains from the outer nonlinear search. Correct structure and
coordinate choices remain explicit assistance.

## Two stages within one budget

1. Run the M14 fitter to its initial training prediction target. Independently
   check the selected parameters with Radau and DOP853 on all training
   trajectories, using the original complete equations. Require mean worst-solver
   NMSE at most `1e-6`, each trajectory at most `1e-5`, and normalized solver
   disagreement at most `1e-5`. Durably seal this first prediction checkpoint.
2. If it already satisfies the stricter targets, stop. Otherwise, if the remaining
   time and call allowances suffice, start one polishing solve at those fitted
   parameters. Require mean training NMSE at most `1e-8` and every trajectory at
   most `1e-7`; the solver-disagreement gate remains `1e-5`. This is a parameter
   warm start: TRF optimizer state restarts once. Both arms use this procedure.
3. Independently verify the polishing endpoint. Replace the first checkpoint only
   if the new vector still passes the initial prediction gates and has lower
   independently checked mean training NMSE. Otherwise preserve the first vector.
   Report whether the retained vector passed the stricter gates separately.

Total fitting allowance remains **1,200 seconds and 900 actual residual calls**
across both stages. The time ceiling includes fitting setup and independent
training checks. Each launched stage reserves up to 180 seconds of the remaining
time for its check; this does not add to the ceiling. Certificate integrations
are accounted separately from optimizer residual calls. Native convergence,
stagnation and per-evaluation time limits remain unchanged. If the first attempt
never certifies, it retains its available endpoint without initiating polishing.
Unknown residual usage blocks a new phase rather than treating missing cost as
zero. Supervisor cleanup uncertainty blocks evaluation and further optimization.

The tighter target is an empirical precision test, not a mathematical guarantee
of coefficient accuracy. In particular, the solver-disagreement gate itself is
not tightened to `1e-8`; each solver must independently meet the error target.
All coefficients and initial errors are assessed separately afterward.

## Evaluation, reporting and interruption

Freeze the final selected backend before opening validation data or reference
coefficients. The evaluator then measures training/validation NMSE, per-coefficient
absolute and relative errors, and latent-initial errors at the final vector and
at the first certified checkpoint. Each replay has its own 300-second allowance,
outside fitting. Identical first/final vectors reuse the same evaluation.
Before/after replay results never affect the sealed selection.

`summary.json` reports all six tasks, stage policies, optimizer traces, actual
residual calls, fitting/polishing seconds, the first certificate, retained stage,
strict certification, and first/final coefficient-recovery counts. All original
starts remain in denominators. Final assessment gates remain maximum coefficient
relative error 1% and latent-initial absolute error 0.01. Prediction accuracy and
parameter recovery are distinct. A complete campaign means all required records
exist, not that all fits succeeded.

Completed tasks resume without repeating fits or evaluations. A coordinator
interrupted mid-fit becomes terminal on resume, preserves its durable first
checkpoint, and marks unknown cost explicitly; it never refreshes the fitting
allowance. Unfinished evaluation uses the existing evaluation journal. Exact
input/source/runtime identities and scheduler receipts protect deterministic
resume. Do not resubmit blindly after an uncertain scheduler acknowledgement.

This milestone changes neither production fitter defaults nor benchmark data or
prompts. It does not introduce a strategy for general coefficients inside coupled
state equations. That discussion follows this experiment.

## Delta commands

Upload `phase-c-fitting-m15.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
The portable archive contains pinned source, focused tests, configuration and the
unchanged sealed inputs. Use the existing Delta fitting environment:

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m15"
tar -xzf "$AF_BASE/phase-c-fitting-m15.tar.gz" -C "$AF_BASE/code/fitting-m15"
export AF_POLISHING_ROOT="$AF_BASE/fitting-polishing-v1"
bash "$AF_BASE/code/fitting-m15/scripts/hpc/submit_phase_c_fitting_polishing_delta.sh"
BASH
```

The default is **six concurrent one-CPU tasks**, each with 16 GB and a 35-minute
wall limit. Set `AF_CONCURRENCY` to reduce this if needed. Preparation runs focused
tests; the report depends on array termination, including failures. The wall limit
covers 20 minutes fitting, up to ten minutes of before/after replay and overhead.
No GPUs or LLM calls are needed.

Inspect and package the results:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m15/scripts/hpc/inspect_phase_c_fitting_polishing_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-polishing-v1
```

Download the printed `review-*.tar.gz` archive. The inspector prints the explicit
root, scheduler states and before/after summaries, avoiding reliance on variables
from earlier terminal sessions.

## Local qualification

The focused suite passes 99 tests both in the project and in the portable bundle.
It covers stricter policy validation, shared time/call budgets, unknown usage,
cleanup uncertainty, failed or worse polishing, exact certificate/vector binding,
interruption without budget refresh, post-selection before/after evaluation,
all-start accounting and idempotent CPU submission. Bundle checksums, six-task
preparation and the inspector/archive path are checked without fitting the hard
benchmark.

The small independently generated two-state smoke exercises both fitting stages
and independent original-equation replay. Final maximum parameter absolute
errors are below `2e-6` for both methods, with 14 and eight residual calls; a
completed resume spends no new fitting budget. This checks correctness, not
benchmark speed or generic recovery. The six hard-case results are still pending.
