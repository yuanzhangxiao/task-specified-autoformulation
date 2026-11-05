# Construction meal fitting reliability

4 November 2026. This training-only CPU milestone follows the completed
`construction-meal-joint-profile-1` campaign. Root-only and joint profiling
reached numerically tied endpoints in all five comparisons. The small-step
probes encountered step-dependent slopes without failed evaluations. Ordinary
curvature near a small gradient can explain that pattern; it is not sufficient
evidence of unreliable evaluation or a need to change the equations.

The new campaign improves the local diagnostic and tests whether genuinely
different saved starts can find a better fit after releasing the meal rate.
Every task uses the existing certified joint profiler. No equations, parameter
domains, public data, prompts, or production defaults change.

Local qualification: 330 regression tests passed. Both representative saved
endpoints completed all derivative stencils without failed calls. The warm fit
stayed near NMSE 0.423866885; a finite probe from the formerly fixed `1/T` start
improved 0.457389536 to 0.455793553, verified by independent original-equation
rollout, but remains worse than the warm incumbent. One warm gradient remains
step-sensitive, so stronger local conclusions remain deferred. The new outer
restart searches have not been run locally.

## Five tasks with distinct purposes

| Index | Saved source | New work |
| --- | --- | --- |
| 0 | Warm joint-profile endpoint | Curvature audit only; no new outer search |
| 1 | Joint endpoint previously fixed at `0.001/T` | Release the rate, search, then audit |
| 2 | Joint endpoint previously fixed at `0.01/T` | Release the rate, search, then audit |
| 3 | Joint endpoint previously fixed at `0.1/T` | Release the rate, search, then audit |
| 4 | Joint endpoint previously fixed at `1/T` | Release the rate, search, then audit |

`T=300` is the public training horizon used in the historical controls. Those
four rates identify the source endpoints; **they are not fixed in the new
searches**. Each restart begins at its own saved verified coefficients and
initials, with the meal rate made free. This is a set of different initial
points in the same model family, not another fixed-rate profile comparison.

The joint inner solve retains two shared coefficients—the meal forcing gain
`g=k_abs*k_abs_rate` and the auxiliary drift—and 32 trajectory-specific initial
roots. Each source remains on the strictly positive meal-rate branch with a
finite original-parameter lift. The physical zero-rate branch is not included.
Conditional optimization does not guarantee outer convergence or parameter
identifiability.

The shared verified incumbent and every source trial remain available even
if setup, qualification, search or diagnosis fails. Source and trial points
are independently replayed before new numerical work. Any useful point found
by the audit must pass a separate original-equation replay and the existing
improvement rule before it can replace an incumbent. Numerical audit scores
alone cannot promote a candidate.

## What the improved audit measures

The audit freezes scales at the selected endpoint. Positive rate/timescale
coordinates use logarithmic perturbations; other coordinates use scaled
physical differences. Every trial re-solves the same bounded inner problem.
Profiled variables are excluded from the outer audit, and parameter bounds
determine the available perturbation directions.

The diagnostic records:

- Centered or feasible one-sided gradient estimates at two step sizes.
- Whether the inner bounded least-squares active set changes across probes.
- The profiled residual Jacobian and its Gauss–Newton curvature approximation.
- Mixed second differences of the actual loss at two step sizes, together with
  actual checks of coupled candidate directions. When the derivative model is
  unresolved, these remain exploratory finite probes.
- The tested neighborhood, numerical resolution, missing or failed directions,
  and actual evaluation, persistence and total costs.

The Gauss–Newton matrix is explicitly distinct from the actual loss Hessian.
The full six-dimensional numerical matrices are informative when all required
probes complete and agree. Positive numerical eigenvalues provide local evidence
within that checked neighborhood; they do not establish global optimality.
Negative curvature alone does not prove feasible descent at a parameter bound
or when the gradient is nonzero; an evaluated feasible improvement is stronger
evidence.
Active-set changes or insufficient numerical resolution limit smooth-curvature
interpretations. Missing probes remain missing rather than being counted as
evidence of no descent.

The output distinguishes observed feasible improvement, local numerical evidence,
and unresolved diagnostics. Finite probes do not formally certify stationarity,
parameter confidence, scientific correctness, or equation inadequacy. The campaign
produces training reconstructions with fitted training roots, not new
prefix-conditioned, validation or test forecasts. It calls no LLM.

## Budgets and deterministic recovery

The four restart tasks each receive **1800 numerical-active search seconds**,
at most **2000 objective attempts**, and a **2700-second overall search cap**
including explicit 900-second persistence headroom. The warm task has no search
reservation. Each task's curvature audit has **360 numerical-active seconds**,
at most **220 profile calls**, and a **600-second overall cap**. Independent
verification of a useful audit candidate has its own reservation.

Setup, qualification, replay, search, audit and conditional verification costs
are recorded separately. Every strict search improvement remains durable;
routine progress uses the existing 30-second cadence. Stage journals prevent
interrupted work from silently renewing its allowance. Historical and new
campaign costs remain distinguishable, with overlapping source histories
explicitly labeled.

The Slurm array has **five tasks, concurrency two, one CPU and 8 GB per task**.
Workers have a four-hour ceiling; the dependent report has thirty minutes.
Default total reservations are **3060 seconds for the warm audit task** and
**6180 seconds per restart task**, within the 13,200-second launcher limit.
The scheduler ceiling is not an expected runtime. Keep the submitted code
directory unchanged until completion: its source, interpreter, package versions,
launchers, account and resource settings are sealed.

## Upload and submit

The code-only bundle contains committed source, `SOURCE_COMMIT` and
`SHA256SUMS`. The completed joint-profile source campaign must already exist
on Delta. From the **Mac terminal**:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/construction-meal-reliability-delivery-20261104/construction-meal-reliability-code.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:~/construction-meal-reliability-code.tar.gz
```

On **Delta**:

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE="$HOME/construction-meal-reliability-code.tar.gz"
test -f "$AF_BUNDLE"
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-meal-reliability.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
sha256sum --quiet -c SHA256SUMS
cat SOURCE_COMMIT
export AF_MEAL_RELIABILITY_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-meal-joint-profile-v1
export AF_MEAL_RELIABILITY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-reliability-v1
bash scripts/hpc/start_construction_meal_reliability_delta.sh run
BASH
```

`run` authenticates the source, freezes a new plan and submits the array and
report. It does not fit on the login node. Progress is flushed before filesystem
and scheduler operations. Confirmed job receipts are reused. Ambiguous scheduler
replies stop automatic resubmission: inspect saved intent/reply records and
`squeue`/`sacct` before recovery. Do not delete seals or choose a fresh root merely
to bypass an identity mismatch.

## Status, inspection and download

From **any Delta directory**, including a fresh SSH session:

```bash
bash <<'BASH'
set -euo pipefail
export AF_MEAL_RELIABILITY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-reliability-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_MEAL_RELIABILITY_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_meal_reliability_delta.sh"
test -f "$AF_WORKER"
bash "$AF_WORKER" status
# Once the jobs have finished:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small receipts and shows each task's source and mode.
`inspect` reconstructs the report and writes `inspection.tar.gz` without fitting
or submitting. Report-only failures do not require rerunning numerical tasks.
Queue check:

```bash
squeue -u "$USER" -o "%.22i %.48j %.12T %R" |
  awk 'NR == 1 || /construction-meal-reliability/'
```

Download from the **Mac terminal**:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-meal-reliability-v1/inspection.tar.gz \
  ~/Downloads/construction-meal-reliability-inspection.tar.gz
```

Retain both campaigns' plans/inputs, submitted code, execution/submission seals,
operation receipts, checkpoints, terminal results and logs. Campaign-specific
overrides are `AF_MEAL_RELIABILITY_ROOT`, `AF_MEAL_RELIABILITY_SOURCE`,
`AF_MEAL_RELIABILITY_CONFIG` and `AF_MEAL_RELIABILITY_CONCURRENCY` (1 or 2,
fixed before submission). Common `AF_PYTHON`, `AF_CASADI_ROOT` and `AF_ACCOUNT`
settings remain supported. Older campaign variables cannot redirect this campaign.
