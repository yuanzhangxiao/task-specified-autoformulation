# Construction meal joint profiling

4 November 2026. This training-only CPU milestone compares root-only profiling
with joint profiling of two shared gains and all training initial roots. Both
arms use the same five verified model-15 starts from
`construction-meal-scale-recovery-1`, the same equations, parameter domains,
training data, and outer numerical allowances. It does not revise the model.

## What is being compared

For the certified positive meal branch, write `B=k*A` and `g=k*q`, where
`k=k_abs` and `q=k_abs_rate`. The reduced affine dynamics include

```text
B' = g*r - k*B
G' = r - C - X + B
I' = r + a*G - lambda*I + b*u
X' = c*I - d*X
C' = s
```

Here `s` is the existing shared coefficient
`k_Uii_term_12_0_244d72574a`; it is not an added physiological term. Conditional
on the other shared parameters, the predicted outputs depend affinely on the
shared gains `g,s` and the independent initial roots. A certified joint design
matrix therefore permits a single bounded least-squares solve across all
training trajectories. The two global gains are shared across trajectories;
the 32 training initial roots remain trajectory-specific.

| Arm | Inner bounded least-squares variables | Outer variables |
| --- | --- | --- |
| `root_only` | 32 training initial roots | Eight shared coordinates, or seven at a fixed meal rate |
| `joint` | The same 32 roots plus shared `g,s` | Six shared coordinates, or five at a fixed meal rate |

The joint solve must preserve the original coefficient domains and finite lift
back to the original model. It applies only after the structural certificate
establishes the required affine dependence and exclusive uses. Predictions are
checked against original-equation propagation. Uncertified or numerically
unreliable evaluations fail qualification rather than supplying an optimistic
objective. The physical zero-rate branch remains distinct and is not tested in
this campaign.

There are **ten tasks**: each of five frozen starts (`warm-flux`,
`fixed-0p001`, `fixed-0p01`, `fixed-0p1`, `fixed-1`) is evaluated under both
arms. The fixed rates remain `0.001/T`, `0.01/T`, `0.1/T`, and `1/T`, with
`T=300` from the public training horizon. Each pair receives exactly the same
saved parameters and initial roots before its arm-specific conditional solve.
No new results are ranked to choose starting points.

The global verified incumbent and each fixed family's verified trial are
preserved independently. Compare the trial endpoints and their costs within
each matched pair. A retained global NMSE may belong to the free-rate incumbent
and must not be mistaken for the achieved loss at a fixed rate.

## Budgets, diagnostics and interpretation

Each task has **1800 numerical-active search seconds**, at most **2000 objective
attempts**, and a **2700-second overall search cap** including explicit
900-second persistence headroom. Every strict feasible improvement remains
durable; routine counters use the existing 30-second cadence. Preparation,
conditional qualification, independent verification and endpoint diagnostics
have separate recorded reservations. Costs of these stages are reported rather
than hidden within a nominally matched outer-search budget.

Endpoint probes attempt positive and negative `1e-3` coordinate steps and
half-steps in an endpoint-frozen chart, respecting parameter bounds. They have
a separate 180-second numerical allowance and 300-second overall cap. They
can exhibit a remaining improvement or numerical inconsistency;
absence of an improving sampled direction is not a stationarity or global
optimality certificate. Raw optimizer success remains separate from a finite
paired terminal evaluation. Conditional least-squares optimality does not
establish convergence of the outer problem.

The probes refreeze units at the diagnostic endpoint using the search's scaling
rule. Positive timescales/rates use logarithmic steps; other coordinates use
scaled physical differences. Frozen and profiled coordinates are excluded.
Every point re-solves the same inner problem. A material decrease at both steps
with consistent slopes records feasible descent. Step-size sensitivity is
reported separately from failed numerical evaluations: ordinary curvature can
make these finite slopes disagree. The loss-resolution threshold is a reporting
convention, not a rigorous integration-error bound. Probed candidates remain
diagnostic records requiring independent verification before any future adoption.

Training roots are estimated from training trajectories. These experiments
provide no new prefix-conditioned forecasts, validation/test scores, parameter
confidence levels, or physiological correctness claims. They call no LLM and
do not change production defaults, benchmark inputs, prompts or equations.

The Slurm array requests **one CPU and 8 GB per task, concurrency two**, with
a four-hour worker ceiling and a thirty-minute dependent report. Reservations
total 5,520 seconds per worker by default and must fit within 13,200 seconds.
Interrupted stages never silently
renew; original and new evidence remain in separate campaign roots. Source,
runtime, interpreter, launcher files and resource settings are sealed. Leave
the submitted code directory unchanged until completion.

## Qualification at the five saved endpoints

Local bounded checks performed no outer optimization. All ten conditional
endpoints (two arms at each start) agree with independent original-equation
Radau integration; maximum joint loss discrepancy is 3.27e-12. All joint design
matrices have 14,448 rows and 34 columns, with numerical rank 34; normalized
KKT residuals are at most 5.91e-13. This rank result holds with the remaining
coefficients fixed and does not establish joint parameter identifiability.

| Start | Root-only NMSE | Joint NMSE |
| --- | ---: | ---: |
| Warm | 0.423866958337 | 0.423866885105 |
| 0.001/T | 0.426489814386 | 0.426489814382 |
| 0.01/T | 0.426269576741 | 0.426269576741 |
| 0.1/T | 0.424601448721 | 0.424601381729 |
| 1/T | 0.457389602578 | 0.457389537135 |

The immediate reductions are tiny. Every joint solve places the nonnegative
auxiliary drift exactly at zero, eliminating that troublesome small outer
coordinate. Whether joint profiling improves outer-search progress remains the
question for the matched Delta run.

A bounded warm-endpoint probe completed 25/25 profiles without failed
evaluations in 51.6 wall seconds. None improved on the jointly profiled baseline.
Eight direction slopes were step-sensitive; the others showed no resolved
descent. This is incomplete evidence about stationarity, not a convergence claim.

The final focused integration/regression suite passes **225 tests**, including
the new numerical/campaign/launcher tests and existing root-profile, meal-chart,
termination, checkpoint and recovery tests. Scoped Ruff, CLI-help, shell syntax,
production-source prepare/verify, and report/inspection smoke checks pass.
Repository-wide Ruff reports 37 existing unrelated findings under `analysis/claude`.

## Upload and submit

The code-only bundle contains committed source, `SOURCE_COMMIT`, and
`SHA256SUMS`. The completed source campaign must already exist on Delta.
From the **Mac terminal**:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/construction-meal-joint-profile-delivery-20261104/construction-meal-joint-profile-code.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:~/construction-meal-joint-profile-code.tar.gz
```

On **Delta**:

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE="$HOME/construction-meal-joint-profile-code.tar.gz"
test -f "$AF_BUNDLE"
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-meal-joint-profile.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
sha256sum --quiet -c SHA256SUMS
cat SOURCE_COMMIT
export AF_MEAL_JOINT_PROFILE_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-meal-scale-recovery-v1
export AF_MEAL_JOINT_PROFILE_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-joint-profile-v1
bash scripts/hpc/start_construction_meal_joint_profile_delta.sh run
BASH
```

`run` authenticates the source, freezes the new plan and submits the array/report;
it does not fit on the login node. Progress is flushed before potentially slow
filesystem or scheduler operations. Confirmed job IDs are reused. An ambiguous
scheduler reply blocks automatic resubmission: inspect saved intent/reply files
and `squeue`/`sacct` before recovery. Do not delete seals or choose a new root to
bypass an identity error.

## Status, inspection and download

From **any Delta directory**, including a fresh SSH terminal:

```bash
bash <<'BASH'
set -euo pipefail
export AF_MEAL_JOINT_PROFILE_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-joint-profile-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_MEAL_JOINT_PROFILE_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_meal_joint_profile_delta.sh"
test -f "$AF_WORKER"
bash "$AF_WORKER" status
# Once the jobs have finished:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small receipts and identifies the arm/start for each task.
`inspect` reconstructs the report and creates `inspection.tar.gz` without
fitting or submitting. A report-only failure does not require rerunning fitting.
Queue check:

```bash
squeue -u "$USER" -o "%.22i %.48j %.12T %R" |
  awk 'NR == 1 || /construction-meal-joint-profile/'
```

Download from the **Mac terminal**:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-meal-joint-profile-v1/inspection.tar.gz \
  ~/Downloads/construction-meal-joint-profile-inspection.tar.gz
```

Retain both campaigns' plans/inputs, submitted code, execution/submission seals,
operation receipts, checkpoints, terminals and logs. The campaign-specific
overrides are `AF_MEAL_JOINT_PROFILE_ROOT`, `AF_MEAL_JOINT_PROFILE_SOURCE`,
`AF_MEAL_JOINT_PROFILE_CONFIG` and `AF_MEAL_JOINT_PROFILE_CONCURRENCY` (1 or 2,
fixed before submission). Common `AF_PYTHON`, `AF_CASADI_ROOT` and `AF_ACCOUNT`
settings remain supported. Older campaign environment variables do not select
this campaign's paths.
