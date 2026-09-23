# Dalla Man mechanism pilots and joint-output fitting

This is a fresh development pilot, not a continuation or a new benchmark release.
Use `configs/dalla_mechanism_pilots_v1.json` with the existing
`review-deadline-2` constructor and whole-model revision path.

| Canonical named cell | Required generated targets | Seeds | Arm | Visits |
|---|---|---|---|---|
| T1-hard | Gp | 0, 1 | Full | construction + 2 revisions |
| T2-easy | Gp, I, U | 0, 1 | Full | construction + 2 revisions |
| T2-hard | Gp, I | 0, 1 | Full | construction + 2 revisions |

There are six lineages and 18 planned task visits. A failed construction or
ineligible revision can reduce the number of actual fits. `full_only=true` is
opt-in; historical matrices still create their original arms in the original
order. T3/T4 are not submitted in this pilot.

## Numerical contract

`collocation-multi-target-v1` explicitly enables all required public outputs.
The equations, parameter vector and causal initialization policies are shared
across training trajectories. Each target has its own training-derived scale
`s_c`, using the existing `TrainingScaler` convention. Both collocation and free
rollout refinement minimize the sum of squared normalized residuals over every
training trajectory, target and sample. Reported aggregate NMSE divides that
sum by the total number of target observations. Targets share each trajectory's
sample grid, so this is also the mean of the per-target NMSEs.

Residual/Jacobian row order is trajectory, target, sample. Observation derivatives
include both state sensitivity and direct parameter dependence. Thus an algebraic
output such as U contributes its own residual and derivatives; it is not treated
as a dynamic state or supplied as a measured future forcing. Penalty residuals
and Jacobians retain the full output dimension on failed integrations/timeouts.

The initializer uses training observations only. Its node states are discarded.
Final training and validation scores are causal free rollouts. Validation initials
come from permitted initial information; validation parameters are not fitted.
Training evidence, residual feedback, capability checks, revision warm starts,
and per-target reporting retain all outputs. A missing output cannot produce a
complete fit. No target is dropped or substituted by a measured trajectory.

The new profile keeps the existing numerical settings: 120 s initialization,
180 s refinement, 240 refinement evaluations, Radau, rtol 1e-7, atol 1e-9,
feasibility recovery and the existing piecewise policy. These are per-fit budgets,
not guarantees of convergence. More outputs can cost more computation under the
same limit. The old `collocation-feasible-v1` and `collocation-single-target-v2`
restrictions remain. Single-output predictions and residuals are unchanged by
using the new profile. Cross-protocol continuation/recovery tools are not enabled
for multi-target campaigns in this milestone.

## Verification

Local milestone verification: 2,973 tests passed, eight optional Torch tests
skipped. Both real-fit smokes passed. Changed Python files pass Ruff; the full
repository check still reports 37 pre-existing errors in `analysis/claude`.

From the project root (or a pinned ACES checkout with modules loaded):

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python -m pytest -q tests/test_multi_target_profile.py tests/test_dalla_pilot.py tests/test_single_target_profile.py tests/test_sibling_fit.py tests/test_review_deadline_submission.py
.venv/bin/python scripts/smoke_multi_target_fitting.py --output /tmp/af-multi-target-smoke
```

On ACES, use `"$AF_PYTHON"` instead of `.venv/bin/python`; the pinned checkout
shares the existing environment. The submission's CPU prepare job automatically
runs the relevant tests and smokes before the GPU job becomes eligible. Use a
new smoke output directory after a code change; exact resume checks code identity.

The synthetic smoke fits two distinct dynamic observations, then adds an
algebraic third observation with a parameter identifiable only through that
output. It tests a changed validation input schedule, a real warm-started child,
per-target accuracy and exact resume. This verifies execution, not Dalla Man
scientific recovery. Other tests compare all sensitivity columns to production
finite differences and verify future validation labels cannot affect rollouts.

## Staging and launch

Do not reuse a historical campaign root or mutate a frozen plan. Pin the new
commit in a clean ACES checkout. Only four development files per cell are needed:
`manifest.json`, `proposer_prompt.txt`, `train.csv`, `validation.csv`.

The Mac transfer helper now accepts `AF_CONFIG`:

```bash
cd /Users/yuanzhangxiao/Projects/autoformalism
AF_CONFIG="$PWD/configs/dalla_mechanism_pilots_v1.json" \
AF_PUBLIC_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-pilots-public-v1 \
bash scripts/hpc/stage_review_deadline_public.sh
```

It uses the existing `delta` and `aces` SSH aliases and the unchanged Delta public
release `/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3`. It copies exactly
12 development files and verifies hashes on ACES. It refuses to replace differing
existing files. No test CSV is transferred. The user runs this transfer.

On ACES set the following after entering the clean pinned checkout:

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT="$PWD"
export AF_CONFIG="$PWD/configs/dalla_mechanism_pilots_v1.json"
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export AF_PUBLIC_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-pilots-public-v1
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-pilots-v1
export AF_IPC_TMP_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/af-ipc
bash scripts/hpc/submit_review_deadline_v2_aces.sh
```

Before any submission the launcher checks every public file, freezes hashes, and
prints the exact matrix, profile, commit, and output root. It uses the existing
image/cache/account defaults. Each proposer visit uses one H100; numerical fits
are separate CPU jobs. Visits are submitted sequentially through dependency
barriers and durable receipts. Repeating the same launch does not duplicate a
completed submission. An uncertain submission remains an explicit inspection
case; do not erase its intent/receipt files to force a retry.

## Results and interpretation

```bash
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-pilots-v1
sacct -j "$(jq -r '.jobs | [.[]] | unique | join(",")' "$AF_OUTPUT_ROOT/submission_manifest.json")" \
  --format=JobID,JobName%28,State,ExitCode,Elapsed
module load GCCcore/13.2.0 Python/3.11.5
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" report --root "$AF_OUTPUT_ROOT"
jq '{planned_rounds,status_counts,rows:[.rows[] | {cell,seed,round,status,retained_train_nmse,retained_validation_nmse,retained_train_per_target_nmse,retained_validation_per_target_nmse,all_graph_requirements_certified,budget_exhausted}]}' "$AF_OUTPUT_ROOT/summary.json"
```

These commands report development results only. Keep all trials and incumbents.
Inspect fitted signs, active meal/insulin pathways, time scales and all individual
validation trajectories before claiming an intervention example. A graph pass
does not certify physiological coefficients or signs, and a low aggregate error
does not establish a good fit for every target. T1 still has no declared insulin
forcing. For T2, verify the released validation schedules actually vary insulin
independently before making an insulin-intervention claim. New diagnostic probes,
if needed, must be labeled separately from registered validation/test results.
