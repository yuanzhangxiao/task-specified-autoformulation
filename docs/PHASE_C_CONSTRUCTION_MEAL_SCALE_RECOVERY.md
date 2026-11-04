# Construction meal-scale recovery

3 November 2026. A training-only CPU follow-up to
`construction-meal-continuation-1`. The returned warm continuation achieved
training NMSE **0.42386730309124343** with finite meal-memory timescale about
1422. Four fixed-rate controls were compromised by excessive optimizer gain
units. This milestone corrects those units, preserves all verified endpoints,
and provides explicit persistence headroom. It does not revise equations.

## Coordinate correction

For the certified reservoir `A'=q*r-k*A` and sole output contribution `+k*A`,
the strictly positive chart uses `g=k*q` and reports `B0=k*A0`. The original
ODE and bounded least-squares solution for training initial roots are unchanged.
The previous unit for g multiplied a stale original-q unit by a newly seeded k.
Consequently, preserving physical g while changing k could enlarge optimizer
steps by factors of 1e5–1e8.

`positive-meal-flux-coordinates-2` freezes a characteristic g unit at chart
certification: the positive source g, or, when g=0, source k times the existing
positive original-q unit. The latter is an explicit numerical fallback, not
scientific knowledge about the gain. Later reseeding does not change this unit.
The search freezes its scales once per run, uses logarithmic coordinates for
positive rates/timescales, and retains original domains and precision checks.
The audit records source k/q/g, original q unit, selected scale, fallback and
protocol. No positive lower bound is added to the physical rate.

## Frozen tasks and retention

The source plan is
`f2bf759f4b34ff303fbdf05408cb1576bdd1d7b97b22c65d5db1a1b63f180434`.
Five authenticated source records determine the new starts without ranking
new results. `T=300` is the maximum public training horizon.

| Task | New start | Rate |
| --- | --- | --- |
| Warm continuation | Verified 0.423867303091 endpoint | Free |
| Fixed 0.001/T | Own verified 0.426489875347 trial | Fixed |
| Fixed 0.01/T | Own verified 0.426269638539 trial | Fixed |
| Fixed 0.1/T | Own verified 0.425224094783 trial | Fixed |
| Fixed 1/T | Own verified 0.521512048455 trial | Fixed |

Each fixed task resumes its feasible trial, not its invalid optimizer terminal
or the inherited free-rate point. The global warm incumbent and each fixed
family's saved trial are retained independently; failed or worse new work
cannot erase them. Global and trial points are independently replayed before
qualification/search. Compare `trial_training_nmse` across fixed rates, not
`training_reconstruction_nmse`, which can refer to the free-rate incumbent.

Exact zero remains a distinct original-model stratum and is not rerun here.
Neither finite-rate losses nor a valid optimizer exit certify a profile minimum,
convergence, coefficient accuracy or physiological correctness. No causal-prefix,
validation or test forecasts are produced. No LLM is called.

## Time and durable progress

Each task has **1800 numerical-active seconds**, at most **2000 objective
attempts**, and a separate **2700-second overall search cap**. The added
900 seconds accommodates measured persistence latency; it does not increase
the numerical allowance. Every strict feasible improvement is still saved
immediately. Routine counters retain the existing 30-second checkpoint cadence.
Blocking I/O remains subject to cooperative deadlines and the scheduler ceiling.

Setup, source/trial verification, profiling qualification and final independent
verification have separate durable reservations. Interrupted operations do not
renew. Historical source costs are retained and labeled as overlapping rather
than summed as independent experiments. A fresh campaign is an explicit new
allowance, not an implicit replay of the interrupted search.

The Slurm array has **five tasks, concurrency two, one CPU and 8 GB per task**;
workers have a four-hour ceiling and the dependent report has thirty minutes.
Source, runtime, interpreter, resources and launchers are sealed. Do not edit
submitted code or reuse a historical result root for this campaign.

## Upload and submit

The code-only bundle contains committed source, `SOURCE_COMMIT` and
`SHA256SUMS`. The completed source campaign must already exist on Delta.
From the **Mac terminal**:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/construction-meal-scale-recovery-delivery-20261103/construction-meal-scale-recovery-code.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:~/construction-meal-scale-recovery-code.tar.gz
```

On **Delta**:

```bash
bash <<'BASH'
set -euo pipefail
AF_BUNDLE="$HOME/construction-meal-scale-recovery-code.tar.gz"
test -f "$AF_BUNDLE"
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-meal-scale-recovery.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
sha256sum --quiet -c SHA256SUMS
cat SOURCE_COMMIT
export AF_MEAL_SCALE_RECOVERY_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-meal-continuation-v1
export AF_MEAL_SCALE_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-scale-recovery-v1
bash scripts/hpc/start_construction_meal_scale_recovery_delta.sh run
BASH
```

The wrapper prints progress before source authentication and scheduler calls.
`run` prepares and submits; it does not fit on the login node. Existing confirmed
job IDs are reused. Ambiguous scheduler replies stop resubmission: inspect saved
intent/reply files and scheduler accounting before recovering. Do not remove seals
or select a new root solely to bypass an identity error.

## Status, inspection and download

From **any Delta directory**, including a fresh SSH session:

```bash
bash <<'BASH'
set -euo pipefail
export AF_MEAL_SCALE_RECOVERY_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-meal-scale-recovery-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_MEAL_SCALE_RECOVERY_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_meal_scale_recovery_delta.sh"
test -f "$AF_WORKER"
bash "$AF_WORKER" status
# Once the jobs have finished:
bash "$AF_WORKER" inspect
BASH
```

`status` reads small receipts; `inspect` rebuilds the report and produces
`inspection.tar.gz` without fitting or submitting. Report-only failure therefore
does not require rerunning numerical tasks. Queue check:

```bash
squeue -u "$USER" -o "%.22i %.48j %.12T %R" |
  awk 'NR == 1 || /construction-meal-scale-recovery/'
```

Download from the **Mac terminal**:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-meal-scale-recovery-v1/inspection.tar.gz \
  ~/Downloads/construction-meal-scale-recovery-inspection.tar.gz
```

Retain source and new plans/inputs, submitted code, execution/submission seals,
operation receipts, best checkpoints, terminals and logs. The launchers use only
`AF_MEAL_SCALE_RECOVERY_*` campaign settings plus common runtime/account settings;
old portfolio environment variables cannot redirect this campaign.

## Local qualification

The saved-model preflight checked all five starts and both characteristic
optimizer directions for every active shared coordinate: **77/77** bounded
profiles agreed with independent original-equation Radau rollouts. Rate and
timescale probes used the actual logarithmic search rule. All ten gain probes
passed; source gain units are now 0.02543–0.03191. The largest absolute loss
discrepancy across these probes was 2.59e-9, within the unchanged relative/absolute
comparison. These are preflight checks, not full refits or convergence evidence.

The focused integration suite passed 255 tests before the final policy/retention
additions; all 19 campaign tests passed after those additions. Scoped Ruff, shell
syntax and CLI smoke checks pass. Repository-wide Ruff still reports 37 existing
unrelated findings under `analysis/claude`. All six historical source terminals
still reconstruct exactly from their saved operation receipts.
