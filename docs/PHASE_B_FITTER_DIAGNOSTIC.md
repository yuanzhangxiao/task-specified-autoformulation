# Phase B frozen fitter diagnostic on Delta

This development experiment asks whether the shared numerical fitter can recover
or improve a parameter vector that a frontier agent already found for the **same
equations**. It is the first fitter milestone, before changing optimization
algorithms or coupling fitting into the staged proposer. Sol can continue the
proposer on ACES while this diagnostic runs on Delta CPUs.

## Frozen comparison

The versioned configuration is `configs/phase_b_fitter_diagnostic_v1.json`.
It includes all three existing agent repetitions, without selecting by outcome,
for these two development cells:

| Cell | Source |
| --- | --- |
| `phase_b_dalla_man_t2_canonical_named_easy` | Prompt-v3 refresh |
| `phase_b_anonymous_system_task_canonical_opaque_hard` | Historical full run |

Each of the six source models receives these four tasks (24 tasks total):

| Arm | Parameters at first evaluation | Starts | Maximum `nfev` per start |
| --- | --- | ---: | ---: |
| `agent_replay` | Exact returned agent vector, no optimization | 0 | 0 |
| `warm_1` | Exact returned agent vector | 1 | 300 |
| `cold_1` | Existing runtime default start | 1 | 300 |
| `cold_3` | Runtime default plus two seeded random starts | 3 | 100 |

The fitting arms each have a 900-second fitting allowance. The three-start arm
shares this allowance across starts. SciPy's `nfev` counter excludes some numerical
Jacobian evaluations: these are matched evaluation **ceilings**, not claims of
equal actual CPU cost. Diagnostics retain attempted starts, evaluations,
convergence messages, integration failures and fitted parameter vectors.

The integrator is the existing adaptive `solve_ivp`/RK45 implementation, with
relative tolerance `1e-7` and absolute tolerance `1e-9`. Derivative regression,
CasADi initialization and proposal calls are disabled. Parameter domains, legacy
bounds and initial-condition declarations are preserved; an infeasible warm start
is recorded as a failure, not clipped or silently replaced.

Each vector is independently replayed on train and validation using standard
deviations fitted only on train. No target state resets occur after initialization.
All trajectories must succeed to report an aggregate NMSE; a failed trajectory
makes the split score unavailable and retains its error. The native fitter metrics
are also retained in the fit checkpoint for comparison. Validation does not fit
parameters or select a restart; the fitter chooses its restart using training cost.
Sources with unresolved fitted latent initial conditions are marked unsupported
so the diagnostic cannot introduce validation-fitted initial conditions.

Only `manifest.json`, `proposer_prompt.txt`, `train.csv`, and `validation.csv` are
copied into the frozen public snapshot. The manifest still declares the sealed
test fingerprint, but test bytes are neither required nor read. A loader fix makes
`load_development()` validate only the train/validation fingerprints; full manifest
validation still checks all three splits. No hidden contracts or test evaluator
are part of this experiment.

## Run on Delta

Use an isolated checkout so active ACES/proposer work and existing Delta jobs keep
their own code. From the existing Delta repository:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-v21
git fetch origin codex/fitter-diagnostic
git worktree add --detach ../autoformalism-fitter-diagnostic origin/codex/fitter-diagnostic
cd ../autoformalism-fitter-diagnostic
bash scripts/hpc/submit_phase_b_fitter_diagnostic_delta.sh
```

If that worktree already exists, use the existing checkout at the same commit for
resume. Do not update its code while its jobs are running. The submitter requires
a clean checkout, pins the submitted commit, and sets `PYTHONPATH` to that
checkout's `src`; the shared virtual environment is not reinstalled or edited.

Defaults match the established Delta setup:

| Setting | Default for `yxiao2` |
| --- | --- |
| Python | `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python` |
| Public data | `/work/hdd/bibo/yxiao2/phase_b/inputs/public-prompt-v3` |
| Historical sources | `/work/hdd/bibo/yxiao2/phase_b/raw-data-agent-fitted-v1` |
| Refresh sources | `/work/hdd/bibo/yxiao2/phase_b/raw-data-agent-fitted-prompt-v3-refresh-v1/runs` |
| Output | `/work/hdd/bibo/yxiao2/phase_b/fitter-diagnostic-v1` |

The refresh-root setting itself points to the directory **above** `runs`.
Environment overrides are `AF_REPO_ROOT`, `AF_PYTHON`, `AF_PUBLIC_DATA_ROOT`,
`AF_HISTORICAL_RAW_ROOT`, `AF_REFRESH_RAW_ROOT`, `AF_OUTPUT_ROOT`, `AF_CONFIG`,
and `AF_ARRAY_CONCURRENCY` (default 6). No provider credentials are required.

The submitter schedules preparation, a dependent CPU array, and a summary after
all array tasks terminate. It uses account `bibo-delta-cpu`, partition `cpu`,
one CPU and 8 GB per task, and a 25-minute Slurm allocation. Numerical thread pools
are restricted to one thread. The supervisor bounds each fitting task to 1,140
seconds, including replay and a 60-second grace period, and each exact replay to
240 seconds. The maximum combined worker budget is about 6.1 CPU-hours, plus
preparation, summary and scheduler overhead. No GPUs are requested.

Preparation prints the counts of ready, missing and invalid source models. All
six remain in the ledger. Missing runs are not replaced by a better repetition.
Prompt hashes are checked against the configuration; source run identity and
train/validation/prompt hashes must match the supplied public release. An invalid
source gets four explicit failure records. A mismatched public package aborts
preparation. Check its log first if the array is waiting on a failed dependency.

## Inspect and resume

The output directory contains:

- `freeze.json`: plan, source provenance, runtime fingerprint and 24 task identities.
- `public/` and `candidates/`: immutable development inputs and executable snapshots.
- `results/NNN.fit.json`: completed fit checkpoint, parameters and optimizer diagnostics.
- `results/NNN.json` and `.log`: final task status, fresh replay metrics and execution log.
- `summary.json` and `summary.md`: all six models, four arms, and paired validation ratios.
- `logs/`: Slurm output and submission/job IDs.

Re-running the same submit command resumes the same frozen experiment. Completed
tasks, including recorded failures, are retained. A task interrupted before its
final checkpoint runs again; a completed fit checkpoint lets it resume at replay.
An interrupted optimizer restarts from the same seeded initialization, not an
internal SciPy solver state. Wall-time cutoffs can produce different stopping
points under different machine load; bitwise equivalence is not promised.

Every resume checks frozen asset hashes, package source, runner source, Python and
numerical dependency versions before accepting an existing checkpoint. Changed
protocols, inputs or dependencies require a new output root. To deliberately retry
terminal failures under a larger budget, use a new versioned configuration/output
root and retain the original results. Do not delete failed rows to improve the
reported denominator.

Summarize a partially completed experiment from the dedicated checkout:

```bash
PYTHONPATH="$PWD/src" /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  scripts/run_phase_b_fitter_diagnostic.py summarize \
  --output-root /work/hdd/bibo/yxiao2/phase_b/fitter-diagnostic-v1
```

## How the results guide the next fitter milestone

Compare all three repetitions within each cell, alongside failures and actual
runtime. Ratios below one mean lower validation NMSE than exact agent replay.
They describe the fitter diagnostic, not a method-level benchmark victory.

1. If warm fits retain good fits but cold fits fail, prioritize initialization,
   parameter scaling and continuation from previously fitted candidates.
2. If warm fits worsen training loss or fail on an executable starting vector,
   inspect bound handling, integration failures and optimizer termination before
   adding more restarts. A warm-start retention safeguard is then a concrete next
   implementation, subject to this evidence.
3. If training improves while validation worsens, prioritize regularization or
   simpler structures; additional optimization alone is unlikely to address that
   pattern.
4. If all starts recover comparable fits, move effort toward topology/structure
   search and integrating numerical feedback into Sol's staged proposer.
5. If exact replay itself fails, first inspect execution/initialization compatibility.
   The source is not evidence of an attainable fit in this runtime until replay
   succeeds.

This branch adds a diagnostic, not a new optimizer or a demonstrated improvement
to the proposed method. Conclusions are limited to two development cells and six
agent-proposed structures. The paper and final test evaluation should wait for
subsequent method improvements and a frozen evaluation plan.

## Parallel integration contract

Astra owns the diagnostic module, its CLI, configuration, tests and Delta scripts
on `codex/fitter-diagnostic`. Sol owns proposer construction and ACES experiments.
The only existing shared file changed here is `src/autoformalism/data/loader.py`;
Sol has acknowledged that boundary. `fit_candidate`, `FitConfig`, `FitResult`, the
restricted compiler, candidate schemas and production fitting defaults are unchanged.
Integrate this branch after review; continue experiments from pinned commits and
separate output roots.
