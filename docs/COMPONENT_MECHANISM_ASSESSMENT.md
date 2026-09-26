# Component-model mechanism assessment

The numerical assessment is the same `fitted-public-mechanism-tests-1` rubric
used for the external baselines. New tools only export retained models and
reaggregate saved outcomes. No fitter, model, scientific threshold, prompt, or
benchmark data is changed. No test data, LLM calls or optimizer calls are used.

## Snapshot and scope

`export_component_mechanisms.py` reads the sealed final-component campaign plan
and saved round results. Its default submission policy freezes the latest
recorded **retained incumbent** per lineage, using the existing validation-based
selection. Later empty interrupted records do not erase an earlier incumbent.
An explicit `--round 5` instead requires that exact recorded round; it never
substitutes a different round. No comparison of compliance scores, refitting,
new NMSE ranking or selection among unretained trials occurs.

The export retains all planned lineages, including missing models; every row
records its source round, selected model's origin, observed rounds, task flags,
and source digests. Full and Brief-only are separate. Critic/verifier treatment
groups cover nine cases; shared-process-off groups cover only the four Dalla
cases. These search snapshots are **before final pruning** and cannot be called
the completed final/pruning comparison. Unequal search budgets remain visible.

The export verifies the selected fit receipt, request, lowered equations, exact
parameter set and development-data digests. It reproduces the causal initializer
lowering and retains learned initializer parameters; it does not replace them
with defaults or optimizer guesses. Changed or corrupt identities stop export.
An existing sealed snapshot is reused even if the source campaign later advances.

The preparation job first writes `inputs/models.json` and `inputs/LEGACY.md`.
The latter reports graph requirements, target predicates, and runtime validity
separately under the same post-hoc checks for verifier-on and verifier-off arms.
It is not the stronger functional assessment. An unavailable model leaves its
legacy evidence unresolved. Graph counts use the public requirement roster;
unavailable target/runtime evidence is reported as an unresolved model-level entry.

## ACES execution

Use a pinned source archive and the existing environment. The new source archive
must contain these scripts. Set `AF_REPO_ROOT`, `AF_COMMIT` and `AF_PYTHON` first:

```bash
module load GCCcore/13.2.0 Python/3.11.5
export PYTHONPATH="$AF_REPO_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_component_mechanisms.py" \
  --source /scratch/group/p.nairr260351.000/u.yx126462/final-components-v1 \
  --root /scratch/group/p.nairr260351.000/u.yx126462/component-mechanisms-v1
```

The launcher submits a CPU preparation job, a **four-task worker array**, and a
reporting job. Each worker requests one CPU, 8 GB and six hours and processes a
deterministic shard of the 160 planned lineages sequentially. `--workers` changes
the worker count. The previous one-task-per-model array could exceed the ACES
per-user submission limit: `%8` constrained execution concurrency, but all 160
array elements still counted toward that limit. The pool defaults to only six
submitted tasks including preparation and reporting. Per-model numerical budgets,
locks, checkpoints and scoring remain unchanged. If a worker reaches its wall
limit, completed results remain available for explicit checkpointed recovery.

The source campaign's `public/` directory supplies public data; its
training/validation identities and prompt hashes are checked by the same adapter
used for the baselines. Preparation needs 16 GB for the embedded campaign plan.
Account defaults to `156264627414` and can be set with `AF_ACCOUNT`.

Submission receipts are durable. Repeating an identical successful submission
returns its saved job IDs; partial scheduler submissions require inspection of
the receipts, not silent resubmission. Worker results and numerical rollout
checkpoints retain the original assessment's immutable resume rules.

### Recover the rejected 160-task submission

Use the updated source archive with the same Python environment and original
source/output paths, then run:

```bash
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_component_mechanisms.py" \
  --source /scratch/group/p.nairr260351.000/u.yx126462/final-components-v1 \
  --root /scratch/group/p.nairr260351.000/u.yx126462/component-mechanisms-v1 \
  --workers 4 --resume-qos-rejection
```

This requires the saved `QOSMaxSubmitJobPerUserLimit` rejection with empty stdout,
no assessment job ID or report submission, and a matching preparation receipt.
It handles the ACES wrapper returning exit code zero despite rejection. The
existing preparation job (2164259 in the reported failure) is reused via an
`afterok` dependency; no second export or preparation job is submitted. Original
receipts are preserved in `submission-intent`; new receipts are written under
`submission-pool-intent`, and success writes the usual `submission.json`.
An ambiguous scheduler timeout or possible accepted array is not auto-resubmitted.
Workers verify the prepared numerical code/environment identity before assessing
models; changing only these orchestration scripts does not change that identity.

```bash
AF_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/component-mechanisms-v1
AF_JOBS=$(jq -r '.jobs | [.prepare,.assess,.report] | join(",")' \
  "$AF_ROOT/submission.json")
sacct -X -j "$AF_JOBS" --format=JobID,JobName%27,State,ExitCode,Elapsed
cat "$AF_ROOT/inputs/LEGACY.md"
cat "$AF_ROOT/mean-sd/SUMMARY.md"
```

`mean-sd/summary.json` contains the new scores and per-case values. Headline means
give equal weight to planned benchmarks; SD is sample SD across benchmark means.
Per-case SD is over repetitions. Unknown evidence is retained in both confirmed
and possible bounds. This is not a change to any per-model scientific verdict.
The original `summary.json` remains the numerical protocol's median/MAD report.

## Reaggregate completed baseline jobs on Delta

With a checkout containing the new reporting script, run:

```bash
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/summarize_functional_mechanisms.py" \
  --root /work/hdd/bibo/yxiao2/phase_b/functional-mechanism-tests-v1 \
  --output /work/hdd/bibo/yxiao2/phase_b/functional-mechanism-tests-v1/mean-sd
```

Only sealed JSON inputs are read. No original data directories, numerical worker
runtime identity, GPUs, fitting or LLM access are needed for this reaggregation.

## Limits of the currently available local results

The local nine-case search summary contains only the Boolean
`all_graph_requirements_certified`, not all predicate records. For both critic
and verifier on, the latest recorded rows give 15/18 confirmed Full lineages
and 16/18 confirmed Brief-only lineages. Treating unconfirmed entries as zero
**confirmed certification**, the equal-case means (sample SD) are 83.3% (35.4 pp)
and 88.9% (33.3 pp). Full includes one missing T2-easy model. Both include two
unresolved CSTR records; these are not evidence of false scientific equations.
The same figures happen to hold at the common zero-based round 5.

The four-case Dalla inspection ZIP supplies complete certificates for seven
available Full and eight available Brief-only retained models. All those models
pass their recorded graph, target, and runtime checks (conditional mean 100%,
SD 0). Coverage is 7/8 and 8/8, respectively; no nine-case target/runtime mean can
be reconstructed from these files. Do not combine this restricted available-only
result with the nine-case confirmed-certification figures.

## Verification

Initial exporter validation passed the full suite (3,528 passed, eight skipped),
with nine focused export/report/submission tests passing after the final
submission test was added.
A read-only smoke check reproduced the lowered equations and fitted parameter
names for all 78 available retained models in the Dalla inspection archive.
CLI help, shell syntax and changed-file Ruff checks passed. Repository-wide Ruff
reported 37 existing findings in unrelated `analysis/claude` files.

The worker-pool recovery passed 17 focused tests and 48 related mechanism tests.
A subprocess CLI smoke assessed five unavailable-model records in two shards and
preserved all saved results on resume. A byte comparison against the preparation
commit verified that all 316 numerical source, assessment CLI and rubric files
were unchanged. The recovery tests cover the site's zero-exit-code rejection,
reuse of preparation, exact 160-model coverage, duplicate submission prevention,
and refusal of ambiguous job receipts or changed source identities.
