# Saved baseline validation replay

**D3 semantics correction (2026-09-16):** the native D3 implementation fits
sample increments, `x_next = x + f`, without a dt multiplier. Its historical
common ODE replay is an ODE reinterpretation diagnostic, not native D3
performance. Historical candidates also cannot populate different Phase-B
cells. Use [fresh Phase-B native validation](PHASE_B_D3_NATIVE_VALIDATION.md)
for new discovery and correctly labeled one-step/recursive scores. Preserve
older results under their original protocol rather than relabeling them.

This milestone reuses existing SINDy/PySR, GPT-5.6 Sol and D3 candidates and
fitted parameters. It performs no fitting, parameter/initial-state optimization,
LLM generation, test-data loading, or private-reference evaluation. It requires
neither Julia/PySR nor PyTorch: saved equations run through the restricted ODE
runtime on CPUs.

The frozen roster `configs/baseline_validation_v1.json` contains:

- the six review Phase-B cells, three repetitions, four methods (72 expected);
- the six historical rebuttal D3 cells at hard tier, five seeds (30 expected).

For the entire existing 40-cell Phase-B matrix, set
`AF_ROSTER="$AF_REPO_ROOT/configs/baseline_validation_full_v1.json"` before
submission. That roster has 480 Phase-B entries plus the same 30 historical D3
entries; missing Phase-B D3 entries remain explicit. Use a separate output root
from the six-cell run. No new discovery is triggered by the larger roster.

Historical D3 remains on its original registered benchmark and validation
split. It cannot stand in for a missing Phase-B D3 model. Its source-directory
precedence is copied from the existing Phase-A3 rebuttal cohort configurations;
no score participates in choosing between production and retry artifacts.
The selected generation and equations must agree between result and checkpoint.
The old `d3_no_tools` / `restricted_schema` implementation stored a later
train-plus-validation refit in its result. For that recognized format, replay
uses the original selected **training checkpoint** values and records that the
refitted result values were ignored. Native D3 result/checkpoint scalars must agree. Preserve
both `result.json` and its sibling `d3_checkpoint.json` when moving artifacts.
Older results containing already-computed test scores are accepted for D3, but
those scores are not copied, ranked, or used during this replay; prior evaluation
is explicitly recorded. No original artifacts are overwritten.

## Evaluation contract

The new CLI is `scripts/baseline_validation.py` with `inventory`, `prepare`,
`run`, and `report` commands. It does not call the test-only postfreeze endpoint
and never relabels validation data as test data.

Each compatible model uses Radau with rtol=1e-6, atol=1e-8 and 120 seconds per
trajectory. Models use permitted inputs/auxiliary paths and initial observations;
observed states are never reset later. Target forcing that would reveal the
validation target path is rejected. Frozen initial maps/values and all fitted
parameters remain unchanged. Errors use training standard deviations, all
samples including the initial sample, pooled samples per target, and equal
weight across targets. Any failed trajectory makes aggregate NMSE unavailable;
per-trajectory diagnostics remain visible. Solver settings and source/library
versions are frozen before evaluation.

GPT sources must use the fitted-model contract and exactly match the saved
public prompt hash and train/validation fingerprints. This automatically rejects
old named-Dalla prompts and accepts compatible refresh sources. SINDy/PySR must
be original development results; the train-plus-validation SINDy finalization
is inadmissible. Classical/legacy D3 result schemas omit original data/prompt
hashes: the report explicitly labels that provenance unverified. Recording the
current evaluation data hashes does not retroactively prove what data trained
a historical model. Treat those rows as provisional until original release
provenance is checked.

These are validation-selected models, so these scores are development results.
They are not independent test estimates, and native one-step or teacher-forced
scores are shown separately from common open-rollout scores. The method does
not claim equal discovery compute or native fitting semantics.

## Commands on Delta

Run these yourself in an authenticated Delta shell. No SSH session or job is
started by the local agent. Set `AF_REPO_ROOT` to the new pinned clean checkout;
the final message gives the exact commit. Do not update active campaign checkouts.

The defaults use the existing Delta Python environment and artifact paths.
The historical D3 path is an expectation from the rebuttal cohort manifests,
not a claim that the files have been found on Delta. Inventory reports missing
roots and missing models without launching replacement discovery.

```bash
export AF_REPO_ROOT=/projects/bibo/$USER/repos/autoformalism-baseline-validation-v1
export AF_PYTHON=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/baseline-validation-v1
export AF_PUBLIC_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3
export AF_LEGACY_ROOT=/projects/bibo/$USER/repos/autoformalism-v21/data_raw
export AF_D3_ROOT=/projects/bibo/$USER/repos/autoformalism-v21/artifacts/rebuttal/consolidated_inputs/vm/vm2-results/artifacts/baselines
bash "$AF_REPO_ROOT/scripts/hpc/submit_baseline_validation_delta.sh"
```

The launcher submits one preparation CPU job, eight single-CPU replay tasks,
and a report job. No GPUs are requested. It records each sbatch receipt and
refuses an uncertain duplicate submission. The same completed submission command
prints existing job IDs. A job may be terminal even when individual sources or
rollouts failed; inspect counts rather than interpreting `complete` as success
for every model.

If D3 is stored elsewhere, set `AF_D3_ROOT` to the smallest directory containing
its original run directories **before submission**. To locate saved checkpoints
without opening benchmark data:

```bash
find /work/hdd/bibo/$USER /projects/bibo/$USER/repos/autoformalism-v21/artifacts \
  -type f -name d3_checkpoint.json -print 2>/dev/null
```

Do not choose roots using test or newly replayed validation scores. If independent
sources still collide at one cell/repetition, the row is marked unavailable with
all paths recorded. Narrow the inventory to the historically intended experiment.

## Inspect results

```bash
jq '{status,groups,limitation}' "$AF_OUTPUT_ROOT/summary.json"
jq '.rows[] | {cohort,source_kind,benchmark_id,tier,repetition,status,normalized_mse,audit,error}' \
  "$AF_OUTPUT_ROOT/summary.json"
```

If the summary is absent or jobs failed:

```bash
cat "$AF_OUTPUT_ROOT/submission/manifest.json"
tail -n 60 "$AF_OUTPUT_ROOT"/logs/*.err
jq '{missing_roots,errors,rows_found:(.rows|length)}' "$AF_OUTPUT_ROOT/inventory.json"
jq '.rows[] | select(.status!="ready") | {index,benchmark_id,source_kind,error,rejected_sources}' \
  "$AF_OUTPUT_ROOT/plan.json"
```

`plan.json` seals both ready and unavailable sources. To add newly located
artifacts, create a new output root and inventory; do not mutate the frozen plan.
If a replay array times out, the commands below submit only replay/report against
the existing plan. Existing row checkpoints are reused; per-row locks prevent
concurrent duplicate simulation. Interrupted rows restart, completed rows do not.

```bash
export AF_COMMIT=$(git -C "$AF_REPO_ROOT" rev-parse HEAD)
af_replay=$(sbatch --parsable --account=bibo-delta-cpu --partition=cpu \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=12:00:00 \
  --array=0-7%8 --export=ALL --job-name=baseline-val-resume \
  --output="$AF_OUTPUT_ROOT/logs/resume-%A_%a.out" \
  --error="$AF_OUTPUT_ROOT/logs/resume-%A_%a.err" \
  "$AF_REPO_ROOT/scripts/hpc/run_baseline_validation_delta.sh" run)
af_replay=${af_replay%%;*}
sbatch --account=bibo-delta-cpu --partition=cpu --nodes=1 --ntasks=1 \
  --cpus-per-task=1 --mem=4G --time=00:10:00 --export=ALL \
  --dependency="afterany:$af_replay" --job-name=baseline-val-report \
  --output="$AF_OUTPUT_ROOT/logs/report-%j.out" \
  --error="$AF_OUTPUT_ROOT/logs/report-%j.err" \
  "$AF_REPO_ROOT/scripts/hpc/run_baseline_validation_delta.sh" report
```

## Local verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_baseline_validation.py tests/test_baseline_development_adapter.py tests/test_final_evaluation_adapters.py
PYTHONPATH=src .venv/bin/python scripts/smoke_baseline_validation.py
.venv/bin/ruff check .
.venv/bin/pytest -q
```
