# Fresh Phase-B D3 native validation

The historical D3 models were discovered on different benchmark cells. They do
not populate the Phase-B matrix. This campaign makes fresh proposals on the
exact Phase-B public prompts and development data, leaving test data sealed.
It uses the existing D3-native-no-tools adapter (upstream revision
`ee86212dfd5935bb0c9626eaa0570223ff7ecf1c`), a user-selected OpenAI API model,
and CPU Adam fitting. It requests no local GPUs.

## Semantics and limitations

The native implementation computes **`x_next = x + f(x, inputs)` without a dt
multiplier**. Treating the fitted expression as an ODE rate changes its meaning.
The earlier `baseline-validation-d3-v2` common ODE replay is therefore an ODE
reinterpretation diagnostic, not D3's native validation performance. Preserve
those historical artifacts; do not relabel them as this experiment.

Each proposal has exactly the observed target and auxiliary states. Native D3
does not introduce latent states. Adam fits the raw one-step loss across all
observed states using training data; validation selects retained optimizer
checkpoints. Across generations, the adapter selects the smallest native
one-step target NMSE. This native selection rule is unchanged.

The selected candidate and exact fitted coefficients receive two evaluations:

| Endpoint | Target history | Auxiliary paths | Update | Samples scored |
|---|---|---|---|---|
| Native one-step diagnostic | Measured at each step | Supplied | `x + f` | After initial |
| Phase-B recursive target rollout | Initial observation only, then predictions | Supplied throughout | `x + f` | Including initial |

Both use target standard deviations fitted on training data. Scores pool samples
per target, then average targets equally. Recursive evaluation cannot change
the selected model or feed back to the proposer. A failed trajectory produces a
null aggregate, and failed/missing tasks stay in the planned denominator.
Summary medians explicitly describe finite results only.

The replay must reproduce the saved native one-step NMSE within relative
`1e-6`, absolute `1e-10`; otherwise the row is `native_check_failed`. Native Adam
does not enforce declared bounds. Replay preserves its finite fitted values and
records bounds violations; it does not clip, refit, or discard them just because
of those declarations.

This is a **discrete predictive baseline under Phase-B information permissions**.
It does not certify the continuous-time or scientific requirements in the task
prompt. No scientific judge runs. It is not a compute-matched comparison, a test
estimate, or proof of recovered latent mechanisms. Model IDs/repetitions of
hosted APIs are not reproducible random seeds across methods.

## Frozen budgets and resume

- Pilot: `configs/phase_b_d3_pilot_v1.json`, six review cells, one repetition,
  five generations each (at most 30 proposal calls in an uninterrupted run).
- Full: `configs/phase_b_d3_full_v1.json`, the established 40-cell matrix,
  three repetitions, five generations each (at most 600 proposal calls).
- One provider attempt per logical call; maximum 8,192 output tokens per call.
  No tool execution, external feature acquisition, or scientific-judge calls.
- Native Adam defaults: CPU float64, learning rate 0.01, up to 2,000 epochs,
  validation every 10 epochs, patience 100 checks. Generational patience is five.
- Each array task gets one CPU, 16 GB memory, two hours. At most two concurrent
  tasks by default. The replay limit is 120 seconds per trajectory/endpoint.

`plan.json` freezes the model ID, full Python source hash, library versions,
public prompt hashes, train/validation fingerprints, roster, and budgets before
calls. The wrapper adds the exact campaign/task identity to request cache keys
and explains the native increment convention without changing benchmark prompts.

Each task saves cached calls, call logs, the native generation checkpoint, a
sealed `native-selection.json`, and a sealed `result.json`. A finished task
requires no new calls or fitting. An interrupted task resumes completed
generations; an interrupted in-progress fit restarts from its cached proposal.
If a process dies after the provider receives a request but before a response
can be cached, a resumed call can incur another charge. Provider nondeterminism
before a response is saved cannot be eliminated. Unknown usage remains explicit.
Terminal discovery failures remain terminal; the launcher does not silently
grant extra generations. A fresh experiment needs a new output root.

## Delta commands

Use the pinned commit given with the implementation. The commands below assume
`AF_REPO_ROOT` names that clean checkout. No remote commands are run by Codex.
Load your existing `OPENAI_API_KEY` securely into the environment; never paste
the key into chat or print it. Set `AF_D3_MODEL` to the exact intended API model
ID. The historical D3 model ID has not been established by the recovered metrics.

```bash
export AF_REPO_ROOT=/projects/bibo/$USER/repos/autoformalism-phase-b-d3-v1
export AF_PYTHON=/projects/bibo/$USER/venvs/autoformalism-v21/bin/python
export AF_PUBLIC_ROOT=/work/hdd/bibo/$USER/phase_b/inputs/public-prompt-v3
export AF_OUTPUT_ROOT=/work/hdd/bibo/$USER/phase_b/d3-native-pilot-v1
export AF_D3_CONFIG="$AF_REPO_ROOT/configs/phase_b_d3_pilot_v1.json"
read -r -p 'OpenAI API model ID for D3: ' AF_D3_MODEL
export AF_D3_MODEL
bash "$AF_REPO_ROOT/scripts/hpc/submit_phase_b_d3_delta.sh"
```

The launcher checks Torch/OpenAI availability before submission. It submits a
prepare/test/smoke job, the discovery array after successful preparation, and a
report job after the array terminates. Running the launcher again prints the
existing submission receipt instead of duplicating paid jobs. The preparation
job runs the real Torch parity smoke before any API call. No previous historical
D3 model is injected into the Phase-B prompt or used as a starting candidate.

If Torch is absent from the shared environment, install the repository's pinned
optional D3 dependency into an environment intended for this job:

```bash
"$AF_PYTHON" -m pip install 'torch==2.8.0'
```

Inspect results when the jobs finish:

```bash
jq '{status,model,expected,counts,metrics,accounting,limitation}' \
  "$AF_OUTPUT_ROOT/summary.json"
jq '.rows[] | {index,benchmark_id,repetition,status,error,
  one_step:.evaluation.native_one_step.normalized_mse,
  rollout:.evaluation.phase_b_rollout.normalized_mse,
  native_check:.evaluation.saved_one_step_matches,
  bounds_violations:.evaluation.bounds_violations}' "$AF_OUTPUT_ROOT/summary.json"
```

If a report is missing, generate it without calling a provider:

```bash
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" \
  "$AF_REPO_ROOT/scripts/phase_b_d3.py" report --root "$AF_OUTPUT_ROOT"
```

If preparation failed, inspect its logs first; there may be no plan to report:

```bash
cat "$AF_OUTPUT_ROOT/submission/manifest.json"
tail -n 60 "$AF_OUTPUT_ROOT"/logs/*.err
```

After ensuring previous discovery jobs have terminated, resume only pending
tasks with the **same checkout, environment, and output root**:

```bash
export AF_COMMIT=$(git -C "$AF_REPO_ROOT" rev-parse HEAD)
af_pending=$(jq -r '[.rows[] | select(.status=="pending") | .index] | join(",")' \
  "$AF_OUTPUT_ROOT/summary.json")
if [[ -n "$af_pending" ]]; then
  sbatch --account=bibo-delta-cpu --partition=cpu --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=16G --time=02:00:00 --export=ALL \
    --array="$af_pending%2" --job-name=d3-phaseb-resume \
    --output="$AF_OUTPUT_ROOT/logs/resume-%A_%a.out" \
    --error="$AF_OUTPUT_ROOT/logs/resume-%A_%a.err" \
    "$AF_REPO_ROOT/scripts/hpc/run_phase_b_d3_delta.sh" run
fi
```

Then rerun the report command. Per-task locks prevent concurrent duplicate work.
Review the pilot before using the full config with a **fresh** output root.

## Local checks

```bash
.venv/bin/python -m pytest -q tests/test_d3_rollout.py tests/test_phase_b_d3.py
.venv/bin/python scripts/smoke_phase_b_d3.py
# On the fitting environment, require the actual Torch path:
.venv/bin/python scripts/smoke_phase_b_d3.py --require-torch
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```
