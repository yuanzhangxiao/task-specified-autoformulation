# Autoformalism

Autoformalism discovers and validates continuous-time dynamical-system
structures from benchmark trajectories and task prompts. The Phase 1 pipeline
uses restricted expression parsing, bounded numerical fitting, whole-term
pruning, structured proposer and judge responses, checkpointed beam search,
validation-based structure selection, and a one-time final test evaluation.

Python 3.11 or newer is required.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Set the public benchmark root either in the environment or on each command:

```bash
export AUTOFORMALISM_DATA_ROOT=/path/to/data_raw
```

Only registered public benchmark directories are loaded. Private or hidden
benchmark paths are rejected.

## Validate a dataset

```bash
python scripts/inspect_dataset.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark original_b1 \
  --tier easy
```

Supported Phase 1 identifiers can be listed with:

```bash
python scripts/inspect_dataset.py --list
```

## Run an experiment

Model arguments use `provider:model`. Supported providers are `openai` and
`ollama`; a model without a provider prefix defaults to OpenAI.

```bash
python scripts/run_experiment.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --seed 7 \
  --proposer-model openai:YOUR_PROPOSER_MODEL \
  --judge-model openai:YOUR_JUDGE_MODEL \
  --iteration-budget 10 \
  --beam-size 3 \
  --llm-timeout-seconds 900 \
  --llm-max-output-tokens 2048 \
  --output-root artifacts/runs
```

The OpenAI client reads `OPENAI_API_KEY` from the environment through the
official SDK. The key is not written to configuration, checkpoints, caches, or
logs.

For a local free model, use an Ollama identifier such as
`ollama:gpt-oss:20b`. Proposer and judge models may use different providers.
Local structured-output requests can take several minutes on first load.
`--llm-timeout-seconds` defaults to 900 seconds. `--llm-max-output-tokens`
defaults to 2048 and prevents an incomplete structured response from generating
until the request timeout. Ollama thinking is disabled for structured calls so
reasoning tokens cannot consume the JSON response budget.
One-step rolling fits default to three multistarts and 400 residual evaluations
per start. Use `--fit-starts` and `--fit-max-nfev` to increase these limits for
final experiments after a smaller pilot succeeds.
Candidates whose complete state vector and derivatives are observed use fast
bounded derivative regression during parameter fitting. Candidates with any
genuine latent state automatically use the ODE-rollout fallback. Both paths are
ranked using causal one-step rollout error.

### Dry run

A dry run validates the benchmark, prompts, paths, roles, and split
fingerprints without constructing an LLM client or creating an experiment:

```bash
python scripts/run_experiment.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id benchmark5 \
  --tier medium \
  --dry-run
```

### Offline mock and synthetic smoke run

`--mock-llm` uses deterministic structured candidates and judges while still
running validation, fitting, pruning, checkpointing, final refitting, and test
evaluation. The built-in `synthetic` benchmark needs no benchmark files, but
`--data-root` must name an existing directory:

```bash
python scripts/run_experiment.py \
  --data-root . \
  --benchmark-id synthetic \
  --tier easy \
  --seed 0 \
  --iteration-budget 1 \
  --beam-size 1 \
  --output-root /tmp/autoformalism-smoke \
  --mock-llm
```

## Baselines

The shared baseline runner preserves trajectory boundaries, fits only on the
training split, selects SINDy sparsity on validation one-step rollout MSE, and
opens test exactly once after selection:

```bash
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method persistence \
  --seed 0

python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method sindy \
  --seed 0
```

`sindy` uses supplied training derivative labels with a degree-two polynomial
library, unary `tanh` features, and deterministic STLSQ. It does not create
explicit lag columns. `llm_feature_sindy` makes one cached proposer call and
accepts only parameter-free algebraic features composed from the exact supplied
symbol contract:

```bash
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method llm_feature_sindy \
  --model openai:MODEL_NAME
```

PySR is optional because it installs a Julia-backed runtime:

```bash
pip install -e '.[pysr]'
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method pysr \
  --pysr-iterations 40 \
  --maximum-expression-size 30
```

The no-judge Autoformalism ablation uses the ordinary experiment CLI. It makes
no judge calls, returns no judge feedback to the proposer, and ranks with
validation fit, simplicity, and deterministic diagnostics:

```bash
python scripts/run_experiment.py [ordinary arguments] --no-judge
```

The upstream D3 repository is an application tied to its own Hydra environments,
executes LLM-generated Python, and exposes test data during candidate fitting.
The default `d3_no_tools` implementation therefore preserves D3's iterative
propose-fit-reflect workflow while replacing executable code with this project's
restricted candidate schema. It disables external feature-acquisition tools,
uses the fixed observed-state skeleton, checkpoints every generation, selects on
validation, and opens test once after selection:

```bash
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method d3_no_tools \
  --model openai:MODEL_NAME \
  --d3-generations 20
```

This is a security- and leakage-safe D3 adaptation, not byte-for-byte execution
of the upstream repository. The audited upstream reference is pinned to commit
`ee86212dfd5935bb0c9626eaa0570223ff7ecf1c`. For audit or compatibility experiments,
`d3_no_tools` also supports a versioned external bridge contract.
`--d3-command` receives `REQUEST_JSON RESPONSE_JSON`; the request has the task
prompt, train/validation trajectories and derivatives,
`external_tools_enabled: false`, and `candidate_submission_enabled: true`, but
no test data. The bridge must write
`{"equations": {"TARGET": "RHS"}}`. The equations are then parsed safely and
evaluated by this repository. This keeps upstream D3 code and dependencies in a
separate environment:

```bash
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method d3_no_tools \
  --d3-command /path/to/d3-environment/bin/python /path/to/d3_bridge.py
```

## Resume

Experiment directories are deterministic:
`<output-root>/<benchmark>_<tier>_seed<seed>`. A normal run refuses to
overwrite an existing checkpoint. Resume with the same data, prompts, models,
seed, budget, and beam configuration:

```bash
python scripts/resume_experiment.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --seed 7 \
  --proposer-model openai:YOUR_PROPOSER_MODEL \
  --judge-model openai:YOUR_JUDGE_MODEL \
  --iteration-budget 10 \
  --beam-size 3 \
  --llm-timeout-seconds 900 \
  --llm-max-output-tokens 2048 \
  --output-root artifacts/runs
```

`run_experiment.py --resume` is equivalent. Checkpoint fingerprints prevent
resuming with incompatible data, prompts, configuration, schemas, or numerical
implementation.

## Summarize results

Summarize one run:

```bash
python scripts/summarize_results.py \
  artifacts/runs/original_b1_easy_seed7
```

Summarize every completed run beneath an output root:

```bash
python scripts/summarize_results.py artifacts/runs
python scripts/summarize_results.py artifacts/runs --json
```

Each run stores:

- `run_config.json`: redacted execution configuration and split fingerprints;
- `summary.json`: selected equations, fitted parameters, and final metrics;
- `checkpoints/`: atomic stage and final-evaluation checkpoints;
- provider-specific caches and JSONL event logs for real LLM runs.

Test metrics are created only after validation selection is frozen and are
never inserted into proposer or judge feedback.

## Colab

[notebooks/run_autoformalism_colab.ipynb](notebooks/run_autoformalism_colab.ipynb)
mounts Google Drive, clones or updates the repository, installs the package,
reads `OPENAI_API_KEY` from Colab Secrets, validates benchmark paths, runs one
benchmark, resumes it, and summarizes the saved results.

## Development verification

```bash
pytest
ruff check .
```

Benchmark data, finalized prompts, API responses, caches, and experiment
artifacts should not be committed.
