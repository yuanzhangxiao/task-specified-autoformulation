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
  --output-root artifacts/runs
```

The OpenAI client reads `OPENAI_API_KEY` from the environment through the
official SDK. The key is not written to configuration, checkpoints, caches, or
logs.

For a local free model, use an Ollama identifier such as
`ollama:gpt-oss:20b`. Proposer and judge models may use different providers.

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
