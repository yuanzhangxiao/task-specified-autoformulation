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

Model arguments use `provider:model`. Supported providers are `openai`,
`gemini`, and `ollama`; a model without a provider prefix defaults to OpenAI.

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

For Gemini, install `pip install -e '.[gemini]'`, set `GEMINI_API_KEY`, and use
an identifier such as `gemini:gemini-3.6-flash`. Proposer and judge models may
come from different providers, for example an OpenAI proposer with a Gemini
judge. Gemini calls use JSON-schema structured output and are cached and logged
under the same rules as other providers.

The preregistered family study uses `original_b1/hard` and `benchmark6/hard`,
five seeds, and the GPT→Gemini, Gemini→GPT, and Gemini→Gemini cells. Run it with
`python scripts/run_family_study.py --data-root "$AUTOFORMALISM_DATA_ROOT"
--output-root artifacts/family-study-v1`. The GPT→GPT cell is intentionally
omitted because it is reused from the main Autoformalism experiment. The runner
skips completed runs and resumes partial checkpoints.

For a local free model, use an Ollama identifier such as
`ollama:gpt-oss:20b`. Proposer and judge models may use different providers.
Local structured-output requests can take several minutes on first load.
`--llm-timeout-seconds` defaults to 900 seconds. `--llm-max-output-tokens`
defaults to 2048 and prevents an incomplete structured response from generating
until the request timeout. Ollama thinking is disabled for structured calls so
reasoning tokens cannot consume the JSON response budget.
One-step search fits default to one start, 50 residual evaluations, a
300-second per-candidate wall-clock limit, and fixed-step RK4 integration.
Configure these limits with `--fit-starts`, `--fit-max-nfev`, and
`--fit-timeout-seconds`. A timeout rejects and checkpoints only the affected
candidate; later search rounds continue. After validation selection, the frozen
structure is warm-started from its selected fitted parameters and refitted with
adaptive `solve_ivp`, one start, 150 evaluations, and a 300-second limit. The
last two limits are configurable with `--final-fit-max-nfev` and
`--final-fit-timeout-seconds`. If that adaptive refit times out, the controller
deterministically falls back to the fixed-step refit rather than losing the
completed search.
Candidates whose complete state vector and derivatives are observed use fast
bounded derivative regression during parameter fitting. Candidates with any
genuine latent state automatically use the ODE-rollout fallback. Both paths are
ranked using causal one-step rollout error.
Search-time pruning evaluates one conservative reduced support and refits it
once. This avoids an exhaustive series of optimizer runs for every candidate;
the unpruned fit remains eligible when the reduced support fails validation.
The proposal boundary also removes unenforceable constraints attached only to
undeclared prose concepts, records that repair, and keeps recent structural
duplicates in bounded feedback so the proposer is told to change equations or
dependencies rather than merely rename symbols.

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

Every baseline CLI run is supervised by a separate process and has a hard
wall-clock limit (30 minutes by default). Configure it with
`--wall-timeout-seconds`. The runner writes `run_status.json` with
`complete`, `failed`, or `timed_out`; a timeout terminates the complete child
process group while preserving any checkpoints already written.

`sindy` uses supplied training derivative labels with a degree-two polynomial
library, unary `tanh` features, and deterministic STLSQ. It does not create
explicit lag columns. `llm_feature_sindy` makes one cached proposer call and
accepts only parameter-free algebraic features composed from the exact supplied
symbol contract. The ordinary degree-two polynomial/tanh library is built over
the supplied variables; LLM-designed features enter linearly so SINDy can select
them without generating unrequested quadratic cross-products:

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
  --maximum-expression-size 30 \
  --wall-timeout-seconds 1800
```

The same limit is passed to PySR's native timeout and is also enforced by the
outer supervisor, which cleans up Julia child processes if the limit expires.

The no-judge Autoformalism ablation uses the ordinary experiment CLI. It makes
no judge calls, returns no judge feedback to the proposer, and ranks with
validation fit, simplicity, and deterministic diagnostics:

```bash
python scripts/run_experiment.py [ordinary arguments] --no-judge
```

`d3_native_no_tools` preserves D3's iterative propose-fit-reflect workflow,
native PyTorch Adam optimizer, and teacher-forced forward-Euler state update.
It models every dynamic channel observed in the selected tier for which the
benchmark supplies derivative labels. External feature-acquisition tools are
disabled, every generation is checkpointed, validation selects the candidate,
and test is opened once after selection:

```bash
python scripts/run_baseline.py \
  --data-root "$AUTOFORMALISM_DATA_ROOT" \
  --benchmark-id original_b1 \
  --tier easy \
  --method d3_native_no_tools \
  --model openai:MODEL_NAME \
  --d3-generations 20
```

Install its optional PyTorch dependency first:

```bash
pip install -e '.[d3]'
```

The upstream defaults are learning rate `1e-2`, 2,000 maximum epochs,
validation every 10 epochs, and 100 non-improving validation checks before
early stopping. The selected validation parameters are frozen for test; there
is no Autoformalism warm start, bounded least-squares fit, RK4 screening, or
`solve_ivp` refit. The complete baseline is protected by
`--wall-timeout-seconds`.

For security, expressions are returned through the restricted schema and
compiled directly into differentiable PyTorch operations; arbitrary
LLM-generated Python is never executed. Thus the proposal representation and
leakage-safe split harness are adaptations, while the numerical fitting method
matches upstream D3. The audited upstream reference is pinned to commit
`ee86212dfd5935bb0c9626eaa0570223ff7ecf1c`.

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
