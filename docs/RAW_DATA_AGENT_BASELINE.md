# Raw-data frontier-agent baseline

This baseline asks whether a current hosted model can solve a public discovery
cell directly, without Autoformalism's iterative proposer, scientific judge, or
search controller.

## Information and tools

Each independent repetition receives exactly:

- the finalized public proposer prompt;
- the public development `train.csv` and `validation.csv` files;
- the public runtime symbol roles; and
- a provider-hosted Python/code-execution tool.

The agent may inspect trajectories, write analysis code, fit parameters, and use
validation to choose a structure. It cannot access test data, private equations,
web search, or external data. It must return one strict
`ProposerCandidateV2`. The response and request are content-addressed, cached,
logged, and checkpointed. Credentials are read only from environment variables
and are never written to artifacts.

The returned structure is compiled and refit by the shared Autoformalism
runtime using train only for continuous optimization and validation for the
reported development NMSE. No deterministic pruning or scientific judge is
applied. Test data remains unopened. This makes the baseline strong but
auditable: it gets the same public data and task, while its own hosted analysis
is the discovery algorithm.

## Two-cell pilot

The frozen pilot contains two contrasting Phase-B cells:

1. named Dalla Man T2, easy tier;
2. anonymous-system task, opaque hard tier.

It uses GPT-5.6 Sol and Gemini 3.1 Pro Preview with three independent
repetitions each, for 12 tasks total. Each task is bounded to one hosted agent
response, at most 12 OpenAI tool calls where the provider exposes that limit,
30,000 output tokens, 20 minutes per provider request, and two provider
attempts. The shared refit uses one start, 50 function evaluations, and a
five-minute fit timeout.

The three repetitions estimate stochastic variability; they are not treated as
scientifically meaningful seeds and hosted providers may not guarantee exact
seed reproducibility. The pilot is a development comparison, not a final test
evaluation.

## Outputs

Each run directory contains `run_config.json`, `agent_result.json`,
`candidate.json`, `evaluation.json`, `status.json`, a content-addressed cache,
and an append-only event log. `scripts/summarize_raw_data_agent_pilot.py`
creates a compact `summary.csv` after the array completes.
