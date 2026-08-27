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

The returned structure is compiled by the shared Autoformalism runtime. One
cached diagnostics-only repair turn is allowed for a typed syntax or runtime
contract failure; it receives the candidate, public symbol contract, and
deterministic errors, but no data, fit score, or scientific feedback. This
matches the non-scientific repair allowance in the main pipeline. The valid
structure is then refit using train only for continuous optimization and
validation for the reported development NMSE. No deterministic pruning or
scientific judge is applied. Test data remains unopened. This makes the
baseline strong but auditable: it gets the same public data and task, while its
own hosted analysis is the discovery algorithm.

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

The value 12 is a predeclared pilot compute budget, not a claim that 12 is an
intrinsically fair or sufficient number. Every response now records the
requested limit, the provider-echoed limit when available, and the observed
code-interpreter call count. `scripts/audit_raw_data_agent_budget.py` inspects
the cached provider response offline, without making another paid call. A
matched GPT-5.6 sensitivity changes only this limit from 12 to 24. Results from
the 12-call pilot must therefore be described as budget-conditioned; if the
24-call run materially improves validity or fit, both operating points should
be reported.

The three repetitions estimate stochastic variability; they are not treated as
scientifically meaningful seeds and hosted providers may not guarantee exact
seed reproducibility. The pilot is a development comparison, not a final test
evaluation.

## Outputs

Each run directory contains `run_config.json`, `agent_result.json`,
`candidate.json`, `evaluation.json`, `status.json`, a content-addressed cache,
and an append-only event log. `scripts/summarize_raw_data_agent_pilot.py`
creates a compact `summary.csv` after the array completes.

## Common refit and scientific comparison

The provider-agent outcome and the numerical evaluator are reported
separately. The original five-minute fit remains useful as an end-to-end
failure outcome, but it is not the final cross-method numerical comparison.
The common evaluator first screens every frozen candidate with bounded
fixed-step RK4 and then warm-starts a longer `solve_ivp` refit. The exact same
configuration accepts either a raw-agent run (`--source-run`) or an
Autoformalism search summary (`--source-summary`). This prevents a stiff or
poorly initialized raw structure from being assigned a sentinel NMSE solely
because one short adaptive fit timed out, while preserving that initial timeout
as part of the raw-agent system result.

NMSE is not a mechanism-validity metric. For tasks with a frozen
Autoformalism reference, `scripts/build_raw_agent_method_pairs.py` creates
identity-blinded, metric-blinded pairs and the validated 120B
paired-question-consensus judge evaluates deterministic validity, absolute
scientific questions, and direct comparative questions in both orientations.
The pair winner is intentionally unlabeled: this is a descriptive comparison,
not an accuracy test whose truth is assumed to be Autoformalism. The report
must show question-level assessments, orientation disagreements, response
coverage, and the consensus preference alongside common-refit NMSE.
