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

The agent may inspect trajectories, write analysis code, fit parameters on the
training split, and use validation to choose a structure. It cannot access test
data, private equations, web search, or external data. The primary baseline must
return one strict `RawAgentFittedModel`: a `ProposerCandidateV2`, one fitted
value for every global parameter, and a short fit-method summary. The response
and request are content-addressed, cached, logged, and checkpointed. Credentials
are read only from environment variables and are never written to artifacts.

The returned candidate is compiled by the shared Autoformalism runtime. One
cached diagnostics-only repair turn is allowed for a typed syntax or runtime
contract failure; it receives the candidate, public symbol contract, and
deterministic errors, but no data, fit score, or scientific feedback. This
matches the non-scientific repair allowance in the main pipeline. Parameter
names, uniqueness, completeness, bounds, and finiteness are schema-validated.
The evaluator then simulates the candidate with exactly the agent-returned
values; it fits no model parameter or trajectory-specific initial condition.
Only target normalization scales are estimated from train. Validation is used
for the frozen development report and test remains unopened. No deterministic
pruning or scientific judge is applied during generation or selection.

The earlier `structure_only` contract is retained as an explicitly labeled
ablation. It returns only `ProposerCandidateV2` and therefore necessarily uses
Autoformalism parameter fitting. Results from that contract must be called
"GPT structure + Autoformalism fit," not a full fitted-model agent baseline.

## Two-cell pilot

The frozen pilot contains two contrasting Phase-B cells:

1. named Dalla Man T2, easy tier;
2. anonymous-system task, opaque hard tier.

The original structure-only pilot used GPT-5.6 Sol and Gemini 3.1 Pro Preview
with three independent repetitions each. The fitted-model confirmation begins
with GPT-5.6 Sol only: the same two cells and three repetitions, for six tasks.
Each task is bounded to one hosted agent response, at most 12 OpenAI tool calls,
30,000 output tokens, 20 minutes per provider request, and two provider
attempts. The fitted-model evaluator performs direct `solve_ivp` simulation and
does not run an optimizer.

The value 12 is a predeclared pilot compute budget, not a claim that 12 is an
intrinsically fair or sufficient number. Every response now records the
requested limit, the provider-echoed limit when available, and the observed
code-interpreter records by status. In the Responses API, completed,
incomplete, and failed records are terminal processed calls; `in_progress` and
`interpreting` records are nonterminal and are reported separately. This avoids
mistaking a provider-retained ignored/nonterminal attempt at the tool ceiling
for an additional completed call. `scripts/audit_raw_data_agent_budget.py`
inspects the cached provider response offline, without making another paid
call. Both total records and processed calls remain visible. A matched GPT-5.6
sensitivity changes only this limit from 12 to 24. Results from the 12-call
pilot must therefore be described as budget-conditioned; if the 24-call run
materially improves validity or fit, both operating points should be reported.

The three repetitions estimate stochastic variability; they are not treated as
scientifically meaningful seeds and hosted providers may not guarantee exact
seed reproducibility. The pilot is a development comparison, not a final test
evaluation.

## Full Phase-B primary baseline

The full development experiment applies the unchanged GPT-5.6 Sol fitted-model
contract to all 40 registered Phase-B cells with three independent repetitions,
for 120 expected models. It covers every Dalla Man task, canonical and perturbed
dynamics, named and obfuscated variants, both difficulty tiers, both CSTR
semantic variants and tiers, and both alien-device semantic variants and tiers.
The six exact pilot runs are reused only if the content-addressed request hash
matches the full request; otherwise execution fails closed rather than mixing
protocols.

Complete evaluation means more than NMSE, but it does not collapse unlike
evidence into one opaque score. Each run reports:

- hosted-agent completion and contract-repair provenance;
- exact-value train and validation rollout NMSE;
- failed trajectories and soft-constraint violations;
- deterministic validity and hard public-requirement compliance;
- every structure-only absolute scientific verdict from the validated 120B
  paired-question-consensus judge; and
- response, orientation, and neutral missing-unit-repair coverage.

The scientific judge does not see parameter values, trajectories, or NMSE, so
its result remains an evaluation of submitted model semantics rather than a
second fit assessor. Natural-candidate verdicts are descriptive and do not have
gold-label accuracy. The combined report separately identifies candidates with
known scientific failures and candidates for which all applicable questions
passed. Test data remain unopened in this development matrix.

## Outputs

Each run directory contains `run_config.json`, `agent_result.json`,
`candidate.json`, `evaluation.json`, `status.json`, a content-addressed cache,
and an append-only event log. Fitted-model artifacts additionally record the
agent-provided parameter values, fit-method summary, and
`parameter_refit_applied: false`. `scripts/summarize_raw_data_agent_pilot.py`
creates a compact `summary.csv` after the array completes.

For the full matrix, `scripts/summarize_raw_data_agent_full_evaluation.py`
joins numerical artifacts, the authoritative offline tool-budget audit, and the
scientific self-audit. It writes a run-level CSV plus JSON and Markdown
aggregates. The run-level `tool_call_count` in the earliest six artifacts counts
one retained nonterminal `interpreting` record; the offline audit is
authoritative for processed-call limits. New calls exclude `interpreting` and
`in_progress` records at ingestion.

## Common refit and scientific comparison

The provider-agent outcome and numerical evaluator are reported separately.
The primary fitted-model result uses the agent's exact values. A common-refit
ablation may subsequently refit frozen structures under identical
Autoformalism settings, but it answers a different question. Its fixed-RK4
screen is only an optional warm start: a screening failure falls back to an
independent `solve_ivp` start and cannot suppress the final refit. The same
configuration accepts either a raw-agent run (`--source-run`) or an
Autoformalism search summary (`--source-summary`).

NMSE is not a mechanism-validity metric. Two fit-free audits are therefore
separate from numerical fitting. First,
`scripts/build_raw_agent_scientific_audit_pairs.py` duplicates each prior GPT
structure into an identity-blinded self-pair. The validated 120B judge reads it
in both orientations, and the report retains deterministic validity, public
requirement compliance, target/initialization semantics, and every absolute
scientific assessment. Direct comparative answers are discarded because the
two candidates are identical. This audit is descriptive and claims no judge
accuracy. Second, when a frozen Autoformalism reference exists,
`scripts/build_raw_agent_method_pairs.py` creates identity- and metric-blinded
cross-method pairs. Their winner is also unlabeled; question-level assessments,
orientation disagreements, response coverage, and consensus preference are
reported without assuming Autoformalism is ground truth.
