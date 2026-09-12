# Diagnosis/repair redesign and pre-fitting judge ablation

This opt-in milestone replaces the special-case v6 repair loop in a **new campaign**,
not in production search. It does not implement MCTS, change finalized prompts,
alter benchmark data, or change the final evaluation judge.

## Frozen comparison

| Arm | Runtime evidence and redesigned edits | Pre-fit scientific judge | Fitter |
| --- | --- | --- | --- |
| `redesigned_runtime` | Yes | None (no ad hoc critic either) | Verified feasibility baseline |
| `redesigned_prefit_judge` | Identical | Established paired hybrid protocol, GPT-OSS-120B | Identical |

The same two unresolved opaque-hard public candidates, seeds 0 and 1, enter both
arms. Each arm independently assesses the baseline and permits four repair rounds.
Thus there are four tasks, at most 16 repair rounds and 20 fitted candidate versions
including baselines. Each task has 40 physical proposer requests, 393216 tokens,
and five attempts per transaction. Both proposers are GPT-OSS-20B with the existing
frozen revision and inference settings. The plan lists counterbalanced arm order
by seed, but the split jobs execute independently: there is no enforced temporal
ordering across arms. Serving within the judge arm is grouped by model, not
synchronized with the runtime-only job.

The comparison isolates **adding pre-fit scientific feedback conditional on the
redesigned repair mechanism**. Historical v6 is not a matched control for the
redesign: fitter and initialization changes also intervene. A separate old-loop
arm would be necessary to estimate the redesign's standalone causal contribution.
There is no automatic winner, weighted score, or claim of statistical power from
two seeds. This is an integration/diagnostic pilot, not a benchmark-wide result.

Fitting is pinned to Astra's verified `549e039` feasibility baseline (plus its
piecewise and initialization dependencies): 120 seconds collocation, 180 seconds
refinement, 240 evaluations, `rollout_or_observed`, five-second node warmup,
`ftol=None`, `recovery_policy=feasible`, ten starts and ten-second probes. The mesh
remains one substep. The newer experimental mesh settings are not silently adopted.

Numeric latent initial conditions become **shared values learned on training**.
Their old numeric values are only starting guesses. Direct observed states use the
allowed first measurement. Causal maps use first public measurements/inputs with
coefficients learned on training and frozen for validation. The same policy applies
to both arms. No per-validation trajectory latent-state fitting is performed.

## Evidence and routing

Each finding includes its source, stage, candidate hash, component (when known),
code, observation, evidence, uncertainty, blocking flag, and recheck instruction.
The sources stay separate:

1. **Runtime, pre-fit:** grammar/schema/symbols, closure, algebraic cycles, target
   mappings, necessary input/memory/nonlinearity paths, and conservative domains.
2. **Scientific judge, pre-fit, judge arm only:** scientific absolute concerns and
   comparative disadvantages under the established rubric; always advisory.
3. **Fitter, post-fit:** initializer/refinement outcomes, finite-evaluation counts,
   fresh causal rollout status, failed trajectory IDs, and train/validation NMSE.

The runtime prioritizes one category: blocking runtime contract, scientific
requirement concern, unresolved domain, numerical feasibility, then predictive
refinement. It **does not assign a scientific cause or mandate an equation**.
The proposer chooses a compact hypothesis and a coherent bounded edit. A timeout
is not evidence of a missing mechanism; finite rollouts are not scientific recovery.

Domain analysis covers division, negative integer powers, logarithms and square
roots. Unknown ranges produce `DOMAIN_UNRESOLVED`, not a claim of reachable
singularity and not an automatic pre-fit rejection. A certified violation is
blocking. Public supplied bounds and effective parameter domains may support an
interval certificate; legacy proposer magnitude ranges do not. There is no claim
that all dangerous trajectories can be detected statically. The compiler/fitter
may separately decline unsupported expressions, labelled as capability/contract
failures rather than scientific invalidity.

Memory-path and nonlinear-path checks are **necessary structural conditions**, not
complete scientific mechanism compliance. For this frozen prompt the runtime checks
an internal dynamic state on an input-to-target path and a nonlinear target pathway.
It does not certify that the proposed dynamics correctly realize memory, coupling
or feedback. The scientific judge evaluates scientific requirements independently.

The next request contains the public task (obsolete response-schema section
removed), current equations/kinds, short parameter aliases and immutable parent
roles, forcing channels, target mappings, initialization policy, structured report,
recent action outcomes, and pending/provisional edits. Numerical evidence is compact;
full solver artifacts remain on disk. Formatting failures never erase the last
numerical assessment. A best finite evaluated candidate is retained separately from
the current working candidate; validation NMSE selects that incumbent, not a blended
judge/numerical score.

## Proposed action and transaction

The actual provider JSON schema is generated from `RepairAction` using the same
strict-schema transport as the construction pilot. Its fields are:

```json
{
  "scope": "function",
  "hypothesis": "Replace the uncertain saturating denominator without removing inputs.",
  "equations": [
    {"component": "f", "kind": "preserve", "expression": "k*tanh(m)-rate*f", "parameters": []}
  ],
  "remove": [],
  "mappings": [],
  "initializers": [],
  "keep": []
}
```

This illustrative equation is valid only when those symbols and dependencies
belong to its parent; it is not a benchmark recommendation.

- `scope`: `function`, `model`, or `no_change`.
- Equation `kind`: `preserve`, `dynamic`, or `algebraic`. A new variable needs an
  explicit kind and a complete defining equation.
- Up to six equation edits, two added variables, two removed variables, three target
  mappings, four initializer edits, and twelve generated variables overall.
- Function scope preserves signed additive hyperedge/source contracts. Model scope
  explicitly permits dependency/sign changes, inventory/type changes, target
  mappings, and initializer changes, including edits outside the named finding.
- Parent parameter names are automatically available and roles remain immutable.
  New shared parameters may be declared once. Direct gains and offsets have roles
  inferred; ambiguous internal parameters need a qualitative role. New direct
  gains are nonnegative magnitudes with the equation's chosen outer sign; offsets
  are unrestricted. Existing unrestricted parameters stay unrestricted.
- A mapping edit has `channel` and `expression`. A latent initializer edit has
  `state` and `causal_map`; null restores a shared training-fitted value. A map has
  `mode: map`, a restricted expression and local parameters. The proposer does
  not estimate numerical latent initial values.
- Known extra `keep` entries (including supplied inputs) are harmless. Unknown
  symbols, contradictory actions and ambiguous shared declarations are rejected.

Local equation checks retain valid edits provisionally and name the remaining
failures. Related mapping/initializer/removal edits also survive retries. The next
request exposes that draft; omission of a pending component cannot silently commit
a partial repair. Explicit keep can abandon a proposed component edit. Explicit
`no_change` abandons the whole draft, without hidden changes or a repeated fit.

The complete transaction is committed only after whole-model compilation,
dependency closure, public structural obligations, domain and initialization
checks succeed. Model-scope edits do not silently redefine a supplied forcing
channel in this pilot. Unsupported whole-model repairs remain explicit failures.
Two consecutive no-change decisions stop that task as `stopped_no_progress`, not
success; other seeds/arms continue. Exhausted revision attempts consume a round
and preserve the last candidate/evidence.

## Established judge, not the former ad hoc critic

The judge arm calls `PairedHybridJudge`: atomic evidence, blinded forward/reverse
orientation, consensus, and existing hybrid scoring. The second seed is the
existing fallback after terminal failure, not an extra independent vote. Initial
absolute review uses a self-pair; later reviews compare parent and challenger.
The frozen plan records all judge prompt strings, scoring, model revision and
inference settings. The model revision resolves once at freeze time.

The judge receives public task/context and the two **unfitted** candidate artifacts,
including symbolic training-fitted initializer parameters. It does not receive
NMSEs, fitted values, trajectories, previous repair hypotheses or the external
domain certificate. The established judge's own internal deterministic assessments
remain part of its unchanged implementation. This is different from handing it the
search runtime's external findings.

Only consensus absolute FAILs and comparative parent-preferred concerns become
advisory scientific feedback. Indeterminate means unavailable evidence, not pass.
The original rubric is not retuned here to discount parameters expected to be
pruned. Its behavior on unpruned pre-fit candidates is precisely an experimental
limitation, and comparative parsimony may be less reliable at this stage. Final
evaluation remains outside this pilot and unchanged.

## Checkpoints, costs and hardware

Plan and public asset hashes, runtime source and launcher hashes must match on
resume. Tasks have separate state/transactions/calls. Every response and named
diagnostic is persisted. Provider budgets survive model-server swaps. Once a
native fit or scientific review is marked started, interruption without a terminal
artifact is retained as interruption; it does not receive a fresh hidden budget.
A fitter infrastructure/capability failure blocks that task instead of generating
scientific repair instructions from it. See per-task state for details.

`results/summary.json` separates arms, baselines, attempted edits, fit outcomes,
finite candidates, best NMSE, solver messages, revision diagnostics, proposer and
judge request/token costs. Missing usage is explicitly counted. Full transactions
and numerical artifacts remain beside it. `status=complete` means all task outcomes
are terminal, not that every model succeeded.

ACES now submits **two independent arm jobs**, each with its own CPU preparation
dependency (regressions, real synthetic fitter smoke and pinned snapshots):

| Arm | GPU allocation | Wall limit | Models downloaded/served |
| --- | --- | --- | --- |
| `redesigned_runtime` | 1 H100 | 2 hours | 20B proposer only |
| `redesigned_prefit_judge` | 2 H100 | 4 hours | 20B proposer and 120B judge, swapping servers |

The runtime-only job does not wait for the judge job or its 120B model download.
Two H100s preserve the existing judge serving configuration (tensor parallelism 2,
32768-token context); this is not a claim that all possible 120B serving setups
require two GPUs. The proposer remains tensor parallelism 1 in both jobs. Smaller
resource requests may schedule sooner, but queue time is not guaranteed. Per-fit,
per-request and per-task scientific budgets are unchanged; allocation expiry uses
the existing checkpoint/resume policy rather than silently shrinking those budgets.
The SIF has the previously confirmed SHA
`9c56389af06cafbf4aa8bd825393f606ac7f2a18e1a700c5999b1ed47f7f9c1e`.
No automatic image rebuild or hash override occurs.

Run from a clean pinned worktree with `AF_REPO_ROOT`, `AF_OUTPUT_ROOT` set. Use
a **new output root** for this launcher revision; do not run an old combined job
and its split replacement. Run the two submissions sequentially (not concurrently
while first freezing the shared plan). The GPU jobs can subsequently run in parallel.
Either arm can be submitted later without changing the plan:

```bash
AF_ARM=redesigned_runtime bash scripts/hpc/submit_repair_comparison_aces.sh
AF_ARM=redesigned_prefit_judge bash scripts/hpc/submit_repair_comparison_aces.sh
```

`AF_ARM` defaults to `redesigned_runtime`. Each invocation submits only that arm.
Both use the same immutable plan and source hashes. Arm-specific worker locks
prevent duplicate execution while allowing the other arm to proceed. Manual
all-arm CLI execution takes both locks. Summary writes are serialized separately.

Each manifest is `submissions/<arm>/manifest.json`, with `prepare_job`, `job_id`,
and the common `plan_sha256`. Each result is `results/summary-<arm>.json`; it can
be complete while the other arm is pending. `results/summary.json` remains the
combined comparison and is incomplete until both arms are terminal. To inspect
one arm without waiting for the other:

```bash
python scripts/repair_feedback_comparison.py summary --root "$AF_OUTPUT_ROOT" --arm redesigned_runtime
```

Defaults use the known ACES venv, SIF, HF cache and the frozen
`staged-fitter-rescue-v1-c1754fe` source. `AF_VLLM_IMAGE`, `AF_PYTHON`,
`AF_HF_HOME`, `AF_SOURCE_RESCUE_ROOT`, `AF_JUDGE_REVISION` can be set explicitly.
Repeating submission prints that arm's existing manifest. An ambiguous scheduler
timeout leaves an arm-specific intent guard: inspect accounting before taking any
further action. It does not block submission of the other arm.
After an unambiguously terminal GPU job with successful CPU preparation, resume
the same frozen experiment using `AF_RESUME=1` and the same `AF_ARM`; queued or
uncertain jobs are refused. Preparation/resume has no cross-arm dependencies.

CPU preparation and GPU workers both explicitly request one node and one task,
as required by the ACES submission policy. The scheduler stub rejects missing
allocation flags so this check runs locally before submission. A rejected initial
submission may leave an intent directory without a job ID; do not delete guards
after an ambiguous scheduler timeout. For a launcher fix, use a new pinned worktree
and output root, preserving the rejected attempt and its frozen plan for audit.

For Delta, use the focused regression suite and
`scripts/smoke_repair_feedback_comparison.py` in its existing Python 3.12 venv.
Do not duplicate the GPU comparison on a different configuration until this matched
ACES integration result is reviewed. Scripts work independently of notebooks.

## Local evidence and limits

- Forty-nine focused tests exercise failure and success paths, the real paired-judge
  adapter with recorded-style synthetic replies, budget/resume, and scheduler stubs.
  Split-launcher regressions cover independent dependencies/resources, ambiguous
  submissions, selected snapshot downloads, arm isolation, locking and summaries.
- The split-launcher full local suite finished with 1379 passed, three optional
  Torch skips and the same 39 missing-benchmark-fixture failures as the preceding
  milestone. Ruff and shell syntax checks passed.
- The real fitter smoke learned a shared latent boundary near 2.0 from a zero
  initial guess, with validation NMSE about 3e-16; no live LLM or hidden latent
  training data entered the fit.
- Offline audit of 26 public v5 replies: 9 committed, 12 rejected, 5 unchanged.
  Rejections: 10 required explicit model scope, one lost public nonlinearity,
  one undeclared symbol. Historical responses are syntactically adapted without
  changing their mathematical expressions. This is not counterfactual search.
- Private-data-dependent legacy tests need files absent from the clean worktree.
  Do not populate private references merely to prepare this public experiment.
- General certificates are incomplete; timeouts remain possible; no universal
  nonsmooth/domain support or scientific recovery is claimed. MCTS, learned
  impact-ranking, and final pruning/evaluation are deferred.
