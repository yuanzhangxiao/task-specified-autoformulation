# Phase 1 Pipeline Design

## Purpose and invariants

Phase 1 discovers a task-sufficient continuous-time model from
trajectory data and natural-language requirements under partial
observability. The system jointly searches structure, latent states,
observation mappings, and bounded continuous parameters. Proposer-facing
equations use an explicit left-hand side: `dx/dt` or `d(x)/dt` denotes a
derivative, while every plain identifier—including `x_rate`—defines an
algebraic process. A named process becomes a derivative only through an explicit
`derivative_of` link to a declared state.
Normalization converts these forms into the internal state-equation/process
representation before deterministic validation. Numeric bounds are required for
fitted parameters, not for states; qualitative state constraints do not require
invented ranges.

The following are architectural invariants:

1. Train, validation, and test data are distinct typed objects.
2. Proposal and iterative selection can see train results and
   validation results, but never test values or metrics.
3. Current and future target values are never inputs. Every task is
   one-step-ahead, so target sample i-1 may be used causally while predicting i.
4. Only prompt/spec-declared auxiliaries and external inputs may be linearly
   interpolated. Permitted target history uses interval-wise sample-and-hold,
   never interpolation involving a future target sample.
5. At each interval boundary, directly observed candidate states are reset
   from the target/auxiliary values then available. Latent states are never
   revealed or reset; each begins from a declared fixed value or a restricted
   analytic expression of known initial observations/inputs/covariates, then
   propagates from causally maintained estimates.
6. LLM output is untrusted and is parsed through schemas and a
   restricted expression grammar; no `eval`, `exec`, arbitrary Python,
   or unrestricted lambdification is used.
7. Global parameters are shared across trajectories. Latent initial values
   are fixed or computed analytically from causally known initial data.
8. Every LLM call is content-addressed, cached, and logged.
9. Every stage is checkpointed with enough state for deterministic
   resume.
10. Final structure selection uses validation data. The test split is
   opened once, only after selection is frozen.
11. Private/hidden benchmark material is never a runtime input.

## Core typed interfaces

Names below describe contracts, not an imposed module layout.

### Benchmark discovery and loading

```python
class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

class ChannelRole(str, Enum):
    TARGET = "target"
    AUXILIARY = "auxiliary"
    EXTERNAL_INPUT = "external_input"
    FIXED_COVARIATE = "fixed_covariate"

@dataclass(frozen=True)
class TierSpec:
    benchmark_id: str
    tier: str
    proposer_prompt: Path
    judge_prompt: Path
    manifest_path: Path
    split_paths: Mapping[SplitName, SplitPaths]
    time_column: str
    trajectory_id_column: str | None
    sampling_interval: float
    targets: tuple[str, ...]
    auxiliaries: tuple[str, ...]
    external_inputs: tuple[str, ...]
    fixed_covariates: tuple[str, ...]

@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    time: NDArray[float64]
    targets: Mapping[str, NDArray[float64]]
    auxiliaries: Mapping[str, NDArray[float64]]
    external_inputs: Mapping[str, NDArray[float64]]
    fixed_covariates: Mapping[str, float | str]

@dataclass(frozen=True)
class DatasetSplit:
    name: SplitName
    trajectories: tuple[Trajectory, ...]
    fingerprint: str
```

`BenchmarkRegistry.discover(root) -> DiscoveryReport` is read-only and
reports all paths and inconsistencies. `BenchmarkLoader.load_spec(...)`
accepts only a normalized `TierSpec`, checks it against public
manifest/prompt declarations, and fails closed. `load_split(spec,
split)` performs keyed or verified positional joins, validates monotone
time and finite numeric data, checks the sampling interval, and rejects
unavailable channels. Loading Phase-B `TEST` requires `FrozenTestAccess` tied
to the registered cell, tier, and validation-frozen selection hash. Eager
all-split loading is rejected for Phase B, so ordinary development code cannot
accidentally obtain test data.

### Proposal and canonical candidate schemas

```python
class ProposerCandidateV2(BaseModel):
    schema_version: Literal["2"]
    candidate_id: str
    parent_candidate_id: str | None
    states: list[ProposedStateV2]  # rhs, channel identity, initial, constraints
    algebraics: list[ProposedAlgebraicV2]
    parameters: list[ProposedParameter]
    # Mechanism tags are embedded; target mappings are inferred from context.

class CandidateModel(BaseModel):
    schema_version: Literal["1"]
    candidate_id: str
    parent_ids: list[str]
    states: list[StateSpec]
    processes: list[ProcessSpec]
    equations: list[EquationSpec]
    observations: list[ObservationSpec]
    parameters: list[ParameterSpec]
    initial_conditions: list[InitialConditionSpec]
    channel_usage: ChannelUsage
    rationale: str

class CompiledModel(Protocol):
    state_names: tuple[str, ...]
    parameter_layout: ParameterLayout
    def rhs(self, t, state, parameters, forcing) -> NDArray: ...
    def observe(self, t, state, parameters, forcing) -> Mapping[str, float]: ...
```

Expressions are tokenized and parsed into an internal AST. Allowed
nodes are numeric literals, declared symbols, arithmetic operators,
comparisons/piecewise forms explicitly supported by the grammar, and an
extensible registry of safe mathematical functions. The registry is an execution
capability boundary, not a claim that the current function set exhausts the
scientifically meaningful analytical forms. Unsupported named functions must be
lowered to registered mathematical primitives; they can never invoke proposer
code, imports, attributes, callbacks, or arbitrary Python.
Validation rejects unknown syntax, undefined symbols, cycles among
algebraic processes, missing state equations, invalid bounds,
unavailable channels, target leakage, and unsafe domains. Every V2 state
embeds its derivative RHS, so deterministic enrichment creates exactly
one canonical state equation per state. It also fills routine metadata and
uses parameter bounds as the default optimizer initialization range.
Observed states identify their data channel and omit initialization; latent
states provide exactly one fixed or analytic initialization. Target mappings are
inferred by matching benchmark targets to observed channels or same-named
states/algebraics. Constraints and task-mechanism tags are component-local.
Constraint provenance and enforcement are runtime-owned. Constraints originating
in proposer output are labeled `proposer` and treated as soft, editable scientific
hypotheses: fitting receives a smooth violation penalty and train/validation
artifacts report normalized maximum, mean, and frequency of violation. Constraints
from an explicit benchmark/domain contract may be labeled `benchmark` and `hard`;
the proposer cannot grant itself that authority or remove such a constraint.
Runtime finiteness and expression-domain safety remain hard independently of any
proposer declaration. Legacy canonical artifacts without provenance retain hard
behavior for checkpoint compatibility.
Potentially-zero denominators produce a recorded warning and use a
sign-preserving `1e-12` runtime guard. Unsafe logarithm and square-root
domains remain hard validation errors because clipping them changes model
semantics. Constraints whose subjects are undeclared prose concepts are
deterministically removed and recorded before validation; this never creates a
new model symbol or changes an equation. Compilation
walks the validated AST directly; it never evaluates proposer text.

### LLM boundary

```python
class LLMProvider(Protocol):
    def structured_response(
        self,
        request: LLMRequest,
        response_schema: type[BaseModel],
    ) -> LLMResult: ...

class ProposalContext(BaseModel):
    static_prompt: str
    controller_instructions: str
    selected_history: list[CandidateSummary]
    allowed_symbols: AllowedSymbolTable

class ScientificJudgeResult(BaseModel):
    schema_version: Literal["2"]
    category_scores: ScientificCategoryScores
    # aggregate_score is computed by the runtime, not emitted by the LLM
    feedback: list[JudgeFeedback]
```

`ScientificCategoryScores` has explicit fields for mechanistic coherence,
source/sink balance semantics, dynamic plausibility, mechanism coupling and
task sufficiency, nonredundancy/accounting, and latent-state complexity
justification. The runtime computes the weighted aggregate from those fields.
The fixed object prevents invented or omitted categories and avoids arbitrary-key
JSON Schema features unsupported by strict hosted-provider structured outputs.
Historical schema-v1 judge results remain loadable for checkpoint compatibility,
but no new calls request the overlapping task-compliance rubric.

The provider wrapper hashes model name, provider settings, schema,
prompt, and request payload. It atomically reads/writes the response
cache and appends JSONL request/result events with secrets excluded.
One logical proposer round may contain multiple bounded provider attempts for
repairable response-contract errors. Logical calls, provider attempts, and repair
attempts are counted separately. Repair prompts are constructed only from typed,
bounded diagnostics derived from the public prompt, runtime symbol contract,
response schema, and the proposer's own response. They never include reference
equations, private benchmark metadata, test information, or simulator-derived
ground truth. Numerical fit, scientific plausibility, mechanism adequacy, and
structural novelty failures end the logical proposal round and enter ordinary
search feedback rather than contract repair.
For seeded Ollama requests, the first provider attempt uses the configured seed
and bounded repair attempts use deterministic consecutive fallback seeds. The
successful attempt seed is retained in raw-response provenance.
An Ollama response that contains reasoning but no final content receives a
contract-only finalization instruction. The historical `json_schema` control
retries through Ollama's OpenAI-compatible strict-JSON-schema endpoint with
reasoning disabled. A prose-wrapped response is accepted only if it contains
exactly one object that passes the full local schema; post-schema validators
still gate the result. Reasoning text is never parsed as the final response.
The manifest-pinned `json_schema_native_retry` ablation instead keeps the native
`/api/chat` endpoint, configured thinking level, and unchanged schema while
advancing the deterministic attempt seed. Its name is included in the request
hash, so it cannot reuse or overwrite control responses.
Invalid structured responses are recorded and returned as failures,
not passed downstream. The proposer sees compact summaries rather than
raw full history. The judge sees the benchmark judge prompt, certified
deterministic validity facts, and candidate structure, but no numerical fit
metric; deterministic code owns fit scoring and all blocking validity checks.
The judge is asked only for scientific semantic assessment. LLM category scores,
red flags, missing requirements, and edits are advisory. They are retained
for feedback and score tie-breaking but cannot reject a deterministically
valid, successfully fitted candidate.

The calibration-only hybrid judge is specified separately in
`docs/JUDGE_HYBRID_PROTOCOL.md`. It extracts frozen requirements only from the
public task, preserves proposer claims at lower authority, certifies graph facts
for the canonical executable candidate, requests separate absolute A/B semantic
predicates, and retains direct comparative judgments as a distinct residual.
Generic conjunctive groups and all numeric aggregation are runtime-owned. This
prospective protocol must pass held-out question-level calibration before it may
replace the production category-score judge.

Calibration provider failures are first-class outcomes. After bounded
contract-only retries, a terminal structured-response failure is stored in an
append-only ledger and the shard continues. The analyzer reports both conditional
scientific accuracy among valid responses and end-to-end accuracy including these
failures; no hidden reasoning is parsed and no missing score is imputed.
Hybrid calibration resume is additionally guarded by an immutable per-shard run
manifest and exact planned-key validation, including the pair-file digest and
selected pair IDs.
For Ollama calibration, schema-constrained final content is the primary response
transport. Always-on schema-validated tool calls and JSON-primary/tool-fallback
are manifest-pinned experimental ablations. Although all paths ignore hidden
reasoning and retain identical scientific questions and runtime aggregation, the
tool transport materially changed frozen-pair preferences and A/B order
consistency. A tool-call result therefore cannot silently replace a missing
JSON-schema judgment in the primary protocol. Terminal JSON failures remain
first-class outcomes.
One observed Ollama tool serialization defect may receive an exact contract-only
repair: `ver verdict` is normalized to `verdict` only in an evidence-bearing
verdict object with no correct-key collision and a closed-schema verdict literal.
The runtime records the repair count and still applies the complete local schema
and expected-unit validators; no fuzzy field repair or scientific-value repair is
allowed.

The frozen 250-call JSON-schema dataset supports an offline operating-point
analysis that enumerates one-order/one-call and both-order configurations with
one through five repetitions. Repetition subsets are evaluated without new LLM
calls. Confidence intervals cluster by calibration pair, and reports separate
response success, complete-call coverage, conditional pair accuracy, strict
end-to-end pair accuracy, order consistency, and decision variability. This
analysis selects a call budget before any hybrid judge is integrated into search.
It also simulates a bounded adaptive policy that obtains one valid response in
each candidate orientation, retries only a missing orientation with a new seed,
and withholds the pair decision if symmetric evidence remains incomplete. This
policy is evaluated offline before it may become a search-time judge contract.
The selected policy is stored in a versioned JSON configuration. A held-out pair
builder compares canonical repaired baseline-structure fingerprints against the
calibration set and fails closed if the requested number of unseen structures is
unavailable. Held-out labels remain limited to deterministic graph facts and the
pre-existing mutation contracts; the protocol configuration and evaluation gates
are frozen before calls begin.

### Simulation, fitting, metrics, and pruning

```python
class Simulator(Protocol):
    def simulate(
        self,
        model: CompiledModel,
        trajectory: Trajectory,
        parameters: ParameterVector,
        latent_initials: LatentInitialVector,
    ) -> SimulationResult: ...

class Fitter(Protocol):
    def fit(
        self,
        model: CompiledModel,
        train: DatasetSplit,
        validation: DatasetSplit,
        config: FitConfig,
        seed: int,
    ) -> FitResult: ...
```

The forcing adapter supplies piecewise-linear interpolation only for
declared auxiliary/input channels and preserves per-trajectory
boundaries. Iterative screening uses deterministic fixed-step classical RK4 at
measurement intervals, avoiding one adaptive solver launch per prediction
slot. Final refitting and frozen test evaluation use `solve_ivp`. Both backends
check finite values, time coverage, and configured state constraints.

`Fitter` constructs one bounded vector containing global parameters only. If
every candidate state maps directly to an observed target/auxiliary with a
training derivative label, it fits normalized RHS derivative residuals using
deterministic bounded multistart `least_squares` without ODE integration.
Candidates containing genuine latent states fall back to rollout residuals.
Search defaults to one bounded start and 50 residual evaluations. A monotonic
wall-clock deadline is checked inside fixed steps and adaptive RHS calls; expiry
returns a checkpointable failed fit instead of raising through the controller.
The frozen train-plus-validation refit uses the selected search parameters as
its first bounded optimizer point, adaptive `solve_ivp`, one start, 150 residual
evaluations, and a separate 300-second deadline by default.
Failed starts remain diagnostics rather than disappearing. Regardless of the
fitting backend, training and validation rankings use causal one-step rollout
metrics; validation never tunes the parameters.

`MetricEvaluator` computes per-channel normalized MSE using
train-derived normalization scales, then aggregates with an explicit
configured policy. It reports train and validation independently.
Zero/near-zero scales use a documented floor. No code in this service
can load test data during iteration.

`TermPruner` operates on whole AST terms. It measures normalized
trajectory contribution at interval boundaries and interiors, proposes
low-contribution removals only up to a configured maximum threshold, and in
search mode evaluates and refits only the first conservative distinct reduced
support. It accepts/rejects that support using validation impact plus validity
and simulation checks; the unpruned fit remains eligible. An explicit
all-supports mode remains available for final diagnostic analysis. Terms
containing declared external inputs are preserved because sparse
events can have important interval effects despite zero sampled-boundary RMS.
Every target-producing dependency retains nonzero dynamics. The one-step
persistence MSE is reported separately as a baseline and is never represented
as a fully pruned discovered model. Raw coefficient magnitude is never the
pruning criterion.

### Controller, persistence, and final evaluation

```python
class CandidateStage(str, Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    FITTED = "fitted"
    PRUNED = "pruned"
    JUDGED = "judged"
    RANKED = "ranked"

class BeamController:
    def run(self, experiment: ExperimentSpec) -> FrozenSelection: ...

class CheckpointStore(Protocol):
    def save_stage(self, record: CandidateRecord) -> None: ...
    def restore(self, experiment_id: str) -> ResumeState: ...
```

Candidate IDs and stage IDs are hashes of canonical inputs. Stage
records are immutable and atomically written. A checkpoint records RNG
states/seeds, configuration, code/config/data/prompt fingerprints,
parent IDs, cache keys, fit results, diagnostics, and controller beam.
Resume verifies fingerprints and skips only complete matching stages.

Beam ranking uses either the historical validation-first tuple or the configured
normalized weighted scalar
`z(log(validation_nmse)) + lambda_judge * z(-log(judge_score + epsilon))`.
Normalization uses only the current development candidate pool. The frozen
selection records both normalized components and the resulting objective.
Invalid candidates cannot enter the beam. Diversity/deduplication is
based on canonical candidate structure, not prose.
The bounded feedback payload retains the most recent rejected candidate when the
beam is full and explicitly states that alpha-renaming does not constitute a new
structure.

After the controller emits `FrozenSelection`, the selection artifact is
hashed and immutable. `FinalEvaluator` then loads test exactly once,
fits no new structure or parameters using test data, simulates the
frozen model, emits test metrics, and never returns them to the
controller or proposer.

## Data flow

```text
AUTOFORMALISM_DATA_ROOT
  -> discovery report
  -> normalized TierSpec + public prompts
  -> validated train/validation DatasetSplit objects
  -> proposer request/cache/log
  -> compact ProposerCandidate schema
  -> deterministic enrichment to CandidateModel
  -> deterministic semantic + expression validation
  -> CompiledModel
  -> bounded multistart fit on train
  -> train/validation simulation and normalized metrics
  -> whole-term pruning + refit
  -> judge request/cache/log
  -> CandidateRecord checkpoint
  -> beam selection and next proposal context
  -> frozen validation-selected structure
  -> capability-gated one-time test evaluation
```

Every arrow produces a typed result with either a value or structured
failure diagnostics. Numerical or LLM failures do not crash the whole
experiment; they close the affected candidate and are available to the
controller as bounded summaries.

## Experiment configuration and outputs

An `ExperimentSpec` declares benchmark/tier, seed, provider/model,
budgets, beam width, iteration count, optimizer/solver tolerances,
normalization and ranking policies, and output/cache locations. It may
reference no private or hidden path. Secrets are read from the
environment only by the provider adapter.

The CLI is the authoritative entry point and must support at least
`inventory`, `validate-data`, `run`, `resume`, and
`evaluate-frozen`. Notebooks may call public APIs but are not required
for execution.

Outputs are small versioned JSON/JSONL artifacts:

- resolved redacted experiment configuration;
- data/prompt/code fingerprints;
- append-only event log;
- LLM cache references;
- immutable candidate stage records;
- controller checkpoints;
- frozen selection;
- final evaluation.

Raw API payloads, datasets, secrets, and large generated trajectories
must not be committed.

## Failure and security behavior

- Missing `AUTOFORMALISM_DATA_ROOT`, contradictory metadata, unsafe
  paths, split overlap, row misalignment, nonmonotone time, or prompt
  channel mismatches fail before an LLM call.
- Paths are resolved beneath the configured public benchmark root;
  `private` and `hidden*` paths are rejected.
- Parser resource limits bound expression length, AST depth, symbol
  count, state count, and numeric literal magnitude.
- Simulation and optimization have time/evaluation limits and preserve
  failure diagnostics.
- Logs redact credentials and avoid raw datasets or unrestricted LLM
  content where a content hash and bounded summary suffice.
- Test access is separated at the API/type boundary and audited in the
  event log.

## Initial end-to-end slice

The first runnable slice should use one explicitly reconciled benchmark
and tier, one seed, one proposer call, one valid candidate, bounded
fitting, validation scoring, one judge call, a beam of one, checkpoint
and deterministic resume, followed by frozen one-time test evaluation.
The current obfuscated-perturbed cases 02–04 are not eligible until
their public metadata contradictions are resolved without changing
finalized prompts unless explicitly authorized.

## Baseline runtime boundary

Baseline commands run inside a supervised process group with a configurable
hard wall-clock limit. The supervisor records `run_status.json` and terminates
the worker and descendants on timeout, so Julia-backed PySR processes cannot be
orphaned. D3 checkpoints remain resumable after termination.

LLM-Feature-SINDy applies the degree-two polynomial/tanh library to supplied
variables and appends proposed algebraic features as linear candidate terms.
It does not recursively generate products or tanh transforms of LLM features.

D3-native-no-tools models every observed dynamic channel with supplied
derivative labels. It uses upstream D3's PyTorch Adam defaults and
teacher-forced Euler objective. Validation selects frozen parameters and test is
opened exactly once; no Autoformalism refit is applied. Expressions use the
restricted AST-to-PyTorch compiler rather than executing generated Python.
