# Phase 1 Pipeline Design

## Purpose and invariants

Phase 1 discovers a task-sufficient continuous-time model from
trajectory data and natural-language requirements under partial
observability. The system jointly searches structure, latent states,
observation mappings, and bounded continuous parameters.

The following are architectural invariants:

1. Train, validation, and test data are distinct typed objects.
2. Proposal and iterative selection can see train results and
   validation results, but never test values or metrics.
3. Target trajectories are never prediction-horizon inputs.
4. Only prompt/spec-declared auxiliaries and external inputs may be
   interpolated over the horizon.
5. LLM output is untrusted and is parsed through schemas and a
   restricted expression grammar; no `eval`, `exec`, arbitrary Python,
   or unrestricted lambdification is used.
6. Global parameters are shared across trajectories. Only explicitly
   declared latent initial conditions may be trajectory-specific.
7. Every LLM call is content-addressed, cached, and logged.
8. Every stage is checkpointed with enough state for deterministic
   resume.
9. Final structure selection uses validation data. The test split is
   opened once, only after selection is frozen.
10. Private/hidden benchmark material is never a runtime input.

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
unavailable channels. Loading `TEST` requires a separate
`FrozenSelection` capability, so ordinary controller code cannot
accidentally obtain test data.

### Candidate schema and expressions

```python
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
comparisons/piecewise forms explicitly supported by the grammar, and a
small whitelist such as `exp`, `tanh`, `abs`, `min`, and `max`.
Validation rejects unknown syntax, undefined symbols, cycles among
algebraic processes, missing state equations, invalid bounds,
unavailable channels, target leakage, and unsafe domains. Compilation
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

class JudgeResult(BaseModel):
    category_scores: dict[str, confloat(ge=0, le=1)]
    overall_score: confloat(ge=0, le=1)
    feedback: list[JudgeFeedback]
```

The provider wrapper hashes model name, provider settings, schema,
prompt, and request payload. It atomically reads/writes the response
cache and appends JSONL request/result events with secrets excluded.
Invalid structured responses are recorded and returned as failures,
not passed downstream. The proposer sees compact summaries rather than
raw full history. The judge sees the benchmark judge prompt and
candidate structure, but no numerical fit metric; deterministic code
owns fit scoring.

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
boundaries. `Simulator` uses `solve_ivp`, checks solver success,
finite values, time coverage, and configured state constraints.

`Fitter` constructs one bounded vector containing global parameters
and explicitly allowed per-trajectory latent initial conditions. It
runs deterministic seeded multistart `least_squares`; failed starts
remain diagnostics rather than disappearing. Training residuals drive
optimization. Validation is evaluated after each fitted start for
selection among starts and later beam ranking, never used as a forcing.

`MetricEvaluator` computes per-channel normalized MSE using
train-derived normalization scales, then aggregates with an explicit
configured policy. It reports train and validation independently.
Zero/near-zero scales use a documented floor. No code in this service
can load test data during iteration.

`TermPruner` operates on whole AST terms. It measures normalized
trajectory contribution, proposes one deterministic removal at a time,
refits, and accepts/rejects using validation impact plus validity and
simulation checks. Raw coefficient magnitude is never the pruning
criterion.

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

Beam ranking uses an explicit tuple or configured scalar over
validation fit, judge compliance, parsimony, and numerical reliability.
Invalid candidates cannot enter the beam. Diversity/deduplication is
based on canonical candidate structure, not prose.

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
  -> CandidateModel schema
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
