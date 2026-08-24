# Autoformalism Phase 1 Implementation Context

## Scientific objective

Implement task-specified dynamical-system discovery under partial
observability.

A candidate model must satisfy both:

1. trajectory fit on observed target channels;
2. task requirements stated in a natural-language proposer prompt.

The state representation, latent states, equations, and continuous
parameters are jointly discovered.

## Benchmark suite

The repository contains:

1. Original Dalla Man benchmarks B01–B04 (use B01 only for now).
2. Perturbed Dalla Man benchmarks B01–B04 (use B01 only for now).
3. Obfuscated original Dalla Man cases 01–04 (use case 01 only for now).
4. Obfuscated perturbed Dalla Man cases 01–04 (use case 01 only for now).
5. Benchmark 5: an obfuscated three-state nonlinear process.
6. Benchmark 6: a procedurally generated alien-device system.

The historical Phase 1 benchmarks may have easy, medium, and hard observability
tiers. The redesigned Phase-B suite instead has two mechanism/data difficulty
tiers for every family. Semantic controls are separate: named/obfuscated for
Dalla Man and CSTR, and functional/opaque for the alien device. The 40 Phase-B
cells are registered through a distinct tidy-split layout. Their materialized
public tables are external release artifacts and are produced only by the
audited release command.

## Prompt conventions

Each benchmark prompt defines:

- target channels;
- supplied auxiliary channels;
- external inputs;
- task-required mechanisms;
- data availability over the prediction horizon.

Target channels are observed for fitting and evaluation. Every Phase 1 task
uses one-step-ahead prediction: target values through sample i-1 may be used
causally when predicting sample i, while the current and future target values
remain unavailable.

Supplied auxiliary channels may be used as exogenous trajectories over
the prediction horizon.

No unlisted observed trajectory may be assumed supplied.

Latent states may be introduced.

The benchmark-specific task specification is authoritative. Do not
rewrite or weaken benchmark prompts unless explicitly instructed.

## Phase 1 pipeline

The pipeline is iterative:

1. A proposer LLM reads:
   - the static benchmark proposer prompt;
   - the high-level proposer-controller instructions;
   - summaries of selected previous candidates;
   - fitted training and validation errors;
   - judge scores and feedback;
   - numerical and pruning diagnostics.

2. The proposer outputs a compact machine-readable candidate model:
   - dynamic states, each containing its governing RHS;
   - algebraic processes;
   - component-level target mappings, constraints, and mechanism tags;
   - free parameters;
   - parameter bounds and optional initialization ranges;
   - global versus trajectory-specific parameter scope.

3. The runtime deterministically enriches the proposal into the complete
   internal candidate schema, then validation checks:
   - JSON/schema validity;
   - undefined symbols;
   - equation closure;
   - use of unavailable channels;
   - target leakage;
   - invalid parameter declarations;
   - unsafe or unsupported expressions. Potential zero denominators are
     guarded and recorded as warnings; unsafe log and square-root domains
     remain errors.

4. A numerical module fits all continuous parameters:
   - parameters outside nonlinear functions;
   - parameters inside exp, tanh, rational, and other expressions;
   - global parameters used by analytic latent-initialization expressions.

5. During iterative screening, the model is simulated with deterministic
   fixed-step RK4 updates at the observed sampling intervals. The frozen final
   structure is refitted and evaluated with adaptive `solve_ivp`.

6. Normalized training and validation MSEs are computed.

7. Terms are pruned using normalized contribution and validation impact. Search
   rounds refit at most one conservative reduced support; exhaustive support
   sweeps are reserved for explicit follow-up analysis.

8. A judge LLM evaluates scientific semantics and returns structured category
   scores in [0,1] plus advisory red flags and edits. The runtime supplies its
   deterministic validity checks as certified facts, so the judge does not
   rescore syntax, closure, symbol availability, mappings, causal channel
   access, parameter bounds, or expression executability. Only deterministic
   or numerical checks may block a candidate.

9. A beam-search controller chooses candidates for refinement.

10. The final structure is selected using validation data. Test data
    are evaluated only once after final selection. The train-plus-validation
    final refit is warm-started from the selected search fit.

## Important implementation decisions

- Do not require all parameters to be linear.
- Do not require closed-form least-squares fitting.
- The graph representation is an expression/dependency graph, not a
  restriction to linear-in-parameter models.
- The LLM proposes parameter identities, locations, bounds, and roles.
- Numerical optimization determines final parameter values.
- Never ask the LLM to numerically tune continuous parameters from
  prose feedback.
- Never use Python eval on LLM-generated expressions.
- Use a restricted AST parser and explicit function whitelist.
- The prospective v2 judge evaluates scientific coherence, balance semantics,
  dynamic plausibility, mechanism coupling, nonredundancy, and justified
  complexity—not trajectory fit or deterministic validity.
- Judge red flags are advisory and never override deterministic validation.
- The prospective hybrid judge separates public task requirements, proposer
  claims, deterministic graph facts, absolute semantic predicates, and direct
  comparative residuals. It remains calibration-only until question-level
  held-out evaluation is complete.
- Ollama hybrid-judge calibration uses JSON-schema output as its primary
  transport. Always-on tool calls and JSON-primary/tool-fallback remain explicit
  transport ablations only: completed frozen-pair tests found that their small
  completion gain did not preserve scientific accuracy or A/B order consistency.
  Tool-call results must not silently replace a missing JSON-schema judgment.
- The first frozen hybrid operating point requires both candidate orientations,
  retries only a missing orientation for at most five distinct seeds, and
  abstains if symmetric evidence remains incomplete. Its next evaluation holds
  out canonical candidate structures while retaining audited mutation contracts;
  search integration remains blocked until the predeclared held-out gates pass.
- A baseline-structure-held-out run exposed an Ollama protocol-completion
  failure: all accepted judgments completed on the first native JSON-schema
  attempt, while the OpenAI-compatible reasoning-disabled repair path recovered
  none. The separately named `json_schema_native_retry` ablation preserves
  `/api/chat`, low reasoning, and the same schema on contract-only retries. It
  has a distinct cache/manifest identity and cannot silently alter the frozen
  JSON-schema control.
- Native low-reasoning retries recovered 97 of 100 calls on the same ten
  held-out pairs and passed the frozen adaptive pair-level gates. Because this
  transport was selected after the original held-out completion failure, it is
  engineering validation rather than a new pristine paper holdout. A bounded
  OpenAI-compatible low-reasoning retry ablation isolates endpoint behavior from
  reasoning suppression before any serving-engine change.
- The low-reasoning OpenAI-compatible retry completed all calls but reintroduced
  Candidate-A order bias, so native JSON retry remains the Ollama choice. The
  next cross-runtime diagnostic uses a pinned vLLM container and a one-A40
  structured-output smoke test before implementing a new judge provider. Its
  first deployment attempt failed during Apptainer SquashFS creation on the
  shared `/work/hdd` filesystem, before vLLM started. The failure persisted on
  node-local NVMe, isolating it to SquashFS creation rather than storage space.
  The smoke job now retains its OCI cache persistently but builds and executes a
  bounded job-local sandbox, bypassing SquashFS entirely. The smoke succeeded on
  one A40 with nonempty strict JSON output, but its single low-reasoning verdict
  was scientifically incorrect. A frozen four-pair vLLM pilot therefore compares
  low and high reasoning under identical seeds, candidate orientations, schemas,
  and scoring without changing the production judge. The four-pair pilot selected
  low reasoning operationally: it completed 40/40 calls and achieved 0.900
  end-to-end preference accuracy, while high reasoning completed 36/40, achieved
  0.875 end-to-end accuracy, consumed about 6.8 times as much aggregate GPU time,
  and degraded atomic source-role accuracy despite better conditional comparative
  consistency. The frozen expansion added the six untouched held-out pairs and
  merged them with the reused low-reasoning calls. The complete 100-call set had
  perfect response success, 0.810 per-call combined accuracy, 0.900 full pair
  aggregation, and 0.740 order consistency. Its predeclared two-orientation
  adaptive operating point reached 0.880 pair accuracy, so the transport passed
  but the scientific protocol missed its accuracy and order-consistency gates.
  Offline pair/criterion attribution and structure-aware aggregation sensitivity
  are diagnostics only; any revised protocol requires new baseline structures.
  The stored-rationale audit traced the weak categories to missed signed terms,
  dismissal of exact repetition as algebraically mergeable, and comparative ties
  based only on declaration counts. A frozen matched development pilot therefore
  adds symmetric syntax-only signed-term and exact-repeat facts plus general rubric
  clarification without changing the LLM, seeds, schema, retries, or score weights.
  That matched pilot solved exact-repeat detection (`0.95`) but failed source-role
  (`0.25`), comparative (`0.45`), and order-consistency (`0.55`) gates. The next
  frozen development protocol therefore separates expected scientific direction
  from the candidate's actual outer sign. A first sign-blinded call infers atomic
  directions and exact-repeat relationships; runtime checks polarity; a second
  call receives those findings for comparative assessment. No task-specific role
  answer is disclosed. A matched 20B/120B vLLM factorial tests question
  presentation and model-scale capability separately before any new-structure
  confirmation or search integration. The 120B condition completed all calls,
  achieved `0.925` conditional/end-to-end preference accuracy, perfect pair
  aggregation, `0.850` order consistency, and `0.975` targeted atomic accuracy.
  Frozen decision decomposition showed that direct comparative judgments, not
  the saturated conjunctive absolute groups, supplied nearly all separation.
  Comparative-criterion ablation found a broad stable weight/threshold region
  and no benefit from removing any of the three general comparative questions.
  Protocol, weights, threshold, reasoning effort, seeds, and retry budget are
  therefore frozen unchanged for a new canonical-structure confirmation. The
  confirmation builder excludes every structure appearing in any opened pair
  file, records those files and hashes in a manifest, and selects only the two
  prespecified wrong-sink and duplicated-flux mutation families. Passing requires
  every predeclared response, pair, order, atomic, and comparative gate; no
  parameter may be tuned on the confirmation outcomes.
- Numerical fit is computed deterministically.
- Raw coefficient magnitude alone is not a valid pruning criterion.
- Prune whole terms using normalized trajectory contribution and
  validation-error impact.
- Global model parameters are shared across trajectories.
- Latent initial conditions are not free fitted values.
  Each must be a fixed value or a safe analytic expression of known initial
  observations, inputs, and covariates.
- Never use current or future validation/test targets as model inputs. All
  benchmarks permit strictly lagged target history for one-step-ahead
  prediction.
- Never expose test error to the iterative proposer.
- Cache every LLM request and response.
- Checkpoint after every candidate stage.
- A per-fit wall-clock deadline returns a structured candidate failure and does
  not terminate subsequent search rounds.
- Constraints attached to undeclared prose concepts are discarded with a
  deterministic repair diagnostic; constraints on declared model or supplied
  forcing symbols remain enforceable.
- Recent structural-duplicate failures remain in bounded proposer feedback even
  when the active beam is full. Renaming symbols alone is explicitly non-novel.

## Phase 1 technology choices

Use:

- Python 3.11 or newer;
- Pydantic v2;
- NumPy;
- SciPy;
- pandas;
- PyYAML;
- SymPy only for symbolic validation/simplification where useful;
- OpenAI Responses API through an abstract LLM provider interface;
- pytest;
- ruff;
- JSONL event logs.

## Initial implementation scope

First support:

- explicit finite-dimensional ODE systems;
- algebraic generated processes;
- global bounded parameters;
- fixed or analytically initialized latent states;
- supplied time-varying auxiliary channels and inputs;
- piecewise-linear interpolation of supplied inputs;
- SciPy solve_ivp;
- bounded multistart least-squares;
- structured proposer and judge responses;
- checkpoint/resume;
- one benchmark and one seed end to end.

Do not begin with:

- bootstrap pruning;
- automatic dimensional analysis;
- distributed execution;
- GPU optimization;
- Bayesian parameter inference;
- arbitrary Python generated by the proposer.

Baseline experiments have a separate wall-clock supervisor. Completion,
failure, and timeout are explicit result states; a timed-out method must not
stop other scheduled runs. D3-native-no-tools uses the native Adam and
teacher-forced Euler fitting protocol over all eligible observed states, with
external tools disabled and safe restricted expressions in place of arbitrary
generated Python. Its numerical fitting configuration must be reported.
