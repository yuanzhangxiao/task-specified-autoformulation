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
