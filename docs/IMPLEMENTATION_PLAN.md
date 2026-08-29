# Phase 1 Implementation Plan

## Delivery rules

Milestones are intentionally small and ordered. Each milestone ends
with its own tests; implementation does not proceed while those tests
fail. After every meaningful change run the focused tests, and before
completing Phase 1 run the full `pytest`, `ruff check .`, and the
documented smoke tests.

Benchmark data and finalized prompts remain read-only. Every fixture
needed by tests is synthetic and minimal; tests do not rewrite or copy
private benchmark material. Public APIs use typed functions with
docstrings.

## M0 — Package and quality skeleton

Deliver a Python 3.11+ package, CLI shell, Pydantic v2 configuration,
pytest/ruff configuration, and deterministic temporary output layout.

Acceptance tests:

- importing the package has no filesystem or network side effects;
- `autoformalism --help` succeeds independently of notebooks;
- invalid config fields and unsafe output paths fail clearly;
- `pytest` and `ruff check .` pass on the skeleton.

## M1 — Benchmark registry and inventory validation

Implement `AUTOFORMALISM_DATA_ROOT` resolution, public-root path
containment, normalized `TierSpec`, manifest readers, discovery
reports, and explicit adapters for the four layouts in
`DATA_INVENTORY.md`. Do not load numerical arrays yet.

Acceptance tests:

- the registry exposes original B1, perturbed B1,
  obfuscated-original case 01, obfuscated-perturbed case 01,
  benchmark 5, and benchmark 6, with three tiers each;
- a configured root takes precedence and a missing variable/root
  produces an actionable error;
- paths containing `private` or `hidden` are rejected;
- supported manifest roles are reconciled through explicit registry
  specifications and disagreements fail closed;
- unsupported B2–B4 and cases 02–04 are not inferred or loaded;
- discovery performs no LLM call and modifies no source file.

## M2 — Split loader and leakage guards

Load X/Y/input or metadata tables into typed trajectories. Implement
verified positional joins where keys are absent from X/Y, trajectory
segmentation, monotone time checks, finite-value checks, sampling
checks, column-role projection, fingerprints, and split-disjointness
checks.

Acceptance tests:

- one single-trajectory and one multi-trajectory synthetic fixture
  load with correct shapes, IDs, targets, auxiliaries, and inputs;
- shuffled/missing/extra input rows, duplicate/nonmonotone times,
  inconsistent sampling, NaN/Inf, and column mismatches fail;
- target channels cannot be requested as contemporaneous or future forcing;
  every benchmark may use one-slot-lagged history;
- validation/test targets are never included in a proposer data view;
- identical content across splits triggers the configured overlap
  guard;
- clean/noisy variants require an explicit selection policy.

## M3 — Candidate response schemas

Define versioned Pydantic schemas for states, processes, equations,
observations, parameters, bounds, initialization scope, channel usage,
proposal context, judge result, and structured diagnostics.

Acceptance tests:

- a minimal valid ODE candidate and judge result round-trip through
  canonical JSON;
- duplicate names, invalid bounds, unknown scope, missing observation
  mapping, extra fields, NaN/Inf, and out-of-range judge scores fail;
- only declared latent initial conditions accept trajectory-specific
  scope;
- canonical serialization is stable across runs.

## M4 — Restricted expression parser and semantic validator

Implement the tokenizer, parser, internal AST, whitelist, resource
limits, symbol table, dependency graph, closure checks, channel-usage
checks, and safe-domain checks. No runtime compiler is included until
these checks are complete.

Acceptance tests:

- supported arithmetic and whitelisted nonlinear/piecewise expressions
  parse to expected ASTs;
- undefined symbols, algebraic cycles, missing state equations,
  unsupported functions/operators, excessive depth/length, invalid
  numeric literals, and dangerous Python syntax fail;
- attempts using `eval`, attribute access, imports, comprehensions, or
  arbitrary calls fail as data;
- unavailable auxiliaries and target-as-exogenous leakage fail;
- successful and failing cases have stable diagnostic codes.

## M5 — Safe model compiler

Compile only validated AST nodes into deterministic numeric callables
for ODE right-hand sides and observation mappings. Implement parameter
layout and global/per-trajectory initial-condition layout.

Acceptance tests:

- compiled results match hand-computed values for every supported AST
  node;
- parameter ordering is canonical;
- undeclared runtime symbols and shape mismatches fail;
- source inspection/tests establish that no `eval`, `exec`, or
  unrestricted lambdification path exists;
- the same global parameter vector is used for multiple trajectories.

## M6 — Forcing interpolation and simulation

Implement per-trajectory piecewise-linear forcing, a fixed-step RK4 screening
simulator, and a `solve_ivp` final-evaluation simulator with explicit
tolerances, evaluation grids, constraints, limits, and diagnostics.

Acceptance tests:

- interpolation matches endpoints/interior points and never crosses a
  trajectory boundary;
- analytic constant, decay, and driven ODE fixtures meet numerical
  tolerance;
- irregular evaluation grids work while input support is enforced;
- unavailable forcing, horizon extrapolation, solver failure,
  nonfinite states, blow-up, and constraint violation return structured
  failures;
- repeated runs with identical inputs are identical.

## M7 — Metrics and normalization

Implement channel-wise normalized MSE, train-derived scales, explicit
aggregation, scale floors, and split-labeled reports.

Acceptance tests:

- metrics match hand calculations for multi-channel/multi-trajectory
  fixtures;
- validation uses training scales only;
- constant and near-zero channels use the documented floor;
- missing predictions and nonfinite values fail;
- metric APIs cannot accept auxiliary values as target predictions and
  cannot load a split themselves.

## M8 — Bounded multistart fitting

Implement seeded parameter initialization, bounded SciPy
`least_squares`, global parameters, allowed per-trajectory latent
initials, failed-start diagnostics, and deterministic best-start
selection. Enforce a monotonic wall-clock deadline that closes only the affected
candidate. Warm-start the frozen train-plus-validation refit from the selected
search parameters and give it a separately configured numerical budget.

Acceptance tests:

- synthetic one- and two-parameter ODEs recover known parameters
  within tolerance;
- two trajectories share global parameters while latent initials may
  differ only when declared;
- all bounds are respected, including parameters inside nonlinear
  expressions;
- the same seed/config/data produce the same starts and result;
- failed starts are retained and all-failed fitting closes the
  candidate cleanly;
- test data cannot be passed into fitting.

## M9 — LLM provider, caching, and event logging

Add an abstract provider and a fake provider first, then the OpenAI
Responses API adapter. Implement structured responses,
content-addressed atomic cache entries, redacted JSONL logs, retries
with bounded policy, and request budgets.

Acceptance tests:

- fake provider responses validate against proposal/judge schemas;
- an identical request is served from cache without a second provider
  call; a material request change misses;
- malformed output, timeout, and provider failure are logged and
  returned structurally;
- logs/cache keys contain no API secret;
- every provider attempt and cache hit has a correlation ID;
- proposer payload snapshots contain no test data/metrics or private
  paths.

## M10 — Candidate stage runner and checkpoints

Compose proposal, validation, compilation, fitting, simulation,
metrics, and immutable per-stage records. Add atomic checkpoints and
fingerprint-verified deterministic resume.

Acceptance tests:

- a fake-provider candidate advances through proposed, validated, and
  fitted stages;
- an invalid proposal stops before fitting and preserves diagnostics;
- interruption after each stage resumes at the next incomplete stage
  without repeating an LLM call or completed fit;
- changed data, prompt, config, or code-schema fingerprint refuses
  incompatible resume;
- rerunning a completed stage does not produce divergent records.

## M11 — Structured judge

Build judge context from the static judge prompt and candidate
structure, validate category scores in `[0,1]`, and attach feedback to
the candidate record independently of numerical fit.

Acceptance tests:

- valid fake judge output is stored and malformed/out-of-range output
  closes judging safely;
- judge requests include no test information;
- numerical fit values are not represented as judge category scores
  or fed to the judge unless the finalized judge contract explicitly
  requires a bounded non-test diagnostic;
- cache/resume behavior matches proposer behavior.

## M12 — Whole-term pruning

Implement AST term enumeration, normalized contribution diagnostics,
conservative support selection, semantic revalidation, one search-time refit,
validation-impact measurement, and deterministic accept/reject policy. Retain
an explicit exhaustive-support mode for follow-up analysis.

Acceptance tests:

- a negligible synthetic term is removed while an essential term is
  retained;
- pruning removes whole terms, not arbitrary coefficients;
- coefficient magnitude alone cannot trigger removal;
- invalid or numerically unstable removals are rejected;
- pruning is deterministic and checkpointed.

## M13 — Beam controller

Implement canonical structural deduplication, explicit ranking,
beam-width enforcement, bounded history summaries, proposal budgets,
and next-iteration refinement.

Acceptance tests:

- hand-built candidate records rank in the documented order;
- invalid candidates never enter the beam;
- structural duplicates collapse deterministically;
- a recent structural duplicate remains visible in bounded feedback even when
  the active beam is full, with alpha-renaming identified as non-novel;
- beam width and LLM/fit budgets are enforced;
- proposal history contains train/validation summaries and judge
  feedback but no test information;
- controller checkpoint/resume reproduces the same beam.

## M14 — Frozen selection and one-time test gate

Create immutable `FrozenSelection`, separate test-access capability,
audited final evaluation, and protection against feedback into the
controller.

Acceptance tests:

- validation selects and hashes a frozen structure before test loading;
- test cannot be loaded through proposer, fitter, judge, pruner, or
  controller APIs;
- final evaluation runs once for a frozen selection and is idempotent
  on resume;
- test metrics never appear in later proposal or selection events;
- attempts to refit structure or parameters on test fail.

## M15 — CLI integration and first end-to-end smoke test

Expose `inventory`, `validate-data`, `run`, `resume`, and
`evaluate-frozen`. Run one explicitly reconciled benchmark/tier, one
seed, fake proposer/judge, beam width one, and a minimal budget from
data loading through frozen test evaluation. Then enable a separately
marked live-provider smoke test.

Acceptance tests:

- the offline smoke test completes from a clean output directory;
- interruption and resume produce byte-equivalent canonical selection
  and metrics artifacts;
- the CLI works without a notebook;
- live-provider testing is skipped without credentials and never logs
  them;
- artifact inspection finds no raw dataset, private/hidden path,
  secret, or unbounded API response.

## M16 — Validated judge integration and matched no-judge ablation

Freeze the selected paired-question-consensus operating point inside the
incumbent-relative search controller, then run a matched development experiment
against validation-only search with all judge calls disabled. Hold benchmark
cells, repetitions, proposer model, sampling settings, fitting budget, beam,
initial request cache, and post-freeze evaluator fixed. Only the scientific
selection and feedback path may differ.

Acceptance tests:

- the frozen plan expands to the exact arm-by-cell-by-repetition cross-product;
- the judge arm uses `incumbent_relative_hybrid`, both candidate orientations,
  question consensus, the fixed indeterminate denominator, and the frozen
  science weight;
- the no-judge arm uses `validation_only` and makes no judge call;
- both arms use a one-member beam, identical numerical and proposal budgets,
  deterministic checkpoints, and development-only search without test access;
- the judge arm populates the content-addressed cache before the no-judge arm,
  so identical initial proposer requests are reused exactly while later
  feedback-dependent requests remain independently hashed;
- terminal method failures remain planned outcomes rather than being omitted;
- selections are content-frozen before common target, mechanism-compliance,
  hidden-subspace, intervention, complexity, and reliability evaluation;
- no weighted overall evaluation score is introduced;
- deterministic resume does not duplicate completed proposer or judge calls.

## Baseline runtime controls

Run each baseline in a supervised process group with a configurable hard
wall-clock deadline. Record completion, failure, or timeout in a small status
artifact. Pass the deadline through to PySR and terminate its Julia descendants
on expiry. Run D3 with native Adam, teacher-forced Euler updates, all eligible
observed states, validation early stopping, and frozen test parameters.

Acceptance tests:

- a forced timeout exits with a distinct status and writes `timed_out`;
- PySR receives its native timeout configuration;
- D3 result metadata identifies Adam/Euler settings and modeled channels;
- timeout cleanup does not delete D3 checkpoints or affect other runs.

## Phase 1 completion gate

Phase 1 is complete only when:

1. `pytest` passes;
2. `ruff check .` passes;
3. registry/data-validation smoke tests pass on all noncontradictory
   public configurations;
4. the offline end-to-end and deterministic-resume smoke tests pass;
5. the live-provider smoke test passes in an authorized environment;
6. changed files and schema versions are summarized;
7. remaining limitations are recorded, including unsupported
   expression grammar, numerical identifiability limits, unresolved
   benchmark metadata contradictions, and the fact that the initial
   slice is one benchmark and one seed rather than a suite-scale
   scientific result.
