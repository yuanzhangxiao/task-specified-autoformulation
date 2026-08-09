# Post-rebuttal experimental roadmap

## Objective

The objective is to produce the strongest possible resubmission while keeping
hosted-LLM costs controlled. The roadmap separates uncertainty from data and
numerical fitting, model selection, and stochastic LLM proposal generation.
Only the last category necessarily requires new proposer calls.

The scientific priorities are:

1. demonstrate held-out and intervention generalization rather than interpolation;
2. quantify statistical uncertainty and failure rates honestly;
3. validate mechanistic and latent-state recovery independently of trajectory fit;
4. fix the current reliability failures, especially obfuscated-perturbed tasks;
5. establish performance-versus-compute and performance-versus-API-cost tradeoffs;
6. make the pipeline reproducible with shared caches and, eventually, open models.

## Experimental invariants

- Freeze one tagged implementation before confirmatory experiments.
- Use matched train/validation/test splits and seeds across methods.
- Never expose test metrics to proposal generation, fitting choices, pruning, or
  selection.
- Record completion, failure, and timeout separately from predictive metrics.
- Report performance conditional on completion and a prespecified failure-aware
  sensitivity analysis; do not silently average sentinel values such as `1e12`.
- Record proposer, judge, fitting, and data seeds separately.
- Cache and log every LLM call using a content-addressed request hash.
- A replay experiment must fail closed on a cache miss rather than making an
  unplanned external call.
- New method comparisons use the same wall-clock, candidate, or token budget.

## Phase A: zero-new-LLM-call work

### A1. Consolidate and audit existing evidence

- Merge LLM caches from every VM and HPC archive by request hash.
- Detect conflicting responses for the same request hash.
- Index every valid and invalid proposal, round checkpoint, judge response,
  fitted parameter vector, and frozen final selection.
- Produce counts by benchmark, tier, seed, method, provider, cache status, and
  terminal run status.
- Recover token usage and estimate historical cost per valid candidate and per
  completed run.

Deliverables:

- a cache manifest;
- a frozen candidate-pool manifest;
- a completeness and conflict report;
- a provider-disabled replay smoke test.

Implemented in the first post-rebuttal milestone: the cache audit resolves
metadata-only duplicates, quarantines semantic conflicts, and the experiment and
LLM-baseline CLIs provide a fail-closed cache-only mode.

### A2. Statistical analysis of existing matched runs

- Mean and sample standard deviation.
- Median and interquartile range.
- Paired bootstrap 95% confidence intervals.
- Paired permutation tests or Wilcoxon signed-rank tests.
- Paired effect sizes.
- Holm correction across prespecified comparisons.
- Completion-rate confidence intervals.
- Hierarchical analysis with method, tier, and method-by-tier fixed effects and
  benchmark/seed random effects where supported by the available sample size.
- Sensitivity analyses with and without documented numerical outliers.

Core paired comparisons, confidence intervals, multiplicity correction,
completion-rate intervals, full/no-judge stratification, judge associations,
and frozen alternative-selection refits are complete in the Phase A2
milestone. A hierarchical mixed-effects model remains optional because the
available benchmark count is small and its assumptions require separate
justification.

### A2.5. Selection/evaluation contract and judge audit

- Separate deterministic runtime validity, public structural compliance,
  qualitative LLM assessment, and private structural recovery.
- Specify which signals are permitted during proposal, fitting, selection, and
  final evaluation.
- Audit overlap between the six historical judge categories and deterministic
  validators.
- Retain private structural validity, hidden trajectories, interventions, and
  test metrics strictly for post-freeze evaluation.

Implemented in `docs/SELECTION_EVALUATION_CONTRACT.md` and
`docs/JUDGE_RUBRIC_AUDIT.md`. The retrospective candidate pool contains only
runtime-valid candidates and cannot be used to invent a new public
structural-compliance signal after the fact. The 336-call adversarial judge
study is now analyzed in `docs/JUDGE_CALIBRATION_ANALYSIS.md` with a
leave-one-benchmark-out calibration protocol. Historical scores are repeatable
and discriminate genuine dynamics mutations, but calibrated category weights
do not improve held-out pairwise accuracy. No selector change is warranted.

Before Phase-B production selection, audit the new common judge's score
distribution separately for candidates with and without hard red flags. Report
the fraction capped at exactly `0.4`, the uncapped-score distribution, category
distributions, and ranking ties. If `0.4` is a dominant point mass, keep hard
flags as a separate validity indicator and reconsider whether a numeric cap is
useful for ranking rather than silently accepting a low-resolution selector.

### A3. Frozen-model intervention and distribution-shift evaluation

**Status: complete.** The final frozen evaluation includes Benchmarks 5 and 6,
an increasing-noise sweep, and faithful original/perturbed Dalla Man B1
simulators evaluated in semantic and obfuscated representations. Hidden-state
alignment, coverage, direction, response shape, timing, failure accounting, and
paired uncertainty analyses are implemented. See `docs/INTERVENTION_SUITE.md`
and `docs/PHASE_A3_INTERVENTION_CONCLUSION.md`.

Across the core, noise, and Dalla Man suites, 645 frozen model/case evaluations
completed without intervention-time simulation failures. Three historical
No-judge Benchmark 5 cells remain unavailable after failed final refits and are
kept as explicit cohort gaps. Results and interpretation are documented in
`docs/INTERVENTION_RESULTS.md` and
`docs/PHASE_A3_INTERVENTION_CONCLUSION.md`.

Evaluate selected models without structural refitting on:

- unseen event magnitudes and event timing;
- multiple events;
- different initial conditions;
- changed fixed covariates;
- longer horizons;
- simulator parameter shifts;
- denser and sparser observation grids;
- increasing measurement noise;
- input profiles outside the training distribution.

Report observed target MSE, hidden-mechanism NMSE where defined, qualitative
intervention correctness, failure rate, and degradation relative to in-distribution
performance.

### A4. Counterfactual mechanism tests

For each benchmark, construct private post-selection interventions with known
simulator consequences:

- increase, remove, or delay an input;
- suppress an intermediate mechanism;
- reverse a regulatory sign;
- change a time constant or saturation threshold;
- modify a causal driver.

Score direction of change, sign, temporal ordering, qualitative dose response,
and approximate response magnitude. These private references remain evaluation
inputs only and are never exposed during search.

### A5. Numerical-fitting robustness on cached structures

Refit frozen candidates under:

- one, three, and five starts;
- current versus scale-aware parameter bounds;
- linear versus log parameterization of positive parameters;
- current versus larger function-evaluation budgets;
- RK4 versus adaptive integration;
- warm versus cold final initialization;
- several solver tolerances.

Measure fit success, validation/test MSE, runtime, parameter stability, and how
often the selected structure changes. This separates structural-search failures
from continuous-optimization failures.

### A6. Frozen-pool selection studies

Using existing fit and judge records, compare:

- validation MSE only;
- judge score only;
- ratio objective;
- weighted-sum objective;
- Pareto selection;
- complexity-penalized selection;
- structure-stability-aware selection.

Also cluster alpha-normalized structures across seeds and test consensus
selection followed by a common final refit.

The first A6 milestone is implemented with robustly normalized weighted-sum,
Pareto-compromise, and epsilon-constrained selectors. It produces a complete
development-only sensitivity surface and a leave-one-benchmark-out robustness
analysis without reading test metrics or private mechanism references.
The prespecified post-freeze confirmation is also complete for validation-only,
moderate normalized weighted, and 20% epsilon-constrained selection. Neither
alternative improved the vector of test error, structural validity, hidden
error, complexity, and refit reliability. Production validation-only selection
therefore remains unchanged. Structure-stability-aware consensus selection
remains future A6 work. Judge calibration is complete; failure recovery is now
the next priority.

### A7. Failure recovery and reliability audit

Classify every failed or extreme-error run as:

- no valid proposal;
- malformed or truncated structured output;
- deterministic validation failure;
- numerical fitting failure;
- final refit failure;
- structurally incomplete model;
- correct structure with poor parameters;
- memorized original mechanism on a perturbed task.

Replay cached proposals after targeted numerical and deterministic fixes. The
obfuscated-perturbed benchmark is the highest-priority reliability case.

The first A7 integrity milestone is complete in
`docs/FAILURE_RECOVERY_AUDIT.md`. It reconciles all 486 core cells, identifies
19 unresolved algorithm/run cells, reclassifies 15 LLM-feature-SINDy numerical
sentinels that were incorrectly marked complete, and confirms that all 19
historical D3 failures were superseded. The baseline result boundary now rejects
failed final rollouts. The zero-LLM-call numerical refit of the 31 cached
structures for the missing full-method Benchmark5 hard seed is also complete:
scale-aware initialization recovered a frozen model
with validation NMSE 0.003468, test NMSE 0.007494, structural validity 1.0,
hidden NMSE 0.6504, and eight additive terms. Its provenance remains explicitly
marked as numerical recovery because the historical candidate failed before a
judge response was produced.

### A8. Human and deterministic structural validation

- Obtain blinded expert ratings for a stratified sample of full, ablated,
  baseline, and adversarial candidates.
- Measure inter-rater agreement and agreement with deterministic structural
  validity and LLM judge scores.
- Complete edge, sign, dynamic-memory, hidden-trajectory, and surviving-term
  evaluations against private simulator references.

## Phase B0: benchmark validity, predictability, and identifiability audit

**Status: core audit complete; nonlinear profile likelihood remains an optional
deep follow-up.** Shortcut, excitation, downsampling, response-phase,
initial-state observability, parameter/flux sensitivity, and frozen free-rollout
diagnostics are implemented. This
phase precedes additional method optimization. Its purpose is to determine what
the current observations can support scientifically, and whether the evaluation
distinguishes dynamical discovery from smooth one-step interpolation.

### B0.1. Shortcut and forecast-horizon audit

- Fit constant, persistence, scalar autoregressive (AR), and input-aware ARX
  baselines using training data only.
- Evaluate validation and test predictions at prespecified horizons of 1, 5,
  10, and 30 samples, recursively propagating AR/ARX predictions within each
  horizon.
- Report whole-trajectory and event-window raw MSE/NMSE separately.
- Report the variance of adjacent target increments relative to target variance,
  external-input activity/change rates, distinct input levels, and the fraction
  of samples near externally driven events.
- Repeat after prespecified temporal downsampling and add uninterrupted free
  rollout where a baseline has a meaningful autonomous state realization.

The model-independent first pass is implemented in
`scripts/audit_benchmark_predictability.py`. It makes no LLM calls and never
fits on validation or test data. Physical-time-matched downsampling confirms
that the forecast-horizon conclusion is not an artifact of merely counting
samples. Reference-defined rise/recovery/equilibrium strata isolate where
shortcuts fail within each held-out response.

### B0.2. Full glucose-insulin task audit

Audit all four original Dalla Man tasks (T1 meal appearance, T2 absorption
action, T3 hepatic regulation, and T4 flux portrait), not only T1. For every
task and information tier, record:

- public targets, auxiliaries, inputs, covariates, and semantic information;
- number and diversity of trajectories, initial conditions, interventions, and
  event regimes;
- shortcut predictability and forecast-horizon degradation;
- private-state/output sensitivities and empirical observability evidence;
- which task-required mechanisms are uniquely recoverable, recoverable only up
  to an equivalence class, or unsupported by the supplied observations.

T4 and any other underdetermined tier will be retained as a stress test or
negative control rather than being described as an ordinary recoverable task.
Adding these tasks to the production registry and prompts is a separate
milestone; the finalized data and prompts themselves remain unchanged.

### B0.3. Factorial interpretation of semantic and perturbation conditions

Treat the four Dalla Man B1 variants as a 2-by-2 diagnostic design:

| | Canonical dynamics | Perturbed dynamics |
|---|---|---|
| Semantic names/context | Original | Perturbed |
| Obfuscated names/context | Obfuscated-original | Obfuscated-perturbed |

- Original versus perturbed tests adaptation rather than canonical-model
  memorization when semantic context is available.
- Semantic versus obfuscated tests dependence on semantic priors.
- Obfuscated-perturbed tests whether data alone reveal the perturbation after
  both semantic guidance and the canonical equation prior are unavailable.

Obfuscated-perturbed is therefore not automatically a separate headline
benchmark. Its incremental value must be demonstrated by the interaction in
this factorial comparison. If it adds no information beyond obfuscated-original,
report it as a combined negative control or move it to the appendix rather than
claiming redundant evidence.

### B0.4. Observability and identifiability audit

- Use private simulators only after task definition, never as runtime inputs.
- Compute local output-sensitivity matrices over multiple initial conditions and
  input protocols, their singular spectra/effective ranks, and empirical
  observability-Gramian diagnostics where appropriate.
- Use profile likelihood or multi-start parameter recovery to distinguish
  structural from practical non-identifiability.
- Test recovery of individual hidden coordinates only when justified; otherwise
  score the identifiable hidden subspace or input-output equivalence class.
- Select additional training interventions by their ability to separate
  competing mechanism classes, not merely by increasing sample count.

The first local empirical-observability pass is implemented in
`scripts/audit_dalla_observability.py`. It measures scaled output sensitivities
to task-relevant private initial states under the historical single-meal and a
multi-meal timing/magnitude protocol. This is a practical local diagnostic, not
a proof of global structural identifiability. Parameter and derived-flux
identifiability remain outstanding.

The extended audit now includes a second initial condition and local parameter
sensitivities for both public outputs and private task fluxes. T3/T4 are
practically ill-conditioned despite numerical full rank at permissive
thresholds. See `docs/PHASE_B0_BENCHMARK_AUDIT.md`.

### B0.5. Decision rules before new expensive runs

- Retain one-step MSE only as a diagnostic.
- Make unseen-condition free rollout and input/initial-condition interventions
  the primary predictive evaluation.
- Use task-structural compliance as a transparent benchmark-specific metric.
- Report hidden alignment only for states or subspaces supported by the
  identifiability audit.
- Do not optimize the judge or weighted objective until the audit establishes
  that the selection metrics reward scientifically meaningful behavior.
- If persistence/ARX remains competitive only at short horizons, revise the
  evaluation protocol. If it remains competitive in long free rollout, enrich
  the trajectories with diverse initial conditions and persistently exciting
  inputs.

The minimum-data and evaluation redesign is frozen conceptually in
`docs/PHASE_B_BENCHMARK_REDESIGN.md`. Historical Phase A3 evaluations are now
correctly classified as reset-based distribution-shift tests. A new explicit
free-rollout mode shows one-to-three-order-of-magnitude degradation and confirms
that rollout-aware fitting/validation must precede objective reweighting.

The Phase-B v1 suite has two mechanism/observability difficulty tiers for every
family, with input schedules and trajectory counts held fixed across tiers.
Dalla Man and CSTR use named/obfuscated semantic controls; alien device uses a
functional/opaque semantic control, which measures task-information value
rather than model retrieval. Dalla Man canonical versus perturbed is a third,
independent dynamics factor. The versioned contract is
`configs/benchmarks/phase_b_suite_v1.json`. Do not launch new production runs
until generated data and prompts conform to it.
The exact pre-generation design is documented in
`docs/PHASE_B_EXACT_BENCHMARK_PROTOCOL.md`.

**Implementation status (2026-08-07):** the private Phase-B generation layer
now executes all frozen 16/4/6 protocols for Dalla Man, CSTR, and alien device
without modifying the historical registry. All 20 distinct numerical cases
pass the complete simulator-only pre-release gates after the protocol-approved
minimal-information/equivalence-subspace remedies documented in
`docs/PHASE_B_PRE_RELEASE_AUDIT.md`. Test sealing remains opt-in. Public channel
channel mappings, semantic-control prompts, staging manifests, and automated
leakage validation are now implemented. Every staging package is explicitly
non-registered, excludes test by default, and commits to numeric content
independently of channel names. The complete 40-cell staging audit passes with
40/40 leakage-clean packages, matching semantic-pair numeric commitments, and
zero sealed test cells. Manual review of all proposer prompts and the shared
judge prompt is complete, and the reviewed files are byte-identical to fresh
generator output. Production-registry integration, strict sealed-test access,
and the explicit 40-cell release command are now implemented. A temporary
release rehearsal passed all 40 cells and opened no test metric. Materialized
release tables remain external artifacts and are created only by the deliberate
release command documented in `docs/PHASE_B_RELEASE_INTEGRATION.md`.

## Phase B: low-call targeted experiments

### B1. Sequential additional seeds

**Decision gate:** defer paid additional seeds until the redesigned trajectories
and rollout-aware validation protocol are frozen. More seeds under the
historical one-step objective would narrow uncertainty around the wrong primary
endpoint.

Do not immediately double every cell. Add seeds one at a time to:

1. Benchmark5 hard, to repair the missing confirmatory cell;
2. obfuscated-perturbed hard, to estimate the current instability;
3. matched original/perturbed hard, to strengthen the memorization test;
4. Benchmark6 hard, if a non-biomedical control requires narrower intervals.

Stop when a prespecified confidence-interval width or maximum seed count is
reached. Perform a power analysis from the existing paired effects before
purchasing calls.

### B2. Cost-aware search variants

**Next implementation priority after the benchmark freeze:** first replay cached
structures with rollout-aware fitting and no LLM calls. Only variants that pass
that numerical pilot should receive new proposer calls.

**Pilot status (2026-08-09): passed.** The exact-contract audit found 74, 85,
45, and 43 unique reusable structures for named canonical easy/hard and named
perturbed easy/hard Dalla Man T1, respectively. A minimal two-structure
open-loop smoke test obtained at least one valid fit in every cell without LLM
calls or test access. See `docs/PHASE_B_FROZEN_REPLAY_PILOT.md`. The next action
is a larger development-only replay sweep to freeze fitting budgets and a
cached-structure baseline. Paid proposal generation remains deferred until that
sweep is analyzed.

The budget calibration now supports a staged policy: screen each structure at
three evaluations/60 seconds, then warm-start refinement of validation-leading
structures at ten evaluations/120 seconds. A five-structure-per-cell sequential
calibration produced valid screens in all 20 attempts; refinement improved two
cells and was safely rejected in two. The full 247-structure screen is queued
for isolated CPU jobs rather than an oversubscribed laptop run.

- Judge only deterministically valid top-k candidates.
- Invoke the judge only after stagnation or for finalists.
- Stop after validation and structural thresholds are satisfied.
- Stop repeated or alpha-equivalent proposals early.
- Use a small/local model for schema repair and a strong hosted model only for
  semantic generation or final judgment.

Treat these as explicit cost-aware variants, not silent changes to the primary
method.

## Phase C: allocation-backed experiments

### Allocation application plan

Apply to ACCESS and NAIRR in parallel, but give them complementary scopes and
never charge the same run to both awards.

#### ACCESS

1. Submit a **Discover ACCESS** request first. It accepts requests continuously,
   provides 1.5 million ACCESS Credits, and requires a one-page proposal. This is
   a better initial fit than Explore because the work is a research study with a
   defined scaled experiment, not only a porting exercise.
2. Request a GPU resource suitable for reproducible open-weight inference. The
   preferred match is **NCSA DeltaAI** (96 GB H100/Grace-Hopper GPUs); accept an
   equivalent ACCESS production resource with at least 80 GB per GPU if queue or
   architecture constraints make it more suitable.
3. Use the University of Hawaiʻi HPC allocation for CPU-only fitting and analysis.
   Ask ACCESS primarily for GPU inference, modest project storage, and data
   transfer rather than duplicating readily available CPU capacity.
4. Profile 100 representative proposer and judge calls on 8B, 14B, and 70B-class
   open models before finalizing the exchange of ACCESS Credits. Convert measured
   GPU-seconds, tokens, and peak memory into the full request rather than relying
   on vendor throughput claims.
5. Initial planning envelope: 2,000--5,000 H100/A100-equivalent GPU-hours,
   1--2 TB project storage, and only the CPU credits required for preprocessing
   colocated with the GPUs. Use the ACCESS Credit Calculator for the exact
   resource-unit exchange in the submission.
6. If the pilot establishes that the procedural benchmark and 70B model require
   more than the Discover ceiling, submit an **Accelerate ACCESS** upgrade
   (3 million credits; three-page maximum) with measured utilization and scaling
   evidence. Reserve Maximize for a genuinely larger follow-on study.

ACCESS proposal outline:

- scientific aim: trustworthy mechanistic discovery under partial observability;
- why advanced computing is essential: thousands of long structured generations
  and judge calls with open-weight models, plus controlled model-family scaling;
- prior evidence: the completed six-benchmark hosted-model study and cached pilot
  results;
- computation: model sizes, quantization policy, concurrency, tokens per call,
  measured GPU-hours, storage, and checkpointing;
- validation: held-out interventions, structural/hidden-state recovery, failure
  rates, and cost-quality curves;
- reproducibility: immutable containers/environment locks, cached calls, tagged
  code, deterministic resume, and release of non-sensitive manifests;
- milestones: port/profile, open-model pilot, scaled study, confirmatory freeze.

#### NAIRR

1. Apply through the current **NAIRR research allocation** opportunity. NAIRR
   requests are matched to contributed resources after review, so describe
   capability requirements (80 GB+ GPUs, multi-GPU inference, storage, model
   serving) as well as preferred systems rather than depending on one machine.
2. Frame the work around **safe and trustworthy AI for scientific discovery**:
   detecting memorization, grounding LLM judgments in numerical evidence,
   adversarial judge robustness, and validating mechanistic claims.
3. Request the scale-up portion that complements ACCESS: multi-GPU inference for
   70B-class open models, the procedural benchmark suite, and robust judge/model
   family comparisons. Do not ask NAIRR to fund ordinary CPU fitting already
   available at UH.
4. Use the same 100-call profiling evidence and request a planning envelope of
   5,000--10,000 accelerator-hours, 2 TB storage, and access for the PI plus named
   collaborators. Adjust to the units and resource catalog in the live call.
5. The application should explicitly state that no protected or sensitive data
   are used, all test references remain isolated from LLM prompts, and outputs
   are schema-validated as untrusted text.

NAIRR's current process uses a short research request followed by independent
review and resource matching. Before submission, verify the live opportunity,
deadline, eligibility, and offered resources in the portal because contributed
capacity changes over time.

#### Application schedule

- Week 1: create/update ACCESS and NAIRR profiles; identify PI/advisor and users;
  run the 100-call GPU profiling pilot; export utilization evidence.
- Week 2: write the ACCESS one-page narrative and resource justification; submit
  Discover ACCESS.
- Week 2: adapt the same technical core into the NAIRR research narrative, with
  stronger trustworthy-AI and resource-matching sections; submit if the current
  call is open or subscribe for the next round.
- While under review: complete all Phase A analyses on local/UH resources and
  package a container plus scheduler scripts.
- After award: run a small reproducibility gate, then the open-model pilot; only
  launch the full matrix after the predefined quality and throughput gates pass.

### C1. Open-weight proposer and judge study

Use allocated GPUs to evaluate a small, prespecified set of capable open-weight
models as proposer, judge, and structured-output repair model. Retain hosted GPT
and Gemini on a smaller reference subset.

Compare:

- hosted proposer / hosted judge;
- open proposer / hosted judge;
- hosted proposer / open judge;
- open proposer / open judge.

Report quality, completion, tokens, GPU-hours, latency, energy/compute proxy, and
estimated dollar cost.

### C2. Procedural benchmark suite

Generate 50–200 systems with controlled state count, hidden-state count,
nonlinearity, feedback, time-scale separation, noise, input sparsity, and
observability. Because ground truth is known, report target error, hidden-state
recovery, edge/sign recovery, intervention accuracy, term recovery, and failure
rate.

### C3. Additional computational baselines

Prioritize baselines that directly address partial observability:

- latent neural ODE;
- weak/integral SINDy;
- delay-coordinate SINDy;
- universal differential equation;
- state-space symbolic regression.

## Confirmatory study after reliability fixes

Only after the protocol is frozen and failure modes are corrected:

- run ten matched seeds over the frozen Phase-B cases and both tiers if power
  analysis supports the expense;
- run up to twenty seeds on selected hard-tier tasks only when sequential
  confidence intervals remain too wide;
- preserve a smaller untouched confirmatory set after development decisions.

## API-cost controls

- Maintain a single deduplicated content-addressed cache manifest.
- Separate proposal, judge, fit, and data seeds.
- Record planned and actual calls/tokens/cost for every job.
- Require an explicit call budget in each production configuration.
- Provide an offline/replay flag that disables provider construction.
- Preflight every batch with expected calls, cache hits, cache misses, and cost.
- Never treat reuse of an identical cached response as an independent LLM seed.

## Recommended execution order

1. Cache/candidate audit and fail-closed replay.
2. Existing-run statistical analysis.
3. Frozen-model intervention and counterfactual evaluation.
4. Numerical-fitting and frozen-pool selection studies.
5. Obfuscated-perturbed failure recovery.
6. Human structural validation.
7. Power analysis and targeted sequential seeds.
8. ACCESS/NAIRR open-model and procedural-benchmark work.
9. Final frozen confirmatory benchmark.

## Decision gates

- Do not buy more LLM seeds until cache reuse and expected call cost are audited.
- Do not run a full confirmatory matrix until obfuscated-perturbed failures are
  understood.
- Do not claim LLM-search variance from fitter/bootstrap replicates.
- Do not claim exact equation identification when only mechanism-level structure
  is stable.
- Prefer intervention generalization and failure-aware statistics over adding
  many redundant in-distribution runs.
