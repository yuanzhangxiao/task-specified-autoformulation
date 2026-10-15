# Phase C: numerical feedback for interaction proposals

Design recorded 15 October 2026. The first
[offline adapter and synthetic qualification (M37)](PHASE_C_FITTING_INTERACTION_ADAPTER.md)
are implemented. The [M38 discrepancy extraction and conditional draft](PHASE_C_FITTING_FEEDBACK_DRAFTS.md)
now adds competing explanations, predicted observable improvements and
disconfirmation criteria to those verified packets. Current-constructor function binding, model edits and the live
paired experiment below remain later milestones. The public
specification, observations, input semantics and observation/initialization
contract stay fixed. Revising scientific claims or sensor assumptions belongs to
the [Vega research plan](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md).

## What we can reuse now

The fitter is substantially stronger than the Phase B backend: accurate rollout,
guarded scaling, eligible coefficient/initial-value profiling, continuation and
different starts, incumbent preservation, and separately budgeted diagnostics.
This is a qualified portfolio on tested problems, not a guarantee that correct
equations always yield the correct coefficients.

| Existing component | Reuse and remaining connection |
| --- | --- |
| [Current construction evidence](PHASE_C_EVIDENCE_FEEDBACK.md) | Public requirements, shared-process roles, dependency paths and factual training highlights. Topology checks do not establish quantitative function behavior. |
| [Public fitting adapter](PHASE_C_FITTING_PUBLIC_INTEGRATION.md) | Compile a complete executable request, preserve all targets and causal initial rules, and fit under a frozen policy. Fresh current constructions still need qualification. |
| [M31 feedback](PHASE_C_FITTING_FEEDBACK.md), [M32 staged diagnostics](PHASE_C_FITTING_STAGED_FEEDBACK.md), [M33 portable verification](PHASE_C_FITTING_DIAGNOSTIC_SUPPLEMENT.md) | Candidate/vector-bound training evidence, independent replay, search status, local joint information and diagnostic availability/cost. The M33 remote closeout is separate from the completed M36 experiment. |
| `search/response_evidence.py`, `search/response_revision.py` | Historical response-shape summaries and compact target-balanced packets: excursion, peak timing, half-return and ambiguity flags. Adapt their identities and assumptions; do not equate their historical replay with the newer independent-solver check. |
| `search/numerical_sibling.py` | Historical proposal protocol: measured mismatch, conditional function hypothesis and predicted observable improvement; explicit function/no-change/topology outcomes. Its older residual schema is not a ready-made M32 adapter. |
| [M34–M36 measurement work](PHASE_C_NOISE_AWARE_FITTING_DESIGN.md) | Explicit observation/preparation assumptions and conditional information. No implicit drift correction, robust-loss switch or outlier deletion in the current clean benchmark protocol. |

The historical generic `search/feedback_routing.py` includes validation metrics
and failures; `tests/test_staged_search_integration.py` explicitly checks this.
The new Phase C connection must construct a **training-only** packet directly.
It must not forward that generic payload and merely rename it training evidence.
No validation/test scores, clean truth, reference coefficients or fault labels
may enter the proposer packet. Evaluation is a separate frozen operation.

## The feedback should answer four practical questions

1. **Can we trust the numerical evidence at this fitted point?** State independent
   rollout agreement, domain/integration failures, derivative scope, missing
   probes and exact retained candidate/vector. A timeout is not instability.
2. **Is more fitting likely to help?** Preserve recent verified improvements,
   tried starts, remaining budget, and the scope of any stationarity or optimum
   certificate. Budget exhaustion alone does not explain the residual error.
3. **What response does this model get wrong?** Summarize all targets, the worst
   trajectories and a bounded selection of informative input/response windows.
   Compare predicted versus observed excursion, timing and recovery where those
   quantities are defined. Retain physical units and exact artifact references.
4. **What changes are permitted and worth testing?** Show relevant dependency
   paths, current functions, shared-process consumers, source sets, outer signs,
   parameter domains and functional obligations. These guide hypotheses without
   assigning causal blame to a term from an output residual alone.

A proposed packet therefore combines identity, numerical/search status, per-target
response evidence, conditional parameter information, and an immutable edit
contract. Preserve multiple flags: a bound can be active at a rank-deficient
point. Local rank is neither calibrated confidence nor scientific correctness.
Do not attach an incumbent's stationarity certificate to a different vector.

Use a compact target overview plus, initially, up to two response examples per
target under a fixed prompt-size allowance. M37 describes early, middle and late
windows within those selected trajectories. Full arrays remain in hashed
artifacts. Record omitted windows and why examples were selected. Training
responses with overlapping inputs, repeated peaks, a peak outside the recorded
window, or a recovery not reached must retain their ambiguity flags; they must
not become a fabricated delay or a zero decay time. Fitted latent trajectories
are model predictions, never measured evidence. Analyses of sampled term signs
must not be described as global monotonicity proofs.

## Route the work without overstating the diagnosis

| Evidence | Appropriate action |
| --- | --- |
| Rollout/input execution unverified | Repair the execution problem or obtain the missing numerical check before interpreting residuals |
| Verified fit still improving | Continue the incumbent within the controller's allowance; a separately allocated function hypothesis remains possible |
| Poor fit after local stationarity or repeated plateau | Permit a conditional function hypothesis or a different start; neither route proves structural impossibility |
| Certified optimum of a restricted candidate/domain still poor | Review that candidate's function family, allowed dependencies and constraints; the certificate does not refute the public specification |
| Accurate outputs, weak/ambiguous parameter information | Preserve alternatives and report the ambiguity; do not force an equation repair or claim recovered coefficients |
| Accurate outputs and supported local information | Retain; evaluate mechanism obligations separately and do not equate local information with ground-truth recovery |

The controller decides the effort allocation. Once it routes a visit to function
revision, the prompt asks for a concrete hypothesis and predicted response change,
not an explanation that optimization ran out of time. It does not require proof
that the parent is impossible. An explicit no-change response remains legitimate
if the available evidence cannot support a hypothesis; name the missing evidence.

For example, an **illustrative**, not measured, message would be:

> Independent rollouts agree. Across the cited training inputs, the fitted model
> returns toward baseline earlier than the observations. Recent continuation has
> made little improvement; global optimality is unproven. Inspect the displayed
> functions along the response pathway. Propose one allowed functional change,
> explain conditionally how it could lengthen the response, and predict which
> observed window should improve after refitting. If that requires a missing
> state or source, request topology revision explicitly.

There may be several explanations: a decay coefficient, nonlinear saturation,
a downstream observation law, or missing memory. The evidence must not supply a
ground-truth formula or declare one explanation uniquely correct. Likewise,
insensitivity to a parameter is a diagnostic, not permission to prune a
task-required mechanism.

## Planned application and evaluation of revisions as complete candidates

The planned edit-enabled version permits one existing interaction or one shared
process law per visit. A shared law changes every consumer consistently. The proposer preserves
the declared source set, public mechanism obligations, outer topology signs,
variable inventory, observation map and initialization contract. A missing
dependency, state or necessary observation/initialization revision requires an
explicit broader-stage request; it cannot be smuggled into a function slot.

The runtime validates the restricted expression and parameter declarations,
namespaces/inherits parameter identities under the established policy, and
records the edit atomically. Mistyped evidence references are recorded as
unverified, with no invented replacement or evidence credit; they do not by
themselves discard an otherwise admissible equation. A string match can verify
a reference's existence, not that its scientific interpretation is correct.

Refit all permitted parameters and initials jointly, using profiling only where
the **integrated outputs**, not merely the RHS, have the required affine form.
Preserve all observed targets and verify the resulting complete rollout. Compare
per-target and worst-trajectory changes so an aggregate gain cannot hide damage
to another target. Preserve the parent on invalid edits, fitting failures or
unverified improvements. Parent/child IDs, diagnostic costs, all failed attempts,
and deterministic resume histories remain explicit.

## First implementation milestone and experiment

**M37 implements the offline adapter and small synthetic controls.** Astra's
supplied topology report has no executable functions, so its models cannot yet
be fitted. The first real-construction qualification remains pending. Take one
current topology after executable functions and observation/initialization maps
are available.
Use a public basin case with complete required outlet coverage as the first
integration target; that choice is provisional until the artifact is available.
Run the current fitter and produce a bounded packet consumable by the function
revision interface. Do not simultaneously change the optimizer or benchmark.

Before live proposals, require controls for: correct equations with an unfinished
fit; a wrong function with a solved restricted fit; ambiguous parameter/initial
combinations; missing dependencies requiring topology work; multiple outputs and
shared-process edits; stale hashes; unavailable diagnostics; citation mistakes;
validation leakage; and interrupted resume. Expected labels and reference
solutions stay evaluator-only. An existing measured summary is not a substitute
for these interface checks.

Then freeze a small paired experiment, initially three proposal seeds and one
revision visit per seed, with:

- unchanged-model continuation/refit;
- function revision using the same training response summaries;
- function revision using those summaries plus numerical/search and joint
  parameter-information diagnostics.

Use the same parent checkpoints and frozen fitting/restart-generation policies,
seeds and allocations. Preserve compatible inherited coordinates and generate
new coordinates deterministically within their declared domains. Different
function forms can have different parameter dimensions, so identical numerical
restart vectors across arms are not generally possible. Match refitting ceilings
across arms and proposal ceilings across revision arms; charge diagnostic
overhead separately. Every unchanged, failed,
topology-request and accepted-child outcome remains in the denominator. Cache
and log every LLM call. Keep the parent/child retention rule training-only for
this qualification, freeze endpoints, and evaluate validation separately without
returning it to the proposer or selecting the best proposal seed. Test stays
sealed. Any later validation-selection policy needs its own frozen declaration.

The experiment asks whether qualified feedback improves *function revisions*,
not merely whether extra fitting helps. Report paired train/validation changes
for every target, harmful revisions on correct controls, admissibility, preserved
mechanism obligations, equation complexity, ambiguous/unresolved outcomes and
total compute/tokens. It cannot establish calibrated causal fault localization.

This ordering returns fitting to the current paper now, while preserving the
identities, assumptions and alternatives needed for Vega's future work. General
noise estimation, sensor-hypothesis selection, near-confounding calibration and
fallible-specification discovery remain later research milestones.
