# Vega handoff for discovery under fallible scientific specifications

Prepared 6 October 2026. Repository: `task-specified-autoformulation`.

Vega will develop the research design for an Autoformalism extension that can
assess and revise fallible scientific assumptions. Codex continues the Phase C
construction runtime; Orion continues fitting and benchmark qualification. This
handoff assigns a bounded first milestone. It does not start an agent, a campaign,
or a change to the production method.

## Research objective

The current framework constructs a dynamical model that conforms to a supplied
task specification. In open scientific discovery, some scientific assumptions
in that specification may be wrong, approximate, or insufficiently supported.
The next system should identify which claims the evidence supports, which claims
are challenged, and which additional evidence would resolve uncertainty.

The initial paper question is:

> Can a system identify and minimally revise an incorrect scientific assumption
> while preserving correct assumptions and distinguishing scientific disagreement
> from insufficient observations, construction failures, and fitting failures?

The first milestone is a design and diagnostic specification, not an unrestricted
search algorithm. Start with a small, explicitly bounded model family and
reviewed mathematical claims. Natural-language interpretation is a separate
source of error to introduce after the diagnosis problem is well defined.

## Current foundation and limitations

Autoformalism has staged variable, process, topology and function construction;
shared process laws reused across equations; latent initialization; parameter
fitting; deterministic validity checks; a scientific LLM critic; bounded
revision; and pruning. Train, validation and test roles are separate. Exact
prompts, replies, budgets, model artifacts and resume identities are recorded.

Phase B compared external baselines and component ablations. Its model inspection
showed that low prediction error, a valid graph, and plausible terminology do not
establish mechanism recovery. Signs can become wrong after evaluating inner
functions or latent states; fitted gains can suppress declared mechanisms;
initial conditions can compensate for an incorrect model or input interpretation.

Phase C separates construction diagnosis from fitting diagnosis. It uses corrected
public development assets and inspects equations and mechanism evidence alongside
NMSE. The current narrow construction confirmation has six basin tasks across
three construction schedules and stops before function generation or fitting.
Its structural checks are not a scientific correctness certificate.

Orion's M13 fitting study is an especially relevant negative control: the same
correct alien-device equations recover accurately from two generic starts and
poorly from a third under the tested allowance. A fitting failure therefore cannot
be treated as evidence that the scientific specification is impossible. Those
are three starts on one problem, not three independent benchmarks.

Earlier input-pulse and trajectory defects also show why an apparent scientific
contradiction may originate in data generation or an execution interface.
Historical releases stay frozen. New diagnosis experiments need isolated,
qualified inputs and explicit provenance.

## Outcomes and evidence requirements

Model conformance and physical adequacy are different questions. A formal checker
may establish that a model satisfies a mathematical claim. Empirical evidence
can challenge or support that claim about the physical system within stated
conditions; it cannot establish unrestricted truth.

| Outcome | Required interpretation and evidence |
| --- | --- |
| Internally inconsistent | Identify conflicting formal claims and give a certificate where the bounded formalism permits one. |
| Empirically challenged | Identify the claim or assumption bundle contradicted by observations or interventions, conditional on the observation model, noise assumptions and examined model family. |
| Approximately valid | State the operating domain and error tolerance under which the approximation is adequate. |
| Supported within scope | Provide a satisfying model and its tested consequences, without claiming uniqueness or universal correctness. |
| Underdetermined | Exhibit unresolved alternatives or a documented evidence gap, and identify a potentially discriminating measurement or intervention. |
| Computationally unresolved | Preserve construction, fitting, numerical or budget failures separately from scientific verdicts. |

Different claims in the same specification can receive different outcomes.
Successful fitting demonstrates that an adequate candidate was found. Failure
to find one does not establish that none exists. A formal impossibility statement
must name the model class, domains, tolerances and other limits of its certificate.

Do not assign physical truth probabilities from an LLM's self-reported confidence.
Any proposed uncertainty measure must name its event, assumptions and calibration
experiment. Candidate agreement alone is insufficient: all candidates may share
the same omission. Abstention is a legitimate result.

## Representing and revising claims

Represent each claim with a stable identifier, original wording, reviewed formal
meaning, affected variables, scope, provenance, and evidence. Distinguish:

1. Measurement definitions and experimental protocols.
2. Scientific hypotheses, such as transfer or delayed response.
3. Approximation assumptions, such as instantaneous equilibrium.
4. User objectives, such as the required prediction target.

Only explicitly designated hypotheses and approximations are eligible for
automatic revision. Suspected measurement errors should trigger a documented
interface diagnosis. They do not authorize rewriting observations or units.
Prediction targets cannot disappear to improve apparent fit.

A conceptual formulation is

\[
p(\mathcal M,C'\mid D,C_0)
\propto
p(D\mid\mathcal M)\,p(\mathcal M\mid C')\,p(C'\mid C_0),
\]

where \(C_0\) is the original specification and \(C'\) a revision. This is a
proposed research formulation, not a claim that the current pipeline optimizes
this posterior. Model complexity and revision costs must prevent arbitrary
models or deletion of every inconvenient claim from becoming trivial solutions.
Vega should propose concrete operational alternatives to exact posterior inference.

Start with a bounded revision vocabulary: retain, relax a stated tolerance,
restrict an applicability domain, remove a revisable claim, or substitute one
reviewed alternative hypothesis. Preserve the original specification and every
revision. Compare candidates using training and development evidence; reserve
fresh interventions for evaluating the resulting scientific conclusions.

## First diagnostic study

Design an initial matrix with three candidate families: coupled storage basins,
a thermal system, and a mechanical oscillator. These are proposed new diagnostic
fixtures; inspect existing assets and coordinate with Orion before choosing their
actual equations. Do not alter existing benchmark data or finalized prompts.

For each family, specify the following six conditions where they are meaningful.
The proposed maximum is 18 conditions, not an authorized production campaign.

| Condition | What the evaluation must distinguish |
| --- | --- |
| Correct and identifiable | Preserve the correct claims and recover their testable consequences. |
| One false scientific claim | Localize the offending claim rather than broadly discarding the specification. |
| Approximation with a limited regime | Identify where it ceases to be adequate. |
| Observationally ambiguous alternatives | Abstain or return alternatives until discriminating evidence is available. |
| Correct structure with difficult fitting | Report computational uncertainty rather than a false scientific accusation. |
| Input or observation-interface defect | Diagnose the interface issue separately from an incorrect physical mechanism. |

Include at least one logical contradiction as a small formal fixture. An
unexcited coupling should not be declared absent merely because no effect is
visible. A difficult optimization start should have an independently established
adequate solution to show why a failure verdict would be wrong.

Before live proposer experiments, require reference/oracle controls to establish
the intended properties of each fixture. Independent integration and checked
input semantics are necessary. Numerical replay agreement establishes numerical
consistency, not scientific truth. Keep private reference evidence outside the
discovery interface and disclose assisted/oracle experiments separately.

Compare fixed-specification construction, unconstrained relaxation, and a bounded
claim-revision method under matched declared budgets. A fitting-only control is
needed to identify improvements obtainable without revising the specification.
Vega should identify the smallest defensible baseline set, rather than propose a
Cartesian sweep of all models, prompts and search strategies.

Primary endpoints are false accusation of correct claims, localization of false
claims, appropriate abstention, revision size, and prediction of fresh
interventions. Calibration or coverage is an endpoint only after defining the
uncertainty quantity. Also report mechanism behavior, NMSE, runtime, provider
calls and tokens. Report failed and unavailable attempts explicitly. Preserve the
project convention of median and unscaled MAD for predictive errors and mean and
SD for compliance, with declared denominators and equal-case aggregation.

## Other paper directions retained from the research discussion

These are follow-on ideas, not concurrent assignments for Vega.

**Choosing the next scientific action.** Decide whether to refit, revise a claim,
ask a scientist, obtain another measurement, run an intervention, or change the
representation. For an accumulator versus a decaying memory, a pulse followed by
observation may resolve uncertainty more efficiently than additional equation
repair. Evaluate information gained and reliable conclusions per total cost.
MCTS is a possible implementation mechanism, not the contribution by itself.

**Discovering identifiable scientific content.** Return adequate competing models,
the claims they share, their disagreements, and experiments that discriminate
them. Separate genuine mechanism differences from latent coordinate or scale
changes. A constructive pair of equally adequate observational models with
different intervention predictions is a useful scientific output. Exact recovery
of one textbook equation string is not always the right endpoint.

**Discovering validity domains.** Learn a model together with operating conditions
and an approximation error claim. A quasi-equilibrium assumption can work under
slow forcing and fail under fast forcing. Assess domain coverage and boundary
reliability on fresh conditions; prevent trivial success by shrinking the domain
to nearly nothing. This can become an extension of the first paper if diagnosis
is already reliable.

**Discovering representations.** Eventually allow evidence to motivate an ODE
with more latent states, delay dynamics, hybrid events, stochastic dynamics, or
a different observation map. Different futures from apparently identical measured
states and subsequent inputs may indicate missing history, after accounting for
noise and measurement error. Model-class changes need bounded alternatives and
explicit costs. This is a longer-term project.

## Prior work and novelty assessment

The following are starting points, not an exhaustive novelty review.

- [AI-Hilbert](https://www.nature.com/articles/s41467-024-50074-w) combines data
  and mathematical background knowledge and addresses inconsistent or incomplete
  theories. Revising scientific assumptions alone is not a new contribution.
  Investigate the distinct challenges of partial observability, dynamic models,
  diagnosis of computational failures and uncertainty about individual claims.
- [AI-Descartes](https://www.nature.com/articles/s41467-023-37236-y) connects
  symbolic discovery and derivability from background theory. Compare its role
  for assumptions with the proposed empirical diagnosis problem.
- [AutoCog](https://arxiv.org/abs/2606.26448), a 2026 preprint, studies automated
  theory and experiment loops. A future active-discovery paper needs a sharper
  contribution than adding an experiment-design agent.
- [Model-based experimental design in sloppy systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC5140062/)
  provides relevant context for separating parameter uncertainty from inadequate
  models. More informative experiments can expose missing physics.
- [dReal](https://dreal.github.io/) is relevant to bounded nonlinear verification;
  do not generalize supported formal certificates to an unrestricted equation
  discovery problem.

## Vega milestone one

Read `AGENTS.md`, `docs/PHASE_C_START_HERE.md`, relevant parts of
`docs/PROJECT_CONTEXT.md` and `docs/PIPELINE_DESIGN.md`, then:

- `work/agent-comms/2026-09-26-model-inspection-findings-and-roadmap.md`
- `work/agent-comms/2026-09-26-cstr-reference-defect-summary.md`
- `docs/PHASE_C_CONSTRUCTION_COMPARISON.md`
- `docs/PHASE_C_DATASET_READINESS.md`
- `docs/PHASE_C_FITTING_M13_RESULTS_2026-10-06.md`

Use these to identify reusable interfaces. Do not assume every historical
discussion or diagnostic has been promoted into the production method.

Deliver within the first work package:

1. A concise research design with a falsifiable central claim and a primary-source
   comparison against the closest prior work.
2. A typed claim/evidence record proposal with examples of supported, challenged,
   ambiguous and computationally unresolved outcomes.
3. The proposed diagnostic matrix, including oracle controls, observation
   assumptions, measurement semantics and intervention separation.
4. A minimal implementation map: reusable components, proposed new files, tests,
   and estimated CPU/GPU/provider-call budgets with assumptions.
5. A short recommendation on whether the study is worth implementing, including
   the strongest counterargument and unresolved scientific choices.

Write these under `work/vega/fallible-specifications/`. Keep implementation,
benchmark releases, production configuration and existing results read-only for
this first milestone. No live LLM calls, optimization, scheduler submissions or
test-data access are part of this design assignment. A proposed controlled
experiment may be described without being executed.

Codex reviews construction integration and evidence semantics; Orion reviews
reference equations, input contracts, fitting controls and numerical claims.
Return a compact completion record with deliverable paths, sources, assumptions,
unresolved questions and measured usage if available. Do not create new agent
threads or send messages to other chats solely because this document names them.

The milestone succeeds when the proposed study could distinguish an incorrect
scientific assumption from a correct assumption that the current solver cannot
fit. A plan that equates high NMSE with a false specification is not acceptable.

## Allocation planning context

ACES remains useful for controlled open-weight inference and independent
construction tasks; Delta supports fitting diagnostics, while Jetstream2 supports
managed model calls and suitable CPU orchestration. Keep provider identity and
numerical environment recorded. Do not merge partially copied mutable campaigns
across sites or silently substitute providers within a matched experiment.

The [ACES batch guide](https://hprc.tamu.edu/kb/User-Guides/ACES/Batch/#batch-queues),
checked 6 October 2026, lists 40 maximum running jobs for CPU and GPU queues.
Available GPUs and account/QOS policies still constrain execution. This is not
40 guaranteed GPU allocations. Prior project logs include a submission-limit
rejection; the live account limits must be inspected before increasing arrays.

`docs/FINAL_COMPONENT_CAMPAIGN.md` specifies an eight-proposer, 32-fit-worker
default wave, plus four critic and eight pruning workers. These are configured
requests, not measured simultaneous utilization. Current topology confirmations
use one proposer job and iterate through tasks sequentially in
`scripts/phase_c_construction_comparison.py`. Small diagnostic batches explain
some deliberate restraint, but this architecture leaves independent-case
parallelism available for larger studies.

The investigator reports approximately 5 percent of the ACES allocation consumed
and an end date of 4 November 2026. This is not a measured GPU-utilization rate.
Historical GPU waits, file-count quota failures, debugging, work at other sites,
and small sequential diagnostic runs all matter; the available evidence does not
assign a percentage of the low consumption to any one cause.

For a subsequent production plan, measure ready-work backlog, eligible-to-start
wait, running concurrency, useful work per allocation, GPU activity and failed
startup cost. Reconstruct concurrency from overlapping allocation start/end
times, excluding duplicate batch/extern records. Queue-reason snapshots are
needed to separate resource contention from dependencies or holds. Increase
independent proposer services gradually and size CPU arrays to the ready backlog
and live policy limits. Avoid adding workers that merely wait for upstream work.

The accompanying progress report requests an extension through 4 May 2027,
the six-month extension selected by the investigator. Its 5 percent figure comes from the investigator;
no live allocation ledger or complete scheduler history was retrieved for it.
