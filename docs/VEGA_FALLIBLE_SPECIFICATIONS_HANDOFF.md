# Vega handoff: fallible scientific specifications and imperfect data

Prepared 6 October 2026; updated 15 October 2026 with completed M34–M36
measurement qualifications and the priority of connecting fitting evidence to
the current paper's interaction/function stage.
Repository: `task-specified-autoformulation`. The existing filename and work
directory are retained so earlier handoffs remain discoverable.

Vega will develop the research design for an Autoformalism extension that can
assess fallible scientific assumptions together with imperfect observations and
input records. Astra leads the Phase C construction workstream; this fitting
workstream continues numerical qualification and the feedback interface. This
handoff assigns a bounded first milestone. It does not start an agent, a campaign,
or a change to the production method.

## Research objective

The current framework constructs a dynamical model that conforms to a supplied
task specification. In open scientific discovery, some scientific assumptions
in that specification may be wrong, approximate, or insufficiently supported.
Observations and recorded interventions can also contain noise, calibration
errors, missing values, timing errors, or incorrect input magnitudes. The next
system should identify which claims and measurement assumptions the evidence
supports or challenges, and which additional evidence would resolve uncertainty.

The initial paper question is:

> When a dynamical model disagrees with observations, can a system distinguish
> fallible scientific assumptions, observation/input defects, inadequate candidate
> equations, and computational failure; make minimal supported revisions; and
> abstain or seek new evidence when the explanations are indistinguishable?

Causes are not mutually exclusive. A false assumption and a faulty sensor may
coexist. The intended contribution is reliable diagnosis and choice of the next
scientific action under partial observability, not merely adding a robust loss
or allowing the system to rewrite whichever evidence is inconvenient.

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
NMSE. Construction confirmations still stop at topology, before function
generation or fitting. Their structural checks are not a scientific correctness
certificate. The six-basin-task confirmation described in the original handoff
is historical; current construction policies and rosters are recorded separately.

Orion's M13 fitting study is an especially relevant negative control: the same
correct alien-device equations recover accurately from two generic starts and
poorly from a third under the tested allowance. A fitting failure therefore cannot
be treated as evidence that the scientific specification is impossible. Those
are three starts on one problem, not three independent benchmarks.

M29 and M30 extend this evidence through the public fitting-request adapter.
Three known-structure controls recover accurately, while none of the available
archived constructions becomes accurate. These are September 30 constructions,
not outputs of the current topology work. In independent basin, an exact
training-only optimum establishes that the candidate's restricted output equation
cannot fit adequately; its omitted drainage remains a construction defect.
M30 removes redundant optimization for that solved output problem without
repairing its equations. The other cases retain missing output dependencies,
scientifically wrong signs, unresolved searches or unavailable numerical rollouts.
Those findings have different evidential strength and must remain separate.

M30's local stationarity witnesses do not certify global optimality or physical
adequacy, and a witness from another search branch cannot describe the retained
parameter vector. The three accurate controls establish recovery on those
controls, not arbitrary constructions or noisy data. General computational
speedup also remains unproven: total recorded task time increased in the single
paired run despite the independent-basin saving.

M31/M32 add a candidate-bound, training-only fitting-feedback interface. M32
accounts for all 23 tasks, with twelve passing synthetic diagnostic controls.
The known-structure alien-device and coupled-basin controls now have supported
local sensitivity evidence. CSTR preserves accurate output replay and passes
seven coefficient directions, but exhausts the diagnostic allowance before
qualifying its two hidden initials and weakest joint direction. A real kink and
a bound-limited control correctly withhold stronger derivative conclusions.
These are limits of the assessment, not proofs of unidentifiability. The archived
constructed models are unchanged. See the
[M32 review](PHASE_C_FITTING_M32_RESULTS_2026-10-13.md).

[M33](PHASE_C_FITTING_DIAGNOSTIC_SUPPLEMENT.md) implements portable verification
of SVD-derived evidence and a separately budgeted CSTR diagnostic supplement.
It preserves sealed Jacobians and actual perturbation plans, distinguishing
numerical/subspace equivalence from exact artifact identity. The user has
submitted its qualification; its separate remote closeout is not established by
the M36 archive. Neither local rank nor the stronger fitting backend establishes
correctness under an uncertain observation model.

M34–M36 now qualify bounded measurement extensions on isolated synthetic
controls. They prepare the future research topic; they do not change Phase C
benchmark observations or silently introduce sensor corrections into current
model discovery.

| Milestone | Completed evidence | Remaining boundary |
| --- | --- | --- |
| [M34: known-noise measurement fitting](PHASE_C_FITTING_MEASUREMENT.md) | 36/36 numerical replays, 30/30 applicable affine optimum checks, 18/18 direct/profiled objective agreements; explicit exact or shared uncertain initials | Fixed known diagonal noise; multi-output profiling remains unqualified and uses joint fallback |
| [M35: fixed Huber loss](PHASE_C_FITTING_ROBUST_MEASUREMENT.md) | 96/96 numerical replays, 72/72 applicable affine optimum checks, 48/48 paired agreements; lower clean-trajectory error in all nine contaminated case/seed pairs | Robust weights are influence controls, not corruption labels; no calibrated robust confidence |
| [M36: shared sensor drift](PHASE_C_FITTING_SENSOR_DRIFT.md) | 144/144 independent rollout/objective and derivative checks, 120/120 applicable affine optimum checks, 72/72 paired agreements; exact ambiguity and resolving-input controls | Small known equations; observation-hypothesis selection, broad fault attribution and statistical coverage are unqualified |

In M36, `y(t)=x(t)+b*(t-t_ref)+epsilon(t)`: drift changes a sensor reading, not
the physical ODE. With `x'=g*u`, known initial state and common constant input
`u0`, the data constrain only `g*u0+b`. The full joint diagnostic detects this
ambiguity; optimizing more accurately cannot separate the two. Differing known
input levels across runs resolve this example. This result does not resolve
the physical-gain/sensor-gain product ambiguity described below.

All 72 no-drift fits had locally full-rank information, including fits to drifting
readings. Thus rank is conditional on the assumed equations and observation
contract, not a truth certificate. The drift-aware nonlinear controls also
retained coefficient error despite full rank. Numerical reliability, local
parameter information, practical precision and scientific correctness must be
separate outputs. Preserve simultaneous flags such as rank deficiency and an
active parameter bound, rather than hiding one behind a headline status.

M36's separate diagnostics cost 52.89 CPU seconds against 195.14 fitting CPU
seconds (27.1% additional), with all 1,200 calls accounted. The next research
design should include such overhead and qualify practical uncertainty on noisy,
near-confounded cases before offering a numerical confidence level.

[M39 practical uncertainty](PHASE_C_FITTING_PRACTICAL_UNCERTAINTY.md) qualifies
five input separations, three known-noise levels and five paired noise seeds on
one affine physical/sensor/initial-value family. It compares nuisance-reoptimized gain
profiles with frozen-nuisance slices, preserving domain-assisted bounds and exact
confounding. The returned run verifies all 150 fit endpoints and 75 method pairs;
141 profiles complete and 9 hit their separate wall allowance. Proper support
widens while the joint design remains full rank and measured prediction error
stays nearly unchanged. This separates practical uncertainty from fitting failure.
Profiles add 48.80 CPU seconds to 58.26 fitting CPU seconds (83.8% overhead in this
tiny control). Diagnostic completion and wall/I/O timing are next; nonlinear
profiles and calibrated coverage remain unqualified. Objective-rise support sets
have no assigned confidence percentage. This prepares uncertainty feedback while
Astra completes interaction construction; it does not launch the broader model/data attribution
study or alter the clean benchmark measurement contract.

Earlier input-pulse and trajectory defects also show why an apparent scientific
contradiction may originate in data generation or an execution interface.
Historical releases stay frozen. New diagnosis experiments need isolated,
qualified inputs and explicit provenance.

## Distinguish noise, defects, and model discrepancy

| Source of disagreement | Interpretation and candidate response |
| --- | --- |
| Ordinary measurement noise | Correct measurements with finite precision; use a declared stochastic or bounded-error observation model. Nonzero residuals are expected. |
| Sparse erroneous readings | Possible outliers or transcription faults; investigate provenance and a bounded contamination model. A large residual alone is not an outlier label. |
| Systematic observation defect | Sensor bias/gain, saturation, wrong units or channel identity; compare explicit calibration hypotheses and independent calibration evidence. |
| Input/time/initialization defect | The applied intervention may differ from its log; timestamps, pulse duration, or initial observations may be wrong. Output-only robust loss need not repair this. |
| Scientific/model discrepancy | An incorrect claim, approximation or candidate equation may explain systematic error even when measurements are sound. |
| Numerical/search failure | Inaccurate integration, derivative uncertainty, initialization failure or exhausted optimization remains a separate cause. |

These are hypotheses to investigate, not classifications inferred directly from
residual shape. For example, if `x' = a*u - b*x`, `x(0)=0`, and a sensor reports
`y = s*x`, then `y' = (s*a)*u - b*y`. Without calibration or another informative
measurement, the physical gain `a` and sensor gain `s` cannot be separated: the
observations depend on them only through their product. Fitting more accurately cannot resolve this
ambiguity. A timing offset can likewise mimic an apparent response delay over
some protocols; test distinguishability rather than assume it.

Retain competing explanations and identify the measurement that could separate
them: sensor calibration, repeated measurements, a second channel, an independent
input log, or a discriminating intervention. Changing the input schedule cannot
break every ambiguity (including the gain-product example above).

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
| Observation/input explanation unresolved | State the competing physical and measurement hypotheses; do not turn a residual anomaly into a confirmed data error. |

Different claims in the same specification can receive different outcomes.
Successful fitting demonstrates that an adequate candidate was found. Failure
to find one does not establish that none exists. A formal impossibility statement
must name the model class, domains, tolerances and other limits of its certificate.

Preserve three levels of evidence:

1. **Candidate fit:** report what a specified search found for one executable
   candidate and parameter vector. Exhausted search, numerical unavailability and
   local stationarity do not establish the candidate's best achievable error.
2. **Candidate equation adequacy:** an exact optimum or other bounded certificate
   may establish that these equations cannot meet a stated error tolerance on
   these observations. M29/M30 independent basin provides this narrow example.
   It does not rule out another construction satisfying the original claims.
3. **Claim-family incompatibility:** challenging a scientific claim requires
   evidence concerning the declared family of models that preserve it, conditional
   on qualified observations, input semantics, noise assumptions and scope.
   Name the coverage and limits of any empirical comparison; only an appropriate
   formal certificate licenses a bounded impossibility statement. Improving one
   candidate after deleting a claim is insufficient to establish this level.

Each level is conditional on a stated data/observation contract. For a simple
example, a nondecreasing model cannot explain a measured decline greater than
`2*epsilon` between two times when each reading has a certified absolute error
bound `epsilon`. Under unbounded Gaussian noise, the same decline is not an
absolute impossibility: it supports a statistical test only under specified
noise assumptions and a calibrated decision procedure. Near-zero NMSE thresholds
from noiseless Phase C controls must not become noisy-data rejection thresholds.

An adequate, conforming model under the original claims and a stated observation
contract is a counterexample to an accusation that those claims cannot explain
the examined observations under that contract. Its existence still does not
establish uniqueness, correctness of the observation contract, or unrestricted
physical truth.

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

In the proposed future study, only explicitly designated scientific hypotheses,
approximations, and separately designated observation/input-error hypotheses
would be eligible for revision. Prediction targets and user objectives remain
fixed. Preserve raw readings, timestamps, units and input logs immutably; a
proposed calibration, mask or correction is a separate versioned interpretation
with its own assumptions, cost and evidence. Neither this handoff nor Phase C
preparation authorizes automatic data cleaning or specification revision.

A conceptual formulation is

\[
p(\mathcal M,C',\psi,Z\mid D_{\rm raw},C_0,H_0)
\propto
p(D_{\rm raw}\mid\mathcal M,Z,\psi)\,
p(Z\mid\psi,H_0)\,p(\psi\mid H_0)\,
p(\mathcal M\mid C')\,p(C'\mid C_0),
\]

where \(C_0\) is the original specification, \(C'\) a revision, \(H_0\) the
recorded measurement/input contract, and \(\mathcal M\) includes equations,
parameters and initial conditions. \(\psi\) selects a bounded observation/error
model; optional \(Z\) contains uncertain true inputs, true sampling times or
contamination indicators. For exact public inputs those components of \(Z\)
remain fixed. Ordinary latent ODE states remain in \(\mathcal M\), not an
unrestricted per-sample data correction. All likelihoods compare predictions
with the immutable raw records through the declared observation model.

This is a proposed research formulation, not implemented posterior inference.
Model complexity, claim revision, calibration and contamination costs must
prevent arbitrary explanations. Do not allow free offsets for every observation,
unbounded noise inflation, unlimited discarded points or deletion of all claims.
Vega should propose operational alternatives and examine sensitivity to these
penalties; a preferred posterior explanation is conditional on its priors.

Start with a bounded revision vocabulary: retain, relax a stated tolerance,
restrict an applicability domain, remove a revisable claim, or substitute one
reviewed alternative hypothesis. Preserve the original specification and every
revision. Compare candidates using training and development evidence; reserve
fresh interventions for evaluating the resulting scientific conclusions.

## First diagnostic study

Start with one tractable basin family and a small factorial diagnostic design:
correct versus one false revisable claim, crossed with four data conditions:
clean, declared ordinary measurement noise, sparse outliers, and one reviewed
systematic timing or calibration defect. This gives eight condition types,
including simultaneous specification/data defects. Predeclare disturbance sizes
and replicate seeds; eight types do not mean eight statistical samples. Include
conditions in which observation error obscures evidence for the false claim.

Add a small set of targeted controls for an exact observation/parameter ambiguity,
a difficult fitting start with a known adequate solution, and a limited-regime
approximation. Introduce thermal and mechanical fixtures only after this pilot
can separate recoverable from intrinsically ambiguous cases. This replaces the
earlier proposed 18-condition breadth-first plan; it is still a design, not an
authorized campaign or permission to change benchmark data/finalized prompts.

Generate disturbances only in isolated derived development fixtures. Keep clean
reference trajectories, corruption masks, true calibration and random seeds
evaluator-only. The discovery interface receives the deliberately available
measurement contract, without hidden fault labels or a clean companion trace.

Include at least one logical contradiction as a small formal fixture. An
unexcited coupling should not be declared absent merely because no effect is
visible. A difficult optimization start should have an independently established
adequate solution to show why a failure verdict would be wrong.

Pair a defective construction with an adequate, conforming witness under the
same correct specification. Separately include a false revisable claim with a
reviewed alternative and evidence that addresses the bounded claim-constrained
family. This makes construction failure and scientific disagreement distinguishable
without treating an improvement after claim deletion as proof. Retain ambiguous
alternatives whose predictions separate only under a fresh intervention.

Before live proposer experiments, require reference/oracle controls to establish
the intended properties of each fixture. Independent integration and checked
input semantics are necessary. Numerical replay agreement establishes numerical
consistency, not scientific truth. Keep private reference evidence outside the
discovery interface and disclose assisted/oracle experiments separately.

Compare fixed-claim fitting, bounded claim revision, observation-error/robust
fitting with fixed claims, and bounded joint diagnosis under matched declared
budgets. Include an oracle-known observation correction as a separately labeled
oracle reference control when informative. A permissive relaxation method is optional
as a failure demonstration, not a substitute for these matched controls. Robust
loss reduces outlier influence; it does not establish which readings are wrong.
Do not select whichever loss or noise scale makes a preferred hypothesis win.

Primary endpoints are false accusations of correct claims and sound data,
localization of claim/data defects (allowing both), appropriate abstention,
revision/correction size, and prediction of fresh interventions. Report defect
localization only where designed to be distinguishable. Score raw-observation
likelihood and clean physical prediction separately; clean truth is evaluator-only
in synthetic studies. Never silently score a cleaned trace against itself.
Calibration or coverage requires a defined uncertainty event and independent
replication. Also report mechanism behavior, NMSE, runtime, provider
calls and tokens. Report failed and unavailable attempts explicitly. Preserve the
project convention of median and unscaled MAD for predictive errors and mean and
SD for compliance, with declared denominators and equal-case aggregation.

## Phase C preparation while construction remains at topology

The heading is retained for existing links. Construction is approaching the
interaction/function stage. **The immediate priority is now current-paper
integration**, rather than another sensor-error experiment. M1–M30 provide the
fitting portfolio; M31–M33 provide candidate-bound diagnostic infrastructure;
M34–M36 add bounded measurement qualifications useful for future work. None of
these establish reliable fitting or diagnosis for every newly proposed equation.

The [interaction feedback plan](PHASE_C_INTERACTION_FITTING_FEEDBACK.md) specifies
the next bounded step: connect an executable candidate to the qualified fitter,
then expose compact training-response mismatches, numerical/search status and
conditional parameter information to a function proposer. A revision must state
a testable hypothesis and an expected observable improvement. Preserve the
incumbent, refit the child, and verify its complete rollout independently.
Changes to source sets or latent inventory return to topology explicitly.

Existing topology evidence, historical response-window feedback and M31/M32
packets are distinct interfaces. They are not yet connected for current
constructions. In particular, the historical generic feedback route includes
validation metrics; the new connection must explicitly enforce training-only
disclosure. Validation is a separate frozen evaluator role and test data stay
sealed. This is an interface qualification plan, not a completed live experiment.

The public specification and measurement/input contract remain fixed in this
current-paper loop. We do not add sensor drift, choose a new loss, relabel samples
or relax scientific claims merely because a candidate fits poorly. The reusable
lesson from M34–M36 is to declare assumptions and preserve ambiguity. Revision
of those assumptions belongs to Vega's future study.

**The three preparation designs now have different implementation status:**

| Design | Concrete deliverable and boundary |
| --- | --- |
| [Measurement evidence contract](PHASE_C_MEASUREMENT_EVIDENCE_DESIGN.md) | A bounded raw-reading, calibration, masking and preparation subset is implemented in M34; the broader provenance design remains a roadmap. |
| [Noise-aware fitting](PHASE_C_NOISE_AWARE_FITTING_DESIGN.md) | P1/P2 known-sigma objectives and weighted profiling, P3 fixed Huber, and P4 bounded shared drift are qualified by M34–M36. No general uncertainty calibration or noisy-input inference. |
| [Controlled diagnosis study](PHASE_C_SPEC_DATA_DIAGNOSTIC_STUDY.md) | Still a design: finite basin catalog, competing claim/data hypotheses, mixed causes and ambiguity controls. It has not been launched by completing M36. |

The initial-value rule remains essential: a noisy first target reading cannot
also be treated as an exact initial state. M34–M36 qualify prepared initials and
explicitly shared unknown preparations, not arbitrary trajectory-specific initial
inference. Do not use held-out observations retrospectively.

Vega can reuse the completed interfaces and controls rather than reconstructing
them. Novelty analysis, practical uncertainty calibration, competing claim/data
explanations and scientific action selection remain future research. Near-
confounding experiments are deferred while the current interaction connection
is qualified. No new numerical campaign or agent is launched by this handoff.

**Design now, add incrementally in separately reviewed milestones:** extend the
evidence contract with the following proposed fields. These are not all
implemented by M32.

| Evidence group | What to preserve and why |
| --- | --- |
| Data/experiment identity | Immutable raw-data hash; channel units and calibration; timestamps/time base; intervention interpolation and pulse semantics; initial-observation contract; missing/censored readings and acquisition provenance. |
| Uncertainty assumptions | Per-channel noise/error model, covariance or error bounds when justified, source of each estimate, and known/estimated/unavailable status. Solver tolerances are not measurement uncertainty. |
| Objective identity | Loss, noise scale, normalization, weights/masks and estimation split. A changed objective is a new experiment, not a transparent resume of the previous optimum. |
| Residual evidence | Raw as well as standardized/weighted residual summaries by target and trajectory; temporal correlation, repeatability, pulse-aligned errors and localized large residuals. Preserve exact sample references in artifacts, with compact proposer summaries. These are observations, not fault verdicts. |
| Joint uncertainty | Physical coefficients, latent initials and any permitted observation/input nuisance parameters assessed together. Declared calibration bounds and priors remain visible; do not freeze an uncertain nuisance quantity merely to manufacture parameter confidence. |
| Diagnostic execution | Point-bound stationarity, qualified derivative scope, missing probes, actual perturbation vectors, numerical-library/BLAS provenance, and separate fitting/diagnostic costs. |
| Interpretation | Evidence level, assumptions required for a stronger claim, unresolved alternatives and potentially discriminating measurements. Keep fitter `specification_verdict=not_assessed`; no automatic data-error verdict either. |

For a declared nonsingular covariance `Sigma`, a possible noise-aware diagnostic
uses the whitened residual `r_w = Sigma^(-1/2) * (prediction - observation)` and
the joint Jacobian with respect to coefficients, initials and eligible nuisance
parameters. This is a proposed interface, not a qualified M32 noise model.
Prediction error should be interpreted against a justified noise scale and
residual adequacy, rather than the existing near-zero noiseless thresholds.
Residual-estimated noise can absorb missing mechanisms, so document its source
and compare against replicate/calibration evidence when available. Gaussian
likelihood, bounded-error checks and robust losses require different uncertainty
interpretations; full rank alone provides no calibrated confidence level.

Qualify noise-aware fitting on known equations first. In a later isolated pilot,
compare ordinary/weighted least squares, bounded robust fitting, and an explicitly
specified observation-error model. Existing coefficient profiling is reusable
only when its conditional subproblem remains appropriate: fixed Gaussian weights
preserve linear least squares, whereas Huber-type loss requires a different
convex inner solve for affine coefficients. Unknown timing, inputs or calibration
can introduce new nonlinearities and confounding. Keep all nuisance-search,
diagnostic and optional profile-minimization overhead visible.

Noise realizations, outlier locations and clean companion trajectories belong to
isolated derived fixtures; do not modify Phase C benchmark releases or finalized
prompts. Construction feedback remains training-only, with validation/reference/
test evidence excluded. Any later development-level protocol selection must stay
outside per-candidate fitter/proposer feedback, with a fresh evaluation set.
Oracle controls qualify fixtures but never supply hidden corrections to discovery.
Process noise and a move from ODEs to stochastic dynamics remain a separate future
scope decision.

The reusable implemented boundaries are documented in
[M31 feedback](PHASE_C_FITTING_FEEDBACK.md) and
[M32 staged diagnostics](PHASE_C_FITTING_STAGED_FEEDBACK.md). Proposed extensions
above do not change production fitting, permit data edits, or establish fresh
construction success. High NMSE alone still cannot determine whether the next
action should revise equations, question a claim, recalibrate a sensor or refit.

## Other paper directions retained from the research discussion

These are follow-on ideas, not concurrent assignments for Vega.

**Choosing the next scientific action.** Decide whether to refit, revise a claim,
ask a scientist, check an input log, recalibrate a sensor, obtain a repeat or new
measurement, run an intervention, or change the
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

- [Brynjarsdottir and O'Hagan, model discrepancy](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf)
  examines how physical parameters can be confounded with model inadequacy and
  how ignoring discrepancy can yield biased, overconfident inferences. Distinguish
  physical discrepancy from measurement error; flexible residual corrections
  alone do not establish either mechanism recovery or scientific novelty.
- [SciPy's robust least-squares documentation](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.least_squares.html)
  provides a mature baseline for reducing outlier influence. Robust optimization
  by itself neither localizes true data faults nor decides which scientific
  claims should be revised. Vega should extend the primary-source review to
  errors-in-variables identification, fault diagnosis and robust system discovery.
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
- `docs/PHASE_C_FITTING_M29_RESULTS_2026-10-12.md`
- `docs/PHASE_C_FITTING_M30_RESULTS_2026-10-12.md`
- `docs/PHASE_C_FITTING_FEEDBACK.md`
- `docs/PHASE_C_FITTING_M31_RESULTS_2026-10-13.md`
- `docs/PHASE_C_FITTING_STAGED_FEEDBACK.md`
- `docs/PHASE_C_FITTING_M32_RESULTS_2026-10-13.md`
- `docs/PHASE_C_FITTING_DIAGNOSTIC_SUPPLEMENT.md`
- `docs/PHASE_C_FITTING_MEASUREMENT.md`
- `docs/PHASE_C_FITTING_ROBUST_MEASUREMENT.md`
- `docs/PHASE_C_FITTING_SENSOR_DRIFT.md`
- `docs/PHASE_C_INTERACTION_FITTING_FEEDBACK.md`
- `docs/PHASE_C_MEASUREMENT_EVIDENCE_DESIGN.md`
- `docs/PHASE_C_NOISE_AWARE_FITTING_DESIGN.md`
- `docs/PHASE_C_SPEC_DATA_DIAGNOSTIC_STUDY.md`

Use these to identify reusable interfaces. Do not assume every historical
discussion or diagnostic has been promoted into the production method.

Deliver within the first work package:

1. A concise research design with a falsifiable central claim and a primary-source
   comparison against the closest prior work.
2. A typed claim/observation/evidence record proposal with supported, challenged,
   ambiguous, mixed-cause and computationally unresolved examples, preserving the
   three evidence levels and reusing training-only fitting evidence. Identify
   which fields exist today and which require future implementation.
3. The narrow diagnostic matrix, with matched clean/noisy/defective observations,
   repeat seeds, oracle controls, a provably confounded example, and a justified
   calibration or intervention that can resolve at least one ambiguous case.
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

The milestone succeeds when the proposed study could separate scientific,
measurement and computational explanations where evidence permits, recognize
mixed causes, and abstain when they are observationally indistinguishable. A plan
that equates high NMSE with a false specification, or treats improved fit after
discarding readings as proof of erroneous data, is not acceptable.

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
