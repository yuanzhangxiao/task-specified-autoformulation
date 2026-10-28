# Theory handoff for scientific premise revision in dynamical models

Prepared by Vega, 25 October 2026. Working formulation for a new ChatGPT 6
theoretical collaborator. Vega owns experimental design and implementation;
the GEMOT collaborators provide biological expertise and review. This document
is self-contained; linked repository documents provide additional history.

**28 October clarification:** the [current task contracts](VEGA_PREMISE_REVISION_TASK_SELECTION.md)
distinguish useful prediction, joint premise revision and constituent attribution.
Claim content and scientific relevance, not IDs, determine success. Requiring
the returned model to remain in the evidence-compatible set is an explicit output
option, not a universal definition of premise revision. The
[exploration applicability audit](VEGA_EXPLORATION_REGULARITY_AUDIT.md) records a
restricted baseline reduction and a qualification to a cited sufficient condition.
These refinements preserve the world/premise/evidence semantics below.

## 1 Assignment and intended contribution

We want a system that notices when a biological model rests on a missing,
mistaken or inapplicable premise, identifies what the evidence actually challenges,
and proposes a justified revision or an informative next observation. The primary
object of revision is a scientific premise, not an arbitrary equation or function
form. Examples include a constant extracellular stimulus, a representative-cell
approximation, negligible downstream binding load, rapid binding equilibrium,
and a fully reset physiological initial state.

The immediate assignment for the theory collaborator is to **critique and refine
the formulation below**, identify tractable guarantees and impossibility results,
and return a precise contract that Vega can implement and falsify experimentally.
Do not assume that the proposed definitions constitute a finished theory or a
novel contribution. The elementary confidence-set argument below is a baseline.

| Participant | Responsibility |
| --- | --- |
| ChatGPT 6 theory collaborator | Formal semantics, identifiability, decision rules, proofs/counterexamples, and explicit assumptions required by guarantees |
| Vega | Controlled fixtures, numerical evidence interfaces, reproducible implementation, matched baselines, experimental error/power/cost measurements, and counterexamples to theory |
| GEMOT biological collaborators | Review case realism, premise meaning, measurement interpretation, feasible interventions, and independently available evidence; specific contributions remain to be agreed |
| Shared decisions | Which biological question matters, what is held fixed, admissible revisions, meaningful effect sizes, and acceptance criteria |

GEMOT is an intended collaborator and potential shared modeling platform. The
project is not framed as a competition with GEMOT. Its authors' expertise should
shape the cases before the biological benchmark is frozen. The user will share
this file with the new agent; this document does not create or message another
chat, initiate experiments, or commit the biological collaborators to a protocol.

## 2 Existing foundation and the gap

Autoformalism constructs dynamical models from scientific task descriptions.
Construction includes variables, processes, dependencies, function forms and
initialization. Orion's fitting work now supplies more than parameter estimates:
independent rollout checks, search status, local sensitivity, nuisance-aware
profile support, and measured response mismatches with conditional feedback.

Relevant evidence already available in the repository:

- M37/M38 produce verified training-response summaries, competing investigations
  and testable predictions. Equation references are places to investigate, not
  diagnoses of faulty terms.
- M39-M42 qualify bounded profile calculations, including nonlinear nuisance
  parameters. Good output predictions and full local rank can coexist with broad
  physical-parameter support. These support sets have no calibrated confidence
  percentage; incomplete calculations remain unresolved.
- M46/M47 distinguish prediction from hidden-initial-state recovery. Identifiable
  combinations can support useful predictions without identifying every state.
  Their conclusions remain conditional on the control's observation and fitted-
  parameter assumptions.
- Construction/fitting integration has advanced beyond the earlier October 23
  story document. Current saved-construction follow-ups include numerical recovery
  and exact candidate-specific necessary-condition checks. On October 25, two
  starts substantially improved the same retained candidate without equation
  changes; its full-coordinate diagnostic requalification remains pending.

The last point is a warning against promoting poor fit into premise rejection.
Numerical verification, optimization adequacy, candidate-family incompatibility
and scientific premise incompatibility are different claims. None of the current
work establishes automatic, general biological-premise diagnosis.

Sources: [feedback](PHASE_C_FITTING_FEEDBACK_DRAFTS.md),
[nonlinear uncertainty](PHASE_C_FITTING_NONLINEAR_PROFILE_SUPPLEMENT.md),
[hidden-state reliability](PHASE_C_FITTING_PREFIX_NOISE_RELIABILITY.md),
[paired construction audit](PHASE_C_PAIRED_CONSTRUCTION_NUMERICAL_HANDOFF.md), and
[October 25 numerical follow-up](PHASE_C_CONSTRUCTION_ENDPOINT_REQUALIFICATION.md).

## 3 The basic objects

Let an experiment specification be

\[
a_e=(v_e,\mathcal T_e,\mathcal M_e,c_e),
\]

where \(v_e\) is the commanded intervention, \(\mathcal T_e\) the sampling
schedule, \(\mathcal M_e\) the measurement channels/procedure, and \(c_e\)
recorded context such as medium volume or preparation history. Commanded input
and actual delivered input are different objects.

Let \(D=(a_{1:E},Y_{1:E})\) contain immutable protocol records and raw readings.
Fix the scientific prediction target, required evaluation domain and information
access rules in a task contract \(G\). A revision cannot make its error disappear
by changing the target or silently discarding difficult observations.

A **world** is an idealized generative configuration sufficient to predict raw
observations under the admitted experiments:

\[
w=(m,\theta,u,\rho,\Pi,h,\nu,\lambda)\in\Omega_H.
\]

| Component | Meaning |
| --- | --- |
| \(m\) | Mechanisms and their composition, drawn from a reviewed bounded library |
| \(\theta\) | Physical coefficients and other continuous quantities with explicit domains |
| \(u\) | Actual input/delivery or environmental dynamics; a constrained model, not a free correction at each sample |
| \(\rho\) | Initial state or preparation-conditioned initial distribution |
| \(\Pi\) | Population heterogeneity, including admitted differences in states, parameters or response timing |
| \(h\) | Observation map and aggregation, including calibration and measurement weights |
| \(\nu\) | Observation noise and other explicitly admitted acquisition nuisances |
| \(\lambda\) | Parameter-sharing and invariance rules across cells, runs and interventions |

The envelope \(\Omega_H\) enforces the reviewed component library, domains and
hard anchors \(H\). It may be a union of finitely many structural branches with
bounded continuous coordinates. Coordinates and dimensions may differ between
branches. A world is not a claim to represent every aspect of biological reality.

Each world induces a law \(P_w^a\) for the raw observations in experiment \(a\).
For example, a single-cell deterministic component could be

\[
\dot x_e=f_m(x_e,u_e;\theta),\qquad
x_e(0)\sim\rho_e,\qquad
Y_{e,k}\sim\mathcal N(h_e(x_e(t_k)),\Sigma_e).
\]

This illustration does not assume that every case has Gaussian noise or a single
cell. The chosen observation law must match the admitted setting.

Hard anchors are **conditioning assumptions**, not infallible physical truths.
An uncertain calibration or a noisy initial reading belongs in the inferential
model, not among exact anchors. A reading cannot simultaneously be treated as an
exact initial state and an independent noisy observation of that state. Protocol
records remain immutable even when actual delivery is uncertain.

The first milestone should fix the mechanism library and elementary rate-law
families. This prevents arbitrary equation search from obscuring the question of
premise revision. It does not imply that the library contains the true biology.

## 4 Premises need meaning, scope and provenance

A premise is more than a sentence attached to an equation. Proposed record:

| Field | Required content |
| --- | --- |
| Identity | Stable ID, version, original wording and source reference |
| Origin | Scientist assertion, protocol report, model-builder default, or explicit approximation; record missing information separately |
| Kind | Physical mechanism, input/environment, preparation, population, observation, cross-condition sharing, or approximation |
| Scope | Applicable experiments, times, interventions, targets and operating conditions |
| Semantics | Reviewed predicate or a specified approximation criterion |
| Bindings | Component IDs, ports, parameters, initializers and observations affected |
| Dependencies | Required definitions, logical implications, and assumptions used in its interpretation |
| Revision policy | Permitted alternatives, tolerance changes or scope qualifications and their costs |
| Evidence | Data/calibration references, numerical status and the exact families examined |

For a well-defined applicable query, write

\[
p_j(w;s_j)\in\{0,1\},\qquad
\Omega_j=\{w\in\Omega_H:p_j(w;s_j)=1\}.
\]

Here \(s_j\) fixes the scope and interpretation. Unknown truth is an epistemic
state: both values may remain possible. It is not a third physical mechanism.
An inapplicable or undefined premise needs a separate status; it must not be
silently counted as false. The simple results below assume every reported
predicate is well defined on the entire relevant envelope. Guards depending on
unknown world variables require a more explicit conditional-claim treatment.

The registry also distinguishes **asserted**, **assumed by default**, and
**not specified**. Missing experimental information does not mean the scientist
asserted its negation. Asking for a protocol detail can be the correct next step.
A statistical rejection of a constructor's default must not be attributed to
the scientist as a false biological belief.

### Exact premises and approximation premises

An exact premise can constrain an input, initial state or coefficient, for example
\(u_e(t)=u_{e0}\), \(N_{D,e}(0)=0\), or specified shared parameter values.

An approximation premise instead states that a reduction preserves a specified
quantity closely enough. Given a reviewed reduction \(\mathcal R_j\), target
\(q\), experiment class \(\mathcal A_{s_j}\), distance \(d_j\) and tolerance
\(\varepsilon_j\), one possible definition is

\[
p_j(w;s_j,\varepsilon_j)=
\mathbf 1\!\left[
\sup_{a\in\mathcal A_{s_j}}
d_j\!\left(q_w^a,q_{\mathcal R_j(w)}^a\right)
\le\varepsilon_j
\right].
\]

Specify whether \(q\) means a physical trajectory, expected measured output,
probability distribution or decision. Define the state projection and initial
preparation used by the reduction so the two quantities are comparable.
“Rapid equilibrium” should usually be interpreted this way, rather than as the
literal physical assertion that a rate is infinite.

Approximation validity is target- and protocol-dependent. A model can predict a
mean adequately while misrepresenting cell-level dynamics. Known binding sites
can exert a negligible effect; “negligible load” does not mean those sites do not
exist. The reference mechanism and reduction themselves require review. Without
them, this error criterion is not operational merely because a formula is written.

Tolerance and scope are fixed before the relevant evaluation. Restricting a
claim's applicability may be an honest conclusion, but leaves the original task
unsolved outside that scope. It cannot silently alter \(G\).

## 5 Link premises to components and executable models

Maintain a typed dependency graph with nodes for protocol facts, premises,
components, compilation steps and evidence. Useful edge types include
`constrains`, `uses_definition`, `implies`, `compiled_into`, and `tested_by`.
This is a semantic/provenance graph, not automatically a causal graph.

A compiler maps a reviewed configuration to executable dynamics, observation and
initialization code. It should retain a trace from each changed premise to the
affected components and generated equations. Distinguish:

1. Exact equivalent implementations, with a justified map back to the same world.
2. Reduced implementations, with explicit semantic approximation error.
3. Unsupported or unverified translations, which cannot support stronger claims.

Solver error is separate from reduction error. Independent integrator agreement
does not establish that an approximation is scientifically adequate. A changed
coordinate system must preserve domains and sharing constraints; removing a
factorization while enlarging its feasible range changes the scientific family.

Premise implications should be retained. Counting synonymous or logically
duplicated sentences as independent faults makes revision cost depend on wording.
The theory should specify how semantically equivalent premise descriptions are
handled before claiming “minimal revision.”

An illustrative, nonimplemented record for the first story is:

```yaml
premise_id: input.bioactive_ligand_constant
origin: model_builder_default
source_statement: "TGF-beta was added at the start"
missing_information: "Subsequent ligand bioavailability was not reported"
kind: input_environment
bindings: [environment.ligand, intracellular_model.stimulus_port]
scope:
  experiments: declared_bolus_runs
  time_interval: declared_observation_window
semantics:
  predicate_id: relative_ligand_change_at_most
  arguments: [initial_ligand, declared_tolerance, declared_window]
allowed_revisions:
  - measured_ligand_with_uncertainty
  - reviewed_finite_pool_model
preserve: [raw_readings, target_channels, intracellular_rate_law_library]
status: not_assessed
```

Names such as `declared_tolerance` refer to frozen configuration values, not
values chosen after examining residuals. The record is a proposed contract,
not a claim that this schema exists in the repository. Natural-language-to-
predicate interpretation must be reviewed separately from inference correctness.

## 6 Revision and diagnosis are different operations

For a baseline set \(B_0\) of asserted/default premises, define

\[
\Omega_B=\{w\in\Omega_H:p_j(w)=1\text{ for every }j\in B\}.
\]

The inference envelope must include reviewed alternatives beyond
\(\Omega_{B_0}\); otherwise the method has assumed the very premises it seeks
to test. Allowed revisions may remove a restriction, substitute a reviewed
alternative, relax a declared tolerance, qualify scope, or elicit missing context.

Removing \(p_j\) permits both its truth values; it does **not** assert
\(\neg p_j\). A replacement is a separate declared constraint. Two labeled
branches may overlap: a heterogeneous family permitting zero variance includes
homogeneous worlds. Physical premise truth must be evaluated on the world, not
read off a branch name or a fitted best-model label.

Only pure removal of restrictions is automatically a nested relaxation.
Changing reductions, initial maps or observation interpretations need not create
nested families. Compare their induced observation laws and preserve raw-data
likelihood normalization; transformed-data losses are not automatically comparable.

For each world define its violated baseline premises

\[
V(w)=\{j\in B_0:p_j(w)=0\}.
\]

These sets admit simultaneous violations. There is no mutually exclusive label
“wrong biology versus wrong data.” A numerical failure is a separate evidence
status that can coexist with either or both.

A preferred repair can minimize a declared semantic cost among compatible
revisions. This expresses a preference, not proof of the physical cause. Keep
repair ranking and claims forced by the evidence in separate outputs. In
particular, selecting one minimum-cost repair does not justify its accusations.
Even all inclusion-minimal violation patterns omit nonminimal mixed-cause worlds;
for a finite exact family they do preserve the common intersection, so this
compression is not inherently unsound for that narrower question.

## 7 Evidence semantics with incomplete computation

Let \(C(D)\subseteq\Omega_H\) be a specified scientific compatibility or
confidence set. The next section provides one deliberately simple construction.
Maintain two computational objects:

\[
I(D)\subseteq C(D)\subseteq O(D)\subseteq\Omega_H,
\]

where \(I\) contains verified feasible witnesses and \(O\) conservatively retains
everything not soundly excluded. These need not be explicitly enumerated sets;
they can be represented by parameter regions, certificates and unresolved branches.

| Evidence | Permitted conclusion |
| --- | --- |
| \(I\cap\Omega_j\ne\varnothing\) | A compatible witness preserves premise \(j\); this does not establish its physical truth |
| \(O\cap\Omega_j\ne\varnothing\), without a witness | Premise \(j\) is not excluded; compatibility remains computationally unresolved |
| \(O\ne\varnothing\) and \(O\cap\Omega_j=\varnothing\) | Premise \(j\) is rejected within the envelope and observation assumptions |
| Witnesses with both \(p_j=0\) and \(p_j=1\) | Demonstrated ambiguity among admitted explanations |
| Both truth values remain only in \(O\) | Unresolved alternatives; do not claim that both have verified fits |
| \(I\ne\varnothing\) and \(O\subseteq\Omega_j\) | Evidence supports \(j\) within scope, without a universal-truth claim |
| \(O=\varnothing\) | Collective rejection of the admitted family/contract; suppress vacuous individual accusations |
| \(I=\varnothing\), \(O\ne\varnothing\) | Overall compatibility is computationally unresolved |

A **joint conflict** \(S\subseteq B_0\) satisfies

\[
O\cap\bigcap_{j\in S}\Omega_j=\varnothing.
\]

When \(O\ne\varnothing\), this says the premises in \(S\) cannot all hold
within the retained possibilities. It need not identify any one false premise.
All other revisable premises are free in this definition. A check that instead
holds other premises fixed must state that additional conditioning.

Logical inconsistency, \(\Omega_B=\varnothing\) before data, differs from this
data-dependent incompatibility. A failed candidate or a local optimizer's poor
endpoint establishes neither. To reject a premise, the exclusion must cover every
admitted construction preserving it, not merely the candidates an LLM happened
to propose. Unexplored regions remain in \(O\).

The ideal diagnosis set is \(\mathcal D(D)=\{V(w):w\in C(D)\}\). When \(C(D)\)
is nonempty, intersecting the sets in \(\mathcal D(D)\) gives forced violations;
their union gives possible violations. Inner and outer computational analogues
should preserve the distinction between demonstrated
and merely unresolved patterns. A preferred repair is an additional decision.

## 8 An elementary statistical baseline and its limits

Start with a fixed, nonadaptive design, \(n\) raw observations, exact mean
predictions and a known positive-definite covariance \(\Sigma\):

\[
Y\sim\mathcal N(\mu(w),\Sigma),\qquad
Q_w(Y)=(Y-\mu(w))^\top\Sigma^{-1}(Y-\mu(w)),
\]

\[
C_\alpha(Y)=\{w\in\Omega_H:Q_w(Y)\le\chi^2_{n,1-\alpha}\}.
\]

If the true world \(w_\star\) belongs to \(\Omega_H\), then
\(\Pr_{w_\star}(w_\star\in C_\alpha)=1-\alpha\). The degrees of freedom are
**\(n\), not \(n\) minus the number of fitted parameters**: this is test
inversion at each specified mean, not a fitted-residual heuristic.

With sound outer containment, report

\[
R(Y)=\{j:O(Y)\ne\varnothing,\ O(Y)\cap\Omega_j=\varnothing\}.
\]

Then the elementary simultaneous guarantee is

\[
\Pr_{w_\star}\!\left[
\exists j\in R(Y):p_j(w_\star)=1
\right]\le\alpha.
\]

Proof: whenever \(w_\star\in C_\alpha\subseteq O\), a true premise has the
retained witness \(w_\star\) and cannot be rejected. Any false accusation
therefore implies failure of whole-world coverage. Projecting the same valid
set onto multiple premises does not require a separate multiplicity penalty.
This is not a posterior probability that a premise is true and is not claimed
as a novel theorem. Its usefulness depends on power and computational cost.

For a parameter region \(A\), exclusion needs a valid lower bound

\[
L_A\le\inf_{w\in A}Q_w,
\qquad L_A>\chi^2_{n,1-\alpha}.
\]

A local optimum is an upper bound on the infimum and cannot provide this
certificate. A validated objective upper bound at one world below the threshold
can establish an inner witness. Finite grid checks do not cover continuous gaps
unless accompanied by a valid bound. If means are numerical enclosures, bounds
must contain the exact objective; merely finding a favorable value somewhere
inside an enclosure is insufficient to certify a witness.

The guarantee assumes correct observation semantics, true-world inclusion,
sound compilation and valid numerical bounds. Estimated noise, selected masks,
unknown inputs treated as exact, unchecked reduction error, or unvalidated
data-dependent catalog pruning invalidate this particular argument. Those extensions may be
tractable, but need their own valid construction. Current objective-rise profiles
are not automatically this confidence set. Sequential acquisition and repeated
stopping also require a separate validity argument.

## 9 Identifiability and the next action

For a declared class of fixed experiments \(\mathcal A\), define

\[
w\equiv_{\mathcal A}w'
\quad\Longleftrightarrow\quad
P_w^a=P_{w'}^a\quad\text{for every }a\in\mathcal A.
\]

A premise is structurally identifiable on this envelope only if its truth is
constant on each relevant equivalence class. If two equivalent worlds disagree
on the premise, an observation-based procedure cannot distinguish them. In
particular, a level-\(\alpha\) rejection test at the true-premise world has
power at most \(\alpha\) at its observation-equivalent false-premise counterpart.

For adaptive, dependent experiments, compare the joint observation laws under
every admissible policy instead. Equality of isolated marginal measurements is
not sufficient. An action that produces different laws makes discrimination
possible in principle, but does not establish finite-sample power. Near-equivalence,
nuisance uncertainty and arbitrarily small premise violations require effect-size
and sample-size conditions.

The action menu may include asking for a protocol fact, measuring an input,
calibrating a sensor, observing single cells, changing an intervention, or computing
a stronger bound. More computation can shrink \(O\) toward \(C\); it cannot
break observational equivalence. A biological action must be evaluated against
all relevant surviving alternatives, not only two favored fitted candidates.

For the first experiment, use a fixed follow-up schedule or separately held-out
confirmatory observations. A later theory may choose actions by expected or
worst-case reduction in decision error per cost. Expected information gain needs
a probability model over worlds; a possibility set alone does not supply one.
Neither an action rule nor sequential error control is settled here.

## 10 Formal mappings of the five biological stories

The biological precedents motivate these abstractions; they do not validate every
model below. Variables in each local reaction system are defined in its paragraph.
All comparisons need fixed targets, explicit shared parameters and independent
evidence for disputed inputs or preparations.

### TGF-beta and the constant-stimulus premise

Fix intracellular dynamics and readout:

\[
\dot x_e=f(x_e,L_e;\theta),\qquad \mu_e(t)=h(x_e(t)).
\]

The exact premise is \(L_e(t)=L_{e0}\) on the specified window. A practical
version bounds relative change. Revise the environmental component using an
independently measured ligand trajectory with uncertainty or a reviewed
finite-dimensional delivery/depletion model. Preserve \(f,h\) and their local
rate-law families. Initial dose alone does not establish a clamped input.
Independent ligand bioactivity measurements can distinguish this explanation
from some intracellular alternatives; arbitrary timewise fitted inputs cannot.

### NF-kappaB and population representation

Let \(z\) index initial states, admitted kinetic variation and timing. Every
cell follows the same network family, producing \(x_e^z(t)\). A measured mean is

\[
\mu_e(t)=
\frac{\int \omega_e(z)h(x_e^z(t))\,d\Pi_e(z)}
     {\int \omega_e(z)\,d\Pi_e(z)},
\]

with nonnegative declared weights and a positive denominator. Physical population
homogeneity, \(\Pi_e=\delta_{z_e}\), is stronger than adequacy of a representative
mean trajectory. Identical coefficients, identical initials and synchronized
response timing should be separate premises when the question requires it.
Revise only reviewed low-dimensional heterogeneity and aggregation components.
Longitudinal cell observations constrain them. A fixed initial phase spread
alone need not produce progressively increasing dephasing; justified period
variation or evolving timing differences may matter. The input treatment under
each timing hypothesis must be explicit.

### Regulatory DNA and negligible downstream load

A schematic fixed chemistry for free regulator \(R\), bound regulator \(B\)
and fixed binding-site abundance \(S_T\) is

\[
\begin{aligned}
\dot R&=p(u)-\delta_fR-k_{\rm on}R(S_T-B)+k_{\rm off}B,\\
\dot B&=k_{\rm on}R(S_T-B)-(k_{\rm off}+\delta_b)B.
\end{aligned}
\]

Production \(p\) is fixed and nonnegative, rates are nonnegative, and
\(R\ge0\), \(0\le B\le S_T\). Degradation here releases binding sites.
The premise usually concerns the accuracy of dropping the load coupling for a
specified output. If the construct is known to carry sites, do not replace that
fact with \(S_T=0\). Compare the no-load reduction with the full declared chemistry
at an explicit tolerance. Matched binding-defective constructs, copy-number
information and upstream measurements help distinguish binding load from general
expression burden. Free and total regulator are different observation targets.

### Drug binding and rapid equilibrium

For a deliberately simplified fixed-total-target control, let \(b\) be occupancy:

\[
\dot b=k_{\rm on}D_{\rm free}(t)(1-b)-k_{\rm off}b,
\qquad
b_{\rm eq}(t)=\frac{D_{\rm free}(t)}{K_d+D_{\rm free}(t)},
\quad K_d=k_{\rm off}/k_{\rm on}.
\]

Take \(k_{\rm on},k_{\rm off}>0\), \(D_{\rm free}\ge0\) and \(b(0)\in[0,1]\).
The premise is adequacy of replacing \(b\) by \(b_{\rm eq}\), not merely a
statement about equilibrium affinity. It depends on initial occupancy, protocol
timescales, targets and tolerance. With genuinely zero free drug, the kinetic
model gives \(b(t)=b(0)e^{-k_{\rm off}t}\). Extracellular washout does not establish
zero free intracellular drug. Independent exposure and occupancy measurements
matter. This is not a complete lapatinib model; conformational mechanisms, if used,
must be specified before comparing a reduction with its unreduced realization.

### Bacterial preparation and initial physiological composition

A fixed illustrative system for active cells \(N_A\) and non-growing cells
\(N_D\) is

\[
\dot N_A=rN_A+\omega N_D-k_AuN_A,\qquad
\dot N_D=-\omega N_D-k_DuN_D,\qquad \mu=N_A+N_D.
\]

Rates and exposure are nonnegative. The disputed initial constraint is
\(N_D(0)=0\), or a prescribed equilibrium fraction. A reviewed alternative is
\(N_D(0)=pN_0\), \(N_A(0)=(1-p)N_0\), with \(N_0>0\) and \(p\in[0,1]\)
linked to preparation history and independently constrained. Keep the compartments
and kinetics fixed in the strict control. If \(r=0\) and
\(k_A=k_D=k\), then \(\dot\mu=-ku\mu\), independent of initial composition
and awakening. Total counts cannot identify \(p\) in that control. Persistence
is not universally pre-existing dormancy, and the two-state biology requires review.

Biological sources and the proposed empirical checks are collected in the
[five-story discussion](../work/vega/fallible-specifications/BIOLOGICAL_ASSUMPTION_STORIES_2026-10-23.md).
Representative primary sources are
[Clarke et al. on TGF-beta](https://pmc.ncbi.nlm.nih.gov/articles/PMC2668365/),
[Zambrano et al. on NF-kappaB](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0090104),
[Jayanthi et al. on downstream load](https://web.mit.edu/ddv/www/papers/ACS2013.pdf),
[Wood et al. on drug binding](https://pubmed.ncbi.nlm.nih.gov/15374980/), and
[Jõers et al. on preparation history](https://pubmed.ncbi.nlm.nih.gov/20435730/).

## 11 A tractable first fixture with an exact ambiguity

Consider a signaling-inspired diagnostic control:

\[
\dot L=-\kappa L,\qquad
\dot R=-\gamma R,\qquad
\dot x=aLR-bx,
\]

where \(L\) is available stimulus and \(R\) effective response capacity.
Initially take \(L_0,R_0,a,b>0\) known, \(x_0=0\), and bounded nonnegative
\(\kappa,\gamma\). Observe \(x\) with fixed known Gaussian noise. These equations
are a methodological control, not a validated TGF-beta mechanism.

The original premise bundle is

\[
P_L:\kappa=0\quad\text{and}\quad P_R:\gamma=0.
\]

Writing \(s=\kappa+\gamma\), the mean response is

\[
x(t)=aL_0R_0\frac{e^{-st}-e^{-bt}}{b-s},
\]

with continuous limit \(aL_0R_0te^{-bt}\) when \(b=s\). Thus observations of
\(x\) under this bolus protocol depend on the sum of the two decay rates.
For \(b=1\), the worlds

\[
(\kappa,\gamma)=(0.2,0),\quad(0,0.2),\quad(0.1,0.1)
\]

have identical output means and, under the same noise law, identical observation
distributions. Their violated-premise sets are respectively
\(\{P_L\}\), \(\{P_R\}\), and \(\{P_L,P_R\}\).

With adequate data, the constant-stimulus/constant-capacity conjunction can be
challenged within this family. Neither individual premise is thereby identified
as false. A single preferred fit or minimum-cardinality repair can conceal this
exact ambiguity. More precise fitting or changing only bolus amplitude cannot
separate these worlds under the stated protocol.

An independent measurement of \(L(t)=L_0e^{-\kappa t}\) at positive time identifies
\(\kappa\) in the ideal noiseless control. With \(a,b,L_0,R_0\) known, the
output then identifies the summed rate and hence \(\gamma\). Noisy observations
give uncertain estimates rather than exact recovery. Replenishment is another
possible action but changes the experiment class; it requires a separately defined
input protocol and analysis. If \(b\) is unknown, the interchange \(b\leftrightarrow s\)
can create ambiguity even with known response scale. Additional unknowns must be
included deliberately.

Approximate constancy is also explicit: on \([0,T]\), with \(T,L_0>0\) and
\(0\le\varepsilon_L<1\),

\[
\sup_t\frac{|L(t)-L_0|}{L_0}=1-e^{-\kappa T}\le\varepsilon_L.
\]

This is a different predicate from \(\kappa=0\). It defines a scientifically
meaningful tolerance and makes the operating window part of the claim. It does
not guarantee uniform detection power arbitrarily close to the tolerance boundary.

Proposed fixture conditions are healthy, stimulus-only change, capacity-only
change, both changes, weak/noisy ambiguity, and informative added measurement.
Add separate numerical-failure and out-of-envelope controls. The latter cannot
be guaranteed detectable if observationally equivalent to an admitted world.
Use a declared finite rate catalog initially for exact enumeration; do not call
it a certification of a continuous rate domain. A later continuous version needs
validated bounds, not denser grids alone.

## 12 Experimental contract for Vega

The theory should produce something that can be tested at a small scale before
an LLM is involved. Proposed sequence:

1. **Semantics and analytic controls.** Implement reviewed predicates, component
   bindings and exact mean formulas; verify the ambiguity and its resolving action.
2. **Statistical attribution.** Compare set-based attribution with fixed-premise
   fitting, best-fit/ranked repair, and conventional joint diagnosis using exactly
   the same world catalog, observations, allowed actions and budgets.
3. **Incomplete computation and nuisance uncertainty.** Deliberately withhold
   certificates, use limited searches, add independently bounded calibration or
   initial-state uncertainty, and verify that unavailable evidence increases
   abstention rather than false accusations.
4. **Approximation premises.** Add a reviewed kinetic/reduced pair, with explicit
   target, tolerance and protocol scope. Qualify semantic and numerical error
   separately before claiming an approximation-validity guarantee.
5. **Biological and language stages.** After collaborator review, instantiate a
   biological case. Only then add natural-language premise extraction/proposal as
   a separately measured error source. LLM proposals do not certify predicates.

Measure simultaneous false-premise rejection, joint-conflict correctness,
identifiable violation detection, retained mixed-cause explanations, demonstrated
versus computational ambiguity, and success/cost of a follow-up action. Define
denominators using independent experiments or noise acquisitions, not optimizer
starts. An always-abstain baseline must fail usefulness/power requirements.
Separate raw-observation prediction from hidden-state recovery in synthetic controls.

Every experiment needs a frozen plan, immutable raw data, explicit split and
action access, deterministic resume, and complete fitting/diagnostic costs.
Synthetic truth and injected labels are evaluator-only. Development results may
select a policy; locked final observations and test metrics cannot enter proposal
generation. No benchmark data or finalized prompts are edited. Every eventual
LLM call must be cached and logged, with schema-validated restricted output and
no execution of generated Python or unrestricted expressions.

This is an experiment proposal, not a new campaign or an implemented inference
layer. Preserve the fitter's `specification_verdict=not_assessed`; a separate,
qualified premise-inference layer would own scoped conclusions.

## 13 Theory questions that need decisions

Please address these in priority order:

1. **Semantic sufficiency.** Is the world/predicate/component formulation sufficient
   for all five stories? Where do guarded premises, population laws, reductions or
   protocol uncertainty require different objects?
2. **Attribution target.** Should the principal output be forced premise violations,
   joint conflicts, a set of repairs, or task-dependent decisions? Define each and
   show where they differ on the analytic fixture.
3. **Soundness under incomplete computation.** What certificate interface preserves
   \(I\subseteq C\subseteq O\) across branches, approximation bounds and solver
   errors? Which portions can realistically be certified in a bounded first family?
4. **Identifiability and useful power.** Characterize premise equivalence classes,
   necessary additional measurements, effect-size margins and sample requirements.
   Explain why parameter rank alone is insufficient.
5. **Approximate premises.** How should validity of a reduction be inferred when
   the detailed state is latent? What reference-model and uniform-error conditions
   are actually needed? Can different premises compose without untracked error?
6. **Action and sequential validity.** Under which acquisition policies can an
   error guarantee be maintained? Distinguish information gained by measurement
   from uncertainty removed by computation. State any prior needed for utility.
7. **Representation and revision costs.** What invariance can we require under
   reparameterization, equivalent compilation, or splitting one premise into
   synonymous statements? Which minimality notion has scientific meaning?
8. **Novelty.** Which proposed result goes beyond established diagnosis, model
   invalidation, test inversion and experiment design? If none yet does, specify
   what additional difficulty could support a substantive theoretical contribution.

Do not assume that a universal solution is required for the first paper. A precise
limited result with meaningful power and a biologically consequential example is
preferable to a broad guarantee resting on an unavailable consistency oracle.

## 14 Requested return from the theory collaborator

Please return one Markdown response that Vega can translate into a finite experiment:

1. A revised notation table and formal problem statement, distinguishing accepted
   definitions from unresolved choices.
2. At most three candidate propositions/theorems with exact hypotheses, proof
   sketches, novelty assessment and counterexamples when hypotheses are removed.
3. A complete analysis of the two-premise analytic fixture, including a case where
   attribution must remain unresolved and the added observation that changes it.
4. A minimal evidence contract: required inputs, outputs, certificates and statuses.
   Use explicit quantities and tolerances rather than phrases such as “high confidence.”
5. A smallest implementable experiment and matched baselines, with acceptance
   criteria and failure cases. Identify any guarantee that depends on an oracle.
6. Questions requiring biological review, separated from mathematical choices.

Vega will respond with numerical counterexamples, feasibility/cost evidence and
measured error/power results. Changes to the formulation should be versioned so
we can tell whether an experiment tests the same theorem and premise semantics.

## 15 Literature positioning and supporting documents

Assumption revision, uncertain-observation diagnosis, minimal repairs, model
invalidation and discriminating experiments have substantial prior art. Relevant
starting points include [Reiter's diagnosis framework](https://www.cs.ru.nl/P.Lucas/teaching/KeR/reiter.pdf),
[Cazes and Kalech on uncertain observations](https://ojs.aaai.org/index.php/AAAI/article/view/5664),
[biological model invalidation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2704209/),
and [AI-Hilbert](https://www.nature.com/articles/s41467-024-50074-w).
The novelty question concerns what new semantic, computational or statistical
result this project can establish, not whether an agent can suggest revising an
assumption. The confidence-set projection argument above is elementary.

Additional context, not prerequisites for reading this file:

- [Literature survey and GEMOT collaboration plan](../work/vega/fallible-specifications/LITERATURE_SURVEY_2026-10-13.md).
- [Biological stories and source-specific caveats](../work/vega/fallible-specifications/BIOLOGICAL_ASSUMPTION_STORIES_2026-10-23.md).
- [Vega research handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md).
- [Current project context](PROJECT_CONTEXT.md), whose recent entries supersede
  older milestone snapshots when their implementation status differs.
- [Pipeline contracts](PIPELINE_DESIGN.md) and
  [measurement-evidence design](PHASE_C_MEASUREMENT_EVIDENCE_DESIGN.md).

All mathematical guarantees in this draft are conditional on their stated
envelope, semantics and observation assumptions. Biological validity, general
coverage under unknown noise, adaptive inference and practical certification
for unrestricted constructions remain open work.
