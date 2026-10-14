# Literature survey of fallible scientific specifications and GEMOT collaboration

Prepared by Vega, 13 October 2026, Hawaii; revised 14 October to reflect the
intended collaboration with GEMOT's authors. Research assessment and proposed
next milestone; no implementation or experimental result is claimed.

## Assessment

The project is worth pursuing, but the broad idea is not new. Scientific belief
revision, diagnosis with uncertain observations, multiple simultaneous faults,
model invalidation, robust equation discovery, and discriminating experiments all
have substantial prior art. Recent agentic systems also combine several of these
ingredients. A claim such as “the first AI scientist that revises assumptions when
data disagree” would be untenable.

The promising contribution is a **validated procedure for deciding which
scientific conclusions are warranted when claims, observation contracts,
candidate equations, and computation can all fail**. Its output should preserve
competing explanations and distinguish a supported revision from a convenient
improvement in fit. This is a proposed contribution, not a confirmed novelty claim:
the surveyed work does not establish this entire capability under the handoff's
combined evaluation conditions, but that does not prove that no prior system does.

The intended collaboration with GEMOT's authors gives this project a concrete
scientific setting. Their biological expertise can guide realistic scenarios,
measurement assumptions and feasible experiments; Vega's primary responsibility
is the diagnostic methodology and its evaluation. GEMOT provides relevant prior
work and a potential shared modeling platform. A paired study within that platform
would measure the value of the added methodology, not rank the two teams.

The novelty assessment still requires comparison with Model Discovery Agent,
EvoSCM, AI-Hilbert, POPPER, statistical model criticism, and classical
diagnosis/model invalidation. Biological relevance strengthens the research
question and evaluation; it does not by itself establish a new algorithm.

## Scope and source coverage

This is a targeted primary-source survey through 13 October 2026, with citation
following across four areas: mechanistic discovery, logical diagnosis and repair,
statistical identification under imperfect observations, and active discrimination.
It is not a systematic review with an exhaustive database inclusion protocol.
Publication dates and versions are recorded where recent revisions materially
affect the comparison. Preprint results remain authors' reported results.

The assessment uses the [Vega handoff](../../../docs/VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md),
[Phase C entry point](../../../docs/PHASE_C_START_HERE.md), relevant scientific
objectives and invariants in the project/pipeline documents, the
[M32 review](../../../docs/PHASE_C_FITTING_M32_RESULTS_2026-10-13.md), the three
October 13 preparation designs, the public basin prompt, and relevant existing
feedback tests. It does not promote historical diagnostics into implemented
noise-aware fitting or assume that M33's pending remote qualification succeeded.

GEMOT's main text and Appendices A and B were read directly in the browser; the
ordinary web fetch returned an access error. Other major comparisons below use
primary full-text methods/results and relevant appendices. “Full-text review”
means those relevant portions were inspected, not that every listing was audited.
Abstract-only or limited-access sources are identified. No external implementation
was executed, no source repository was audited, and no benchmark test data were
opened. Absence statements concern demonstrated evidence in the inspected paper.

## GEMOT as a foundation for joint research

[de Pomereu, Snelling and Fröhlich, GEMOT](https://www.biorxiv.org/content/10.64898/2026.09.26.754619v1.full)
is a September 28, 2026 preprint. It combines agent judgment, Antimony model
representation, an MCP boundary, and AMICI/pyPESTO/Fides evaluation. Biological
mechanisms and readout mappings are revised; supplied observation/noise templates
remain fixed, while eligible calibration/noise parameters are fitted. Training
diagnostics include residual structure, local parameter geometry, and numerical
failures. These checks explicitly do not certify scientific adequacy.

Across 18 problems and 90 principal runs, 78 models meet its reference-relative
BIC criterion, whereas 55 improve training likelihood. Five systems have held-out
evaluations, with differing protocols: Fujita fixes fitted parameters; several
others assess transfer with refitting. Predictive success therefore is not uniform
evidence of mechanism identification. The discussion acknowledges underdetermination
and leaves experimental design unimplemented. Code is withheld until publication.
These details are supported by Sections 2–3, Appendices A.3–A.4 and B.7, and the
reproducibility statement.

The proposed joint extension is **adjudicating explicitly fallible contracts**
in biologically meaningful settings. Existing capabilities should be credited
and reused; new work should establish when a revision or follow-up experiment
is supported, and measure errors in those decisions.

## Closest scientific discovery systems

| Primary work and review scope | Established overlap | Consequence for Vega |
| --- | --- | --- |
| **AI-Hilbert**, Cory-Wright et al., Nature Communications, 2024. [Paper](https://www.nature.com/articles/s41467-024-50074-w). Full-text inconsistent-background and overall-formulation sections. | Combines data with incomplete or inconsistent polynomial background knowledge; selects axiom subsets and uses sum-of-squares derivability certificates. | Assumption revision and formal reasoning are already central contributions. A discarded axiom need not be physically false. Compare explicit claim-family compatibility under competing measurement explanations, rather than merely deriving a better-fitting law from fewer axioms. |
| **AI-Descartes**, Cornelio et al., Nature Communications, 2023. [Paper](https://www.nature.com/articles/s41467-023-37236-y), [full text](https://arxiv.org/html/2109.01634). Theory comparison and approximation examples. | Links symbolic regression to theorem proving and reasoning-based error; compares alternative theories and examines approximation validity. | Conformance, empirical adequacy, and domain validity should stay separate, but that separation is not itself new. Useful as a reference for bounded formal reasoning. |
| **Model Discovery Agent**, Murphy, 2026, v5 dated September 27. [Full text](https://arxiv.org/html/2608.09696v5), Sections 2–3, A.4, A.6–A.9, C.4. | Combines model proposals, approximate Bayesian inference over structures/parameters/latent trajectories, residual-triggered expansion, and value-of-information design. Its glucose example adds sensor lag alongside physiological dynamics. Finite-pool weights omit unproposed model mass. | Probably the strongest recent comparison for the later active-discovery project. Neither sensor dynamics nor active latent-model discovery distinguishes Vega. Evaluate justified attribution and abstention, and do not reinterpret finite-pool weights as unconditional physical truth probabilities. |
| **EvoSCM**, Zhao et al., 2026, v2 dated September 28. [Full text](https://arxiv.org/html/2609.01526v2), Section 3 and Appendices C/E. | Maintains causal hypotheses, assumptions, precommitted predictions and discriminating experiments; minimally revises edges, latent variables, mechanisms and parameters. Its deduction prompt uses LLM validation; the physics setup includes paired simulator branches sharing noise. | Structured belief revision is a direct overlap. Compare on the same acquisition interface, without granting simulator noise cancellation unavailable in a real experiment. LLM coherence checks are not infeasibility certificates. |
| **CriticAL**, Li et al., 2024 preprint, called CALM in the ICLR submission. [Full text](https://arxiv.org/html/2411.06590), Sections 3–4. | LLM-proposed statistics and posterior predictive checks filter model critiques; controlled examples assess detection and false critiques. Uses multiplicity correction. | Statistical validation of agent criticism and measuring false critiques are precedented. Use a discrepancy-testing comparator; extend the target to attribution among claim, measurement and computational explanations. Its discrepancy tests do not by themselves identify the cause. |
| **POPPER**, Huang et al., ICML 2025. [Proceedings](https://proceedings.mlr.press/v267/huang25n.html), [paper](https://www-cs.stanford.edu/people/jure/pubs/auto-hypothesis-validation-icml25.pdf), Section 2.3 and Appendices C/H. | Sequential agent-generated falsification tests use e-values and an e-process for error control, subject to valid sub-null implications and conditional validity. Reported failures include incorrect testing code and p-value interpretation. | Controlled automated falsification is not new. Later sequential claim testing must verify those prerequisites, including nuisance treatment and selection, rather than assign significance to a generated explanation. |
| **APOD**, Ferrara et al., October 5, 2026 preprint. [Full text](https://arxiv.org/html/2610.06227v1), Sections 2–4. | Discovers population ODEs and variability structures from sparse noisy data, using residual, predictive, parameter and plausibility diagnostics. Residual-error models and outputs are supplied. | A current comparator for noisy mechanistic discovery. The inspected evaluation does not establish joint acquisition-fault versus scientific-claim localization. Avoid claims that rich fitting diagnostics are unique to this project. |
| **AutoCog**, Jagadish et al., June 2026 preprint. [Full text](https://arxiv.org/html/2606.26448v1), system, discussion and confirmation study. | Runs a theory/experiment/revision loop with executable cognitive theories, human behavioral experiments and prospective confirmation. | Adding an experiment-design agent is insufficient. Its exploratory/confirmatory separation is useful for the proposed later scientific-action study. |
| **D3**, Holt et al., NeurIPS 2024. [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/aea8bdc42d8ba3a67a69b3f18be93f69-Paper-Conference.pdf), Section 3. | Modeling, feature acquisition and evaluation agents improve white-box/hybrid dynamics with fitting and feedback. | A comparator for iterative model improvement. Distinguish acquisition of an already available feature from a new calibrated measurement or physical intervention. Better validation error does not establish fault attribution. |
| **DASyR-LLM**, Aliaga Medina et al., August 2026 preprint. [Full text](https://arxiv.org/html/2608.05120v1), Sections 3–5. | Combines physicochemical critique, symbolic regression, fitting and model-based experiment design. Examples use fully measured species and additive Gaussian noise; the discussion excludes systematic bias and missing observations. | Reinforces the crowded active-discovery landscape. Partial observability and structured measurement faults are useful evaluation differences, provided they are actually tested. |
| **Mechanistic World Models**, Posner, Lei and Schölkopf, 2026. [Full text](https://arxiv.org/html/2607.12474), Sections 3–4. | Conceptual framework for mechanisms, variables, bindings, reuse and experimental discovery, including nonidentifiability. | Use as framing, not as an empirical diagnosis baseline. A new umbrella architecture is a weaker contribution than a falsifiable attribution result. |

## Diagnosis and formal model invalidation

The central methodological risk is rediscovering classical diagnosis in agentic
terminology. A scientific assumption can be represented as a diagnosable premise;
sensor assumptions can be premises too. What makes the current application hard
is obtaining trustworthy consistency evidence for continuous, partially observed
dynamics with incomplete computation. Even this difficulty has strong precedents.

| Primary work and review scope | Established overlap and implication |
| --- | --- |
| **Reiter, A Theory of Diagnosis from First Principles**, 1987. [Paper](https://www.cs.ru.nl/P.Lucas/teaching/KeR/reiter.pdf). Full-text review. | Minimal diagnoses correspond to hitting sets of conflicts. Competing diagnoses and additional observations are foundational. Numerical search failure cannot serve as the consistency oracle required to create a conflict; Vega must expose the strength of the underlying oracle. |
| **de Kleer and Williams, Diagnosing Multiple Faults**, 1987. [Paper](https://dekleer.org/Publications/Diagnosing%20Multiple%20Faults%20AIJ%20reprint.pdf). Full-text review. | Multiple faults, minimal violated-assumption sets, incremental evidence and information-based measurement selection are established. Mixed-cause diagnosis and informative next measurements are therefore essential baselines, not standalone novelty. |
| **de Kleer and Williams, Diagnosis with Behavioral Modes**, 1989. [Paper](https://www.ijcai.org/Proceedings/89-2/Papers/076.pdf). Full-text review. | Sherlock includes alternative failure modes and an unknown mode, with ranked explanations and probing. Unknown physical behavior should still be separated from unknown numerical results; generic abstention is not a new invention. |
| **Cazes and Kalech, Model-Based Diagnosis with Uncertain Observations**, AAAI 2020. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5664). Full-text review. | Directly studies possibly erroneous observations in consistency-based and abductive diagnosis, with complete/ranked diagnosis algorithms. Finite discrete observables and independent observation distributions distinguish its formulation from shared drift, calibration and trajectory uncertainty. This is a major threat to a broad “joint model/data diagnosis” claim. |
| **Anderson and Papachristodoulou, On Validation and Invalidation of Biological Models**, 2009. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC2704209/). | Uses sum-of-squares methods and error lower bounds to invalidate models with uncertain parameters, initials and measurements, including partial observations. Explains why local fitting failure is insufficient. Claim-family incompatibility with bounded certificates has substantial precedent. Certificate degree, formalism and numerical verification remain substantive limits. |
| **Harirchi and Ozay, Guaranteed Model-Based Fault Detection in Cyber–Physical Systems**, Automatica 2018. [Preprint](https://arxiv.org/pdf/1609.05921), [publication](https://doi.org/10.1016/j.automatica.2018.03.040). Full-text preprint review. | Uses model invalidation for hidden-mode switched affine systems with parametric uncertainty and bounded noise; feasibility and detectability establish when faults must be distinguishable. This is a strong comparator for finite families, uncertain observations and guaranteed detection, even without scientific natural-language claims. |
| **Ghosh et al., Diagnosis and Repair for Synthesis from Signal Temporal Logic Specifications**, HSCC 2016. [Paper](https://arxiv.org/abs/1602.01883). Full-text review. | Diagnoses inconsistency and repairs predicates/timing within a bounded repair space. Its adversarial procedure can return UNKNOWN when termination assumptions fail. Minimal repair and preserving computational uncertainty are established practices; empirical physical attribution is a different target. |
| **de Kleer and Raiman, Trading off the Costs of Inference vs. Probing in Diagnosis**, IJCAI 1995. [Paper](https://www.ijcai.org/Proceedings/95-2/Papers/092.pdf). Full-text review. | Explicitly compares further computation with physical probing using their costs and benefits. A refit-versus-measure decision is not new in general. Vega would need to demonstrate useful choices among scientifically distinct actions and charge their full cost. |

Related foundations include [AGM theory revision](https://doi.org/10.2307/2274239)
and [Console and Torasso's diagnosis semantics](https://doi.org/10.1111/j.1467-8640.1991.tb00388.x)
(abstract-level review). They motivate distinguishing consistency from positive
explanation and distinguishing set-minimal, minimum-cardinality and minimum-cost
repairs. None of these objectives identifies a physical cause merely by preference.

## Uncertainty and imperfect observations

| Primary work and review scope | Established overlap and implication |
| --- | --- |
| **Kennedy and O'Hagan, Bayesian Calibration of Computer Models**, 2001. [Paper](https://www.asc.ohio-state.edu/statistics/comp_exp/jour.club/KennedyOHagan.pdf). Primary-paper review. | Joint treatment of parameter uncertainty, simulator discrepancy and observation uncertainty is foundational. The handoff's joint posterior is a useful formulation within this tradition, not sufficient novelty on its own. |
| **Brynjarsdóttir and O'Hagan, Learning about Physical Parameters**, 2014. [Author manuscript](https://www.tonyohagan.co.uk/academic/pdf/simmach.pdf). Full-text review. | Ignoring discrepancy can give biased, overconfident physical estimates; flexible discrepancy can improve predictions without identifying physical parameters. Keep sensitivity to correction/revision priors visible and avoid interpreting a preferred explanation as uniquely identified. |
| **Agüero and Goodwin, Identifiability of Errors in Variables Dynamic Systems**, 2008. [Publication](https://doi.org/10.1016/j.automatica.2007.06.011). Abstract/introduction reviewed. | Studies identifiability with noisy inputs and outputs under explicit signal/noise conditions. Logged-input uncertainty is not solved by output robust loss. Full technical comparison needs the inaccessible portions of the paper. |
| **Raue et al., Structural and Practical Identifiability Analysis**, 2009. [Publication](https://doi.org/10.1093/bioinformatics/btp358). Abstract and indexed primary sections reviewed. | Profile likelihood distinguishes structural dependencies and finite-data limitations in partially observed dynamics. Joint nuisance analysis is necessary; a failed numerical profile cannot establish an exact symmetry. |
| **Villaverde, Tsiantis and Banga, Full Observability and Estimation of Unknown Inputs, States and Parameters**, 2019. [Primary record](https://pmc.ncbi.nlm.nih.gov/articles/PMC6685009/). Abstract/indexed primary text reviewed. | FISPO analyzes inputs, states and parameters together. Vega should assess the augmented physical/measurement system; analyzing physical coefficients while fixing uncertain sensors can manufacture apparent information. |
| **Villaverde et al., A Protocol for Dynamic Model Calibration**, 2021 online. [Paper](https://arxiv.org/pdf/2105.12008). Full-text workflow review. | Separates identifiability, objective formulation, optimization, uncertainty and validation. Orion's safeguards align with established practice. The new work must show how those safeguards affect attribution, not claim the safeguards alone as discoveries. |
| **Champion et al., A Unified Sparse Optimization Framework**, 2020. [Full text](https://arxiv.org/html/1906.10612v2). | SR3 combines sparse dynamics, trimming, physical constraints and parameterized libraries. Examples can trim legitimate off-attractor behavior; stationarity is not global adequacy. Joint equation fitting and data trimming already exist. Include rare valid transients that a robust method might wrongly discard. |
| **Hirsh, Barajas-Solano and Kutz, Sparsifying Priors for Bayesian Uncertainty Quantification in Model Discovery**, 2022. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC8864363/). | UQ-SINDy estimates coefficient/term uncertainty under library, noise and prior assumptions. These probabilities do not account for every missing mechanism or establish the truth of an informal scientific claim. |
| **Fasel et al., Ensemble-SINDy**, 2022. [Paper](https://arxiv.org/pdf/2111.10992). Full-text review. | Uses ensembles for robust discovery, inclusion statistics and active data acquisition. Ensemble agreement can conceal shared misspecification; disagreement alone cannot break exact observation/parameter symmetries. |
| **Skanda and Lebiedz, A Robust Optimization Approach to Experimental Design for Model Discrimination**, 2013. [Paper](https://arxiv.org/pdf/1101.3663). Full-text review. | Designs dynamical experiments by worst-case discrimination across parameter uncertainty, including measurement timing and perturbations. Active discrimination under nuisance uncertainty is established; the distinctive test is choosing the right kind of evidence for the suspected ambiguity. |
| **White et al., The Limitations of Model-Based Experimental Design and Parameter Estimation in Sloppy Systems**, 2016. [Full text](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005227). | Informative experiments can activate omitted mechanisms and invalidate an approximation. The limited-regime fixture should measure whether the diagnosis identifies a scope failure, rather than automatically blaming acquisition or refining coefficients. |
| **Talts et al., Validating Bayesian Inference Algorithms with Simulation-Based Calibration**, 2018, revised 2020. [Full text](https://arxiv.org/html/1804.06788v2). | Calibration checks inference computation under a declared generative model. It does not validate the physical correctness of that model. Separate inference checks, predictive coverage, and false-attribution rates. |
| **Hansen, Lunde and Nason, The Model Confidence Set**, 2011. [Paper](https://onlinelibrary.wiley.com/doi/pdf/10.3982/ECTA5771). Abstract and paper opening reviewed. | Sets of statistically unresolved predictive competitors are established. Covering a best predictive model is not the same event as covering the true physical cause; state which uncertainty is being assessed. |
| **Yang and Zhu, Bayesian Selection of Misspecified Models Is Overconfident**, 2018. [Author manuscript](https://discovery.ucl.ac.uk/10040828/1/Yang_2017YangZhuPNAS.pdf). Primary-paper review. | Model probabilities can become highly concentrated among wrong alternatives. An explicit family-inadequacy outcome and shared-omission controls are needed; model weights cannot repair missing support. |

Two recent watchlist items reinforce these conclusions but received only abstract
review: [robust SINDy via least trimmed squares](https://arxiv.org/abs/2606.28584)
(June 2026 preprint) and [Fung's ODR-BINDy work](https://doi.org/10.1145/3831701)
(institutional record lists acceptance in July 2026). They should receive a deeper
technical comparison before a paper makes claims about robust or errors-in-variables
equation discovery.

## A narrower contribution that can be falsified

Proposed central claim:

> Within a declared dynamical and observation family, an evidence-qualified joint
> diagnostic procedure reduces false accusations against correct scientific claims
> and sound acquisition contracts, while retaining useful detection of identifiable
> single and mixed causes, and abstaining on observational or computational ambiguity.

This would be a systems/methodology and evaluation contribution unless a genuinely
new inference or verification result emerges. “We combine existing components” is
not enough. The convincing result would identify a reproducible failure mode in
appropriate baselines and show that the new decision procedure prevents it at a
useful detection rate and acceptable cost.

The distinction between four objects must be operational:

1. **A candidate and fitted point.** A retained parameter vector explains, or fails
   to explain, particular readings under a specific objective.
2. **A candidate equation family.** A checked optimum or lower bound concerns all
   admissible parameters of those equations.
3. **A scientific claim family.** Every allowed construction preserving a reviewed
   claim must be considered before declaring that claim incompatible within scope.
4. **A measurement/input contract.** Every verdict is conditional on the admitted
   noise, calibration, timing, forcing and preparation assumptions.

Computation is an evidence status across these objects, not another physical
generating mechanism to be assigned a posterior probability. An unfinished
search can coexist with a real sensor defect. Report both facts without making
the computational status compete with the sensor in a mutually exclusive label.

The first finite catalog is particularly useful because coverage can be explicit.
Its limitation is equally important: it does not demonstrate unrestricted model
discovery, and the one-state basin is only a foundation for the eventual latent
system problem. Do not advertise the first pilot as solving general partial
observability simply because the surrounding architecture allows latent states.

## A principled baseline for the first Gaussian milestone

The following is an elementary confidence-set construction, proposed here as a
baseline and evidence specification, **not a new theorem**. It gives the first
pilot a sharper target than a tuned residual threshold.

Let h index a predeclared physical/observation branch and let theta include every
uncertain physical coefficient, prepared initial quantity and allowed sensor
nuisance. With fixed sampling/input design, exact mean predictions, n independent
Gaussian observations and known positive sigma_i, define

\[
Q(h,\theta)=\sum_{i=1}^{n}
\left(\frac{y_i-\mu_i(h,\theta)}{\sigma_i}\right)^2,
\qquad
\mathcal C_\alpha=
\{(h,\theta):Q(h,\theta)\leq\chi^2_{n,1-\alpha}\}.
\]

If the true pair belongs to the catalog, Q at that pair has a chi-square
distribution with **n** degrees of freedom. This is inversion of a test at each
specified mean; it does not use a fitted-residual n-minus-parameter heuristic.
Consequently the true pair lies in the set with probability 1-alpha. Projecting
the same set onto claim and measurement assumptions gives simultaneous protection:
any false accusation made because every retained world violates an actually true
assumption implies that the true world was excluded. The bound is conditional on
the declared experiment/model, not on the unrestricted physical world.

For the proposed finite rate catalog and bounded linear drift, each branch has a
convex weighted affine subproblem. A verified lower bound above the threshold
can eliminate a branch; a feasible point below it supplies a surviving witness.
An unresolved calculation must remain in a conservative outer set. Within a
branch, a claim such as zero drift requires its own constrained subproblem; one
representative fitted point does not describe all surviving parameter values.
In particular, a branch called “sensor drift” that permits zero drift includes
healthy sensors. Claim truth is a property of the model and parameters, not its
branch name. Sound numerical exclusion must also account for rounding error.

If the set is empty, report collective rejection of the catalog/observation
assumptions at the stated level. Suppress vacuous accusations of every individual
claim. If alternatives remain, their common consequences may be reported, while
disputed consequences remain unresolved. A minimum-cost preferred repair can be
offered separately, without erasing alternatives or acquiring a truth guarantee.

The construction has strict limits. Estimated sigma, correlated noise, data-based
sample deletion, approximate unchecked means, repeated testing/stopping, and a
catalog that excludes the true mechanism invalidate the stated guarantee unless
handled explicitly. Noisy y(0) cannot be plugged in as an exact state; include an
eligible true initial quantity and count the initial reading once. Data-dependent
proposals cannot silently discard ungenerated branches from the outer set.
Bounded numerical uncertainty needs conservative enclosures. Outliers, unknown
noise laws and input/timing faults need separately qualified procedures. Known-sigma
Gaussian attribution is therefore an initial increment, not a solution to the
entire eight-condition study. Its conservatism and power must also be measured.

## Proposed collaboration and biological case selection

The shared question is: **When a mechanistic model disagrees with biological
measurements, what evidence warrants revising the biology, revising the measurement
interpretation, or collecting additional evidence?** The collaboration should
shape this question before the benchmark is frozen, so the method addresses
decisions that matter in actual biological modeling.

The following division of work is proposed for discussion; it does not record
commitments already made by GEMOT's authors.

| Work | Proposed lead and joint contribution |
| --- | --- |
| Scientific cases and consequences | GEMOT collaborators identify meaningful competing biological explanations, plausible acquisition limitations, and consequences of a wrong revision. Vega helps turn these into testable questions. |
| Measurement and intervention semantics | GEMOT collaborators review assay interpretation, preparation, input records, feasible measurements and intervention costs. Both teams agree which assumptions may vary and what evidence could settle them. |
| Diagnostic methodology | Vega develops the claim/observation representation, evidence requirements, attribution and abstention rules, uncertainty treatment and computational safeguards. Biological review checks that formal claims preserve their intended meaning. |
| Evaluation and interpretation | Both teams choose acceptance criteria and interpret scientific outcomes. Vega implements matched baselines, reproducible evaluation and error accounting; collaborators assess biological relevance and proposed next actions. |

For each candidate case, jointly prepare a short case packet containing the
scientific question, competing mechanisms, observed and latent quantities,
readout interpretation, uncertain inputs/initials, plausible error magnitudes,
available follow-up actions and their costs. Include raw-data provenance and the
independent evidence, if any, that could establish a disputed explanation.
Preserve disagreement between experts as uncertainty where evidence cannot settle it.

Possible discussion prompts include a feedback explanation versus an uncertain
readout delay, an omitted biological process versus a measurement artifact, or a
response change versus uncertainty in the delivered perturbation. These are
candidate contrasts for collaborators to assess, not established properties of
any GEMOT dataset. Select cases because both explanations and a consequential
decision are credible, rather than because a chosen method can separate them.

Use three levels of evidence:

1. **Mathematical qualification.** Retain the small basin/affine controls to check
   the procedure and its stated guarantees. They establish technical behavior.
2. **Biologically grounded controlled cases.** With collaborator review, construct
   synthetic or semi-synthetic variants of a selected biological model. Keep
   disturbance labels private and preserve plausible scales and observation
   constraints. Known injected causes support error-rate evaluation, conditional
   on the simulator and injection design.
3. **Empirical biological cases.** Use suitable existing data and, where feasible,
   independently collected follow-up evidence. Assess whether conclusions and
   proposed actions are useful. An accepted model or expert preference alone is
   insufficient to label the true cause; unresolved cases retain that status.

Begin biological case selection alongside mathematical qualification. Confirm
scenario assumptions and splits before evaluating the method; reserve independent
cases or interventions for final assessment. This prevents one collaboratively
developed example from serving as both the method's design target and its sole
validation. A prospective laboratory experiment is an optional later milestone,
not a prerequisite or an assumed collaborator commitment.

## Methodology and software integration

I recommend an **attribution layer around a mechanistic proposer and numerical
evaluator**, with a stable interface that can later accept either Autoformalism or
GEMOT candidates. This is a proposed integration; software compatibility remains
unverified. Use the existing project interfaces for the first finite-catalog
qualification while developing biological cases with collaborators. Shared case
design can proceed before software integration. The latter needs an agreed
interface and access to a suitable implementation through collaboration or public
release; public availability is not a prerequisite for the collaboration itself.

| Proposed component | Responsibility and evidence boundary |
| --- | --- |
| Claim registry | Stable IDs, original wording, reviewed mathematics, scope, provenance, fixed objectives, and explicit revision permissions. A prose tag is not a proof that an equation has the claimed behavior. |
| Observation contract registry | Immutable raw readings/input logs and separate versioned hypotheses for calibration, timing, initial preparation and noise. Known, estimated and unavailable values remain distinct. |
| Candidate adapter | Maps equations, readouts, initial conditions, parameter roles and units into a restricted representation. Reject unsupported semantics rather than silently approximate them. |
| Numerical evidence adapter | Returns point identity, objective, residuals, bounds, completion state and numerical qualifications. Evidence strength is preserved; local stationarity stays local. |
| Attribution engine | Compares claim/observation branches and retains all unresolved alternatives. Only appropriate evidence can remove a branch or support a named accusation. |
| Scientific action selector | Chooses among additional computation, equation exploration, sensor calibration, input-log verification, repeat observation and intervention, with explicit costs and predicted consequences. |

For a later external adapter, review event semantics, pre-equilibration,
condition-specific parameters, transformed observations, likelihood normalization,
noise parameters, readout expressions and unsupported model-language constructs.
The existing restricted parser policy prohibits executing proposer-supplied Python.
An SBML/Antimony translation would need an allowlist and independent qualification;
a format conversion alone does not preserve scientific meaning.

A supplied contract should never become editable merely because fitting improves.
The experiment author designates fallible fields before a run. The proposer can
request a reviewed alternative, but the evaluator owns the interpretation and
scores the immutable raw observations through that alternative. Likelihood
comparisons across transformations must use a common raw-data measure and the
appropriate Jacobian/normalization; robust loss values cannot be substituted
directly into Gaussian likelihood comparisons.

The joint explanation can be recorded as
`(claim_packet, equations, observation_contract, fitted_coordinates)` with a
separate evidence envelope. Useful proposed envelope fields are
`scope`, `assumptions`, `candidate_hash`, `objective_hash`, `parameter_hash`,
`coverage_of_family`, `witness_or_bound`, `numerical_status`, `calibration_protocol`,
`unresolved_alternatives`, and `permitted_next_actions`. These are design fields,
not a claim that a matching schema currently exists.

Reuse the [measurement evidence design](../../../docs/PHASE_C_MEASUREMENT_EVIDENCE_DESIGN.md),
[noise-aware fitting design](../../../docs/PHASE_C_NOISE_AWARE_FITTING_DESIGN.md),
and implemented training-only feedback rather than duplicate them. The fitter
should retain `specification_verdict=not_assessed`; the new attribution layer owns
scoped claim conclusions. A backend-specific packet must never be promoted into
a stronger common label merely to make adapters appear interchangeable.

The first methodology study would hold the candidate pool and evidence interface
fixed while varying the attribution policy. A later, jointly agreed GEMOT study
could compare its existing workflow with the same workflow plus attribution,
using matched data and total budgets. This within-platform ablation asks whether
the added method changes conclusions or next actions usefully. It must also
include conventional joint diagnosis with the same hypothesis access; a larger
catalog alone is not evidence for a better decision procedure. Later experiments
could vary the proposer to test transfer across construction systems. Changing
both proposal and attribution policies at once would confound their effects.

## Examples that distinguish explanation from convenient repair

For the handoff's linear system

\[
\dot x=a u-bx,\quad x(0)=0,\quad y=sx,
\]

the output satisfies `y'=(s*a)u-b*y`. For any admissible positive c, replacing
`a` by `c*a`, `s` by `s/c`, and `x` by `c*x` preserves observations for every input.
This is an exact observational equivalence, not merely a small singular value.
More pulses cannot distinguish the gains. A calibrated physical-state reading or
independent known-level sensor calibration can add the missing information;
calibration uncertainty must then be propagated. A nonzero known physical initial
state would change this particular ambiguity and must not be smuggled into it.

For a different case, an accumulator and a leaking reservoir can agree closely
over a short regime but diverge after inflow stops. A sufficiently informative
drawdown measurement can discriminate them, conditional on the measurement model.
If sensor drift can mimic the decline, the action must be assessed against that
joint alternative as well. Merely maximizing disagreement between two physical
models can choose the wrong experiment.

If an omitted-drainage candidate fails while another adequate candidate preserves
the same scientific claims, the proper response is equation revision. If an
adequate candidate exists but a particular start fails, the proper status is
computationally unresolved. If all examined constructions fail but family coverage
is incomplete, the scientific claim remains unrefuted. These examples define
different required behavior even when their displayed residuals are similar.

## Changes recommended for the proposed pilot

Keep the existing [two-claim by four-observation-condition design](../../../docs/PHASE_C_SPEC_DATA_DIAGNOSTIC_STUDY.md),
balanced truth directions, evaluator-only disturbance labels, and staged qualification.
The literature motivates these additions to its future review, not edits to its
frozen predecessor or authorization to launch it:

1. **Add a conventional joint-catalog baseline.** Give it the same hypothesis
   vocabulary, costs, evidence, and compute budget. Include confidence-set or
   set-membership diagnosis where assumptions permit. Beating claim-only and
   observation-only arms may only show the value of a larger catalog.
2. **Separate accusation from ranking.** Compare ranked best-fit/BIC/penalized
   explanations, discrepancy tests, and set-valued attribution. Score false
   claim accusations, false data accusations, power and retained ambiguity.
3. **Add shared misspecification.** Use an omitted nonlinear mechanism or an
   unlisted sensor defect. Require an “unexplained within catalog” option and
   report its detection/power limits. No finite-data procedure can guarantee
   detection of every out-of-family world, especially an observationally equivalent one.
4. **Add a rare valid transient.** Test whether robust fitting or trimming falsely
   labels unusual but correct readings as corrupted. A contamination score is
   not an error label without further evidence.
5. **Keep three action controls.** One ambiguity resolved by calibration, one by
   physical intervention, and one unresolved by every available action. Score
   actual resolution, not an LLM's claim that an experiment is informative.
6. **Require a separate latent-system stage.** After the affine basin pilot,
   qualify at least one partially observed example with independently checked
   structural ambiguity. Keep the one-state success claim appropriately narrow.

Useful comparisons are the existing four arms plus the conventional joint
diagnosis baseline and always-abstain/wording-only controls. The proposed GEMOT
collaboration would add a within-platform ablation, with its scope agreed jointly.
MDA/EvoSCM-style policies are later external comparators; a reimplementation must
be labeled inspired by the paper unless it reproduces the authors' method
faithfully. A CriticAL-style discrepancy screen separates
detecting mismatch from localizing it. Formal model invalidation is a benchmark
oracle or baseline only on its supported mathematical class.

Keep paired source blocks as the independent units. Select policies using
development blocks, freeze them, then assess new blocks and fresh interventions.
Do not tune thresholds on locked outcomes or interpret correlated trajectories
as independent repetitions. Report error rates separately for both claim truth
directions and for identifiable versus ambiguous conditions. An always-abstain
system must fail the power/usefulness gates. The existing 64-block pilot cannot
establish arbitrarily small rare-error rates.

The known-sigma affine qualification remains the next implementation increment.
This survey adds no experiment to its budget. The existing main-study estimate
of 70.4 CPU-hours is a ceiling for its stated fit/diagnostic attempts only; new
baselines and controls require additional budgeting. No GPU/provider budget is
needed for the deterministic first stage. Live proposer costs, action costs and
external software availability remain future planning items.

## Recommendation and unresolved choices

Proceed with the narrow study, with **conditional diagnostic reliability** as the
central methodological result, and develop its biological cases jointly with
GEMOT's authors. Treat a later full discovery/action system as a second stage.
Before claiming a methodological advance, decide whether the paper contributes
a new attribution algorithm, a rigorous diagnostic benchmark, or both. The first
Gaussian confidence-set construction is a strong baseline but too elementary
to serve as the principal theoretical novelty.

The strongest objection is that the project integrates classical diagnosis,
model invalidation and uncertainty-aware fitting around an LLM proposer. A useful
answer must demonstrate a failure of strong, matched baselines and show that the
evidence distinctions change conclusions or experimental choices. If conventional
joint diagnosis already meets the target, the honest contribution is the benchmark
and interface, or the research should move to a harder supported family.

Open choices are the precise accusation event, minimum scientifically meaningful
defect size, admissible observation alternatives, treatment of unknown noise,
repair preference versus evidence, and sequential error accounting. Formalized
claim meaning must be reviewed before testing; later natural-language
misinterpretation is a separate source of failure. External integration needs
an accessible implementation, agreed usage terms and semantic adapter qualification.
Joint planning also needs a suitable biological case, documented measurement
semantics, feasible follow-up actions and clarity about independently available
validation evidence.

No result here establishes field validity, unrestricted scientific truth,
calibrated probabilities for LLM beliefs, or novelty against all literature.
The next research deliverables are a diagnostic decision specification with the
elementary baseline, matched comparator contracts and acceptance gates, and a
proposed biological case packet developed with collaborators. The former supports
the bounded implementation milestone; the latter determines which biological
extension would be scientifically useful and what evidence it would require.
