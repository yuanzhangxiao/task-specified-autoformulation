# Screening ODE topologies before proposal

Design note for review · Phase C · 22 November 2026

## 1. Decision problem and deliverables

Which information must be capable of reaching an output in time for a shared dynamical model to explain the training trajectories? This question can be asked before a proposer names a latent state. Different physical paths can carry the same information. Choosing a path and estimating a rate law are subsequent decisions.

**Input.** Supply (i) multiple training trajectories with clocks, units, observed intervals and measurement masks, (ii) a public forecast contract specifying inputs, interpolation, available baseline measurements, allowed initialization policies, outputs and required or forbidden mechanisms, and (iii) a finite catalogue of admissible information sources or complete paths with explicit costs. Record bounds on new states and edges if graph realizations are requested. A public output whose type is unspecified may be evaluated under more than one type hypothesis.

For run r, let x denote differential states and z algebraic quantities. A graph G specifies dependency sets P. One admissible model has the form

$$
\dot x_i^{(r)}(t)=f_i(P_i^{(r)}(t);\theta), \qquad
z_j^{(r)}(t)=h_j(P_j^{(r)}(t);\theta).
$$

$$
x^{(r)}(t_0)=g_\eta(b^{(r)}).
$$

The laws and parameters are shared across runs. The initialization information b is available at forecast time. An observed initial coordinate is fixed to its authorized reading. A latent coordinate can use a shared constant or an allowed map of baseline readings and earlier initializer outputs. A prefix estimator is admissible only under a declared prefix protocol. Its input ends at the forecast origin and scoring starts after the prefix. Assume deterministic evolution with unique solutions on the common observation interval and well-defined algebraic outputs. Resets, interventions, prehistories and nonunique algebraic solutions require explicit semantics.

**Output.** For each observed target return all inclusion-minimal information sets that pass the specified screen, their costs, the cheapest sets, and the training evidence for each decision. Return contradictions and undecidable cases separately. A requested second output is a portfolio of globally valid graph realizations with explicit initializers, complexity, numerical scores and uncertainty. An unmeasured latent state enters through its observable consequences.

**Two meanings of optimal.** The finite screen admits a provable minimum-cost solution. A scientific objective could instead minimize graph cost subject to an achievable rollout error at most ε, with the infimum taken over one admissible set of shared laws, parameters and deployable initialization rules. Specifying that family, loss, tolerance and cost is part of defining the second problem. A positive error lower bound above ε rules out a candidate family. A zero bound leaves its best achievable error open.

$$
\min_{G}\ c(G)\qquad \mathrm{subject\ to}\quad
C(G)=1,\quad \inf_{\phi\in\mathcal{F}(G)} L_{\mathrm{train}}(G,\phi)\leq\varepsilon .
$$

This is a best achievable error criterion. At a threshold equal to an unattained infimum, an actual model attaining the threshold requires an existence check. Crossing the training threshold is a separate outcome from predicting new trajectories successfully.

## 2. Witnesses and the equality test

Fix a target and a common forecast origin. A witness w = (r,s,t) consists of two runs and a common time with disjoint observed target intervals. The signature σⱼ(r,t) is the information option j makes available to the target by t. Examples include a baseline reading carried by a latent initializer, or the admissible history of a forcing channel driving a differential state. Include fixed information such as the target's measured initial value in every candidate.

**Equality principle.** If two runs give a closed upstream subsystem equal initial states and equal forcing histories, the common initial-value problem has the same solution. A deterministic observation of that subsystem must agree. The contrapositive gives a necessary information constraint for each witness. A different source history allows different predictions. It does not specify their sign, size or physiological cause.

For a differential target, compare its initial information and the causal forcing prefix. A step in a bounded forcing at the endpoint alone cannot instantaneously move a continuous state. For an algebraic target, a direct source may act at that same instant. Propagate histories through any differential ancestors of the algebraic target. A pure baseline map cannot depend on later observations. A generated glucose series must come from its own upstream model in open-loop prediction. Its measured future values cannot be introduced as independent forcing.

For each witness and available option record one of three facts. **Different** means a verified difference can reach the target by the deadline. **Equal** means equal information is certified. **Unknown** includes missing records, overlapping uncertain input intervals and unresolved initializer dependencies. Equal covariates fed to a shared initializer give equal initials. Unequal covariates only permit different initials, and a fixed map may still give equal initials. An unconstrained initial value fitted separately to each complete trajectory provides no deployable source of distinction. If fixed information distinguishes a witness, omit that row from the remaining obligations. For a candidate, any all-equal row proves a contradiction. A row with an unknown and no known difference is unresolved. Passing requires a known distinction in every remaining row.

**What a pass proves.** With complete signatures and exact scalar labels, passing is necessary and sufficient for a time-indexed deterministic lookup on the observed signatures. Assign the observed value to each equal-signature group. For scalar interval labels, common intersection within each group suffices. Reusing one ODE across times and trajectories imposes more conditions. For example, two scalar ODE solutions with the same forcing and unique evolution cannot reverse their ordering. Distinct initial values can pass the equality screen even when measured final values reverse that ordering.

**A source can matter only jointly.** Consider three runs with common initial x = 0, constant input pairs (0,0), (1,0), (0,1), and slopes 0, 1, 2. Each input alone leaves two runs indistinguishable. The pair admits the shared law dx/dt = u + 2v. A permanent classification of individual sources into adequate and inadequate would lose this solution.

## 3. Exact solution and worst-case cost

For one target, let m be the number of remaining witnesses and p the number of optional sources or complete causal paths. Column j costs a positive amount cⱼ. Set A to one exactly when option j carries a verified distinction for row w. In this section, options can coexist, their distinctions combine by union and costs add. Write sⱼ = 1 when option j is selected. The minimum-cost problem is weighted set cover:

$$
\min_{s\in\{0,1\}^{p}}\ \sum_{j=1}^{p}c_j s_j
\qquad\mathrm{subject\ to}\qquad
\sum_{j=1}^{p} A_{wj}s_j \geq 1 \quad (w=1,\ldots,m).
$$

Each row specifies a set of alternatives; the answer must hit every such set. A zero row means no available option certifies a distinction for that witness. Report whether all options were certified equal or some were unknown. With no outstanding witnesses the empty addition is optimal, and the data provide no positive evidence for an optional dependency. Keep mandatory public mechanisms in the base contract.

**Minimum versus minimal.** Positive costs make an optimum inclusion-minimal. Other inclusion-minimal sets may be more expensive and scientifically different. A cover is inclusion-minimal exactly when every selected option has a private witness that no other selected option covers. This yields an exact test while iterating over subsets. Nonminimal sets may improve fits by providing extra interactions.

**Work.** For n aligned runs with T samples, one target has at most T·n(n−1)/2 rows. For simple complete baseline and forcing options, incrementally compare prefixes on the common grid in O(n²Tp) time per target under unit-cost sample comparisons. Arbitrary time alignment, uncertain interpolation or graph reachability must be costed separately. The dense witness matrix costs O(mp) space. Brute force costs O(2ᵖ p(m+1)) time including checking minimality. Integer programming or branch-and-bound can certify the same optimum. For a few witnesses one can instead use dynamic programming over covered witness masks, with O(p·2ᵐ) transitions plus mask-operation costs.

**Hardness on actual trajectories.** Reduce Set Cover to one scalar, smooth ODE with common initial value zero. A reference run has constant inputs all zero and observed x(1) = 0. For each universe element e add a run with x(1) = 1 and binary input j indicating membership of e in set j. An uncovered element has the reference signature. Conversely, any cover S realizes the entire dataset with the shared law below.

$$
\dot x=1-\prod_{j\in S}(1-u_j).
$$

Input size is O(mp). Verifying a chosen finite cover is polynomial. Hence the budgeted screen decision is NP-complete and its optimization is NP-hard. This proof makes no NP-membership claim for general nonlinear ODE fitting. Listing all minimal sets can itself require exponential output: m disjoint witnesses, each with two private alternatives, have 2ᵐ minimal covers.

## 4. Approximation and graph realization

**Exact first.** With eight optional sources there are 255 nonempty subsets. Exhaustive search gives an interpretable baseline, including ties, for testing faster solvers. Deduplicate identical witness masks for computation while retaining their scientific identities for ranking. If learning the direct effect versus a mediated effect is important, keep both realizations even when their masks match.

**When one low-cost cover suffices.** Repeatedly choose the option covering the most new witnesses per unit cost. If all rows are coverable, the cost is at most H_d times optimum, where d is the largest number of witnesses covered by one option, and H_d ≤ 1 + ln d. Direct rescoring costs O(mp·min(m,p)) after building the matrix. This is Chvátal's weighted set-cover guarantee [3]. Greedy returns one cover. Exact enumeration is required if completeness over minimal alternatives is a deliverable.

**When a budget allows residual conflicts.** For unit-cost options and a budget of k selections, maximize the sum of nonnegative weights of the witnesses covered. Greedy by marginal gain achieves at least 1−1/e of optimum in O(kmp) time by direct rescoring [4]. Mark uncovered hard contradictions explicitly. Such a partial cover diagnoses missing information or prioritizes further construction. It cannot earn structural acceptance while a certified hard contradiction remains.

**Why sources and graphs need separate variables.** A source, its route, and its initializer need not be the same decision. Suppose the only path is u → M → y. Adding either edge alone to an empty graph transmits no information. Selecting both can resolve a witness. Alternatively, two outputs may share one memory state and should incur its cost only once. Independent set-cover costs and approximation guarantees do not apply to these raw edge decisions.

For a bounded graph catalogue define binary state, edge and initializer decisions g, public admissibility predicates C(g), total cost c(g), and a predicate R(g,w) computed by tracing permitted roots to the target before the witness deadline. A joint structural optimum is

$$
\min_{g\in\mathcal{G}} c(g)
\qquad\mathrm{subject\ to}\qquad
C(g)=1,\quad R(g,w)=1\ \mathrm{for\ every}\ w.
$$

Here R = 1 requires a certified distinguishing route. Materialize and recheck ownership, initializer availability, shared-process constraints, well-posed algebraic evaluation and all affected targets. With B binary catalogue decisions, exhaustive checking costs O(2ᴮ · Q), where Q bounds the chosen finite graph and witness checks for one assignment. Treat complete paths as independent columns only where composition and additive costs have been verified. Otherwise use exact graph search or a heuristic portfolio with no claimed cover-cost guarantee.

The general nonlinear rollout objective has no complexity bound in p and m alone. The rate-law representation, parameter domains, simulation accuracy and optimizer all matter. A useful tractable special case is a fixed linear-in-parameters integral regression with N measured windows and k basis terms, N ≥ k. Dense QR gives a least-squares optimum in O(Nk²) arithmetic operations for full column rank. Its guarantee concerns that regression score. Fitted latent states and joint nonlinear rollouts require a different solver.

## 5. Quantifying conflict and ranking survivors

For target i at time t, partition comparable runs into groups E whose predictions the candidate forces to agree. For exact recorded values y with observation weights ω, the best common prediction in a group is its weighted mean. For observed intervals J, use distance to the interval. A lower bound on the weighted, unnormalized squared error against the allowed observations is

$$
B_i(G)=\sum_t\sum_E\ \min_{a\in\mathbb{R}}
\sum_{r\in E}\omega_{rt}\,\mathrm{dist}(a,J_{irt})^2 .
$$

In the point-observation case, divide by the sum of weights and take the square root for an RMSE floor. Disjoint groups avoid double counting pairs. Allowing each group a freely chosen prediction at every time relaxes the shared ODE, proving the lower bound. Use singleton groups when equality is unverified. Report group sizes: all singleton groups yield zero even when a dataset contains almost no information for this test. A source may remove exact contradictions because of a tiny difference with negligible scientific significance.

**Seed 102.** With 16 public training runs and 301 minutes, grouping by observed initial insulin and infusion history gives 91 contradictory insulin pairs and an RMSE floor of 92.71 pmol/L. Supplying meal history gives 10 pairs and a floor of 0.95. Supplying both meal history and initial glucose through an admissible route leaves 3 exact conflicts. The largest remaining observed gap is 1.62 × 10⁻⁸ pmol/L. These figures illustrate why the report should show both exact counts and physical magnitude. Numerical tolerances need independent calibration. Approximate equality of two forcing prefixes requires a justified sensitivity bound before it can support a rejection.

**Predictive ranking is a different optimization.** Among screen survivors, record graph complexity, scientific mechanism coverage, the lower bound and a training-only predictive score. One inexpensive score fits the same class of rate laws on training folds and predicts target increments on entirely omitted training trajectories. A held trajectory's observed states may be used in a clearly labeled conditional rate diagnostic; open-loop proxies must generate every model-owned series. Fit shared dynamics and initializer parameters on each fold's training runs. Apply the same permitted initialization protocol and forecast horizon on the remaining runs. Compare full models with matched fitting budgets and independent validation under a frozen selection protocol.

No logarithmic approximation bound transfers to the improvement in squared error. As a counterexample, two balanced binary inputs with an XOR output give no decrease in within-group squared error from either input alone. Both together explain the output exactly. The numerical and scientific scores break ties and prioritize alternatives; choosing a poor function or encountering a poor optimizer optimum cannot establish that the topology is impossible.

## 6. Prior art and the first design decisions

**Direct predecessors.** Davies and Russell [7] formulated minimum feature selection for a consistent mapping from pairs of examples and proved NP-completeness. Ideker, Thorsson and Karp [1] compared Boolean perturbation experiments with different target outcomes, formed sets of differing candidate inputs, and inferred smallest regulator sets by minimum cover. Jarrah et al. [2] found all inclusion-minimal wiring sets consistent with finite-field discrete transitions. There an eligible set can support an interpolating function on the observed transitions.

**Continuous-time neighbors.** CausalKinetiX [5] uses stability of kinetic laws across experiments to rank candidate structures. Streif et al. [6] study bounded-error consistency and initial-condition/parameter sets for specified continuous-time systems, using occupation measures and relaxations. Those are useful precedents for ranking and for certifying feasibility once a model family is fixed. The proposed contribution to evaluate is the combination of causal histories, deployable initialization, joint graph realization and a scientific proposer. This targeted literature review establishes direct overlaps. Priority for the full combination remains open.

**Decisions for the first implementation.** Start with noiseless or independently bounded scalar targets, aligned clocks, recorded initial conditions and forcing interpolation known from the public contract. Return a witness table and all minimal covers for small catalogues. Permit bounded latent and initializer alternatives, with an explicit unresolved outcome when evidence is missing. Compare costs, screen yield and leave-one-trajectory-out predictions separately. Preserve competing realizations for fitting, particularly when identical masks hide different biological mechanisms.

Use a case with a known contradiction, a case with distinct baseline states that should pass the equality check, and synthetic cases with complementary sources, a forbidden initializer, and two targets sharing one memory. Check exhaustive agreement on small instances. Report how often the data exclude a hypothesis, how many alternatives survive, and whether their training-only ranking predicts subsequent rollout performance. Reserve validation and test trajectories for their declared purposes.

The LLM proposes meaningful latent mechanisms, possible mediated routes and scientific priorities within the public contract. The algorithm supplies the unmet witnesses, enumerates or bounds the alternatives, and rechecks additions to the catalogue. An empty cover identifies the limits of that catalogue or of the available evidence. It justifies declaring the public specification inconsistent only with an independent argument that all its permitted realizations are excluded.

**Primary references**

1. Ideker, Thorsson & Karp (2000). Discovery of regulatory interactions through perturbation: inference and experimental design, §3.1. [Paper](https://idekerlab.ucsd.edu/wp-content/uploads/publications/ideker_psb2000.pdf)
2. Jarrah, Laubenbacher, Stigler & Stillman (2007). Reverse-engineering of polynomial dynamical systems. [Paper](https://doi.org/10.1016/j.aam.2006.08.004)
3. Chvátal (1979). A greedy heuristic for the set-covering problem. [Paper](https://doi.org/10.1287/moor.4.3.233)
4. Nemhauser, Wolsey & Fisher (1978). An analysis of approximations for maximizing submodular set functions—I. [Paper](https://doi.org/10.1007/BF01588971)
5. Pfister, Bauer & Peters (2019). Learning stable and predictive structures in kinetic systems: benefits of a causal approach. [Paper](https://arxiv.org/abs/1810.11776)
6. Streif, Rumschinski, Henrion & Findeisen (2013). Estimation of consistent parameter sets for continuous-time nonlinear systems using occupation measures and LMI relaxations. [Paper](https://arxiv.org/abs/1303.4615)
7. Davies & Russell (1994). NP-completeness of searches for smallest possible feature sets. [Paper](https://cdn.aaai.org/Symposia/Fall/1994/FS-94-02/FS94-02-011.pdf)
