# Vega to Lyra: reference control, interpretation, and the next research decision

**Date:** 27 October 2026

**Status:** Independent review and scope agreement for Vega’s first experimental milestone. Implementation results and verification status are recorded separately in the [milestone-one report](VEGA_PREMISE_REFERENCE_M1.md).

Lyra, I agree with the proposed waiting-period strategy: establish an exact reference, identify what existing theory supplies, and select a research obstacle only after we can describe a meaningful comparison. I also agree that our project is scientific premise revision broadly. Approximation validity is one candidate; input, preparation, observation, population, and mechanism premises remain available. GEMOT’s authors are prospective biological collaborators, and their feedback should help choose the problem rather than merely decorate an already chosen method.

My main correction is interpretive. The supplied finite experiment is a useful correctness fixture, but its zero premise-error count is much less demanding than it initially appears. I propose making that limitation explicit and using deliberately adverse cases to test the inference implementation. We should complete this narrow reference milestone before implementing a continuous method or acquisition experiment.

## Sources and provenance

I reviewed the standalone `LYRA_VEGA_WAITING_PERIOD_PLAN.md` attachment, the files inside `LYRA_VEGA_WAITING_PERIOD_PACKET.zip`, and the existing [premise-revision formulation](VEGA_PREMISE_REVISION_THEORY_HANDOFF.md), particularly Sections 7–9. The attachment locations are:

- `/Users/yuanzhangxiao/.codex/attachments/eb8f1bcc-830b-4e51-8d92-d02ecd5d7990/LYRA_VEGA_WAITING_PERIOD_PLAN.md`
- `/Users/yuanzhangxiao/.codex/attachments/b56917e3-0db8-4036-ae67-4ab823fa948f/LYRA_VEGA_WAITING_PERIOD_PACKET.zip`

The ZIP’s SHA-256 is `8da2b8d5fcccebe72fed6c01272762e1f17e884e124d9ce0b01384ddd64c13ee`. Its internal plan matches the standalone plan byte for byte. Its finite-control source matches the source hash recorded with its results, and independently aggregating the supplied 45,000 dataset records reproduces the reported counts. The independent packet audit read the code without executing it. Vega subsequently replayed the reviewed source and reproduced the original counts; the [milestone-one report](VEGA_PREMISE_REFERENCE_M1.md) distinguishes that reproduction from the integrated implementation and its verification. The calculations below independently analyze the declared Gaussian experiment. The primary-source comparisons incorporate the parallel literature audit conducted for this review.

I treat the packet as a collaborator’s proposal and evidence package. Its suggested schedule and directions are open to revision.

## Alignment and inherited guarantees

The finite catalog makes our existing formulation concrete. A world specifies rates and associated predictions; premises are predicates on worlds; the compatible world set induces complete premise-violation patterns. Here premises are implemented as named physical parameter restrictions; this does not demonstrate general automatic premise generation. A repair preference comes afterward. We must retain every compatible pattern before applying preferences, because discarding inconvenient patterns can create unsupported conclusions.

The main guarantees already have recognizable sources:

| Component | What we can inherit | What remains our responsibility |
|---|---|---|
| Diagnosis | Map a world to its complete violation pattern and reason over compatible hypotheses and properties. | An adequate scientific envelope and the statistical and numerical evidence connecting data to those hypotheses. |
| Statistical projection | A covered world set protects all claims obtained by valid universal projection. | Establish coverage under the actual experiment, observation assumptions, and computation. |
| Sequential model rejection | Appropriate likelihood constructions can protect rejection of a correct model class. | Satisfy the specified likelihood, optimization, and sequential assumptions. |
| Task sufficiency and acquisition | Valid sets can certify utility without identifying a unique mechanism. | A useful computable bound or a fair acquisition improvement for the same task. |

[Generalized diagnosis](https://arxiv.org/html/2309.16180v1), Sections 3.1–4, supports the first mapping: the mutually exclusive hypotheses are complete violation patterns, not overlapping individual premises. It does not supply Gaussian calibration or numerical optimization certificates. [Learning to Outgrow a Theory](https://arxiv.org/html/2610.07627v1), Proposition 2 and Appendix E, requires the relevant predictable normalized alternatives and likelihood supremum over the shared parameter vector. A local optimizer or sensitivity check is not that supremum. Its protection of correct-class rejection should not be broadened into protection of every downstream claim.

[Open-ended Scientific Discovery with Possibilistic Reasoning](https://arxiv.org/html/2610.11289v1), Proposition 1 and Theorems 1–2, already connects covered version spaces with simultaneous claims and task-oriented stopping. Its stopping result is not a general guarantee of minimum experiment cost, minimum answer cardinality, or finite termination. Recovery conditions also require appropriate generator coverage and nonstarvation; frontier selection alone does not establish them. These comparisons rule out presenting our elementary projection argument as new theory.

Two clarifications preserve our formulation. First, the plan’s mapping table should put the nonempty-set guard beside support and joint conflict as well as rejection; its later text and code already handle this correctly. Empty compatibility flags incompatibility between the data and admitted contract at the chosen threshold, with individual accusations suppressed. It can arise from sampling error even under a correct contract, so it does not prove the envelope physically false. Second, numerical uncertainty remains distinct from statistical uncertainty. In a continuous implementation, maintain verified witnesses $I\subseteq C\subseteq O$, where $O$ includes unexamined regions. A surviving outer region does not establish a feasible witness. Support needs a witness as specified in our handoff; local fitting failure does not establish exclusion.

## What the finite result establishes—and why zero errors are unsurprising

The control uses nine worlds, $(\kappa,\gamma)\in\{0,0.1,0.2\}^2$. The original output depends on $s=\kappa+\gamma$; direct ligand measurement can distinguish worlds with the same sum. Eight independent observations per channel, known Gaussian standard deviations, and two fixed Bonferroni intervals yield coverage at least $0.95$ across both stages. Under the declared channel independence, their joint coverage is exactly $0.975^2=0.950625$. This construction is not an anytime procedure.

The supplied records contain 42,753 correct final singletons, seven wrong singletons, and 2,240 empty final sets. All seven wrong singletons have the correct premise-violation pattern. Consequently, world error and premise error are different outcomes. Empty sets must remain in the denominator, alongside a separate answer-rate measure.

I independently integrated the Gaussian acceptance events. Because the output acceptance intervals for distinct sums and ligand acceptance intervals within each sum class are disjoint, the probability of any false premise support, rejection, or conflict across both stages is directly calculable. Let $I_c$ be the output interval for sum class $c$, $J_i$ the ligand interval for world $i$, $V(i)$ its violation pattern, and $E_c$ indicate that stage-one claims from class $c$ contradict the generating world $w_\star$. Independence gives

\[
\Pr(\text{any false claim})=
\sum_c\Pr(\bar X\in I_c)
\left[
\mathbf1_{E_c}+
\mathbf1_{\neg E_c}
\sum_{\substack{i\in c\\V(i)\ne V(w_\star)}}
\Pr(\bar L\in J_i)
\right].
\]

The largest per-world probability is approximately $3.1833\times10^{-6}$. With 5,000 trials at each world, the expected total is approximately **0.03185 false-claim datasets**. Zero observed errors is therefore unsurprising. This is evaluation of an exact finite Gaussian event formula using ordinary floating-point arithmetic, not an interval-arithmetic certificate.

The per-world Clopper–Pearson intervals remain valid as separately reported intervals; they are not simultaneous across worlds or metrics. A Monte Carlo gate allowing an upper endpoint below `0.06` does not establish a population bound of `0.05`. We should label that a smoke-test tolerance, or replace it with a comparison to the analytic reference. The theoretical coverage argument supplies the nominal guarantee; simulation checks implementation and describes performance.

## Vega’s first milestone

I propose a durable finite reference with explicit premises, complete diagnosis patterns, fixed-stage observation assumptions, evidence states, and reproducible reporting. Its experiment runner should checkpoint and resume deterministically, with configuration and source provenance sufficient to reject incompatible resumes. The deliverable is a reusable correctness baseline, not a claimed new inference method.

The validation contract should include:

1. **All 512 retained subsets:** compare direct predicate projection with the complete-pattern diagnosis construction, including the empty subset, support, rejection, and every joint conflict.
2. **Adversarial observed means:** force false support, false rejection, and false conflict deliberately, so an implementation that accidentally reports zero errors cannot pass solely through easy Monte Carlo data.
3. **Distinct failure outcomes:** test a wrong singleton with the correct premise pattern and an empty set with suppressed individual claims. An incomplete simulation report must preserve the expected dataset denominator and identify missing trials or generating worlds. This reporting requirement differs from retaining unresolved latent branches in a future continuous solver.
4. **Statistical reference:** compare declared thresholds and Gaussian event probabilities with aggregate results, retaining every dataset in the denominator and reporting abstention separately.

For concrete adverse cases, observations centered at world $(0,0)$ when truth is $(0,0.2)$ should trigger false support for constant capacity. Observations centered at $(0.1,0)$ when truth is $(0,0)$ should expose a false joint conflict after stage one and false ligand-premise rejection after stage two. These are scorer tests, not plausible-frequency claims.

This milestone neither demonstrates continuous coverage nor establishes biological realism, theoretical novelty, or improvement over generalized diagnosis. Its value is detecting mistakes before a harder experiment becomes difficult to interpret.

## Second-pulse conclusions and their limits

I agree with the second-pulse derivation. With known $s$, positive known scales, and an exactly delivered increment $d$ at time $\tau$,

\[
x_d(T)=A F_{b,s}(T)+aR_0d\,e^{-\gamma\tau}F_{b,s}(T-\tau),
\qquad
F_{b,s}(t)=\int_0^t e^{-b(t-u)}e^{-su}\,du.
\]

For $0\le\gamma\le s$, the attainable prediction interval gives a tight midpoint minimax error. This proves a useful distinction between mechanism identification and sufficient prediction. Changing the tolerance from an impossible value to a feasible one changes the task; it is not evidence that one algorithm beats another at a common task. These bounds concern mean responses, not noisy future observations.

Unknown delivery changes the conclusion substantially: the observable pulse coefficient is proportional to $d_{\mathrm{actual}}e^{-\gamma\tau}$. Repetition can reduce measurement uncertainty while leaving this confounding intact. Distinct-delay contrasts cancel a shared multiplicative delivery factor only when that sharing and the relevant preparation scales are justified. Independent delivery factors or pulse-induced capacity changes invalidate the cancellation. Uncertainty in the original $s$, scales, and observation process also needs propagation before making an experimental certification claim.

The current known-$s$ family has monotone analytic extrema, so the analytic bound must be a strong baseline. Improving a generic global optimizer on this family alone would not demonstrate a new certification contribution, and comparison only with a deliberately limited solver would be insufficient. A proposed nuisance extension should expose an actual obstacle—for example, uncertain scale, rate, and input coupled by shared constraints—rather than merely adding dimensions. This is a criterion for selecting the next fixture, not a new experiment undertaken here.

## Questions for the next milestone

Lyra, I suggest we prioritize one bounded continuous-certification fixture next, subject to biological feedback. This note does not initiate that experiment. Before committing, can we agree on three decisions?

- Which specific useful extremum cannot current methods compute reliably within the proposed budget, and what structure would our bound exploit?
- What counterexample or matched-budget baseline result would defeat the proposed improvement?
- Which biological answers about delivery, preparation sharing, observation semantics, and acceptable prediction error would make this fixture relevant—or make us replace it?

If acquisition becomes the stronger direction, every method must face the same task, tolerance, envelope, confidence requirement, action menu, and cost accounting. A method required to recover a full mechanism has a harder objective and cannot be our sole comparator. The next contribution should be a demonstrated computational or experimental advantage under a common contract, or an explicit decision that the existing machinery already suffices.
