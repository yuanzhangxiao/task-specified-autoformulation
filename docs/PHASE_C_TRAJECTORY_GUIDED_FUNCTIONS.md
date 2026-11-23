# Using training trajectories to construct functions

**Design for discussion.** Phase C already passes training evidence to the function stage. Given an admitted topology, the proposer writes expressions for its processes and equations. I propose that the runtime first compare a few function families against measurements conditional on the sources of that particular equation. The proposer can use scientific knowledge to choose among them or offer another expression.

The new topology export gives each variable’s type, RHS sources and initialization. It leaves open how to group the sources into terms. For example, the same edges from glucose and insulin can represent an additive production and loss law or a joint response. First assemble candidate balances under the public requirements. In the basin task one volumetric transfer contributes $-Q/A_{\rm up}$ upstream and $+Q/A_{\rm down}$ downstream. Then choose forms for the unknown processes.

## What a trajectory tells us

| Equation | Evidence the runtime can use | What the evidence can distinguish |
|---|---|---|
| Observed algebraic variable, e.g. \(U=H(x)\) | Measured \(U\) against observed argument values at the same time, across training trajectories | An offset, a slope, or a change of slope when the other arguments are comparable. A flattened time trace alone gives little evidence for saturation in \(H\). |
| Observed differential state, e.g. \(\dot I=F(x)\) | For each interval, compare \(I(b)-I(a)\) with \(\int_a^b F(x(t))\,dt\). Use several interval widths and respect input discontinuities. | The observations constrain the sum of all rates in the insulin balance. An insulin peak says that production and loss balance at that time. It does not identify either function individually. |
| Equation with latent arguments or a shared process | Simulate the coupled equations with their declared initializations, using one parameter vector and the same process law in every equation that uses it. | Different hidden mechanisms can give the same observable trajectory. Preserve alternatives where the available observations cannot distinguish them. |

## When the balance isolates one function

The coupled-basin public specification gives the basin areas, the overflow crests, and the two measured inflows. In training event `train_003` the upstream water starts at its crest and upstream inflow is zero throughout. The upstream basin therefore supplies zero transfer. The downstream balance directly identifies the average outlet discharge on each interval.

\[
\overline Q_{\rm out}=\overline Q_{\rm in,down}-A_{\rm down}\frac{h(b)-h(a)}{b-a}.
\]

Compare three one-parameter laws, \(Q=k[\max(h-h_{\rm crest},0)]^p\), with \(p=1,1.5,2\). Fit \(k\) separately for each power by least squares over the 60 four-minute intervals of `train_003`. Average the inlet at the two endpoints using its public linear interpolation. Approximate the interval mean of the powered head by averaging its two endpoint values.

A second training event supplies a check at lower water levels. In `train_000`, the upstream basin needs \(800(0.35-0.20)=120\) m³ of water to reach its crest. Integrated upstream inflow first reaches that volume at minute 16. The four complete intervals up to that minute also have zero upstream transfer.

| Trial power | Outlet-flow RMSE on the four intervals of the second training event (m³/min) |
|---:|---:|
| 1 | 0.970 |
| 1.5 | 0.083 |
| 2 | 0.289 |

This favors the three-halves law for the outlet among the three candidates. The four check intervals cover a narrow range of downstream depths. Whole-system rollout across other events remains the relevant validation. The reusable algorithmic operation is to isolate an unknown process when the public balance and measured inputs permit it. When several rates remain unknown, fit them together.

## When two different histories give similar arguments

Take the saved T2 inventory-0 graph `g_9885c80dc2db2b32`, which admits \(\dot I=F(G_p)\). On the public training trajectory `train_001`, the administered insulin is zero throughout. The two-minute average rate is \([I(t+1)-I(t-1)]/2\) on this one-minute grid.

| Minute | Plasma glucose mass (mg/kg) | Plasma insulin (pmol/L) | Average insulin rate (pmol/L/min) |
|---:|---:|---:|---:|
| 20 | 184.966 | 50.454 | +3.550 |
| 193 | 184.923 | 55.396 | −0.596 |

At nearly the same glucose value, insulin is rising in one interval and falling in the other. Four- and six-minute windows give the same sign pattern. This is evidence to *compare* models. The glucose values differ slightly, the rates are interval averages, and the two times also differ in history. Calling this a deterministic contradiction would require stronger evidence or explicit bounds on the function.

First test a simple glucose-dependent production law on its own terms. On graphs that also admit insulin as a source, compare a production term and an insulin-dependent loss, e.g. \(\dot I=P(G_p)-kI\). A graph with a suitable hidden state permits another explanation involving delayed production. Compare these candidates on multiple training trajectories and report which uncertainty remains. If the evidence instead points to a source missing from a graph, return the observation to topology repair. An expression search should respect the graph it is evaluating.

## A first experiment

1. **Hold the graphs fixed.** Take a small, structurally diverse sample spanning the corrected algorithm-generated and proposer-repaired graphs. Include T2-easy and the coupled basin. Compare the same graphs across arms and record inventory, provenance, initialization and shared processes.
2. **Build a short list per equation.** Start with the simplest permissible form. Add a saturating family for a repeatedly flattening *conditional* response, or a two-source interaction when the slope with respect to one measured source changes with the other. Record how many trajectories and how much of the argument range support the comparison. When evidence is weak, keep competing forms. Maintain any public constraints on units, signs and parameter sharing. A peak time can guide a diagnostic without prescribing a delay or an equation.
3. **Compare three ways to choose expressions.** Use the current proposer, the algorithm's short list alone, and a hybrid in which the proposer sees the evidence and may substitute its own expression. Give each arm the same number of full-model fits, optimizer starts and compute cap per graph. Give the two proposer arms the same model and call budget. Include initializer formulas in the fitting contract. Score models with latent sources by simulating the whole system from the permitted initial conditions.
4. **Evaluate the decision.** During construction, use only training trajectories. Hold out entire trajectories *within training* to compare families where the number of trajectories permits it. Freeze each arm's candidates before applying the independent validation trajectories. Compare the fraction of graphs that yield a stable rollout, paired validation NMSE by target, public mechanism checks and compute used. Track whether the hybrid's winning expression came from its own proposal or the supplied short list. Reserve test trajectories for final evaluation.

Kacprzyk, Piskorz and van der Schaar model trajectory shapes directly. Their formulation motivates recording shape as evidence. Inferring a governing function under a fixed graph is a separate inverse problem. Messenger and Bortz give a weak-form approach to learning dynamics without pointwise numerical differentiation, relevant to the interval comparison above. [Skip the Equations](https://proceedings.mlr.press/v267/kacprzyk25a.html) · [Weak SINDy](https://doi.org/10.1137/20M1343166)
