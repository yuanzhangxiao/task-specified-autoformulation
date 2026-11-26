# Choosing the next step in ODE construction

The output is an executable model with differential and algebraic equations, their dependency graph, initialization rules and fitted parameters. The decision at each step is whether a computation is worth its cost given the evidence already collected. A list of diagnostics alone cannot make that decision.

## What the search optimizes

Write a model as $M=(G,\Phi,H_0,\theta)$. Here $G$ is the graph, $\Phi$ the equations, $H_0$ the initialization rules, and $\theta$ the fitted coefficients. The graph includes any unobserved states. The equation and initializer families define a set of possible trajectories as their shared coefficients and permitted initial values vary. A fitted model is one member of that set.

For a new event, assign loss equal to a prespecified normalized forecast error if the model qualifies on the full event. Record failure separately if it violates a required condition, cannot simulate, or abstains. For comparing construction policies, specify either a minimum acceptable qualification rate or an explicit cost for failure before comparing forecast loss. Otherwise a policy can appear to improve by dropping difficult events. Any preference for simpler models, shorter runs or fewer LLM calls also belongs in the design. In the T2 experiment below we enforce qualification, report coverage, and measure whole-event error with equal weights for the three public targets.

The search state contains the surviving graphs and equation families, certified equivalences and contradictions, fitted models, evidence available on the permitted split, and the remaining budget. Possible actions include asking for new states, extending a graph, choosing equations from an admissible family, applying a diagnostic, fitting, forecasting on a selection event, and returning a model or abstention. A test on a fitted coefficient becomes available only after fitting.

Given a distribution over tasks and action outcomes, a budget of $b$ units and a computation cost weight $\lambda$, an ideal policy would minimize

$$
V(S,b)=\min\left\{R_{\rm stop}(S),
  \min_{a:c_a\le b}\left[\lambda c_a+
  \mathbb{E}\{V(S',b-c_a)\mid S,a\}\right]\right\}.
$$

Here $R_{\rm stop}$ is the predicted cost of returning the best qualified model or abstaining now. Choosing a lower NMSE on observations already used to fit is insufficient to estimate this risk. One can estimate the outcome and cost of each action from development runs on other tasks, log them prospectively, and compare policies on untouched tasks. The equation defines an optimal decision *given those inputs*. We have yet to estimate a reliable transition model or an optimal policy across benchmarks. Selecting computations by their expected decision value has precedents. Our research question is how the algebra, observability and public requirements of ODE construction change these decisions. [Hay et al., Selecting Computations](https://arxiv.org/abs/1207.5879)

## How evidence changes the search

| What the runtime established | What may be done immediately | Where search continues |
| --- | --- | --- |
| An explicit public identity plus recorded quantities determines a component on the fitting trajectories. | Substitute the identity and reconstruct that component. | Its dependence on latent states and its extrapolation usually remain open. |
| Two measurements must coincide for **all** parameters in a given family, yet differ beyond the registered tolerance. | Eliminate the family and all descendants that preserve the contradiction. Store the witness and assumptions. | Open branches that can break the equality, such as an additional observed dependency or a history-dependent state. For noisy observations use a defensible uncertainty bound. |
| Distinct formulas give identical predictions over the permitted input support. | Share valid calculations and report the inability to distinguish them on that support. | Preserve alternatives if future inputs can separate them. A data-supported tie cannot establish a universal identity. |
| Shape, an LLM or a residual pattern suggests a new graph or equation. | Nominate a candidate and record the cost. | Check it against public constraints and forecasts. A correlation or an optimizer's failure supplies no universal pruning certificate. |
| A fitted candidate fails a sign or trajectory check. | Reject that parameter vector at the affected decision. | Refit only if plausible improvement is worth its cost. Prune the whole family only with a parameter-independent argument. |
| The remaining models pass necessary checks yet predict different trajectories. | Fit under matched budgets and choose using every registered selection event. | Freeze graph, equations, initializers, parameters and decision before validation. Measure failure rate as well as NMSE. |
| The registered public hypothesis itself contradicts measurements under all allowed realizations. | Surface the conflicting hypothesis, evidence and measurement assumptions. | A relaxed hypothesis is a separately labeled branch that requires authorization if it changes the task. |

An identifiable functional form may be fixed if the admissible class, input variation and observability rule out all alternatives within that class. A local curvature signature with several compatible laws instead narrows the candidate set. Store the assumptions with the conclusion so that changing a latent state or adding noise reopens the affected branches.

These decisions work at different levels of the same search. A proposer can introduce a scientifically motivated latent state when the current graph families are exhausted. The algorithm can enumerate alternatives when the choices are bounded, enforce the identities once during generation, and direct computation to unresolved distinctions. Full ODE forecasts bring back couplings that a local derivative or reconstruction cannot resolve.

A practical first comparison can operate without learned probabilities. First apply exact substitutions during generation. Schedule an inexpensive, certificate-producing check before a fit whenever rejection would avoid that fit. Among remaining branches, keep a fixed share of the budget for different scientific hypotheses so that early estimates cannot eliminate all alternatives. Compare this policy with the fixed stage order using the same total fit and proposer budgets. The more ambitious policy would learn the expected change in terminal loss per unit cost from logged attempts, including failed attempts, and decide when another fit, another equation or stopping is worth more. The allocation among branches is an open experimental question.

## An exact decision available today

Suppose $K$ fitted alternatives remain and there are $m$ selection events, equally weighted, with nonnegative loss. A candidate qualifies only if it passes every event. Make the graph and functional form a single candidate. Evaluate in order of increasing fitting NMSE, breaking ties by candidate ID. Use a fixed event order. After one candidate qualifies on all events, let $B$ be the sum of its event losses.

1. For each new candidate, accumulate the selection losses. A failed event disqualifies it.
2. If the partial sum strictly exceeds $B$, its final loss must exceed the incumbent’s. Stop evaluating that candidate.
3. On completing all events, update the incumbent by total loss, then parameter count, then stable ID. Return an abstention if there is no qualified incumbent.

The winner and its score agree exactly with exhaustive selection on the same fitted candidates. The winning model always undergoes all $m$ evaluations. Strict inequality preserves ties. Worst-case cost is $Km$ forecasts plus $O(K\log K + m\log m)$ ordering work (apart from the cost of the forecasts themselves). With an eligible first candidate, as few as $m+K-1$ forecasts may suffice. Sorting by training loss is a heuristic for finding a strong incumbent early. Only the stopping criterion has an optimality guarantee about the selected outcome. If hard constraints involve other events or pooled scores have candidate-dependent weights or denominators, a new valid lower bound is required. The implementation checks the bounded decision against an exhaustive decision on the same receipts.

## Evaluating the workflow as a scientific contribution

1. Freeze the task splits, permitted evidence, grammar, fitter, budgets, forecast loss, coverage measure and tie-breaking rules. Preserve a sealed test split for final claims.
2. Compare a fixed stage order with a policy that chooses the next computation. Give both the same action set and total budget. Also measure the cost to attain the same qualified forecast error. Record actual full-model simulations, optimizer evaluations, proposer calls and wall time separately.
3. Separate the sources of improvement. Compare generation under public constraints with generate-then-check; test certified pruning, heuristic candidate nomination, and adaptive budget allocation separately. Count families eliminated with certificates, failed selections, and valid alternatives lost to heuristic search.
4. Evaluate on multiple benchmark families and unseen tasks. Report the loss–cost and coverage–cost curves, per-task paired results, and remaining nonidentifiability. Include misspecified public requirements and realistic noise as separate experiments. Measurements on one T2 component cannot establish an overall architecture ranking.

The T2 disposal experiment instantiates reconstruction, certified rejection, budgeted fitting, four-event selection, and export of the entire selected ODE. It provides an opportunity to measure which of these decisions affected model quality and which saved computation. [T2 experiment and results](PHASE_C_BALANCED_DISPOSAL.md)
