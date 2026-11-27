# When should the constructor screen, fit, or stop?

Fix a finite menu of candidate ODE families, a fitting procedure, a qualification rule, and a compute budget. The output is one qualified fitted ODE or an abstention. The **policy** chooses which computation to run next given what the constructor has learned. A family contradiction is a *possible outcome* of a diagnostic. The policy may choose to pay for that diagnostic. It may remove a family only when the returned evidence proves the contradiction under the stated assumptions.

## Which evidence enters a decision?

| Evidence available at decision time | Consequence |
| --- | --- |
| Public variables, units, balance, allowed inputs and initializer rules | Define admissible families. Apply a known substitution once when generating candidates. |
| Training observations and their uncertainty, together with a family invariant | A contradiction can rule out every parameter vector in that family. A screen without a witness leaves the family open. Count rows or events on which the invariant was testable. |
| Structure shared by candidates, previous fit outcomes on separate development cases, or an informative training signature | Change the estimated probabilities of what a future check or fit would reveal. Probability alone cannot certify a contradiction. |
| A returned parameter vector, qualification status, and scores on predefined training-selection events | Add a qualified fitted model to the available terminal choices, or record that a particular fit failed. Refit using another registered start if the expected benefit pays for it. |
| Actual charges and remaining budget | Limit the affordable next actions and compare the value of spending them now or later. |

In T2 disposal, $D=U-U_{ii}$ is computable at measured times. For a proposed instantaneous $D=f(I)$, identical insulin at two observed times requires identical disposal if the measured arguments are exact. Half the spread of the two disposal values is a lower bound on the largest prediction error. The family is contradicted only when that bound exceeds the allowed error. Adding latent insulin action changes the premise. Two equal insulin measurements supply no such certificate for $D=f(A)$ unless the graph also forces the two $A$ values to coincide. [T2 disposal witnesses](PHASE_C_BALANCED_DISPOSAL.md)

## Optimal policy for a specified decision problem

Let $\omega$ describe the uncertain outcomes of **all** allowed computations and the terminal losses. For example, one simulated scenario can specify which families a check would reject, which candidate fits would succeed under the allocated starts, and how well their fitted equations would forecast. A prior $p(\omega)$ can encode shared graph failures and correlations across candidates. A computation reveals only its registered observation $O_a(\omega)$. Terminal losses for genuinely held-out events remain unknown while constructing a model.

Let $h$ be the history of actions and revealed outcomes, $b$ the remaining budget, and $Q(h)$ the set of qualified, already fitted models. With a finite scenario model and deterministic observations conditional on $\omega$, the posterior redistributes weight among scenarios consistent with the observed results.

$$
p_h(\omega) =
\frac{p(\omega)\,{\bf 1}[\omega\text{ matches every observation in } h]}
     {\sum_{\omega'} p(\omega')\,{\bf 1}[\omega'\text{ matches }h]} .
$$

If real receipts can contain outcomes absent from the finite support, specify a fallback before the run. Preserve physical certificates and actual qualification decisions under that fallback. Training examples used to build or calibrate the scenario model must be separate from held-out policy evaluation. The cost of acquiring any pilot evidence belongs in the comparison.

Assign loss $L_i(\omega)$ to returning fitted model $i$ and $\rho$ to abstaining. These encode the operational value of a forecast, including a failure loss if appropriate. An experiment can use a training-selection score or its discretization as a loss proxy, then measure the actual held-out error and coverage after freezing the decision. The distinction matters whenever in-sample ranking transfers poorly. The posterior risk of stopping is

$$
R(h)=\min\{\rho,\ \min_{i\in Q(h)}\mathbb E_h[L_i]\}.
$$

Each action costs $c_a>0$ budget units with a price $\lambda\ge 0$ per unit in loss units. A hard budget remains in force even at zero price. For every legal affordable action, compute its **cost plus expected best achievable future risk**.

$$
J_a(h,b)=\lambda c_a+
\sum_y P_h(O_a=y)\,V(h\cup\{(a,y)\},b-c_a),\qquad
V(h,b)=\min\{R(h),\min_a J_a(h,b)\}.
$$

Evaluate from the last affordable action backwards. Stop when $R(h)$ is no larger than any $J_a$. Otherwise perform the minimizing action, record its evidence, and repeat. This is optimal in expected terminal loss plus computation price for the specified action set, positive costs, prior, observations and budget. At a terminal history $R$ is achievable by construction. Inductively, conditioning on the next observation gives the exact cost of each possible first action under optimal continuation. Taking the minimum proves the result. There is no distribution-free optimal order when two observationally indistinguishable task populations can assign different prospects to the same fit.

**A fitting example with an exact answer.** Two families are available. Exactly one will succeed, each with probability one half. A successful fit has loss zero, a failed fit leaves only abstention with loss one. A single diagnostic costing one unit identifies which family is contradicted. Fitting either family costs six units. With a budget of seven and price $\lambda=0.1$, stopping costs 1. Fitting blindly costs $0.6+0.5=1.1$ in expectation. Screening and stopping costs 1.1 as well. Screening and then fitting the surviving family costs **0.7**. A one-action lookahead therefore stops, while the optimal policy screens. This constructed example shows why counting how many equations a check rejects, or how much it immediately improves a forecast, is insufficient.

**A useful special case with a simple threshold.** Consider one optional screen before an already planned fit. Suppose fitting without screening performs equivalent final qualification, the screen costs $c_s$, a certificate saves the whole fit cost $c_f$, the certificate probability is $p$, and both workflows return the same terminal decisions. Their expected compute costs are $c_s+(1-p)c_f$ and $c_f$. Screen exactly when $p c_f>c_s$ to reduce expected cost, with indifference at equality. This threshold changes when evidence can also redirect a fit to another family, when the diagnostic is already free, or when selecting a different model has a different loss.

## Computation and approximation

With $n$ one-use actions, at most $r$ observation labels per action, $S$ positive-weight joint scenarios, and maximum horizon $H$, an upper bound on distinct information states is

$$
\sum_{k=0}^{\min(H,n)} {n\choose k}\min(r^k,S).
$$

For any set of $k$ paid actions there are at most $r^k$ possible observed patterns and at most $S$ possible scenario-supported patterns. The bound assumes costs, admissibility and terminal choices depend only on revealed action–outcome pairs. Memoization can merge histories that visit them in different orders. Exhaustive search still grows exponentially with the action count. For a larger menu, truncate lookahead at $d$, use $R(h)$ as the terminal value, and replan after the actual observation. A depth-$d$ tree has at most order $(nr)^d$ leaves per decision. The example above shows why one-step greedy can miss a useful screen.

Two bounds can assess approximate policies in a finite scenario model. The expected loss of an implementable continuation is an upper bound on the optimal risk. Giving a policy all future outcomes free of charge and allowing it to choose a feasible model with this advance knowledge yields a lower bound. Their gap measures residual optimization uncertainty **under the scenario model**. Calibration uncertainty is separate. For example, if all total losses lie in $[0,M]$ and the true law is within total variation distance $\epsilon$ of the planning law on the same scenario space, evaluating any fixed policy under the two laws changes its expected loss by at most $M\epsilon$. Applying this bound to both optimal policies bounds the planning policy's excess risk by $2M\epsilon$. Unbounded NMSE or an unknown $\epsilon$ prevents treating this as a numerical guarantee on the benchmark.

## What is established and what needs testing?

The T2 and basin pilot restricts the policy to two registered families per episode. A separate diagnostic costs one unit. T2 compares a lower bound on disposal error with an accuracy tolerance. Basin intersects the coefficient ranges consistent with downstream depth changes in dry intervals. In the forecast objective, exceeding a fitting tolerance is evidence of tension and leaves the family eligible to compete on forecast risk. In a precision objective, the same certificate rules out the family for that registered requirement. A particular basin fit can also fail the precision gate when its fitted coefficient is outside the allowed range despite a nonempty interval. The basin precision study uses 0.1, 1 and 10 mm accuracy requirements. Fitting costs four units and includes mandatory qualification even if the separate screen was skipped. A fit reveals a failure or a binned training-selection score. A qualified fit is required to recommend an ODE. The primary budget is six units, enough for both diagnostics and one fit. The nine-unit sensitivity permits two fits and a diagnostic. The pilot prices one unit at 0.03 and assigns abstention loss one. These settings describe an experimental decision maker. Actual optimizer evaluations, forecast quality and abstentions must be reported separately.

The pilot constructs its joint outcome model using training evidence from the other benchmark, mixed with a broad independent distribution that allows unseen outcome combinations. Its terminal forecast-risk proxy is the registered loss assigned to the selection-score bin. The primary prior gives weight 0.75 to the other benchmark and 0.25 to the broad distribution. A sensitivity experiment gives the broad distribution all the weight. For the retrospective payoff, each delivered model receives its actual qualified validation NMSE capped at two, and abstention or a failed validation forecast costs one. Joint scenarios can capture dependence between action outcomes even though the broad component factorizes. Official validation events can then assess whether the chosen model's forecast risk transferred. This experiment estimates the value of **acting under a specified prior**. Calibrating that prior and the price of computation for deployment requires more independent development tasks.

Choosing computations by their expected effect on a terminal decision is established Bayesian metareasoning. Our scientific question concerns which ODE-specific certificates, model equivalences, fit outcomes and forecast checks provide useful observations, and whether exploiting them improves the quality and cost of a **complete** model on new tasks. Greedy approximation guarantees from adaptive submodularity require diminishing conditional returns. A screen that makes fitting another family more valuable violates that intuition, as in the example. [Hay et al., Selecting Computations](https://proceedings.mlr.press/r10/hay12a.html) · [Golovin and Krause, Adaptive Submodularity](https://arxiv.org/abs/1003.3967)

Use the same candidate pool, admissibility checks, initialization policy, fitting budget and action prices for competing workflows. Freeze the costs, prior, terminal loss, tie policy and unseen-outcome behavior before selecting on a new episode. Record the policy's predicted action values alongside the revealed evidence. Evaluate qualified coverage, whole-event held-out loss, and actual resource usage. Sensitivity to abstention cost, loss bins, prior weights and transfer between domains reveals when an apparent policy gain depends on the chosen decision problem. [Previous construction framework](PHASE_C_MODEL_CONSTRUCTION_DECISIONS.md)

## What the matched replay established

We used existing optimizer receipts to replay the computations a policy would purchase. We then simulated only the chosen fitted ODEs on official validation, with their parameters frozen. This comparison required no new proposer calls or optimizer runs. Each T2 episode fitted on eight training events and used four other training events for the fit decision. Each basin episode fitted on five training events and used the sixth for the fit decision. The official validation releases contained four and three events respectively. Folds overlap, and both releases have been used in previous studies. The benchmark test split remained unopened.

| Forecast objective | Exact policy | Fixed alternative | Interpretation |
| --- | --- | --- | --- |
| T2, six units | Returned 4/8 models, validation NMSE 0.78820, all-episode loss including cost 1.01410. | Extended-first gave the same outcomes. Simple-first returned 4/8, NMSE 0.73830, loss including cost 0.98915. | The six-unit budget permits just one fit. Across the three episodes where both simple-first and the exact policy returned a model, their respective NMSEs were 0.75894 and 0.79805. |
| Basin, six units | Returned 5/6 models, validation NMSE 0.00002816, loss including cost 0.28669. | Extended-first was identical. Simple-first returned the same five episodes, NMSE 0.08229 and loss including cost 0.35525. | The estimated action values of the two fits were exactly equal. The lexical tie-break chose the three-halves power. This result supplies no evidence for scientific ranking by the policy. |
| Basin, nine units | Requested seven fits across the six episodes. Returned 5/6, validation NMSE 0.00002816. | Fitting both candidates requested twelve fits, with the same returned models. | On five episodes a good first fit made stopping preferable. The sixth triggered a second fit and finally abstention. Replay charges correspond to 942 versus 1,722 archived optimizer evaluations. |
| T2, nine units | Requested all sixteen fits, returned 5/8, NMSE 0.76583. | Fitting both candidates was identical. | There was no saving on these episodes. |

For the primary six-unit experiment the exact policy chose no standalone checks. With the **separate 0.1 mm basin objective**, five of the six fitting panels yielded contradictory coefficient intervals for both powers. On the sixth both intervals were nonempty but their archived fitted coefficients lay outside. All policies abstained. Screening both families first would have requested one fit across all six panels. The policy informed by T2 would have requested six. Thus the comparison contains a real case where the cross-domain estimate of the value of a check was inadequate. The fit price of four times a check's price is an experimental assumption. Measuring actual action costs and estimating their outcomes conditional on the type and support of each diagnostic would be the next test.

### A basin witness a constructor can inspect

Let $x$ be downstream head above the crest. During a dry interval with zero local inflow and provably zero upstream overflow, $A\dot{x}=-k x^p$. Therefore $k=(A/\Delta t)\int_{x_1}^{x_0} x^{-p}\,dx$. Allowing an error of ±1 mm in each measured head gives an interval for $k$. One shared parameter must lie in the intersection across all usable intervals. For the linear law in basin fold 0, observations in train_003 between minutes 96 and 100 require $k\ge40.20$, while minutes 176 to 180 require $k\le23.14$ (bounds rounded outwards). No fitting start can satisfy both. For the three-halves law the intersection on that fold was approximately $[53.31,55.93]$. Passing this necessary test leaves fitting and held-out forecasting to decide how well the law works. The figures assume exact supplied geometry and prescribed inflows, nonnegative outlet gain and the stated tolerance on the measured downstream depth.

This example also identifies three quantities needed to design a better policy. First, estimate the probability that a proposed diagnostic is *informative on this particular dataset*, such as the number and head range of isolated dry intervals. Second, estimate the distribution of qualified whole-event forecast loss after a fit, conditional on the candidate, the diagnostic and any incumbent. Third, measure the incremental cost of each computation, accounting for calculations already shared across candidates. Estimate these quantities from other independent training tasks, reserve separate tasks to assess calibration, and freeze them before a prospective policy comparison. Candidate roles like “simple” and “extended” are a weak substitute for these measurements.

### Reproduction

Use the registered T2 balanced-disposal artifacts, coupled-basin leave-one-training-event-out fit receipts and public release files already present in the experiment checkout. The comparison verifies their hashes. Each stage resumes from immutable receipts.

~~~bash
python -m scripts.compare_evidence_decisions select
python -m scripts.compare_evidence_decisions evaluate
python -m scripts.report_evidence_decisions
~~~

[Results and decision traces](../output/evidence-decisions-20261126/REPORT.html) · [Frozen decisions](../output/evidence-decisions-20261126/choices.json) · [Official validation results](../output/evidence-decisions-20261126/results.json)
