# T2 disposal with an explicit balance

On the official T2-easy validation events, generating five disposal laws subject to $U=U_{ii}+D$ gave mean whole-model NMSE **0.62216**. Generating the same five laws with a freely fitted coefficient of $U_{ii}$ gave **0.58308**. These means compare the seven graph and selection splits that qualified in both arms. The free arm qualified in all eight. This experiment supports reconstructing and screening a component when its balance is known. It yielded no aggregate forecast improvement from enforcing this balance under the matched budget.

## What the fitting observations identify

The public task describes $U_{ii}$ as the supplied insulin-independent contribution to total disposal $U$ and requires a delayed pathway from plasma insulin to the insulin-dependent contribution. Under the specified balance, the latter contribution is observable at the training sample times as $D=U-U_{ii}$. The latent insulin-action state $A$ remains unobserved. Training and validation forecasts generate $G_p$, $I$, $A$ and $U$ jointly. No future observed target is used to drive the forecasts.

On one eight-event fitting fold, train_001 and train_011 share $I(0)=25.5878134643$ pmol/L. They have $D(0)=0.9157859801$ and $0.9890358777$ mg/kg/min. The best common prediction for these two observations still misses one by at least half their difference, **0.036625**. The other fold has a witness of **0.040357**. If measurement uncertainty tolerates these differences, this evidence alone does not reject the approximate equation.

| Candidate for $D$ | Minimum proven maximum absolute error, first / second fold (mg/kg/min) | What is identified |
| --- | --- | --- |
| One shared constant | 4.71127 / 3.58599 | A time- and event-invariant value cannot reproduce disposal. |
| Instantaneous $f(I)$ | 0.03662 / 0.04036 | Insulin alone at the current time cannot recover exact disposal when measured arguments are reproduced. |
| Instantaneous $f(G_p)$ or $f(G_t)$ | 0 / 0 | Equal-argument comparisons supply no positive lower bound. A complete candidate must still realize the required delayed pathway. |
| $f(A)$, $f(A,G_p)$, $f(A,G_t)$, $f(A,G_p,G_t)$ | 0 / 0 on both graphs | All survive. A particular glucose argument is not shown to be necessary. |

For the latent argument, certification requires equality of the action initializer at the initial time or equality of the upstream initial states and full forcing histories thereafter under a unique deterministic solution. On the action-only hypothesis only 26–147 of 2,408 fitting rows per graph and fold had a repeated certified argument. A zero lower bound here frequently reflects the small number of informative comparisons. [Dependency results](../output/balanced-disposal-20261126/dependencies/summary.json)

## Matched experiment

Use the two frozen T2 parents with algebraic $U$ (graph02 and graph03). Each graph uses eight events for fitting. The two fitting folds are complementary halves of the 16 training events. Among the other eight events, four choose a law and four assess it. Two rotations exchange the latter sets. All four selection events count equally, and a failed event disqualifies that candidate. The graph, formula, initialization and fitted parameters freeze before scoring on the four official validation events. These four events have been used in earlier benchmark experiments. They remain separate from the selection events in this experiment. The sealed test set was left unopened.

With $P$ and $T$ representing plasma and tissue glucose standardized using the fitting events, define $L=b+kA$. Generate five laws for $D$.

| Law | Disposal |
| --- | --- |
| Insulin action | $L$ |
| Plasma glucose | $L+pP$ |
| Tissue glucose | $L+tT$ |
| Both glucose measurements | $L+pP+tT$ |
| Interaction | $L+tT+jAT$ |

Pair each balanced law with a free readout having the same dependencies, an intercept and a fitted coefficient on $U_{ii}$. The parent readout is preserved in the reference slot (action for graph02, both glucose measurements for graph03). All other equations and start policies are paired. The glucose term uses predicted $G_p$ during forecasting. Each candidate received two starts, each capped at 500 evaluations and 90 seconds, with the same sampled nonnegativity constraints on $G_p,I,U$. Both arms thus had 20 fitted models and 20,000 allowed optimizer evaluations. Actual use was 19,519 (balanced) and 19,520 (free), with 39/40 fitted starts in each arm and at least one qualified fit for all 40 models. $D<0$ was separately audited. Across all opposite-training-event forecasts there were such samples for some models, even with the explicit identity. On official validation all selected models kept $D\ge-10^{-9}$ on the scoring grid.

| Split and comparison | Qualified in free / balanced | Common selections | Free NMSE | Balanced NMSE | Free / balanced $U$ NMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Omitted training events, five laws each | 7/8 / 5/8 | 5 | 0.43003 | 0.48905 | 0.45912 / 0.43286 |
| Official validation, five laws each | 8/8 / 7/8 | 7 | 0.58308 | 0.62216 | 0.49066 / 0.53864 |
| Official validation, only the original dependency choice in both arms | 8/8 / 6/8 | 6 | 0.84463 | 0.83392 | 0.87250 / 0.86511 |
| Official validation, original free law versus five balanced laws | 8/8 / 7/8 | 7 | 0.84547 | 0.62216 | 0.85449 / 0.53864 |

NMSE is the mean of the post-initial squared errors normalized by fitting-fold target variance, averaged equally over the three targets and then the events. Each row's two means use its own common qualifying selections. In the equal-portfolio comparison balanced won four of seven paired validation selections. On graph03, fold 1, rotation 0, the additive balanced law selected on training scored 1.14953 while the free law with an interaction scored 0.78314. The magnitude of this loss matters more for mean NMSE than the number of wins. One balanced selection on graph02 abstained because all five laws failed qualification on a selection event. Every candidate actually sent to official validation passed there. Rotations reuse the same four validation events and some fitted models; seven paired selections do not constitute seven independent experiments.

Coverage changes the comparison if abstaining has an assigned cost. Across all eight decisions the free arm's total validation loss was 5.31431. The balanced arm's was 4.35510 plus the cost of its one abstention. The methods break even at an abstention cost equivalent to 0.95922 NMSE. A lower cost favors the balanced policy on these eight decisions, and a higher cost favors the free policy. This sensitivity calculation uses the frozen decisions. Future policy comparisons should register the cost or minimum coverage in advance.

Across **every fitting and validation event** $U_{ii}=1$. For a free readout $U=aU_{ii}+b+F$, the balanced readout $U=U_{ii}+(b+a-1)+F$ gives identical predictions on this support. This equality holds for any choices of the other state variables, including during endogenous simulation. The experiment can distinguish disposal dependency choices and optimizer endpoints. It cannot empirically determine the $U_{ii}$ coefficient, attribute a forecast gain to that coefficient, or forecast what would happen if $U_{ii}$ varied. The identity can still be imposed from the task interpretation. [Forecast curves, scores and chosen formulas](../output/t2-balanced-disposal-20261126/REPORT.html)

## Selecting the final graph and its equations

A second decision pooled the ten balanced graph-and-law choices per split and used only training events to select a single executable ODE. Each export contains the graph, all governing equations, causal initializers and fitted coefficients. All four chose graph03.

| Eight-event fitting fold / selection rotation | Chosen arguments in $D$ | Selection NMSE | Official validation NMSE |
| --- | --- | ---: | ---: |
| 0 / 0 | $A,G_p$ | 0.38670 | 0.34786 |
| 0 / 1 | $A,G_p,G_t$ | 0.23331 | 0.32445 |
| 1 / 0 | $A,G_p,G_t$ | 1.89850 | 1.14953 |
| 1 / 1 | $A$ | 0.73128 | 0.67638 |

For example, the first row gives $U=U_{ii}+1.91450+0.15792 A+1.71528(G_p-207.16691)/53.31256$. The complete model specifies how $A$ evolves and how to initialize it. The mean official NMSE of the four frozen graph-and-law selections was 0.62455. No validation-based choice of one deployment model was made. [Example complete ODE](../output/t2-balanced-disposal-20261126/model-choice/fold0-rotation0.json) · [All four choices](../output/t2-balanced-disposal-20261126/model-choice/summary.json)

This pool allowed an exact test of computational savings. The objective sums four nonnegative event losses. Once a candidate's partial sum exceeds the total for a fully qualified incumbent, no remaining event can let it win. Replaying this bound picked exactly the exhaustive winner in all four decisions using **114/160** candidate–event evaluations. Stopping only on the first failed qualification needed 147. Thus 13 were avoidable from failure alone and a further **33** from the loss bound. The run computed all curves to verify the result, so these are avoidable selection forecasts in replay, with no claimed reduction in fitting time. [Decision framework](PHASE_C_MODEL_CONSTRUCTION_DECISIONS.md)

## Reproduce

The commands require the frozen public training packets and validation intake from the matched-equation experiment in this checkout. Use the project Python environment. The scripts check registered identities and resume from content-addressed receipts.

~~~bash
python -m scripts.screen_t2_disposal_dependencies
python -m scripts.export_t2_disposal_models plan
python -m scripts.run_t2_balanced_disposal --workers 4
python -m scripts.report_t2_balanced_disposal plan
python -m scripts.report_t2_balanced_disposal select
python -m scripts.export_t2_disposal_models export
python -m scripts.report_t2_balanced_disposal internal
python -m scripts.report_t2_balanced_disposal freeze-validation
python -m scripts.report_t2_balanced_disposal validate
python -m scripts.report_t2_balanced_disposal report
~~~
