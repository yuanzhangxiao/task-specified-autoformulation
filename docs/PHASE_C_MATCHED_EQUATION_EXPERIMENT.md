# Can trajectory shape guide equation construction on a fixed graph?

We compared constructors on the same six T2-easy graphs. We selected one graph from each combination of three variable inventories and two graph origins using the archived order, before scoring formulas. Each graph had two complementary splits of the 16 public training events. Each constructor received eight events and two formula slots per split. We fitted both slots, selected the lower verified training NMSE, and forecast the other eight events with fixed coefficients. The graph search had previously seen all 16 events. This study measures generalization of equation construction and fitting on a fixed topology. The official validation and test sets were unopened.

The original constructor scored a small family of affine, saturating and product equations. The new constructor ranks analytic functions using values of algebraic variables and changes in differential states over several time windows. A third arm retains the new constructor's first model and gives GPT-OSS-20B the second slot. It may choose the second numerical nominee or submit its own complete model. The numerical constructors receive no scientific text. All arms inherit the same prior graph, initializer choices and fitting bounds for glucose, insulin and utilization. [Algorithm 1](PHASE_C_GENERAL_EQUATION_CONSTRUCTOR.md)

## Three constructors on T2-easy

| Outcome | Original | Trajectory-informed | Trajectory-informed + 20B |
|---|---:|---:|---:|
| Verified fitted formulas out of 24 slots | 24 | 22 | 18 |
| Graph/fold combinations with a held forecast out of 12 | 12 | 11 | 11 |
| Graph/fold combinations with all bounded targets nonnegative at scored times out of 12 | 5 | 9 | 9 |
| Graphs with nonnegative forecasts in both folds out of 6 | 0 | 3 | 3 |

The new constructor had lower held NMSE in six of 11 comparable splits and in two of five graphs where both methods produced forecasts for both folds. On those five graphs, its median paired change in NMSE was **+0.0134**, where lower is better. For one graph and split neither new formula produced a verified training fit. The optimizer found no feasible start or subsequent feasible vector in that run. This leaves the feasibility of the equations open.

In one concrete glucose equation the graph permitted utilization, insulin-independent utilization and meal rate. On the fitting events, the new constructor compared an affine RHS to an affine RHS plus a hinge in utilization. The hinge reduced omitted-event RMSE for the average rate of change of glucose from 1.000 to 0.967. Its threshold of about 2.23 in the original utilization units was the median on the fitting events. Across the eight independently forecast events, the selected model improved glucose NMSE from 1.300 to 0.429. Averaging glucose, insulin and utilization gave the opposite result, from 1.311 to 1.360. Other equations and their fitted coefficients could also differ between these two models. [Reproduce the example](../scripts/plot_equation_nomination_example.py)

Of 52 equations in the first nominees across all splits, only 14 had a measured response and all declared sources measured. For 22 the constructor could score the observed part, and for 16 it had to rely on simulation of the complete model. Its first nominees contained 47 affine equations and five hinges. The restricted search and missing observations both limit what shape information can select.

### What the 20B proposer contributed

All 12 ACES requests returned normally (job `2415338`, 63,559 total tokens). Six selected numerical candidate B and one rewrote B exactly, including its coefficients. Five proposed alternatives failed validation. One reproduced all of B's equations and initializers but omitted its parameter declarations. A second omitted coefficients and added an undeclared glucose self-dependency. One changed the permitted glucose sources, one repeated A, and one produced an initializer in which its required EGP dependence canceled. No newly authored equation system survived.

With A always retained, the hybrid matched the numerical method in ten of the 11 comparable held forecasts. In the remaining split (`graph01-fold1`), the invalid 20B reply left only A and increased pooled NMSE from 1.360 to 1.503. Both methods lacked a training winner on the twelfth split. This experiment tested a single 20B draw per graph/fold with a two-slot maximum. A next comparison can keep both numerical nominees and ask 20B for a targeted change to just one equation, giving the algorithm-only control an equal third slot. For terms with hidden sources, compare complete candidate rollouts on inner held events before promoting a change. This would directly test scientific edits and interactions that local observed-rate errors cannot distinguish.

## Portability to coupled detention basins

The same code ran on both archived basin graphs. Each graph used two complementary three-event fitting and three-event scoring splits, two formulas per arm and the same optimization budget as T2. This separate control supplied no discharge formula, balance, crest, depth-sign constraint or hydraulic factorization.

| Downstream depth NMSE on omitted events | Original | Trajectory-informed |
|---|---:|---:|
| Graph A, first split | 5.80 | 837.00 |
| Graph A, second split | 61.21 | 141.52 |
| Graph B, first split | 2,946.46 | 4,174.82 |
| Graph B, second split | 4,644.58 | 4,649.44 |

All 16 fits finished. The new constructor reduced training NMSE in **all four** comparisons and increased held-event error in **all four**. Every selected model predicted some negative downstream depths. Both graphs gave the downstream equation an unobserved upstream depth as a source and omitted a direct dependence on downstream depth or downstream inflow. The result establishes that the software transfers between these tasks. It gives no evidence of reliable basin forecasts on these two graphs. The earlier basin pilot supplied balances and a crest and answers a different question.

## Does the constructor recognize a known shape?

We generated independent synthetic training and forecast events for a measured scalar ODE driven by two step-held inputs. The simulator supplied noise-free trajectories on irregular time grids. The constructor received the graph, channel values and interpolation convention. The simulator's law and parameters were available only to the experiment's evaluator. We used eight fixed random seeds for each law, six fitting events and six forecast events per seed. We scored the first nominee with its estimated starting coefficients, with no subsequent fitting.

| Actual shape of forcing term | First nominee recovered the family and source | Median held NMSE |
|---|---:|---:|
| Affine | 7/8 | 0.00000202 |
| Hinge | 8/8 | 0.000190 |
| Smooth saturation (tanh) | 8/8 | 0.000200 |
| Square | 8/8 | 0.00000215 |
| Two-source product | 8/8 | 0.00000226 |
| Sine of a combination of the inputs, outside the menu | 0/8 | 0.856 |

This measures recognition given a correct graph, informative forcing, complete state measurements and a restricted list of possible shapes. With the sine control all eight first choices were affine. A low local score or a best available candidate does not certify the functional form. [Synthetic evaluation](../scripts/check_constructor_recovery.py)

## Reproduction

The registered graph, trajectory and fit-constraint archives must be present under the project's existing output directories. The scripts check their identities and preserve per-slot receipts.

```bash
.venv/bin/python -m scripts.matched_equation_experiment prepare
.venv/bin/python -m scripts.matched_equation_experiment fit --arms original shape
.venv/bin/python -m scripts.matched_equation_experiment score --arms original shape
.venv/bin/python -m scripts.audit_equation_constructor_names
.venv/bin/python -m scripts.check_constructor_portability
.venv/bin/python -m scripts.check_constructor_recovery
```

The matched run archives the 12 raw responses under `output/matched-equation-construction-20261125/hybrid/generate-responses`. With those receipts in place the hybrid fits, scoring and final report can be replayed as follows.

```bash
.venv/bin/python -m scripts.matched_equation_experiment fit --arms hybrid
.venv/bin/python -m scripts.matched_equation_experiment score --arms hybrid
.venv/bin/python -m scripts.report_matched_equation_experiment
.venv/bin/python -m scripts.plot_equation_nomination_example output/matched-equation-construction-20261125
```

The opaque-name check replaced every scientific variable name in all 12 T2 inputs. All 24 proposed equation systems retained their analytic expressions and numerical starts after undoing the renaming, including when the opaque keys were resorted. The order of sources within each equation was preserved.

[Matched experiment](../scripts/matched_equation_experiment.py) · [Forecast evaluation](../scripts/matched_equation_forecasts.py) · [Basin control](../scripts/check_constructor_portability.py) · [Name audit](../scripts/audit_equation_constructor_names.py)
