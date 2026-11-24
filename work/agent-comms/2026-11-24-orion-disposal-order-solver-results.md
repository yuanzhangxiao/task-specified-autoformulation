# Disposal ordering as a fitting constraint and as a screen

**For Astra.** Keep the direct time-zero constraint check. In this fixed delayed model, projecting the starting vector onto the feasible set did not improve the fit or reduce the cost of solving it. A short free rollout did save work when screening an already specified collection of parameter vectors.

## Matched fits

Both solvers used the same archived delayed equation, frozen training-only insulin law, 16 public training events, two raw starts, scaled coordinates for computing the nearest projection, physical coefficient bounds, midpoint domain checks, and limit of 80 forward evaluations per start. We kept the previously published unrestricted, 1% and 50% fits as references. The fitting objective averages the training-variance-normalized squared errors of Gp, I and U with equal target weight. Terminal qualification uses an independent full rollout. Official validation and test remained closed.

The original solver uses SLSQP with explicit linear inequalities. In the second solver, evaluating the declared initializer and algebraic readout at time zero gives an affine map in the free coefficients. Its predictions agreed with the compiled model within 6.7 × 10⁻¹⁶ mg/kg/min over four checked vectors per start. For this model and training set, two independent inequalities imply all 29 pair inequalities. We verified that reduction and used distances from those two boundaries as bound-constrained coordinates. SLSQP then runs from the closest feasible point. Both methods use the same full Jacobian at the archived vector to set their coordinate scales. That vector violates the 50% rule. The projection itself takes about half a millisecond.

| Constraint and solver | Start | NMSE Gp / I / U | Pooled NMSE | Max abs U error, 0 / 15 min | Worst 50% gap deficit, 0 / 15 min | F / J calls | Event rollouts F / J | Fit time (s) |
|---|---|---|---:|---|---|---|---|---:|
| Unrestricted, reference | Warm | .03037 / .12679 / .12583 | .09433 | 2.093 / 3.370 | 3.005 / 3.518 | 1 / 1 | 16 / 16 | 0.27 |
| Unrestricted, reference | Displaced | .03037 / .12679 / .12583 | .09433 | 2.093 / 3.370 | 3.005 / 3.518 | 29 / 27 | 464 / 432 | 4.76 |
| 1%, reference | Warm | .15485 / .12679 / .18747 | .15637 | .092 / 1.716 | .075 / 2.091 | 23 / 18 | 368 / 288 | 3.21 |
| 1%, reference | Displaced | .15485 / .12679 / .18747 | .15637 | .092 / 1.716 | .075 / 2.091 | 24 / 19 | 384 / 304 | 3.42 |
| 50%, reference | Warm | .16421 / .12679 / .19129 | .16076 | .051 / 1.676 | .000 / 2.046 | 22 / 17 | 352 / 272 | 3.05 |
| 50%, projected | Warm | .16421 / .12679 / .19129 | .16076 | .051 / 1.676 | .000 / 2.046 | 36 / 25 | 562 / 400 | 4.53 |
| 50%, reference | Displaced | .16421 / .12679 / .19129 | .16076 | .051 / 1.676 | .000 / 2.046 | 25 / 19 | 400 / 304 | 3.28 |
| 50%, projected | Displaced | .16421 / .12679 / .19129 | .16076 | .051 / 1.676 | .000 / 2.046 | 39 / 28 | 610 / 448 | 4.86 |

Disposal errors and deficits are in mg/kg/min. Deficit is the positive part of the worst shortfall from half of the observed positive gap. This common diagnostic can be positive for fits with a looser training constraint. The worst signed *enforced* 50% margin across the four fits was −3.33 × 10⁻¹⁶, within the 10⁻⁹ feasibility tolerance. All eight endpoints passed their own constraints. F counts evaluations of whole-horizon residual batches and J counts whole-horizon Jacobian evaluations, including setup. Event counts are actual event integrations started. Each projected start had one numerically unstable trial that ended after two events, and its line search recovered. No fit called the minute-15 partial simulator. The independent endpoint audit adds 16 partial and 16 matched full simulations per vector, plus 32 higher-resolution full event simulations and 16 scalar insulin checks.

The constrained wall times are medians of three serial runs per start, with alternating execution order. The two comparison controls were timed once. Setup / projection / solver times in seconds were .276 / 0 / 2.774 and .250 / .0006 / 4.280 at the warm start, and .264 / 0 / 3.015 and .241 / .0005 / 4.624 at the displaced start (reference then projected). Timing medians are taken independently, so components can differ slightly from the total. The reference had 3 and 6 order-violating forward evaluations. The projected fits had zero, but projection moved each start to a poor trajectory fit (pooled NMSE 9.11 and 4.66). After the common scaling Jacobian, they also used zero order-violating Jacobians. Both approaches converged to equivalent losses within 2 × 10⁻¹² for either raw start. These results support a reproducible tradeoff for these starts and solvers. A global lower bound on trajectory error remains open.

## Minute 15 and construction feedback

We used the time proposed in Astra's handoff. At minute 15 the observed disposal differences exceeding 0.01 mg/kg/min define 97 ordered pairs. This tolerance is a diagnostic setting. Each 50% fit missed the diagnostic half-gap on 16 of the 97 pairs, all with reversed predictions. The worst pair was train_011 versus train_005. Observed disposal was 0.041 higher on train_011. The free forecast put it 2.026 lower. Supplying measured Gp at minute 15 and integrating action from recorded insulin reduces the worst absolute U error from 1.676 to 0.439 across all events, and the worst half-gap deficit from 2.046 to 0.581. That conditional calculation still reverses this pair. Both the propagated states and the algebraic disposal readout deserve inspection.

Relative to the unrestricted control, imposing the initial order hurts train_005 most in both Gp and U, followed by train_004. For train_005 the 50% fit predicts glucose 265.95 at minute 170 against 179.95 observed (unrestricted 187.81 mg/kg). At minute 122 it predicts disposal 6.836 against 9.974 observed (unrestricted 8.543 mg/kg/min). Fixing the initial order improves initial levels while leaving a sizeable error during the meal response. Use these event, time and target comparisons in construction alongside the separate no-meal equation analysis. The direct order constraint contains no absolute-level or timing information.

As a separate fixed-vector experiment, we applied the minute-15 half-gap as a hypothetical rejection rule to the six archived unrestricted, 1% and 50% endpoints. Both the partial and full implementations rejected all six, with identical per-event U(15) values and pair decisions. The partial path required 96 prefix integrations and zero full integrations (11,568 RK4 steps, 0.018 s). The full path required 96 full integrations (231,696 steps, 0.121 s). This saving depends on the rejection rate. No endpoint selection or fitting decision used the minute-15 rule.

[Per-start results and event differences](../../output/t2-order-solver-comparison-20261124/summary.json) · [Complete fitted equations and vectors](../../output/t2-order-solver-comparison-20261124/fitted-models.json) · [Partial versus full screen](../../output/t2-order-solver-comparison-20261124/fixed-vector-screen.json) · [No-meal term attribution and suggested balance](2026-11-24-orion-disposal-dip-equations.md)

Run or resume from the project checkout using the existing environment and training artifacts.

```sh
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m scripts.run_t2_order_solver_comparison
```
