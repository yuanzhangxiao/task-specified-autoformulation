# Orion handoff · Fitting with disposal-order constraints

**November 24, 2026 · T2-easy · public training events**

Orion, please test whether checking disposal order before expensive trajectory evaluations saves work while recovering equally good feasible fits. Start with the existing delayed model and the same training objective and parameter bounds. Your persistent-versus-fading holdout comparison is complete. The new experiment concerns how to fit a fixed model. [Previous fit and audit](../work/agent-comms/2026-11-23-initial-disposal-order-handoff.md) · [Completed holdout comparison](../work/agent-comms/2026-11-24-orion-initial-memory-holdouts.md)

## What we already know at time zero

The disposal equation is affine in plasma glucose, tissue glucose, insulin action and supplied insulin-independent disposal. In the original models, initial action is a shared affine function of initial insulin. Substituting that initializer into the equation lets us evaluate initial disposal for every event without integrating an ODE.

Within those equations, one shared choice of coefficients gives the initial prediction

\[
\widehat U_j(0)=1.33223250+0.00446056358\,G_{t,j}(0).
\]

We chose the slope and intercept to minimize the largest absolute error using only the initial observations. The resulting maximum error across 16 events is 0.001866 mg/kg/min. There are three distinct observed levels. The middle event's model inputs are exactly the midpoint of the two extremes, yet its measured disposal is 0.003732 above their observed midpoint. An affine equation preserves midpoints, so at least one initial prediction must miss by half that amount or more. The displayed line attains that bound to rounding. Initial plasma and tissue glucose also sum to a constant across all 16 events. These observations leave the physiological source and the later dynamics unidentified. [Saved predictions and coefficients](../output/t2-initial-order-20261123/audit.json) · [Calculation](../scripts/t2_initial_disposal_family.py)

## First experiment · enforce order at time zero

For each pair with distinguishable observed initial disposal, label the higher event \(h\) and the lower event \(l\). Write \(\Delta_{hl}=U_h^{\rm obs}(0)-U_l^{\rm obs}(0)>0\). Keep the trajectory loss as the objective and impose a separate inequality

\[
\widehat U_h(0;\theta)-\widehat U_l(0;\theta)\ \ge\ \rho\,\Delta_{hl}.
\]

1. **Reproduce the reference.** The existing SLSQP fit already enforces the 50% rule (\(\rho=0.5\)) as a hard constraint on the retained solution and checks it by independent replay. It still computes a whole-trajectory residual and Jacobian while setting up the initial constraints. Start with this same 50% problem to compare costs. The 50% margin is an experimental setting. In noisy data, define which observed gaps are distinguishable before constructing inequalities. [Reference fitter](../scripts/fit_t2_initial_disposal_order.py)
2. **Build constraints without a rollout.** With the action gain held fixed as in the reference, the initial pair differences are affine in the free coefficients. Construct the matrix from the declared initializer and disposal equation. Verify its values against direct initial predictions at several parameter vectors. Find the nearest feasible start to each existing start in the same scaled coordinates by minimizing squared distance subject to the linear inequalities and parameter bounds. If other coefficients are freed and produce products, recompute the constraint type.
3. **Compare at matched settings.** Use the same raw starts, training events, parameter bounds, domain checks and simulation budget. Record projection and solver time as well as the number of full trajectories and Jacobians evaluated. SLSQP can evaluate infeasible trial vectors. For an arm designed to avoid such evaluations, use a feasible start and an optimizer that preserves the linear inequalities, e.g. SciPy `trust-constr` with `LinearConstraint(..., keep_feasible=True)` and analytic derivatives. Instrument actual objective calls. Independently replay the selected vectors and report the worst signed margin with numerical tolerance. Keep the reference fits for the unrestricted, 1% and 50% problems for interpreting the accuracy tradeoff. [SciPy linear constraints](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.LinearConstraint.html) · [SciPy constrained minimization](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html)

## Second experiment · check a later time

Start with one declared early time on the training grid, such as minute 15. Determine the observed order **at that time**. Omit ties and gaps within the chosen measurement tolerance. Two trajectories can cross after time zero.

- When all arguments of an algebraic equation are supplied at that time, evaluate it directly. In this model plasma glucose and latent insulin action are generated. A cheap diagnostic can insert recorded training glucose and reconstruct action from recorded insulin history using the action ODE. Label the resulting error as conditional on those measurements. A free forecast can differ.
- For a check on the actual model prediction, propagate the required states of both events through the selected time using their permitted initial values and forcing. Compare the result against the same time in a full replay. If a fixed vector violates an explicit hard constraint beyond numerical tolerance, its remaining trajectory cannot change that earlier prediction. A solver that needs values or derivatives at infeasible trials still needs a consistent evaluation strategy. Benchmark skipping the rest first in candidate screening or in a solver designed for feasible steps.
- Initially report later-time order **and absolute errors** as diagnostics. A common shift of every predicted value can preserve every ordering. In the completed fading-model forecast for omitted event `train_011`, predicted disposal at minute 9 was 1.175 against 1.997 mg/kg/min observed. This explains why native-unit errors matter. The previously inspected held-event result should stay separate from selecting checkpoint times, tolerances or model parameters.

## What to return

Please send one table with trajectory NMSE by target, worst initial and later disposal errors in mg/kg/min, worst order violation, full and partial simulations, Jacobian evaluations, and wall time per start. Include the saved parameter vectors and a reproducible comparison of partial and full replay on the same vectors. Use the fitting fold for preparing coefficients, scales, constraints and starts, and keep official validation and test closed.

If a better optimizer finds a feasible fit close to the unrestricted score for the same equation, improve fitting first. If several qualified starts and solvers reproduce a substantial tradeoff, identify which events, times and targets deteriorate and send those to construction as evidence for a specific revision. That tradeoff is empirical. A proof that the equation must fail requires a bound that applies to every allowed parameter vector.
