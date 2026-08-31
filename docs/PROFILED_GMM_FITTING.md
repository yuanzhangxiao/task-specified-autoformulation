# Exact-derivative graph-meta-model fitting

## Scope

This is the first, deliberately oracle, milestone for Eqs. (10)-(11) in the
paper. It does not replace the existing bounded rollout fitter. A run must opt
in with:

```text
--parameter-fit-strategy exact_derivative_linear_ridge
```

The data loader tags numerical finite differences and supplied derivative files
as `estimated`. The oracle backend refuses them. It runs only when every training
and validation trajectory carries derivatives explicitly tagged `exact`.

## Division of responsibility

The proposer owns the state graph and each nonlinear basis function
`phi(state, forcing)`. The runtime owns every fitted graph weight `theta`.
Nonlinear shape constants are therefore fixed numeric literals in the proposed
basis. For example:

```text
theta * exp(-0.7 * x)       accepted
theta * exp(-shape * x)     rejected when shape is fitted
```

After recursively expanding algebraic processes, an optimized parameter may:

- multiply a parameter-free basis function;
- appear in sums or differences of such weighted bases.

It may not occur in a denominator, exponent, approved nonlinear-function
argument, product with another optimized parameter, observation mapping, or
initial-condition expression. The restricted AST is checked deterministically
before fitting.

Every modeled state must be directly observed in this first milestone. Its
initial value is read from the first observation and is not optimized. This is
why the oracle milestone does not yet implement the paper's latent-state
proposal path.

## Solver

For each training sample, the runtime evaluates the affine RHS at a legal anchor
parameter vector and at one coordinate probe per parameter. The certified affine
contract makes these differences exact design-matrix columns. Parameter-free RHS
terms become a fixed offset. Stacking all state equations gives

```text
y_adjusted = Phi theta.
```

The runtime computes

```text
theta = solve(Phi.T @ Phi + lambda * I, Phi.T @ y_adjusted)
```

with a least-squares fallback only if the linear solve is singular. This is one
solve (`function_evaluations = 1`), not iterative nonlinear optimization.
The solution is rejected if it is nonfinite or outside proposer-declared bounds;
it is not clipped because clipping would no longer be Eq. (11).

Only training derivatives determine `theta`. Candidate selection still uses
causal train/validation rollout NMSE, and test data remains sealed until the
structure is frozen. The final train-plus-validation refit uses the same
closed-form contract.

## Relationship to the broader profiled-ODE plan

The attached profiled-ODE specification describes a useful later hierarchy:
estimated derivatives, separable nonlinear least squares, collocation, multiple
shooting, and likelihood-based noise models. Those components are intentionally
not part of this milestone. The next numerical milestone should first compare
documented derivative estimators under a frozen protocol, then decide whether
variable projection over a small set of nonlinear shape parameters is warranted.

## Proposer feedback milestone

The controller does not send only NMSE and a final judge score. Active-beam
feedback now contains a `deterministic_runtime` block with public-target
predicates, fit backend and status, optimizer diagnostics, bound hits, and failed
trajectories. Deterministically rejected proposals retain structured diagnostic
codes, locations, and messages. Under paired-question consensus, the existing
incumbent-relative block continues to include every absolute and comparative
judge answer, evidence, and orientation disagreement. The controller prompt
explicitly tells the proposer to repair those individual findings rather than
react only to aggregate scores.

## Known limitations

- No estimated derivatives are accepted.
- No unobserved dynamic state can be fitted.
- Ridge regularization is scale-dependent and is predeclared, not tuned on test.
- Eq. (11) is unconstrained; out-of-bound solutions fail rather than invoking a
  constrained optimizer.
- Fixed nonlinear shape constants are searched structurally by the proposer,
  not refined continuously.
