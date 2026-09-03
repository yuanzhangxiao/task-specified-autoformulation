# Profiled latent-basis affine fitting

`profiled_latent_basis_linear_ridge` extends the fixed-latent-basis backend to
fit a small set of nonlinear latent-shape parameters without exposing true
latent trajectories or latent derivatives.

## Parameter partition

The runtime derives the partition from the executable candidate:

- every parameter used by a latent-state RHS is an outer latent-shape
  parameter;
- every remaining parameter is an inner affine RHS weight.

Outer parameters may occur nonlinearly. Thus a latent relaxation such as
`dX/dt = -X / tau_x + I` is supported. Conditional on the outer values, every
inner weight must be affine after algebraic-process expansion. Inner weights
cannot occur in denominators, exponents, nonlinear functions, or products with
one another. Parameters remain forbidden in observation mappings and initial
conditions. Each latent initial value is fixed or parameter-free.

## Variable-projection solver

At each bounded outer-optimizer step, the runtime:

1. integrates only the candidate's latent subsystem while conditioning it on
   measured public observed-state paths;
2. constructs the exact-observed-derivative design matrix for the inner weights;
3. solves the bounded ridge problem for those weights; and
4. returns the profiled derivative-plus-ridge residual to the outer optimizer.

Each observed-state derivative block is divided by its training-only standard
deviation before the joint solve, preventing a high-magnitude channel from
silently dominating a multi-output fit.

This is separable nonlinear least squares, or variable projection. It avoids
sending affine weights through the nonlinear optimizer and makes proposer
bounds explicit constraints instead of rejecting an otherwise useful structure
after an unconstrained solve.

Only exact derivatives of observed states are accepted in this milestone.
Neither latent values nor latent derivatives are inputs. Training derivatives
fit the parameters; validation and test use ordinary causal rollouts of the
complete ODE model.

## Limitations and next step

This is not the full collocation or multiple-shooting method. Latent paths are
candidate-generated conditional trajectories, not independently optimized
nuisance variables. The next search milestone can route separate topology,
functional-form, deterministic-contract, and numerical-fit feedback into staged
candidate revisions. Estimated-derivative and derivative-free profiled fitting
remain later protocol variants and must be evaluated separately.
