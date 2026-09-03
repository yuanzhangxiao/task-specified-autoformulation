# Profiled latent-basis affine fitting

`profiled_latent_basis_linear_ridge` extends the fixed-latent-basis backend to
fit a small set of nonlinear latent-shape parameters without exposing true
latent trajectories or latent derivatives.

## Parameter partition

The runtime first derives effective observability from public mappings. An
identity mapping such as `Gp <- Gp` makes that state observed even if the
proposer labeled it latent. A nonidentity mapping such as
`v01 <- k1 * x + k2 * y` does not expose `x` or `y`; they remain latent. The
declared label is preserved for audit, and disagreements are reported.

The runtime then derives the parameter partition from the executable candidate:

- every parameter used by an effectively latent-state RHS is an outer
  latent-shape parameter;
- every parameter used non-affinely in an observed-state RHS or observation
  mapping is an outer parameter; and
- every remaining parameter is an inner affine RHS or observation weight.

For example, in `v01 = gain * sigmoid(shape * z)`, `shape` is outer and `gain`
is inner. In `U = Uii + p_U * X`, `p_U` is an inner observation weight. Outer
parameters may occur nonlinearly. Thus a latent relaxation such as
`dX/dt = -X / tau_x + I` is supported. Conditional on the outer values, every
inner weight must be affine after algebraic-process expansion. Inner weights
cannot occur in denominators, exponents, nonlinear functions, or products with
one another. Parameters remain unsupported in initial-condition expressions.
Each latent initial value is fixed or parameter-free.

## Variable-projection solver

At each bounded outer-optimizer step, the runtime:

1. integrates only the candidate's latent subsystem while conditioning it on
   measured public observed-state paths;
2. constructs one linear system from exact derivatives of identity-mapped
   target states and measured values of nonidentity target mappings;
3. solves the ridge problem for all inner RHS and observation weights; and
4. returns the profiled derivative-plus-ridge residual to the outer optimizer.

Each observed-state derivative block is divided by its training-only standard
deviation before the joint solve, preventing a high-magnitude channel from
silently dominating a multi-output fit.

This is separable nonlinear least squares, or variable projection. It avoids
sending affine weights through the nonlinear optimizer. Across the affine
linear backends, proposer-supplied ranges are initialization/search suggestions
and diagnostic reference ranges by default. Inner weights use the direct ridge
solution and may leave those suggestions; every such estimate is reported. A
`hard` compatibility policy retains rejection or bounded linear least squares
for frozen comparisons. The ordinary bounded-rollout backend is unchanged and
continues to enforce all declared bounds.
Independently supplied hard benchmark, runtime, or deterministic constraints
remain binding under the suggested-range policy; only proposer ranges are
relaxed.

The diagnostic separately reports derivative-equation rows,
observation-mapping rows, linear-system rank and condition number, inferred
observed-state labels, and affine estimates outside suggested ranges.

## Certified reciprocal coordinates

A positive time constant can be represented one-to-one by a positive rate:
`k = 1 / tau`. The runtime now certifies this transformation only when all of
the following hold:

- the declared parameter bounds and initialization range are strictly positive;
- every occurrence of the parameter is the complete denominator of a division,
  such as `x / tau` rather than `x / (tau + 1)`; and
- after replacing each division by multiplication with the reciprocal, the
  expanded state RHS is affine in the reciprocal coordinate.

The outer optimizer then uses bounds `[1 / tau_upper, 1 / tau_lower]`, maps each
trial point back to the candidate's declared `tau`, and reports the
transformation in the fitting diagnostic. Original- and reciprocal-coordinate
conditions draw the same starts in physical parameter space before the latter
is transformed, so their multistart comparison is matched. Unsafe or ambiguous
uses are not rewritten. The feature is separately switchable so its effect can
be measured against optimization in the original coordinate.

This reparameterization does not turn a partially observed latent-dynamics fit
into a closed-form linear solve. Even when `dZ/dt = -k * Z` is affine in `k` at
one instant, the generated path `Z(t; k)` depends nonlinearly on `k`. Therefore
`k` remains an outer variable unless the state path is observed or supplied,
which this method does not assume.

Only exact derivatives of identity-mapped target states are accepted. A fully
latent candidate with a nonidentity observation mapping can instead be fitted
from measured target levels and needs no derivative input. Neither latent
values nor latent derivatives are inputs. Training evidence fits the
parameters; validation and test use ordinary causal rollouts of the complete
ODE model.

## Frozen public pilot

`configs/phase_b_reciprocal_fitting_pilot_v1.json` defines a development-only,
two-benchmark by three-seed comparison. Each of six frozen repaired GPT-OSS
candidates is fitted under three conditions: bounded rollout fitting, profiled
fitting in the original parameter coordinate, and profiled fitting with
certified reciprocal coordinates. The latter two receive exact derivatives of
observed public training channels from a separately versioned trusted-simulator
overlay. Validation derivatives are not supplied. Neither arm receives latent
values or latent derivatives. Only train data is used for fitting, validation
is scored by causal rollout, and test remains closed.

The report keeps fit compatibility, success, train/validation NMSE, function
evaluations, integration failures, wall time, bound contacts, and reciprocal
certification separate. It declares no weighted score or automatic winner.

## Limitations and next step

This is not the full collocation or multiple-shooting method. Latent paths are
candidate-generated conditional trajectories, not independently optimized
nuisance variables. The next search milestone can route separate topology,
functional-form, deterministic-contract, and numerical-fit feedback into staged
candidate revisions. Estimated-derivative and derivative-free profiled fitting
remain later protocol variants and must be evaluated separately. The current
direct ridge solve is closed form in the numerical linear-algebra sense; it does
not impose signs or other hard scientific constraints on affine weights.
