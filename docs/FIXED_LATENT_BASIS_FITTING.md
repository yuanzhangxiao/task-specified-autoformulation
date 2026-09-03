# Fixed latent-basis affine fitting

`fixed_latent_basis_linear_ridge` is the first partially observed extension of
the Eq. (10)-(11) affine-weight fitter. It does not expose true latent states or
latent derivatives.

## Contract

- Exact derivative labels are required for directly observed dynamic states.
- Every fitted parameter must enter the expanded state equations affinely.
- Latent dynamics contain proposer-owned numeric shape constants and no fitted
  parameters.
- Every latent state has a fixed or parameter-free analytic initialization.
- Every observed dynamic state has an identity observation in the public data.
- Proposer bounds remain enforced. A bound contact is reported as an advisory
  scale or identifiability diagnostic, not as proof of a structural error.

## Fitting and evaluation

For each training trajectory, the runtime integrates only the candidate's
parameter-free latent subsystem while conditioning it on the measured observed
state path. This constructs candidate-specific latent basis functions without
opening a private reference. It then evaluates the observed-state right-hand
sides at an anchor and one coordinate probe per affine parameter, constructs the
linear design matrix, and solves the ridge normal equations once.

Validation and test metrics are computed by the existing rollout evaluator. No
derivative labels, fitted latent paths, or private information are used in model
selection or held-out scoring.

## Scope and next milestone

This backend is appropriate when the proposer can choose scientifically useful
latent basis dynamics in advance. It cannot fit a latent time constant such as
`tau` in `dX/dt = -X / tau + I`, because that would make the latent basis depend
on an optimized parameter. Variable projection is the planned extension for
those parameterized latent dynamics: outer optimization would update the small
set of nonlinear latent-shape parameters, while the inner affine weights remain
a deterministic linear solve.
