# Optional CasADi nonlinear initialization

The production fitting path remains the existing typed fitter and causal rollout
evaluator. CasADi is an opt-in, training-only initializer for candidates whose
nonlinear parameter search is difficult from generic runtime starting points.

## Numerical boundary

The initializer:

- translates only expression trees that already passed the restricted parser;
- uses direct multiple shooting with fixed-step RK4 dynamics and IPOPT;
- optimizes global parameters and internal latent shooting states;
- uses only public training targets and supplied training inputs;
- never receives validation/test trajectories, latent labels, or latent
  derivatives;
- returns a parameter starting point to the existing fitter; and
- records its objective, iteration count, elapsed time, and parameter estimates.

The existing fitter then performs the configured affine/profiled/nonlinear solve.
The existing causal evaluator remains authoritative for train/validation metrics.
CasADi therefore does not change model-selection evidence or test isolation.

The initializer is disabled by default. Enable it with
`--nonlinear-initializer casadi_multiple_shooting`. The `continue` failure policy
falls back to ordinary runtime starts; `raise` fails closed for controlled
validation experiments.

## Qualitative parameter domains

New proposer responses declare a qualitative parameter role rather than numeric
bounds. The runtime maps roles to domains:

| Role | Runtime domain |
| --- | --- |
| `rate`, `time_constant`, `scale`, `positive_shape` | positive |
| `nonnegative_coefficient` | nonnegative |
| `coefficient`, `offset`, `shape` | real |

These are sign/domain constraints, not magnitude guesses. Trusted benchmark or
runtime constraints may further restrict a domain. Historical numeric proposer
ranges remain readable for deterministic replay but are not part of the new
proposer contract.

For affine parameters, unconstrained real coefficients continue to use the
closed-form least-squares solve. A qualitative nonnegative/positive domain uses
the existing bounded linear least-squares solve. CasADi is used only for the
optional nonlinear initialization stage.

## Current milestone limitations

- CasADi initialization currently supports global parameters only.
- The symbolic initializer enforces parameter domains and trusted parameter
  constraints; final state-constraint validity is still decided by the existing
  causal evaluator.
- Long trajectories are deterministically thinned for initialization only. The
  final fitter and evaluator still use the complete configured data.
- CasADi/IPOPT may consume more memory per fit than SciPy. It should be evaluated
  as an adaptive rescue path before production enablement, not assumed to be a
  universal replacement.
