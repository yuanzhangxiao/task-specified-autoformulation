# Stormwater detention basins — development specification v1

## A. Scientific task

Model the downstream water level in two stormwater detention basins in series.
Measured runoff enters each basin. Water can leave the upstream basin through
an overflow structure, enter the downstream basin, and ultimately leave the
system through the downstream outlet. The connecting channel has negligible
travel time and storage on the sampling timescale. Tailwater does not submerge
either overflow structure in the specified operating range. Evaporation,
infiltration and other unmeasured gains/losses are negligible over each event.

## B. Observations and known forcing

The sole prediction target is downstream water depth `h_down`, in metres.
The external inputs `inflow_up` and `inflow_down` are measured volumetric runoff
in cubic metres per minute, supplied over the prediction horizon. They are
linearly interpolated between samples. Time is measured in minutes. Rainfall is
not the model input: no unobserved rainfall-to-runoff model needs to be inferred.
Upstream depth is unobserved after the initial instant. There are no auxiliary
time series. Rollouts start from the initial downstream observation; subsequent
target observations cannot reset or drive the prediction.

## C. Surveyed geometry and preparation

Fixed covariates give constant horizontal water-surface areas `area_up` and
`area_down` (square metres), crest heights `crest_up` and `crest_down` above
their respective basin floors (metres), and `initial_up`, a measured upstream
depth at the initial instant. This one initial reading is available equally
on every development and future evaluation trajectory. It is not an upstream
trajectory or a parameter to fit separately on validation. The downstream
warning level is the supplied covariate `warning_depth`.
Unknown hydraulic coefficients are shared across events. Generated trajectories
remain below the surveyed bank levels and within the free-overflow regime.

## D. Required mechanisms

- Represent accumulation in both physical basins and causal upstream memory.
- Represent one internal volumetric transfer: what leaves upstream enters
  downstream, with the appropriate area conversion if states are depths.
- Represent threshold-dependent overflow and the downstream discharge.
- Preserve water balance, nonnegative stored water, and no artificial source
  or sink between basins under the stated assumptions.

Discover the discharge laws and unknown coefficients from the development data
and scientific context. No particular process name, equation syntax, or latent
coordinate naming is required. A shared named law and algebraically equivalent
inlined equations receive the same mechanism credit. Depths have different
areas: equal-and-opposite depth derivatives are not a conservation law.

## E. Evaluation boundaries

Fit parameters using training only and evaluate continuous output rollouts.
Validation may inform later model selection; no validation-specific initial
state fitting is allowed. Hidden storage/flow labels and generating equations
are evaluator-only diagnostics. This is a synthetic engineering development
case, not field validation or a released held-out benchmark.
