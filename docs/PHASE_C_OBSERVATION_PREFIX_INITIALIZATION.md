# Initial-state estimation from an observation prefix: proposed scope

This document records an implementation audit and a proposed extension. It does
not enable a new fitting or evaluation policy. The currently runnable change is
the separate [readable initializer diagnostic](PHASE_C_INITIALIZER_READABLE_RETEST.md).

## What the current implementation estimates

Orion's public Phase C C+S and transcription paths estimate shared latent initial
values and shared initializer-map coefficients jointly with dynamics parameters
from complete TRAIN trajectories. The map is evaluated on each trajectory's
permitted first readings. A shared constant is the same across trajectories; a
shared map can produce different values from different permitted readings.

Relevant implementation:

- `fitting/initialization.py:lower_initialization` lowers these quantities to global
  fitted parameters; the plan records `scope=shared_training` and no fitted
  validation initials.
- `fitting/collocation_sensitivity.py` passes training data to initialization and
  refinement. `sensitivity_probe.py` constructs residuals over training target
  samples. Its validation replay disables trajectory-specific initial fitting.
- `fitting/transcription_solver.py` constrains first nodes to the declared initial
  rule. Later shooting/collocation nodes are optimization variables for numerical
  transcription, not free initial states exported to held-out replay.
- `fitting/simulation.py:trajectory_initial_state` evaluates the shared initial
  rule from the new trajectory's permitted initial data. Qualification performs
  a free rollout and does not fit validation initial values.

There is a separate legacy generic backend in `fitting/fitter.py` with ranged
per-trajectory initial variables and full-validation-trajectory initial fitting.
The public fitting boundary rejects that initialization scope. Its existence must
not be described as the current public Phase C protocol, and it must not be reused
unchanged for future-only prefix evaluation.

## Information available at prediction time

Estimating an initial state is not itself leakage. The allowed information and
scoring horizon determine the claim:

| Evaluation task | Allowed state-estimation data | What the score measures |
| --- | --- | --- |
| Prediction/control from the first instant | Initial measurements, pre-existing history and a prior | Prospective behavior from the start |
| Prediction after an observation window | A predeclared prefix ending at time tau | Future predictions strictly after tau |
| Retrospective reconstruction | The entire held-out observed trajectory | Reconstruction conditional on those observations |

Full-trajectory state fitting can be useful for investigating whether equations
can explain data. It does not demonstrate that the model could have predicted
those observations or supported decisions before they arrived. Observed output
channels remain unavailable before their measurement time, even when a separate
benchmark target is hidden. An online observer can update as data arrive; a
retrospective smoother uses future observations to estimate earlier states.
Future exogenous inputs may be supplied only under the declared forecast scenario
(for example, a known control schedule). Future measured auxiliary channels cannot
be treated as known inputs merely because the dataset calls them auxiliary.

For a fixed learned model theta, a minimal prefix estimator is

    z_hat(0) = argmin_z sum_{t_k <= tau} ||y_hat_theta(t_k; z) - y(t_k)||^2_W + R(z).

Only the latent boundary is optimized here. Shared dynamics, observation-map and
initializer parameters are frozen. Propagate the resulting state to tau, then
score samples with t_k > tau. Bounds, prior/regularizer R, solver budget, observation
channels, tau and normalization must be fixed using training/validation decisions,
never selected from the final test outcomes. Failure to estimate a state remains
visible. The prefix may not identify every hidden state; a good fit need not imply
a unique state or correct physics.

For control beginning at time zero, a later prefix cannot be used retrospectively
for the early decisions. Predeployment history, a prior with uncertainty, or causal
online estimation is needed for that period. This is a different experiment from
waiting until tau to begin prediction.

## Minimal implementation boundary

1. **Evaluation contract and schemas.** Add a separate opt-in initialization policy
   describing allowed observed channels, prefix cutoff, latent coordinates/bounds,
   estimation budget and scoring start. Keep time-zero-only policy as a baseline.
2. **Numerical state estimator.** With theta frozen, optimize only latent initial
   state coordinates using a physically restricted prefix view. Return a sealed
   inferred state and diagnostics. Existing rollout/sensitivity infrastructure can
   be reused; this is not a new dynamics fitter. Do not call the legacy
   full-validation initial-state routine.
3. **Simulation and metrics.** Carry the inferred state to the cutoff, define the
   treatment of observed components there, then perform free future rollout with
   no future target or unavailable auxiliary forcing. Apply a strict future-only
   metric mask. Use the same
   prefix and scoring horizon for every compared method.
4. **Construction and compatibility feedback.** Retain variables, RHS topology,
   interaction grammar and scientific requirements. Record the estimator policy
   as part of initialization. Requalify matched-trajectory checks: equal t0
   observations no longer imply equal latent estimates when different permitted
   histories may be used. Initializer-dependent checks are conditional or deferred
   until that policy is known. Existing shared constants/maps remain testable in
   the current strict protocol.
5. **Reporting, provenance and tests.** Separate cold-start, prefix-conditioned and
   retrospective scores; report prefix duration and inference cost. Add cache keys,
   deterministic resume and a test that changing observations strictly after tau
   cannot change the estimated initial state. Test cutoff boundaries, unavailable
   observations, failed/infeasible estimates and latent-state nonidentifiability.

The minimal estimator does not solve a separate training restriction: if training
trajectories genuinely need different hidden initials and the current t0 map cannot
represent them, freezing a poorly fitted shared-initial model is insufficient.
Supporting that situation additionally requires per-trajectory TRAIN initial
coordinates (or a learned history initializer), while retaining shared dynamics.
The public fitting boundary, parameter indexing/sensitivities and transcription
boundary constraints would need an explicit new opt-in policy. Training-only free
initials still do not specify how to initialize an unseen trajectory; the prefix
estimator supplies that missing inference procedure. This broader fitting change
should be coordinated separately with Orion rather than silently activated here.

A first milestone can use known equations and a small synthetic partially observed
system, with a fixed prefix length and no proposer changes. It should demonstrate
causal access, future-only scoring and reproducibility before any campaign change.
A learned history encoder or online observer can follow if repeated numerical
state estimation is too expensive. A neural encoder is not required for this first
milestone. Existing benchmark files and historical results need not be rewritten.

## Literature context

There is no general convention banning fitted initial states. Beintema, Toth and
Schoukens (2021) discuss optimizing section initial states and instead learn an
encoder from past input/output samples to initialize unseen sequences:
https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf

Rubanova et al. (2019), section 4.2, demonstrate extrapolation by conditioning a
latent-state encoder on the first half of observations and predicting the second:
https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf

MathWorks `compare` documents initial-state estimation from comparison data, which
illustrates why a reported model-fit score is not automatically a prospective
forecast score:
https://www.mathworks.com/help/ident/ref/compare.html
