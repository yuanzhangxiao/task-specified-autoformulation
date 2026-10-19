# Phase C initialization contract: construction and fitting

Updated 2026-10-19. This is the shared handoff for Astra (construction) and Orion
(fitting). It records the two agreed extensions and their first opt-in numerical
qualification, M43. Historical public fitting policies and finalized benchmark
prompts remain unchanged. Passing a synthetic qualification is not yet evidence
that the construction pipeline and all fitting backends support this contract.

## The agreement in plain language

1. A latent initial value may be an unknown constant shared across trajectories.
   Another initial value may depend on that constant, on other declared initial
   states, and on permitted public initial observations. Fit the independent
   constants and map coefficients; calculate dependent initial values.
2. A latent initial value may instead be an independent unknown for each training
   trajectory. For an unseen trajectory, infer that unknown from a declared short
   observation window while holding the learned dynamics and initializer-map
   coefficients fixed. Forecast and score only after that window.

These options can coexist in one model. A dependent initializer is not an
additional independent variable. For example, in

    z1_j(0) = alpha
    z2_j(0) = a * z1_j(0) + b * y_j(0),

the unknowns are alpha, a and b, all shared. There is no separate free z2_j(0).
Replacing alpha by a trajectory root xi_j permits different hidden preparations
for identical initial measured values. The shared coefficients a and b do not
become trajectory-specific.

A shared map already allows different initial values when its permitted inputs
differ. Neither prefix inference nor a new latent root is necessary merely
because two trajectories have different observations. They become useful when
available initial measurements cannot determine the hidden preparation.

## Construction-to-fitting interface

Use an explicit `latent-initialization-graph-1` graph with exactly one rule for
**each latent state**. Identity-observed states receive their own permitted first
measurement in this milestone. Construction must not declare a free initial value
for an observed state to evade that equality.

| Initial rule | Independent fitted quantities | Application to a new trajectory |
| --- | --- | --- |
| `known` | None | Use the justified declared value |
| `value`, `shared_training` | One shared scalar | Reuse the learned scalar |
| `value`, `trajectory` | One scalar per training trajectory | Infer a new scalar using the prefix policy |
| `map` | Shared declared map coefficients | Evaluate the map in dependency order |

Dependencies are explicit aliases. `state_initials` denotes an initial value of a
named model state; `public_initials` denotes a first public reading, input,
covariate or initial time. These have distinct meanings even if the public channel
and a model variable have similar names. A map cannot silently read a whole
trajectory, a future value, a generated process, or undeclared equation parameters.
Map-local parameters are namespaced by state. The first version permits acyclic
dependencies only; it does not solve implicit simultaneous initializer equations.

For the example above, the graph is:

```json
{
  "protocol": "latent-initialization-graph-1",
  "rules": {
    "z1": {
      "initial": {
        "mode": "value", "guess": 1.0,
        "role": "nonnegative_coefficient", "scope": "shared_training"
      }
    },
    "z2": {
      "initial": {
        "mode": "map", "expression": "a * root + b * reading",
        "parameters": [
          {"name": "a", "role": "nonnegative_coefficient", "guess": 0.5},
          {"name": "b", "role": "coefficient", "guess": 0.5}
        ],
        "state_initials": {"root": "z1"},
        "public_initials": {"reading": "y"}
      }
    }
  }
}
```

Changing only z1's scope to `trajectory` gives the second option. Generated
coordinates are `init_z1_value`, `init_z2_a`, and `init_z2_b`. The fitted root
`init_z1_value` then has a separate assignment per trajectory in the new adapter;
its occurrence in a compiled expression does not make it a shared parameter.
The graph audit preserves this distinction. Guess values are starting points, not
fixed values or biological claims. Numerical domains follow declared parameter
roles and the fitting bound policy; these coefficient bounds do not prove that
every dependent state expression stays in its physical domain. The first numerical
adapter rejects unsupported state-constraint models rather than silently dropping
those declarations.

Construction should export this graph with the candidate, the initialization
policy, and the provenance of any fixed initial value or domain claim. It should
not export independently optimized values for dependent map outputs. The fitter
returns both independent coordinates and realized complete initial states, with
shared versus trajectory scope preserved.

## What training solves

Let theta be shared equation parameters, phi shared initializer-map parameters,
c shared constant roots, and xi_j the independent roots of training trajectory j.
The acyclic initializer graph defines the complete boundary

    x_j(0) = G(y_j(0), u_j(0), covariates_j; c, phi, xi_j).

Direct rollout fitting minimizes

    sum_j sum_k || h(x_j(t_k; theta, G_j)) - y_j(t_k) ||^2_W

jointly over theta, phi, c and all training xi_j, subject to their declared domains.
Only independent roots and coefficients enter the optimization. Sensitivities
include the derivative through the complete initializer graph and are assembled
into shared columns plus the appropriate trajectory's root columns.

Complete training trajectories may be used to learn these quantities. This yields
a **training reconstruction score**. It must not be confused with a training
forecast score: to report prefix-conditioned training forecasts, freeze theta,
phi and c and re-infer each xi_j using only its prefix, exactly as for a new row.

## What is permitted on a new trajectory

Prefix duration is a fitting hyperparameter, not a proposer decision. A duration
sweep must hold shared parameters fixed and compare candidates on a common
future horizon. The [M44 qualification](PHASE_C_FITTING_PREFIX_WINDOW_SELECTION.md)
selects the shortest adequate window using predeclared validation criteria,
while reporting root information and inference cost separately. Duration tuning
uses validation, so selected validation scores are not independent test estimates.
The complete duration-selection policy must be frozen before test evaluation.

The prefix policy fixes its duration, available observation channels, known future
exogenous schedules, root domains, starting rule, numerical budget and training
normalization before evaluation. For shared learned parameters theta_hat, phi_hat
and c_hat, inference solves only

    xi_hat_j = argmin_xi sum_{t_k <= tau_j}
        || h(x_j(t_k; theta_hat, G_j(c_hat, phi_hat, xi))) - y_j(t_k) ||^2_W.

Here tau_j is the declared cutoff relative to the trajectory's initial time.
The initial milestone uses no learned encoder or extra regularizer. Additional
priors would require an explicit new policy rather than an undocumented default.

The initial adapter uses all declared target channels as prefix observations;
auxiliary measurements are available to initializer maps only at time zero.
Assimilating additional measured auxiliary histories requires an explicit future
extension of the observation contract.

The estimator receives a typed prefix view, not an entire validation trajectory.
Shared parameters and training-derived normalization are frozen. Forecasting gets
initial permitted readings and declared future exogenous inputs; it does not get
future targets or an unapproved future measured auxiliary. The inferred boundary
is propagated continuously through the cutoff without resetting observed states
to their last measured values. Scoring uses samples strictly **after** the cutoff.
Replaying the complete interval from the inferred initial state is equivalent to
carrying the same simulated state forward at the cutoff.

| Evaluation claim | Data used to infer roots | Scored observations |
| --- | --- | --- |
| Time-zero initialization | Permitted first readings / shared roots | Declared horizon after initialization |
| Prefix-conditioned forecast | Fixed observation window | Strictly after the window |
| Retrospective reconstruction | Complete trajectory | Reconstruction, never called prospective prediction |

Comparisons must use the same permitted window and scoring horizon. A prefix is
not available for control decisions made before that prefix ends. Future input
schedules may be known as part of an intervention; future sensor readings are not
known simply because a dataset labels a column auxiliary. Altering a held-out
suffix must not alter inferred roots or model predictions.

## Construction checks that must change

- Same time-zero observations imply the same initializer output only when the
  declared rule is a deterministic time-zero map with shared roots.
- Different permitted prefixes may justify different inferred roots. A witness
  using only equality at time zero must not reject a prefix-conditioned model.
- Training roots can differ without revealing the hidden simulator state. They are
  inferred from observations, not copied from benchmark truth.
- Prefix inference does not guarantee uniqueness. A root with no observable
  influence, a redundant parameterization, or an insufficient window remains
  ambiguous. A good trajectory fit is not a certificate of a unique physical state.
- Adding dependent maps does not remove scale symmetries. If only a product such
  as a*alpha is observed, separate recovery of both may remain impossible.
- Initialization cannot repair a missing dynamical mechanism automatically. Poor
  prediction can still reflect equations, optimization, data, or inadequate state
  information. Report these separately rather than assigning a cause by default.

## Ownership and first integration gate

**Astra / construction:** declare independent roots, scopes and a complete acyclic
initializer graph; preserve mechanistic justification; adapt conditional equality
witnesses; supply candidate analytic equations before numerical fitting; retain
current prompt/replay policies until the opt-in interface is explicitly selected.

**Orion / fitting:** validate and lower the graph; fit shared and per-trajectory
coordinates; infer unseen roots from the declared prefix; preserve split boundaries;
report fit quality, numerical reliability, root information and inference costs.

M43 implements a separate adapter and known-equation qualification before an
end-to-end construction campaign. Its source entry points are:

- `src/autoformalism/fitting/initialization_graph.py`: typed graph and restricted
  AST expansion, with cycle, namespace and dependency checks.
- `src/autoformalism/fitting/prefix_initialization.py`: shared/per-trajectory direct
  rollout fitting, prefix-only root estimation and future-only scoring.
- `src/autoformalism/fitting/initialization_qualification.py`: frozen synthetic
  qualification and durable operation records.
- `scripts/phase_c_initialization_qualification.py`: independent CLI.

The experiment must demonstrate recovery on shared-root and trajectory-root
controls, exact derivatives through dependencies, frozen shared parameters,
suffix-independent inference, and independent numerical replay. It must preserve
failure/ambiguity reports and separate full-training reconstruction from
prefix-conditioned forecast. Both local and returned Delta qualifications pass
all four tasks and 20 prefix forecasts. The Delta review verifies unchanged shared
parameters and prefix-only inference on unseen rows; worst validation suffix NMSE
is 1.98e-20 on these noiseless known-equation controls. The opt-in construction
exporter is implemented separately in
`PHASE_C_INITIALIZATION_CONSTRUCTION_GRAPH.md`; a complete exported-model fitting
experiment and benchmark evaluation remain pending. No LLM calls,
benchmark rewriting or test access were used by M43. Delta CPU commands are in the
[M43 experiment document](PHASE_C_FITTING_INITIALIZATION_QUALIFICATION.md).

This first adapter reuses direct rollout and forward sensitivities with bounded,
scaled optimization. It does not yet extend collocation, the legacy public-fit
schema, all profiling paths, or production construction. Identity-observed first
readings are treated as exact boundaries; noisy first-state estimation is a
separate observation-model choice. A local rank diagnostic is not a calibrated
confidence interval or a proof of global identifiability.

Operations preserve terminal results on resume. Interrupted bounded operations
remain charged and visible; they are not silently rerun with a replenished budget.
Native optimizer internal-state continuation is not claimed.

## Historical implementation boundary

`fitting/initialization.py:apply_initialization_plan` implements the existing
`shared-latent-initialization-1` contract: shared scalar roots or shared map
coefficients applied to permitted first readings, with no dependent latent
initializers. Public C+S/transcription paths learn those parameters on complete
training trajectories and freeze them for validation. Shooting/collocation nodes
are numerical transcription variables, not extra exported free initial states.

The legacy generic `fitter.py` also contains full-held-out-trajectory initial
fitting. That behavior is outside the public evaluation contract and is not reused
for prefix forecasts. The separate noisy-measurement fitting adapter has its own
explicit initial-boundary contract and is not silently changed by M43.

## Background references

These references motivate conditioning on available observations; they do not
establish implementation coverage or guarantees for this adapter.

- [Beintema, Toth and Schoukens (2021)](https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf):
  section initial states and encoders from past input/output samples.
- [Rubanova et al. (2019)](https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf):
  extrapolation after conditioning on an observed prefix.
- [MathWorks `compare`](https://www.mathworks.com/help/ident/ref/compare.html):
  initial-state estimation illustrates why reconstruction fit need not be a
  prospective forecast measure.
