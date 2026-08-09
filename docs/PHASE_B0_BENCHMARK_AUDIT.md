# Phase B0 benchmark audit

## Scope and leakage boundary

This audit asks whether the public trajectories can distinguish dynamical-system
discovery from short-horizon interpolation before any further search objective
or judge optimization. The shortcut models are fitted on training data only.
Validation and test data are used only for evaluation. No private hidden state,
test metric, or LLM call is used to fit a shortcut.

The reproducible outputs are:

- `artifacts/phase-b0/predictability-all-tasks-v2/shortcut_metrics.csv`;
- `artifacts/phase-b0/predictability-all-tasks-v2/excitation_metrics.csv`;
- `artifacts/phase-b0/predictability-all-tasks-v2/downsampling_metrics.csv`;
- `artifacts/phase-b0/predictability-all-tasks-v2/response_phase_metrics.csv`;
- `artifacts/phase-b0/predictability-all-tasks-v2/manifest.json`;
- `artifacts/phase-b0/observability-v1/observability.csv`.
- `artifacts/phase-b0/observability-v2/observability.csv`;
- `artifacts/phase-b0/observability-v2/parameter_sensitivity.csv`;
- `artifacts/phase-b0/free-rollout-core-v1/`;
- `artifacts/phase-b0/free-rollout-dalla-v1/`.

## Implemented diagnostics

Four deliberately weak predictors are evaluated at 1, 5, 10, 30, 60, 120,
and 300 sample horizons:

- constant training mean;
- persistence from the last observed target;
- scalar autoregression fitted on training transitions;
- input-aware scalar ARX fitted on training transitions.

AR and ARX are propagated recursively within each horizon. Event-window metrics
cover the 30 samples surrounding a nonzero or changing numeric input. The audit
also reports target-increment variance relative to target variance, input
activity/change rates, distinct numeric input levels, and event-window coverage.

## First findings

### One-step prediction is an easy shortcut task

Hard-tier test persistence NMSE is 0.00126 for original B1, 0.00141 for
perturbed B1, 0.00902 for Benchmark 5, and 0.000558 for Benchmark 6. These
values explain why a persistence baseline can look competitive under the
historical one-step protocol.

For the Dalla Man training trajectory, the variance of adjacent Gp increments
is only 0.000723 times the variance of Gp itself. The meal forcing is nonzero in
0.33% of samples and has only two observed numeric levels. The dataset contains
one 90 g training meal, one 105 g validation meal, and one 120 g test meal, all
at the same time and basal initial condition. This is weak excitation for
identifying a multi-compartment nonlinear system.

### The shortcut advantage disappears with forecast horizon

Persistence NMSE for original B1 Gp rises from 0.00126 at one minute to 0.0317
at five minutes, 0.125 at ten minutes, 0.836 at thirty minutes, 1.95 at sixty
minutes, and 4.24 at 120 minutes. In the 30-minute post-event window it reaches
4.40 at a 30-minute horizon. Perturbed B1 behaves similarly and reaches NMSE
5.51 at 120 minutes.

Benchmark 5 persistence rises from 0.00902 at one sample to 4.03 at 30 samples;
Benchmark 6 rises from 0.000558 to 0.465. The one-step score therefore
substantially understates dynamical error. The 300-sample endpoint should not be
used as a headline because the Dalla Man trajectory has nearly returned to its
initial basal value, making endpoint persistence deceptively good again.

### Scalar ARX does not solve the Dalla Man identification problem

The input-aware scalar ARX model is nearly indistinguishable from persistence
on Dalla Man. At a 30-minute horizon its whole-trajectory NMSE is 0.830, while
its post-event NMSE is 4.42. A single impulse time and three split-specific meal
masses do not expose delayed absorption sufficiently for a scalar linear input
term to generalize.

### Physical-time downsampling confirms the horizon effect

At the same 30-minute forecast horizon, original-T1 persistence NMSE is 0.836
on the one-minute grid, 0.867 on the five-minute grid, and 0.900 on the
ten-minute grid. The conclusion is stable when physical time rather than sample
count is held fixed. Conversely, a nominal one-step prediction becomes much
harder as the sampling interval grows: the very low historical score depends
directly on resetting from observations every minute.

### Shortcut error concentrates in the scientifically relevant response

For original-T1 Gp at a 30-minute horizon, persistence NMSE is 4.52 during the
rising response, 0.776 during recovery, and only 0.0894 near equilibrium. A
whole-trajectory average therefore dilutes failure where the meal mechanism is
most active. Future reporting should stratify response phases or weight
prespecified event windows, while retaining an unweighted full-trajectory
metric for transparency.

### T2--T4 hard tiers are more demanding tasks over the same outputs

The hard tiers of T2, T3, and T4 all expose the same two dynamic channels, Gp
and I, and the same meal protocol. Their public observed trajectories are
therefore numerically identical; what changes is the requested hidden
mechanistic decomposition:

- T2 adds delayed insulin-dependent disposal;
- T3 additionally requires a separate delayed hepatic-regulation pathway;
- T4 requires a coherent portrait of meal appearance, endogenous production,
  utilization, and insulin secretion.

Consequently, predictive error alone cannot establish that T3 or T4 mechanisms
were recovered. T4 asks for substantially more latent structure without adding
hard-tier measurements or experimental excitation. It should be treated as a
high-ambiguity stress test unless the next identifiability analysis shows that
its required internal quantities are observable from Gp, I, and the meal input.

### Initial hidden states are locally visible but poorly conditioned

A scaled finite-difference sensitivity audit perturbed task-relevant private
initial states and observed only Gp for T1 or Gp and I for T2--T4. All tested
state columns have nonzero numerical rank at relative thresholds of `1e-3` and
`1e-6`. This means those initial-state directions affect the outputs in the
exact simulator; it does **not** establish unique equation, parameter, or flux
recovery.

The spectra are strongly anisotropic. Under the historical single 90 g meal,
condition numbers rise from 83 for T1 to 228 for T4, while stable ranks remain
near one (1.006--1.017). Most output sensitivity is therefore concentrated in
one dominant direction and the remaining hidden directions are practically
weak. A richer multi-meal timing/magnitude protocol improves condition numbers
to 61 for T1/T2 and approximately 115--120 for T3/T4, but does not remove the
imbalance. This supports richer excitation while cautioning against treating
numerical full rank as practical recoverability.

### Parameter and flux sensitivity expose practical non-identifiability

T1 output sensitivity has full local rank for its five audited parameters and
a condition number of 34 under the single meal. T2 remains full-rank but its
condition number grows to 229. T3 has only 7 of 10 parameter directions above a
relative `1e-3` threshold, with condition number 10,642. T4 has only 12 of 18
directions above that threshold and condition number 49,207. A richer multi-meal
protocol improves T3/T4 condition numbers to 2,832 and 12,178, respectively,
but leaves substantial practical ambiguity.

Private flux trajectories are more sensitive to these parameters than the
public outputs, particularly for T3. That does not make the parameters
recoverable: those fluxes are unavailable during discovery. It instead shows
that materially different hidden mechanisms can be weakly distinguished by
Gp/I while producing strongly different private flux portraits.

Repeating the initial-state audit after shifting glucose and insulin initial
conditions changes the condition numbers only modestly. Multiple initial
conditions are useful, but input diversity and task-specific interventions are
also required.

### Historical intervention results were not uninterrupted free rollouts

The Phase A3 evaluator reset observed candidate states from measurements at
every interval. A new explicit frozen free-rollout mode now propagates observed
states without those resets. It preserves the historical protocol as the
default and makes no structural or parameter refit.

Across matched cells, free rollout increases geometric NMSE by approximately
9--524 times on Benchmarks 5/6 and 209--1,029 times on the Dalla Man variants.
Core free-rollout evaluation completed for 165/172 cells; Dalla Man completed
for 320/344. Failures are retained rather than assigned a finite sentinel.

Full Autoformalism is not consistently best under free rollout. For example,
geometric NMSE on Benchmark 5 is 2.15 for Full versus 0.339 for
LLM-feature-SINDy, while on Benchmark 6 it is 1.32 versus 1.14. On original B1,
Full is 5.44 versus approximately 2.62 for both D3 and LLM-feature-SINDy. These
models were selected for one-step correction, so this is evidence of objective
mismatch rather than a fair claim that one baseline solves autonomous
discovery. It does show that weighted re-ranking alone is insufficient.

## Interpretation of obfuscated-perturbed

Original, perturbed, obfuscated-original, and obfuscated-perturbed form a 2-by-2
diagnostic design: semantic context present/absent crossed with canonical/
perturbed dynamics. Obfuscated-perturbed is useful only as the interaction cell:
can the data reveal a noncanonical mechanism when neither meaningful names nor
the canonical semantic prior are available?

Its observed trajectories are exactly the perturbed B1 trajectories under a
symbol renaming, just as obfuscated-original duplicates original B1. It is not
an independent predictive dataset. If the semantic-by-perturbation interaction
does not add a distinct conclusion, this condition should be reported as a
negative control or appendix ablation rather than as another independent
benchmark.

## Provisional decisions

1. One-step MSE becomes a diagnostic, not the primary performance endpoint.
2. Report 5--120-step error curves and event-window errors; do not summarize
   only the terminal 300-minute endpoint.
3. Prioritize free rollout under new input schedules and initial conditions.
4. Add training trajectories with multiple event times, multiple event masses,
   multiple events, and varied initial conditions before claiming unique latent
   recovery.
5. Treat obfuscated tasks as semantic-prior/identifiability stress tests.
6. Treat perturbed semantic tasks as the primary anti-memorization experiment.
7. Qualify T2--T4 claims separately rather than extrapolating evidence from T1.
8. Add rollout-aware fitting and validation before changing judge weights.
9. Distinguish one-step predictors from autonomous ODEs in all result tables.

## Remaining Phase B0 milestones

- nonlinear profile-likelihood analysis for the most confounded T3/T4 parameter
  groups; the current result is a local sensitivity audit;
- a frozen redesign proposal stating the minimum additional trajectories needed
  to distinguish canonical and perturbed mechanism families.

No judge weighting or weighted-sum search change should be finalized until
these remaining diagnostics identify which signals genuinely correspond to
scientifically meaningful generalization.
