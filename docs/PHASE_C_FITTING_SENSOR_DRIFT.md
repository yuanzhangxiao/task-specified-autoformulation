# M36: shared sensor drift and physical parameter ambiguity

Implemented as an opt-in qualification after
[M35 robust measurement fitting](PHASE_C_FITTING_ROBUST_MEASUREMENT.md).
The 144-attempt Delta qualification completed in the October 15 review.
Benchmark data, finalized task prompts and production defaults stay
unchanged. This milestone implements P4 of the
[noise-aware design](PHASE_C_NOISE_AWARE_FITTING_DESIGN.md), not the full
specification/data diagnosis study.

## Question and measurement contract

Can a bounded, shared measurement explanation accommodate systematic sensor
drift while exposing cases where physical dynamics and sensor effects cannot
be distinguished? The same immutable acquisition is fitted under two explicit
observation hypotheses:

\[
H_0:\quad \mu_{ji}=H(x_j(t_{ji});\theta),\qquad
H_1:\quad \mu_{ji}=H(x_j(t_{ji});\theta)+b(t_{ji}-t_{\rm ref}).
\]

Here `b` is one shared sensor slope, with a predeclared finite domain, numerical
scale, channel, unit, time reference and public provenance. It is an observation
nuisance parameter, not a new ODE source or a physical latent state. The slope
unit is the target analysis unit divided by the common analysis-time unit.
The fixed calibration and raw acquisition bytes are preserved. Predicted physical
outputs and predicted readings including drift remain distinct.

In these controls, the shared slope is defined against each run's declared
analysis clock and common reference zero. This assumes the same instrument-start
protocol, including zero drift offset at that reference. It does not silently
allow a separate intercept, arbitrary trajectory correction, unknown gain,
time warping, or an unknown input. Initial conditions remain exact preparation
or one explicitly shared unknown initial value; noisy first readings are never
treated as exact initial states.

Both hypotheses use the same training data, physical equation, starting physical
parameters, fitting allowance and fixed known sigma. Gaussian and fixed-threshold
Huber losses are compared separately. A lower optimized training objective under
the larger drift model is not itself evidence that drift exists: the hypotheses
are reported as alternatives without automatic selection or a fault verdict.

## Numerical implementation

The new typed `DriftProblem` wraps an unchanged `MeasurementProblem`. Its complete
serialization, observation hypothesis, loss, source, runtime and policy bind
checkpoint identities. The existing fitter's opt-in dispatch extends the rollout
vector by one bounded sensor coordinate. Original-equation DOP853/Radau checks
both compose the declared observation model after solving physical dynamics.

Where the physical predictions have a certified affine form conditional on
nonlinear parameters, adding the drift appends the fixed column
`(time-time_reference)/sigma` to the complete whitened design. Physical gains,
eligible shared initials and sensor slope are solved together by the same bounded
Gaussian or Huber projection. The column's nonlinear derivative is zero, but
the full inner solve retains its cross-coupling with other columns. An ineligible
or unreliable profile falls back to direct joint fitting within the remaining
allowance. The observation hypothesis never creates an affine certificate for
an ineligible physical model.

## Separate local information diagnostic

After a fitted endpoint is sealed, a separately budgeted diagnostic checks the
raw standardized residual Jacobian by finite differences at two step sizes.
It uses original-equation DOP853 rollouts, central differences where feasible,
and explicitly recorded second-order one-sided differences near a bound.
Known calls and elapsed cost are journaled; interruption does not renew the
allowance or silently turn missing results into successful checks.

Let `v=(theta,nu)` contain physical coefficients and all allowed shared-initial
and sensor nuisance coordinates. With fixed observation whitening `W` and frozen
numerical scales `D`, the qualified matrix is

\[
J= W\frac{\partial\mu}{\partial v}D=[J_\theta\;J_\nu],\qquad
J_{\theta\mid\nu}=(I-Q_\nu Q_\nu^T)J_\theta,
\]

where `Q_nu` is a rank-qualified orthonormal basis of the nuisance columns.
SVD is applied directly to these matrices, rather than determining rank from
squared normal equations. Reports retain the joint spectrum, null directions,
coordinate scales, physical spectrum before/after nuisance projection, active
bounds, finite-difference checks and costs.

Numerical rank uses a declared relative tolerance 1e-8 with absolute floor
1e-10. After nuisance projection, its threshold is anchored to the unprojected
physical spectral norm so cancellation roundoff cannot become apparent information.
A full-rank smallest/largest singular-value ratio below 1e-6 is marked
weak under the declared coordinates. These are numerical conventions, not
statistical significance thresholds. Active bounds withhold interior conclusions.
Gaussian Gram matrices have only the local expected-information interpretation
conditional on the declared model and known sigma. Under Huber the raw Jacobian
remains descriptive; it is not relabeled robust likelihood information.
No confidence percentage, global identifiability certificate, sensor-fault
accusation or specification verdict is emitted.

## Frozen experiment

Six acquisition cases, three seeds, two observation hypotheses, two losses and
two strategies yield 144 attempts. All physical controls use the known equation
`x' = g*u-k*x`; the decay may be fixed or fitted. Generating values, clean
trajectories and ambiguity witnesses are evaluator-only.

| Case | Purpose |
| --- | --- |
| Clean, exact preparation | Measure the effect of allowing unnecessary drift |
| Drift, exact preparation | Separate physical gain from a shared sensor trend |
| Drift, shared unknown initial | Assess initial/physical/sensor information jointly |
| Drift, unknown decay | Retain nonlinear outer fitting and conditional profiling |
| Zero decay, common constant input | Exact physical-gain/sensor-drift ambiguity |
| Zero decay, varied inputs | Resolve that ambiguity through input variation |

When `k=0` and every trajectory has the same constant input `u0`,
`mu=x0+(g*u0+b)*t` at reference zero. The data constrain the combination
`g*u0+b`, not both individual coefficients. Different known input levels across
trajectories can distinguish them. The fitter receives neither the witness
parameters nor the expected diagnostic label. An independent evaluator checks
the mathematical witness and reports physical versus measured-output error
separately; a nonunique optimum is never judged by recovery of one arbitrary
parameter vector alone.

Analytic own-loss optimum checks apply to affine controls; nonlinear cases have
independent endpoint prediction/objective checks without a global optimum claim.
No held-out observations select endpoints. Diagnostics do not alter fit retention.
Failed, missing and interrupted outcomes remain visible in all denominators and
cost summaries.

## Completed Delta qualification, 15 October 2026

Review archive `review-20261015-064857.tar.gz`, plan
`ce527d5817b7a8a80abcb57afa71a9528e3a002b80ea74aa798e524a5405c5b3`,
was verified and its report reproduced locally. All 144 fits completed, all
144 independent rollout/objective checks and derivative checks passed, and all
120 applicable affine optimum checks passed. All 72 direct/profiled objective
pairs agreed; the maximum absolute gap was `5.03e-12`. These are checks of the
declared optimization problem, not 144 demonstrations of physical recovery.

The following four rows are selected synthetic acquisition cases, not four
benchmarks. Each uses `x' = g*u-k*x` and drifting noisy readings. The last two
columns are median evaluator-only physical-trajectory RMSE over three seeds,
using the Gaussian loss and profiled strategy. They are not training NMSE or
scores available to the fitter.

| Case | Unknown physical quantities | Ignore drift | Allow shared drift |
| --- | --- | ---: | ---: |
| `drift_exact` | Gain; decay and prepared initial value known | 0.1891 | 0.00100 |
| `drift_shared_initial` | Gain and one shared initial value; decay known | 0.1956 | 0.01206 |
| `drift_nonlinear_decay` | Gain and decay; initial value known | 0.1939 | 0.02036 |
| `resolved_varied_input` | Gain with zero decay; differing known constant inputs | 0.1947 | 0.00348 |

The other two cases were the clean negative control and the deliberately
confounded constant-input control. All twelve shared-drift fits in the latter
had joint rank one and zero physical information after nuisance projection.
Ten received the summary label `local_rank_deficient`; two were labeled
`boundary_limited` and also had rank deficiency. Reports should preserve both
facts, rather than interpreting an exclusive headline label as their absence.
All twelve corresponding varied-input fits had joint rank two and conditional
physical rank one. Input variation resolved this particular ambiguity; it does
not resolve arbitrary physical/calibration confounding.

Ignoring drift can yield a locally full-rank fit to the wrong observation model.
Huber alone did not remove the systematic trend. Conversely, adding a drift
coordinate can slightly overfit a clean acquisition, and even a full-rank
nonlinear fit retained appreciable coefficient error. No automatic observation
hypothesis selection, fault diagnosis or confidence percentage is qualified.
Three Gaussian profiles in the confounded control used the declared joint
fallback, so the successful profiled arm is a portfolio result.

Summed fitting CPU time was 195.14 seconds: 121.00 direct and 74.14 profiled.
Separately reported diagnostics used 52.89 CPU seconds, an additional 27.1%,
and 1,200 fully accounted calls. Summed fitting wall time was 599.56 seconds;
diagnostic wall time was 193.25 seconds, an additional 32.2%. These sums exclude
queue time and are not campaign elapsed time. No LLM or test data was used.

The immediate follow-up is the current-paper
[interaction feedback integration](PHASE_C_INTERACTION_FITTING_FEEDBACK.md).
Near-confounding and statistical uncertainty calibration remain future work.

## Local verification

The combined M36, measurement-fitting and robust-fitting regression suite passes
243 tests. A 48-fit smoke covers all six cases at seed zero under both hypotheses,
both losses and both strategies: all independent numerical/objective replay
checks pass, all 40 applicable affine optimum checks pass, and all 24 strategy
pairs agree in their own loss. The shared-drift constant-input control is correctly
rank deficient; equal measured-output objectives can accompany different physical
trajectories. This local smoke is not the full three-seed Delta qualification.

Changed Python files pass Ruff; shell syntax and diff checks pass. The complete
repository test run was bounded at five minutes: 1,041 tests passed, with seven
failures and one setup error before interruption. Most failures report changing
source/frozen identities during concurrent development; an unrelated construction
test also fails. The full repository is therefore not claimed to pass. Repository
Ruff reports 37 existing issues in unrelated `analysis/claude` scripts.

## Delta commands

Upload the pinned `phase-c-fitting-m36.tar.gz` bundle to your Delta home directory:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/sensor-drift-m36.XXXXXX)
tar -xzf "$HOME/phase-c-fitting-m36.tar.gz" -C "$AF_CODE"
cd "$AF_CODE"
sha256sum -c SHA256SUMS --quiet
bash scripts/hpc/submit_phase_c_sensor_drift_delta.sh
BASH
```

One CPU per attempt, two concurrent by default; no GPU, LLM, API key or test
dataset is needed. Preserve the extracted code and submission journal for resume.
Do not blindly resubmit when the scheduler receipt is uncertain.

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/sensor-drift-fitting-v1
AF_CODE=$(jq -er '.identity.repo' "$AF_ROOT/submission_manifest.json")
bash "$AF_CODE/scripts/hpc/inspect_phase_c_sensor_drift_delta.sh" "$AF_ROOT"
```

The inspector reports completed fits, own-loss strategy comparisons, information
diagnostics and separate overhead, then writes a portable review archive.

## Remaining scope

This qualification uses small known-equation controls and fixed known sigma.
It does not calibrate robust uncertainty, estimate unknown noise, identify
arbitrary sensor errors, select scientific claims, or establish performance on
the full Phase C benchmark roster. Confounding is a valid outcome. Local
information diagnostics alone cannot establish global uniqueness or which of
two competing explanations is scientifically correct.
