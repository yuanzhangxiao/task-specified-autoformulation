# M34: fixed-noise measurement fitting

Implemented 14 October 2026 for a CPU-only qualification on small synthetic
controls. Remote results are pending. This is the first numerical increment of
the [noise-aware design](PHASE_C_NOISE_AWARE_FITTING_DESIGN.md), separate from the
submitted M33 diagnostic supplement. Existing benchmark releases, public prompts,
clean fitting interfaces and production defaults remain unchanged.

## What this increment establishes

The new opt-in request binds an equation candidate to raw training readings,
their acquisition identities, fixed conversions, known positive measurement
standard deviations, missing-reading masks and an explicit preparation contract.
The objective is

\[
Q(v)=\frac12\sum_{i\in\mathcal O}
\left(\frac{\mu_i(v)-y_i}{\sigma_i}\right)^2,
\qquad
-\log p(y\mid v)=Q(v)+\sum_{i\in\mathcal O}\log\sigma_i
+\frac{|\mathcal O|}{2}\log(2\pi).
\]

All three quantities in a standardized residual use the same analysis units.
Raw values and conversions remain available for auditing. Coordinate scaling,
measurement uncertainty and legacy NMSE normalization have different meanings;
the weighted objective is never labeled NMSE.

Initial states are either fixed by explicit trajectory-specific preparation or
controlled by a declared shared global parameter. A noisy first target reading
contributes once to the likelihood; it does not fix the initial state. The
numerical adapter overrides initialization explicitly, including its symbolic
derivatives and profiling eligibility. It does not edit a first observation to
trick the legacy initializer. Shared causal maps, independently fitted
trajectory-specific initials, estimated noise scales, uncertain forcing/clocks,
covariance, censoring and sensor-error correction are deferred.

Missing target readings stay null in the raw contract and contribute no residual.
Any finite placeholder needed by the legacy in-memory trajectory container is
excluded by the frozen mask and cannot initialize a state. Forcing histories
remain complete and use the existing piecewise-linear semantics.

## Numerical procedure

1. Validate and freeze the complete measurement/request/objective identity.
2. Compile the supplied equations and explicit initial rules. Certify affine
   state propagation or eligible terminal-output profiling when possible.
3. Independently check the supplied incumbent using original-equation DOP853 and
   Radau rollouts; keep usable evidence even if a later search fails.
4. Optimize from the supplied start and one predetermined domain-only restart,
   sharing a 100-call, 30-second search ceiling. Reuse bounded TRF and the guarded
   coordinate scaling qualified in earlier milestones. Stop on optimizer
   criteria or budget, never on a near-zero noiseless NMSE gate.
5. For the profiled arm, whiten the affine design, offsets and their derivatives
   before the bounded inner solve. A fixed fully affine problem needs one such
   solve, with no redundant outer optimization. Ineligible or failed profiles
   fall back to joint fitting within the same remaining allowance.
6. Check retained candidates under the same independent solvers, sharing a
   15-second verification allowance. Retain only a verified improvement under
   both solvers; preserve ties and the verified incumbent.

Return physical parameters, the Gaussian objective, raw and standardized residual
evidence, numerical verification, search history, profiling/fallback facts and
separate search/verification/setup costs. Joint whitened sensitivity includes
eligible initials but is descriptive: there is no calibrated interval or
probability, and no specification/data-fault verdict.

Durable stage identities make a completed resume a no-op. Interrupted stages
retain their original allowance charge and unknown usage; they do not restart.
A retained progress vector may undergo a still-unspent independent verification,
but its saved loss is not trusted. Changing sigma, mask, calibration, initial
rules, code, objective or policy requires a distinct experiment root.

## Qualification question and limitations

The paired CPU study uses six small equation/measurement cases, three noise seeds
and two strategies (joint rollout and eligible profiling): 36 attempts. The
cases exercise prepared/shared initials, heteroskedastic weights, active bounds,
missing readings, a nonlinear rate parameter, and multiple outputs requiring
joint fallback. Analytical weighted optima are evaluator-only where available;
the remaining paired comparisons check the same objective and numerical replay.
Truth, noise-generating seeds and clean companion values never enter fitter
inputs or incumbent selection. These are isolated controls, not edited basin
benchmark observations or a test-data run.

Success means the interfaces and numerical methods solve the declared weighted
problem correctly. Noisy fitted coefficients need not equal generating values.
This study does not establish statistical coverage, robustness to corrupted
readings, real-data calibration quality, or specification/data-fault attribution.
It precedes the much broader [diagnosis study](PHASE_C_SPEC_DATA_DIAGNOSTIC_STUDY.md).

Local verification: 197 numerical/legacy regression tests and 35 campaign tests
passed. Twelve actual seed-zero fits passed independent rollout verification;
all ten applicable analytic weighted-optimum checks passed. The final campaign
prepare/fit/completed-resume/report smoke also passed. Changed-file Ruff and
shell syntax checks pass. Repository-wide Ruff reports 37 pre-existing errors
in unrelated analysis files. A broader pytest run was stopped after 382 passes
and three optional-Torch skips; it is not a complete repository-suite result.
The Delta qualification remains pending.

## Why Huber is deferred

Squared loss gives a residual of ten standard deviations one hundred times the
loss of a one-standard-deviation residual. A few erroneous readings can therefore
dominate the fit. Huber keeps quadratic loss near zero and changes to linear
growth for larger standardized residuals. It limits their influence, while
retaining a convex inner problem for an affine output model with fixed scales.

That inner problem is no longer one ordinary least-squares solve; it needs a
qualified convex solver or converged reweighting. A large residual can also come
from a missing mechanism or incorrect initialization. Downweighting it does not
diagnose a faulty sensor. First qualify the Gaussian objective here; qualify
robust fitting and its distinct interpretation in the next increment.

## Delta launch

Use the delivered `phase-c-fitting-m34.tar.gz` from the pinned commit. Upload it
to `$HOME`, then run this on Delta:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/measurement-fitting-m34.XXXXXX)
tar -xzf "$HOME/phase-c-fitting-m34.tar.gz" -C "$AF_CODE"
cd "$AF_CODE"
sha256sum -c SHA256SUMS --quiet
bash scripts/hpc/submit_phase_c_measurement_fitting_delta.sh
BASH
```

Default result root:
`/work/hdd/bibo/yxiao2/phase_c/measurement-fitting-v1`.
The launcher uses the existing CPU environment and fitting dependency overlay,
with two concurrent array tasks. No GPU, API key, source campaign archive or
existing benchmark dataset is required. Preparation runs relevant tests on a
compute node before the array. Resume uses the same extracted code, result root
and submission journal; do not create another extraction merely to resubmit.

Inspect and package the completed results using the supplied
`scripts/hpc/inspect_phase_c_measurement_fitting_delta.sh` with the printed root.
From the same extracted code directory on Delta:

```bash
bash scripts/hpc/inspect_phase_c_measurement_fitting_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/measurement-fitting-v1
```

Missing/failed tasks stay in the report denominator. Download its review archive
for comparison with the analytical controls and numerical audit.
