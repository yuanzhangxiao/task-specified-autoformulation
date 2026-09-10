# Paired collocation and sensitivity experiment

This CPU-only diagnostic combines the two improvements supported by the first
method comparison: collocation initialization and forward-sensitivity refinement.
It also tests whether the benefit survives different starts and noise realizations.
It changes no production fitting defaults, proposer contracts, benchmark prompts,
or benchmark data. There are no LLM calls, test data, or private benchmark metrics.
The user submits the Delta jobs; ACES remains available for Sol's prefit work.

## What the preceding experiment established

On the two latent synthetic systems, forward sensitivities reduced fitting time
by approximately 3.8–4.6 times relative to production finite differences. They did
not fix the inferior solution reached from the broad start in the separated case.
Collocation followed by production finite differences did: clean validation NMSE
was approximately 6.9e-21 without noise and 1.05e-5 with noise, versus approximately
0.024 for the ordinary start. This was one start and one noise realization, so it
does not establish robustness. Every multiple-shooting initializer timed out.

The next experiment drops compiled finite differences and multiple shooting from
the main matrix. It preserves the previous collocation method and compares its
two refinement routes from exactly the same saved initialization. It does not
change the mesh, parameter coordinates, model equations, or optimizer tolerances.

## Frozen experiment

`configs/fitter_methods_v2.json` specifies:

- Two latent fixtures: `moderate` and `separated`, unchanged from v1.
- Noise SD fractions 0 and 0.03, relative to clean training-output SD.
- Three start seeds: 20260909, 20260910, 20260911. Seed 20260909 reproduces the
  original broad start. Positive parameters use the same log-uniform [0.25, 4]
  law; the signed offset uses uniform [-2, 2]. Starts do not depend on truth.
- Replicate noise seeds `20260909 + 104729*replicate`. The existing SHA-based
  case/noise-index/trajectory salt determines each draw. All three routes in a
  pair receive byte-identical observations and the same original start.
- Four training inputs and two validation inputs, horizon 24 and sample step 0.2.
  Only `v01`, time, and supplied `u01` enter fitting. All three initial states
  remain fixed at zero. Hidden reference trajectories never become fitted labels.

For clean data, the three replicates isolate start variation. For noisy data,
both start and noise realization vary across replicates; this is a paired
robustness screen, not a full start-by-noise factorial or a success-rate estimate.

| Route | Initializer | Refinement |
| --- | --- | --- |
| S | Frozen broad start | Forward sensitivities |
| C+FD | Shared collocation parameters | Production finite differences |
| C+S | The same shared collocation parameters | Forward sensitivities |

The total is **36 fits**, preceded by **2 numerical guards** and **12 shared
initializers**: 50 array tasks plus one summary job. Method order rotates by
replicate/noise index to reduce a fixed scheduling-order effect.

Guards independently verify the generating trajectories and compare sensitivity
Jacobians with central differences at truth and all three broad starts. Each fit
ends with fresh production Radau, BDF, and tighter-Radau replays on both splits.
The final simulations discard all intermediate collocation states and restart
from the specified zero initials.

## Shared initialization, budgets, and failures

Each pair has one bounded native collocation solve. Both refinement routes read
that exact checkpoint, with its content hash, training fingerprint, parameter
vector, success status, and elapsed time retained. A failed initializer falls
back to the same frozen broad start in both routes and remains explicitly marked
as a fallback. A missing initializer is an ordering error, not permission to
silently run a different comparison.

Every logical fit has at most 600 seconds, including at most 120 seconds charged
for initialization. Both collocation routes are charged the original initializer
duration even though its CPU work is physically shared. A timeout/native crash
without a usable initializer report charges the full 120 seconds. The refinement
ceiling remains 150 optimizer function evaluations; actual residual calls are
reported separately. Replays are separately limited to 30 seconds per solver and
split. `total_fit_seconds` sums initialization and refinement; it excludes shared
reference generation, data loading, graph setup outside refinement, and replays.
This is time to the final fitted result, not the first time a trajectory-accuracy
threshold was crossed.

Each array task uses one CPU, no GPU, and 8 GB; concurrency defaults to two.
Guard/initializer/fit supervisors allow 660/180/840 seconds including grace.
Slurm limits are 15/5/20 minutes, with a 5-minute summary. The conservative sum
of worker ceilings is about 9.4 CPU-hours, plus summary; successful runs should
use substantially less. Queue wait is not included.

The stage order is guards → initializers → fits → summary. `afterany` dependencies
allow reports to preserve failures rather than losing their denominators. Fits
independently require a passing guard. Plans, source, dependencies, candidates,
starts, and task identities are frozen before numerical work. Repeating preparation
or a completed task verifies and reuses its checkpoint. Repeating submission
returns the recorded jobs; a partial submission stops for queue reconciliation.
An interrupted optimizer restarts at its frozen/cached start; it does not resume
the optimizer's internal iteration state. Native processes have hard wall limits
and process-group supervision.

## Recovery and parameter equivalence

`complete` means the independent numerical replay checks passed. A separate
post-fit diagnostic counts output recovery when clean-reference training **and**
validation NMSE are at most 1e-4. Reference scores do not choose parameters,
initializations, methods, or models. All planned outcomes remain in denominators.
No scientific compliance judgment is inferred from synthetic output accuracy.

For this specific zero-initialized model, let `a=1/tau`, `b=1/tau_p`, and
`q=m+k_p*p`. The input-to-memory transfer is

```text
q/u01 = k_u * (s + b + k_p) / ((s + a)*(s + b)).
```

The alternative parameters

```text
tau_new = tau_p
tau_p_new = tau
k_p_new = k_p + 1/tau_p - 1/tau
```

preserve that transfer, with all other parameters unchanged. The hidden state map
is `p_new=p`, `m_new=m+(a-b)*p`, `f_new=f`, so the nonlinear feedback is preserved
too. The alternative belongs to the allowed domain only when `k_p_new >= 0`.
Thus moderate `(tau,tau_p,k_p)=(2,4,0.4)` and `(4,2,0.15)` are equivalent output
representations. The separated truth's swapped gain is negative and is excluded.

The report retains raw parameter errors and adds the closest of these known
admissible aliases, minimizing `max_n |theta_n-alias_n|/max(1,|alias_n|)`.
This is a diagnostic about a known symmetry, not the complete equivalence class
or an identifiability proof. Neither coordinate convention nor alias distance is
a recovery gate. No transformation is applied inside fitting. This milestone
does not audit the private perturbation metric or establish that arbitrary hidden
coordinate changes preserve every public scientific requirement.

## Weak-matching correction

The previous eight-interval trapezoidal weak rule returned 2.76923077 for exact
data `x(t)=2+3t` governed by `x'=d`, instead of `d=3`. This 7.6923% bias comes from
inconsistent discrete integration by parts, before observational noise matters.

`matching_start` now defaults to `gauss5-linear-v2`: five-point Gauss quadrature on
each sample interval, using the piecewise-linear interpolation of the same
Savitzky–Golay smoothed observations. The nonlinear RHS features are evaluated
at the quadrature states and supplied inputs. The test function remains
`phi(q)=q^2*(1-q)^2`; its exact integral is `window_width/30`. This calculation
integrates a constant RHS and `-phi'` times a linear state exactly. It still has
smoothing/interpolation error on general trajectories and is an initializer,
not a claim of unbiased estimation from noise.

Regression tests cover positive/zero/negative constant drift, different window
lengths, the previous bias, nonlinear observed matching, and unsupported inputs.
The frozen v1 campaign explicitly requests `trapezoid-v1` to retain its historical
numerical behavior. Reproduce the actual v1 run with commit e58f702; new source
must not be used to resume an old freeze. The v2 Delta matrix focuses on latent
fitting and does not rerun the observed matching arms.

## Run and review on Delta

Use the exact commit and existing-directory-safe checkout command from the
handoff. The dedicated `fitter-methods-v1-deps` directory already contains
CasADi 3.7.2 from the previous run and is reused. No package installation or GPU
allocation is required.

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v2 && bash scripts/hpc/submit_fitter_methods_v2_delta.sh
```

When jobs finish, paste this compact report:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v2/summary.md
```

To regenerate a partial or final summary without resubmitting:

```bash
cd /projects/bibo/yxiao2/repos/autoformalism-fitter-methods-v2 && PYTHONPATH=src:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python scripts/run_fitter_methods.py summarize --output /work/hdd/bibo/yxiao2/phase_b/fitter-methods-v2
```

`summary.md` contains stage counts, paired aggregates, and the 36 concise fit rows.
`compact.json` adds alias labels and explicit shared-start checks. `details.md`
contains verbose diagnostics; `summary.json` retains complete records. Optimizer
success and initializer success are distinct from numerical verification and
output recovery. Do not delete submission manifests to force duplicate jobs.

## Limits and next decision

This remains a diagnostic adapter for supported smooth expressions, global
parameters, fixed numeric initials, and free rollouts. A successful result would
justify a separate transfer test on frozen proposer candidates, with Sol, under
their actual public data and prediction contracts. It would not justify enabling
collocation for every generated expression or changing scientific acceptance to
require recovery of one named hidden coordinate system. Mesh refinement, broader
systems, and larger independently crossed start/noise matrices remain later work.

## Local verification

The focused 28 tests passed, including actual shared collocation initialization,
both refinement routes, checkpoint reuse and tampering, initializer timeout
fallback, signed weak-drift consistency, the admissible state/parameter mapping,
and the four-stage launcher with duplicate/partial-submission protection.

Both full-size guards passed at truth and all three broad starts; the largest
relative sensitivity/central-difference discrepancy was 4.72e-5, below the 1e-3
gate. A full-size separated/noiseless C+S smoke completed without fallback and
passed independent replays, with clean train/validation NMSE approximately
2.10e-21/2.01e-21. Completed-task resume also passed. This single local smoke is
not the 36-fit Delta result or a cross-machine timing comparison.

The full regression suite passed: **1,173 passed, 3 skipped**. The skips are
optional Torch baselines unavailable in the local environment. Ruff, shell syntax,
CLI preparation, supervised execution, and compact-summary generation also passed.
