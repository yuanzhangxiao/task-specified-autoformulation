# M27 results: guarded scaling recovers all fifteen tested case/start combinations

Reviewed `review-20261011-210137.tar.gz` on 2026-10-11. All 45 tasks completed,
with complete cost accounting. Guarded relative scaling reaches the independently
verified training target on all fifteen case/start combinations. Fixed and
iterative Jacobian scaling each miss one alien-device start. All methods recover
the nonlinear CSTR and coupled-basin cases.

One guarded basin endpoint has accurate coefficients and passes the fitter's
training check, but the retrospective evaluator flags a validation integrator
discrepancy. That flag remains in the frozen results; a separate fixed-parameter
local diagnostic is described below. Completion, training accuracy, coefficient
error and solver agreement are distinct outcomes.

This strengthens the case for guarded scaling as the first strategy in a
qualified fitting adapter. It does not establish a universal scaling winner or
change production defaults. These are noiseless known-structure experiments,
with five initializations per structure, not fifteen independent scientific
tasks. Alien starts 0–2 repeat the M26 inputs.

## Provenance and audit

- Protocol: `phase-c-broad-scaling-1`; [M27 runbook](PHASE_C_FITTING_BROAD_SCALING.md).
- Source commit: `0864a5f616602d3a387587b7be099367056d8ff0`.
- Plan SHA256: `5bbe6564701978fc1698f6ea0867b781214aeda4566e85bd8f8c561c916f505e`.
- Input SHA256: `1f2cecce17b068e6cf82fbe5b8c3753054555c29f82c26e3f88918b8ffce01b4`.
- Archive SHA256: `4079ee89ac7f417dbbabb86eaf7ce5f4bdf09a9fe40a2f6adfd9860a1b638c82`.
- Delta jobs: prepare `22810795`, array `22810796`, report `22810797`.
- Output root: `/work/hdd/bibo/yxiao2/phase_c/fitting-broad-scaling-v1`.
- Runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3, CasADi 3.7.2,
  Pydantic 2.12.5. One CPU per task; no GPUs or LLM calls.

All 807 sealed JSON artifacts validate. Rebuilding the summary reproduces the
supplied summary exactly. Each retained parameter vector matches its selected
verified checkpoint. All arms use the same frozen original and restart vectors
within each case/start; their first evaluated parameter-vector hashes also
match. Backend policies match the plan and retain the training-only selection
boundary. No validation/reference values are used by the optimizer; no test
data are opened.

All 258 recorded operations complete. Independent retrospective replays cover
735 complete model/trajectory records, each with Radau and DOP853: twenty
trajectories per alien/CSTR endpoint and nine per basin endpoint. The archived
preparation check reports 33 tests passed. Worker stderr contains only the
explicit-integrator warning that its `jac` argument has no effect; this does
not mean the optimizer's residual Jacobian was unused.

## Recovery and cost

The strict training target requires aggregate NMSE at most 1e-12 for both
verification solvers, worst-trajectory NMSE at most 1e-11, and solver agreement.
All arms receive the same six-block ceiling, forty evaluations or eight hundred
seconds per block, and the same predetermined restart portfolio. Actual work
stops adaptively. Fit time includes setup, search, training verification and
final sensitivity; retrospective evaluation is separate. Medians include the
unsuccessful fits.

| Case | Scaling | Verified training target | Frozen coefficient-recovery flag | Median fit minutes | Summed fit minutes |
|---|---|---:|---:|---:|---:|
| Alien-device hard | Iterative Jacobian | 4/5 | 4/5 | 18.02 | 131.60 |
| Alien-device hard | Fixed coordinates | 4/5 | 4/5 | 10.81 | 108.86 |
| Alien-device hard | Guarded relative | 5/5 | 5/5 | 9.50 | 82.14 |
| CSTR hard | Iterative Jacobian | 5/5 | 5/5 | 6.09 | 35.34 |
| CSTR hard | Fixed coordinates | 5/5 | 5/5 | 5.51 | 28.83 |
| CSTR hard | Guarded relative | 5/5 | 5/5 | 5.84 | 29.23 |
| Coupled basin | Iterative Jacobian | 5/5 | 5/5 | 1.93 | 9.32 |
| Coupled basin | Fixed coordinates | 5/5 | 5/5 | 1.94 | 10.65 |
| Coupled basin | Guarded relative | 5/5 | 4/5* | 1.94 | 10.47 |

*The coefficient-recovery flag combines a retrospective coefficient-error
threshold with an integrator-agreement gate. Numerically, all five guarded basin
coefficient vectors satisfy the coefficient threshold; one misses the latter
gate. Across the study, 43/45 reach the training target, 43/45 have coefficient
errors below the retrospective threshold, and 42/45 pass the combined frozen
coefficient-recovery flag. Do not report the flagged endpoint as a coefficient
estimation failure or silently overwrite its verification result.*

Total recorded fitting time is 446.45 task-minutes; retrospective evaluation
adds 43.85 task-minutes. These sums are neither elapsed campaign time nor billed
service units. One alien iterative search starts 224 calls and completes 223
before its allowance ends; the interrupted call remains charged. All other
search calls complete. No domain-failure events are recorded.

## New alien start distinguishes the scaling policies

The alien model has six states, thirteen dynamic coefficients and five shared
hidden initials. Exact terminal-output profiling eliminates four gains, leaving
fourteen outer unknowns. Internal couplings and shapes remain fixed and anchor
the latent coordinates.

| Original start | Scaling | Train NMSE | Validation NMSE | Max. coefficient relative error | Fit minutes |
|---:|---|---:|---:|---:|---:|
| 2 | Iterative Jacobian | 0.1098 | 0.2037 | 715% | 75.27 |
| 2 | Fixed coordinates | 1.13e-16 | 2.41e-16 | 0.0000238% | 10.81 |
| 2 | Guarded relative | 2.18e-15 | 7.72e-15 | 0.0000876% | 12.75 |
| 3 | Iterative Jacobian | 8.68e-14 | 4.38e-13 | 0.000617% | 18.02 |
| 3 | Fixed coordinates | 0.0828 | 0.2009 | 420% | 70.24 |
| 3 | Guarded relative | 4.28e-13 | 9.07e-13 | 0.000682% | 44.75 |

Guarded start 3 needs a genuinely different start. The original start's two
blocks reduce training NMSE only to approximately 0.614 and 0.610. The first
predetermined restart reaches approximately 0.00203, then 4.28e-13 after a
continuation block. Total search use is 137 calls over four blocks. Reserved
restart capacity is therefore part of the successful strategy; scaling alone
does not rescue its original branch.

Iterative scaling succeeds on this start faster than guarded scaling, while
failing start 2. Fixed scaling has the opposite pattern. Guarded scaling is the
only policy successful on every tested alien start, but it is not uniformly
fastest. Its summed alien fitting time is about 25% below fixed scaling and
38% below iterative scaling in this small experiment, including failures.

Successful alien endpoints have maximum dynamic coefficient relative error at
most 0.000899%, and maximum absolute hidden-initial error at most 7.16e-6.
Their largest validation NMSE is 1.39e-12. Both poor endpoints retain full local
joint sensitivity rank 18/18, with condition numbers about 23,519 and 6,802.
Successful endpoints also have full rank, with conditions about 890–1,078.
Rank alone neither certifies a good optimum nor proves global identifiability.
Recovery on the same data with another strategy demonstrates that an accurate
solution is reachable; the two failures cannot be blamed simply on an absence
of any informative solution.

## Nonlinear CSTR and basin results

CSTR jointly fits seven dynamic coefficients, including nonlinear kinetic
parameters, and two shared hidden initials. It uses direct rollout fitting,
without exact terminal-output profiling. All fifteen fits succeed from their
original starts; one iterative fit needs a second block, and the rest need one.
The worst successful coefficient error is 0.00285%; the largest jacket-initial
temperature error is 1.14e-4 K. Maximum validation NMSE is 3.23e-13.
All final local sensitivity ranks are 9/9, with conditions about 540–707.

The basin fits two coefficients inside coupled threshold dynamics; its initial
depths are public, so it has no fitted hidden initials. All fifteen fits reach
the training target, with maximum coefficient error 0.000306%. Local ranks are
2/2. The fixed and guarded arms each use one restart on different starts; all
other basin tasks finish in one block. These results establish successful
branch-local optimization on these instances, not global differentiability at
threshold crossings.

Thus the improved fitting strategy works beyond a profiled terminal output.
The CSTR result is especially useful: nonlinear coupled coefficients and latent
initials are recovered by the joint-rollout route. Noisy measurements, unknown
structure and unanchored latent coordinate systems remain untested here.

## One basin verification discrepancy

`basin_coupled_s0__guarded_relative` retains:

| Parameter | Fitted | Reference |
|---|---:|---:|
| `k_outlet` | 54.00000004124662 | 54 |
| `k_transfer` | 71.99999493730063 | 72 |

Maximum relative coefficient error is 7.03e-8, or 0.00000703%. The fitter's
tightened training check passes, with maximum normalized solver discrepancy
6.71e-11. The separate retrospective evaluator, however, uses its original
rtol=1e-9 and atol=1e-11 and obtains discrepancy 3.03e-4 on `val_001`, above its
1e-4 limit. On that trajectory Radau NMSE is 3.09e-16, while DOP853 NMSE is
1.81e-9. The summary reports Radau scores, which explains why tiny reported
NMSE and a failed solver-agreement flag coexist.

A bounded local diagnostic holds the fitted parameters, data and equations
fixed and repeats only this trajectory. At nominal rtol=1e-9, the discrepancy
does not reproduce locally. Tightening to rtol=1e-11 gives discrepancy 2.41e-9;
rtol=1e-12 gives 8.21e-11. Both tightest solvers give trajectory NMSE about
2.93e-16. No optimization is performed. This supports a numerical verification
issue rather than an inaccurate parameter vector, but does not reproduce or
fully isolate the Delta-specific default-tolerance behavior. The local Python
version is 3.12.10 rather than 3.12.13; listed scientific package versions match.

The diagnostic is retained separately at
`output/fitting-m27-basin-tolerance-diagnostic.json`; archived fitted results and
flags are not changed. It is a reason to add a bounded tighter-tolerance retry
to the retrospective verifier, with separate initial/final diagnostics and cost,
not a reason to loosen its agreement criterion or refit using validation.

## What the scaling experiment actually exercises

Every policy operates on the same existing dimensionless physical coordinates
and normalized mean-square objective. Guarded relative scaling partially balances
Jacobian column magnitudes at each block's start and freezes that adjustment
within the block. The observed scale ranges are:

| Case | Smallest adjustment | Largest adjustment |
|---|---:|---:|
| Alien | 0.281 | 4.901 |
| CSTR | 0.338 | 3.648 |
| Basin | 0.787 | 1.615 |

No scale hits the 0.1/10 clipping limits; no weak-column guard activates.
The experiment supports partial balancing and blockwise freezing together with
the existing restart/retention strategy. It does not measure the empirical
benefit of clipping or isolate each safeguard's contribution.

## Recommended next milestone

Prepare a common fitting adapter around the qualified rollout machinery:

1. Route to exact profiling only when the symbolic certificate applies;
   otherwise fit parameters and hidden initials jointly by rollout.
2. Use guarded scaling as the provisional first strategy, retain fixed scaling
   as a bounded alternative, and preserve capacity for predetermined different
   starts. A combined portfolio needs its own matched-budget check; do not add
   the full budget of each strategy and call it the same experiment.
3. Return separate training fit quality, solver agreement, local joint
   sensitivity, search status and computational cost. Actual coefficient errors
   remain evaluator-only; a fitting confidence report cannot claim them without
   reference values or calibrated uncertainty.
4. Add a bounded fixed-parameter tighter replay when verification disagrees,
   preserving the original failure record and charging the retry. Validate this
   with the basin endpoint and a genuine disagreement/failure case so the retry
   cannot silently convert persistent failures into success.

Qualify this adapter first on the same frozen known-structure inputs and
budgets, including failure, resume and training/validation separation tests.
After that, connect it to constructed candidates and distinguish structural
misfit from numerical fitting failure. Production integration, new remote jobs
and changes to benchmark data are not part of this results review.

## Review verification

The archive checks and the fixed-parameter diagnostic above are review smoke
tests. The following focused pytest command passes all 25 tests:

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q tests/test_broad_scaling.py \
  tests/test_optimizer_scaling.py tests/test_scaling_recovery.py
```

`ruff check .` reports 39 issues in unrelated existing/concurrent files: 37
under `analysis/claude/` and two in construction files being edited separately.
They are left untouched. The documentation diff passes `git diff --check`.
This review changes only this results note and the M27 runbook's status/link;
no implementation files or benchmark artifacts are modified.
