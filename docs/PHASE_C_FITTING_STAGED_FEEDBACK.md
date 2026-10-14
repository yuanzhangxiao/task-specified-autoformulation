# M32: staged diagnostics and piecewise local sensitivity

Implemented October 13, 2026. Experimental opt-in adapter; production fitting,
benchmark data, finalized prompts, and retained models are unchanged. This follows
[M31's result review](PHASE_C_FITTING_M31_RESULTS_2026-10-13.md). Delta qualification
is prepared, not yet completed.

## Why this milestone

M31's alien-device diagnostic exhausted its shared allowance during the second
independent output integration. Basin and CSTR passed replay but withheld stronger
parameter conclusions because their equations contain piecewise primitives. A
blanket piecewise exclusion is unnecessarily conservative; trusting an AD branch
convention alone is too permissive. We now inspect nearby independent rollouts.

The distinction remains: output accuracy, local derivative reliability, parameter
information, and optimization evidence are different claims. An accurate rollout
is not a coefficient-confidence certificate. A failed diagnostic is not evidence
that the fitted equations or public specification are false.

## Frozen numerical check

Write the joint coefficient/initializer vector as `theta`, fixed coordinate units
as the diagonal matrix `S`, and the complete training residual vector as `r(theta)`.
Residual ordering and training-only normalization are identical to the fitter.
The AD Jacobian `J = dr/dtheta` includes every joint coordinate. Structurally
output-inactive parameters are reported separately; weak columns are not removed
merely to improve numerical rank.

For each active coordinate axis, and the weakest right-singular direction of
`J S` restricted to active coordinates, evaluate both signs of

    theta_plus  = theta + h S d
    theta_minus = theta - h S d

at the predeclared dimensionless steps `h = 1e-3, 3e-4`. Every point is evaluated
by DOP853 and Radau, both with relative tolerance `1e-12` and absolute tolerance
`1e-14`. No parameters are optimized. Compare the AD direction `J S d` against

    forward  = [r(theta_plus) - r(theta)] / h
    backward = [r(theta) - r(theta_minus)] / h
    central  = [r(theta_plus) - r(theta_minus)] / (2 h).

Checks require agreement for both solvers, both signs, and both step sizes, plus
agreement between successive step estimates. The default RMS derivative tolerance
is `1e-8 + 0.02 * derivative_magnitude`. Solver discrepancies, a roundoff estimate,
and a residual floor of `1e-12` form an empirical noise floor, multiplied by four.
The observed perturbation signal must exceed eight times the conservative noise
scale. A tiny unresolved column cannot pass solely because all estimates are tiny.
These are explicit numerical qualification thresholds, not statistical confidence
levels or mathematical integration-error bounds.

Out-of-domain perturbations are recorded as infeasible and never clipped. A
boundary result needs a future constrained, one-sided assessment; it does not
prove unidentifiability. Disagreeing one-sided slopes are reported as an
unsupported kink at the tested scales, not as a proof of nondifferentiability.

The adapter also records sampled margins, ties and activity changes for `abs`,
`min`, and `max`, including causal initial maps. Sampling does not certify behavior
between observations. A branch crossing can still pass the numerical derivative
check: a kink in the instantaneous vector field does not automatically imply a
kink in the observed solution map.

Only complete supported checks allow `piecewise_numerically_supported` to enter
the existing joint-rank/conditioning assessment. The result can still be weak or
ambiguous. The packet never claims classical differentiability, global
identifiability, profile-likelihood coverage, or calibrated confidence.

## Reliable diagnostic execution

Each stage has an immutable identity binding the candidate, exact fitted vector,
training data, source/runtime, solver, tolerances and budget:

| Stage | Default allowance |
|---|---:|
| Primary DOP853 outputs | 120 s |
| Independent Radau outputs | 360 s |
| Joint sensitivity integration | 180 s |
| Each perturbation/solver call | At most 120 s |
| All perturbations combined | 1,800 s |

The second solver receives a separate larger allowance because M31's recorded
bottleneck was the strict independent replay. Increasing an allowance does not
relax numerical accuracy criteria. Smooth cases do not run piecewise probes.

A completed stage is sealed immediately. Resume reuses it without new integrations
or budget renewal. A stage started but interrupted is marked interrupted and
charged its full reserved allowance; its unknown wall/CPU/call totals stay
unknown. Unstarted subsequent stages may proceed within their original budgets.
Each failure identifies its stage and records the last trajectory/setup phase.
A sensitivity or perturbation failure preserves completed output evidence. A saved
stationarity witness stays in `historical_search_witness`; it enters the current
stationary-poor recommendation only after strict replay and qualified derivative
evidence at that exact point. Failed piecewise qualification cannot silently
inherit a stronger branch-AD stopping claim.

The packet separately reports started/completed calls, stage completion counts,
wall time, CPU time, budget charge, and historical fitting cost. Diagnostic timing
includes symbolic compilation and integration; lightweight artifact assembly and
hash verification are excluded. Deadlines are cooperative and may slightly
overshoot between checks; the scheduler's three-hour ceiling is an outer limit,
not an instruction to spend three hours.

If a partial replay contradicts the historical accuracy or metric, the adapter
reports `rollout_unverified` and exposes the available new metric. It cannot hide
that contradiction behind the other solver's timeout. Missing diagnostics retain
historical evidence with its origin clearly labeled.

## Qualification roster and exit criteria

The new campaign imports all 16 M31 entries, including its five already-fitted
analytic controls. None is refitted. Two missing sources and the entry without a
retained endpoint remain in the denominator. The previous feedback is used only
by the separate comparison report, never as an input to the diagnostic adapter.

Seven additional small controls cover an inactive guard, clean crossing, kink,
parameter bound, smooth initializer, guarded initializer, and a genuinely
stationary poor fit. Kink/bound controls deliberately verify prescribed points;
they do not claim parameter recovery. The stationary case obtains its witness
from an actual bounded fit at the retained vector. Evaluator expectations stay
outside the fitting/feedback interfaces.

This makes 23 tasks. Qualification requires preserved source availability and fit
identities, verified independent outputs for available retained points, preservation
of previously accurate outputs, expected diagnoses for old/new controls, and
training-only packet boundaries. Piecewise support is required only for the
appropriate positive controls; negative controls must withhold it. The historical
alien-device timeout remains recorded alongside the new result and costs.

An unsuccessful qualification is still a valid result: do not change thresholds
or erase failed tasks in place. Inspect the stage artifacts before designing a
separate follow-up protocol. All equations, parameters and initials remain fixed
throughout this diagnostic comparison.

## Delta commands

No GPU, LLM server, API key, or test data is required. Upload the delivered
`transfers/phase-c-fitting-m32.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`, then run:

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m32.tar.gz
bash phase-c-fitting-m32/scripts/hpc/submit_phase_c_staged_feedback_delta.sh
```

Defaults use the existing fitting Python environment and CasADi dependency folder,
account `bibo-delta-cpu`, one CPU and 16 GB per task, with at most three array tasks
running concurrently. Override `AF_CONCURRENCY` if needed. The archive contains a
sealed training-only M31 export, so no historical result directory is required.
The checksum manifest is verified before submission. The shared scheduler keeps
submission receipts and refuses to guess after an ambiguous scheduler response.
Do not manually duplicate a submission with an unconfirmed receipt.

After completion:

```bash
cd /work/hdd/bibo/yxiao2/phase_c
bash phase-c-fitting-m32/scripts/hpc/inspect_phase_c_staged_feedback_delta.sh
```

Default results:
`/work/hdd/bibo/yxiao2/phase_c/fitting-staged-feedback-v1/summary.json`.
The inspection command prints key results and the timestamped review archive to
return. Run it with an explicit campaign path if overriding the default root.

## Files and limitations

Core: `piecewise_assessment.py`, `staged_feedback.py` and the typed
`schemas/fitting-feedback-v2.schema.json`. The only legacy adapter change adds an
explicit derivative-qualified keyword; its default preserves M31 behavior.
Campaign, fixtures, tests and Delta scripts are separate M32 files.

This milestone diagnoses an unchanged model. It neither repairs piecewise laws nor
smooths them, refits archived models, rewrites specifications, or integrates with
the live construction controller. The two solvers can share numerical bias. Finite
local probes cannot exclude remote alternate parameter solutions. Future work may
add constrained one-sided checks and profile minimization after these diagnostics
are qualified on real archived models.

## Local verification

The frozen source snapshot passed 112 tests covering M32, M31 feedback,
independent sensitivity integration, confidence checks and piecewise fitting.
Three additional legacy tests that failed while the shared source tree was being
edited passed when rerun against this snapshot. The snapshot passed `ruff check .`;
Delta shell syntax, CLI loading, actual 23-entry input preparation/verification,
and an unavailable-source report/resume smoke also passed.

Whole-repository completion is not claimed. The shared-checkout `pytest` run was
stopped after 1,369 passes, three failures and one setup error involving source
identity changes during concurrent work. A broader parallel snapshot run was
stopped after 929 passes and ten failures caused by absent untracked private
benchmark assets and the checkout-relative `.venv` path. These assets are
intentionally absent from the delivery. The original checkout's global Ruff scan
also reports unrelated findings in `analysis/claude`; those files were not edited.
The full 23-task M32 numerical qualification still requires the Delta run above.
