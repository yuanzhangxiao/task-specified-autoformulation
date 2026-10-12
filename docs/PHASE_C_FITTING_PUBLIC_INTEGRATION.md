# M29: qualify the rollout fitter on public construction requests

Implemented October 12, 2026, following the
[M28 review](PHASE_C_FITTING_M28_RESULTS_2026-10-12.md).
This is an opt-in integration qualification, with no new construction calls,
benchmark changes, test access or production-default change. Astra's construction
work is unaffected. The public fitting dispatcher still uses its existing paths.

The [completed M29 review](PHASE_C_FITTING_M29_RESULTS_2026-10-12.md) accounts for
all eleven entries. All three known controls recover accurately; the six available
archived constructions remain inaccurate or numerically unavailable. Equation
inspection identifies missing output dependencies and wrong physical signs. The
independent-basin example reaches its analytic within-model optimum but exposes
flat-outer-profile failures and wasted fallback work. Next qualify output-inactive
coordinate handling and stationary-poor-fit stopping; keep production defaults
unchanged.

## What this milestone tests

`autoformalism.fitting.public_rollout.run(inputs, policy, folder)` accepts a typed
`Inputs` containing a `PublicFitRequest`, a training-only split, and optionally a
complete incumbent vector. Unlike the M28 interface, callers need not construct
optimizer coordinates or restart portfolios themselves.

1. Validate the exact training channel roles against the public request.
2. Lower the candidate and public initialization plan using the existing compiler.
   Preserve equations, shared process coefficients, roles, bounds, observation
   mappings, and initializer expressions. Observed states start at their initial
   observations; hidden initial maps remain globally parameterized.
3. Preserve supplied guesses. For an omitted coefficient, use the midpoint of
   the existing runtime start interval after intersecting it with the declared
   domain. Typically this is 1 for an unrestricted coefficient or 1.5 for a
   positive/nonnegative coefficient. This is a deterministic generic start, not
   a physical estimate. Invalid supplied guesses are reported, not clipped.
4. Derive coordinates from training observations and the supplied public initial
   maps. Generate three domain-only restart vectors using the request's seed.
   Freeze the original plus restarts before numerical fitting. No reference
   coefficient, old fitted endpoint, or validation score selects these vectors.
5. Use the M28 router: exact affine profiling, exact terminal-output profiling,
   or joint rollout when the relevant symbolic certificate does not apply.
   Guarded scaling is the default; the unproven fixed alternative stays optional.
6. Preserve the verified incumbent through failed searches. Seal the result before
   the separate evaluator receives validation. A supplied distinct incumbent is
   checked independently; it does not bypass training verification.

The result retains the complete M28 `FitResult`: parameters, origin, routing,
training verification, separate numerical/fit/search/sensitivity assessments,
and cost accounting. It adds the initialization audit, which guesses were
defaulted, lowered-model identity, coordinates, restart vectors and setup cost.
Interrupted setup is terminal and retains its charged allowance; interrupted
search operations use the existing durable journals without a budget reset.
Terminal reuse checks input/policy/source/runtime and sealed backend identities.

Valid-schema requests that fail lowering, initializer binding, domain checks or
symbolic capability checks produce `setup_unavailable` with the exact diagnostic.
They remain in the report. Fixed models without optimization variables currently
require an evaluation-only path and are explicitly unsupported by this fitter.
Piecewise derivatives require explicit opt-in (enabled for this pilot); they
are branch-local conventions, not global smoothness guarantees. No additional
scientific sign, monotonicity or state-domain certification is inferred here.

## Fixed roster and provenance

Use **Full, seed 0, for every cell** from the September 30 construction archive.
Later construction studies do not yet provide complete equation/initializer
handoffs. This tests actual archived proposer outputs, not the latest constructor
and not models selected for favorable old fitting scores. Two source construction
failures remain in the eleven-task denominator.

| Archived case | States | Targets | Total fitted variables, including initial-map coefficients | Preflight routing |
|---|---:|---:|---:|---|
| Dalla Man T1 easy | 1 | 1 | 2 | Affine profiling |
| Dalla Man T1 hard | 1 | 1 | 1 | Joint rollout |
| Dalla Man T2 easy | — | — | — | Construction failed |
| Dalla Man T2 hard | 3 | 2 | 10 | Joint rollout |
| CSTR easy | 1 | 1 | 3 | Joint rollout |
| Alien-device easy | — | — | — | Construction failed |
| Coupled basins | 2 | 1 | 4 | Joint rollout |
| Independent basins | 2 | 1 | 7 | Terminal-output profiling |
| Known alien hard control | 6 | 1 | 18 | Terminal-output profiling |
| Known basin control | 2 | 1 | 2 | Joint rollout |
| Known nonlinear CSTR hard control | 3 | 1 | 9 | Joint rollout |

The constructed T2 model initializes its hidden state with an affine function
of initial insulin and auxiliary observations. Basin models also use shared
affine initial maps. These are useful integration cases beyond the constant
hidden initials in the known controls. Capability success is not scientific
compliance or successful fitting.

The three known controls use the same seed-0 requests and public observations
as M28. Their restart lists now follow the public-request seed rule rather than
M28's problem-hash seed rule. Thus this is not a matched timing comparison with
M28. Ground-truth coefficient/initial errors are computed only for these three
controls, after fitting; they have no meaning for unmatched discovered states.

- Construction plan: `315ca6edb7c6dac776f00fb450e8a9f0cc584e7a9c0cbc1fbea5e0c47fd92780`.
- M28 input: `eb1d681faa7d98af0d0130e010f7099513b71e041f41c0ae574632770ae142a2`.
- M29 input: `cce09c7f7b25573cd7c0c1549b15534d78a18f0e1fff631ebae8cf0148517744`.

The exporter verifies old plan/proposal seals and task identities, reconstructs
each public request, and checks it against the saved fitting freeze when present.
It copies public arrays without regeneration. It does not use saved fitted
coefficients or scores. The portable export contains no proposer logs or test data.

## Budgets, metrics and exit criteria

Each available model receives the M28 allowance: at most six blocks, forty
evaluations or 800 seconds per block, with at most two consecutive blocks per
start. Restarts share that allowance. Independent training checks and local
sensitivity diagnostics retain their existing allowances. Setup gets a recorded
180-second allowance; setup is cooperative rather than a separate hard-kill
subprocess. All setup/search/evaluator time and unknown interrupted usage are
reported separately. No fit is launched for the two source failures.

The independent evaluator uses unchanged fitted parameters and initializer
coefficients on every training/validation trajectory, with the existing bounded
300+300-second numerical retry. It reports aggregate and per-target NMSE, solver
agreement and the original per-trajectory records. It never refits validation.

Exit criteria: all eleven outcomes accounted for; request and initializer
identities preserved; exact resume and evaluator separation pass; the known
controls still demonstrate accurate prediction and coefficient recovery; every
numerical/domain/capability failure is actionable and retains its costs. Evaluate
the six constructed models as imperfect-structure cases. Poor fit or failure
to reach the noiseless target does not establish structural infeasibility.
Completion alone does not mean the fit is accurate. Local sensitivity is not
global identifiability or calibrated confidence. Noise-aware stopping and broader
coverage remain future work before production promotion.

## Delta commands

Upload `transfers/phase-c-fitting-m29.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. It includes pinned committed source, frozen
inputs, `SOURCE_COMMIT`, `SHA256SUMS`, tests and launchers.

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m29.tar.gz
bash phase-c-fitting-m29/scripts/hpc/submit_phase_c_public_integration_delta.sh
```

Default root: `/work/hdd/bibo/yxiao2/phase_c/fitting-public-integration-v1`.
There are eleven array entries (nine fits, two source-failure records), with
one CPU, 16 GB, three concurrent tasks and a 2.5-hour scheduler ceiling per entry.
No GPUs or LLM calls are needed. The launcher uses the existing Delta fitting
environment and verifies bundle hashes. Set `AF_CONCURRENCY` to adjust only this
array's throttle, or `AF_INTEGRATION_ROOT` to choose a new campaign root.
Repeating the same command resumes confirmed submissions. An uncertain scheduler
reply requires reconciliation, not blind resubmission.

Inspect partial/completed results and create a review archive:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m29/scripts/hpc/inspect_phase_c_public_integration_delta.sh
```

Download the printed `review-*.tar.gz` path. `summary.json` includes every source
failure, setup diagnostic, fit assessment, evaluator score and cost record.

## Local verification

All nine available requests pass numerical capability preparation locally.
The analytic two-output smoke checks the shared causal initializer, independent
validation replay and exact terminal resume; train/validation NMSE is
9.42e-21/1.89e-21. This small smoke is interface verification, not evidence of
performance on the nine full experimental fits. Those remain for Delta.

Tests cover training-only schemas, missing guesses, domain rejection, shared
initialization, multiple targets, incumbent handoff, piecewise opt-in,
interruption accounting, provenance, fixed denominators, post-fit evaluation,
per-target scoring, and idempotent/uncertain scheduler submissions.

The selected regression suite passes 161 tests, including existing public
fitting, initialization, sensitivity, profiling, scaling, adapter and replay
tests. Changed Python files pass Ruff, and all three shell launchers pass syntax
checks. Repository-wide `ruff check .` reports the same 37 pre-existing findings
in four unrelated `analysis/claude` files; those files are unchanged.
