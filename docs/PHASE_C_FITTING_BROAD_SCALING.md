# M27: broader qualification of rollout scaling

Implemented October 11, 2026; remote experiment not yet run. This follows the
[M26 review](PHASE_C_FITTING_M26_RESULTS_2026-10-11.md). Fixed and guarded scaling
both recovered all three alien-device starts in M26. That small comparison does
not justify choosing a universal default.

## Frozen comparison

Run 45 tasks: three correct equation structures, five predetermined original
starts per structure, and three paired scaling policies. Each task gets the same
three further restart candidates as the other policies for its case/start.
All candidates are frozen before any fitting. Retain every outcome, including
unavailable rollouts and missing tasks, in the denominator.

| Case | States | Dynamic parameters | Shared hidden initials | Selected kernel |
|---|---:|---:|---:|---|
| Alien-device hard | 6 | 13 | 5 | Exact terminal-output profiling; 4 gains eliminated |
| CSTR hard with nonlinear parameters | 3 | 7 | 2 | Joint parameter/initial rollout fitting |
| Coupled detention basin | 2 | 2 | 0 | Joint rollout fitting; public initial depths |

The policies are unchanged from M26: `jacobian`, `fixed_coordinates`, and
`guarded_relative`. Exact profiling is used whenever the existing symbolic
certificate applies; nonlinear coupling does not get approximated as linear.
All methods retain the same normalized training objective, independent
DOP853/Radau verification, conservative incumbent retention, continuation rule,
failure accounting and final joint parameter/initial sensitivity diagnostic.

This is an assisted **known-equation** experiment, not construction performance.
Alien internal couplings/shapes remain fixed and anchor its latent coordinates.
CSTR includes unknown nonlinear kinetic parameters and shared concentration and
jacket-temperature initials. Basin coefficients operate inside a coupled
threshold model; the absence of fitted hidden initials follows its public
initial-reading contract. Local sensitivity rank is reported, not assumed to
prove global identifiability. No statistical confidence coverage is claimed.

## Inputs and starts

The exporter copies existing sealed development observations and equation
templates from M3, M6 and M26. It neither regenerates nor edits benchmark arrays,
prompts or released input conventions. The frozen export digest is
`1f2cecce17b068e6cf82fbe5b8c3753054555c29f82c26e3f88918b8ffce01b4`.

Alien starts 0–2 retain the exact M26 requests and coordinates. Restart lists use
the same deterministic rule (local reconstruction differs from the saved Delta
values by at most 4.5e-16 due to floating-point arithmetic). The new plan freezes
one identical list for all three arms on the execution platform.
CSTR and basin starts 0–2 use their original generic-start rules. Starts 3–4
use predetermined seeds and a wider log perturbation window of ±1.5 for the
generic positive coefficient guesses. CSTR concentration initials stay
nonnegative, jacket-temperature guesses use additive perturbations, and alien
hidden initials are signed. Additional restarts use the existing declared-domain
sampler and numerical coordinate units. They are not filtered using outcomes.

The fitter receives only `TrainingProblem` (request, training arrays, frozen
coordinates, original start and initially identical incumbent). Validation and
reference values remain in the evaluator payload. Seal the complete fitter
result before computing validation or coefficient/initial errors. Never select
the best seed by validation for reporting this comparison.

## Piecewise expressions

The original smooth-only sensitivity contract rejects the CSTR temperature
guard and basin overflow hinges. M27 explicitly opts these cases into the
**existing** restricted piecewise expression evaluator through a separate
`PiecewiseTrainingProblem` type. Ordinary `TrainingProblem` serialization and
smooth-only defaults remain unchanged. The equations are not smoothed or edited.

Branch derivatives at ties are numerical conventions, not classical derivative
certificates. Exact-value rollouts remain the acceptance check. Unit tests check
directional derivatives across crossings in a small coupled ODE and verify that
the ordinary contract still rejects it. An additional local preflight on the
actual seed-0 inputs gave relative directional discrepancies of about 3.1e-8
for CSTR and 2.3e-6 for basin using an appropriate finite-difference step.
Basin finite differences are sensitive to integration noise: at step 1e-4 the
default integrator yielded about 4.6e-3; tightening integration reduced this to
2.3e-6. These point checks do not certify all future iterates or change the
matched integration tolerances. Any persistent basin discrepancy warrants a
separate numerical investigation, not an identifiability conclusion.

## Budgets and reporting

Each task has at most six search blocks, 40 charged evaluations or 800 seconds
per block, and at most two consecutive blocks per start. Reserve capacity for
unvisited starts; stop early on independently verified training accuracy.
Verification has a separate 180-second ceiling per check, sensitivity 180 seconds,
and retrospective replay 300 seconds. This matches M26. Search has a maximum
240 evaluations/80 minutes; the Slurm limit is 2.5 hours including other stages.

Use Delta CPUs: one CPU and 16 GB per task, three concurrent tasks by default,
no GPUs or LLM calls. The array throttle is adjustable with `AF_CONCURRENCY`.
It is our launch setting, not a statement about Delta's account limit.

Reports group outcomes by case and scaling policy. Each group retains a
denominator of five and separately reports training accuracy, coefficient
recovery, evaluator availability, fit time and evaluator time. Detailed records
include actual calls, block decisions, kernel routing, fitted coefficients,
joint sensitivity, scale vectors, weak-column guards and clipping behavior.
Median costs include recorded failures; inspect missing costs before comparing.
Unfinished operations retain their charged budgets on deterministic resume.

Use the numerical checks to separate a poor basin from a weakly constrained
coefficient or a verification failure. Ground-truth coefficient and initial
errors are retrospective diagnostics only. Repeated starts are not independent
scientific tasks. This milestone does not promote a production default.

## Delta launch and inspection

Upload `transfers/phase-c-fitting-m27.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`, then run on the Delta login node:

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m27.tar.gz
bash phase-c-fitting-m27/scripts/hpc/submit_phase_c_broad_scaling_delta.sh
```

The archive includes the frozen input file and committed source. It uses the
existing Python/CasADi environment from prior Delta fitting runs. The output
defaults to `/work/hdd/bibo/yxiao2/phase_c/fitting-broad-scaling-v1`.
The launcher checks archive members before submission. Submission intents and
receipts prevent blindly duplicating a job after an ambiguous scheduler reply.

Inspect progress and produce a review archive using:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m27/scripts/hpc/inspect_phase_c_broad_scaling_delta.sh
```

Download the exact `review-*.tar.gz` path printed by that command. `missing`
means a task has no completed record; consult Slurm before treating it as a
scientific failure. Do not resubmit to a new root merely to reset an interrupted
budget. Existing completed tasks are reused exactly on resume.

## Local verification

- 93 focused tests passed across scaling, recovery, profiling, sensitivities,
  campaign sealing, paired starts and submission idempotence.
- Six actual small synthetic fits passed: three policies each on a profiled
  terminal-output model and a joint nonlinear model. Verified training NMSE was
  below 1.5e-12; completed resume returned identical results.
- All 15 real start portfolios constructed successfully. The first three alien
  portfolios reproduce M26 to roundoff; all three structural routes were checked.
- Changed Python files pass Ruff; shell launch/worker/inspection scripts pass
  `bash -n`. Whole-repository Ruff still reports 37 pre-existing violations in
  `analysis/claude/`; unrelated files were left intact.
- No remote submission, benchmark fitting or test-data access was performed.
