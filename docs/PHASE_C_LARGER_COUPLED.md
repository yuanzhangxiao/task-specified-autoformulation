# M18: larger coupled linear fitting controls

Protocol `phase-c-larger-coupled-1`; config
`configs/phase_c_larger_coupled_v1.json`. Approved after M17 recovered all twelve
paired polishing endpoints. This is a numerical qualification, separate from the
28-cell benchmark release. It does not change production fitting or construction.
Nonlinear benchmark blocks and coefficient profiling within collocation come later.

## Question and equations

Does exact rollout profiling remain useful as the coupled block grows? Fewer
nonlinear optimization variables require more integrated basis/sensitivity states,
so success at two states does not establish a speedup at six states.

For n=3 or 6, use the complete known structure

```
x0' = -a0*x0 + w0*x1 + gain*u(t)
xi' = -wi-1*xi-1 - ai*xi + wi*xi+1       (0 < i < n-1)
xn-1' = -wn-2*xn-2 - an-1*xn-1
y = x0
```

All `ai` and `gain` are unknown positive coefficients bounded in [0.01,30].
Couplings are fixed at `(0.9,1.1,0.8,1.2,1.0)`, truncated to the required length.
They anchor the units and order of the latent coordinates. Each hidden initial
value is an unknown signed parameter, shared across trajectories; observed x0
starts at its permitted initial observation. No hidden trajectories or derivatives
enter fitting. This is deliberately not an arbitrary latent A,B,C realization
with a similarity-transform ambiguity.

| Case | Reference diagonal a | Dynamic coefficients | Hidden initials |
|---|---|---:|---:|
| linear3 | 0.4, 0.7, 1.0 | 4 | 2 |
| linear3_fast_slow | 0.15, 1.5, 12 | 4 | 2 |
| linear6 | 0.3, 0.45, 0.65, 0.85, 1.05, 1.25 | 7 | 5 |
| linear6_fast_slow | 0.15, 0.35, 0.8, 2, 5, 12 | 7 | 5 |

Reference gain is 1.4; hidden initials are `(0.35,-0.25,0.2,-0.15,0.1)`, truncated.
These values belong only to the generator/evaluator. Generic starting coefficients
are exp(Uniform(-1,1)); hidden initials are Uniform(-1,1), with three fixed seeds.
For a given dimension, starts are identical across timescale cases and methods.
No candidate/start is selected by reference or validation performance.

The state matrix is negative diagonal plus a known skew-symmetric coupling, so
its symmetric part is strictly negative definite throughout the permitted parameter
domain. This isolates scaling, conditioning and optimization difficulty while
avoiding unstable proposed dynamics. It does not test arbitrary unstable skeletons.
Diagonal separation is not identical to separation of the system eigenvalues;
the evaluator records the real parts of those eigenvalues as well.

## Data and qualification

Each case has three training and two held-out validation input schedules over
[0,24]. The continuous input is piecewise linear with integer knots and levels
rounded to eighths. Observations occur every 1/8 time unit, with additional 1/32
samples in the first unit: 217 samples per trajectory. The binary-exact grid and
levels avoid artificial forcing corners from floating-point rounding; the usual
runtime boundary policy is unchanged. Both fitters see exactly the same samples.

A separate handwritten matrix-exponential simulator generates the noiseless
controls. It augments the linear system with input value and slope on each input
interval. Original-equation Radau and DOP853 replays verify those targets on the
executing host. Preparation requires mean reference NMSE <=1e-10 and normalized
solver discrepancy <=1e-5 on both development splits.

Before fitting, all coefficient and initial-value output sensitivities must have
full column-normalized numerical rank, with smallest/largest singular-value ratio
>1e-8. Export checks finite differences of the matrix-exponential solution; the
host preparation independently repeats the check with forward sensitivities of
the ordinary equations. Reports retain singular values and raw column norms.
These are local excitation/conditioning checks, not proofs of global uniqueness
or noise robustness. A failed case blocks the array; it is not dropped silently.

## Matched methods and budget

Four cases x three starts x two methods = **24 CPU tasks**:

- `rollout_only`: jointly optimize all 2n unknowns through the original ODE.
- `coupled_profiled_rollout`: optimize n diagonal decays; solve bounded least
  squares jointly for the input gain and n-1 hidden initials at every outer point.

Profiling is exact in its inner variables up to numerical tolerances because the
transition matrix depends only on the outer decays. It does not make the outer
objective convex. Full feedback and parameter domains are preserved. With forward
sensitivities, the direct method integrates 21/78 quantities at n=3/6; the current
profiled basis integrates 48/294. This extra work is part of the comparison.

Both methods receive **600 total fitting seconds and 600 residual evaluations**,
including setup, checkpoint handling and one optional polishing restart. Initial
training targets are mean NMSE 1e-8 and worst-trajectory NMSE 1e-7. After independent
certification, retain that vector and use only remaining budget toward 1e-11 and
1e-10. Each phase reserves up to 90 seconds from the shared allowance for checking;
the M17 method-specific point caps and cleanup behavior are unchanged. A failed
polish preserves the certified first vector. No coefficient errors guide stopping.

After final training selection is sealed, independently score first/final distinct
vectors with 180 seconds each. Accuracy requires train and validation NMSE <=1e-6
and solver agreement; coefficient recovery requires every dynamic coefficient
within 1%, and initial recovery requires every hidden initial within 0.001 in
anchored units. Report every start, including numerical failures. Instrumentation
separates imports, integration, linear solves and checkpoint writes, using CPU and
wall time. Inclusive spans overlap and must not be summed indiscriminately.

Completed work resumes without fitting again; interrupted fits preserve checkpoints
without resetting budgets. Source, input, runtime and scheduler identities remain
sealed. An uncertain submission must be reconciled before resubmission. There are
no LLM calls, GPUs or test-data access.

## Delta commands

Upload `transfers/phase-c-fitting-m18.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. It contains pinned code, tests, configuration and
the frozen input file; no previous campaign directory is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m18
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m18.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_larger_coupled_delta.sh"
BASH
```

Default root: `/work/hdd/bibo/yxiao2/phase_c/fitting-larger-coupled-v1`.
Six concurrent one-CPU tasks, 16 GB each, using the existing fitting environment.
The scheduler reserves 35 minutes per task; the algorithmic fitting budget remains
ten minutes. `AF_CONCURRENCY`, `AF_PYTHON`, and `AF_CASADI_ROOT` may override platform
settings without changing fit policies. Do not modify dependencies used by active jobs.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m18/scripts/hpc/inspect_phase_c_larger_coupled_delta.sh
```

This prints all results and creates a dated review archive including qualification,
models, metrics, worker timing and provenance. It handles missing results explicitly.

For a fresh local control export, use `scripts/phase_c_larger_coupled.py export
--output <outside-campaign>/larger-coupled-inputs.json`; the portable campaign uses
the already frozen file. No original benchmark assets are read or modified.

Exit: account for all 24 tasks, compare recovery and compute cost separately by
dimension/timescale, and inspect weakly determined coefficients before choosing
the next nonlinear/collocation experiment. No production promotion is implied.

## Local implementation checks

All 14 new structural/orchestration tests pass; the extracted package passes its
141 preparation tests. Original-equation host qualification passes for all four
cases. The column-normalized sensitivity ratios are approximately 0.0873, 0.00609,
0.0127 and 0.0000395 in table order. Reference replay NMSE is below 6e-22; these
are attainable-reference checks, not estimates recovered from generic starts.

Four real smoke runs use seed 0 on the ordinary 3/6-state cases. All finish with
complete independent replay, timing and exact terminal resume. Recovery outcomes
are kept separate from successful execution:

| States / method | Training NMSE | Validation NMSE | Coefficients / initials recovered |
|---|---:|---:|---|
| 3 / joint | 8.21e-14 | 8.49e-14 | yes / yes |
| 3 / profiled | 7.27e-12 | 6.91e-12 | yes / yes |
| 6 / joint | 2.84e-15 | 2.88e-15 | yes / yes |
| 6 / profiled | 5.18e-7 | 3.68e-7 | no / no |

The last fit stops under the existing stagnation rule after 52 residual calls,
before certification/polishing, with worst coefficient relative error 323%.
It passes the looser 1e-6 evaluation prediction gate, illustrating why the metrics
must remain separate. At that retained point, finite differences agree with the
profiled residual Jacobian to relative column error below 1.7e-7; the physical
gradient components are below 7e-11. This is consistent with a stationary region
of the nonconvex outer objective, not proof of a global or local minimum.
No start, equation, budget or stopping policy was changed after inspecting this
outcome. The full Delta matrix remains prospective, with this seed included.
Local wall times during concurrent verification are not a speed comparison.

Final verification: full `pytest -q -n 4` passes 4,220 tests with eight optional
Torch skips and sixteen warnings. Changed Python files pass Ruff; repository-wide
`ruff check .` retains 37 pre-existing findings in unrelated untracked analysis
files. The extracted bundle passes host qualification, terminal smoke resume, and
the inspector/archive check. No cluster jobs were submitted locally.
