# M17: coupled-block polishing and timeout diagnostics

## Evidence from M16

Source: the user's `review-20261007-051325.tar.gz`, commit
`b0b1b129ab4f619edb723ece5404072d8b5405f4`, plan
`de1ea2b3de2d8e80a6361f48a9f1d97fe9fe276377f018371bc7159dc5279e78`.
The two controls have the correct equations

```
y' = z
z' = -a*z - b*y + c*u(t)
```

The ordinary case uses `(a,b,c,z0)=(0.8,1.3,1.6,0.4)`; the fast/slow case uses
`(18,2,2,0.4)`. Both have the same three generic starts, three training schedules
and two validation schedules. These are noiseless development controls, not a
regenerated benchmark release. Joint rollout optimizes all four parameters.
Coupled profiling integrates an affine basis and solves bounded least squares
for `c,z0` at each outer `a,b` point. The full two-state feedback system is linear
in states, although its rollout is nonlinear in `a,b`.

| Case/method | Completed | Worst relative coefficient error among completed | Worst absolute initial error |
|---|---:|---:|---:|
| Ordinary, joint | 3/3 | 0.0148% | 0.0000994 |
| Ordinary, profiled | 3/3 | 0.00231% | 0.0000159 |
| Fast/slow, joint | 2/3 | 0.0119% | 0.0000456 |
| Fast/slow, profiled | 2/3 | 0.404% | 0.00158 |

All ten completed fits have validation NMSE below `8e-9`. Both methods pass the
prediction and 1% coefficient gates on **5/6 original starts**. Initial recovery
at absolute tolerance `0.001` is 5/6 for joint and 3/6 for profiling. Across the
five completed matched pairs, residual calls total 49 versus 34. Fit wall time
totals 187.39 versus 146.52 seconds, but this small run cannot establish a stable
speedup.

The profiled fast/slow fits stopped as soon as training NMSE crossed `1e-8`.
Their final coefficient and initial errors need not indicate optimizer bias:
joint fits happened to overshoot that stopping threshold substantially. In
`y''/a + y' + (b/a)y = (c/a)u`, the ratios are better determined than the separate
coefficients when `a` is large. Accurate predictions and separate coefficient
recovery remain distinct measurements.

Both failures were fast/slow seed 2, before a complete first candidate. The
profiled history records a point deadline after about 80 seconds despite a
30-second cooperative point cap. Both children eventually exited cleanly, with
roughly 172 seconds of parent-observed elapsed time. These starts use the same
model, input and initial parameter vector as successful ordinary-case seed 2;
only observed targets differ. A local first-point integration replay took under
one second. Runtime interference is plausible, but the existing logs cannot
assign the cause to imports, filesystem waits or CPU scheduling.

## Prospective experiment

Protocol: `phase-c-coupled-polishing-1`. Exact M16 input digest:
`425186a426cc3ea90dd7fd1246a71851c58bdc13887669d3aa7daf8bdf423e22`.
The portable bundle carries that frozen input file; preparation rejects changed
or regenerated inputs. No original result is overwritten.

1. **Repeat cohort, two tasks:** fast/slow seed 2, both methods, original stopping
   thresholds and limits. These are new explicit experiments, not automatic
   retry/resume or replacements for the original failures.
2. **Polishing cohort, twelve tasks:** both cases, all three original generic
   starts, both methods. After independent training certification at mean NMSE
   `1e-8` and worst-trajectory NMSE `1e-7`, preserve the first vector. If needed,
   warm-start once toward `1e-11` and `1e-10`, respectively.

Each task has **300 total fitting seconds and 300 residual calls**. Polishing
receives only the remaining time and calls. Worker imports, setup, checkpoint writes,
both optimizer phases and training certification are within the wall envelope;
the supervisor has a separate ten-second emergency cleanup grace. Each fit stage
reserves up to 60 seconds for an independent Radau/DOP853 training check. That
reserve is deducted from remaining time, not added to it. Existing method-specific
per-point limits remain unchanged. In particular, joint fitting's initial primal
screen uses a fraction of its allowance, while subsequent joint/profiled points
have 30-second cooperative caps. M17 does not equate these distinct safety caps.

A polished vector replaces the first only if independent original-equation
training rollouts remain certified and mean training NMSE improves. A failed
polish preserves the certified first vector. Unknown call counts or unconfirmed
child cleanup prevent unsafe continuation. Interrupted work remains terminal on
resume; restarting the same command never silently refreshes its budget.

Reference coefficients and validation are withheld from fitting. The first and
final vectors are scored only after final selection is sealed, with a separate
120-second evaluator allowance per distinct vector. Reports retain every planned
start, original M16 failure counts, separate repeat outcomes, coefficient and
initial errors before/after polishing, optimizer stages and timing records.

## Timing instrumentation

Enabled only by M17's fit wrapper. Ordinary M15/M16 paths remain uninstrumented.
A standalone worker entrypoint starts clocks before importing the package.
Records separate imports, symbolic model construction, profile construction,
trajectory integration, residual points, bounded linear solves, independent
training checking, JSON reads and durable checkpoint writes. Every category
records inclusive wall time and process CPU time; only the twenty longest spans
are retained. No parameters, trajectories, credentials or environment dumps enter
the timing report.

Nested spans overlap: never sum residual-point time with trajectory or write
time. A wall/CPU gap does not by itself distinguish descheduling from I/O waits.
The parent also records total subprocess elapsed time, allowing import/entry
gaps to be inspected. A killed process may have no final timing file; reports
keep this missing rather than claiming zero work. Timing metadata itself is
written once after work, under the existing parent deadline; atomic checkpoint
publication and `fsync` behavior are unchanged.

## Delta commands

Upload `transfers/phase-c-fitting-m17.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`, then run:

```bash
AF_CODE=/work/hdd/bibo/yxiao2/phase_c/code/fitting-m17
mkdir -p "$AF_CODE"
tar -xzf /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m17.tar.gz -C "$AF_CODE"
bash "$AF_CODE/scripts/hpc/submit_phase_c_coupled_polishing_delta.sh"
```

The launcher verifies bundle checksums and the exact input digest, freezes the
plan, runs preparation tests, then submits 14 CPU tasks with six-way concurrency,
one CPU and 16 GiB per task, followed by a report job. The scheduler's 35-minute
reservation accommodates startup, checking and reporting; fitting still has the
five-minute algorithmic ceiling. No GPU, LLM calls or test data are involved.
The default output root is
`/work/hdd/bibo/yxiao2/phase_c/fitting-coupled-polishing-v1`.

After completion:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m17/scripts/hpc/inspect_phase_c_coupled_polishing_delta.sh
```

That command reports status, metrics and timing, and makes a dated review archive
including sealed source/provenance, worker logs and before/after evaluations.

## Exit criteria and next scope

First determine whether the separate repeats recover and whether their timings
identify a bottleneck. Then compare first/final prediction, coefficients and
initial values under the matched polishing budget, counting failures explicitly.
A larger coupled block is the next milestone, conditional on these findings.
There is no production promotion, nonlinear coupled-block qualification,
structural discovery experiment or claim of universal identifiability here.

## Local implementation qualification

The four-task small-control smoke completed both failed-start repeats and both
fast/slow seed-0 polishing arms. Exact terminal resume launched no new work.
Joint seed 0 already reached the strict target in its first phase. Profiled seed
0 required two additional calls after its original nine: independently rescored
training NMSE fell from `8.14e-9` to `2.39e-13`, validation NMSE was `2.33e-13`,
and both coefficient and latent-initial recovery gates passed. Total fitting time
was approximately 28 seconds locally. Both repeated seed-2 starts also passed all
three evaluation gates locally. These are implementation checks, not substitutes
for the prospective Delta roster or evidence of reliable cluster timing.

The extracted portable bundle passed its 127 preparation tests. The same files
must remain pinned throughout the campaign; source changes invalidate resume.

Final verification: full `pytest -q -n 4` passed 4,201 tests, with eight
Torch-dependent skips and eleven warnings. All changed Python files pass Ruff.
Repository-wide `ruff check .` retains 37 pre-existing findings in unrelated
untracked analysis files; those files were left untouched. The four-task real
smoke and all 127 portable preparation tests passed.
