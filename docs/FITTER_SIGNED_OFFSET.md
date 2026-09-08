# Signed-offset control and independent replay

This development experiment changes only the already-reviewed candidate's `c`
parameter role from `nonnegative_coefficient` to `offset`. The resulting domain
changes from `[0, infinity)` to the real line. Equations, other parameter roles,
sharing, fixed initial states, inputs and data remain identical. The original
candidate is copied without rewriting its bytes. This is a deliberate paired
model intervention, not a silent reinterpretation of an old checkpoint.

The source is the completed `fitter-runtime-v2` campaign at `cde5da1`. Its code,
plan, original candidate and public-snapshot ancestry are pinned in
`configs/fitter_signed_offset_v1.json`. Source paths contain only the original
candidate, source metadata and public train/validation data; the test table is
never loaded. The original benchmark prompt is preserved byte-for-byte.

## Questions and task matrix

1. Does the signed-offset model improve beyond its nonnegative-offset control
   under identical fitting conditions?
2. Does that conclusion hold from both prespecified starts?
3. Can BDF independently verify the two previously fitted, very stiff Radau
   parameter vectors without the DOP853 replay timeouts?

| Index | Task | Optimization |
| --- | --- | --- |
| 0 | Numerical guard on two common starts and an analytic baseline vector | None |
| 1 | Original domains, all-one start | Radau, 900 seconds |
| 2 | Signed offset, identical all-one start | Radau, 900 seconds |
| 3 | Original domains, fixed perturbed start | Radau, 900 seconds |
| 4 | Signed offset, identical perturbed start | Radau, 900 seconds |
| 5 | Verify saved runtime-v2 Radau-relative vector | None |
| 6 | Verify saved runtime-v2 Radau-scaled vector | None |

The second start samples every named parameter log-uniformly between 0.5 and 2,
using seed 20260909 in sorted parameter order. Its exact values are frozen before
jobs run and are identical between paired variants. Neither start is selected
using a fit score. The training mean is used only for a separate, nonoptimized
analytic baseline control, not as a privileged start in either fitting arm.

All fits use the corrected piecewise input integration, Radau with
`rtol=1e-7, atol=1e-9`, scaled finite differences with factor `1e-4` and scale
floor 1, at most 150 optimizer function evaluations, and one start per task.
The actual residual-call count, optimizer status, parameter movement and stopping
reason are retained separately from numerical verification and fit quality.

## Numerical verification and reporting

The guard checks two frozen training trajectories. At each common start and the
signed-baseline vector it compares Radau at fit tolerance, Radau at
`1e-9/1e-11`, BDF at `1e-9/1e-11`, and refined Radau at `1e-10/1e-12`.
It additionally checks that the signed-baseline vector gives the constant
training mean. Fitting is blocked unless all guard cases pass.

Every final or saved vector is replayed on **all** training and validation
trajectories with Radau-tight, BDF-tight and Radau-refined. Each of these six
solver/split combinations has its own 120-second budget. Failure or slowness on
training cannot consume validation's budget. Completed individual trajectories
are cached, including prediction arrays, parameter-bound provenance through the
task identity, solver configuration, numerical-array hashes and elapsed time.

Verification compares normalized residuals point by point, rather than relying
on aggregate scores to agree: RMS difference must be at most `1e-6` and maximum
absolute difference at most `1e-5`. Scores are computed from unclipped residuals
using training normalization for both splits. A failed split receives no NMSE;
its trajectory IDs and failure messages remain visible. No successful partial
trajectory average is presented as a whole-split score.

The report includes zero and training-mean baselines; each solver's train and
validation score; prediction mean, RMS and standard deviation; all parameter
vectors; and every numerical check. An optimizer stop does not establish useful
mechanism recovery. A completed verification does not establish a good fit.
All four fitting arms remain in the report; no model is selected and no new
candidate is proposed from these results automatically.

## Resources, checkpointing and commands

Entry point: `bash scripts/hpc/submit_fitter_offset_delta.sh`.

Defaults:

- Source: `/work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v2`.
- Output: `/work/hdd/bibo/yxiao2/phase_b/fitter-signed-offset-v1`.
- Python: `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`.
- One CPU and 8 GB per task, zero GPUs, at most two concurrent array tasks.
- Guard: 15-minute Slurm allocation, at most 660 worker seconds.
- Remaining tasks: 30-minute allocation; fit workers capped at 1,680 seconds,
  replay-only workers at 780 seconds. Total worker caps are 149 CPU minutes,
  excluding preparation and summary.

The user submits the jobs. The launcher records each returned job ID immediately
and uses dependencies to run the guard before fits and the summary afterward.
An existing submission manifest prevents duplicate submission. An interrupted
submission with no manifest leaves an intent marker requiring queue reconciliation.
The launcher does not guess whether an uncertain submission succeeded.

Preparation and task execution are locked. Repeating a task command reuses its
completed result. If a worker was killed before its fit checkpoint, its residual
and iteration logs survive; an explicitly resumed unfinished fit starts again
from the same prespecified vector in a new attempt directory. Internal optimizer
state is not serialized. Terminal failures remain terminal, rather than silently
receiving a larger budget. Code/data/configuration changes require a new freeze
and output directory.

After completion:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-signed-offset-v1/summary.md
```

To regenerate a missing summary from existing checkpoints, in the pinned checkout:

```bash
PYTHONPATH=src /projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python \
  scripts/run_fitter_offset.py summarize \
  --output /work/hdd/bibo/yxiao2/phase_b/fitter-signed-offset-v1
```

Limitations: the model intervention isolates one domain mistake and does not
guarantee scientifically adequate dynamics. Two starts do not establish a global
optimum. BDF verification may itself fail or time out, in which case the result
stays unverified with the explicit reason. The protocol deliberately does not
change production solver defaults, prune mechanisms, loosen time-constant domains,
or rerun proposal generation on Delta.
