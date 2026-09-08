# Corrected input integration and fitter experiment

The runtime-v1 experiment exposed a missed input pulse. At its intermediate
parameter vector, tight and reference Radau never sampled the nonzero input
on train_001. The proposed memory equation therefore stayed at zero. Both
tolerances agreed because they made the same mistake. Preferring that reference
then rejected the other integrators and skipped all six planned fits.

The previous reference rule was insufficient: agreement after tightening one
solver's tolerance does not establish that the solver saw the supplied input.
The correction changes numerical integration, not the candidate's equations,
parameter domains, input interpolation, data, or initial conditions.

## Production correction

Adaptive free rollouts now stop and restart at the union of changes in slope
of all declared supplied forcing channels. Each segment starts from the final
state of the previous segment. There are no target resets or measured-target
dependencies in this procedure. Exactly equal adjacent slopes are combined;
approximate equality is not used to discard small changes. Constant inputs
require only one integration segment. Irregular sampling and single-sample
pulses are supported by the same piecewise-linear interpretation.

`FitConfig.maximum_integration_step` optionally supplies a strictly positive,
finite step cap. Its default is unset; breakpoint segmentation is always used
for adaptive free rollouts. A cap also applies to adaptive one-step intervals.
The fixed RK4 backend retains its existing sampling-interval integration.

Adaptive free observations reuse their immutable forcing interpolator instead
of rebuilding and validating it at every output time. The one-step causal
target-forcing path preserves its per-index interpolator. Timed and untimed
rollouts use the same numerical path. Segment counts and state ranges are
included in the profiling artifacts.

Analytic synthetic tests check the convolution of a linear memory equation
with delayed pulses, several integration tolerances and methods, irregular
sampling, small auxiliary changes, continuity across segment boundaries, and
independence from a parameter confined to another state equation. These
checks use proposed synthetic equations, not private benchmark equations.

## Frozen runtime-v2 campaign

The new protocol is `fitter-runtime-2`, configuration
`configs/fitter_runtime_v2.json`, output `fitter-runtime-v2`. Runtime-v1 remains
available at commit 65c9219; its output must not be overwritten or reused by
the corrected code. The submitter still reads the original
`fitter-stagnation-v1` source snapshot. It verifies the historical plan exactly
as stored and loads its original integration settings with the new optional
step-cap default unset.

All original model/public-asset checks and the three reviewed anchor hashes
remain enforced. The four completed stagnation-arm results and selected
parameter vectors are also checked against their task identities and frozen
as additional replay inputs. There are no LLM calls, test data, private
reference equations, or model selection.

### Three profile tasks

Each anchor uses the same first two training trajectories and full-training
normalization as runtime-v1. There are 18 rollout cases per anchor:

- RK45, DOP853 and Radau at current 1e-7/1e-9 and tight 1e-9/1e-11
  tolerances, each with two distinct physical repeats: 12 cases.
- DOP853 and Radau at reference 1e-10/1e-12 tolerances: two cases.
- Each reference repeated with maximum step equal to the minimum supplied
  sample spacing and half that spacing: four cases.

Both reference methods must agree pointwise, each must agree with its tight
run, and each must agree with both step-limited controls. Every comparison
requires normalized residual RMS <=1e-6 and maximum absolute difference <=1e-5.
There is no fallback to one unverified solver. Failure or disagreement is
reported as `reference_unverified` and prevents the matched fitting stage.

Candidate method/tolerance accuracy remains RMS <=1e-5 and maximum <=1e-4
against the verified reference, for both repeats. A fitting method must pass
its current tolerance at all three anchors.

The three derivative policies remain relative 1e-4, scaled 1e-4, and scaled
1e-5, with scale floor one. At every unique base and perturbation vector,
tight Radau is independently checked by tight DOP853. Pointwise residual
disagreement beyond the reference thresholds invalidates the derivative check
and blocks fits. Each policy also reports cross-solver Jacobian disagreement;
that diagnostic is not an additional fitted threshold or a solver selector.
Agreement between perturbation sizes alone does not certify sensitivity.

Compute timing excludes numerical-array serialization and checkpoint I/O,
which are reported separately. Detailed simulation timings remain nested.
A separate intrusive Python profile uses the first profiled trajectory with
forcing breakpoints, falling back to the first trajectory if none has them.
Its time is excluded from solver comparisons.

### Six matched fits and four previous-vector replays

The six fits retain the all-one start, seven parameters, fixed zero initials,
full training split, 900-second fitting budget, 100 reported-nfev maximum,
current integration tolerances, and the RK45/DOP853/Radau by relative/scaled
derivative factorial. Each resulting vector receives a tight DOP853 and Radau
replay on all training and validation trajectories. Each replay has 120 seconds.
The reported score is DOP853's; `complete` requires both finite scores to agree
within 1e-5 separately on train and validation. Optimizer timeout or
non-convergence remains explicit even when replay succeeds.

Four additional tasks replay the selected vectors from `current_default`,
`current_large_step`, `tight_default`, and `tight_large_step` under the corrected
integrator. They perform no optimization, and report old scores beside corrected
scores plus solver agreement. These diagnostic replays run even if a fitting
guard fails, but remain `replay_unverified` unless their own checks pass.
Final replay agreement compares aggregate scores; the profile reference checks
compare pointwise residuals. Neither is a mathematical global error guarantee.

## Resources, checkpointing and commands

Thirteen tasks use one CPU and 8 GB each, with default concurrency two and no
GPUs. Profiles have 600 seconds of numerical budget, 60 seconds per residual,
and a 660-second worker cap within a 15-minute allocation. Fit workers have
a 1,200-second cap including replays and margin within a 25-minute allocation.
Previous-vector replay workers have a 300-second cap. The 13 worker caps total
173 CPU-minutes, excluding preparation and summary. Invalid budget overrides
that exceed the frozen allocations are rejected before submission.

Profiles run as array 0-2. A dependent array 3-12 contains the six fits followed
by four previous-vector replays, and a final summary job follows that array.
Completed case arrays, derivative points, fits and replays are checkpointed
under immutable identities. Terminal failures remain visible. An interrupted
unfinished optimizer restarts deterministically from the same all-one start;
its internal trust-region state is not serialized and elapsed time can change
how far a restart gets. Use a new frozen output for intentional reruns.

The user runs the commands on Delta after Codex tests and pushes the commit:

```bash
(
set -e
cd /projects/bibo/yxiao2/repos/autoformalism-v21
git fetch origin codex/fitter-forcing-breakpoints
runtime_commit=$(git rev-parse FETCH_HEAD)
if [ ! -d ../autoformalism-fitter-runtime-v2 ]; then
  git worktree add --detach ../autoformalism-fitter-runtime-v2 "$runtime_commit"
fi
cd ../autoformalism-fitter-runtime-v2
test "$(git rev-parse HEAD)" = "$runtime_commit"
bash scripts/hpc/submit_fitter_runtime_delta.sh
)
```

Use the exact pushed commit from the handoff as runtime_commit for a pinned
release. If the directory already exists, the block verifies its commit and
stops on a mismatch instead of submitting from an unknown checkout.
The source remains `/work/hdd/bibo/yxiao2/phase_b/fitter-stagnation-v1` and the
Python remains `/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python`.

After the summary job finishes:

```bash
cat /work/hdd/bibo/yxiao2/phase_b/fitter-runtime-v2/summary.md
```

Re-running the submitter prints an existing submission manifest without
duplicating its jobs. A partially failed submission must be reconciled with
the queue before adding missing stages. Codex does not submit or monitor these
jobs; the user returns results for discussion before further implementation.
