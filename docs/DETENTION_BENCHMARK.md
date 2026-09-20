# Stormwater detention: development benchmark qualification

This is the first step toward a seventh benchmark family, in civil engineering
and hydrology. It creates a new isolated development release and qualifies its
reference skeleton using the existing `collocation-single-target-v2` fitter.
Existing benchmark data, prompts, registration, proposer campaigns and fitting
algorithms are unchanged. Shared-process review and model/reasoning calibration
are the next milestone, after this case's physical and numerical checks succeed.

The eight CPU tasks are coupled/independent basins × noiseless/1% observation
noise × two fixed broad parameter starts. No LLM or GPU is used. This is an
attainability audit with a known skeleton and known rating-curve shape, not a
discovery result or proof of unique latent/parameter identification. There is
no automatic promotion to the benchmark suite and no automatic next campaign.

## Physical model and source basis

Two lined, vertical-sided detention cells collect measured storm runoff. Their
constant surface areas are 800 and 1,200 m². Upstream overflow enters downstream
through a short conveyance with negligible storage and travel time; the latter
also receives local runoff and releases to a channel. The upstream floor is
5 m above the downstream floor. Bank depths are 4 m; all reference trajectories
are checked below those banks, so the upper overflow remains unsubmerged.
Receiving-channel stage is below the downstream crest. Evaporation, seepage and
additional spill routes are neglected over the four-hour event. This is a
documented synthetic engineering idealization, not field-validated infrastructure.

With depth states h_up and h_down, input I in m³/min, and time in minutes:

```
area_up * h_up'     = inflow_up   - q_transfer
area_down * h_down' = inflow_down + q_transfer - q_outlet
```

The shared internal flow cancels in total stored water. The unequal areas mean
depth derivatives need not have equal magnitudes. In the independent negative
control the upper discharge goes to its own receiving channel; it never enters
the downstream equation. The correct target-generating skeleton then needs only
the downstream state and outlet coefficient. An unidentifiable irrelevant upper
coefficient is not added to its fitting problem. The negative control is not an
additional independent benchmark family.

The modeling basis is USACE's level-pool reservoir/storage continuity and
head–discharge relationships:

- [Reservoir modeling concepts and equations](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/reservoir-modeling/reservoir-modeling-concepts-and-equations)
- [Storage/discharge and outflow-curve options](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/reservoir-modeling/available-reservoir-routing-and-storage-methods)
- [Spillway discharge equations](https://www.hec.usace.army.mil/confluence/hmsdocs/hmstrm/reservoir-modeling/reservoir-modeling-concepts-and-equations/spillways)

V1's **exact generating law is a piecewise-linear rating table**, interpolated
at excess heads [0, .05, .1, .2, .35, .5, .75, 1, 1.5, 2, 3, 4] m, whose knot
values are H^(3/2). Discharge is K times this curve and is zero below the crest.
Crests are .35 and .12 m; reference K values are 72 and 54 m^(3/2)/min. These
correspond, for example, to the usual broad-crested coefficient convention with
1.2 m^(1/2)/s and effective crest lengths 1 and .75 m. These are illustrative
engineering parameters, not calibrated field measurements.

The direct square-root representation of the exact H^(3/2) law at zero fails
the existing sensitivity-domain certification. We do not modify the frozen
fitter or hide an epsilon smoothing. Tabulated rating curves are an established
engineering option. The generator uses table lookup; the diagnostic candidate
uses its algebraically equivalent hinge expansion through the restricted parser.
The release records the rating-table approximation error against H^(3/2),
including approximately 4.55% maximum relative discharge difference for H≥5 cm
on a 1 mm comparison grid. Below 5 cm relative error is not bounded by that
figure. Table approximation is part of the declared generator, not solver error.
This scope does not demonstrate support for every continuous piecewise formula.

## Public and private information

`public/{case}/noise{0,1}/` contains sealed JSON envelopes with `value` and
`sha256`: the public specification and `train.json` / `val.json`. These use the
existing `PublicSplit` and channel conventions, but are not yet registered as a
Phase-B release. A future proposer exporter should copy only this directory.

The only target is `h_down`. Inputs are measured upstream and downstream runoff,
linearly interpolated at 4 min intervals over 240 min. Rainfall-to-runoff
inference is deliberately outside the task. Surveyed areas/crests and one
calibrated upstream initial gauge reading are public fixed covariates. The
target's initial reading is calibrated too; noise is added only after time zero.
No initial condition is optimized separately on validation and no observed
trajectory resets the open rollout. The latent initial map simply uses the
public `initial_up` scalar.

Six training events include isolated and paired upstream storms, local-only
forcing, coincident storms and zero-input recession. Three validation events use
new storm timings and overlaps. Noise is independent Gaussian noise with SD 1%
of clean training output SD; the same observations are reused across starts.
There is no clipping of noisy observations. No test trajectories are generated
or opened. A future held-out release needs separately frozen seeds and forcing
regimes after model/protocol selection.

`diagnostic/` contains evaluator-only hidden states, integrated flows, reference
parameters and exact-skeleton requests. It must never be supplied to the
proposer, critic or scientific feedback. All equations retain physical units;
the audit fits only k_transfer/k_outlet from guesses (30,30) or (120,100).
Independent controls fit only k_outlet. Known rating shapes are oracle assistance
and must be reported as such; success does not establish discovery capability.

The public prompt requires physical storage, shared-transfer consistency and
threshold-dependent release. It does not prescribe a process name, coefficient,
rating table, or exact equation. Equivalent inline expressions deserve identical
mechanism credit. Current compiled-equation balance probes are regression
witnesses over physical depth coordinates, not a universal scientific verifier
for arbitrary latent transformations. The later proposer comparison needs its
own equation-derived mechanism assessment before reporting compliance scores.

## Gates, reporting and resume

Preparation checks:

- Independent DOP853/Radau reference solves with input-knot restarts, tight
  tolerances and 1 min maximum steps; depth disagreement below 1e-7 m.
- Integrated water conservation, nonnegative storage and operating-range bounds.
- Compiled named/inlined RHS values and physical balance probes, including
  below-crest, threshold and active-flow values.
- Full production BDF/Radau replay at reference parameters: clean NMSE <1e-8
  and prediction differences <1e-5 of the training output SD.

Each fitting task uses the unchanged 120 s collocation + 180 s refinement
profile, 240-call ceiling and its existing piecewise policy. Imports, setup and
post-fit diagnostics are outside those numerical budgets. Native fitting status,
observed train/validation NMSE, post-fit clean NMSE, independent solver agreement,
peak depth/timing errors and warning-depth classification disagreement are
recorded separately. Peak times and warning classifications use the 4 min grid;
they are not continuously located event times. Clean labels never enter fitting.
Strict recovery is both clean NMSEs ≤1e-4 plus replay agreement; the separately
reported practical band is ≤0.01. Neither certifies scientific correctness.

Preparation, requests and observations are sealed with source and data hashes.
Completed fits are reused unchanged. Interrupted fits retain the original
started marker and receive no fresh optimization budget. Post-fit replay is a
separately bounded 240 s attempt with its own marker. Missing, failed, unsupported
and interrupted tasks remain in denominators. Reports do not start new work.
The scheduler records submission intent before sbatch and refuses uncertain
resubmission, avoiding duplicate arrays. Every change needs a new experiment
root. Group scratch and disabled bytecode caching reduce inode pressure.

## ACES commands

Use the full pushed commit accompanying this milestone. Existing data staging,
model downloads and GPU reservations are unnecessary. Run the following inside
your ACES session; the subshell preserves the login session on an error.

```bash
(
  set -euo pipefail
  : "${AF_COMMIT:?set the full milestone commit first}"
  AF_BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  AF_GROUP=/scratch/group/p.nairr260351.000/u.yx126462
  export AF_REPO_ROOT="$AF_GROUP/repos/detention-${AF_COMMIT:0:7}"
  export AF_OUTPUT_ROOT="$AF_GROUP/detention-development-v1"
  export AF_PYTHON="$AF_BASE/.venv/bin/python"
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  module load GCCcore/13.2.0 Python/3.11.5
  export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
  "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_detention_benchmark.py" \
    --output "$AF_OUTPUT_ROOT"
)
```

This submits preparation/verification, eight fits (four concurrent, one CPU and
16 GB per task), and a summary. No automatic follow-up follows. If scheduler
submission is uncertain, inspect the saved receipts and queue before retrying.

After all jobs finish:

```bash
ROOT=/scratch/group/p.nairr260351.000/u.yx126462/detention-development-v1
cat "$ROOT/submission_manifest.json"
cat "$ROOT/SUMMARY.md"
jq '{status_counts, generation_gate_passed, rows}' "$ROOT/summary.json"
jq '.value | {audits, rating_table_max_relative_weir_difference_above_5cm}' \
  "$ROOT/plan.json"
```

Share `SUMMARY.md` and `summary.json`; use `plan.json` for gate diagnostics.
Success permits a bounded proposer/shared-process pilot, not opening test data.

## Local verification

```bash
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/pytest -q tests/test_detention_benchmark.py tests/test_detention_submission.py
.venv/bin/python scripts/smoke_detention_benchmark.py --output /tmp/detention-smoke-v1
```

Use a fresh output root after source changes. Full tests, repository-wide Ruff,
shell syntax, focused tests and the real fitting smoke are required before push.

The initial local qualification passed all 14 focused tests and the unchanged
resume smoke. Coupled/noiseless/start 0 recovered the output with clean
validation NMSE 1.70e-9 and BDF/Radau agreement; mean validation peak-depth error
was 2.33e-5 m. Collocation converged, while the subsequent refinement exhausted
its existing budget. These are one-arm development results, not the completed
eight-arm comparison or evidence of overall optimizer convergence.

Repository verification on 2026-09-19: 2,436 tests passed, six skipped, and 11
D3 tests failed because a concurrently changed function signature disagreed
with the test fixtures already loaded by pytest. Rerunning the current D3
modules gave 23 passed and two optional-Torch skips. Those unrelated edits are
excluded from this milestone. Changed-file Ruff and shell syntax passed;
repository-wide Ruff still reports 37 findings in unrelated `analysis/claude`
scripts. The full-suite run is therefore recorded as failed, not silently
reported as clean.
