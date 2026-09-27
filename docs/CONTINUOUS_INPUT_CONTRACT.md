# Continuous input contract and finite-duration ingestion

`continuous-rates-1` is a new, opt-in benchmark version. Its generator metadata
is `reference-rates-1`. Historical inputs, finalized prompts and fitted results
are not reinterpreted. New cell IDs append `_rates_v1` and public release files
live under `phase_b_continuous_inputs_v1/`.

## Input meaning

For an amount M scheduled at time s, the ingestion rate is triangular over the
fixed interval [s, s + D], with D = 10 minutes in the registered Dalla protocols:

```
u(t) = (2 M / D) max(0, 1 - abs(2 (t - s) / D - 1))
integral u(t) dt = M
```

The named public channel is `meal_rate_g_per_min`; an obfuscated version keeps
`u01` and describes it as a finite-duration input rate. Rate samples must not be
divided by the sampling interval or interpreted as instantaneous amounts.
The duration and triangular shape are fixed experimental settings, not fitted
physiology or a universal assertion about eating behavior.

Rates of overlapping or simultaneous meals add. A time-zero meal has its full
area and leaves the initial physical state unchanged. A meal too close to the
final horizon to finish is rejected; it is never truncated or silently moved.
Onset, midpoint and completion are mandatory input knots, even between the
ordinary observation times. The generated target and auxiliary samples include
these times. The opt-in loader accepts those explicit, possibly irregular times.
Numerically equivalent observation/event times are merged in favor of the
declared event; distinct input knots below floating-point resolution are rejected.

All external inputs in this version are defined by linear interpolation of the
exported table. CSTR/alien references and the Dalla infusion channels now use
that same table interpolant, including its ramps near nominal step boundaries.
Analytic sine waves and ideal rectangular callbacks are not silently substituted
for that table during reference generation. Refining the observation grid holds
the input schedule fixed through the independently recorded `input_dt`.

The existing adaptive production rollout already integrates separately at every
change of input slope. The new contract uses this path directly; there is no
meal-specific hidden-state reset in a discovered model. Candidate equations
still decide how the input influences their own states.

## Trusted reference changes and limits

The private Dalla reference adds `1000 * meal_rate_g_per_min` to the first
stomach derivative, converting grams/minute to mg/minute, and removes meal
jumps in that mode. All 12 physical states remain continuous. Its existing
gastric-normalization convention at meal onset (remaining stomach content plus
the newly scheduled meal amount) is retained. This is a declared extension of
the old private reference, not a new independent physiological validation of
the gastric law. In particular, finite overlapping ingestion uses summed rates
with that same meal-onset normalization convention.

Targets, permitted auxiliaries and derivative labels are regenerated together.
The rate, onset and duration are public experimental information; the private
stomach equations, normalization bookkeeping and fitted coefficients are not
provided to a proposer. The runtime is responsible for numerical execution;
the proposer remains responsible for the discovered mechanisms.

Default low-level historical APIs still use `legacy-events-1`. The new release
command defaults to `continuous-rates-1`; the audit requires an explicit flag.
Existing campaign configurations remain on their frozen old IDs. New dataset
consumers can use:

```python
from autoformalism.data import BenchmarkLoader
from autoformalism.data.registry import BenchmarkRegistry, continuous_phase_b_specs

loader = BenchmarkLoader(BenchmarkRegistry(continuous_phase_b_specs()))
```

The registry checks the input-contract marker and channel names. A legacy
trajectory cannot be exported as a rate trajectory merely by changing a label.
The ordinary train/validation/test access rules continue to apply.

## Audit and tests

From a checkout with the existing private reference specifications:

```bash
export PYTHONPATH="$PWD/src:$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_continuous_inputs.py tests/test_continuous_input_submission.py
.venv/bin/python scripts/audit_reference_integrity.py \
  --output-root artifacts/continuous-input-audit-v1 \
  --input-contract continuous-rates-1 --include-test-protocols
```

Repeating the audit command verifies and reuses sealed checkpoints. Changed
source, settings or reference specifications require a new output directory.
The audit checks independent solvers, observation-grid refinement with fixed
inputs, exact linear-compartment convolution, integrated meal mass, glucose/gut
balances and all public projections. Production-rollout tests independently
reproduce the reference stomach response from the exported rate alone, including
time-zero and overlapping off-grid meals. No parameter fitting or LLM calls occur.

`--include-test-protocols` checks newly generated private reference schedules;
it never reads existing test datasets or scores candidates. Omit it to check
development schedules only. Public audit exports contain train/validation only.
Private audit NPZs and the bundled private system specifications are evaluator
assets, not proposer inputs.

Local verification on 2026-09-26 passed all 260 numerical protocols and 40 public
projections, then resumed the same audit successfully. The largest scaled
independent-solver difference was 1.66e-7; the largest output-grid-refinement
difference was 1.78e-13. The 31 new numerical/submission tests and 15 historical
hidden-contract tests passed. Historical protocol checksums retain their original
payload and explicitly reject the new input contract. Changed Python files pass
Ruff; repository-wide Ruff also reports 37 existing issues in `analysis/claude/`.

The full regression command exercised 3,542 bulk tests (8 Torch-dependent skips)
and 66 serial tests. Its first pass was not clean: three historical protocol
checksum tests exposed the compatibility issue above, and eight other checks
across the bulk/serial runs detected source changes during execution. All those
failures passed on subsequent focused/serial runs, including the complete
15-test hidden-contract module after the fix. The whole suite was not repeated
as a single invocation after that fix; the pinned cluster bundle avoids testing
against an actively edited checkout.

## CPU jobs on ACES or Delta

The supplied `continuous-inputs-COMMIT.tar.gz` contains a pinned source snapshot,
`SOURCE_COMMIT` and the two small private system specifications required for this
audit. It contains no historical trajectory datasets, model checkpoints, API
keys or model weights. Replace COMMIT by the package identifier supplied with it.

ACES, after uploading the archive into your group scratch directory:

```bash
AF_BASE=/scratch/group/p.nairr260351.000/u.yx126462
AF_PACKAGE=continuous-inputs-COMMIT
mkdir -p "$AF_BASE/repos"
tar -xzf "$AF_BASE/$AF_PACKAGE.tar.gz" -C "$AF_BASE/repos"
bash "$AF_BASE/repos/$AF_PACKAGE/scripts/hpc/submit_continuous_input_audit.sh" aces
```

Delta, after uploading into `/work/hdd/bibo/yxiao2/phase_b`:

```bash
AF_BASE=/work/hdd/bibo/yxiao2/phase_b
AF_PACKAGE=continuous-inputs-COMMIT
mkdir -p "$AF_BASE/repos"
tar -xzf "$AF_BASE/$AF_PACKAGE.tar.gz" -C "$AF_BASE/repos"
bash "$AF_BASE/repos/$AF_PACKAGE/scripts/hpc/submit_continuous_input_audit.sh" delta
```

Use one site. Both launchers request one CPU, 8 GB, and a one-hour ceiling,
using the existing project Python environments. They run the focused numerical
tests, the full private audit, and an identical resume. Override `AF_PYTHON`,
`AF_ACCOUNT` or `AF_OUTPUT_ROOT` if needed. Submission prints the exact output
directory and confirmed job ID. It saves the command, stdout, stderr and return
code and refuses blind resubmission after an ambiguous Slurm reply.

Inspect the printed directory:

```bash
AF_ROOT=/path/printed/by/launcher
cat "$AF_ROOT/submission/job.id"
sacct -j "$(cat "$AF_ROOT/submission/job.id")" \
  --format=JobID,JobName,State,ExitCode,Elapsed
jq '{input_contract,numerical_passed,numerical_protocols,public_cells_passed,
     public_cells,input_contract_verified,failed_protocols,remaining_contract_issue}' \
  "$AF_ROOT/audit/summary.json"
```

On a Jetstream2 VM with the project Python environment installed, unpack the same
archive and run the launcher with `jetstream2`. It executes locally on the VM,
without Slurm or an LLM API; set `AF_PYTHON` and `AF_OUTPUT_ROOT` as appropriate.

## Regeneration is separate from rerunning experiments

The audit does not overwrite or promote datasets, reopen saved model selection,
or refit any model. A separate release can be generated with
`scripts/release_phase_b_public_suite.py --public-data-root NEW_ROOT
--input-contract continuous-rates-1`; this also creates newly generated sealed
test files. Keep that release distinct from the audit and all historical roots.
Scientific release gates and fresh matched fitting/selection/evaluation for
every compared method remain necessary. Existing event-based fitted parameters
and historical scores are not results on the new rate-based benchmark.
