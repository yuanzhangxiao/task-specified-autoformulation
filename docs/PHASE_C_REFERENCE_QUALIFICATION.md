# Phase C: corrected release and reference fitting qualification

Owner: Orion. This implements the benchmark portion of milestone 1 in
[PHASE_C_START_HERE.md](PHASE_C_START_HERE.md). Historical campaign reconciliation
has a separate owner and is not changed here.

The four ACES fits and their diagnostics have now been inspected. The coupled
reference is attainable on development data, but only easy/reference-near met
the low-error target. See [the bounded fitting follow-up](PHASE_C_CSTR_REFINEMENT.md)
for the observed failure modes and the separately frozen numerical comparisons.

## What this milestone establishes

The supplied ACES audit reported 260/260 numerical protocols and 40/40 public
projections passing under `continuous-rates-1`. The new release command consumes
those exact, checksummed reference checkpoints. It does not regenerate missing
records or silently mix them with historical data. It publishes all 40 existing
library cells under `phase_b_continuous_inputs_v1/*_rates_v1`, with an explicit
opt-in registry. T3/T4 remain library assets; this does not add them to the paper
or the next discovery pilot.

Publication includes the previously audited evaluator-only test trajectories,
exported behind the existing logical test-access seal. No candidate is evaluated
on test. This is not filesystem access control: do not give an LLM the release
root or evaluator metadata. Discovery consumes its cell's public prompt and
the loader's authorized channels. Historical directories are never overwritten.

An audit passing means the reference integration, input encoding and public
projection are consistent. It does **not** establish that a model with the
public observations and allowed initial information can reproduce the reference.
The next diagnostic tests that distinction before more discovery runs.

## First bounded roster

Four CPU fits: named CSTR easy/hard, each with `reference_near` and `generic`
starts. Equations are supplied by the evaluator. There are no proposer, critic,
pruning or test-evaluation calls. The public fitting profile is unchanged:
`collocation-single-target-v2` (120-second initializer, 180-second refinement,
240 residual evaluations and its existing bounded recovery settings).

The reference balances are transcribed using

```
shape = exp(-activation * (350 / max(T, 250) - 1)) * max(C, 0)
C'  = flow * (Cf - C) - rate * shape
T'  = flow * (Tf - T) + heat_rate * shape - exchange * (T - Tj)
Tj' = jacket_flow * (Tjf - Tj) + jacket_exchange * (T - Tj)
```

The reference parameterization is `activation=(E/R)/350`,
`rate=k0*exp(-(E/R)/350)`, and `heat_rate=source_gain*rate`.
This avoids directly optimizing the enormous Arrhenius prefactor. All dynamic
coefficients have nonnegative domains; these are fitted, not fixed to truth.
Separate concentration and heat gains still permit the reference balance.

Both tiers model all three coupled states. Easy initializes concentration and
jacket temperature from their first public auxiliary values, then integrates
them freely; it does not substitute their future observed values into the ODE.
Hard has T alone observed. Hidden initial values use
shared train-fitted affine maps `a + b*(T(0)-350)`. T is initialized from its
first public observation. No trajectory identifier, private state time series,
private per-trajectory initial value or derivative enters the fitting objective.

The affine initial-map form is evaluator assistance. It covers the existing
training offset design and equilibrium validation initials. It is not a promise
for independently varied hidden initial states, including different test initial
conditions. Reference-map parameters are supplied only for diagnostic replay
and the explicitly assisted start; both map parameters remain train-fitted.

`reference_near` starts dynamic parameters at 0.8 times reference values and the
initial-map parameters at their reference values. `generic` uses a frozen vector
unrelated to fit scores. Report both, including failures; do not select a winning
start and describe it as unassisted recovery. Numerical parameter uniqueness is
not claimed, even when prediction error is small.

## Reduced auxiliary-driven diagnostic control

The pilot also retains a reference-parameter replay of the temperature ODE alone,
with interpolated C and Tj supplied as forcing. It receives no optimization calls.
Local full-development checks found enormous rollout error in this representation,
while the coupled equations reproduce both tiers at about 2e-17 NMSE. The easy
prompt explicitly permits dynamically modeling supplied auxiliaries, so the
coupled witness respects the public interface.

Substituting a concentration trajectory removes the feedback whereby increasing
temperature consumes more reactant and limits subsequent heating. In the reduced
equation, temperature still accelerates the reaction but cannot deplete its
externally imposed concentration. Small interpolation or numerical discrepancies
can therefore grow rapidly. This is a conditional representation/conditioning
issue, not evidence that the original three-state reference is incorrect or that
an optimizer failed. The control keeps this failure visible in the final report.

## Checks and interpretation

1. Publish/resume each cell atomically, comparing every file against a fresh
   projection of the audited cache. Reject missing, changed or failed records.
2. Freeze four requests with the existing public fitter, including actual public
   arrays, source identity, profile and initializer. Test CSVs are never read by
   preparation, fitting or reporting.
3. Replay reference parameters with Radau and DOP853 using the production public
   rollout and one training-derived temperature standard deviation. Require both
   development split NMSEs <= 0.001 and maximum solver disagreement <= 0.0001 of
   that scale. If the witness fails, record it and do not spend fitting budget.
4. Execute each fixed numerical attempt, then independently replay retained
   fitted parameters with both solvers. Record every trajectory's error and
   solver failures. The diagnostic low-error target is NMSE <= 0.01 on both
   development splits with the same solver-agreement tolerance. This is a
   predeclared diagnostic threshold, not a method-ranking or selection gate.
5. Keep fitting status, budget exhaustion and native optimizer convergence
   separate from accuracy. `status=complete` means the diagnostic has terminal
   records, not that all fits succeeded. An interrupted public fit never receives
   a fresh attempt budget on resume. Read-only replay can be repeated if its
   checkpoint was not completed.

If reference replay succeeds but a fit is inaccurate, the supplied equations
have an admissible low-error witness on these development data. The failure
therefore cannot be attributed to missing proposer mechanisms. It still does
not identify whether local optimization, conditioning, initialization or a time
limit is responsible; the saved fitter diagnostics distinguish those cases.

Dalla Man and alien-device exact-structure fitting remain explicitly **pending**.
Dalla Man still has private gastric-normalization bookkeeping at meal onset;
an exact transcription under the public ODE interface needs review. The alien
device's differing hidden initial states need a public-information attainability
check. Neither issue is repaired by exposing private trajectories as inputs.
Whole-suite scientific qualification is not claimed by this CSTR pilot.

A public-data inspection already establishes one alien-hard limit. In
`phase_b_alien_device_unknown_device_mechanism_canonical_functional_hard_rates_v1`,
training trajectories `train_000`, `train_014`, and `train_015` have identical
complete input/time tables, no supplied auxiliaries, and initial target zero.
Their later target values differ by up to 3.08837. Under deterministic equations
and one initializer using permitted public information, all three predictions
must coincide. Even an unrestricted common prediction curve has a training NMSE
floor of approximately 0.005616 from this group alone, using the training-wide
target variance and sample weighting. This is an information limit, not an
optimizer failure; it does not exclude useful approximate prediction. Decide
whether to expose legitimate initial measurements, change the initial-condition
design in a new version, or explicitly evaluate irreducible uncertainty before
requiring near-zero reference recovery. Do not use trajectory IDs as hidden
state labels. The inspected training CSV SHA-256 is
`64db6d82a5626c54b9e16f7fd1fda52d7857eb917d74798e8cecf218a2bf70c6`.

## Run on ACES

Upload the supplied source bundle to your group directory and extract into one
new source folder. The bundle includes the two local private evaluator specs,
which are not committed to Git. Keep it private. Set `AF_REPO_ROOT` to that
folder, then use the existing CPU launcher:

```bash
export AF_REFERENCE_MODE=qualify
export AF_AUDIT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/continuous-inputs-be31404/audit
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-reference-v1
bash "$AF_REPO_ROOT/scripts/hpc/submit_continuous_input_audit.sh" aces
```

One CPU, 8 GB, one hour; no GPU or API key. The launcher retains the scheduler
command, reply and job ID. An ambiguous submission reply stops; inspect its
receipt and accounting before attempting any recovery. Do not delete the receipt
to force resubmission. Source/data identity changes require a distinct experiment
root; a cleanly resumed fit retains its consumed budget.
Standalone execution also checks the prepared source and Python/package versions.
The worker writes an incomplete summary after preparation and refreshes it after
every fit, so an interruption does not hide the completed tasks.

```bash
AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/phase-c-reference-v1
AF_JOB=$(cat "$AF_OUTPUT_ROOT/submission/job.id")
sacct -j "$AF_JOB" --format=JobID,JobName%28,State,ExitCode,Elapsed
cat "$AF_OUTPUT_ROOT/qualification/summary.json"
```

For Delta, transfer the complete audited cache (including `plan.json`,
`summary.json` and `private/`) and the source bundle; set `AF_AUDIT_ROOT` to that
copy, `AF_OUTPUT_ROOT` beneath `/work/hdd/bibo/yxiao2/phase_b`, and pass `delta`
to the same launcher. There is no benefit to duplicating these four fits on two
clusters. Jetstream2 can use `jetstream2` with an existing Python environment;
it runs directly without creating a VM.

Individual stages are ordinary CLIs:

```bash
PYTHONPATH=src python scripts/release_phase_b_public_suite.py \
  --audit-root "$AF_AUDIT_ROOT" --public-data-root "$AF_OUTPUT_ROOT/public"
PYTHONPATH=src python scripts/qualify_reference_fitting.py prepare \
  --public-data-root "$AF_OUTPUT_ROOT/public" --root "$AF_OUTPUT_ROOT/qualification"
PYTHONPATH=src python scripts/qualify_reference_fitting.py run \
  --root "$AF_OUTPUT_ROOT/qualification" --index 0
PYTHONPATH=src python scripts/qualify_reference_fitting.py report \
  --root "$AF_OUTPUT_ROOT/qualification"
```

Indices 0–3 are easy/reference_near, easy/generic, hard/reference_near,
hard/generic. Replace `python` with the site environment's explicit interpreter.
Heavy fitting belongs on compute nodes. Preserve all four diagnostic directories,
release metadata, source, audit and submission receipts for interpretation.

## Local verification

- Full regression run: 3,639 passed, eight skipped because Torch is unavailable.
  After final execution-runtime guarding, 74 focused tests passed.
- Full 40-cell publication from the 260-protocol audit and identical resume passed.
- Coupled CSTR reference replay, all 16 training and four validation trajectories,
  both tiers: approximately 2e-17 training and validation NMSE; maximum two-solver
  discrepancy below 7.1e-7 training standard deviations.
- The existing analytical public-fitting smoke passed with unchanged resume.
- Changed Python files pass Ruff; repository-wide Ruff still reports 37 existing
  issues in unrelated `analysis/claude` scripts. Shell syntax checks pass.

The four production-size attempts were run on ACES, not on the laptop. Their
diagnostics motivated the linked follow-up. The remaining families' attainable
reference contracts remain open research work.
