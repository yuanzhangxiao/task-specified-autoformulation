# Matched parent–child fitting comparison

Implemented 22 October 2026 as the opt-in
`construction-parent-child-fit-1` protocol. The question is whether the returned
model revisions improve training reconstruction and validation forecasts when
parents and children receive the same fitting procedure and allowances.

## Population and numerical procedure

The frozen roster contains **eight unchanged parents and all 16 accepted changed
children**, all from the same Dalla Man T2-easy public task. These are 24 models,
not 24 benchmarks. The retained revision aliases its unchanged parent; the seven
rejected episodes remain in provenance and do not become fit tasks. No child is
preselected using its equations, an earlier score, or validation data.

The numerical routines are based on Orion's **`e7067f3`**. This campaign adds
orchestration and held-out evaluation without changing those fitter kernels:

1. Certify affine state/input dynamics and propagate every piecewise-linear
   forcing interval using the matrix exponential. Unsupported dynamics fail
   qualification rather than being silently linearized.
2. Apply only certified, domain-preserving reductions of isolated gain products.
   Lift every fitted endpoint back to the original equations and parameter
   domains. This is an exact coordinate change, not a model simplification.
3. Give every model two fresh starts: its declared values and a deterministic
   perturbation keyed by parameter name and seed 1729. Matching parameter names
   receive the same perturbation fraction. **No historical fitted warm starts**
   or historical optimization budgets enter this comparison.
4. Check derivatives with the precision-aware policy and verify original-equation
   predictions before searching. Each start permits **600 numerical-active
   seconds and 1,000 oracle calls**. Invalid nonfinite trial points may be
   rejected; they cannot become accepted checkpoints.
5. Independently verify returned endpoints against original-equation Radau
   rollouts using the fixed tolerance ladder. Qualification and diagnosis each
   have 120-second allowances; each verification has 180 seconds. Separate
   inclusive limits cover construction and checkpoint overhead.
6. Freeze the verified endpoint with the lowest **training** error across the
   two starts, including verified starting points. Diagnosis retains the selected
   endpoint's own optimization history. Validation cannot change this choice.

All models receive the same policy, but actual work can differ when qualification
fails or optimization stops early. A verified starting point can remain available
even if its search is unavailable. Reports distinguish this from a converged fit.
Numerical agreement and local optimization evidence do not certify a global
optimum, parameter identifiability, or scientific correctness.

## Initialization, data access, and scoring

Use all 16 public training and four public validation trajectories. Equations,
parameter domains, initializer graphs and root scopes are preserved. Training
jointly estimates shared dynamics/map parameters and permitted training roots
from complete training trajectories. Its NMSE is therefore **training
reconstruction**, not a causal forecast from an early window.

For validation, freeze all shared parameters and use the predeclared **eight-minute
observation window**. Only declared trajectory-specific roots can be estimated,
with one declared start, a 60-second/100-call search allowance, and separate
180-second original-equation verification. Analytic-map models have no free
validation roots and incur no root search. This root inference is numerically
checked but carries no optimization-convergence or identifiability claim.

Forecast from the original boundary without later observed-target resets, and
score only samples strictly after the common window. Forecast verification has
its own 180-second allowance and compares certified affine propagation with
original-equation Radau. It receives a target-free schedule. Publicly authorized
EGP/Uii auxiliary trajectories remain allowed forcing, so these are forecasts
conditional on those supplied observations. The window and information boundary
are fixed for this diagnostic, not optimized or substituted for historical scores.

For each target, normalize residuals by its pooled training standard deviation,
floored at `1e-12`; average squared normalized residuals across scored samples and
targets. Use these same frozen scales for validation. The report includes target
errors, verification and optimization statuses, per-start outcomes and costs.
A validation aggregate requires all four forecasts; missing or failed rows do
not silently disappear from its denominator.

For each eligible child report
`delta = child NMSE - its parent's NMSE`; negative values indicate improvement.
Report the median and **unscaled MAD** of available paired changes. Also take the
median child change within each parent, then report the median and MAD across
these parent summaries, so parents with more accepted children do not receive
extra weight. Availability counts accompany both summaries. Children share
parents, and all models share one task: these are descriptive comparisons, not
independent benchmark replications or proof that any improvement arose solely
from construction rather than optimization.

Known equation concerns from the saved scientific audit remain annotations; they
do not remove accepted children from fitting. This experiment performs no new
mechanism-compliance assessment, LLM call, test-data access, automatic model
selection, promotion, revision, or follow-up experiment.

## Run on ACES and inspect

Upload and extract the delivered bundle under
`/scratch/group/p.nairr260351.000/u.yx126462`. In an **ACES shell**, run from that
bundle directory:

```bash
bash run_fit.sh run
bash run_fit.sh status
bash run_fit.sh inspect
```

The bundle wrapper resolves its own code directory. The default results root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-paired-fit-v1`.
Preparation authenticates inputs and runtime without fitting on the login node.
Because ACES previously reached its file quota, check `showquota` before
extracting; allow roughly 3,500 free file/directory entries for this bundle and
its bounded receipts. No GPU/compiler cache is created by this launch.
The array contains **24 CPU tasks, at most eight concurrent**, each requesting
one CPU, 8 GB RAM and three hours. It requests no GPU. A separate report job runs
after the array, including failed tasks. Scheduler wall time is a ceiling, not a
promised duration or numerical fitting charge.

Inspect `report/SUMMARY.md` and `report/summary.json`; download
`inspection.tar.gz` from the results root. `status` and `inspect` reconstruct
reports from saved receipts and do not restart fitting. Source/runtime identities,
durable reservations and checkpoints support deterministic resume; an interrupted
operation does not silently receive another allowance. Keep the extracted code
and experiment settings unchanged while jobs are active.

If Slurm returns an ambiguous submission reply, reconcile its saved intent with
`squeue`/`sacct` before resubmitting: earlier socket timeouts have still created
jobs. A successful scheduler exit alone does not prove all forecasts are available.

Delta is an optional alternative for this same CPU experiment, using a separate
results root and the existing Delta scientific Python/dependency environment:

```bash
export AF_PLATFORM=delta
export AF_PAIRED_FIT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-paired-fit-v1
bash run_fit.sh run
bash run_fit.sh status
bash run_fit.sh inspect
```

Platform/Python/dependency overrides must be set before preparation. Do not reuse
an already sealed ACES campaign root with a different platform or interpreter.
