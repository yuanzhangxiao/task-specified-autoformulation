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

Upload the delivered bundle under
`/scratch/group/p.nairr260351.000/u.yx126462`. Check current file quotas before
choosing a source destination; both scratch locations have approached their file
limits. Use the verified installer below if extraction has failed. Results remain
in group scratch. In an **ACES shell**, run from the complete bundle directory:

```bash
bash run_fit.sh run
bash run_fit.sh status
bash run_fit.sh inspect
```

The bundle wrapper resolves its own code directory. The default results root is
`/scratch/group/p.nairr260351.000/u.yx126462/phase-c-paired-fit-v2`.
Preparation authenticates inputs and runtime without fitting on the login node.
The required scientific packages are NumPy, SciPy, pandas, Pydantic and CasADi.
The first launcher mistakenly also required SymPy, which these fitter kernels
do not use. A missing-SymPy failure occurs before preparation/submission. Use
the corrected source bundle without installing SymPy or changing the numerical
environment. Do not patch an existing sealed execution: keep it intact and choose
a fresh results root if preparation or submission had already succeeded there.
Because ACES previously reached its file quota, check `showquota` before
extracting; budget roughly 700 file/directory entries at the source destination
and 2,500 in group scratch for the bounded receipts. Leave additional room for
other work sharing the allocation. No GPU/compiler cache is created by this launch.
The launcher first submits a five-minute CPU preflight that verifies the saved
execution identity on a compute node, without fitting. The array depends on its
successful completion; a failed preflight prevents all 24 fits from starting.
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
export AF_PAIRED_FIT_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-paired-fit-v2
bash run_fit.sh run
bash run_fit.sh status
bash run_fit.sh inspect
```

Platform/Python/dependency overrides must be set before preparation. Do not reuse
an already sealed ACES campaign root with a different platform or interpreter.

### Recovery from the first ACES array

The returned `2246406` array failed before fitting. The saved identity included
EasyBuild's helper directory, which the Python module prepended to `PYTHONPATH`
during submission. Workers inherited the loaded module environment, reset
`PYTHONPATH`, and therefore differed from the sealed identity. This was a launcher
environment inconsistency, not numerical evidence about the 24 models.

The corrected launcher restores the explicit source/dependency path **after**
module loading and repeats the import check on that exact path. Actual Python,
package, source and entrypoint pins remain enforced. Launcher-identity errors now
list the differing fields and their saved/current values; scientific source and
package verification remain earlier gates with their own errors. Execution protocol 2 adds
the compute preflight described above and uses the separate `v2` output root.
Keep `phase-c-paired-fit-v1` and its job/error records intact; do not edit its
sealed identity to make verification pass. The models, inputs, fitter kernels,
two starts and numerical budgets remain unchanged.

### Recovery from incomplete extraction

The `0bf4747` extraction subsequently hit the file quota. Its missing
`autoformalism.targets` import is a consequence of the incomplete source copy.
Do not launch it, install dependencies to compensate, or edit a sealed execution.
The maintenance helpers below use only Python's standard library and do not
import or change the fitter. They can be delivered together as the single-file
`paired-fit-quota-recovery.pyz` to avoid extracting another source tree.

The helpers require Python 3.9 or newer. ACES's default `python3` was confirmed
to be Python 3.6, which cannot parse their annotations. Use the existing
`/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python -S`
for both maintenance commands. No scientific imports or new installation are
needed. The delivered entrypoint now checks the version before importing either
helper and is itself parseable by Python 3.6. The earlier zipapp also works with
the explicit Python 3.11 executable; this error does not require another upload.

The explicit `phase-c` profile of `archive_finished_component_caches.py` accepts
only a `phase-c-runtime-cache` root and explicitly listed numeric jobs. It requires
an empty current-user queue and an owned terminal allocation record from `sacct`.
It accepts only the known compiler-cache directories (`triton`, `torchinductor`,
`vllm`, `xdg`, `cuda`); LLM responses, model results and environments remain outside
its scope. Every selected cache is archived, read back and checksum-verified;
source contents and the queue are checked again before loose files are removed.
Default invocation produces a read-only plan; `--apply` performs the archival.
The original component profile retains its stricter successful-proposer rule.

The latest returned inventory has no active jobs and one remaining Phase C cache,
job `2206843`, occupying approximately 2,960 entries. Its terminal state is still
verified when the helper runs. Store its archive and receipt in a small directory
under home, which has separate quota headroom. Missing accounting or a new active
job stops the cleanup; never bypass those checks by blindly deleting caches.

`install_verified_paired_bundle.py` checks the supplied whole-archive SHA-256,
every `MANIFEST.sha256` entry, the exact file roster and required launch inputs
before extraction. It rejects links, unsafe paths and special members. It stages
the source privately and publishes a complete directory only after readback
verification. Existing incomplete destinations are preserved and refused; a
complete matching destination can be reused. A failed write removes only the
installer's own staging files. It does not launch anything.

The unchanged `phase-c-paired-fit-0bf4747.tar.gz` contains 565 regular files and
has SHA-256
`8eeb5026909d68a728deb3d779471f1aeea9a2008017c43f5c7996dc2ee939dc`.
After archiving the selected cache, install under group scratch, preserving the
partial user-scratch copy and all v1 results. Chain verification and launch with
`&&` so any failed installation prevents submission. Continue to use the v2
results root, the unchanged 24-model roster and the compute-node preflight.
All operations still require a few available file slots; this is bounded recovery,
not a durable increase to the allocation's shared file quota.
