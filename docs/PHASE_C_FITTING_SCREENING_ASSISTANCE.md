# Phase C fitting M9 — bounded screening and assisted collocation

## Questions and scope

M8 reduced checkpoint overhead but did not recover the alien-device model from
poor starts. Screening a single difficult iterate sometimes consumed the remaining
fit budget. M9 tests two distinct questions:

1. Does bounded screening retain more useful iterates, and does Radau help relative
   to the existing RK45 screening integrator?
2. Can collocation represent and retain an already attainable solution when given
   a good **training-fitted** start? This separates local transcription/mesh trouble
   from the difficulty of finding the right optimization basin.

This is an opt-in fitting diagnostic. Production fitting defaults, construction,
benchmark arrays, public prompts and previous campaign results are unchanged.
Correct structures, shared initials and known parameter blocks remain the same as
M6–M8. No GPU, LLM call, test data, or hidden trajectory is needed.

## Frozen roster

`configs/phase_c_fitting_screening_assistance_v1.json` specifies two cases
(basin and alien hard), three starts and eight arms: **48 entries**.

- Dense collocation, reduced collocation and L-BFGS multiple shooting each run
  with bounded RK45 or bounded Radau screening: 36 generic-start fits.
- Dense and reduced collocation each run the assisted fixed-then-free procedure:
  12 entries. Five source starts qualify, giving 10 fits; the two alien seed-2
  entries are explicitly `assisted_source_ineligible`, not omitted.

The generic arms retain M8's compact checkpoint policy, equations, ordinary starts,
coordinates, full observation weights and forcing interpolation. Both screeners use
relative tolerance `1e-8`, absolute tolerance `1e-10`, and the same 900-second total
fit ceiling. Native solve allowance still depends on actual remaining time, so this
is a matched total-budget comparison, not identical native iteration counts.
Independent Radau/DOP853 train/validation replay has a separate 300-second ceiling.
Replay timeout is unavailable evidence, not a partial-data score.

## Bounded screening

Every full-training candidate evaluation runs in a killable child process.
The 20-second point ceiling includes imports/startup, integration and result writes;
ordinary pre-NLP screening also retains its existing 15-second phase ceiling.
The outer fitter's 900-second deadline covers all phases. Process termination has
at most a two-second cleanup grace. Harder candidates may consequently be left
unevaluated; this is recorded rather than interpreted as a scientific rejection.

`fit/screening.json` journals the identity and start of every attempt **before**
launch, then records completion, timeout or unavailability. Separate phase/point
directories prevent overwritten logs and reset call numbering. An interrupted
phase closes without retries or a new budget. Full campaign resume reuses terminal
results; interrupted consumed fits follow the existing no-budget-reset policy.
Only complete, finite, unclipped training residuals can replace `fit/best.json`.
A failed candidate cannot replace the verified incumbent with a penalty score.

## Assisted starts and provenance

`scripts/export_fitting_assisted_starts.py` verifies the sealed M7 plan and each
`rollout_continue` backend's exact training worker payload. It exports all six
starts and their source plan/backend/request hashes, observed time and residual
call count. Eligibility requires verified training-only selection and training
NMSE at most `1e-8`; validation and reference errors are not consulted.
The failed alien seed-2 endpoint remains recorded but supplies no assisted vector.

The provided source is the previously downloaded M7 review. Its original input
value hash is `97aacd4d137ad3631b11798795ad857d4febcae5780565d351277806119adfc6`.
The new sealed bundle adds assistance metadata and exact frozen ordinary-start
requests, coordinates and node guesses. This avoids cross-platform last-bit
differences from regenerating random starts. It does not change any benchmark
array. Source fitting cost is reported separately and is not charged as if free
work by M9. Counts/time are descriptive, not hardware-independent work units.

For each eligible assisted arm:

1. Recheck the source vector by a bounded complete Radau training rollout.
2. Generate node guesses by integrating that fitted model, at mesh boundaries and
   internal Radau nodes. All original forcing knots are retained. Intermediate
   target interpolation only supplies a trajectory container; it is never scored
   or used as additional observations. This phase has at most 120 seconds.
3. Fix **all** coefficients and shared initials at that vector. Optimize only
   trajectory node states under the existing dynamic equalities, with at most
   250 seconds. For alien this removes 18 optimization variables. Record initial
   and final nodal observation loss, dynamic defects and native termination.
4. Build a separate graph with those 18 quantities free again. Transfer boundary
   and internal node values from the fixed solve; no dual state is transferred.
   Both graph builds are charged. The release solve receives 75% of the remaining
   total budget, leaving time for bounded checkpoint screening.
5. Screen the released final vector first, then the remaining checkpoint pool.
   Select only using actual complete training rollouts, retaining the assisted
   input if no better verified point appears. Independently replay the selected
   endpoint and evaluate coefficient/initial accuracy afterward.

A missing fixed-stage final node record stops the diagnostic explicitly. Native
nonconvergence with a finite final record remains visible; it does not certify
feasibility. No true parameter values or hidden node labels initialize either NLP.
Numerical coordinates remain those of the original generic-start experiment.

## Interpreting the results

Do not combine assisted and generic-start success rates. In assisted rows,
excellent selected-endpoint NMSE alone says little: the supplied fit was already
good. Inspect `assisted.phases`, initial/final nodal defects, native convergence,
`released_final_training_nmse`, its screen status and coefficient diagnostics,
and whether `retained_assisted_input` is true. The final-vector diagnostics are
separate from the training-selected endpoint; a failed final screen has no NMSE.

A fixed-stage large nodal error or failed solve from accurate integrated nodes
suggests investigating mesh/transcription/conditioning. A good fixed stage but
bad released stage directs attention to joint optimization. Success in both
supports prioritizing poor-start search, scaling, continuation or block methods.
None of these observations alone proves the cause of every M7/M8 failure, global
identifiability, or readiness to replace the production fitter.

## Delta launch and inspection

Upload `phase-c-fitting-m9.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
It includes committed code, tests and sealed development inputs plus fitted-start
provenance. No cluster-to-cluster copy, new fitting environment, or M7 rerun is
required. Use a fresh code directory and campaign root:

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m9"
tar -xzf "$AF_BASE/phase-c-fitting-m9.tar.gz" -C "$AF_BASE/code/fitting-m9"
bash "$AF_BASE/code/fitting-m9/scripts/hpc/submit_phase_c_fitting_screening_delta.sh"
BASH
```

Default account is `bibo-delta-cpu`; one CPU and 16 GB per fit, concurrency two,
25-minute scheduler ceiling (15-minute fit + five-minute replay + overhead).
There are 46 actual fits and two cheap ineligible entries. The fit+replay ceilings
sum to about 15.3 CPU-hours; successful early exits can use less. Preparation is
separate. Increase `AF_CONCURRENCY` only if desired; no GPU allocation is needed.

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m9/scripts/hpc/inspect_phase_c_fitting_screening_delta.sh
```

This regenerates the report, shows scheduler state, and prints the exact review
archive to download. Missing result files remain visible until jobs finish.
The default output is `/work/hdd/bibo/yxiao2/phase_c/fitting-screening-assistance-v1`.
Uncertain `sbatch` replies retain submission intents and must be reconciled with
Slurm before resubmission; do not delete receipts or reuse old campaign roots.

For local regression testing only:

```bash
PYTHONPATH=src .venv/bin/python scripts/smoke_fitting_screening_diagnostic.py
```

This small control first fits its own source vector using training data, exercises
all eight arms, checks both assisted phases and independent replay, then confirms
exact completed-result resume without additional fitting.
