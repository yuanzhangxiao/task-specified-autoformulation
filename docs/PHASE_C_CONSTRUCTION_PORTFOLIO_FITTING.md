# Fitting all 23 construction models

29 October 2026. This opt-in CPU campaign implements the receiver review in
[the intake note](../work/agent-comms/2026-10-29-scaled-readout-fitting-intake-review.md).
It is separate from the pending LLM-SR detention continuation and preserves all
23 model/domain identities, the public data, and the eight-unit prefix contract.
It makes no claim that the models have already been fitted successfully.

Returned v2 results (30 October): all 23 tasks are recorded; 15 affine models
entered fitting and eight nonlinear models were blocked during qualification.
The best reconstruction NMSEs are 0.564509 and 0.564608, with optimization
adequacy still unresolved. Prefix inference/forecasting still use a generic ODE
route and fail on several stiff fitted endpoints. See the
[returned-results audit and next milestone](../work/agent-comms/2026-10-30-construction-portfolio-fitting-results.md)
before interpreting missing forecasts or sending function-form revision feedback.

The separate [numerical recovery campaign](PHASE_C_CONSTRUCTION_PORTFOLIO_RECOVERY.md)
now preserves these fitted endpoints while correcting prefix propagation and
requalifying blocked nonlinear starts. Its new root leaves v2 evidence intact;
the full Delta recovery results are pending.

## Worker launch correction after the first returned inspection

The first returned archive contains the authenticated 23-model inputs and accepted
Slurm receipts (`run=23332572`, `report=23332573`), but **zero result records and
zero numerical operation reservations**. It is not evidence of failed model
fitting. The archive omitted worker logs, so the historical Slurm exit states
cannot be established from it alone.

A local Slurm-copy simulation reproduced a concrete launcher bug: `sbatch`
executes a copied script from its spool directory, and the original wrapper
unconditionally recomputed `AF_REPO_ROOT` relative to that copy. This overwrote
the correct exported checkout and failed before Python with
`Use the extracted source bundle, not an empty directory.` Both fitting and
report workers used the affected path.

The corrected worker requires and preserves the exported checkout. Login-side
commands still derive it from the actual launcher path, ignoring stale inherited
repository variables. Tests execute copies of both worker stages in a simulated
Slurm spool directory, cover missing exports and stale login environments, and
confirm normal stage routing. Inspection archives now include bounded stdout
and stderr tails (last 64 KiB per file, up to 128 files), with truncation/omission
metadata in `inspection_log_tails.json`. Log tails are diagnostic evidence, not
numerical operation receipts.

Use the corrected bundle in a new persistent code directory and the fresh
`construction-portfolio-fitting-v2` root. Preserve v1 and its receipts; do not
overwrite its sealed launcher or reset its journal. Equations, input data,
initial guesses, reduction policy and fitting budgets are unchanged. No remote
jobs were submitted or cancelled during this repair.

Original returned archive SHA256:
`965d388b2ded40989eaf8caa26bcecf093ab96fd5e1f7faa6301186dd1d32002`.

Repair verification: **242 focused tests passed**, including 35 launcher tests
and the bounded-log cases. The returned empty report reconstructed exactly.
Changed Python files and Bash syntax pass their checks; repository-wide Ruff
still has the same 37 unrelated analysis findings. The original full-suite
attempt and its limitation remain recorded below.

## Why another adapter is needed

Earlier experiments also had adapters. Their runners were specialized to the
model family being tested: the returned-eight construction campaign assumed an
eight-model roster; its later continuation used certified affine propagation;
the Model-6 profiler assumed distinct identity-state outputs; the detention
experiments used their own parameter charts. Those assumptions are not valid for
this complete new population.

The general model compiler already understands these equations and initializer
graphs. The new adapter joins that compiler to the improved search driver and
the existing certified reductions. It does not invent a new modeling language
or edit a candidate to fit a numerical method. Fifteen models qualify for exact
affine state/input propagation; eight require the general numerical ODE solver.
Parameter dependence can be nonlinear even in the affine group.

All 23 have composite output readouts, so the existing Model-6 output profiler
cannot be applied unchanged. Composite-readout profiling remains a separate
possible extension; this milestone uses direct trajectory fitting.

## Inputs and exact reductions

`construction_portfolio_intake.py` authenticates the complete delivery manifest,
roster, original problems and training split before creating compact campaign
inputs. It checks both existing digest conventions rather than confusing their
different JSON serialization rules. Missing, duplicate, modified or extra model
files, unauthorized data fields and unsafe paths fail before submission.

Every candidate starts from its supplied guesses. Shared dynamics and map
parameters are fitted jointly with each training trajectory's declared
independent latent roots. Dependent initials are evaluated through the graph.
There are 35–53 original joint coordinates and 27–45 after certified reductions.
The reductions combine eligible isolated gain products and remove exactly unused
coordinates; their certificates and lift back to original parameters are saved.
They do not assert identifiability of the original factors. Unsupported duplicate
terms and potential latent-state scale symmetries remain explicit limitations.

Candidate-01 retains the complete permitted `Uii(t)` forcing schedule. The other
22 use only the two external inputs as ongoing forcing. No auxiliary is silently
replaced by its initial value; no extra physiological bounds are introduced.

The executable problem is unchanged, including its restricted-expression guards.
Rational denominators and other domain issues are diagnosed as numerical outcomes,
not silently repaired by changing a function or parameter domain.

## Numerical sequence

Each model owns durable stages with independent, nonrenewable allowances:

1. Compile and record the route, declared start and reduction certificates.
2. Verify the starting model against an independent original-equation replay.
3. Qualify local derivatives of the actual reduced fitting objective. This also
   checks predictions after lifting the reduced parameters back to the original
   equations. Unqualified starts remain visible and do not enter gradient search.
4. Run one bounded lazy-Jacobian search from the original declared start, with
   fixed physical scaling plus the guarded-relative scaling policy. Improving
   finite residual-only checkpoints are retained even if subsequent derivatives
   fail. No held-out score or other candidate's fitted values select a start.
5. Independently verify the best saved search point. A failed, unverified or
   worse candidate cannot displace a verified starting endpoint. Saved useful
   progress from an interrupted search can be verified in its separately reserved
   verification stage; the interrupted search is not rerun.
6. Diagnose the retained point. Numerical agreement, local stationarity,
   remaining descent, parameter ambiguity and scientific validity are separate.
   Accepted-step history is used only if it belongs to that exact endpoint.
7. Freeze shared parameters, infer independent roots afresh from each training
   prefix, and independently replay each forecast. Score strictly after the
   cutoff. Full-training reconstruction is a different reported metric.

The generic nonlinear solver can give slightly different predictions in a
state-only solve and a solve augmented with sensitivity equations. The adapter
uses state-only residuals as the objective and pairs a Jacobian with that baseline
only after checking normalized residual agreement to `1e-7`. It still requires
local finite-difference derivative qualification. A bounded eight-point cache
includes the model, data, chart, parameter vector, method and solver settings;
hits, residual solves, sensitivity solves and consistency failures are recorded.
This consistency check is not a derivative-accuracy certificate by itself.

Validation is not in the delivered intake. This milestone reports training
reconstruction and prefix-conditioned training forecasts, with validation marked
pending authentication of the previously frozen matching split. It creates no
replacement split and opens no test data. Later validation must use the same
prefix-only root inference rule. These forecast diagnostics do not select shared
parameters or trigger proposal feedback. The separate joint fitting objective
does use the complete training trajectories, including their suffixes; these
training forecasts are therefore not independent generalization estimates.

## Budgets and failure reporting

The common search allowance is 900 seconds and at most 1,000 driver calls per
model. Starting replay, derivative qualification, final replay and endpoint
diagnosis each receive 180 numerical-active seconds, separately reserved.
Checkpoint overhead has bounded additional wall allowances. Each training row
has a 30-second prefix search, a 15-second verification, and a 60-second forecast
verification allowance. Inclusive reservations total 5,220 seconds per model.
Slurm requests two hours, one CPU and 8 GB, with two simultaneous model tasks by
default. The report receives 30 minutes. No GPU or LLM is used.

These limits are ceilings, not a promise of convergence or an estimate of queue
time. A completed task means its recorded stages terminated; inspect
`fitting_status`, qualifications, retained verification and assessment as well.
Every model remains in the report. Incomplete forecasts have no complete-model
aggregate score. Summaries give medians, unscaled MADs and availability counts.

Started operations without terminal receipts appear as running-or-interrupted,
with their allowance charged and unknown actual totals. Repeating a worker uses
completed receipts and proceeds only with unattempted stages. It does not reset
an interrupted stage. Actual operation wall/CPU costs include failed work and
checkpoints; runtime validation, reporting, queue time and adapter reconstruction
on partial resume are excluded explicitly. Deadlines cannot forcibly interrupt
every solver setup or blocked filesystem call.

## Delta commands

The supplied portable bundle includes source, the authenticated 23-model intake
under `inputs/scaled-readout-intake`, and checksums. Upload that bundle from the
Mac, using its supplied filename:

```bash
scp /absolute/local/path/to/phase-c-construction-portfolio-COMMIT.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

Then on Delta, substitute the same filename below. Extract into a persistent
project directory and retain this exact code directory for the campaign:

```bash
set -euo pipefail
AF_BUNDLE=/projects/bibo/yxiao2/phase-c-construction-portfolio-COMMIT.tar.gz
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-portfolio.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
sha256sum -c SHA256SUMS --quiet
export AF_CONSTRUCTION_PORTFOLIO_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2
bash scripts/hpc/start_construction_portfolio_fitting_delta.sh run
printf 'Keep this code directory: %s\n' "$AF_REPO"
```

Preparation authenticates the inputs; it does not run a broad numerical test suite
on the login node. Progress is printed before metadata work and submissions.
Confirmed Slurm IDs and the exact source/runtime identity are saved. Ambiguous
scheduler replies block blind resubmission. Do not create an empty new checkout
for inspection or change code under a running campaign.

From that same extracted directory, use:

```bash
export AF_CONSTRUCTION_PORTFOLIO_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2
bash scripts/hpc/start_construction_portfolio_fitting_delta.sh status
squeue -u "$USER" -o '%.22i %.42j %.12T %R'
bash scripts/hpc/start_construction_portfolio_fitting_delta.sh inspect
```

Inspection rebuilds the report from recorded evidence and writes
`$AF_CONSTRUCTION_PORTFOLIO_ROOT/inspection.tar.gz`. Download it from the Mac:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-fitting-v2/inspection.tar.gz \
  "$HOME/Downloads/construction-portfolio-fitting-v2-inspection.tar.gz"
```

The launcher deliberately ignores old detention output-root variables. An
explicit `AF_CONSTRUCTION_PORTFOLIO_INTAKE` can point to the same independently
extracted delivery; otherwise the bundled intake is used. Concurrency can be
set before first submission through `AF_CONSTRUCTION_PORTFOLIO_CONCURRENCY`;
changing it after submission changes the sealed execution identity.

## Implementation locations

- Intake authentication: `research/construction_portfolio_intake.py`.
- Mixed numerical adapter: `fitting/prefix_portfolio.py`; the generic training
  oracle adds its training-data metadata for existing reduction certificates.
- Campaign journals and reports: `research/construction_portfolio_fitting.py`.
- CLI: `scripts/run_construction_portfolio_fitting.py` and the Delta wrapper.
- Frozen policy: `configs/construction_portfolio_fitting_v1.json`.

No production fitter default, benchmark, prompt or scientific requirement changed.
Returned Delta evidence will determine whether numerical recovery is still needed
before any equation-revision feedback is justified.

## Local verification

The focused intake, adapter, campaign, launcher and existing numerical regression
suite passed **230 tests**. All 23 delivered packets also passed authentication,
campaign preparation and static adapter construction: 15 affine routes, eight
general ODE routes, and 27–45 reduced coordinates. These checks did not fit or
roll out the real candidates locally. Synthetic tests exercise the fitting,
independent verification, prefix inference, reporting and deterministic resume.

Changed Python files pass Ruff; the shell wrapper passes `bash -n`. Repository
`ruff check .` reports 37 pre-existing findings under unrelated `analysis/`
files. A repository-wide `pytest -q -n 4` attempt was stopped at the five-minute
verification limit, around 28% progress, without a displayed failure; it is not
a full-suite pass. The focused suite completed independently.
