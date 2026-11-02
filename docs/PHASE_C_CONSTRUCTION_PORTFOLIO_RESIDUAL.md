# Construction portfolio: residual controls and profiled continuation

2 November 2026. Opt-in CPU follow-up to
`PHASE_C_CONSTRUCTION_PORTFOLIO_PROFILE.md`. This campaign uses unchanged
equation families and training data. It makes no construction revision, LLM
call, held-out evaluation or production-policy change.

## Purpose and source

The previous campaign verified substantial training improvements for models
12/15 from their slower-action starts, ending at NMSE .449981937/.439421177.
Those searches stopped at the 1,000-evaluation cap, before using their full
1,800-second numerical allowance. Models 10/14 passed reduction and original
rollout checks but could not finish their augmented sensitivity prerequisites.
Their searches did not start.

This separate campaign authenticates the completed source at
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-profile-v1` and freezes a
compact projection with exact source identities. Historical plans, results,
receipts and budgets remain unchanged. The new root is
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1`.
The prepared input projection is approximately **2.918 MB**, compared with
138.865 MB for the preceding nested input archive. All four production record
digests, the shared training payload and numerical settings are explicitly
pinned; the smaller copy does not relax source authentication.

| Task | Numerical work |
| --- | --- |
| candidate-10-residual | Derivative-free search from the verified retained endpoint |
| candidate-12-profiled | Warm continuation from the successful profiled endpoint |
| candidate-14-residual | Derivative-free search from the verified retained endpoint |
| candidate-15-profiled | Warm continuation from the successful profiled endpoint |

Each search has at most **2,000 evaluations and 1,800 numerical seconds**. Other
stages have separately recorded reservations. The launcher uses the campaign's
computed maximum reservation (**4,620 seconds** under the default policy) and
rejects a plan exceeding its four-hour worker
envelope. Each worker requests one CPU and 8 GB; the array permits at most two
concurrent tasks. The dependent report has a 30-minute limit. GPUs are unused.

## Numerical contract

For 10/14, the residual-only control uses the same certified coefficient
reductions and original parameter domains. Before search it must establish
finite, repeatable residual evaluation, valid lifts, and original-equation
agreement. The unavailable Jacobian is not required by this derivative-free
optimizer. This does not qualify gradients, certify stationarity or establish
parameter confidence; those conclusions retain their separate requirements.

For 12/15, bounded least squares solves the 32 trajectory initial roots at each
shared-parameter trial. All shared parameters, including both timescales, remain
free in the outer search. Starts are the complete saved parameter vectors from
the successful `action_tau_10` arms, not the earlier fixed-timescale diagnostic
points. The new search is a **warm continuation with a new allowance**, not an
exact Powell-state resume: the old internal Powell directions were not saved.

The campaign explicitly chooses the original shared-parameter token
`k_Uii_term_12_0_244d72574a` for the exact-zero boundary control. Its spelling
does not cause the generic fitter to infer scientific meaning. The control
requires zero to be feasible in both coordinate representations, holds the
other shared coefficients fixed and reestimates initial roots. It tests whether
replacing a tiny coefficient with its feasible boundary changes the objective
beyond numerical precision.
The boundary comparison and its original-equation verification remain visible
separately. It may replace an incumbent only after independent verification and
a meaningful training improvement beyond the declared numerical tie tolerance.
A tied or worse boundary result preserves the incumbent. Zero feasibility does
not imply that a task-required mechanism may be removed or that a scientifically
interpreted effect has been established.

Rejected trials preserve bounded failure counts and representative numerical
diagnostics. An inner least-squares or domain failure is not silently converted
to a good fit. Independent original-equation replay gates endpoint retention;
an unsuccessful search cannot erase the source incumbent. Meaningful
improvement is distinguished from a numerical tie, and evaluation-limit
termination is distinguished from convergence.

Costs retain explicit operation/view scopes. Nested counters must not be added
as if every scope represents disjoint physical evaluations. Checkpoints remain
durable and numerical reservations do not renew on repeated worker execution.
Model 03's derivative error-floor investigation is outside this four-task run;
its unresolved qualification is preserved.

All NMSEs here are training reconstruction, with roots estimated from training
observations. No new causal-prefix forecasts, validation or test scores are
computed. A new fitted endpoint must receive its own causal-prefix evaluation
before making a held-out prediction claim.

## Submit on Delta

Extract the committed code bundle into a new repository directory and enter
that directory before the initial submission. Keep that directory unchanged
while jobs run. The native interpreter, scientific package versions, Python
source tree and launcher entrypoints are bound to the execution identity.

```bash
export AF_PORTFOLIO_RESIDUAL_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-profile-v1
export AF_PORTFOLIO_RESIDUAL_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1
bash scripts/hpc/start_construction_portfolio_residual_delta.sh run
```

`run` authenticates the source, prepares the new plan and submits the four-task
array plus dependent report. It flushes progress before filesystem and Slurm
operations. Preparation and submission can also be run separately with
`prepare`, `verify` and `submit`.

## Status and download from any working directory

After submission, recover the original code directory from the saved execution
record rather than creating an empty directory or relying on the current
terminal location. Run this entire block on Delta:

```bash
bash <<'BASH'
set -euo pipefail
export AF_PORTFOLIO_RESIDUAL_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1
AF_REPO=$(jq -er '.value.repo | select(type == "string" and length > 0)' \
  "$AF_PORTFOLIO_RESIDUAL_ROOT/launcher_execution.json")
AF_WORKER="$AF_REPO/scripts/hpc/start_construction_portfolio_residual_delta.sh"
test -f "$AF_WORKER" || { echo "Recorded worker is missing: $AF_WORKER" >&2; exit 1; }
bash "$AF_WORKER" status
bash "$AF_WORKER" inspect
BASH
```

`status` reads small sealed terminal receipts and does not run fitting or an
expensive report. `inspect` reconstructs numerical evidence and produces
`$AF_PORTFOLIO_RESIDUAL_ROOT/inspection.tar.gz`; it does not fit models. During
execution, run only the `status` line if an archive is not yet needed.

From the **local** terminal, download the archive:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-residual-v1/inspection.tar.gz ~/Downloads/construction-portfolio-residual-inspection.tar.gz
```

## Recovery and retention

Repeated preparation/submission with the same identities reuses confirmed
scheduler receipts. An ambiguous Slurm reply blocks automatic resubmission:
inspect the saved intent and reconcile it with `squeue`/`sacct` before taking
another submission action. Do not delete submission records or choose a new
root to bypass an uncertain job. A disconnected terminal alone is not grounds
to submit the array again.

If only reporting fails, use the same recovered worker path with `report` or
`inspect`. If a worker is interrupted, preserve its reservation and completed
stage receipts; a worker retry can reuse recorded work but cannot obtain a fresh
search budget. A deliberately new fitting allowance requires a separately
identified campaign.

Retain the source campaign, frozen inputs/plan, execution and submission
identities, all operation/result receipts and logs, and exact source code.
Operational `complete` means outcomes are recorded. It may still contain an
unqualified residual, unsuccessful inner solve, time-limited search or unresolved
scientific interpretation.
