# Construction portfolio: coefficient reduction and initial-value profiling

1 November 2026. Opt-in numerical follow-up to
`PHASE_C_CONSTRUCTION_PORTFOLIO_SEARCH.md`. This campaign preserves the supplied
equations, public data and scientific requirements. It makes no automatic
construction revision or production-policy change.

## Motivation and frozen source

The reviewed five-model search under plan
`6cddde26ba8ae9b91b5e086f3bde345b4e9e625fadeba397bbc4bd11394b4a9a`
produced independently verified improvements in models 10/14, but their final
sensitivity solves timed out. Models 12/15 admitted slightly better
insulin-action timescales, but their 120-call Powell searches changed only 4–6
coordinates and never adjusted the 32 trajectory initial roots. Model 03
recovered one derivative direction while three remained unresolved.

The source is normally
`/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-search-v1`.
Preparation authenticates the saved plan, models, operation receipts and
terminal reconstruction before freezing a new input snapshot. Historical
receipts and budgets are preserved. Old operational completion remains distinct
from numerical convergence.

## Scope and interpretation

The seven CPU tasks comprise:

- Model 03: complete missing local probes for coordinates 10/22 using
  authenticated saved prerequisites and successful probes. Direction 8 remains
  unresolved near the precision floor, and the previous direction-19 pass is
  preserved. No model search is authorized by passing this subset.
- Models 10 and 14: certify remaining coefficient combinations, qualify the
  reduced numerical representation and continue from verified retained fits.
- Models 12 and 15: compare two starts per model—its incumbent and the
  independently verified slower insulin-action point from profile 09—with
  initial-root profiling and matched search allowances. These use the complete
  saved parameter vectors, including their initial roots. Both timescales remain
  free during the new search; the slower-action start is not a fixed-timescale
  constraint.

Every fitting arm has at most 1,800 numerical seconds and 1,000 outer evaluations
for search, plus separately reserved qualification, original-equation verification
and diagnosis. The largest task reservation is 6,060 seconds. The launcher
validates the campaign's total reservation against a four-hour worker envelope.
Each worker requests one CPU and 8 GB, with at most two concurrent
tasks. A separate report job has 30 minutes. No GPU, LLM call, validation or test
access is needed.

For nonlinear reductions, exclusive-use and domain/lift certificates are
required before merging coefficient sums or products. A smaller coordinate
vector must represent the same feasible equations; a convenient positive bound
or a nonzero divisor cannot silently remove a legitimate branch.
For models 10/14, the certified representation reduces the global coordinate
count from 29 to 24. Trajectory-local sensitivity propagation removes roots
owned by other trajectories and reduces that derivative system from 21 to 9
parameter directions. A separate reduction-equivalence operation precedes
qualification and search.

At fixed shared coefficients, an eligible affine model has a residual of the
form `r = A z0 + b` in its training initial roots. The inner bounded
least-squares problem estimates those roots while the outer search adjusts the
shared coefficients. Certification must cover initialization dependencies,
observation maps, residual weights and feasible root domains. Unsupported
nonlinear or coupled constraints must remain explicit blockers. Inner solve
completion does not establish outer optimality, identifiability or scientific
validity.
For models 12/15 this eliminates the 32 training-root coordinates from the outer
search, leaving 9/8 shared coordinates respectively. The earlier estimate of
8/7 referred to a fixed-timescale profile; the present searches allow both
timescales to vary.

New endpoints must pass original-equation replay before retention. Exact
numerical prerequisites may be shared only for identical model, data, parameter,
coordinate and numerical-policy identities. Retained-best and last-accepted
histories remain separate even when a numerical evaluation can be reused.
The root-only solution found during qualification is independently replayed and
can be retained even if the later outer search is interrupted or fails to
improve it. A worse or unavailable attempted endpoint cannot erase the source
incumbent.

Per-operation, per-view evaluation ledgers and inner-profile execution ledgers
make fitting, profiling and verification scopes visible. They can be nested and
must not be summed as disjoint physical calls. Legacy aggregate aliases are
explicitly non-exhaustive; historical accounting corrections are additive.

All scores are training reconstruction. This campaign does not reestimate
causal observation prefixes or measure new held-out predictions. Previously
reported prefix scores belong to their original endpoints. Better aggregate
NMSE can hide per-target tradeoffs, so Gp, I and U are reported separately.

## Local saved-endpoint checks

The bounded local checks authenticate the saved models and exercise equivalence
before remote work. At the slower-action seeds, conditional root fitting alone
produced independently verified training NMSE **0.5341373823842465** for model
12 and **0.5370171573682042** for model 15. These are initial-root-only training
improvements, not outer convergence or held-out forecasting results.

For 10/14, the algebraic reductions and trajectory-local coordinate counts were
checked. Short 20-second sensitivity attempts did not complete qualification;
the remote qualification remains a blocking prerequisite for continuation.
Increasing the qualification budget is not itself a pass. No equation-family
inadequacy or parameter-confidence claim follows from these checks.

## Delta commands

Extract the delivered source bundle into a new repository directory, then enter
that directory. Keep it unchanged while its jobs are active; Slurm workers bind
the code location and native interpreter to the frozen execution identity.

```bash
export AF_PORTFOLIO_PROFILE_SOURCE=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-search-v1
export AF_PORTFOLIO_PROFILE_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-profile-v1
bash scripts/hpc/start_construction_portfolio_profile_delta.sh run
```

`run` performs source authentication and preparation, then submits the array and
dependent report. It prints progress before potentially slow filesystem and
scheduler operations. These steps can be performed separately:

```bash
bash scripts/hpc/start_construction_portfolio_profile_delta.sh prepare
bash scripts/hpc/start_construction_portfolio_profile_delta.sh verify
bash scripts/hpc/start_construction_portfolio_profile_delta.sh submit
```

Status reads small sealed terminal receipts without rerunning fitting or report
construction. Inspection reconstructs results and writes a portable archive.

```bash
bash scripts/hpc/start_construction_portfolio_profile_delta.sh status
bash scripts/hpc/start_construction_portfolio_profile_delta.sh inspect
```

The archive is `$AF_PORTFOLIO_PROFILE_ROOT/inspection.tar.gz`. From a local
terminal, download it with:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-portfolio-profile-v1/inspection.tar.gz ~/Downloads/construction-portfolio-profile-inspection.tar.gz
```

## Recovery and retention

Repeated preparation/submission with identical inputs reuses confirmed records.
A repeated worker execution reuses completed numerical receipts and preserves
nonrenewable reservations. It does not restart an interrupted search with a new
budget. An ambiguous Slurm reply blocks automatic resubmission: first compare
the saved submission intent with `squeue`/`sacct` and reconcile the actual job.
Do not delete intents or change roots to bypass uncertainty.

If only reporting fails, rerun `report` or `inspect` using the same committed
source and root; these commands do not fit models. If an array task is canceled,
inspect its receipt status and scheduler history before arranging a worker retry.
Do not resubmit the full array solely because the terminal disconnected.

Preserve the source campaign, new `plan.json` and `inputs.json`, launcher and
submission identities, all operation/result receipts and logs, and the exact
source commit or bundle. An operational `complete` report can still contain
unqualified derivatives, time-limited optimization, or unresolved parameter
confidence; those outcomes must remain visible.
