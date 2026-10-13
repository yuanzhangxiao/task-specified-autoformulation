# M30: output dependencies and stationary poor-fit qualification

Implemented October 12, 2026, following the [M29 review](PHASE_C_FITTING_M29_RESULTS_2026-10-12.md).
This milestone is opt-in. Production fitting/construction defaults, historical
results, public prompts, benchmark observations and model equations are unchanged.
No experiments are launched by the agent; use the Delta commands below.

The [completed M30 results review](PHASE_C_FITTING_M30_RESULTS_2026-10-12.md)
verifies all 22 entries: the independent-basin optimizer defect is resolved and
known controls retain accurate recovery. General stationary stopping has not
demonstrated a whole-suite speedup.

## Changes and mathematical scope

The recovery policy gains an optional `poor_fit` configuration. `null` retains
M29 search behavior. The new path makes three changes.

### 1. Certify output dependencies before optimization

Expand shared processes using the existing restricted expression grammar. Start
from every observation expression, follow all state RHS dependencies backwards
to closure, and include parameters in those RHSs, observations and relevant
causal initial maps. Initial observation names have their existing data namespace;
they do not introduce dependencies on later measured values.

A parameter outside this closure cannot affect the fitted outputs through the
declared equations. Its value stays at the supplied incumbent value, or original
start value when no separate incumbent exists, throughout all restarts. The
parameter, state and process declarations remain in the returned model. The
assessment explicitly lists `unestimated_parameters` and the dependency
certificate. This is not scientific pruning, a new prior or removal of a small
coefficient. Algebraic cancellations conservatively remain active; small sampled
Jacobian columns never authorize removal.

All original equations are still integrated and verified. Disconnected states
can still cause a numerical failure; output independence does not certify their
feasibility. Their equations are not silently removed. Exact latent scaling
symmetries, such as the T2 insulin-action symmetry, are also not eliminated by
this syntactic dependency check.

### 2. Return fully profiled output solutions directly

The existing symbolic affine/terminal-output kernels select only output-active
inner coefficients. If all active unknowns belong to the bounded linear inner
problem, solve

\[
\min_{\ell\le g\le u}\|A g+b\|_2^2
\]

once, preserving the original bounds. There is no outer trust-region search when
no active outer coordinates remain. A model whose outputs are independent of all
declared parameters similarly needs only evaluation. This does not yet add an
interface for candidates declaring no parameters at all.

Before ending the restart portfolio, require independent DOP853/Radau verification
of the original equations and agreement with the profiled loss to
`max(1e-10, 1e-7 * abs(profile_loss))`. Failed verification retains its costs and
does not certify the solution. Rank-deficient inner designs remain unavailable
and use the existing budgeted fallback rather than gaining a false uniqueness
claim. The inner solver's certificate is numerical, within its stated tolerances.

An independently verified but inaccurate all-inner/fixed-output fit reports
`search_status=output_problem_solved`, with `recommended_action=review_equations`.
This means the particular output-fitting problem has been solved, not that the
equations are scientifically correct. It is stronger than a local plateau.

### 3. Leave stationary poor starts without weakening accurate-fit checks

Let \(r(q)\) be the normalized training residual vector in fixed dimensionless
parameter coordinates, \(J=\partial r/\partial q\), and \(N\) its length. Compute
the projected MSE gradient \(2J^Tr/N\), setting an outward-pointing component to
zero at an active parameter bound. Also compute each non-active column's absolute
residual correlation

\[
\rho_j=\frac{|J_j^Tr|}{\|J_j\|_2\|r\|_2}.
\]

This second check protects against declaring stationarity merely because a column
is small in magnitude. These diagnostics use the unamplified residuals, independent
of the optimizer's objective multiplier.

For loss greater than both `1e-6` and `100 * target_nmse`, the default rule requires
projected gradient and maximum correlation at most `1e-8`, loss variation at most
`1e-10 * max(1, loss)`, and coordinate movement at most `1e-6` over five evaluated
points. It stops the current search block, not the whole portfolio. Trial points
are labeled evaluations; they are not misrepresented as accepted solver steps.
Independent rollout must validate the returned best point before a final
stationary-poor-fit assessment can be attached to it.
The stationarity witness is bound to the evaluated parameter-vector hash; a
better checkpoint from another evaluation does not inherit that finding.

There are two explicit initial-point exceptions to the history requirement:
an exactly zero Jacobian, or projected gradient and column correlation both
within `64 * machine_epsilon`. These are labeled `locally_flat_start` or
`first_order_stationary_start`. They permit leaving a degenerate local search
without dividing by zero inside the trust-region solver. They do **not** certify
a minimum or remove any coordinates; a stationary point can be a saddle or
maximum. All predetermined distinct starts remain eligible.

The existing strict accurate-fit target, worst-trajectory check, numerical
verification and incumbent-preservation rules remain in force. The poor-fit rule
does not stop the near-zero-loss coefficient-polishing path. Where the retained
point has independently checked stationary-search evidence, report
`stationary_poor_fit` and `review_equations_or_new_starts`. Budget exhaustion and
unavailable rollouts retain their separate existing diagnostics. No outcome
implies calibrated coefficient confidence or global identifiability.

## Paired experiment

Reuse the exact M29 input artifact:
`cce09c7f7b25573cd7c0c1549b15534d78a18f0e1fff631ebae8cf0148517744`.
Run two arms from fresh:

- `legacy`: the M29 guarded-only policy, with `poor_fit=null`.
- `output_aware`: the same policy, with output-dependency handling and the
  stationarity thresholds above enabled.

There are **22 entries**: nine available models times two arms, plus the two
original construction failures retained in both arms. Six available models are
September 30 Full/seed-0 constructions; three are known-structure controls. This
does not evaluate the latest constructor. No new LLM calls or test access occur.

Original requests, training arrays, coordinates, raw restart vectors and budget
ceilings are identical within each pair. In the new arm only, certified inactive
coordinates stay at incumbent values rather than taking arbitrary restart values;
the effective vectors are saved separately. The report verifies equality of the
raw starts and training problems. A reference coefficient is never a fitting
input; known-control errors are calculated by the post-fit evaluator only.

Both arms retain at most six search blocks, forty calls or 800 seconds per block,
at most two consecutive blocks per start, the original plus three domain-only
starts, and the existing verification/sensitivity allowances. Savings must come
from less unnecessary work, not a larger ceiling. Frozen setup, every search
operation, fallback and independent evaluator remain checkpointed. Interrupted
operations retain their charged allowance; resume cannot reset budgets.

Validation uses exactly the sealed fitted parameters and shared initializers.
The known-control coefficient/initial diagnostics, independent solver agreement,
per-target NMSE, all failures and cost totals remain in the archive.

## Preflight and exit criteria

Local preflight on all nine real requests finds:

- Independent basin: six inactive unknowns, one active gain, a fully profiled
  terminal-output problem.
- T2 hard: the unused `U` coefficient is inactive; the insulin-action state and
  all of its initializer coefficients remain active.
- Other constructed models and all three known controls: no inactive parameters.
- Every pair has identical raw starts and typed training problem.

Synthetic tests cover expanded process paths, multiple outputs, initial maps,
conservative syntactic dependencies, bounded scalar profiling, no-outer-variable
cases, unverified output rejection, stationary-start diversification and exact
resume. A two-output/shared-initializer smoke retains accurate prediction.
The focused regression suite passes 124 tests. The smoke's training/validation
NMSEs are `9.42e-21` and `1.89e-21`, with unchanged deterministic resume. Ruff
passes all changed Python files; the repository-wide check still reports the
37 pre-existing issues in `analysis/claude`.

Remote success requires all 22 outcomes accounted for; no parameter/domain or
data-boundary violations; no redundant flat-outer-profile failures on independent
basin; preservation of its analytically known training optimum (about 2.633,
which remains poor); and retained recovery of prediction, coefficients and shared
initials on all three known controls. Compare actual search calls and total
setup/fitting/verification/evaluation time, not just scheduler elapsed time.
Inspect any changed endpoint before interpreting a speedup. Wall-clock cutoffs
and hardware variation can still change the search path under identical ceilings.

The CSTR jacket sign and basin balance defects are not repaired by this milestone.
Noisy observations, statistical thresholds, broader construction coverage and
production promotion remain future work; see the completed results review above.

## Delta commands

Upload `transfers/phase-c-fitting-m30.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. The bundle contains committed code, frozen inputs,
configuration, tests, `SOURCE_COMMIT` and per-file hashes. It uses the existing
Delta fitting environment; no new installation or GPU is needed.

```bash
cd /work/hdd/bibo/yxiao2/phase_c
tar -xzf phase-c-fitting-m30.tar.gz
bash phase-c-fitting-m30/scripts/hpc/submit_phase_c_poor_fit_delta.sh
```

Default result root: `/work/hdd/bibo/yxiao2/phase_c/fitting-poor-fit-v1`.
Each array entry requests one CPU and 16 GB, with three concurrent entries and a
2.5-hour scheduler ceiling. `AF_CONCURRENCY` changes only the throttle. Use
`AF_POOR_FIT_ROOT` for a different fresh output directory. Existing M29 results
remain untouched. Repeating the launcher resumes confirmed submissions;
uncertain scheduler receipts require reconciliation, not blind resubmission.

Inspect and package results:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/phase-c-fitting-m30/scripts/hpc/inspect_phase_c_poor_fit_delta.sh
```

The command prints paired scores, parameter diagnostics, search calls and costs,
then the exact review archive path to download. The archive includes both arms'
full checkpoints. Production defaults are unchanged until qualification succeeds.
