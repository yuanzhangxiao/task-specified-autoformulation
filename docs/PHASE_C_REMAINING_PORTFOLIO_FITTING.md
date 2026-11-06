# Fitting the remaining 22 construction candidates

5 November 2026. This campaign continues the 22 unchanged parents while Astra
revises candidate 15 separately. They are alternatives for the same Dalla Man
T2 canonical named easy task. The purpose is to improve each parent's fit and
collect concrete training mismatches for subsequent construction revisions.

## Numerical work

| Candidates | Route |
| --- | --- |
| 00, 05–09, 11–12, 16, 18–22 | Exact affine propagation. Fit shared coefficients while conditionally solving the 32 training initial roots at every evaluation. Run a warm continuation and a predetermined distinct shared-parameter start. |
| 10, 14 | Continue the verified nonlinear fits with residual-only optimization. Fit each small shared-coordinate block, then release all coordinates, including training initial roots. Repeat from a predetermined distinct start. |
| 03 | Qualify repeatable state-only residuals and independent original-equation replay before the same derivative-free route. The unresolved derivative evidence remains unresolved. |
| 01, 02, 04, 13, 17 | Try five predetermined starts within the declared parameter domains. Use the first independently qualified start for bounded block fitting and full release. Report all unsuccessful attempts. |

Certified reductions remove redundant optimizer coordinates while retaining an
explicit lift to the original model and its parameter domains. The affine route
uses the generic initial-root profiler. Candidate 15's specialized meal-forcing
profile is not assumed valid for these other equations.

The second start changes shared coordinates relative to the frozen starting
vector. Recovery probes also vary independent training roots. Their order and
values are fixed before seeing their fitting losses. No physiological bounds,
ground-truth parameter values, equation changes, or additional observations are
introduced. A finite unsuccessful set of starts cannot establish that the model
family has no usable parameters.

Each phase records its own stopping reason and evaluated best point. Independent
original-equation replay is required before an endpoint can replace the retained
fit. The source incumbent survives failed, interrupted or worse searches. A
conditional optimum for the roots does not certify an optimum for the shared
coefficients. A block stop does not certify stationarity of the full problem.

Ordinary candidates receive up to 900 numerical seconds per start, for two
starts. Nonlinear fitting divides each start's allowance across all three shared
blocks and full release. Recovery candidates screen at most five starts for up to 120
seconds each, then give the first qualified start 1,800 fitting seconds. Each
original-equation replay has a separate 180-second allowance. Feedback receives
another 180 seconds. Inclusive reservations total at most 6,300 seconds per
candidate, including setup and persistence allowances.

## Sources and reporting

The compact intake authenticates the original 23-model portfolio, the later
10/12/14 endpoints, and the 14 conditional-root fits from 5 November. Candidate
12 retains its exact-zero-decay control separately from its positive-rate warm
start. The intake includes only public training data and selected numerical
receipts. The launch binds this intake to source code, runtime and resource
settings in a fresh root.

Each candidate returns its source and retained NMSE, per-target errors, attempted
starts, qualification outcomes, stopping reasons and operation costs. A separate
bounded original-equation rollout produces predictions for plotting and
descriptive mismatch records. These include samplewise peak timing and height,
initial/final error, and errors during ingestion, before ingestion, after
ingestion, and on schedules with no meals. Input phases are derived from the
supplied ingestion rate. Peak plateaus and endpoint maxima remain visible.

These records describe full-training reconstruction with fitted training roots.
They do not establish prefix-only forecasting or held-out performance. Fresh
eight-minute-prefix evaluation of the improved endpoints is deferred to a
separate milestone. No validation or test data are opened, and no LLM is called.
Residual windows locate a mismatch but do not automatically identify its cause,
certify convergence, or prescribe an equation change.

Every numerical operation owns a durable, nonrenewable reservation. Resume reuses
completed receipts and continues unattempted stages. It does not renew an
interrupted search. Useful partial search checkpoints can enter their separately
reserved verification stage. Reports preserve missing tasks and numerical
failures even when Slurm reports that a worker completed.

## Delta launch

Upload the supplied bundle from the Mac. It contains the committed source and
the authenticated compact intake. Keep the exact extracted code directory for
the lifetime of this campaign.

```bash
scp /absolute/local/path/construction-remaining-portfolio-bundle.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

On Delta, use a new persistent code directory and the new campaign root.

```bash
set -euo pipefail
AF_BUNDLE=/projects/bibo/yxiao2/construction-remaining-portfolio-bundle.tar.gz
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/construction-remaining-portfolio.XXXXXX)
tar -xzf "$AF_BUNDLE" -C "$AF_REPO"
cd "$AF_REPO"
sha256sum -c SHA256SUMS --quiet
export AF_REMAINING_PORTFOLIO_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1
bash scripts/hpc/start_construction_remaining_portfolio_delta.sh run
printf 'Keep this code directory: %s\n' "$AF_REPO"
```

The default uses the existing project Python and CasADi dependency directory,
the CPU account `bibo-delta-cpu`, one CPU and 8 GB per task, and two concurrent
tasks. There are 22 array tasks. Each requests four hours including numerical
work, verification and filesystem headroom. The report requests one hour and
runs after the array terminates, including unsuccessful workers. These are
scheduler ceilings, not expected durations. Login-side preparation authenticates
the compact intake and does not run a fitting experiment or test suite.

From that same code directory, inspect progress and package the results.

```bash
export AF_REMAINING_PORTFOLIO_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1
bash scripts/hpc/start_construction_remaining_portfolio_delta.sh status
squeue -u "$USER" -o '%.22i %.42j %.12T %R'
bash scripts/hpc/start_construction_remaining_portfolio_delta.sh inspect
```

If the terminal session is new, recover the existing code directory instead of
creating an empty one.

```bash
export AF_REMAINING_PORTFOLIO_ROOT=/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1
AF_REPO=$(jq -er '.value.repo' "$AF_REMAINING_PORTFOLIO_ROOT/launcher_execution.json")
bash "$AF_REPO/scripts/hpc/start_construction_remaining_portfolio_delta.sh" status
bash "$AF_REPO/scripts/hpc/start_construction_remaining_portfolio_delta.sh" inspect
```

Download from the Mac after inspection has written the archive.

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/construction-remaining-portfolio-v1/inspection.tar.gz \
  "$HOME/Downloads/construction-remaining-portfolio-inspection.tar.gz"
```

An ambiguous scheduler reply blocks automatic resubmission. Inspect its saved
submission intent and Slurm accounting before adopting an already accepted job.
Changing the code or concurrency under this root changes its sealed identity.

## Implementation

- `fitting/remaining_portfolio_search.py` provides route qualification, legal
  predetermined starts, block masks and the existing profilers/searchers.
- `fitting/remaining_portfolio_feedback.py` produces descriptive training records.
- `research/construction_remaining_portfolio.py` authenticates the intake,
  journals all 22 tasks, preserves incumbents, and reconstructs reports.
- `scripts/run_construction_remaining_portfolio.py` and the two Delta shell
  wrappers provide the independent CLI and scheduler boundary.

The experiment is opt-in. Stronger fitting-adequacy and parameter-confidence
conclusions require their own completed diagnostics after this bounded pass.
