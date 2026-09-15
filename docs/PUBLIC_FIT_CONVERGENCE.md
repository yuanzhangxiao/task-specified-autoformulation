# Public fitting convergence diagnostic

This is the user-authorized follow-up to the completed one-window continuation
in `PUBLIC_FIT_CONTINUATION.md`. Its purpose is to measure how much further
training fit improves, and when the optimizer meets a local stopping criterion,
when given substantially more time. **The final pipeline continuation limit
remains undecided.** The fitting algorithm and existing numerical profiles are
unchanged.

## Frozen experiment

`configs/public_fit_convergence_v1.json` identifies the completed continuation
`46e9914d…` and its exact backend digest `d92f0871…`. The experiment starts from
its complete retained vector: eight equation parameters and five learned causal
initializer coefficients. The supplied starting evidence is:

- training cost: 3978.0498151453285;
- training output NMSE: 0.8273814091604574;
- validation output NMSE: 0.92851911527396;
- 36 cumulative residual calls;
- previous continuation stopped at its 180-second budget, with no recorded
  integration failures and continuing accepted-step improvements.

The diagnostic permits **20 additional windows of at most 180 numerical seconds
and 240 residual calls each**, allowing up to 3600 additional numerical seconds.
The original collocation allowance was 120 seconds; the original refinement and
completed continuation each allowed 180 seconds. These are allocated budgets,
not measurements of elapsed optimizer time.

Each new window uses the existing `run_extension` implementation unchanged:
forward-sensitivity least squares, `ftol=None`, the original domains, tolerances,
training normalization and data. It starts from the entire retained vector,
including learned initialization-map parameters. There is no collocation rerun,
random start, model revision, LLM call, hidden-label access or test access.

**A window is a fresh optimizer invocation initialized at the retained point.**
It does not restore trust-region radius or other solver state. Consequently this
experiment measures repeated warm-start refinement, not one uninterrupted
3600-second solve. Setup, repeated start checks and scoring overhead are visible;
setup and scoring are outside the numerical allowance. The start check itself is
charged to the window. The ACES job requests three hours for preflight and this
overhead, using one CPU, 16 GB and no GPU.

## Selection and stopping

After each window, retain an improvement only when its full-training objective
and production training NMSE both improve. Otherwise keep the incumbent. Score
validation only after training selection; validation never chooses parameters,
extends the budget, or stops the diagnostic. Validation scoring failures remain
visible without consuming a proposer round or causing a fitting-policy decision.

The diagnostic stops on one of these conditions:

1. **Local stationarity criterion met:** native optimizer success at the exact
   retained parameter vector, a fresh successful training-rollout verification,
   and a finite, nonnegative reported optimality measure at most `1e-6`.
2. **Numerical failure:** a non-budget training/integration error or unsuccessful
   production training scoring. The best already verified incumbent is kept.
3. **Interruption:** a window was started but no complete backend result survived.
   It is consumed and not rerun with a fresh allowance.
4. **Diagnostic cap reached:** all 20 authorized windows have been used.

A native `xtol` success with a large optimality measure is recorded but does not
satisfy condition 1. Slow improvement (less than 1% training-cost reduction in a
window) is descriptive only; there is no plateau-based early stop in this
experiment. A timeout inside a window can continue to the next authorized window.

Neither local numerical stationarity nor consistent production rollouts certify
scientific correctness, uniqueness or global optimality. Reaching the cap means
stationarity was not observed within this experiment; it cannot show that the
model would never converge. NMSE measures only observed `v01`, not latent-state
error. Independent BDF/Radau replay is not part of this diagnostic.

## Checkpoint and provenance contract

Both entire historical experiment roots remain read-only. Preparation validates
historical freeze/result/backend hashes, the old pilot gate and ledger, the full
parameter layout and domains, and reconstructed public data/lowering/settings.
Historical source hashes are preserved without requiring the current source to
be the old version. The new freeze binds the current source, runtime, selected
seed and numerical allowances.

A separate reservation permits one diagnostic for this selected continuation.
Changing the output directory or selection cannot silently create new budget.
The historical one-extension reservation remains intact. This is explicit new
authorization, not resetting the old pilot.

Every window has a started marker, exact input-vector hash, previous-result hash,
backend evidence and sealed terminal record. Resume skips completed windows and
runs only unstarted, authorized windows. A complete backend whose final record
was not published can be finalized without rerunning optimization. An incomplete
started window terminates as interrupted and preserves all partial evidence;
unknown time/call counts are marked unknown, and reported sums are lower bounds.
No automatic replacement run, cap extension or proposer revision follows.

## Running and reviewing on ACES

From a clean pinned checkout, run
`scripts/hpc/submit_public_fit_convergence_aces.sh` with `AF_REPO_ROOT` set to that
checkout. Defaults are:

```text
AF_PARENT_FIT=/scratch/user/u.yx126462/phase_b/prefit-public-fit-v1/fit
AF_CONTINUATION_FIT=/scratch/user/u.yx126462/phase_b/public-fit-continuation-v1/continuation
AF_OUTPUT_ROOT=/scratch/user/u.yx126462/phase_b/public-fit-convergence-v1
AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
```

The launcher binds the checkout and all six historical JSON artifacts before
submission, then verifies them again after dependency tests and a numerical smoke
control. Repeating submission with the same completed manifest prints the
existing job ID; it does not submit another job.

Review the table after each window, including while the job runs:

```bash
cat /scratch/user/u.yx126462/phase_b/public-fit-convergence-v1/diagnostic/SUMMARY.md
```

`diagnostic/summary.json` is updated after every window. The top-level
`summary.json` is published after the job's complete diagnostic run. Each
`diagnostic/windows/NNN/backend_result.json` contains the full parameter vector,
accepted iteration costs, native status, optimality, failure records, start check,
setup/numerical/scoring times and retained selection. Its `calls/` directory
contains detailed numerical checkpoints.

For an interrupted job, first inspect scheduler state and the last artifacts.
The CLI `run --output .../diagnostic` on the exact checkout/runtime resumes only
unstarted windows and never refreshes an interrupted window. `report` and
`inspect` perform no numerical work. They require the diagnostic lock to be free;
use `cat` on the atomically published summary during an active run.

## Validation of this milestone

Controller tests cover complete initializer carryover, worsening/unavailable
validation without routing changes, flat-progress continuation, native stop
qualification, interruption, partial publication, resume, immutable source/runtime
and lineage, and budget reservations. Scheduler tests cover both historical roots,
queued source/artifact drift, duplicate submission prevention and the CLI chain.
The numerical smoke uses a toy observed-output recovery example. Its original
parent timeout history is explicitly a fixture; its continuation and diagnostic
numerical solves are real. It is an implementation check, not evidence that this
public benchmark will converge.
