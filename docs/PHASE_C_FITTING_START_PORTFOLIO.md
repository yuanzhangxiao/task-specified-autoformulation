# M13: preserve optimization starts and allocate rollout fitting by trials

Protocol: `phase-c-start-portfolio-1`.
Configuration: `configs/phase_c_start_portfolio_v1.json`.
This implements the user's approved follow-up to the
[M12 review](PHASE_C_FITTING_M12_RESULTS_2026-10-06.md). It changes no construction
stage, benchmark data, finalized public prompt or production fitting default.

## Question and controls

Rollout fitting recovered two of three original alien-device hard starts in M12.
The progressive-mesh arm recovered exactly the same vectors after falling back
to the original starts; it established no mesh-assisted recovery. Medium
collocation supplied better current training scores but worse subsequent fits.
An incumbent for deployment and a start for optimization must therefore be kept
separately. This experiment asks whether short optimization trials improve start
selection and recovery within the same CPU allowance.

Keep all three original generic starts and the frozen M12 data. The diagnostic
supplies the correct six-state equations and anchored numerical coordinates, with
13 dynamic coefficients and five shared latent-initial parameters. It uses 16
training and four validation trajectories. This assistance is not a construction
result, and three starts on one problem are not independent scientific tasks.

| Arm | Starting pairs and allocation |
|---|---|
| `rollout_only` | Existing M12 joint forward-sensitivity TRF from the original generic pair |
| `rollout_portfolio` | Original pair plus two deterministic generic perturbations; short trials, then bounded continuation of eligible candidates |
| `checkpoint_portfolio` | One medium collocation attempt, then original pair, one screened collocation pair and one generic perturbation; same trial policy |

There are nine tasks, not nine restarts inside one task. Each has the same
1,200-second fitting ceiling including preparation, collocation where applicable,
screening, trials and training certificates. The last 180 seconds are reserved
for training certification. Independent final evaluation has a separate
300-second allowance. Early termination can leave budget unused; these are equal
ceilings, not forced equal CPU consumption.

The rollout-only control retains its previous optimization policy. Portfolio
trials reuse that same joint forward-sensitivity TRF implementation; this milestone
tests allocation among starts, not a new derivative or integration algorithm.

## Start generation and preservation

The original physical parameter/initial pair is always first. Two deterministic
perturbations use a random seed derived from that original vector's content hash,
never from reference coefficients, validation scores or the identity of a failed
seed. For an original positive parameter with a nonnegative lower bound, use a
multiplicative log-normal perturbation with log standard deviation 0.7. Signed
parameters and shared initials receive normal perturbations with standard
deviation 0.7 times their existing numerical coordinate scale. Clip to the same
declared physical bounds used by rollout fitting; do not narrow bounds around the
truth. Save all generated pairs and remove exact duplicates while retaining the
original. Parameter and initial-value blocks stay paired throughout.

The checkpoint arm attempts the existing medium mesh (62,226 variables in this
case), retaining every observation and the exact public input interpolation. Its
native allowance is 250 seconds. At most two recorded collocation points get
90-second training-rollout screens. The best complete screened pair replaces one
perturbation in the candidate list, **never the original pair**. If none is
available, both generic perturbations remain. All collocation and screening costs
come out of the same ceiling. A timeout does not make an intermediate iterate
scientifically invalid; an actual complete training rollout is required to score it.

## Trial and continuation policy

Each distinct candidate gets one joint rollout-fitting trial with at most
120 seconds and 60 residual calls. Trials occur before allocating longer fitting
to any candidate, unless an independent training certificate already passes or
the shared allowance ends. Each trial includes its ordinary rollout screen and
can preserve every useful complete parameter/initial pair.

After these trials, continue the eligible candidate with the lowest *post-trial*
training NMSE. Ties favor higher measured improvement per second, then stable
source order. A continuation has at most 240 seconds and 120 calls. All rollout
trials together have at most 900 charged residual calls. After two continuations
with relative NMSE improvement below 1%, retire that candidate from further
allocation and consider another. Optimizer termination, numerical failure,
stagnation or a per-point sensitivity timeout also retires only that candidate.
Its best verified model remains eligible to be the final incumbent.

The ordinary 30-second sensitivity point limit remains unchanged. A derivative
failure no longer ends the entire portfolio while other candidates and budget
remain. This milestone switches starting pairs; it does not claim to repair the
failed derivative or add a new stiff sensitivity solver. A candidate stopped on
the optimizer's training-accuracy threshold but rejected by the independent
certificate is retired, avoiding repeated termination at the same unverified point.

Continuation means a new TRF solve warm-started from the candidate's best complete
pair. It does not restore the previous optimizer's trust-region state. Repeated
screens and startup costs are charged. The deployable incumbent is the lowest
complete training NMSE over all evaluated pairs, including retired candidates.
No parameter block from one pair is combined with initials from another.

## Checkpoint publication repair

`transcription_solver` now records final publication separately from native
optimizer success. A rejected checkpoint records its raw parameters, iteration,
objective/defect when finite, domain violation magnitudes and rejection reason.
Nonfinite parameter values are represented as null in the diagnostic record.
Intermediate rejection counts and the latest rejection are also saved. A
successful final publication retains nodes and internal Radau nodes as before.

The new checkpoint arm explicitly disables IPOPT's bound relaxation and honors
original bounds. Historical native solver options remain unchanged unless this
opt-in flag is set. The strict physical checkpoint gate is not loosened. This
removes one plausible cause of the M12 missing endpoint; M12 did not record its
actual rejection reason, so that cause remains unproven. A process killed before
final capture is labeled as such, not as a successful published endpoint.

Subprocess receipts now hash checkpoint pools, rejection records and final
publication status along with existing outputs. Tests exercise both a real
successful small solve and deliberately rejected final values even when the
native solver reports success. A converged NLP and a usable checkpoint are
distinct facts.

## Safety, costs and final assessment

Every subprocess has an intent, payload, deadline, output hashes and terminal
receipt. A timed-out worker can contribute its durable complete training
checkpoint only after process termination is confirmed. If it did not publish a
terminal residual-call count, charge its entire assigned call cap; unknown calls
never become free. An unconfirmed exit blocks subsequent fitting and evaluation.

Completed tasks resume exactly. An interrupted coordinator closes with its last
saved evidence and `interrupted_no_fit_restart`; rerunning does not reset the
20-minute budget. Post-fit evaluation can finish separately after safe closure.
Keep old M12 roots and code frozen; M13 uses a fresh root and protocol.

The training certificate uses complete independent Radau and DOP853 trajectories
and the M12 thresholds: mean worst-solver/channel NMSE at most `1e-6`, every
trajectory at most `1e-5`, normalized solver disagreement at most `1e-5`.
Passing permits early stopping. This is an empirical training-prediction check,
not a global optimizer or parameter-identifiability certificate.

Only after the selected pair and backend are sealed does the separate evaluator
read validation or reference coefficients. Keep prediction, coefficient recovery
and latent-initial recovery separate. The inherited post-fit gates are NMSE at
most `0.01` on train/validation, maximum coefficient relative error at most 1%,
and maximum initial absolute error at most `0.01`. Report every original start,
total elapsed cost, actual/unknown call accounting, retired candidates and their
reasons. More candidates within one task are not additional independent samples.

## Delta commands

Upload the supplied `phase-c-fitting-m13.tar.gz` to
`/work/hdd/bibo/yxiao2/phase_c/`. It contains pinned code, focused tests, configuration
and the unchanged sealed generic inputs; no new environment, source checkout,
GPU, API key or old experiment directory is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m13"
tar -xzf "$AF_BASE/phase-c-fitting-m13.tar.gz" -C "$AF_BASE/code/fitting-m13"
export AF_PORTFOLIO_ROOT="$AF_BASE/fitting-start-portfolio-v1"
bash "$AF_BASE/code/fitting-m13/scripts/hpc/submit_phase_c_start_portfolio_delta.sh"
BASH
```

Nine tasks use one CPU and 16 GB each, at most two concurrently. The scheduler
limit is 35 minutes per task to cover fitting, final evaluation and orchestration.
Preparation runs focused tests; the report runs after the array, including after
failures. Submission intents prevent blind resubmission after an uncertain
scheduler reply. Repeating a confirmed submission reuses its job IDs.

Check and package results with the explicit output root:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m13/scripts/hpc/inspect_phase_c_start_portfolio_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-start-portfolio-v1
```

The inspector verifies the new protocol, prints scheduler and per-task recovery
results plus candidate retirement/trial counts, and gives the review archive path.
Download that archive for inspection. A complete campaign only means all tasks
have terminal records; robustness is established by the recovery results.

## Verification and limitations

Local verification: full `pytest -q -n 4` completed with **4,094 passed,
8 skipped** (optional Torch tests). Repository-wide `ruff check .` reports 37
existing findings in unrelated, untracked `analysis/claude` files; all changed
Python files pass. `git diff --check` passes.

Targeted tests cover deterministic bounded starts, original-pair preservation,
local derivative failure, uncertain call accounting, interrupted budgets,
independent-certificate rejection, wrong worker identity, post-selection evaluation,
confirmed scheduler resume and native checkpoint publication/rejection.
The native linear smoke passes all three arms and exact terminal resume. Its
portfolio arms certify after their first trial, so controlled tests separately
exercise switching and continued allocation. Large-case optimization runs on Delta.

This is a bounded portfolio heuristic, not guaranteed global recovery. Restart
spread, trial duration and stagnation thresholds may be suboptimal. Correct
equations, known numerical coordinates and the single noiseless diagnostic remain
assistance; a positive result would still need broader qualification before
production integration. No Phase-C construction changes are included.
