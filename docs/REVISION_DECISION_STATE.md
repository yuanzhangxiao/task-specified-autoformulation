# Revision decision state and action contracts (v6)

This milestone changes the pre-fitting search interface, not the fitter. It uses
the same two public source candidates, four rounds, model, sampling settings,
request budget and numerical settings as v5. No benchmark data or finalized prompt
is changed. Private generator details are not included in proposer requests.

## What the evidence does and does not establish

Zero external input is not equivalent to equilibrium. An autonomous source or a
non-equilibrium initial state can produce a changing output without forcing.
The generic public generation implementation permits autonomous dynamics. Public
`train_000` therefore does not establish a ground-truth error. The observed defect
was narrower: the audited candidates satisfy F(0,0;theta)=0 and use zero initials,
so their zero solution cannot reproduce this changing-output trajectory.

No recorded metric is invalidated by that observation. However, the three stable
v5 rounds were nearly zero-output predictors: their scores matched the zero
baseline, not useful predictive recovery. Successful compilation, domain checks,
rollout completion and optimizer success must remain distinct endpoints.

Astra's September 11 update also identifies a Delta /projects and /taiga outage
during experiments. Missing or stalled jobs are unavailable evidence, not failed
numerical algorithms. Astra owns a separate, pinned-code recovery campaign. This
milestone makes no collocation/sensitivity or numerical-default changes.

## Three separate artifacts

- `decision_state.json`: current committed candidate digest, last numerical
  evidence and its candidate/round, history, best evaluated candidate, and a
  compact pending-action description. A failed revision cannot replace the last
  numerical feedback with a transport diagnostic.
- `pending_revision.json`: parent/contract-bound provisional candidate, accepted
  edits, explicitly kept companions, pending components, parameter aliases and
  every attempt. It survives retry exhaustion and process interruption. Nothing
  is fitted until the whole action commits.
- `round_NNN/round_record.json`: immutable completed decision record, including
  not-run outcomes. Resume consumes this record without repeating completed
  calls or fitting. `best_evaluated_candidate.json` retains the best finite
  validation-scored candidate even when later search steps fail or worsen.

Best-evaluated retention is a diagnostic safeguard, not a new scientific scoring
rule. The current exploratory lineage and original routing heuristic remain in
place; changing incumbent acceptance and stagnation routing is a later milestone.

## Provider contract

The v6 response contains `revisions` and `keep_components`. A revision still has
`component`, `expression`, and `parameters`; a parameter has `name` and nullable
`role`. Every pending component appears exactly once in one of the two lists.
The runtime also treats an executable-equivalent replacement as a logged keep.
An all-keep action is `no_change`, not a rejected reply or a new fitting task.

Requests expose each equation's signed additive interactions: sign, complete
nonparameter source set and multiplicity. A function action must preserve that
multiset, not merely the equation's source union. New offsets and cross-variable
couplings can require topology scope. Rejections show the removed and added
interactions and identify the required scope; no scientific edit is silently made.

Parent parameters use short, reversible request-local aliases. Their canonical
identities and roles remain fixed and do not need redeclaration. Genuinely new
parameters still need declaration. A constant-scaled offset such as `b*1` is
recognized as an offset, but adding it is not automatically allowed by a frozen
function topology. Sign ownership is unchanged.

The reviewed public nonlinear obligation is pinned to its public prompt SHA and
exact requirement quote. A revision cannot remove all non-affine generated-state
syntax from a required target path. This is only a necessary syntax/path check:
it does not certify nonlinear feedback, exclude algebraic cancellation, or ensure
the fitted weights activate the mechanism. Scientific mechanism scoring remains
separate.

Feedback adds fitted parameters, public training RMS and a clearly labeled pooled
zero-prediction baseline, observed zero-input/changing-output trajectories, fixed
initializers and numerical history. These are observations, not hidden-reference
answers or proof of an impossible model. No validation/test initial states are
estimated from future observations.

## Offline verification before another allocation

`scripts/replay_staged_revision_actions.py --source-root SOURCE --output REPORT`
reconstructs each stored v5 request's actual provisional equations and parameter
registry, then tests its unchanged response with the new action checker. It reads
only the frozen candidate, plan and public call records. It makes no LLM call,
fit, judge request, or test/private access. Every reply is retained in accounting.
This is a same-decision-point audit, not a counterfactual search or a prediction of
what the model will do under the new prompt.

Local replay of all 26 v5 replies produced 5 committed actions, 5 no-change actions
and 16 rejected actions. It retained 14 denominator findings, exposed 6 exact
signed-interaction changes, and detected 1 lost public nonlinear obligation.
The remaining source-set, missing-component and unavailable-symbol defects remain
visible. Acceptance is not the sole goal: genuine contract failures must survive.

## Live experiment and limitations

Use `configs/staged_multiround_feedback_v6.json` with the existing ACES submission
script and the completed `staged-fitter-rescue-v1-c1754fe` source. Use a fresh output
root; never resume v5 artifacts under changed source code. Explicitly set the
existing vLLM image path and shared Python executable.

The shared shell launcher supports protocol v6. The submitter runs its actual
protocol/platform dispatch using `--check-config` before checking large inputs or
calling Slurm. Regression tests exercise every committed multiround config,
unknown-protocol rejection, and the production v6 dispatch path without GPUs.
Job 2110814 stopped at the previously missing shell allowlist entry before any
provider or fitting work; retry in a fresh root with the corrected launcher hash.

The summary separates recorded rounds, attempted fits, all-keep rounds, rejected
provider attempts and rejected component entries. Legacy rates remain for
historical comparison; their denominators include not-run round records.

Still deferred: learned global latent initials, causal initializer learning,
broader inventory edits, sensitivity-based localization, stagnation routing,
scientific judge integration, proof of trajectory reachability, and automatic
promotion of a challenger over the incumbent. These must not be conflated with
this action-interface milestone or Astra's infrastructure recovery.
