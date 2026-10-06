# M12: generic-start recovery with training-verified mesh stopping

Protocol: `phase-c-generic-recovery-1`.
Configuration: `configs/phase_c_generic_recovery_v1.json`.
This implements the user's approved next fitting milestone. Construction remains
Astra's responsibility. No benchmark arrays, public prompts, production fitting
defaults or test data change.

## Why this follows M11

The M11 review archive `review-20261006-023756.tar.gz` accounts for all ten tasks,
32 native solves and complete independent evaluation of 16 released endpoints
and ten retained models. Its plan identity is
`5e23339747332d2d0effef6840f7455cf5fb1c90b2c43bd6f50e1d9543699a56`.
Both supplied training-fitted starts approach the same solution; they are not
independent optimization basins. Independent and continuation endpoints agree
numerically. Representative released-endpoint errors are:

| Collocation variables | Train NMSE | Validation NMSE | Maximum coefficient relative error |
|---:|---:|---:|---:|
| 33,906 | 5.8863e-4 | 9.8607e-4 | 9.68% |
| 59,994 | 1.8885e-8 | 3.1353e-8 | 0.117% |
| 89,994 | 1.1996e-9 | 1.0629e-9 | 0.0134% |
| 115,218 | 7.8442e-10 | 6.6593e-10 | 0.0128% |

The coarse solve has tiny enforced collocation defects despite appreciable
rollout and coefficient error. Mesh refinement addresses this discretization
bias. It does not establish recovery from a generic initialization. M11 retains
the original, even more accurate M7 model in every task; those selected scores
are inherited success, not new recovery. The full continuation also took more
native process time than the direct dense solve from that good starting point.
There is no demonstrated end-to-end speedup yet.

M12 returns to **all three original generic starts**, including the previously
unsuccessful seed. No fitted M7/M11 parameter vector or node trajectory is
supplied. The exporter removes assisted starts and saved candidate pools from
the verified development export. Correct equations, the existing parameter
domains and anchored numerical coordinates remain diagnostic assistance.

## Frozen comparison

Keep the alien-device hard case: six states, 13 dynamic coefficients, five
shared latent initial-value parameters, 16 training trajectories and four
validation trajectories. Each arm receives the same generic physical parameter
and initial-value vector for a seed. There are nine tasks:

| Arm | Procedure |
|---|---|
| `rollout_only` | Joint forward-sensitivity least squares from the generic vector |
| `medium_rollout` | One medium collocation mesh, then rollout fitting from the best retained training checkpoint |
| `mesh_rollout` | A nested coarse-to-fine collocation ladder, then rollout fitting from the best retained training checkpoint |

Every arm has a **1,200-second fitting ceiling**, including process startup,
mesh planning, screens and training certificates. This is a maximum, not a fixed
duration or promise that all mesh levels execute. Final validation/reference
evaluation has a separate 300-second allowance. Process cleanup can add up to
ten seconds after a timeout and is reported; it grants no further optimizer work.

For the two collocation arms, the collocation allocation is 75% of the time left
after reserving 180 seconds for final certification: 765 seconds. Each native
solve has at most 250 seconds. A training screen has at most 90 seconds.
Unspent time returns to rollout refinement, which has at most 900 residual calls
and must leave the certificate reserve. Rollout-only has the same overall
ceiling and reserve. These are equal ceilings, not equal stage allocations.

## Initial mesh and refinement

Every level preserves the supplied piecewise-linear input function's corners,
including pulse boundaries. All original observations remain in the objective
with their existing weights, even where state nodes are sparse.

Training observations add a bounded set of anchors: up to eight separated
high-slope/high-curvature locations per trajectory, with adjacent observation
times to bracket each change. This is a mesh-placement heuristic, not an
inference of latent-state events. It has not been qualified on noisy data.
The same anchors appear at every level so that meshes remain nested.

| Requested variable target | Actual variables, including mandatory/observation anchors |
|---:|---:|
| 6,000 with input/resolution floor | 37,110 |
| 60,000 | 62,226 |
| 90,000 | 90,954 |
| All observation times | 115,218 |

The first guess interpolates the frozen generic node guesses. Parameters and
shared initials are released immediately: unlike M11's discretization control,
there is no stage holding a previously accurate fitted parameter vector fixed.
On successful native convergence, complete finite endpoint nodes with scaled
discrete defect at most `1e-6` can transfer by the full Radau polynomial to the
next mesh. A timed-out or unsuccessful solve stops mesh expansion and leaves
the remaining allowance for rollout refinement.

The continuation endpoint and deployed incumbent are separate. The final native
iterate and one distinct low-collocation-objective checkpoint can receive bounded
training-rollout screens. Only a complete rollout can replace the incumbent, and
only if its training NMSE improves. A low nodal objective or tiny enforced defect
alone cannot replace it. The rollout stage keeps parameter/initial-value pairs
intact; it does not mix a parameter block from one checkpoint with initial values
from another without reoptimization.

This remains a fixed nested ladder with observation anchors and early stopping,
not automatic local error-based mesh subdivision. It tests whether these pieces
improve generic-start recovery before adding that further complexity.

## What early stopping certifies

A promising retained model is independently simulated over every complete
training trajectory using both Radau and DOP853 (`rtol=1e-9`, `atol=1e-11`).
The training-only check requires:

- mean per-trajectory worst-solver/channel NMSE at most `1e-6`;
- every trajectory/channel/solver NMSE at most `1e-5`;
- maximum solver disagreement, divided by the training output scale, at most
  `1e-5`;
- complete finite predictions for all training trajectories.

Passing stops further meshes and rollout fitting. A partial or timed-out check
does not pass. This is an **empirical training-prediction certificate**, not a
mathematical integration error bound, parameter-recovery certificate, or guarantee
on unseen interventions. It need not require collocation convergence when an
intermediate checkpoint already yields accurate independent rollouts.

Training selection is sealed before validation or reference coefficients are
read by the separate evaluator. Post-fit reports retain prediction accuracy,
coefficient error and shared-initial error as separate outcomes. None controls
the fit. All starts remain in the denominator; no best-seed selection is reported
as initialization robustness. The inherited evaluation gates remain unchanged.

## Checkpointing, failure and interpretation

Journal every process start and terminal receipt. Completed tasks resume exactly.
An interrupted fitting coordinator closes with its last retained evidence and
`interrupted_no_fit_restart`; it does not grant a fresh twenty minutes. Resuming
a sealed backend can finish its separate evaluation journal. If a worker's exit
cannot be confirmed, no subsequent fit or independent evaluation is launched.

The report distinguishes missing tasks, unavailable evaluation, native solve
outcomes, early training certification and post-fit accuracy. Failing an accuracy
gate is still a recorded experiment. A campaign marked `complete` means every
task has a terminal record, not that every fit recovered the truth.

The primary question is whether a mesh-assisted arm recovers the difficult
original start within the same ceiling while preserving the previously successful
starts. Report all nine outcomes and actual elapsed costs. A negative result
would motivate restart/continuation or identifiability work, not omission of that
seed. This one correct-equation example cannot establish production robustness.

## Delta commands

Upload `phase-c-fitting-m12.tar.gz` to `/work/hdd/bibo/yxiao2/phase_c/`.
It includes the sealed original generic starts and arrays, so no old experiment
directory, new environment, GPU or API credential is needed.

```bash
bash <<'BASH'
set -euo pipefail
AF_BASE=/work/hdd/bibo/yxiao2/phase_c
mkdir -p "$AF_BASE/code/fitting-m12"
tar -xzf "$AF_BASE/phase-c-fitting-m12.tar.gz" -C "$AF_BASE/code/fitting-m12"
export AF_RECOVERY_ROOT="$AF_BASE/fitting-generic-recovery-v1"
bash "$AF_BASE/code/fitting-m12/scripts/hpc/submit_phase_c_generic_recovery_delta.sh"
BASH
```

Preparation runs focused tests. Nine array tasks use one CPU and 16 GB each,
with at most two running concurrently. Their scheduler limit is 35 minutes to
cover the fitting ceiling, final evaluation and orchestration. A dependent report
runs after the array, including after failures. Repeating a completed submission
reuses saved job IDs; uncertain scheduler replies require reconciliation.

Check results and produce the review archive:

```bash
bash /work/hdd/bibo/yxiao2/phase_c/code/fitting-m12/scripts/hpc/inspect_phase_c_generic_recovery_delta.sh \
  /work/hdd/bibo/yxiao2/phase_c/fitting-generic-recovery-v1
```

The inspector accepts the explicit root and verifies the protocol first. It does
not reuse an old `AF_CAMPAIGN` shell variable. It prints every arm's scores,
coefficient/initial errors, elapsed time and meshes, then the archive to download.

## Verification

Focused tests cover the complete generic roster, removal of fitted assistance,
training-only mesh/certificate boundaries, nested input preservation, bounded
observation anchors, partial/disagreeing rollouts, early stopping, incumbent
preservation after an unfinished solve, interrupted budgets, post-fit evaluation,
cleanup blocking, scheduler receipts and wrong-campaign inspection.

A native linear smoke exercises all three arms, independent prediction checks,
post-fit evaluation and exact terminal resume. Hard-case work locally is limited
to sealed-input verification and mesh planning; the nine optimizations run on Delta.

The native linear smoke passes all three arms, coefficient and initial-value
recovery, and exact terminal resume. Training/validation NMSEs are respectively
`4.85e-10/5.83e-10`, `7.77e-11/7.49e-11` and `2.42e-9/2.88e-9` for rollout-only,
medium and progressive meshes. These are small-control checks, not new alien
recovery results. All 48 focused tests pass in the portable tree; changed Python
passes Ruff, shell entry points pass syntax checks, and repository-wide Ruff
still reports 37 unrelated pre-existing findings under `analysis/claude`.

The generic-only input body hash is
`ed71b7ee77608f94b66e7c16a91c0a0c29f9c05ec7b25bc594694ba079d42d09`.
All three starts give the mesh counts above; planning verifies nesting and
unchanged observations/forcing without running a hard-case optimizer locally.
