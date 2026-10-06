# M12 results: generic-start recovery remains unresolved

Reviewed `review-20261006-154711.tar.gz` on 2026-10-06. All nine tasks and their
independent evaluations completed. Rollout-only recovers two of three generic
starts. The progressive-mesh arm recovers the same two by falling back to the
original starts; the medium-mesh arm recovers none. No progressive run advances
beyond its first mesh. This is not evidence that adaptive mesh recovery works.
The native final-checkpoint recording issue below also limits that comparison.

## Frozen scope and audit

- Protocol: `phase-c-generic-recovery-1`; [M12 runbook](PHASE_C_FITTING_GENERIC_RECOVERY.md).
- Source commit: `07cb022319ac629c076fe4050157332e25d5060c`.
- Plan SHA256: `fcddaf8ff1c6c08cf598850189111f942add765d87a3ebb6ffd578003daad072`.
- Archive SHA256: `196ed31905e7f7fd62eff71527d64b952d8e6b9a415210c5946fc60688f19077`.
- Delta jobs: prepare `22696819`, array `22696820`, report `22696821`.
- Correct alien-device hard equations: six states, 13 dynamic coefficients,
  five shared latent-initial parameters, 16 training and four validation
  trajectories. Three original generic starts; no fitted M7/M11 assistance.
- Each arm has a 1,200-second fitting ceiling, including screening and training
  certification. Final evaluation has a separate allowance. These are matched
  ceilings, not identical allocations among stages.

Verified plan/input seals, nine result/backend identities, exact agreement of
summary rows with results, 42 subprocess receipts and 46 recorded output hashes.
All subprocess exits are confirmed and none restarts its consumed budget.
Every selected model has complete Radau and DOP853 replay over all 20 trajectories
(180 model/trajectory records, 601 samples each). Maximum normalized solver
disagreement across the final replays is `3.05e-8`. No test data or LLM calls
are involved; validation and reference coefficients are evaluated after selection.
This local review performs no new hard-case optimization or remote operation.

## Prediction and parameter recovery

All errors below are independently replayed selected-model scores. Maximum
coefficient error is relative and expressed as a percentage. Maximum initial
error is absolute in the diagnostic model's physical coordinates. Time includes
fitting orchestration and training checks, but excludes final evaluation.

| Start | Arm | Train NMSE | Validation NMSE | Max. coefficient error | Max. initial error | Fit minutes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Rollout-only | 3.337e-9 | 3.635e-9 | 0.0428% | 3.828e-4 | 8.82 |
| 0 | Medium + rollout | 0.656115 | 0.691067 | 8,685.7% | 5.748 | 18.73 |
| 0 | Progressive meshes + rollout | 3.337e-9 | 3.635e-9 | 0.0428% | 3.828e-4 | 14.28 |
| 1 | Rollout-only | 1.548e-9 | 4.747e-9 | 0.1156% | 5.978e-4 | 10.73 |
| 1 | Medium + rollout | 0.855292 | 0.989782 | 28,706.0% | 4.248 | 18.91 |
| 1 | Progressive meshes + rollout | 1.548e-9 | 4.747e-9 | 0.1156% | 5.978e-4 | 17.80 |
| 2 | Rollout-only | 0.124009 | 0.121818 | 1,033.5% | 2.718 | 19.47 |
| 2 | Medium + rollout | 0.248914 | 0.359401 | 36,399.1% | 26.703 | 11.01 |
| 2 | Progressive meshes + rollout | 0.124095 | 0.123774 | 1,025.4% | 2.401 | 18.90 |

Prediction, coefficient and initial-value recovery gates all pass for the same
four tasks: starts 0/1 in rollout-only and progressive-mesh arms. Thus recovery
is respectively **2/3, 0/3 and 2/3**. These are three starts on one problem, not
independent scientific tasks or a population success-rate estimate.

Those four tasks also pass the stricter training-only early-stop certificate.
The other five complete their checks but fail the prediction thresholds. Early
stopping saved time on successful fits; it did not certify convergence of meshes.

## What the meshes actually did

Each medium solve has 62,226 decision variables and reaches its 250-second native
process limit. Coarse solves have 37,110 variables. Coarse starts 0/1 also time out.
Their screened collocation candidates do not improve the generic incumbent, so
rollout refinement starts from the original generic parameter/initial pair.

For starts 0/1, progressive-mesh and rollout-only arms return **exactly identical
parameter vectors** and follow the same rollout optimization histories. The
mesh attempts add 328 and 425 seconds, respectively. Their success therefore
cannot be credited to mesh continuation.

Coarse start 2 is different: IPOPT reports `Solve_Succeeded` after 113 iterations,
but `final_checkpoint_diagnostics.json` is absent. The last recorded checkpoint
is iteration 101, with nodal NMSE `0.629443` and scaled defect `3.88e-8`. It is not
the final iterate. Without final node trajectories, mesh transfer correctly does
not proceed. The screened saved points have rollout NMSE about `0.8868`, worse
than the original generic score `0.8727`, so rollout fitting again uses the latter.

Code inspection identifies an observability defect in
`transcription_solver.solve`: `snapshot` can silently return on nonfinite values
or a parameter-bound violation exceeding `1e-10`, while the native result still
reports optimizer success. The archive does not preserve the rejected final
values or rejection reason, so it does **not** establish which guard fired.
IPOPT bound relaxation is a hypothesis, not a measured explanation. Do not
attribute the missing endpoint to a failed defect test or silently loosen domains.

## Lower current loss is not enough to choose an optimization start

The medium-mesh arm replaces the original generic pair with a collocation pair
that has better *actual rollout* training NMSE. That selection is honest, but
subsequent recovery is worse:

| Start | Original generic NMSE | Selected medium checkpoint NMSE | Medium then rollout, final train | Original then rollout, final train |
|---:|---:|---:|---:|---:|
| 0 | 1.22514 | 0.84577 | 0.65611 | 3.337e-9 |
| 1 | 1.59213 | 0.97029 | 0.85529 | 1.548e-9 |
| 2 | 0.87267 | 0.24891 | 0.24891 | 0.12401 |

This directly supports the user's concern about discarding promising starting
pairs based only on present NMSE. A best deployable incumbent and a promising
optimization start serve different purposes. It does not prove a particular
local minimum: time allocations, numerical conditioning and evaluation cost
also differ. All parameter/initial pairs stayed intact; no blocks were mixed.

For medium start 2, the first sensitivity evaluation exhausts its 30-second
per-point allowance. Refinement stops with `point_allowance_exhausted` even though
its stage had 555 seconds available. The selected collocation pair is retained;
the whole fitting task uses only 661 seconds. This is a local derivative-cost
failure, not exhaustion of the full 1,200-second campaign allowance. A completed
ordinary rollout screen does not imply that sensitivity integration is affordable.

## The difficult original start

Rollout-only start 2 improves from about `0.873` to `0.124`, but later progress is
slow: best NMSE is `0.124156` at call 46 and `0.124009` at call 80. The progressive
arm follows the same path for 61 calls, having spent part of its budget elsewhere.
Both retain `output_gain_1` at approximately `1.39e-17`, its zero lower boundary;
the post-fit reference value is about `0.4612`. Several other parameters and
latent initials compensate and remain far from their references.

This is a poor boundary solution under the tested optimization and budget. It
does not prove unidentifiability, structural infeasibility, or a local/global
optimum. Two other starts recover both coefficients and initials accurately.
The poor scores also cannot be explained by disagreement between the two final
integration solvers. Increasing mesh density alone has not addressed this issue.

## Recommended next milestone, not yet implemented

1. **Make endpoint rejection explicit.** Persist the raw final parameter values,
   bound/nonfinite diagnostics and checkpoint disposition separately from native
   optimizer status. Preserve eligible final nodes and report why any endpoint
   cannot transfer. Test both successful saving and deliberate rejection. Never
   silently treat native convergence as successful checkpoint publication.
2. **Preserve several optimization starts under one budget.** Keep the original
   generic pair, a bounded number of distinct screened collocation pairs and
   reproducible generic perturbations. Give candidates short optimization trials
   before allocating the remainder using training loss, improvement and measured
   cost. Always retain the best verified incumbent. Do not start by replacing the
   original pair merely because another pair currently has a lower NMSE.
3. **Handle expensive derivative trials locally.** A point timeout should trigger
   a recorded bounded alternative or a return to another start, rather than end
   all fitting while budget remains. Reuse the existing shared budget and cache;
   do not label unavailable derivatives as evidence against the equation model.

Keep the three original starts and rollout-only control in the next comparison.
Report recovery, coefficients/initials and total cost, including unsuccessful
trials. Freeze restart generation and allocation using training evidence only;
do not inject reference values, use validation to choose starts, or retune only
for the known failed seed. First isolate checkpoint reliability and start
allocation; a broader solver/mesh search would confound this diagnosis.

No algorithm, benchmark, production default or historical artifact is changed by
this review. It records negative and positive outcomes for follow-up; generic
initialization robustness is still an open milestone.

## Review verification

The frozen portable bundle's Python source identity matches the returned plan;
the inspected fitting solver/controller/worker files also match that bundle.
The current checkout includes a later construction commit, so its whole-source
identity differs. An old local smoke correctly refuses resume under that changed
identity. A fresh linear smoke passes all three arms and exact terminal resume;
its progressive arm reaches three meshes. This small-case success does not
exercise the hard-case missing-endpoint failure.

Focused `pytest` passes 38 tests covering generic recovery, mesh transfer and
screening replay (one existing SciPy warning). Reviewed fitting files pass Ruff.
`ruff check .` still reports 37 pre-existing findings in untracked
`analysis/claude` files, which this documentation review leaves unchanged.
`git diff --check` passes. Only this report and the fitting-plan/Phase-C entry
links are changed; archived outputs remain uncommitted.
