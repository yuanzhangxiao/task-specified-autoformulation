# M32 results: piecewise support works; CSTR and portable verification remain open

Reviewed `review-20261014-014403.tar.gz` on October 13, 2026 (Hawaii).
The recorded qualification **passes**: all 23 outcomes are accounted for, all
12 diagnostic controls behave as expected, and all five previously accurate
outcomes retain verified accuracy. This is a diagnostic milestone, not a new fit
or a demonstration of improved constructed equations.

## Provenance and verification

- Implementation: `4627433629b2c283cc5773a68471f7537aa7d35d`;
  [M32 protocol and commands](PHASE_C_FITTING_STAGED_FEEDBACK.md).
- Plan SHA256: `44c2867ca2f9ae9987cd5856d29288defa62961e3ec39b45ae8465a6d6bfc3e1`.
- Source SHA256: `c79a763193068a0a09ea7479a69fefad2277d2190daf3a92fe4567cbdbf8ec55`.
- Inputs SHA256: `67d115820a3c8ebb10f69db7c0f468a46a64f96c2414bed3a9355bde27b520dc`.
- Archive SHA256: `58714a0a78fa5a8bfc4f8a68ff20d37425c50fc716f901b8ff32f942c441b08c`.
- Delta prepare / array / report: `22878682` / `22878683` / `22878684`.
- Output: `/work/hdd/bibo/yxiao2/phase_c/fitting-staged-feedback-v1`.
- Recorded runtime: Python 3.12.13; NumPy 2.3.5, SciPy 1.16.3,
  CasADi 3.7.2, Pandas 2.3.3, Pydantic 2.12.5.

All 939 sealed JSON hashes validate. Frozen delivery source and imported inputs
match the experiment. Twenty endpoints have verified independent output replay;
one intentionally skips diagnostics because it has no retained endpoint. Two
unavailable sources remain in the denominator. No validation/test observations or
live LLM calls were used. All sixteen prior outcomes remain frozen; only seven
new tiny diagnostic fixtures receive fitting work.

Local report reconstruction found a portability defect: `completed()` passes
13 of 23 rows but rejects ten because it requires exact equality of recomputed
SVD-derived floating-point evidence. The sealed source records remain intact.
Inspection of all reconstructed packets finds unchanged diagnoses, actions,
ranks, statuses, costs and identities. The 111 differing numeric leaves concern
SVD results and arithmetic involving their weak directions. Consequently, exact
whole-report reproduction on this local runtime is **not** claimed.

For example, alien's condition number is 1078.132126137753 in the archive and
1078.1321261377511 locally. In nearly singular cases the differences are more than
last-bit arithmetic: a T2 candidate's condition number changes from 5.94327e19 to
5.96925e19, and the chosen weak initializer direction rotates within a poorly
determined subspace. The ambiguous synthetic control's condition number changes
from 3.81186e16 to 3.42168e16; both diagnoses remain weak sensitivity. The local
Python is 3.12.10; listed package versions agree, but BLAS/platform identity was
not recorded. These differences do not justify loosening artifact hashes or
silently replacing the direction used to conduct an experiment.

The 63 M32 regression tests pass again during review. `git diff --check` passes;
repository-wide Ruff still reports the 37 unrelated findings in `analysis/claude`.
No numerical fitting or integration of the archived models was rerun locally.
Only this review and the Phase C handoff status are changed in the checkout.

## Main outcomes

The NMSEs below are training-only at unchanged retained parameters.

| Known-structure control | Training NMSE | M31 conclusion | M32 conclusion | New diagnostic duration |
|---|---:|---|---|---:|
| Alien-device hard | 8.36e-15 | New diagnostic timed out | Accurate, local sensitivity evidence available | 287.73 s |
| Coupled basin | 1.56e-16 | Piecewise derivatives unqualified | Accurate, piecewise sensitivity supported | 359.45 s |
| CSTR hard | 3.39e-13 | Piecewise derivatives unqualified | Accurate, joint parameter assessment incomplete | 1,869.64 s |

Alien's Radau output replay now completes in 243.93 seconds under its separate
360-second allowance, followed by a 26.23-second joint sensitivity calculation.
Its joint coefficient/initializer Jacobian has rank 18/18 and scaled condition
number about 1078. The result provides local evidence, not calibrated confidence
or exclusion of remote equivalent parameter sets.

Basin qualifies both coefficient axes and the weakest joint direction at both
step sizes, both perturbation signs and with both independent solvers. Its
Jacobian has rank 2/2 and condition number 7.63. Sampled branch activity includes
switches and ties, so this example goes beyond a model whose guards are all
inactive. Piecewise vector fields can have useful local solution sensitivities;
the nearby-rollout checks support that claim here. They do not prove classical
differentiability or certify behavior between stored samples.

CSTR verifies its output and obtains a descriptive rank-9/9 Jacobian, condition
number 584.71. All **seven coefficient axes pass** their perturbation checks.
The two hidden-initial-value axes and weakest joint direction remain unqualified
because the 1,800-second aggregate perturbation budget is exhausted. Of forty
required perturbed points, 29 have both solver evaluations. Across the task,
62 of 63 started evaluation calls complete; one Radau call is limited to the
remaining 31.89 seconds and stops during trajectory `train_009`.

This is a computation-availability limitation, not observed derivative failure
or evidence of unidentifiability. Sampled CSTR states stay away from its guards:
`C >= 0.15354` and `T - 250 >= 99.86068`, with no sampled switches or ties.
Those samples alone cannot certify the entire continuous trajectory or all
perturbations. The joint conclusion appropriately remains `accurate_unresolved`.

## Controlled diagnoses and unchanged constructions

All twelve synthetic control checks pass: five carried from M31 and seven new.
Inactive guards, clean branch crossings, and a guarded initial map receive
supported local evidence. An actual kink and a bound-limited perturbation
withhold it. The genuine retained-point stationary poor fit receives
`stationary_poor_fit`, while the ambiguity and insufficient-search controls keep
their distinct diagnoses. These are intentionally designed controls, not twelve
independent benchmark recovery successes.

The eight archived construction cells are unchanged: four `search_incomplete`,
one `solved_restricted_problem_poor_fit`, one `rollout_unverified` without a
retained endpoint, and two `source_unavailable`. The seven overall
`accurate_local_evidence` packets consist of the alien/basin known-structure
controls and five synthetic fixtures. No newly successful constructed model is
established here. The two T1 output replays remain verified, but their stricter
sensitivity gates remain unverified.

The aggregate qualification can pass while CSTR's parameter assessment remains
unfinished: its predeclared gates require verified outputs, preserved accuracy,
correct synthetic-control behavior, packet boundaries and complete cost
accounting. They do not require stronger parameter claims for every real model.

## Cost and next milestone

New diagnostics total **2,687.22 seconds** of task duration (44.79 minutes),
2,391.20 CPU seconds, and 214 started / 213 completed evaluation calls. These
are sums across tasks, not campaign elapsed time or queue time. CSTR accounts for
about 70% of diagnostic duration; its Radau calls use 1,559.10 seconds. The seven
new synthetic fixtures add a separate 22.21 seconds of fitting, 72 completed
search calls. Historical fitting and M31 diagnostic costs are not new work.

Recommended next steps, without refitting these models:

1. Make diagnostic inspection portable. Keep exact hashes and point/source
   identities, freeze the actually executed perturbation plan, and validate
   recomputed numerical evidence using declared tolerances and subspace-aware
   checks. A numerically arbitrary weak vector must not be regenerated as a new
   experimental input during report reconstruction. Record linear algebra
   environment details and test both benign variation and tampered artifacts.
2. Complete CSTR in an explicitly linked supplemental diagnostic with a new,
   recorded incremental budget. Reuse successful sealed evaluations; preserve
   the original exhausted attempt and charge. Twenty-one further solver calls
   are required if all completed calls are reused. Keep the same fitted vector,
   perturbation points, thresholds and data. Do not overwrite M32 or reset its
   budget under the guise of resume.
3. Before applying this to every search candidate, qualify a cost-aware schedule:
   test hidden initials and the weak joint direction early, and reserve the
   exhaustive assessment for retained endpoints. Measure any runtime/setup reuse;
   Radau integration currently dominates, so construction caching alone should
   not be assumed to solve the cost problem.

Bound-aware one-sided diagnostics and profile likelihood remain later work.
Neither a local sensitivity pass nor accurate predictions certify scientific
mechanisms, global parameter uniqueness, or a probability of correctness.
