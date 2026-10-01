# Phase C fitting M2: identifiable-control results

Reviewed 2026-10-01. All 27 endpoints pass the frozen output, coefficient,
latent-initial and latent-trajectory recovery gates. This closes the M2
qualification on its three supplied controls. It does not qualify arbitrary
discovered models or resolve CSTR joint fitting.

## Provenance and integrity

- Protocol: `phase-c-identifiable-fitting-1`.
- Execution commit: `f620f2a24dc960d1bcb08e46cbf57b75a2ddf0b8`.
- Delta jobs: preparation `22594280`, fit array `22594281`, report `22594282`.
- Plan SHA256: `f13e60096491ecaa8cf5fd26647a43816844fa35180b0be4460cac0c0367d7e1`.
- Uploaded archive: `review-20261001-172440.tar.gz`, SHA256
  `decf3aa2c61e23c68c9d3a88dff78f513aa94de6d3ba47f06421c9d493bfff64`.
- Local review copy: `artifacts/phase-c-fitting-m2-review-decf3aa2c6`.

All 132 sealed artifacts verify. Regenerating the summary reproduces its contents
exactly. The source fingerprint matches the reviewed checkout. Each arm uses the
same frozen physical node guesses and shared collocation pool for its case/start;
the plan, input, shared-pool and backend identities match. Logged residual-call
files agree with every reported call count. All endpoints retain a training fit
at least as good as their best screened checkpoint. No new fitting, LLM calls,
test-data access or remote sessions were performed in this review.

## What was recovered

The supplied system is `y' = z`, `z' = -a*z - b*y + c*u - d*y^3`, with y observed
and z hidden. The linear control omits d. We fit the coefficients and a shared
unknown z(0), using three training trajectories, then freeze the fitted values
for two validation trajectories. The fitter receives neither true coefficients
nor hidden trajectories. The control's public preparation contract permits the
shared initial value; arbitrary trajectory-specific initialization is not tested.

Each arm succeeds on all three starts of the linear, nonlinear and fast/slow
controls. Both independent rollout solvers agree within the frozen tolerance.
Every endpoint passes all three gates: output NMSE <= 1e-6 on both splits,
maximum relative parameter error <= 1% including z(0), and latent NMSE <= 1e-4
on both splits. Actual errors are substantially below those limits.

| Refinement arm | Recovery | Median residual calls | Total residual calls | Median refinement seconds | Worst validation NMSE | Worst relative parameter error (%) |
|---|---:|---:|---:|---:|---:|---:|
| Joint | 9/9 | 7 | 74 | 18.04 | 1.92e-18 | 1.96e-7 |
| Joint + stopping | 9/9 | 6 | 60 | 8.48 | 6.12e-13 | 0.00350 |
| Conditional rescue + stopping | 9/9 | 39 | 331 | 78.37 | 3.88e-11 | 0.03044 |

Calls and seconds include screening and all refinement blocks, but exclude shared
collocation and final evaluator replay. The worst latent NMSE over all 27
endpoints and both splits is 9.22e-9. None exhausts its refinement budget.

## Where the improvement came from

**Collocation already provides excellent starts for six of the nine pairs.**
The linear and nonlinear initializers all converge. Their best checkpoint's
training rollout NMSE is approximately 2.15e-9 and 4.35--4.48e-8 respectively.
Joint refinement improves those solutions; the stopping arm needs only two
additional sensitivity/residual evaluations after screening in each of these
six cases. Repeated convergence to essentially identical parameter vectors is
expected here, not evidence of independent scientific replications.

**An unfinished collocation run can still be useful.** All three fast/slow
initializers time out. Their best saved checkpoint rollouts have training NMSE
7.86e-9, 1.89e-6 and 0.0161. All are successfully refined. The third case starts
joint refinement from approximately
`(a,b,c,d,z0)=(2.529,0.262,0.269,0.229,0.0883)`, compared with the evaluator's
reference `(18,2,2,0.5,0.4)`. Joint + stopping recovers it with nine refinement
evaluations after two screening calls; its independently replayed validation
NMSE is 1.53e-15. This is evidence that parameter recovery can follow an
unconverged, inaccurate checkpoint. Without a no-collocation comparison, it
does not establish that checkpoints were necessary for success.

**Training-based stopping saves work without losing recovery here.** All nine
ordinary joint fits terminate through the native gradient criterion. All nine
joint + stopping fits stop at the training-accuracy threshold, then pass the
independent evaluator checks. Total residual calls fall from 74 to 60 (18.9%).
Recorded refinement time falls from 164.87 to 105.07 seconds (36.3%). Wall-clock
variation is substantial, so call counts provide the cleaner paired evidence;
the median timing ratio is not a controlled end-to-end speedup. Adding each
pair's actual shared initializer elapsed time gives total attributed
initialization-plus-refinement time of 490.19 versus 430.39 seconds (12.2% less),
excluding evaluation. The initializer was physically run only once per pair.

**Always doing conditional rescue is expensive on these controls.** It makes
331 calls, 5.52 times the stopping-only joint arm, with no recovery-rate gain.
Every conditional endpoint still reaches its accuracy stop during final joint
refinement. The policy first spends work improving separate blocks at several
points, including poor starts, even when the best joint start is already good.
Its internal stages include four stagnation stops and seven point-allowance
timeouts; these do not prevent final recovery. This experiment tests a fixed
conditional-first policy, not every alternating method and not joint versus
alternating collocation. It does not refute the concern that a useful parameter
block can be hidden by a poor initial-state block. No case here requires such
rescue to pass.

## Remaining limitations and timing issue

These are three noiseless, two-state models with one latent state, supplied
correct equation structures, a fixed latent scale, four or five unknowns, and
informative training inputs. Their coefficients enter the vector field linearly,
although fitting through rollout is nonlinear. Intrinsically nonlinear shape
parameters, many hidden states, noise, model mismatch and richer initialization
maps remain unqualified. The mathematical identifiability argument concerns
ideal continuous observations under the excitation condition; numerical local
rank is not a general finite-sample or noise-robustness guarantee.

Unlike M1's guarded CSTR expressions, these controls permit smooth forward
sensitivities throughout. Success here does not isolate scaling, checkpointing,
the Jacobian or problem simplicity as the cause of the difference from M1.

The fast/slow initializer wrapper records 72.11, 81.71 and 83.66 seconds despite
a requested 60-second limit. The saved last-progress times are 52.99, 34.97 and
2.16 seconds, respectively. These logs do not distinguish delayed process
termination, scheduling or storage overhead, and do not establish extra useful
solver work after the deadline. Report actual elapsed time, not the 60-second
accounting allowance, when describing measured cost. Shared pools keep the arm
comparison matched, but timeout lifecycle timing needs instrumentation before
claiming a strict end-to-end wall-clock cap.

## Recommended next milestone

Use joint refinement with training-based stopping as the leading candidate.
Retain conditional rescue as a bounded fallback hypothesis after joint progress
stalls; do not yet promote it or change production defaults.

The next useful bridge is back to the difficult joint CSTR control: verify
sensitivities for its guarded expressions against numerical derivatives in
appropriate smooth regions, explicitly handle unreliable derivatives near
switches, and compare sensitivity-based refinement with the existing polling
refinement from identical frozen pools under equal budgets. Keep observed-error,
parameter/initial recovery and latent recovery separate. Known-block controls
remain evaluator diagnostics. CSTR has encouraging local rank evidence at the
reference point, but global identifiability is not established.

An additional identifiable control with a nonlinear shape parameter would help
isolate that difficulty if the bridge fails. Do not start by increasing budgets
or adding an unconditional rescue stage. This is a recommendation, not an
implemented or submitted follow-up.

## Review verification

- Artifact/hash, paired-input, call-accounting and incumbent checks above passed.
- Delta preparation reports 19 focused tests passed; local rerun also reports
  19 passed (4.02 seconds).
- Repository-wide `ruff check .` still reports the same 37 findings in unrelated
  `analysis/claude` files. No implementation files changed in this review.
- Summary regeneration is the artifact reporting smoke check. The full suite
  and numerical campaign were not rerun for this documentation-only update.
