# M10 review: screening restored, assisted dense collocation succeeds locally

## Main finding

The timing fix removes the previous blocker. All ten calibration points complete,
and all four eligible alien-device assisted fits complete both fixed and released
stages. Dense collocation preserves excellent prediction and coefficient accuracy
after releasing the parameters. Reduced collocation converges to a noticeably
biased parameter vector, even from the same accurate supplied solution.

The experiment is not a new generic-start recovery success. All four assisted
runs retain their supplied M7 training-fitted incumbent because its actual
training rollout remains better than the released collocation endpoint. Saved
generic candidate pools still contain no newly certified accurate model.

The supplied report is **incomplete**: 19 complete independent evaluations,
two explicitly incomplete evaluations, and one missing task record out of 22.
The missing task is a process-cleanup failure, not a pending optimization result.

## Calibration and saved candidate screening

The same three generic starts and two previously training-qualified fitted starts
were evaluated using both integrators on all 16 training trajectories. No new
parameters, benchmark arrays or equation structures were generated for calibration.

| Integrator | Complete calibrations | Observed wall time per point | Frozen allowance |
|---|---:|---:|---:|
| RK45 | 5/5 | 13.5--15.1 s | 60 s |
| Radau | 5/5 | 32.0--39.0 s | 88 s |

This confirms that M9's 20-second allowance was insufficient for even ordinary
Radau evaluations. Startup costs 3.39--4.53 seconds and measured post-import setup
costs 0.07--0.10 seconds in these calibration points. Integration and its associated
progress writes consume most of the remaining time. The two supplied good starts
recheck at training NMSE approximately `1e-15` and `1e-14` with both solvers.

Of 149 pool entries, deduplicated **within each pool**, 35 verified historical
scores are reused. There are 108 new attempted screens: 89 complete, 18 timed out,
and one left in `started` state by the cleanup exception. Six entries in that
failed pool are not attempted. The 17 finalized pools attempt all their uncached
entries under the frozen allowance. These are rescored saved M9 vectors, not new
NLP optimization runs.

| New generic screens | Complete | Timed out | Unfinalized |
|---|---:|---:|---:|
| RK45 | 15 | 17 | 1 |
| Radau | 74 | 1 | 0 |

The methods use their own original pools and different calibrated ceilings; these
counts are not a matched equal-time integrator comparison. They support retaining
an implicit solver option for difficult candidate rollouts. Timeouts alone do not
prove stiffness or scientific invalidity. Some occur late in the trajectory set.

Among the 15 generic pools with complete independent evaluation, the best training
and validation NMSE are `0.260379` and `0.304323`, from reduced Radau, start 0.
The two evaluation-timeout candidates have complete *screening* training scores
of `0.233917` (dense Radau, start 0) and `0.565127` (reduced Radau, start 2).
Neither is an accurate model under the frozen 0.01 prediction threshold. Their
aggregate independent train/validation results correctly remain unavailable.
Thirteen of 17 finalized pools change selected vectors, including nine Radau pools
that had no selected vector under the M9 cap; none passes coefficient recovery.

## Assisted experiments: inspect the released solution, not just the incumbent

Both supplied source vectors come from M7 fitting using training observations.
No true coefficients or reference latent trajectories initialize these solves.
Their fitted models are integrated to supply boundary and internal-node guesses.
The two starts are extremely close to the same solution, so they are not evidence
of robustness across two substantially different attraction regions.

Results after coefficients and shared latent initials are released:

| Mesh | Released variables | Native iterations | Training rollout NMSE | Largest coefficient relative error | Largest initial absolute error |
|---|---:|---:|---:|---:|---:|
| Dense, both starts | 115,218 | 6 | `7.844e-10` | `0.01277%` | `0.0003560` |
| Reduced, both starts | 33,906 | 11 | `5.886e-4` | `9.6797%` | `0.21945` |

Both fixed and released stages report `Solve_Succeeded` in every run. Dense fixed
stages take two iterations and released stages six; reduced stages take four and
eleven. Dense fixed/released worker wall times are 49--56 / 79--82 seconds;
reduced times are 23--28 / 31--37 seconds. Graph construction costs approximately
15 seconds for dense and 6 seconds for reduced. All supplied observations retain
their original weights, and the reduced grid preserves the supplied input
interpolation exactly. Only the state discretization and observation interpolation
through the collocation polynomial change.

The dense released solution meets the coefficient (1%) and initial (0.01 absolute)
tolerances and has excellent training rollout accuracy. The reduced solution still
predicts training trajectories reasonably well, but fails both parameter and
initial recovery tolerances. The largest dynamic error is in `decay_x2`.

**Released endpoints were screened on training only.** Independent train/validation
replay in this campaign evaluates the retained incumbent, not every rejected
released endpoint. We therefore do not report validation accuracy or independent
two-solver certification for the released solutions. All four retained incumbents
have independently verified train/validation NMSE below `1e-14`; this is inherited
M7 accuracy, not a new improvement by collocation.

## Why the reduced mesh shifts otherwise good coefficients

The fixed-parameter stage is especially informative: the accurate coefficients
and initials cannot move, so an increase in observation error cannot be blamed on
incorrect parameter search.

| Quantity, representative start 0 | Dense | Reduced |
|---|---:|---:|
| Nodal observation NMSE before fixed solve | `1.265e-15` | `6.264e-6` |
| Maximum scaled discrete defect before fixed solve | `0.002673` | `0.45835` |
| Observation NMSE after fixed solve | `1.792e-9` | `0.001251` |
| Observation NMSE after releasing parameters | `1.008e-9` | `0.0004099` |

The fixed solves drive their discrete defects nearly to zero, but the reduced
grid moves its state trajectories away from the observations to satisfy those
discrete equations. Releasing parameters then improves that discrete objective by
moving coefficients and initials away from the accurate continuous-time fit.
Actual rollout at the released reduced vector also deteriorates.

This is strong evidence of a mesh/transcription mismatch. A systematic refinement
test is still needed to establish convergence and exclude an implementation issue
specific to reduced-grid interpolation. Small defects at the collocation nodes
alone are not an accuracy certificate: they are the equalities being enforced.
An adaptive indicator should examine between-node dynamics or compare with an
independent integrated trajectory.

Dense collocation's six-iteration success also corrects an overly pessimistic
interpretation of its variable count. The sparse 115,218-variable problem is
solvable quickly **locally with a good initialization**. This does not show that
generic starts will find that solution, nor that its memory cost disappears.
The generic-start difficulty and the reduced-grid accuracy problem are distinct.

## Incomplete records and operational follow-up

The missing entry is array index **6**, task
`pool_alien_hard_s0_fixed_reduced_collocation_bounded_rk45`, in job `22685223_6`.
The log records a 60-second point timeout followed by a second `TimeoutExpired`
when the killed child failed to exit within the two-second cleanup wait. That
second exception escaped, leaving no sealed backend or task result.

The durable progress record says all 16 trajectories completed by 54.23 seconds.
The oracle record also says `valid_full_training_evaluation: true`; its scalar cost
corresponds to NMSE `1.225138464`, agreeing with the ordinary-start calibration.
The final screening result was not published, and a zero-byte progress temporary
file remains. This is consistent with a publication/cleanup problem after numerical
work, but does not establish the underlying OS or storage cause. We do not silently
promote this partial protocol record into a completed campaign result.

Two other selected candidates exceed the 300-second independent replay ceiling.
In both, the explicit DOP853 cross-check times out despite successful Radau
screening. Partial scores remain diagnostic evidence, not aggregate validation.

Recommended operational correction: record cleanup failures as terminal outcomes,
preserve the last valid incumbent, and stop launching further work if child
termination cannot be confirmed. Close this missing record from existing logs
without refreshing its budget. The four assisted results need no rerun.

## Recommended scientific next step

Keep every fitting method and do not promote a production default yet.

1. **Test mesh convergence using the saved assisted solutions.** Compare a small
   refinement ladder and a transfer from reduced to denser grids, retaining all
   observations. Score actual training rollouts and coefficient/initial recovery,
   then independently evaluate frozen endpoints. Use between-node error to decide
   where refinement is needed; preserve the supplied fitted incumbent. This tests
   whether fewer variables can be retained without the observed parameter bias.
2. **Test initialization robustness separately.** Use bounded perturbations of
   training-fitted starts, with physical scales and bounds rather than true values,
   and compare rollout fitting with dense or qualified adaptive collocation under
   matched budgets. This measures how far the local successful region extends.
   It does not replace the generic-start comparison or count fitted assistance as
   a discovery from scratch.

These are proposed diagnostics, not newly implemented campaigns. More blanket NLP
time is not the first priority: all four assisted native solves already converge
well inside their allowances.

## Provenance and checks

- Archive SHA256: `9bbbe3d3d1527ae9e1def4f473c5e217a6d406123697709effc9638f06870ada`.
- Experiment commit: `9d4e5b940cc62be71225e95576ebc7c748c2ce24`.
- Plan SHA256: `ea0692603750012dec09c3f4c527bd93a65a296841eddd17ec4c92458dccb0e6`.
- Input SHA256: `774a2d766a64512edb93ed7d2c1346bc3f08d62882afd46b07e8f1cf439d51cb`.
- Source-code SHA256: `3feed5357efed1ccc66a146471f71db5bb13a7694e57315c11a0697e7c54bede`.
- Delta jobs: prepare `22685221`, calibration `22685222`, run array `22685223`,
  report `22685224`. One CPU per task, run concurrency two, no GPU or LLM call.
- All 92 sealed records verify. All 21 completed backend/result identities and
  replay journals reconcile. The summary reproduces exactly in a separate
  directory. Input export and experiment source identity match the frozen release.
- Coefficient diagnostics recompute from saved selected/released vectors. Every
  finalized selection equals the lowest complete training score, including cached
  incumbents. Every point payload matches its request, training arrays, method and
  vector digest. No partial rollout is counted as a complete screen.
- Maximum independent solver disagreement among the 19 complete evaluations:
  `1.397e-7`, below the `1e-4` agreement gate.
- Recorded new task time: 6,962.6 seconds; calibration: 251.3 seconds. These totals
  exclude independent evaluation time, upstream M7/M9 work and the unsealed failed
  task, so they are not end-to-end campaign cost.
- Local read-only audit: `artifacts/fitting-m10-review-analysis/audit.py` and
  `analysis.json`. No fitting or new hard-case integration was performed.
- Documentation-review verification: 18 focused screening/replay tests pass;
  the eight-case smoke test passes exact resume using the frozen M10 release.
  `ruff check .` reports 37 pre-existing findings in untracked `analysis/claude`
  files. No implementation files were changed in this review.
