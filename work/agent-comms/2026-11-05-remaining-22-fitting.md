# Fitting the other 22 T2-easy parents for construction feedback

Candidate 15 established a practical way to distinguish an optimizer problem from a repeatable trajectory mismatch. The fitting workstream can now cover all 22 other parents while Astra works on 15. The common glucose-balance restriction already identifies one revision target. Each parent's fits can reveal which measured outputs it sacrifices, which discrepancies persist when the parameters move, and whether a repaired child improves on its own parent.

## Where to resume

| Parents | Available evidence | Next fitting operation |
| --- | --- | --- |
| 14 affine models: 00, 05–09, 11–12, 16, 18–22 | Verified shared coefficients and training initials. Each model admits a bounded conditional fit of its 32 hidden initial values. | Search the 7–10 remaining shared coordinates, re-solving the initial values for each evaluation. Start from the verified checkpoints, retain the incumbent, and use a distinct start when a poor fit plateaus. Give 12 a separate boundary check because its zero and near-zero decay fits tie. |
| 10 and 14 | Verified nonlinear fits. Their previous saturation scans stopped before a full outer iteration. | Resume from the saved fits. Optimize a small block of coefficients, rotate or release the held coordinates, then check the full objective and replay the original equations. |
| 03 | A finite training rollout agrees across two stiff solvers. The sensitivity check remains unresolved. | Qualify repeatable residual evaluations on this model's parameterization and run a derivative-free fit. Keep gradient-based conclusions pending. |
| 01, 02, 04, 13, 17 | The provided starts reproducibly encountered a near-zero denominator (first four) or runaway states (17). | Try a bounded, predetermined set of starts in the declared domains. Screen cheaply, then require full-training replay before starting coefficient optimization. If these starts fail, send the trajectory, time, and failure mechanism to construction. |

## What to return for each parent

Record the starting and retained training losses, per-output errors, the actual stopping reason, independent replay agreement, and whether distinct starts reproduced the residual. For successful fits, compare fasting, ingestion, and post-meal windows at fixed fitted parameters. Check forecasting separately by re-estimating the hidden initial values from the public eight-minute observation prefix. Use public training signals for feedback to the proposer.

A credible optimum for the 32 initial values conditional on fixed coefficients leaves the shared-parameter problem open. Likewise, a rollout failure at a supplied starting point supports feedback about that start. Track every parent, including cases that need numerical recovery, and give repaired children matched fitting effort against their parents.

The previous Delta launchers bind exact candidate lists, source results and budgets. Broader shared-coefficient fitting needs a fresh, checkpointed campaign intake. The candidate 15 direct-meal experiment is a separate workstream.

## First numerical pass on the 14 affine parents

I fitted the 32 training initial values for each parent while holding its shared coefficients fixed. The bounded least-squares calculation uses all 16 public training trajectories and the same three target channels as the saved fit. For 12 I used the verified positive-rate free-control checkpoint and retained its exact-zero fit as a separate control. The other 13 started from their verified portfolio checkpoints. All 14 conditional fits passed their projection checks and independent replay of the original equations. Candidate 18 needed a separately recorded 90-second replay allowance after its initial 30-second replay timed out.

| Parent | Previous training NMSE | Replayed training NMSE |
| --- | ---: | ---: |
| 00 | 8.40325 | 8.40111 |
| 11 | 8.23241 | 7.15231 |
| 18 | 7.69364 | 7.60496 |
| 20 | 2.36088 | 2.25695 |
| 22 | 7.80937 | 6.59229 |

Seven of 14 improvements exceeded the existing endpoint comparison tolerance. Besides the five above, 07 and 16 improved by less than 0.00001. The other seven tied within numerical resolution. For 11, glucose NMSE improved from 18.51 to 15.08 while utilization worsened from 4.34 to 4.63. For 21, the conditional solve reproduced 1.55212, with glucose, insulin, and disposal NMSE of 1.00, 0.67, and 2.98. These examples show why each feedback packet needs per-target errors and why the shared coefficients still warrant search.

The training-root pass, 14 sets of saved parameters, replay receipts and final comparisons are available in the local review directory `output/construction-remaining-parents-preflight-20261105/`. `verified-results.json` includes the separately verified candidate 18 result. Operations were journaled in a new directory. The public observation-prefix forecasts and the new shared-coefficient searches are pending. The 54 root-profiling and termination regression tests passed.
