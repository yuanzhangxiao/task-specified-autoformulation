# Three T2-easy insulin-recovery equations for Orion

ACES job `2385821` ran GPT-OSS-20B on two fitted parents. It returned 40 first attempts and spent all 120 corrections. The live editor admitted zero branches. A retrospective replay recovered three distinct, executable revisions of `c15-2` from two sampling seeds. All three compile in the T2 fitting interface and have reproducible starting vectors. Their improvements after optimization are unknown.

## What to fit

Use `orion-fitting-handoff.zip`. Its `manifest.json` lists the three candidate files. Each contains the full `PrefixFitProblem`, the original patch, the decoded patch, the exact fields changed in decoding, the checks, the source receipt hashes, and a tested starting vector including all training initial values. `parent-and-training.json` supplies the fitted parent, its original coefficients and roots, all 16 public training trajectories, and the original forecast policy.

Write \(X\) for the insulin-action state, \(G_*\) for the parent's fitted glucose threshold, and \(D=X(G_p-G_*)\). The entries below are additions to the parent's right-hand sides. The two gains \(a,b\) are positive. In seed 116, \(c\) is nonnegative, \(s\) is positive and \(C\) is a new nonnegative threshold.

| File stem | Change to the insulin and action equations | Fixed-start pooled training NMSE | Fixed-start insulin change, min 240–300 (pmol/L) |
|---|---|---:|---:|
| Fitted parent `c15-2` | Reference | 0.129409 | +60.696 |
| `c15-2-seed109-initial` | \(\Delta\dot I=bD,\quad\Delta\dot X=+aD\) | 0.129411 | +60.732 |
| `c15-2-seed109-correction3` | \(\Delta\dot I=bD,\quad\Delta\dot X=-aD\) | 0.129408 | +60.660 |
| `c15-2-seed116-correction3` | \(\Delta\dot X=cI\,\mathrm{sigmoid}(s(G_p-C))\) | 0.129409 | +60.696 |

The recorded change is **+0.913 pmol/L** on the 150 g meal, `train_005`. At all 61 sampled minutes of that late window, the fitted parent had positive \(X\) and \(G_p<G_*\). At that same state, both seed-109 revisions reduce the instantaneous insulin rate. Their opposite effects on the action state propagate through glucose disposal and secretion. Seed 116 changes insulin through that indirect route.

Compare all three with the parent under matched training data, numerical budget, parameter domains and initialization policy. Report the absolute error in the late insulin change, pooled and per-target NMSE, shifted-90 g insulin peak times, target minima and the early insulin-infusion response. Keep the 16 training roots for each of the two latent states available to the optimizer under the original policy. Preserve observed \(I(0)\) and \(G_p(0)\). The numerical check used the supplied parent parameters and roots. It set \(a=b=10^{-8}\) for seed 109, and \(c=0\) for seed 116. DOP853 and Radau independently reproduced the trajectories at all 301 sample times for all 16 records. These fixed starts are controls for refitting. The model–data checker still reports deferred checks.

## Provenance

The saved edits used the heading `equations` in the component field. Each recovered edit contained an explicit derivative on the left and the correct differential type. The replay resolved the identifier against the parent or an explicit declaration, kept each signed right-hand side and initializer, and reran the committed admission checks. Five qualifying replies reduce to the three equations in the table. A run stopping a branch at its first qualifying reply would have retained seed 109's initial model and seed 116's third correction. The alternate sign in seed 109's third correction is available because the live editor continued that branch. This distinguishes the retrospective pool from a prospective yield estimate.

The `c15-3` peak-timing arm produced zero strict admissions. Seed 103 suggested a second meal-driven reservoir and a term proportional to it in insulin production, but also called the existing algebraic production process differential. Changing that single type field to algebraic passes the structural checks in a separate diagnostic. This interpretation requires proposer confirmation and is outside the fitting packet.

The live constructor was commit `4f972345574780a3e6b699b289c53aacfb6bb87d`, plan `b7db21e307d3c3d8b4bb2ec81d5f8b6278914aa279df83eca5a0253c0b693f47`. The ACES inspection SHA-256 is `6de2c298c4a87e9146f93111c2758ceb695964abd09498e69b07e175b8232863`. Local report and fitting packet are in `output/t2-fitted-breadth-20b-results-20261119/`.
