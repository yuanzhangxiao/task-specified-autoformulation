# T2-easy breadth results and fitting handoff for Orion

Start with **`c15-4`**, the fitted Candidate 15 descendant with an additive basal insulin secretion term. At the original fitted parameters and the two new parameter guesses, it moves the insulin derivative inward at the recorded depletion point and retains the parent’s pooled training NMSE to six decimal places. Its insulin peak is still early. All five admitted, distinct model packets are available for matched fitting and controls.

## What the two ACES jobs produced

Job `2320453` used GPT-OSS-120B on two H100s and incorporated 18 calls from job `2320276`. Both fitted-parent arms requested five alternatives in one call each. The fresh arm used five independent topology seeds, 100–104. Admission counts give the number of distinct branches that had passed the implemented checks by each cutoff.

| Cutoff | Cumulative calls | Candidate 15 | Candidate 21 | Fresh | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial proposals | 7 | 4 | 0 | 0 | 4 |
| One correction allowed | 18 | 4 | 0 | 0 | 4 |
| Two corrections allowed | 29 | 4 | 1 | 0 | 5 |
| Three corrections allowed | 39 | 4 | 1 | 0 | 5 |

`c21-2` passed on call 22. The literal first 27 calls therefore contained all five representatives. Of 15 branches, five were eligible, nine rejected and one unavailable. The last correction of `c15-1` reached the generation limit after three replies had used unsupported inequalities in the equation for its new state. Usage was complete at 210,305 tokens. An offline replay regenerated all 126 branch, depth, reply, call-order and handoff records byte for byte. All five packets passed three public graph-path checks. Training consistency was marked deferred in all five and the Candidate 15 boundary-sign certificate was marked deferred in all four. The campaign did no optimization and used the public training split only.

## Matched numerical starting points

These are free rollouts on all 16 **training** trajectories. Freeze the saved parent coefficients and its fitted training roots, then insert the child’s supplied starting values for any additional parameters. Start measured states at their recorded values, apply the saved initialization graph and linearly interpolate the recorded meal and infusion. The main NMSE averages squared errors for `Gp`, `I` and `U`, each scaled by its pooled training standard deviation. The trajectory column applies the same scales, scores each trajectory separately and reports its median and unscaled MAD. A negative insulin sample falls below −0.0001 pmol/L on the one-minute grid from 0 to 300 minutes.

| Model | Pooled training NMSE | Trajectory NMSE, median [MAD] | Trajectories with negative insulin | Insulin rate at the parent’s depletion state (pmol/L/min) |
| --- | ---: | ---: | ---: | ---: |
| Candidate 15 constrained parent | 0.1655624 | 0.1086 [0.0707] | 0/16 | −0.801736 |
| `c15-0` | 30.68 | 24.94 [23.61] | 16/16 | −0.801736 |
| `c15-2` | 18,601 | 11,739 [6,888] | 0/16 | +1.500000 |
| `c15-3` | 83.83 | 58.25 [58.20] | 8/16 | −0.801736¹ |
| **`c15-4`** | **0.1655623** | **0.1086 [0.0707]** | **0/16** | **+0.698264** |
| Candidate 21 parent | 1.5521231 | 1.411 [0.520] | 0/16 | — |
| `c21-2` | 1,148.53 | 1,057.65 [270.70] | 0/16 | — |

¹ The construction receipt left the rate unresolved because its saved parent state has no coordinate for the child’s new state `M`. Its equation, zero initializer and the recorded 150 g meal determine that coordinate. At the supplied timescale its effect is negligible at minute 300.

At the supplied starts, **only `c15-4` kept all three targets nonnegative on all 16 minute grids**. Counting trajectories with at least one sample below −0.0001 in that target’s units, `c15-2` had negative glucose mass on 16/16, `c21-2` had negative total disposal on 16/16, and `c15-3` had negative glucose mass on 11/16 and negative disposal on 4/16 in addition to its 8/16 insulin failures. The saved proposed models remain eligible for refitting.

An original-equation DOP853 rollout and an independently assembled Radau rollout agreed for all seven models and all 16 trajectories. They reproduced Orion’s two stored parent NMSEs. These child values are starting-point diagnostics. Their achievable NMSEs still require fitting. Training roots were estimated previously from training trajectories. The future 8-minute prefix forecast is a separate evaluation.

## What each repair changes

- **`c15-4` adds a restoring secretion near depletion.** Its insulin equation adds $B[1-\tanh(I/\epsilon)]$. At the 150 g parent witness, $I=r=v=0$ and the remaining rate is $-0.801736$, so the new rate is $B-0.801736$. The saved $B=\epsilon=1.5$ yields $+0.698264$. At minute 300 the observed insulin is 22.620 pmol/L, the constrained parent reaches zero and the child predicts 1.082. Continuing each model from its own endpoint with zero meal and infusion gives −0.794 and +0.938 respectively at minute 301. The child stays positive on every sampled minute through 330 in this diagnostic extension. The 150 g insulin peak still occurs at minute 65, thirty minutes before the observed peak. On the three 90 g schedules (`train_003/006/007`), the parent and child both peak at 65/132/191 minutes, against observed 101/161/221. Fit $B$ and $\epsilon$ against positivity and timing. $B=0$ recovers the parent exactly. A positive $B$ alone cannot certify every boundary state.
- **`c15-0` clips the wrong term for this witness.** In the subtracted decay process it replaces $kI$ with $\max(kI,0)$. Both terms vanish at $I=0$ for any $k$. The signed action contribution producing the outward derivative is untouched. Clipping changes the equation away from zero and negative insulin occurs on all 16 replayed trajectories. Keep as a control for admission and fitting.
- **`c15-2` changes insulin production to a glucose sigmoid.** Its production becomes $P_{\max}\,\sigma(k(G_p-G_*))$, removing the signed action term from that process. The new threshold is 1.5 in the saved model units, far below the reference $G_p=240.85$. The sigmoid is saturated there. The inherited net insulin self-coefficient is positive at approximately $0.01826$ per minute. On the 150 g schedule the model reaches about 78,221 pmol/L at minute 300. Qualify scale-aware glucose threshold and parameter starts when testing this family.
- **`c15-3` introduces a meal-driven state.** It adds $g_1 g_M M$ to insulin production with $\dot M=k_r r-M/\tau_M$ and $M(0)=0$. The saved $\tau_M=1.5$ min leaves essentially no meal contribution at the late depletion point. Its lowest insulin sample is −287.8 pmol/L. The arm tests a meal timing mechanism with an exact nested control $g_M=0$. Check whether fitted timescales and gains improve timing while meeting the concentration checks.
- **`c21-2` makes the glucose source nonzero by adding disposal back.** Write $Q$ for the generated counterpart of the insulin-independent utilization `Uii` and $X$ for insulin action. The model has $U=Q+X$, $Q(0)=1$ on all 16 schedules and $\dot Q=q$ with one shared constant. At zero meal, the edited equation gives $\dot G_p+U=Q=1+qt$, hence $\dot G_p=-X$. It also changes the decay-memory equation from $\dot Z=X-Z$ to $\dot Z=X-k_ZZ$. The glucose gate had correctly rejected the preceding correction with zero net source. The admitted edit now gives a source independent of meal history at a fixed time. On the 90 g training schedule, the three consecutive post-ingestion windows imply mean net supplies 5.49, 7.10 and 6.50 from measured $\Delta G_p$ and trapezoidal integration of $U$. An affine $Q$ cannot rise and then fall over these equal windows. These estimates depend on how the measured disposal is integrated between samples. At the saved $q\approx0$ the child supplies about 1 across all windows. Preserve the branch as a glucose-balance and mechanism control.

## Where to spend the next proposer calls

All **20/20** fresh topology replies had unique component names and at least one differential state. Every reply omitted all three literal targets `Gp`, `I` and `U` from the component-name field. Seeds 100, 102 and 103 used category labels such as `states`, `processes` and `generated_quantities`. Seeds 101 and 104 used prefixed names. Seed 104 supplied both `state_Gp` and `target_Gp`. Merging these automatically would require a scientific choice. The prompt did say to use public names. Its repair feedback repeated a single generic instruction covering missing targets, duplicate names and state count. Ten of the 15 corrections repeated the preceding graph exactly when compared by component names, types, source sets and auxiliary mappings. All 15 changed some reply text. None of the five seeds reached formula construction.

For a targeted retest, give the proposer one row for each exact public output name, let it choose the type and sources, and display any missing names beside the names it submitted. Retest seeds 100–104 before comparing fresh construction with parent revision. Also diagnose unsupported comparison operators on their own. For the insulin gate, specialize the full rate at zero before giving up on a function such as $\max$, and distinguish a proven boundary-equivalent edit from a negative witness at one parameter start. Keep the numerical witness visible to the proposer.

## Numerical handoff

The original ACES package is `/scratch/group/p.nairr260351.000/u.yx126462/phase-c-t2-breadth-results-v2/inspection.tar.gz`. Archive SHA-256 is `5d19239a29a8dd282df627f62eaf48bf1251cf18666e97851eb3078540a8b17e`. Plan SHA-256 is `cefdfbe73bbcc57d1606f3d8e8833fe637aa17c1a447639f5b912ad878649a30`. The assessment used checkout `ec485a5`.

In the portable handoff, `returned/handoff/` contains all five original sealed problems and `returned/inputs.json` holds both parents, their saved parameters and training roots, and the exact training split. The ACES archive has the same files relative to its root. `fixed-training.json` gives each trajectory’s score, minimum, peak time, cross-solver discrepancy and the two 30-minute diagnostic continuations. `fixed_training.py` reproduces these figures with `PYTHONPATH=/path/to/autoformalism:/path/to/autoformalism/src` in the project environment. `verification.json` records the independent replay of the ACES inspection.

For each child, report the inherited start, best verified fitted result, target-specific NMSE, minima, inward/outward rate at depletion, and peak timing on the shifted schedules. For `c15-4`, include the nested zero-basal control and the 150 g post-horizon scenario. For `c21-2`, include the fasting and post-meal glucose-supply windows. Preserve all five branch identities. Document any unsupported solver or unverified search separately. Compare held-out forecasts only with frozen shared coefficients and a matched prefix policy.
