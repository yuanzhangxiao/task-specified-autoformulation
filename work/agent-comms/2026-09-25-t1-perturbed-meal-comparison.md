# Frozen comparison: perturbed-obfuscated T1 meal interventions

Both Full round-12 seeds outperform Sol repetitions 0 and 1 on all eight physical meal probes. Sol repetition 2 outperforms both Full seeds on every one of those probes. This supports a comparison of robustness across the available repetitions, not superiority over the best Sol endpoint. No outcome was omitted or used to refit a model.

## Frozen design and scope

The preceding equation audit chose the cell, five endpoints and schedule grid before these new outcomes were generated. The identity is `65ada007815ce46c842ba185ff30349d43e4faef3ac8e458180f6dc3eaf489da`. This is an exploratory, post-hoc diagnostic selected after equation inspection, not an untouched benchmark test or proof of a mechanism's causal contribution to accuracy.

- Both historical Full R12 seeds in `phase_b_anonymous_system_t1_perturbed_obfuscated_easy`.
- All three frozen GPT-5.6 Sol repetitions in exactly the same cell.
- Single meals of 30, 60 and 120 g at minute 60.
- Total 60 g, split into 30+30 g with gaps of 15, 30, 60 and 120 minutes.
- Two 60 g meals at minutes 60 and 540, observed through minute 900.
- Fasting controls at horizons 300 and 900, and a single 60 g long-horizon control.
- A separate conditional 60 g pulse with fasting auxiliary histories held fixed.

All fitted parameters and causal initializers remain frozen. The trusted `perturbed_b1` reference generator regenerates target and all permitted auxiliary channels jointly for physical schedules. The models retain their public input representation: linearly interpolated one-minute meal pulses. The reference uses exact stomach jumps. Neither representation was tuned or replaced for this comparison. The conditional probe deliberately holds auxiliaries fixed and has no physical reference accuracy score.

No training, fitting, pruning, proposer calls, remote sessions or benchmark test-trajectory access occurred. Reading the existing physical generator is explicitly part of this diagnostic; its equations were not provided to a proposer.

## All physical meal results

These are mean squared target errors divided by the original public perturbed-T1 training variance, using training standard deviation 34.4731605935. The public named channel is an exact physical alias of the anonymous target. Each row uses its entire declared horizon; no time-window or endpoint selection was performed.

| Schedule | Full seed 0 | Full seed 1 | Sol 0 | Sol 1 | Sol 2 |
|---|---:|---:|---:|---:|---:|
| 30 g | 0.16942 | 0.10273 | 47,529.65 | 28,051.41 | **0.04349** |
| 60 g | 0.27666 | 0.21138 | 55,752.34 | 42,416.14 | **0.05042** |
| 120 g | 0.78394 | 0.82308 | 79,198.67 | 60,656.30 | **0.17737** |
| 30+30 g, 15-minute gap | 0.18863 | 0.14326 | 57,205.62 | 40,779.37 | **0.01846** |
| 30+30 g, 30-minute gap | 0.16301 | 0.12604 | 56,870.69 | 37,073.25 | **0.01525** |
| 30+30 g, 60-minute gap | 0.16173 | 0.12376 | 48,807.16 | 30,569.65 | **0.04220** |
| 30+30 g, 120-minute gap | 0.15991 | 0.11366 | 47,739.53 | 28,231.19 | **0.08099** |
| 60+60 g, 480-minute gap, horizon 900 | 0.27473 | 0.45562 | 1,485,400.09 | 1,396,856.08 | **0.03025** |

Median NMSE across these eight probes: Full seed 0 = 0.17903; Full seed 1 = 0.13465; Sol 0 = 56,311.52; Sol 1 = 38,926.31; Sol 2 = 0.04285. These are medians over this probe grid, not benchmark medians. Sol 0/1 predict negative glucose on all eight physical meal probes. Their finite numerical rollouts are not physically valid trajectories.

For completeness, control NMSEs are:

| Control | Full seed 0 | Full seed 1 | Sol 0 | Sol 1 | Sol 2 |
|---|---:|---:|---:|---:|---:|
| Fasting, 300 min | 0.02881 | 0.00885 | 39,914.07 | solver-sensitive | approximately 0 |
| Fasting, 900 min | 0.30872 | 0.46095 | 1,379,563.86 | solver-sensitive | approximately 0 |
| Single 60 g, 900 min | 0.34605 | 0.32574 | 1,484,423.64 | 1,395,908.26 | **0.01765** |

## Response fidelity, rather than baseline offsets

Subtracting each model's own matched fasting trajectory does not reverse the main comparison: Sol 2 has smaller squared meal-response error than either Full seed on all eight physical probes. For the 60 g single meal, relative squared response error is 0.16287 / 0.17780 / 0.04841 for Full 0 / Full 1 / Sol 2. For the 30-minute split it is 0.09208 / 0.10594 / 0.01418. Full's worse accuracy is therefore not just an initial offset.

The narrower difference between a split schedule and a single meal gives Full a smaller error than Sol 2 for gaps 15 and 30 minutes. However, those Full relative squared errors are 3.46/3.60 and 1.22/1.30 respectively: all exceed 1, meaning a zero predicted schedule effect would do better. Sol 2's corresponding errors are 4.91 and 1.75. This is not a convincing positive demonstration and must not substitute for the substantially better Sol 2 absolute trajectories. All schedule-difference metrics remain in the data.

Fasting-relative Sol 1 scores are not numerically reliable because the control itself is solver-sensitive. `verified-metrics.csv` withholds them instead of presenting the primary solver's near-perfect fasting solution as robust. Sol 1's physical meal errors and its comparison against the single-meal long control agree between solvers.

## What the equations explain

The conditional input probe confirms the structural contrast:

- Full 0 and Full 1 have nonnegative, delayed incremental meal responses, peaking at about 76.47 and 79.60 mg/kg.
- Sol 0's conditional response reaches about -44.64 mg/kg. This matches the negative input derivative found in its equation. Tiny positive differences near zero are numerical noise, not a positive pathway.
- Sol 1's response grows to about 7,972 mg/kg by minute 300. Its positive local target feedback amplifies a small input perturbation. Its fasting solution is also sensitive to numerical perturbations near the equilibrium.
- Sol 2 has a positive delayed conditional response, peaking at about 80.52 mg/kg. Its stable three-stage meal pathway and negative target self-coupling are scientifically credible here.

Direction alone does not fix gain and timing. In the physical 60 g schedule, the reference peaks at minute 153 with glucose mass 229.41 mg/kg. Full 0 peaks at minute 116 and 255.18; Full 1 at minute 119 and 251.16; Sol 2 at minute 127 and 236.00. All are early, but Full overshoots more. Its simpler fitted filters and remaining initial transients do not reproduce the reference's gastric/intestinal response accurately.

Long washout also reveals incorrect baseline regulation. At minute 900 of fasting, the reference remains 172.58 mg/kg, while Full 0 is 139.04 and Full 1 is 212.05. Stable eigenvalues do not imply relaxation to the correct physiological baseline. The previously identified very slow latent components matter even without another meal. Sol 2 remains about 172.59.

Sol 2's negative tissue-glucose coefficient was not treated as a sign mistake: the perturbed tissue-return law has a negative local slope near baseline, even though its flux is positive. This distinction between flux direction and differential sensitivity was material to identifying it as a credible competitor.

These results explain why the historical median comparison can favor Full while a particular Sol repetition is better. They do not establish that memory alone causes the advantage over Sol 0/1, or that Full recovered the full physiological system.

## Numerical and implementation verification

- All 55 physical/control rollouts and five conditional rollouts completed with frozen parameters.
- Independent Radau and DOP853 physical references agree across all 11 schedules, maximum discrepancy 1.37e-8 over targets and supplied auxiliaries.
- All 60 model rollouts were independently replayed with tighter DOP853 tolerances. 58 agree with primary Radau at relative tolerance 1e-5 and absolute tolerance 1e-4. The two exceptions are Sol 1 fasting controls. The 900-minute control differs by approximately 44,949 mg/kg. All 40 primary meal-probe comparisons agree, so the reported physical-meal ranking is unaffected.
- The frozen run resumed to an identical summary hash, `47d5c22fac25834d4ce491af7169df48ecce3dce4c9a50b06a17e2138135cad0`.
- 15 focused tests passed, including independent-reference checks, exact resume, missing-model rejection, separation of conditional and physical scores, and propagation of control unreliability to paired metrics.
- The full suite was stopped after 6m40s with 825 passed and 5 dependency skips; it did not complete. All changed Python files pass Ruff. Whole-repository Ruff still reports 37 pre-existing findings under `analysis/claude`.
- New files are three standalone diagnostic/plot/verification scripts and one test file. Production fitting, benchmark data and prompts were not changed.

## Artifacts and reproduction

All local outputs are in `artifacts/t1-perturbed-meal-comparison-2026-09-25/`:

- `plan.json`: exact models, parameters, source identity and frozen schedule/metric design.
- `curves.csv`: all physical curves and matched-fasting response curves.
- `metrics.csv`: raw primary-solver metrics; `verified-metrics.csv`: paired-score reliability and qualified metrics.
- `references/`, `replays/`, `conditional/`, `verification-replays/`: checkpointed trajectories.
- `summary.json`, `solver-audit.json`: full results and independent numerical checks.
- `figures/absolute-all.*` and `relative-all.*`: all models and relevant controls, including outliers, on a symmetric-log axis.
- `figures/absolute-focus.*` and `relative-focus.*`: linear-axis views of both Full seeds and Sol 2, explicitly labeled; other repetitions are in the all-model views.
- `figures/conditional-response.*`: the separate fixed-auxiliary mechanism probe.

From the project checkout, the evaluation can be reproduced or resumed with the existing frozen local inputs:

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
AF_OUT=artifacts/t1-perturbed-meal-comparison-2026-09-25
.venv/bin/python scripts/compare_t1_perturbed_meals.py freeze \
  --models artifacts/t1-equation-comparison-2026-09-25/models.json \
  --public-plan artifacts/t1-external-intervention-replay-2026-09-25-v1/plan.json \
  --root "$AF_OUT"
.venv/bin/python scripts/compare_t1_perturbed_meals.py run --root "$AF_OUT"
.venv/bin/python scripts/verify_t1_perturbed_meals.py --root "$AF_OUT"
.venv/bin/python scripts/plot_t1_perturbed_meals.py --root "$AF_OUT"
```

Model and numerical-runtime identities are checked on resume. The plotting bundle is portable data for Claude; replay under a different runtime should be separately identified rather than bypassing those checks.
