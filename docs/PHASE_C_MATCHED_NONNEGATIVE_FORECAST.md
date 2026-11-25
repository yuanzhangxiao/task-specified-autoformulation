# Refitting the matched graphs with nonnegative predictions

## Comparison

The experiment reuses all 96 formulas from the [matched topology-to-forecast comparison](PHASE_C_MATCHED_TOPOLOGY_FORECAST.md). The formula, graph, variable inventory, initializer, original coefficient guess, rank-specific perturbation, 16 training events and validation release are identical for each corresponding slot. Both arms have 24 graphs and two formulas per graph. The experiment starts cold and keeps the two original starts, 1,000-evaluation limit and 180-second limit per formula. The earlier fitted vectors never enter optimization or selection.

The original fitter minimized normalized rollout error by nonlinear least squares. The new fitter minimizes the same loss with SLSQP under the inequality

```math
\min_{t\in T_j}\hat y_{j,q}(t;\theta)\geq 0
\qquad (q\in\{G_p,I,U\},\ j=1,\ldots,16).
```

Here $T_j$ is the set of recorded times for training event $j$. A complete rollout supplies both the objective and all 48 inequalities. SLSQP can evaluate infeasible trial coefficients while searching. Retention requires a feasible training rollout at both two and four integration substeps and the same numerical resolution check as before. The numerical audit allows roundoff down to −10⁻¹⁰ in each target's units. The finer integration uses the same observation times. Physical positivity between observations, on future inputs or for hidden variables remains a separate question.

Training budgets use the same nominal caps. Each constrained check on the finer grid consumes one of the 1,000 evaluations. The original fitter charged its final grid verification outside that optimizer cap. Differences from the earlier result include the change of solver as well as the constraint. The head-to-head comparison of topology generators uses identical solver code and accounting.

The public T2-easy task explicitly marks plasma glucose mass nonnegative and describes insulin as a concentration and utilization as a disposal rate. The three variable inventories do not give insulin a structured nonnegativity declaration. This experiment declares all three observed targets nonnegative to match the earlier post-fit qualification rule. A future benchmark can supply its own list of physical bounds.

## Selection and scoring

The frozen prefit results still apply to the same equations. A graph qualifies when at least one formula has no demonstrated prefit conflict and attains a numerically verified, nonnegative fit on training. The earlier prefit checks admit formulas on at most 23/24 algorithm graphs and 18/24 repaired 20B graphs in this cohort. No coefficient refit can make a graph whose two formulas were ruled out by those prefit checks qualify in this particular experiment.

The primary comparison in this report selects the qualifying formula with lowest training NMSE within each inventory and arm. We also compute the original lowest-training-error comparison without the prefit filter. All training choices are sealed before development scoring. The four development events have no administered insulin. Only training event `train_015` has nonzero administered insulin. Scores cover glucose mass, insulin and disposal under the four development schedules. The test split stays closed.

## Results

All 96 optimizations returned a numerically verified nonnegative fit and used exactly 1,000 charged evaluations and two starts. All 48 training-selected graph finalists also stayed nonnegative at the four development events' observation times. The earlier formula checks determine the remaining training exclusions: both formulas in one algorithm graph and six 20B-plus-repair graphs had an identified insulin contradiction.

| T2-easy inventory | Qualifying graphs, algorithm / repaired 20B | Development NMSE of best qualifying training choice, algorithm / repaired 20B | Median development NMSE among qualifying graph finalists, algorithm / repaired 20B |
|---|---:|---:|---:|
| Seed 0 | 8/8 · 7/8 | 0.54507 · 0.86034 | 0.80433 · 0.99846 |
| Seed 1 | 8/8 · 3/8 | 0.51652 · 0.64731 | 0.87023 · 0.87517 |
| Seed 3 | 7/8 · 8/8 | 0.43784 · 0.58617 | 0.79358 · 0.86215 |

The algorithm arm went from 0 to 23 qualifying graphs. The repaired 20B arm went from 7 to 18. Among qualifying models, the algorithm has the lower development NMSE for each inventory's training winner. Its median is also lower in each inventory, with little separation in seed 1. If the deterministic prefit contradictions are disregarded and the winners selected on numerical performance alone, the algorithm wins seeds 0 and 3 and the repaired arm wins seed 1 (0.37253 vs 0.51652). The same three inventories share T2-easy data, and multiple graphs in an arm can descend from one original proposal.

For one example where the score corresponds to a visible difference, the measured insulin maximum of development event `val_001` was 364.47 pmol/L at minute 189. The eligible seed-0 algorithm winner reached 308.13 at minute 205. The eligible repaired 20B winner reached 162.65 at minute 96. The figure shows all four events so the comparison can also be inspected away from that peak.

**Optimization control.** Ten previously fitted models already had admissible coefficients. We replayed those saved vectors over both integration grids and confirmed all ten satisfy the new sampled training constraints. All ten cold refits had higher training NMSE than those known feasible vectors. For example, formula `042-0` previously attained 0.58683 and now attained 0.76223. The experiment establishes feasible fits within budget. It cannot attribute the increase in error relative to the old fit entirely to the physical constraint. These archived vectors were used only for this retrospective numerical check.

[Report with paired scores and insulin trajectories](../output/matched-nonnegative-forecast-20261124/REPORT.html) · [Exact comparison values](../output/matched-nonnegative-forecast-20261124/comparison-summary.json)

## Reproduce

Use the same local sealed public training intake, validation release and parent plan as the original experiment. A repeat at the same root verifies receipts and reuses completed fits. A reserved fit without a terminal receipt consumes its slot on recovery. Each worker owns a per-slot lock.

```bash
export PYTHONPATH=src:.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
.venv/bin/python -m scripts.matched_nonnegative_forecast run --workers 6
.venv/bin/python -m scripts.report_nonnegative_topology_forecast
```
