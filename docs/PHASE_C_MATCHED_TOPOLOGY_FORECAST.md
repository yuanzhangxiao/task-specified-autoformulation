# From matched topologies to forecasts

We compared two ways of constructing a topology from the **same previously declared variables**. The first enumerates graphs and screens them against public requirements and training evidence. The second asks GPT-OSS-20B for a graph and applies deterministic repairs to its response. Both arms use saved outputs from the same topology study. The question is whether differences between their admitted graphs survive common equation construction, parameter fitting and forecast evaluation. This comparison holds the variables fixed. The [separate revision study](PHASE_C_ONE_ROUND_REVISION.md) starts from models that have already been fitted.

## Cohort

A graph must use exactly the registered variable names and differential/algebraic types. We deduplicate by structural identity within an inventory and generator. Before constructing equations, we order each pool by a fixed salted hash of its inventory and graph identity. We advance the smaller pool size, up to eight per arm, without replacement. No graph appears in both sampled arms. The 20B seeds refer to the original **inventory** or **proposal** as labeled. They are not independent draws of the forecast experiment.

| Saved inventory | Algorithm graphs available | Distinct 20B-plus-repair graphs with identical variables | Advanced per arm |
|---|---:|---:|---:|
| T2-easy, seed 0 | 8 | 39 | 8 |
| T2-easy, seed 1 | 8 | 34 | 8 |
| T2-easy, seed 3 | 8 | 11 | 8 |
| T2-easy, seed 102 | 8 | 0 | 0 |
| Coupled basin, seed 0 | 8 | 0 | 0 |

Seeds 0 and 3 model utilization `U` as a differential state. Seed 3 also declares a meal-memory state alongside insulin action. Seed 1 models `U` algebraically and retains differential insulin action. Both origins inherit these exact choices for each comparison.

The algorithm counts refer to its archived eight-graph portfolios. The 20B counts include multiple deterministic repairs of one original proposal. For seed 1, 29 further structures introduce an algebraic variable and fall outside the fixed inventory. For seed 3, two proposer replies yield the same 11 structures. Coverage and forecast comparisons therefore have different denominators.

On the three matched inventories, the saved algorithm portfolios contain 24 admitted graphs from 32 graph checks. The 20B arm used nine graph proposals (19,995 observed tokens). Seven original replies contributed to the 24 selected repaired graphs. The deterministic repair search made 5,888 graph checks and returned 124 passing occurrences, including repeated structures and inventories with an extra variable. The three exact-inventory repair pools contain 39 + 34 + 11 = 84 distinct structures counted within inventory. These are measurements for the two implemented searches. A proposer call and a CPU graph check have different costs.

The selected graphs differ in ways the public task makes interpretable. Administered insulin directly enters the plasma insulin derivative in 24/24 repaired graphs and 9/24 algorithm graphs. EGP, excretion and utilization all enter the plasma glucose derivative directly in 24/24 repaired graphs and 1/24 algorithm graphs. The median number of edges on equation right-hand sides is 10 for 20B-plus-repair (range 9–11) and 12 for the algorithm (9–16). An edge can have a negligible fitted coefficient. The graph alone does not establish its sign, strength or physiological validity.

## Equations, fitting and selection

Both arms receive the same two-slot, trajectory-informed constructor. It compares additive affine terms, saturating terms and pairwise interactions where the training trajectories support the comparison. Hidden variables and unobserved equations use its existing fallback rules. Formula construction keeps the selected graph fixed. Each slot has two optimizer starts, a combined limit of 1,000 residual evaluations and a 180-second wall limit. Formula ranks use seeds 100 and 101 on both sides. A fit qualifies numerically when its training forecast is finite and passes the same two-versus-four-substep integration check. Failures stay in the denominator.

We register the graph cohort, both formulas per graph, the training evidence and source-code identities before starting fits. The same 16 public T2-easy training events supply the evidence for every inventory. An equal-budget comparison runs the optimizer on every executable formula. The record retains each prefit conflict even when an executable formula obtains a finite NMSE. The pointwise and ordering certificates flagged the same 2 of 48 algorithm formulas and 12 of 48 repaired-graph formulas. Every other insulin certificate returned “unverified.” Passing this gate means a check found no demonstrated contradiction. It does not certify that an insulin trajectory can be fitted within the declared pointwise tolerance.

All selections below use training alone. Exact ties break by graph identity and formula rank.

- **Forecast comparison.** Within each graph choose the verified formula with lowest training NMSE. Within each inventory and arm choose the graph with the lowest of those scores. Record development scores for the selected formula of every graph and identify the prespecified portfolio winner separately.
- **Scientific eligibility comparison.** Across all original slots in an inventory and arm, require a passing deterministic prefit check, verified training fit and zero predicted samples below −10⁻¹⁰ for every observed training target. Among qualifying slots, choose the lowest training NMSE. Report both the number of distinct graphs that qualify and the winner's development forecast. If no slot qualifies, the arm has no winner for that inventory.
- A separate diagnostic applies the training nonnegativity requirement without the prefit check. This shows whether the two conditions reject different models.

The four development trajectories are used only after freezing the selections and parameter vectors. Only one of the sixteen training events, `train_015`, contains a nonzero administered-insulin input. All four development events have zero administered insulin. Their observed insulin peaks allow evaluation of the modeled response under those schedules, but the development split cannot establish the response to a dose. These trajectories have been used in earlier studies. The score averages squared forecast error over samples for each target, divides by its training variance and averages over the three targets. Report nonnegativity again on development. Scientific requirements that the deterministic checks cannot establish remain open. The test split is unopened.

## Results on November 24

All 96 fits passed the numerical training check. All 48 graph-level winners could be scored on all four development events. Each pair below is algorithm / 20B-plus-repair. Lower NMSE is better. A dash means that no training fit qualified for that inventory and arm.

| T2-easy inventory | Development NMSE of lowest-training-error choice | Median development NMSE across all eight graphs per arm | Graphs meeting training eligibility / out of eight | Development NMSE of best eligible training choice |
|---|---:|---:|---:|---:|
| Seed 0 | 0.08198 / 0.24547 | 0.26522 / 0.47486 | 0 / 3 | — / 0.68218 |
| Seed 1 | 0.06746 / 0.34806 | 0.20175 / 0.52919 | 0 / 2 | — / 0.72291 |
| Seed 3 | 0.11381 / 0.53677 | 0.30278 / 0.69032 | 0 / 2 | — / 0.53677 |

**Forecast error.** The numerically selected algorithm winner has lower development NMSE in all three inventories. A typical graph from that arm also has lower error in each inventory. In the 64 possible across-arm graph pairs within each inventory, the algorithm has lower NMSE in 60, 60 and 62, respectively. These pairs reuse the same graphs and events, so they should not be treated as independent trials.

**Scientific eligibility.** Every one of the 48 algorithm formulas predicts a negative observed quantity on training. Insulin goes negative in 46. The remaining two keep insulin nonnegative but predict negative utilization. The three primary algorithm winners reach training insulin minima of −19.17, −75.32 and −26.35 pmol/L. The 20B-plus-repair arm has 10 qualifying fits across seven of its 24 graphs. Its lowest-training-error choice for seed 1 has a demonstrable prefit contradiction as well as a negative training forecast. Both are displayed in the report even though the formula integrates and obtains finite NMSE.

**What the qualifying models missed.** The three training-selected, scientifically eligible 20B-plus-repair forecasts kept the three observed targets nonnegative on development. Their maximum absolute insulin errors were 284.79, 283.85 and 275.90 pmol/L. For seed 3, at minute 188 of the second development event (`val_001`), predicted insulin was 88.42 pmol/L and observed insulin was 364.31. These models clear the recorded checks and still miss an important response. No forecast is established as physiologically correct by this study.

**Compute.** The official run used all 96 optimizer starts per arm. Actual residual evaluations were 47,675 for the algorithm and 44,597 for 20B-plus-repair, with 44 and 36 formula slots reaching the 1,000-evaluation cap. The cap is equal even when the optimizer stops early. The local CPU run reused archived generation and incurred no new GPU usage. A separate preliminary software run preceded the official run while per-event reporting was being finalized. Its results were excluded. The official plan digest is `62e9bd0fda7c92f0981671278006c81d6ab9d3917a01220881acd3fae83939d5`.

**Next comparison.** Keep the 48 graphs and their equations frozen. Compare the current fit to a fit that explicitly enforces nonnegative training outputs, using matched optimization budgets. Score development NMSE, insulin peak errors, and the fraction of graphs that retain nonnegativity on development. This tests whether the lower-error graphs can survive the physical constraint with different coefficients. The present post-fit screen alone cannot answer that question. Additional inventories and benchmarks would be needed to compare generators more generally.

[Full report with equations, forecast plots and individual graph scores](../output/matched-topology-forecast-20261124/REPORT.html)

## Reproduce or resume

The registered source artifacts must be present locally at the paths in the cohort and fitting adapters. From the project checkout:

```bash
export PYTHONPATH=src:.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
r=output/matched-topology-forecast-20261124
.venv/bin/python -m scripts.matched_topology_forecast prepare --root "$r"
.venv/bin/python -m scripts.topology_forecast_eligible register --root "$r"
.venv/bin/python -m scripts.matched_topology_forecast fit --root "$r" --workers 6
.venv/bin/python -m scripts.matched_topology_forecast freeze --root "$r"
.venv/bin/python -m scripts.topology_forecast_eligible freeze --root "$r"
.venv/bin/python -m scripts.matched_topology_forecast score --root "$r"
.venv/bin/python -m scripts.topology_forecast_eligible score --root "$r"
.venv/bin/python -m scripts.report_topology_forecast_match "$r"
```

Every completed fit has a sealed receipt. A reservation without a completed fit uses its original slot and is recorded as interrupted on restart. Validation forecasts are cached against the frozen fit identity. Re-running these commands at the same root verifies completed work. No GPU, new proposer call or test data are needed.
