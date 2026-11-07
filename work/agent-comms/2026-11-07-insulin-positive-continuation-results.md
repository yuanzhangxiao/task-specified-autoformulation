# Candidate 15 insulin continuation results

All six Delta tasks completed. Each unconstrained continuation lowered its
training NMSE and still predicted negative insulin. Each constrained fit passed
the finer concentration check across all 16 training trajectories. For the
primary pair, keeping insulin nonnegative raised overall NMSE from 0.16164 to
0.16556, a 2.43% increase. The latter value is 36.6% below the earlier direct-meal
model’s 0.26101 and improves on the preliminary constrained smoke result of 0.17269.

| Historical start | Prior NMSE | Continued free NMSE | Free minimum I (pmol/L) | Constrained NMSE | Increase against matched free fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primary, originally zero | 0.16365 | 0.16164 | −31.75 | 0.16556 | 2.43% |
| Originally negative | 0.16743 | 0.16173 | −17.57 | 0.16600 | 2.64% |
| Originally positive | 0.18767 | 0.16169 | −49.06 | 0.18153 | 12.27% |

The labels identify the initial perturbations in the earlier campaign. The
fitted feedback coefficients were negative at all three saved starts. Each pair
uses the same equations, training observations and additional search allowance.
The verification grid has 38,416 times across trajectories. Each constrained
fit has one negative numerical value of magnitude below 4 × 10⁻¹¹ pmol/L, within
the recorded tolerance of 1.12 × 10⁻⁶ pmol/L. The free arms have 762, 343, and
1,254 samples below tolerance, respectively.

## Meal response during training

The predeclared 90 g meals start at minutes 0, 60 and 120. Observed insulin
peaks at minutes 101, 161 and 221, each at 301.2 pmol/L. The earlier direct-meal
model peaked 91 minutes early on all three schedules. In the primary free
continuation the peaks are 35, 29 and 30 minutes early. In the primary
constrained fit they are 36, 29 and 30 minutes early and reach 240.0, 239.9 and
257.9 pmol/L. The fit from the originally positive start is more sensitive to
the constraint and peaks 67, 61 and 62 minutes early.

The ten-minute ingestion windows give more specific feedback than peak time
alone. Observed insulin rises by 1.97 pmol/L during each meal. The primary
constrained fit rises by 162.19, 165.86 and 167.27 pmol/L. Its insulin equation
contains the direct term `c_I * meal_rate_g_per_min`, with `c_I = 1.7141`.
Integrating that term alone over 90 g contributes +154.27 pmol/L during
ingestion. These numbers identify the equation term responsible for most of the
early increase. They provide training evidence for testing a slower meal response.

## The concentration boundary

On the 150 g trajectory, observed insulin at the last observation, minute 300,
is 22.62 pmol/L. The primary free fit predicts −31.75. All three constrained fits
reach approximately zero at that time. In the primary constrained equation,
external insulin and ingestion are zero, the glucose term contributes
+0.123 pmol/L/min, and the term in the delayed insulin-action state contributes
−0.925 pmol/L/min. The insulin self-term vanishes at zero. The resulting net rate
is −0.802 pmol/L/min.

As a post-hoc diagnostic, Orion extended each of the three frozen constrained
models from its original fitted terminal state for one minute, keeping both
public input rates at zero. Radau integration and affine matrix propagation
agree to within 10⁻¹² in final state values. Predicted insulin at minute 301 is
−0.794, −0.742 and −0.292 pmol/L in the table’s order. The last benchmark
observation is at minute 300. The extension tests the fitted equations’ behavior
at depletion and supplies no new prediction-error score.

For Astra, two concrete targets are a smaller insulin rise during ingestion and
a net rate satisfying `dI/dt >= 0` at `I = 0` under admissible conditions. In
the saved fits, the delayed action state continues to suppress insulin when the
available insulin has reached zero. A revision can preserve the task-required
delayed effect of insulin on glucose disposal while changing this feedback into
the insulin balance. The primary constrained fit provides a common comparison
at NMSE 0.16556. The data leave the choice of functional form open.

## Fitting and verification

Each outer search exhausted 1,800 numerical seconds. The free arms completed
1,400, 1,466 and 1,413 objective calls, and the constrained arms completed 888,
850 and 868. All retained conditional solves had rank 35 and satisfied their
recorded optimality tolerances. The local audits found further reductions in
NMSE ranging from 4.08 × 10⁻⁷ to 6.85 × 10⁻⁵. Each task then retained its best
verified probe. The audit therefore describes the point immediately before that
last step. Outer convergence and the best attainable loss in this family
remain open. All scores here reconstruct training trajectories with 32 fitted
initial values. Prefix-conditioned forecasting on unseen schedules remains a
separate evaluation.

Archive SHA-256 `9d821bb258e5141d607c9f1cab6c974faa31ec8c75b9b78c56b90510fa2593a4`,
implementation `5073358`, Delta jobs `23635365` and `23635369`.
The review verified 131 sealed files, reconciled all six final records against
60 operation records, and independently replayed the original equations at
the six frozen parameter vectors. The maximum difference from the saved
normalized predictions was 1.22 × 10⁻⁹. The plan, runtime declaration, source
digest, entrypoint digests and archived starting points were checked.

The local review includes the paired per-target errors, optimizer counts, and
observed and predicted trajectories. [Detailed report](../../output/insulin-positive-continuation-analysis-9d821bb2/REPORT.html)
and [trajectory figure](../../output/insulin-positive-continuation-analysis-9d821bb2/insulin-trajectories.pdf).
