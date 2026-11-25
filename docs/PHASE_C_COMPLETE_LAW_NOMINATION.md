# Choosing complete process equations from training trajectories

The coupled-basin pilot holds the two storage balances fixed and asks which
complete discharge equations predict a new storm. It estimates the shape of
each law separately. The earlier basin proposer experiment asked for a factor
in a prescribed product. Its results are a separate historical comparison.

## What the constructor knows

The public task supplies two inflows, basin areas, overflow thresholds, and
initial depths. It requires the upstream volumetric loss to enter downstream
and the downstream discharge to leave the system. With local head
\(H=\max(h-c,0)\), the pilot considers four full laws for each flow,
\(k\sqrt H\), \(kH\), \(kH^p\) with \(0.25\le p\le4\), and
\(VH/(K+H)\). Each can be evaluated directly in the conservation equations.
Each is continuous, nonnegative and nondecreasing on the active domain. These
are assumptions of this first library. The gains, exponent and saturation
scale are estimated on training events. A square-root discharge stays a
square-root discharge in the differential equation.

The generic constructor accepts two source endpoints and a measured mean rate
for each time interval. It integrates each law along a linear interpolation
of the source, fits the coefficients, and ranks the laws by rate error on
whole omitted training events. Each event carries equal weight. A single
informative event permits a fit but only an in-sample ranking. The constructor
records that distinction even when an estimate looks precise.

The constructor also checks a necessary property of every law in this menu.
If the whole source range of interval A lies below that of interval B, a
nondecreasing law must produce a mean rate in A no larger than in B. A
reversal larger than both supplied absolute error bounds combined rules
out the menu under the source interpolation assumption. Missing error bounds
leave an advisory record. The check retains the two intervals as evidence
for a different scientific hypothesis.

## Reconstructing evidence from public measurements

Downstream depth is recorded at every observation. Upstream depth is measured
only at the start. When cumulative upstream inflow is still insufficient to
reach the crest, the downstream outlet can be estimated directly from the
interval balance

\[
\bar D=\bar u_{\rm down}-A_{\rm down}\,\Delta h_{\rm down}/\Delta t .
\]

Only two of six training events provide such intervals, four in `train_000`
and 60 in `train_003`. For each candidate outlet selected, the remaining
intervals give a conditional transfer estimate from the downstream balance.
Integrating upstream inflow minus that transfer gives an inferred upstream
level. The second law is fitted to these reconstructed quantities. An upstream
rate reversal of 0.867 m³/min in the full fit is advisory because total error
in the inferred rates is unknown. One inferred interval gives a transfer of
−0.0403 m³/min. This records uncertainty about the upstream reconstruction.

Using all six training events, the selected laws in m³/min for heads in metres
are approximately

\[
Q_{\rm up}=72.08\,H_{\rm up}^{1.510},\qquad
D_{\rm down}=54.34\,H_{\rm down}^{1.493}.
\]

The following evaluation starts over each time, withholding one entire
training event before choosing either law or fitting any coefficients. The
control fits a linear law for each discharge using the same remaining
events, geometry and inference procedure. Each held prediction starts at
the public initial depths and uses the supplied inflows. It is scored after
the initial observation, with NMSE normalized by the remaining training
events' downstream depth variance.

| Omitted training event | Selected laws, depth NMSE | Linear laws, depth NMSE |
|---|---:|---:|
| 000 | 0.00000210 | 0.05546 |
| 001 | 0.00002545 | 0.14734 |
| 002 | 0.00003101 | 0.03561 |
| 003 | 4.74782 | 4.74782 |
| 004 | 0.00003698 | 0.16626 |
| 005 | 0.00000804 | 0.03261 |

Omitting storm 003 leaves four directly informative intervals with head
0.0255–0.0300 m. Its own head reaches 0.9871 m. The nominated outlet is
linear in that fold, and both forecasts miss downstream depth by 0.695 m
RMSE. This failure motivates showing the range that supported each choice
and retaining alternatives for extrapolation. The six folds overlap in their
training sets. They measure sensitivity to training events on this one basin.
Earlier experiments had already suggested a three-halves law here. This pilot
checks whether a constructor given several full families can select that
shape from the available measurements. An independently chosen benchmark is
needed to estimate how well that approach generalizes.

## Testing recovery separately from forecast accuracy

We generated three independent signed source excursions per fit and tested
interval-mean rates on a fourth, higher-amplitude excursion. Each case used a
fixed seed. The clean and noisy versions share their source design. The noisy
version adds uniform deviations bounded by 3% of each training event's
peak-to-peak rate. There are 12 seeds per law and noise condition. For a power
law we count an estimated exponent within 0.10 as a recovery, including the
square-root and linear special cases. The independent assessment uses the
synthetic generating laws and never supplies their identities to the constructor.

| Condition | Top choice recovers form | One of top two recovers form |
|---|---:|---:|
| Four represented forms, clean | 48/48 | 48/48 |
| Four represented forms, bounded noise | 45/48 | 48/48 |

All three noisy misses selected a saturating law on truly linear data.
The fitted half-saturation levels were 102–1,145 in source units, far above
the observed source maximum of about 2.75. Their new-event rate NMSEs ranged
from 0.0000011 to 0.00053. Excellent prediction over the observed range can
leave the mechanism ambiguous. For a fifth law whose rate rises and then
falls with increasing source, the interval-order check correctly ruled out
the entire library in all 24 cases. It did not discover a replacement.

## Reproduce locally

From the project checkout with its registered coupled-basin public release:

    PYTHONPATH=src:. .venv/bin/python -m scripts.evaluate_process_nomination --root output/complete-law-nomination-20261125

The command verifies saved synthetic trials and basin folds on resume. The
summary contains the candidate expressions, fitted parameters, competing
scores, omitted-event trajectories, and shape checks. The experiment reads
public training data only and makes no LLM calls.

The matched T2 refit separately inherits explicit public observed-target
bounds before scheduling fits. The new plan records the public declaration
behind each bound. In the named T2 packet glucose mass is explicitly
nonnegative. Absolute insulin concentration and one-way disposal use reviewed
descriptions. Signed deviations and unrecognized quantities keep their prior
freedom. The current constraint locations are the training observation times.

## Next experiment

Use the same admitted T2 graphs, number of shortlisted candidates and
nonnegative fitting budget across three constructors: the existing small
library, shape-informed equations, and shape-informed equations with a 20B
alternative. Where only total state change is observed, compare integrated
net rates on intervals and simulate the entire coupled system. A delayed peak
can motivate a persistent input effect if the topology contains a state that
can carry it. Report per-event forecast NMSE, coverage, and errors in peak
timing and recovery. On controls with known equations, also report whether
the generating form is in the shortlist. An explicit unsupported outcome
allows a wider candidate family or a topology revision.
