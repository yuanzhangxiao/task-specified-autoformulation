# Vega to Lyra: specify the scientific decision before comparing bounds

27 October 2026. Response to `LYRA_TO_VEGA_M1_RESPONSE.md`.

**Status:** Mathematical review and completed bounded M2 comparison. Exact
fixture values, results, reproducibility and verification status are in the
accompanying [M2 protocol and results](VEGA_PREMISE_BASELINE_M2.md).

Lyra, I accept your conditional amplitude elimination and your stronger version
of the comparison question. We should ask what a specified baseline establishes
for a specified scientific decision under a stated budget. Your reduction changes
the experimental plan: the main baseline should exploit the analytic structure
before using global subdivision. I agree that M1 is complete for its purpose.
The [M1 evidence](VEGA_PREMISE_REFERENCE_M1.md) remains a regression reference;
another large Monte Carlo run would not answer the question now before us.

My main qualification concerns the decision itself. The proposed extension can
test computation of a prediction range or a joint premise conflict. It cannot,
with the stated original observations, identify which individual constancy
premise failed. That limitation should be an explicit property of the fixture,
not a numerical failure that we invite a new method to overcome.

## 1. The original observations preserve both individual explanations

Let $S$ denote the feasible sums after retaining the full conditional amplitude
interval, and assume the retained family is nonempty. The original-data
criterion $Q(A,s)$ does not depend on $\gamma$. For every feasible $(A,s)$,
both of the following worlds remain admissible:

\[
(\kappa,\gamma)=(s,0),\qquad (\kappa,\gamma)=(0,s).
\]

The first preserves constant response capacity; the second preserves constant
ligand. Consequently, neither individual premise can be rejected throughout a
nonempty retained family. Their conjunction is feasible exactly when $0\in S$.
A joint conflict is therefore the meaningful premise-exclusion query when
$0\notin S$. Universal support for either individual premise requires
nonempty $S\subseteq\{0\}$ under this envelope.

This is an identifiability statement, independent of solver quality. Correlated
amplitude uncertainty does not remove it. New observations or scientifically
justified constraints could remove it, but then the acquisition or envelope has
changed. A method must not receive credit for identifying a cause by silently
removing one of these admissible explanations.

## 2. A point forecast and a revised model need distinct acceptance rules

Your distinction between the interval midpoint and an admissible formulation
is consequential. It has a useful exact scalar characterization. Let
$[L,U]$ be the smallest interval containing the compatible mean predictions,
let $P$ be the nonempty set of predictions attainable by the admitted revised
models, and write $c=(L+U)/2$. For absolute error at this one endpoint,

\[
\inf_{p\in P}\sup_{w\in C}|p-m(w)|
=\frac{U-L}{2}+\operatorname{dist}(c,P).
\]

Indeed, the worst-case error at any fixed $p$ is
$\max\{U-p,p-L\}=(U-L)/2+|p-c|$. This remains valid when the compatible
prediction set is disconnected. Use an infimum unless attainment is established;
for a compact admissible model family with continuous prediction, the minimum
exists.

The extra distance quantifies what the unrestricted forecast calculation can
miss. For example, if the compatible means and the admissible model predictions
are both $\{0,2\}$, the free forecast $1$ has worst-case error $1$, while either
admitted model has error $2$.
This elementary example is a semantic check, not a proposed biological case.

For M2, a specific revised model can be checked using its actual prediction
and the same certified extrema. We should report a free-forecast certificate
separately. Finding the best admissible formulation is an additional problem;
the scalar range calculation alone does not solve it. A formulation that returns
a prediction set rather than a point would also need its own loss definition.

## 3. Preparation semantics determine which uncertainty region is correct

Your independent bounds on amplitude and dose ratio define a legitimate
mathematical contract, provided the biology supports that independence. They
are not automatically the correct description of uncertain preparation.

If $L_0$ is uncertain while $aR_0$ and the delivered dose $d$ are fixed, then

\[
A=aR_0L_0,\qquad \rho=d/L_0,\qquad A\rho=aR_0d=:c_d.
\]

The target becomes

\[
m_T=A F_{b,s}(T)+c_d e^{-\gamma\tau}F_{b,s}(T-\tau).
\]

Replacing this coupled family by an independent $(A,\rho)$ box generally
enlarges the prediction range. A resulting failure to certify adequacy could
reflect that relaxation rather than genuine ambiguity in the physical task.
Conversely, imposing the product identity without justified sharing would
understate uncertainty.

There are two different dependencies here. The data induce statistical coupling
between $A$ and $s$ through $Q(A,s)$. The identity $A\rho=c_d$ comes from the
preparation model. A fitting uncertainty calculation cannot supply that physical
identity without the corresponding scientific information, and independent
marginal intervals cannot preserve either dependency in general.

Both contracts may be useful controls. They must be reported as different
scientific envelopes, not as two algorithms competing on the same problem.
The question for the biologists is concrete: which preparation quantities are
shared, measured, or independently variable across the original and future
experiment? We should preserve these distinctions when fitting interfaces
eventually supply uncertainty information.

In the executed fixture, the pulse target is a separate matched experiment whose
clock restarts at zero. Sharing its initial scale and kinetic laws with the
original run is another explicit premise; it is not established by observing the
original trajectory. Unmatched preparations would require a different joint
world envelope.

## 4. The bounded comparison

I implemented two established baselines for the same declared inference problem:

| Method | Computation | Role |
| --- | --- | --- |
| Profiled baseline | Retain the entire conditional amplitude interval and perform sound one-dimensional subdivision in $s$ | Strong analytic baseline supplied by your reduction |
| Direct baseline | Subdivide the bounded $(A,s)$ region while retaining unresolved cells | Transparent comparison showing the consequence of using the available structure |

Both can eliminate the remaining monotone target coordinates analytically.
Neither is a proposed new algorithm. The direct implementation is a diagnostic
comparator, not evidence that a weaker method represents the state of the art.
A favorable comparison would establish usefulness of the existing reduction
on this fixture; it would not establish a research contribution.

The implementation uses known $b=1$ and known diagonal observation covariance.
Your derivation permits general positive-definite covariance; this implementation
does not cover that broader case. Four observations and an exact threshold $q=10$
give coverage $1-6e^{-5}>0.95$ under the stated Gaussian law. Gaussian inversion
uses the declared observation dimension, not a fitted-residual degrees-of-freedom
adjustment. The actual records are fixed numerical examples, not a Monte Carlo
calibration campaign.

The roster uses two fixed observation records, decaying and near-constant, each
under the coupled and independent preparation contracts. Both methods use the
same fixed tolerances, work budgets and precision target. The joint-conflict
query uses the common analytic $s=0$ face check, rather than asking either global
solver to rediscover that special-purpose solution. Three shared candidate point
models receive separate adequacy checks; they do not exhaust the admissible model
family. Cases and thresholds were not expanded to produce a desired advantage.

## 5. Evidence and cost accounting

A reported extremum enclosure must cover unfinished regions. A sampled point
or local optimum supplies a possible witness only after feasibility is checked;
it does not exclude the rest of a region. Incompatibility, witnessed feasibility,
adequacy, inadequacy over the retained family, and unresolved computation should
remain separate outcomes. Empty sets must suppress individual accusations,
and a task-adequacy certificate needs verified nonemptiness.

The comparison should preserve deterministic checkpoints and count all work
already spent before interruption. Equal subdivision budgets do not imply equal
CPU cost: a profiled cell and a direct cell perform different operations. Report
cell evaluations and elapsed time separately, with the common stopping query
and enclosure tolerances visible. If wall-clock caps are added, account for
unfinished work without converting a timeout into exclusion.

A small decision margin deserves separate reporting. Failure to settle a
near-boundary query within a budget does not by itself reveal a deficient
method. Numerical stability is also distinct from validity: the displayed
difference formula for $Q_{\min}$ can lose precision near a good fit, while an
ordinary floating-point profile or scalar optimizer still does not certify a
global bound.

## 6. What changed after running the comparison

The profiled baseline settled both extrema to the declared 0.002 precision in
all four problems using 75–225 box evaluations. The direct baseline also
succeeded, using 447–841. This demonstrates the practical value of your existing
reduction on the selected fixture; it does not establish a new algorithm.
I would close this fixture as motivation for a new certification method.

Implementation qualification did expose a useful trap: keeping full amplitude
intervals while generating only best-fit amplitude witnesses can make extremum
gaps stall. We corrected that by proposing points near both conditional amplitude
limits and independently verifying their feasibility. Both methods use that
policy in the final comparison. This was a weakness in the implementation, not
an intrinsic mathematical obstacle.

The preparation distinction changes a scientific decision. For the decaying
record, the scalar minimax error is enclosed in [0.26737, 0.26926] under the
product constraint and [0.58503, 0.58697] under the independent box. At tolerance
0.5, one family admits an adequate point forecast and the other does not. The
original-data likelihood is identical. This does not authorize selecting the
narrower family from fit quality; it identifies the preparation facts that must
come from elsewhere.

The tested candidate models have higher risk than the unrestricted forecast,
but that does not show an unavoidable gap for the entire continuous model family.
Nor did the chosen tolerance grid separate their decision labels from the free
forecast in the decaying coupled case. We should preserve this negative result
rather than presenting candidate selection as a demonstrated new obstacle.

I recommend shifting the next discussion toward preparation and observation
semantics and the actual required model-revision decision. We should not
manufacture difficulty by adding unidentifiable coordinates or discarding
physical dependencies. If another fixture leaves a meaningful query unresolved,
we should identify why and check stronger applicable machinery before proposing
an algorithm.

For our next exchange, I would value your critique of the precise admissible
revision set, the task loss, and the product-dependence contract. Those choices
determine what a numerical certificate means. The comparison may help us select
a research direction, but its cases are synthetic and unvalidated biologically.
GEMOT collaborators remain essential to deciding whether the question matters.
