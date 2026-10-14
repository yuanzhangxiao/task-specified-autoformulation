# Preparation: noise-aware fitting and conditional parameter evidence

Status: implementation design, 13 October 2026. No new fitter, experiment,
benchmark release or production default is introduced by this document.
M33 is submitted separately; its numerical results are pending.

Companion designs: [measurement evidence](PHASE_C_MEASUREMENT_EVIDENCE_DESIGN.md)
and [controlled diagnosis study](PHASE_C_SPEC_DATA_DIAGNOSTIC_STUDY.md).
The broader research question is in the
[Vega handoff](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md).

## The next question

The M29–M33 backend can distinguish several numerical/search failures from an
accurate fit on qualified examples. It cannot yet tell whether a residual is
consistent with measurement uncertainty, nor whether an apparent parameter
constraint survives uncertainty in initial observations or calibration.

The proposed next step is to qualify these questions on fixed, known equations.
The first implementation should add a **training-only observation objective and
an explicit initial-observation contract**, with fixed externally justified
noise scales. Keep the current rollout/profiling portfolio, domain checks,
checkpoint retention and independently verified predictions. A separate objective
identity determines which checkpoints can be compared or resumed.

This does not require a new optimizer before we know what the optimizer should
minimize. It requires separating four quantities currently easy to conflate:

| Quantity | Meaning | How it is chosen |
| --- | --- | --- |
| Coordinate scale | Numerical size of a step in a parameter/initial coordinate | Frozen training/domain-derived coordinate policy |
| NMSE denominator | Descriptive size of variation in an observed channel | Declared training-only reporting convention |
| Measurement standard deviation | Uncertainty of a raw observation in its units | Calibration, replicates, or explicitly declared assumption |
| Integration tolerance | Numerical accuracy requested from the ODE solver | Independent numerical qualification |

The existing `scale_for(train)` normalization in `confidence_checks.Rollouts`
is not a measurement standard deviation. It must not be relabeled as one.
The table describes possible sources, not first-profile eligibility. Initially,
sigma must be fixed and declared known under the measurement protocol. Finite
calibration/replicate estimates remain labeled estimated and are unsupported by
this profile; a later plug-in objective would make its conditional interpretation
and omitted scale uncertainty explicit. Independence from candidate residuals
alone does not make a scale known.

## Observation model and objective

For trajectory \(j\), use the fixed candidate equations

\[
\dot x_j=f(x_j,u_j(t);\theta),\qquad
x_j(0)=g(a_j;\eta),\qquad
\mu_{jic}=H_c(x_j(t_{ji});\psi).
\]

Here \(a_j\) contains only explicitly permitted initial information; \(\eta\)
contains eligible initial-map parameters; \(\psi\) contains eligible measurement
nuisance parameters. In the first implementation the time base, input histories
and calibration are fixed, so \(\psi\) is empty. Any later nuisance expansion is
a declared model with bounds and provenance, not a free correction per sample.

Stack present, permitted training readings once in a frozen index order. Let
\(y\) be the readings after the declared fixed unit/calibration conversion;
retain the original raw readings and conversion alongside this derived view.
Predictions and sigma must use the same analysis units. Let
\(v=(\theta,\eta,\psi)\), \(e(v)=\mu(v)-y\), and let
\(\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_N^2)\), with every
\(\sigma_i>0\) supplied independently of this candidate's residuals. Then

\[
r(v)=We(v),\quad W=\operatorname{diag}(\sigma_i^{-1}),\qquad
L_G(v)=\tfrac12\|r(v)\|_2^2+\sum_i\log\sigma_i+
\tfrac N2\log(2\pi).
\]

Optimize the quadratic part when the remaining terms are constant; retain the
full objective identity for reporting. A sum gives the Gaussian likelihood for
independent readings. Equal weighting of trajectories instead would be a
different objective, not the same likelihood. A later fixed full covariance
uses a Cholesky solve; it must not be represented by diagonal weights while
silently ignoring correlation. Missing entries require an explicit acquisition
mask. Censored entries need a censored likelihood and are initially unsupported.

For exact synthetic observations, retain a separate noiseless objective and
numerical accuracy target. Do not set Gaussian \(\sigma=0\), and do not describe
an arbitrary numerical floor as a measured noise level. If the noise scale is
unavailable, return descriptive residual evidence with statistical adequacy
unassessed. Estimating noise from a poor candidate's residuals could absorb the
missing mechanism. Unknown scales require a later bounded likelihood with its
normalization terms, not minimization of \(e^2/\sigma^2\) alone.

Measurement-model choice affects estimation and interpretation; it is part of
the scientific assumptions. Murphy, Maclaren and Simpson demonstrate this in a
likelihood-based mechanistic-model framework, rather than recommending a universal
error law. [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC10827430/)

## Initial readings cannot silently become exact initial states

If \(y_{j0}=x_j(0)+\epsilon_{j0}\), setting \(x_j(0)=y_{j0}\) exactly conditions
on a noisy measurement as if it were error-free. The resulting trajectory error
can resemble incorrect dynamics. Use one of two explicit designs:

1. **Exact preparation:** a separately justified, calibrated initial state is
   fixed. The first noisy reading is still an observation, not the preparation.
2. **Uncertain preparation:** infer an eligible initial value/map jointly with
   physical parameters. Its first observed reading enters the observation
   likelihood exactly once. An independent preparation measurement may contribute
   its own factor; copying the same reading into a prior would double-count it.

The first adapter extension should cover a shared unknown initial state across
trajectories only when the public preparation protocol justifies sharing, or a
shared causal map with eligible global coefficients and exact public covariates.
Such a map may return different initials without fitting a separate value for
each trajectory. Exact boundaries are recorded by trajectory and may also differ.
The main diagnostic fixture uses exact preparations; a separate qualification
control exercises the shared uncertain initial. This is not a justification to
share arbitrary experimental initials. Per-trajectory initials and maps driven
by noisy initial covariates require a later explicit scope and uncertainty model.
No later observations from a held-out trajectory may initialize its rollout.

Current `public_fitting._lower`, `apply_initialization_plan` and
`SymbolicODE.initial_for` implement explicit existing initialization semantics.
The new opt-in path must reject incompatible hard-bound noisy initials, and
preserve the historical path. An adapter must not overwrite raw first readings
to trick the old initializer into accepting estimated values.

## Preserve profiling where its mathematical certificate still holds

Suppose that, conditional on nonlinear parameters \(\beta\), the *complete
predicted outputs* have a certified affine form

\[
\mu(\beta,\gamma)=a(\beta)+B(\beta)\gamma,
\qquad \ell\leq\gamma\leq u.
\]

The inner variables \(\gamma\) may include eligible linear gains and initial
values. Fixed Gaussian uncertainty preserves the convex subproblem

\[
\gamma^*(\beta)\in\arg\min_{\ell\leq\gamma\leq u}
\tfrac12\|WB(\beta)\gamma+W(a(\beta)-y)\|^2.
\]

Whiten the design, offset and their derivatives consistently. The existing
`profiled_output.project` and `affine_profile.AffineProfile` provide relevant
bounded profiling machinery, but their current residual normalization and
single-output eligibility need an explicit audit before reuse. An RHS linear in
a coefficient does not establish that its integrated outputs are affine in that
coefficient. Ineligible cases retain direct joint rollout optimization.

For a fixed design and bounds, bounded linear least squares is convex; a verified
inner optimum does not establish the optimum over nonlinear \(\beta\).
[SciPy's bounded linear least-squares API](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.lsq_linear.html)
provides the existing numerical building block. Keep inner convergence, active
constraints, any rank deficiency and its compute charge explicit.

The second increment adds a predeclared Huber loss on standardized residuals:

\[
\rho_\delta(r)=
\begin{cases}r^2/2,&|r|\leq\delta,\\
\delta|r|-\delta^2/2,&|r|>\delta.
\end{cases}\qquad L_H=\sum_i\rho_\delta(r_i).
\]

For fixed \(\beta,W,\delta\), the affine inner problem is still convex. It is
not one ordinary least-squares solve. Solve it directly or iterate weighted
subproblems to a declared KKT/objective tolerance; a single reweighting is not
an exact profile. If \(r\ne0\), the usual working weight is
\(\rho_\delta'(r)/r=\min(1,\delta/|r|)\), with weight one at zero. Record
inner iterations, failed solves and approximate-profile status. Nonunique minima,
changing active bounds and Huber transition points need guarded derivatives.

Start with a fixed \(\delta\), such as 1.345, as an experimental setting; do
not select it to make a preferred mechanism win. Huber is a robustness objective
here, not a fitted data-fault probability. Downweighted samples are not confirmed
bad measurements. General robust losses are not all convex. SciPy's
[`least_squares`](https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.optimize.least_squares.html)
supports robust bounded fitting, but ordinary Gaussian confidence formulas cannot
be copied onto a robust result without a suitable statistical analysis.

An output bias may be affine; simultaneous sensor gain and physical gain can be
bilinear or exactly confounded. Certify the chosen inner block after composing
the observation map. Do not profile jointly through an uncertified product.

## Joint information and honest uncertainty

With a fixed whitening matrix and coordinate-scale diagonal \(D\), assess

\[
\widetilde J=W\frac{\partial\mu}{\partial v}D.
\]

Use M32/M33's qualified derivative scope and saved actual perturbation directions.
Include all eligible coefficients, initial values and observation nuisances.
For Gaussian data, \(J_w^TJ_w\) is local expected information under the declared
model; it is not generally the full observed Hessian of a nonlinear objective.
Rank and singular values describe local information in specified coordinates.
Full rank by itself does not establish practically narrow uncertainty or remote
uniqueness, and M33's numerical tolerances are not statistical significance levels.

Writing the whitened Jacobian as \([J_\theta\;J_\nu]\), where \(\nu\) includes
eligible initials and calibration, the unconstrained local information remaining
for physical parameters after profiling nuisance directions is

\[
I_{\theta\mid\nu}=J_\theta^T(I-P_\nu)J_\theta,
\qquad P_\nu=J_\nu J_\nu^+.
\]

Implement with stable QR/SVD projections, retaining rank decisions and scale
definitions. Constraints require tangent-cone/one-sided treatment; the expression
above is an interior linearization. Report data-only information separately from
curvature introduced by priors or regularization. A tight prior can stabilize an
estimate without making the measurements informative.

For a genuine likelihood, a parameter profile is
\(L_i(a)=\min_{v:v_i=a}L_G(v)\), reoptimizing all permitted nuisance variables.
The familiar \(2[L_i(a)-L(\hat v)]\) likelihood-ratio threshold has regularity,
identifiability and asymptotic assumptions. Boundaries, kinks, model/claim
selection and weak information can invalidate its usual calibration. An
unfinished profile is not a wide or narrow confidence interval. Use the first
small fixtures to measure coverage before publishing confidence percentages;
calibration must rerun the intended fitting/selection procedure on independent
replicates. Keep a `confidence_probability: null` default until an event and
calibration experiment are actually defined and supported.

## Fitting output and routing

Return a table of independent axes rather than collapsing them into success:

| Axis | Example output | Permitted interpretation/action |
| --- | --- | --- |
| Numerical agreement | Verified outputs; derivatives unsupported | Keep output evidence, withhold sensitivity claims |
| Search reliability | Improving / locally stationary / restricted optimum / unavailable | Continue, restart, or report the stated optimum scope |
| Observation adequacy | Descriptive residuals; Gaussian adequacy uncalibrated | Do not label a scientific or data fault |
| Parameter information | Joint weak direction; nuisance confounding | Report combination/ambiguity and a discriminating measurement |
| Model/data attribution | Not assessed or competing explanations | Request bounded hypothesis comparisons; preserve alternatives |

A small training residual is not itself the stopping rule for noisy observations.
Use numerical and optimization termination criteria, with optional calibrated
adequacy checks later. The heuristic \(\|r\|^2\approx N-p\) only has a simple
distribution under appropriate linear/unconstrained/correct-model assumptions;
do not impose it as a universal acceptance gate. Training residual correlations,
pulse-aligned errors and outlier-like extremes remain descriptive until tested
under the frozen observation model, sampling design and selection procedure.

Raw residuals and historical NMSE stay visible even when the optimizer uses a
different loss. A lower Huber objective is not numerically comparable to a
Gaussian objective; compare arms through a common evaluation protocol and
separately reported native objectives. Clean physical predictions in synthetic
studies are evaluator-only, never an incumbent-selection criterion.

## Implementation sequence and exit checks

| Increment | Proposed change | Required qualification before expansion |
| --- | --- | --- |
| P1: objective/initial contract | New opt-in wrapper; fixed diagonal sigma; raw and standardized residuals; exact or explicitly shared uncertain preparation | Analytical weighted solution; correct initial treatment; identity/resume rejection; no evaluator leakage |
| P2: preserve profiling | Apply the same fixed weights to direct rollout and certified affine profiles | Same objective/endpoints within numerical tolerance; analytic sensitivities; rank/bound controls; independent rollout |
| P3: robustness | Fixed Huber setting and verified convex affine inner solves | Convex optimum/KKT checks; direct-vs-profile agreement; both clean and contaminated controls; no sample fault verdict |
| P4: bounded measurement hypothesis | One shared sensor drift parameter, then its confounding control | Joint physical/initial/nuisance assessment; raw readings unchanged; identifiable and deliberately ambiguous examples |
| P5: diagnosis study | Freeze the four-arm study and its calibration protocol | Oracle fixture checks, measured cost pilot, locked evaluation, all failures retained |

P1/P2 are the recommended first coded milestone after this preparation. They do
not wait for fresh construction functions, and do not replace M33. P3–P5 are
subsequent milestones; the broad study must not launch just because this design
lists it. No LLM is needed for these stages. Use small local analytical unit
controls and Delta CPUs for measured fitting experiments.

Tests must include negative sigma/nonfinite values, missing unit/noise provenance,
mask or objective changes on resume, double-counted initial observations,
unsupported covariance/censoring/noisy forcing, Gaussian affine optima, unknown
initial/gain confounding, Huber inner-solve failure, and interrupted accounting.
Reuse existing rollout, profiling, activity, piecewise and portable-evidence tests.
Do not change the sealed Phase C datasets to obtain these controls.

Measure fitting, inner profiling, derivative qualification, nuisance profiles,
calibration replications, and artifact preparation separately. Preserve unknown
interrupted usage. Future statistical calibration can cost much more than the
initial point fit; allocate it to retained endpoints and report that cost.
