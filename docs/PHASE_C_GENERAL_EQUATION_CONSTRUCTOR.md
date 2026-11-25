# Algorithm 1. Trajectory-informed equation nomination on a fixed graph

**Question.** Given a graph of permitted dependencies and measured training trajectories, which complete equations should receive the limited budget for fitting and free simulation?

| Contract | Specification |
|---|---|
| Input | A fixed graph listing each generated variable, its differential or algebraic type, its ordered RHS sources, and its allowed initialization. Each training event supplies sampled times, targets, available inputs and auxiliaries, and fixed covariates. The caller identifies time and any sources held at their previous value between samples. |
| Output | Up to two complete models with analytic equations, initializer expressions and shared coefficient starts. Each locally assessed equation has a record of the comparison, the leading families, their errors, and which variables were unobserved. |
| Generic assumptions | Finite-dimensional dynamics on the supplied graph, shared coefficients across events, an affine background and at most one added shape per equation. Differential screening uses numerical quadrature between measurements. These choices restrict the search and are identical across benchmarks. |
| Information boundary | The deterministic constructor uses graph membership, types, training measurements and interpolation metadata. It receives no benchmark identifier, natural-language task, physiology or hydraulic template, simulator equations, hidden trajectories, derivative labels, evaluation scores or validation/test rows. The graph itself can carry scientific choices made upstream. |

## Procedure

Let $v$ denote a generated variable, $S_v$ its declared sources and $O_v$ the subset with measured or supplied values. An input can influence an equation only through a source already in $S_v$.

| Step | Operation |
|---:|---|
| 1 | Verify the training event IDs and channel alignment. Sort events by ID. Preserve the graph and initialization rules. |
| 2 | For each measured $v$ with at least one available source, form an affine equation on all $O_v$. Enumerate one alternative at a time by adding one hinge, smooth bend, square, or two-source product. The expressions below define the full menu. |
| 3 | Split events by alternating sorted ID into two folds. For each fold, estimate source centers, scales, hinge locations and coefficients using only the opposite fold. For an algebraic variable, predict the measured value at each time. For a differential variable, predict net change divided by elapsed time over overlapping windows of 1, 4 and 16 sample intervals, where available. |
| 4 | Compute the omitted-event RMSE of each family, giving each event equal weight and each available window width equal weight within an event. Rank by RMSE. Prefer the affine model when its RMSE is within $2\%$ of the best plus $10^{-10}$ in the equation's units. With only one event, rank using that event and record an in-sample score. |
| 5 | Fit the selected families' coefficient starts on all construction events. Reinsert every unobserved declared source as a free affine term. Mark a score obtained on $O_v\subsetneq S_v$ as a score for the observed part only. For an unobserved response or a response with no measured sources, use the fallback below and give the equation no local shape score. |
| 6 | Assemble model A from each equation's first choice. For model B, replace exactly one equation with its second choice. Among equations with two cross-validated candidates, select the smallest relative gap $\lvert e_1-e_2\rvert/\max(e_1,e_2,10^{-10})$. If none have such scores, use the first available alternative in graph order. If no alternative exists, return A alone. |
| 7 | Bind permitted initializers and distinct free coefficients. Parse the complete expressions in the restricted grammar and verify that every RHS uses its declared sources. Return the models and evidence. Joint fitting and forecast scoring take place downstream. |

### Families and objective

Write $z_i=(x_i-\mu_i)/s_i$ and $m_i=\operatorname{median}(z_i)$. For the measured arguments the candidate families have the form

$$
F(z)=\beta_0+\sum_i\beta_i z_i
\quad\text{or}\quad
F(z)=\beta_0+\sum_i\beta_i z_i+\gamma b(z),
$$

$$
b(z)\in\{\max(z_i-m_i,0),\ \tanh z_i,\ z_i^2,\ z_i z_j\ (i<j)\}.
$$

All coefficients are signed. The fitting fold supplies the pooled sample mean, standard deviation and median. A standard deviation at or below $10^{-9}$ uses $\max(1,\lvert\mu_i\rvert)$ as the scale. Shape alternatives involving a constant source over the construction events are excluded from the ranking. The affine equation still carries that source. Exact score ties follow enumeration order.

For an ODE window $[t_k,t_{k+w}]$ the response is $[v(t_{k+w})-v(t_k)]/(t_{k+w}-t_k)$. Each design column is the time average of the corresponding basis function over the same window, approximated by the composite trapezoidal rule. A step-held source uses its left endpoint at both ends of each step. Actual elapsed times enter the quadrature and the denominator, including on irregular grids. With $J$ fitting events the coefficient starts minimize

$$
\frac{1}{J}\sum_j\frac{1}{|W_j|}\sum_{w\in W_j}
\frac{1}{n_{jw}}\sum_k (y_{jwk}-X_{jwk}\beta)^2
+10^{-6}\|\beta_{\rm slopes}\|_2^2.
$$

Here $n_{jw}$ counts windows of width $w$ in event $j$. Algebraic equations use one block of measured values per event. Overlapping windows summarize dynamics at several timescales. They do not supply independent experimental replicates.

### Latent variables and initialization

For an unobserved response or one with no measured sources, the fallback offers an affine equation, and, if a source exists, an affine equation with a tanh term on the first declared source. With $T$ the median event duration floored at one, unobserved ODE gains start at $0.1/T$ and algebraic gains at $0.1$. In the fallback, a differential self-dependency starts with a restoring slope of $-1/T$ in the original coordinate. For a measured response with no measured source the intercept starts at its average value or average rate. The joint fit can change the values and signs of these numerical starts.

A measured-initial rule uses the declared observation. A shared unknown initial value receives one fitted constant. A declared initial map receives an affine map of exactly its permitted public and state initial values. The map coefficients start at zero. This implementation covers these time-zero rules. A separate construction or fitting interface is needed for per-event prefix estimation.

### Scientific knowledge and the 20B arm

The earlier basin experiment used manually supplied water balances and a known threshold to isolate a discharge process. Algorithm 1 performs no balance isolation and assumes no threshold for a particular variable. Its hinge location is a training statistic. It ranks the total derivative of an observed variable or the observed portion of that derivative. A hidden mechanism can produce the same observed shape as a different equation.

The matched experiment reuses previously admitted graphs and their public fitting constraints across arms. Those inputs have identifiable prior scientific decisions. Generality here concerns the equation constructor conditional on those inputs. The implementation covers scalar ODE and algebraic nodes with unnamed contributions and shared coefficients across events. It does not enforce automatic conservation or parameter sharing between distinct equations. Renaming channels while preserving graph order, roles, interpolation and data leaves the tested numerical comparisons unchanged. Reordering equivalent sources can change tie resolution or the first-source latent fallback.

In the hybrid arm, A is retained. One 20B draw for the second slot can select B or submit a different complete model on the same graph. The study reserves one request for each of 12 graph/fold combinations, with seed 101, temperature 0.7 and at most 2,304 output tokens per request. Only that arm sees names in scientific context, the public task prose, summaries of fitting events, both nominees and the shape evidence. Each of the three arms receives the same two fitting slots per graph and outer fold. The matched fitter uses two starts per formula with a total ceiling of 1,000 residual evaluations and a 180-second optimization limit. The outer fold is held out during equation nomination and coefficient fitting. The lower verified training NMSE selects the model for the outer-fold forecast. The fixed graphs were constructed earlier using the public training collection, so this evaluation isolates the function stage conditional on those graphs.

### Computational cost and guarantees

For $d=|O_v|$ measured arguments, the menu has at most $L=1+3d+d(d-1)/2$ members. Let $N$ denote the total sample count and $W\le3$ the number of window widths. The two folds and all-event refit cost $O(L(WN+d+2)(d+2)^2)$ arithmetic operations per equation using dense least squares. Interval sums are computed once per fold using cumulative integrals. The principal design arrays take $O(WN(d+L))$ space per equation. The cost of subsequent nonlinear ODE fitting and simulation is separate.

For each fixed family and fold, the local regularized problem is convex in the coefficients. All enumerated families are assessed. The procedure supplies no guarantee of globally optimal rollout error, structural identifiability, or recovery of dynamics outside the grammar. Stable prediction and scientific recovery should be measured separately.

[Constructor](../src/autoformalism/research/shape_equation_nomination.py) · [Tests](../tests/test_shape_equation_nomination.py) · [Matched experiment](../scripts/matched_equation_experiment.py)
