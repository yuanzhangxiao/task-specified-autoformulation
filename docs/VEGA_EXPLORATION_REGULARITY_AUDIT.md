# Applicability audit: infinite answers and a finite-answer baseline

Vega, 28 October 2026. Updated after Lyra's exploration-contract response and
independent rechecking. This is a mathematical applicability check, not an acquisition campaign
or a claim of biological or algorithmic novelty.

## 1. Result and source checked

The proposed infinite-answer comparison is appropriate, but one sufficient
condition needs clarification before we use it. A bounded scalar Gaussian example
below has a continuous, compact correct-answer correspondence and a robustly
correct answer at every instance, yet fails the paper's uniform Assumption 3.
This blocks our intended shortcut from continuous prediction intervals to that
assumption. It does not refute results explicitly conditional on Assumption 3.

The source is [Poiani, Bernasconi and Celli, *Pure Exploration with Infinite
Answers*, AISTATS 2026](https://proceedings.mlr.press/v300/poiani26a.html).
The official proceedings PDF linked from that page has SHA-256
`c12d26ed4d78e5cc7a6148704655515b8e0904a46445d57c5954783fa627c45f`.
Its PDF page 4 states Assumptions 1–3 and Theorem 1; pages 13–14 discuss regression,
finite answers and composition. The corresponding
[author text](https://arxiv.org/pdf/2505.22473) has the same relevant statements.
The framework studies regular exploration with compact, potentially infinite
answers. Its composition argument assumes the component problems satisfy the
regularity conditions. Theorem 1 supplies continuity plus compactness as a
sufficient condition for Assumption 3. That last implication is the specific
point examined here; the paper remains a central prior-art comparator.

## 2. A direct boundary check

Consider one independent Gaussian observation per acquisition, with variance one
and mean \(\mu\in M=[0,1]\). Take the allowed mean interval \(\Theta=[0,1]\)
and fixed answer space

\[
X=[-1/10,11/10],\qquad e=1/10,
\qquad X^*(\mu)=[\mu-e,\mu+e].
\]

Thus an answer estimates the mean to absolute error at most \(e\). There is
no clipping of the correct-answer intervals: each full interval is inside
\(X\). Gaussian distributions belong to a regular canonical family on the real
line; restricting their means to this compact interval does not create an
invalid observation law.

For an answer \(x\), let

\[
\operatorname{Alt}(x)=\{\lambda\in M:|x-\lambda|>e\}.
\]

For a neighborhood \(B_\rho(x)=X\cap[x-\rho,x+\rho]\), the common alternatives
are those instances making **every** answer in the neighborhood incorrect:

\[
\operatorname{Alt}(B_\rho(x))
=\{\lambda\in M:\forall y\in B_\rho(x),\ |y-\lambda|>e\}.
\]

With one arm, the weighted divergence reduces to

\[
D(\mu,S)=\inf_{\lambda\in S}\frac{(\mu-\lambda)^2}{2}.
\]

The tested condition requires the difference
\(D(\mu,\operatorname{Alt}(B_\rho(x)))-D(\mu,\operatorname{Alt}(x))\)
to become uniformly small over instances and answers for a common positive
radius, with nonempty common-alternative sets.

### Exact calculation at a strictly correct answer

For any \(0<\rho\le1/100\), choose

\[
\mu=1/20,\qquad x_\rho=1/10+\rho/2.
\]

This answer is strictly correct: its error is \(1/20+\rho/2<1/10\).
Directly from the definitions,

\[
\operatorname{Alt}(x_\rho)
=[0,\rho/2)\ \cup\ (1/5+\rho/2,1],
\]

whereas

\[
\operatorname{Alt}(B_\rho(x_\rho))=(1/5+3\rho/2,1].
\]

The lower alternative interval disappears because the neighborhood contains
answers that remain within tolerance of every mean near zero. The closest
point alternatives approach \(\rho/2\); the closest common alternatives
approach \(1/5+3\rho/2\). The endpoints need not belong to the open intervals
for their divergences to determine the infima. Consequently the difference is

\[
\begin{aligned}
&\frac{(3/20+3\rho/2)^2-(1/20-\rho/2)^2}{2}\\
&\hspace{2em}=\frac1{100}+\frac\rho4+\rho^2
>\frac1{100}.
\end{aligned}
\]

For \(\rho\ge1/100\), fix \(x=21/200\) and the same \(\mu=1/20\).
Increasing the neighborhood can only shrink its common-alternative set.
If that set remains nonempty, its divergence cannot decrease; the gap is at
least its value \(63/5000\) at radius \(1/100\). If the common-alternative
set becomes empty, the nonemptiness requirement already fails. Hence **no
positive radius** satisfies the condition with regularity-error target
\(1/200\). The same conclusion holds for every smaller target.

### The other stated conditions do hold

Both \(M\) and \(X\) are compact. The correct-answer correspondence consists
of intervals whose endpoints vary continuously with \(\mu\), so it is
continuous and compact-valued. Choosing \(x=\mu\) gives an answer separated
from the closure of its alternatives by at least \(e\) in mean space. Every
singleton answer has a nonempty alternative set: at least one endpoint of
\([0,1]\) is more than \(e\) away.

This is an analytical counterexample to applying the **written sufficient
condition** without an additional qualification. The obstruction is a vanishing
alternative interval at the boundary of the allowed instance set, not an
invalid Gaussian endpoint, a clipped correct-answer interval, or an incorrect
answer chosen for the calculation. The counterexample concerns uniformity:
the answer \(x_\rho\) changes with \(\rho\). Pointwise limits at a fixed
answer do not establish the stated uniform condition.

Lyra independently confirmed the calculation and identified the proof step below;
Vega rechecked it against Appendix B.2. The continuity and compactness hypotheses
of the printed Theorem 1 do not imply the printed uniform Assumption 3. This does
not invalidate results explicitly conditional on that assumption, prove the task
unlearnable, or establish a new scientific-revision method.

### The failed compactness step

Using the same example, set an auxiliary radius \(a=1/100\), and write the
quantities in Lemmas 20–21 as

\[
g_a(x,\lambda)=\max_{t\in[0,1],\ |t-\lambda|\le a}
\operatorname{dist}(x,[t-e,t+e]),\qquad
r_a(x)=\min_{\lambda\in\operatorname{cl}\operatorname{Alt}(x)}g_a(x,\lambda).
\]

At \(x=e\), the alternative closure is \([2e,1]\), and its minimum is
\(r_a(e)=a\). For \(0<\delta<a\), the answer \(e+\delta\) has a new lower
alternative component containing zero; it gives \(r_a(e+\delta)=\delta\).
The upper component gives at least \(a\). Thus the claimed continuity fails,
and the infimum over answers is zero without being attained: each individual
answer still has strictly positive \(r_a\).

In particular, \((e+\delta_n,0)\) belongs to the graph of
\(x\mapsto\operatorname{cl}\operatorname{Alt}(x)\) for \(\delta_n\downarrow0\),
but its limit \((e,0)\) does not. Lemma 20 treats a sublevel set closed relative
to this graph as closed in the ambient compact product. The graph need not be
closed. Compact fibers and continuity of the original correct-answer
correspondence do not justify that inference. In fact, choosing sublevel threshold
\(a/2\) and \(0<\delta_n<a/2\) puts that sequence in the sublevel set, while
its limit remains outside: the required compactness fails directly.
Adding a closed-graph condition
has not been checked as a complete repair of the theorem.

## 3. A constructive baseline that avoids that shortcut

A useful positive reduction remains available through finite answers. Its
scientific assumptions are deliberately explicit.

Condition on known source amplitude, known summed rate \(s>0\), known pulse
scale, verified delivery and sharing, and a known observation map. The only
unknown kinetic coordinate is \(\gamma\in[0,s]\). Each arm has IID scalar
Gaussian readings, with independent potential sample sequences across arms,
known variance and mean
\(c_k e^{-\gamma t_k}\), where \(c_k,t_k>0\) are known. If necessary, subtract
a **known** baseline and divide by the known standard deviation to obtain
unit-variance arms. Subtracting a fitted or shared noisy baseline would not
justify this independent-arm model.

The fixed future task uses one coherent response function across a finite
protocol menu:

\[
m_v(a)=b_a+c_a e^{-v\tau_a},\qquad v\in[0,s],
\quad c_a,\tau_a>0.
\]

For maximum absolute prediction error, the known constant

\[
L_G=\max_a c_a\tau_a
\]

satisfies \(\max_a|m_v(a)-m_\gamma(a)|\le L_G|v-\gamma|\).
Choose a fixed grid including 0 and \(s\), with maximum spacing
\(h\le\varepsilon_G/L_G\). Every true rate has a grid representative at
distance at most \(h/2\), giving prediction error at most
\(\varepsilon_G/2\). This preserves positive prediction slack and one shared
parameter assignment; it does not return unrelated forecasts at each protocol.

Let the scientifically allowed labels be

\[
R:\gamma>c_R,\qquad L:\gamma<s-c_L,
\quad c_L,c_R\ge0,\quad c_L+c_R<s.
\]

At every true rate, one label has strict slack at least
\((s-c_L-c_R)/2\). Therefore some grid-model/label pair is correct throughout
a neighborhood of the true rate. Each positive-time observation mean is
continuous and injective in \(\gamma\); its compact image preserves separation
from alternatives outside that neighborhood.

The answer space is the finite product of the grid and \(\{L,R\}\).
Compactness is immediate. A robustly correct answer exists by the preceding
slack argument. For a radius smaller than half the minimum distance between
distinct encoded answers, every answer neighborhood is a singleton. Thus the
two alternative sets are exactly equal, and the uniform divergence difference
is zero. Each pair has a nonempty alternative set because its label fails at
one endpoint. This verifies the relevant regularity directly; it does not use
the disputed continuity shortcut.

The admissible mean set is the compact curve induced by \(\gamma\in[0,s]\)
inside a bounded Gaussian mean box. Correctness on that curve alone does not
make arbitrary extensions of the answer correspondence operationally equivalent.
The printed AISTATS Algorithm 1 restricts stopping and recommendation to
\(X^*(\widehat\mu_t)\), while the empirical mean vector need not lie on the
curve. Its definition does not supply a projection onto the structured set.

For example, let \(M=\{(\theta,\theta):\theta\in[1/4,3/4]\}\), with two
independent unit Gaussian arms. On \(M\), let \(L\) be correct for
\(\theta<3/5\) and \(R\) for \(\theta>2/5\). Extend the correspondence
off \(M\) by allowing only \(R\). At \(\theta=1/4\), only \(L\) is correct,
but empirical means after positive counts lie off the diagonal almost surely.
The statement survives adaptive counts by a countable union over all finite
count pairs. Coordinatewise clipping to \([0,1]\) does not put them on the
interior diagonal segment. A recommendation restricted to this extension is
therefore wrong whenever it terminates; an error bound \(\alpha\) would force
termination probability at most \(\alpha\). This is an operational defect of
that extension, not an impossibility for the finite task.

For our finite baseline, use the stopping construction in
[Degenne and Koolen, Algorithm 1](https://proceedings.neurips.cc/paper/2019/file/60cb558c40e4f18479664069d9642d5a-Paper.pdf): search the
entire fixed answer set for an answer with no incorrect alternative in a valid
confidence region. The paper calls that conservative stopping region \(D_t\);
it is distinct from its more aggressive sampling region \(C_t\). Below we use
our generic retained-set notation \(C_t\) for the valid stopping region.
Our reporting interface also requires verified nonemptiness:

\[
C_t\ne\varnothing,\qquad
\exists x\in X:\ C_t\cap\operatorname{Alt}(x)=\varnothing.
\]

This needs no ambient answer filter. Allowing all finite answers off the curve
would remove this particular filter obstruction in an empirical-answer procedure,
but transferring its full theorem would still need verification. The confidence
construction used below does not inherit another policy's efficiency guarantees.

The condition \(s>c_L+c_R\) means this particular instance family already
excludes joint approximate constancy. It is appropriate to a conditional
attribution question, not a complete healthy-versus-faulty detection study.
Healthy cases require their own acceptable outputs and learnability check.
More explicitly, \(p_L\wedge p_R\) implies \(s\le c_L+c_R\), so the supplied
calibration already establishes the joint conflict. Pulse measurements resolve
the conditional next decision. Total discovery cost must include the evidence
and uncertainty behind that calibration; incremental cost may condition on it
only when that is the declared task for every method.

The existing finite-answer exploration theory is consequently an applicable
baseline for this restricted fixed-output task. The grid rule is a sufficient
accuracy construction, not a claim of minimum acquisition cost or optimality
over every continuous predictor. A method restricted to this grid may be less
efficient than one exploiting all allowed responses. Numerical optimization and
an implemented sequential stopping rule still require verification.

This reduction concerns population-level answer correctness. It does not by
itself reproduce every retained-set certificate in our reporting interface.
For that interface, the task specification gives a conservative construction:
keep a simultaneous confidence interval for the rate, select the grid point
nearest its midpoint, and bound prediction and claim errors over the entire
interval. That policy supplies an explicit certificate under the same calibrated
contract; the exploration theorem's optimality does not automatically transfer
to this conservative policy.

## 4. What this establishes for the project

This positive reduction concerns a predictive approximation with a fixed allowed
class. It does not ensure that the returned grid point remains in a shrinking
transcript-dependent set \(C(D)\). Nor does it establish a reduction for unknown
amplitude, summed rate or scale, uncertain sharing, correlated trajectories,
batch costs, changing observation maps, or general scientific claim libraries.
Those are explicit unproved extensions, not automatic novelty claims.

The accompanying [exact-arithmetic checker](../scripts/check_premise_exploration_contract.py)
checks selected rational instances of the boundary calculation. The all-radius
argument above is the proof; the selected checks do not establish uniformity by
sampling. The finite-grid construction is also analytical, with no fitted data,
sample-cost result or new acquisition policy measured here.

The useful next step is therefore concrete: preserve the infinite-answer paper
as close prior art, preserve the independently checked regularity qualification, and retain
the finite coherent-model grid as a defensible baseline when its scientific
contract is appropriate. Biological feedback should determine whether those
known quantities, independent acquisitions, predictive output, and attribution
labels describe a consequential task. A failure of one sufficient-condition
shortcut is not a reason to invent a harder pulse example or abandon an existing
baseline that already supplies valid answers.
