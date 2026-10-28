# Two matched tasks for scientific premise revision

Vega, 28 October 2026. Bounded selection exercise following Lyra's M2 response.
This specifies possible research tasks and proves analytical controls. It does
not select a biologically validated case, launch an acquisition campaign, or
establish a new methodological contribution. M2 remains closed.

The [reply to Lyra](VEGA_TO_LYRA_M2_TASK_REPLY.md) gives the research judgment;
the [biological discussion aid](VEGA_GEMOT_REVISION_TASK_DISCUSSION.md) asks which
of these requirements matters in practice. The original
[premise formulation](VEGA_PREMISE_REVISION_THEORY_HANDOFF.md) remains in force.

## 1. A shared scientific envelope

Keep the reviewed illustrative dynamics

\[
\dot L=-\kappa L,\qquad \dot R=-\gamma R,\qquad
\dot x=aLR-x,\qquad x(0)=0.
\]

Times and concentrations below are synthetic scaled units. This is a
signaling-inspired control, not a validated TGF-beta model. The source measured
mean is \(A F_s(t)\), where

\[
s=\kappa+\gamma,\qquad
F_s(t)=\frac{e^{-st}-e^{-t}}{1-s}.
\]

Consider separate matched runs, with their clocks reset, sharing the declared
kinetics and initial measured scale. An added, actually delivered dose multiplier
\(d\) at time \(\tau\) produces

\[
m_w(d,\tau,\Delta)
=A F_s(\tau+\Delta)
 +d\eta e^{-\gamma\tau}F_s(\Delta),\qquad \Delta>0.
\]

Here \(\eta\) is the measured pulse coefficient for one reference dose. For
example, with observation gain \(g\), capacity scale \(h=aR_0\), initial ligand
\(L_0\), and fixed reference dose \(d_0\),
\(A=ghL_0\) and \(\eta=ghd_0\). The commanded dose is not automatically the
delivered dose. This control conditions on verified delivery, a linear readout,
and a pulse that does not instantly alter capacity. Unknown delivery or gain
requires additional joint constraints, not an unnoticed change in the meaning
of \(d\).

One possible bounded envelope preserves the independent M2 preparation family:
\(A\in[0.5,2]\), \(s\in[0,0.6]\), \(\gamma\in[0,s]\), and
\(\eta/A\in[0.5,2]\). This last ratio describes the reference dose, while the
external multiplier \(d\) remains an input. The coupled M2 family instead fixes
\(\eta=1\). These remain different scientific contracts. Neither is selected
from source fit quality. The controls below explicitly state when they condition
on known \(A,s\) or a smaller slice of this envelope.

For a concrete *candidate* protocol menu take
\(d\in\{1,2\}\), \(\tau\in\{3,5\}\), and \(\Delta=2\).
Its biological usefulness, precision requirement \(\varepsilon_G\), and costs
are unresolved. The menu is a specification example, not a frozen benchmark.
Changing it changes the task and must be recorded.

The original constancy premises are replaced, where appropriate, by scoped
approximation predicates:

\[
p_L:\kappa\le c_L,\qquad p_R:\gamma\le c_R.
\]

For relative change tolerance \(\epsilon_j\) on a window \([0,T_j]\),
\(c_j=-\log(1-\epsilon_j)/T_j\). Exact constancy has \(c_j=0\).
The thresholds, window, origin of each premise, and interpretation of measured
ligand versus bioactive ligand require review. An omitted protocol fact is not
a false assertion by the biologist.

## 2. Allowed outputs: two tasks with the same data and loss

Let \(C(D)\) be a nonempty retained scientific set. Computation maintains
\(I\subseteq C\subseteq O\), with verified \(I\ne\varnothing\) needed for
positive certificates. Define a candidate returned model by one parameter
assignment \(v\) and its full response function \(f_v(a)=m_v(a)\). The assignment
must be shared across the declared protocol menu: it cannot choose another
\(\eta\) or \(\gamma\) for each requested prediction. Inputs \(d,\tau\)
are supplied by the task, not optimized by the returned model.

Use the same loss in both tasks:

\[
\ell_G(v,w)=\max_{a\in\mathcal A_G}|f_v(a)-m_w(a)|,\qquad
R_G(v,C)=\sup_{w\in C}\ell_G(v,w).
\]

This concerns measured **means**. Individual future observations need a noise
prediction loss. Intervention ranking or dose selection would also be different
losses. Allowed point models here belong to \(C(D)\); an implementable certificate
must verify the returned point, rather than treating an outer cell as feasible.
An unrestricted function is a separately labeled comparator, not an allowed
point model.

| Requirement | Task P: useful predictive revision | Task J: a warranted named revision |
| --- | --- | --- |
| Returned predictive object | One admitted response function over the fixed protocol menu | Exactly the same object and loss |
| Success | \(R_G(v,C)\le\varepsilon_G\) | P-success plus at least one named premise rejected throughout \(C\) |
| Interpretation of chosen nuisance values | Working representatives, not identified physical values | Same; named claims need separate evidence |
| Assumption change | Relax defaults enough to retain compatible explanations; show which alternatives remain | Same relaxation, plus an evidence-supported assertion that a named scoped predicate fails |
| Insufficient evidence | Report remaining risk or unresolved computation | Also report named ambiguity or a joint conflict without claiming J-success |

Task P is a necessary predictive control. Task J is closer to the user's
original goal of supported assumption revision. They are not competing methods.

Concretely, a J output has \(r=(v,B,J,G,E)\): a working model \(v\), revised
assumption contract \(B\), nonempty set of rejected premise IDs \(J\), fixed
scope \(G\), and evidence/provenance \(E\). A conservative relaxation contract
must admit all retained explanations; \(C\subseteq\Omega_B\) is sufficient.
The associated point model is a representative, not the entire revised contract.
For every \(j\in J\), the scientific assertion requires

\[
C(D)\cap\{w:p_j(w)=1\}=\varnothing.
\]

Use \(O\) for a sound implemented exclusion. A joint conflict
\(C\cap\{p_L=p_R=1\}=\varnothing\) is weaker: it may justify withdrawing
the conjunction without naming either premise individually. Removing a premise
permits its negation; it does not assert it. Minimal repair cost remains a
preference separate from warranted blame. Mixed violations remain admitted.

For J, one can equivalently impose a hard semantic admissibility condition and
then optimize predictive loss. No arbitrary exchange rate should let better
prediction compensate for a false scientific assertion.

Abstention is a valid operational outcome, but not successful completion of J.
Usefulness must be evaluated on declared instances admitting an informative
action and a warranted answer; an always-abstain rule fails that requirement.
Healthy worlds and boundary cases need separately declared acceptable outputs.
Recovering the **entire** premise truth signature is a third, stronger task.
It is not silently included in J.

## 3. Evidence menu and what its entries actually buy

| Evidence/action | What it can constrain | Essential qualification |
| --- | --- | --- |
| More baseline observations | \(A,s\) | They do not separate \(\kappa\) and \(\gamma\) in this family. |
| More doses or readouts at the same pulse time | \(z(\tau)=\eta e^{-\gamma\tau}\) and nuisance precision | Even the whole post-pulse trajectory need not separate the two factors. |
| Another pulse time in a matched run | Another \(z(\tau)\) | Requires justified common pulse scale and kinetics across runs. |
| Independent ligand bioactivity measurement | Ligand decay or a scoped exposure statement | Assay specificity, calibration, initial scale and noise belong in the inference. |
| Delivered-dose/readout calibration | The relevant product or its bounds | A calibration need not identify every physical factor. |
| Protocol records or preparation clarification | Supported sharing constraints | A statement can remain uncertain; it is not automatically exact numerical evidence. |
| More certified computation | A tighter outer approximation \(O\) | It cannot distinguish worlds with identical observation laws. |

An eventual cost function should include setup, samples, time, assay precision,
and shared preparations. No numerical costs or optimal policy are supplied here.
The first empirical design should use a fixed acquisition schedule with a valid
joint observation model. If baseline-subtracted increments are used, retain
their covariance and uncertainty in fitted baseline means. Independent Gaussian
errors on raw readings do not imply independent errors on differences.

A confidence set for combined experiments needs its own validity argument. Fixed
design joint Gaussian inversion or an explicitly allocated simultaneous error
budget is available as a baseline. M2's threshold cannot be reused repeatedly
under adaptive stopping while retaining its original coverage claim. Adaptive
design, nonstationary batches and uncertain observation laws are unimplemented.

## 4. Control A: unlimited evidence at one pulse time can leave J impossible

Condition on exact \(A=1,s=0.2\), and shared \(\eta\in[0.5,2]\). Take

\[
w_0:(\kappa,\gamma,\eta)=(0.2,0,1),\qquad
w_1:(\kappa,\gamma,\eta)=(0,0.2,e^{0.6}).
\]

Both satisfy the envelope, since \(e^{0.6}<2\). Their original trajectories
coincide. At \(\tau_0=3\), both have \(z(3)=1\), so every dose and every
post-pulse sampling time has the same mean. With the same complete noise law,
the observations have the same law. This also defeats adaptive selection of
doses/readout times restricted to \(\tau_0\), provided each conditional
acquisition law has the stated invariance. More computation or replication
cannot distinguish the worlds.

For thresholds below 0.2, \(w_0\) violates ligand constancy only and \(w_1\)
capacity constancy only. A forced named answer cannot be uniformly correct with
error below \(1/2\) on this pair: under their common observation law, the two
disjoint correct-answer events cannot both have probability above \(1/2\).
With abstention, uniform wrong-answer probability at most \(\alpha\) limits
the probability of a correct named answer to at most \(\alpha\) in each world
of this pair. The joint disjunction is valid but does not meet J.

For predicting only this pulse time, exact knowledge of \(A,s,z(3)\) suffices
without identifying the cause. This zero-uncertainty statement concerns the
specified oracle control, not the full finite-data M2 set. For prediction at
\(\tau_1=5\), however, the pair's unit-dose means at delay 2 differ by

\[
(1-e^{-0.4})F_{0.2}(2)=0.220467189964\ldots.
\]

Every scalar prediction has worst-case error on this pair at least
\(0.110233594982\ldots\). Thus tolerance 0.1 is impossible for that new
prediction from this evidence menu. This is a mathematical witness with synthetic
units, not a proposed biological tolerance or a calibrated success-rate estimate.

### The strongest elementary baseline

With exact positive responses at two distinct times and a common \(\eta\),

\[
\gamma=\frac{\log z(\tau_1)-\log z(\tau_2)}{\tau_2-\tau_1},
\qquad \eta=z(\tau_1)e^{\gamma\tau_1}.
\]

This is ordinary log-linear identification. One calibrated \(\eta\) and one
positive pulse time can also suffice; an appropriate ligand assay is another
route. Two timings are not a universal minimum-cost prescription.

For simultaneous positive bounds \(z_i\in[\ell_i,u_i]\), eliminate
\(\theta=\log\eta\) by intersecting the affine intervals

\[
[\log\ell_1+\gamma\tau_1,\log u_1+\gamma\tau_1]
\cap[\log\ell_2+\gamma\tau_2,\log u_2+\gamma\tau_2]
\cap[\log\eta_{\min},\log\eta_{\max}]\ne\varnothing.
\]

Together with \(0\le\gamma\le s\), this is an exact projection for the
declared rectangular bounds and known \(s\). It is not necessarily an exact
projection of the original joint likelihood: dependence and uncertain \(A,s\)
must remain in a full finite-data calculation. Bounds touching zero cannot be
logged this way. Interpolation of \(\log z\) between two times is linear;
extrapolation amplifies endpoint error through absolute interpolation weights.
Late timing may also lower signal, so algebraic rate precision alone does not
settle experimental cost.

### Remove the sharing premise and the conclusion can disappear

Allow run-specific \(\eta_i\in[0.5,2]\) on the menu \(\tau\in\{3,5\}\).
One world has \(\gamma=0,\eta_i=0.5\); another has
\(\gamma=0.2,\eta_i=0.5e^{0.2\tau_i}\). The latter scales are approximately
0.91106 and 1.35914, both admitted. Every measured pulse factor is now 0.5 in
both worlds. They retain the same \(A,s\) and opposing constancy violations.

This is a different envelope, explicitly relaxing sharing. It disproves a
general claim that two timings resolve the ambiguity without sharing evidence.
It does not say that bounded scale uncertainty defeats every possible timing
menu. A deterministic scale drift \(\eta(\tau)=\eta_0e^{-\delta\tau}\)
similarly makes these observations depend on \(\gamma+\delta\). A good fit
does not license silently setting \(\delta=0\).

## 5. Control B: one named revision can be easier than full diagnosis

Suppose \(s>0\) is known and the nonempty retained capacity-rate set is a
compact interval \(\Gamma=[l,u]\subseteq[0,s]\). Then

\[
\text{reject }p_R\iff l>c_R,\qquad
\text{reject }p_L\iff u<s-c_L.
\]

If \(c_L+c_R<s\), **at least one** named rejection is guaranteed whenever

\[
u-l<s-c_L-c_R.
\]

Proof: if neither rejection is justified, \(l\le c_R\) and
\(u\ge s-c_L\), hence the width is at least the stated gap. Sharpness over
interval locations follows from \([c_R,s-c_L]\), which preserves each premise
at a different endpoint. The condition is sufficient uniformly over locations,
not necessary for an individual interval: a wider interval near one side may
already reject one premise.

For the declared synthetic rates \(s=0.2,c_L=c_R=0.02\), a width below 0.16
suffices; the boundary interval \([0.02,0.18]\) permits no individual rejection.
At exact constancy, the threshold becomes \(s\). Even if the true
\(\gamma=0\), rejecting the other premise can be easy. Therefore a blanket
claim that J requires resolving arbitrarily small nonzero rates would be wrong.
That difficulty belongs to full truth-signature recovery or uniform confirmation
of exact zero against arbitrarily close alternatives. With unknown \(s\),
project the full joint rate set; this one-dimensional criterion is only a control.

This sharp elementary criterion is useful for testing acquisition objectives.
It is not yet a substantive new theorem. It points directly toward established
multiple-correct-answer exploration, rather than requiring identification of
every latent factor.

## 6. Control C: a real response-function restriction, already tractable

Lyra proves that the best scalar M2 forecast is attainable by a model in all
four executed cases. That result does not extend automatically to multiple
protocols. Here is a separate exact control; it does not alter the M2 roster.

Fix known \(A,s>0,\eta=1\). Require unit-dose mean **increments** at pulse
times \(t,2t\), each measured after the same delay \(\Delta\). Let
\(a=e^{-st}=0.1\). As \(\gamma\in[0,s]\), the allowed predictions are

\[
v_\gamma=F_s(\Delta)(x,x^2),\qquad x=e^{-\gamma t}\in[0.1,1].
\]

The two coordinates have a common positive scale; no differing error weights
are introduced. Whether these two timings are scientifically required remains
open. This is an oracle model-family calculation, not a noisy pulse experiment.

With maximum absolute coordinate error, a free vector center is
\((0.55,0.505)F_s(\Delta)\), attaining risk \(0.495F_s(\Delta)\).
The second-coordinate range alone proves that lower bound. A returned point
model, however, has normalized risk

\[
R(x)=\max\{1-x,x-0.1,1-x^2,x^2-0.01\}.
\]

For \(x\le0.9\), this is \(\max\{1-x^2,x-0.1\}\). The decreasing and
increasing terms cross at

\[
x_*=(\sqrt{5.4}-1)/2=0.661895003862\ldots.
\]

For \(x\ge0.9\), risk is at least 0.8, so the crossing is the global minimum.
The best model risk is \(0.561895003862\ldots F_s(\Delta)\), strictly above
the free-vector optimum by \(0.066895003862\ldots F_s(\Delta)\).

This validates the distinction between a response function and unrelated point
forecasts. It is elementary Chebyshev approximation, with no demonstrated need
for a new solver. Its optimal interior rate is still a working representative,
not evidence that both physical constancy premises fail. A set-valued response
or randomized model would require another explicit loss and output contract.

## 7. Reductions to existing work

The following are primary-source comparisons, not claims that a cited software
implementation already accepts this particular ODE and evidence interface.

| Closest work | What we inherit and what must be checked |
| --- | --- |
| [Letham et al., 2016](https://arxiv.org/pdf/1511.03395), prediction deviation | Compatible ODE-model disagreement and prospective experiment impact are already used for biological prediction. Theorem 1 gives a conditional trajectory bound; Appendix A describes approximation and multistart local optimization. This is a direct baseline for P. A finite-data, numerical, or semantic distinction must be stated precisely, not asserted from vocabulary. |
| [Jalaldoust, Bellot and Bareinboim, 2024](https://papers.nips.cc/paper_files/paper/2024/file/f90d2001ec7595d4b8fdfa4db966d154-Paper-Conference.pdf), partial transportability | Theorems 1–2 and Algorithm 2 address bounds, shared mechanisms and robust prediction; Appendix A includes Bayesian finite-data inference. The setup uses recursive SCMs and supplied sharing diagrams. An ODE embedding and guarantee comparison remain necessary. Bayesian credible summaries, uniform frequentist coverage and numerical certification are different standards; their difference alone is not a contribution. |
| [Skanda and Lebiedz, 2013](https://arxiv.org/pdf/1101.3663), robust dynamic design | Worst-case model discrimination with uncertain parameters and designed times/perturbations is an established comparator for J. Its local numerical strategy is not a proof of global exclusion, but a useful benchmark must use its strongest applicable formulation. |
| [Degenne and Koolen, 2019](https://arxiv.org/pdf/1902.03475), multiple correct answers | A finite evidence menu and finite labels may reduce to structured pure exploration with finite arms and one-parameter exponential-family observations. Their lower bound and Sticky Track-and-Stop results cover this setting. Continuous model outputs, correlated acquisitions, unequal setup costs and batch drift require explicit handling before transferring a theorem. |

For the last reduction, let \(\mu(w)\) collect the menu's observation means,
assuming they determine the full observation laws. If \(I^*(w)\) is the set of
scientifically acceptable answers in world \(w\), the answer set for an
observable instance is

\[
I^*_{\mathrm{obs}}(\mu)
=\bigcap_{w:\mu(w)=\mu} I^*(w).
\]

It is an **intersection**, not a union. An empty intersection rules out a single
answer guaranteed correct in every observationally equivalent world. For finite
answer sets it also precludes arbitrarily high-confidence uniform resolution;
a fixed error target needs a quantitative bound such as Control A. For example,
correct-answer sets \(\{a,b\},\{b,c\},\{a,c\}\) have empty intersection, yet
randomization gives success probability \(2/3\) in each indistinguishable world.
A useful predictor may lie in the intersection while no named revision does.
This is ordinary identifiability. Moreover, J includes a continuous predictor:
reducing its finite label portion alone does not solve the entire task unless predictive
adequacy is independently guaranteed.

## 8. Selection judgment and what would justify another milestone

My preference is to use P as a matched control and ask the biologists whether J
is consequential. We should not substitute a convenient forecast for the user's
premise-revision objective. Equally, we should not demand complete mechanism
identification when one supported revision is sufficient.

The present results are reductions and counterexamples. None establishes a
publishable new acquisition method or computational obstruction. The next useful
theory question is conditional: after biology fixes an output, observation law,
sharing tolerance, allowed actions and costs, does the resulting J task reduce
to known structured exploration/design with an analytical prediction check?
If it does, implement that baseline and evaluate the biological contribution.
If it does not, state the failed reduction and a concrete result beyond it before
funding a new algorithm. Unknown sharing cannot be restricted only for our method.

There is no five-day automation, live biology request, new Monte Carlo study,
or production-agent change associated with this document. The open decisions
are scientific task choices, not permission to continue an otherwise specified
implementation.

## 9. Executable checks and current verification

The small [checker](../scripts/check_premise_revision_controls.py) computes
outward numerical bounds for Controls A–C using the unchanged M2 interval
arithmetic. The equations above supply the global and indistinguishability
proofs; an interval containing an expected value is only a numerical consistency
check. It does not prove equality by itself.

```bash
.venv/bin/python scripts/check_premise_revision_controls.py \
  --output output/vega-lyra-m2-review-20261028/task-control-receipt.json
```

The script binds its source and the reviewed interval module. Completed receipts
are immutable and reused only when identical; publication is atomic, and an
interrupted short calculation can be repeated deterministically. It has no search,
sample generation, fitted data, LLM calls or benchmark access. It is a proof check,
not an acquisition experiment with inferred statistical guarantees.

Lyra's archive SHA-256 is
`9f437f40b43f977d13a1fcfcfa7137a3b662aacc802dbc44ebdc8ad0407c0450`.
All six manifest-listed file hashes matched; its response equals the user's
pasted text. Rerunning the reviewed midpoint checker against the unchanged Vega
M2 packet reproduced all four complete result records exactly. This reuses the
same interval implementation and historical extrema, not an independent global
solver. The original M2 campaign was not rerun in this selection exercise.

Verification logs and receipts are under
`output/vega-lyra-m2-review-20261028/`. The focused invocation passed **155 tests**
in 13.08 seconds: 148 existing M2 tests and seven new arithmetic-check/receipt
tests, including rejected source changes, differing receipts and interrupted
publication. The new Python files pass Ruff. Repository-wide `ruff check .`
reports the same 37 preexisting errors under untracked `analysis/claude/`.
The full repository pytest suite was not rerun for this isolated selection
exercise; earlier M2 full-suite outcomes remain as documented in its report.
The checker CLI wrote its receipt and then reused it without changes.

Changed files are this specification, the reply, the biological discussion aid,
`scripts/check_premise_revision_controls.py`, and
`tests/test_premise_revision_controls.py`. Production source, the M1/M2 solvers,
benchmark data and prompts are unchanged. Numerical results assume the documented
Decimal/libmpdec contracts. Biological validity, finite-sample acquisition
performance and general novelty remain open.
