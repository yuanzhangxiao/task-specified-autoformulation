# The T2 insulin balance in Candidate 15

This comparison is for human review. The canonical Phase C Dalla Man T2-easy simulator generated the observations. Its twelve states, equations and constants are evaluator information. The proposer receives the public task, training observations and evidence from its own models.

## Reference equations

Write $[z]_+=\max(z,0)$, $r$ for the supplied meal rate in g/min and $v$ for the supplied insulin infusion. The targets are plasma glucose mass $G_p$, plasma insulin concentration $I$, and total glucose disposal $U$. The two public mechanisms are a causal meal-to-glucose pathway and a delayed insulin-to-disposal pathway. Several reduced models can meet these requirements. [Public task and release scope](PHASE_C_DATASET_READINESS.md#dalla-man-decision-and-scope)

For the canonical T2 protocols, external glucose infusion is zero. Write $G=[G_p]_+/V_G$, $I=[I_p]_+/V_I$, and $S_\ell=\gamma[I_{po}]_+$. The three gut states are stomach solids, stomach liquid and gut glucose.

$$
\begin{aligned}
\dot Q_1 &=1000r-k_{\rm gri}[Q_1]_+,&
\dot Q_2 &=k_{\rm gri}[Q_1]_+-k_{\rm empt}[Q_2]_+,&
\dot Q_g &=k_{\rm empt}[Q_2]_+-k_{\rm abs}[Q_g]_+,\\
R_a &= f k_{\rm abs}[Q_g]_+/BW, &
E &= k_{e1}[[G_p]_+-k_{e2}]_+, &
EGP &= [k_{p1}-k_{p2}[G_p]_+-k_{p3}I_d-k_{p4}[I_{po}]_+]_+,\\
U_{ii}&=F_{\rm cns},&
U_{id}&=\left[\frac{(V_{m0}+V_{mx}X)[G_t]_+}{K_{m0}+[G_t]_+}\right]_+,&
U &=U_{ii}+U_{id},\\
\dot G_p &=EGP+R_a-U_{ii}-E-k_1[G_p]_++k_2[G_t]_+,&
\dot G_t &=-U_{id}+k_1[G_p]_+-k_2[G_t]_+.
\end{aligned}
$$

$$
\begin{aligned}
\dot I_p &=-(m_2+m_4)[I_p]_+ +m_1 I_\ell+v,\\
\dot I_\ell &=-(m_1+m_3)I_\ell +m_2[I_p]_+ +S_\ell,\\
\dot I_{po} &=-\gamma[I_{po}]_+ +Y+K[\dot G_p/V_G]_+ +S_b,\\
\dot Y &=-\alpha Y+\alpha\max\{\beta(G-h),-S_b\},\\
\dot I_1&=k_i(I-I_1),\qquad \dot I_d=k_i(I_1-I_d),\qquad
\dot X=p_{2U}(I-I_b-X).
\end{aligned}
$$

Here $m_3=m_1 H_E/(1-H_E)$ with $H_E=\operatorname{clip}(m_6-m_5S_\ell,10^{-6},0.99)$. The gastric rate depends on $Q_s=[Q_1]_++[Q_2]_+$ and an event memory $L$ in mg. At meal onset, $L$ is the dose plus the nonnegative residual stomach content. Between onsets it is constant. For $L>0$,

$$
 k_{\rm empt}=k_{\min}+\tfrac12(k_{\max}-k_{\min})
 \left[\tanh\left(\frac{5(Q_s-bL)}{2L(1-b)}\right)
 -\tanh\left(\frac{5(Q_s-dL)}{2Ld}\right)+2\right].
$$

At $L=0$, $k_{\rm empt}=k_{\max}$. The ingested mass enters $\dot Q_1$ over the ten-minute pulse. The plasma insulin loss vanishes at $I_p=0$, leaving $m_1 I_\ell+v\ge0$ for nonnegative liver insulin and infusion. Glucose stimulates secretion, and the action state $X$ modulates glucose disposal. [Reference implementation and parameter defaults](../src/autoformalism/rebuttal/dalla_man.py)

On the 150 g training trajectory, the basal state in the order $(G_p,G_t,Q_1,Q_2,Q_g,I_p,I_\ell,I_{po},Y,I_1,I_d,X)$ is approximately $(172.583340,130.406723,0,0,0,1.279391,4.565405,3.098684,0,25.587813,25.587813,0)$. The initial insulin concentration is $I_p(0)/0.05=25.587813$ pmol/L.

## Proposed models

Here $X$ is the candidate's insulin-action state, $Q$ its baseline disposal state initialized from $U_{ii}(0)$, and $A$ a proposed absorption reservoir. The reference and candidate states named $X$ have different dynamics and initializers. After the addition of $A$, the common reduced equations are

$$
\begin{aligned}
\dot G_p &= r-U+aA,& \dot A &= c_A r-aA,&
\dot X &= eI-fX,& \dot Q &=q,& U&=Q+X,\\
\dot I &=c_Ir+bG_p+\eta I+\kappa X+h_v v .
\end{aligned}
$$

| Step in the lineage | Change | Pooled training NMSE |
| --- | --- | ---: |
| Functional ancestor | $\dot G_p=\rho r-U$, where $\rho$ combines several meal-rate terms. The ancestor has no $A$. Insulin has $c_I=1,\ \kappa=0$. | — |
| Candidate 15 | Adds $A$ and changes $\dot G_p$ as above. Insulin still has $c_I=1,\ \kappa=0$. | 0.423867 |
| One-coefficient numerical control | Frees the direct meal coefficient $c_I$. Joint training gives $c_I=3.11416$ with $\kappa=0$. | 0.261011 |
| GPT-OSS-120B signed-action revision | Adds $\kappa X$ to insulin production. The primary free fit and concentration-constrained fit use the same equations. | 0.161635 free, 0.165562 constrained |
| GPT-OSS-120B depletion revision | Rectifies net production and insulin, retains direct meal forcing and integrates a clock pulse in a new state $D$. | Unfitted |
| GPT-OSS-120B basal revision | Adds $B\ge0$, returns to signed production and uses $D$ without declaring it against the fitted parent. | Rejected, unfitted |

The clock proposal, inspected on a copy with two parameter names spelled as declared, is

$$
\begin{aligned}
 P&=k_G G_p+k_X X,\qquad J=k_v v,\\
 \dot I&=c_Ir+[P]_+-g_2[I]_+ +g_7J+D,\\
 \dot D&=154.27042889042568
 \left[\sigma((t-s)/\tau_r)-\sigma((t-e_t)/\tau_f)\right].
\end{aligned}
$$

Here $\sigma$ is logistic, $g_2=g_7=1$ at the inherited values, and $e_t$ denotes a clock time. The derivative of $D$ is the same function of absolute time across schedules at fixed shared parameters. The raw reply remains rejected for undeclared Greek parameter identifiers. [Construction record](../work/agent-comms/2026-11-07-insulin-boundary-proposal-results.md)

The subsequent basal proposal returned $\dot I=c_Ir+P+B-g_2I+J+D$ with empty
new-state and new-process declarations. At the recorded depleted state this
would give $\dot I=B+D-0.801736$ pmol/L/min. The proposer supplied no equation
for $D$ in the new patch. [Basal repair results](../work/agent-comms/2026-11-08-insulin-basal-repair-results.md)

## The trajectories that crossed zero

The primary fits start on the 150 g training meal (train_005) with a triangular input from minute 0 to 10, peak rate 30 g/min at minute 5, and zero external insulin throughout. The public measurements at time zero are $G_p=172.583340$, $I=25.587813$, $U=1.915786$ and $U_{ii}=1$. The candidate sets $Q(0)=U_{ii}(0)$ and estimates $A(0)$ and $X(0)$ separately for each training trajectory.

| Quantity in the displayed equations | Primary free fit | Primary constrained fit |
| --- | ---: | ---: |
| $A(0)$ | 0 | $7.64\times10^{-16}$ |
| $X(0)$ | 1.16641868 | 1.48246036 |
| $c_I,\ b$ | 1.73036213, 0.000543283901 | 1.71411588, 0.000509943860 |
| $\eta,\ \kappa$ | +0.0194607318, −1.21870529 | +0.0182635481, −1.07258053 |
| $h_v$ | 11.7898015 | 8.83877320 |
| $a,\ c_A$ | 0.00204365144, 23.6885696 | 0.00193579031, 25.3272772 |
| $e,\ f$ | 0.000618688978, 0.0310159757 | 0.000654194397, 0.0308297400 |
| $q$ | 0.000632679448 | $3.59\times10^{-22}$ |

1. **Free fitted trajectory.** The first downward crossing is at minute 269.98053. There $G_p=184.49631$, $X=1.19360$, and $r=v=I=0$ to numerical precision. The glucose term contributes $+0.100234$ and the signed action term contributes $-1.454649$, giving $\dot I=-1.354415$ pmol/L/min. At minute 300 the prediction is $-31.7464$ pmol/L and the observation is $+22.6196$.
2. **Concentration-constrained fit.** The trajectory reaches zero within tolerance at minute 300. Here $G_p=240.84609$ and $X=0.86199$. The same boundary calculation gives $+0.122818-0.924554=-0.801736$ pmol/L/min. Continuing the fitted model for one minute with zero inputs gives $I(301)=-0.794226$. The last observation is at minute 300.
3. **Unfitted clock proposal.** Keep the primary constrained shared coefficients and original five initial values. Set $D(0)=0$, $s=0$, $e_t=10$, $\tau_r=20$, $\tau_f=1$ minute. Both widths are positive. Under this allowed choice, $I(31)=26.09877$ and $I(32)=-1.53443$. At the crossing, minute 31.94482, $P<0$ and $D=-27.09037$, so $\dot I=D<0$. This disproves the proposed guarantee across allowed parameters. The achievable training NMSE remains unknown.

All three free continuations crossed zero. Their minimum concentrations were $-31.75$, $-17.57$ and $-49.06$ pmol/L. On the primary pair, swapping in the constrained initial values alone leaves $I(300)=-26.64$. Swapping in the constrained shared coefficients alone gives $-2.99$. At $I=0$ the term $\eta I$ vanishes in both fits. The nonzero removal by $\kappa X$ explains the crossing at these fitted states. [Orion’s fit and initial-value controls](../work/agent-comms/2026-11-07-insulin-positive-continuation-results.md)

## Deterministic feedback

A benchmark can publicly identify a quantity as an absolute concentration and specify its admissible domain. The runtime can then substitute a reached boundary state into the assembled equation, report contributions, compare measured and fitted changes around recorded inputs, and identify the inputs of a proposed new rate. A counterexample at fitted parameters invalidates that fitted trajectory. A counterexample at an admissible unfitted setting invalidates a claim of positivity for every allowed setting. A biochemical replacement remains the proposer's decision.

This read-only command takes the saved plan and public inputs. Omitting the reply returns evidence for the fitted parent. Supplying a visible proposal JSON also checks the proposed equations and renders applicable feedback. It uses the existing construction gate and restricted expression parser and makes zero provider or optimizer calls.

~~~bash
.venv/bin/python -m scripts.describe_insulin_boundary \
  --root output/insulin-boundary-result-20261107/returned \
  --reply output/insulin-boundary-result-20261107/visible-reply.json
~~~

The command reports domain, timing and meal-window feedback. General nonlinear positivity proofs and automatic discovery of trajectory counterexamples remain open. [Feedback implementation](../scripts/describe_insulin_boundary.py)
