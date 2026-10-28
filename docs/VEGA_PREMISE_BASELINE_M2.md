# Vega M2: bounded comparison of profiled and direct interval baselines

27 October 2026. Companion to the [reply to Lyra](VEGA_TO_LYRA_M2_BASELINE_REPLY.md)
and [M1 reference](VEGA_PREMISE_REFERENCE_M1.md). This is a synthetic baseline
feasibility comparison, not a new certification algorithm or biological study.

## Decision reached

Lyra's analytical reduction is useful and sufficient for this fixture. The
profiled baseline enclosed both endpoint extrema to the declared absolute gap
of 0.002 in all four problems, using **75–225 box evaluations**. Direct
two-dimensional subdivision also succeeded, using **447–841 evaluations**.
Neither result motivates a new numerical method for this family.

The more consequential result concerns the scientific envelope. Preserving a
preparation identity changes an adequacy decision, even though the original-data
likelihood is identical. For the decaying record at tolerance 0.5, the coupled
family permits a uniformly adequate forecast; the independent box does not.
These are different assumptions about the experiment, not competing algorithms
on one problem. The biological interpretation needs independent review.

## Frozen scientific contract

The common laws are

\[
L'=-\kappa L,\quad R'=-\gamma R,\quad x'=aLR-x,
\quad s=\kappa+\gamma,\quad A=aL_0R_0,
\]

with $x(0)=0$, original response $x(t)=A F_{1,s}(t)$, and

\[
F_{1,s}(t)=\frac{e^{-st}-e^{-t}}{1-s}.
\]

The admitted parameter domain is $A\in[0.5,2]$, $s\in[0,0.6]$,
$0\le\gamma\le s$. Known $b=1$ keeps the kernel denominator away from zero.
The original observations occur at times $(1,2,3,5)$, with known independent
Gaussian standard deviation 0.02. The retained family uses

\[
Q(A,s)=\sum_{i=1}^{4}
\left(\frac{y_i-AF_{1,s}(t_i)}{0.02}\right)^2\le10.
\]

The exact integer threshold avoids treating a floating-point quantile as a
validated numerical bound. Under the declared Gaussian observation law and a
true world inside the prior, test inversion has coverage
$1-6e^{-5}\approx0.9595723>0.95$. There is no subtraction of fitted degrees of
freedom. This is a fixed experiment, not an adaptive testing procedure.

The two records are deterministic numerical fixtures, constructed from unit
amplitude responses at $s=0.2$ and $s=0$, respectively, with small fixed offsets.
They are not random draws or an empirical calibration study. The decimal values
below are treated as exact observed constants:

| Time | Decaying record | Near-constant record |
| ---: | ---: | ---: |
| 1 | 0.569564139883174 | 0.624120558828558 |
| 2 | 0.660730953498783 | 0.870664716763387 |
| 3 | 0.627780709657703 | 0.946212931632136 |
| 5 | 0.446426867715446 | 0.998262053000915 |

Generating parameter values are not passed to inference. The four observation
constants, prior, and preparation contract are its inputs.

The future task concerns a **separate matched experiment**, with its clock
restarted at zero and the same initial scale and kinetic laws as the original
unperturbed run. It is not a pulse retroactively added to the run already
observed through time 5. This preparation sharing is a declared premise requiring
biological justification. The target is its mean response at $T=7$ after an
additive ligand pulse at $\tau=3$, with unchanged capacity at the pulse:

\[
m(A,s,\gamma,\rho)=
A F_{1,s}(7)+A\rho e^{-3\gamma}F_{1,s}(4).
\]

Two preparation contracts are evaluated separately:

- **Coupled:** $A\rho=1$. A concrete interpretation is that amplitude uncertainty
  comes entirely from uncertain $L_0$, while $aR_0=1$ and the absolute delivered
  dose $d=1$ are known and shared. Then $A=L_0$, $\rho=1/L_0$, and the pulse term
  is $e^{-3\gamma}F_{1,s}(4)$. Any readout scaling needed for that interpretation
  must also be known or independently calibrated.
- **Independent:** $A\in[0.5,2]$ and $\rho\in[0.5,2]$ vary independently. These
  are the same prior marginal ranges but a larger joint family. Relative to the
  coupled interpretation, it admits combinations violating the fixed-dose
  identity. It may describe another preparation protocol; it must not be chosen
  or rejected simply because it produces a preferred conclusion.

Both contracts have the same original-data quadratic. Those observations cannot
determine which preparation contract is correct. Protocol records or additional
biological evidence must supply that information.

## Queries, models, and budgets

The primary numerical outputs are enclosures
$L_{\rm lo}\le L\le L_{\rm hi}$ and
$U_{\rm lo}\le U\le U_{\rm hi}$ for the minimum and maximum of $m$ over the
retained family. The stopping precision is a gap of at most **0.002 for each
extremum**. The planned box budgets are **64, 256, 1024, 4096**. A root evaluation
followed by two-child splits can use an odd number below a budget; unused work is
not charged. A resolved search stops rather than spending the remaining budget.

For an unrestricted scalar forecast, the minimax absolute error is $(U-L)/2$.
Its enclosure determines adequacy at the fixed diagnostic tolerances
**0.1, 0.25, 0.5**. These tolerances are numerical controls, not biological
acceptance criteria. Nonempty feasibility must be witnessed before reporting
adequacy or a joint conflict.

Three candidate point models are shared across methods within each problem.
A common 33-point rate grid, $s=0.6j/32$, supplies verified feasible amplitudes.
The anchor with the smallest verified upper bound on $Q$ fixes $(A_r,s_r)$;
the three candidates use $\gamma=0,s_r/2,s_r$. They represent the
constant-capacity branch, a balanced-rate choice, and the constant-ligand branch.
Their future dose ratio is $1/A_r$ under the coupled contract and the declared
midpoint 1.25 under the independent contract. All are checked against the data.
When $s_r=0$, the three branch candidates coincide; no revision is forced.

Each candidate's actual prediction receives its own worst-case-error enclosure.
This is **not optimization over every admissible revised model**. A candidate
failing does not establish failure of all models on its branch. A gap between
these candidates and the free forecast can reflect candidate selection; it is
not proof that the full continuous model family cannot attain the midpoint.

The premise query is a joint conflict with constant ligand and constant
capacity. It has a strong common special-purpose calculation: minimize the
convex amplitude quadratic at $s=0$. It does not require global subdivision.
For any feasible $(A,s)$, both $\gamma=0$ and $\gamma=s$ remain admissible, so
neither individual constancy premise can be rejected in this family.

## Numerical baseline and containment contract

Both methods analytically eliminate the monotone pulse coordinates. The direct
baseline subdivides $(A,s)$ boxes and excludes a box only when an outward lower
bound on $Q$ exceeds 10. The profiled baseline uses Lyra's full conditional
amplitude interval and subdivides only $s$. It does not replace uncertainty by
the best-fit amplitude.

For diagonal weights $w_i=1/\sigma_i^2$, the implementation evaluates
$Q_{\min}$ using the nonnegative Gram identity

\[
a_sQ_{\min}(s)=
\sum_{i<j}w_iw_j\big(f_i(s)y_j-f_j(s)y_i\big)^2,
\qquad a_s=\sum_i w_i f_i(s)^2.
\]

Its enclosure is intersected with an independently evaluated residual-form
enclosure. An inconsistent intersection raises an error; it never excludes a
world. Conditional amplitude bounds use a conservative radius. Feasible points
near both amplitude limits are proposed and independently accepted only when
their outward $Q$ upper bound is at most 10.

This last detail was caught during implementation qualification. Using only
best-fit amplitude witnesses left some extremum gaps stalled even though the
outer feasible intervals were retained correctly. Adding verified boundary
witnesses fixes the avoidable numerical bottleneck. This was an implementation
defect in witness selection, not a research obstacle or evidence against Lyra's
reduction. The final experiment uses the corrected policy for both methods.

Arithmetic uses exact decimal inputs and 50-digit directed rounding. Python's
`Decimal.exp` and `sqrt` are rounded to nearest, so their results are expanded by
a representable neighbor on each side; merely setting a directed context would
be insufficient. Decimal endpoints remain strings in saved evidence. Binary
floating point only orders subdivision priorities and never excludes a cell or
certifies a conclusion. Unfinished cells remain on the frontier. These guarantees
depend on the documented Decimal/libmpdec operation contract and the reviewed
implementation; this is not a proof-assistant-verified solver.

Conditional variable elimination is established by [Golub and Pereyra](
https://epubs.siam.org/doi/abs/10.1137/0710036). Validated dynamic parameter
estimation and global bounds also predate this work; see [Lin and Stadtherr's
2006 paper](https://www3.nd.edu/~markst/lin-stadtherr-iecr2006.pdf) and
[2007 paper](https://academicweb.nd.edu/~markst/lin-stadtherr-iecr07.pdf).
Those bounded-error results do not themselves supply our Gaussian calibration.
Arithmetic behavior follows the [Python Decimal documentation](
https://docs.python.org/3.12/library/decimal.html) and [decimal square-root
specification](https://speleotrove.com/decimal/daops.html#refsqrt).

## Recorded comparison

All 8 jobs and 32 planned checkpoint rows completed. Both methods met the
extremum precision target in every problem. Counts below include common witness
preparation where applicable; box counts measure search nodes, while kernel
counts also include preparation and feasibility witnesses.

| Record / contract | Profiled boxes | Direct boxes | Profiled kernel calls | Direct kernel calls |
| --- | ---: | ---: | ---: | ---: |
| Decaying / coupled | 83 | 581 | 1,392 | 5,390 |
| Decaying / independent | 225 | 841 | 4,624 | 8,940 |
| Near-constant / coupled | 75 | 447 | 1,440 | 4,336 |
| Near-constant / independent | 131 | 489 | 2,714 | 5,296 |

Every profiled run resolved at or before the 256-box checkpoint; every direct
run resolved at or before 1024. Equal box budgets are not equal CPU budgets.
The complete uninterrupted run recorded 2,872 box evaluations, 34,132 kernel
evaluations, and 2,372 point-feasibility evaluations. Its measured execution time
was 7.65 seconds, including initialization, search, summarization and checkpoint
publication, but excluding startup and prior-evidence validation. This is one
local timing observation, not a general speedup estimate or a comparison against
published optimized solvers.

The profiled forecast-risk enclosures below are rounded outward for display:

| Record / contract | Optimal scalar forecast risk | At tolerance 0.1 | At 0.25 | At 0.5 |
| --- | --- | --- | --- | --- |
| Decaying / coupled | [0.26737, 0.26926] | Inadequate | Inadequate | Adequate |
| Decaying / independent | [0.58503, 0.58697] | Inadequate | Inadequate | Inadequate |
| Near-constant / coupled | [0.13575, 0.13726] | Inadequate | Adequate | Adequate |
| Near-constant / independent | [0.86187, 0.86379] | Inadequate | Inadequate | Inadequate |

The direct baseline gives the same decisions, with slightly different valid
enclosures. Inadequacy here means that no scalar point forecast can meet the
tolerance uniformly over the **current retained family**. It does not say that
prediction is impossible after new observations or under another justified
scientific contract.

Both methods establish the joint premise conflict for the decaying record. For
the near-constant record they retain a verified $s=0$ witness, so the conjunction
remains possible. Neither method rejects either individual premise.

For the decaying coupled case, the three fixed candidate-model risks are
[0.37519, 0.37711], [0.30033, 0.30219], and [0.40364, 0.40550], respectively.
These exceed the unrestricted scalar optimum, but all three are adequate at 0.5
and inadequate at 0.1 and 0.25. The chosen tolerance grid therefore does not
separate their decision labels from the free forecast in this case. The report
keeps the different risk quantities rather than presenting them as one result.

## Persistence, reproduction, and implementation

| File | Purpose |
| --- | --- |
| `src/autoformalism/research/premise_intervals.py` | Restricted outward-rounded arithmetic and monotone analytic kernel |
| `src/autoformalism/research/premise_baseline.py` | Frozen problems, profiles, verified witnesses, subdivision, and decisions |
| `src/autoformalism/research/premise_baseline_campaign.py` | Source/runtime binding, checkpoint chain, resume and execution accounting |
| `scripts/run_premise_baseline.py` | Standalone run/report CLI |
| Three corresponding `tests/test_premise_*.py` files | Arithmetic, inference, and persistence validation |
| `docs/VEGA_TO_LYRA_M2_BASELINE_REPLY.md` | Scientific response and next decision |

The fixed roster is two records × two preparation contracts × two methods × four
budgets. Each checkpoint records the full search state, bound ledger, unfinished
frontier, verified witnesses, models, cumulative work and report. Source hashes,
runtime and protocol are sealed. Resume rejects mismatches and gaps. Read-only
reports recompute decisions from saved state without generating evidence.

Runtime receipts are separate from deterministic scientific checkpoints. Failed
or interrupted attempts remain visible. A missing completion receipt makes
actual total runtime/work unknown rather than zero. Replaying an unfinished
segment may consume extra physical work; the recorded logical box budget is not
a claim that such replay is free. The main comparison has no interrupted or
unaccounted attempts. Hashes detect accidental changes, not deliberate resealing
of an entire evidence directory; core state is internally generated, not an
untrusted external certificate accepted without provenance.

From the repository root:

```bash
.venv/bin/python scripts/run_premise_baseline.py \
  --output-root output/premise-baseline-reproduction --max-new-checkpoints 5

.venv/bin/python scripts/run_premise_baseline.py \
  --output-root output/premise-baseline-reproduction

.venv/bin/python scripts/run_premise_baseline.py \
  --output-root output/premise-baseline-reproduction --report-only
```

The first command reports 5 of 32 rows and lists every missing row. Changing the
source or scientific contract requires a fresh output root.

Local evidence is under `output/vega-premise-baseline-m2-20261027/`: `main/` holds
the uninterrupted comparison; `resume/` is the independent stop/resume check;
`qualification/` contains implementation qualification, not the final comparison.
Generated checkpoints and execution receipts are ignored by Git.

The focused verification passed **726 tests**: 81 arithmetic tests, 45 baseline
tests, 22 persistence tests, and the 578 existing M1 regressions. Independent
high-precision calculations check conditional amplitude endpoints, kernel and
prediction containment, witness feasibility and query inequalities. Other tests
check preservation of feasible grid points, shared candidate construction,
read-only reporting, corrupt checkpoints, source/runtime drift, and truthful
interruption accounting.

The complete stop/resume check produced **34 byte-identical scientific files**
(plan, 32 checkpoints and summary) relative to the uninterrupted comparison.
Read-only reporting and a completed resume left every saved file unchanged.
Runtime receipts differ as expected. `verification.json` records these checks.
An isolated share packet also successfully reconstructed the saved report using
the byte-identical implementation modules and minimal package wrappers.

Repository-wide `pytest -q -n 4` completed with **11,030 passed, 8 skipped,
6 failed and 1 setup error** in 25 minutes. Six non-passing cases reported
source/freeze identity differences while an unrelated research implementation
commit landed in the shared checkout. The M2 implementation hashes remained
unchanged. The other failure exhausted an existing detention fitting test's
30-second allowance.

A serial rerun of the seven affected tests yielded **6 passed and 1 failed**;
the remaining fitting case then passed alone (**1 passed in 30.22 seconds**).
All affected tests therefore have subsequent passing rechecks, but the original
broad invocation is not a clean pass, and the fitting test remains sensitive to
its time allowance. The eight skips require unavailable optional PyTorch. Logs
are `pytest-full.log`, `pytest-serial-recheck.log`,
`pytest-isolated-fit-recheck.log`, and `pytest-focused.log` in the evidence root.

The new Python files pass Ruff; `ruff check .` reports the same 37 pre-existing
issues in untracked `analysis/claude/` files, which were preserved.

## Implication for the research discussion

This fixture supports Lyra's recommendation to exploit existing structure first.
It supplies no evidence that a new continuous-certification algorithm is needed.
The two implementations are established baselines, and this bounded comparison
does not rank all applicable validated solvers.

The preparation question is a better lead for the GEMOT discussion: does
amplitude uncertainty arise from initial concentration, response capacity,
readout calibration, or their combination, and is the pulse specified in absolute
or relative units? Those facts change the admissible joint worlds. The same
original-data fit cannot answer them automatically. A useful next case should
make these assumptions and the required predictive or model-revision decision
biologically consequential, rather than adding dimensions to make an optimizer
struggle. No new acquisition campaign or biological benchmark is started here.
