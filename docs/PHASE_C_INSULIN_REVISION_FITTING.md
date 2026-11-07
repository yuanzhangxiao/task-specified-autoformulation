# Fitting the admitted insulin feedback revision

6 November 2026. This is a separate CPU fitting campaign for Astra's admitted
revision of candidate 15. The remaining 22-parent campaign can continue from
its existing frozen source bundle and output root.

## Model and comparison

The revised insulin-production process adds `k_insulin_delay * X_insulin_action`.
Its coefficient has a real domain. The resulting effective coupling is the
product of this coefficient and the insulin-production outer gain. It closes
an I → X → I loop; X is the existing insulin-action state, not a newly proposed
pancreatic state. Fitting will determine whether that additional flexibility
improves the responses. Admission does not establish its physiological meaning.

The portable handoff pins the admitted `PrefixFitProblem`, public training rows,
initialization graph, supplied endpoint and construction provenance. It includes
the 32 fitted training roots. Setting the new coefficient to zero recovers the
previous family with training NMSE 0.2610107571549462 on independent replay.
Generic latent guesses in the base candidate are not substitutes for those roots.

Three tasks are predetermined:

| Task | Purpose |
| --- | --- |
| Nested start, new coefficient zero | Primary comparison with the previous verified fit. |
| Positive perturbation | Separately labelled initialization robustness check. |
| Negative perturbation | Separately labelled initialization robustness check. |

The signed starts are ±0.0021326192830597522, one tenth of the larger inherited
absolute I/X diagonal rate. This is a fixed numerical initialization heuristic,
not a physiological bound or a constraint on subsequent signed fitting.

Each task receives 1,800 numerical search seconds and 2,000 calls, matching the
previous fit's per-search allowance. Qualification and original-equation replay
have separate allowances. The two robustness searches are extra computation;
their best result must not be described as a single matched-budget comparison.
All task starts and budgets are frozen before fitting results are available.

## Numerical route

At fixed shared parameters the equations remain affine in states and inputs.
Exact affine propagation supports a bounded conditional least-squares solve
for the forcing coefficients and the 32 training initial roots. The new feedback
coefficient enters the state matrix and stays in the outer nonlinear search.
It cannot be profiled as a forcing-only linear coefficient.

An exact algebraic check removes a shared production-gain redundancy: an outer
gain multiplying two otherwise isolated real amplitudes can be set to one
while absorbing it into both amplitudes. The zero-gain case is represented by
two zero amplitudes. Domain and occurrence checks must pass before applying
this reduction. Existing certified reductions apply separately, and every
retained endpoint must lift to and replay in the admitted original equations.

Outer coordinates are centered and scaled using the net affine coefficients
and their exact parameter derivatives at the starting point. For example,
`-I-b*I` is scaled by its small net damping, rather than by the nearly unit
amplitude `b`. This avoids losing the conditioning improvements from earlier
fits when the new coupled term prevents use of the old timescale adapter.
These scales stay fixed during each search and preserve the parameter domains.
The inherited meal chart searches positive absorption rates; the exact zero-rate
branch is not searched in this milestone. This restriction is reported separately
from the family-preserving production-gain reduction.

The declared real domain of the added coefficient remains unchanged. Instability
at the generic guess of one is not a reason to impose a positive-only domain
or a new stability bound. Conditional stability diagnostics at the endpoint
are distinct from fitting constraints and do not certify the full nonlinear
scientific interpretation or unseen behavior.

The verified nested incumbent survives worse, failed or interrupted searches.
Each operation has a durable, nonrenewable reservation. Resume reuses completed
receipts and preserves saved useful points without resetting search budgets.
The independent replay guards retention, while optimizer termination and
conditional least-squares optimality remain separate from a claim of joint or
global optimality.

## What the report will answer

The report compares the retained revision with the independently replayed nested
reference. It includes aggregate and per-target training NMSE, trajectory median
and unscaled MAD, meal-dose and no-meal controls, and the three shifted 90 g meal
responses. The baseline insulin peaks occur 91 minutes early: predicted at
minutes 10, 70 and 130 versus observed 101, 161 and 221. Both peak timing and
height matter; a smaller average error alone does not resolve this discrepancy.

Effective feedback sign, sampled output positivity and relevant eigenvalues
are reported as diagnostics. Sampled positivity is not a continuous-domain
certificate. The Gp/I/X block excludes the inherited Q drift and meal reservoir;
its stability alone is not a full-model stability certificate.

This is full-training reconstruction with fitted trajectory roots. No validation
or test observations are used, and no LLM is called. A later evaluation must
freeze shared coefficients, infer new roots only from the existing eight-unit
observation prefix, and score beyond that prefix. Good reconstruction would not
by itself establish that forecasting result.

## Delta commands

From the Mac, upload the delivered bundle:

```bash
scp /Users/yuanzhangxiao/Projects/autoformalism/output/insulin-revision-fitting-delivery-20261106/insulin-revision-fitting-bundle.tar.gz \
  yxiao2@login.delta.ncsa.illinois.edu:/projects/bibo/yxiao2/
```

On Delta, use a persistent extracted source directory and a fresh output root:

```bash
set -euo pipefail
AF_REPO=$(mktemp -d /projects/bibo/yxiao2/repos/insulin-revision-fitting.XXXXXX)
tar -xzf /projects/bibo/yxiao2/insulin-revision-fitting-bundle.tar.gz -C "$AF_REPO"
cd "$AF_REPO"
sha256sum -c SHA256SUMS --quiet
export AF_INSULIN_REVISION_FITTING_ROOT=/work/hdd/bibo/yxiao2/phase_c/insulin-revision-fitting-v1
bash scripts/hpc/start_insulin_revision_fitting_delta.sh run
printf 'Keep this source directory: %s\n' "$AF_REPO"
```

Defaults use the existing project Python and CasADi dependency directory,
`bibo-delta-cpu`, one CPU and 8 GB per worker, and at most two concurrent workers.
Each worker requests four hours, including qualification, verification and
filesystem headroom; the report requests one hour. These are scheduler ceilings,
not expected numerical search durations. Login preparation authenticates inputs
and does not run fitting or a test suite.

In a new terminal session, recover the recorded source directory rather than
creating an empty directory:

```bash
export AF_INSULIN_REVISION_FITTING_ROOT=/work/hdd/bibo/yxiao2/phase_c/insulin-revision-fitting-v1
AF_REPO=$(jq -er '.value.repo' "$AF_INSULIN_REVISION_FITTING_ROOT/launcher_execution.json")
bash "$AF_REPO/scripts/hpc/start_insulin_revision_fitting_delta.sh" status
squeue -u "$USER" -o '%.22i %.42j %.12T %R'
# After the workers finish:
bash "$AF_REPO/scripts/hpc/start_insulin_revision_fitting_delta.sh" inspect
```

Download from the Mac:

```bash
scp yxiao2@login.delta.ncsa.illinois.edu:/work/hdd/bibo/yxiao2/phase_c/insulin-revision-fitting-v1/inspection.tar.gz \
  "$HOME/Downloads/insulin-revision-fitting-inspection.tar.gz"
```

Ambiguous scheduler replies block automatic resubmission; reconcile saved intents
with Slurm before retrying. Keep the extracted source unchanged for the campaign.
Source code, runtime, inputs and execution settings have separate sealed identities.

## Returned result

The primary revised fit improved training NMSE from 0.261011 to 0.163647.
The three shifted-meal insulin peaks moved from 91 minutes early to 35, 29 and
30 minutes early. All three starts reached their 1,800-second numerical limit
while still improving; none establishes optimization convergence.

All retained endpoints also predict negative insulin on some training samples
(primary minimum -20.02 pmol/L). These are independently replayed predictions
within the declared real parameter domains, not coefficient-bound violations.
The new signed term became inhibitory and net insulin self-dependence became
positive. The fit improves response timing but does not establish scientific
validity or the proposer's pancreatic interpretation.

The detailed returned review and next-step recommendation are appended to
`work/agent-comms/2026-11-06-insulin-revision-fitting-campaign.md`. The next
experiment should distinguish further numerical continuation from an explicit
positivity-aware control, retaining the original receipts and comparison.
