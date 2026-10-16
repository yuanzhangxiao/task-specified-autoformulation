# M39: practical uncertainty under nearly confounded inputs

Implementation milestone, 15 October 2026. This is an isolated synthetic
qualification for the [Vega research preparation](VEGA_FALLIBLE_SPECIFICATIONS_HANDOFF.md),
while current construction awaits executable interaction laws. It adds no
automatic model/data attribution, confidence percentages, benchmark changes or
production fitting defaults. The M37/M38 training-feedback interface remains
available for the current paper.

## Question and controlled experiment

Can fitted parameters become unreliable even when the sensitivity matrix remains
full rank, and can a nuisance-aware parameter profile expose that loss of
precision? M36 compared exact ambiguity with clearly separating inputs. M39
gradually closes that gap and independently increases known measurement noise.

The fixed physical model and observation contract are

\[
\dot x_r=g u_r,\qquad x_r(0)=x_0,\qquad
y_{ri}=x_r(t_i)+b t_i+\epsilon_{ri},\qquad
\epsilon_{ri}\sim\mathcal N(0,\sigma^2).
\]

Here `g` is a physical gain, `b` is an observation-only sensor drift, and `x0`
is one unknown shared initial value. Three separately acquired initial readings
are noisy observations of the same preparation; they are included once each,
not imposed as three exact initial conditions. The three input schedules are
constant levels `u_r = 1 + delta*s_r`, with `s_r` equal to `-1, 0, 1`.
Samples occur at `t_i = i/4`, `i=0,...,16`.

The frozen factorial roster is:

| Factor | Values |
| --- | --- |
| Input separation `delta` | 0.5, 0.1, 0.02, 0.005, 0 |
| Known noise standard deviation `sigma` | 0.01, 0.05, 0.2 |
| Independent noise seeds | 0, 1, 2, 3, 4 |
| Fit strategy | Direct and eligible affine-profiled rollout fitting |

This gives 75 acquisitions and 150 fitting tasks. The same standardized noise
draw is paired across the separation/noise conditions for a seed; every strategy
sees the same raw observations in its pair. Five seeds support descriptive
qualification, not coverage calibration. Realized coefficient error need not
increase monotonically for every seed.

The generating values (`g=1.3`, `x0=0.4`, `b=0.06`) are evaluator-only. Fit starts
are fixed independently of those values. Domains are declared acquisition/model
assumptions: `g` in `[0.05,3]`, `x0` in `[0.01,2]`, and `b` in `[-2,2]`.
These bounds are not additional measurements. They remain unchanged across
conditions and seeds. All data are newly generated synthetic controls; Phase C
benchmark data and finalized prompts are untouched.

## Profile and reference calculation

The Gaussian objective, with known sigma and parameter-independent constants
omitted, is

\[
L(g,x_0,b)=\frac12\sum_{r,i}
\left[\frac{x_0+(g u_r+b)t_i-y_{ri}}{\sigma}\right]^2.
\]

For each fixed gain `a`, recompute the best permitted initial value and drift:

\[
L_p(a)=\min_{x_0,b\ \mathrm{in\ declared\ domains}}L(a,x_0,b).
\]

This is a bounded convex least-squares problem for this family. The fitter still
uses the existing direct/profiled numerical backend. The diagnostic separately
qualifies the affine structure and checks the retained endpoint against the
global bounded own-loss optimum. An unfinished or nonoptimal fit cannot be used
as an inflated reference objective that makes the profile look artificially
flat. Unsupported nonlinear equations are outside this first profile adapter.

Compare the proper profile with the deliberately conditional slice
`L(a, x0_hat, b_hat)`, freezing nuisance values at the same retained endpoint.
The latter answers a different question and can be misleadingly narrow if
presented as total uncertainty. Do not use true nuisance values in either
calculation.

Report objective-rise support sets

\[
\mathcal S_c=\{a:L_p(a)-L_{\min}\le c\},\qquad c\in\{0.5,2\}.
\]

These are named objective-rise sets, not confidence intervals. No confidence
probability is assigned. The distinction follows the profile-likelihood
framework of [Raue et al. (2009)](https://doi.org/10.1093/bioinformatics/btp358),
while avoiding an unqualified likelihood-ratio calibration at bounds or under
non-identifiability. Incomplete diagnostic work remains unavailable, not a narrow
or wide interval.

For the balanced interior design, the nuisance-projected gain information has
an exact check independent of generating parameter values:

\[
I_{g\mid x_0,b}=\frac{2\delta^2\sum_i t_i^2}{\sigma^2}
=\frac{187\delta^2}{\sigma^2}.
\]

Before any bounds become active, the profile rise is
`0.5*I*(g-g_hat)^2`, so the half-width is `sqrt(2*c/I)`. In contrast, freezing
nuisance values gives curvature `93.5*(3+2*delta^2)/sigma^2`, which remains
large as the input difference disappears. This check explains why full rank or
a condition number alone is insufficient: a single nonzero projected sensitivity
has condition number one regardless of its absolute magnitude.

At zero separation the data constrain `g+b` and `x0`; the joint rank is two for
three unknowns and conditional gain information is zero. A coefficient error
against one generating representative is not failure to recover an identifiable
quantity. Measured predictions, physical predictions and individual coefficients
are reported separately.

## Domain and numerical safeguards

- Report absolute conditional information, joint rank, profile endpoints and
  widths, not only a full-rank/rank-deficient label.
- Identify gain-domain clipping and **all** active nuisance bounds at support
  endpoints. A finite interval induced by drift bounds is domain-assisted, even
  if its endpoints do not touch the gain bounds. It is not data identification.
- Qualify the bounded global objective and every profile solve; retain failed,
  interrupted, missing and unsupported outcomes in all denominators.
- Bind diagnostic results to the exact problem, retained parameter vector and
  policy. Fitting and diagnostic allowances/costs are separate. Repeated commands
  verify completed checkpoints; uncertain interrupted work is not silently
  rerun with a reset budget.
- Keep a compact append-only progress journal. Durable start/finish records and
  completed nodes survive interruption; a torn final record is marked incomplete.
  The journal avoids both one-file-per-node inode growth and rewriting the whole
  history on every callback.
- Check imported acquisition fields against their source bytes and frozen
  factorial design, including calibration, known noise, forcing, time units,
  shared initialization and parameter domains. Copied evaluator readings must
  equal the public acquisition; a matching claimed hash alone is insufficient.
- Keep truth and analytic error assessments in the post-fit evaluator. Profiles
  and fit selection use only the public candidate, readings, domains and known
  noise contract. No validation/test data or LLM calls are used.

## Expected outputs and interpretation

Each completed fit should report its own-loss optimum/replay status, raw output
fit, parameter estimates, conditional information, profiled and frozen-nuisance
support sets, endpoint-bound activity, diagnostic availability and compute cost.
The evaluator adds coefficient error and clean physical/measured-output error.
Grouped results report counts and median/MAD over the fixed seeds, preserving
strategy pairs rather than selecting the best run by truth.

The desired qualification is: broadening profiles reflect the gradual loss of
information, exact confounding remains explicit, and holding nuisance values
fixed understates that ambiguity. It does not establish reliable confidence
percentages, arbitrary nonlinear profile optimization, or a diagnosis of whether
a real discrepancy comes from equations, sensors or scientific assumptions.

The later feedback use is a statement such as: “This gain can change substantially
with little loss in fit when the initial value and sensor trend readjust.” It
should prevent a confident coefficient interpretation and motivate a more
discriminating input or calibration measurement. It does not authorize changes
to the current paper's fixed measurement contract.

## Delta execution

The experiment uses existing Delta CPU dependencies, one CPU per task and two
concurrent tasks by default. No GPU, model server or API key is required. Each
task has a 30-second fitting search allowance, 15-second numerical verification
allowance and a separate 30-second profile allowance. The Slurm time reservation
includes Python startup, artifact verification and evaluator work.

Upload the pinned `phase-c-fitting-m39.tar.gz` bundle into the Delta home folder:

```bash
bash <<'BASH'
set -euo pipefail
AF_CODE=$(mktemp -d /projects/bibo/yxiao2/repos/practical-uncertainty-m39.XXXXXX)
tar -xzf "$HOME/phase-c-fitting-m39.tar.gz" -C "$AF_CODE"
cd "$AF_CODE"
sha256sum -c SHA256SUMS --quiet
bash scripts/hpc/submit_phase_c_practical_uncertainty_delta.sh
BASH
```

Keep the code directory, input seals and submission receipts for deterministic
resume. An uncertain scheduler reply requires inspection, not blind resubmission.

```bash
AF_ROOT=/work/hdd/bibo/yxiao2/phase_c/practical-uncertainty-fitting-v1
AF_CODE=$(jq -er '.identity.repo' "$AF_ROOT/submission_manifest.json")
bash "$AF_CODE/scripts/hpc/inspect_phase_c_practical_uncertainty_delta.sh" "$AF_ROOT"
```

The inspector reports all planned tasks and writes a portable review archive.
The 150-task cluster run has returned; the review below distinguishes complete
fits from nine incomplete diagnostic profiles.

## Delta results, reviewed 15 October 2026

Source archive: `review-20261016-021339.tar.gz` (UTC filename), SHA-256
`ee5ac2c8c8a0499679b1d6305c249ff2a8685a5d9491fea74c84ef6da9a2fb65`.
Execution commit: `6c467ad7b877ed64f4a9caa5a5b2207dd9fb483b`.
Plan: `6bc5c0e385d13728fd1d93df642ae898bda2351a2685d714a37028a30316d9f2`.
Delta jobs: prepare `22930374`, array `22930375`, report `22930376`.

All seals, frozen acquisition checks, nested endpoints and the reproduced summary
agree. No new fitting, integration, test access or LLM calls were performed in
this review.

Review verification also reran 149 profile/fixture tests successfully and
`git diff --check`. Whole-repository Ruff reports 37 unrelated findings in
`analysis/claude`; this review changes documentation only.

- All **150/150 fits** pass independent numerical and analytic objective checks.
  All retained endpoints meet the independently computed bounded global optimum;
  the maximum absolute objective gap is `7.18e-13`.
- All **75/75 direct/profiled-policy pairs** agree in objective, with maximum
  difference `2.48e-12`. At exact confounding, the profiled policy rejects a
  rank-deficient inner design and safely falls back to joint fitting. Its 30
  rejected profiling attempts (two starts for each of 15 cases) explain the
  recorded started/completed search-call difference; they are not missing costs.
- All **150/150** conditional and frozen information checks match the analytic
  design. Completed interior support widths match their analytic reference
  within `1.1e-9` relative error.
- **141/150 profiles complete; 9/150 are partial.** All nine stop at their
  30-second wall allowance. The three recorded failed profile operations are
  deadline overruns, not failed KKT or solver-optimality checks. Partial profiles
  stay excluded from summary widths, even if one threshold completed.

### Practical uncertainty while the design is still full rank

At fixed `sigma=0.01`, every direct/profiled profile completes for all five seeds.
The following table uses the direct arm, objective rise `c=0.5`, and medians over
the same five noise draws. The generating gain `1.3` is evaluator-only.

| Input separation | Joint rank | Absolute gain error | Relative gain error | Proper support width | Frozen-nuisance width | Clean reading RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 3/3 | 0.001422 | 0.109% | 0.002925 | 0.001106 | 0.002610 |
| 0.1 | 3/3 | 0.007109 | 0.547% | 0.014625 | 0.001190 | 0.002610 |
| 0.02 | 3/3 | 0.035544 | 2.734% | 0.073127 | 0.001194 | 0.002610 |
| 0.005 | 3/3 | 0.142175 | 10.937% | 0.292509 | 0.001194 | 0.002610 |
| 0 | 2/3 | Not identifiable individually | Not an identifiable recovery error | 2.950000 | 0.001194 | 0.002227 |

Reducing input contrast by 100-fold widens the proper support by 100-fold while
all nonzero designs remain full rank. The best fit to sensor readings is nearly
unchanged. The equations explain this directly: each reading slope is `g*u_r+b`.
When all `u_r` are close to one, changing `g` and compensating with `b` leaves
nearly the same observations. Freezing `b` suppresses that compensation and
manufactures an unjustifiably narrow slice. At zero contrast the complete gain
domain `[0.05,3]` remains supported; lowering noise cannot resolve this exact
gain/drift ambiguity.

Noise strengthens the same pattern. For the all-complete direct groups at
`delta=0.1`, increasing sigma from `0.01` to `0.05` multiplies support width and
median gain error by five. At `delta=0.005, sigma=0.05`, median relative gain error
is 54.68%, while the complete support ranges touch the gain domain. Their finite
width is therefore not purely data-determined. In the particularly incomplete
`delta=0.02, sigma=0.2` profiled group, widths use only two of five seeds; it must
not be used as an unqualified five-seed trend comparison.

These results qualify useful **uncertainty diagnostics**, not an improvement in
point estimation or calibrated coverage. They demonstrate that a solved fitting
problem can still leave the physical coefficient poorly determined. The proper
profile detects this using only the public acquisition, without knowing the
generating gain. It should prevent overconfident coefficient interpretation or
premature equation criticism in later feedback.

### Diagnostic cost and next bounded step

| Stage | Total recorded CPU seconds | Sum of attempt wall seconds |
| --- | --- | --- |
| Fitting, including verification | 58.261 | 826.039 |
| Separate practical profiles | 48.799 | 952.932 |
| Retrospective evaluator | 0.405 | 25.362 |

Profile overhead is **83.8% of fitting CPU time** in this deliberately tiny
problem. Median profile CPU time is 0.324 seconds; median elapsed time is 2.540
seconds. All 22,893 diagnostic calls have recorded completion, including the
deadline failures; wall/CPU cost accounting is complete even though nine profiles
are not. Wall totals above are sums across attempts, not campaign makespan or
scheduler wait time.

The nine partial profiles used only 0.144–0.431 CPU seconds while consuming
30.16–31.24 wall seconds. This points to waiting rather than expensive numerical
optimization. Frequent durable writes on shared storage are a plausible cause,
but there is no per-operation I/O timing here to establish it conclusively.

Next: preserve all 150 fitted endpoints, instrument diagnostic compute versus
journal/wait time, and complete the nine profiles under a separately recorded
diagnostic extension. Avoid resetting or overwriting their consumed allowances.
Reduce unnecessary profile solves and journal traffic without losing started-work
accounting. After this completion check, extend practical-uncertainty qualification
to a small nonlinear block, where global inner optimality is no longer automatic.
No new benchmark run or parameter refit is needed for the present nine diagnostics.

Reviewed artifacts and a three-panel explanatory figure are under
`output/phase-c-practical-uncertainty-m39-delta-review/`. The figure uses only the
25 complete direct profiles at fixed `sigma=0.01`; its numerical source is
`plot-data.json`. Generated artifacts remain uncommitted.

## Local qualification, 15 October 2026

The four M39 test files pass **184 tests**; relevant M36 regression tests and a
construction source-identity rerun pass another **168 tests**. The immutable
deployment export passes `ruff check .`; changed files also pass Ruff, shell
syntax and whitespace checks. The shared checkout's whole-repository Ruff run
finds unrelated in-progress issues in `analysis/claude` and
`research/saved_topology_repair.py`; these files are not modified by M39.

The full `pytest -q` attempt was stopped at the 300-second local limit: 1,078
passed and three Torch-dependent tests skipped. One construction test caught a
source change during concurrent development; its isolated rerun passed once
the source was stable. This is not a claim that the full repository suite passed.

Eight actual fits on four seed-0 acquisitions, using both strategies, all pass
the independent objective and analytic curvature checks. Completed attempts
resume byte-for-byte without changed modification times; repeated reports are
identical. The final smoke used an immutable export because construction work
was actively changing the shared package source identity.

At objective rise `c=0.5`, both strategies produce the following widths (rounded):

| Separation | Sigma | Proper profile width | Frozen-nuisance width | Interpretation |
| --- | --- | --- | --- | --- |
| 0.5 | 0.01 | 0.002925 | 0.001106 | Data-limited |
| 0.02 | 0.05 | 0.365636 | 0.005970 | Full rank but substantially weaker |
| 0.005 | 0.2 | 2.142467 | 0.011926 | Lower gain domain clips the range |
| 0 | 0.05 | 2.95 | 0.005971 | Entire permitted gain domain; exact confounding |

These are selected smoke conditions, not the completed factorial study. The
first three rows vary both factors; the full campaign separates their effects.
The compact journal keeps all eight profiles below 0.15 seconds each on this
local run (0.817 CPU seconds in total), including journal work. This is a tiny
affine problem, not an overhead estimate for nonlinear profiles or Delta I/O.
Local artifacts are in
`output/phase-c-practical-uncertainty-m39-pinned-smoke/`; generated results are not
committed. General nonlinear profiles, estimated noise scales and calibrated
coverage remain future qualifications.
