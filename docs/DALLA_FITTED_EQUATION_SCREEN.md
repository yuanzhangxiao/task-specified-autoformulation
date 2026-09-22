# Finding fitted Dalla Man models for mechanistic inspection

This is a post-selection diagnostic. It does not change candidates, rerun fits,
choose benchmark winners, call a proposer, or read held-out test files.

## Expanded archive: 217 fitted records

The two September 22 uploads are byte-identical. Their inventory artifact digest
is `b0398ddee4079f2c8d570c33619944ec4474268877363fe85718e3079c944cca`.
They contain 217 saved request/parameter records from six review campaigns,
covering the same four T1 easy cells. These are not 217 independent runs or
necessarily 217 mathematically distinct models. Every exported record has fit
status `complete` and profile `collocation-single-target-v2`.

All 217 candidate hashes and fitted parameter-name sets reproduce under restricted
lowering. The equation screen extracts fitted direct public-channel coefficients
and flags nonlinear cross terms. For 215 polynomial target balances, 1,269 checks
against the production interpreter reproduce the extracted affine coefficients.
Constant denominators use the production signed floor of 1e-12; ignoring that
floor would misstate coefficients in five extreme historical fits. The other
two target balances contain a saturating meal term and no auxiliary pathway;
they were inspected separately. These checks establish numerical interpretation,
not scientific correctness or trajectory accuracy.

The strongest predictive leads and the more plausible direct-balance leads were
then inspected with their full latent equations and fitted initializers:

| Candidate | Source and status | Training NMSE | Validation NMSE | Equation assessment |
| --- | --- | ---: | ---: | --- |
| Full, perturbed named, seed 1 | v5 round 13, retained | 0.0423530 | 0.0593280 | Strong new predictive lead. EGP coefficient −7.1933; several duplicate meal filters, an unused state, and linear exchange despite the perturbed reference. |
| Brief-only, canonical obfuscated, seed 1 | v5 round 13, unretained fitted trial | 0.0189227 | 0.0177609 | Best training fit in the archive. Gt coefficient is much closer to the reference than in the incumbent, but EGP is −2.0251 and direct meal input is negative. |
| Brief-only, canonical obfuscated, seed 1 | v4 round 9, unretained fitted trial | 0.0191307 | 0.0177609 | Similar predictive lead; EGP −1.8583 and direct meal input negative. |
| Brief-only, canonical obfuscated, seed 1 | v3 round 6, unretained fitted trial | 0.0213034 | 0.0164264 | Simpler one-filter lead; EGP −2.0709 and direct meal input negative. |
| Brief-only, canonical obfuscated, seed 1 | Retained incumbent first seen at round 7 | 0.0341485 | 0.0140791 | Previously inspected; exaggerated independent Gt response. |
| Full, perturbed named, seed 1 | v3 round 5, retained then | 0.499526 | 0.434599 | Direct source/sink signs more plausible, but poor fit and incorrect exchange form. |
| Full, perturbed named, seed 0 | v3 round 3, unretained fitted trial | 0.990980 | 1.023660 | Direct balance signs more plausible, but fitted meal-memory coupling is exactly zero and the fit is poor. |

Thus the expanded archive **does reveal substantially better Full fitting than
the previously downloaded round-12 endpoints**, but it still supplies no clear
example combining a good fitted physiological mechanism with good training fit.
The three earlier Brief-only trials are legitimate exploratory candidates; they
are not official retained endpoints and were not rejected as invalid models.
Their validation errors exceed the incumbent's, despite their lower training
errors. Selecting one now for an illustration must remain explicitly post hoc.

### Best new Full candidate

ID: `9d60ddc59cf797f44ef7fc3cbe9505b041245187a4c0046ac99f96246b7c73cf`.
The fitted target balance, rounded, is

```text
dGp/dt = -7.19331 EGP + 164.09475 Uii - 0.1 E
         + 0.0437753 Gt - 0.154562 Gp
         + 0.00773513 A + 0.00107150 Z + 0.254050 I - 6.09465 Y.
```

The latent equations are driven only by meals and other latent states. Thus there
is no additional latent EGP pathway that restores a positive production response.
The reference balance has coefficient +1 on EGP. Uii is constant in the original
data, so its large coefficient is functioning as an intercept rather than an
identified utilization sensitivity. The symbol I here denotes a discovered
latent state, not an observed insulin channel.

The M state is disconnected from the output. X, A and Z have identical forcing
and decay coefficients; I uses a different forcing coefficient with the same
decay. The decay rate is approximately 0.00107169/min, or a 933-minute time scale.
Y is driven by X with a similarly slow decay. Initial values differ, and latent
scaling is arbitrary; these are redundancy and transient-compensation concerns,
not a claim that a large or negative latent value alone invalidates a model.
The perturbed reference's exchange is state-dependent, while this complete model
is affine in its states and supplied channels.

### Best-training Brief-only trial

ID: `a276318babbbc64b105676549d20eb813157492c7909fd3ce662abe604f3ee7e`.
Decoding names for post-selection interpretation, its fitted equations are

```text
dw/dt = -0.0168879 w + u,   w(0) = 61.8892
dz/dt = -0.0246568 z + u,   z(0) = -145.7544
dGp/dt = -0.503914 u - 2.02505 EGP + 3.07589 Uii + 153.6985 E
         + 0.121739 Gt - 0.0908364 Gp + 0.0783253 w + 0.0246568 z.
```

Its direct Gt coefficient is 1.54 times canonical +0.079, substantially closer
than the incumbent's 12.12 times. The latent filters have 59.2- and 40.6-minute
time scales, making this an interesting candidate for the *existing* diagnostic
probes. However, these filters depend only on u and cannot restore the missing
positive EGP response. The subsequent frozen-parameter replay below tests this
lead directly rather than inferring interventional accuracy from coefficients.

All four leading new/trial records in the table exhausted their fitting allocation
and do not report verified native convergence. They are usable retained parameter
vectors, not proof that their structures cannot fit better. No refitting or
post-inspection editing was performed.

### Round-13 trajectory inspection

The strongest two new leads were replayed on all 16 original training and four
validation trajectories, with their saved parameters and initial-condition maps.
The Full round-12 parent was also replayed. All six aggregate scores reproduce
the saved values within 2e-15. The compact inventory omitted historical data
hashes; the replay freezes the previously authenticated public plan for each
matching benchmark cell and verifies this score reproduction.

| Frozen model | Training NMSE | Validation NMSE | Status |
| --- | ---: | ---: | --- |
| Full, perturbed named T1, seed 1, round 12 | 0.352996 | 0.384974 | Previous retained model |
| Full, same cell and seed, round 13 | 0.042353 | 0.059328 | Retained model |
| Brief-only, canonical obfuscated T1, seed 1, round 13 | 0.018923 | 0.017761 | Unretained fitted trial |
| Brief-only incumbent, same cell and seed | 0.034149 | 0.014079 | Retained model |

Full round 13 improves the meal-response amplitudes substantially: 13 of its
16 training trajectories have NMSE below 0.05. It still responds too early on
some schedules and undershoots the late validation tails. This is a useful
predictive improvement, despite the physiological coefficient problems above.

The Brief-only trial has 14 of 16 training trajectories below 0.05 and all four
validation trajectories below 0.05. It also shows artificial short dips at meal
arrival, consistent with its fitted negative direct meal coefficient. These
features are visible in the absolute trajectories, not merely inferred from
the equations.

The Brief-only trial was additionally evaluated on the **same seven previously
frozen canonical diagnostic cases**: fasting and meal responses with initial
tissue glucose unchanged or changed by +/-20%, and a split-meal schedule. All
other initial physical states are held fixed for the tissue-glucose preparation;
the supplied auxiliary trajectories come from the corresponding reference
simulation. These are post-selection diagnostic cases, not registered test data.
The canonical probes were not applied to the perturbed Full model.

For fasting with initial tissue glucose +20%, the relative squared error in
the *change from each model's own unperturbed control* is 0.271 for the new
Brief-only trial, versus 1.449 for its incumbent and 1.543 for Sol repetition 0
with no extra latent states. This confirms a substantially better incremental
response to this preparation. However, absolute-trajectory NMSE is 0.0161,
0.0111 and 0.0118 respectively: baseline offsets can reverse the comparison.
Both absolute trajectories and matched-control differences are therefore shown.
The trial also has worse split-meal effect error (0.118 versus 0.0735 and 0.0749),
with sharp dips from its direct negative meal term. It is not an across-the-board
interventional improvement or a clean mechanism-recovery example.

All 67 new rollouts completed; deterministic resume reused every checkpoint
without a new integration. Eight selected trajectories were independently
checked with DOP853 at tighter tolerances and a 0.5-minute maximum step; the
largest prediction difference from the Radau replay was 3.50e-6 mg/kg. No
parameters were refitted and no model calls or test data were used.

Curves, all-trajectory sheets, overview figures, exact model identities and
verification records are in `artifacts/t1-round13-inspection-2026-09-22/`.
The archive still contains no round-15 records, so this inspection does not
assess the user's completed final round. Refresh the checkpoint inventory
before drawing conclusions about rounds 13-15 as a whole.

### Partial status and wider benchmark coverage

- Five checkpoint entries lack either the fitted parameter vector or lowered
  candidate identity. Four are round-zero entries; the v5 round-14 Full entry
  preserved its retained round-13 model before the trial export failed. The
  export does not retain enough failure detail to diagnose those missing fits.
- The 43 unsupported roots are other protocol formats, mostly earlier pilots.
  They are not 43 failed current-model fits. Three declare T2 tasks: two are
  construction-only audits, while `prefit-training-evidence-v1-e7ffd12` is the
  older fitting pilot. Its previously recovered report shows no complete finite
  validation for the three T2 Brief-only fits; the sole finite training-evidence
  T2 fit has train/validation NMSE about 1.43e8/1.36e8. It is not an accurate
  candidate and is not the current Full protocol.
- The 65 missing checkpoint paths comprise 48 rounds in the old v4 branch,
  all 16 planned round-15 records in v5, and one v5 round-14 record. These are
  snapshot omissions, not evidence of 65 independent failed experiments.
- This bundle does not establish fitted current-method T1-hard, T2, T3 or T4
  coverage. It establishes the four T1-easy conditions and identifies the older
  T2 pilot separately. Other machines and unsupported formats remain outside
  the exported fitted-model inventory.

Complete numerical equations, screening facts and coefficient verification are
saved under `artifacts/dalla-equation-search-2026-09-22/`. The communication note
is `work/agent-comms/2026-09-22-codex-expanded-dalla-equation-search.md`.
Production code, fitting settings, benchmark data and official selection are
unchanged.

## Actual current-method coverage

The known review campaign has four Dalla Man cells: T1 canonical and perturbed,
each named and obfuscated, all easy. Each has Full and Brief-only at two seeds.
These are four variants of **T1**, not tasks T1–T4. The full Dalla Man public
matrix has 32 cells (four tasks × two dynamics × two names × two tiers).
The external baselines' larger matrix does not establish matching coverage for
our method. An absent local checkpoint is not evidence that no remote run exists.

The round-12 downloads contain 11 retained Full/Brief-only endpoints:

| Condition | Full models available locally | Brief-only models available locally |
| --- | --- | --- |
| T1 canonical named easy | 2 | 1; seed 1 has no retained endpoint |
| T1 canonical obfuscated easy | 2 | 2 |
| T1 perturbed named easy | Scores only for 2 | 2 |
| T1 perturbed obfuscated easy | Scores only for 2 | 2 |
| T1 hard and all T2–T4 | No current-method checkpoints located | No current-method checkpoints located |

The known summary contains scores for four additional perturbed Full endpoints.
Their equations are needed before assessing them. Older historical candidate
pools cover the original and perturbed T1 contracts; those are a different
protocol and cannot stand in for current Full/Brief-only runs or T2–T4.

## What makes a fitted equation promising?

Inspect the numerical equation after inserting its retained fitted parameters,
including the fitted initial-condition maps. A topology edge alone is insufficient:
its fitted coefficient can be zero, negligible, or canceled by another term.

For a T1 easy model, the public physiological channels make the plasma balance
particularly informative. In the positive region, canonical reference dynamics are

```text
dGp/dt = EGP + Ra - Uii - E - 0.065 Gp + 0.079 Gt.
```

The perturbed reference changes the two exchange fluxes to state-dependent laws.
The meal-to-appearance path must also represent ingestion/absorption dynamics;
the public meal signal is not itself the glucose appearance rate. Both the
balance and the complete input-to-output dynamics matter.

Screen for:

1. Correct active input paths and source/sink roles under that cell's public contract.
2. Fitted response strengths and time scales consistent with those roles.
3. Necessary dynamic memory that actually affects the target, allowing legitimate
   reduced models and rescaling or recombination of latent coordinates.
4. Sensible initialization and basal behavior; large latent values alone are not
   evidence of an error because latent scales are arbitrary.
5. Acceptable original training fit as a prerequisite for the desired illustration.

Exact recovery of every hidden physiological state or parameter is not required.
The comparison of numerical coefficients is meaningful for *direct public
channels in their specified units*, not arbitrary latent coordinates. Constant
Uii and nearly inactive E cannot identify general responses to those channels.
Failure to recover their reference coefficients is therefore not, by itself,
proof of failure on the observed trajectories. Nor is a correct local sign enough
to establish a correct full model.

For T2–T4, the rubric must include all required outputs and their insulin/glucose
input paths, using that task's actual auxiliary contract. A T1 glucose equation
cannot certify a multi-output model. Obfuscated names may be decoded for this
post-selection inspection only; no reference information is sent to discovery.

## Findings from the available round-12 equations

No convincing reference-consistent model was identified among the 11 retained
endpoints. This is a qualitative equation screen, not an automated correctness
score or a claim that no good model exists elsewhere.

| Model | Train / validation NMSE | Main obstacle |
| --- | --- | --- |
| Full, canonical named, seeds 0/1 | .333/.264; .314/.445 | Fitted Gt coefficients are −2.40e−6 and −3.60e−6 instead of +.079; direct exchange is effectively removed. |
| Full, canonical obfuscated, seeds 0/1 | .148/.275; .157/.240 | Both ignore all supplied auxiliary channels, including Gt. They cannot respond to an independent Gt preparation with the same target initial value and meal input. |
| Brief-only, canonical named, seed 0 | 2.099/2.549 | Direct balance has +Uii, +E and −Gt, and fits poorly. |
| Brief-only, canonical obfuscated, seed 0 | .412/.397 | Gt absent; redundant equal-rate parallel meal filters. |
| Brief-only, canonical obfuscated, seed 1 | **.0341/.0141** | Best original predictive fit, but EGP coefficient −18.74 and Gt coefficient +.9573, versus reference +1 and +.079. Independent Gt probes already show exaggerated responses. |
| Brief-only, perturbed named, seed 0 | .336/.442 | All auxiliaries omitted; direct Gp decay nearly zero. |
| Brief-only, perturbed named, seed 1 | 7.780/8.583 | Fitted Gt and Uii coefficients zero; poor fit. |
| Brief-only, perturbed obfuscated, seed 0 | .142/.236 | Gt absent, unused state, weak EGP drive. |
| Brief-only, perturbed obfuscated, seed 1 | **.0847/.0767** | Better predictive fit, but negative EGP, positive Uii/E, and linear rather than perturbed exchange. |

The same downloads also contain 10 further saved trial/request records. These
were screened without promoting them to retained endpoints. Several reproduce
the retained equations. The changed trials mainly add meal-driven states or
alter fitted coefficients while preserving the missing/wrong Gt pathway or
incorrect direct balance; none supplied a stronger mechanistic example. In all
21 saved request/vector records, current restricted lowering exactly reproduced
the saved candidate digest and parameter-name set.

Local detailed artifacts are under
`artifacts/dalla-equation-inventory-2026-09-22/`. The earlier complete trajectory
and diagnostic exports remain under `artifacts/t1-brief-only-inspection-2026-09-22/`
and `artifacts/t1-intervention-full-2026-09-22/`.

## Collect all saved rounds before running more experiments

`scripts/collect_dalla_fitted_models.py` is a standard-library-only exporter.
It scans every immediate child with a plan, supports review-deadline-1 through
-5, and collects every planned Full/Brief-only Dalla Man round's retained endpoint
and fitted trial. It includes public briefs, exact fit requests, fitted parameters,
training/validation scores and provenance. No trajectory tables, residual packets,
provider logs, or test evaluation files are exported. There are no model calls,
simulations or fits. Unknown campaign protocols, missing rounds and corrupted
artifacts remain explicit. Candidate execution/lowering is deferred to inspection.

The 64 coverage rows represent 32 cells × two arms. Counts refer to distinct
saved requests and parameter vectors, not independent runs or mathematical
equivalence classes. Continuations and copied incumbents must not be treated as
additional independent seeds. Requests are deduplicated exactly; occurrence
records retain their campaign, round, retained/trial role and fit evidence.

Run this on ACES, replacing `REV` with the commit supplied in the accompanying
message. It exports one JSON file for downloading, without changing an active
experiment checkout. Additional `--scan` or `--root` options can cover other
known result locations. A new snapshot needs a new output path if data changed.

```bash
(
  set -euo pipefail
  module load GCCcore/13.2.0 Python/3.11.5
  REV=REPLACE_WITH_REPORTED_COMMIT
  BASE=/scratch/user/u.yx126462/repos/autoformalism-e432fe3
  git -C "$BASE" fetch origin codex/prefit-aces-v1
  OUT=$(mktemp -d /scratch/user/u.yx126462/phase_b/dalla-equations.XXXXXX)
  git -C "$BASE" show "${REV}:scripts/collect_dalla_fitted_models.py" \
    > "$OUT/collect_dalla_fitted_models.py"
  python3 "$OUT/collect_dalla_fitted_models.py" \
    --scan /scratch/user/u.yx126462/phase_b \
    --out "$OUT/dalla-fitted-models.json"
)
```

Download the printed `dalla-fitted-models.json`. This allows equation-first
inspection of earlier retained models and unretained trials, the four missing
perturbed Full endpoints, newer continuations, and any T2–T4 campaigns actually
present in a supported format. Inspect unsupported-protocol declarations before
concluding a benchmark was never run.

## Verification

```bash
.venv/bin/pytest -q tests/test_collect_dalla_fitted_models.py \
  tests/test_t1_brief_only_replay.py tests/test_t1_curve_replay.py
.venv/bin/ruff check scripts/collect_dalla_fitted_models.py \
  tests/test_collect_dalla_fitted_models.py
```

The exporter can be smoke-tested against the two local downloaded `inputs`
directories with repeated `--root`. Repeating an unchanged export must produce
an identical artifact. Changed output is refused rather than silently replaced.

Local verification on September 22: 21 focused tests passed, the real downloaded
checkpoint export and its repeat were identical, and all 21 saved records passed
the independent lowering/parameter-identity checks. Changed Python files pass
Ruff. Repository-wide Ruff still reports 37 unrelated errors in `analysis/claude`.
The full pytest run was stopped after 4 minutes 14 seconds with 422 passed and
five skipped; it was not completed and no whole-suite pass is claimed.
