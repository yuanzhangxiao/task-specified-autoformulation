# Finding fitted Dalla Man models for mechanistic inspection

This is a post-selection diagnostic. It does not change candidates, rerun fits,
choose benchmark winners, call a proposer, or read held-out test files.

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
