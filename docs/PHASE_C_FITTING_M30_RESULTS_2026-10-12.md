# M30 results: fully profiled stopping works; general speedup remains unproven

Reviewed `review-20261013-024409.tar.gz` on October 12, 2026 (Hawaii time;
the archive name uses UTC). All 22 entries are accounted for. Both arms retain
eight usable fitted endpoints, one numerically unavailable endpoint and two
original construction failures. All three known-structure controls remain
accurate. No new accurate constructed model emerges.

The exact output-dependency fix succeeds: independent basin needs one search
evaluation instead of 158 and returns the same analytically optimal gain and
prediction scores. Its remaining error comes from its restricted output equation.
The general stationary-poor-fit rule fires conservatively but does not establish
a broad computational saving. Whole-campaign recorded task time increases from
107.03 to 113.00 minutes in this single paired run.

## Provenance and verification

- Protocol: `phase-c-poor-fit-qualification-1`; [runbook](PHASE_C_FITTING_STATIONARY.md).
- Code commit: `879bdcdf03e6ad2740048b3e98ec0f7b03527d59`.
- Source-tree SHA256: `9df8689ce5092ca6d2078593c360691c07e1300195bba90be5077dc21671fa6d`.
- Plan SHA256: `ef6f49623e2efed94618e7e9f3a5bd635951db8ff85e32a372982ee10360d4bc`.
- Input SHA256: `cce09c7f7b25573cd7c0c1549b15534d78a18f0e1fff631ebae8cf0148517744`.
- Archive SHA256: `a99567235f64d6adf933c68e1ae0b7614658563815e522cd8f3865b4fc610698`.
- Delta jobs: preparation `22856746`, array `22856747`, report `22856748`.
- Output root: `/work/hdd/bibo/yxiao2/phase_c/fitting-poor-fit-v1`.
- Runtime: Python 3.12.13, NumPy 2.3.5, SciPy 1.16.3, CasADi 3.7.2,
  Pydantic 2.12.5. One CPU per entry, three concurrent entries, no GPU/LLM calls.

All 664 sealed JSON artifacts validate. Rebuilding the paired report reproduces
the supplied summary exactly. The source hash matches the pinned delivery;
subsequent construction-only commits in the project do not alter this experiment.
The inputs equal the frozen M29 export. All nine available pairs have identical
training problems and raw starts. All eighteen lowered requests match their
recorded candidate hashes; all sixteen returned parameter vectors respect their
declared bounds. Known-control coefficient errors recompute exactly.

There are 154 terminal fitting operations: 146 complete and eight unavailable
checks. All sixteen independent evaluator attempts finish successfully. Accounting
is complete in both arms, including failed work. Validation is evaluated only
after fitting; no reference coefficients enter the fitter and no test data are
opened. Frozen inactive coordinates stay at their incumbent values in every
effective restart and returned model.

These are the same **September 30 archived constructions** as M29, not current
construction outputs. The counts include both arms: sixteen `complete`, two
`retained_unverified`, and four `source_unavailable`. A complete fit does not
mean an accurate fit.

## Matched prediction and computation

Times below include request preparation, fitting/checks and the separate
independent evaluator. They are sums of recorded task time, not queue time,
campaign elapsed time or billed service units. Scores are independently replayed
primary-solver NMSEs after the two-solver agreement check.

| Available model | Calls, legacy → new | Total seconds, legacy → new | New train / validation NMSE |
|---|---:|---:|---:|
| Constructed T1 easy | 104 → 104 | 798.9 → 775.3 | 0.6532 / 0.5568 |
| Constructed T1 hard | 57 → 57 | 165.0 → 176.1 | 0.6600 / 0.5288 |
| Constructed T2 hard | 240 → 240 | 617.9 → 588.3 | 0.9842 / 1.0071 |
| Constructed CSTR easy | 171 → 169 | 3518.4 → 4081.3 | 1.0363 / 0.4342 |
| Constructed coupled basin | 4 → 4 | 13.6 → 16.6 | unavailable |
| Constructed independent basin | 158 → 1 | 238.1 → 33.0 | 2.6330 / 0.8338 |
| Known alien hard | 12 → 12 | 562.6 → 572.8 | 8.36e-15 / 1.91e-14 |
| Known coupled basin | 7 → 7 | 152.5 → 177.5 | 1.51e-16 / 2.57e-16 |
| Known CSTR hard | 28 → 28 | 355.0 → 359.1 | 3.39e-13 / 1.21e-13 |

T1 and constructed CSTR retain exactly the same parameters and scores. T2 changes
slightly after removing its inactive coefficient from optimization: legacy
train/validation NMSE is 0.984611/1.007569, new is 0.984182/1.007117. Both remain
poor. Basin and known-control differences are at numerical precision.

Total search calls decrease from 781 to 622; 157 of the 159 saved calls come from
independent basin. Total recorded time rises by 5.58%, and fitting CPU time rises
from 5080.0 to 5234.0 seconds. There is no demonstrated whole-suite speedup.
Timing variation and wall-clock cutoffs matter; this is one run per arm.

## Exact independent-basin result

The measured state obeys

\[
\dot h=g_0 u_{\rm down}(t)/A_{\rm down},\qquad g_0\ge0.
\]

The omitted drainage term remains omitted. The other state, two additional gains
and four shared-initializer coefficients do not influence this measured state.
The dependency analysis correctly marks six of seven unknowns unestimated and
preserves their values and declarations.

The bounded inner solve returns `g0 = 0.08996330802042875`, agreeing with the
training-only closed-form optimum derived in the [M29 review](PHASE_C_FITTING_M29_RESULTS_2026-10-12.md).
Training NMSE remains `2.6329948436703523`; validation remains about
`0.833750286431`. Independent verification accepts the original equations.

The legacy arm repeats four flat outer-optimizer failures and uses joint
fallback. The new arm finishes in one block/one call with
`search_status=output_problem_solved` and `recommended_action=review_equations`.
Search time falls from 202.05 to 2.20 seconds; total time from 238.06 to 33.03
seconds, about a 7.2-fold total speedup. The case-specific optimizer defect is
resolved. The best achievable fit of these equations is still poor.

## What the general stopping rule does and does not establish

The rule fires in two T1-hard branches and one CSTR continuation. All four frozen
starts are still visited. None of these three witnesses certifies global
optimality, scientific correctness or parameter accuracy.

T1 hard uses exactly the same 57 calls. In two branches, the new stationarity rule
stops at the evaluation where the legacy solver also terminates. The retained
point came from the first branch, which has no new stationarity certificate, so
its final assessment correctly remains `further_search_needed`. A witness on a
different point must not be silently attached to that retained parameter vector.

CSTR similarly retains the first branch's parameters. The new stationarity rule
acts only on a later continuation. Its first two branches have identical evaluated
loss sequences and best parameters across arms, but take longer in the new arm.
Two subsequent new-arm blocks hit the 800-second limit, whereas the corresponding
legacy blocks reach their call ceilings. This changes the intermediate continuation
point and reduces completed work. The two fewer calls therefore cannot be credited
as evidence of successful early stopping. The archive does not isolate the cause
of per-evaluation timing differences; it would be unjustified to attribute the
whole slowdown to the stopping-rule calculations or to hardware alone.

The other structural findings from M29 persist: CSTR's jacket exchange has the
wrong outer sign, coupled basin adds discharge to storage and fails with
`symbolic RHS is nonfinite` at every start, and the T2 output ignores its declared
`U` process. Only the unused T2 coefficient is frozen; its active insulin-action
scale symmetry remains. These equation findings are separate from optimizer
certificates. No signs, mechanisms or public requirements were changed here.

## Accurate recovery is preserved

All three known controls reach the strict training target in the first block.
The new rule does not interrupt near-zero-loss coefficient polishing.

| Control | Maximum coefficient relative error (fraction) | Maximum hidden-initial absolute error |
|---|---:|---:|
| Alien hard | 1.22e-6 | 8.27e-7 |
| Coupled basin | 7.03e-8 | no fitted hidden initials |
| CSTR hard | 1.81e-5 | 7.56e-5 |

Errors are effectively unchanged across arms. Joint sensitivity ranks remain
18/18, 2/2 and 9/9 respectively. Initial errors use each parameter's physical
units. These results do not establish global identifiability or performance on
arbitrary constructions or noisy data.

## Recommended next step

Retain exact output-dependency handling and the all-inner return path. Keep the
stationarity rule conservative, without advertising a broad speedup or loosening
accurate-fit tolerances based on this run. Its classification behavior is useful,
but its general computational benefit remains unproven.

The next useful qualification should use frozen candidates from the current
construction stages through the existing public-request adapter, with the same
training-only fitting contract. Prepare compact fitting feedback for construction:

1. Separate verified accurate fit, solved-but-poor output problem, locally
   stationary search, exhausted search and unavailable numerical rollout.
2. Include per-target training residuals, certified inactive parameters and
   missing output paths, with equation/parameter provenance. Report sensitivity
   on active unknowns separately from already-certified inactive coordinates;
   include latent initial-map coefficients in the active joint analysis.
3. Keep scientific sign/mechanism findings advisory and separately sourced.
   Fitting must not silently repair a candidate. Validation and reference
   coefficients must stay out of proposer feedback and fitting decisions.

More budget on these same archived defective structures is a low priority.
Production promotion and broader efficiency claims require fresh construction
coverage. No subsequent experiment is implemented or launched by this review.

## Local verification

The 63-test preparation suite passes locally and in the archived Delta log.
The two-output/shared-initializer smoke passes, including unchanged resume and
independent replay. No benchmark fit was rerun locally. Repository-wide
`ruff check .` retains the same 37 pre-existing issues under `analysis/claude`.
This review changes documentation only.
