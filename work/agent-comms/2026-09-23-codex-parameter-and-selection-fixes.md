# Public T2 requirement and two runtime corrections

The frozen canonical named T2 easy/hard public prompts require meal effects on
glucose and delayed insulin action on disposal. They require generating insulin,
but do not prescribe a meal/glucose-to-insulin production pathway. Consequently,
no extra scientific gate, dependency hint, or benchmark modification was added.
Observed-versus-predicted training responses remain the evidence for inferring
missing dependencies.

Implemented as prospective `review-deadline-8`:

1. Explicit conflicting roles for existing parameters now reject the transaction
   with a structured repair message. Canonical names and displayed aliases are
   both checked. The proposer can retain the existing meaning or explicitly use
   a fresh name; the runtime does not guess. Compatible references still inherit
   declarations without fixing the numerical value.
2. Validation selection uses `1e-8 + 1e-6 * max(abs(a), abs(b))` as a fixed
   comparison band. Fewer outer additive terms win within the band; equal
   complexity retains the incumbent. Round records contain the comparison audit.

Historical protocol dispatch and saved results retain their earlier semantics.
The new phase imports incumbents exactly; it does not retroactively remove a
state selected by v7. The fitter and public response-summary contents are unchanged.

Offline checks on the supplied final-v7 archive caught all four saved role
conflicts (T1-hard seed 0 rounds 15–17; T2-easy seed 1 round 16). The v8 comparison
kept the incumbent for the T2-hard seed-0 round-16 change from 9 to 11 terms with
only `3.3997e-11` lower validation NMSE. These checks used saved responses and
scores, with no new provider calls or fitting, and are not counterfactual live
repair success measurements.

Focused regression tests: 73 passed. Both v7 and v8 real synthetic three-output
fitting smoke tests passed, including caching, exact resume and immutable source
imports. `ruff check src scripts tests` and shell syntax checks passed.
Whole-repository Ruff reported 37 existing findings under untracked
`analysis/claude/` files, which were left untouched.

The implementation was included in shared-checkout commit `f9c7392` by another
task while the broad regression run was in progress. That commit also removes
an unrelated roster-median tool and its tests. Its history has not been rewritten.
Separate concurrent edits to the judge and LLM-SR modules appeared during
verification. Full-suite results:

- Initial parallel run: **3,120 passed, 8 skipped, 7 failed, 1 setup error**.
  Several failures explicitly rejected changed source/checkpoint identities.
  The skips are missing optional Torch dependencies.
- Serial rerun of those eight cases: **5 passed, 3 failed**; a child-fit identity
  guard still detected changing source, and two scaled-fitting cases failed.
- To remove source-edit interference, the eight cases were rerun against a static
  copy of committed `0600c53` inside ignored project artifacts, with one numerical
  thread: **5 passed, 3 failed**. All source/checkpoint identity failures passed.
  Remaining failures were the existing fitter-recovery submission test's
  20-second subprocess timeout and both scaled-alternating fitting tests' strict
  recovery assertions. These tests and their fitter implementations were not
  changed in this milestone. They remain unresolved; this is not a clean full-suite
  pass. No numerical budgets or assertions were weakened to make them pass.

Implementation/launcher files changed: the v4 parameter resolver and multi-output
adapter, response-revision dispatch, deadline plan/import/selection/reporting,
normal and recovery selection, response smoke, and ACES continuation launchers.
The new `review_integrity` module defines the fixed selection policy. New regression
tests exercise aliases, canonical collisions, explicit correction, immutable
incumbents, version isolation, finite-score selection, frozen policies and resume.

The complete policy, limitations and ACES continuation commands are in
[`docs/REVIEW_INTEGRITY_V8.md`](../../docs/REVIEW_INTEGRITY_V8.md). The new launcher
imports global round 17 and defaults to three new visits, with CPU regression and
smoke gates before H100 proposer work. No remote sessions or jobs were started.
