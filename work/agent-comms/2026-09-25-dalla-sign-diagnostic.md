# CPU-only assisted R4 sign diagnostic

The user authorized proceeding while the automatic sign-review jobs remain
queued. This change adds a separate `dalla-sign-diagnostic-1` experiment, without
modifying those jobs, their frozen inputs, or the automatic discovery pipeline.

## Decisions and scope

The configuration selects the original rescued `full_perturbed_r4` model by
result, canonical request, and public-context hashes. Production is additive;
utilization and excretion are subtractive. Positive meal-filter contribution and
negative glucose self-clearance are explicit structural assumptions. Tissue
coupling remains unrestricted because the public channel name does not specify
a transfer direction. Inner functions, topology, and initialization policies are
preserved. Changed gain declarations receive the existing compatible-seed
policy's role-based starts, not absolute values of historical coefficients.

Every export records `assistant_specified_under_user_authorization`. This is a
post-hoc assisted diagnostic, not autonomous recovery or scientific
certification. There is no fabricated provider receipt or assessor response.
No reference model, test split, or intervention score enters fitting or selection.

## Implementation

- New diagnostic module validates source identity, applies the explicit patch,
  freezes repaired/control inputs, and exports labeled results and sign audits.
- New CLI and CPU submitter reuse existing rescue fitting and pruning. Two
  independent array elements each request one CPU, 16 GB and 2h15; a final report
  depends on array termination. No preparation or GPU dependency is needed.
- Submission intents, replies and confirmed IDs are persisted. An uncertain
  scheduler reply requires verified job adoption, not a blind retry.
- Portable ACES/Delta shell entrypoints use an immutable upload archive containing
  the original public rescue packet. The packet and archive are not committed.
- The configuration, operational documentation, tests and synthetic smoke are
  included. Project-context and pipeline-design notes describe the diagnostic.

## Validation

- 89 relevant pytest tests passed, including both sites, partial reporting,
  identity failures, no provider calls, independent arms, and exact resume.
- The synthetic smoke passed with real toy fitting and pruning, zero LLM calls,
  no benchmark data, and byte-identical resumed outputs.
- Importing the actual R4 packet validated its identities, preserved the
  initialization plan and unrestricted tissue term, and constrained exactly five
  outer gains. No benchmark fitting was run locally.
- Changed Python files pass Ruff; shell syntax and `git diff --check` pass.
- Repository-wide `ruff check .` reports 37 existing errors in unrelated
  untracked `analysis/claude` plotting files. Those files were left untouched.
- The full repository test suite was not rerun; validation used the relevant
  diagnostic, rescue, sign-repair and directional-review suites.

## Next inspection

Run the CPU diagnostic on one cluster. Download its top-level `summary.json` and
`models.json` to inspect retained equations, parameters, pruning and signs.
Only after freezing those endpoints should we inspect exploratory intervention
rollouts with fixed parameters. Compare with the separately saved automatic GPU
review when it completes. Correct signs do not guarantee an accurate model.

Commands and interpretation are in `docs/DALLA_SIGN_DIAGNOSTIC.md`.
