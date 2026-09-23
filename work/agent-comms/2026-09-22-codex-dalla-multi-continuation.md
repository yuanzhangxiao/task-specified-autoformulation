# Dalla Man continuation corrections

The user authorized implementing the three controller corrections while the
original ACES round-2 retry runs. No remote session or job submission was made
from this task. The experiment runbook is
`docs/DALLA_MULTI_TARGET_CONTINUATION.md`.

## Decisions

- New `review-deadline-6` imports complete round-2 checkpoints into a separate
  campaign. Twelve additional visits give fifteen total visits (indices 0–14).
  It reuses the original joint-output fitting profile and budgets.
- Revision responses use explicit target-addressed output mappings; omitted
  mappings and equations remain unchanged. Existing parameter declarations and
  causal initialization remain authoritative.
- Unresolved citations are warnings, never verified evidence. Executable model
  checks and public scientific predicates still gate fitting.
- The current frozen target contract expects algebraic `U`. This is the existing
  runtime interpretation of its public rate description, not a theorem excluding
  differential rate models. The interpretation is now exposed before construction
  and on repair requests. Benchmark prompts and gate semantics were not changed.
- T2-easy's recorded unfitted drafts are copied with their source hashes and can
  receive three repair attempts per visit. Their prompts contain public findings
  and descriptive training evidence, without fabricated fitting evidence. A type
  conversion must be explicitly proposed. Failing drafts can continue later.
- Failed revisions preserve fitted incumbents and use at most the current visit's
  unchanged-model refit allocation. Invalid/worse fits cannot erase the incumbent.
- New-protocol fit jobs depend on proposer success; finish jobs fail on missing
  results and later dispatch requires successful finish. Historical launch paths
  retain their old behavior.

## Operations and verification

Use an independent group-scratch clone, group-scratch compiler caches and short
node-local IPC paths. A linked worktree may still write metadata into the old
quota-exhausted personal repository. Keep the original pilot checkout at 4113fc1.
The continuation importer refuses incomplete source rounds. No test/private
reference material is imported or opened.

Final focused regression run: 124 passed. Both the new joint-output smoke and
the historical revision smoke passed with real fitting and mocked LLM transport.
The full suite produced 3,012 passes and eight skips (Torch unavailable), plus
four failures and one setup error, all in frozen-source identity checks. Other
source files appeared in the shared checkout during that run. All five affected
checks passed on a subsequent rerun without changes to their implementation.
The final focused tests also cover refit-only controls and unexecutable fresh
construction, added after that full run.

Ruff passes on all changed Python files; shell syntax and `git diff --check`
pass. Repository-wide Ruff currently reports 38 issues outside this change:
37 under untracked `analysis/claude`, and one in the concurrently added
`src/autoformalism/rebuttal/general_critic.py`. These files were preserved.

The new tests establish interface behavior, budget preservation and deterministic
resume. They do not establish improved live scientific recovery. ACES confirmation
and inspection of fitted per-target trajectories and mechanisms remain necessary.
