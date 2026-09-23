# Multi-target Dalla Man pilots

The user asked Codex to implement the other pilots directly, including T2's
multiple required targets. This supersedes the earlier advisory-only scope for
this milestone. No message was sent to another agent; no remote session or job
submission was performed.

The implementation and user-run commands are documented in
`docs/DALLA_MULTI_TARGET_PILOTS.md`. The default pilot is canonical named T1-hard,
T2-easy and T2-hard, Full only, seeds 0/1, three visits per lineage. There are six
lineages and 18 planned visits. This replaces the earlier T1-only implementation
handoff with one combined fresh pilot; do not independently submit a duplicate
T1-hard pilot for these seeds without checking the user's submission records.

The versioned `collocation-multi-target-v1` profile jointly fits all required
outputs with separate training scales, including algebraic U. It preserves the
existing numerical budgets and causal replay contract. Old single-target
profiles and default campaign matrices retain their restrictions and ordering.
The new opt-in `full_only` configuration is restricted to fresh v2 campaigns.
Multi-target continuation/import/recovery research protocols are not enabled.

Reports now retain per-target training and validation errors for both trial and
selected models. Training evidence and revision feedback already handled all
channels; tests verify this. The Mac public-file staging helper accepts the
pilot config. The ACES launcher prints the frozen matrix and public hashes before
submitting, and its CPU preparation checks multi-target fitting before GPU work.

Scientific interpretation remains pending live results. The local controls
verify all-output residuals/Jacobians, per-target fitting, algebraic revisions,
causal validation, failure dimensions, immutable resume, and submission behavior.
They do not demonstrate Dalla Man recovery or predict intervention accuracy.

Local verification: full `pytest -q -n 4` completed with 2,973 passed and eight
skipped (optional Torch unavailable). The two-/three-output real-fit smoke and
the existing three-visit revision smoke passed, including exact resume. Shell
syntax and Ruff on every changed Python file passed. Repository-wide Ruff
reported 37 existing errors under untracked `analysis/claude`; those files were
not changed or committed.
