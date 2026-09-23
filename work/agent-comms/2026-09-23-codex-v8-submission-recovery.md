# V8 prepare submission recovery

The user confirmed job 2157971 is the accepted prepare job for
`dalla-review-integrity-r17-v1`. Slurm's recorded submission command and output
paths match the intended pinned checkout. The site wrapper returned code zero,
empty stdout and a socket-timeout error, so the ordinary launcher stopped without
recording a job ID or submitting proposer/fit/finish/dispatch jobs.

Added `scripts/recover_review_continuation_submission.py`, a standalone scheduler
helper that is uploaded outside the running checkout. It verifies the scheduler
record, user, complete original argv, clean original Git commit, sealed plan and
imported prerequisites, adopts the accepted job, and submits only the missing
chain. Successful completed preparation can be verified through accounting after
its controller record expires. Original receipts, scientific inputs, code and
budgets remain unchanged. The original dispatcher handles later rounds.

The helper is intentionally limited to the first continuation submission with a
known accepted prepare job. It refuses failed preparation, wrong identities and
any further uncertain scheduler submission. A confirmed partial prefix and a
completed recovery are idempotent. It does not claim generic Slurm recovery.

Verification: 36 tests passed across the new recovery tests and existing
continuation tests. Tests use real frozen synthetic v8 plans and mocked scheduler
responses, including the observed false-success socket timeout. CLI help smoke,
changed-file Ruff and Git whitespace checks passed. Whole-tree `ruff check .`
still reports 37 pre-existing issues in untracked `analysis/claude` files.
No ACES/Delta session was opened and no remote experiment was submitted locally.

Usage and exact ACES paths are documented in `docs/REVIEW_SUBMISSION_RECOVERY.md`.
