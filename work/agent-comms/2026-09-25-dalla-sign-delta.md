# R4 sign pilot: Delta execution alternative

The user's ACES R4 sign-review job 2162092 is pending on Priority; CPU fit
2162093 and report 2162094 wait on dependencies. The H100 serves GPT-OSS-20B
for the directional proposal and semantic assessment, not GPT-OSS-120B. The
4h30 CPU ceiling accommodates two sequential rescue/pruning arms (up to six
fits), plus replay/recovery/scoring overhead. It is not observed wall time.

Added `--delta` to `scripts/submit_dalla_sign_repair.py`. It reuses the shared
one-A40 server, pins the actual existing SIF digest in a separate immutable
configuration/plan, and uses Delta accounts and filesystem constraints.
`scripts/hpc/run_dalla_sign_repair_aces.sh` now selects site initialization from
the frozen platform; Delta preparation caches the exact model revision and
verifies vLLM 0.27.1. Review and fitting budgets remain unchanged. The archive
launcher is `scripts/hpc/launch_dalla_sign_delta.sh`; operational instructions
and resource explanations are in `docs/DALLA_SIGN_REPAIR_DELTA.md`.

Only `full_perturbed_r4` is selected from the original local public-only rescue
packet, verified before packaging. Its file SHA256 is
`08d883b623ce31235c979fc21cd4063327982de6504cce6564e461934ba25f94`.
Generated transfer archives and model/data packets are not committed.
There were no remote logins, live LLM calls, benchmark refits or test-data reads.

Validation:

- 87 focused tests passed: Delta routing/config immutability, ACES regression,
  directional review, sign repair, recovery and rescue.
- Synthetic `smoke_dalla_sign_repair.py --protocol dalla-sign-repair-2` passed,
  with two mocked provider calls and exact resume.
- Original packet import and selected-task verification passed.
- Changed Python files pass Ruff; both shell entrypoints pass `bash -n`.
- `ruff check .` still reports 37 existing issues in `analysis/claude/`.
- Full `pytest -q --maxfail=1` was started and interrupted after 153 passes and
  three missing-Torch skips (about 49 seconds); the full suite is not certified.

Delta GPU startup, model download and scheduler acceptance remain untested
locally. Hardware/image changes are explicit; bitwise equivalence and faster
queueing are not promised. Hold the pending ACES review before trying Delta;
do not run both and choose the outcome by intervention performance. The existing
ACES-specific recovery helper must not be used to recover a Delta submission.
