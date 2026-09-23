# Recover an accepted prepare job after a scheduler timeout

An ACES site wrapper can return exit code zero with empty stdout and a scheduler
socket timeout on stderr even though Slurm accepted the job. The original
continuation launcher correctly retains this uncertain receipt and stops before
submitting dependent jobs. Its nonempty intent directory must not be deleted to
force another submission.

`scripts/recover_review_continuation_submission.py` repairs this specific first
chain without editing the clean scientific checkout used by the accepted job.
Copy the script **outside** that checkout. Set `AF_REPO_ROOT` and `AF_COMMIT` to
the original checkout and commit; the helper imports its original plan verifier
and submitter. It checks the frozen plan, imported checkpoints, saved submission
prerequisites, Git identity, job owner, name, state and full submission command.
The original account and scientific resource allocations are retained.

The supplied prepare job is recorded in a separate adoption audit; the original
intent and timeout reply stay intact. A pending/running prepare remains an
`afterok` prerequisite. A completed prepare must have exit code `0:0`; successful
durable accounting is sufficient if its controller record has expired, and no
stale dependency is then submitted. Unavailable or incomplete accounting fails
closed. The helper queues proposer, fitting array, finish and the original
next-round dispatcher, and publishes the standard submission manifests. No
scientific artifacts, models, budgets, prompts or benchmark data are changed.

Completed recovery is idempotent. A recorded partial prefix is reused with exact
argument checks. Any later intent without an accepted ID stops recovery; another
ambiguous scheduler response is never blindly retried. This helper does not
recover failed preparation, later-round dispatchers, or an uncertain proposer,
fit, finish or dispatch submission. Those need separate scheduler inspection.

## ACES v8 invocation

Upload the recovery script to
`/scratch/group/p.nairr260351.000/u.yx126462/repos/` alongside the pinned checkout.
The downloaded archive checkout remains at commit
`f9c739273c6cef5fb6fe0e19fb9c6dfb459cc3e1`.

```bash
(
set -euo pipefail
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/repos/autoformalism-review-integrity-v8-upload
export AF_COMMIT=f9c739273c6cef5fb6fe0e19fb9c6dfb459cc3e1
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-review-integrity-r17-v1
export AF_VLLM_IMAGE=/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif
export AF_HF_HOME=/scratch/user/u.yx126462/huggingface-cache
export AF_COMPUTE_CACHE_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/runtime-cache/review
export AF_IPC_TMP_ROOT=/tmp/af-ipc-u.yx126462
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$AF_PYTHON" /scratch/group/p.nairr260351.000/u.yx126462/repos/recover_review_continuation_submission.py \
  --root "$AF_OUTPUT_ROOT" --prepare-job 2157971
)
```

Verification uses fake Slurm responses and real frozen synthetic v8 plans. It
checks exact resume, dependency wiring, unchanged science/receipts, wrong-owner
and wrong-command rejection, failed preparation, expired controller records,
and refusal to repeat another uncertain submission:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_review_continuation_submission_recovery.py tests/test_review_continuation.py
.venv/bin/ruff check scripts/recover_review_continuation_submission.py \
  tests/test_review_continuation_submission_recovery.py
```
