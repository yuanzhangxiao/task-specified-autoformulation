# Response-oriented feedback pilot

`review-deadline-7` continues fitted development checkpoints with
`training-response-evidence-1`. It is opt-in. Previously submitted protocol-6
jobs and their pinned checkouts keep their original behavior.

The prospective [protocol-8 correction](REVIEW_INTEGRITY_V8.md) adds explicit
repair of conflicting existing parameter declarations and validation tolerance
before complexity tie-breaking. It preserves this response-evidence policy and
the public scientific requirements.

The diagnosed T2-easy requests had about 39,200–40,200 raw message tokens before
chat formatting, against a 32,768-token server context and an 8,192-token output
allowance. The old packet was not literally every sample: it repeated summaries
for 48 trajectory/target rows and 144 time windows, plus six detailed sample
sets. Its 198 reference identifiers also inflated repair feedback.

## Evidence shown to the proposer

The fitter still uses all training observations. Fixed-parameter training replay
computes these features on the complete time grid, independently of the sparse
samples stored in the older packet:

- Each target has an overview: sample-weighted NMSE, signed bias in training-SD
  units, trajectory error range, and trajectory/sample counts.
- Each target gets at most three trajectory examples: largest error, smallest
  error, then an input-departure pattern contrast when available. Ties are
  deterministic. All target overviews remain present as examples are reduced.
- Each example compares observed/predicted initial and final values, range,
  signed dominant excursion, peak timing, sampled half-return time, turning-point
  count, and early/middle/late NMSE.
- Input summaries include initial/final/range, first sampled departure from the
  initial level, and the number of departures. A peak delay relative to that
  departure is shown only for one changing input with one departure and a unique
  interior response peak after the departure. Repeated/overlapping inputs retain
  their response shapes without a claim that a particular input caused the peak.
- At most three already recorded diagnostic samples per selected example may
  supplement these summaries. Samples are removed first when packing a request.

For a sampled signal `y(t_i)`, the dominant excursion is the signed value
`y(t_k)-y(t_0)` at the maximum absolute departure from the initial level. A flat
signal, tied maximum, or endpoint maximum has no uniquely reported interior
peak time. For a unique interior peak, half-return is the first later sampled
time with `|y(t_i)-y(t_0)| <= |y(t_k)-y(t_0)|/2`, minus `t_k`. This is a sampled
return toward the initial value, **not an estimated exponential decay constant**.
An unfinished return is marked `not_reached`. A peak-time difference is reported
only when both unique interior excursions have the same sign. Time units are
the public trajectory units; no interpolation or latent truth is introduced.

The presentation uses six significant digits. Equations, parameter values,
initialization rules and saved fitting artifacts keep their original precision.
Only displayed `R...` references can resolve in the citation audit. Unresolved
citations remain advisory and cannot certify evidence or bypass equation checks.

## Context and delivery handling

The serving tokenizer checks the complete system/user messages before generation.
The input allowance is `min(24000, 32768 - output_allowance - 512)`. Packing tries
three examples per target with samples, three without samples, two, then one.
The public task, equations, parameters, initialization and all target overviews
are retained. Oversized requests are recorded without being sent to generation.
Tokenization requests have their own cache and do not count as LLM generations.
The endpoint contract uses vLLM's documented
[TokenizeChatRequest and TokenizeResponse](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/openai/protocol.py).
Each GPU allocation probes its actual endpoint before running the proposer.

An HTTP/delivery error records its bounded response body and stops proposal
retries for that visit. The fitted incumbent is retained without an unnecessary
fallback fit. The finish job writes the report, then fails the dependency barrier
so automatic dispatch does not repeat the delivery failure in future rounds.
Invalid *equations* still receive up to three correction attempts and the
existing bounded fallback-fit policy. These are distinct outcomes.

## Run on ACES

Use a clean checkout pinned to the commit supplied with this change. Do not pull
inside the checkout used by already submitted jobs. No new data or fitter profile
is needed. The default source is the user's six-task T1-hard/T2-easy/T2-hard
continuation. All tasks must have a fitted incumbent at the imported round.

If protocol 6 still has queued future rounds, optionally stop its pending
**dispatchers only**, allowing already submitted proposer/fit/finish work to
complete. Inspect the listed jobs before the cancellation command:

```bash
AF_SOURCE_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-continuation-v2
AF_DISPATCHERS=$(jq -r '.jobs | to_entries[] | select(.key | startswith("dispatch-")) | .value' "$AF_SOURCE_ROOT/submission_manifest.json" | paste -sd, -)
if [ -n "$AF_DISPATCHERS" ]; then
  squeue -h -j "$AF_DISPATCHERS" -t PENDING -o '%i %j %T'
  squeue -h -j "$AF_DISPATCHERS" -t PENDING -o '%i' | xargs -r scancel
fi
```

Wait for the already submitted source round to finish before importing. Import
uses an exclusive execution lock and refuses to race an active source worker.
The launcher chooses the latest round with a result for every task, by completion
order rather than scores. It records the chosen global round and exact result
hashes. An existing destination retains its original choice on rerun.

```bash
module load GCCcore/13.2.0 Python/3.11.5
export AF_REPO_ROOT=/path/to/the/new/pinned/checkout
export AF_PYTHON=/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python
export AF_SOURCE_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-mechanism-continuation-v2
export AF_OUTPUT_ROOT=/scratch/group/p.nairr260351.000/u.yx126462/dalla-response-feedback-v1
export AF_ADDITIONAL_VISITS=3
unset AF_SOURCE_ROUND
bash "$AF_REPO_ROOT/scripts/hpc/submit_review_response_aces.sh"
```

The CPU prepare job runs focused tests and a synthetic smoke test, then computes
one bounded fixed-parameter training replay per imported model. It saves a
readable `response-preview.json`. The GPU proposer starts only after prepare
succeeds. Each of the three visits gets the existing proposer/CPU-fit/finish
chain. No live LLM calls occur during preparation. The fitter remains the
source's `collocation-multi-target-v1` profile. New trial fits compute their
response features during the already required training replay.

Check submissions and generate the report (the full rows are in the file,
not in the CLI's shortened stdout):

```bash
AF_JOBS=$(jq -r '.jobs | [.[]] | unique | join(",")' "$AF_OUTPUT_ROOT/submission_manifest.json")
squeue -j "$AF_JOBS"
sacct -j "$AF_JOBS" --format=JobID,JobName%28,State,ExitCode,Elapsed
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_deadline.py" report --root "$AF_OUTPUT_ROOT"
jq '{status_counts, proposal_status_counts, delivery_failure_visits, max_prompt_input_tokens, fallback_fits,
     latest: ([.rows[] | select(.status != "missing")] | group_by(.task_id) | map(max_by(.round) |
       {cell,seed,round,proposal_status,retained_train_per_target_nmse,retained_validation_per_target_nmse}))}' "$AF_OUTPUT_ROOT/summary.json"
PYTHONPATH="$AF_REPO_ROOT/src" "$AF_PYTHON" "$AF_REPO_ROOT/scripts/review_response.py" audit --root "$AF_OUTPUT_ROOT" > "$AF_OUTPUT_ROOT/request-audit.json"
jq '{physical_generation_requests, delivery_failures, requests}' "$AF_OUTPUT_ROOT/request-audit.json"
```

After CPU prepare, inspect `response-preview.json` for the actual response
evidence. Cached generated requests contain the exact packed prompt and its
`prompt_preflight` token counts. `calls/tokenization/` holds tokenizer results.

## Verification and limits

Local checks:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/test_response_evidence.py tests/test_response_revision.py tests/test_review_multi.py tests/test_review_continuation.py
PYTHONPATH=src:. .venv/bin/python scripts/smoke_review_response.py --output /tmp/response-smoke
PYTHONPATH=src:. .venv/bin/python -m pytest -q -n 4
.venv/bin/ruff check .
```

The tests cover sampled peak/return definitions, ambiguous timing, repeated input
events, train-only identity checks, 16-trajectory/three-target packing, omitted
citations, output reservation, HTTP diagnostics, incumbent preservation, exact
resume and the continuation stop barrier. The smoke uses real CPU fits with a
mock proposer/tokenizer. Live serving compatibility and better scientific
revisions still require the ACES pilot; software tests do not establish those.

The current response builder requires the saved residual packet to include every
trajectory/target summary (the six-task pilot's largest packet has 48 rows,
below its 64-row limit). Features describe the whole trajectory's dominant
excursion, not separate impulse responses to each meal. Auxiliary effects,
overlapping inputs, initial transients and coarse sampling can confound timing.
No fitted coefficient, task requirement or scientific conclusion is inferred
solely from these shapes. Full numerical fitting still determines rollout error.
Validation remains model-selection evidence; it is never proposer feedback.
