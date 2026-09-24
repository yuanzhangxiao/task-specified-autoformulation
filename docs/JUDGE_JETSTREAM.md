# Jetstream paired-judge compatibility pilot

This is one separately recorded execution of the existing scientific judge, not
a replacement for the pending ACES/Delta sign-recheck campaigns or a new judge
calibration. No optimizer, proposer, solver rollout, model selection, or test
data is involved. Run the client on the Mac; no GPU or Jetstream instance is
required for the authenticated hosted API.

The input is the self-sealed, already prepared ACES sign-recheck `plan.json`.
The portable importer verifies request hashes, model/context schemas, the frozen
scientific protocol, sign-evidence changes and placement linkage. It never
opens the original cluster paths. Seals verify internal consistency, not
external authenticity. The output contains a copy of the symbolic input.

By default, choose the affected request with the largest sorted JSON byte
representation, breaking ties by source index. This is a deterministic size
heuristic, not a token count or score-based choice. `freeze --review-index N`
can instead select one explicit affected source index. Exactly one pair is
frozen; other requests cannot be run by this pilot.

## What stays fixed

`repair_scientific_judge.perform_review` accepts an optional client factory.
Without that argument its existing behavior is unchanged. The pilot supplies
a `VLLMClient` subclass with a different HTTP transport. Prompts, scientific
schemas, post-schema validators, evidence, scoring, orientation consensus,
and cache/repair behavior remain in the existing implementation. Requests use
corrected `outer-factor-sign-2` evidence, low reasoning, temperature 0.2,
6144 output tokens, ten attempts per structured call and the existing two-seed
fallback policy. Seed attempts derive from the saved request.

Both atomic and comparative stages execute in both orientations. Usually this
needs four physical requests. Identical self-pair requests can use the existing
cache and need fewer; retries can need more. A transport cap of 80 requests is
the existing theoretical maximum (two seeds, two orientations, two stages,
ten attempts). It is not a new token or scientific-complexity limit.

Only the transport endpoint and model alias change:

| Item | Value |
| --- | --- |
| HTTP endpoint | `https://llm.jetstream-cloud.org/api/chat/completions` |
| Served alias | `gpt-oss-120b` |
| Prompt/protocol model name | `openai/gpt-oss-120b` (unchanged) |
| Authentication | Hidden prompt, or `AF_JETSTREAM_API_KEY` environment variable |

No credential is accepted as a CLI argument or written into artifacts. Redirects
are refused. Responses/errors are scrubbed for credential echoes. The managed
model revision, serving software version, and actual forwarding of requested
settings remain unverified. A returned alias or accepted seed field does not
prove that the original immutable weights or all settings were used. Keep this
execution separate; passing compatibility is not passing calibration.

## Run on the Mac

First download this file using the ACES file browser:

```text
/scratch/group/p.nairr260351.000/u.yx126462/judge-sign-recheck-v1/plan.json
```

Save it as `~/Downloads/judge-sign-aces-plan.json`. It contains public symbolic
requests, not trajectory datasets or model weights. The existing Delta copy
`/work/hdd/bibo/yxiao2/phase_b/judge-sign-aces-plan.json` is equivalent input.

Use the pinned commit supplied with the change. Extract an immutable source
archive under the local project's `tmp/` directory, keeping the existing
checkout and virtual environment available. For example, after setting
`AF_COMMIT` to that full commit:

```bash
(
  set -euo pipefail
  AF_BASE=/Users/yuanzhangxiao/Projects/autoformalism
  : "${AF_COMMIT:?Set the supplied commit first}"
  AF_REPO_ROOT="$AF_BASE/tmp/jetstream-${AF_COMMIT:0:7}"
  AF_PYTHON="$AF_BASE/.venv/bin/python"
  AF_OUTPUT_ROOT="$HOME/Downloads/judge-jetstream-pilot-v1"
  AF_SOURCE_PLAN="$HOME/Downloads/judge-sign-aces-plan.json"

  [[ -f "$AF_SOURCE_PLAN" ]] || {
    echo "Download the ACES plan as $AF_SOURCE_PLAN first."
    exit 1
  }
  git -C "$AF_BASE" fetch origin codex/prefit-aces-v1
  git -C "$AF_BASE" cat-file -e "$AF_COMMIT^{commit}"
  if [[ ! -e "$AF_REPO_ROOT" ]]; then
    mkdir -p "$AF_REPO_ROOT"
    git -C "$AF_BASE" archive "$AF_COMMIT" | tar -x -C "$AF_REPO_ROOT"
    printf '%s\n' "$AF_COMMIT" > "$AF_REPO_ROOT/SOURCE_COMMIT"
  fi
  [[ "$(cat "$AF_REPO_ROOT/SOURCE_COMMIT")" == "$AF_COMMIT" ]]
  export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
  export PYTHONDONTWRITEBYTECODE=1
  cd "$AF_REPO_ROOT"

  "$AF_PYTHON" scripts/judge_jetstream.py freeze \
    --source-plan "$AF_SOURCE_PLAN" --root "$AF_OUTPUT_ROOT"
  caffeinate -i "$AF_PYTHON" scripts/judge_jetstream.py run \
    --root "$AF_OUTPUT_ROOT"
)
```

Enter the API key at the hidden prompt. Keep the terminal open and the laptop
awake. Each HTTP attempt has the original 900-second timeout. `caffeinate`
prevents idle sleep; closing the laptop can still interrupt the run.

Repeating the block verifies the frozen identity and reuses a completed result
without asking for a key or making new calls. An interrupted in-flight review
retains its partial records and becomes `interrupted` on resume; it does not
silently obtain a fresh review budget. Do not delete markers to restart it.
`report --root ...` can inspect partial output without a key or new requests.

## Inspect

```bash
ROOT="$HOME/Downloads/judge-jetstream-pilot-v1"
cat "$ROOT/SUMMARY.md"
python3 -m json.tool "$ROOT/summary.json"
```

`summary.json` reports paired-review availability, physical attempts, token
usage, latency, schema attempts, stop reasons, returned model aliases, and any
returned system fingerprints. Missing usage stays missing, including interrupted
HTTP requests. `calls/call-NNN/` retains each request, dispatch marker and response,
including failed or schema-invalid attempts. `judge/` contains the usual parsed
caches and event log. The separately sealed `result.json` contains the review.
Scientific indeterminacy and disagreement remain valid outcomes; successful API
delivery is not a certification that either model is scientifically correct.

After live schema/long-request compatibility succeeds, a separately selected set
of existing labeled calibration cases is the next gate. This pilot does not run
that gate or automatically extend to the remaining saved pairs.

Official service documentation:
- https://docs.jetstream-cloud.org/inference-service/api/
- https://docs.jetstream-cloud.org/inference-service/api-examples/

Offline verification uses the real paired orchestration and provider parser with
mocked HTTP responses. Tests check prompt identity, fixed settings, self-pair
cache reuse, retries, accounting, key redaction, rejected redirects, tampering,
interruption, and deterministic resume. `scripts/smoke_judge_jetstream.py` runs
the complete offline import/review/report path without network access.
