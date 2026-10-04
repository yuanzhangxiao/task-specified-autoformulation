#!/bin/bash
# Smoke-test, then start, the Phase C LLM-SR budget pilot on a Jetstream2 VM.
#
# The smoke run makes two requests and goes through the whole path: the hosted
# endpoint, LLM-SR's own evaluator, our conversion and refit, a development
# rollout and the sealed result. Only if it finishes does the 10,000-sample
# pilot start, in the background, detached from this login.
#
# Rerunning is safe. A finished smoke test or pilot is never repeated, and a
# running pilot is not started twice. A pilot stopped part-way (a reboot, a
# killed process) restarts from its first sample, because LLM-SR cannot resume;
# the stopped attempt is kept beside it as results/0.interrupted-N.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly py="${AF_HOME}/venv/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly smoke="${AF_HOME}/runs/phase-c-llm-sr-smoke-v1"
readonly pilot="${AF_HOME}/runs/phase-c-llm-sr-budget-pilot-v1"

[[ -x "${py}" && -f "${AF_HOME}/setup_record.txt" ]] || {
  echo "run scripts/jetstream/setup_vm.sh first" >&2
  exit 2
}
cd "${repo}"
git diff --quiet && git diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export AF_LLM_SR_ROOT="${AF_HOME}/vendor/LLM-SR" AF_ENDPOINT_KIND=jetstream2_hosted

freeze() {  # config, root
  mkdir -p "$2"
  "${py}" scripts/phase_c_llm_sr.py prepare \
    --config "$1" --release "${release}" --root "$2" >"$2/prepare.json"
  git rev-parse HEAD >"$2/source_commit"
}
status_of() {  # root
  "${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$1/results/0/result.json"
}

echo "== smoke test: two requests through the whole path"
freeze configs/phase_c_llm_sr_smoke_v1.json "${smoke}"
if ! "${py}" scripts/phase_c_llm_sr.py run --root "${smoke}" --index 0 \
  >"${smoke}/run.log" 2>&1; then
  tail -n 40 "${smoke}/run.log"
  echo "the smoke test failed (log: ${smoke}/run.log); the pilot was not started" >&2
  exit 1
fi
smoke_status="$(status_of "${smoke}")"
echo "smoke test finished: ${smoke_status}"
[[ "${smoke_status}" != "endpoint_unavailable" ]] || {
  echo "the hosted endpoint did not answer; the pilot was not started" >&2
  exit 1
}

echo "== budget pilot: 10,000 samples"
freeze configs/phase_c_llm_sr_budget_pilot_v1.json "${pilot}"
if [[ -f "${pilot}/results/0/result.json" ]]; then
  echo "the pilot already finished: $(status_of "${pilot}")"
  exit 0
fi
if [[ -f "${pilot}/pid" ]] && kill -0 "$(cat "${pilot}/pid")" 2>/dev/null; then
  echo "the pilot is already running as process $(cat "${pilot}/pid")"
  exit 0
fi
nohup setsid "${py}" scripts/phase_c_llm_sr.py run --root "${pilot}" --index 0 \
  >>"${pilot}/run.log" 2>&1 </dev/null &
echo "$!" >"${pilot}/pid"
sleep 5
kill -0 "$(cat "${pilot}/pid")" 2>/dev/null || {
  tail -n 40 "${pilot}/run.log"
  echo "the pilot stopped at once (log: ${pilot}/run.log)" >&2
  exit 1
}
echo "the pilot is running as process $(cat "${pilot}/pid"); log ${pilot}/run.log"
