#!/bin/bash
# Smoke-test, then start, the Phase C LLM-SR budget pilot on a Jetstream2 VM.
#
#   start_llm_sr_pilot.sh               smoke test, then the pilot
#   start_llm_sr_pilot.sh --smoke-only  smoke test only
#
# The smoke run makes two requests and goes through the whole path: the hosted
# endpoint, LLM-SR's own evaluator, our conversion and refit, a development
# rollout and the sealed result. The 10,000-sample pilot starts, in the
# background and detached from this login, only if the smoke run finished, at
# least one sample the model wrote scored, and no answer was replayed from the
# hosted service's cache. LLM-SR's own starting program always scores, so a
# finished smoke run alone does not show that the model's replies can be read;
# v1 of this pilot ran for hours on replies that could not. Its two requests
# usually carry the same prompt, as LLM-SR's islands all begin from one, so a
# cache the requests failed to bypass would show here.
#
# Rerunning is safe. A finished smoke test or pilot is never repeated, and a
# running pilot is not started twice. A pilot stopped part-way (a reboot, a
# killed process) restarts from its first sample, because LLM-SR cannot resume;
# the stopped attempt is kept beside it as results/0.interrupted-N.

set -euo pipefail

smoke_only=0
case "${1:-}" in
  "") ;;
  --smoke-only) smoke_only=1 ;;
  *)
    echo "usage: $0 [--smoke-only]" >&2
    exit 2
    ;;
esac

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly py="${AF_HOME}/venv/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly smoke="${AF_HOME}/runs/phase-c-llm-sr-smoke-v2"
readonly pilot="${AF_HOME}/runs/phase-c-llm-sr-budget-pilot-v2"

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
result_value() {  # root, dotted key; 0 when the result does not carry it
  "${py}" - "$1/results/0/result.json" "$2" <<'EOF'
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value.get(key, 0) if isinstance(value, dict) else 0
print(value)
EOF
}

echo "== smoke test: two requests through the whole path"
freeze configs/phase_c_llm_sr_smoke_v2.json "${smoke}"
# A search waits out a hosted-service outage for hours; this one runs in the
# foreground and normally takes a few minutes, so it is not allowed to.
if ! timeout 30m "${py}" scripts/phase_c_llm_sr.py run --root "${smoke}" \
  --index 0 >"${smoke}/run.log" 2>&1; then
  tail -n 40 "${smoke}/run.log"
  echo "the smoke test failed (log: ${smoke}/run.log); the pilot was not started" >&2
  exit 1
fi
smoke_status="$(result_value "${smoke}" status)"
written="$(result_value "${smoke}" accounting.model_samples)"
scored="$(result_value "${smoke}" accounting.model_samples_scored)"
replayed="$(result_value "${smoke}" accounting.cache_hits)"
echo "smoke test finished: ${smoke_status}"
echo "samples the model wrote: ${written}; of those, scored by LLM-SR: ${scored}"
echo "answers replayed from the service's cache: ${replayed}"
[[ "${smoke_status}" != "endpoint_unavailable" ]] || {
  echo "the hosted endpoint did not answer; the pilot was not started" >&2
  exit 1
}
[[ "${scored}" -gt 0 ]] || {
  echo "no sample the model wrote could be scored; the pilot was not started" >&2
  exit 1
}
[[ "${replayed}" -eq 0 ]] || {
  echo "the service replayed cached answers; the pilot was not started" >&2
  exit 1
}
if [[ "${smoke_only}" == 1 ]]; then
  echo "smoke test only; the pilot was not started"
  exit 0
fi

echo "== budget pilot: 10,000 samples"
freeze configs/phase_c_llm_sr_budget_pilot_v2.json "${pilot}"
if [[ -f "${pilot}/results/0/result.json" ]]; then
  echo "the pilot already finished: $(result_value "${pilot}" status)"
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
