#!/bin/bash
# Run every task of one nine-case Phase B plan against the Jetstream2 hosted
# model, detached from the login, a few at a time.
#
#   run_phase_b_hosted.sh METHOD CONFIG NAME [TASKS]
#
#   METHOD  d3, hdtwingen or spec_sindy; it must be the plan's method
#   CONFIG  a plan declaring the jetstream2_hosted endpoint
#   NAME    the run's directory under ~/af/runs
#   TASKS   how many tasks run at once (default 1)
#
# The staged nine-case inputs must be at ~/af/release/phase-b-nine-case-v1; a
# plan that reads the Phase C release (inputs_protocol phase-c-development-2)
# reads it at ~/af/release/phase-c-public-v2.
# Rerunning is safe: a finished task is not repeated and a running one is
# refused by its lock. A task an outage stopped (exit status 3) resumes where
# it stopped. When every task has ended, the report is NAME/report.json.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly self="${repo}/scripts/jetstream/run_phase_b_hosted.sh"
readonly py="${AF_HOME}/venv/bin/python"
readonly hosted="https://llm.jetstream-cloud.org/gpt-oss-120b"

usage() {
  sed -n '2,11p' "${self}" >&2
  exit 2
}

script_for() {  # the CLI that runs a method's plans
  case "$1" in
    d3 | hdtwingen) echo "scripts/phase_b_nine_case_d3.py" ;;
    spec_sindy) echo "scripts/phase_b_nine_case_spec_sindy.py" ;;
    *) usage ;;
  esac
}

worker() {  # root, script, count, tasks: runs in the foreground
  local root="$1" script="$2" count="$3" tasks="$4"
  echo "started $(date -u +%FT%TZ): ${count} tasks, ${tasks} at a time"
  seq 0 $((count - 1)) | xargs -P "${tasks}" -I{} bash -c \
    '"$1" "$2" run --root "$3" --index "$4" >>"$3/logs/task-$4.log" 2>&1 || echo "task $4 exited with status $?"' \
    _ "${py}" "${script}" "${root}" {}
  "${py}" "${script}" report --root "${root}" >"${root}/report.json" ||
    echo "the report failed"
  echo "finished $(date -u +%FT%TZ)"
}

cd "${repo}"
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export AF_ENDPOINT_KIND=jetstream2_hosted

if [[ "${1:-}" == --worker ]]; then
  shift
  worker "$@"
  exit 0
fi

method="${1:-}" config="${2:-}" name="${3:-}" tasks="${4:-1}"
script="$(script_for "${method}")"
[[ -f "${config}" && "${name}" =~ ^[A-Za-z0-9._-]+$ && "${tasks}" =~ ^[1-9][0-9]*$ ]] || usage
readonly root="${AF_HOME}/runs/${name}"

[[ -x "${py}" && -f "${AF_HOME}/setup_record.txt" ]] || {
  echo "run scripts/jetstream/setup_vm.sh first" >&2
  exit 2
}
protocol="$("${py}" -c 'import json, sys
print(json.load(open(sys.argv[1]))["inputs_protocol"])' "${config}")"
case "${protocol}" in
  phase-b-nine-case-inputs-1) inputs="${AF_HOME}/release/phase-b-nine-case-v1" receipt=receipt.json ;;
  phase-c-development-2) inputs="${AF_HOME}/release/phase-c-public-v2" receipt=summary.json ;;
  *) echo "${config} reads unknown inputs: ${protocol}" >&2; exit 2 ;;
esac
[[ -f "${inputs}/${receipt}" ]] || {
  echo "the inputs of ${config} are not at ${inputs}" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
if [[ -f "${root}/pid" ]] && kill -0 "$(cat "${root}/pid")" 2>/dev/null; then
  echo "${name} is already running as process $(cat "${root}/pid")"
  exit 0
fi
curl -sf --max-time 30 -o /dev/null "${hosted}/v1/models" || {
  echo "the hosted service is not answering at ${hosted}; try again later" >&2
  exit 2
}

mkdir -p "${root}/logs"
"${py}" "${script}" prepare \
  --config "${config}" --inputs "${inputs}" --root "${root}" >"${root}/prepare.json"
git rev-parse HEAD >"${root}/source_commit"
read -r count planned < <("${py}" -c 'import json, sys
value = json.load(open(sys.argv[1]))
print(value["expected"], value.get("method", "spec_sindy"))' "${root}/prepare.json")
[[ "${count:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo "could not read the task count from ${root}/prepare.json" >&2
  exit 2
}
[[ "${planned}" == "${method}" ]] || {
  echo "${config} is a ${planned} plan, not ${method}" >&2
  exit 2
}

logged=0
[[ ! -f "${root}/runner.log" ]] || logged="$(wc -l <"${root}/runner.log")"
nohup setsid bash "${self}" --worker "${root}" "${script}" "${count}" "${tasks}" \
  >>"${root}/runner.log" 2>&1 </dev/null &
echo "$!" >"${root}/pid"
sleep 5
if ! kill -0 "$(cat "${root}/pid")" 2>/dev/null; then
  # Finished tasks return at once, so a run with nothing left ends quickly.
  if tail -n +"$((logged + 1))" "${root}/runner.log" | grep -q '^finished'; then
    echo "every task of ${name} has ended; report: ${root}/report.json"
    exit 0
  fi
  tail -n 40 "${root}/runner.log"
  echo "the run stopped at once (log: ${root}/runner.log)" >&2
  exit 1
fi
echo "running ${count} tasks of ${name}, ${tasks} at a time, as process $(cat "${root}/pid")"
echo "log ${root}/runner.log; one log per task in ${root}/logs"
