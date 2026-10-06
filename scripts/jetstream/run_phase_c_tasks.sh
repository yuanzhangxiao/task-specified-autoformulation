#!/bin/bash
# Run every task of one Phase C LLM-SR, LLM-ODE or D3 plan against this VM's
# model server, several at a time, detached from the login.
#
#   run_phase_c_tasks.sh METHOD CONFIG NAME TASKS
#
#   METHOD  llm_sr, llm_ode or d3
#   CONFIG  a frozen plan declaring the vm_local_vllm endpoint
#   NAME    the run's directory under ~/af/runs
#   TASKS   how many tasks run at once; start the server for at least as many
#           (scripts/jetstream/gpu_server.sh start MODEL TASKS)
#
# Rerunning is safe: a finished task is not repeated, a running one is refused
# by its lock, and an interrupted one restarts with its partial attempt kept
# beside it; a D3 task resumes from its last finished generation. Each start
# copies the server's record into the run, so the run names the model revision,
# image and settings that answered it. When every task has ended, the method's
# report is written to NAME/report.json.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly self="${repo}/scripts/jetstream/run_phase_c_tasks.sh"
readonly py="${AF_HOME}/venv/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly server_record="${AF_HOME}/server.json"

usage() {
  sed -n '2,11p' "${self}" >&2
  exit 2
}

worker() {  # root, method, count, tasks: runs in the foreground
  local root="$1" method="$2" count="$3" tasks="$4"
  echo "started $(date -u +%FT%TZ): ${count} tasks, ${tasks} at a time"
  # run() seals a failed search as a result, so a nonzero exit is a crash, or for
  # D3 (status 3) a server that stopped answering for longer than its patience.
  seq 0 $((count - 1)) | xargs -P "${tasks}" -I{} bash -c \
    '"$1" "scripts/phase_c_$2.py" run --root "$3" --index "$4" >>"$3/logs/task-$4.log" 2>&1 || echo "task $4 exited with status $?"' \
    _ "${py}" "${method}" "${root}" {}
  "${py}" "scripts/phase_c_${method}.py" report --root "${root}" >"${root}/report.json" ||
    echo "the report failed"
  echo "finished $(date -u +%FT%TZ)"
}

cd "${repo}"
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export AF_ENDPOINT_KIND=vm_local_vllm AF_VLLM_BASE_URL=http://127.0.0.1:8000
export AF_LLM_SR_ROOT="${AF_HOME}/vendor/LLM-SR" AF_LLM_ODE_ROOT="${AF_HOME}/vendor/llm-ode"

if [[ "${1:-}" == --worker ]]; then
  shift
  worker "$@"
  exit 0
fi

method="${1:-}" config="${2:-}" name="${3:-}" tasks="${4:-}"
[[ "${method}" == llm_sr || "${method}" == llm_ode || "${method}" == d3 ]] || usage
[[ -f "${config}" && "${name}" =~ ^[A-Za-z0-9._-]+$ && "${tasks}" =~ ^[1-9][0-9]*$ ]] || usage
readonly root="${AF_HOME}/runs/${name}"

[[ -x "${py}" && -f "${AF_HOME}/setup_record.txt" ]] || {
  echo "run scripts/jetstream/gpu_server.sh setup first" >&2
  exit 2
}
git diff --quiet && git diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
[[ -f "${server_record}" ]] || {
  echo "no model server is running; start one with scripts/jetstream/gpu_server.sh start" >&2
  exit 2
}
if [[ -f "${root}/pid" ]] && kill -0 "$(cat "${root}/pid")" 2>/dev/null; then
  echo "${name} is already running as process $(cat "${root}/pid")"
  exit 0
fi

mkdir -p "${root}/logs" "${root}/servers"
"${py}" "scripts/phase_c_${method}.py" prepare \
  --config "${config}" --release "${release}" --root "${root}" >"${root}/prepare.json"
git rev-parse HEAD >"${root}/source_commit"

# The plan names its model; the server must be serving exactly that one.
read -r count model < <("${py}" -c '
import json, sys
value = json.load(open(sys.argv[1]))
print(value["expected"], value["model"])' "${root}/prepare.json")
[[ "${count:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo "could not read the task count from ${root}/prepare.json" >&2
  exit 2
}
served="$(curl -sf --max-time 10 "${AF_VLLM_BASE_URL}/v1/models" |
  "${py}" -c 'import json, sys; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))')" || {
  echo "the model server at ${AF_VLLM_BASE_URL} is not answering" >&2
  exit 2
}
[[ "${served}" == "${model}" ]] || {
  echo "the plan names ${model}, but the server is serving ${served}" >&2
  exit 2
}
cp "${server_record}" "${root}/servers/$(date -u +%Y%m%dT%H%M%SZ).json"
server_tasks="$("${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["tasks"])' "${server_record}")"
((tasks <= server_tasks)) ||
  echo "note: the server was started for ${server_tasks} tasks; ${tasks} will queue"

logged=0
[[ ! -f "${root}/runner.log" ]] || logged="$(wc -l <"${root}/runner.log")"
nohup setsid bash "${self}" --worker "${root}" "${method}" "${count}" "${tasks}" \
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
