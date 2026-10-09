#!/bin/bash
# Run every task of one Phase C plan against the Jetstream2 hosted model,
# detached from the login, a few at a time.
#
#   run_phase_c_hosted.sh METHOD CONFIG NAME [TASKS]
#
#   METHOD  d3, llm_ode or llm_sr
#   CONFIG  a frozen plan declaring the jetstream2_hosted endpoint
#   NAME    the run's directory under ~/af/runs
#   TASKS   how many tasks, or LLM-SR target searches, run at once (default 1)
#
# Rerunning is safe: a finished task is not repeated and a running one is
# refused by its lock. A D3 task that an outage outlasting its patience stopped
# (exit status 3) resumes from its last finished generation; an interrupted
# LLM-ODE task restarts with its partial attempt kept beside it. LLM-SR first
# searches every target of every task as its own process, then assembles and
# seals each task's model from its finished searches; a finished search is
# kept, and one that was stopped starts over beside the stopped attempt. When
# every task has ended, the method's report is written to NAME/report.json.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly self="${repo}/scripts/jetstream/run_phase_c_hosted.sh"
readonly py="${AF_HOME}/venv/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly hosted="https://llm.jetstream-cloud.org/gpt-oss-120b"

usage() {
  sed -n '2,10p' "${self}" >&2
  exit 2
}

searches() {  # plan: one "index target" line per LLM-SR target search
  "${py}" -c 'import json, sys
for row in json.load(open(sys.argv[1]))["rows"]:
    for target in row["searched_targets"]:
        print(row["index"], target)' "$1"
}

worker() {  # root, method, count, tasks: runs in the foreground
  local root="$1" method="$2" count="$3" tasks="$4"
  echo "started $(date -u +%FT%TZ): ${count} tasks, ${tasks} at a time"
  if [[ "${method}" == llm_sr ]]; then
    echo "searching $(searches "${root}/plan.json" | wc -l | tr -d " ") targets first"
    searches "${root}/plan.json" | xargs -P "${tasks}" -L 1 bash -c \
      '"$1" scripts/phase_c_llm_sr.py run --root "$2" --index "$3" --target "$4" >>"$2/logs/task-$3-$4.log" 2>&1 || echo "task $3 target $4 exited with status $?"' \
      _ "${py}" "${root}"
    echo "target searches ended $(date -u +%FT%TZ); sealing each task"
  fi
  seq 0 $((count - 1)) | xargs -P "${tasks}" -I{} bash -c \
    '"$1" "scripts/phase_c_$2.py" run --root "$3" --index "$4" >>"$3/logs/task-$4.log" 2>&1 || echo "task $4 exited with status $?"' \
    _ "${py}" "${method}" "${root}" {}
  "${py}" "scripts/phase_c_${method}.py" report --root "${root}" >"${root}/report.json" ||
    echo "the report failed"
  echo "finished $(date -u +%FT%TZ)"
}

cd "${repo}"
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0 OMP_NUM_THREADS=1
export AF_ENDPOINT_KIND=jetstream2_hosted
export AF_LLM_ODE_ROOT="${AF_HOME}/vendor/llm-ode"
export AF_LLM_SR_ROOT="${AF_HOME}/vendor/LLM-SR"

if [[ "${1:-}" == --worker ]]; then
  shift
  worker "$@"
  exit 0
fi

method="${1:-}" config="${2:-}" name="${3:-}" tasks="${4:-1}"
[[ "${method}" == d3 || "${method}" == llm_ode || "${method}" == llm_sr ]] || usage
[[ -f "${config}" && "${name}" =~ ^[A-Za-z0-9._-]+$ && "${tasks}" =~ ^[1-9][0-9]*$ ]] || usage
readonly root="${AF_HOME}/runs/${name}"

[[ -x "${py}" && -f "${AF_HOME}/setup_record.txt" ]] || {
  echo "run scripts/jetstream/setup_vm.sh first" >&2
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
"${py}" "scripts/phase_c_${method}.py" prepare \
  --config "${config}" --release "${release}" --root "${root}" >"${root}/prepare.json"
git rev-parse HEAD >"${root}/source_commit"
count="$("${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["expected"])' \
  "${root}/prepare.json")"
[[ "${count:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo "could not read the task count from ${root}/prepare.json" >&2
  exit 2
}

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
