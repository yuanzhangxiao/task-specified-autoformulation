#!/bin/bash
# Run every task of one Phase C classical plan (SINDy, PySR) on this Jetstream2
# CPU VM, detached from the login.
#
#   run_phase_c_classical.sh CONFIG NAME
#
#   CONFIG  a plan declaring the jetstream2_cpu platform
#   NAME    the run's directory under ~/af/runs
#
# Each task gets the CPUs its plan declares, pinned as a Slurm allocation would
# pin them, and as many tasks run at once as the VM has room for. Nothing else
# of ours may be running here: the tasks' wall clocks assume those CPUs are
# theirs. Rerunning is safe: a task that recorded an outcome (complete, failed
# or timed_out) is never run again, so none gets a second try at its budget,
# and one that was interrupted starts again. When every task has ended, the
# summary is written to NAME/summary.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly self="${repo}/scripts/jetstream/run_phase_c_classical.sh"
readonly py="${AF_HOME}/venv-classical/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly depot="${AF_HOME}/julia-depot-pysr-1.5.9"

usage() {
  sed -n '2,9p' "${self}" >&2
  exit 2
}

lane() {  # root, first task, stride, task count, CPU list: runs in the foreground
  local root="$1" first="$2" stride="$3" count="$4" cpus="$5" index
  for ((index = first; index < count; index += stride)); do
    if [[ "$(git rev-parse HEAD)" != "$(cat "${root}/source_commit")" ]] ||
      ! git diff --quiet || ! git diff --cached --quiet; then
      echo "the checkout changed under the run; task ${index} and later were not started"
      return
    fi
    taskset -c "${cpus}" "${py}" scripts/phase_c_classical.py run --root "${root}" \
      --index "${index}" --release "${release}" --julia-depot "${depot}" \
      >>"${root}/logs/task-${index}.log" 2>&1 ||
      echo "task ${index} exited with status $?"
  done
}

worker() {  # root, task count, lanes, CPUs per task: runs in the foreground
  local root="$1" count="$2" lanes="$3" width="$4" slot
  echo "started $(date -u +%FT%TZ): ${count} tasks, ${lanes} at a time, ${width} CPUs each"
  for ((slot = 0; slot < lanes; slot++)); do
    lane "${root}" "${slot}" "${lanes}" "${count}" \
      "$((slot * width))-$((slot * width + width - 1))" &
  done
  wait
  "${py}" scripts/summarize_phase_b_public_baseline_pilot.py \
    --task-plan "${root}/frozen/task_plan.jsonl" --runs-root "${root}/runs" \
    --output-root "${root}/summary" >"${root}/summary.log" 2>&1 ||
    echo "the summary failed (log: ${root}/summary.log)"
  echo "finished $(date -u +%FT%TZ)"
}

cd "${repo}"
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0

if [[ "${1:-}" == --worker ]]; then
  shift
  worker "$@"
  exit 0
fi

config="${1:-}" name="${2:-}"
[[ -f "${config}" && "${name}" =~ ^[A-Za-z0-9._-]+$ ]] || usage
readonly root="${AF_HOME}/runs/${name}"

[[ -x "${py}" && -f "${AF_HOME}/setup_classical_record.txt" ]] || {
  echo "run scripts/jetstream/setup_classical.sh first" >&2
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
for other in "${AF_HOME}"/runs/*/pid; do
  [[ -f "${other}" && "${other}" != "${root}/pid" ]] || continue
  if kill -0 "$(cat "${other}")" 2>/dev/null; then
    echo "$(basename "$(dirname "${other}")") is running on this VM as process" \
      "$(cat "${other}"); these tasks need their CPUs to themselves" >&2
    exit 2
  fi
done

mkdir -p "${root}/logs"
"${py}" scripts/phase_c_classical.py prepare \
  --config "${config}" --release "${release}" --root "${root}" >"${root}/prepare.json"
commit="$(git rev-parse HEAD)"
if [[ -f "${root}/source_commit" ]]; then
  [[ "$(cat "${root}/source_commit")" == "${commit}" ]] || {
    echo "${name} was started from $(cat "${root}/source_commit"); check that" \
      "commit out to resume it" >&2
    exit 2
  }
else
  echo "${commit}" >"${root}/source_commit"
fi
read -r count width pysr < <("${py}" -c '
import json, sys
summary = json.load(open(sys.argv[1]))
print(summary["expected"], summary["cpus_per_task"], summary["tasks_by_method"].get("pysr", 0))
' "${root}/prepare.json")
[[ "${count}" =~ ^[1-9][0-9]*$ && "${width}" =~ ^[1-9][0-9]*$ ]] || {
  echo "could not read the task count from ${root}/prepare.json" >&2
  exit 2
}
lanes=$(($(nproc) / width))
((lanes >= 1)) || {
  echo "this VM has $(nproc) CPUs and each task declares ${width}" >&2
  exit 2
}

# PySR's runtime record, as Delta's prepare job writes it; each PySR task
# checks it against this digest before starting.
if ((pysr > 0)) && [[ ! -f "${root}/runtime/pysr_runtime.json.sha256" ]]; then
  mkdir -p "${root}/runtime"
  JULIA_DEPOT_PATH="${depot}" "${py}" scripts/check_pysr_runtime.py \
    --output "${root}/runtime/pysr_runtime.json" >/dev/null
  (cd "${root}/runtime" && sha256sum pysr_runtime.json >pysr_runtime.json.sha256)
fi
{
  echo "launch $(date -u +%FT%TZ)"
  echo "flavor $(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/instance-type || echo unknown)"
  echo "cpus $(nproc)"
  echo "model $(lscpu | sed -n 's/^Model name: *//p' | head -n 1)"
  echo "lanes ${lanes} of ${width} CPUs"
} >>"${root}/machine.txt"

logged=0
[[ ! -f "${root}/runner.log" ]] || logged="$(wc -l <"${root}/runner.log")"
nohup setsid bash "${self}" --worker "${root}" "${count}" "${lanes}" "${width}" \
  >>"${root}/runner.log" 2>&1 </dev/null &
echo "$!" >"${root}/pid"
sleep 5
if ! kill -0 "$(cat "${root}/pid")" 2>/dev/null; then
  # Tasks with an outcome return at once, so a run with nothing left ends quickly.
  if tail -n +"$((logged + 1))" "${root}/runner.log" | grep -q '^finished'; then
    echo "every task of ${name} has ended; summary: ${root}/summary"
    exit 0
  fi
  tail -n 40 "${root}/runner.log"
  echo "the run stopped at once (log: ${root}/runner.log)" >&2
  exit 1
fi
echo "running ${count} tasks of ${name}, ${lanes} at a time, as process $(cat "${root}/pid")"
echo "log ${root}/runner.log; one log per task in ${root}/logs"
