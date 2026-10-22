#!/bin/bash
# Run the public mechanism tests on every Phase C external baseline's sealed
# picks, detached from the login.
#
#   run_phase_c_baseline_mechanisms.sh [WORKERS]
#
#   WORKERS  rows assessed at once (default 12)
#
# Reads the frozen release, the SINDy/PySR, D3, LLM-ODE and LLM-SR runs under
# ~/af/runs, and the LLM-SR transcriptions in this checkout. Development data
# only; no model is called and nothing is fitted. Writes to
# ~/af/runs/phase-c-baseline-mechanisms-v1. Rerunning is safe: the plan is
# sealed, and an assessed row is reused. See docs/PHASE_C_BASELINE_MECHANISMS.md.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly self="${repo}/scripts/jetstream/run_phase_c_baseline_mechanisms.sh"
readonly py="${AF_HOME}/venv/bin/python"
readonly release="${AF_HOME}/release/phase-c-public-v2"
readonly runs="${AF_HOME}/runs"
readonly root="${runs}/phase-c-baseline-mechanisms-v1"

worker() {  # workers: runs in the foreground
  echo "started $(date -u +%FT%TZ): $1 rows at a time"
  "${py}" scripts/assess_phase_c_baseline_mechanisms.py run \
    --root "${root}" --workers "$1" || echo "some rows did not finish (status $?)"
  "${py}" scripts/assess_phase_c_baseline_mechanisms.py report --root "${root}" \
    >/dev/null || echo "the report failed"
  echo "finished $(date -u +%FT%TZ)"
}

cd "${repo}"
export PYTHONPATH="${repo}/src" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "${1:-}" == --worker ]]; then
  worker "$2"
  exit 0
fi

workers="${1:-12}"
[[ "${workers}" =~ ^[1-9][0-9]*$ ]] || { sed -n '2,7p' "${self}" >&2; exit 2; }
[[ -x "${py}" ]] || { echo "no interpreter at ${py}" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 2
}
# The rollouts have wall-clock limits, so nothing else may share the CPUs.
for pid in "${runs}"/*/pid; do
  [[ -f "${pid}" ]] || continue
  if kill -0 "$(cat "${pid}")" 2>/dev/null; then
    echo "another run is alive: ${pid}" >&2
    exit 2
  fi
done
for run in phase-c-classical-v1 phase-c-d3-hosted-120b-v2 \
  phase-c-llm-ode-hosted-120b-v1 phase-c-llm-ode-published-prompt-hosted-120b-v1 \
  phase-c-llm-sr-hosted-120b-v1; do
  [[ -d "${runs}/${run}" ]] || { echo "missing run ${runs}/${run}" >&2; exit 2; }
done

mkdir -p "${root}"
if [[ -f "${root}/source_commit" ]] &&
  [[ "$(cat "${root}/source_commit")" != "$(git rev-parse HEAD)" ]]; then
  echo "${root} was prepared from another commit" >&2
  exit 2
fi
"${py}" scripts/assess_phase_c_baseline_mechanisms.py prepare \
  --release "${release}" \
  --construction-config configs/phase_c_construction_v1.json \
  --root "${root}" \
  --classical "${runs}/phase-c-classical-v1" \
  --d3 "${runs}/phase-c-d3-hosted-120b-v2" \
  --llm-ode "${runs}/phase-c-llm-ode-hosted-120b-v1" \
  --llm-ode-published-prompt "${runs}/phase-c-llm-ode-published-prompt-hosted-120b-v1" \
  --llm-sr "${runs}/phase-c-llm-sr-hosted-120b-v1" \
  --llm-sr-transcriptions configs/phase_c_llm_sr_transcriptions_v1.json
git rev-parse HEAD >"${root}/source_commit"

nohup setsid bash "${self}" --worker "${workers}" >>"${root}/runner.log" 2>&1 </dev/null &
echo "$!" >"${root}/pid"
echo "assessing as process $(cat "${root}/pid"); log ${root}/runner.log"
