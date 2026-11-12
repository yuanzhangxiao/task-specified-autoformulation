#!/bin/bash
# Run the nine-case smokes, then the full campaigns once every smoke passed.
#
#   run_phase_b_nine_case_campaigns.sh
#
# Each run goes through run_phase_b_hosted.sh, started by ~/af/start_when_up.sh
# so that a hosted outage at launch is waited out. The smokes run first, all
# at once. When each has finished, its report must be complete with no task
# that failed to discover, failed its replay check, or failed to fit (a SINDy
# fit that reached its wall-clock limit, or whose rollouts failed at every
# threshold, is the method's outcome, not a failure). Only
# then do the three full campaigns start, four tasks at a time each. If a
# smoke fails the check, nothing more starts and the reason is printed.
# The check reads each run's summary.json, which has one row per task.
# The v1 smokes ran at bba1565 and stopped at a check that read the
# row-less report.json; the smokes run again as v2.

set -euo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly launch="${repo}/scripts/jetstream/run_phase_b_hosted.sh"
readonly runs="${AF_HOME}/runs"
readonly py="${AF_HOME}/venv/bin/python"

readonly smokes=(
  "d3 configs/phase_b_nine_case_d3_smoke_v1.json phase-b-nine-case-d3-smoke-v2 1"
  "hdtwingen configs/phase_b_nine_case_hdtwingen_smoke_v1.json phase-b-nine-case-hdtwingen-smoke-v2 1"
  "spec_sindy configs/phase_b_nine_case_spec_sindy_smoke_v1.json phase-b-nine-case-spec-sindy-smoke-v2 2"
)
readonly campaigns=(
  "spec_sindy configs/phase_b_nine_case_spec_sindy_hosted_120b_v1.json phase-b-nine-case-spec-sindy-hosted-120b-v1 4"
  "d3 configs/phase_b_nine_case_d3_hosted_120b_v1.json phase-b-nine-case-d3-hosted-120b-v1 4"
  "hdtwingen configs/phase_b_nine_case_hdtwingen_hosted_120b_v1.json phase-b-nine-case-hdtwingen-hosted-120b-v1 4"
)

start() {  # method config name tasks
  bash "${AF_HOME}/start_when_up.sh" bash "${launch}" "$@"
}

finished() {  # name: whether the run's last log line says it finished
  tail -n 1 "${runs}/$1/runner.log" 2>/dev/null | grep -q '^finished'
}

cd "${repo}"
# start_when_up.sh retries a status of 2, which the launcher also gives for a
# setup problem; check those first so that one fails here instead.
[[ -f "${AF_HOME}/release/phase-b-nine-case-v1/receipt.json" ]] || {
  echo "the staged nine-case inputs are not at ${AF_HOME}/release/phase-b-nine-case-v1" >&2
  exit 1
}
[[ -x "${py}" && -f "${AF_HOME}/setup_record.txt" && -f "${AF_HOME}/start_when_up.sh" ]] || {
  echo "the VM setup or ${AF_HOME}/start_when_up.sh is missing" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "the checkout at ${repo} has local changes" >&2
  exit 1
}
PYTHONPATH="${repo}/src" "${py}" scripts/stage_phase_b_nine_case.py verify \
  --root "${AF_HOME}/release/phase-b-nine-case-v1"
for entry in "${smokes[@]}" "${campaigns[@]}"; do
  read -r _ _ name _ <<<"${entry}"
  [[ ! -e "${runs}/${name}" ]] || {
    echo "${runs}/${name} exists already; this script starts fresh runs only" >&2
    exit 1
  }
done
echo "smokes starting $(date -u +%FT%TZ) from $(git rev-parse --short HEAD)"
for entry in "${smokes[@]}"; do
  # shellcheck disable=SC2086
  start ${entry}
done
reports=()
for entry in "${smokes[@]}"; do
  read -r _ _ name _ <<<"${entry}"
  until finished "${name}"; do sleep 60; done
  reports+=("${runs}/${name}/summary.json")
done
echo "smokes finished $(date -u +%FT%TZ)"
PYTHONPATH="${repo}/src" "${py}" scripts/check_phase_b_smokes.py "${reports[@]}"
for entry in "${campaigns[@]}"; do
  # shellcheck disable=SC2086
  start ${entry}
done
echo "campaigns started $(date -u +%FT%TZ)"
