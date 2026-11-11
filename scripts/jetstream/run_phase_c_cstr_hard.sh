#!/bin/bash
# Run the CSTR-hard baseline plans on this Jetstream2 VM, one step after another:
#
#   1. SINDy and PySR, which need the VM to themselves;
#   2. D3, alone;
#   3. LLM-SR in both forms and LLM-ODE under both prompts, side by side.
#
# Each step starts when the previous one has ended. A hosted step starts once
# the hosted model has answered three times in a row, two minutes apart, and
# starts again if its launcher finds the service down (status 2). Rerunning
# resumes: a run that has ended returns at once. Start it detached:
#
#   nohup bash scripts/jetstream/run_phase_c_cstr_hard.sh > ~/af/runs/cstr-hard-chain.log 2>&1 < /dev/null &

set -uo pipefail

: "${AF_HOME:=${HOME}/af}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly repo
readonly hosted_url="https://llm.jetstream-cloud.org/gpt-oss-120b/v1/models"
cd "${repo}" || exit 1

stamp() {
  echo "$(date -u +%FT%TZ) $*"
}

wait_for() {  # run name: until its detached worker has ended
  local pid="${AF_HOME}/runs/$1/pid"
  while [[ -f "${pid}" ]] && kill -0 "$(cat "${pid}")" 2>/dev/null; do
    sleep 60
  done
}

when_up() {  # command...: run it once the hosted model answers steadily
  local answered status
  while true; do
    answered=0
    while ((answered < 3)); do
      if curl -sf --max-time 30 -o /dev/null "${hosted_url}"; then
        answered=$((answered + 1))
      else
        answered=0
      fi
      ((answered < 3)) && sleep 120
    done
    "$@"
    status=$?
    ((status == 2)) || return "${status}"
    stamp "status 2; waiting for the hosted model again"
    sleep 120
  done
}

hosted() {  # method, plan stem, run name, tasks at once
  when_up bash scripts/jetstream/run_phase_c_hosted.sh \
    "$1" "configs/$2.json" "$3" "$4" || stamp "$3 did not start (status $?)"
}

stamp "step 1: SINDy and PySR"
bash scripts/jetstream/run_phase_c_classical.sh \
  configs/phase_c_cstr_hard_public_baseline_jetstream_cpu_v1.json \
  phase-c-cstr-hard-classical-v1 || stamp "the classical run did not start"
wait_for phase-c-cstr-hard-classical-v1

stamp "step 2: D3"
hosted d3 phase_c_cstr_hard_d3_hosted_120b_v2 phase-c-cstr-hard-d3-hosted-120b-v2 4
wait_for phase-c-cstr-hard-d3-hosted-120b-v2

stamp "step 3: LLM-SR and LLM-ODE"
hosted llm_sr phase_c_cstr_hard_llm_sr_hosted_120b_v1 \
  phase-c-cstr-hard-llm-sr-hosted-120b-v1 3
hosted llm_sr phase_c_cstr_hard_llm_sr_expression_hosted_120b_v1 \
  phase-c-cstr-hard-llm-sr-expression-hosted-120b-v1 3
hosted llm_ode phase_c_cstr_hard_llm_ode_hosted_120b_v1 \
  phase-c-cstr-hard-llm-ode-hosted-120b-v1 3
hosted llm_ode phase_c_cstr_hard_llm_ode_published_prompt_hosted_120b_v1 \
  phase-c-cstr-hard-llm-ode-published-prompt-hosted-120b-v1 3
stamp "every step has started"
