#!/bin/bash
# Same public contexts, fresh inventories, three schedules; no fitting jobs.
set -euo pipefail
AF_TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
AF_GROUP=${AF_GROUP:-/scratch/group/p.nairr260351.000/u.yx126462}
[[ -d "$AF_GROUP" ]] || { echo 'Run on ACES.' >&2; exit 1; }
module load GCCcore/13.2.0 Python/3.11.5
export AF_PYTHON=${AF_PYTHON:-/scratch/user/u.yx126462/repos/autoformalism-e432fe3/.venv/bin/python}
export AF_REPO_ROOT="$AF_TOOLS"
AF_STUDY=${AF_COMPARISON_STUDY:-comparison}
case "$AF_STUDY" in
  comparison) AF_DEFAULT_ROOT=phase-c-construction-comparison-v9 ;;
  live_confirmation) AF_DEFAULT_ROOT=phase-c-construction-live-v8 ;;
  basin_confirmation) AF_DEFAULT_ROOT=phase-c-construction-basins-v3 ;;
  shared_law_comparison) AF_DEFAULT_ROOT=phase-c-shared-laws-v2 ;;
  prompt_comparison) AF_DEFAULT_ROOT=phase-c-minimal-prompts-v1 ;;
  refinement_confirmation) AF_DEFAULT_ROOT=phase-c-prompt-refinement-v1 ;;
  variable_checklist_confirmation) AF_DEFAULT_ROOT=phase-c-variable-checklist-v1 ;;
  stage_check_confirmation) AF_DEFAULT_ROOT=phase-c-stage-checks-v1 ;;
  deferred_interaction_confirmation) AF_DEFAULT_ROOT=phase-c-minimal-deferred-v2 ;;
  focused_prompt_confirmation) AF_DEFAULT_ROOT=phase-c-focused-prompts-seeds5-v1 ;;
  readable_prompt_confirmation) AF_DEFAULT_ROOT=phase-c-readable-prompts-seeds5-v1 ;;
  public_coverage_confirmation) AF_DEFAULT_ROOT=phase-c-public-coverage-seeds5-v1 ;;
  coverage_decision_confirmation) AF_DEFAULT_ROOT=phase-c-coverage-decisions-seeds5-v1 ;;
  completion_feedback_confirmation) AF_DEFAULT_ROOT=phase-c-completion-feedback-seeds5-v2 ;;
  *) echo 'Unknown AF_COMPARISON_STUDY.' >&2; exit 2 ;;
esac
export AF_OUTPUT_ROOT=${AF_COMPARISON_ROOT:-$AF_GROUP/$AF_DEFAULT_ROOT}
export PYTHONPATH="$AF_REPO_ROOT/src" PYTHONDONTWRITEBYTECODE=1
case "${1:?Use run or inspect}" in
  run)
    export AF_VLLM_IMAGE=${AF_VLLM_IMAGE:-/scratch/user/u.yx126462/containers/vllm-openai-v0.27.1.sif}
    export AF_HF_HOME=${AF_HF_HOME:-/scratch/user/u.yx126462/huggingface-cache}
    export AF_COMPUTE_CACHE_ROOT="$AF_GROUP/phase-c-runtime-cache"
    export AF_IPC_TMP_ROOT=/tmp/phase-c-ipc-u.yx126462
    if [[ ! -f "$AF_OUTPUT_ROOT/plan.json" ]]; then
      AF_PREPARE_EXTRA=()
      if [[ "$AF_STUDY" == focused_prompt_confirmation || "$AF_STUDY" == readable_prompt_confirmation || "$AF_STUDY" == completion_feedback_confirmation || "$AF_STUDY" == public_coverage_confirmation || "$AF_STUDY" == coverage_decision_confirmation ]]; then
        read -r -a AF_SEEDS <<< "${AF_COMPARISON_SEEDS:-0 1 2 3 4}"
        AF_PREPARE_EXTRA+=(--seeds "${AF_SEEDS[@]}")
      fi
      if [[ "$AF_STUDY" == refinement_confirmation ]]; then
        AF_PREPARE_EXTRA+=(--repair-source "${AF_REPAIR_SOURCE:-$AF_GROUP/phase-c-current-bookkeeping-v2}")
      fi
      "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_construction_comparison.py" prepare \
        --source "${AF_COMPARISON_SOURCE:-$AF_GROUP/phase-c-topology-v1}" \
        --root "$AF_OUTPUT_ROOT" --study "$AF_STUDY" "${AF_PREPARE_EXTRA[@]}"
    else
      "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_construction_comparison.py" verify --root "$AF_OUTPUT_ROOT"
    fi
    [[ "$(jq -r '.study // "comparison"' "$AF_OUTPUT_ROOT/plan.json")" == "$AF_STUDY" ]] || {
      echo 'Study differs from saved plan; use its original study/root.' >&2; exit 2;
    }
    if [[ -n "${AF_COMPARISON_SEEDS:-}" ]]; then
      AF_SAVED_SEEDS=$(jq -r '(.sampling_seeds // [0]) | map(tostring) | join(" ")' "$AF_OUTPUT_ROOT/plan.json")
      [[ "$AF_COMPARISON_SEEDS" == "$AF_SAVED_SEEDS" ]] || {
        echo 'Seeds differ from saved plan; use a fresh root for a different roster.' >&2; exit 2;
      }
    fi
    exec "$AF_PYTHON" "$AF_REPO_ROOT/scripts/submit_phase_c_baseline.py" \
      --root "$AF_OUTPUT_ROOT" --construction-comparison --wave "${AF_COMPARISON_WAVE:-comparison-1}"
    ;;
  inspect)
    "$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_construction_comparison.py" report --root "$AF_OUTPUT_ROOT"
    cat "$AF_OUTPUT_ROOT/SUMMARY.md"
    AF_CONTENTS=(plan.json summary.json SUMMARY.md TOPOLOGY.html)
    if [[ "$AF_STUDY" == variable_checklist_confirmation || "$AF_STUDY" == stage_check_confirmation || "$AF_STUDY" == deferred_interaction_confirmation || "$AF_STUDY" == focused_prompt_confirmation || "$AF_STUDY" == readable_prompt_confirmation || "$AF_STUDY" == completion_feedback_confirmation || "$AF_STUDY" == public_coverage_confirmation || "$AF_STUDY" == coverage_decision_confirmation ]]; then
      cat "$AF_OUTPUT_ROOT/VARIABLES.md"
      AF_CONTENTS+=(VARIABLES.json VARIABLES.md)
    fi
    if [[ "$AF_STUDY" == stage_check_confirmation || "$AF_STUDY" == deferred_interaction_confirmation || "$AF_STUDY" == focused_prompt_confirmation || "$AF_STUDY" == readable_prompt_confirmation || "$AF_STUDY" == completion_feedback_confirmation || "$AF_STUDY" == public_coverage_confirmation || "$AF_STUDY" == coverage_decision_confirmation ]]; then
      cat "$AF_OUTPUT_ROOT/SHARED_PROCESSES.md"
      AF_CONTENTS+=(SHARED_PROCESSES.json SHARED_PROCESSES.md)
    fi
    for AF_PART in results logs submissions runtime; do
      [[ ! -d "$AF_OUTPUT_ROOT/$AF_PART" ]] || AF_CONTENTS+=("$AF_PART")
    done
    tar -czf "$AF_OUTPUT_ROOT/inspection.tar.gz" -C "$AF_OUTPUT_ROOT" "${AF_CONTENTS[@]}"
    printf '\nDownload: %s\n' "$AF_OUTPUT_ROOT/inspection.tar.gz"
    ;;
  *) echo 'Use run or inspect.' >&2; exit 2 ;;
esac
