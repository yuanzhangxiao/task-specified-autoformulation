#!/bin/bash
# Submit or resume the frozen external-baseline (SINDy, PySR, Sol) chain.
# Native D3 stays in the roster as evaluator_unsupported until its increment lands.
#
#   AF_RESUME_FROM   prepare|postfreeze|merge|hidden|finalize (default prepare)
#   AF_POSTFREEZE_ARRAY / AF_HIDDEN_ARRAY  resubmit only these indices
#
# Resuming reuses the existing output root and per-task checkpoints; a new root
# for every retry would discard completed CPU evaluation.

set -euo pipefail
readonly af_user="${USER:?}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_OUTPUT_ROOT:=${AF_WORK}/phase_b/external-baseline-evaluation-v1}"
: "${AF_ACCOUNT:=bibo-delta-cpu}"
# Which plan this evaluation runs. D3 is scored under its own receipt, so
# it uses the single-method plan and its own output root.
: "${AF_EVAL_PLAN:=${AF_REPO_ROOT}/configs/external_baseline_frozen_test_evaluation_v2.json}"
: "${AF_RESUME_FROM:=prepare}"
: "${AF_POSTFREEZE_ARRAY:=0-23%12}"
: "${AF_HIDDEN_ARRAY:=0-23%12}"
export AF_REPO_ROOT
cd "${AF_REPO_ROOT}"

readonly commit="$(git rev-parse HEAD)"
export AF_EVAL_PLAN
readonly digest="$(bash scripts/hpc/external_baseline_inputs_digest.sh)"
echo "plan=${AF_EVAL_PLAN}"
echo "output_root=${AF_OUTPUT_ROOT}"
readonly ledger="${AF_OUTPUT_ROOT}/submission_ledger.jsonl"
readonly receipt="${AF_OUTPUT_ROOT}/submission_manifest.json"

case "${AF_RESUME_FROM}" in
  prepare|postfreeze|merge|hidden|finalize) ;;
  *) echo "unknown AF_RESUME_FROM: ${AF_RESUME_FROM}" >&2; exit 2 ;;
esac
if [[ -f "${receipt}" && "${AF_RESUME_FROM}" == "prepare" ]]; then
  echo "a full submission already exists; set AF_RESUME_FROM to resume" >&2
  cat "${receipt}"
  exit 2
fi

# A resumed stage must match the code identity sealed by the original run, not
# a digest recomputed now: the same commit can hold different local edits.
readonly sealed_record="${AF_OUTPUT_ROOT}/frozen/execution_record.json"
if [[ "${AF_RESUME_FROM}" != "prepare" && -f "${sealed_record}" ]]; then
  sealed_digest="$(jq -r '.chain_inputs_digest' "${sealed_record}")"
  if [[ "${sealed_digest}" != "${digest}" ]]; then
    echo "chain inputs differ from the sealed execution record" >&2
    echo "  sealed=${sealed_digest}" >&2
    echo "  actual=${digest}" >&2
    echo "restore the sealed checkout, or start a new output root" >&2
    exit 2
  fi
fi
mkdir -p logs "${AF_OUTPUT_ROOT}"

readonly common="ALL,AF_REPO_ROOT=${AF_REPO_ROOT},AF_OUTPUT_ROOT=${AF_OUTPUT_ROOT},AF_EVALUATOR_CODE_COMMIT=${commit},AF_CHAIN_INPUTS_DIGEST=${digest},AF_EVAL_PLAN=${AF_EVAL_PLAN}"
# Plain variables rather than an associative array: bash 3.2 lacks -A.
job_prepare=""
job_postfreeze=""
job_merge=""
job_hidden=""
job_finalize=""
previous=""

# Record each submission as it happens, so an interrupted chain is recoverable.
note() {
  printf '{"stage":"%s","job":"%s","commit":"%s","inputs_digest":"%s","submitted_at":"%s"}\n' \
    "$1" "$2" "${commit}" "${digest}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${ledger}"
}

submit() {
  local stage="$1" script="$2" array="${3:-}"
  local -a options=(--parsable --account="${AF_ACCOUNT}" --export="${common}")
  [[ -n "${array}" ]] && options+=(--array="${array}")
  [[ -n "${previous}" ]] && options+=(--dependency="afterok:${previous}" --kill-on-invalid-dep=yes)
  local job
  job="$(sbatch "${options[@]}" "scripts/hpc/${script}")"
  previous="${job}"
  note "${stage}" "${job}"
  printf '%s_JOB=%s\n' "${stage}" "${job}"
}

started=0
for stage in prepare postfreeze merge hidden finalize; do
  [[ "${stage}" == "${AF_RESUME_FROM}" ]] && started=1
  ((started)) || continue
  case "${stage}" in
    prepare)    submit prepare external_baseline_eval_prepare.slurm
                job_prepare="${previous}" ;;
    postfreeze) submit postfreeze external_baseline_eval_postfreeze.slurm "${AF_POSTFREEZE_ARRAY}"
                job_postfreeze="${previous}" ;;
    merge)      submit merge external_baseline_eval_merge.slurm
                job_merge="${previous}" ;;
    hidden)     submit hidden external_baseline_eval_hidden.slurm "${AF_HIDDEN_ARRAY}"
                job_hidden="${previous}" ;;
    finalize)   submit finalize external_baseline_eval_finalize.slurm
                job_finalize="${previous}" ;;
  esac
done

jq -n --arg commit "${commit}" --arg digest "${digest}" \
  --arg resume "${AF_RESUME_FROM}" \
  --arg prepare "${job_prepare}" --arg postfreeze "${job_postfreeze}" \
  --arg merge "${job_merge}" --arg hidden "${job_hidden}" \
  --arg final "${job_finalize}" \
  '{schema_version: "phase-b-external-baseline-submission-2",
    evaluator_code_commit: $commit, chain_inputs_digest: $digest,
    resumed_from: $resume, prepare_job: $prepare, postfreeze_job: $postfreeze,
    postfreeze_merge_job: $merge, hidden_job: $hidden, finalize_job: $final}' \
  > "${receipt}"
cat "${receipt}"
