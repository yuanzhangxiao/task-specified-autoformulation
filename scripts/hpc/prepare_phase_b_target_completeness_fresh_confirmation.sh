#!/bin/bash
set -euo pipefail

readonly af_user="${USER:-}"
: "${af_user:?cannot determine the user name}"
: "${AF_PROJECT:=/projects/bibo/${af_user}}"
: "${AF_WORK:=/work/hdd/bibo/${af_user}}"
: "${AF_REPO_ROOT:=${AF_PROJECT}/repos/autoformalism-v21}"
: "${AF_PYTHON:=${AF_PROJECT}/venvs/autoformalism-v21/bin/python}"
: "${AF_DEVELOPMENT_ROOT:=${AF_WORK}/phase_b/target-completeness-absolute-v7}"
: "${AF_EXPERIMENT_ROOT:=${AF_WORK}/phase_b/target-completeness-fresh-confirmation-v1}"
: "${AF_PUBLIC_DATA_ROOT:=${AF_WORK}/phase_b/judge-hybrid-target-mapping-v4-clean-names/public}"

readonly development_analysis="${AF_DEVELOPMENT_ROOT}/target_completeness_validation.json"
readonly development_pairs="${AF_DEVELOPMENT_ROOT}/pairs.jsonl"
readonly development_run_manifest="${AF_DEVELOPMENT_ROOT}/gpt-oss-120b/shards/shard_0/target_completeness_run_manifest.json"
readonly development_config="${AF_REPO_ROOT}/configs/target_completeness_absolute_v7.json"
readonly confirmation_config="${AF_REPO_ROOT}/configs/target_completeness_fresh_confirmation_v1.json"

for af_required in \
  "${development_analysis}" \
  "${development_pairs}" \
  "${development_run_manifest}" \
  "${development_config}" \
  "${confirmation_config}"; do
  [[ -f "${af_required}" ]] || {
    echo "missing prerequisite: ${af_required}" >&2
    exit 2
  }
done

mapfile -t run_roots < <(
  find "${AF_WORK}/phase_b" -mindepth 2 -maxdepth 3 -type d -name runs \
    -print | LC_ALL=C sort
)
mapfile -t exclusion_files < <(
  find "${AF_WORK}/phase_b" -mindepth 2 -maxdepth 4 -type f -name pairs.jsonl \
    ! -path "${AF_EXPERIMENT_ROOT}/pairs.jsonl" -print | LC_ALL=C sort
)
(( ${#run_roots[@]} > 0 )) || {
  echo "no completed Phase-B run roots found" >&2
  exit 2
}
(( ${#exclusion_files[@]} > 0 )) || {
  echo "no opened pair files found" >&2
  exit 2
}

run_args=()
for af_path in "${run_roots[@]}"; do
  run_args+=(--runs-root "${af_path}")
done
exclusion_args=()
for af_path in "${exclusion_files[@]}"; do
  exclusion_args+=(--exclude-pairs "${af_path}")
done

mkdir -p "${AF_EXPERIMENT_ROOT}"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/build_target_completeness_confirmation_pairs.py" \
  "${run_args[@]}" \
  "${exclusion_args[@]}" \
  --data-root "${AF_PUBLIC_DATA_ROOT}" \
  --development-analysis "${development_analysis}" \
  --development-run-manifest "${development_run_manifest}" \
  --development-pairs "${development_pairs}" \
  --development-config "${development_config}" \
  --protocol-config "${confirmation_config}" \
  --output "${AF_EXPERIMENT_ROOT}/pairs.jsonl" \
  --manifest "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation_manifest.json"

cp "${confirmation_config}" "${AF_EXPERIMENT_ROOT}/protocol_config.json"
"${AF_PYTHON}" "${AF_REPO_ROOT}/scripts/verify_target_completeness_confirmation.py" \
  --pairs "${AF_EXPERIMENT_ROOT}/pairs.jsonl" \
  --manifest "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation_manifest.json" \
  --config "${AF_EXPERIMENT_ROOT}/protocol_config.json" \
  --development-analysis "${development_analysis}"

wc -l \
  "${AF_EXPERIMENT_ROOT}/pairs.jsonl" \
  "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation_manifest.json"
sha256sum \
  "${AF_EXPERIMENT_ROOT}/pairs.jsonl" \
  "${AF_EXPERIMENT_ROOT}/target_completeness_confirmation_manifest.json" \
  "${AF_EXPERIMENT_ROOT}/protocol_config.json"
