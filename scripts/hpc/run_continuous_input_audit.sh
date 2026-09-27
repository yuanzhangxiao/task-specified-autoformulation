#!/bin/bash
# CPU-only input-contract verification. No discovery fitting or LLM requests.
set -euo pipefail
: "${AF_REPO_ROOT:?}" "${AF_OUTPUT_ROOT:?}" "${AF_PYTHON:?}" "${AF_COMMIT:?}"
if [[ "${AF_AUDIT_SITE:?}" == aces ]]; then
  module load GCCcore/13.2.0 Python/3.11.5
fi
cd "$AF_REPO_ROOT"
if [[ -f SOURCE_COMMIT ]]; then
  [[ "$(cat SOURCE_COMMIT)" == "$AF_COMMIT" ]]
else
  [[ "$(git rev-parse HEAD)" == "$AF_COMMIT" ]]
fi
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$AF_PYTHON" -m pytest -q -p no:cacheprovider \
  tests/test_continuous_inputs.py tests/test_continuous_input_submission.py
"$AF_PYTHON" scripts/audit_reference_integrity.py \
  --output-root "$AF_OUTPUT_ROOT/audit" \
  --input-contract continuous-rates-1 --include-test-protocols
"$AF_PYTHON" scripts/audit_reference_integrity.py \
  --output-root "$AF_OUTPUT_ROOT/audit" \
  --input-contract continuous-rates-1 --include-test-protocols
echo "Audit and deterministic resume completed: $AF_OUTPUT_ROOT/audit/summary.json"
