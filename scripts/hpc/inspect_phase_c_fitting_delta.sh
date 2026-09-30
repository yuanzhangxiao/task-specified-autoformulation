#!/bin/bash
# Refresh a summary and package compact evidence; no fitting or submission.
set -euo pipefail
AF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AF_PYTHON="${AF_PYTHON:-/projects/bibo/yxiao2/venvs/autoformalism-v21/bin/python}"
AF_OUTPUT_ROOT="${AF_OUTPUT_ROOT:-/work/hdd/bibo/yxiao2/phase_c/fitting-m1}"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT${AF_CASADI_ROOT:+:$AF_CASADI_ROOT}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ -d /projects/bibo/yxiao2/venvs/fitter-methods-v1-deps && -z "${AF_CASADI_ROOT:-}" ]]; then
  export PYTHONPATH="$PYTHONPATH:/projects/bibo/yxiao2/venvs/fitter-methods-v1-deps"
fi
"$AF_PYTHON" "$AF_REPO_ROOT/scripts/phase_c_fitting.py" report --root "$AF_OUTPUT_ROOT"
AF_DEST=$(mktemp -d /work/hdd/bibo/yxiao2/phase_c/fitting-review.XXXXXX)
"$AF_PYTHON" - "$AF_OUTPUT_ROOT" "$AF_DEST/review.tar.gz" <<'PY'
import sys, tarfile
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
names = {'plan.json', 'inputs.json', 'summary.json', 'submission_manifest.json'}
patterns = ('qualification/*.json', 'results/*/result.json', 'results/*/backend.json',
            'results/*/replay.json', 'results/*/fit/coordinates.json',
            'results/*/fit/collocation/progress.json', 'logs/*.err')
paths = [root/n for n in names if (root/n).is_file()]
paths += [p for pattern in patterns for p in root.glob(pattern)]
with tarfile.open(output, 'w:gz') as archive:
    for path in sorted(set(paths)):
        if path.is_file() and not path.is_symlink():
            archive.add(path, arcname=str(path.relative_to(root)))
print(f'Download: {output}')
PY
