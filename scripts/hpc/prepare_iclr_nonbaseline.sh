#!/usr/bin/env bash
# Prepare public-only ACES plans on the actual scientific runtime. No model calls.
set -euo pipefail
: "${AF_REPO_ROOT:?set the checkout directory}"
: "${AF_INPUTS:?set the extracted public package directory}"
: "${AF_CAMPAIGN:?set a fresh writable ACES output parent}"
: "${AF_PYTHON:?set the project Python executable}"
cd "$AF_REPO_ROOT"
export PYTHONPATH="$AF_REPO_ROOT/src:$AF_REPO_ROOT" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$AF_PYTHON" - "$AF_INPUTS" "$AF_CAMPAIGN" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from autoformalism.rebuttal.iclr_controlled_search import prepare
from autoformalism.rebuttal.iclr_search_controls import ARMS
inputs, destination = map(Path, sys.argv[1:])
roster = json.loads((inputs / "roster.json").read_text())
root = inputs / "public"
for cell, hashes in roster["public_files"].items():
    for name, expected in hashes.items():
        if hashlib.sha256((root / cell / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"public input differs: {cell}/{name}")
for lane, arms in (("anchors", list(ARMS)), ("stress", ["full", "fixed_gamma"])):
    cells = roster[lane]
    result = prepare(destination / lane, root, cells, arms, [0, 1, 2], 15, True, True)
    print(f"{lane}: {len(result['blocks'])} blocks, seal {result['artifact_sha256']}")
PY
