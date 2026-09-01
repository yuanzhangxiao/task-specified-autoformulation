#!/usr/bin/env python3
"""Initialize the optional PySR/Julia runtime and record its identity."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path


def main() -> None:
    """Import PySR once so array tasks share an initialized Julia depot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import pysr
    from juliacall import Main as jl

    payload = {
        "schema_version": "phase-b-pysr-runtime-1",
        "status": "ready",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "pysr_version": getattr(pysr, "__version__", "unknown"),
        "julia_version": str(jl.seval("VERSION")),
        "julia_depot_path": os.environ.get("JULIA_DEPOT_PATH"),
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
