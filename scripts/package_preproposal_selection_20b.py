"""Package the committed screened-selection worker and verified train-only cases."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.package_t2_breadth import package_committed
from scripts.preproposal_selection_20b import ENTRYPOINTS, input_payload


def package(destination: Path, case: Path, harmony: Path) -> Path:
    payload = input_payload(case)
    with TemporaryDirectory() as folder:
        source = Path(folder) / "preproposal.json"
        source.write_text(json.dumps(payload, ensure_ascii=False))
        return package_committed(
            destination,
            "phase-c-preproposal-selection-20b-v1",
            ["src", "pyproject.toml", *ENTRYPOINTS],
            "inputs/preproposal.json",
            source,
            harmony,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "out", "harmony"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(package(args.out, args.case, args.harmony))
