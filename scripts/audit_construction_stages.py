"""Index saved Phase C variables and shared-use conflicts without remote calls."""

import argparse
import json
from pathlib import Path

from autoformalism.research.construction_stage_audit import audit, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    write_report(result, args.output)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
