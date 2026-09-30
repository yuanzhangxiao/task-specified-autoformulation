#!/usr/bin/env python3
"""Export, freeze, qualify, run and inspect the CPU-only Phase C fitting study."""

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.fitting_qualification_inputs import export_inputs
from autoformalism.fitting import qualification as q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("export", "prepare", "qualify", "run", "report")
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/phase_c_fitting_v1.json",
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "export":
        if args.release is None or args.inputs is None:
            parser.error("export requires --release and --inputs")
        result = export_inputs(args.release, args.data_root, args.inputs)
    else:
        if args.root is None:
            parser.error("require --root")
        if args.command == "prepare":
            if args.inputs is None:
                parser.error("prepare requires --inputs")
            result = q.prepare(
                args.inputs,
                args.root,
                q.QualificationConfig.model_validate_json(args.config.read_text()),
            )
        elif args.command == "qualify":
            result = q.qualify(args.root)
            if not result["passed"]:
                raise SystemExit(
                    "Reference replay failed; inspect qualification/result.json"
                )
        elif args.command == "run":
            if args.index is None:
                parser.error("run requires --index")
            try:
                result = q.run_task(args.root, args.index)
            finally:
                q.report(args.root)
        else:
            result = q.report(args.root)
            result = {
                k: result[k]
                for k in ("status", "expected", "recorded", "status_counts", "groups")
            }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
