#!/usr/bin/env python3
"""CPU-only generation and frozen-fitter audit of a new development benchmark."""

import argparse
import json
from pathlib import Path

from autoformalism.fitting.public_fitting import _jsonable
from autoformalism.rebuttal.detention_benchmark import prepare, report, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/detention_benchmark_v1.json")
    )
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        plan = prepare(args.output, args.config)
        result = {
            "tasks": len(plan["tasks"]),
            "generation_gate_passed": plan["generation_gate_passed"],
            "audits": plan["audits"],
        }
    elif args.stage == "fit":
        if args.index is None:
            parser.error("fit requires --index")
        value = run_task(args.output, args.index)
        result = {
            "task": value["task"],
            "status": value["fit"]["status"],
            "training": value["fit"]["training"],
            "validation": value["fit"]["validation"],
            "diagnostic": value["diagnostic"],
        }
    else:
        result = report(args.output)
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
