#!/usr/bin/env python3
"""Run complete private pre-release gates for one Phase-B cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoformalism.benchmarks import audit_task_gates, mechanism_gate_definition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", required=True, choices=("dalla_man", "cstr", "alien_device")
    )
    parser.add_argument("--task", choices=("T1", "T2", "T3", "T4"))
    parser.add_argument("--tier", required=True, choices=("easy", "hard"))
    parser.add_argument(
        "--dynamics", choices=("canonical", "perturbed"), default="canonical"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    definition = mechanism_gate_definition(
        args.family,
        args.tier,
        task=args.task,
        data_root=args.data_root,
    )
    report = audit_task_gates(
        definition,
        dynamics=args.dynamics,
        data_root=args.data_root,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
