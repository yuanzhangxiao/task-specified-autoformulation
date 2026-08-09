#!/usr/bin/env python3
"""Generate and audit private Phase-B reference trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import (
    audit_basic_gates,
    phase_b_protocols,
    simulate_phase_b,
    write_private_bundle,
)

TARGETS = {"dalla_man": "Gp", "cstr": "T", "alien_device": "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=tuple(TARGETS))
    parser.add_argument("--task", choices=("T1", "T2", "T3", "T4"))
    parser.add_argument(
        "--dynamics", choices=("canonical", "perturbed"), default="canonical"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seal-test",
        action="store_true",
        help="Generate the frozen test references exactly once after audits pass.",
    )
    args = parser.parse_args()

    protocols = phase_b_protocols(args.family, task=args.task)
    selected = tuple(
        item for item in protocols if args.seal_test or item.split != "test"
    )
    trajectories = tuple(
        simulate_phase_b(
            item,
            dynamics=args.dynamics,
            data_root=args.data_root,
        )
        for item in selected
    )
    output = args.output_root.resolve()
    write_private_bundle(output, selected, trajectories)
    train = tuple(
        trajectory
        for protocol, trajectory in zip(selected, trajectories, strict=True)
        if protocol.split == "train"
    )
    report = audit_basic_gates(
        args.family,
        train,
        target_name=TARGETS[args.family],
    )
    (output / "basic_gate_report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
