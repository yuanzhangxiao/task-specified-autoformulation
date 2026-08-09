#!/usr/bin/env python3
"""Stage audited Phase-B public assets without registering the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import (
    audit_public_bundle,
    phase_b_protocols,
    phase_b_public_spec,
    simulate_phase_b,
    write_public_staging_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--family", required=True, choices=("dalla_man", "cstr", "alien_device")
    )
    parser.add_argument("--task", choices=("T1", "T2", "T3", "T4"))
    parser.add_argument("--tier", required=True, choices=("easy", "hard"))
    parser.add_argument(
        "--semantic-variant",
        required=True,
        choices=("named", "obfuscated", "functional", "opaque"),
    )
    parser.add_argument(
        "--dynamics", choices=("canonical", "perturbed"), default="canonical"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--seal-test",
        action="store_true",
        help=(
            "Include test assets only after selection/evaluation protocols are "
            "frozen."
        ),
    )
    args = parser.parse_args()

    spec = phase_b_public_spec(
        args.family,
        args.tier,
        args.semantic_variant,
        task=args.task,
        dynamics=args.dynamics,
        data_root=args.data_root,
    )
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
    write_public_staging_bundle(
        args.output_root,
        spec,
        trajectories,
        seal_test=args.seal_test,
    )
    report = audit_public_bundle(args.output_root, spec)
    print(json.dumps(report.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
