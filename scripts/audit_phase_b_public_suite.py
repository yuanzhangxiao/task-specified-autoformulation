#!/usr/bin/env python3
"""Stage and audit all 40 public Phase-B cells without sealing test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks import (
    audit_public_bundle,
    load_suite_spec,
    phase_b_protocols,
    phase_b_public_spec,
    simulate_phase_b,
    write_public_staging_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/benchmarks/phase_b_suite_v1.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    suite = load_suite_spec(args.suite)
    records: list[dict[str, object]] = []
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            protocols = tuple(
                item
                for item in phase_b_protocols(family.family, task=task_argument)
                if item.split != "test"
            )
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                trajectories = tuple(
                    simulate_phase_b(
                        item,
                        dynamics=dynamics,
                        data_root=args.data_root,
                    )
                    for item in protocols
                )
                for tier in family.tiers:
                    pair_commitments: dict[str, dict[str, str]] = {}
                    for variant in family.semantic_variants:
                        spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=args.data_root,
                        )
                        cell_root = args.output_root / spec.benchmark_id
                        write_public_staging_bundle(cell_root, spec, trajectories)
                        report = audit_public_bundle(cell_root, spec)
                        manifest = json.loads(
                            (cell_root / "manifest.json").read_text(encoding="utf-8")
                        )
                        pair_commitments[variant] = manifest["numeric_payload_sha256"]
                        records.append(
                            {
                                "benchmark_id": spec.benchmark_id,
                                "passed": report.passed,
                                "test_sealed": manifest["test_sealed"],
                                "violations": list(report.violations),
                            }
                        )
                    commitments = list(pair_commitments.values())
                    if any(item != commitments[0] for item in commitments[1:]):
                        raise RuntimeError(
                            f"semantic pair numeric mismatch: {family.family}/{task}/"
                            f"{dynamics}/{tier.name}"
                        )

    summary = {
        "schema_version": "phase_b_public_suite_audit_v1",
        "expected_cells": suite.number_of_cells,
        "staged_cells": len(records),
        "passed_cells": sum(bool(item["passed"]) for item in records),
        "test_cells_sealed": sum(bool(item["test_sealed"]) for item in records),
        "public_staging_audit_passed": (
            len(records) == suite.number_of_cells
            and all(bool(item["passed"]) for item in records)
            and not any(bool(item["test_sealed"]) for item in records)
        ),
        "cells": records,
    }
    (args.output_root / "public_suite_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cells"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
