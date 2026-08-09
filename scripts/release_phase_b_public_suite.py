#!/usr/bin/env python3
"""Materialize all frozen Phase-B v1 public cells with sealed test splits."""

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
    write_public_production_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("configs/benchmarks/phase_b_suite_v1.json"),
    )
    parser.add_argument("--private-data-root", type=Path, default=Path("data_raw"))
    parser.add_argument("--public-data-root", type=Path, required=True)
    args = parser.parse_args()

    suite = load_suite_spec(args.suite)
    release_root = args.public_data_root / "phase_b_v1"
    records: list[dict[str, object]] = []
    for family in suite.families:
        for task in family.tasks:
            task_argument = task if family.family == "dalla_man" else None
            protocols = phase_b_protocols(family.family, task=task_argument)
            for condition in family.dynamics_conditions:
                dynamics = "canonical" if condition == "not_applicable" else condition
                trajectories = tuple(
                    simulate_phase_b(
                        protocol,
                        dynamics=dynamics,
                        data_root=args.private_data_root,
                    )
                    for protocol in protocols
                )
                for tier in family.tiers:
                    commitments: list[dict[str, str]] = []
                    for variant in family.semantic_variants:
                        spec = phase_b_public_spec(
                            family.family,
                            tier.name,
                            variant,
                            task=task_argument,
                            dynamics=dynamics,
                            data_root=args.private_data_root,
                        )
                        cell_root = release_root / spec.benchmark_id
                        write_public_production_bundle(cell_root, spec, trajectories)
                        report = audit_public_bundle(cell_root, spec)
                        manifest = json.loads(
                            (cell_root / "manifest.json").read_text(encoding="utf-8")
                        )
                        commitments.append(manifest["numeric_payload_sha256"])
                        records.append(
                            {
                                "benchmark_id": spec.benchmark_id,
                                "passed": report.passed,
                                "test_sealed": manifest["test_sealed"],
                                "status": manifest["status"],
                            }
                        )
                    if any(item != commitments[0] for item in commitments[1:]):
                        raise RuntimeError(
                            "semantic pair numeric commitment mismatch for "
                            f"{family.family}/{task}/{dynamics}/{tier.name}"
                        )

    passed = (
        len(records) == suite.number_of_cells
        and all(bool(item["passed"]) for item in records)
        and all(bool(item["test_sealed"]) for item in records)
        and all(item["status"] == "production_registered" for item in records)
    )
    summary = {
        "schema_version": "phase_b_public_release_audit_v1",
        "expected_cells": suite.number_of_cells,
        "released_cells": len(records),
        "passed_cells": sum(bool(item["passed"]) for item in records),
        "test_cells_sealed": sum(bool(item["test_sealed"]) for item in records),
        "release_audit_passed": passed,
        "cells": records,
    }
    (release_root / "release_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    headline = {key: value for key, value in summary.items() if key != "cells"}
    print(json.dumps(headline, indent=2))
    if not passed:
        raise SystemExit("Phase-B public release audit failed")


if __name__ == "__main__":
    main()
