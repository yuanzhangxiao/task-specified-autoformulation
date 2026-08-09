#!/usr/bin/env python3
"""Generate hashed private references for a prespecified intervention suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from autoformalism.rebuttal.interventions import (
    ReferenceTrajectory,
    file_sha256,
    load_intervention_suite,
    load_system_spec,
    simulate_reference,
    suite_benchmarks,
)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _write_trajectory(path: Path, trajectory: ReferenceTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}."
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            forcing_count = len(trajectory.forcing[0])
            state_count = len(trajectory.states_clean[0])
            writer.writerow(
                ["time"]
                + [f"forcing_{index}" for index in range(forcing_count)]
                + [f"state_{index}_clean" for index in range(state_count)]
                + [f"state_{index}_observed" for index in range(state_count)]
            )
            for row in zip(
                trajectory.time,
                trajectory.forcing,
                trajectory.states_clean,
                trajectory.states_observed,
                strict=True,
            ):
                time, forcing, clean, observed = row
                writer.writerow([time, *forcing, *clean, *observed])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def freeze_suite(
    *, suite_path: Path, data_root: Path, output_root: Path
) -> dict[str, Any]:
    """Generate references and return a provenance-rich frozen manifest."""

    suite = load_intervention_suite(suite_path)
    specs: dict[str, dict[str, Any]] = {}
    spec_sources: dict[str, dict[str, str]] = {}
    for benchmark_id in suite_benchmarks(suite.cases):
        path, spec = load_system_spec(data_root, benchmark_id)
        specs[benchmark_id] = spec
        spec_sources[benchmark_id] = {
            "path": str(path.relative_to(data_root)),
            "sha256": file_sha256(path),
        }

    records: list[dict[str, Any]] = []
    for case in suite.cases:
        trajectory = simulate_reference(case, system_spec=specs[case.benchmark_id])
        relative_path = Path("references") / f"{case.case_id}.csv"
        output_path = output_root / relative_path
        _write_trajectory(output_path, trajectory)
        records.append(
            {
                "case_id": case.case_id,
                "benchmark_id": case.benchmark_id,
                "shift_types": list(case.shift_types),
                "samples": len(trajectory.time),
                "reference_path": str(relative_path),
                "reference_sha256": file_sha256(output_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": "1",
        "suite_id": suite.suite_id,
        "suite_fingerprint": suite.fingerprint,
        "suite_source": str(suite_path),
        "suite_source_sha256": file_sha256(suite_path),
        "frozen_before_evaluation": True,
        "uses_private_reference": True,
        "available_to_proposal_fit_or_selection": False,
        "system_spec_sources": spec_sources,
        "generator_sources": {
            str(path): file_sha256(path)
            for path in (
                Path("src/autoformalism/rebuttal/interventions.py"),
                Path("src/autoformalism/rebuttal/dalla_man.py"),
            )
            if path.is_file()
        },
        "cases": records,
    }
    _atomic_text(
        output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_suite(
        suite_path=args.suite,
        data_root=args.data_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
