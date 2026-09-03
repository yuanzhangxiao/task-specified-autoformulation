#!/usr/bin/env python3
"""Build a sealed exact-observed-derivative overlay for a frozen Phase-B pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.benchmarks import (
    PhaseBPublicSpec,
    phase_b_protocols,
    phase_b_public_spec,
)
from autoformalism.benchmarks.phase_b_generation import (
    PrivateTrajectory,
    simulate_phase_b,
)

_NOMINAL_ABSOLUTE_TOLERANCE = 1e-7
_NOMINAL_RELATIVE_TOLERANCE = {
    "dalla_man": 1e-4,
    "cstr": 1e-4,
    # The frozen Alien public solve is unusually sensitive to SciPy versions.
    # This is a provenance-alignment guard, not an evaluation threshold.
    "alien_device": 2e-3,
}


def build_overlay(
    config_path: Path,
    public_data_root: Path,
    private_data_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Generate train/validation derivatives without opening test trajectories."""
    config_path = config_path.expanduser().resolve()
    public_data_root = public_data_root.expanduser().resolve()
    private_data_root = private_data_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    config = _read_object(config_path)
    if config.get("schema_version") != "phase-b-reciprocal-fitting-pilot-1":
        raise ValueError("unexpected reciprocal fitting pilot schema")
    results: list[dict[str, object]] = []
    for cell in config["cells"]:
        if not isinstance(cell, dict):
            raise ValueError("pilot cell must be an object")
        benchmark_id = str(cell["benchmark_id"])
        spec = phase_b_public_spec(
            str(cell["family"]),
            str(cell["tier"]),
            str(cell["semantic_variant"]),
            task=str(cell["task"]),
            dynamics=str(cell["dynamics"]),
            data_root=private_data_root,
        )
        if spec.benchmark_id != benchmark_id:
            raise ValueError(f"public specification ID differs: {benchmark_id}")
        source_root = public_data_root / "phase_b_v1" / benchmark_id
        prompt_path = source_root / "proposer_prompt.txt"
        if _sha256(prompt_path) != cell["public_prompt_sha256"]:
            raise ValueError(f"public proposer prompt differs: {benchmark_id}")
        protocols = tuple(
            item
            for item in phase_b_protocols(
                str(cell["family"]),
                task=(str(cell["task"]) if cell["family"] == "dalla_man" else None),
            )
            if item.split == "train"
        )
        trajectories = tuple(
            simulate_phase_b(
                protocol,
                dynamics=str(cell["dynamics"]),
                data_root=private_data_root,
            )
            for protocol in protocols
        )
        split_rows: dict[str, object] = {}
        for split_name in ("train",):
            members = tuple(
                item
                for item in trajectories
                if item.protocol_id.startswith(f"{split_name}_")
            )
            public_path = source_root / f"{split_name}.csv"
            channels = _validate_public_alignment(public_path, spec, members)
            output_path = output_root / benchmark_id / f"{split_name}.csv"
            _write_derivatives(output_path, spec, members, channels)
            split_rows[split_name] = {
                "public_split_sha256": _sha256(public_path),
                "derivative_overlay_sha256": _sha256(output_path),
                "trajectory_count": len(members),
                "derivative_channels": list(channels),
                "nominal_alignment_relative_tolerance": (
                    _NOMINAL_RELATIVE_TOLERANCE[spec.family]
                ),
                "nominal_alignment_absolute_tolerance": (
                    _NOMINAL_ABSOLUTE_TOLERANCE
                ),
            }
        results.append(
            {
                "benchmark_id": benchmark_id,
                "public_prompt_sha256": str(cell["public_prompt_sha256"]),
                "splits": split_rows,
            }
        )
    manifest = {
        "schema_version": "phase-b-exact-observed-derivative-overlay-1",
        "status": "complete",
        "development_only": True,
        "oracle_training_observed_derivatives_supplied": True,
        "validation_derivatives_supplied": False,
        "latent_values_supplied": False,
        "latent_derivatives_supplied": False,
        "private_simulator_used_by_overlay_builder": True,
        "private_reference_available_to_fitter": False,
        "test_data_opened": False,
        "config_sha256": _sha256(config_path),
        "cells": results,
    }
    manifest_path = output_root / "manifest.json"
    _write_once(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(
        output_root / "manifest.json.sha256",
        f"{_sha256(manifest_path)}  manifest.json\n".encode(),
    )
    return manifest


def _validate_public_alignment(
    path: Path,
    spec: PhaseBPublicSpec,
    trajectories: tuple[PrivateTrajectory, ...],
) -> tuple[str, ...]:
    """Check regenerated public values before releasing derivative labels."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = sum(len(item.time) for item in trajectories)
    if len(rows) != expected_count:
        raise ValueError(f"public row count differs: {path}")
    derivative_channels = tuple(
        channel.public_name
        for channel in spec.channels
        if channel.role != "external_input"
        and all(channel.private_source in item.derivatives for item in trajectories)
    )
    required_targets = {
        channel.public_name for channel in spec.channels if channel.role == "target"
    }
    if not required_targets.issubset(derivative_channels):
        missing = sorted(required_targets - set(derivative_channels))
        raise ValueError(f"exact target derivatives are unavailable: {missing}")
    cursor = 0
    for trajectory_index, trajectory in enumerate(trajectories):
        split = trajectory.protocol_id.split("_", maxsplit=1)[0]
        neutral_id = f"{split}_{trajectory_index:03d}"
        for time_index, time in enumerate(trajectory.time):
            row = rows[cursor]
            cursor += 1
            if row.get("trajectory_id") != neutral_id or not np.isclose(
                float(row["t"]), float(time), rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"public trajectory alignment differs: {path}")
            for channel in spec.channels:
                private_values = _private_array(
                    trajectory, channel.private_source
                )
                expected = private_values[time_index]
                if not np.isclose(
                    float(row[channel.public_name]),
                    float(expected),
                    rtol=_NOMINAL_RELATIVE_TOLERANCE[spec.family],
                    atol=_NOMINAL_ABSOLUTE_TOLERANCE,
                ):
                    raise ValueError(
                        "public/private nominal channel differs beyond overlay "
                        f"tolerance: {path}:{neutral_id}:{channel.public_name}"
                    )
    return derivative_channels


def _write_derivatives(
    path: Path,
    spec: PhaseBPublicSpec,
    trajectories: tuple[PrivateTrajectory, ...],
    channels: tuple[str, ...],
) -> None:
    by_public = {item.public_name: item.private_source for item in spec.channels}
    header = ["trajectory_id", "t", *(f"d__{name}" for name in channels)]
    lines: list[list[str]] = [header]
    for trajectory_index, trajectory in enumerate(trajectories):
        split = trajectory.protocol_id.split("_", maxsplit=1)[0]
        neutral_id = f"{split}_{trajectory_index:03d}"
        for index, time in enumerate(trajectory.time):
            lines.append(
                [
                    neutral_id,
                    f"{float(time):.17g}",
                    *(
                        f"{float(trajectory.derivatives[by_public[name]][index]):.17g}"
                        for name in channels
                    ),
                ]
            )
    rendered = "\n".join(",".join(row) for row in lines) + "\n"
    _write_once(path, rendered.encode())


def _private_array(trajectory: PrivateTrajectory, name: str) -> np.ndarray:
    if name in trajectory.state_names:
        return trajectory.states[:, trajectory.state_names.index(name)]
    if name in trajectory.input_names:
        return trajectory.inputs[:, trajectory.input_names.index(name)]
    if name in trajectory.derived:
        return trajectory.derived[name]
    raise ValueError(f"private source is unavailable: {name}")


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"write-once derivative artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--public-data-root", type=Path, required=True)
    parser.add_argument("--private-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = build_overlay(
        config_path=arguments.config,
        public_data_root=arguments.public_data_root,
        private_data_root=arguments.private_data_root,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
