"""Read separately versioned exact-observed-derivative overlays."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from autoformalism.data.models import (
    DatasetSplit,
    DerivativeProvenance,
    Trajectory,
)


def attach_exact_derivative_overlay(
    split: DatasetSplit,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> DatasetSplit:
    """Attach an exact derivative CSV after strict ID/time alignment checks.

    The overlay format is tidy: ``trajectory_id``, ``t``, and one or more
    columns named ``d__<public_channel>``. The public observations are retained
    byte-for-byte in memory; only the derivative mapping and its provenance are
    replaced.
    """
    path = path.expanduser().resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"exact derivative overlay SHA-256 differs: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields[:2] != ("trajectory_id", "t"):
            raise ValueError(
                "exact derivative overlay must begin with trajectory_id,t"
            )
        derivative_fields = fields[2:]
        if not derivative_fields or any(
            not name.startswith("d__") for name in derivative_fields
        ):
            raise ValueError(
                "exact derivative columns must use the d__<channel> convention"
            )
        channels = tuple(name.removeprefix("d__") for name in derivative_fields)
        if len(channels) != len(set(channels)):
            raise ValueError("exact derivative overlay has duplicate channels")
        rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            rows[str(row["trajectory_id"])].append(row)

    expected_ids = tuple(item.trajectory_id for item in split.trajectories)
    if set(rows) != set(expected_ids):
        raise ValueError(
            "exact derivative trajectory IDs differ: "
            f"expected={sorted(expected_ids)}, actual={sorted(rows)}"
        )
    trajectories: list[Trajectory] = []
    for trajectory in split.trajectories:
        selected = rows[trajectory.trajectory_id]
        try:
            time = np.asarray([float(row["t"]) for row in selected], dtype=float)
            derivatives = {
                channel: np.asarray(
                    [float(row[f"d__{channel}"]) for row in selected],
                    dtype=float,
                )
                for channel in channels
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"exact derivative overlay is nonnumeric: {trajectory.trajectory_id}"
            ) from exc
        if time.shape != trajectory.time.shape or not np.allclose(
            time, trajectory.time, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "exact derivative times differ for " + trajectory.trajectory_id
            )
        if not all(np.isfinite(values).all() for values in derivatives.values()):
            raise ValueError(
                "exact derivative overlay is nonfinite: " + trajectory.trajectory_id
            )
        available = set(trajectory.targets) | set(trajectory.auxiliaries)
        unknown = sorted(set(derivatives) - available)
        if unknown:
            raise ValueError(
                "exact derivative overlay references unavailable channels: "
                + ", ".join(unknown)
            )
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory.trajectory_id,
                time=trajectory.time.copy(),
                targets={
                    name: values.copy() for name, values in trajectory.targets.items()
                },
                auxiliaries={
                    name: values.copy()
                    for name, values in trajectory.auxiliaries.items()
                },
                external_inputs={
                    name: values.copy()
                    for name, values in trajectory.external_inputs.items()
                },
                fixed_covariates=dict(trajectory.fixed_covariates),
                derivatives=derivatives,
                derivative_provenance=DerivativeProvenance.EXACT,
            )
        )
    combined = hashlib.sha256(f"{split.fingerprint}:{digest}".encode()).hexdigest()
    return DatasetSplit(split.name, tuple(trajectories), combined)
