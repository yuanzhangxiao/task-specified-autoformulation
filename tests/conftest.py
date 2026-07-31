"""Synthetic public benchmark fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autoformalism.data.models import BenchmarkSpec, TierRoles
from autoformalism.data.registry import BenchmarkRegistry


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """Create a small two-trajectory, three-split benchmark."""
    benchmark_root = tmp_path / "public" / "synthetic"
    tier_root = benchmark_root / "easy"
    tier_root.mkdir(parents=True)
    manifest = {
        "benchmark_id": "synthetic",
        "time_column": "t",
        "trajectory_id_column": "trajectory_id",
        "sampling_interval": 1.0,
        "tiers": {"easy": ["y", "a"]},
    }
    (benchmark_root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    for split_index, split in enumerate(("train", "val", "test")):
        offset = split_index * 100.0
        observations = pd.DataFrame(
            {
                "y": [offset + 1, offset + 2, offset + 3, offset + 4],
                "a": [10 + offset, 11 + offset, 12 + offset, 13 + offset],
            }
        )
        derivatives = pd.DataFrame(
            {"dy_dt": [1.0] * 4, "da_dt": [1.0] * 4}
        )
        inputs = pd.DataFrame(
            {
                "trajectory_id": [f"{split}_0"] * 2 + [f"{split}_1"] * 2,
                "t": [0.0, 1.0, 0.0, 1.0],
                "u": [offset, offset + 1, offset + 2, offset + 3],
                "c": [1.0, 1.0, 2.0, 2.0],
            }
        )
        observations.to_csv(tier_root / f"X_{split}.csv", index=False)
        observations.to_csv(tier_root / f"X_{split}_clean.csv", index=False)
        derivatives.to_csv(tier_root / f"Y_{split}.csv", index=False)
        inputs.to_csv(benchmark_root / f"input_{split}.csv", index=False)
    return tmp_path


@pytest.fixture
def synthetic_registry() -> BenchmarkRegistry:
    """Return a registry containing only the synthetic benchmark."""
    spec = BenchmarkSpec(
        benchmark_id="synthetic",
        relative_root=Path("public/synthetic"),
        manifest_relative_path=Path("public/synthetic/manifest.json"),
        tier_roles={
            "easy": TierRoles(targets=("y",), auxiliaries=("a",)),
        },
        time_column="t",
        trajectory_id_column="trajectory_id",
        external_inputs=("u",),
        fixed_covariates=("c",),
        input_filename_template="input_{split}.csv",
        sampling_interval=1.0,
    )
    return BenchmarkRegistry((spec,))

