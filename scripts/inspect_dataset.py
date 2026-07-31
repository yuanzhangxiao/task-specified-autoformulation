#!/usr/bin/env python3
"""Inspect a registered dataset without printing trajectory values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autoformalism.config import DataConfig, load_config
from autoformalism.data.loader import BenchmarkLoader
from autoformalism.data.models import DatasetSplit
from autoformalism.data.registry import BenchmarkRegistry
from autoformalism.data.scaling import TrainingScaler


def _split_summary(split: DatasetSplit) -> dict[str, Any]:
    return {
        "name": split.name.value,
        "number_of_trajectories": len(split.trajectories),
        "number_of_rows": sum(item.number_of_rows for item in split.trajectories),
        "trajectory_ids": [item.trajectory_id for item in split.trajectories],
        "fingerprint": split.fingerprint,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--tier", choices=("easy", "medium", "hard"))
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--list", action="store_true", help="list benchmark IDs")
    return parser


def main() -> None:
    """Load configuration, validate data, and print structural metadata."""
    args = build_parser().parse_args()
    registry = BenchmarkRegistry()
    if args.list:
        print("\n".join(registry.identifiers()))
        return

    settings = load_config(args.env_file)
    config = DataConfig(
        root=args.data_root or settings.data_root,
        benchmark_id=args.benchmark or settings.benchmark_id,
        tier=args.tier or settings.tier,
        use_clean_observations=(
            settings.use_clean_observations if args.clean is None else args.clean
        ),
        scaling_epsilon=settings.scaling_epsilon,
    )
    dataset = BenchmarkLoader(registry).load(config)
    scaler = TrainingScaler(config.scaling_epsilon).fit(dataset.train)
    summary = {
        "benchmark_id": dataset.benchmark_id,
        "tier": dataset.tier,
        "targets": list(dataset.roles.targets),
        "auxiliaries": list(dataset.roles.auxiliaries),
        "splits": [
            _split_summary(dataset.train),
            _split_summary(dataset.validation),
            _split_summary(dataset.test),
        ],
        "training_scales": {
            name: {
                "mean": scale.mean,
                "standard_deviation": scale.standard_deviation,
            }
            for name, scale in scaler.scales.items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

