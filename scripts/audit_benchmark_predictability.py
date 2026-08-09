#!/usr/bin/env python3
"""Run Phase B0 shortcut and excitation diagnostics on public splits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from autoformalism.config import DataConfig
from autoformalism.data.loader import BenchmarkLoader
from autoformalism.data.models import BenchmarkSpec, TierRoles
from autoformalism.data.registry import BenchmarkRegistry
from autoformalism.rebuttal.benchmark_audit import (
    audit_excitation,
    audit_response_phases,
    audit_shortcuts,
    downsample_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="*")
    parser.add_argument("--tiers", nargs="*", default=["easy", "medium", "hard"])
    parser.add_argument("--horizons", nargs="*", type=int, default=[1, 5, 10, 30])
    parser.add_argument("--downsample-strides", nargs="*", type=int, default=[1, 5, 10])
    parser.add_argument("--include-original-dalla-tasks", action="store_true")
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    registry = BenchmarkRegistry(
        BenchmarkRegistry().specs()
        + (_original_dalla_task_specs() if args.include_original_dalla_tasks else ())
    )
    benchmarks = args.benchmarks or list(registry.identifiers())
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shortcut_rows: list[dict[str, object]] = []
    excitation_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []
    downsample_rows: list[dict[str, object]] = []
    for benchmark in benchmarks:
        for tier in args.tiers:
            dataset = BenchmarkLoader(registry).load(
                DataConfig(
                    root=args.data_root,
                    benchmark_id=benchmark,
                    tier=tier,
                    use_clean_observations=args.clean,
                )
            )
            for target in dataset.roles.targets:
                excitation_rows.append(
                    {
                        "benchmark": benchmark,
                        "tier": tier,
                        **asdict(audit_excitation(dataset.train, target)),
                    }
                )
                for split_name, split in (
                    ("validation", dataset.validation),
                    ("test", dataset.test),
                ):
                    for record in audit_shortcuts(
                        dataset.train, split, target, tuple(args.horizons)
                    ):
                        shortcut_rows.append(
                            {
                                "benchmark": benchmark,
                                "tier": tier,
                                "target": target,
                                "split": split_name,
                                **asdict(record),
                            }
                        )
                    for record in audit_response_phases(
                        dataset.train, split, target, tuple(args.horizons)
                    ):
                        phase_rows.append(
                            {
                                "benchmark": benchmark,
                                "tier": tier,
                                "target": target,
                                "split": split_name,
                                **asdict(record),
                            }
                        )
                for stride in args.downsample_strides:
                    physical_horizons = tuple(
                        horizon // stride
                        for horizon in args.horizons
                        if horizon >= stride and horizon % stride == 0
                    )
                    if not physical_horizons:
                        continue
                    sampled_train = downsample_split(dataset.train, stride)
                    sampled_test = downsample_split(dataset.test, stride)
                    for record in audit_shortcuts(
                        sampled_train,
                        sampled_test,
                        target,
                        physical_horizons,
                        event_radius=max(1, 30 // stride),
                    ):
                        downsample_rows.append(
                            {
                                "benchmark": benchmark,
                                "tier": tier,
                                "target": target,
                                "stride": stride,
                                "physical_horizon": record.horizon * stride,
                                **asdict(record),
                            }
                        )
    _write_csv(output_root / "shortcut_metrics.csv", shortcut_rows)
    _write_csv(output_root / "excitation_metrics.csv", excitation_rows)
    _write_csv(output_root / "response_phase_metrics.csv", phase_rows)
    _write_csv(output_root / "downsampling_metrics.csv", downsample_rows)
    manifest = {
        "schema_version": "1",
        "benchmarks": benchmarks,
        "tiers": args.tiers,
        "horizons": args.horizons,
        "downsample_strides": args.downsample_strides,
        "use_clean_observations": args.clean,
        "shortcut_rows": len(shortcut_rows),
        "excitation_rows": len(excitation_rows),
        "phase_rows": len(phase_rows),
        "downsample_rows": len(downsample_rows),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows generated for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _original_dalla_task_specs() -> tuple[BenchmarkSpec, ...]:
    """Return analysis-only specs for all four finalized original tasks."""

    task_channels = (
        (
            "B1_meal_appearance",
            {
                "easy": ("Gp", "EGP", "Uii", "E", "Gt"),
                "medium": ("Gp", "EGP", "Uii"),
                "hard": ("Gp",),
            },
        ),
        (
            "B2_absorption_action",
            {
                "easy": ("Gp", "I", "EGP", "Uii", "E", "Gt", "U"),
                "medium": ("Gp", "I", "U", "Uii"),
                "hard": ("Gp", "I"),
            },
        ),
        (
            "B3_hepatic_regulation",
            {
                "easy": ("Gp", "I", "EGP", "Uii", "E", "Gt", "U", "Ipo"),
                "medium": ("Gp", "I", "EGP"),
                "hard": ("Gp", "I"),
            },
        ),
        (
            "B4_flux_portrait",
            {
                "easy": ("Gp", "I", "Uii", "E", "Gt", "Ipo"),
                "medium": ("Gp", "I", "Uii", "E", "Gt"),
                "hard": ("Gp", "I"),
            },
        ),
    )
    specs: list[BenchmarkSpec] = []
    for index, (task, channels) in enumerate(task_channels, start=1):
        roles = {
            tier: TierRoles(targets=tuple(channels[tier]))
            for tier in ("easy", "medium", "hard")
        }
        specs.append(
            BenchmarkSpec(
                benchmark_id=f"original_t{index}",
                relative_root=Path(f"benchmark1_original_dalla_man/benchmarks/{task}"),
                manifest_relative_path=Path(
                    "benchmark1_original_dalla_man/manifest.json"
                ),
                tier_roles=roles,
                time_column="time",
                external_inputs=("meal_event_g",),
                fixed_covariates=("body_weight_kg",),
                input_filename_template="metadata_{split}.csv",
                sampling_interval=1.0,
            )
        )
    return tuple(specs)


if __name__ == "__main__":
    main()
