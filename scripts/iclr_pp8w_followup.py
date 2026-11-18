"""Freeze training-count and CSTR acquisition studies without opening test data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_stress_data as data

PROTOCOL = "iclr-pp8w-followup-1"
SOURCE = Path("output/iclr-stress-final-20261115/design")
SEEDS = (0, 1, 2)
SOURCES = {"cstr": "sensor_00", "t2": "acquisition_00"}
# Chosen from public input schedules, never from fitting or held-out scores.
# CSTR: baseline and one upward intervention on each of its three drivers.
# T2: fasting, 90 g meal, shifted 90 g meal, and 90 g meal with insulin.
SUBSETS = {
    "cstr": {4: (0, 1, 5, 9), 8: (0, 1, 2, 5, 6, 9, 10, 15)},
    "t2": {4: (0, 3, 6, 15), 8: (0, 1, 3, 5, 6, 8, 11, 15)},
}
FILES = ("train.json", "val.json", "specification.json", "proposer_prompt.txt")


def publish_cell(
    root: Path,
    name: str,
    spec: dict,
    train: list[dict],
    val: list[dict],
    unchanged: dict[str, Path],
) -> dict:
    """Preserve original split bytes when only the training roster changes."""
    directory = root / "public" / name
    for split, rows in (("train", train), ("val", val)):
        if split in unchanged:
            value = read_seal(unchanged[split])
            if value["rows"] != rows:
                raise ValueError("unchanged split rows differ")
        else:
            value = data.PublicSplit.model_validate(
                {"name": split, "rows": rows, "fingerprint": data.content_sha256(rows)}
            ).model_dump(mode="json")
        seal(directory / f"{split}.json", value)
    seal(directory / "specification.json", {**spec, "benchmark_id": name})
    prompt = spec["public_prompt"].rstrip("\n") + "\n"
    path = directory / "proposer_prompt.txt"
    if path.exists() and path.read_text() != prompt:
        raise ValueError("frozen public prompt differs")
    path.write_text(prompt)
    return {
        "cell": name,
        "public_spec_sha256": data.digest(directory / "specification.json"),
        "public_files": {f: data.digest(directory / f) for f in FILES},
    }


def selected_rows(rows: list[dict], family: str, count: int) -> list[dict]:
    """Return the predeclared nested subset with no response-based selection."""
    expected = {f"train_{i:03d}" for i in range(16)}
    by_id = {r["trajectory_id"]: r for r in rows}
    if len(rows) != 16 or set(by_id) != expected:
        raise ValueError("expected exactly the sixteen frozen training trajectories")
    if family not in SUBSETS or count not in (4, 8, 16):
        raise ValueError("unsupported family or trajectory count")
    indices = range(16) if count == 16 else SUBSETS[family][count]
    return [by_id[f"train_{i:03d}"] for i in indices]


def prepare(source: Path, root: Path, *, include_eight: bool = False) -> dict:
    """Copy public conditions into a separate immutable, resumable experiment."""
    if root.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(root.resolve()):
        raise ValueError("source and destination must be disjoint")
    manifest = sealed_read(source / "manifest.json")
    original = {c["cell"]: c for c in manifest["cells"]}
    fingerprints = {}
    inputs = {}
    for family, key in SOURCES.items():
        directory = source / "public" / key
        fingerprints[family] = {f: data.digest(directory / f) for f in FILES}
        if fingerprints[family] != original[key]["public_files"]:
            raise ValueError("source public files differ from frozen stress design")
        inputs[family] = (
            read_seal(directory / "specification.json"),
            read_seal(directory / "train.json")["rows"],
            read_seal(directory / "val.json")["rows"],
        )
    identity = sealed_write(
        root / "design.json",
        {
            "protocol": PROTOCOL,
            "source_manifest": manifest["artifact_sha256"],
            "public_source_files": fingerprints,
            "implementation_sha256": data.digest(Path(__file__)),
            "acquisition_implementation_sha256": data.digest(Path(data.__file__)),
            "subsets": json.loads(json.dumps(SUBSETS)),
            "include_eight": include_eight,
            "seeds": list(SEEDS),
            "arms": ["full"],
            "visits": 15,
            "pruning": True,
            "selection": "Unchanged validation and complexity rule; no test access",
            "subset_scope": (
                "One fixed subset per count; search seeds, not data resamples"
            ),
            "validation": (
                "Full original validation roster; noise/stride applied only "
                "in acquisition arms"
            ),
            "training_normalization": (
                "Recomputed by the unchanged fitter on each condition's training rows"
            ),
            "evaluation_normalization": (
                "Common clean sixteen-trajectory training scales"
            ),
            "evaluation_panels": {
                "cstr": "existing 24 interventions",
                "t2": "existing six-case stress panel",
            },
            "serving": (
                "Locally hosted 20B proposer and 120B critic on Delta; "
                "fresh matched sixteen-trajectory controls"
            ),
            "test_data_opened": False,
        },
    )
    cells = []
    counts = (4, 16, 8) if include_eight else (4, 16)
    for count in counts:
        for family in SOURCES:
            spec, train, val = inputs[family]
            rows = selected_rows(train, family, count)
            name = f"{family}_trajectories_{count:02d}"
            source_cell = source / "public" / SOURCES[family]
            unchanged = {"val": source_cell / "val.json"}
            if count == 16:
                unchanged["train"] = source_cell / "train.json"
            cell = publish_cell(root, name, spec, rows, val, unchanged)
            cells.append(
                cell
                | {
                    "family": family,
                    "source": original[SOURCES[family]]["source"],
                    "study": "training_count",
                    "train_trajectories": count,
                    "training_ids": [r["trajectory_id"] for r in rows],
                    "noise": 0,
                    "stride": 1,
                    "mask": list(spec["auxiliaries"]),
                    "scales": data.train_scales(train),
                    "validation_rows_unchanged": True,
                    "historical_source_cell": SOURCES[family],
                }
            )
    spec, train, val = inputs["cstr"]
    scales = data.train_scales(train)
    for name, noise, stride in (("cstr_noise_10pct", 0.1, 1), ("cstr_stride_04", 0, 4)):
        altered = [
            [
                data.acquire(r, scales=scales, fraction=noise, stride=stride)
                for r in split
            ]
            for split in (train, val)
        ]
        cell = data.write_cell(
            root,
            name,
            {**spec, "noise_sd_fraction": noise, "initial_observation_noise": noise},
            *altered,
        )
        cells.append(
            cell
            | {
                "family": "cstr",
                "source": original["sensor_00"]["source"],
                "study": "data_quality",
                "train_trajectories": 16,
                "training_ids": [r["trajectory_id"] for r in train],
                "noise": noise,
                "stride": stride,
                "mask": [],
                "scales": scales,
                "nominal_sampling_minutes": 0.1 * stride,
                "input_corners_retained": True,
                "noise_seed": data.NOISE_SEED,
                "historical_source_cell": "sensor_00",
            }
        )
    return sealed_write(
        root / "manifest.json",
        {
            "protocol": PROTOCOL,
            "design_sha256": identity["artifact_sha256"],
            "cells": cells,
            "algorithm_seeds": list(SEEDS),
            "arms": ["full"],
            "runs": [
                {"benchmark_id": c["cell"], "arm": "full", "repetition": s}
                for c in cells
                for s in SEEDS
            ],
            "test_data_opened": False,
            "provider_calls": 0,
        },
    )


def main() -> None:
    """Create the public-only package before any discovery or scoring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-eight", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source, args.root, include_eight=args.include_eight)
    print(
        json.dumps(
            {
                "manifest_sha256": result["artifact_sha256"],
                "cells": len(result["cells"]),
                "runs": len(result["runs"]),
            }
        )
    )


if __name__ == "__main__":
    main()
