"""Evaluate private hidden references only after selections are frozen."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from autoformalism.rebuttal.hidden import hidden_mechanism_nmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-selection-manifest", type=Path, required=True)
    parser.add_argument("--private-aligned-values", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--allow-signed", action="store_true")
    args = parser.parse_args()
    if not args.frozen_selection_manifest.is_file():
        raise SystemExit(
            "frozen selection manifest is required before hidden evaluation"
        )
    # Parse the manifest before touching private values. It is deliberately not
    # imported by the ordinary experiment pipeline.
    json.loads(args.frozen_selection_manifest.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with args.private_aligned_values.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["artifact_id"], row["mechanism_id"])
            split = row["split"]
            grouped[key][f"{split}_candidate"].append(float(row["candidate_value"]))
            grouped[key][f"{split}_reference"].append(float(row["reference_value"]))
    output = []
    for (artifact_id, mechanism_id), values in sorted(grouped.items()):
        metric = hidden_mechanism_nmse(
            np.asarray(values["train_candidate"]),
            np.asarray(values["train_reference"]),
            np.asarray(values["test_candidate"]),
            np.asarray(values["test_reference"]),
            allow_signed_scale=args.allow_signed,
        )
        output.append(
            {
                "artifact_id": artifact_id,
                "mechanism_id": mechanism_id,
                **metric.model_dump(mode="json"),
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "hidden_mechanism_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = tuple(output[0]) if output else ("artifact_id", "mechanism_id")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
