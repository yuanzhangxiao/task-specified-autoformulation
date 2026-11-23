"""Package the committed 20B function comparison and separate train/validation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts.package_t2_breadth import package_committed
from scripts.preproposal_selection_20b import ENTRYPOINTS as SELECTION
from scripts.submit_structural_revision_aces import file_hash
from scripts.submit_trajectory_functions import ENTRYPOINTS

NAME = "phase-c-trajectory-functions-20b-v1"


def package(
    destination: Path, inputs: Path, harmony: Path, *, basin_review=False
) -> Path:
    """Verify the split binding and write a manifest of the delivery bytes."""
    train, validation = inputs / "train.json", inputs / "validation.json"
    training, holdout = sealed_read(train), sealed_read(validation)
    if (
        holdout["split"] != "val"
        or holdout["training_intake_sha256"] != training["artifact_sha256"]
        or holdout["source_files"] != training["validation_files"]
    ):
        raise ValueError("Validation packet does not match the training intake")
    tracked = list(
        dict.fromkeys(
            [
                "src",
                "pyproject.toml",
                *ENTRYPOINTS,
                *SELECTION,
                "docs/PHASE_C_TRAJECTORY_GUIDED_FUNCTIONS.md",
            ]
        )
    )
    return package_committed(
        destination,
        "phase-c-basin-function-review-20b-v1" if basin_review else NAME,
        tracked,
        "inputs/train.json",
        train,
        harmony,
        extra_inputs=(("inputs/validation.json", validation),),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("out", "inputs", "harmony"):
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--basin-review", action="store_true")
    args = parser.parse_args()
    archive = package(
        args.out, args.inputs, args.harmony, basin_review=args.basin_review
    )
    print(json.dumps({"archive": str(archive), "sha256": file_hash(archive)}, indent=2))
