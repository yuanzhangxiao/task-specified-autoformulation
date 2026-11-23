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
    destination: Path,
    inputs: Path,
    harmony: Path,
    *,
    basin_review=False,
    basin_processes=False,
    pointwise_repair=False,
    fit_campaign: Path | None = None,
) -> Path:
    """Verify the split binding and write a manifest of the delivery bytes."""
    if fit_campaign is not None and not pointwise_repair:
        raise ValueError("Parent fit receipts apply only to pointwise repair")
    if sum((basin_review, basin_processes, pointwise_repair)) > 1:
        raise ValueError("Choose one experiment")
    tracked = list(
        dict.fromkeys(
            [
                "src",
                "pyproject.toml",
                *ENTRYPOINTS,
                *SELECTION,
                "scripts/replay_trajectory_function_pointwise.py",
                "scripts/replay_trajectory_function_revisions.py",
                "docs/PHASE_C_TRAJECTORY_GUIDED_FUNCTIONS.md",
            ]
        )
    )
    if pointwise_repair:
        from scripts.pointwise_repair_trial import read_source

        read_source(inputs)
        source_files = (
            "plan.json",
            "intake.json",
            "generation.json",
            "candidates/002.json",
            "candidates/014.json",
            "candidates/015.json",
        )
        extra = [(f"inputs/source/{p}", inputs / p) for p in source_files]
        if fit_campaign is not None:
            from scripts.pointwise_repair_fitting import SLOTS, parents_from_archive

            parents_from_archive(fit_campaign)
            fit_files = [
                "plan.json",
                "intake.json",
                "generation.json",
                "finalists.json",
            ]
            for i in SLOTS:
                fit_files.extend(
                    (f"candidates/{i:03}.json", f"fits/{i:03}/result.json")
                )
            extra.extend((f"inputs/parents/{p}", fit_campaign / p) for p in fit_files)
        return package_committed(
            destination,
            "phase-c-pointwise-repair-20b-v1",
            [
                *tracked,
                "scripts/pointwise_repair_trial.py",
                "scripts/pointwise_repair_fitting.py",
                "scripts/trajectory_boundary_trial.py",
                "scripts/hpc/start_pointwise_repair.sh",
            ],
            "inputs/source/admission.json",
            inputs / "admission.json",
            harmony,
            extra_inputs=tuple(extra),
        )
    train, validation = inputs / "train.json", inputs / "validation.json"
    training, holdout = sealed_read(train), sealed_read(validation)
    if (
        holdout["split"] != "val"
        or holdout["training_intake_sha256"] != training["artifact_sha256"]
        or holdout["source_files"] != training["validation_files"]
    ):
        raise ValueError("Validation packet does not match the training intake")
    name = "phase-c-basin-function-review-20b-v1" if basin_review else NAME
    if basin_processes:
        name = "phase-c-basin-process-laws-v1"
        tracked += [
            "scripts/basin_function_family.py",
            "scripts/basin_family_stability.py",
            "scripts/basin_process_proposals.py",
            "scripts/fit_basin_process_proposals.py",
            "scripts/replay_basin_process_proposals.py",
            "scripts/hpc/start_basin_process_laws.sh",
        ]
    return package_committed(
        destination,
        name,
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
    parser.add_argument(
        "--fit-campaign",
        type=Path,
        help="Bundle the authenticated training parents for subsequent CPU fitting",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--basin-review", action="store_true")
    group.add_argument("--basin-processes", action="store_true")
    group.add_argument(
        "--pointwise-repair",
        action="store_true",
        help="Use a sealed pointwise replay campaign as --inputs",
    )
    args = parser.parse_args()
    archive = package(
        args.out,
        args.inputs,
        args.harmony,
        basin_review=args.basin_review,
        basin_processes=args.basin_processes,
        pointwise_repair=args.pointwise_repair,
        fit_campaign=args.fit_campaign,
    )
    print(json.dumps({"archive": str(archive), "sha256": file_hash(archive)}, indent=2))
