"""Audit saved training equations for opposite directions at comparable forcing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research.affine_direction_certificate import certify
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from scripts import trajectory_boundary_trial as boundary_trial
from scripts import trajectory_functions as trial


def audit(campaign: Path, boundary: Path, tolerance: float = 1.0) -> dict:
    """Inspect the sealed formula plan, using its source's public training cohort."""
    source, intake, generation, finalists = boundary_trial._sources(campaign)
    plan = sealed_read(boundary / "plan.json")
    if (
        plan["protocol"] != "trajectory-boundary-trial-1"
        or plan["source_plan"] != source["artifact_sha256"]
        or plan["source_finalists"] != finalists["artifact_sha256"]
    ):
        raise ValueError(
            "The boundary formulas do not belong to this training campaign"
        )
    outcome = []
    graph_rows = trial.slots(intake)
    for model in plan["models"]:
        index = model["source_slot"]
        candidate = trial._read_candidate(campaign, source, index)
        fit = sealed_read(campaign / f"fits/{index:03d}/result.json")
        if (
            candidate["artifact_sha256"] != model["parent_candidate"]
            or fit["artifact_sha256"] != model["parent_fit"]
            or generation["candidates"][index] != candidate["artifact_sha256"]
        ):
            raise ValueError("Candidate or fit differs from the source model")
        g = graph_rows[index][0]
        frozen = trial.training_case(intake, intake["graphs"][g])
        evidence = TrainingCompatibilityEvidence.model_validate(frozen["evidence"])
        result = certify(
            model["graph"],
            model["formula"],
            plan["target"],
            evidence,
            tolerance=tolerance,
        )
        outcome.append(
            {
                "slot": index,
                "arm": model["arm"],
                "evidence": evidence.sha256,
                "source_fit": model["parent_fit"],
                **result,
            }
        )
    return {
        "protocol": "saved-affine-direction-audit-1",
        "boundary_plan": plan["artifact_sha256"],
        "source_plan": source["artifact_sha256"],
        "tolerance": tolerance,
        "rows": outcome,
        "optimizer_calls": 0,
        "llm_calls": 0,
        "test_data_opened": False,
        "validation_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    saved = sealed_write(
        args.output, audit(args.campaign, args.boundary, args.tolerance)
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "artifact_sha256": saved["artifact_sha256"],
                "rows": [
                    {k: r[k] for k in ("slot", "arm", "status")} for r in saved["rows"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
