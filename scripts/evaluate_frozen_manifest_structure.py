"""Evaluate private structural validity for artifacts in a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = {
        artifact.artifact_id: artifact
        for line in args.candidate_pool.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for artifact in (CandidateArtifact.model_validate_json(line),)
    }
    selected_ids = {
        str(row["artifact_id"])
        for row in json.loads(args.manifest.read_text(encoding="utf-8"))
    }
    specs = {
        spec.benchmark_id: spec
        for path in args.config_root.glob("*.json")
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }
    rows = []
    for artifact_id in sorted(selected_ids):
        artifact = pool[artifact_id]
        result = evaluate_mechanisms(
            artifact.candidate, specs[artifact.benchmark_id]
        )
        rows.append(
            {
                "artifact_id": artifact_id,
                "benchmark": artifact.benchmark_id,
                "tier": artifact.tier,
                "seed": artifact.seed,
                "structural_validity": result.structural_validity,
                "mechanism_coverage": result.mechanism_coverage,
                "manual_review_required": result.manual_review_required,
                "source": artifact.source_checkpoint,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
