"""Evaluate structural validity of frozen structured-method selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel


def _candidate(path: Path, method: str) -> CandidateModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method in {"full", "nojudge"}:
        return CandidateModel.model_validate(payload["frozen"]["candidate"])
    checkpoint = json.loads(
        (path.parent / "d3_checkpoint.json").read_text(encoding="utf-8")
    )
    generation = int(payload["selected_hyperparameters"]["selected_generation"])
    return CandidateModel.model_validate(checkpoint["records"][generation]["candidate"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = {
        spec.benchmark_id: spec
        for path in args.config_root.glob("*.json")
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }
    runs = pd.read_csv(args.runs)
    runs = runs[
        runs.method.isin(("full", "nojudge", "d3_native_no_tools"))
        & runs.test_mse.notna()
    ]
    rows = []
    for row in runs.itertuples(index=False):
        spec = specs.get(row.benchmark)
        if spec is None:
            continue
        result = evaluate_mechanisms(_candidate(Path(row.source), row.method), spec)
        rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "tier": row.tier,
                "seed": row.seed,
                "structural_validity": result.structural_validity,
                "source": row.source,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
