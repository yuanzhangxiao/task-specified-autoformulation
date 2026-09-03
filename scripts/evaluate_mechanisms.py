"""Evaluate public graph-based mechanism specifications on candidate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    specifications = {
        (spec.benchmark_id, spec.tier): spec
        for path in sorted(args.config_root.glob("*.json"))
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }
    records = (
        CandidateArtifact.model_validate_json(line)
        for line in args.candidate_pool.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    rows: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    for item in records:
        spec = specifications.get((item.benchmark_id, item.tier))
        if spec is None:
            continue
        result = evaluate_mechanisms(item.candidate, spec)
        rows.append(
            {
                "artifact_id": item.artifact_id,
                "benchmark_id": item.benchmark_id,
                "tier": item.tier,
                "seed": item.seed,
                "mechanism_coverage": result.mechanism_coverage,
                "graph_mechanism_compliance": (
                    result.graph_mechanism_compliance
                ),
                "graph_mechanism_compliance_complete": (
                    result.graph_mechanism_compliance_complete
                ),
                "mechanism_annotation_compliance": (
                    result.mechanism_annotation_compliance
                ),
                "mechanism_annotation_compliance_complete": (
                    result.mechanism_annotation_compliance_complete
                ),
                # Retained as aliases for existing consumers.
                "mechanism_compliance": result.mechanism_compliance,
                "mechanism_compliance_complete": (
                    result.mechanism_compliance_complete
                ),
                "structural_validity": result.structural_validity,
                "manual_review_required": result.manual_review_required,
            }
        )
        for predicate in result.predicates:
            if predicate.status == "ambiguous":
                reviews.append(
                    {
                        "artifact_id": item.artifact_id,
                        **predicate.model_dump(mode="json"),
                    }
                )
    args.output_root.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.output_root / "mechanism_metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_root / "mechanism_eval_manual_review.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviews),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
