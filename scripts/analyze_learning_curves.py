"""Extract checkpoint learning curves and fit/structure Pareto records."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from autoformalism.rebuttal.artifacts import CandidateArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--mechanism-metrics", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    metrics = _read_metrics(args.mechanism_metrics)
    records = [
        CandidateArtifact.model_validate_json(line)
        for line in args.candidate_pool.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[
        tuple[str, str, int, str], list[CandidateArtifact]
    ] = defaultdict(list)
    for item in records:
        grouped[
            (item.benchmark_id, item.tier, item.seed, item.run_directory)
        ].append(item)
    rows = []
    pareto = []
    for key, values in sorted(grouped.items()):
        best_mse = float("inf")
        best_judge = float("-inf")
        best_validity = 0.0
        for item in sorted(values, key=lambda record: record.round_index):
            best_mse = min(best_mse, item.validation_mse)
            if item.judge_score is not None:
                best_judge = max(best_judge, item.judge_score)
            item_metrics = metrics.get(item.artifact_id, {})
            structural = float(item_metrics.get("structural_validity", 0.0))
            coverage = float(item_metrics.get("mechanism_coverage", 0.0))
            best_validity = max(best_validity, structural)
            row = {
                "benchmark_id": key[0],
                "tier": key[1],
                "seed": key[2],
                "run_directory": key[3],
                "round_index": item.round_index,
                "artifact_id": item.artifact_id,
                "validation_mse": item.validation_mse,
                "best_validation_mse": best_mse,
                "judge_score": item.judge_score,
                "best_judge_score": None if best_judge == float("-inf") else best_judge,
                "mechanism_coverage": coverage,
                "structural_validity": structural,
                "best_structural_validity": best_validity,
                "term_count": item.term_count,
            }
            rows.append(row)
            pareto.append(
                {
                    "artifact_id": item.artifact_id,
                    "benchmark_id": key[0],
                    "tier": key[1],
                    "seed": key[2],
                    "validation_mse": item.validation_mse,
                    "structural_validity": structural,
                    "mechanism_coverage": coverage,
                    "term_count": item.term_count,
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write(args.output_root / "learning_curves.csv", rows)
    _write(args.output_root / "pareto_fit_mechanism.csv", pareto)


def _read_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["artifact_id"]: row for row in csv.DictReader(handle)}


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
