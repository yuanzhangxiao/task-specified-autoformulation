"""Compare ratio and scaled weighted-sum rankings without test leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.objectives import (
    compare_ratio_and_weighted_sum,
    ratio_objective,
    weighted_sum_objective,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument(
        "--lambda-multipliers",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    records = _read_pool(args.candidate_pool)
    grouped: dict[tuple[str, str], list[CandidateArtifact]] = defaultdict(list)
    for item in records:
        if item.judge_score is not None:
            grouped[(item.benchmark_id, item.tier)].append(item)
    args.output_root.mkdir(parents=True, exist_ok=True)
    comparisons = []
    ranking_rows = []
    selected_rows = []
    for (benchmark, tier), candidates in sorted(grouped.items()):
        if len(candidates) < 2:
            continue
        median_loss = sorted(item.validation_mse for item in candidates)[
            len(candidates) // 2
        ]
        for multiplier in args.lambda_multipliers:
            comparison = compare_ratio_and_weighted_sum(
                candidates,
                lambda_multiplier=multiplier,
                epsilon=args.epsilon,
                top_k=args.top_k,
            )
            comparisons.append(
                {
                    "benchmark_id": benchmark,
                    "tier": tier,
                    **comparison.model_dump(mode="json"),
                }
            )
            for item in candidates:
                ratio = ratio_objective(
                    item.validation_mse, float(item.judge_score), args.epsilon
                )
                weighted = weighted_sum_objective(
                    item.validation_mse,
                    float(item.judge_score),
                    comparison.lambda_value,
                    args.epsilon,
                )
                ranking_rows.append(
                    {
                        "benchmark_id": benchmark,
                        "tier": tier,
                        "lambda_multiplier": multiplier,
                        "lambda_value": comparison.lambda_value,
                        "artifact_id": item.artifact_id,
                        "validation_mse": item.validation_mse,
                        "judge_score": item.judge_score,
                        "ratio_objective": ratio,
                        "weighted_sum_objective": weighted,
                        "complexity_terms": item.term_count,
                    }
                )
            selected_rows.extend(
                (
                    {
                        "benchmark_id": benchmark,
                        "tier": tier,
                        "lambda_multiplier": multiplier,
                        "policy": "ratio",
                        "artifact_id": comparison.ratio_selected_artifact_id,
                    },
                    {
                        "benchmark_id": benchmark,
                        "tier": tier,
                        "lambda_multiplier": multiplier,
                        "policy": "weighted_sum",
                        "artifact_id": comparison.weighted_sum_selected_artifact_id,
                    },
                )
            )
        assert median_loss >= 0.0
    _write_csv(args.output_root / "ratio_map_candidate_ranking.csv", ranking_rows)
    _write_csv(args.output_root / "ratio_map_selected_models.csv", selected_rows)
    (args.output_root / "ratio_map_rank_correlations.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # This manifest freezes every development-only selection before any separate
    # test-evaluation command is permitted to load test data.
    (args.output_root / "frozen_selection_manifest.json").write_text(
        json.dumps(selected_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_pool(path: Path) -> tuple[CandidateArtifact, ...]:
    return tuple(
        CandidateArtifact.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
