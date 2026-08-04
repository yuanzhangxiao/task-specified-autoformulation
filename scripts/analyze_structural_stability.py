"""Compare final candidate topology and target terms across seeds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.structure import pairwise_similarities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    records = [
        CandidateArtifact.model_validate_json(line)
        for line in args.candidate_pool.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # One development-selected structure per run; test metrics never participate.
    by_run: dict[tuple[str, str, int], list[CandidateArtifact]] = defaultdict(list)
    for item in records:
        by_run[(item.benchmark_id, item.tier, item.seed)].append(item)
    finals = {
        key: min(values, key=lambda item: (item.validation_mse, item.term_count))
        for key, values in by_run.items()
    }
    rows = []
    by_context: dict[tuple[str, str], list[CandidateArtifact]] = defaultdict(list)
    for (benchmark, tier, _), item in finals.items():
        by_context[(benchmark, tier)].append(item)
    for (benchmark, tier), values in sorted(by_context.items()):
        items = tuple(
            (item.artifact_id, item.candidate)
            for item in sorted(values, key=lambda record: record.seed)
        )
        for similarity in pairwise_similarities(items):
            rows.append(
                {
                    "benchmark_id": benchmark,
                    "tier": tier,
                    **similarity.model_dump(mode="json"),
                }
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "structural_similarity_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = tuple(rows[0]) if rows else (
            "benchmark_id",
            "tier",
            "left_id",
            "right_id",
            "edge_jaccard",
            "term_jaccard",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
