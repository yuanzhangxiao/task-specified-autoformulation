"""Extract checkpoint learning curves and fit/structure Pareto records."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from autoformalism.rebuttal.artifacts import CandidateArtifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--mechanism-metrics", type=Path)
    parser.add_argument(
        "--target-scales",
        type=Path,
        help="CSV with benchmark, tier, and target_scale columns.",
    )
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
    if args.target_scales is not None:
        scales = _read_target_scales(args.target_scales)
        _write(
            args.output_root / "learning_curve_iteration_summary.csv",
            _iteration_summary(records, scales),
        )


def _read_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["artifact_id"]: row for row in csv.DictReader(handle)}


def _read_target_scales(path: Path) -> dict[tuple[str, str], float]:
    """Read one target normalization scale for each benchmark-tier cell."""
    values: dict[tuple[str, str], set[float]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("target_scale"):
                continue
            benchmark = row.get("benchmark") or row.get("benchmark_id")
            if benchmark is None:
                raise ValueError("target-scale CSV lacks a benchmark column")
            values[(benchmark, row["tier"])].add(float(row["target_scale"]))
    ambiguous = {key: item for key, item in values.items() if len(item) != 1}
    if ambiguous:
        raise ValueError(f"expected one target scale per cell: {ambiguous}")
    return {key: next(iter(item)) for key, item in values.items()}


def _iteration_summary(
    records: list[CandidateArtifact],
    target_scales: dict[tuple[str, str], float],
    *,
    iterations: int = 8,
) -> list[dict[str, object]]:
    """Aggregate cumulative development curves across independent runs."""
    by_run: dict[
        tuple[str, str, str, int, str], list[CandidateArtifact]
    ] = defaultdict(list)
    for item in records:
        method = "full" if item.use_judge else "nojudge"
        by_run[
            (method, item.benchmark_id, item.tier, item.seed, item.run_directory)
        ].append(item)

    group_sizes: dict[tuple[str, str, str], int] = defaultdict(int)
    curves: dict[
        tuple[str, str, str, int], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for (method, benchmark, tier, _seed, _directory), values in by_run.items():
        group_sizes[(method, benchmark, tier)] += 1
        scale = target_scales[(benchmark, tier)]
        ordered = sorted(values, key=lambda item: item.round_index)
        for iteration in range(iterations):
            available = [item for item in ordered if item.round_index <= iteration]
            if not available:
                continue
            best_mse = min(item.validation_mse for item in available)
            bucket = curves[(method, benchmark, tier, iteration)]
            bucket["validation_nmse"].append(best_mse)
            bucket["validation_raw_mse"].append(best_mse * scale**2)
            scores = [
                float(item.judge_score)
                for item in available
                if item.judge_score is not None
            ]
            if scores:
                bucket["judge_score"].append(max(scores))

    output: list[dict[str, object]] = []
    for (method, benchmark, tier, iteration), metrics in sorted(curves.items()):
        raw = metrics["validation_raw_mse"]
        normalized = metrics["validation_nmse"]
        judges = metrics["judge_score"]
        output.append(
            {
                "benchmark_id": benchmark,
                "tier": tier,
                "method": method,
                "iteration": iteration + 1,
                "contributing_runs": len(raw),
                "total_runs": group_sizes[(method, benchmark, tier)],
                "validation_raw_mse_mean": mean(raw),
                "validation_raw_mse_sd": stdev(raw) if len(raw) > 1 else 0.0,
                "validation_nmse_mean": mean(normalized),
                "validation_nmse_sd": (
                    stdev(normalized) if len(normalized) > 1 else 0.0
                ),
                "judge_score_mean": mean(judges) if judges else None,
                "judge_score_sd": stdev(judges) if len(judges) > 1 else 0.0,
            }
        )
    return output


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
