#!/usr/bin/env python3
"""Synthesize frozen Phase A3 intervention batches without refitting models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from autoformalism.rebuttal.statistics import paired_log_comparison


def _load_rows(inputs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in inputs:
        suite = root.name
        for row in json.loads((root / "evaluations.json").read_text(encoding="utf-8")):
            item = dict(row)
            item["suite"] = suite
            _, method, seed = item["model_label"].split(":", 2)
            item["method"] = method
            item["seed"] = int(seed)
            rows.append(item)
    return rows


def method_rollups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate log-scale errors while retaining benchmark strata."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["benchmark_id"], row["method"])].append(row)
    output = []
    for (suite, benchmark, method), items in sorted(groups.items()):
        successful = [item for item in items if item["success"]]
        nmses = np.asarray([item["target_nmse"] for item in successful], dtype=float)
        shapes = [
            float(item["response_shape_correlation"])
            for item in successful
            if item["response_shape_correlation"] is not None
        ]
        timings = [
            float(item["peak_timing_error_fraction"])
            for item in successful
            if item["peak_timing_error_fraction"] is not None
        ]
        hidden = [
            float(item["hidden_alignment_nmse"])
            for item in successful
            if item["hidden_alignment_nmse"] is not None
        ]
        output.append(
            {
                "suite": suite,
                "benchmark_id": benchmark,
                "method": method,
                "evaluations": len(items),
                "success_rate": len(successful) / len(items),
                "target_nmse_geometric_mean": float(
                    10 ** np.mean(np.log10(nmses))
                ),
                "target_nmse_median": float(np.median(nmses)),
                "response_shape_correlation_mean": mean(shapes) if shapes else None,
                "peak_timing_error_fraction_mean": mean(timings) if timings else None,
                "hidden_alignment_nmse_mean": mean(hidden) if hidden else None,
                "hidden_alignment_nmse_sd": stdev(hidden)
                if len(hidden) > 1
                else 0.0
                if hidden
                else None,
            }
        )
    return output


def paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare Full against each alternative on identical suite/case/seed cells."""

    lookup = {
        (
            row["suite"],
            row["benchmark_id"],
            row["case_id"],
            row["seed"],
            row["method"],
        ): row
        for row in rows
        if row["success"]
    }
    alternatives = sorted({row["method"] for row in rows}.difference({"Full"}))
    output = []
    strata = sorted({(row["suite"], row["benchmark_id"]) for row in rows})
    for suite, benchmark in strata:
        for alternative in alternatives:
            pairs = []
            for key, full in lookup.items():
                row_suite, row_benchmark, case, seed, method = key
                if (row_suite, row_benchmark, method) != (suite, benchmark, "Full"):
                    continue
                other = lookup.get((suite, benchmark, case, seed, alternative))
                if other is not None:
                    pairs.append(
                        (float(full["target_nmse"]), float(other["target_nmse"]))
                    )
            if not pairs:
                continue
            comparison = paired_log_comparison(
                np.asarray([pair[0] for pair in pairs]),
                np.asarray([pair[1] for pair in pairs]),
                permutation_samples=10_000,
                random_seed=20260806,
            )
            output.append(
                {
                    "suite": suite,
                    "benchmark_id": benchmark,
                    "alternative": alternative,
                    **comparison.model_dump(),
                }
            )
    return output


def _sample(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4g}"


def render_markdown(
    rollups: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> str:
    lines = [
        "# Phase A3 intervention synthesis",
        "",
        "Geometric means summarize heterogeneous positive NMSEs without allowing "
        "a few catastrophic cells to dominate arithmetically. Paired comparisons "
        "use only identical suite/case/seed cells and report Full/alternative ratios.",
        "",
        "## Method rollups",
        "",
        "| Suite | Benchmark | Method | Geometric NMSE | Median NMSE | Shape r | "
        "Peak error | Hidden NMSE | Success |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rollups:
        lines.append(
            f"| {row['suite']} | {row['benchmark_id']} | {row['method']} | "
            f"{_sample(row['target_nmse_geometric_mean'])} | "
            f"{_sample(row['target_nmse_median'])} | "
            f"{_sample(row['response_shape_correlation_mean'])} | "
            f"{_sample(row['peak_timing_error_fraction_mean'])} | "
            f"{_sample(row['hidden_alignment_nmse_mean'])} | "
            f"{row['success_rate']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Paired Full-method comparisons",
            "",
            "A ratio below one favors Full; the confidence interval is a "
            "nonparametric bootstrap over matched evaluation cells.",
            "",
            "| Suite | Benchmark | Alternative | Pairs | Full win rate | "
            "Geometric ratio [95% CI] | Sign-flip p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['suite']} | {row['benchmark_id']} | {row['alternative']} | "
            f"{row['pair_count']} | {row['first_win_rate']:.1%} | "
            f"{row['geometric_mean_ratio']:.3g} "
            f"[{row['geometric_ratio_ci_low']:.3g}, "
            f"{row['geometric_ratio_ci_high']:.3g}] | "
            f"{row['sign_flip_p_value']:.3g} |"
        )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_rows(args.input)
    rollups = method_rollups(rows)
    comparisons = paired_comparisons(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_root / "method_rollups.csv", rollups)
    _write_csv(args.output_root / "paired_comparisons.csv", comparisons)
    (args.output_root / "summary.md").write_text(
        render_markdown(rollups, comparisons), encoding="utf-8"
    )
    print(
        f"evaluations={len(rows)} rollups={len(rollups)} "
        f"paired_comparisons={len(comparisons)}"
    )


if __name__ == "__main__":
    main()
