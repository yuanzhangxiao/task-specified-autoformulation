"""Build compact baseline, family, objective, and diagnostic rebuttal tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = (
    "sindy",
    "d3_native_no_tools",
    "llm_feature_sindy",
    "nojudge",
    "full",
)
DISPLAY = {
    "sindy": "SINDy",
    "d3_native_no_tools": "D3",
    "llm_feature_sindy": "LLM-feature-SINDy",
    "nojudge": "No-judge",
    "full": "Ours",
}
BENCHMARKS = (
    "original_b1",
    "perturbed_b1",
    "obfuscated_original_case01",
    "obfuscated_perturbed_case01",
    "benchmark5",
    "benchmark6",
)


def _summary(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[np.isfinite(clean) & (clean < 1e11)]
    if clean.empty:
        return "N/A"
    if len(clean) == 1:
        return f"{clean.iloc[0]:.4g}"
    return f"{clean.mean():.4g} ± {clean.std(ddof=1):.3g}"


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _baseline_table(root: Path, output: Path) -> None:
    runs = pd.read_csv(root / "authoritative_runs.csv")
    selected = pd.read_csv(root / "selected_structural_validity.csv")
    llm = pd.read_csv(root / "llm_feature_structural_validity.csv")
    structural = pd.concat((selected, llm), ignore_index=True)
    hidden = pd.read_csv(root / "selected_hidden_dynamics.csv")
    rows = []
    csv_rows = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            subset = runs[
                (runs.benchmark == benchmark)
                & (runs.tier == "hard")
                & (runs.method == method)
            ]
            structure = structural[
                (structural.benchmark == benchmark)
                & (structural.tier == "hard")
                & (structural.method == method)
            ]
            hidden_group = hidden[
                (hidden.benchmark == benchmark)
                & (hidden.tier == "hard")
                & (hidden.method == method)
            ]
            spec = "N/A" if method == "sindy" else _summary(
                structure.structural_validity
            )
            row = (
                benchmark,
                DISPLAY[method],
                _summary(subset.test_mse),
                spec,
                _summary(hidden_group.hidden_mse),
                _summary(subset.term_count),
            )
            rows.append(row)
            csv_rows.append(
                dict(
                    zip(
                        (
                            "benchmark",
                            "method",
                            "target_nmse",
                            "structural_validity",
                            "hidden_nmse",
                            "terms",
                        ),
                        row,
                        strict=True,
                    )
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(output / "baseline_ablation_hard.csv", index=False)
    (output / "baseline_ablation_hard.md").write_text(
        "# Adapted baselines and ablations — hard tier\n\n"
        + _table(
            (
                "Benchmark",
                "Method",
                "Target NMSE ↓",
                "Structural validity ↑",
                "Hidden NMSE ↓",
                "Terms ↓",
            ),
            rows,
        )
        + "\n",
        encoding="utf-8",
    )


def _family_table(root: Path, output: Path) -> None:
    runs = pd.read_csv(root / "authoritative_runs.csv")
    family = runs[runs.method.str.contains("_proposer__", regex=False, na=False)]
    rows = []
    for (arm, benchmark), group in family.groupby(["method", "benchmark"]):
        proposer, judge = arm.split("_proposer__")
        rows.append(
            (
                proposer.upper(),
                judge.replace("_judge", "").upper(),
                benchmark,
                _summary(group.test_mse),
                _summary(group.structural_validity),
                _summary(group.judge_score),
                f"{group.test_mse.count()}/5",
            )
        )
    (output / "llm_family_by_benchmark.md").write_text(
        "# LLM family study\n\n"
        + _table(
            (
                "Proposer",
                "Judge",
                "Benchmark",
                "Target NMSE ↓",
                "Structural validity ↑",
                "Judge score ↑",
                "Complete",
            ),
            rows,
        )
        + "\n",
        encoding="utf-8",
    )


def _objective_table(root: Path, output: Path) -> None:
    frame = pd.read_csv(root / "objective_summary.csv")
    rows = [
        (
            f"{row.lambda_multiplier:g}",
            str(int(row.contexts)),
            f"{row.same_top1_rate:.3f}",
            f"{row.median_spearman:.3f}",
            f"{row.median_kendall:.3f}",
            f"{row.mean_top5_overlap_fraction:.3f}",
        )
        for row in frame.itertuples(index=False)
    ]
    (output / "ratio_vs_weighted_sum.md").write_text(
        "# Ratio versus weighted-sum objective\n\n"
        + _table(
            (
                "λ multiplier",
                "Contexts",
                "Same top-1 ↑",
                "Median Spearman ↑",
                "Median Kendall ↑",
                "Top-5 overlap ↑",
            ),
            rows,
        )
        + "\n",
        encoding="utf-8",
    )


def _diagnostic_tables(root: Path, output: Path) -> None:
    learning = pd.read_csv(root / "learning_curve_summary.csv")
    learning["method"] = np.where(
        learning.run_directory.str.contains("noj-"), "No-judge", "Ours"
    )
    hard = learning[learning.tier == "hard"]
    curve_rows = []
    for (benchmark, method), group in hard.groupby(["benchmark_id", "method"]):
        curve_rows.append(
            (
                benchmark,
                method,
                str(len(group)),
                _summary(group.valid_rounds),
                _summary(group.relative_improvement),
            )
        )
    stability = pd.read_csv(root / "structural_stability_summary.csv")
    stability = stability[stability.tier == "hard"]
    stability_rows = [
        (
            row.benchmark_id,
            str(int(row.pairs)),
            f"{row.mean_edge_jaccard:.3f}",
            f"{row.median_edge_jaccard:.3f}",
            f"{row.mean_term_jaccard:.3f}",
            f"{row.median_term_jaccard:.3f}",
        )
        for row in stability.itertuples(index=False)
    ]
    (output / "learning_and_stability.md").write_text(
        "# Learning curves and structural stability\n\n"
        "## Hard-tier checkpoint improvement\n\n"
        + _table(
            ("Benchmark", "Method", "Runs", "Valid rounds", "Relative improvement"),
            curve_rows,
        )
        + "\n\n## Hard-tier structural stability\n\n"
        + _table(
            (
                "Benchmark",
                "Pairs",
                "Mean edge Jaccard",
                "Median edge Jaccard",
                "Mean term Jaccard",
                "Median term Jaccard",
            ),
            stability_rows,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    _baseline_table(args.analysis_root, args.output_root)
    _family_table(args.analysis_root, args.output_root)
    _objective_table(args.analysis_root, args.output_root)
    _diagnostic_tables(args.analysis_root, args.output_root)


if __name__ == "__main__":
    main()
