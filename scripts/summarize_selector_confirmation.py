"""Summarize multidimensional outcomes of frozen selector policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoformalism.rebuttal.statistics import paired_log_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--authoritative-runs", type=Path, required=True)
    parser.add_argument("--production-structure", type=Path, required=True)
    parser.add_argument("--production-hidden", type=Path, required=True)
    parser.add_argument("--alternative-runs", type=Path, required=True)
    parser.add_argument("--alternative-structure", type=Path, required=True)
    parser.add_argument("--alternative-hidden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    details = assemble_policy_outcomes(
        selections=pd.read_csv(args.selections),
        authoritative=pd.read_csv(args.authoritative_runs),
        production_structure=pd.read_csv(args.production_structure),
        production_hidden=pd.read_csv(args.production_hidden),
        alternative_runs=pd.read_csv(args.alternative_runs),
        alternative_structure=pd.read_csv(args.alternative_structure),
        alternative_hidden=pd.read_csv(args.alternative_hidden),
    )
    summary, paired = summarize_policy_outcomes(details)
    details.to_csv(args.output_root / "selector_confirmation_details.csv", index=False)
    summary.to_csv(args.output_root / "selector_confirmation_summary.csv", index=False)
    paired.to_csv(
        args.output_root / "selector_confirmation_paired.csv", index=False
    )
    (args.output_root / "selector_confirmation_report.md").write_text(
        _report(summary, paired), encoding="utf-8"
    )


def assemble_policy_outcomes(
    *,
    selections: pd.DataFrame,
    authoritative: pd.DataFrame,
    production_structure: pd.DataFrame,
    production_hidden: pd.DataFrame,
    alternative_runs: pd.DataFrame,
    alternative_structure: pd.DataFrame,
    alternative_hidden: pd.DataFrame,
) -> pd.DataFrame:
    """Join frozen choices to post-freeze outcomes without scalarizing them."""
    keys = ["benchmark", "tier", "seed"]
    production = authoritative[authoritative.method == "full"][[
        *keys, "test_mse"
    ]].rename(columns={"test_mse": "production_test_mse"})
    production = production.merge(
        production_structure[production_structure.method == "full"][[
            *keys, "structural_validity"
        ]].rename(columns={"structural_validity": "production_structural_validity"}),
        on=keys,
        how="left",
    ).merge(
        production_hidden[production_hidden.method == "full"][[
            *keys, "hidden_mse"
        ]].rename(columns={"hidden_mse": "production_hidden_mse"}),
        on=keys,
        how="left",
    )
    alternatives = alternative_runs[[
        "artifact_id", "status", "test_mse"
    ]].rename(
        columns={"status": "alternative_status", "test_mse": "alternative_test_mse"}
    )
    structure_by_source = alternative_structure.copy()
    if "artifact_id" not in structure_by_source:
        structure_by_source["artifact_id"] = structure_by_source.source.map(
            lambda value: Path(value).parent.name
        )
    hidden_by_source = alternative_hidden.copy()
    hidden_by_source["artifact_id"] = hidden_by_source.source.map(
        lambda value: Path(value).parent.name
    )
    alternatives = alternatives.merge(
        structure_by_source[["artifact_id", "structural_validity"]],
        on="artifact_id",
        how="left",
    ).merge(
        hidden_by_source[["artifact_id", "hidden_mse"]],
        on="artifact_id",
        how="left",
    )
    frame = selections.merge(production, on=keys, how="left").merge(
        alternatives, on="artifact_id", how="left"
    )
    # Use an explicit baseline mapping rather than relying on row order.
    baseline_ids = frame[frame.confirmation_policy == "validation_only"][[
        "run_directory", "artifact_id"
    ]].rename(columns={"artifact_id": "production_artifact_id"})
    frame = frame.merge(baseline_ids, on="run_directory", how="left")
    unchanged = frame.artifact_id == frame.production_artifact_id
    frame["changed_selection"] = ~unchanged
    frame["completion"] = np.where(
        unchanged,
        frame.production_test_mse.notna(),
        frame.alternative_status.eq("complete"),
    )
    frame["test_mse"] = np.where(
        unchanged, frame.production_test_mse, frame.alternative_test_mse
    )
    frame["structural_validity"] = np.where(
        unchanged,
        frame.production_structural_validity,
        frame.structural_validity,
    )
    frame["hidden_mse"] = np.where(
        unchanged, frame.production_hidden_mse, frame.hidden_mse
    )
    return frame


def summarize_policy_outcomes(
    details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report marginal vectors and paired predictive comparisons."""
    summaries = []
    for policy, group in details.groupby("confirmation_policy", sort=False):
        valid_test = group.test_mse.notna() & np.isfinite(group.test_mse)
        summaries.append(
            {
                "policy": policy,
                "runs": len(group),
                "completed": int(group.completion.sum()),
                "completion_rate": float(group.completion.mean()),
                "changed_selections": int(group.changed_selection.sum()),
                "changed_completed": int(
                    (group.changed_selection & group.completion).sum()
                ),
                "test_mse_geometric_mean": _geometric_mean(
                    group.loc[valid_test, "test_mse"]
                ),
                "structural_validity_mean": float(group.structural_validity.mean()),
                "hidden_mse_geometric_mean": _geometric_mean(group.hidden_mse.dropna()),
                "term_count_mean": float(group.term_count.mean()),
            }
        )
    baseline = details[details.confirmation_policy == "validation_only"][[
        "run_directory", "test_mse", "structural_validity", "hidden_mse", "term_count"
    ]].rename(columns={
        "test_mse": "baseline_test_mse",
        "structural_validity": "baseline_structural_validity",
        "hidden_mse": "baseline_hidden_mse",
        "term_count": "baseline_term_count",
    })
    paired_rows = []
    for policy in ("weighted_j0.5_s0.1", "epsilon_d0.2"):
        selected = details[details.confirmation_policy == policy].merge(
            baseline, on="run_directory", how="inner"
        )
        valid = (
            selected.test_mse.notna()
            & selected.baseline_test_mse.notna()
            & np.isfinite(selected.test_mse)
            & np.isfinite(selected.baseline_test_mse)
            & (selected.test_mse > 0)
            & (selected.baseline_test_mse > 0)
        )
        comparison = paired_log_comparison(
            selected.loc[valid, "test_mse"].to_numpy(),
            selected.loc[valid, "baseline_test_mse"].to_numpy(),
            permutation_samples=20_000,
        )
        dominance = _dominance_counts(selected[valid])
        changed_valid = valid & selected.changed_selection
        changed_comparison = (
            paired_log_comparison(
                selected.loc[changed_valid, "test_mse"].to_numpy(),
                selected.loc[changed_valid, "baseline_test_mse"].to_numpy(),
                permutation_samples=20_000,
            )
            if changed_valid.any()
            else None
        )
        paired_rows.append(
            {
                "policy": policy,
                "failed_or_missing": int((~valid).sum()),
                **comparison.model_dump(mode="json"),
                "changed_successful_pairs": (
                    changed_comparison.pair_count if changed_comparison else 0
                ),
                "changed_first_win_rate": (
                    changed_comparison.first_win_rate
                    if changed_comparison
                    else float("nan")
                ),
                "changed_geometric_mean_ratio": (
                    changed_comparison.geometric_mean_ratio
                    if changed_comparison
                    else float("nan")
                ),
                "changed_geometric_ratio_ci_low": (
                    changed_comparison.geometric_ratio_ci_low
                    if changed_comparison
                    else float("nan")
                ),
                "changed_geometric_ratio_ci_high": (
                    changed_comparison.geometric_ratio_ci_high
                    if changed_comparison
                    else float("nan")
                ),
                **dominance,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(paired_rows)


def _dominance_counts(frame: pd.DataFrame) -> dict[str, int]:
    alternative = 0
    production = 0
    incomparable = 0
    for row in frame.itertuples(index=False):
        pairs = [
            (row.test_mse, row.baseline_test_mse, "lower"),
            (row.structural_validity, row.baseline_structural_validity, "higher"),
            (row.term_count, row.baseline_term_count, "lower"),
        ]
        if not pd.isna(row.hidden_mse) and not pd.isna(row.baseline_hidden_mse):
            pairs.append((row.hidden_mse, row.baseline_hidden_mse, "lower"))
        alt_no_worse = all(
            first <= second if direction == "lower" else first >= second
            for first, second, direction in pairs
            if not pd.isna(first) and not pd.isna(second)
        )
        base_no_worse = all(
            second <= first if direction == "lower" else second >= first
            for first, second, direction in pairs
            if not pd.isna(first) and not pd.isna(second)
        )
        alt_strict = any(
            first < second if direction == "lower" else first > second
            for first, second, direction in pairs
            if not pd.isna(first) and not pd.isna(second)
        )
        base_strict = any(
            second < first if direction == "lower" else second > first
            for first, second, direction in pairs
            if not pd.isna(first) and not pd.isna(second)
        )
        if alt_no_worse and alt_strict:
            alternative += 1
        elif base_no_worse and base_strict:
            production += 1
        else:
            incomparable += 1
    return {
        "alternative_pareto_dominates": alternative,
        "production_pareto_dominates": production,
        "pareto_incomparable_or_equal": incomparable,
    }


def _geometric_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric) & (numeric > 0)]
    return float(np.exp(np.mean(np.log(numeric)))) if len(numeric) else float("nan")


def _report(summary: pd.DataFrame, paired: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Frozen selector multidimensional confirmation",
            "",
            "Selections and hyperparameters were frozen before these test and "
            "private-reference outcomes were joined. Metrics remain separate; "
            "no post-selection scalar quality score is constructed.",
            "",
            "## Marginal outcomes",
            "",
            _markdown(summary),
            "",
            "## Paired with production validation-only selection",
            "",
            _markdown(paired),
            "",
            "A test-MSE ratio below one favors the alternative. Pareto dominance "
            "uses test MSE, structural validity, term count, and hidden MSE only "
            "when hidden MSE exists for both choices.",
            "",
        ]
    )


def _markdown(frame: pd.DataFrame) -> str:
    headers = [str(column).replace("_", " ") for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            f"{value:.4g}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
