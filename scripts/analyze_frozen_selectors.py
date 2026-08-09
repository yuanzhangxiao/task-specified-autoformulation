"""Compare leak-free selectors on a frozen development-only candidate pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.objectives import select_frozen_candidate

JUDGE_WEIGHTS = (0.1, 0.25, 0.5, 1.0)
SPARSITY_WEIGHTS = (0.0, 0.05, 0.1, 0.25)
EPSILON_FRACTIONS = (0.01, 0.05, 0.1, 0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = _read_pool(args.candidate_pool)
    groups = _group_candidates(records)
    selections = _select_grid(groups)
    summary = _summarize(selections)
    lobo_folds = _leave_one_benchmark_out(summary, selections)
    categories = _judge_category_audit(records)

    selections.to_csv(args.output_root / "frozen_selector_selections.csv", index=False)
    summary.to_csv(args.output_root / "frozen_selector_sensitivity.csv", index=False)
    lobo_folds.to_csv(
        args.output_root / "frozen_selector_lobo_development.csv", index=False
    )
    categories.to_csv(args.output_root / "judge_category_audit.csv", index=False)
    manifest = {
        "candidate_pool": str(args.candidate_pool.resolve()),
        "candidate_artifacts": len(records),
        "judged_run_pools": len(groups),
        "uses_test_metrics": False,
        "uses_private_mechanism_references": False,
        "judge_weights": JUDGE_WEIGHTS,
        "sparsity_weights": SPARSITY_WEIGHTS,
        "epsilon_fractions": EPSILON_FRACTIONS,
        "lobo_rule": (
            "On non-held-out benchmarks, require median validation ratio <=1.05 "
            "and 90th percentile <=1.25; maximize mean judge gain, then minimize "
            "mean term change and validation ratio."
        ),
    }
    (args.output_root / "frozen_selector_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "frozen_selector_report.md").write_text(
        _report(summary, lobo_folds, categories, len(groups)), encoding="utf-8"
    )


def _read_pool(path: Path) -> tuple[CandidateArtifact, ...]:
    return tuple(
        CandidateArtifact.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _group_candidates(
    records: tuple[CandidateArtifact, ...],
) -> dict[str, tuple[CandidateArtifact, ...]]:
    grouped: dict[str, list[CandidateArtifact]] = defaultdict(list)
    for item in records:
        if item.judge_score is not None:
            grouped[item.run_directory].append(item)
    result = {}
    for run_directory, candidates in grouped.items():
        unique: dict[str, CandidateArtifact] = {}
        for item in candidates:
            current = unique.get(item.structural_hash)
            if current is None or (
                item.validation_mse,
                -float(item.judge_score),
                item.artifact_id,
            ) < (
                current.validation_mse,
                -float(current.judge_score),
                current.artifact_id,
            ):
                unique[item.structural_hash] = item
        result[run_directory] = tuple(
            sorted(unique.values(), key=lambda item: item.artifact_id)
        )
    return result


def _configurations() -> tuple[dict[str, float | str], ...]:
    configurations: list[dict[str, float | str]] = [
        {
            "config_id": "validation_only",
            "policy": "validation_only",
            "judge_weight": 0.0,
            "sparsity_weight": 0.0,
            "epsilon_fraction": 0.0,
        }
    ]
    for policy in ("normalized_weighted_sum", "pareto_compromise"):
        for judge_weight in JUDGE_WEIGHTS:
            for sparsity_weight in SPARSITY_WEIGHTS:
                configurations.append(
                    {
                        "config_id": (
                            f"{policy}__j{judge_weight:g}__s{sparsity_weight:g}"
                        ),
                        "policy": policy,
                        "judge_weight": judge_weight,
                        "sparsity_weight": sparsity_weight,
                        "epsilon_fraction": 0.0,
                    }
                )
    for fraction in EPSILON_FRACTIONS:
        configurations.append(
            {
                "config_id": f"epsilon_constrained__d{fraction:g}",
                "policy": "epsilon_constrained",
                "judge_weight": 0.0,
                "sparsity_weight": 0.0,
                "epsilon_fraction": fraction,
            }
        )
    return tuple(configurations)


def _select_grid(
    groups: dict[str, tuple[CandidateArtifact, ...]],
) -> pd.DataFrame:
    rows = []
    for run_directory, candidates in sorted(groups.items()):
        exemplar = candidates[0]
        for config in _configurations():
            result = select_frozen_candidate(
                candidates,
                policy=str(config["policy"]),  # type: ignore[arg-type]
                judge_weight=float(config["judge_weight"]),
                sparsity_weight=float(config["sparsity_weight"]),
                epsilon_fraction=float(config["epsilon_fraction"]),
            )
            rows.append(
                {
                    "run_directory": run_directory,
                    "benchmark": exemplar.benchmark_id,
                    "tier": exemplar.tier,
                    "seed": exemplar.seed,
                    **config,
                    **result.model_dump(mode="json"),
                }
            )
    return pd.DataFrame(rows)


def _relative_to_validation(selections: pd.DataFrame) -> pd.DataFrame:
    baseline = selections[selections.config_id == "validation_only"][[
        "run_directory",
        "artifact_id",
        "validation_mse",
        "judge_score",
        "term_count",
    ]].rename(
        columns={
            "artifact_id": "baseline_artifact_id",
            "validation_mse": "baseline_validation_mse",
            "judge_score": "baseline_judge_score",
            "term_count": "baseline_term_count",
        }
    )
    joined = selections.merge(baseline, on="run_directory", how="inner")
    joined["same_as_validation"] = (
        joined.artifact_id == joined.baseline_artifact_id
    )
    joined["validation_ratio"] = (
        joined.validation_mse / joined.baseline_validation_mse
    )
    joined["judge_gain"] = joined.judge_score - joined.baseline_judge_score
    joined["term_change"] = joined.term_count - joined.baseline_term_count
    return joined


def _summarize(selections: pd.DataFrame) -> pd.DataFrame:
    joined = _relative_to_validation(selections)
    rows = []
    config_columns = [
        "config_id",
        "policy",
        "judge_weight",
        "sparsity_weight",
        "epsilon_fraction",
    ]
    for values, group in joined.groupby(config_columns, sort=False, dropna=False):
        rows.append(
            {
                **dict(zip(config_columns, values, strict=True)),
                "runs": len(group),
                "same_as_validation_rate": float(group.same_as_validation.mean()),
                "median_validation_ratio": float(group.validation_ratio.median()),
                "validation_ratio_p90": float(group.validation_ratio.quantile(0.9)),
                "within_1pct_validation_rate": float(
                    (group.validation_ratio <= 1.01).mean()
                ),
                "within_5pct_validation_rate": float(
                    (group.validation_ratio <= 1.05).mean()
                ),
                "median_judge_gain": float(group.judge_gain.median()),
                "mean_judge_gain": float(group.judge_gain.mean()),
                "median_term_change": float(group.term_change.median()),
                "mean_term_change": float(group.term_change.mean()),
                "changed_runs": int((~group.same_as_validation).sum()),
                "changed_median_validation_ratio": _changed_median(
                    group, "validation_ratio"
                ),
                "changed_median_judge_gain": _changed_median(group, "judge_gain"),
                "changed_median_term_change": _changed_median(group, "term_change"),
            }
        )
    return pd.DataFrame(rows)


def _changed_median(group: pd.DataFrame, column: str) -> float:
    changed = group[~group.same_as_validation]
    if changed.empty:
        return 1.0 if column == "validation_ratio" else 0.0
    return float(changed[column].median())


def _leave_one_benchmark_out(
    summary: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    del summary  # summaries are recomputed within each development fold
    joined = _relative_to_validation(selections)
    rows = []
    for benchmark in sorted(joined.benchmark.unique()):
        training = joined[joined.benchmark != benchmark]
        heldout = joined[joined.benchmark == benchmark]
        training_summary = _summarize(training.drop(columns=[
            "baseline_artifact_id",
            "baseline_validation_mse",
            "baseline_judge_score",
            "baseline_term_count",
            "same_as_validation",
            "validation_ratio",
            "judge_gain",
            "term_change",
        ]))
        nonbaseline = training_summary[
            training_summary.config_id != "validation_only"
        ]
        admissible = nonbaseline[
            (nonbaseline.median_validation_ratio <= 1.05)
            & (nonbaseline.validation_ratio_p90 <= 1.25)
        ]
        if admissible.empty:
            admissible = nonbaseline
        chosen = admissible.sort_values(
            [
                "mean_judge_gain",
                "mean_term_change",
                "median_validation_ratio",
                "judge_weight",
                "sparsity_weight",
                "epsilon_fraction",
                "config_id",
            ],
            ascending=[False, True, True, True, True, True, True],
        ).iloc[0]
        fold = heldout[heldout.config_id == chosen.config_id]
        rows.append(
            {
                "heldout_benchmark": benchmark,
                "selected_config_id": chosen.config_id,
                "training_median_validation_ratio": chosen.median_validation_ratio,
                "training_validation_ratio_p90": chosen.validation_ratio_p90,
                "training_median_judge_gain": chosen.median_judge_gain,
                "training_median_term_change": chosen.median_term_change,
                "heldout_runs": len(fold),
                "heldout_same_as_validation_rate": float(
                    fold.same_as_validation.mean()
                ),
                "heldout_median_validation_ratio": float(
                    fold.validation_ratio.median()
                ),
                "heldout_validation_ratio_p90": float(
                    fold.validation_ratio.quantile(0.9)
                ),
                "heldout_median_judge_gain": float(fold.judge_gain.median()),
                "heldout_median_term_change": float(fold.term_change.median()),
            }
        )
    return pd.DataFrame(rows)


def _judge_category_audit(
    records: tuple[CandidateArtifact, ...],
) -> pd.DataFrame:
    names = sorted(
        {name for item in records for name in item.judge_category_scores}
    )
    rows = []
    for name in names:
        pairs = [
            (item.judge_category_scores[name], float(item.judge_score))
            for item in records
            if item.judge_score is not None and name in item.judge_category_scores
        ]
        values = np.asarray([pair[0] for pair in pairs], dtype=float)
        aggregate = np.asarray([pair[1] for pair in pairs], dtype=float)
        correlation = (
            float(spearmanr(values, aggregate).statistic)
            if len(set(values)) > 1 and len(set(aggregate)) > 1
            else float("nan")
        )
        rows.append(
            {
                "category": name,
                "candidate_count": len(pairs),
                "mean_score": float(values.mean()),
                "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "spearman_with_aggregate": correlation,
            }
        )
    return pd.DataFrame(rows)


def _report(
    summary: pd.DataFrame,
    lobo: pd.DataFrame,
    categories: pd.DataFrame,
    run_count: int,
) -> str:
    nonbaseline = summary[summary.config_id != "validation_only"].copy()
    admissible = nonbaseline[
        (nonbaseline.median_validation_ratio <= 1.05)
        & (nonbaseline.validation_ratio_p90 <= 1.25)
    ].sort_values(
        ["mean_judge_gain", "mean_term_change", "median_validation_ratio"],
        ascending=[False, True, True],
    )
    display = admissible.head(10)
    lines = [
        "# Frozen selector development analysis",
        "",
        f"This analysis compares {len(summary)} prespecified selector settings "
        f"over {run_count} judged run-level candidate pools. It reads no test "
        "metrics and no private mechanism references.",
        "",
        "All candidates passed deterministic runtime validation. Consequently, "
        "this retrospective pool can compare fit, historical judge score, and "
        "post-pruning complexity, but it cannot retrospectively construct a new "
        "public structural-compliance signal.",
        "",
        "## Most judge-improving admissible settings",
        "",
        "Admissible settings have median validation-loss ratio at most 1.05 and "
        "90th-percentile ratio at most 1.25 relative to validation-only selection.",
        "",
        _markdown(
            display,
            (
                "config_id",
                "same_as_validation_rate",
                "median_validation_ratio",
                "validation_ratio_p90",
                "changed_runs",
                "changed_median_validation_ratio",
                "changed_median_judge_gain",
                "changed_median_term_change",
            ),
        ),
        "",
        "## Leave-one-benchmark-out development robustness",
        "",
        "For each row, configuration choice used only the other five benchmarks. "
        "Held-out columns remain development metrics, not test outcomes.",
        "",
        _markdown(
            lobo,
            (
                "heldout_benchmark",
                "selected_config_id",
                "heldout_runs",
                "heldout_median_validation_ratio",
                "heldout_validation_ratio_p90",
                "heldout_median_judge_gain",
                "heldout_median_term_change",
            ),
        ),
        "",
        "## Historical judge-category behavior",
        "",
        _markdown(
            categories,
            (
                "category",
                "candidate_count",
                "mean_score",
                "sample_sd",
                "spearman_with_aggregate",
            ),
        ),
        "",
        "These tables are sensitivity evidence, not a declaration that a higher "
        "historical judge score defines a scientifically better model. Any final "
        "selector must be frozen before joining its choices to test MSE, private "
        "structural validity, or hidden-trajectory error.",
        "",
    ]
    return "\n".join(lines)


def _markdown(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    headers = [name.replace("_", " ") for name in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values = [
            f"{value:.4g}" if isinstance(value, float) else str(value)
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
