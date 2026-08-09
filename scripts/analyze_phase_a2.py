"""Run frozen Phase A2 statistics and judge-utility analyses without LLM calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from autoformalism.rebuttal.artifacts import CandidateArtifact
from autoformalism.rebuttal.objectives import (
    ratio_objective,
    weighted_sum_objective,
)
from autoformalism.rebuttal.statistics import (
    holm_adjust,
    paired_log_comparison,
    wilson_interval,
)

EXPECTED = {
    "persistence": 18,
    "sindy": 18,
    "pysr": 90,
    "d3_native_no_tools": 90,
    "llm_feature_sindy": 90,
    "nojudge": 90,
    "full": 90,
}
COMPARATORS = (
    "nojudge",
    "d3_native_no_tools",
    "llm_feature_sindy",
    "pysr",
    "sindy",
    "persistence",
)
DISPLAY = {
    "full": "Full method",
    "nojudge": "No-judge",
    "d3_native_no_tools": "D3",
    "llm_feature_sindy": "LLM-feature-SINDy",
    "pysr": "PySR",
    "sindy": "SINDy",
    "persistence": "Persistence",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--full-candidate-pool", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--lambda-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--policy-test-metrics",
        type=Path,
        help="Optional frozen post-selection test metrics from the evaluator.",
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs)
    completion = _completion(runs)
    comparisons = _paired_comparisons(runs)
    full_nojudge_strata = _full_nojudge_strata(runs)
    selected_associations = _selected_associations(runs)
    candidates = _read_pool(args.full_candidate_pool)
    within_run, selections, policy_summary = _judge_policy_analysis(
        candidates,
        epsilon=args.epsilon,
        lambda_multiplier=args.lambda_multiplier,
    )

    completion.to_csv(args.output_root / "completion_rates.csv", index=False)
    comparisons.to_csv(args.output_root / "paired_method_comparisons.csv", index=False)
    full_nojudge_strata.to_csv(
        args.output_root / "full_vs_nojudge_stratified.csv", index=False
    )
    selected_associations.to_csv(
        args.output_root / "selected_judge_associations.csv", index=False
    )
    within_run.to_csv(
        args.output_root / "within_run_judge_correlations.csv", index=False
    )
    selections.to_csv(args.output_root / "judge_policy_selections.csv", index=False)
    policy_summary.to_csv(args.output_root / "judge_policy_summary.csv", index=False)
    policy_test_details = pd.DataFrame()
    policy_test_summary = pd.DataFrame()
    if args.policy_test_metrics is not None:
        policy_test_details, policy_test_summary = _policy_test_comparison(
            selections,
            runs,
            pd.read_csv(args.policy_test_metrics),
        )
        policy_test_details.to_csv(
            args.output_root / "judge_policy_test_comparisons.csv", index=False
        )
        policy_test_summary.to_csv(
            args.output_root / "judge_policy_test_summary.csv", index=False
        )
    manifest = selections[
        [
            "run_directory",
            "benchmark",
            "tier",
            "seed",
            "policy",
            "artifact_id",
        ]
    ].to_dict(orient="records")
    (args.output_root / "frozen_judge_policy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validation_ids = {
        row["run_directory"]: row["artifact_id"]
        for row in manifest
        if row["policy"] == "validation_only"
    }
    changed_manifest = [
        row
        for row in manifest
        if row["policy"] in {"ratio", "weighted_sum"}
        and row["artifact_id"] != validation_ids[row["run_directory"]]
    ]
    (args.output_root / "frozen_changed_judge_policy_manifest.json").write_text(
        json.dumps(changed_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "phase_a2_report.md").write_text(
        _report(
            completion,
            comparisons,
            full_nojudge_strata,
            selected_associations,
            policy_summary,
            within_run,
            policy_test_summary,
        ),
        encoding="utf-8",
    )


def _policy_test_comparison(
    selections: pd.DataFrame,
    runs: pd.DataFrame,
    test_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare frozen changed policy selections with production selections."""
    validation = selections[selections.policy == "validation_only"][[
        "run_directory", "artifact_id"
    ]].rename(columns={"artifact_id": "validation_artifact_id"})
    changed = selections[selections.policy.isin(("ratio", "weighted_sum"))].merge(
        validation, on="run_directory", how="inner"
    )
    changed = changed[changed.artifact_id != changed.validation_artifact_id].copy()
    metric_columns = test_metrics[["artifact_id", "status", "error", "test_mse"]]
    changed = changed.merge(metric_columns, on="artifact_id", how="left")
    current = runs[runs.method == "full"][[
        "benchmark", "tier", "seed", "test_mse"
    ]].rename(columns={"test_mse": "production_test_mse"})
    changed = changed.merge(current, on=["benchmark", "tier", "seed"], how="left")
    changed = changed.rename(columns={"test_mse": "policy_test_mse"})
    changed["valid_pair"] = _valid_error(changed.policy_test_mse) & _valid_error(
        changed.production_test_mse
    )
    changed["policy_wins"] = changed.policy_test_mse < changed.production_test_mse
    changed["test_mse_ratio"] = (
        changed.policy_test_mse / changed.production_test_mse
    )

    rows = []
    for policy in ("ratio", "weighted_sum"):
        subset = changed[changed.policy == policy]
        valid = subset[subset.valid_pair]
        comparison = paired_log_comparison(
            valid.policy_test_mse.to_numpy(),
            valid.production_test_mse.to_numpy(),
            permutation_samples=20_000,
        )
        rows.append(
            {
                "policy": policy,
                "changed_selections": len(subset),
                "successful_test_refits": len(valid),
                "failed_test_refits": len(subset) - len(valid),
                **comparison.model_dump(mode="json"),
            }
        )
    return changed, pd.DataFrame(rows)


def _valid_error(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.isfinite(numeric) & (numeric > 0) & (numeric < 1e11)


def _completion(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, expected in EXPECTED.items():
        subset = runs[runs.method == method]
        complete = len(subset)
        valid = int(_valid_error(subset.test_mse).sum())
        low, high = wilson_interval(complete, expected)
        valid_low, valid_high = wilson_interval(valid, expected)
        rows.append(
            {
                "method": method,
                "expected": expected,
                "complete_artifacts": complete,
                "completion_rate": complete / expected,
                "completion_ci_low": low,
                "completion_ci_high": high,
                "valid_test_results": valid,
                "valid_result_rate": valid / expected,
                "valid_result_ci_low": valid_low,
                "valid_result_ci_high": valid_high,
            }
        )
    return pd.DataFrame(rows)


def _paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["benchmark", "tier", "seed"]
    full = runs[runs.method == "full"][[*keys, "test_mse"]].rename(
        columns={"test_mse": "full_test_mse"}
    )
    rows = []
    for comparator in COMPARATORS:
        other = runs[runs.method == comparator][[*keys, "test_mse"]].rename(
            columns={"test_mse": "comparator_test_mse"}
        )
        paired = full.merge(other, on=keys, how="inner")
        valid = _valid_error(paired.full_test_mse) & _valid_error(
            paired.comparator_test_mse
        )
        result = paired_log_comparison(
            paired.loc[valid, "full_test_mse"].to_numpy(),
            paired.loc[valid, "comparator_test_mse"].to_numpy(),
        )
        rows.append(
            {
                "first_method": "full",
                "comparator": comparator,
                **result.model_dump(mode="json"),
                "matched_artifact_pairs": len(paired),
            }
        )
    adjusted = holm_adjust([float(row["sign_flip_p_value"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_sign_flip_p_value"] = value
    return pd.DataFrame(rows)


def _full_nojudge_strata(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["benchmark", "tier", "seed"]
    full = runs[runs.method == "full"][[*keys, "test_mse"]].rename(
        columns={"test_mse": "full_test_mse"}
    )
    nojudge = runs[runs.method == "nojudge"][[*keys, "test_mse"]].rename(
        columns={"test_mse": "nojudge_test_mse"}
    )
    paired = full.merge(nojudge, on=keys, how="inner")
    strata = [
        ("tier", value, paired[paired.tier == value])
        for value in ("easy", "medium", "hard")
    ] + [
        ("benchmark", value, paired[paired.benchmark == value])
        for value in sorted(paired.benchmark.unique())
    ]
    rows = []
    for stratum, value, group in strata:
        valid = _valid_error(group.full_test_mse) & _valid_error(group.nojudge_test_mse)
        result = paired_log_comparison(
            group.loc[valid, "full_test_mse"].to_numpy(),
            group.loc[valid, "nojudge_test_mse"].to_numpy(),
            permutation_samples=20_000,
        )
        rows.append(
            {
                "stratum": stratum,
                "value": value,
                **result.model_dump(mode="json"),
            }
        )
    return pd.DataFrame(rows)


def _selected_associations(runs: pd.DataFrame) -> pd.DataFrame:
    full = runs[runs.method == "full"]
    rows = []
    for metric in ("validation_mse", "test_mse"):
        subset = full[["judge_score", metric]].dropna()
        subset = subset[_valid_error(subset[metric])]
        correlation = spearmanr(subset.judge_score, subset[metric]).statistic
        rows.append(
            {
                "scope": "selected_full_runs",
                "metric": metric,
                "n": len(subset),
                "spearman_judge_vs_error": float(correlation),
                "desired_direction": "negative",
            }
        )
    return pd.DataFrame(rows)


def _read_pool(path: Path) -> tuple[CandidateArtifact, ...]:
    return tuple(
        CandidateArtifact.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _judge_policy_analysis(
    records: tuple[CandidateArtifact, ...],
    *,
    epsilon: float,
    lambda_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped: dict[str, list[CandidateArtifact]] = {}
    for item in records:
        if item.judge_score is not None:
            grouped.setdefault(item.run_directory, []).append(item)
    correlation_rows = []
    selection_rows = []
    for run_directory, raw_candidates in sorted(grouped.items()):
        candidates = _deduplicate_structures(raw_candidates)
        exemplar = candidates[0]
        losses = np.asarray([item.validation_mse for item in candidates])
        scores = np.asarray([float(item.judge_score) for item in candidates])
        if len(candidates) >= 3 and len(set(losses)) > 1 and len(set(scores)) > 1:
            correlation_rows.append(
                {
                    "run_directory": run_directory,
                    "benchmark": exemplar.benchmark_id,
                    "tier": exemplar.tier,
                    "seed": exemplar.seed,
                    "candidate_count": len(candidates),
                    "spearman_judge_vs_validation_mse": float(
                        spearmanr(scores, losses).statistic
                    ),
                }
            )
        lambda_value = float(np.median(losses)) * lambda_multiplier
        policies = {
            "validation_only": min(
                candidates,
                key=lambda item: (
                    item.validation_mse,
                    -float(item.judge_score),
                    item.term_count,
                    item.structural_hash,
                ),
            ),
            "judge_only": min(
                candidates,
                key=lambda item: (
                    -float(item.judge_score),
                    item.validation_mse,
                    item.term_count,
                    item.structural_hash,
                ),
            ),
            "ratio": min(
                candidates,
                key=lambda item: ratio_objective(
                    item.validation_mse, float(item.judge_score), epsilon
                ),
            ),
            "weighted_sum": min(
                candidates,
                key=lambda item: weighted_sum_objective(
                    item.validation_mse,
                    float(item.judge_score),
                    lambda_value,
                    epsilon,
                ),
            ),
        }
        for policy, selected in policies.items():
            selection_rows.append(
                {
                    "run_directory": run_directory,
                    "benchmark": exemplar.benchmark_id,
                    "tier": exemplar.tier,
                    "seed": exemplar.seed,
                    "candidate_count": len(candidates),
                    "policy": policy,
                    "artifact_id": selected.artifact_id,
                    "validation_mse": selected.validation_mse,
                    "judge_score": selected.judge_score,
                    "term_count": selected.term_count,
                    "lambda_value": lambda_value,
                }
            )
    correlations = pd.DataFrame(correlation_rows)
    selections = pd.DataFrame(selection_rows)
    baseline = selections[selections.policy == "validation_only"].set_index(
        "run_directory"
    )
    summary_rows = []
    for policy in ("validation_only", "judge_only", "ratio", "weighted_sum"):
        subset = selections[selections.policy == policy].set_index("run_directory")
        common = subset.join(
            baseline[["artifact_id", "validation_mse", "judge_score"]],
            lsuffix="_policy",
            rsuffix="_validation",
        )
        summary_rows.append(
            {
                "policy": policy,
                "runs": len(subset),
                "same_as_validation_rate": float(
                    np.mean(common.artifact_id_policy == common.artifact_id_validation)
                ),
                "median_validation_mse": float(subset.validation_mse.median()),
                "median_judge_score": float(subset.judge_score.median()),
                "median_validation_loss_ratio_vs_validation": float(
                    np.median(
                        common.validation_mse_policy / common.validation_mse_validation
                    )
                ),
                "median_judge_score_change_vs_validation": float(
                    np.median(common.judge_score_policy - common.judge_score_validation)
                ),
                "changed_runs": int(
                    np.sum(common.artifact_id_policy != common.artifact_id_validation)
                ),
                "changed_median_validation_loss_ratio": _changed_median(
                    common,
                    "validation_mse_policy",
                    "validation_mse_validation",
                    ratio=True,
                ),
                "changed_median_judge_score_change": _changed_median(
                    common,
                    "judge_score_policy",
                    "judge_score_validation",
                    ratio=False,
                ),
            }
        )
    return correlations, selections, pd.DataFrame(summary_rows)


def _changed_median(
    frame: pd.DataFrame,
    first: str,
    second: str,
    *,
    ratio: bool,
) -> float:
    changed = frame[frame.artifact_id_policy != frame.artifact_id_validation]
    if changed.empty:
        return 1.0 if ratio else 0.0
    values = (
        changed[first] / changed[second] if ratio else changed[first] - changed[second]
    )
    return float(np.median(values))


def _deduplicate_structures(
    candidates: list[CandidateArtifact],
) -> list[CandidateArtifact]:
    unique: dict[str, CandidateArtifact] = {}
    for item in candidates:
        existing = unique.get(item.structural_hash)
        if existing is None or (
            item.validation_mse,
            -float(item.judge_score),
            item.artifact_id,
        ) < (
            existing.validation_mse,
            -float(existing.judge_score),
            existing.artifact_id,
        ):
            unique[item.structural_hash] = item
    return sorted(unique.values(), key=lambda item: item.artifact_id)


def _report(
    completion: pd.DataFrame,
    comparisons: pd.DataFrame,
    full_nojudge_strata: pd.DataFrame,
    associations: pd.DataFrame,
    policies: pd.DataFrame,
    within_run: pd.DataFrame,
    policy_test_summary: pd.DataFrame,
) -> str:
    nojudge = comparisons[comparisons.comparator == "nojudge"].iloc[0]
    weighted = policies[policies.policy == "weighted_sum"].iloc[0]
    validation = associations[associations.metric == "validation_mse"].iloc[0]
    test = associations[associations.metric == "test_mse"].iloc[0]
    lines = [
        "# Phase A2 frozen-evidence analysis",
        "",
        "All model selection inputs in this analysis are development-only. Test "
        "MSE is used only for post-selection evaluation and paired reporting.",
        "",
        "## Completion and valid-result rates",
        "",
        _markdown(
            completion,
            (
                "method",
                "expected",
                "complete_artifacts",
                "completion_rate",
                "valid_test_results",
                "valid_result_rate",
            ),
        ),
        "",
        "## Paired test-NMSE comparisons",
        "",
        "The ratio is full-method NMSE divided by comparator NMSE; values below "
        "one favor the full method.",
        "",
        _markdown(
            comparisons,
            (
                "comparator",
                "pair_count",
                "first_win_rate",
                "geometric_mean_ratio",
                "geometric_ratio_ci_low",
                "geometric_ratio_ci_high",
                "holm_sign_flip_p_value",
            ),
        ),
        "",
        "## Judge diagnosis",
        "",
        f"Across {int(nojudge.pair_count)} valid matched full/no-judge pairs, the "
        f"full method wins {nojudge.first_win_rate:.1%} of pairs and has a "
        f"geometric-mean NMSE ratio of {nojudge.geometric_mean_ratio:.3f} "
        f"(95% bootstrap CI {nojudge.geometric_ratio_ci_low:.3f}--"
        f"{nojudge.geometric_ratio_ci_high:.3f}).",
        "",
        "Stratified full/no-judge results:",
        "",
        _markdown(
            full_nojudge_strata,
            (
                "stratum",
                "value",
                "pair_count",
                "first_win_rate",
                "geometric_mean_ratio",
                "geometric_ratio_ci_low",
                "geometric_ratio_ci_high",
            ),
        ),
        "",
        f"For selected full-method models, judge score has Spearman correlation "
        f"{validation.spearman_judge_vs_error:.3f} with validation NMSE and "
        f"{test.spearman_judge_vs_error:.3f} with test NMSE. Negative is the "
        "desired direction; these associations are modest rather than decisive.",
        "",
        f"Within the {len(within_run)} runs having at least three nonconstant "
        "judged candidates, the median judge-versus-validation Spearman "
        f"correlation is {within_run.spearman_judge_vs_validation_mse.median():.3f}; "
        f"it has the desired negative sign in "
        f"{(within_run.spearman_judge_vs_validation_mse < 0).mean():.1%} of runs.",
        "",
        "The production controller currently ranks candidates by validation NMSE "
        "first and uses judge score only as a tie-breaker. It does not use test "
        "MSE, but it also does not implement a genuine weighted-sum objective.",
        "",
        f"At the prespecified 1x scaled judge weight, weighted-sum selection agrees "
        f"with validation-only selection in {weighted.same_as_validation_rate:.1%} "
        f"of frozen runs. Its median validation-loss ratio is "
        f"{weighted.median_validation_loss_ratio_vs_validation:.3f}, while its "
        f"median judge-score change is "
        f"{weighted.median_judge_score_change_vs_validation:+.3f}.",
        f"Among its {int(weighted.changed_runs)} changed selections specifically, "
        f"the median validation-loss ratio is "
        f"{weighted.changed_median_validation_loss_ratio:.3f} and the median "
        f"judge-score increase is {weighted.changed_median_judge_score_change:+.3f}.",
        "",
    ]
    if policy_test_summary.empty:
        lines.extend(
            [
                "Alternative policy selections are frozen in "
                "`frozen_judge_policy_manifest.json`. Their test performance has "
                "not yet been supplied to this analysis.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Frozen alternative-policy test evaluation",
                "",
                "Only selections that differ from the production validation-only "
                "choice require a new deterministic refit. These choices were "
                "frozen before any test evaluation. A ratio below one favors the "
                "alternative policy.",
                "",
                _markdown(
                    policy_test_summary,
                    (
                        "policy",
                        "changed_selections",
                        "successful_test_refits",
                        "failed_test_refits",
                        "first_win_rate",
                        "geometric_mean_ratio",
                        "geometric_ratio_ci_low",
                        "geometric_ratio_ci_high",
                        "sign_flip_p_value",
                    ),
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _markdown(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    headers = [name.replace("_", " ") for name in columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4g}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
