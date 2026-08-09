"""Calibrate historical judge categories on held-out adversarial benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr

CATEGORIES = (
    "task_output_coverage",
    "mechanism_state_adequacy",
    "mathematical_completeness",
    "data_causal_consistency",
    "constraint_compliance",
    "parsimony_interpretability",
)
NARRATIVE_MUTATION = "narrative_equation_mismatch"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    raw = _expand_categories(pd.read_csv(args.scores))
    averaged = _average_repetitions(raw)
    fixed = _fixed_score_metrics(averaged, args.bootstrap_samples)
    lobo, coefficients = _lobo_calibration(averaged)
    reliability = _reliability(raw, averaged)
    mutation = _mutation_metrics(averaged)

    fixed.to_csv(args.output_root / "fixed_score_metrics.csv", index=False)
    lobo.to_csv(args.output_root / "lobo_calibration_metrics.csv", index=False)
    coefficients.to_csv(
        args.output_root / "lobo_calibration_coefficients.csv", index=False
    )
    reliability.to_csv(args.output_root / "judge_reliability.csv", index=False)
    mutation.to_csv(args.output_root / "mutation_sensitivity.csv", index=False)
    (args.output_root / "judge_calibration_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source": str(args.scores.resolve()),
                "raw_calls": len(raw),
                "averaged_candidates": len(averaged),
                "categories": CATEGORIES,
                "primary_scope": "dynamics_only",
                "excluded_from_primary": [NARRATIVE_MUTATION],
                "l2_penalty": 1.0,
                "calibration_split": "leave_one_benchmark_out",
                "uses_test_metrics": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "judge_calibration_report.md").write_text(
        _report(fixed, lobo, reliability, mutation), encoding="utf-8"
    )


def _expand_categories(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = frame.copy()
    payloads = expanded.category_scores.map(json.loads)
    for category in CATEGORIES:
        expanded[category] = payloads.map(
            lambda item, key=category: float(item[key])
        )
    expanded["valid_label"] = (expanded.known_label == "valid").astype(int)
    expanded["qualitative_score"] = expanded[
        ["mechanism_state_adequacy", "parsimony_interpretability"]
    ].mean(axis=1)
    return expanded


def _average_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["aggregate_score", "qualitative_score", *CATEGORIES]
    keys = [
        "pair_id",
        "benchmark_id",
        "mutation_type",
        "known_label",
        "judge_model",
        "valid_label",
    ]
    return frame.groupby(keys, as_index=False)[columns].mean()


def _fixed_score_metrics(frame: pd.DataFrame, samples: int) -> pd.DataFrame:
    rows = []
    for scope, subset in _scopes(frame):
        for model, group in subset.groupby("judge_model"):
            for score_name in (
                "aggregate_score",
                "qualitative_score",
                *CATEGORIES,
            ):
                metrics = _score_metrics(group, score_name)
                low, high = _bootstrap_pair_accuracy(
                    group, score_name, samples=samples
                )
                rows.append(
                    {
                        "scope": scope,
                        "judge_model": model,
                        "score": score_name,
                        **metrics,
                        "pair_accuracy_ci_low": low,
                        "pair_accuracy_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def _lobo_calibration(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    coefficient_rows = []
    for scope, scoped in _scopes(frame):
        for model, model_frame in scoped.groupby("judge_model"):
            predictions = []
            for heldout in sorted(model_frame.benchmark_id.unique()):
                train = model_frame[model_frame.benchmark_id != heldout]
                test = model_frame[model_frame.benchmark_id == heldout].copy()
                fitted = fit_ridge_logistic(
                    train.loc[:, CATEGORIES].to_numpy(),
                    train.valid_label.to_numpy(),
                )
                test["calibrated_score"] = predict_ridge_logistic(
                    fitted, test.loc[:, CATEGORIES].to_numpy()
                )
                predictions.append(test)
                for category, coefficient in zip(
                    CATEGORIES, fitted["coefficients"], strict=True
                ):
                    coefficient_rows.append(
                        {
                            "scope": scope,
                            "judge_model": model,
                            "heldout_benchmark": heldout,
                            "category": category,
                            "standardized_coefficient": coefficient,
                        }
                    )
            combined = pd.concat(predictions, ignore_index=True)
            for score_name in (
                "aggregate_score",
                "qualitative_score",
                "calibrated_score",
            ):
                low, high = _bootstrap_pair_accuracy(
                    combined, score_name, samples=20_000
                )
                metric_rows.append(
                    {
                        "scope": scope,
                        "judge_model": model,
                        "score": score_name,
                        **_score_metrics(combined, score_name),
                        "pair_accuracy_ci_low": low,
                        "pair_accuracy_ci_high": high,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def fit_ridge_logistic(
    features: np.ndarray, labels: np.ndarray, *, l2_penalty: float = 1.0
) -> dict[str, np.ndarray | float]:
    """Fit a deterministic standardized ridge-logistic calibration model."""
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if features.ndim != 2 or labels.shape != (len(features),):
        raise ValueError("features and labels have incompatible shapes")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("both binary labels are required")
    location = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    standardized = (features - location) / scale
    design = np.column_stack((np.ones(len(features)), standardized))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        probabilities = expit(logits)
        loss = float(
            np.mean(np.logaddexp(0.0, logits) - labels * logits)
            + 0.5 * l2_penalty * np.sum(parameters[1:] ** 2) / len(labels)
        )
        gradient = design.T @ (probabilities - labels) / len(labels)
        gradient[1:] += l2_penalty * parameters[1:] / len(labels)
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(design.shape[1]),
        method="L-BFGS-B",
        jac=True,
    )
    if not result.success:
        raise RuntimeError(f"logistic calibration failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "coefficients": np.asarray(result.x[1:], dtype=float),
        "location": location,
        "scale": scale,
    }


def predict_ridge_logistic(
    fitted: dict[str, np.ndarray | float], features: np.ndarray
) -> np.ndarray:
    standardized = (
        np.asarray(features, dtype=float) - np.asarray(fitted["location"])
    ) / np.asarray(fitted["scale"])
    return expit(
        float(fitted["intercept"])
        + standardized @ np.asarray(fitted["coefficients"])
    )


def _score_metrics(frame: pd.DataFrame, score: str) -> dict[str, float | int]:
    pivot = frame.pivot(index="pair_id", columns="known_label", values=score)
    margins = pivot["valid"] - pivot["adversarial"]
    labels = frame.valid_label.to_numpy()
    values = frame[score].to_numpy()
    return {
        "candidate_count": len(frame),
        "pair_count": len(margins),
        "pair_accuracy": float((margins > 0).mean()),
        "false_preference_rate": float((margins < 0).mean()),
        "tie_rate": float((margins == 0).mean()),
        "mean_margin": float(margins.mean()),
        "auroc": _auroc(values[labels == 1], values[labels == 0]),
        "brier_score": float(np.mean((values - labels) ** 2)),
    }


def _bootstrap_pair_accuracy(
    frame: pd.DataFrame,
    score: str,
    *,
    samples: int,
    random_seed: int = 20260805,
) -> tuple[float, float]:
    pivot = frame.pivot(index="pair_id", columns="known_label", values=score)
    correct = (pivot["valid"] > pivot["adversarial"]).to_numpy(dtype=float)
    generator = np.random.default_rng(random_seed)
    draws = generator.choice(correct, size=(samples, len(correct)), replace=True)
    low, high = np.percentile(draws.mean(axis=1), (2.5, 97.5))
    return float(low), float(high)


def _reliability(raw: pd.DataFrame, averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in raw.groupby("judge_model"):
        matrix = group.pivot(
            index=["pair_id", "known_label"],
            columns="repetition",
            values="aggregate_score",
        ).to_numpy()
        rows.append(
            {
                "judge_model": model,
                "targets": len(matrix),
                "repetitions": matrix.shape[1],
                "icc_1_1": _icc_one_way(matrix),
                "mean_within_target_sd": float(np.mean(matrix.std(axis=1, ddof=0))),
            }
        )
    models = sorted(averaged.judge_model.unique())
    if len(models) == 2:
        left = averaged[averaged.judge_model == models[0]][
            ["pair_id", "known_label", "aggregate_score"]
        ]
        right = averaged[averaged.judge_model == models[1]][
            ["pair_id", "known_label", "aggregate_score"]
        ]
        joined = left.merge(
            right,
            on=["pair_id", "known_label"],
            suffixes=("_left", "_right"),
        )
        correlation = float(
            spearmanr(
                joined.aggregate_score_left,
                joined.aggregate_score_right,
            ).statistic
        )
        rows.append(
            {
                "judge_model": f"{models[0]} vs {models[1]}",
                "targets": len(joined),
                "repetitions": 1,
                "icc_1_1": correlation,
                "mean_within_target_sd": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _mutation_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, mutation), group in frame.groupby(
        ["judge_model", "mutation_type"]
    ):
        rows.append(
            {
                "judge_model": model,
                "mutation_type": mutation,
                **_score_metrics(group, "aggregate_score"),
            }
        )
    return pd.DataFrame(rows)


def _scopes(frame: pd.DataFrame):
    yield "all", frame
    yield "dynamics_only", frame[frame.mutation_type != NARRATIVE_MUTATION]


def _icc_one_way(matrix: np.ndarray) -> float:
    targets, repetitions = matrix.shape
    target_means = matrix.mean(axis=1)
    grand_mean = matrix.mean()
    between = repetitions * np.sum((target_means - grand_mean) ** 2) / (
        targets - 1
    )
    within = np.sum((matrix - target_means[:, None]) ** 2) / (
        targets * (repetitions - 1)
    )
    return float((between - within) / (between + (repetitions - 1) * within))


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    wins = np.sum(positive[:, None] > negative[None, :])
    ties = np.sum(positive[:, None] == negative[None, :])
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def _report(
    fixed: pd.DataFrame,
    lobo: pd.DataFrame,
    reliability: pd.DataFrame,
    mutation: pd.DataFrame,
) -> str:
    primary = fixed[fixed.scope == "dynamics_only"]
    calibrated = lobo[lobo.scope == "dynamics_only"]
    return "\n".join(
        [
            "# Judge calibration analysis",
            "",
            "Primary results exclude the prose-only narrative mutation because it "
            "does not deterministically alter the candidate dynamics. The all-"
            "mutation result is retained as sensitivity analysis.",
            "",
            "## Fixed historical scores on dynamics mutations",
            "",
            _markdown(primary),
            "",
            "## Leave-one-benchmark-out category calibration",
            "",
            "Each calibrated score is fitted on three benchmarks and evaluated on "
            "the fourth. No trajectory test metric is used.",
            "",
            _markdown(calibrated),
            "",
            "## Repeat and cross-family reliability",
            "",
            _markdown(reliability),
            "",
            "## Mutation-level aggregate-score sensitivity",
            "",
            _markdown(mutation),
            "",
            "A calibrated adversarial classifier is not automatically a valid "
            "model-selection score. It establishes measurement sensitivity only; "
            "prospective selection influence requires a separate frozen study.",
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
