"""Audit unresolved and superseded experiment failures without LLM calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS = (
    "original_b1",
    "perturbed_b1",
    "obfuscated_original_case01",
    "obfuscated_perturbed_case01",
    "benchmark5",
    "benchmark6",
)
TIERS = ("easy", "medium", "hard")
EXPECTED_SEEDS = {
    "persistence": (0,),
    "sindy": (0,),
    "pysr": tuple(range(5)),
    "d3_native_no_tools": tuple(range(5)),
    "llm_feature_sindy": tuple(range(5)),
    "nojudge": tuple(range(5)),
    "full": tuple(range(5)),
}
RUN_PATTERN = re.compile(
    r"^(?P<benchmark>.+)_(?P<tier>easy|medium|hard)_seed(?P<seed>\d+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--failure-sentinel", type=float, default=1e11)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.runs)
    runs = runs[runs.method.isin(EXPECTED_SEEDS)].copy()
    expected = expected_cells()
    completed = set(
        zip(runs.method, runs.benchmark, runs.tier, runs.seed, strict=False)
    )
    historical = historical_failures(args.artifact_root, completed)
    gaps = unresolved_cells(expected, completed, historical)
    extremes = extreme_completed_runs(runs, args.failure_sentinel)
    recovery = recovery_manifest(gaps, extremes, args.artifact_root)

    historical.to_csv(args.output_root / "historical_failures.csv", index=False)
    gaps.to_csv(args.output_root / "unresolved_cells.csv", index=False)
    extremes.to_csv(args.output_root / "invalid_complete_runs.csv", index=False)
    recovery.to_csv(args.output_root / "recovery_manifest.csv", index=False)
    summary = build_summary(expected, runs, historical, gaps, extremes, recovery)
    (args.output_root / "failure_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_root / "failure_audit_report.md").write_text(
        report(summary, gaps, extremes, recovery), encoding="utf-8"
    )


def expected_cells() -> set[tuple[str, str, str, int]]:
    return {
        (method, benchmark, tier, seed)
        for method, seeds in EXPECTED_SEEDS.items()
        for benchmark in BENCHMARKS
        for tier in TIERS
        for seed in seeds
    }


def historical_failures(
    root: Path,
    completed: set[tuple[str, str, str, int]],
) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("run_status.json")):
        payload = _read_json(path)
        if payload.get("status") != "failed":
            continue
        identity = _identity_from_status(path)
        if identity is None:
            continue
        method, benchmark, tier, seed = identity
        error = str(payload.get("error") or "")
        rows.append(
            {
                "method": method,
                "benchmark": benchmark,
                "tier": tier,
                "seed": seed,
                "failure_category": classify_failure(error),
                "superseded_by_complete": identity in completed,
                "cache_replay_candidate": cache_replay_candidate(error),
                "error": error,
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def unresolved_cells(
    expected: set[tuple[str, str, str, int]],
    completed: set[tuple[str, str, str, int]],
    historical: pd.DataFrame,
) -> pd.DataFrame:
    failures: dict[tuple[str, str, str, int], pd.Series] = {}
    for row in historical.itertuples(index=False):
        key = (row.method, row.benchmark, row.tier, int(row.seed))
        failures[key] = row
    rows = []
    for method, benchmark, tier, seed in sorted(expected - completed):
        failure = failures.get((method, benchmark, tier, seed))
        rows.append(
            {
                "method": method,
                "benchmark": benchmark,
                "tier": tier,
                "seed": seed,
                "failure_category": (
                    failure.failure_category if failure is not None else "nonfinal_run"
                ),
                "cache_replay_candidate": (
                    bool(failure.cache_replay_candidate)
                    if failure is not None
                    else method == "full"
                ),
                "source": failure.source if failure is not None else "",
            }
        )
    return pd.DataFrame(rows)


def extreme_completed_runs(runs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    numeric = pd.to_numeric(runs.test_mse, errors="coerce")
    subset = runs[np.isfinite(numeric) & (numeric >= threshold)].copy()
    subset["failure_category"] = "numerical_failure_sentinel"
    subset["reported_status"] = subset.status
    subset["cache_replay_candidate"] = subset.method.eq("llm_feature_sindy")
    return subset[
        [
            "method",
            "benchmark",
            "tier",
            "seed",
            "failure_category",
            "reported_status",
            "test_mse",
            "cache_replay_candidate",
            "source",
        ]
    ]


def recovery_manifest(
    gaps: pd.DataFrame, extremes: pd.DataFrame, artifact_root: Path
) -> pd.DataFrame:
    rows = []
    for row in gaps.itertuples(index=False):
        if row.method == "full":
            action = "cached_candidate_numerical_refit"
            priority = 1
        elif row.method in {"sindy", "pysr"}:
            action = "retain_explicit_algorithm_failure"
            priority = 3
        else:
            action = "inspect_nonfinal_run"
            priority = 2
        rows.append({**row._asdict(), "priority": priority, "action": action})
    for row in extremes.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "tier": row.tier,
                "seed": row.seed,
                "failure_category": row.failure_category,
                "cache_replay_candidate": row.cache_replay_candidate,
                "source": row.source,
                "priority": 1,
                "action": "cached_feature_safe_rollout_replay",
            }
        )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        return manifest
    manifest["candidate_structure_count"] = manifest.apply(
        lambda row: _candidate_structure_count(row, artifact_root), axis=1
    )
    return manifest.sort_values(
        ["priority", "method", "benchmark", "tier", "seed"]
    ).reset_index(drop=True)


def classify_failure(error: str) -> str:
    lowered = error.lower()
    if "could not convert string to float" in lowered:
        return "structured_metadata_ingestion"
    if "invalid json" in lowered or "eof while parsing" in lowered:
        return "truncated_structured_output"
    if "syntax_error" in lowered or "cannot be parsed" in lowered:
        return "structured_expression_validation"
    if "no pysr expression passed safe validation rollout" in lowered:
        return "safe_rollout_failure"
    if "no sindy support passed safe validation rollout" in lowered:
        return "safe_rollout_failure"
    if "no valid fitted candidates" in lowered:
        return "no_valid_fitted_candidate"
    if "final refit failed" in lowered:
        return "final_refit_failure"
    if "numerical fit failed" in lowered or "fitting wall-clock" in lowered:
        return "numerical_fitting_failure"
    return "other_failure"


def cache_replay_candidate(error: str) -> bool:
    category = classify_failure(error)
    return category in {
        "no_valid_fitted_candidate",
        "final_refit_failure",
        "numerical_fitting_failure",
    }


def build_summary(
    expected: set[tuple[str, str, str, int]],
    runs: pd.DataFrame,
    historical: pd.DataFrame,
    gaps: pd.DataFrame,
    extremes: pd.DataFrame,
    recovery: pd.DataFrame,
) -> dict[str, object]:
    valid_completed = len(runs) - len(extremes)
    return {
        "schema_version": "1",
        "expected_core_cells": len(expected),
        "authoritative_completed_rows": len(runs),
        "valid_completed_rows_after_sentinel_check": valid_completed,
        "unresolved_cells": len(gaps),
        "invalid_complete_sentinel_rows": len(extremes),
        "historical_failure_records": len(historical),
        "superseded_historical_failures": int(
            historical.superseded_by_complete.sum()
        ),
        "recovery_actions": dict(Counter(recovery.action)) if len(recovery) else {},
        "uses_test_for_selection": False,
        "test_metric_use": "failure-integrity audit only",
    }


def report(
    summary: dict[str, object],
    gaps: pd.DataFrame,
    extremes: pd.DataFrame,
    recovery: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Failure recovery audit",
            "",
            "This is an integrity and recovery audit, not a model-selection step. "
            "Test metrics are used only to identify the documented numerical-failure "
            "sentinel.",
            "",
            "## Summary",
            "",
            *(f"- {key.replace('_', ' ')}: {value}" for key, value in summary.items()),
            "",
            "## Unresolved expected cells",
            "",
            _markdown(gaps),
            "",
            "## Completed rows carrying a failure sentinel",
            "",
            _markdown(extremes),
            "",
            "## Recovery actions",
            "",
            _markdown(recovery),
            "",
        ]
    )


def _identity_from_status(path: Path) -> tuple[str, str, str, int] | None:
    match = RUN_PATTERN.match(path.parent.name)
    method = _method_from_path(path)
    if match is None or method is None:
        return None
    return (
        method,
        match.group("benchmark"),
        match.group("tier"),
        int(match.group("seed")),
    )


def _method_from_path(path: Path) -> str | None:
    parts = set(path.parts)
    if "d3_native_no_tools" in parts:
        return "d3_native_no_tools"
    if "llm_feature_sindy" in parts:
        return "llm_feature_sindy"
    for method in ("persistence", "pysr", "sindy"):
        if method in parts:
            return method
    return None


def _candidate_structure_count(row: pd.Series, root: Path) -> int:
    if row["method"] != "full":
        return 0
    name = f"{row['benchmark']}_{row['tier']}_seed{int(row['seed'])}"
    structures = set()
    for path in root.rglob(f"{name}/checkpoints/round_*.json"):
        if any(part.startswith("noj-") for part in path.parts):
            continue
        payload = _read_json(path)
        structural_hash = payload.get("structural_hash")
        if structural_hash:
            structures.add(structural_hash)
    return len(structures)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None."
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
