#!/usr/bin/env python3
"""Resolve a frozen model cohort, evaluate interventions, and summarize results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

try:
    from scripts.evaluate_intervention_suite import evaluate_suite
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from evaluate_intervention_suite import evaluate_suite


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_seed(path: Path, payload: dict[str, Any]) -> int | None:
    if payload.get("seed") is not None:
        return int(payload["seed"])
    for parent in path.parents:
        match = re.search(r"_seed(\d+)$", parent.name)
        if match:
            return int(match.group(1))
    return None


def _is_complete(path: Path, *, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if _artifact_seed(path, payload) != seed:
        return False
    if path.name == "final.json":
        return payload.get("stage") == "complete"
    return payload.get("status") == "complete" or payload.get("stage") == "complete"


def resolve_cohort(
    cohort_path: Path, *, project_root: Path
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve one authoritative artifact per expected method/benchmark/seed cell."""

    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    specifications: list[str] = []
    manifest: list[dict[str, Any]] = []
    for method, method_config in cohort["methods"].items():
        for benchmark, patterns in method_config["patterns"].items():
            for seed in method_config["expected_seeds"]:
                selected: Path | None = None
                for pattern in patterns:
                    candidate = project_root / pattern.format(seed=seed)
                    if _is_complete(candidate, seed=seed):
                        selected = candidate
                        break
                record: dict[str, Any] = {
                    "method": method,
                    "benchmark_id": benchmark,
                    "tier": cohort["tier"],
                    "seed": seed,
                    "status": "missing" if selected is None else "complete",
                    "path": None,
                    "sha256": None,
                }
                if selected is not None:
                    relative = selected.relative_to(project_root)
                    label = f"{benchmark}:{method}:{seed}"
                    specifications.append(f"{label}={relative}")
                    record["path"] = str(relative)
                    record["sha256"] = _sha256(selected)
                manifest.append(record)
    return specifications, manifest


def _sample(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6g}"


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate failure-aware intervention metrics across available seeds."""

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        method = row["model_label"].split(":", 2)[1]
        groups[(row["benchmark_id"], row["case_id"], method)].append(row)
    summaries: list[dict[str, Any]] = []
    for (benchmark, case_id, method), items in sorted(groups.items()):
        successful = [item for item in items if item["success"]]
        mses = [float(item["target_mse"]) for item in successful]
        nmses = [float(item["target_nmse"]) for item in successful]
        degradation = [
            float(item["nmse_degradation_ratio"])
            for item in successful
            if item["nmse_degradation_ratio"] is not None
        ]
        directions = [
            float(item.get("response_direction_correct"))
            for item in successful
            if item.get("response_direction_correct") is not None
        ]
        shapes = [
            float(item.get("response_shape_correlation"))
            for item in successful
            if item.get("response_shape_correlation") is not None
        ]
        timings = [
            float(item.get("peak_timing_error_fraction"))
            for item in successful
            if item.get("peak_timing_error_fraction") is not None
        ]
        hidden = [
            float(item.get("hidden_alignment_nmse"))
            for item in successful
            if item.get("hidden_alignment_nmse") is not None
        ]
        hidden_coverage = [
            float(item.get("hidden_state_coverage"))
            for item in successful
            if item.get("hidden_alignment_nmse") is not None
        ]
        summaries.append(
            {
                "benchmark_id": benchmark,
                "case_id": case_id,
                "method": method,
                "attempted": len(items),
                "successful": len(successful),
                "completion_rate": len(successful) / len(items),
                "target_mse_mean": mean(mses) if mses else None,
                "target_mse_sd": stdev(mses)
                if len(mses) > 1
                else 0.0
                if mses
                else None,
                "target_nmse_mean": mean(nmses) if nmses else None,
                "target_nmse_sd": (
                    stdev(nmses) if len(nmses) > 1 else 0.0 if nmses else None
                ),
                "degradation_ratio_mean": mean(degradation) if degradation else None,
                "direction_accuracy": mean(directions) if directions else None,
                "response_shape_correlation_mean": mean(shapes) if shapes else None,
                "peak_timing_error_fraction_mean": mean(timings)
                if timings
                else None,
                "hidden_alignment_nmse_mean": mean(hidden) if hidden else None,
                "hidden_alignment_nmse_sd": stdev(hidden)
                if len(hidden) > 1
                else 0.0
                if hidden
                else None,
                "hidden_state_coverage_mean": mean(hidden_coverage)
                if hidden_coverage
                else None,
            }
        )
    return summaries


def render_markdown(
    summaries: list[dict[str, Any]], manifest: list[dict[str, Any]]
) -> str:
    """Render a compact publication-oriented table plus cohort accounting."""

    lines = [
        "# Frozen intervention evaluation",
        "",
        "Models were selected and fitted before this private evaluation. No structure "
        "or parameter was refitted. MSE is computed against clean private target "
        "trajectories; the noise cases use noisy observations only as permitted "
        "lagged history.",
        "",
        "| Benchmark | Intervention | Method | Target MSE | Target NMSE | "
        "Direction | Shape r | Peak error | Hidden NMSE | Hidden coverage | Success |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        mse = f"{_sample(row['target_mse_mean'])} ± {_sample(row['target_mse_sd'])}"
        nmse = f"{_sample(row['target_nmse_mean'])} ± {_sample(row['target_nmse_sd'])}"
        lines.append(
            f"| {row['benchmark_id']} | {row['case_id']} | {row['method']} | "
            f"{mse} | {nmse} | {_sample(row['direction_accuracy'])} | "
            f"{_sample(row['response_shape_correlation_mean'])} | "
            f"{_sample(row['peak_timing_error_fraction_mean'])} | "
            f"{_sample(row['hidden_alignment_nmse_mean'])} | "
            f"{_sample(row['hidden_state_coverage_mean'])} | "
            f"{row['successful']}/{row['attempted']} |"
        )
    missing = [item for item in manifest if item["status"] != "complete"]
    lines.extend(
        [
            "",
            "## Cohort accounting",
            "",
            f"Resolved frozen cells: {len(manifest) - len(missing)}/{len(manifest)}.",
            "The No-latent ablation is prespecified for three seeds; other methods "
            "use five.",
        ]
    )
    if missing:
        lines.extend(["", "Missing cells:"])
        for item in missing:
            lines.append(
                f"- {item['method']} / {item['benchmark_id']} / seed {item['seed']}"
            )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--free-rollout", action="store_true")
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    specifications, manifest = resolve_cohort(args.cohort, project_root=project_root)
    rows = evaluate_suite(
        suite_path=args.suite,
        data_root=args.data_root,
        tier="hard",
        model_specs=specifications,
        reset_observed_states=not args.free_rollout,
    )
    summaries = summarize(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "cohort_manifest.json").write_text(
        json.dumps(
            {
                "cohort_sha256": _sha256(args.cohort),
                "suite_sha256": _sha256(args.suite),
                "evaluation_protocol": (
                    "free_rollout" if args.free_rollout else "one_step_reset"
                ),
                "records": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_root / "evaluations.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_root / "evaluations.csv", rows)
    _write_csv(args.output_root / "summary.csv", summaries)
    (args.output_root / "summary.md").write_text(
        render_markdown(summaries, manifest), encoding="utf-8"
    )
    successful = sum(bool(row["success"]) for row in rows)
    print(
        f"models={len(specifications)} evaluations={len(rows)} "
        f"successful={successful} failed={len(rows) - successful}"
    )


if __name__ == "__main__":
    main()
