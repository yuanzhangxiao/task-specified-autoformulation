"""Summarize the hard-tier no-persistent-latent ablation against full runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autoformalism.rebuttal.artifacts import _term_count
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.schemas import CandidateModel


def _summary(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[np.isfinite(clean) & (clean < 1e11)]
    if clean.empty:
        return "N/A"
    if len(clean) == 1:
        return f"{clean.iloc[0]:.4g}"
    return f"{clean.mean():.4g} ± {clean.std(ddof=1):.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative-runs", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    specs = {
        spec.benchmark_id: spec
        for path in args.config_root.glob("*_hard.json")
        for spec in (
            MechanismEvaluationSpec.model_validate_json(
                path.read_text(encoding="utf-8")
            ),
        )
    }

    records: dict[tuple[str, int], dict[str, object]] = {}
    for root in args.input_root:
        for path in sorted(root.rglob("checkpoints/final.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("stage") != "complete":
                continue
            run_name = path.parent.parent.name
            benchmark, marker, seed_text = run_name.rpartition("_hard_seed")
            if not marker or not seed_text.isdigit():
                continue
            candidate = CandidateModel.model_validate(
                payload["frozen"]["candidate"]
            )
            structural = evaluate_mechanisms(candidate, specs[benchmark])
            records[(benchmark, int(seed_text))] = {
                "benchmark": benchmark,
                "seed": int(seed_text),
                "no_latent_test_nmse": payload["test_metrics"]["normalized_mse"],
                "structural_validity": structural.structural_validity,
                "dynamic_terms": sum(
                    _term_count(item.rhs) for item in candidate.state_equations
                ),
                "source": str(path),
            }
    no_latent = pd.DataFrame(records.values())
    full = pd.read_csv(args.authoritative_runs)
    full = full[(full.method == "full") & (full.tier == "hard")][
        ["benchmark", "seed", "test_mse"]
    ].rename(columns={"test_mse": "full_test_nmse"})
    merged = no_latent.merge(full, on=["benchmark", "seed"], how="left")
    merged["nmse_ratio_no_latent_over_full"] = (
        merged.no_latent_test_nmse / merged.full_test_nmse
    )

    rows = []
    for benchmark, group in merged.groupby("benchmark", sort=True):
        rows.append(
            {
                "benchmark": benchmark,
                "completed": len(group),
                "full_test_nmse": _summary(group.full_test_nmse),
                "no_latent_test_nmse": _summary(group.no_latent_test_nmse),
                "structural_validity": _summary(group.structural_validity),
                "dynamic_terms": _summary(group.dynamic_terms),
                "nmse_ratio": _summary(group.nmse_ratio_no_latent_over_full),
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    merged.sort_values(["benchmark", "seed"]).to_csv(
        args.output_root / "no_latent_runs.csv", index=False
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_root / "no_latent_summary.csv", index=False)
    lines = [
        "# Persistent-latent-state ablation",
        "",
        "| Benchmark | Complete | Full NMSE ↓ | No-latent NMSE ↓ | "
        "Structural validity ↑ | Terms ↓ | Ratio ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.benchmark} | {row.completed}/3 | {row.full_test_nmse} | "
            f"{row.no_latent_test_nmse} | {row.structural_validity} | "
            f"{row.dynamic_terms} | {row.nmse_ratio} |"
        )
    (args.output_root / "no_latent_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
