"""Build detailed rebuttal tables and an equation audit from consolidated runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ORDER = (
    "persistence",
    "sindy",
    "pysr",
    "d3_native_no_tools",
    "llm_feature_sindy",
    "nojudge",
    "full",
)
BENCHMARK_ORDER = (
    "original_b1",
    "perturbed_b1",
    "obfuscated_original_case01",
    "obfuscated_perturbed_case01",
    "benchmark5",
    "benchmark6",
)
TIERS = ("easy", "medium", "hard")
DISPLAY = {
    "persistence": "Persistence",
    "sindy": "SINDy",
    "pysr": "PySR",
    "d3_native_no_tools": "D3-native-no-tools",
    "llm_feature_sindy": "LLM-feature-SINDy",
    "nojudge": "No-judge Autoformalism",
    "full": "Full method",
}


def _number(value: float) -> str:
    return f"{value:.3g}"


def _cell(values: pd.Series, expected: int) -> str:
    clean = values.dropna().astype(float)
    clean = clean[np.isfinite(clean) & (clean < 1e12)]
    if clean.empty:
        return f"Failure (0/{expected})"
    return (
        f"{_number(float(clean.median()))} "
        f"[{_number(float(clean.min()))}, {_number(float(clean.max()))}] "
        f"({len(clean)}/{expected})"
    )


def _expected(method: str, benchmark: str, tier_count: int = 1) -> int:
    seeds = 5 if method in {
        "pysr",
        "d3_native_no_tools",
        "llm_feature_sindy",
        "nojudge",
        "full",
    } else 1
    return seeds * tier_count


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _extract_equations(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data.get("equations"), dict):
        return {str(k): str(v) for k, v in data["equations"].items()}
    candidate = ((data.get("frozen") or {}).get("candidate") or {})
    equations = {
        f"d({row['state']})/dt": str(row["rhs"])
        for row in candidate.get("state_equations", [])
    }
    equations.update(
        {
            str(row["name"]): str(row["expression"])
            for row in candidate.get("processes", [])
        }
    )
    return equations


def main() -> None:
    output = Path("artifacts/rebuttal/analysis")
    runs = pd.read_csv(output / "authoritative_runs.csv")

    sections = [
        "# Detailed generalization tables",
        "",
        "Cells report median [minimum, maximum] test NMSE and valid/expected runs. "
        "The range is shown deliberately because IQR can hide isolated "
        "catastrophic runs.",
    ]
    for tier in TIERS:
        rows = []
        for method in METHOD_ORDER:
            row = [DISPLAY[method]]
            for benchmark in BENCHMARK_ORDER:
                subset = runs[
                    (runs.method == method)
                    & (runs.benchmark == benchmark)
                    & (runs.tier == tier)
                ]
                row.append(_cell(subset.test_mse, _expected(method, benchmark)))
            rows.append(row)
        sections.extend(
            [
                "",
                f"## {tier.capitalize()} tier",
                "",
                _markdown_table(
                    ["Method", *BENCHMARK_ORDER],
                    rows,
                ),
            ]
        )

    rows = []
    for method in METHOD_ORDER:
        row = [DISPLAY[method]]
        for benchmark in BENCHMARK_ORDER:
            subset = runs[(runs.method == method) & (runs.benchmark == benchmark)]
            row.append(_cell(subset.test_mse, _expected(method, benchmark, 3)))
        rows.append(row)
    sections.extend(
        [
            "",
            "## All tiers pooled",
            "",
            "This pools all available seeds across easy, medium, and hard; it does "
            "not average the three tier medians.",
            "",
            _markdown_table(["Method", *BENCHMARK_ORDER], rows),
            "",
        ]
    )

    average_rows = []
    for method in METHOD_ORDER:
        row = [DISPLAY[method]]
        for benchmark in BENCHMARK_ORDER:
            tier_medians = []
            for tier in TIERS:
                values = runs[
                    (runs.method == method)
                    & (runs.benchmark == benchmark)
                    & (runs.tier == tier)
                ].test_mse.dropna()
                values = values[np.isfinite(values) & (values < 1e12)]
                if len(values):
                    tier_medians.append(float(values.median()))
            if len(tier_medians) != len(TIERS):
                row.append(f"N/A ({len(tier_medians)}/3 tiers)")
            else:
                arithmetic = float(np.mean(tier_medians))
                geometric = float(np.exp(np.mean(np.log(tier_medians))))
                row.append(f"{_number(arithmetic)} / {_number(geometric)}")
        average_rows.append(row)
    sections.extend(
        [
            "## Average over tiers",
            "",
            "Each cell is arithmetic mean / geometric mean of the three tier-level "
            "medians, giving every tier equal weight.",
            "",
            _markdown_table(["Method", *BENCHMARK_ORDER], average_rows),
            "",
        ]
    )
    (output / "generalization_all_tiers.md").write_text(
        "\n".join(sections), encoding="utf-8"
    )

    family = runs[runs.method.str.contains("_proposer__", regex=False, na=False)]
    family_rows = []
    for arm, group in family.groupby("method", sort=True):
        for benchmark in ("original_b1", "benchmark6"):
            subset = group[group.benchmark == benchmark]
            valid = subset.test_mse.dropna()
            family_rows.append(
                {
                    "arm": arm,
                    "benchmark": benchmark,
                    "complete": int(valid.count()),
                    "expected": 5,
                    "median_test_mse": float(valid.median()) if len(valid) else np.nan,
                    "min_test_mse": float(valid.min()) if len(valid) else np.nan,
                    "max_test_mse": float(valid.max()) if len(valid) else np.nan,
                    "mechanism_coverage": subset.mechanism_coverage.mean(),
                    "structural_validity": subset.structural_validity.mean(),
                    "judge_score": subset.judge_score.mean(),
                }
            )
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_csv(output / "family_by_benchmark.csv", index=False)
    family_md_rows = []
    for row in family_frame.itertuples(index=False):
        proposer, judge = row.arm.split("_proposer__")
        family_md_rows.append(
            [
                proposer.upper(),
                judge.replace("_judge", "").upper(),
                row.benchmark,
                f"{row.complete}/{row.expected}",
                "N/A" if np.isnan(row.median_test_mse) else (
                    f"{_number(row.median_test_mse)} "
                    f"[{_number(row.min_test_mse)}, {_number(row.max_test_mse)}]"
                ),
                (
                    "N/A"
                    if np.isnan(row.structural_validity)
                    else f"{row.structural_validity:.3f}"
                ),
                "N/A" if np.isnan(row.judge_score) else f"{row.judge_score:.3f}",
            ]
        )
    (output / "family_by_benchmark.md").write_text(
        "# LLM family study by benchmark\n\n"
        + _markdown_table(
            [
                "Proposer",
                "Judge",
                "Benchmark",
                "Complete",
                "Median [min, max] test NMSE",
                "Structural validity",
                "Judge score",
            ],
            family_md_rows,
        )
        + "\n",
        encoding="utf-8",
    )

    equation_rows = []
    for row in runs.itertuples(index=False):
        if row.benchmark not in {
            "original_b1",
            "perturbed_b1",
            "obfuscated_original_case01",
            "obfuscated_perturbed_case01",
        } or row.tier != "hard" or row.method not in METHOD_ORDER:
            continue
        equations = _extract_equations(Path(row.source))
        equation_rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "seed": row.seed,
                "test_mse": row.test_mse,
                "equations_json": json.dumps(equations, sort_keys=True),
                "source": row.source,
            }
        )
    equation_frame = pd.DataFrame(equation_rows)
    equation_frame.to_csv(output / "dalla_man_hard_selected_equations.csv", index=False)
    lines = [
        "# Selected hard-tier Dalla Man equations",
        "",
        "These are frozen selected equations for manual dynamics review. Baseline "
        "equations are RHS predictors; Autoformalism entries explicitly label state "
        "derivatives and algebraic processes.",
    ]
    for row in equation_frame.itertuples(index=False):
        lines.extend(
            [
                "",
                f"## {DISPLAY[row.method]} — {row.benchmark} — seed {row.seed}",
                "",
                f"Test NMSE: {_number(row.test_mse)}",
                "",
                "```text",
            ]
        )
        for name, expression in json.loads(row.equations_json).items():
            lines.append(f"{name} = {expression}")
        lines.extend(["```", ""])
    (output / "dalla_man_hard_selected_equations.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
