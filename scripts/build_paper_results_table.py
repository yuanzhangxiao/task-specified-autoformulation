"""Build paper-format benchmark-tier tables from consolidated evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = (
    "sindy",
    "pysr",
    "d3_native_no_tools",
    "full",
)
DISPLAY = {
    "sindy": "SINDy",
    "pysr": "PySR",
    "d3_native_no_tools": "D3",
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
BENCHMARK_DISPLAY = {
    "original_b1": "Original Dalla Man",
    "perturbed_b1": "Perturbed Dalla Man",
    "obfuscated_original_case01": "Obfuscated original",
    "obfuscated_perturbed_case01": "Obfuscated perturbed",
    "benchmark5": "Benchmark 5",
    "benchmark6": "Benchmark 6",
}
SPEC_NA = {"sindy", "pysr"}


def _summary(values: pd.Series, *, deterministic: bool = False) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return "N/A"
    if deterministic or len(clean) == 1:
        return f"{clean.iloc[0]:.4g}"
    return f"{clean.mean():.4g} ± {clean.std(ddof=1):.3g}"


def _load_structural(selected: Path, llm_feature: Path) -> pd.DataFrame:
    return pd.concat(
        (pd.read_csv(selected), pd.read_csv(llm_feature)), ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--selected-structural", type=Path, required=True)
    parser.add_argument("--llm-feature-structural", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    observed = pd.read_csv(args.observed)
    structural = _load_structural(
        args.selected_structural, args.llm_feature_structural
    )
    hidden = pd.read_csv(args.hidden)
    rows: list[dict[str, object]] = []
    markdown = [
        "# Paper-format benchmark results",
        "",
        "Values are mean ± sample SD across stochastic seeds; deterministic "
        "single runs are shown as one value. Obs. MSE is the raw test MSE "
        "under the registered benchmark evaluation protocol. Spec. success is "
        "the mean graph-predicate structural-validity score. Hidden MSE is "
        "test NMSE after choosing and affine-aligning a generated latent/process "
        "coordinate on training data only. D3 algebraic processes are evaluated "
        "teacher-forced on observed states; Ours uses its simulated latent/process "
        "trajectory. N/A means the method does not "
        "produce an internally interpretable mechanism/state representation.",
        "",
        "| Benchmark | Tier | Method | Obs. MSE ↓ | Spec. success ↑ | "
        "Hidden MSE ↓ | # Terms ↓ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for benchmark in BENCHMARKS:
        for tier_index, tier in enumerate(("easy", "medium", "hard")):
            for method_index, method in enumerate(METHODS):
                obs = observed[
                    (observed.benchmark == benchmark)
                    & (observed.tier == tier)
                    & (observed.method == method)
                ]
                # The sentinel denotes an invalid rollout, not a poor finite fit.
                valid_obs = obs[obs.one_step_normalized_mse < 1e11]
                struct = structural[
                    (structural.benchmark == benchmark)
                    & (structural.tier == tier)
                    & (structural.method == method)
                ]
                hidden_group = hidden[
                    (hidden.benchmark == benchmark)
                    & (hidden.tier == tier)
                    & (hidden.method == method)
                ]
                deterministic = method == "sindy"
                obs_text = _summary(
                    valid_obs.one_step_raw_mse, deterministic=deterministic
                )
                terms_text = _summary(
                    valid_obs.dynamic_terms, deterministic=deterministic
                )
                if obs.empty:
                    obs_text = "Failed"
                    terms_text = "—"
                spec_text = (
                    "N/A"
                    if method in SPEC_NA
                    else _summary(struct.structural_validity)
                )
                hidden_text = _summary(hidden_group.hidden_mse)
                record = {
                    "benchmark": benchmark,
                    "tier": tier,
                    "method": method,
                    "observed_raw_mse": obs_text,
                    "structural_validity": spec_text,
                    "hidden_affine_aligned_test_nmse": hidden_text,
                    "dynamic_terms": terms_text,
                }
                rows.append(record)
                benchmark_text = (
                    BENCHMARK_DISPLAY[benchmark]
                    if tier_index == 0 and method_index == 0
                    else ""
                )
                tier_text = tier.title() if method_index == 0 else ""
                markdown.append(
                    "| "
                    + " | ".join(
                        (
                            benchmark_text,
                            tier_text,
                            DISPLAY[method],
                            obs_text,
                            spec_text,
                            hidden_text,
                            terms_text,
                        )
                    )
                    + " |"
                )
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root / "paper_format_table.csv", index=False)
    (args.output_root / "paper_format_table.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
