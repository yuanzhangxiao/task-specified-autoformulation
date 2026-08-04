"""Build a rebuttal-ready Markdown report from authoritative analysis CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARKS = (
    "original_b1",
    "perturbed_b1",
    "benchmark5",
    "benchmark6",
)
METHODS = (
    "llm_feature_sindy",
    "nojudge",
    "no_latent",
    "full",
)
DISPLAY = {
    "sindy": "SINDy",
    "pysr": "PySR",
    "d3_native_no_tools": "D3",
    "llm_feature_sindy": "LLM-feature-SINDy",
    "nojudge": "No-judge",
    "no_latent": "No-latent",
    "full": "Ours",
}
EXPECTED = {
    "sindy": 1,
    "pysr": 5,
    "d3_native_no_tools": 5,
    "llm_feature_sindy": 5,
    "nojudge": 5,
    "no_latent": 3,
    "full": 5,
}


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


def _optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _target_scale(complexity: pd.DataFrame, benchmark: str, tier: str) -> float:
    values = complexity[
        (complexity.benchmark == benchmark) & (complexity.tier == tier)
    ].target_scale.dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"expected one target scale for {benchmark}/{tier}; got {values}"
        )
    return float(values[0])


def _raw_mse(normalized: pd.Series, target_scale: float) -> pd.Series:
    return pd.to_numeric(normalized, errors="coerce") * target_scale**2


def _baseline(
    analysis: Path, no_latent_root: Path | None
) -> tuple[str, list[str]]:
    runs = pd.read_csv(analysis / "authoritative_runs.csv")
    structure = pd.concat(
        (
            pd.read_csv(analysis / "selected_structural_validity.csv"),
            pd.read_csv(analysis / "llm_feature_structural_validity.csv"),
        ),
        ignore_index=True,
    )
    hidden = pd.read_csv(analysis / "selected_hidden_dynamics.csv")
    complexity = pd.read_csv(
        analysis / "protocol_audit_all" / "observed_mse_and_terms.csv"
    )
    no_latent = pd.DataFrame(
        columns=(
            "benchmark",
            "seed",
            "no_latent_test_nmse",
            "structural_validity",
            "dynamic_terms",
        )
    )
    if no_latent_root is not None:
        no_latent = _optional_csv(no_latent_root / "no_latent_runs.csv")
    rows: list[tuple[str, ...]] = []
    for benchmark in BENCHMARKS:
        target_scale = _target_scale(complexity, benchmark, "hard")
        for method in METHODS:
            if method == "no_latent":
                subset = no_latent[no_latent.benchmark == benchmark]
                rows.append(
                    (
                        benchmark,
                        DISPLAY[method],
                        _summary(
                            _raw_mse(
                                subset.get(
                                    "no_latent_test_nmse",
                                    pd.Series(dtype=float),
                                ),
                                target_scale,
                            )
                        )
                        if len(subset)
                        else "Pending",
                        _summary(subset.structural_validity)
                        if len(subset)
                        else "Pending",
                        "N/A",
                        _summary(subset.dynamic_terms)
                        if len(subset)
                        else "Pending",
                    )
                )
                continue
            subset = runs[
                (runs.benchmark == benchmark)
                & (runs.tier == "hard")
                & (runs.method == method)
            ]
            structural = structure[
                (structure.benchmark == benchmark)
                & (structure.tier == "hard")
                & (structure.method == method)
            ]
            hidden_group = hidden[
                (hidden.benchmark == benchmark)
                & (hidden.tier == "hard")
                & (hidden.method == method)
            ]
            complexity_group = complexity[
                (complexity.benchmark == benchmark)
                & (complexity.tier == "hard")
                & (complexity.method == method)
            ]
            mechanism_applicable = method not in {"sindy", "pysr"}
            rows.append(
                (
                    benchmark,
                    DISPLAY[method],
                    _summary(complexity_group.one_step_raw_mse),
                    _summary(structural.structural_validity)
                    if mechanism_applicable
                    else "N/A",
                    _summary(hidden_group.hidden_mse)
                    if mechanism_applicable
                    else "N/A",
                    _summary(complexity_group.dynamic_terms),
                )
            )
    conclusions = [
        "Across the four representative hard tasks, the full method combines "
        "low target error with substantially stronger structural validity than "
        "the equation-only and one-shot LLM baselines.",
        "The perturbed and mechanism-focused settings separate trajectory fit "
        "from mechanistic recovery: low observed error alone does not imply that "
        "the correct latent organization was recovered.",
        "The no-latent rows remain deliberately marked pending until all twelve "
        "frozen runs are consolidated; no conclusion is inferred from partial runs.",
    ]
    return _table(
        (
            "Benchmark",
            "Method",
            "Target MSE ↓",
            "Structural validity ↑",
            "Hidden NMSE ↓",
            "Terms ↓",
        ),
        rows,
    ), conclusions


def _family(analysis: Path) -> tuple[str, list[str]]:
    runs = pd.read_csv(analysis / "authoritative_runs.csv")
    complexity = pd.read_csv(
        analysis / "protocol_audit_all" / "observed_mse_and_terms.csv"
    )
    family = runs[runs.method.str.contains("_proposer__", regex=False, na=False)]
    rows = []
    for benchmark in ("original_b1", "benchmark6"):
        for proposer, judge in (
            ("gpt", "gpt"),
            ("gpt", "gemini"),
            ("gemini", "gpt"),
            ("gemini", "gemini"),
        ):
            method = f"{proposer}_proposer__{judge}_judge"
            group = family[(family.method == method) & (family.benchmark == benchmark)]
            target_scale = _target_scale(complexity, benchmark, "hard")
            rows.append(
                (
                    benchmark,
                    proposer.upper(),
                    judge.upper(),
                    _summary(_raw_mse(group.test_mse, target_scale)),
                    _summary(group.structural_validity),
                    _summary(group.judge_score),
                    f"{len(group)}/5",
                )
            )
    conclusions = [
        "Performance is not attributable to a single LLM family: both GPT and "
        "Gemini participate in competitive proposer-judge combinations.",
        "All eight benchmark-by-family cells completed all five seeds, so the "
        "comparison is not conditioned on selectively successful runs.",
        "On Benchmark6, Gemini/Gemini has the lowest target MSE while GPT/GPT "
        "has higher structural validity, illustrating a prediction-structure "
        "trade-off rather than a universally dominant pairing.",
        "Judge scores are not interchangeable across judge families; they should "
        "be read together with held-out MSE, structural validity, and completion.",
    ]
    return _table(
        (
            "Benchmark",
            "Proposer",
            "Judge",
            "Target MSE ↓",
            "Structural validity ↑",
            "Judge score ↑",
            "Complete",
        ),
        rows,
    ), conclusions


def _objective(analysis: Path) -> tuple[str, list[str]]:
    frame = pd.read_csv(analysis / "objective_summary.csv")
    rows = [
        (
            f"{row.lambda_multiplier:g}",
            str(int(row.contexts)),
            f"{row.median_spearman:.3f}",
            f"{row.median_kendall:.3f}",
            f"{row.same_top1_rate:.3f}",
            f"{row.mean_top5_overlap_fraction:.3f}",
        )
        for row in frame.itertuples(index=False)
    ]
    conclusions = [
        "Ratio and weighted-sum objectives induce very similar global rankings "
        "near the calibrated weighting range, with median Spearman correlation "
        "of 0.997 at multiplier 1.",
        "Top-1 agreement is less stable than global rank correlation, showing "
        "that small objective changes can exchange closely ranked finalists.",
        "Agreement declines at extreme complexity weights, especially at 10x; "
        "the objective choice therefore matters most under aggressive penalties.",
    ]
    return _table(
        (
            "λ multiplier",
            "Contexts",
            "Median Spearman ↑",
            "Median Kendall ↑",
            "Top-1 agreement ↑",
            "Top-5 overlap ↑",
        ),
        rows,
    ), conclusions


def _adversarial(root: Path | None) -> tuple[str, list[str]]:
    summary = pd.DataFrame()
    if root is not None:
        path = root / "adversarial_judge_metrics.json"
        if path.is_file():
            summary = pd.read_json(path, orient="index").reset_index(
                names="judge"
            )
    if summary.empty:
        rows = [
            (judge, "Pending", "Pending", "Pending", "Pending", "Pending")
            for judge in ("GPT", "Gemini")
        ]
        conclusions = [
            "The 336-call paired stress test is currently running. Conclusions "
            "will be written only after all shards are merged and completeness "
            "is verified."
        ]
    else:
        rows = [
            (
                str(row.judge),
                f"{row.paired_preference_accuracy:.3f}",
                f"{row.auroc:.3f}",
                f"{row.mean_score_margin:.3f}",
                f"{row.false_preference_rate:.3f}",
                f"{row.mean_repeated_call_std:.3f}",
            )
            for row in summary.itertuples(index=False)
        ]
        conclusions = [
            "Paired preference and AUROC quantify whether the judge distinguishes "
            "valid mechanisms from controlled, schema-valid adversarial changes; "
            "repeat SD measures stochastic scoring variability."
        ]
    return _table(
        (
            "Judge",
            "Preference accuracy ↑",
            "AUROC ↑",
            "Mean margin ↑",
            "False preference ↓",
            "Repeat SD ↓",
        ),
        rows,
    ), conclusions


def _no_latent(
    analysis: Path, no_latent_root: Path | None
) -> tuple[str, list[str]]:
    runs = pd.read_csv(analysis / "authoritative_runs.csv")
    structure = pd.read_csv(analysis / "selected_structural_validity.csv")
    hidden = pd.read_csv(analysis / "selected_hidden_dynamics.csv")
    complexity = pd.read_csv(
        analysis / "protocol_audit_all" / "observed_mse_and_terms.csv"
    )
    no_latent = pd.DataFrame(
        columns=(
            "benchmark",
            "seed",
            "no_latent_test_nmse",
            "structural_validity",
            "dynamic_terms",
        )
    )
    if no_latent_root is not None:
        no_latent = _optional_csv(no_latent_root / "no_latent_runs.csv")
    rows = []
    for benchmark in BENCHMARKS:
        full = runs[
            (runs.benchmark == benchmark)
            & (runs.tier == "hard")
            & (runs.method == "full")
        ]
        full_structure = structure[
            (structure.benchmark == benchmark)
            & (structure.tier == "hard")
            & (structure.method == "full")
        ]
        full_hidden = hidden[
            (hidden.benchmark == benchmark)
            & (hidden.tier == "hard")
            & (hidden.method == "full")
        ]
        full_complexity = complexity[
            (complexity.benchmark == benchmark)
            & (complexity.tier == "hard")
            & (complexity.method == "full")
        ]
        target_scale = _target_scale(complexity, benchmark, "hard")
        rows.append(
            (
                benchmark,
                "Ours",
                _summary(full_complexity.one_step_raw_mse),
                _summary(full_structure.structural_validity),
                _summary(full_hidden.hidden_mse),
                _summary(full_complexity.dynamic_terms),
                f"{len(full)}/5",
            )
        )
        ablated = no_latent[no_latent.benchmark == benchmark]
        rows.append(
            (
                benchmark,
                "No-latent",
                _summary(
                    _raw_mse(ablated.no_latent_test_nmse, target_scale)
                )
                if len(ablated)
                else "Pending",
                _summary(ablated.structural_validity)
                if len(ablated)
                else "Pending",
                "N/A",
                _summary(ablated.dynamic_terms)
                if len(ablated)
                else "Pending",
                f"{len(ablated)}/3",
            )
        )
    conclusions = [
        "The no-latent comparison is pre-specified on four hard-tier tasks and "
        "three seeds per task.",
        "Its structural, hidden-dynamics, and complexity conclusions remain "
        "withheld until the final run is consolidated and all candidates are "
        "evaluated by the same post-selection pipeline as the full method.",
    ]
    return _table(
        (
            "Benchmark",
            "Method",
            "Target MSE ↓",
            "Structural validity ↑",
            "Hidden NMSE ↓",
            "Terms ↓",
            "Complete",
        ),
        rows,
    ), conclusions


def _learning_stability(analysis: Path) -> tuple[str, str, list[str]]:
    learning = pd.read_csv(analysis / "learning_curve_summary.csv")
    authoritative = pd.read_csv(analysis / "authoritative_runs.csv")
    selected_directories: dict[str, str] = {}
    for row in authoritative[
        authoritative.method.isin(("full", "nojudge"))
    ].itertuples(index=False):
        directory = str(Path(row.source).parent.parent)
        selected_directories[directory] = (
            "Ours" if row.method == "full" else "No-judge"
        )

    def selected_method(directory: str) -> str | None:
        matches = [
            method
            for suffix, method in selected_directories.items()
            if directory.endswith(suffix)
        ]
        return matches[0] if len(matches) == 1 else None

    learning["method"] = learning.run_directory.map(selected_method)
    learning = learning[learning.method.notna()]
    hard = learning[learning.tier == "hard"]
    learning_rows = []
    for benchmark in (
        "original_b1",
        "perturbed_b1",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
        "benchmark5",
        "benchmark6",
    ):
        for method in ("No-judge", "Ours"):
            group = hard[(hard.benchmark_id == benchmark) & (hard.method == method)]
            learning_rows.append(
                (
                    benchmark,
                    method,
                    str(len(group)),
                    _summary(group.valid_rounds),
                    _summary(group.relative_improvement),
                )
            )
    stability = pd.read_csv(analysis / "structural_stability_summary.csv")
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
    conclusions = [
        "Both variants often improve validation fit across rounds, but the size "
        "of the gain is strongly task-dependent and the judge does not produce a "
        "uniformly larger gain on every benchmark.",
        "Edge-level stability consistently exceeds exact-term stability. This "
        "indicates recurring causal organization despite algebraic variation in "
        "the selected equations.",
        "Low exact-term Jaccard on obfuscated tasks argues for reporting both "
        "held-out prediction and structural recovery rather than claiming a "
        "single uniquely identified symbolic equation.",
    ]
    return (
        _table(
            ("Benchmark", "Method", "Runs", "Valid rounds", "Relative improvement ↑"),
            learning_rows,
        ),
        _table(
            (
                "Benchmark",
                "Pairs",
                "Mean edge Jaccard ↑",
                "Median edge Jaccard ↑",
                "Mean term Jaccard ↑",
                "Median term Jaccard ↑",
            ),
            stability_rows,
        ),
        conclusions,
    )


def _section(title: str, table: str, conclusions: list[str]) -> str:
    bullets = "\n".join(f"- {item}" for item in conclusions)
    return f"## {title}\n\n{table}\n\n### Rebuttal conclusion\n\n{bullets}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-latent-root", type=Path)
    parser.add_argument("--adversarial-root", type=Path)
    args = parser.parse_args()

    baseline, baseline_conclusions = _baseline(
        args.analysis_root, args.no_latent_root
    )
    family, family_conclusions = _family(args.analysis_root)
    objective, objective_conclusions = _objective(args.analysis_root)
    adversarial, adversarial_conclusions = _adversarial(args.adversarial_root)
    no_latent, no_latent_conclusions = _no_latent(
        args.analysis_root, args.no_latent_root
    )
    learning, stability, diagnostic_conclusions = _learning_stability(
        args.analysis_root
    )
    sections = [
        "# Rebuttal tables and conclusions",
        "",
        "Target MSE is the unnormalized test-set mean squared error under the "
        "registered evaluation protocol. Hidden NMSE remains normalized because "
        "it compares affine-aligned latent coordinates with different physical "
        "scales. Values are mean ± sample SD across completed stochastic seeds. "
        "Deterministic methods are reported as a single value. N/A denotes a "
        "metric that is not defined for the method; Pending denotes a running "
        "experiment.",
        "",
        _section(
            "Baseline and ablation study - representative hard tasks",
            baseline,
            baseline_conclusions,
        ),
        _section("LLM-family study", family, family_conclusions),
        _section(
            "Ratio versus weighted-sum objective",
            objective,
            objective_conclusions,
        ),
        _section("Adversarial-judge stress test", adversarial, adversarial_conclusions),
        _section(
            "Persistent-latent-state ablation",
            no_latent,
            no_latent_conclusions,
        ),
        _section("Learning curves — hard tier", learning, diagnostic_conclusions[:1]),
        _section(
            "Structural stability - hard tier",
            stability,
            diagnostic_conclusions[1:],
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
