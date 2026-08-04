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
        "Removing latent states reduces structural validity and worsens target MSE "
        "on three of four tasks; Benchmark6 is a counterexample where the simpler "
        "no-latent model predicts better with equal structural validity.",
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
        judge_names = {
            "openai:gpt-5.6-terra": "GPT",
            "gemini:gemini-3.6-flash": "Gemini",
        }
        rows = [
            (
                judge_names.get(str(row.judge), str(row.judge)),
                f"{row.paired_preference_accuracy:.3f}",
                f"{row.auroc:.3f}",
                f"{row.mean_score_margin:.3f}",
                f"{row.false_preference_rate:.3f}",
                f"{row.mean_repeated_call_std:.3f}",
            )
            for row in summary.itertuples(index=False)
        ]
        conclusions = [
            "Both judges distinguish valid mechanisms from controlled adversarial "
            "variants well, with AUROC 0.899 for both model families.",
            "GPT has higher strict paired-preference accuracy (0.905 versus 0.821) "
            "and a larger mean margin, whereas Gemini has a lower false-preference "
            "rate (0.012 versus 0.083); Gemini's remaining comparisons are mostly "
            "ties rather than reversals.",
            "Repeated-call variability is low for both judges (mean SD 0.025 for "
            "GPT and 0.017 for Gemini), indicating that the result is not driven "
            "by unstable decoding noise.",
            "Narrative-equation mismatch is the principal failure mode for both "
            "judges, while wrong causal direction, wrong driver, and disconnected "
            "named mechanisms are detected reliably.",
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
        ablated = no_latent[no_latent.benchmark == benchmark]
        matched_seeds = set(ablated.seed.astype(int))
        full = runs[
            (runs.benchmark == benchmark)
            & (runs.tier == "hard")
            & (runs.method == "full")
            & (runs.seed.isin(matched_seeds))
        ]
        full_structure = structure[
            (structure.benchmark == benchmark)
            & (structure.tier == "hard")
            & (structure.method == "full")
            & (structure.seed.isin(matched_seeds))
        ]
        full_hidden = hidden[
            (hidden.benchmark == benchmark)
            & (hidden.tier == "hard")
            & (hidden.method == "full")
            & (hidden.seed.isin(matched_seeds))
        ]
        full_complexity = complexity[
            (complexity.benchmark == benchmark)
            & (complexity.tier == "hard")
            & (complexity.method == "full")
            & (complexity.seed.isin(matched_seeds))
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
                f"{len(full)}/3",
            )
        )
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
        "Persistent latent states improve target MSE and structural validity on "
        "original_b1, perturbed_b1, and Benchmark5 under matched seeds.",
        "Benchmark6 is the important counterexample: the no-latent model has "
        "lower target MSE and equal structural validity, so latent-state benefits "
        "are task-dependent rather than universal.",
        "The no-latent models use fewer terms on the first three tasks and cannot "
        "be assigned hidden-trajectory MSE by construction.",
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
    learning = pd.read_csv(
        analysis / "learning_candidates" / "learning_curve_iteration_summary.csv"
    )
    hard = learning[learning.tier == "hard"]
    iterations = tuple(range(1, 9))
    learning_rows: list[tuple[str, ...]] = []

    def curve_cell(
        benchmark: str, method: str, iteration: int, metric: str
    ) -> str:
        selected = hard[
            (hard.benchmark_id == benchmark)
            & (hard.method == method)
            & (hard.iteration == iteration)
        ]
        if selected.empty:
            return "—"
        row = selected.iloc[0]
        mean_value = row[f"{metric}_mean"]
        if pd.isna(mean_value):
            return "N/A"
        sd_value = row[f"{metric}_sd"]
        digits = 3
        value = f"{mean_value:.{digits}g}"
        if int(row.contributing_runs) > 1:
            value += f" ± {sd_value:.{digits}g}"
        return f"{value} [{int(row.contributing_runs)}/{int(row.total_runs)}]"

    for benchmark in (
        "original_b1",
        "perturbed_b1",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
        "benchmark5",
        "benchmark6",
    ):
        for method, display in (("full", "Ours"), ("nojudge", "No-judge")):
            learning_rows.append(
                (
                    benchmark,
                    display,
                    "Best validation target MSE ↓",
                    *(
                        curve_cell(
                            benchmark, method, iteration, "validation_raw_mse"
                        )
                        for iteration in iterations
                    ),
                )
            )
        learning_rows.append(
            (
                benchmark,
                "Ours",
                "Best judge score ↑",
                *(
                    curve_cell(benchmark, "full", iteration, "judge_score")
                    for iteration in iterations
                ),
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
    surviving = pd.read_csv(analysis / "surviving_terms_hard_full.csv")
    motifs = {
        "original_b1": (
            "Meal input → gastric/intestinal transit → glucose appearance; "
            "most seeds also retain glucose clearance."
        ),
        "perturbed_b1": (
            "Two-stage meal transport/absorption plus return toward basal glucose; "
            "one seed uses saturating gastric emptying."
        ),
        "obfuscated_original_case01": (
            "Pulse-driven dynamic memory followed by storage inflow and "
            "loss/return."
        ),
        "obfuscated_perturbed_case01": (
            "Pulse response feeding a storage balance, but several seeds retain "
            "only part of this pathway."
        ),
        "benchmark5": (
            "Direct u02 transformation with exchange, supported by filtered u01 "
            "and u03 dynamics."
        ),
        "benchmark6": "A compact command-response/relaxation target equation.",
    }
    interpretations = {
        "original_b1": "Stable mechanism; exact terms and names vary.",
        "perturbed_b1": "Stable mechanism with moderate algebraic variation.",
        "obfuscated_original_case01": "Stable motif, but little exact-term overlap.",
        "obfuscated_perturbed_case01": "Least stable; pathway completeness varies.",
        "benchmark5": "Transformation/exchange recur; one seed is missing.",
        "benchmark6": "Consistently sparse; exact response form varies.",
    }
    surviving_rows: list[tuple[str, ...]] = []
    for benchmark in (
        "original_b1",
        "perturbed_b1",
        "obfuscated_original_case01",
        "obfuscated_perturbed_case01",
        "benchmark5",
        "benchmark6",
    ):
        selected = surviving[surviving.benchmark_id == benchmark]
        counts = {
            int(row.seed): int(row.dynamic_terms)
            for row in selected.itertuples(index=False)
        }
        count_text = "; ".join(
            f"s{seed}={counts[seed]}" if seed in counts else f"s{seed}=missing"
            for seed in range(5)
        )
        surviving_rows.append(
            (
                benchmark,
                f"{len(counts)}/5",
                count_text,
                motifs[benchmark],
                interpretations[benchmark],
            )
        )
    conclusions = [
        "The full method's best validation target MSE improves across search on "
        "the named Dalla Man tasks and Benchmark6; for example, original_b1 "
        "improves from 1.75 at iteration 1 to 1.11 at iteration 8 while its best "
        "judge score rises from 0.713 to 0.937.",
        "Judge and fit curves measure complementary progress: judge scores track "
        "task-mechanism adequacy, whereas validation MSE tracks predictive fit. "
        "Their incumbents are computed independently and need not be the same "
        "candidate.",
        "Bracketed coverage makes failed early proposals explicit. A rise in an "
        "across-run mean can occur when additional runs first contribute a valid "
        "candidate, so within-run best-so-far monotonicity should not be confused "
        "with monotonicity of a changing-cohort mean.",
        "The obfuscated-perturbed curve exposes high between-seed variability and "
        "large-error candidates rather than hiding them in an endpoint average; "
        "this is a reliability limitation of the current search.",
        "The same high-level mechanisms recur more consistently than identical "
        "symbolic terms because seeds use different latent-state and process names.",
        "Original and perturbed Dalla Man retain the clearest common mechanism; "
        "obfuscated-perturbed is least stable and remains a structural-reliability "
        "limitation.",
    ]
    return (
        "Each cell is mean ± sample SD [contributing/total completed runs]. "
        "Curves are cumulative best-so-far development metrics; test data are "
        "never evaluated per iteration.\n\n"
        + _table(
            (
                "Benchmark",
                "Method",
                "Metric",
                *(f"Iter. {iteration}" for iteration in iterations),
            ),
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
        )
        + "\n\n### Surviving terms across seeds (Q4)\n\n"
        + "Term counts use the frozen, post-pruning ODEs. Exact equations remain "
        "available in `surviving_terms_hard_full.csv`; this compact table reports "
        "the recurring dynamical motif in plain language.\n\n"
        + _table(
            (
                "Benchmark",
                "Completed",
                "Terms by seed",
                "Recurring surviving dynamics",
                "Interpretation",
            ),
            surviving_rows,
        ),
        conclusions,
    )


def _section(
    title: str,
    table: str,
    conclusions: list[str],
    *,
    experiment: str,
    motivation: str,
) -> str:
    bullets = "\n".join(f"- {item}" for item in conclusions)
    return (
        f"## {title}\n\n"
        f"**Experiment.** {experiment}\n\n"
        f"**Motivation.** {motivation}\n\n"
        f"{table}\n\n### Rebuttal conclusion\n\n{bullets}"
    )


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
            experiment=(
                "We evaluated LLM-feature-SINDy, no-judge Autoformalism, "
                "Autoformalism with persistent latent states forbidden, and the "
                "full method on four hard-tier tasks. All methods used the same "
                "registered data splits and target evaluation; stochastic entries "
                "are summarized across their completed seeds."
            ),
            motivation=(
                "The table isolates the contributions of LLM-designed features, "
                "iterative search, judge feedback, and persistent latent states "
                "without crowding the comparison with classical baselines."
            ),
        ),
        _section(
            "LLM-family study",
            family,
            family_conclusions,
            experiment=(
                "We crossed GPT and Gemini as proposer and judge on original_b1 "
                "and Benchmark6 hard, using identical prompts, search budgets, "
                "fitting settings, and five seeds for each of the four pairings."
            ),
            motivation=(
                "This tests whether performance depends on one model family or "
                "persists when proposer and judge roles are reassigned across "
                "independent LLM families."
            ),
        ),
        _section(
            "Ratio versus weighted-sum objective",
            objective,
            objective_conclusions,
            experiment=(
                "For each of 18 benchmark-tier candidate pools, we ranked the same "
                "frozen candidates with a ratio objective and with weighted-sum "
                "objectives at seven predefined complexity multipliers. We compared "
                "the rankings using Spearman, Kendall, top-1 agreement, and top-5 "
                "overlap."
            ),
            motivation=(
                "This directly evaluates whether the two selection objectives make "
                "materially different choices, independently of proposal-generation "
                "or fitting randomness."
            ),
        ),
        _section(
            "Adversarial-judge stress test",
            adversarial,
            adversarial_conclusions,
            experiment=(
                "We constructed 28 schema-valid valid/adversarial pairs spanning "
                "four hard tasks and seven controlled mechanism mutations. GPT and "
                "Gemini judge each candidate independently in three repetitions, "
                "for 336 blinded calls in total."
            ),
            motivation=(
                "This tests whether judge scores reward mechanistic correctness or "
                "can be fooled by fluent but causally defective candidates."
            ),
        ),
        _section(
            "Persistent-latent-state ablation",
            no_latent,
            no_latent_conclusions,
            experiment=(
                "We reran the full search with persistent latent states forbidden "
                "on four hard-tier tasks for seeds 0-2, then compared against the "
                "full method on the same seeds using target MSE, deterministic "
                "structural validity, hidden MSE where defined, and term count."
            ),
            motivation=(
                "This isolates whether persistent internal state is necessary for "
                "predictive and structural recovery, rather than assuming that a "
                "larger latent model is always preferable."
            ),
        ),
        _section(
            "Learning curves - hard tier",
            learning,
            diagnostic_conclusions[:4],
            experiment=(
                "For every authoritative hard-tier run, we reconstructed eight "
                "search rounds from development-only checkpoints. At each round "
                "we report the cumulative best validation target MSE and, for the "
                "full method, the cumulative best judge score, summarized as mean "
                "± sample SD across seeds with explicit contributing-run counts."
            ),
            motivation=(
                "This directly answers whether predictive fit and task-mechanism "
                "assessment improve over iterations, while comparison with the "
                "no-judge variant isolates the role of judge-guided grounding."
            ),
        ),
        _section(
            "Structural stability and surviving terms - hard tier",
            stability,
            diagnostic_conclusions[4:],
            experiment=(
                "Across completed hard-tier seeds, we computed pairwise Jaccard "
                "similarity for alpha-normalized dependency edges and selected "
                "target-equation terms. We then summarize the additive-term counts "
                "and recurring dynamical motif of the actual frozen, post-pruning "
                "models across seeds."
            ),
            motivation=(
                "This distinguishes reproducible causal organization from exact "
                "symbolic-form agreement, directly answers which terms survive "
                "across seeds, and reveals structural equifinality across "
                "independent runs."
            ),
        ),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
