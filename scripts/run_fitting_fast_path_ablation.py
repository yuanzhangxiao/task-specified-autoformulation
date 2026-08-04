"""Compare derivative fast-path and generic rollout fits on frozen structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic

import pandas as pd

from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import FitConfig, evaluate_fitted_candidate, fit_candidate
from autoformalism.schemas import CandidateModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", default="benchmark6")
    parser.add_argument("--tier", default="hard")
    parser.add_argument("--max-nfev", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    runs = pd.read_csv(args.runs)
    runs = runs[
        (runs.method == "full")
        & (runs.benchmark == args.benchmark)
        & (runs.tier == args.tier)
        & runs.test_mse.notna()
    ]
    loader = BenchmarkLoader(BenchmarkRegistry())
    config = DataConfig(
        root=args.data_root.resolve(),
        benchmark_id=args.benchmark,
        tier=args.tier,
    )
    development = loader.load_development(config)
    test = loader.load_test(config)
    rows = []

    def checkpoint() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output, index=False)
        if frame.empty:
            return
        summary = (
            frame.groupby("variant", as_index=False)
            .agg(
                completed=("seed", "count"),
                success_rate=("success", "mean"),
                elapsed_seconds=("elapsed_seconds", "mean"),
                function_evaluations=("function_evaluations", "mean"),
                validation_nmse=("validation_nmse", "mean"),
                test_nmse=("test_nmse", "mean"),
            )
            .sort_values("variant")
        )
        lines = [
            "# Frozen-structure fitting ablation",
            "",
            "| Variant | Runs | Success ↑ | Seconds ↓ | NFEV ↓ | "
            "Validation NMSE ↓ | Test NMSE ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in summary.itertuples(index=False):
            lines.append(
                f"| {item.variant} | {item.completed} | {item.success_rate:.3f} | "
                f"{item.elapsed_seconds:.3g} | {item.function_evaluations:.3g} | "
                f"{item.validation_nmse:.4g} | {item.test_nmse:.4g} |"
            )
        args.output.with_suffix(".md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    for run in runs.itertuples(index=False):
        source = Path(run.source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if "final_fit" not in payload:
            continue
        candidate = CandidateModel.model_validate(payload["frozen"]["candidate"])
        arguments = ExecutionArguments(
            data_root=config.root,
            benchmark_id=args.benchmark,
            tier=args.tier,
            seed=int(run.seed),
            proposer_model=None,
            judge_model=None,
            iteration_budget=1,
            beam_size=1,
            output_root=args.output.parent,
            resume=False,
            dry_run=False,
            mock_llm=True,
            use_clean_observations=False,
        )
        model = compile_candidate(candidate, _context(arguments, development))
        warm = payload["final_fit"]["global_parameters"]
        for label, allow_fast_path in (
            ("derivative_fast_path", True),
            ("generic_rollout", False),
        ):
            settings = FitConfig(
                number_of_starts=1,
                random_seed=int(run.seed),
                integration_backend="fixed_rk4",
                fixed_step_substeps=4,
                allow_derivative_regression=allow_fast_path,
                maximum_function_evaluations=args.max_nfev,
                maximum_wall_time_seconds=args.timeout_seconds,
            )
            started = monotonic()
            fitted = fit_candidate(
                model,
                development.train,
                development.validation,
                settings,
                initial_global_parameters=warm,
            )
            elapsed = monotonic() - started
            test_mse = None
            if fitted.success:
                _, metrics = evaluate_fitted_candidate(
                    model,
                    test,
                    global_parameters=fitted.global_parameters,
                    global_initial_conditions=fitted.global_initial_conditions,
                    target_scales=fitted.target_scales,
                    config=settings,
                    fit_trajectory_initial_conditions=False,
                )
                test_mse = metrics.normalized_mse
            rows.append(
                {
                    "benchmark": args.benchmark,
                    "tier": args.tier,
                    "seed": int(run.seed),
                    "variant": label,
                    "success": fitted.success,
                    "elapsed_seconds": elapsed,
                    "function_evaluations": sum(
                        item.function_evaluations for item in fitted.diagnostics
                    ),
                    "backend": ";".join(
                        sorted({item.backend for item in fitted.diagnostics})
                    ),
                    "training_nmse": fitted.training_metrics.normalized_mse,
                    "validation_nmse": fitted.validation_metrics.normalized_mse,
                    "test_nmse": test_mse,
                    "message": fitted.message,
                    "source": run.source,
                }
            )
            checkpoint()
    checkpoint()


if __name__ == "__main__":
    main()
