"""Audit raw one-step MSE, free-rollout MSE, and selected ODE term counts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from autoformalism.baselines.core import candidate_from_equations
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import ExecutionArguments, _context
from autoformalism.expressions import ModelValidationError, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.artifacts import _term_count
from autoformalism.schemas import CandidateModel

METHODS = (
    "persistence",
    "sindy",
    "pysr",
    "d3_native_no_tools",
    "llm_feature_sindy",
    "nojudge",
    "full",
)
DISPLAY = {
    "persistence": "Persistence",
    "sindy": "SINDy",
    "pysr": "PySR",
    "d3_native_no_tools": "D3-native-no-tools",
    "llm_feature_sindy": "LLM-feature-SINDy",
    "nojudge": "No-judge Autoformalism",
    "full": "Full method",
}


class _ParameterSubstitution(ast.NodeTransformer):
    def __init__(self, parameters: dict[str, float]) -> None:
        self._parameters = parameters

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id not in self._parameters:
            return node
        return ast.copy_location(ast.Constant(self._parameters[node.id]), node)


def _substitute_parameters(expression: str, parameters: dict[str, float]) -> str:
    tree = ast.parse(expression, mode="eval")
    replaced = _ParameterSubstitution(parameters).visit(tree)
    ast.fix_missing_locations(replaced)
    return ast.unparse(replaced)


def _symbolic_complexity(expressions: list[str]) -> tuple[int, int]:
    operation_types = (
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Compare,
        ast.BoolOp,
        ast.IfExp,
    )
    operations = 0
    tree_nodes = 0
    for expression in expressions:
        nodes = tuple(ast.walk(ast.parse(expression, mode="eval")))
        operations += sum(isinstance(node, operation_types) for node in nodes)
        tree_nodes += sum(
            isinstance(node, (*operation_types, ast.Name, ast.Constant))
            for node in nodes
        )
    return operations, tree_nodes


def _candidate_and_fit(
    path: Path, method: str
) -> tuple[CandidateModel | dict[str, str], dict[str, float], int, int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method in {"nojudge", "full"}:
        candidate = CandidateModel.model_validate(payload["frozen"]["candidate"])
        parameters = dict(payload["final_fit"]["global_parameters"])
        expressions = [item.rhs for item in candidate.state_equations]
        terms = sum(_term_count(item) for item in expressions)
        operations, tree_nodes = _symbolic_complexity(expressions)
        return candidate, parameters, terms, operations, tree_nodes

    equations = {str(k): str(v) for k, v in payload["equations"].items()}
    parameters: dict[str, float] = {}
    if method == "d3_native_no_tools":
        raw = payload.get("selected_hyperparameters", {}).get(
            "selected_parameters", "{}"
        )
        parameters = {
            str(key): float(value) for key, value in json.loads(raw).items()
        }
        equations = {
            name: _substitute_parameters(expression, parameters)
            for name, expression in equations.items()
        }
    terms = sum(_term_count(expression) for expression in equations.values())
    operations, tree_nodes = _symbolic_complexity(list(equations.values()))
    # Baseline equations have their fitted constants embedded after substitution.
    return equations, {}, terms, operations, tree_nodes


def _free_rollout(
    *,
    source: Path,
    method: str,
    development: object,
    test: object,
    arguments: ExecutionArguments,
    target: str,
    scale: float,
) -> tuple[float, float, str | None, int | None, int | None, int | None]:
    if method == "persistence":
        return np.nan, np.nan, "persistence is not an autonomous ODE", None, None, None
    try:
        candidate_or_equations, parameters, terms, operations, tree_nodes = (
            _candidate_and_fit(source, method)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return (
            np.nan,
            np.nan,
            f"missing frozen fit: {type(exc).__name__}: {exc}",
            None,
            None,
            None,
        )
    context = _context(arguments, development)
    candidate = (
        candidate_or_equations
        if isinstance(candidate_or_equations, CandidateModel)
        else candidate_from_equations(
            candidate_or_equations,
            context,
            identifier=f"free_rollout_{method}",
        )
    )
    try:
        model = compile_candidate(candidate, context)
    except (ModelValidationError, RuntimeError, TypeError, ValueError) as exc:
        return (
            np.nan,
            np.nan,
            f"{type(exc).__name__}: {exc}",
            terms,
            operations,
            tree_nodes,
        )
    squared: list[np.ndarray] = []
    settings = FitConfig(
        integration_backend="fixed_rk4",
        fixed_step_substeps=4,
        maximum_wall_time_seconds=30.0,
    )
    for trajectory in test.trajectories:
        simulation = simulate_trajectory(
            model,
            trajectory,
            parameters,
            {},
            settings,
            reset_observed_states=False,
        )
        if not simulation.success:
            return (
                np.nan,
                np.nan,
                simulation.message,
                terms,
                operations,
                tree_nodes,
            )
        error = simulation.predictions[target] - trajectory.targets[target]
        squared.append(error**2)
    raw = float(np.mean(np.concatenate(squared)))
    return raw, raw / scale**2, None, terms, operations, tree_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="*")
    args = parser.parse_args()

    runs = pd.read_csv(args.runs)
    runs = runs[runs.method.isin(METHODS)].copy()
    if args.benchmarks:
        runs = runs[runs.benchmark.isin(args.benchmarks)].copy()
    loader = BenchmarkLoader(BenchmarkRegistry())
    cache: dict[tuple[str, str], tuple[object, object, str, float]] = {}
    rows: list[dict] = []
    for row in runs.itertuples(index=False):
        key = (row.benchmark, row.tier)
        if key not in cache:
            config = DataConfig(
                root=args.data_root.resolve(),
                benchmark_id=row.benchmark,
                tier=row.tier,
            )
            development = loader.load_development(config)
            test = loader.load_test(config)
            if len(development.roles.targets) != 1:
                raise ValueError("audit currently requires one target per benchmark")
            target = development.roles.targets[0]
            scale = max(
                float(
                    np.std(
                        np.concatenate(
                            [
                                item.targets[target]
                                for item in development.train.trajectories
                            ]
                        )
                    )
                ),
                1e-8,
            )
            cache[key] = (development, test, target, scale)
        development, test, target, scale = cache[key]
        execution_arguments = ExecutionArguments(
            data_root=args.data_root.resolve(),
            benchmark_id=row.benchmark,
            tier=row.tier,
            seed=int(row.seed),
            proposer_model=None,
            judge_model=None,
            iteration_budget=1,
            beam_size=1,
            output_root=args.output_root,
            resume=False,
            dry_run=False,
            mock_llm=True,
            use_clean_observations=False,
        )
        free_raw, free_normalized, error, terms, operations, tree_nodes = (
            _free_rollout(
                source=Path(row.source),
                method=row.method,
                development=development,
                test=test,
                arguments=execution_arguments,
                target=target,
                scale=scale,
            )
        )
        rows.append(
            {
                "method": row.method,
                "benchmark": row.benchmark,
                "tier": row.tier,
                "seed": int(row.seed),
                "target_scale": scale,
                "one_step_normalized_mse": row.test_mse,
                "one_step_raw_mse": float(row.test_mse) * scale**2,
                "free_rollout_normalized_mse": free_normalized,
                "free_rollout_raw_mse": free_raw,
                "dynamic_terms": terms,
                "symbolic_operations": operations,
                "symbolic_tree_nodes": tree_nodes,
                "free_rollout_error": error,
            }
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_root / "observed_mse_and_terms.csv", index=False)
    _write_summary(frame, args.output_root)


def _summary_value(values: pd.Series) -> str:
    clean = values.dropna().astype(float)
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return "N/A"
    if len(clean) == 1:
        return f"{clean.iloc[0]:.4g}"
    return f"{clean.mean():.4g} ± {clean.std(ddof=1):.3g}"


def _write_summary(frame: pd.DataFrame, output_root: Path) -> None:
    records = []
    markdown = [
        "# Observed-trajectory protocol audit",
        "",
        "Values with multiple stochastic runs are mean ± sample SD; a single "
        "deterministic run is shown as one value. Free rollout uses the frozen "
        "one-step-fitted model with four fixed RK4 substeps per observation interval.",
    ]
    for benchmark in sorted(frame.benchmark.unique()):
        for tier in ("easy", "medium", "hard"):
            subset = frame[(frame.benchmark == benchmark) & (frame.tier == tier)]
            if subset.empty:
                continue
            markdown.extend(
                [
                    "",
                    f"## {benchmark} — {tier}",
                    "",
                    "| Method | Raw one-step MSE | Raw free-rollout MSE | "
                    "Dynamic terms | Symbolic tree size | Free valid |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for method in METHODS:
                group = subset[subset.method == method]
                if group.empty:
                    continue
                record = {
                    "method": method,
                    "benchmark": benchmark,
                    "tier": tier,
                    "one_step_raw_mse_mean": group.one_step_raw_mse.mean(),
                    "one_step_raw_mse_sd": group.one_step_raw_mse.std(ddof=1),
                    "one_step_n": group.one_step_raw_mse.count(),
                    "free_rollout_raw_mse_mean": group.free_rollout_raw_mse.mean(),
                    "free_rollout_raw_mse_sd": group.free_rollout_raw_mse.std(
                        ddof=1
                    ),
                    "free_rollout_n": group.free_rollout_raw_mse.count(),
                    "dynamic_terms_mean": group.dynamic_terms.mean(),
                    "dynamic_terms_sd": group.dynamic_terms.std(ddof=1),
                    "dynamic_terms_n": group.dynamic_terms.count(),
                    "symbolic_operations_mean": group.symbolic_operations.mean(),
                    "symbolic_operations_sd": group.symbolic_operations.std(ddof=1),
                    "symbolic_tree_nodes_mean": group.symbolic_tree_nodes.mean(),
                    "symbolic_tree_nodes_sd": group.symbolic_tree_nodes.std(ddof=1),
                }
                records.append(record)
                markdown.append(
                    "| "
                    + " | ".join(
                        (
                            DISPLAY[method],
                            _summary_value(group.one_step_raw_mse),
                            _summary_value(group.free_rollout_raw_mse),
                            _summary_value(group.dynamic_terms),
                            _summary_value(group.symbolic_tree_nodes),
                            f"{group.free_rollout_raw_mse.count()}/{len(group)}",
                        )
                    )
                    + " |"
                )
    pd.DataFrame(records).to_csv(output_root / "protocol_summary.csv", index=False)
    (output_root / "protocol_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
