#!/usr/bin/env python3
"""Run one leakage-safe baseline and write a JSON result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autoformalism.baselines.d3 import run_d3_no_tools
from autoformalism.baselines.models import BaselineConfig
from autoformalism.baselines.runner import run_baseline
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import _development_forcing_bounds, _parse_model
from autoformalism.expressions import ValidationContext
from autoformalism.llm import LLMConfig, create_llm_client


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline CLI parser for execution and argument tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "AUTOFORMALISM_DATA_ROOT", "data_raw")))
    parser.add_argument("--benchmark-id", default="original_b1")
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument("--method", required=True, choices=(
        "persistence", "sindy", "pysr", "llm_feature_sindy", "d3_no_tools"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/baselines"))
    parser.add_argument("--d3-command", nargs="+")
    parser.add_argument(
        "--pysr-iterations",
        type=int,
        default=40,
        help="Number of PySR evolutionary iterations per target (default: 40).",
    )
    parser.add_argument(
        "--maximum-expression-size",
        type=int,
        default=30,
        help="Maximum PySR expression-tree size (default: 30).",
    )
    parser.add_argument(
        "--d3-generations",
        type=int,
        default=20,
        help="Maximum number of D3 propose-fit-reflect generations (default: 20).",
    )
    parser.add_argument(
        "--d3-patience",
        type=int,
        default=20,
        help="Stop D3 after this many generations without improvement.",
    )
    return parser


def main() -> None:
    """Parse, execute, and persist one baseline run."""
    parser = build_parser()
    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    registry = BenchmarkRegistry()
    spec = registry.get(args.benchmark_id)
    loader = BenchmarkLoader(registry)
    data_config = DataConfig(root=root, benchmark_id=args.benchmark_id, tier=args.tier)
    dataset = loader.load_development(data_config)
    loader.validate_test_paths(data_config)
    context = ValidationContext(
        targets=dataset.roles.targets,
        auxiliaries=dataset.roles.auxiliaries,
        external_inputs=spec.external_inputs,
        fixed_covariates=spec.fixed_covariates,
        lagged_targets=dataset.roles.targets if spec.one_step_target_history else (),
        forcing_bounds=_development_forcing_bounds(dataset, include_targets=True),
    )
    prompt_path = root / spec.relative_root / args.tier / "proposer_prompt.txt"
    prompt = prompt_path.read_text(encoding="utf-8")
    config = BaselineConfig(
        method=args.method,
        seed=args.seed,
        llm_model=args.model,
        pysr_iterations=args.pysr_iterations,
        maximum_expression_size=args.maximum_expression_size,
        d3_generations=args.d3_generations,
        d3_patience=args.d3_patience,
    )
    output = args.output_root.resolve() / args.method / (
        f"{args.benchmark_id}_{args.tier}_seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.method == "d3_no_tools":
        client = None
        if not args.d3_command:
            if not args.model:
                parser.error("--model or --d3-command is required for d3_no_tools")
            provider, model = _parse_model(args.model)
            client = create_llm_client(LLMConfig(
                provider=provider,
                model=model,
                cache_directory=output / "llm_cache",
                log_path=output / "llm_events.jsonl",
                proposal_target_channels=dataset.roles.targets,
                timeout_seconds=900.0,
            ))
        result = run_d3_no_tools(
            config, dataset, lambda: loader.load_test(data_config), context,
            task_prompt=prompt, command=args.d3_command or (), work_directory=output,
            llm_client=client,
        )
    else:
        client = None
        if args.method == "llm_feature_sindy":
            if not args.model:
                parser.error("--model is required for llm_feature_sindy")
            provider, model = _parse_model(args.model)
            client = create_llm_client(LLMConfig(
                provider=provider,
                model=model,
                cache_directory=output / "llm_cache",
                log_path=output / "llm_events.jsonl",
                proposal_target_channels=dataset.roles.targets,
                timeout_seconds=900.0,
            ))
        result = run_baseline(
            config, dataset, lambda: loader.load_test(data_config), context,
            llm_client=client, proposer_prompt=prompt,
        )
    payload = result.model_dump(mode="json")
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
