#!/usr/bin/env python3
"""Run one leakage-safe baseline and write a JSON result."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from autoformalism.baselines.d3 import run_d3_native_no_tools
from autoformalism.baselines.models import BaselineConfig, BaselineRunStatus
from autoformalism.baselines.runner import run_baseline
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.execution import _development_forcing_bounds, _parse_model
from autoformalism.expressions import ValidationContext
from autoformalism.llm import LLMConfig, create_llm_client


def build_parser() -> argparse.ArgumentParser:
    """Build the baseline CLI parser for execution and argument tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUTOFORMALISM_DATA_ROOT", "data_raw")),
    )
    parser.add_argument("--benchmark-id", default="original_b1")
    parser.add_argument("--tier", choices=("easy", "medium", "hard"), default="easy")
    parser.add_argument(
        "--method",
        required=True,
        choices=(
            "persistence",
            "sindy",
            "pysr",
            "llm_feature_sindy",
            "d3_native_no_tools",
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument(
        "--llm-cache-only",
        action="store_true",
        help="fail closed on an LLM cache miss without contacting a provider",
    )
    parser.add_argument(
        "--llm-cache-root",
        type=Path,
        help="shared flat cache directory for cache-only baseline replay",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/baselines"))
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
    parser.add_argument(
        "--wall-timeout-seconds",
        type=float,
        default=1_800.0,
        help="Hard wall-clock limit for the complete baseline run (default: 1800).",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Parse, execute, and persist one baseline run."""
    parser = build_parser()
    args = parser.parse_args()
    if args.wall_timeout_seconds <= 0:
        parser.error("--wall-timeout-seconds must be positive")
    if not args._worker:
        _supervise(args)
        return
    _run_worker(args, parser)


def _run_worker(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Execute one baseline inside the killable supervised worker process."""
    started = time.monotonic()
    output = _output_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = _execute_baseline(args, parser, output)
    except Exception as exc:
        elapsed = time.monotonic() - started
        _write_status(
            output,
            BaselineRunStatus(
                status="failed",
                elapsed_wall_seconds=elapsed,
                wall_timeout_seconds=args.wall_timeout_seconds,
                error=f"{type(exc).__name__}: {str(exc)[:2000]}",
            ),
        )
        raise
    elapsed = time.monotonic() - started
    result = result.model_copy(
        update={
            "elapsed_wall_seconds": elapsed,
            "wall_timeout_seconds": args.wall_timeout_seconds,
        }
    )
    payload = result.model_dump(mode="json")
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_status(
        output,
        BaselineRunStatus(
            status="complete",
            elapsed_wall_seconds=elapsed,
            wall_timeout_seconds=args.wall_timeout_seconds,
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _execute_baseline(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    output: Path,
):
    """Load leakage-safe inputs and dispatch one configured baseline method."""

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
        wall_timeout_seconds=args.wall_timeout_seconds,
    )
    if args.method == "d3_native_no_tools":
        if not args.model:
            parser.error("--model is required for d3_native_no_tools")
        provider, model = _parse_model(args.model)
        cache_directory = args.llm_cache_root or output / "llm_cache"
        client = create_llm_client(
            LLMConfig(
                provider=provider,
                model=model,
                cache_directory=cache_directory,
                log_path=output / "llm_events.jsonl",
                proposal_target_channels=dataset.roles.targets,
                timeout_seconds=900.0,
                cache_only=args.llm_cache_only,
            )
        )
        result = run_d3_native_no_tools(
            config,
            dataset,
            lambda: loader.load_test(data_config),
            context,
            task_prompt=prompt,
            work_directory=output,
            llm_client=client,
        )
    else:
        client = None
        if args.method == "llm_feature_sindy":
            if not args.model:
                parser.error("--model is required for llm_feature_sindy")
            provider, model = _parse_model(args.model)
            cache_directory = args.llm_cache_root or output / "llm_cache"
            client = create_llm_client(
                LLMConfig(
                    provider=provider,
                    model=model,
                    cache_directory=cache_directory,
                    log_path=output / "llm_events.jsonl",
                    proposal_target_channels=dataset.roles.targets,
                    timeout_seconds=900.0,
                    cache_only=args.llm_cache_only,
                )
            )
        result = run_baseline(
            config,
            dataset,
            lambda: loader.load_test(data_config),
            context,
            llm_client=client,
            proposer_prompt=prompt,
        )
    return result


def _supervise(args: argparse.Namespace) -> None:
    """Run the worker in its own process group and enforce a hard deadline."""
    output = _output_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--_worker",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = process.wait(timeout=args.wall_timeout_seconds)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        elapsed = time.monotonic() - started
        status = BaselineRunStatus(
            status="timed_out",
            elapsed_wall_seconds=elapsed,
            wall_timeout_seconds=args.wall_timeout_seconds,
            error="baseline wall-clock limit reached",
        )
        _write_status(output, status)
        print(json.dumps(status.model_dump(mode="json"), indent=2, sort_keys=True))
        raise SystemExit(124) from None
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise
    if return_code != 0:
        raise SystemExit(return_code)


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate the worker and descendants, including Julia child processes."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10.0)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _output_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root.resolve()
        / args.method
        / (f"{args.benchmark_id}_{args.tier}_seed{args.seed}")
    )


def _write_status(output: Path, status: BaselineRunStatus) -> None:
    destination = output / "run_status.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(status.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


if __name__ == "__main__":
    main()
