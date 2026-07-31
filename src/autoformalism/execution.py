"""User-facing experiment assembly shared by command-line entry scripts."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autoformalism.config import DataConfig
from autoformalism.data import (
    BenchmarkLoader,
    BenchmarkRegistry,
    DatasetSplit,
    DevelopmentDataset,
    SplitName,
    Trajectory,
)
from autoformalism.expressions import ValidationContext
from autoformalism.fitting import FitConfig
from autoformalism.llm import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    MockLLMClient,
    create_llm_client,
)
from autoformalism.pruning import PruningConfig
from autoformalism.schemas import CandidateModel, JudgeResult
from autoformalism.search import FinalEvaluation, SearchConfig, SearchController

_CONTROLLER_PROMPT = """
Return exactly one complete CandidateModel. Treat all target channels as generated
outputs, never as supplied forcing. Use only declared auxiliaries, external inputs,
fixed covariates, states, processes, and parameters. Expressions must follow the
restricted grammar. Use the beam feedback to make a structurally meaningful
exploratory proposal; do not merely tune numeric values.
""".strip()


@dataclass(frozen=True)
class ExecutionArguments:
    """Normalized command-line values for one experiment."""

    data_root: Path
    benchmark_id: str
    tier: str
    seed: int
    proposer_model: str | None
    judge_model: str | None
    iteration_budget: int
    beam_size: int
    output_root: Path
    resume: bool
    dry_run: bool
    mock_llm: bool
    use_clean_observations: bool


class _RoleClient:
    """Delegate proposer and judge calls to independently configured clients."""

    def __init__(self, proposer: LLMClient, judge: LLMClient) -> None:
        self._proposer = proposer
        self._judge = judge

    def propose(self, *, system_prompt: str, user_prompt: str):
        return self._proposer.propose(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

    def judge(self, *, system_prompt: str, user_prompt: str):
        return self._judge.judge(system_prompt=system_prompt, user_prompt=user_prompt)


def build_experiment_parser(
    *,
    description: str,
    default_resume: bool = False,
) -> argparse.ArgumentParser:
    """Build the common run/resume command-line parser."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUTOFORMALISM_DATA_ROOT", "data_raw")),
    )
    parser.add_argument("--benchmark-id", default="original_b1")
    parser.add_argument(
        "--tier", choices=("easy", "medium", "hard"), default="easy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--proposer-model",
        help="provider:model, for example openai:gpt-5.2 or ollama:gpt-oss:20b",
    )
    parser.add_argument(
        "--judge-model",
        help="provider:model; defaults to --proposer-model",
    )
    parser.add_argument("--iteration-budget", type=int, default=5)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=default_resume,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser


def arguments_from_namespace(namespace: argparse.Namespace) -> ExecutionArguments:
    """Validate and normalize parsed command-line arguments."""
    if namespace.iteration_budget < 1:
        raise SystemExit("--iteration-budget must be at least 1")
    if namespace.beam_size < 1:
        raise SystemExit("--beam-size must be at least 1")
    if (
        not namespace.mock_llm
        and not namespace.dry_run
        and not namespace.proposer_model
    ):
        raise SystemExit("--proposer-model is required unless --mock-llm is used")
    return ExecutionArguments(
        data_root=namespace.data_root.expanduser().resolve(),
        benchmark_id=namespace.benchmark_id,
        tier=namespace.tier,
        seed=namespace.seed,
        proposer_model=namespace.proposer_model,
        judge_model=namespace.judge_model or namespace.proposer_model,
        iteration_budget=namespace.iteration_budget,
        beam_size=namespace.beam_size,
        output_root=namespace.output_root.expanduser().resolve(),
        resume=namespace.resume,
        dry_run=namespace.dry_run,
        mock_llm=namespace.mock_llm,
        use_clean_observations=namespace.clean,
    )


def execute(arguments: ExecutionArguments) -> dict[str, Any]:
    """Validate inputs, optionally dry-run, then execute or resume one run."""
    dataset, test_loader, proposer_prompt, judge_prompt = _load_inputs(arguments)
    experiment_directory = _experiment_directory(arguments)
    checkpoint_directory = experiment_directory / "checkpoints"
    plan = {
        "benchmark_id": dataset.benchmark_id,
        "tier": dataset.tier,
        "seed": arguments.seed,
        "targets": list(dataset.roles.targets),
        "auxiliaries": list(dataset.roles.auxiliaries),
        "iteration_budget": arguments.iteration_budget,
        "beam_size": arguments.beam_size,
        "proposer_model": arguments.proposer_model,
        "judge_model": arguments.judge_model,
        "mock_llm": arguments.mock_llm,
        "experiment_directory": str(experiment_directory),
        "split_fingerprints": {
            "train": dataset.train.fingerprint,
            "validation": dataset.validation.fingerprint,
        },
    }
    if arguments.dry_run:
        return {"status": "dry_run", **plan}

    run_metadata = checkpoint_directory / "run.json"
    if arguments.resume and not run_metadata.exists():
        raise SystemExit(
            f"cannot resume; checkpoint does not exist: {checkpoint_directory}"
        )
    if not arguments.resume and run_metadata.exists():
        raise SystemExit(
            f"run already exists at {experiment_directory}; pass --resume"
        )
    experiment_directory.mkdir(parents=True, exist_ok=True)
    client = _make_client(arguments, dataset, experiment_directory)
    context = _context(arguments, dataset)
    search_config = SearchConfig(
        checkpoint_directory=checkpoint_directory,
        maximum_iterations=arguments.iteration_budget,
        beam_size=arguments.beam_size,
        stagnation_iterations=max(2, min(5, arguments.iteration_budget)),
        validation_mse_target=0.0,
        cheap_prefit_judge=False,
        proposer_system_prompt=(
            f"Configured proposer model: {arguments.proposer_model or 'mock'}\n\n"
            f"{proposer_prompt}\n\nController requirements:\n{_CONTROLLER_PROMPT}"
        ),
        judge_system_prompt=(
            f"Configured judge model: {arguments.judge_model or 'mock'}\n\n"
            f"Proposer task:\n{proposer_prompt}\n\nJudge task:\n{judge_prompt}"
        ),
        fit_config=FitConfig(random_seed=arguments.seed),
        pruning_config=PruningConfig(),
    )
    result = SearchController(
        llm_client=client,
        context=context,
        training=dataset.train,
        validation=dataset.validation,
        test_loader=test_loader,
        config=search_config,
    ).run()
    summary = _result_summary(arguments, result)
    (experiment_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (experiment_directory / "run_config.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _load_inputs(
    arguments: ExecutionArguments,
) -> tuple[DevelopmentDataset, Callable[[], DatasetSplit], str, str]:
    if arguments.benchmark_id == "synthetic":
        if not arguments.data_root.is_dir():
            raise SystemExit(f"data root is not a directory: {arguments.data_root}")
        return (
            _synthetic_development_dataset(),
            _synthetic_test_split,
            "Generate a one-state continuous-time decay model for target.",
            "Require a closed causal state equation and target observation mapping.",
        )
    data_config = DataConfig(
        root=arguments.data_root,
        benchmark_id=arguments.benchmark_id,
        tier=arguments.tier,
        use_clean_observations=arguments.use_clean_observations,
    )
    registry = BenchmarkRegistry()
    loader = BenchmarkLoader(registry)
    dataset = loader.load_development(data_config)
    loader.validate_test_paths(data_config)
    spec = registry.get(arguments.benchmark_id)
    prompt_root = (arguments.data_root / spec.relative_root / arguments.tier).resolve()
    proposer_path = prompt_root / "proposer_prompt.txt"
    judge_path = prompt_root / "judge_prompt.txt"
    for path in (proposer_path, judge_path):
        if not path.is_file():
            raise SystemExit(f"benchmark prompt is missing: {path}")
        if not path.is_relative_to(arguments.data_root):
            raise SystemExit(f"benchmark prompt escapes data root: {path}")
    return (
        dataset,
        lambda: loader.load_test(data_config),
        proposer_path.read_text(encoding="utf-8"),
        judge_path.read_text(encoding="utf-8"),
    )


def _context(
    arguments: ExecutionArguments,
    dataset: DevelopmentDataset,
) -> ValidationContext:
    if arguments.benchmark_id == "synthetic":
        return ValidationContext(targets=("target",))
    spec = BenchmarkRegistry().get(arguments.benchmark_id)
    return ValidationContext(
        targets=dataset.roles.targets,
        auxiliaries=dataset.roles.auxiliaries,
        external_inputs=spec.external_inputs,
        fixed_covariates=spec.fixed_covariates,
    )


def _make_client(
    arguments: ExecutionArguments,
    dataset: DevelopmentDataset,
    experiment_directory: Path,
) -> LLMClient:
    if arguments.mock_llm:
        candidates = _mock_candidates(dataset, arguments.iteration_budget)
        judges = [_mock_judge()] * (2 * arguments.iteration_budget)
        return MockLLMClient(
            proposer_responses=candidates,
            judge_responses=judges,
        )
    assert arguments.proposer_model is not None
    assert arguments.judge_model is not None
    proposer_provider, proposer_model = _parse_model(arguments.proposer_model)
    judge_provider, judge_model = _parse_model(arguments.judge_model)
    cache_root = experiment_directory / "llm_cache"
    proposer = create_llm_client(
        LLMConfig(
            provider=proposer_provider,
            model=proposer_model,
            cache_directory=cache_root / "proposer",
            log_path=experiment_directory / "proposer_events.jsonl",
        )
    )
    judge = create_llm_client(
        LLMConfig(
            provider=judge_provider,
            model=judge_model,
            cache_directory=cache_root / "judge",
            log_path=experiment_directory / "judge_events.jsonl",
        )
    )
    return _RoleClient(proposer, judge)


def _parse_model(value: str) -> tuple[LLMProvider, str]:
    if ":" not in value:
        return LLMProvider.OPENAI, value
    provider_name, model = value.split(":", 1)
    try:
        provider = LLMProvider(provider_name)
    except ValueError as exc:
        raise SystemExit(
            f"unsupported model provider {provider_name!r}; use openai or ollama"
        ) from exc
    if not model:
        raise SystemExit("model identifier cannot be empty")
    return provider, model


def _experiment_directory(arguments: ExecutionArguments) -> Path:
    name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        f"{arguments.benchmark_id}_{arguments.tier}_seed{arguments.seed}",
    )
    return arguments.output_root / name


def _synthetic_split(
    name: SplitName,
    identifier: str,
    duration: float,
) -> DatasetSplit:
    time = np.linspace(0.0, duration, int(duration * 10) + 1)
    target = 1.25 * np.exp(-0.55 * time)
    trajectory = Trajectory(
        identifier,
        time,
        {"target": target},
        {},
        {},
        {},
        {},
    )
    return DatasetSplit(name, (trajectory,), f"synthetic-{name.value}")


def _synthetic_development_dataset() -> DevelopmentDataset:
    from autoformalism.data.models import TierRoles

    return DevelopmentDataset(
        benchmark_id="synthetic",
        tier="easy",
        roles=TierRoles(targets=("target",)),
        train=_synthetic_split(SplitName.TRAIN, "train", 2.0),
        validation=_synthetic_split(
            SplitName.VALIDATION, "validation", 2.5
        ),
    )


def _synthetic_test_split() -> DatasetSplit:
    return _synthetic_split(SplitName.TEST, "test", 3.0)


def _mock_candidates(
    dataset: DevelopmentDataset,
    count: int,
) -> list[CandidateModel]:
    target_names = dataset.roles.targets
    combined = {
        target: np.concatenate(
            [trajectory.targets[target] for trajectory in dataset.train.trajectories]
        )
        for target in target_names
    }
    candidates: list[CandidateModel] = []
    for round_index in range(count):
        identifier = f"mock_candidate_{round_index}"
        states = [f"state_{index}" for index in range(len(target_names))]
        parameters: list[dict[str, Any]] = []
        equations: list[dict[str, str]] = []
        for index, state in enumerate(states):
            decay = f"decay_{index}"
            parameters.append(_parameter(decay, 0.0, 5.0))
            rhs = f"-{decay} * {state}"
            if round_index:
                coefficient = f"nonlinear_{round_index}_{index}"
                parameters.append(_parameter(coefficient, -2.0, 2.0))
                rhs += f" + {coefficient} * {state} ** {round_index + 1}"
            equations.append({"state": state, "rhs": rhs})
        payload = {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "change_summary": "Deterministic offline mock proposal.",
            "states": [
                {
                    "name": state,
                    "kind": "observed",
                    "unit": "data_unit",
                    "description": f"Generated state for {target}.",
                }
                for state, target in zip(states, target_names, strict=True)
            ],
            "state_equations": equations,
            "observation_mappings": [
                {
                    "channel": target,
                    "expression": state,
                    "unit": "data_unit",
                }
                for state, target in zip(states, target_names, strict=True)
            ],
            "parameters": parameters,
            "initial_conditions": [
                {
                    "state": state,
                    "scope": "global",
                    "initialization_range": _initial_range(combined[target]),
                }
                for state, target in zip(states, target_names, strict=True)
            ],
        }
        candidates.append(CandidateModel.model_validate(payload))
    return candidates


def _parameter(name: str, lower: float, upper: float) -> dict[str, Any]:
    return {
        "name": name,
        "scope": "global",
        "bounds": {"lower": lower, "upper": upper},
        "initialization_range": {"lower": lower, "upper": upper},
        "unit": "1/time",
        "description": f"Mock parameter {name}.",
    }


def _initial_range(values: np.ndarray) -> dict[str, float]:
    lower = float(np.min(values))
    upper = float(np.max(values))
    padding = max(1e-3, (upper - lower) * 0.25)
    return {"lower": lower - padding, "upper": upper + padding}


def _mock_judge() -> JudgeResult:
    return JudgeResult.model_validate(
        {
            "hard_red_flags": [],
            "category_scores": {"schema_compliance": 1.0},
            "aggregate_score": 1.0,
            "missing_requirements": [],
            "actionable_edits": [],
        }
    )


def _result_summary(
    arguments: ExecutionArguments,
    result: FinalEvaluation,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "benchmark_id": arguments.benchmark_id,
        "tier": arguments.tier,
        "seed": arguments.seed,
        "stopping_reason": result.stopping_reason,
        "completed_iterations": result.completed_iterations,
        "selection_hash": result.frozen_selection.selection_hash,
        "selected_candidate": result.frozen_selection.candidate.model_dump(
            mode="json"
        ),
        "selection_validation_normalized_mse": (
            result.frozen_selection.validation_mse
        ),
        "final_global_parameters": dict(result.final_fit.global_parameters),
        "final_training_normalized_mse": (
            result.final_fit.training_metrics.normalized_mse
        ),
        "test_normalized_mse": result.test_metrics.normalized_mse,
        "test_per_target_normalized_mse": dict(
            result.test_metrics.per_target_normalized_mse
        ),
        "test_failed_trajectories": list(result.test_metrics.failed_trajectories),
    }
