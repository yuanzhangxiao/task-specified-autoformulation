#!/usr/bin/env python3
"""Run one frozen public-only incremental-construction task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autoformalism.baselines.core import baseline_validation_context
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader, BenchmarkRegistry
from autoformalism.llm import VLLMClient, VLLMReasoningEffort
from autoformalism.rebuttal.incremental_construction_experiment import (
    build_incremental_construction_tasks,
    load_incremental_construction_plan,
)
from autoformalism.rebuttal.incremental_construction_pilot import (
    FUNCTIONAL_ACTION_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    TOPOLOGY_ACTION_SYSTEM_PROMPT,
    IncrementalConstructionPilotConfig,
    build_public_construction_problem,
    construct_public_candidates,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.search.incremental_proposer import IncrementalProposerConfig
from autoformalism.targets import PublicTargetContract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--mechanism-spec-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vllm-base-url", required=True)
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    plan = load_incremental_construction_plan(plan_path)
    tasks = build_incremental_construction_tasks(plan)
    if not 0 <= args.task_index < len(tasks):
        raise SystemExit(f"task index is outside [0, {len(tasks) - 1}]")
    task = tasks[args.task_index]
    run_name = f"{task.benchmark_id}_{task.tier}_seed{task.repetition}"
    run_root = args.output_root.expanduser().resolve() / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    prompt_path = (
        args.data_root.expanduser().resolve()
        / "phase_b_v1"
        / task.benchmark_id
        / "proposer_prompt.txt"
    )
    target_path = (
        args.target_contract_root.expanduser().resolve()
        / "specs"
        / f"{task.benchmark_id}.json"
    )
    mechanism_path = (
        args.mechanism_spec_root.expanduser().resolve()
        / "specs"
        / f"{task.benchmark_id}.json"
    )
    _require_sha(prompt_path, task.public_prompt_sha256)
    _require_sha(target_path, task.public_target_contract_sha256)
    _require_sha(mechanism_path, task.public_mechanism_spec_sha256)

    registry = BenchmarkRegistry()
    specification = registry.get(task.benchmark_id)
    dataset = BenchmarkLoader(registry).load_development(
        DataConfig(
            root=args.data_root,
            benchmark_id=task.benchmark_id,
            tier=task.tier,
        )
    )
    context = baseline_validation_context(dataset, specification)
    target_contract = PublicTargetContract.model_validate_json(
        target_path.read_text(encoding="utf-8")
    )
    mechanism_spec = MechanismEvaluationSpec.model_validate_json(
        mechanism_path.read_text(encoding="utf-8")
    )
    if set(context.targets) != {
        item.target_channel for item in target_contract.targets
    }:
        raise ValueError("public targets differ from the target contract")
    public_problem = build_public_construction_problem(
        public_prompt=prompt_path.read_text(encoding="utf-8"),
        context=context,
        target_contract=target_contract,
        mechanism_spec=mechanism_spec,
    )
    run_fingerprint = hashlib.sha256(
        plan_path.read_bytes()
        + b"\0"
        + json.dumps(
            task.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    model = plan.model_contract
    client = VLLMClient(
        model=model.model,
        cache_directory=run_root / "llm_cache",
        log_path=run_root / "proposer_events.jsonl",
        base_url=args.vllm_base_url,
        reasoning_effort=VLLMReasoningEffort(model.reasoning_effort),
        timeout_seconds=model.request_timeout_seconds,
        max_output_tokens=model.max_output_tokens,
        temperature=model.temperature,
        seed=task.repetition,
        max_attempts=model.maximum_provider_attempts,
        initial_backoff_seconds=1.0,
        max_backoff_seconds=30.0,
        jitter_fraction=0.0,
    )
    budget = plan.construction_budget
    result = construct_public_candidates(
        client=client,
        proposer_config=IncrementalProposerConfig(
            checkpoint_directory=run_root / "checkpoints",
            run_fingerprint=run_fingerprint,
            intent_system_prompt=INTENT_SYSTEM_PROMPT,
            topology_action_system_prompt=TOPOLOGY_ACTION_SYSTEM_PROMPT,
            functional_action_system_prompt=FUNCTIONAL_ACTION_SYSTEM_PROMPT,
        ),
        pilot_config=IncrementalConstructionPilotConfig(
            topology_branch_count=budget.topology_branch_count,
            function_children_per_topology=(
                budget.function_children_per_topology
            ),
            maximum_topology_action_steps=(
                budget.maximum_topology_action_steps
            ),
            maximum_functional_action_steps=(
                budget.maximum_functional_action_steps
            ),
        ),
        public_problem=public_problem,
        context=context,
        target_contract=target_contract,
        mechanism_spec=mechanism_spec,
        output_path=run_root / "construction_result.json",
    )
    print(result.model_dump_json(indent=2))


def _require_sha(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(
            f"frozen public input differs: path={path}, "
            f"observed={observed}, expected={expected}"
        )


if __name__ == "__main__":
    main()
