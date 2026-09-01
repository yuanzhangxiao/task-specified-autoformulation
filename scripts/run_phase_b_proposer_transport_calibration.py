#!/usr/bin/env python3
"""Run matched round-zero proposer calls without fitting or judge evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from autoformalism.execution import (
    ExecutionArguments,
    _context,
    _load_inputs,
    _load_public_target_contract,
    proposer_system_prompt,
)
from autoformalism.expressions import (
    ModelValidationError,
    compile_candidate,
    repair_protected_declarations,
)
from autoformalism.llm import VLLMClient, VLLMReasoningEffort
from autoformalism.llm.exceptions import LLMProviderError, LLMResponseError
from autoformalism.rebuttal.proposer_transport_calibration import (
    ProposerCalibrationResult,
    build_proposer_calibration_tasks,
    load_proposer_calibration_plan,
)
from autoformalism.search.controller import _structural_hash
from autoformalism.targets import evaluate_public_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vllm-base-url", required=True)
    args = parser.parse_args()

    plan_path = args.plan.expanduser().resolve()
    plan = load_proposer_calibration_plan(plan_path)
    tasks = build_proposer_calibration_tasks(plan)
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise SystemExit(f"task index is outside [0, {len(tasks) - 1}]")
    task = tasks[args.task_index]
    output_root = args.output_root.expanduser().resolve()
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()

    for budget in plan.model_contract.max_output_token_budgets:
        result_path = (
            output_root
            / "results"
            / f"task_{task.task_index:03d}"
            / f"budget_{budget:06d}.json"
        )
        if result_path.is_file():
            restored = ProposerCalibrationResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
            if (
                restored.plan_sha256 != plan_sha256
                or restored.task_index != task.task_index
                or restored.max_output_tokens != budget
            ):
                raise ValueError(f"calibration checkpoint differs: {result_path}")
            print(
                f"restored task={task.task_index} budget={budget} "
                f"success={restored.response_success}"
            )
            continue
        result = _run_budget(
            plan_path=plan_path,
            task=task,
            budget=budget,
            data_root=args.data_root.expanduser().resolve(),
            target_contract_root=args.target_contract_root.expanduser().resolve(),
            output_root=output_root,
            vllm_base_url=args.vllm_base_url,
        )
        _atomic_json(result_path, result.model_dump(mode="json"))
        print(
            f"completed task={task.task_index} budget={budget} "
            f"response={result.response_success} "
            f"valid={result.deterministic_valid} "
            f"target={result.public_target_passed}"
        )


def _run_budget(
    *,
    plan_path: Path,
    task: Any,
    budget: int,
    data_root: Path,
    target_contract_root: Path,
    output_root: Path,
    vllm_base_url: str,
) -> ProposerCalibrationResult:
    plan = load_proposer_calibration_plan(plan_path)
    condition_root = (
        output_root
        / "conditions"
        / f"task_{task.task_index:03d}"
        / f"budget_{budget:06d}"
    )
    target_contract_path = (
        target_contract_root / "specs" / f"{task.benchmark_id}.json"
    )
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=task.benchmark_id,
        tier=task.tier,
        seed=task.repetition,
        proposer_model=f"vllm:{plan.model_contract.model}",
        judge_model=f"vllm:{plan.model_contract.model}",
        iteration_budget=1,
        beam_size=1,
        output_root=condition_root,
        resume=False,
        dry_run=False,
        mock_llm=False,
        use_clean_observations=False,
        llm_timeout_seconds=plan.model_contract.request_timeout_seconds,
        llm_max_output_tokens=budget,
        use_judge=False,
        development_only=True,
        public_target_contract=target_contract_path,
        vllm_base_url=vllm_base_url,
        vllm_proposer_reasoning_effort=VLLMReasoningEffort(
            plan.model_contract.reasoning_effort
        ),
        vllm_judge_reasoning_effort=VLLMReasoningEffort.LOW,
        vllm_temperature=plan.model_contract.temperature,
        vllm_seed=task.repetition,
    )
    dataset, _test_loader, public_prompt, _judge_prompt = _load_inputs(arguments)
    context = _context(arguments, dataset)
    target_contract = _load_public_target_contract(
        arguments,
        proposer_prompt=public_prompt,
        targets=dataset.roles.targets,
    )
    assert target_contract is not None
    system_prompt = proposer_system_prompt(
        arguments,
        public_prompt=public_prompt,
        context=context,
    )
    user_prompt = json.dumps(
        {"round": 0, "proposal_mode": "exploratory", "beam_feedback": []},
        sort_keys=True,
    )
    event_log = condition_root / "proposer_events.jsonl"
    client = VLLMClient(
        model=plan.model_contract.model,
        cache_directory=condition_root / "cache",
        log_path=event_log,
        base_url=vllm_base_url,
        reasoning_effort=VLLMReasoningEffort(
            plan.model_contract.reasoning_effort
        ),
        timeout_seconds=plan.model_contract.request_timeout_seconds,
        max_output_tokens=budget,
        temperature=plan.model_contract.temperature,
        seed=task.repetition,
        max_attempts=plan.model_contract.maximum_provider_attempts,
        proposal_target_channels=dataset.roles.targets,
    )

    response_success = False
    deterministic_valid = False
    public_target_passed = False
    first_attempt_response_success = False
    request_hash: str | None = None
    cache_hit = False
    latency_ms: float | None = None
    candidate_hash: str | None = None
    diagnostics: list[dict[str, object]] = []
    error_type: str | None = None
    error: str | None = None
    try:
        call = client.propose(system_prompt=system_prompt, user_prompt=user_prompt)
        response_success = True
        request_hash = call.request_hash
        cache_hit = call.cache_hit
        latency_ms = call.latency_ms
        first_attempt_response_success = call.attempts in (0, 1)
        candidate, repairs = repair_protected_declarations(call.parsed, context)
        diagnostics.extend(
            {"code": "DETERMINISTIC_REPAIR", "message": item}
            for item in repairs
        )
        try:
            compile_candidate(candidate, context)
            deterministic_valid = True
            target_result = evaluate_public_targets(candidate, target_contract)
            public_target_passed = target_result.passed
            diagnostics.extend(
                item.model_dump(mode="json")
                for item in target_result.predicates
                if item.status != "passed"
            )
            candidate_hash = _structural_hash(candidate)
            _atomic_json(
                condition_root / "candidate.json",
                candidate.model_dump(mode="json"),
            )
        except ModelValidationError as exc:
            diagnostics.extend(
                {
                    "code": item.code,
                    "location": item.location,
                    "message": item.message,
                }
                for item in exc.diagnostics
            )
    except (LLMProviderError, LLMResponseError) as exc:
        error_type = type(exc).__name__
        error = str(exc)[:4000]
        raw = getattr(exc, "raw_response", None)
        if isinstance(raw, dict):
            request_hash = _request_hash_from_events(event_log)
    events = _events(event_log)
    accounting = _attempt_accounting(events, request_hash=request_hash)
    if response_success and request_hash is None:
        request_hash = _request_hash_from_events(event_log)
    return ProposerCalibrationResult(
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        task_index=task.task_index,
        benchmark_id=task.benchmark_id,
        tier=task.tier,
        repetition=task.repetition,
        model=plan.model_contract.model,
        reasoning_effort=plan.model_contract.reasoning_effort,
        max_output_tokens=budget,
        request_hash=request_hash,
        cache_hit=cache_hit,
        response_success=response_success,
        first_attempt_response_success=first_attempt_response_success,
        provider_attempt_count=accounting["attempt_count"],
        provider_input_tokens=accounting["input_tokens"],
        provider_output_tokens=accounting["output_tokens"],
        provider_total_tokens=accounting["total_tokens"],
        successful_attempt_input_tokens=accounting[
            "successful_input_tokens"
        ],
        successful_attempt_output_tokens=accounting[
            "successful_output_tokens"
        ],
        successful_attempt_total_tokens=accounting[
            "successful_total_tokens"
        ],
        latency_ms=latency_ms,
        length_exhausted_attempt_count=accounting["length_exhausted"],
        reasoning_character_count=accounting["reasoning_characters"],
        deterministic_valid=deterministic_valid,
        public_target_passed=public_target_passed,
        candidate_structural_sha256=candidate_hash,
        deterministic_diagnostics=tuple(diagnostics),
        error_type=error_type,
        error=error,
        test_data_opened=False,
        scientific_judge_called=False,
        parameter_fitting_performed=False,
    )


def _events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, dict):
            events.append(value)
    return events


def _request_hash_from_events(path: Path) -> str | None:
    hashes = [
        item.get("request_hash")
        for item in _events(path)
        if isinstance(item.get("request_hash"), str)
    ]
    return hashes[-1] if hashes else None


def _attempt_accounting(
    events: list[dict[str, object]], *, request_hash: str | None
) -> dict[str, int | None]:
    relevant = [
        item
        for item in events
        if request_hash is None or item.get("request_hash") == request_hash
    ]
    failures = [item for item in relevant if item.get("event") == "llm_failure"]
    responses = [item for item in relevant if item.get("event") == "llm_response"]
    response = responses[-1] if responses else None
    attempts = (
        int(response.get("attempts", 0))
        if response is not None and not response.get("cache_hit")
        else 0
    )
    if not attempts and failures:
        attempts = max(
            int(item.get("attempt", 0))
            for item in failures
            if isinstance(item.get("attempt"), int)
        )
    raw_attempts = [item.get("raw_response") for item in failures]
    successful_raw = response.get("raw_response") if response is not None else None
    if response is not None and not response.get("cache_hit"):
        raw_attempts.append(successful_raw)
    input_tokens = 0
    output_tokens = 0
    observed_usage = False
    length_exhausted = 0
    reasoning_characters = 0
    for raw in raw_attempts:
        if not isinstance(raw, dict):
            continue
        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int) and not isinstance(prompt, bool):
                input_tokens += prompt
                observed_usage = True
            if isinstance(completion, int) and not isinstance(completion, bool):
                output_tokens += completion
                observed_usage = True
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
            length_exhausted += 1
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict):
            reasoning = (
                message.get("reasoning")
                or message.get("reasoning_content")
                or message.get("thinking")
            )
            if isinstance(reasoning, str):
                reasoning_characters += len(reasoning)
    successful_usage = _token_usage(successful_raw)
    return {
        "attempt_count": attempts,
        "input_tokens": input_tokens if observed_usage else None,
        "output_tokens": output_tokens if observed_usage else None,
        "total_tokens": (
            input_tokens + output_tokens if observed_usage else None
        ),
        "successful_input_tokens": successful_usage[0],
        "successful_output_tokens": successful_usage[1],
        "successful_total_tokens": (
            None
            if successful_usage[0] is None and successful_usage[1] is None
            else (successful_usage[0] or 0) + (successful_usage[1] or 0)
        ),
        "length_exhausted": length_exhausted,
        "reasoning_characters": reasoning_characters,
    }


def _token_usage(raw: object) -> tuple[int | None, int | None]:
    """Return prompt/completion usage for one provider attempt."""
    if not isinstance(raw, dict):
        return None, None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    input_tokens = (
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    )
    output_tokens = (
        completion
        if isinstance(completion, int) and not isinstance(completion, bool)
        else None
    )
    return input_tokens, output_tokens


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
