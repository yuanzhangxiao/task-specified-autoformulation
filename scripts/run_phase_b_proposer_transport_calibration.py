#!/usr/bin/env python3
"""Run matched round-zero proposer calls without fitting or judge evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
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

    efforts = plan.model_contract.evaluated_reasoning_efforts
    for effort in efforts:
        for budget in plan.model_contract.max_output_token_budgets:
            result_path = _condition_path(
                output_root / "results",
                task_index=task.task_index,
                effort=effort,
                budget=budget,
                multiple_efforts=len(efforts) > 1,
            ).with_suffix(".json")
            if result_path.is_file():
                restored = ProposerCalibrationResult.model_validate_json(
                    result_path.read_text(encoding="utf-8")
                )
                if (
                    restored.plan_sha256 != plan_sha256
                    or restored.task_index != task.task_index
                    or restored.reasoning_effort != effort
                    or restored.max_output_tokens != budget
                ):
                    raise ValueError(
                        f"calibration checkpoint differs: {result_path}"
                    )
                print(
                    f"restored task={task.task_index} effort={effort} "
                    f"budget={budget} success={restored.response_success}"
                )
                continue
            result = _run_budget(
                plan_path=plan_path,
                task=task,
                reasoning_effort=effort,
                budget=budget,
                data_root=args.data_root.expanduser().resolve(),
                target_contract_root=(
                    args.target_contract_root.expanduser().resolve()
                ),
                output_root=output_root,
                vllm_base_url=args.vllm_base_url,
            )
            _atomic_json(result_path, result.model_dump(mode="json"))
            print(
                f"completed task={task.task_index} effort={effort} "
                f"budget={budget} response={result.response_success} "
                f"valid={result.deterministic_valid} "
                f"target={result.public_target_passed}"
            )


def _run_budget(
    *,
    plan_path: Path,
    task: Any,
    reasoning_effort: str,
    budget: int,
    data_root: Path,
    target_contract_root: Path,
    output_root: Path,
    vllm_base_url: str,
) -> ProposerCalibrationResult:
    plan = load_proposer_calibration_plan(plan_path)
    condition_root = _condition_path(
        output_root / "conditions",
        task_index=task.task_index,
        effort=reasoning_effort,
        budget=budget,
        multiple_efforts=(
            len(plan.model_contract.evaluated_reasoning_efforts) > 1
        ),
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
            reasoning_effort
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
            reasoning_effort
        ),
        timeout_seconds=plan.model_contract.request_timeout_seconds,
        max_output_tokens=budget,
        continuation_max_output_tokens=(
            plan.model_contract.continuation_max_output_tokens
            if plan.model_contract.incomplete_response_strategy == "continue_once"
            else None
        ),
        temperature=plan.model_contract.temperature,
        seed=task.repetition,
        max_attempts=plan.model_contract.maximum_provider_attempts,
        proposal_target_channels=dataset.roles.targets,
        proposal_protected_parameter_names=tuple(
            sorted(
                {
                    *context.targets,
                    *context.auxiliaries,
                    *context.external_inputs,
                    *context.fixed_covariates,
                }
            )
        ),
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
    logical_latency_ms: float | None = None
    logical_started = time.perf_counter()
    try:
        try:
            call = client.propose(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        finally:
            logical_latency_ms = (time.perf_counter() - logical_started) * 1000.0
        response_success = True
        request_hash = call.request_hash
        cache_hit = call.cache_hit
        latency_ms = call.latency_ms
        candidate, repairs = repair_protected_declarations(call.parsed, context)
        diagnostics.extend(
            {"code": "DETERMINISTIC_REPAIR", "message": item}
            for item in repairs
        )
        pre_schema = call.raw_response.get("_autoformalism_pre_schema_repair")
        if isinstance(pre_schema, dict):
            pre_schema_repairs = pre_schema.get("repairs")
            if isinstance(pre_schema_repairs, list):
                diagnostics.extend(
                    {
                        "code": "PRE_SCHEMA_REPAIR",
                        "repair": item,
                    }
                    for item in pre_schema_repairs
                    if isinstance(item, dict)
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
    if logical_latency_ms is None:
        logical_latency_ms = (time.perf_counter() - logical_started) * 1000.0
    accounting = _attempt_accounting(events, request_hash=request_hash)
    first_attempt_response_success = bool(
        response_success and accounting["attempt_count"] == 1
    )
    if response_success and request_hash is None:
        request_hash = _request_hash_from_events(event_log)
    return ProposerCalibrationResult(
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        task_index=task.task_index,
        benchmark_id=task.benchmark_id,
        tier=task.tier,
        repetition=task.repetition,
        model=plan.model_contract.model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=budget,
        incomplete_response_strategy=(
            plan.model_contract.incomplete_response_strategy
        ),
        continuation_request_count=accounting["continuation_requests"],
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
        logical_latency_ms=logical_latency_ms,
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


def _condition_path(
    root: Path,
    *,
    task_index: int,
    effort: str,
    budget: int,
    multiple_efforts: bool,
) -> Path:
    """Keep legacy single-effort paths while separating factorial conditions."""
    path = root / f"task_{task_index:03d}"
    if multiple_efforts:
        path /= f"effort_{effort}"
    return path / f"budget_{budget:06d}"


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
    physical_attempts: list[dict[str, object]] = []
    continuation_requests = 0
    for raw in raw_attempts:
        segments = _physical_responses(raw)
        physical_attempts.extend(segments)
        continuation_requests += max(0, len(segments) - 1)
    if physical_attempts:
        attempts = len(physical_attempts)
    input_tokens = 0
    output_tokens = 0
    observed_usage = False
    length_exhausted = 0
    reasoning_characters = 0
    for raw in physical_attempts:
        usage = raw.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
            completion = usage.get(
                "completion_tokens", usage.get("output_tokens")
            )
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
        if raw.get("status") == "incomplete":
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
        reasoning_characters += _responses_reasoning_characters(raw)
    successful_usage = _summed_token_usage(
        _physical_responses(successful_raw)
    )
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
        "continuation_requests": continuation_requests,
    }


def _token_usage(raw: object) -> tuple[int | None, int | None]:
    """Return prompt/completion usage for one provider attempt."""
    if not isinstance(raw, dict):
        return None, None
    usage = raw.get("usage")
    if not isinstance(usage, dict):
        return None, None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    input_tokens = (
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None
    )
    output_tokens = (
        completion
        if isinstance(completion, int) and not isinstance(completion, bool)
        else None
    )
    return input_tokens, output_tokens


def _physical_responses(raw: object) -> list[dict[str, object]]:
    """Expand a Responses continuation aggregate into physical requests."""
    if not isinstance(raw, dict):
        return []
    continuation = raw.get("_autoformalism_continuation")
    if isinstance(continuation, dict):
        segments = continuation.get("segments")
        if isinstance(segments, list) and all(
            isinstance(item, dict) for item in segments
        ):
            expanded = list(segments)
            request_count = continuation.get("request_count")
            if (
                isinstance(request_count, int)
                and not isinstance(request_count, bool)
                and request_count > len(expanded)
            ):
                expanded.extend({} for _ in range(request_count - len(expanded)))
            return expanded
    return [raw]


def _summed_token_usage(
    responses: list[dict[str, object]],
) -> tuple[int | None, int | None]:
    """Sum input/output usage for one successful restart or continuation chain."""
    usages = [_token_usage(item) for item in responses]
    inputs = [item[0] for item in usages if item[0] is not None]
    outputs = [item[1] for item in usages if item[1] is not None]
    return (
        sum(inputs) if inputs else None,
        sum(outputs) if outputs else None,
    )


def _responses_reasoning_characters(raw: dict[str, object]) -> int:
    """Count returned reasoning text in one Responses API segment."""
    output = raw.get("output")
    if not isinstance(output, list):
        return 0
    total = 0
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for key in ("content", "summary"):
            value = item.get(key)
            if isinstance(value, str):
                total += len(value)
            elif isinstance(value, list):
                for part in value:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total += len(part["text"])
    return total


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
