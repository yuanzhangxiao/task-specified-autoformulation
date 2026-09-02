#!/usr/bin/env python3
"""Replay frozen proposer first attempts through a lossless repair protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from autoformalism.execution import (
    ExecutionArguments,
    _context,
    _load_inputs,
    _load_public_target_contract,
)
from autoformalism.rebuttal.proposer_repair_replay import (
    replay_proposer_first_attempt,
)
from autoformalism.rebuttal.proposer_transport_calibration import (
    ProposerCalibrationResult,
    ProposerRepairReplayPlan,
    build_proposer_calibration_tasks,
    load_proposer_calibration_plan,
)


def main() -> None:
    """Validate the frozen source, replay it, and write a hash-bound bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-plan", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-contract-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = replay_frozen_first_attempts(
        args.replay_plan,
        args.experiment_root,
        data_root=args.data_root,
        target_contract_root=args.target_contract_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def replay_frozen_first_attempts(
    replay_plan_path: Path,
    experiment_root: Path,
    *,
    data_root: Path,
    target_contract_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Replay selected conditions without invoking a provider."""
    replay_plan_path = replay_plan_path.expanduser().resolve()
    experiment = experiment_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    replay_plan = ProposerRepairReplayPlan.model_validate_json(
        replay_plan_path.read_text(encoding="utf-8")
    )
    source_plan_path = experiment / "frozen" / "plan.json"
    if _sha256(source_plan_path) != replay_plan.source_plan_sha256:
        raise ValueError("source proposer calibration plan differs")
    source_plan = load_proposer_calibration_plan(source_plan_path)
    source_analysis_path = (
        experiment / "analysis" / "proposer_transport_calibration.json"
    )
    source_analysis = json.loads(source_analysis_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source_analysis, dict)
        or source_analysis.get("schema_version")
        != "phase-b-proposer-transport-calibration-analysis-1"
        or source_analysis.get("status") != "fail"
        or source_analysis.get("selected_max_output_tokens") is not None
    ):
        raise ValueError("source calibration is not the frozen failed analysis")
    tasks = build_proposer_calibration_tasks(source_plan)
    available = {
        (effort, budget)
        for effort in source_plan.model_contract.evaluated_reasoning_efforts
        for budget in source_plan.model_contract.max_output_token_budgets
    }
    requested = {
        (item.reasoning_effort, item.max_output_tokens)
        for item in replay_plan.conditions
    }
    if not requested <= available:
        raise ValueError("repair replay condition is absent from the source plan")

    copied: list[dict[str, object]] = []
    for source, relative, role in (
        (replay_plan_path, Path("inputs/replay_plan.json"), "replay_plan"),
        (source_plan_path, Path("inputs/source_plan.json"), "source_plan"),
        (
            experiment / "frozen" / "freeze_manifest.json",
            Path("inputs/source_freeze_manifest.json"),
            "source_freeze_manifest",
        ),
        (
            source_analysis_path,
            Path("inputs/source_analysis.json"),
            "source_analysis",
        ),
    ):
        copied.append(_copy(source, output / relative, output, role=role))

    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for condition in replay_plan.conditions:
        effort = condition.reasoning_effort
        budget = condition.max_output_tokens
        for task in tasks:
            condition_name = f"{effort}_{budget:06d}"
            source_result = (
                experiment
                / "results"
                / f"task_{task.task_index:03d}"
                / f"effort_{effort}"
                / f"budget_{budget:06d}.json"
            )
            result = ProposerCalibrationResult.model_validate_json(
                source_result.read_text(encoding="utf-8")
            )
            if (
                result.plan_sha256 != _sha256(source_plan_path)
                or result.benchmark_id != task.benchmark_id
                or result.tier != task.tier
                or result.repetition != task.repetition
                or result.task_index != task.task_index
                or result.reasoning_effort != effort
                or result.max_output_tokens != budget
            ):
                raise ValueError(f"source result identity differs: {source_result}")
            event_log = (
                experiment
                / "conditions"
                / f"task_{task.task_index:03d}"
                / f"effort_{effort}"
                / f"budget_{budget:06d}"
                / "proposer_events.jsonl"
            )
            event = _first_attempt_event(event_log, result.request_hash)
            raw_response = event.get("raw_response")
            content = _chat_content(raw_response)
            if content is None:
                raise ValueError(
                    f"first attempt has no replayable chat content: {event_log}"
                )
            dataset, context, contract = _public_context(
                data_root=data_root,
                target_contract_root=target_contract_root,
                benchmark_id=task.benchmark_id,
                tier=task.tier,
                repetition=task.repetition,
                scratch_root=output / ".context",
            )
            protected = tuple(
                sorted(
                    {
                        *context.targets,
                        *context.auxiliaries,
                        *context.external_inputs,
                        *context.fixed_covariates,
                    }
                )
            )
            replay, candidate = replay_proposer_first_attempt(
                content,
                target_channels=dataset.roles.targets,
                protected_parameter_names=protected,
                context=context,
                target_contract=contract,
            )
            row = {
                "task_index": task.task_index,
                "benchmark_id": task.benchmark_id,
                "tier": task.tier,
                "repetition": task.repetition,
                "reasoning_effort": effort,
                "max_output_tokens": budget,
                "source_result_sha256": _sha256(source_result),
                "source_event_log_sha256": _sha256(event_log),
                **replay.model_dump(mode="json"),
                "schema_version": "phase-b-proposer-repair-replay-row-1",
            }
            rows.append(row)
            grouped[(effort, budget)].append(row)
            task_label = f"task_{task.task_index:03d}"
            copied.append(
                _copy(
                    source_result,
                    output / "sources" / condition_name / f"{task_label}.json",
                    output,
                    role="source_calibration_result",
                )
            )
            copied.append(
                _write_json_artifact(
                    output
                    / "first_attempts"
                    / condition_name
                    / f"{task_label}.json",
                    {
                        "event": event,
                        "event_log_sha256": _sha256(event_log),
                    },
                    output,
                    role="frozen_first_attempt",
                )
            )
            copied.append(
                _write_json_artifact(
                    output / "replays" / condition_name / f"{task_label}.json",
                    row,
                    output,
                    role="repair_replay_result",
                )
            )
            if candidate is not None:
                copied.append(
                    _write_json_artifact(
                        output
                        / "finalists"
                        / condition_name
                        / f"{task_label}.json",
                        candidate.model_dump(mode="json"),
                        output,
                        role="repaired_finalist_candidate",
                    )
                )

    operating_points = [
        _operating_point(
            condition.reasoning_effort,
            condition.max_output_tokens,
            grouped,
        )
        for condition in replay_plan.conditions
    ]
    selected = next((item for item in operating_points if item["passed"]), None)
    rows_path = output / "repair_replay_rows.jsonl"
    _write_once(
        rows_path,
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows).encode(),
    )
    copied.append(_ledger_record(rows_path, output, role="repair_replay_rows"))
    report_path = output / "proposer_repair_replay.md"
    _write_once(report_path, _markdown(operating_points, selected).encode())
    copied.append(_ledger_record(report_path, output, role="repair_replay_report"))
    ledger_path = output / "artifact_ledger.jsonl"
    _write_once(
        ledger_path,
        "".join(
            json.dumps(item, sort_keys=True) + "\n"
            for item in sorted(copied, key=lambda item: str(item["path"]))
        ).encode(),
    )
    manifest = {
        "schema_version": "phase-b-proposer-repair-replay-manifest-1",
        "status": "pass" if selected is not None else "fail",
        "source_experiment_status": "fail",
        "source_plan_sha256": _sha256(source_plan_path),
        "source_analysis_sha256": _sha256(
            source_analysis_path
        ),
        "replay_plan_sha256": _sha256(replay_plan_path),
        "replay_result_count": len(rows),
        "operating_points": operating_points,
        "selected_reasoning_effort": (
            None if selected is None else selected["reasoning_effort"]
        ),
        "selected_max_output_tokens": (
            None if selected is None else selected["max_output_tokens"]
        ),
        "artifact_count": len(copied),
        "artifact_ledger_sha256": _sha256(ledger_path),
        "new_llm_calls_made": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    manifest_path = output / "proposer_repair_replay.json"
    _write_once(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_once(
        output / "proposer_repair_replay.json.sha256",
        f"{_sha256(manifest_path)}  proposer_repair_replay.json\n".encode(),
    )
    return manifest


def _public_context(
    *,
    data_root: Path,
    target_contract_root: Path,
    benchmark_id: str,
    tier: str,
    repetition: int,
    scratch_root: Path,
) -> tuple[object, object, object]:
    contract_path = target_contract_root / "specs" / f"{benchmark_id}.json"
    arguments = ExecutionArguments(
        data_root=data_root,
        benchmark_id=benchmark_id,
        tier=tier,
        seed=repetition,
        proposer_model=None,
        judge_model=None,
        iteration_budget=1,
        beam_size=1,
        output_root=scratch_root,
        resume=False,
        dry_run=True,
        mock_llm=True,
        use_clean_observations=False,
        use_judge=False,
        development_only=True,
        public_target_contract=contract_path,
    )
    dataset, _test_loader, prompt, _judge_prompt = _load_inputs(arguments)
    context = _context(arguments, dataset)
    contract = _load_public_target_contract(
        arguments,
        proposer_prompt=prompt,
        targets=dataset.roles.targets,
    )
    if contract is None:
        raise ValueError(f"missing public target contract: {benchmark_id}")
    return dataset, context, contract


def _first_attempt_event(path: Path, request_hash: str | None) -> dict[str, object]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    relevant = [
        item
        for item in events
        if isinstance(item, dict)
        and (request_hash is None or item.get("request_hash") == request_hash)
    ]
    failures = [
        item
        for item in relevant
        if item.get("event") == "llm_failure" and item.get("attempt") == 1
    ]
    if len(failures) == 1:
        return failures[0]
    successes = [
        item
        for item in relevant
        if item.get("event") == "llm_response" and item.get("attempts") == 1
    ]
    if len(successes) != 1:
        raise ValueError(f"cannot identify one first attempt: {path}")
    return successes[0]


def _chat_content(raw: object) -> str | None:
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else None


def _operating_point(
    effort: str,
    budget: int,
    grouped: dict[tuple[str, int], list[dict[str, object]]],
) -> dict[str, object]:
    rows = grouped[(effort, budget)]
    count = len(rows)
    schema = sum(bool(item["schema_valid_after_repair"]) for item in rows)
    valid = sum(bool(item["deterministic_valid"]) for item in rows)
    target = sum(bool(item["public_target_passed"]) for item in rows)
    repair = sum(bool(item["pre_schema_repairs"]) for item in rows)
    passed = count > 0 and schema == valid == target == count
    return {
        "reasoning_effort": effort,
        "max_output_tokens": budget,
        "trial_count": count,
        "pre_schema_repair_activation_rate": repair / count,
        "schema_valid_after_repair_rate": schema / count,
        "deterministic_validity": valid / count,
        "public_target_pass_rate": target / count,
        "passed": passed,
    }


def _markdown(
    operating_points: list[dict[str, object]],
    selected: dict[str, object] | None,
) -> str:
    lines = [
        "# Offline proposer first-attempt repair replay",
        "",
        "No new LLM calls, parameter fitting, scientific judge, test data, or "
        "private reference were used.",
        "",
        "| Reasoning | Tokens | Trials | Repair activation | Schema-valid | "
        "Deterministic-valid | Public-target pass | Result |",
        "|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in operating_points:
        lines.append(
            f"| {item['reasoning_effort']} | {item['max_output_tokens']} | "
            f"{item['trial_count']} | "
            f"{item['pre_schema_repair_activation_rate']:.3f} | "
            f"{item['schema_valid_after_repair_rate']:.3f} | "
            f"{item['deterministic_validity']:.3f} | "
            f"{item['public_target_pass_rate']:.3f} | "
            f"{'pass' if item['passed'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            (
                "Selected repaired transport condition: **none**."
                if selected is None
                else "Selected repaired transport condition: "
                f"**{selected['reasoning_effort']} / "
                f"{selected['max_output_tokens']} tokens**."
            ),
            "",
            "The original frozen calibration remains FAIL; this report evaluates "
            "a separately versioned deterministic repair protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_json_artifact(
    path: Path,
    payload: object,
    root: Path,
    *,
    role: str,
) -> dict[str, object]:
    _write_once(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return _ledger_record(path, root, role=role)


def _copy(
    source: Path,
    destination: Path,
    root: Path,
    *,
    role: str,
) -> dict[str, object]:
    _write_once(destination, source.read_bytes())
    return _ledger_record(destination, root, role=role)


def _ledger_record(path: Path, root: Path, *, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != data:
            raise ValueError(f"frozen replay artifact differs: {path}")
        return
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
