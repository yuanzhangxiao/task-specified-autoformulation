"""Matched atomic-versus-same-LHS function generation on frozen topologies."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
)
from autoformalism.rebuttal.staged_function_campaign import (
    FunctionCampaignConfig,
    SelectedTopology,
)
from autoformalism.rebuttal.staged_topology_campaign import (
    StagedCampaignConfig,
    runtime_source_hash,
)
from autoformalism.schemas.staged_topology import PublicScientificBrief
from autoformalism.search.staged_function_runner import (
    FunctionGenerationGranularity,
    run_staged_functions,
)
from autoformalism.staged_topology import content_hash


class FunctionGranularityCampaignConfig(StagedCampaignConfig):
    """Two-arm generation policy bound to a completed function-campaign plan."""

    protocol: Literal["scientific-staged-function-granularity-1"]
    source_function_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_topologies: tuple[SelectedTopology, ...] = Field(
        min_length=1,
        max_length=8,
    )
    generation_granularities: tuple[FunctionGenerationGranularity, ...] = Field(
        min_length=2,
        max_length=2,
    )

    @model_validator(mode="after")
    def matched_arms_and_unique_sources(self) -> FunctionGranularityCampaignConfig:
        """Require the prespecified matched comparison without diagnostics."""
        if set(self.generation_granularities) != {
            "atomic_interaction",
            "equation_batch",
        }:
            raise ValueError("granularity campaign requires both prespecified arms")
        names = [item.task_id for item in self.selected_topologies]
        if len(names) != len(set(names)):
            raise ValueError("duplicate selected topology")
        if self.diagnostic_fixtures:
            raise ValueError("matched granularity campaign excludes toy diagnostics")
        return self


def granularity_launcher_hash() -> str:
    """Bind the new CLI and cluster wrappers to the frozen plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_function_granularity_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_function_granularity_aces.slurm",
        "scripts/hpc/submit_staged_function_granularity_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_granularity_campaign(
    config_path: Path,
    source_function_plan_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze matched arms from exact topologies embedded in a prior plan."""
    config = FunctionGranularityCampaignConfig.model_validate_json(
        config_path.read_text()
    )
    source_plan = json.loads(source_function_plan_path.read_text())
    source_digest = content_hash(
        {
            key: value
            for key, value in source_plan.items()
            if key != "plan_sha256"
        }
    )
    if (
        source_digest != source_plan.get("plan_sha256")
        or source_digest != config.source_function_plan_sha256
    ):
        raise ValueError("source function plan digest mismatch")
    if (
        source_plan.get("test_data_opened") is not False
        or source_plan.get("private_reference_opened") is not False
    ):
        raise ValueError("source function plan crossed the public-only boundary")
    source_config = FunctionCampaignConfig.model_validate(source_plan["config"])
    _require_matched_source_settings(config, source_config)

    indexed: dict[str, list[dict[str, Any]]] = {}
    for task in source_plan["tasks"]:
        reviewed = task.get("reviewed_source_task")
        if task.get("kind") == "benchmark" and isinstance(reviewed, str):
            indexed.setdefault(reviewed, []).append(task)

    tasks = []
    for selection in config.selected_topologies:
        if selection.task_id.rsplit("_seed", 1)[0] not in config.public_cells:
            raise ValueError("selection is not an allowed public benchmark task")
        source_tasks = indexed.get(selection.task_id, [])
        if not source_tasks:
            raise ValueError(f"selected topology is absent: {selection.task_id}")
        canonical = source_tasks[0]
        source = canonical["source"]
        if content_hash(source) != selection.result_sha256:
            raise ValueError("selected topology digest mismatch")
        if (
            not source.get("complete_topology")
            or not source.get("public_structure_checks_passed")
            or source.get("test_data_opened") is not False
            or source.get("private_reference_opened") is not False
        ):
            raise ValueError("selected topology failed the public reviewed handoff")
        for duplicate in source_tasks[1:]:
            if (
                duplicate["source"] != source
                or duplicate["brief"] != canonical["brief"]
                or duplicate["context"] != canonical["context"]
            ):
                raise ValueError(
                    "source function plan contains inconsistent topology copies"
                )
        for seed in config.seeds:
            pair_id = f"{selection.task_id}_seed{seed}"
            arms = (
                config.generation_granularities
                if seed % 2 == 0
                else tuple(reversed(config.generation_granularities))
            )
            for granularity in arms:
                tasks.append(
                    {
                        "task_id": f"{pair_id}_{granularity}",
                        "pair_id": pair_id,
                        "kind": "benchmark",
                        "seed": seed,
                        "generation_granularity": granularity,
                        "brief": canonical["brief"],
                        "context": canonical["context"],
                        "source": source,
                        "reviewed_source_task": selection.task_id,
                    }
                )

    plan = {
        "schema_version": "scientific-staged-function-granularity-plan-1",
        "config": config.model_dump(mode="json"),
        "source_function_plan_sha256": source_digest,
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": granularity_launcher_hash(),
        "tasks": tasks,
        "comparison_policy": (
            "Matched atomic-interaction versus same-LHS equation batches on "
            "identical reviewed topologies; latent initial calls remain separate"
        ),
        "arm_launch_order": (
            "counterbalanced_by_seed: atomic first on even seeds; equation batch "
            "first on odd seeds"
        ),
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen granularity campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def _require_matched_source_settings(
    config: FunctionGranularityCampaignConfig,
    source: FunctionCampaignConfig,
) -> None:
    """Hold inference, source cells, seeds and serving resources fixed."""
    comparisons = (
        (config.model_settings, source.model_settings, "model settings"),
        (config.platform, source.platform, "platform"),
        (config.serving_image_sha256, source.serving_image_sha256, "image"),
        (config.served_context_tokens, source.served_context_tokens, "context"),
        (config.public_cells, source.public_cells, "public cells"),
        (config.seeds, source.seeds, "seeds"),
        (config.limits, source.limits, "modeling limits"),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise ValueError(f"granularity campaign {label} differ from source")


def run_granularity_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run all matched tasks once with deterministic drain and resume."""
    plan = json.loads(plan_path.read_text())
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen granularity plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen granularity campaign")
    if granularity_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen granularity campaign")
    config = FunctionGranularityCampaignConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    records: list[dict[str, Any]] = []
    try:
        for task in plan["tasks"]:
            root = output / task["task_id"]
            root.mkdir(parents=True, exist_ok=True)
            with (root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                identity = content_hash([plan["plan_sha256"], task])
                terminal = root / "terminal.json"
                if terminal.exists():
                    record = json.loads(terminal.read_text())
                    if record["identity"] != identity:
                        raise ValueError(
                            "terminal result belongs to another frozen task"
                        )
                else:
                    client = StagedTopologyClient(
                        settings=config.model_settings,
                        base_url=base_url,
                        directory=root / "calls",
                        namespace=identity,
                        seed=task["seed"],
                        can_start=lambda: not stop
                        and time.monotonic()
                        < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        result = run_staged_functions(
                            PublicScientificBrief.model_validate(task["brief"]),
                            ValidationContext.model_validate(task["context"]),
                            task["source"],
                            client,
                            root,
                            generation_granularity=task["generation_granularity"],
                        )
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "pair_id": task["pair_id"],
                        "generation_granularity": task["generation_granularity"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(
                    output / "summary.json",
                    summarize_granularity(plan, records),
                )
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_granularity(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def summarize_granularity(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep every planned failure in denominators and avoid a scalar winner."""
    by_id = {record["task_id"]: record for record in records}
    rows = [_summary_row(task, by_id.get(task["task_id"])) for task in plan["tasks"]]
    config = FunctionGranularityCampaignConfig.model_validate(plan["config"])
    groups = []
    for granularity in config.generation_granularities:
        selected = [
            row for row in rows if row["generation_granularity"] == granularity
        ]
        complete = [row for row in selected if row["complete_model"]]
        nonlinear_applicable = [
            row
            for row in complete
            if row["required_nonlinearity_has_syntax_evidence"] is not None
        ]
        relaxation_applicable = [
            row
            for row in complete
            if row["all_tagged_relaxation_outer_signs_subtractive"] is not None
        ]
        groups.append(
            {
                "generation_granularity": granularity,
                "planned_tasks": len(selected),
                "terminal_results": sum(row["result_present"] for row in selected),
                "complete_models": len(complete),
                "complete_model_rate": _ratio(len(complete), len(selected)),
                "function_first_attempt_step_success_rate": _ratio(
                    sum(
                        row["function_first_attempt_accepted_steps"]
                        for row in selected
                    ),
                    sum(row["function_attempted_steps"] for row in selected),
                ),
                "function_rejected_attempts": sum(
                    row["function_rejected_attempts"] for row in selected
                ),
                "latent_initial_first_attempt_step_success_rate": _ratio(
                    sum(
                        row["latent_initial_first_attempt_accepted_steps"]
                        for row in selected
                    ),
                    sum(row["latent_initial_attempted_steps"] for row in selected),
                ),
                "latent_initial_rejected_attempts": sum(
                    row["latent_initial_rejected_attempts"] for row in selected
                ),
                "physical_requests": sum(row["physical_requests"] for row in selected),
                "observed_total_tokens": sum(
                    row["observed_total_tokens"] for row in selected
                ),
                "provider_seconds": sum(row["provider_seconds"] for row in selected),
                "required_nonlinearity_syntax_rate": _ratio(
                    sum(
                        row["required_nonlinearity_has_syntax_evidence"] is True
                        for row in nonlinear_applicable
                    ),
                    len(nonlinear_applicable),
                ),
                "relaxation_outer_sign_rate": _ratio(
                    sum(
                        row["all_tagged_relaxation_outer_signs_subtractive"] is True
                        for row in relaxation_applicable
                    ),
                    len(relaxation_applicable),
                ),
                "candidates_requiring_human_scientific_review": len(complete),
            }
        )
    terminal_results = sum(row["result_present"] for row in rows)
    return {
        "schema_version": "scientific-staged-function-granularity-summary-1",
        "status": (
            "complete" if terminal_results == len(plan["tasks"]) else "incomplete"
        ),
        "plan_sha256": plan["plan_sha256"],
        "source_function_plan_sha256": plan["source_function_plan_sha256"],
        "planned_tasks": len(plan["tasks"]),
        "terminal_results": terminal_results,
        "groups": groups,
        "rows": rows,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def _summary_row(task: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    """Build one planned row whether or not its terminal artifact exists."""
    result = record["result"] if record is not None else None
    events = result["events"] if result is not None else []
    function_events = [
        event for event in events if not event["step"].startswith("initial_")
    ]
    initial_events = [event for event in events if event["step"].startswith("initial_")]
    function_steps = {event["step"] for event in function_events}
    initial_steps = {event["step"] for event in initial_events}
    function_first_attempt = {
        event["step"]
        for event in function_events
        if event["accepted"] and event["attempt"] == 0
    }
    initial_first_attempt = {
        event["step"]
        for event in initial_events
        if event["accepted"] and event["attempt"] == 0
    }
    facts = result.get("scientific_review_facts") if result is not None else None
    candidate = result.get("candidate") if result is not None else None
    return {
        "task_id": task["task_id"],
        "pair_id": task["pair_id"],
        "reviewed_source_task": task["reviewed_source_task"],
        "seed": task["seed"],
        "generation_granularity": task["generation_granularity"],
        "result_present": record is not None,
        "status": result["status"] if result is not None else None,
        "complete_model": bool(result and result["complete_model"]),
        "error": result.get("error") if result is not None else None,
        "function_attempted_steps": len(function_steps),
        "function_first_attempt_accepted_steps": len(function_first_attempt),
        "function_rejected_attempts": sum(
            not event["accepted"] for event in function_events
        ),
        "latent_initial_attempted_steps": len(initial_steps),
        "latent_initial_first_attempt_accepted_steps": len(initial_first_attempt),
        "latent_initial_rejected_attempts": sum(
            not event["accepted"] for event in initial_events
        ),
        "physical_requests": result["physical_requests"] if result is not None else 0,
        "observed_total_tokens": (
            result["observed_total_tokens"] if result is not None else 0
        ),
        "provider_seconds": result["provider_seconds"] if result is not None else 0,
        "required_nonlinearity_has_syntax_evidence": (
            facts["required_nonlinearity_has_syntax_evidence"] if facts else None
        ),
        "all_tagged_relaxation_outer_signs_subtractive": (
            facts["all_tagged_relaxation_outer_signs_subtractive"] if facts else None
        ),
        "cross_lhs_shared_parameters": (
            facts["cross_lhs_shared_parameters"] if facts else None
        ),
        "human_review_required": facts["human_review_required"] if facts else [],
        "candidate_id": candidate["candidate_id"] if candidate else None,
        "state_equations": candidate["state_equations"] if candidate else [],
        "processes": candidate["processes"] if candidate else [],
        "parameter_count": len(candidate["parameters"]) if candidate else None,
        "latent_state_count": (
            sum(state["kind"] == "latent" for state in candidate["states"])
            if candidate
            else None
        ),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
