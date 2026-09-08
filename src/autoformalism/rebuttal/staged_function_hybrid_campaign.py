"""Targeted public-only evaluation of batched functions with atomic repair."""

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
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import content_hash


class HybridFunctionCampaignConfig(StagedCampaignConfig):
    """One prespecified hybrid arm on one reviewed topology."""

    protocol: Literal[
        "scientific-staged-function-hybrid-repair-1",
        "scientific-staged-function-hybrid-repair-2",
    ]
    source_function_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_topology: SelectedTopology
    generation_granularity: Literal["equation_batch_atomic_repair"]
    function_repair_policy: Literal["legacy", "certified_outer_gain"] = "legacy"

    @model_validator(mode="after")
    def targeted_public_probe(self) -> HybridFunctionCampaignConfig:
        """Keep this optimization probe small and scientifically targeted."""
        if len(self.public_cells) != 1:
            raise ValueError("hybrid repair pilot requires exactly one public cell")
        if self.diagnostic_fixtures:
            raise ValueError("hybrid repair pilot excludes toy diagnostics")
        expected_policy = (
            "legacy"
            if self.protocol == "scientific-staged-function-hybrid-repair-1"
            else "certified_outer_gain"
        )
        if self.function_repair_policy != expected_policy:
            raise ValueError(
                f"{self.protocol} requires function repair policy {expected_policy}"
            )
        return self


def hybrid_launcher_hash() -> str:
    """Bind the hybrid CLI and allocation wrappers to the frozen plan."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_function_hybrid_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_function_hybrid_repair_aces.slurm",
        "scripts/hpc/submit_staged_function_hybrid_repair_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_hybrid_campaign(
    config_path: Path,
    source_function_plan_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Freeze an opaque-only hybrid run from the exact reviewed source plan."""
    config = HybridFunctionCampaignConfig.model_validate_json(config_path.read_text())
    source_plan = json.loads(source_function_plan_path.read_text())
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
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

    selection = config.selected_topology
    expected_cell = selection.task_id.rsplit("_seed", 1)[0]
    if expected_cell != config.public_cells[0]:
        raise ValueError("selection is not the prespecified public cell")
    source_tasks = [
        task
        for task in source_plan["tasks"]
        if task.get("kind") == "benchmark"
        and task.get("reviewed_source_task") == selection.task_id
    ]
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
            raise ValueError("source plan contains inconsistent topology copies")

    tasks = [
        {
            "task_id": f"{selection.task_id}_hybrid_seed{seed}",
            "kind": "benchmark",
            "seed": seed,
            "generation_granularity": config.generation_granularity,
            "function_repair_policy": config.function_repair_policy,
            "brief": canonical["brief"],
            "context": canonical["context"],
            "source": source,
            "reviewed_source_task": selection.task_id,
        }
        for seed in config.seeds
    ]
    plan = {
        "schema_version": (
            "scientific-staged-function-hybrid-repair-plan-1"
            if config.protocol == "scientific-staged-function-hybrid-repair-1"
            else "scientific-staged-function-hybrid-repair-plan-2"
        ),
        "config": config.model_dump(mode="json"),
        "source_function_plan_sha256": source_digest,
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": hybrid_launcher_hash(),
        "tasks": tasks,
        "comparison_policy": (
            "Opaque reviewed topology only; same-LHS equation batches are audited "
            "per interaction and only rejected terms receive atomic repair"
        ),
        "parameter_identity_policy": (
            "interaction-local by default; intentional cross-interaction sharing "
            "is deferred"
        ),
        "baseline_reference": {
            "protocol": "scientific-staged-function-granularity-1",
            "same_source_plan_sha256": source_digest,
            "same_reviewed_topology": selection.model_dump(mode="json"),
            "same_model_settings": True,
            "same_seeds": True,
        },
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen hybrid campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def _require_matched_source_settings(
    config: HybridFunctionCampaignConfig,
    source: FunctionCampaignConfig,
) -> None:
    """Hold inference, serving resources, seeds and modeling limits fixed."""
    comparisons = (
        (config.model_settings, source.model_settings, "model settings"),
        (config.platform, source.platform, "platform"),
        (config.serving_image_sha256, source.serving_image_sha256, "image"),
        (config.served_context_tokens, source.served_context_tokens, "context"),
        (config.seeds, source.seeds, "seeds"),
        (config.limits, source.limits, "modeling limits"),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise ValueError(f"hybrid campaign {label} differ from source")
    if any(cell not in source.public_cells for cell in config.public_cells):
        raise ValueError("hybrid campaign public cell is absent from source")


def run_hybrid_campaign(
    plan_path: Path,
    output: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run each frozen seed once with deterministic drain and exact resume."""
    plan = json.loads(plan_path.read_text())
    if content_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    ) != plan.get("plan_sha256"):
        raise ValueError("frozen hybrid plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen hybrid campaign")
    if hybrid_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen hybrid campaign")
    config = HybridFunctionCampaignConfig.model_validate(plan["config"])
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
                            function_repair_policy=task.get(
                                "function_repair_policy", "legacy"
                            ),
                        )
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "seed": task["seed"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(output / "summary.json", summarize_hybrid(plan, records))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_hybrid(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def summarize_hybrid(
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Retain all planned seeds and keep transport, repair and science separate."""
    by_id = {record["task_id"]: record for record in records}
    rows = [_summary_row(task, by_id.get(task["task_id"])) for task in plan["tasks"]]
    complete = [row for row in rows if row["complete_model"]]
    audits = [audit for row in rows for audit in row["batch_term_audits"]]
    repairs = [audit for audit in audits if audit["atomic_repair_attempted"]]
    deterministic_repairs = [
        repair
        for audit in audits
        for key in (
            "deterministic_role_repairs",
            "atomic_deterministic_role_repairs",
        )
        for repair in audit.get(key, [])
    ]
    nonlinear = [
        audit
        for audit in audits
        if audit["functional_obligation"]["requires_nonlinear_source_dependence"]
    ]
    terminal_results = sum(row["result_present"] for row in rows)
    return {
        "schema_version": (
            "scientific-staged-function-hybrid-repair-summary-2"
            if plan["config"].get("function_repair_policy") == "certified_outer_gain"
            else "scientific-staged-function-hybrid-repair-summary-1"
        ),
        "status": (
            "complete" if terminal_results == len(plan["tasks"]) else "incomplete"
        ),
        "plan_sha256": plan["plan_sha256"],
        "source_function_plan_sha256": plan["source_function_plan_sha256"],
        "planned_tasks": len(plan["tasks"]),
        "terminal_results": terminal_results,
        "complete_models": len(complete),
        "complete_model_rate": _ratio(len(complete), len(rows)),
        "batch_first_attempt_step_success_rate": _ratio(
            sum(row["batch_first_attempt_accepted_steps"] for row in rows),
            sum(row["batch_attempted_steps"] for row in rows),
        ),
        "batch_rejected_attempts": sum(row["batch_rejected_attempts"] for row in rows),
        "latent_initial_first_attempt_step_success_rate": _ratio(
            sum(row["latent_initial_first_attempt_accepted_steps"] for row in rows),
            sum(row["latent_initial_attempted_steps"] for row in rows),
        ),
        "latent_initial_rejected_attempts": sum(
            row["latent_initial_rejected_attempts"] for row in rows
        ),
        "batch_term_count": len(audits),
        "batch_term_acceptance_rate": _ratio(
            sum(audit["batch_accepted"] for audit in audits), len(audits)
        ),
        "atomic_repair_activation_count": len(repairs),
        "atomic_repair_activation_rate": _ratio(len(repairs), len(audits)),
        "atomic_repair_success_rate": _ratio(
            sum(audit["atomic_repair_succeeded"] for audit in repairs),
            len(repairs),
        ),
        "deterministic_outer_gain_role_repair_count": len(deterministic_repairs),
        "nonlinear_obligation_term_count": len(nonlinear),
        "nonlinear_obligation_final_acceptance_rate": _ratio(
            sum(audit["final_source"] is not None for audit in nonlinear),
            len(nonlinear),
        ),
        "required_nonlinearity_syntax_rate": _ratio(
            sum(
                row["required_nonlinearity_has_syntax_evidence"] is True
                for row in complete
            ),
            len(complete),
        ),
        "models_with_cross_lhs_shared_parameters": sum(
            bool(row["cross_lhs_shared_parameters"]) for row in complete
        ),
        "physical_requests": sum(row["physical_requests"] for row in rows),
        "observed_total_tokens": sum(row["observed_total_tokens"] for row in rows),
        "provider_seconds": sum(row["provider_seconds"] for row in rows),
        "rows": rows,
        "automatic_winner_defined": False,
        "parameter_fitting_performed": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def _summary_row(task: dict[str, Any], record: dict[str, Any] | None) -> dict[str, Any]:
    """Expose repair decisions and candidate equations for every planned seed."""
    result = record["result"] if record is not None else None
    facts = result.get("scientific_review_facts") if result is not None else None
    candidate = result.get("candidate") if result is not None else None
    audits = result.get("batch_term_audits", []) if result is not None else []
    events = result.get("events", []) if result is not None else []
    batch_events = [
        event for event in events if event["step"].startswith("equation_functions_")
    ]
    initial_events = [event for event in events if event["step"].startswith("initial_")]
    batch_steps = {event["step"] for event in batch_events}
    initial_steps = {event["step"] for event in initial_events}
    return {
        "task_id": task["task_id"],
        "reviewed_source_task": task["reviewed_source_task"],
        "seed": task["seed"],
        "result_present": record is not None,
        "status": result["status"] if result is not None else None,
        "complete_model": bool(result and result["complete_model"]),
        "error": result.get("error") if result is not None else None,
        "batch_term_audits": audits,
        "batch_attempted_steps": len(batch_steps),
        "batch_first_attempt_accepted_steps": len(
            {
                event["step"]
                for event in batch_events
                if event["accepted"] and event["attempt"] == 0
            }
        ),
        "batch_rejected_attempts": sum(not event["accepted"] for event in batch_events),
        "latent_initial_attempted_steps": len(initial_steps),
        "latent_initial_first_attempt_accepted_steps": len(
            {
                event["step"]
                for event in initial_events
                if event["accepted"] and event["attempt"] == 0
            }
        ),
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
        "cross_lhs_shared_parameters": (
            facts["cross_lhs_shared_parameters"] if facts else None
        ),
        "human_review_required": facts["human_review_required"] if facts else [],
        "candidate_id": candidate["candidate_id"] if candidate else None,
        "state_equations": candidate["state_equations"] if candidate else [],
        "processes": candidate["processes"] if candidate else [],
        "parameters": candidate["parameters"] if candidate else [],
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
