"""Frozen handoff experiment on reviewed topologies, with a shared model worker."""

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
from autoformalism.rebuttal.staged_topology_campaign import (
    StagedCampaignConfig,
    diagnostic_task,
    runtime_source_hash,
)
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_topology import (
    EquationDefinition,
    ModelingLimits,
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.staged_topology import content_hash, lower_topology


class SelectedTopology(StrictSchema):
    """A reviewed source outcome, pinned before any function-generation call."""

    task_id: Identifier
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FunctionCampaignConfig(StagedCampaignConfig):
    """Explicitly separate selected-topology handoff results from proposal rates."""

    protocol: Literal["scientific-staged-functions-1"]
    source_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_topologies: tuple[SelectedTopology, ...] = Field(
        min_length=1, max_length=8
    )

    @model_validator(mode="after")
    def unique_selections(self) -> FunctionCampaignConfig:
        """Reject duplicated reviewed sources before resolving input artifacts."""
        names = [item.task_id for item in self.selected_topologies]
        if len(set(names)) != len(names):
            raise ValueError("duplicate selected topology")
        return self


def function_launcher_hash() -> str:
    """Bind CLI and serving/allocation wrappers in addition to Python modules."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_function_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_topology_aces.slurm",
        "scripts/hpc/staged_topology_delta.slurm",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def function_diagnostic(name: str, limits: ModelingLimits) -> dict[str, Any]:
    """Exercise joint-source functions and generated auxiliaries on public toys."""
    task = diagnostic_task(name, limits)
    if name not in {"driven_memory", "generated_auxiliary"}:
        raise ValueError("unsupported function diagnostic")
    hidden = "z" if name == "driven_memory" else "a"
    equations = (
        EquationDefinition(
            name="x",
            definition="differential",
            terms=(
                {
                    "sources": ["x", hidden],
                    "outer_sign": "add",
                    "scientific_role": "joint driven response and self-relaxation",
                },
            ),
        ),
        EquationDefinition(
            name=hidden,
            definition="differential" if hidden == "z" else "algebraic",
            terms=(
                {
                    "sources": ["u", "z"] if hidden == "z" else ["u"],
                    "outer_sign": "add",
                    "scientific_role": "input response with relaxation when dynamic",
                },
            ),
        ),
    )
    topology, _ = lower_topology(
        PublicScientificBrief.model_validate(task["brief"]),
        tuple(
            ScientificVariable.model_validate(item)
            for item in task["initial_inventory"]
        ),
        equations,
        ValidationContext.model_validate(task["context"]),
    )
    return {
        "task_id": f"function_diagnostic_{name}",
        "kind": "diagnostic",
        "seed": 0,
        "brief": task["brief"],
        "context": task["context"],
        "source": {
            "complete_topology": True,
            "inventory": task["initial_inventory"],
            "equations": [e.model_dump(mode="json") for e in equations],
            "topology": topology.model_dump(mode="json"),
        },
    }


def freeze_function_campaign(
    config_path: Path, topology_plan: Path, topology_results: Path, output: Path
) -> dict[str, Any]:
    """Verify source plan, terminal identity and reviewed result digests."""
    config = FunctionCampaignConfig.model_validate_json(config_path.read_text())
    parent = json.loads(topology_plan.read_text())
    digest = content_hash(
        {key: value for key, value in parent.items() if key != "plan_sha256"}
    )
    if digest != config.source_plan_sha256 or digest != parent["plan_sha256"]:
        raise ValueError("source plan digest mismatch")
    indexed = {task["task_id"]: task for task in parent["tasks"]}
    tasks = []
    for selection in config.selected_topologies:
        original = indexed[selection.task_id]
        if (
            original["kind"] != "benchmark"
            or selection.task_id.rsplit("_seed", 1)[0] not in config.public_cells
        ):
            raise ValueError("selection is not an allowed public benchmark task")
        root = topology_results / selection.task_id
        source = json.loads((root / "result.json").read_text())
        terminal = json.loads((root / "terminal.json").read_text())
        if (
            content_hash(source) != selection.result_sha256
            or terminal["identity"] != content_hash([digest, original])
            or terminal["result"] != source
        ):
            raise ValueError("reviewed topology result or terminal identity differs")
        if (
            not source["complete_topology"]
            or not source["public_structure_checks_passed"]
        ):
            raise ValueError(
                "selected public topology did not pass the reviewed handoff gate"
            )
        for seed in config.seeds:
            tasks.append(
                {
                    "task_id": f"{selection.task_id}_functions_seed{seed}",
                    "kind": "benchmark",
                    "seed": seed,
                    "brief": original["brief"],
                    "context": original["context"],
                    "source": source,
                    "reviewed_source_task": selection.task_id,
                }
            )
    diagnostics = [
        function_diagnostic(name, config.limits) for name in config.diagnostic_fixtures
    ]
    plan = {
        "config": config.model_dump(mode="json"),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": function_launcher_hash(),
        "tasks": [*diagnostics[:1], *tasks, *diagnostics[1:]],
        "selection_policy": (
            "Two human-reviewed public-scientific finalists; conditional handoff "
            "experiment, not topology success-rate estimation"
        ),
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen function campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def run_function_campaign(
    plan_path: Path, output: Path, base_url: str, *, wall_seconds: float | None = None
) -> dict[str, Any]:
    """Run each frozen task once, draining and replaying exact cached attempts."""
    plan = json.loads(plan_path.read_text())
    if (
        content_hash(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        != plan["plan_sha256"]
    ):
        raise ValueError("frozen plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from frozen function campaign")
    if function_launcher_hash() != plan["launcher_sha256"]:
        raise ValueError("launcher differs from frozen function campaign")
    config = FunctionCampaignConfig.model_validate(plan["config"])
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
                        )
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "kind": task["kind"],
                        "result": result,
                    }
                    atomic_json(terminal, record)
                records.append(record)
                atomic_json(output / "summary.json", summarize_functions(plan, records))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize_functions(plan, records)
    atomic_json(output / "summary.json", summary)
    return summary


def summarize_functions(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report selected-topology function completion separately from diagnostics."""
    tasks = [
        {
            "task_id": item["task_id"],
            "kind": item["kind"],
            **{
                key: item["result"][key]
                for key in (
                    "status",
                    "complete_model",
                    "physical_requests",
                    "budget_charge",
                    "observed_total_tokens",
                    "unmeasured_requests",
                    "provider_seconds",
                )
            },
            "rejected_attempts": sum(
                not event["accepted"] for event in item["result"]["events"]
            ),
        }
        for item in records
    ]
    return {
        "plan_sha256": plan["plan_sha256"],
        "planned": len(plan["tasks"]),
        "finished": len(records),
        "tasks": tasks,
        "selected_topology_models": sum(
            item["kind"] == "benchmark" and item["complete_model"] for item in tasks
        ),
        "parameter_fitting_performed": False,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
