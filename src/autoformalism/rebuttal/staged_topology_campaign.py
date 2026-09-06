"""Frozen public-only tasks for one warm model server across a bounded allocation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from autoformalism.data import BenchmarkRegistry
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    atomic_json,
)
from autoformalism.rebuttal.mechanisms import MechanismEvaluationSpec
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.staged_topology import (
    ModelingLimits,
    PublicScientificBrief,
    PublicVariable,
    ScientificRequirement,
    ScientificVariable,
)
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.staged_topology import build_scientific_brief, content_hash
from autoformalism.targets import PublicTargetContract


class StagedCampaignConfig(StrictSchema):
    """Small prospective probe; larger matrices require separately frozen configs."""

    protocol: Literal["scientific-staged-topology-1"]
    purpose: str
    platform: Literal["aces-h100x1", "aces-h100x2", "delta-a40x1"]
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(ge=16384)
    public_cells: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    diagnostic_fixtures: tuple[
        Literal["driven_memory", "generated_auxiliary", "two_memories"], ...
    ]
    limits: ModelingLimits
    wall_seconds: int = Field(ge=60)
    shutdown_margin_seconds: int = Field(ge=30)

    @model_validator(mode="after")
    def bounded_unique_tasks(self) -> StagedCampaignConfig:
        """Reject duplicates and an impossible deadline before freezing."""
        for values in (self.public_cells, self.seeds, self.diagnostic_fixtures):
            if len(set(values)) != len(values):
                raise ValueError("duplicate campaign tasks")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if (
            not self.model_settings.timeout_seconds
            < self.shutdown_margin_seconds
            < self.wall_seconds
        ):
            raise ValueError("require timeout < shutdown margin < worker wall time")
        return self


def runtime_source_hash() -> str:
    """Bind resume to every Python module in the installed source package."""
    package = Path(__file__).resolve().parents[1]
    return content_hash(
        {
            str(path.relative_to(package)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(package.rglob("*.py"))
        }
    )


def public_validation_context(benchmark_id: str) -> ValidationContext:
    """Use only the committed public registry; never instantiate a data loader."""
    spec = BenchmarkRegistry().get(benchmark_id)
    if spec.data_layout != "tidy_split_file":
        raise ValueError("this probe supports registered Phase-B public cells only")
    tier = benchmark_id.rsplit("_", 1)[-1]
    roles = spec.tier_roles[tier]
    return ValidationContext(
        targets=roles.targets,
        auxiliaries=roles.auxiliaries,
        external_inputs=spec.external_inputs,
        fixed_covariates=spec.fixed_covariates,
        time_symbol=spec.time_column,
        lagged_targets=roles.targets if spec.one_step_target_history else (),
    )


def diagnostic_task(name: str, limits: ModelingLimits) -> dict[str, Any]:
    """Public toy fixtures isolate Level 2 without supplying benchmark answers."""
    variables = [
        ScientificVariable(
            name="x", definition="differential", scientific_role="observed output"
        ),
        ScientificVariable(
            name="u", definition="supplied", scientific_role="external drive"
        ),
    ]
    auxiliaries: tuple[str, ...] = ()
    requirement = ScientificRequirement(
        id="response",
        public_requirement="A delayed response of x to u",
        targets=("x",),
        drivers=("u",),
        positive_requirements=("Represent dynamic memory.",),
    )
    context_text = (
        "Construct a driven continuous-time response. "
        "Generate x from u through internal memory z."
    )
    if name == "driven_memory":
        variables.append(
            ScientificVariable(
                name="z", definition="differential", scientific_role="memory of drive u"
            )
        )
    elif name == "generated_auxiliary":
        auxiliaries = ("a",)
        variables.append(
            ScientificVariable(
                name="a",
                definition="algebraic",
                scientific_role="instantaneous drive transformed from u and used by x",
            )
        )
        context_text = (
            "Generate x from a. The auxiliary a is measured but this candidate "
            "generates a algebraically from u; do not supply its trajectory."
        )
        requirement = requirement.model_copy(
            update={
                "public_requirement": "A response of x to u through generated a",
                "positive_requirements": (),
            }
        )
    elif name == "two_memories":
        variables.extend(
            ScientificVariable(
                name=label, definition="differential", scientific_role=role
            )
            for label, role in (
                ("fast", "fast memory of u"),
                ("slow", "slow memory of u"),
            )
        )
        context_text = (
            "Generate x using two scientifically distinct dynamic memories of u: "
            "a fast and a slow response. Preserve both accepted hypotheses."
        )
    else:
        raise ValueError("unknown diagnostic fixture")
    context = ValidationContext(
        targets=("x",), auxiliaries=auxiliaries, external_inputs=("u",)
    )
    brief = PublicScientificBrief(
        scientific_context=context_text,
        public_variables=(
            PublicVariable(name="x", data_role="target"),
            PublicVariable(name="u", data_role="external_input"),
            *(PublicVariable(name=a, data_role="auxiliary") for a in auxiliaries),
        ),
        requirements=(requirement,),
        limits=limits,
    )
    return {
        "task_id": f"diagnostic_{name}",
        "kind": "diagnostic",
        "seed": 0,
        "brief": brief.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "initial_inventory": [item.model_dump(mode="json") for item in variables],
    }


def freeze_campaign(
    config_path: Path, public_root: Path, repository: Path, output: Path
) -> dict[str, Any]:
    """Freeze exact prompts, contexts, settings and task order before model calls."""
    config = StagedCampaignConfig.model_validate_json(config_path.read_text())
    tasks: list[dict[str, Any]] = []
    for cell in config.public_cells:
        prompt = (public_root / "phase_b_v1" / cell / "proposer_prompt.txt").read_text()
        target = PublicTargetContract.model_validate_json(
            (
                repository / "configs/target_eval/phase_b_v2/specs" / f"{cell}.json"
            ).read_text()
        )
        mechanism = MechanismEvaluationSpec.model_validate_json(
            (
                repository / "configs/mechanism_eval/phase_b_v1/specs" / f"{cell}.json"
            ).read_text()
        )
        context = public_validation_context(cell)
        brief = build_scientific_brief(
            prompt, context, target, mechanism, limits=config.limits
        )
        for seed in config.seeds:
            tasks.append(
                {
                    "task_id": f"{cell}_seed{seed}",
                    "kind": "benchmark",
                    "seed": seed,
                    "brief": brief.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                    "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "initial_inventory": None,
                }
            )
    diagnostics = [
        diagnostic_task(name, config.limits) for name in config.diagnostic_fixtures
    ]
    # One simple stage fixture validates serving/schema first; the remaining
    # fixtures remain useful even if both public constructions fail.
    ordered = [*diagnostics[:1], *tasks, *diagnostics[1:]]
    plan = {
        "config": config.model_dump(mode="json"),
        "runtime_source_sha256": runtime_source_hash(),
        "tasks": ordered,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    if output.exists():
        if json.loads(output.read_text()) != plan:
            raise ValueError("existing frozen campaign differs")
    else:
        atomic_json(output, plan)
    return plan


def run_campaign(
    plan_path: Path, output: Path, base_url: str, *, wall_seconds: float | None = None
) -> dict[str, Any]:
    """Drain a frozen task list against one already-loaded model server."""
    plan = json.loads(plan_path.read_text())
    if (
        content_hash(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        != plan["plan_sha256"]
    ):
        raise ValueError("frozen plan digest mismatch")
    if runtime_source_hash() != plan["runtime_source_sha256"]:
        raise ValueError("runtime source differs from the frozen campaign")
    config = StagedCampaignConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    old_handlers = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    results: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    try:
        for task in plan["tasks"]:
            task_root = output / task["task_id"]
            task_root.mkdir(parents=True, exist_ok=True)
            with (task_root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                terminal_path = task_root / "terminal.json"
                identity = content_hash([plan["plan_sha256"], task])
                if terminal_path.exists():
                    record = json.loads(terminal_path.read_text())
                    if record["identity"] != identity:
                        raise ValueError(
                            "terminal result belongs to another frozen task"
                        )
                else:
                    client = StagedTopologyClient(
                        settings=config.model_settings,
                        base_url=base_url,
                        directory=task_root / "calls",
                        namespace=identity,
                        seed=task["seed"],
                        can_start=lambda: not stop
                        and time.monotonic()
                        < deadline - config.shutdown_margin_seconds,
                    )
                    try:
                        result = run_staged_topology(
                            PublicScientificBrief.model_validate(task["brief"]),
                            ValidationContext.model_validate(task["context"]),
                            client,
                            task_root,
                            initial_inventory=(
                                tuple(
                                    ScientificVariable.model_validate(item)
                                    for item in task["initial_inventory"]
                                )
                                if task["initial_inventory"] is not None
                                else None
                            ),
                        )
                    except DeferredCall:
                        break
                    record = {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "kind": task["kind"],
                        "result": result,
                    }
                    atomic_json(terminal_path, record)
                results.append(record)
                atomic_json(output / "summary.json", summarize_tasks(plan, results))
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    summary = summarize_tasks(plan, results)
    atomic_json(output / "summary.json", summary)
    return summary


def summarize_tasks(
    plan: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep diagnostic fixtures out of benchmark completion denominators."""
    endpoints = [
        {
            "task_id": item["task_id"],
            "kind": item["kind"],
            **{
                key: item["result"][key]
                for key in (
                    "status",
                    "complete_topology",
                    "public_structure_checks_passed",
                    "physical_requests",
                    "budget_charge",
                    "observed_total_tokens",
                    "unmeasured_requests",
                    "provider_seconds",
                )
            },
        }
        for item in records
    ]
    return {
        "plan_sha256": plan["plan_sha256"],
        "planned": len(plan["tasks"]),
        "finished": len(records),
        "tasks": endpoints,
        "benchmark_topologies": sum(
            item["kind"] == "benchmark" and item["complete_topology"]
            for item in endpoints
        ),
        "benchmark_requirements_passed": sum(
            item["kind"] == "benchmark" and item["public_structure_checks_passed"]
            for item in endpoints
        ),
        "test_data_opened": False,
        "private_reference_opened": False,
    }
