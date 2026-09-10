"""Two-round public function-first feedback pilot on frozen failed candidates."""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import re
import shutil
import signal
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.expressions import (
    RestrictedParser,
    ValidationContext,
    compile_candidate,
)
from autoformalism.fitting.collocation_sensitivity import (
    CollocationSensitivityConfig,
    fit_collocation_forward_sensitivity,
)
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedTopologyClient,
    atomic_json,
    visible_response,
)
from autoformalism.rebuttal.staged_prefit_fitting_campaign import load_public_data
from autoformalism.rebuttal.staged_topology_campaign import (
    StagedCampaignConfig,
    runtime_source_hash,
)
from autoformalism.schemas import (
    CandidateModel,
    ParameterScope,
    ParameterSpec,
)
from autoformalism.schemas.base import Identifier, StrictSchema
from autoformalism.schemas.staged_functions import FunctionParameter
from autoformalism.search.identity import candidate_identity
from autoformalism.staged_topology import content_hash


class ComponentRevision(StrictSchema):
    """One complete RHS replacement for a runtime-selected generated component."""

    component: Identifier
    expression: str = Field(min_length=1, max_length=4096)
    parameters: tuple[FunctionParameter, ...] = Field(max_length=64)

    @model_validator(mode="after")
    def unique_parameters(self) -> ComponentRevision:
        names = [item.name for item in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("duplicate local parameter declaration")
        return self


class ComponentRevisionReply(StrictSchema):
    """A coherent bounded revision of all runtime-selected components."""

    revisions: tuple[ComponentRevision, ...] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def unique_components(self) -> ComponentRevisionReply:
        names = [item.component for item in self.revisions]
        if len(names) != len(set(names)):
            raise ValueError("duplicate component revision")
        return self


class MultiRoundFeedbackConfig(StagedCampaignConfig):
    """Frozen two-task, two-round function-first routing experiment."""

    protocol: Literal["scientific-staged-multiround-feedback-1"]
    source_task_indices: tuple[int, int]
    round_count: Literal[2] = 2
    fit: CollocationSensitivityConfig
    routing_policy: Literal[
        "function_first_then_topology_if_persistent_instability"
    ] = "function_first_then_topology_if_persistent_instability"

    @model_validator(mode="after")
    def bounded_public_pilot(self) -> MultiRoundFeedbackConfig:
        if len(self.public_cells) != 1 or self.public_cells[0] != (
            "phase_b_anonymous_system_task_canonical_opaque_hard"
        ):
            raise ValueError("pilot requires the one reviewed opaque hard cell")
        if self.source_task_indices != (3, 4) or self.seeds != (0, 1):
            raise ValueError("pilot is fixed to unresolved source tasks 3 and 4")
        if self.diagnostic_fixtures:
            raise ValueError("pilot excludes diagnostic fixtures")
        return self


FUNCTION_REVISION_SYSTEM_PROMPT = """You repair functions in a fitted
continuous-time model.
The runtime selected a small set of generated components implicated by a numerical
failure. Return exactly one complete expression and its fitted parameter names and
qualitative roles for every selected component. Preserve the displayed nonparameter
source set exactly; therefore this is a function revision, not a topology revision.
Avoid positive superlinear closed-feedback growth and prefer bounded or saturating
nonlinear response when scientifically compatible with the public requirement.
Do not change states, processes, target mappings, initial values, or other equations.
Do not provide ranges, scopes, units, prose, scores, or a complete model."""

TOPOLOGY_REVISION_SYSTEM_PROMPT = """You backtrack one equation-dependency
decision in a continuous-time model after a bounded function repair remained
numerically unstable.
Return exactly one complete expression and its fitted parameter names and qualitative
roles for every selected generated component. You may add or remove dependencies,
but only from the displayed existing public and generated variables. Preserve the
variable inventory, target mapping, all unselected equations, and the required causal
pathway. Prefer the smallest scientifically motivated dependency correction that
removes the diagnosed unstable closed loop. Do not provide ranges, scopes, units,
prose, scores, or a complete model."""


def launcher_hash() -> str:
    """Bind the complete checkpoint-producing campaign surface."""
    repository = Path(__file__).resolve().parents[3]
    paths = (
        "scripts/staged_multiround_feedback_campaign.py",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/staged_multiround_feedback_aces.slurm",
        "scripts/hpc/submit_staged_multiround_feedback_aces.sh",
    )
    return content_hash(
        {
            path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
            for path in paths
        }
    )


def freeze_campaign(
    config_path: Path,
    source_rescue_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze two unresolved candidates and their public development inputs."""
    config = MultiRoundFeedbackConfig.model_validate_json(config_path.read_text())
    source_plan_path = source_rescue_root / "plan.json"
    source_summary_path = source_rescue_root / "summary" / "summary.json"
    source_plan = _read_object(source_plan_path)
    source_summary = _read_object(source_summary_path)
    source_digest = content_hash(
        {key: value for key, value in source_plan.items() if key != "plan_sha256"}
    )
    if source_digest != source_plan.get("plan_sha256"):
        raise ValueError("source rescue plan digest differs")
    if (
        source_plan.get("schema_version") != "scientific-staged-fitter-rescue-plan-1"
        or source_summary.get("status") != "complete"
        or source_summary.get("terminal_results") != 6
        or source_summary.get("test_data_opened") is not False
        or source_summary.get("private_reference_opened") is not False
    ):
        raise ValueError("source rescue campaign is not an eligible public source")
    frozen = output_root / "frozen"
    public_ledger: dict[str, str] = {}
    for relative, expected in source_plan["public_asset_ledger"].items():
        source = source_rescue_root / relative
        if _sha256(source) != expected:
            raise ValueError(f"source public artifact differs: {relative}")
        destination = frozen / "public" / Path(relative).relative_to("frozen/public")
        _copy_once(source, destination)
        public_ledger[str(destination.relative_to(output_root))] = _sha256(destination)

    tasks: list[dict[str, Any]] = []
    for ordinal, source_index in enumerate(config.source_task_indices):
        source_task = source_plan["tasks"][source_index]
        if (
            source_task["benchmark_id"] != config.public_cells[0]
            or int(source_task["seed"]) != config.seeds[ordinal]
        ):
            raise ValueError("selected source task identity differs")
        candidate_source = source_rescue_root / source_task["candidate_path"]
        rescue_source = source_rescue_root / "tasks" / f"task_{source_index:03d}.json"
        if (
            _sha256(candidate_source) != source_task["candidate_file_sha256"]
            or not rescue_source.is_file()
        ):
            raise ValueError("selected source candidate or rescue record differs")
        rescue = _read_object(rescue_source)
        if rescue.get("attribution") != "unresolved_after_bounded_rescue":
            raise ValueError("pilot source is no longer an unresolved rescue task")
        candidate_path = frozen / "candidates" / f"candidate_{ordinal:03d}.json"
        rescue_path = frozen / "source_rescue" / f"task_{ordinal:03d}.json"
        _copy_once(candidate_source, candidate_path)
        _copy_once(rescue_source, rescue_path)
        tasks.append(
            {
                "task_index": ordinal,
                "task_id": (
                    f"{source_task['benchmark_id']}_seed"
                    f"{source_task['seed']}_multiround"
                ),
                "benchmark_id": source_task["benchmark_id"],
                "tier": source_task["tier"],
                "seed": source_task["seed"],
                "source_task_index": source_index,
                "candidate_path": str(candidate_path.relative_to(output_root)),
                "candidate_file_sha256": _sha256(candidate_path),
                "source_rescue_path": str(rescue_path.relative_to(output_root)),
                "source_rescue_file_sha256": _sha256(rescue_path),
            }
        )
    plan = {
        "schema_version": "scientific-staged-multiround-feedback-plan-1",
        "config": config.model_dump(mode="json"),
        "source_rescue_plan_sha256": source_digest,
        "source_rescue_summary_file_sha256": _sha256(source_summary_path),
        "public_asset_ledger": public_ledger,
        "public_asset_ledger_sha256": content_hash(public_ledger),
        "runtime_source_sha256": runtime_source_hash(),
        "launcher_sha256": launcher_hash(),
        "tasks": tasks,
        "routing_priority": [
            "deterministic_contract_failure",
            "function_revision_for_numerical_instability",
            "topology_dependency_backtrack_after_persistent_instability",
        ],
        "parameter_fitting_method": "collocation_then_forward_sensitivity",
        "validation_used_for_parameter_fitting": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }
    plan["plan_sha256"] = content_hash(plan)
    _write_once_json(output_root / "plan.json", plan)
    _copy_once(source_plan_path, frozen / "source_plan.json")
    _copy_once(source_summary_path, frozen / "source_summary.json")
    return plan


def run_campaign(
    plan_path: Path,
    output_root: Path,
    base_url: str,
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run and resume every selected candidate through two routed rounds."""
    plan = _verified_plan(plan_path.parent)
    config = MultiRoundFeedbackConfig.model_validate(plan["config"])
    deadline = time.monotonic() + (wall_seconds or config.wall_seconds)
    stop = False

    def drain(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous = {
        sig: signal.signal(sig, drain) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        for task in plan["tasks"]:
            if stop or time.monotonic() >= deadline - config.shutdown_margin_seconds:
                break
            root = output_root / task["task_id"]
            root.mkdir(parents=True, exist_ok=True)
            with (root / "worker.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                terminal = root / "terminal.json"
                identity = content_hash([plan["plan_sha256"], task])
                if terminal.exists():
                    record = _read_object(terminal)
                    if record.get("identity") != identity:
                        raise ValueError("terminal result belongs to another task")
                    continue
                client = StagedTopologyClient(
                    settings=config.model_settings,
                    base_url=base_url,
                    directory=root / "calls",
                    namespace=identity,
                    seed=task["seed"],
                    can_start=lambda: not stop
                    and time.monotonic() < deadline - config.shutdown_margin_seconds,
                )
                try:
                    result = run_task(plan_path.parent, task, root, client)
                except DeferredCall:
                    break
                except Exception as exc:  # one failed candidate must not hide the other
                    progress = (
                        _read_object(root / "progress.json")
                        if (root / "progress.json").exists()
                        else {"rounds": []}
                    )
                    result = {
                        "schema_version": (
                            "scientific-staged-multiround-feedback-result-1"
                        ),
                        "status": "failed",
                        "task_id": task["task_id"],
                        "benchmark_id": task["benchmark_id"],
                        "seed": task["seed"],
                        "rounds": progress["rounds"],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:8000],
                        "physical_requests": len(client.records),
                        "observed_total_tokens": sum(
                            int(item.get("observed_total_tokens") or 0)
                            for item in client.records
                        ),
                        "provider_seconds": sum(
                            float(item.get("latency_seconds") or 0.0)
                            for item in client.records
                        ),
                        "parameter_fitting_method": (
                            "collocation_then_forward_sensitivity"
                        ),
                        "validation_used_for_parameter_fitting": False,
                        "scientific_judge_called": False,
                        "test_data_opened": False,
                        "private_reference_opened": False,
                        "automatic_winner_defined": False,
                    }
                atomic_json(
                    terminal,
                    {
                        "identity": identity,
                        "task_id": task["task_id"],
                        "result": result,
                    },
                )
                atomic_json(output_root / "summary.json", summarize(plan, output_root))
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    summary = summarize(plan, output_root)
    atomic_json(output_root / "summary.json", summary)
    return summary


def run_task(
    campaign_root: Path,
    task: Mapping[str, Any],
    output: Path,
    client: StagedTopologyClient,
) -> dict[str, Any]:
    """Apply function-first routing and fit each revision before continuing."""
    plan = _verified_plan(campaign_root)
    config = MultiRoundFeedbackConfig.model_validate(plan["config"])
    candidate_path = campaign_root / str(task["candidate_path"])
    rescue_path = campaign_root / str(task["source_rescue_path"])
    if (
        _sha256(candidate_path) != task["candidate_file_sha256"]
        or _sha256(rescue_path) != task["source_rescue_file_sha256"]
    ):
        raise ValueError("frozen task input differs")
    current = CandidateModel.model_validate_json(candidate_path.read_text())
    source_rescue = _read_object(rescue_path)
    dataset, context = load_public_data(
        campaign_root / "frozen" / "public",
        str(task["benchmark_id"]),
        str(task["tier"]),
    )
    prompt_path = (
        campaign_root
        / "frozen"
        / "public"
        / "phase_b_v1"
        / str(task["benchmark_id"])
        / "proposer_prompt.txt"
    )
    scientific_context = re.split(
        r"(?m)^F\.\s+Required response\s*$", prompt_path.read_text(), maxsplit=1
    )[0].rstrip()
    rounds: list[dict[str, Any]] = []
    prior_feedback = _source_feedback(source_rescue)
    for round_index in range(1, config.round_count + 1):
        round_root = output / f"round_{round_index:03d}"
        round_root.mkdir(parents=True, exist_ok=True)
        candidate_checkpoint = round_root / "candidate.json"
        revision_checkpoint = round_root / "revision.json"
        fit_checkpoint = round_root / "fit.json"
        route = _route(round_index, rounds)
        selected = (
            tuple(rounds[-1]["selected_components"])
            if rounds
            else select_revision_components(current)
        )
        if not selected:
            raise ValueError("runtime could not identify a bounded revision scope")
        if candidate_checkpoint.exists() and revision_checkpoint.exists():
            revised = CandidateModel.model_validate_json(
                candidate_checkpoint.read_text()
            )
            revision_record = _read_object(revision_checkpoint)
            if revision_record.get("parent_sha256") != _candidate_hash(current):
                raise ValueError("revision checkpoint parent differs")
        else:
            revised, revision_record = _request_revision(
                client=client,
                route=route,
                candidate=current,
                selected=selected,
                context=context,
                scientific_context=scientific_context,
                numerical_feedback=prior_feedback,
                round_index=round_index,
            )
            _write_once_json(revision_checkpoint, revision_record)
            _write_once_json(candidate_checkpoint, revised.model_dump(mode="json"))
        if fit_checkpoint.exists():
            fit = _read_object(fit_checkpoint)
        else:
            try:
                model = compile_candidate(revised, context)
                fit = fit_collocation_forward_sensitivity(
                    model,
                    dataset.train,
                    dataset.validation,
                    config.fit,
                    round_root / "numerical",
                )
            except Exception as exc:  # generated expressions remain untrusted
                fit = {
                    "schema_version": "collocation-forward-sensitivity-fit-1",
                    "status": "fit_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:8000],
                    "training": None,
                    "validation": None,
                }
            _write_once_json(fit_checkpoint, fit)
        round_record = {
            "round_index": round_index,
            "route": route,
            "selected_components": list(selected),
            "parent_candidate_sha256": _candidate_hash(current),
            "candidate_sha256": _candidate_hash(revised),
            "revision": revision_record,
            "fit": fit,
            "candidate_equations": _component_expressions(revised),
            "numerically_stable": fit.get("status") == "complete",
            "training_normalized_mse": _score(fit, "training"),
            "validation_normalized_mse": _score(fit, "validation"),
        }
        rounds.append(round_record)
        atomic_json(output / "progress.json", {"rounds": rounds})
        current = revised
        prior_feedback = _round_feedback(round_record)
    return {
        "schema_version": "scientific-staged-multiround-feedback-result-1",
        "status": "complete",
        "task_id": task["task_id"],
        "benchmark_id": task["benchmark_id"],
        "seed": task["seed"],
        "rounds": rounds,
        "physical_requests": len(client.records),
        "observed_total_tokens": sum(
            int(item.get("observed_total_tokens") or 0) for item in client.records
        ),
        "provider_seconds": sum(
            float(item.get("latency_seconds") or 0.0) for item in client.records
        ),
        "parameter_fitting_method": "collocation_then_forward_sensitivity",
        "validation_used_for_parameter_fitting": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }


def select_revision_components(candidate: CandidateModel) -> tuple[str, ...]:
    """Select the smallest nonlinear generated SCC implicated in blow-up risk."""
    expressions = _component_expressions(candidate)
    generated = set(expressions)
    parameters = {item.name for item in candidate.parameters}
    dependencies = {
        name: set(RestrictedParser().parse(expression, location=name).symbols)
        & generated
        for name, expression in expressions.items()
    }

    def reaches(start: str, target: str) -> bool:
        frontier = list(dependencies[start])
        seen: set[str] = set()
        while frontier:
            item = frontier.pop()
            if item == target:
                return True
            if item not in seen:
                seen.add(item)
                frontier.extend(dependencies[item] - seen)
        return False

    risky: list[tuple[str, ...]] = []
    for name, expression in expressions.items():
        parsed = RestrictedParser().parse(expression, location=name)
        source_symbols = set(parsed.symbols) - parameters
        if not _has_superlinear_dependence(parsed.tree, source_symbols & generated):
            continue
        component = tuple(
            sorted(
                item
                for item in generated
                if (item == name or reaches(name, item))
                and (item == name or reaches(item, name))
            )
        )
        if component and component not in risky:
            risky.append(component)
    if risky:
        selected = min(risky, key=lambda item: (len(item), item))
        return selected[:4]
    nonlinear = [
        name
        for name, expression in expressions.items()
        if _has_superlinear_dependence(
            RestrictedParser().parse(expression, location=name).tree,
            generated,
        )
    ]
    return tuple(nonlinear[:2] or list(expressions)[:2])


def apply_component_revision(
    candidate: CandidateModel,
    reply: ComponentRevisionReply,
    context: ValidationContext,
    *,
    selected: tuple[str, ...],
    route: Literal["function_revision", "function_refinement", "topology_revision"],
) -> tuple[CandidateModel, dict[str, Any]]:
    """Apply one closed-schema revision and certify its structural granularity."""
    if tuple(item.component for item in reply.revisions) != selected:
        raise ValueError(
            "revision components must match the selected order exactly: "
            f"expected={list(selected)}"
        )
    expressions = _component_expressions(candidate)
    unknown = set(selected) - set(expressions)
    if unknown:
        raise ValueError(f"selected components do not exist: {sorted(unknown)}")
    generated = set(expressions)
    supplied = set(context.forcing_channels) | {context.time_symbol}
    old_parameters = {item.name: item for item in candidate.parameters}
    revised_parameters: dict[str, ParameterSpec] = {}
    replacement: dict[str, str] = {}
    source_diffs: dict[str, dict[str, list[str]]] = {}
    for item in reply.revisions:
        declared = {parameter.name for parameter in item.parameters}
        if declared & (generated | supplied):
            raise ValueError("revision parameter collides with a scientific symbol")
        parsed = RestrictedParser().parse(item.expression, location=item.component)
        actual = set(parsed.symbols) - declared
        illegal = actual - generated - supplied
        if illegal:
            raise ValueError(
                f"revision contains unavailable symbols: {sorted(illegal)}"
            )
        old = RestrictedParser().parse(
            expressions[item.component], location=item.component
        )
        old_sources = set(old.symbols) - set(old_parameters)
        if route != "topology_revision" and actual != old_sources:
            raise ValueError(
                "function revision changed topology sources: "
                f"missing={sorted(old_sources - actual)}, "
                f"extra={sorted(actual - old_sources)}"
            )
        source_diffs[item.component] = {
            "removed": sorted(old_sources - actual),
            "added": sorted(actual - old_sources),
        }
        replacement[item.component] = item.expression
        for parameter in item.parameters:
            spec = ParameterSpec(
                name=parameter.name,
                scope=ParameterScope.GLOBAL,
                role=parameter.role,
            )
            previous = revised_parameters.get(parameter.name)
            if previous is not None and previous.role is not spec.role:
                raise ValueError("parameter role conflicts across revised components")
            if (
                parameter.name in old_parameters
                and old_parameters[parameter.name].role is not spec.role
            ):
                raise ValueError("reused parameter role conflicts with the parent")
            revised_parameters[parameter.name] = spec
    final_expressions = {**expressions, **replacement}
    active_symbols = set().union(
        *(
            set(RestrictedParser().parse(value, location=name).symbols)
            for name, value in final_expressions.items()
        ),
        *(
            set(
                RestrictedParser()
                .parse(item.expression, location=f"mapping:{item.channel}")
                .symbols
            )
            for item in candidate.observation_mappings
        ),
    )
    parameters = {
        name: spec for name, spec in old_parameters.items() if name in active_symbols
    }
    parameters.update(revised_parameters)
    states = tuple(
        item.model_copy(update={"rhs": replacement.get(item.state, item.rhs)})
        for item in candidate.state_equations
    )
    processes = tuple(
        item.model_copy(
            update={"expression": replacement.get(item.name, item.expression)}
        )
        for item in candidate.processes
    )
    parent_identity = candidate_identity(candidate)
    payload = candidate.model_dump(mode="json")
    payload.update(
        candidate_id=(
            "round_"
            + content_hash([candidate.candidate_id, reply.model_dump(mode="json")])[:16]
        ),
        parent_candidate_id=candidate.candidate_id,
        change_summary=f"runtime-applied {route} of {', '.join(selected)}",
        state_equations=[item.model_dump(mode="json") for item in states],
        processes=[item.model_dump(mode="json") for item in processes],
        parameters=[item.model_dump(mode="json") for item in parameters.values()],
    )
    revised = CandidateModel.model_validate(payload)
    compile_candidate(revised, context)
    revised_identity = candidate_identity(revised)
    if revised_identity.functional_sha256 == parent_identity.functional_sha256:
        raise ValueError("revision did not change the executable functions")
    topology_changed = (
        revised_identity.topology_sha256 != parent_identity.topology_sha256
    )
    if route == "topology_revision" and not topology_changed:
        raise ValueError("topology backtrack did not change any dependency")
    if route != "topology_revision" and topology_changed:
        raise ValueError("function revision changed the topology identity")
    old_reachability = _target_reachability(candidate, context)
    revised_reachability = _target_reachability(revised, context)
    lost_reachability = {
        target: sorted(old_reachability[target] - revised_reachability[target])
        for target in old_reachability
        if old_reachability[target] - revised_reachability[target]
    }
    if route == "topology_revision" and lost_reachability:
        raise ValueError(
            "topology backtrack disconnected an existing target pathway: "
            f"{lost_reachability}"
        )
    return revised, {
        "parent_identity": parent_identity.model_dump(mode="json"),
        "revised_identity": revised_identity.model_dump(mode="json"),
        "topology_changed": topology_changed,
        "source_diffs": source_diffs,
        "target_reachability_preserved": not lost_reachability,
    }


def _request_revision(
    *,
    client: StagedTopologyClient,
    route: str,
    candidate: CandidateModel,
    selected: tuple[str, ...],
    context: ValidationContext,
    scientific_context: str,
    numerical_feedback: Mapping[str, Any],
    round_index: int,
) -> tuple[CandidateModel, dict[str, Any]]:
    diagnostic: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    expressions = _component_expressions(candidate)
    parameter_names = {item.name for item in candidate.parameters}
    request = {
        "schema_version": "component-revision-request-1",
        "route": route,
        "scientific_context": scientific_context,
        "selected_components": [
            {
                "component": name,
                "current_expression": expressions[name],
                "required_nonparameter_sources": sorted(
                    set(
                        RestrictedParser()
                        .parse(expressions[name], location=name)
                        .symbols
                    )
                    - parameter_names
                )
                if route != "topology_revision"
                else None,
            }
            for name in selected
        ],
        "all_current_equations": expressions,
        "available_nonparameter_symbols": sorted(
            set(expressions) | set(context.forcing_channels) | {context.time_symbol}
        ),
        "numerical_feedback": dict(numerical_feedback),
        "runtime_priority": (
            "repair numerical instability at function level before topology"
            if route != "topology_revision"
            else "function repair was exhausted; minimally backtrack dependencies"
        ),
    }
    system = (
        TOPOLOGY_REVISION_SYSTEM_PROMPT
        if route == "topology_revision"
        else FUNCTION_REVISION_SYSTEM_PROMPT
    )
    for attempt in range(client.settings.attempts_per_step):
        payload = {**request, "runtime_diagnostic": diagnostic}
        record = client.call(
            system=system,
            user=json.dumps(payload, sort_keys=True, ensure_ascii=False),
            response_model=ComponentRevisionReply,
            step=f"round_{round_index}_{route}",
            attempt=attempt,
        )
        rejected: Any = None
        try:
            rejected = visible_response(record)
            reply = ComponentRevisionReply.model_validate(rejected)
            revised, audit = apply_component_revision(
                candidate,
                reply,
                context,
                selected=selected,
                route=route,  # type: ignore[arg-type]
            )
        except (ValueError, TypeError, KeyError) as exc:
            diagnostic = {
                "rejected_response": rejected,
                "error": str(exc)[:6000],
                "instruction": "repair only the stated contract failure",
            }
            attempts.append(
                {
                    "attempt": attempt,
                    "request_hash": record["request_hash"],
                    "accepted": False,
                    "error": str(exc)[:6000],
                }
            )
            continue
        attempts.append(
            {
                "attempt": attempt,
                "request_hash": record["request_hash"],
                "accepted": True,
            }
        )
        return revised, {
            "schema_version": "component-revision-record-1",
            "route": route,
            "parent_sha256": _candidate_hash(candidate),
            "reply": reply.model_dump(mode="json"),
            "audit": audit,
            "attempts": attempts,
        }
    raise ValueError(f"bounded component revision exhausted for {route}")


def _route(round_index: int, rounds: list[dict[str, Any]]) -> str:
    if round_index == 1:
        return "function_revision"
    if rounds[-1]["numerically_stable"]:
        return "function_refinement"
    return "topology_revision"


def _source_feedback(source: Mapping[str, Any]) -> dict[str, Any]:
    attempts = source.get("rescue_attempts", [])
    failures = []
    for attempt in attempts:
        score = attempt.get("fresh_training_score") or {}
        for item in score.get("failures", []):
            message = str(item.get("message", ""))
            if message and message not in failures:
                failures.append(message)
    return {
        "source_attribution": source.get("attribution"),
        "source_fit_success": source.get("source_fit_success"),
        "full_training_rollout_complete": False,
        "failure_messages": failures[:5],
        "diagnosis": (
            "full-horizon integration instability persisted across bounded starts; "
            "inspect positive superlinear closed feedback before changing topology"
        ),
    }


def _round_feedback(round_record: Mapping[str, Any]) -> dict[str, Any]:
    fit = round_record["fit"]
    failures: list[str] = []
    for split in ("training", "validation"):
        payload = fit.get(split) or {}
        failures.extend(str(item) for item in payload.get("failed_trajectories", []))
    return {
        "previous_route": round_record["route"],
        "full_training_rollout_complete": round_record["numerically_stable"],
        "training_normalized_mse": round_record["training_normalized_mse"],
        "validation_normalized_mse": round_record["validation_normalized_mse"],
        "initializer_success": bool((fit.get("initializer") or {}).get("success")),
        "initializer_message": (fit.get("initializer") or {}).get("message"),
        "optimizer_success": bool(
            (fit.get("refinement") or {}).get("optimizer_success")
        ),
        "optimizer_message": (fit.get("refinement") or {}).get("message"),
        "fit_error": fit.get("error"),
        "failure_messages": failures[:5],
    }


def summarize(plan: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in plan["tasks"]:
        terminal = output_root / task["task_id"] / "terminal.json"
        if not terminal.exists():
            rows.append({"task_id": task["task_id"], "result_present": False})
            continue
        result = _read_object(terminal)["result"]
        compact_rounds = [_compact_round(item) for item in result["rounds"]]
        rows.append(
            {
                "task_id": task["task_id"],
                "benchmark_id": task["benchmark_id"],
                "seed": task["seed"],
                "result_present": True,
                "status": result["status"],
                "rounds": compact_rounds,
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "physical_requests": result["physical_requests"],
                "observed_total_tokens": result["observed_total_tokens"],
                "provider_seconds": result["provider_seconds"],
            }
        )
    rounds = [item for row in rows for item in row.get("rounds", [])]
    route_counts = Counter(item["route"] for item in rounds)
    stable = [item for item in rounds if item["numerically_stable"]]
    first_rounds = [item for item in rounds if item["round_index"] == 1]
    final_rounds = [row["rounds"][-1] for row in rows if row.get("rounds")]
    return {
        "schema_version": "scientific-staged-multiround-feedback-summary-1",
        "status": "complete"
        if all(row["result_present"] for row in rows)
        else "incomplete",
        "planned_tasks": len(rows),
        "terminal_results": sum(row["result_present"] for row in rows),
        "planned_rounds": len(rows) * int(plan["config"]["round_count"]),
        "completed_rounds": len(rounds),
        "numerically_stable_round_rate": _rate(len(stable), len(rounds)),
        "round_one_numerically_stable_rate": _rate(
            sum(item["numerically_stable"] for item in first_rounds),
            len(first_rounds),
        ),
        "final_numerically_stable_rate": _rate(
            sum(item["numerically_stable"] for item in final_rounds),
            len(final_rounds),
        ),
        "route_counts": dict(sorted(route_counts.items())),
        "topology_backtrack_count": route_counts.get("topology_revision", 0),
        "collocation_initializer_success_rate": _rate(
            sum(
                bool((item["fit"].get("initializer") or {}).get("success"))
                for item in rounds
            ),
            len(rounds),
        ),
        "forward_sensitivity_optimizer_success_rate": _rate(
            sum(
                bool((item["fit"].get("refinement") or {}).get("optimizer_success"))
                for item in rounds
            ),
            len(rounds),
        ),
        "median_validation_nmse_stable_rounds": (
            float(np.median([item["validation_normalized_mse"] for item in stable]))
            if stable
            else None
        ),
        "physical_requests": sum(int(row.get("physical_requests", 0)) for row in rows),
        "observed_total_tokens": sum(
            int(row.get("observed_total_tokens", 0)) for row in rows
        ),
        "provider_seconds": sum(
            float(row.get("provider_seconds", 0.0)) for row in rows
        ),
        "rows": rows,
        "parameter_fitting_method": "collocation_then_forward_sensitivity",
        "validation_used_for_parameter_fitting": False,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "automatic_winner_defined": False,
    }


def _compact_round(item: Mapping[str, Any]) -> dict[str, Any]:
    fit = item["fit"]
    revision = item.get("revision") or {}
    audit = revision.get("audit") or {}
    initializer = fit.get("initializer") or {}
    refinement = fit.get("refinement") or {}
    return {
        "round_index": item["round_index"],
        "route": item["route"],
        "selected_components": item["selected_components"],
        "candidate_sha256": item["candidate_sha256"],
        "topology_changed": audit.get("topology_changed"),
        "numerically_stable": item["numerically_stable"],
        "training_normalized_mse": item["training_normalized_mse"],
        "validation_normalized_mse": item["validation_normalized_mse"],
        "fit_status": fit.get("status"),
        "fit_error_type": fit.get("error_type"),
        "fit_error": fit.get("error"),
        "collocation_initializer_success": initializer.get("success"),
        "collocation_initializer_message": initializer.get("message"),
        "forward_sensitivity_optimizer_success": refinement.get("optimizer_success"),
        "forward_sensitivity_optimizer_message": refinement.get("message"),
    }


def _component_expressions(candidate: CandidateModel) -> dict[str, str]:
    return {
        **{item.state: item.rhs for item in candidate.state_equations},
        **{item.name: item.expression for item in candidate.processes},
    }


def _candidate_hash(candidate: CandidateModel) -> str:
    return content_hash(candidate.model_dump(mode="json"))


def _has_superlinear_dependence(tree: ast.AST, sources: set[str]) -> bool:
    def names(node: ast.AST) -> set[str]:
        return {
            item.id
            for item in ast.walk(node)
            if isinstance(item, ast.Name) and item.id in sources
        }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Pow)
            and names(node.left)
            and (not isinstance(node.right, ast.Constant) or node.right.value > 1)
        ):
            return True
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and names(node.left)
            and names(node.right)
        ):
            return True
    return False


def _target_reachability(
    candidate: CandidateModel, context: ValidationContext
) -> dict[str, set[str]]:
    """Return generated/supplied ancestors of each public target expression."""
    expressions = _component_expressions(candidate)
    parameters = {item.name for item in candidate.parameters}
    scientific = set(expressions) | set(context.forcing_channels)
    dependencies = {
        name: (set(RestrictedParser().parse(value, location=name).symbols) - parameters)
        & scientific
        for name, value in expressions.items()
    }

    def ancestors(symbols: set[str]) -> set[str]:
        seen: set[str] = set()
        frontier = list(symbols & scientific)
        while frontier:
            symbol = frontier.pop()
            if symbol in seen:
                continue
            seen.add(symbol)
            frontier.extend(dependencies.get(symbol, set()) - seen)
        return seen

    return {
        mapping.channel: ancestors(
            set(
                RestrictedParser()
                .parse(mapping.expression, location=f"mapping:{mapping.channel}")
                .symbols
            )
            - parameters
        )
        for mapping in candidate.observation_mappings
    }


def _score(fit: Mapping[str, Any], split: str) -> float | None:
    payload = fit.get(split)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("normalized_mse")
    return (
        float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
    )


def _verified_plan(output_root: Path) -> dict[str, Any]:
    plan = _read_object(output_root / "plan.json")
    if content_hash({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get(
        "plan_sha256"
    ):
        raise ValueError("multiround plan digest differs")
    if plan.get("runtime_source_sha256") != runtime_source_hash():
        raise ValueError("runtime source differs from frozen multiround plan")
    if plan.get("launcher_sha256") != launcher_hash():
        raise ValueError("launcher differs from frozen multiround plan")
    for relative, expected in plan["public_asset_ledger"].items():
        if _sha256(output_root / relative) != expected:
            raise ValueError(f"frozen public artifact differs: {relative}")
    if _sha256(output_root / "frozen/source_summary.json") != plan.get(
        "source_rescue_summary_file_sha256"
    ):
        raise ValueError("frozen source rescue summary differs")
    for task in plan["tasks"]:
        for path_key, digest_key in (
            ("candidate_path", "candidate_file_sha256"),
            ("source_rescue_path", "source_rescue_file_sha256"),
        ):
            if _sha256(output_root / task[path_key]) != task[digest_key]:
                raise ValueError(f"frozen task artifact differs: {task[path_key]}")
    return plan


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"required artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise ValueError(f"frozen artifact differs: {destination}")
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"checkpoint differs: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    temporary.replace(path)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


__all__ = [
    "ComponentRevision",
    "ComponentRevisionReply",
    "MultiRoundFeedbackConfig",
    "apply_component_revision",
    "freeze_campaign",
    "run_campaign",
    "run_task",
    "select_revision_components",
    "summarize",
]
