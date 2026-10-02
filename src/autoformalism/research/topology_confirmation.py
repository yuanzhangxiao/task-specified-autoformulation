"""Continue saved variable decisions through optional processes and topology only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from autoformalism.expressions import ValidationContext
from autoformalism.fitting import public_fitting as public
from autoformalism.llm.construction import ConstructionClient
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.research import construction_baseline as baseline
from autoformalism.research import construction_contract as contract
from autoformalism.research import construction_trace, phase_c_inputs
from autoformalism.research import variable_confirmation as variables
from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import variable_bindings, variable_equation_usage
from autoformalism.search.shared_construction import _stage
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import TrainingEvidence
from autoformalism.search.variable_inventory_review import replace_inventory

PROTOCOL = "phase-c-topology-confirmation-1"
REPO = baseline.REPO


class Config(baseline.Config):
    """Retain proposer settings; inherited fitter fields are unused in this stage."""

    protocol: Literal["phase-c-topology-confirmation-1"] = PROTOCOL


def source_identity() -> dict:
    """Seal the runtime and launcher code used by this continuation."""
    return {
        **baseline.source_identity(),
        "topology_launchers": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "scripts/phase_c_topology.py",
                "scripts/hpc/start_phase_c_topology.sh",
            )
        },
    }


def _config(plan: dict) -> Config:
    return Config.model_validate(
        {
            **{k: plan["config"][k] for k in Config.model_fields if k != "protocol"},
            "protocol": PROTOCOL,
        }
    )


def _roster(tasks: list[dict], cases) -> None:
    expected = {
        (case, seed, arm)
        for case in cases
        for seed in (0, 1)
        for arm in ("full", "brief_only")
    }
    actual = [(t["benchmark_id"], t["seed"], t["arm"]) for t in tasks]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("requires the complete case/seed/prompt roster")
    ids = [t["task_id"] for t in tasks]
    if len(ids) != len(set(ids)) or any(
        Path(n).name != n or n in {".", ".."} for n in ids
    ):
        raise ValueError("task IDs must be unique simple names")
    if any(
        not t.get("shared_processes") or not t.get("scientific_verifier") for t in tasks
    ):
        raise ValueError("this confirmation retains shared processes and verifier")


def _inventory(cell: dict, saved: dict):
    """Validate the imported declarations and bindings without changing them."""
    reply = variable_bindings.BoundVariableReply.model_validate(
        {
            "variables": saved["inventory"],
            "mechanism_bindings": [
                {"requirement_id": k, "memory_states": v}
                for k, v in saved["memory_candidates"].items()
            ],
        }
    )
    candidate, bindings = replace_inventory(
        PublicScientificBrief.model_validate(cell["brief"]),
        reply,
        contract.target_definitions(cell),
    )
    if {v.name: v.model_dump(mode="json") for v in candidate} != {
        v["name"]: v for v in saved["inventory"]
    } or {k: sorted(v) for k, v in bindings.items()} != saved["memory_candidates"]:
        raise ValueError("import would change the saved variable decisions")
    return tuple(ScientificVariable.model_validate(v) for v in saved["inventory"])


def freeze(base_source: Path, usage_source: Path, root: Path) -> dict:
    """Import twenty earlier inventories plus twelve targeted replacements."""
    sources = {"variables_v2": base_source, "usage_diagnostic": usage_source}
    plans = {name: sealed_read(path / "plan.json") for name, path in sources.items()}
    old, new = plans.values()
    for plan in plans.values():
        if (
            plan["protocol"] != variables.PROTOCOL
            or plan.get("test_data_opened") is not False
        ):
            raise ValueError("requires sealed development-only variable plans")
    _roster(old["tasks"], phase_c_inputs.ROSTER)
    _roster(new["tasks"], variables.TARGETED_CASES)
    if (
        old["source_plan_sha256"] != new["source_plan_sha256"]
        or _config(old) != _config(new)
        or new.get("equation_usage_policy") != variable_equation_usage.POLICY
    ):
        raise ValueError("source public campaign, settings or usage policy differ")
    old_tasks = {t["task_id"]: t for t in old["tasks"]}
    if any(old_tasks.get(t["task_id"]) != t for t in new["tasks"]):
        raise ValueError("replacement task identity differs")
    if any(old["cells"].get(k) != v for k, v in new["cells"].items()):
        raise ValueError("replacement public cells differ")
    cells = {
        name: {
            k: old["cells"][name][k]
            for k in ("brief", "context", "target_contract", "evidence")
        }
        for name in phase_c_inputs.ROSTER
    }
    replacements = {t["task_id"] for t in new["tasks"]}
    inputs = {}
    for task in old["tasks"]:
        source = (
            "usage_diagnostic" if task["task_id"] in replacements else "variables_v2"
        )
        plan = plans[source]
        directory = baseline.location(sources[source], task)
        saved = baseline.read_outcome(directory / "proposal.json", plan, task)
        if not saved or saved["status"] != "variables_complete":
            raise ValueError(f"no completed source inventory: {task['task_id']}")
        records = _cache_records(directory / "calls", baseline.namespace(plan, task))
        if saved["cost"] != _cost(records):
            raise ValueError("source variable-call accounting differs")
        _inventory(cells[task["benchmark_id"]], saved)
        inputs[task["task_id"]] = {
            "inventory": saved["inventory"],
            "memory_candidates": saved["memory_candidates"],
            "historical_variable_cost": saved["cost"],
            "origin": {
                "source": source,
                "plan_sha256": plan["artifact_sha256"],
                "proposal_sha256": saved["artifact_sha256"],
                "request_hashes": sorted(r["request_hash"] for r in records),
            },
        }
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": _config(old).model_dump(mode="json"),
                "tasks": old["tasks"],
                "cells": cells,
                "inputs": inputs,
                "source_identity": source_identity(),
                "source_plans": {
                    name: {
                        "artifact_sha256": p["artifact_sha256"],
                        "source_identity": p["source_identity"],
                    }
                    for name, p in plans.items()
                },
                "topology_context_policy": variable_equation_usage.TOPOLOGY_POLICY,
                "scope": (
                    "saved inventories; optional processes and topology; "
                    "no functions or fitting"
                ),
                "budget_scope": (
                    "new topology episode; historical variable cost reported separately"
                ),
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def verify(root: Path) -> dict:
    """Resume a frozen continuation without requiring its old checkouts."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("topology confirmation source/protocol differs")
    Config.model_validate(plan["config"])
    _roster(plan["tasks"], phase_c_inputs.ROSTER)
    if (
        plan["test_data_opened"]
        or plan["automatic_followup"]
        or plan["topology_context_policy"] != variable_equation_usage.TOPOLOGY_POLICY
        or set(plan["cells"]) != set(phase_c_inputs.ROSTER)
        or set(plan["inputs"]) != {t["task_id"] for t in plan["tasks"]}
    ):
        raise ValueError("topology scope differs")
    for task in plan["tasks"]:
        _inventory(plan["cells"][task["benchmark_id"]], plan["inputs"][task["task_id"]])
    return plan


def propose(root: Path, plan: dict, task: dict, base_url: str, **kwargs) -> dict:
    """Reuse process/topology stages and same-inventory fallback, then stop."""
    directory = baseline.location(root, task)
    identity = baseline.namespace(plan, task)
    with public._lock(directory / "construction-lock"):
        _cache_records(directory / "calls", identity)
        if directory.exists():
            construction_trace.render(directory, identity)
        previous = baseline.read_outcome(directory / "proposal.json", plan, task)
        if previous:
            return previous
        client = ConstructionClient(
            settings=Config.model_validate(plan["config"]).model_settings,
            directory=directory / "calls",
            namespace=identity,
            seed=task["seed"],
            base_url=base_url,
            **kwargs,
        )
        cell = plan["cells"][task["benchmark_id"]]
        saved = plan["inputs"][task["task_id"]]
        inventory = _inventory(cell, saved)
        attempts = []
        try:
            for route in ("process", "ordinary_fallback"):
                output = directory / "construction" / route
                topology = _stage(
                    output / "topology_stage.json",
                    lambda output=output, route=route: run_staged_topology(
                        contract.visible_brief(cell, task),
                        ValidationContext.model_validate(cell["context"]),
                        client,
                        output / "topology",
                        initial_inventory=inventory,
                        initial_memory_candidates=saved["memory_candidates"],
                        hybrid_variable_construction=True,
                        explicit_mechanism_bindings=True,
                        target_definitions=contract.target_definitions(cell),
                        training_evidence=TrainingEvidence.model_validate(
                            cell["evidence"]
                        )
                        if task["arm"] == "full"
                        else None,
                        audit_public_polarity_policy=True,
                        proposer_owns_unfixed_signs=True,
                        shared_process_guidance=True,
                        optional_process_review=route == "process",
                        bind_shared_processes=route == "process",
                        signed_shared_processes=route == "process",
                        complete_process_context=True,
                        topology_inventory_context=True,
                    ),
                )
                # The stage has tuple-valued checks in memory; expose the same
                # JSON representation on the first return and cached resume.
                topology = json.loads(json.dumps(topology))
                complete = (
                    topology["complete_topology"]
                    and topology["public_structure_checks_passed"]
                )
                attempts.append(
                    {
                        "route": route,
                        "status": topology["status"],
                        "error": topology.get("error"),
                        "complete": complete,
                    }
                )
                if complete or topology["status"] == "inventory_revision_requested":
                    break
                if not (topology.get("process_review") or {}).get("bindings"):
                    break  # Empty/invalid optional suggestions add no retry allowance.
            return sealed_write(
                directory / "proposal.json",
                {
                    "identity": identity,
                    "status": "topology_complete"
                    if complete
                    else (
                        "inventory_revision_requested"
                        if topology["status"] == "inventory_revision_requested"
                        else "topology_failed"
                    ),
                    "selected_route": route,
                    "fallback_used": route == "ordinary_fallback",
                    "attempts": attempts,
                    "topology_result": topology,
                    "cost": _cost(client.records),
                    "origin": saved["origin"],
                    "function_generation_performed": False,
                    "parameter_fitting_performed": False,
                },
            )
        finally:
            construction_trace.render(directory, identity)


def report(root: Path, plan: dict) -> dict:
    """Render exact skeletons, structural witnesses and costs without new calls."""
    from autoformalism.research.topology_inspection import report as inspect

    return inspect(root, plan)
