"""Public-only optional-process pilot; frozen data, call budgets and numerical fit."""

from __future__ import annotations

import hashlib
import signal
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.expressions import ValidationContext
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.llm.staged_topology import DeferredCall, StagedModelSettings
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import BudgetedRepairClient
from autoformalism.rebuttal.review_deadline_pipeline import _bundle, request_for
from autoformalism.rebuttal.shared_process_audit import inventory as model_inventory
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import (
    EvidenceSettings,
    TrainingEvidence,
    build_training_evidence,
)
from autoformalism.staged_topology import content_hash
from autoformalism.targets import _dependency_graph, _reaches

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "detention-process-pilot-1"


class PilotConfig(StrictSchema):
    """Sixteen fresh models, one construction/fit each; no revision or test access."""

    protocol: Literal["detention-process-pilot-1"] = PROTOCOL
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    model_settings: StagedModelSettings
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    served_context_tokens: Literal[32768] = 32768
    seeds: tuple[Literal[0], Literal[1]] = (0, 1)
    noise_index: Literal[1] = 1
    limits: ModelingLimits
    evidence: EvidenceSettings = EvidenceSettings()
    fit_profile: Literal["collocation-single-target-v2"] = (
        "collocation-single-target-v2"
    )
    scientific_judge: Literal["off"] = "off"
    wall_seconds: Literal[21600] = 21600
    shutdown_margin_seconds: Literal[300] = 300


def tasks(config):
    """Counterbalance arm order while matching cases, seeds and initial information."""
    result = []
    for case_index, case in enumerate(("coupled", "independent")):
        for seed in config.seeds:
            for arm in ("full", "brief_only"):
                for enabled in (
                    (False, True)
                    if (case_index + seed + (arm == "full")) % 2
                    else (True, False)
                ):
                    result.append(
                        {
                            "index": len(result),
                            "case": case,
                            "seed": seed,
                            "arm": arm,
                            "review": enabled,
                            "task_id": (
                                f"{case}_seed{seed}_{arm}_review_"
                                + ("on" if enabled else "off")
                            ),
                        }
                    )
    return result


def launcher_hash():
    """Bind entry points in addition to the complete Python package source hash."""
    paths = [
        "scripts/detention_process_pilot.py",
        "scripts/submit_detention_process_pilot.py",
        "scripts/hpc/run_detention_process_pilot_aces.sh",
        "scripts/hpc/submit_detention_process_pilot_aces.sh",
        "scripts/hpc/run_staged_topology_server.sh",
    ]
    return content_hash(
        {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths}
    )


def _public_asset(path):
    """Verify the release exporter's compact-JSON seal, not staged-call hashes."""
    value = public._read(path)
    if public.content_sha256(value["value"]) != value["sha256"]:
        raise ValueError(f"source public asset seal differs: {path}")
    return value["value"]


def freeze(source: Path, root: Path, config_path: Path):
    """Read only six public assets and the qualification manifest, never diagnostic/."""
    config = PilotConfig.model_validate_json(config_path.read_text())
    qualification = _public_asset(source / "plan.json")
    if (
        qualification["protocol"] != "detention-development-1"
        or not qualification["generation_gate_passed"]
    ):
        raise ValueError("requires qualified detention development release")
    binding = content_hash(qualification)
    with public._lock(root):
        if (root / "plan.json").exists():
            plan = verify(root)
            if plan["source_plan_sha256"] != binding or plan[
                "config"
            ] != config.model_dump(mode="json"):
                raise ValueError("source or configuration differs")
            for relative, digest in plan["source_public_files"].items():
                if (
                    hashlib.sha256((source / relative).read_bytes()).hexdigest()
                    != digest
                ):
                    raise ValueError("source public asset changed")
            return plan
        cells, files = {}, {}
        for case in ("coupled", "independent"):
            assets = {}
            for name in ("specification", "train", "val"):
                relative = f"public/{case}/noise1/{name}.json"
                path = source / relative
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if qualification["files"].get(relative) != digest:
                    raise ValueError("source public asset changed")
                files[relative] = digest
                assets[name] = _public_asset(path)
            spec = assets["specification"]
            context = ValidationContext(
                targets=tuple(spec["targets"]),
                external_inputs=tuple(spec["external_inputs"]),
                fixed_covariates=tuple(spec["fixed_covariates"]),
            )
            variables = [
                {"name": name, "data_role": role}
                for role, names in (
                    ("target", context.targets),
                    ("external_input", context.external_inputs),
                    ("covariate", context.fixed_covariates),
                )
                for name in names
            ]
            requirements = [
                {
                    "id": "local_balance",
                    "public_requirement": (
                        "Downstream accumulation and threshold-dependent discharge "
                        "respond to local runoff."
                    ),
                    "targets": ["h_down"],
                    "drivers": ["inflow_down"],
                }
            ]
            if case == "coupled":
                requirements.append(
                    {
                        "id": "upstream_memory",
                        "public_requirement": (
                            "Represent upstream storage memory and its "
                            "water-conserving "
                            "transfer into the downstream basin, using surveyed areas."
                        ),
                        "targets": ["h_down"],
                        "drivers": ["inflow_up"],
                        "requires_dynamic_memory": True,
                    }
                )
            brief = PublicScientificBrief(
                scientific_context=spec["public_prompt"],
                public_variables=variables,
                requirements=requirements,
                limits=config.limits,
            )
            training = PublicSplit.model_validate(assets["train"])
            validation = PublicSplit.model_validate(assets["val"])
            if training.name != "train" or validation.name != "val":
                raise ValueError("development splits differ")
            evidence = build_training_evidence(
                public.unpack_split(training), context, config.evidence
            )
            cells[case] = {
                "brief": brief.model_dump(mode="json"),
                "context": context.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json"),
                "training": training.model_dump(mode="json"),
                "validation": validation.model_dump(mode="json"),
            }
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": config.model_dump(mode="json"),
                "source_plan_sha256": binding,
                "source_public_files": files,
                "cells": cells,
                "tasks": tasks(config),
                "source_sha256": public._source_identity(),
                "launcher_sha256": launcher_hash(),
                "runtime": public._runtime(),
                "test_data_opened": False,
                "private_reference_opened": False,
            },
        )


def verify(root):
    """Reject source, launch or matrix drift without reading the old release."""
    plan = sealed_read(root / "plan.json")
    config = PilotConfig.model_validate(plan["config"])
    if plan["protocol"] != PROTOCOL or plan["tasks"] != tasks(config):
        raise ValueError("pilot contract differs")
    if (
        plan["source_sha256"] != public._source_identity()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("source or launcher differs")
    return plan


def make_client(root, plan, task, base_url, can_start=lambda: True, transport=None):
    """One durable request ledger includes optional review and fallback costs."""
    return BudgetedRepairClient(
        settings=PilotConfig.model_validate(plan["config"]).model_settings,
        directory=root / "results" / task["task_id"] / "calls",
        namespace=content_hash([plan["artifact_sha256"], task]),
        seed=task["seed"],
        base_url=base_url,
        can_start=can_start,
        **({"transport": transport} if transport else {}),
    )


def _stage(path, build):
    """Seal stages so resumed call counts cannot change downstream identities."""
    return (
        sealed_read(path)["result"]
        if path.exists()
        else sealed_write(path, {"result": build()})["result"]
    )


def structure(candidate, case):
    """Conservative dependency witness, including public initial-condition maps."""
    graph = _dependency_graph(candidate)
    for initial in candidate.initial_conditions:
        if initial.expression:
            parsed = RestrictedParser().parse(initial.expression, location="initial")
            for symbol in parsed.symbols:
                graph.setdefault(symbol, set()).add(initial.state)
    forbidden = (
        [
            name
            for name in ("inflow_up", "initial_up")
            if _reaches(graph, name, "target:h_down")
        ]
        if case == "independent"
        else []
    )
    return {
        "forbidden_upstream_dependencies": forbidden,
        "eligible": not forbidden,
        "check": "syntactic_dependency_including_initials",
        "conservation_certified": False,
        "broader_science": "not_certified",
    }


def construct(root, plan, task, client):
    """Fallback to the unchanged inventory after failed optional additions.

    The same total request/token caps include both attempts. No additional fit is
    granted. Allocation interruption is resumable and is not an invalid review.
    """
    directory = root / "results" / task["task_id"]
    path = directory / "proposal.json"
    if path.exists():
        return sealed_read(path)
    cell = plan["cells"][task["case"]]
    brief = PublicScientificBrief.model_validate(cell["brief"])
    context = ValidationContext.model_validate(cell["context"])
    evidence = (
        TrainingEvidence.model_validate(cell["evidence"])
        if task["arm"] == "full"
        else None
    )
    attempts, bundle = [], None
    for route in ("review", "fallback") if task["review"] else ("control",):
        output = directory / route
        topology, functions, error, structural = None, None, None, None
        try:
            topology = _stage(
                output / "topology_stage.json",
                lambda output=output, route=route: run_staged_topology(
                    brief,
                    context,
                    client,
                    output / "topology",
                    hybrid_variable_construction=True,
                    audit_public_polarity_policy=True,
                    proposer_owns_unfixed_signs=True,
                    training_evidence=evidence,
                    shared_process_guidance=True,
                    optional_process_review=(route == "review"),
                ),
            )
            if (
                topology["complete_topology"]
                and topology["public_structure_checks_passed"]
            ):
                functions = _stage(
                    output / "function_stage.json",
                    lambda output=output, topology=topology: run_staged_functions(
                        brief,
                        context,
                        topology,
                        client,
                        output / "functions",
                        generation_granularity="equation_batch_atomic_repair",
                        function_repair_policy="certified_outer_gain",
                        initialization_policy="causal_training",
                        training_evidence=evidence,
                        shared_process_guidance=True,
                    ),
                )
            if functions and functions["complete_model"]:
                bundle = _bundle(cell, task, brief, topology, functions)
                structural = structure(
                    CandidateModel.model_validate(bundle["candidate"]), task["case"]
                )
                if not structural["eligible"]:
                    bundle = None
                    error = "disconnected basin has an upstream dependency"
            else:
                error = (
                    (functions or {}).get("error")
                    or topology.get("error")
                    or "construction incomplete"
                )
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)[:6000]
        attempts.append(
            {
                "route": route,
                "error": error,
                "topology_status": (topology or {}).get("status"),
                "function_status": (functions or {}).get("status"),
                "process_review": (topology or {}).get("process_review"),
                "structural": structural,
            }
        )
        if bundle is not None:
            break
        # No alteration means no new failure attributable to optional additions.
        if route != "review" or not ((topology or {}).get("process_review") or {}).get(
            "added_names"
        ):
            break
    records = client.records
    usage = {
        "physical_calls": len(records),
        "observed_total_tokens": sum(
            r.get("observed_total_tokens") or 0 for r in records
        ),
        "unmeasured_calls": sum(
            r.get("observed_total_tokens") is None for r in records
        ),
        "budget_charge": sum(r.get("budget_charge", 0) for r in records),
        "provider_seconds": sum(r.get("latency_seconds", 0) for r in records),
    }
    for key in ("prompt_tokens", "completion_tokens"):
        values = [
            (r.get("raw_response", {}).get("usage") or {}).get(key) for r in records
        ]
        usage[key] = sum(v for v in values if isinstance(v, int) and v >= 0)
        usage[key + "_unmeasured_calls"] = sum(
            not isinstance(v, int) or v < 0 for v in values
        )
    candidate = CandidateModel.model_validate(bundle["candidate"]) if bundle else None
    return sealed_write(
        path,
        {
            "task": task,
            "status": "constructed"
            if bundle
            else "requirement_failed"
            if structural and not structural["eligible"]
            else "construction_failed",
            "bundle": bundle,
            "attempts": attempts,
            "fallback_used": len(attempts) > 1,
            "usage": usage,
            "structural": structural,
            "equation_inventory": model_inventory(
                candidate,
                ValidationContext.model_validate(bundle["initialization"]["context"]),
            )
            if bundle
            else None,
            "test_data_opened": False,
            "private_reference_opened": False,
        },
    )


def run_proposals(root, base_url, wall_seconds):
    """One allocation serves all tasks; every physical request is resumable."""
    plan = verify(root)
    draining = False

    def stop(*args):
        nonlocal draining
        draining = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    deadline = monotonic() + min(wall_seconds, plan["config"]["wall_seconds"]) - 300
    for task in plan["tasks"]:
        if draining or monotonic() >= deadline:
            return {"status": "deferred"}
        with public._lock(root / "results" / task["task_id"]):
            try:
                construct(
                    root,
                    plan,
                    task,
                    make_client(
                        root,
                        plan,
                        task,
                        base_url,
                        lambda: not draining and monotonic() < deadline,
                    ),
                )
            except DeferredCall:
                return {"status": "deferred"}
    return {"status": "complete"}


def replay(request, parameters, cell):
    """Bounded numerical consistency; only noisy public outputs are read."""
    model, _, _ = public._lower(request)
    scale = np.std(
        [v for r in cell["training"]["rows"] for v in r["targets"]["h_down"]]
    )
    errors, differences = [], []
    deadline = monotonic() + 240
    for key in ("training", "validation"):
        for row in public.unpack_split(
            PublicSplit.model_validate(cell[key])
        ).trajectories:
            values = []
            for method in ("BDF", "Radau"):
                result = simulate_trajectory(
                    model,
                    row,
                    parameters,
                    {},
                    FitConfig(
                        integration_backend="solve_ivp",
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    deadline=deadline,
                    reset_observed_states=False,
                )
                if not result.success:
                    errors.append(
                        f"{key}/{row.trajectory_id}/{method}: {result.message}"
                    )
                    break
                values.append(result.predictions["h_down"])
            if len(values) == 2:
                differences.append(
                    float(np.max(np.abs(values[0] - values[1])) / max(scale, 1e-12))
                )
    return {
        "replay_agreement": bool(
            not errors and differences and max(differences) < 1e-5
        ),
        "maximum_scaled_difference": max(differences, default=None),
        "errors": errors,
        "accuracy_certified": False,
    }


def fit_task(root, index):
    """Reuse frozen fitter markers; no fresh attempt on interruption or rerun."""
    plan = verify(root)
    if index not in range(len(plan["tasks"])):
        raise ValueError("task index outside plan")
    task = plan["tasks"][index]
    directory = root / "results" / task["task_id"]
    with public._lock(directory):
        if (directory / "result.json").exists():
            return sealed_read(directory / "result.json")
        if not (directory / "proposal.json").exists():
            return {"status": "proposal_missing"}
        proposal = sealed_read(directory / "proposal.json")
        result = {
            "task": task,
            "proposal_sha256": proposal["artifact_sha256"],
            "status": proposal["status"],
            "fit": None,
            "replay": None,
        }
        if proposal["status"] == "constructed":
            request = request_for(proposal["bundle"], plan, task, 0)
            cell = plan["cells"][task["case"]]
            public.prepare_fit(
                request,
                PublicSplit.model_validate(cell["training"]),
                PublicSplit.model_validate(cell["validation"]),
                directory / "fit",
            )
            fit = public.execute_fit(directory / "fit")
            result.update(status=fit.status, fit=fit.model_dump(mode="json"))
            if fit.status == "complete":
                marker = directory / "replay_started.json"
                if marker.exists():
                    result["replay"] = {
                        "status": "interrupted",
                        "replay_agreement": False,
                    }
                else:
                    sealed_write(
                        marker,
                        {"fit_sha256": content_hash(fit.model_dump(mode="json"))},
                    )
                    try:
                        result["replay"] = replay(request, dict(fit.parameters), cell)
                    except (ValueError, RuntimeError, TimeoutError) as exc:
                        result["replay"] = {
                            "status": "failed",
                            "replay_agreement": False,
                            "error": str(exc),
                        }
        return sealed_write(directory / "result.json", result)


def report(root):
    """Every planned model remains in denominators; no success from process counts."""
    plan = verify(root)
    rows = []
    equations = [
        "# Discovered equations — optional process review",
        "",
        "Proposer equations and shared initial maps, before fitting. "
        "Inspect area conversions and conservation; "
        "reuse counts alone certify neither.",
    ]
    for task in plan["tasks"]:
        directory = root / "results" / task["task_id"]
        proposal = (
            sealed_read(directory / "proposal.json")
            if (directory / "proposal.json").exists()
            else {}
        )
        result = (
            sealed_read(directory / "result.json")
            if (directory / "result.json").exists()
            else {}
        )
        if result and result["proposal_sha256"] != proposal["artifact_sha256"]:
            raise ValueError("result proposal identity differs")
        fit = result.get("fit") or {}
        inv = proposal.get("equation_inventory") or {}
        bundle = proposal.get("bundle")
        equations.extend(["", "## " + task["task_id"], ""])
        if bundle:
            candidate = bundle["candidate"]
            equations.extend(
                f"- `{p['name']} = {p['expression']}`" for p in candidate["processes"]
            )
            equations.extend(
                f"- `{p['state']}' = {p['rhs']}`" for p in candidate["state_equations"]
            )
            equations.extend(
                f"- `observe {p['channel']} = {p['expression']}`"
                for p in candidate["observation_mappings"]
            )
            equations.extend(
                f"- Initial `{p['state']}`: "
                f"`{p.get('expression') or p.get('fixed_value')}`"
                for p in candidate["initial_conditions"]
            )
        else:
            equations.append("No retained constructed candidate.")
        rows.append(
            {
                "task": task,
                "status": result.get("status", "missing"),
                "proposal_status": proposal.get("status", "missing"),
                "attempts": proposal.get("attempts"),
                "fallback_used": proposal.get("fallback_used"),
                "usage": proposal.get("usage"),
                "structure": proposal.get("structural"),
                "complexity": inv.get("counts"),
                "shared_governing_laws": sum(
                    sum(not n.startswith("output:") for n in p["consumers"]) >= 2
                    for p in inv.get("existing_shared_processes", [])
                )
                if inv
                else None,
                "training": fit.get("training"),
                "validation": fit.get("validation"),
                "replay": result.get("replay"),
            }
        )
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "rows": rows,
        "scientific_correctness_certified": False,
        "test_data_opened": False,
        "automatic_followup": False,
    }
    public._write(root / "summary.json", summary)
    lines = [
        "# Optional process review — basin pilot",
        "",
        "Same guidance, model, seeds, data and total budgets; optional review on/off.",
        (
            "No private equations, hidden trajectories or test data. "
            "Named reuse is not conservation certification."
        ),
        "",
        f"Status counts: {summary['status_counts']}",
        "",
        (
            "| Task | Status | Review | Fallback | Val NMSE | "
            "Shared governing laws | Tokens |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        audit = ((r["attempts"] or [{}])[0].get("process_review") or {}).get(
            "status", "off"
        )
        fields = [
            r["task"]["task_id"],
            r["status"],
            audit,
            r["fallback_used"],
            (r["validation"] or {}).get("normalized_mse"),
            r["shared_governing_laws"],
            (r["usage"] or {}).get("observed_total_tokens"),
        ]
        lines.append("| " + " | ".join(str(v) for v in fields) + " |")
    temporary = root / "SUMMARY.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(root / "SUMMARY.md")
    temporary = root / "EQUATIONS.md.tmp"
    temporary.write_text("\n".join(equations) + "\n")
    temporary.replace(root / "EQUATIONS.md")
    return summary
