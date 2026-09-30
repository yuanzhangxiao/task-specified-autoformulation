"""Small public-only baseline for inspecting stages, equations and fitted behavior."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.llm.staged_topology import StagedModelSettings, atomic_json
from autoformalism.rebuttal import mechanism_functional as functional
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_construction_campaign import _cache_records, _cost
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_comparison import (
    BudgetedRepairClient,
    RepairBudgetExceeded,
)
from autoformalism.rebuttal.staged_topology_campaign import runtime_source_hash
from autoformalism.research import basin_construction_assessment as basin
from autoformalism.research import construction_trace, phase_c_inputs
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit
from autoformalism.schemas.staged_topology import ModelingLimits
from autoformalism.search.shared_construction import construct
from autoformalism.search.training_evidence import EvidenceSettings
from autoformalism.staged_topology import content_hash

PROTOCOL = "phase-c-construction-baseline-1"
REPO = Path(__file__).resolve().parents[3]


class Config(StrictSchema):
    """One construction per case, seed and prompt; no scientific revisions."""

    protocol: Literal["phase-c-construction-baseline-1"] = PROTOCOL
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    limits: ModelingLimits
    evidence: EvidenceSettings
    served_context_tokens: int = Field(default=32768, ge=8192)
    wall_seconds: int = Field(default=21600, ge=600, le=21600)
    fit_profile: Literal["collocation-multi-target-v1"] = "collocation-multi-target-v1"
    assessment: functional.Settings = functional.Settings()


def source_identity() -> dict:
    """Pin shared implementations and the small operational entry points."""
    names = (
        "scripts/phase_c_baseline.py",
        "scripts/submit_phase_c_baseline.py",
        "scripts/hpc/run_phase_c_baseline.sh",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/hpc/start_foundation_aces.sh",
    )
    return {
        "runtime": runtime_source_hash(),
        "launchers": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in names
        },
    }


def freeze(release: Path, root: Path, config_path: Path) -> dict:
    """Copy only the qualified public train/validation release into a sealed plan."""
    config = Config.model_validate_json(config_path.read_text()).model_dump(mode="json")
    receipt, cells = phase_c_inputs.qualified_public_cells(release, config)
    tasks = [
        {
            "task_id": f"cell{i:02d}_seed{seed}_{arm}",
            "benchmark_id": name,
            "seed": seed,
            "arm": arm,
            "shared_processes": True,
            "scientific_verifier": True,
        }
        for i, name in enumerate(cells)
        for seed in (0, 1)
        for arm in ("full", "brief_only")
    ]
    with public._lock(root):
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "config": config,
                "cells": cells,
                "tasks": tasks,
                "source_identity": source_identity(),
                "release_receipt_sha256": receipt,
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def verify(root: Path) -> dict:
    """Resume only the same source, settings and sealed public task matrix."""
    plan = sealed_read(root / "plan.json")
    if plan["protocol"] != PROTOCOL or plan["source_identity"] != source_identity():
        raise ValueError("baseline source/protocol differs; use the original checkout")
    Config.model_validate(plan["config"])
    return plan


def namespace(plan: dict, task: dict) -> str:
    return content_hash([plan["artifact_sha256"], task])


def location(root: Path, task: dict) -> Path:
    return root / "results" / task["task_id"]


def read_outcome(path: Path, plan: dict, task: dict) -> dict | None:
    if not path.exists():
        return None
    result = sealed_read(path)
    if result["identity"] != namespace(plan, task):
        raise ValueError("outcome belongs to a different task")
    return result


def propose(root: Path, plan: dict, task: dict, base_url: str, **kwargs) -> dict:
    """Use existing stages and retain exact request/response evidence."""
    directory = location(root, task)
    identity = namespace(plan, task)
    with public._lock(directory / "construction-lock"):
        previous = read_outcome(directory / "proposal.json", plan, task)
        if previous:
            construction_trace.render(directory, identity)
            return previous
        _cache_records(directory / "calls", identity)
        client = BudgetedRepairClient(
            settings=StagedModelSettings.model_validate(
                plan["config"]["model_settings"]
            ),
            directory=directory / "calls",
            namespace=identity,
            seed=task["seed"],
            base_url=base_url,
            **kwargs,
        )
        # Assessment rules and validation data are deliberately absent from this view.
        cell = plan["cells"][task["benchmark_id"]]
        view = {
            k: cell[k]
            for k in (
                "brief",
                "context",
                "target_contract",
                "mechanism_spec",
                "training",
                "evidence",
            )
        }
        try:
            result = construct(
                view,
                task,
                client,
                directory / "construction",
                build_bundle=pipeline._bundle,
                certificate_for=pipeline.certificates,
                retain_failed_draft=True,
            )
            return sealed_write(
                directory / "proposal.json",
                {
                    "identity": identity,
                    **result,
                    "cost": _cost(client.records),
                },
            )
        except RepairBudgetExceeded as exc:
            return sealed_write(
                directory / "proposal.json",
                {
                    "identity": identity,
                    "status": "provider_budget_exhausted",
                    "error": str(exc),
                    "cost": _cost(client.records),
                },
            )
        finally:
            construction_trace.render(directory, identity)


def fit(root: Path, plan: dict, task: dict) -> dict:
    """Fit one mechanically admitted model, retaining the production fitter's resume."""
    directory = location(root, task)
    with public._lock(directory / "fit-lock"):
        previous = read_outcome(directory / "fitted.json", plan, task)
        if previous:
            return previous
        proposal = read_outcome(directory / "proposal.json", plan, task)
        if proposal is None:
            return {"status": "pending_construction"}
        value = {"identity": namespace(plan, task), "status": proposal["status"]}
        if proposal["status"] == "constructed":
            cell = plan["cells"][task["benchmark_id"]]
            request = pipeline.request_for(proposal["bundle"], plan, task, 0)
            public.prepare_fit(
                request,
                PublicSplit.model_validate(cell["training"]),
                PublicSplit.model_validate(cell["validation"]),
                directory / "fit",
            )
            result = public.execute_fit(directory / "fit")
            value.update(status=result.status, fit=result.model_dump(mode="json"))
        return sealed_write(directory / "fitted.json", value)


def assess(root: Path, plan: dict, task: dict) -> dict:
    """Measure finite public mechanism predicates independently of admission."""
    directory = location(root, task)
    with public._lock(directory / "assessment-lock"):
        previous = read_outcome(directory / "assessment.json", plan, task)
        if previous:
            return previous
        fitted = read_outcome(directory / "fitted.json", plan, task)
        if fitted is None:
            return {"status": "pending_fit"}
        cell = plan["cells"][task["benchmark_id"]]
        rules = cell["independent_rules"]
        row = None
        if fitted["status"] == "complete":
            proposal = read_outcome(directory / "proposal.json", plan, task)
            model, _, _ = public._lower(
                pipeline.request_for(proposal["bundle"], plan, task, 0)
            )
            row = {
                "candidate": model.validated.candidate.model_dump(mode="json"),
                "context": model.validated.context.model_dump(mode="json"),
                "parameters": fitted["fit"]["parameters"],
                "initials": {},
                "semantics": "continuous_time",
            }
            train = public.unpack_split(PublicSplit.model_validate(cell["training"]))
            settings = functional.Settings.model_validate(plan["config"]["assessment"])
            findings = (
                basin.assess(row, cell, train, settings, directory / "probes")
                if cell.get("assessment_policy") == basin.POLICY
                else functional.assess(
                    row, rules, train, settings, directory / "probes"
                )
            )
            status = "assessed"
        else:
            status = "model_unavailable"
            findings = [
                {
                    "id": r["id"],
                    "public_requirement": r["public_requirement"],
                    "status": "unresolved",
                    "reason": fitted["status"],
                }
                for r in rules
            ]
        counts = Counter(f["status"] for f in findings)
        return sealed_write(
            directory / "assessment.json",
            {
                "identity": namespace(plan, task),
                "status": status,
                "model": row,
                "findings": findings,
                "counts": dict(counts),
                "confirmed": counts["pass"] / len(rules),
                "possible": (counts["pass"] + counts["unresolved"]) / len(rules),
                "llm_calls": 0,
                "optimizer_calls": 0,
                "test_data_opened": False,
            },
        )


def report(root: Path, plan: dict) -> dict:
    """Expose every task and equation, including failures and incomplete work."""
    from autoformalism.research.construction_statistics import aggregate

    rows, equations = [], ["# Phase C construction equations", ""]
    for task in plan["tasks"]:
        directory = location(root, task)
        proposal = read_outcome(directory / "proposal.json", plan, task)
        fitted = read_outcome(directory / "fitted.json", plan, task)
        assessment = read_outcome(directory / "assessment.json", plan, task)
        cost = _cost(_cache_records(directory / "calls", namespace(plan, task)))
        metrics = (fitted or {}).get("fit", {})
        row = {
            **task,
            "assessment_policy": plan["cells"][task["benchmark_id"]].get(
                "assessment_policy", "fitted-public-mechanism-tests-1"
            ),
            "status": (fitted or {}).get("status", "pending"),
            "proposal_status": (proposal or {}).get("status", "pending"),
            "assessment_status": (assessment or {}).get("status", "pending"),
            "training_nmse": (metrics.get("training") or {}).get("normalized_mse"),
            "validation_nmse": (metrics.get("validation") or {}).get("normalized_mse"),
            "confirmed": (assessment or {}).get("confirmed"),
            "possible": (assessment or {}).get("possible"),
            "mechanisms": (assessment or {}).get("counts"),
            "cost": cost,
        }
        rows.append(row)
        equations.extend(
            [
                f"## {task['task_id']}",
                "",
                f"{task['benchmark_id']} — {row['status']}",
                "",
            ]
        )
        if proposal:
            bundle = proposal.get("bundle") or proposal.get("construction_draft")
            if bundle:
                equations.extend(
                    [
                        "```json",
                        json.dumps(
                            {
                                "candidate": bundle["candidate"],
                                "initialization": bundle["initialization"],
                                "fitted_parameters": metrics.get("parameters"),
                                "independent_findings": (assessment or {}).get(
                                    "findings"
                                ),
                            },
                            indent=2,
                        ),
                        "```",
                        "",
                    ]
                )
        if directory.exists():
            construction_trace.render(directory, namespace(plan, task))
    value = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "rows": rows,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "assessment_counts": dict(Counter(r["assessment_status"] for r in rows)),
        "aggregate": aggregate(rows),
        "aggregate_by_assessment_policy": {
            policy: aggregate([r for r in rows if r["assessment_policy"] == policy])
            for policy in sorted({r["assessment_policy"] for r in rows})
        },
        "test_data_opened": False,
    }
    atomic_json(root / "summary.json", value)
    (root / "EQUATIONS.md").write_text("\n".join(equations))
    lines = [
        "# Phase C construction baseline",
        "",
        "Validation NMSE and independent finite public mechanism tests.",
        "No test access.",
        "Scientific adequacy of stage replies requires equation/trace review.",
        "",
        "| Task | Status | Val NMSE | Confirmed | Possible | Calls | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['task_id']} | {r['status']} | {r['validation_nmse']} | "
            f"{r['confirmed']} | {r['possible']} | {r['cost']['physical_requests']} | "
            f"{r['cost']['observed_tokens']} |"
        )
    lines.extend(
        [
            "",
            "Aggregates: case medians then macro median/unscaled MAD for NMSE;",
            "case means then macro mean/sample SD for compliance. Pending tasks block",
            "aggregates; unavailable models get NMSE +infinity, unresolved predicates.",
            "Basin scores cover fixed finite predicates, not every public requirement.",
            "The independent basin is a negative control within the basin family.",
            "```json",
            json.dumps(
                {k: value[k] for k in ("aggregate", "aggregate_by_assessment_policy")},
                indent=2,
            ),
            "```",
        ]
    )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return value
