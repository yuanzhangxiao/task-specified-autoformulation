"""General process-pruning pilot with an equal-allocation unchanged-refit control."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.pruning import process_aware as pruning
from autoformalism.rebuttal import review_deadline_io as history
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.review_deadline_pipeline import certificates
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "general-process-pruning-1"
REPO = Path(__file__).resolve().parents[3]
POLICY = {
    "maximum_candidates_per_parent": 1,
    "contribution_seconds": 300,
    "validation_relative_tolerance": 0.01,
    "validation_absolute_floor": 1e-6,
    "ranking": "minimum maximum consumer term RMS / defining law RMS; training only",
    "selection": (
        "Pareto smaller and within tolerance of best parent/control validation"
    ),
    "unresolved_public_requirements": (
        "skip automatic deletion; permit unchanged control"
    ),
    "fit_profile": "collocation-single-target-v2",
}


def launcher_hash() -> str:
    """Seal local orchestration as well as package code."""
    return content_hash(
        {
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
            for p in (
                "scripts/process_pruning.py",
                "scripts/submit_process_pruning.py",
                "scripts/hpc/run_process_pruning_aces.sh",
                "scripts/smoke_process_pruning.py",
                "scripts/submit_shared_process_pilot.py",
                "scripts/submit_review_continuation.py",
            )
        }
    )


def certificate(request: PublicFitRequest, cell: dict, task: dict) -> dict:
    """Recheck actual rebuilt equations under the original public contract."""
    model, _, _ = public._lower(request)
    return certificates(
        {
            "candidate": model.validated.candidate.model_dump(mode="json"),
            "initialization": {
                "context": model.validated.context.model_dump(mode="json")
            },
        },
        cell,
        task,
    )


def certified(value: dict) -> bool:
    """Ambiguous obligations remain unresolved, never silently counted as passes."""
    return (
        value["eligible_for_development_selection"]
        and value["all_public_graph_requirements_certified"]
        and all(p["status"] == "satisfied" for p in value["targets"]["predicates"])
    )


def score(fit: dict | None) -> float | None:
    """Only complete, finite predictions are eligible for numerical selection."""
    if not fit or fit["status"] != "complete" or fit.get("parameters") is None:
        return None
    for split in ("training", "validation"):
        metric = fit[split]
        value = metric.get("normalized_mse")
        if (
            not metric["available"]
            or value is None
            or not math.isfinite(value)
            or value < 0
            or metric.get("failed_trajectories")
        ):
            return None
    return fit["validation"]["normalized_mse"]


def snapshot(source: Path) -> tuple[dict, list[dict]]:
    """Read only public development assets and the completed two-round lineage."""
    history.require_open(source)
    old = history.verify(source, execution=False)
    if (
        old["protocol"] != history.INTEGRATION_PROTOCOL
        or old["config"]["fit_profile"] != POLICY["fit_profile"]
    ):
        raise ValueError(
            "requires completed general integration pilot with frozen fitter"
        )
    rows = []
    for task in old["tasks"]:
        previous = None
        for index in (0, 1):
            result = history.read_round(source, task, index)
            proposal = sealed_read(
                history.round_path(source, task, index) / "proposal.json"
            )
            if (
                result is None
                or result["task"] != task
                or result["round"] != index
                or result["test_data_opened"]
            ):
                raise ValueError("missing or invalid completed development round")
            prior = previous["artifact_sha256"] if previous else None
            if (
                result["parent_sha256"] != prior
                or proposal["parent_sha256"] != prior
                or result["proposal_sha256"] != proposal["artifact_sha256"]
            ):
                raise ValueError("historical lineage differs")
            previous = result
        selected = result["selected"]
        if not selected or score(selected["fit"]) is None:
            raise ValueError("each parent requires complete finite development metrics")
        request = PublicFitRequest.model_validate(selected["request"])
        model, _, _ = public._lower(request)
        fit = PublicFitResult.model_validate(selected["fit"])
        cell = old["cells"][task["cell"]]
        origin = selected["origin_round"]
        if selected["origin_task"] != task["task_id"] or origin not in (0, 1):
            raise ValueError("selected fit origin differs")
        fit_directory = history.round_path(source, task, origin) / "fit"
        envelope = public._read(fit_directory / "result.json")
        if (
            public.content_sha256(envelope) != selected["fit_result_sha256"]
            or envelope["result"] != selected["fit"]
            or envelope["sha256"] != public.content_sha256(selected["fit"])
        ):
            raise ValueError("selected fit artifact differs")
        if fit.backend_result_sha256 is not None:
            backend = public._read(fit_directory / "backend_result.json")
            if public.content_sha256(backend) != fit.backend_result_sha256:
                raise ValueError("historical backend artifact differs")
        if (
            model.validated.candidate.model_dump(mode="json")
            != selected["bundle"]["candidate"]
            or fit.lowered_candidate_sha256
            != public.content_sha256(model.validated.candidate.model_dump(mode="json"))
            or fit.request_sha256
            != public.content_sha256(request.model_dump(mode="json"))
            or fit.training_content_sha256 != public.content_sha256(cell["training"])
            or fit.validation_content_sha256
            != public.content_sha256(cell["validation"])
            or request.profile != POLICY["fit_profile"]
        ):
            raise ValueError("selected handoff or development data identity differs")
        training = PublicSplit.model_validate(cell["training"])
        sibling_fit.compatible_seed(request, request, dict(fit.parameters), training)
        checks = certificate(request, cell, task)
        if not checks["eligible_for_development_selection"]:
            raise ValueError("parent fails a hard public requirement")
        rows.append(
            {
                "task": task,
                "parent": selected,
                "source_result_sha256": result["artifact_sha256"],
                "certificate": checks,
            }
        )
    return old, rows


def freeze(source: Path, root: Path) -> dict:
    """Copy public inputs and freeze policy before ranking or refitting."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("output must be separate from historical campaign")
    old, rows = snapshot(source)
    payload = {
        "protocol": PROTOCOL,
        "policy": POLICY,
        "source": str(source),
        "source_plan_sha256": old["artifact_sha256"],
        "cells": old["cells"],
        "rows": rows,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "launcher_sha256": launcher_hash(),
        "test_data_opened": False,
        "llm_calls": 0,
        "automatic_followup": False,
    }
    with public._lock(root):
        return sealed_write(root / "plan.json", payload)


def verify(root: Path) -> dict:
    """Resume only under identical code, public inputs and numerical profile."""
    history.require_open(root)
    plan = sealed_read(root / "plan.json")
    history.require_open(Path(plan["source"]))
    if (
        plan["protocol"] != PROTOCOL
        or plan["policy"] != POLICY
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("pruning execution identity differs")
    return plan


def choose(row: dict, cell: dict) -> dict:
    """Freeze one candidate from training evidence before any child validation score."""
    request = PublicFitRequest.model_validate(row["parent"]["request"])
    original = certificate(request, cell, row["task"])
    if not certified(original):
        return {
            "status": "public_requirements_unresolved",
            "certificate": original,
            "request": None,
        }
    unchecked = [
        c.model_dump(mode="json")
        for c in request.base_candidate.constraints
        if c.kind.value
        in {"conservation", "custom", "monotone_increasing", "monotone_decreasing"}
    ]
    if unchecked:
        return {
            "status": "declared_constraints_require_review",
            "request": None,
            "certificate": original,
            "unverified_constraints": unchecked,
        }
    units, blocked = pruning.removal_units(request)
    viable, seen = [], set()
    for unit in units:
        try:
            child, audit = pruning.remove(request, unit)
            checks = certificate(child, cell, row["task"])
            if not certified(checks):
                blocked.append(
                    {
                        "id": unit["id"],
                        "reason": "public_requirements_not_certified",
                        "certificate": checks,
                    }
                )
                continue
            digest = content_hash(
                [
                    child.base_candidate.model_dump(mode="json"),
                    child.initialization_plan.model_dump(mode="json"),
                ]
            )
            if digest in seen:
                continue
            seen.add(digest)
            viable.append(
                {
                    "unit": unit,
                    "request": child.model_dump(mode="json"),
                    "audit": audit,
                    "certificate": checks,
                }
            )
        except (ValueError, TypeError) as error:
            blocked.append({"id": unit["id"], "reason": str(error)})
    if not viable:
        return {"status": "no_legal_removal", "request": None, "blocked": blocked}
    contribution = pruning.contributions(
        request,
        row["parent"]["fit"]["parameters"],
        PublicSplit.model_validate(cell["training"]),
        seconds=POLICY["contribution_seconds"],
    )
    for item in viable:
        item["rank"] = max(contribution[k] for k in item["unit"]["slots"])
    selected = min(viable, key=lambda x: (x["rank"], x["unit"]["id"]))
    return {
        "status": "ready",
        **selected,
        "contributions": contribution,
        "blocked": blocked,
        "ranking": [{"id": v["unit"]["id"], "rank": v["rank"]} for v in viable],
    }


def decide(
    parent: dict, control: dict | None, child: dict | None, choice: dict
) -> dict:
    """Predeclared validation/complexity rule; unchanged parent is always retained."""
    options = [(score(parent), "parent", parent)]
    if score(control) is not None:
        options.append((score(control), "control", control))
    baseline, origin, fit = min(options, key=lambda x: (x[0], x[1] != "parent"))
    tolerance = max(
        POLICY["validation_absolute_floor"],
        POLICY["validation_relative_tolerance"] * abs(baseline),
    )
    accepted = (
        choice["status"] == "ready"
        and certified(choice["certificate"])
        and pruning.smaller(choice["audit"]["before"], choice["audit"]["after"])
        and score(child) is not None
        and score(child) <= baseline + tolerance
    )
    return {
        "selected": "pruned" if accepted else origin,
        "fit": child if accepted else fit,
        "pruned_accepted": accepted,
        "baseline_validation_nmse": baseline,
        "allowed_nmse_increase": tolerance,
        "paired_control_available": score(control) is not None,
    }


def checked_result(directory: Path, plan: dict, row: dict) -> dict:
    """Join a published decision to its sealed candidate and backend receipts."""
    result = sealed_read(directory / "result.json")
    choice = sealed_read(directory / "choice.json")
    if (
        result["identity"] != plan["artifact_sha256"]
        or result["task"] != row["task"]
        or result["choice"] != choice
    ):
        raise ValueError("result or candidate identity differs")
    for arm, fit in result["fits"].items():
        if fit is not None:
            observed = sibling_fit.inspect_child_fit(directory / arm)
            if observed["result"] != fit:
                raise ValueError("pruning fit receipt differs")
    expected = decide(
        row["parent"]["fit"],
        result["fits"]["control"],
        result["fits"]["pruned"],
        choice,
    )
    if expected != result["selection"]:
        raise ValueError("pruning selection differs")
    return result


def run_one(root: Path, index: int) -> dict:
    """One consumed ranking attempt and up to two checkpointed frozen-profile fits."""
    plan = verify(root)
    if index not in range(len(plan["rows"])):
        raise ValueError("task index outside frozen matrix")
    row = plan["rows"][index]
    directory = root / "results" / row["task"]["task_id"]
    return execute_row(directory, plan, row, index)


def execute_row(directory: Path, plan: dict, row: dict, index: int) -> dict:
    """Apply the same bounded pruning experiment to a verified parent snapshot."""
    cell = plan["cells"][row["task"]["cell"]]
    with public._lock(directory):
        result_path = directory / "result.json"
        if result_path.exists():
            return checked_result(directory, plan, row)
        choice_path = directory / "choice.json"
        if choice_path.exists():
            choice = sealed_read(choice_path)
        else:
            marker = directory / "ranking_started.json"
            if marker.exists():
                value = {"status": "ranking_interrupted", "request": None}
            else:
                sealed_write(
                    marker, {"identity": plan["artifact_sha256"], "index": index}
                )
                try:
                    value = choose(row, cell)
                except (ValueError, TimeoutError) as error:
                    value = {
                        "status": "ranking_unavailable",
                        "request": None,
                        "message": str(error),
                    }
            choice = sealed_write(
                choice_path, {"identity": plan["artifact_sha256"], **value}
            )
        if choice["identity"] != plan["artifact_sha256"]:
            raise ValueError("candidate identity differs")
        parent = PublicFitRequest.model_validate(row["parent"]["request"])
        fits = {}
        order = ["control", "pruned"] if index % 2 == 0 else ["pruned", "control"]
        for arm in order:
            if arm == "pruned" and choice["request"] is None:
                fits[arm] = None
                continue
            child = (
                parent
                if arm == "control"
                else PublicFitRequest.model_validate(choice["request"])
            )
            fit_directory = directory / arm
            sibling_fit.prepare_child_fit(
                parent,
                child,
                row["parent"]["fit"]["parameters"],
                PublicSplit.model_validate(cell["training"]),
                PublicSplit.model_validate(cell["validation"]),
                fit_directory,
                lineage={
                    "campaign": plan["artifact_sha256"],
                    "task": row["task"],
                    "arm": arm,
                    "choice": choice["artifact_sha256"],
                },
                allow_initialization_changes=True,
            )
            fits[arm] = sibling_fit.execute_child_fit(fit_directory).model_dump(
                mode="json"
            )
        return sealed_write(
            result_path,
            {
                "identity": plan["artifact_sha256"],
                "task": row["task"],
                "status": "complete",
                "choice": choice,
                "fits": fits,
                "execution_order": order,
                "selection": decide(
                    row["parent"]["fit"], fits["control"], fits["pruned"], choice
                ),
                "parent_complexity": pruning.complexity(parent),
                "test_data_opened": False,
                "llm_calls": 0,
                "independent_replay": "not_performed",
            },
        )


def report(root: Path) -> dict:
    """Report missing, skipped, numerical and requirement evidence independently."""
    plan = verify(root)
    rows = []
    for row in plan["rows"]:
        path = root / "results" / row["task"]["task_id"] / "result.json"
        result = checked_result(path.parent, plan, row) if path.exists() else None
        rows.append(
            {
                "task": row["task"]["task_id"],
                "status": result["status"] if result else "missing",
                "parent": row["parent"]["fit"],
                "parent_certificate": row["certificate"],
                "result": result,
            }
        )
    payload = {
        "protocol": PROTOCOL,
        "identity": plan["artifact_sha256"],
        "policy": POLICY,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "pruning_choice_counts": dict(
            Counter(r["result"]["choice"]["status"] for r in rows if r["result"])
        ),
        "fit_status_counts": dict(
            Counter(
                f["status"]
                for r in rows
                if r["result"]
                for f in r["result"]["fits"].values()
                if f is not None
            )
        ),
        "selection_counts": dict(
            Counter(r["result"]["selection"]["selected"] for r in rows if r["result"])
        ),
        "rows": rows,
        "llm_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
        "independent_replay": "not_performed",
    }
    public._write(root / "summary.json", payload)
    text = [
        "# General process-aware pruning",
        "",
        "One training-ranked deletion and one unchanged-refit control per parent. "
        "Frozen fitter; no LLM calls or test access.",
        "Public ambiguities skip deletion; graph checks do not certify scientific "
        "correctness. Contributions depend on fitted parameters.",
        "Historical parents have different total budgets. Control/pruned refits "
        "have equal caps; no independent solver replay or automatic promotion.",
        "",
        "| Task | Pruning choice | Parent val NMSE | Control val NMSE | "
        "Pruned val NMSE | Selected |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        result = row["result"]
        text.append(
            "| "
            + " | ".join(
                str(v)
                for v in (
                    row["task"],
                    result["choice"]["status"] if result else "missing",
                    score(row["parent"]),
                    score(result["fits"]["control"]) if result else None,
                    score(result["fits"]["pruned"]) if result else None,
                    result["selection"]["selected"] if result else None,
                )
            )
            + " |"
        )
    (root / "SUMMARY.md").write_text("\n".join(text) + "\n")
    return payload
