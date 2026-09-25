"""Explicitly assisted sign hypotheses with paired CPU rescue/pruning fits.

This is a development diagnostic, not a successful autonomous sign review.
No provider, reference simulator or intervention-evaluation path is invoked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import dalla_rescue as rescue
from autoformalism.rebuttal.dalla_sign_repair import sign_integrity
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.search import directional_sign_review as directional
from autoformalism.search import sign_review
from autoformalism.staged_topology import content_hash

PROTOCOL = "dalla-sign-diagnostic-1"
REPO = rescue.REPO
LABEL = "assistant_specified_under_user_authorization"
LIMITATION = (
    "Assisted, post-hoc sign diagnostic on a selected historical model, not "
    "autonomous discovery or scientific certification. Directions have explicit "
    "public-role or structural-hypothesis rationales; ambiguity remains visible. "
    "Fitting and pruning select on training. No intervention or test scoring."
)


class Decisions(StrictSchema):
    """Complete explicit decisions, bound to an exact model and public context."""

    protocol: Literal["dalla-sign-diagnostic-1"] = PROTOCOL
    decision_source: Literal["assistant_specified_under_user_authorization"] = LABEL
    task_id: str = Field(pattern=r"^[a-z0-9_]+$")
    source_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    assumptions: tuple[str, ...] = Field(min_length=1)
    review: directional.DirectionReview


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launcher_hash() -> str:
    """Pin the diagnostic entrypoints as well as reused fitting orchestration."""
    names = (
        "scripts/dalla_sign_diagnostic.py",
        "scripts/submit_dalla_sign_diagnostic.py",
        "scripts/smoke_dalla_sign_diagnostic.py",
        "scripts/hpc/run_dalla_sign_diagnostic.sh",
        "scripts/hpc/launch_dalla_sign_diagnostic.sh",
    )
    return content_hash({n: _digest(REPO / n) for n in names})


def derive(source: Path, decisions_path: Path) -> dict:
    """Apply declared hypotheses mechanically; never infer signs from fit values."""
    packet = public._read(source)
    decisions = Decisions.model_validate(public._read(decisions_path))
    if (
        packet.get("protocol") != rescue.PROTOCOL
        or packet.get("test_data_opened") is not False
        or packet.get("interventions_evaluated") is not False
    ):
        raise ValueError("requires original public-only rescue models.json")
    matches = [m for m in packet["models"] if m["task"]["task_id"] == decisions.task_id]
    if len(matches) != 1:
        raise ValueError("requires exactly one matching source task")
    entry = matches[0]
    value = entry["result"]
    if (
        value["artifact_sha256"] != decisions.source_result_sha256
        or value["artifact_sha256"]
        != content_hash({k: v for k, v in value.items() if k != "artifact_sha256"})
        or value["task"] != entry["task"]
        or value.get("test_data_opened") is not False
    ):
        raise ValueError("source result identity differs")
    parent = PublicFitRequest.model_validate(value["selected_request"])
    if parent.profile != rescue.PROFILE:
        raise ValueError("diagnostic preserves the existing rescue profile")
    cell = packet["cells"][entry["task"]["cell"]]
    context = sign_review.context(parent, cell["brief"])
    if (
        content_hash(parent.model_dump(mode="json")) != decisions.request_sha256
        or content_hash(context) != decisions.public_context_sha256
    ):
        raise ValueError("decision model or public context differs")
    declared = decisions.review.model_dump(mode="json")
    directional.proposals(
        parent, declared, {s["slot_id"] for s in context["eligible_slots"]}
    )
    # Reuse only scope/flow-consistency checks, never fabricate a semantic assessment.
    raw = {
        "decisions": [
            {k: v for k, v in d.items() if k in sign_review.SignDecision.model_fields}
            for d in declared["decisions"]
        ]
    }
    patch = sign_review.apply(parent, cell["brief"], raw, advisory_citations=True)
    if not patch["changed"]:
        raise ValueError("diagnostic requires at least one explicit sign constraint")
    child = PublicFitRequest.model_validate(patch["request"])
    training = PublicSplit.model_validate(cell["training"])
    parameters = value["selected_fit"]["parameters"]
    warm = sibling_fit.compatible_seed(parent, child, parameters, training)
    provenance = {
        "protocol": PROTOCOL,
        "decision_source": LABEL,
        "decisions_sha256": content_hash(decisions.model_dump(mode="json")),
        "semantic_assessor_called": False,
        "scientific_correctness_certified": False,
    }
    rows = []
    for arm, req, vector in (
        ("repaired", child.model_dump(mode="json"), warm["parameters"]),
        ("unchanged_control", parent.model_dump(mode="json"), parameters),
    ):
        rows.append(
            {
                "task": {**entry["task"], "task_id": arm, "diagnostic": provenance},
                "request": req,
                "parameters": vector,
                "warm_start_audit": warm if arm == "repaired" else None,
            }
        )
    return {
        "source_task": entry["task"],
        "decisions": decisions.model_dump(mode="json"),
        "patch": patch,
        "provenance": provenance,
        "inputs": {
            "protocol": rescue.INPUT_PROTOCOL,
            "cells": {entry["task"]["cell"]: cell},
            "rows": rows,
            "test_data_opened": False,
        },
    }


def freeze(source: Path, decisions: Path, root: Path, *, site: str = "aces") -> dict:
    """Freeze a separate paired diagnostic without any numerical fit or LLM call."""
    source, decisions, root = source.resolve(), decisions.resolve(), root.resolve()
    if site not in {"aces", "delta"}:
        raise ValueError("unsupported CPU site")
    if source.is_relative_to(root) or decisions.is_relative_to(root):
        raise ValueError("source and explicit decisions must be outside output root")
    rescue.pruning.history.require_open(root)
    derived = derive(source, decisions)
    with public._lock(root):
        inputs = sealed_write(root / "paired-inputs.json", derived["inputs"])
        fitted = rescue.freeze(root / "paired-inputs.json", root / "fitting")
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "site": site,
                "source": str(source),
                "source_file_sha256": _digest(source),
                "decisions_path": str(decisions),
                "decisions_file_sha256": _digest(decisions),
                **{k: v for k, v in derived.items() if k != "inputs"},
                "input_sha256": inputs["artifact_sha256"],
                "fitting_plan_sha256": fitted["artifact_sha256"],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": launcher_hash(),
                "live_llm_calls": 0,
                "test_data_opened": False,
                "interventions_evaluated": False,
                "limitation": LIMITATION,
            },
        )


def verify(root: Path) -> dict:
    """Verify model/decision identity and both paired fitting inputs without fitting."""
    rescue.pruning.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
        or _digest(Path(plan["source"])) != plan["source_file_sha256"]
        or _digest(Path(plan["decisions_path"])) != plan["decisions_file_sha256"]
    ):
        raise ValueError("diagnostic identity differs")
    derived = derive(Path(plan["source"]), Path(plan["decisions_path"]))
    if any(plan[k] != v for k, v in derived.items() if k != "inputs"):
        raise ValueError("explicit patch differs from decision record")
    inputs = sealed_read(root / "paired-inputs.json")
    if (
        inputs["artifact_sha256"] != plan["input_sha256"]
        or {k: v for k, v in inputs.items() if k != "artifact_sha256"}
        != derived["inputs"]
        or rescue.verify(root / "fitting")["artifact_sha256"]
        != plan["fitting_plan_sha256"]
    ):
        raise ValueError("paired diagnostic inputs differ")
    return plan


def run_one(root: Path, index: int) -> dict:
    """Each CPU array element runs one independent, equal-budget rescue arm."""
    plan = verify(root)
    if index not in (0, 1):
        raise ValueError("expected repaired=0 or unchanged_control=1")
    result = rescue.run_one(root / "fitting", index)
    audit = None
    if index == 0 and result["selected_fit"]:
        audit = sign_integrity(
            {"patch": plan["patch"]}, result["selected_request"], result["selected_fit"]
        )
    return {
        "status": result["status"],
        "arm": result["task"]["task_id"],
        "sign_integrity": audit,
    }


def report(root: Path) -> dict:
    """Export labeled partial or complete endpoints without fitting or promotion."""
    with public._lock(root / "reporting"):
        if not (root / "plan.json").exists():
            result = {"protocol": PROTOCOL, "status": "not_prepared", "rows": []}
            public._write(root / "summary.json", result)
            return result
        plan = verify(root)
        result = rescue.report(root / "fitting")
        models = public._read(root / "fitting/models.json")
        audit = None
        for item in models["models"]:
            value = item["result"]
            if item["task"]["task_id"] == "repaired" and value["selected_fit"]:
                audit = sign_integrity(
                    {"patch": plan["patch"]},
                    value["selected_request"],
                    value["selected_fit"],
                )
        labels = {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "decision_source": LABEL,
            "semantic_assessor_called": False,
            "scientific_correctness_certified": False,
            "sign_integrity": audit,
            "decisions": plan["decisions"],
            "source_task": plan["source_task"],
            "limitation": LIMITATION,
        }
        public._write(root / "models.json", {**models, **labels, "live_llm_calls": 0})
        result.update(labels)
        public._write(root / "summary.json", result)
        return result
