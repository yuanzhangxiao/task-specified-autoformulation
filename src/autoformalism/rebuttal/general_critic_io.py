"""Immutable handoff from pruning into the general advisory-critic pilot."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import process_pruning as pruning
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.rebuttal.repair_scientific_judge import judge_protocol
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
)
from autoformalism.staged_topology import content_hash

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = "general-advisory-critic-1"
POLICY = {
    "revision_episodes": 1,
    "revision_attempts": 3,
    "fit_profile": "collocation-single-target-v2",
    "critic_can_veto": False,
    "critic_can_select": False,
    "routing": "proposer chooses a coherent revision; runtime checks and budgets",
    "selection": "finite validation NMSE, additive terms, incumbent on exact tie",
    "numerical_feedback": "training residuals conditional on retained parameters",
    "benchmark_specific_probes": False,
}


def launcher_hash() -> str:
    """Pin the pilot's entry points alongside package source."""
    names = (
        "scripts/general_critic.py",
        "scripts/submit_general_critic.py",
        "scripts/hpc/run_general_critic_aces.sh",
        "scripts/hpc/submit_general_critic_aces.sh",
        "scripts/smoke_general_critic.py",
        "scripts/submit_shared_process_pilot.py",
        "scripts/submit_review_continuation.py",
    )
    return content_hash(
        {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names}
    )


def make_bundle(request: PublicFitRequest, cell: dict, task: dict) -> dict:
    """Rebuild current equations and boundaries, discarding stale construction slots."""
    model, guesses, audit = public._lower(request)
    candidate = model.validated.candidate.model_dump(mode="json")
    return {
        "source_task": task,
        "brief": cell["brief"],
        "context": request.context.model_dump(mode="json"),
        "candidate": candidate,
        "initialization": {
            "base_candidate": request.base_candidate.model_dump(mode="json"),
            "base_context": request.context.model_dump(mode="json"),
            "plan": request.initialization_plan.model_dump(mode="json"),
            "candidate": candidate,
            "context": model.validated.context.model_dump(mode="json"),
            "guesses": guesses,
            "audit": audit,
        },
    }


def historical_fit(
    directory: Path, request: PublicFitRequest, expected: dict, parent: dict, cell: dict
) -> None:
    """Verify historical receipts without requiring the old executable package."""
    frozen = public._read(directory / "freeze.json")
    body = {k: v for k, v in frozen.items() if k != "identity"}
    public_fit = frozen["public_fit"]
    if (
        public.content_sha256(body) != frozen["identity"]
        or public.content_sha256(
            {k: v for k, v in public_fit.items() if k != "identity"}
        )
        != public_fit["identity"]
        or public_fit["request"] != request.model_dump(mode="json")
        or public_fit["training"] != cell["training"]
        or public_fit["validation"] != cell["validation"]
        or frozen["parent_request"] != parent["request"]
        or frozen["parent_parameters"] != parent["fit"]["parameters"]
    ):
        raise ValueError("historical sibling inputs differ")
    model, _, _ = public._lower(request)
    if public_fit["lowered_candidate"] != model.validated.candidate.model_dump(
        mode="json"
    ):
        raise ValueError("historical lowered equations differ")
    result = sibling_fit._read_result(directory, frozen, request)
    if result is None or result.model_dump(mode="json") != expected:
        raise ValueError("historical sibling result differs")


def snapshot(source: Path) -> tuple[dict, list[dict], dict]:
    """Import the actual pruning decision, not a new score-based choice of parent."""
    pruning.history.require_open(source)
    old = sealed_read(source / "plan.json")
    if old["protocol"] != pruning.PROTOCOL or old["policy"] != pruning.POLICY:
        raise ValueError("requires the completed Milestone 3 pruning pilot")
    # Independently check the original development handoff and its public assets.
    original = Path(old["source"])
    integration, original_rows = pruning.snapshot(original)
    if (
        integration["artifact_sha256"] != old["source_plan_sha256"]
        or original_rows != old["rows"]
        or integration["cells"] != old["cells"]
    ):
        raise ValueError("pruning source lineage differs")
    cells = {}
    for name, cell in old["cells"].items():
        prompt = (
            original / "public/phase_b_v1" / name / "proposer_prompt.txt"
        ).read_text()
        if (
            hashlib.sha256(prompt.encode()).hexdigest()
            != cell["assets"]["proposer_prompt.txt"]
        ):
            raise ValueError("original public prompt differs")
        cells[name] = {**cell, "public_prompt": prompt}
    rows = []
    for row in old["rows"]:
        task, parent = row["task"], row["parent"]
        directory = source / "results" / task["task_id"]
        result = sealed_read(directory / "result.json")
        choice = sealed_read(directory / "choice.json")
        if (
            result["identity"] != old["artifact_sha256"]
            or result["task"] != task
            or result["choice"] != choice
            or result["status"] != "complete"
            or result["test_data_opened"]
        ):
            raise ValueError("pruning result differs or is incomplete")
        expected = pruning.decide(
            parent["fit"], result["fits"]["control"], result["fits"]["pruned"], choice
        )
        if expected != result["selection"]:
            raise ValueError("pruning selection differs")
        for arm, fit in result["fits"].items():
            if fit is not None:
                raw = choice["request"] if arm == "pruned" else parent["request"]
                historical_fit(
                    directory / arm,
                    PublicFitRequest.model_validate(raw),
                    fit,
                    parent,
                    cells[task["cell"]],
                )
        origin = expected["selected"]
        raw = choice["request"] if origin == "pruned" else parent["request"]
        request = PublicFitRequest.model_validate(raw)
        fit = PublicFitResult.model_validate(expected["fit"])
        if pruning.score(fit.model_dump(mode="json")) is None:
            raise ValueError("import requires a finite completed parent")
        certificate = pruning.certificate(request, cells[task["cell"]], task)
        if not certificate["eligible_for_development_selection"]:
            raise ValueError("imported parent fails an executable public requirement")
        rows.append(
            {
                "task": task,
                "request": raw,
                "fit": fit.model_dump(mode="json"),
                "bundle": make_bundle(request, cells[task["cell"]], task),
                "certificate": certificate,
                "pruning_origin": origin,
                "source_result_sha256": result["artifact_sha256"],
                "original_parent": parent,
            }
        )
    return old, rows, cells


def freeze(source: Path, root: Path, judge_revision: str) -> dict:
    """Freeze policy, exact judge/proposer settings and public development lineage."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and output must be separate")
    if not re.fullmatch(r"[a-f0-9]{40}", judge_revision):
        raise ValueError("judge revision must be a full immutable model commit")
    with public._lock(root):
        if (root / "plan.json").exists():
            plan = verify(root)
            if (
                plan["source"] != str(source)
                or plan["judge_revision"] != judge_revision
            ):
                raise ValueError("frozen source or model revision differs")
            return plan
        old, rows, cells = snapshot(source)
        cfg = public._read(REPO / "configs/shared_process_integration_v1.json")
        return sealed_write(
            root / "plan.json",
            {
                "protocol": PROTOCOL,
                "policy": POLICY,
                "source": str(source),
                "source_plan_sha256": old["artifact_sha256"],
                "rows": rows,
                "cells": cells,
                "judge_protocol": judge_protocol(),
                "judge_revision": judge_revision,
                "proposer_settings": cfg["model_settings"],
                "serving_image_sha256": cfg["serving_image_sha256"],
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "launcher_sha256": launcher_hash(),
                "test_data_opened": False,
                "automatic_followup": False,
            },
        )


def verify(root: Path) -> dict:
    """Allow checkpoint execution only under the pinned pilot identity."""
    pruning.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    pruning.history.require_open(Path(plan["source"]))
    if (
        plan["protocol"] != PROTOCOL
        or plan["policy"] != POLICY
        or plan["judge_protocol"] != judge_protocol()
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("critic execution identity differs")
    return plan
