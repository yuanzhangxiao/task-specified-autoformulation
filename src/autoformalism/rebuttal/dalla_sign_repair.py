"""Public-only sign review followed by paired rescue fits and existing pruning."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.models import FitConfig
from autoformalism.llm.staged_topology import (
    DeferredCall,
    StagedModelSettings,
    StagedTopologyClient,
    http_transport,
    visible_response,
)
from autoformalism.rebuttal import dalla_rescue as rescue
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.search import directional_sign_review, sign_review
from autoformalism.staged_topology import content_hash

PROTOCOL = "dalla-sign-repair-1"
REPO = rescue.REPO


class Config(StrictSchema):
    """Pinned local serving and per-model physical request allocation."""

    protocol: Literal["dalla-sign-repair-1", "dalla-sign-repair-2"] = PROTOCOL
    platform: Literal["aces-h100x1"] = "aces-h100x1"
    serving_image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_settings: StagedModelSettings
    served_context_tokens: int = Field(default=32768, ge=8192)
    wall_seconds: int = Field(default=3000, ge=60, le=7200)
    selected_task_ids: tuple[str, ...] = ()


def launcher_hash() -> str:
    """Pin this orchestration and its reused scheduler/server helpers."""
    names = (
        "scripts/dalla_sign_repair.py",
        "scripts/submit_dalla_sign_repair.py",
        "scripts/smoke_dalla_sign_repair.py",
        "scripts/hpc/run_dalla_sign_repair_aces.sh",
        "scripts/hpc/run_staged_topology_server.sh",
        "scripts/submit_shared_process_pilot.py",
        "scripts/submit_review_continuation.py",
        "scripts/recover_review_continuation_submission.py",
    )
    return public.content_sha256(
        {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names}
    )


def freeze(source: Path, config_path: Path, root: Path) -> dict:
    """Import every selected rescue endpoint without ranking or numerical work."""
    source, root = source.resolve(), root.resolve()
    rescue.pruning.history.require_open(root)
    if source.is_relative_to(root):
        raise ValueError("source must be outside output root")
    packet = public._read(source)
    config = Config.model_validate(public._read(config_path))
    if (
        packet.get("protocol") != rescue.PROTOCOL
        or packet.get("test_data_opened") is not False
        or packet.get("interventions_evaluated") is not False
    ):
        raise ValueError("requires original public-only rescue models.json")
    rows = []
    selected = set(config.selected_task_ids)
    available = {item["task"]["task_id"] for item in packet["models"]}
    if len(selected) != len(config.selected_task_ids) or not selected <= available:
        raise ValueError("selected task IDs must be unique and present in source")
    if config.protocol == "dalla-sign-repair-1" and selected:
        raise ValueError("task selection requires the v2 protocol")
    adapter = (
        directional_sign_review
        if config.protocol == "dalla-sign-repair-2"
        else sign_review
    )
    for item in packet["models"]:
        if selected and item["task"]["task_id"] not in selected:
            continue
        value = item["result"]
        # Rescue results are sealed with content_hash, not the compact JSON
        # convention used for public fitting identities.
        if value["artifact_sha256"] != content_hash(
            {k: v for k, v in value.items() if k != "artifact_sha256"}
        ):
            raise ValueError(f"source result digest differs: {item['task']['task_id']}")
        if value.get("test_data_opened") is not False or value["task"] != item["task"]:
            raise ValueError("source task or data boundary differs")
        request = PublicFitRequest.model_validate(value["selected_request"])
        cell = packet["cells"][item["task"]["cell"]]
        train, val = (
            PublicSplit.model_validate(cell[k]) for k in ("training", "validation")
        )
        public._bundle(request, train, val)
        parameters = value["selected_fit"]["parameters"]
        sibling_fit.compatible_seed(request, request, parameters, train)
        rows.append(
            {
                "task": item["task"],
                "request": request.model_dump(mode="json"),
                "parameters": parameters,
                "original_fit": value["selected_fit"],
                "source_result_sha256": value["artifact_sha256"],
                "review_context": adapter.context(request, cell["brief"]),
            }
        )
    import re

    ids = [r["task"]["task_id"] for r in rows]
    if (
        not 1 <= len(rows) <= 12
        or len(set(ids)) != len(ids)
        or any(not re.fullmatch(r"[a-z0-9_]+", n) for n in ids)
    ):
        raise ValueError("requires 1-12 unique safe task identifiers")
    plan = {
        "protocol": config.protocol,
        "config": config.model_dump(mode="json"),
        "source": str(source),
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_plan_sha256": packet["plan_sha256"],
        "rows": rows,
        "cells": packet["cells"],
        "runtime": public._runtime(),
        "source_sha256": public._source_identity(),
        "launcher_sha256": launcher_hash(),
        "sign_policy": adapter.POLICY,
        "fit_policy": rescue.POLICY,
        "test_data_opened": False,
    }
    with public._lock(root):
        return sealed_write(root / "plan.json", plan)


def verify(root: Path) -> dict:
    """Reject changed input, runtime, policies or code instead of changing a run."""
    rescue.pruning.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    config = Config.model_validate(plan["config"])
    adapter = (
        directional_sign_review
        if config.protocol == "dalla-sign-repair-2"
        else sign_review
    )
    if (
        plan["protocol"] != config.protocol
        or plan["sign_policy"] != adapter.POLICY
        or plan["fit_policy"] != rescue.POLICY
        or plan["runtime"] != public._runtime()
        or plan["source_sha256"] != public._source_identity()
        or plan["launcher_sha256"] != launcher_hash()
        or hashlib.sha256(Path(plan["source"]).read_bytes()).hexdigest()
        != plan["source_file_sha256"]
    ):
        raise ValueError("sign-repair execution identity differs")
    Config.model_validate(plan["config"])
    return plan


def _review_path(root: Path, row: dict) -> Path:
    return root / "results" / row["task"]["task_id"] / "review.json"


def checked_review(root: Path, plan: dict, row: dict) -> dict:
    """Recompute accepted sign patches before their numerical handoff."""
    value = sealed_read(_review_path(root, row))
    if value["identity"] != plan["artifact_sha256"] or value["task"] != row["task"]:
        raise ValueError("review identity differs")
    if value["status"] in {"repaired", "unchanged"}:
        request = PublicFitRequest.model_validate(row["request"])
        brief = plan["cells"][row["task"]["cell"]]["brief"]
        if plan["protocol"] == "dalla-sign-repair-2":
            rebuilt = directional_sign_review.finish(
                request, brief, value["direction_history"]
            )
            if (
                rebuilt["reply"] != value["reply"]
                or rebuilt["status"] != value["status"]
            ):
                raise ValueError("saved semantic review differs")
            expected = rebuilt["patch"]
        else:
            expected = sign_review.apply(request, brief, value["reply"])
        if (
            value["patch"] != expected
            or (value["status"] == "repaired") != expected["changed"]
        ):
            raise ValueError("saved repair differs from its explicit decisions")
    return value


def review_one(
    root: Path,
    index: int,
    *,
    base_url: str,
    transport=http_transport,
    can_start=lambda: True,
) -> dict:
    """Cached bounded repair attempts, with deterministic diagnostics on retries."""
    plan = verify(root)
    if index not in range(len(plan["rows"])):
        raise ValueError("index outside campaign")
    row = plan["rows"][index]
    directory = _review_path(root, row).parent
    with public._lock(directory):
        if _review_path(root, row).exists():
            return checked_review(root, plan, row)
        attempts, reply, patch = [], None, None
        client = StagedTopologyClient(
            settings=StagedModelSettings.model_validate(
                plan["config"]["model_settings"]
            ),
            base_url=base_url,
            directory=directory / "calls",
            namespace=plan["artifact_sha256"] + str(index),
            seed=row["task"]["seed"],
            transport=transport,
            can_start=can_start,
        )
        status, extra = "no_eligible_gains", {}
        if (
            row["review_context"]["eligible_slots"]
            and plan["protocol"] == "dalla-sign-repair-2"
        ):
            reviewed = directional_sign_review.run(
                client,
                PublicFitRequest.model_validate(row["request"]),
                plan["cells"][row["task"]["cell"]]["brief"],
            )
            status, reply, patch, attempts = (
                reviewed[k] for k in ("status", "reply", "patch", "attempts")
            )
            extra["direction_history"] = reviewed["direction_history"]
            if patch["changed"]:
                certificate = rescue.pruning.certificate(
                    PublicFitRequest.model_validate(patch["request"]),
                    plan["cells"][row["task"]["cell"]],
                    row["task"],
                )
                if not certificate["eligible_for_development_selection"]:
                    status = "admission_failed"
                    extra["admission_certificate"] = certificate
        elif row["review_context"]["eligible_slots"]:
            status = "attempts_exhausted"
            diagnostic = None
            for attempt in range(client.settings.attempts_per_step):
                user = json.dumps(
                    {**row["review_context"], "repair_diagnostic": diagnostic},
                    sort_keys=True,
                )
                record = client.call(
                    system=sign_review.SYSTEM,
                    user=user,
                    response_model=sign_review.SignReview,
                    step="outer_sign_review",
                    attempt=attempt,
                )
                try:
                    raw = visible_response(record)
                    patch = sign_review.apply(
                        PublicFitRequest.model_validate(row["request"]),
                        plan["cells"][row["task"]["cell"]]["brief"],
                        raw,
                    )
                    # Compile the actual child and recheck its public obligations.
                    certificate = rescue.pruning.certificate(
                        PublicFitRequest.model_validate(patch["request"]),
                        plan["cells"][row["task"]["cell"]],
                        row["task"],
                    )
                    if not certificate["eligible_for_development_selection"]:
                        raise ValueError("repaired candidate fails public admission")
                    reply = raw
                    attempts.append(
                        {
                            "attempt": attempt,
                            "accepted": True,
                            "request_hash": record["request_hash"],
                        }
                    )
                    status = "repaired" if patch["changed"] else "unchanged"
                    break
                except (ValueError, TypeError, KeyError) as error:
                    patch = None
                    diagnostic = str(error)[:3000]
                    attempts.append(
                        {
                            "attempt": attempt,
                            "accepted": False,
                            "error": diagnostic,
                            "request_hash": record["request_hash"],
                        }
                    )
        return sealed_write(
            _review_path(root, row),
            {
                "identity": plan["artifact_sha256"],
                "task": row["task"],
                "status": status,
                "reply": reply,
                "patch": patch,
                "attempts": attempts,
                "physical_requests": len(client.records),
                "observed_tokens": sum(
                    r.get("observed_total_tokens") or 0 for r in client.records
                ),
                "budget_charge": sum(r.get("budget_charge", 0) for r in client.records),
                "unknown_usage_requests": sum(
                    r.get("observed_total_tokens") is None for r in client.records
                ),
                "scientific_correctness_certified": False,
                "test_data_opened": False,
                **extra,
            },
        )


def run_reviews(root: Path, base_url: str, wall_seconds: int) -> dict:
    """One server for the frozen set; a drain preserves remaining request budget."""
    import signal

    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    previous = signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + min(
        wall_seconds, verify(root)["config"]["wall_seconds"]
    )
    try:
        for i in range(len(verify(root)["rows"])):
            try:
                review_one(
                    root,
                    i,
                    base_url=base_url,
                    can_start=lambda: not stopped and time.monotonic() + 190 < deadline,
                )
            except DeferredCall:
                break
    finally:
        signal.signal(signal.SIGTERM, previous)
    return report(root)


def prepare_fits(root: Path, plan: dict, row: dict, review: dict) -> Path:
    """Use the existing rescue runner for a repaired/unchanged allocation pair."""
    cell = plan["cells"][row["task"]["cell"]]
    parent = PublicFitRequest.model_validate(row["request"])
    child = PublicFitRequest.model_validate(review["patch"]["request"])
    warm = sibling_fit.compatible_seed(
        parent, child, row["parameters"], PublicSplit.model_validate(cell["training"])
    )
    inputs = root / "fit-inputs" / (row["task"]["task_id"] + ".json")
    sealed_write(
        inputs,
        {
            "protocol": rescue.INPUT_PROTOCOL,
            "test_data_opened": False,
            "cells": {row["task"]["cell"]: cell},
            "rows": [
                {
                    "task": {**row["task"], "task_id": arm},
                    "request": req,
                    "parameters": parameters,
                    "sign_review_sha256": review["artifact_sha256"],
                    "warm_start_audit": warm if arm == "repaired" else None,
                }
                for arm, req, parameters in (
                    ("repaired", child.model_dump(mode="json"), warm["parameters"]),
                    ("unchanged_control", row["request"], row["parameters"]),
                )
            ],
        },
    )
    output = root / "fitting" / row["task"]["task_id"]
    rescue.freeze(inputs, output)
    return output


def sign_integrity(review: dict, request: dict, fit: dict) -> dict:
    """Check retained fixed domains/values and report gains removed by pruning."""
    model, _, _ = public._lower(PublicFitRequest.model_validate(request))
    specs = {p.name: p for p in model.validated.candidate.parameters}
    removed, checked = [], []
    for slot in review["patch"]["provenance"]["fixed_slots"]:
        name = slot["parameter"]
        if name not in specs:
            removed.append(name)
            continue
        spec = specs[name]
        bound = _parameter_variable(model, spec, FitConfig())
        if (
            spec.domain.value != "nonnegative"
            or not bound.lower <= fit["parameters"][name] <= bound.upper
        ):
            raise ValueError("fixed sign domain or fitted bound violated")
        sign_review.check_fixed_term(
            model.validated.candidate.model_dump(mode="json"), slot
        )
        checked.append(name)
    return {
        "passed": True,
        "checked_fixed_gains": checked,
        "pruned_fixed_gains": removed,
        "scope": (
            "fixed outer operators, inner laws, magnitude domains and fitted bounds; "
            "not global monotonicity"
        ),
    }


def fit_one(root: Path, index: int) -> dict:
    """Keep repaired and unchanged rescue/pruning allocations separate."""
    plan = verify(root)
    if index not in range(len(plan["rows"])):
        raise ValueError("index outside campaign")
    row = plan["rows"][index]
    with public._lock(root / "fit-locks" / row["task"]["task_id"]):
        review = checked_review(root, plan, row)
        if review["status"] != "repaired":
            return {"status": "not_fitted", "reason": review["status"]}
        directory = prepare_fits(root, plan, row, review)
        # Independent, already checkpointed allocations allow partial numerical resume.
        repaired = rescue.run_one(directory, 0)
        rescue.run_one(directory, 1)
        audit = None
        if repaired["selected_fit"]:
            audit = sign_integrity(
                review, repaired["selected_request"], repaired["selected_fit"]
            )
        result = rescue.report(directory)
        return {"status": result["status"], "sign_integrity": audit}


def report(root: Path) -> dict:
    """Report incomplete, unchanged and failed repairs with all endpoints."""
    with public._lock(root / "reporting"):
        if not (root / "plan.json").exists():
            result = {"protocol": PROTOCOL, "status": "not_prepared", "rows": []}
            public._write(root / "summary.json", result)
            return result
        plan = verify(root)
        rows, models = [], []
        for row in plan["rows"]:
            review = (
                checked_review(root, plan, row)
                if _review_path(root, row).exists()
                else None
            )
            status = review["status"] if review else "review_pending"
            fits = {}
            numerical = root / "fitting" / row["task"]["task_id"]
            if review and review["status"] == "repaired":
                status = "fit_pending"
                if (numerical / "plan.json").exists():
                    fit_plan = rescue.verify(numerical)
                    for fit_row in fit_plan["rows"]:
                        directory = numerical / "results" / fit_row["task"]["task_id"]
                        if (directory / "result.json").exists():
                            fits[fit_row["task"]["task_id"]] = rescue.checked(
                                directory, fit_plan, fit_row
                            )
                    if len(fits) == 2:
                        status = (
                            "complete"
                            if fits["repaired"]["selected_fit"]
                            else "repair_fit_failed"
                        )
            audit = None
            if fits.get("repaired", {}).get("selected_fit"):
                audit = sign_integrity(
                    review,
                    fits["repaired"]["selected_request"],
                    fits["repaired"]["selected_fit"],
                )
            rows.append(
                {
                    "task": row["task"],
                    "status": status,
                    "review_status": review["status"] if review else None,
                    "fixed_gains": len(review["patch"]["provenance"]["fixed_slots"])
                    if review and review["patch"]
                    else 0,
                    "physical_requests": review["physical_requests"]
                    if review
                    else None,
                    "observed_tokens": review["observed_tokens"] if review else None,
                    "budget_charge": review["budget_charge"] if review else None,
                    "unknown_usage_requests": review["unknown_usage_requests"]
                    if review
                    else None,
                    "original_training": row["original_fit"]["training"][
                        "normalized_mse"
                    ],
                    "original_validation": row["original_fit"]["validation"][
                        "normalized_mse"
                    ],
                    "arms": {
                        arm: {
                            "status": v["status"],
                            "training": v["selected_fit"]["training"]
                            if v["selected_fit"]
                            else None,
                            "validation": v["selected_fit"]["validation"]
                            if v["selected_fit"]
                            else None,
                            "pruning_selected": v["pruning"]["selection"]["selected"]
                            if v["pruning"]
                            else None,
                        }
                        for arm, v in fits.items()
                    },
                    "sign_integrity": audit,
                    "citation_audit": review["patch"]["provenance"].get(
                        "citation_audit", []
                    )
                    if review and review["patch"]
                    else [],
                    "unresolved_slots": review["patch"]["provenance"].get(
                        "unresolved_slots", []
                    )
                    if review and review["patch"]
                    else [],
                }
            )
            models.append(
                {
                    "task": row["task"],
                    "original_request": row["request"],
                    "original_fit": row["original_fit"],
                    "review": review,
                    "arms": fits,
                }
            )
        result = {
            "protocol": plan["protocol"],
            "plan_sha256": plan["artifact_sha256"],
            "status": "partial"
            if any(r["status"] in {"review_pending", "fit_pending"} for r in rows)
            else "complete",
            "expected": len(rows),
            "rows": rows,
            "test_data_opened": False,
            "limitation": (
                "Exploratory public-context sign repair. Equal fitting allocations, "
                "different warm starts. No scientific or intervention certification."
            ),
        }
        public._write(
            root / "models.json",
            {
                "protocol": plan["protocol"],
                "plan_sha256": plan["artifact_sha256"],
                "cells": plan["cells"],
                "models": models,
                "test_data_opened": False,
                "interventions_evaluated": False,
            },
        )
        public._write(root / "summary.json", result)
        return result
