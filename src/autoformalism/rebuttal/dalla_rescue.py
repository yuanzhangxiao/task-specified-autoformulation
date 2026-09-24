"""Imported demonstration candidates: bounded refit and existing paired pruning.

No proposer, reference simulator or intervention data is used by this campaign.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import re
import signal
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.fitter import evaluate_fitted_candidate
from autoformalism.rebuttal import process_pruning as pruning
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.search import review_revision_multi

PROTOCOL = "dalla-demonstration-rescue-1"
INPUT_PROTOCOL = "dalla-rescue-inputs-1"
PROFILE = "collocation-rescue-v1"
REPO = Path(__file__).resolve().parents[3]
POLICY = {
    "profile": PROFILE,
    "seed_replay_seconds": 600,
    "refits_per_seed": 1,
    "refit_selection": "lowest complete training NMSE; seed wins ties",
    "pruning": {**pruning.POLICY, "fit_profile": PROFILE},
    "interventions_used_for_selection": False,
}


def launcher_hash() -> str:
    """Pin orchestration, including scheduler receipt/recovery helpers."""
    names = (
        "scripts/dalla_rescue.py",
        "scripts/build_dalla_rescue_inputs.py",
        "scripts/smoke_dalla_rescue.py",
        "scripts/submit_dalla_rescue.py",
        "scripts/hpc/run_dalla_rescue_aces.sh",
        "scripts/submit_shared_process_pilot.py",
        "scripts/submit_review_continuation.py",
        "scripts/recover_review_continuation_submission.py",
    )
    return public.content_sha256(
        {n: hashlib.sha256((REPO / n).read_bytes()).hexdigest() for n in names}
    )


def correct_declaration(bundle: dict, raw: dict, old: str, new: str) -> dict:
    """Explicit saved-patch correction; never silently change an incumbent role.

    The caller chooses the fresh name. Only references within the submitted
    equation patch are renamed; existing equations and declarations are untouched.
    Citation claims are left unverified when the historical catalog is absent.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", new) or new == old:
        raise ValueError("a distinct safe fresh name is required")
    try:
        review_revision_multi.apply_edits(
            bundle, None, raw, reject_existing_role_conflicts=True
        )
    except ValueError as error:
        if (
            getattr(error, "code", None) != "EXISTING_PARAMETER_ROLE_CONFLICT"
            or error.details["referenced_name"] != old
        ):
            raise
    else:
        raise ValueError("saved patch has no identified declaration conflict")
    text = str(bundle)
    if re.search(rf"\b{re.escape(new)}\b", text):
        raise ValueError("fresh name is already occupied")
    changed = copy.deepcopy(raw)
    declarations = [p for p in changed["new_parameters"] if p["name"] == old]
    if len(declarations) != 1:
        raise ValueError("requires one explicit conflicting declaration")
    declarations[0]["name"] = new

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            return (
                ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
                if (node.id == old)
                else node
            )

    touched = []
    for equation in changed["equations"]:
        tree = ast.parse(equation["expression"], mode="eval")
        if any(isinstance(n, ast.Name) and n.id == old for n in ast.walk(tree)):
            equation["expression"] = ast.unparse(Rename().visit(tree))
            touched.append(equation["component"])
    if not touched:
        raise ValueError("new declaration has no references in the patch")
    revised = review_revision_multi.apply_edits(
        bundle, None, changed, reject_existing_role_conflicts=True
    )
    if revised["bundle"] is None:
        raise ValueError("corrected patch did not produce a candidate")
    return {
        "original_patch_sha256": public.content_sha256(raw),
        "corrected_patch": changed,
        "renamed_declaration": {"old": old, "new": new, "components": touched},
        "revision": revised,
        "verified_scientific_citations": False,
    }


def _request(raw: dict) -> PublicFitRequest:
    return PublicFitRequest.model_validate({**raw, "profile": PROFILE})


def freeze(source: Path, root: Path) -> dict:
    """Freeze public inputs and compatible complete starts without fitting."""
    source, root = source.resolve(), root.resolve()
    pruning.history.require_open(root)
    if source.is_relative_to(root):
        raise ValueError("source must be outside the output directory")
    inputs = sealed_read(source)
    if (
        inputs["protocol"] != INPUT_PROTOCOL
        or inputs.get("test_data_opened") is not False
    ):
        raise ValueError("requires a sealed public-only rescue input packet")
    rows, seen = [], set()
    for entry in inputs["rows"]:
        row = copy.deepcopy(entry)
        label = row["task"]["task_id"]
        if not re.fullmatch(r"[a-z0-9_]+", label) or label in seen:
            raise ValueError("task IDs must be unique safe directory names")
        seen.add(label)
        request = _request(row["request"])
        cell = inputs["cells"][row["task"]["cell"]]
        train = PublicSplit.model_validate(cell["training"])
        val = PublicSplit.model_validate(cell["validation"])
        public._bundle(request, train, val)
        sibling_fit.compatible_seed(request, request, row["parameters"], train)
        row["request"] = request.model_dump(mode="json")
        row["certificate"] = pruning.certificate(request, cell, row["task"])
        if not row["certificate"]["eligible_for_development_selection"]:
            raise ValueError(f"hard public requirement failed: {label}")
        rows.append(row)
    if not 1 <= len(rows) <= 12:
        raise ValueError("rescue packet must contain 1-12 explicit starts")
    payload = {
        "protocol": PROTOCOL,
        "policy": POLICY,
        "source": str(source),
        "input_sha256": inputs["artifact_sha256"],
        "cells": inputs["cells"],
        "rows": rows,
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "launcher_sha256": launcher_hash(),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        return sealed_write(root / "plan.json", payload)


def verify(root: Path) -> dict:
    """Fail closed on source, runtime, input or policy drift."""
    pruning.history.require_open(root)
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["policy"] != POLICY
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or plan["launcher_sha256"] != launcher_hash()
    ):
        raise ValueError("rescue execution identity differs")
    if sealed_read(Path(plan["source"]))["artifact_sha256"] != plan["input_sha256"]:
        raise ValueError("rescue input packet changed")
    return plan


@contextmanager
def _deadline(seconds: int):
    """Bound the complete seed replay on a Unix CPU worker, not each sample."""

    def expired(*_):
        raise TimeoutError("seed replay allocation exhausted")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def replay_seed(row: dict, cell: dict) -> dict:
    """Evaluate supplied parameters without fitting any global or local value."""
    request = PublicFitRequest.model_validate(row["request"])
    model, _, _ = public._lower(request)
    train = public.unpack_split(PublicSplit.model_validate(cell["training"]))
    scales = TrainingScaler().fit(train).scales
    scales = {
        t: scales[f"target:{t}"].standard_deviation for t in request.context.targets
    }
    config = CollocationSensitivityConfig.model_validate(
        public.profile_settings(request)
    ).fit_config()
    metrics = {}
    with _deadline(POLICY["seed_replay_seconds"]):
        for name in ("training", "validation"):
            split = (
                train
                if name == "training"
                else public.unpack_split(PublicSplit.model_validate(cell[name]))
            )
            _, raw = evaluate_fitted_candidate(
                model,
                split,
                global_parameters=row["parameters"],
                global_initial_conditions={},
                target_scales=scales,
                config=config,
                fit_trajectory_initial_conditions=False,
            )
            metrics[name] = public._metrics(
                public._jsonable(raw), request.context.targets
            ).model_dump(mode="json")
    return {
        "status": "complete"
        if all(m["available"] for m in metrics.values())
        else "rollout_failed",
        "parameters": row["parameters"],
        **metrics,
        "parameter_fitting_performed": False,
        "validation_initials_fitted": False,
    }


def retain(seed: dict | None, fitted: dict | None) -> tuple[str | None, dict | None]:
    """Training-only comparison; keep the seed on exact ties."""
    choices = [("seed", seed), ("refit", fitted)]
    valid = [(name, fit) for name, fit in choices if pruning.score(fit) is not None]
    return (
        min(valid, key=lambda x: x[1]["training"]["normalized_mse"])
        if valid
        else (None, None)
    )


def _fit_health(directory: Path) -> dict:
    path = directory / "backend_result.json"
    if not path.exists():
        return {"backend_available": False}
    raw = public._read(path)
    initializer = raw.get("initializer") or {}
    return {
        "backend_available": True,
        "initializer_success": initializer.get("success"),
        "initializer_message": initializer.get("message"),
        "refinement_failure_counts": dict(
            Counter(
                e.get("status", "unknown")
                for e in (raw.get("refinement") or {}).get("failure_evidence", [])
            )
        ),
    }


def _seed_checkpoint(directory: Path, plan: dict, row: dict, cell: dict) -> dict:
    path, marker = directory / "seed.json", directory / "seed_started.json"
    if path.exists():
        saved = sealed_read(path)
        if saved["identity"] != plan["artifact_sha256"]:
            raise ValueError("seed replay identity differs")
        return saved
    if marker.exists():
        if sealed_read(marker)["identity"] != plan["artifact_sha256"]:
            raise ValueError("consumed seed allocation identity differs")
        value = {
            "status": "interrupted",
            "parameters": None,
            "allocation_consumed": True,
        }
    else:
        sealed_write(marker, {"identity": plan["artifact_sha256"]})
        try:
            value = replay_seed(row, cell)
        except Exception as error:
            value = {
                "status": "rollout_failed",
                "parameters": None,
                "message": str(error),
            }
    return sealed_write(path, {"identity": plan["artifact_sha256"], **value})


def checked(directory: Path, plan: dict, row: dict) -> dict:
    """Verify every published decision against the original fit receipts."""
    result = sealed_read(directory / "result.json")
    if result["identity"] != plan["artifact_sha256"] or result["task"] != row["task"]:
        raise ValueError("rescue result identity differs")
    seed = sealed_read(directory / "seed.json")
    fitted = sibling_fit.inspect_child_fit(directory / "refit")["result"]
    if result["seed"] != seed or result["refit"] != fitted:
        raise ValueError("rescue receipt differs")
    origin, parent = retain(seed, fitted)
    if result["pre_pruning_origin"] != origin or result["pre_pruning_fit"] != parent:
        raise ValueError("rescue training selection differs")
    expected_fit, expected_request = parent, row["request"] if parent else None
    if parent is not None:
        pruning_row = {
            "task": row["task"],
            "parent": {"request": row["request"], "fit": parent},
        }
        p = pruning.checked_result(directory / "pruning", plan, pruning_row)
        if p != result["pruning"]:
            raise ValueError("pruning receipt differs")
        expected_fit = p["selection"]["fit"]
        if p["selection"]["selected"] == "pruned":
            expected_request = p["choice"]["request"]
    if (
        result["selected_fit"] != expected_fit
        or result["selected_request"] != expected_request
    ):
        raise ValueError("selected model differs")
    return result


def run_one(root: Path, index: int) -> dict:
    """One refit and one existing paired pruning pass per explicit seed."""
    plan = verify(root)
    if index not in range(len(plan["rows"])):
        raise ValueError("task index outside frozen rescue packet")
    row = plan["rows"][index]
    cell = plan["cells"][row["task"]["cell"]]
    directory = root / "results" / row["task"]["task_id"]
    with public._lock(directory):
        if (directory / "result.json").exists():
            return checked(directory, plan, row)
        seed = _seed_checkpoint(directory, plan, row, cell)
        request = PublicFitRequest.model_validate(row["request"])
        sibling_fit.prepare_child_fit(
            request,
            request,
            row["parameters"],
            PublicSplit.model_validate(cell["training"]),
            PublicSplit.model_validate(cell["validation"]),
            directory / "refit",
            lineage={"campaign": plan["artifact_sha256"], "task": row["task"]},
        )
        fitted = sibling_fit.execute_child_fit(directory / "refit").model_dump(
            mode="json"
        )
        origin, parent = retain(seed, fitted)
        pruned, selected_request, selected_fit = None, None, None
        if parent is not None:
            pruning_row = {
                "task": row["task"],
                "parent": {"request": row["request"], "fit": parent},
            }
            pruned = pruning.execute_row(
                directory / "pruning", plan, pruning_row, index
            )
            selected_fit = pruned["selection"]["fit"]
            selected_request = (
                pruned["choice"]["request"]
                if (pruned["selection"]["selected"] == "pruned")
                else row["request"]
            )
        return sealed_write(
            directory / "result.json",
            {
                "identity": plan["artifact_sha256"],
                "task": row["task"],
                "status": "complete" if parent is not None else "no_finite_model",
                "seed": seed,
                "refit": fitted,
                "refit_health": _fit_health(directory / "refit"),
                "pre_pruning_origin": origin,
                "pre_pruning_fit": parent,
                "pruning": pruned,
                "pruning_fit_health": {
                    arm: _fit_health(directory / "pruning" / arm)
                    for arm in ("control", "pruned")
                }
                if pruned
                else None,
                "selected_request": selected_request,
                "selected_fit": selected_fit,
                "public_certificate": pruning.certificate(
                    PublicFitRequest.model_validate(selected_request), cell, row["task"]
                )
                if selected_request
                else None,
                "scientific_correctness_certified": False,
                "intervention_performance": "not_evaluated",
                "test_data_opened": False,
                "live_llm_calls": 0,
            },
        )


def _report(root: Path) -> dict:
    """Always write a readable progress report, including preparation failures."""
    if not (root / "plan.json").exists():
        result = {
            "status": "not_prepared",
            "rows": [],
            "message": "Inspect prepare job logs.",
        }
        root.mkdir(parents=True, exist_ok=True)
        public._write(root / "summary.json", result)
        return result
    plan = verify(root)
    rows, models = [], []
    for row in plan["rows"]:
        directory = root / "results" / row["task"]["task_id"]
        value = (
            checked(directory, plan, row)
            if (directory / "result.json").exists()
            else None
        )
        fit = value["selected_fit"] if value else None
        rows.append(
            {
                "task": row["task"],
                "status": value["status"] if value else "missing",
                "training_nmse": fit["training"]["normalized_mse"] if fit else None,
                "validation_nmse": fit["validation"]["normalized_mse"] if fit else None,
                "pre_pruning_origin": value["pre_pruning_origin"] if value else None,
                "pruning_selected": value["pruning"]["selection"]["selected"]
                if value and value["pruning"]
                else None,
                "refit_health": value["refit_health"] if value else None,
                "seed_status": value["seed"]["status"] if value else None,
                "refit_status": value["refit"]["status"] if value else None,
                "refit_budget_exhausted": value["refit"]["budget_exhausted"]
                if value
                else None,
                "pruning_fit_health": value["pruning_fit_health"] if value else None,
            }
        )
        if value:
            models.append({"task": row["task"], "original_seed": row, "result": value})
    public._write(
        root / "models.json",
        {
            "protocol": PROTOCOL,
            "plan_sha256": plan["artifact_sha256"],
            "cells": plan["cells"],
            "models": models,
            "test_data_opened": False,
            "interventions_evaluated": False,
        },
    )
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": plan["artifact_sha256"],
        "status": "complete" if len(models) == len(rows) else "partial",
        "expected": len(rows),
        "recorded": len(models),
        "rows": rows,
        "live_llm_calls": 0,
        "test_data_opened": False,
        "limitation": (
            "Exploratory rescue of selected historical structures. Public graph "
            "checks are not fitted mechanism or intervention certification. "
            "Different benchmark variants remain separate."
        ),
    }
    public._write(root / "summary.json", result)
    return result


def report(root: Path) -> dict:
    """Serialize manual and scheduled report writers without locking workers."""
    with public._lock(root / "reporting"):
        return _report(root)
