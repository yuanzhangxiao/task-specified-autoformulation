"""Fresh shared-process search followed by the existing bounded pruning protocol."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import process_pruning as pruning
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_reporting as reporting
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search.review_integrity import PARAMETER_POLICY, selection_policy


def policy() -> dict:
    """Freeze composition explicitly without changing any historical experiment."""
    return {
        "construction": "general-shared-construction-1",
        "gain_assembly": "declared_or_independent",
        "public_target_contract_before_construction": True,
        "revision": "response-oriented-revision-1",
        "shared_revision": "general-shared-revision-1",
        "parameter_roles": PARAMETER_POLICY,
        "selection": selection_policy(),
        "fit_profile": "collocation-multi-target-v1",
        "failed_revision": "retain incumbent; same visit fit allowance",
        "pruning": {
            **pruning.POLICY,
            "fit_profile": "collocation-multi-target-v1",
            "placement": "once after final search round",
            "no_finite_parent": "record skipped; no fit",
        },
        "scientific_critic": "off",
        "automatic_test_access": False,
        "fresh_construction": True,
        "imported_models": 0,
    }


def final_row(root: Path, plan: dict, task: dict) -> dict | None:
    """Join the retained parent to its actual fit receipt before pruning."""
    last = plan["config"]["rounds"] - 1
    result = io.read_round(root, task, last)
    if result is None:
        return None
    if result["task"] != task or result["round"] != last:
        raise ValueError("final search result identity differs")
    parent = result["selected"]
    row = {
        "task": task,
        "parent": parent,
        "source_result_sha256": result["artifact_sha256"],
    }
    if not parent or pruning.score(parent["fit"]) is None:
        return row
    origin = parent["origin_round"]
    if parent["origin_task"] != task["task_id"] or origin not in range(last + 1):
        raise ValueError("parent fit origin differs")
    receipt = public._read(io.round_path(root, task, origin) / "fit/result.json")
    if (
        public.content_sha256(receipt) != parent["fit_result_sha256"]
        or receipt["result"] != parent["fit"]
        or receipt["sha256"] != public.content_sha256(parent["fit"])
    ):
        raise ValueError("retained parent fit receipt differs")
    request = PublicFitRequest.model_validate(parent["request"])
    model, _, _ = public._lower(request)
    cell = plan["cells"][task["cell"]]
    fit = parent["fit"]
    if (
        request.profile != policy()["fit_profile"]
        or fit["request_sha256"] != public.content_sha256(parent["request"])
        or fit["lowered_candidate_sha256"]
        != public.content_sha256(parent["bundle"]["candidate"])
        or model.validated.candidate.model_dump(mode="json")
        != parent["bundle"]["candidate"]
        or fit["training_content_sha256"] != public.content_sha256(cell["training"])
        or fit["validation_content_sha256"] != public.content_sha256(cell["validation"])
    ):
        raise ValueError("retained parent/data identity differs")
    checks = pruning.certificate(request, cell, task)
    if not checks["eligible_for_development_selection"]:
        raise ValueError("retained parent fails public requirements")
    return row


def _binding(plan: dict, row: dict) -> dict:
    return {
        "identity": plan["artifact_sha256"],
        "task": row["task"],
        "source_result_sha256": row["source_result_sha256"],
        "parent_sha256": public.content_sha256(row["parent"]),
    }


def _checked(directory: Path, plan: dict, row: dict) -> dict:
    binding = sealed_read(directory / "parent.json")
    if {k: v for k, v in binding.items() if k != "artifact_sha256"} != _binding(
        plan, row
    ):
        raise ValueError("pruning parent identity differs")
    if not row["parent"] or pruning.score(row["parent"]["fit"]) is None:
        result = sealed_read(directory / "result.json")
        if result != _skipped(plan, row, seal=True):
            raise ValueError("skipped pruning receipt differs")
        return result
    return pruning.checked_result(directory, plan, row)


def _skipped(plan: dict, row: dict, *, seal: bool = False) -> dict:
    value = {
        "identity": plan["artifact_sha256"],
        "task": row["task"],
        "source_result_sha256": row["source_result_sha256"],
        "status": "skipped_no_finite_parent",
        "test_data_opened": False,
        "llm_calls": 0,
    }
    if seal:
        from autoformalism.staged_topology import content_hash

        value["artifact_sha256"] = content_hash(value)
    return value


def run_one(root: Path, index: int) -> dict:
    """Use Astra's exact training ranking, atomic deletion and paired refit logic."""
    io.require_open(root)
    plan = io.verify(root)
    if plan["protocol"] != io.FRESH_PROTOCOL or index not in range(len(plan["tasks"])):
        raise ValueError("requires a frozen fresh shared task")
    row = final_row(root, plan, plan["tasks"][index])
    if row is None:
        raise ValueError("final search round missing; pruning cannot start")
    directory = root / "pruning" / row["task"]["task_id"]
    sealed_write(directory / "parent.json", _binding(plan, row))
    if not row["parent"] or pruning.score(row["parent"]["fit"]) is None:
        with public._lock(directory):
            return sealed_write(directory / "result.json", _skipped(plan, row))
    return pruning.execute_row(directory, plan, row, index)


def report(root: Path) -> dict:
    """Report incomplete work, final retained models and original search history."""
    plan = io.verify(root)
    if plan["protocol"] != io.FRESH_PROTOCOL:
        raise ValueError("requires a fresh shared campaign")
    search = reporting.report(root)
    rows = []
    for task in plan["tasks"]:
        row = final_row(root, plan, task)
        directory = root / "pruning" / task["task_id"]
        result = (
            _checked(directory, plan, row)
            if row and (directory / "result.json").exists()
            else None
        )
        selection = (result or {}).get("selection")
        parent = (row or {}).get("parent")
        request = parent["request"] if parent else None
        fit = parent["fit"] if parent else None
        if selection:
            fit = selection["fit"]
            if selection["selected"] == "pruned":
                request = result["choice"]["request"]
        rows.append(
            {
                "task": task,
                "status": result["status"] if result else "missing",
                "pruning_choice": result.get("choice", {}).get("status")
                if result
                else None,
                "selected": selection["selected"]
                if selection
                else "search_parent"
                if parent
                else None,
                "search_parent_validation_nmse": pruning.score(parent["fit"])
                if parent
                else None,
                "final_validation_nmse": pruning.score(fit),
                "final_request": request,
                "final_fit": fit,
            }
        )
    value = {
        "protocol": io.FRESH_PROTOCOL,
        "identity": plan["artifact_sha256"],
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        else "partial",
        "search_status_counts": search["status_counts"],
        "pruning_status_counts": dict(Counter(r["status"] for r in rows)),
        "pruning_choice_counts": dict(
            Counter(r["pruning_choice"] for r in rows if r["pruning_choice"])
        ),
        "selection_counts": dict(Counter(r["selected"] for r in rows if r["selected"])),
        "rows": rows,
        "test_data_opened": False,
        "scientific_critic_called": False,
        "limitation": (
            "Integration experiment, not a controlled old/new pipeline comparison. "
            "Public graph checks and pruning do not certify scientific correctness. "
            "Models remain development-selected."
        ),
    }
    public._write(root / "pruning_summary.json", value)
    return value
