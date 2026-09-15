"""Verified saved requirement repair to the frozen public numerical interface."""

from __future__ import annotations

import fcntl
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import Field

from autoformalism.expressions import ModelValidationError
from autoformalism.fitting.public_fitting import (
    content_sha256,
    execute_fit,
    inspect_fit,
    prepare_fit,
)
from autoformalism.llm.staged_topology import visible_response
from autoformalism.rebuttal import prefit_construction_campaign as construction
from autoformalism.rebuttal import prefit_requirements as requirements
from autoformalism.rebuttal.prefit_construction_audit import (
    _inputs,
    read_audit,
    reconstruct,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import FitProfile, PublicFitRequest, Sha256
from autoformalism.search.repair_provenance import (
    STAGE_AWARE,
    attempt_feedback,
    repair_provenance,
    with_repair_provenance,
)
from autoformalism.search.requirement_feedback import (
    apply_requirement_repair,
    diagnose_requirements,
)
from autoformalism.staged_topology import content_hash

PROTOCOL = "prefit-public-fit-handoff-1"
CELL = "phase_b_anonymous_system_task_canonical_opaque_hard"


class HandoffSelection(StrictSchema):
    """Declare the source before observing any numerical result."""

    protocol: Literal["prefit-public-fit-handoff-1"] = PROTOCOL
    source_plan_sha256: Sha256
    construction_plan_sha256: Sha256
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    cell: Literal["phase_b_anonymous_system_task_canonical_opaque_hard"] = CELL
    profile: FitProfile = "collocation-feasible-v1"
    random_seed: int = Field(default=20260913, ge=0, le=2**32 - 1)


@contextmanager
def _lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".handoff.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("handoff directory is in use") from error
        yield


def _disjoint(root: Path, *sources: Path) -> None:
    for source in sources:
        a, b = root.resolve(), source.resolve()
        if a.is_relative_to(b) or b.is_relative_to(a):
            raise ValueError("handoff destination must be separate from all sources")


def _unique(rows: list[dict], field: str, value: str) -> dict:
    matching = [row for row in rows if row[field] == value]
    if len(matching) != 1:
        raise ValueError(f"expected exactly one declared {field}: {value}")
    return matching[0]


def _construction_bundle(root: Path, plan: dict, bundle: dict) -> dict:
    """Rebuild the sealed construction without executing its historical runtime."""
    task = _unique(plan["tasks"], "task_id", bundle["source_task"]["task_id"])
    if (
        task != bundle["source_task"]
        or "/" in task["task_id"]
        or ".." in task["task_id"]
    ):
        raise ValueError("source construction task differs")
    saved, binding, _ = _inputs(root, plan, task)
    audit = read_audit(root / "results" / task["task_id"], binding)
    if saved is None or audit is None or audit["status"] != "passed":
        raise ValueError("construction needs a sealed passing audit")
    cell = plan["cells"][task["cell"]]
    rebuilt = reconstruct(cell, saved)
    if not rebuilt["certificate"]["passed"]:
        raise ValueError("construction revalidation failed")
    for key in ("certificate", "function_slots", "handoff", "scientific_review_facts"):
        if audit[key] != rebuilt[key]:
            raise ValueError("construction differs from reconstructed audit")
    expected = {
        "source_task": task,
        "source_audit_sha256": audit["artifact_sha256"],
        "source_construction_sha256": saved["artifact_sha256"],
        "brief": cell["brief"],
        "context": cell["context"],
        "topology": {
            key: saved["topology"][key]
            for key in ("inventory", "equations", "topology")
        },
        "candidate": rebuilt["handoff"]["candidate"],
        "initialization": rebuilt["handoff"]["initialization"],
        "slots": rebuilt["function_slots"],
        "scientific_review_facts": rebuilt["scientific_review_facts"],
    }
    if bundle != expected:
        raise ValueError("requirement parent differs from original construction")
    return cell


def verify_source(
    source: Path, construction_root: Path, selection: HandoffSelection
) -> dict:
    """Replay saved replies and audit exact acceptance; no calls, data or fitting."""
    plan = sealed_read(source / "plan.json")
    parent = sealed_read(construction_root / "plan.json")
    if (
        plan["protocol"] != requirements.PROTOCOL
        or parent["protocol"] != construction.CONSTRUCTION_ONLY_PROTOCOL
        or plan["artifact_sha256"] != selection.source_plan_sha256
        or parent["artifact_sha256"] != selection.construction_plan_sha256
        or plan["source_plan_sha256"] != parent["artifact_sha256"]
    ):
        raise ValueError("source plans differ from the declared selection")
    config = requirements.RequirementConfig.model_validate(plan["config"])
    task = _unique(plan["tasks"], "task_id", selection.task_id)
    case = _unique(plan["cases"], "case_id", task["case_id"])
    bundle = case["bundle"]
    if content_hash(bundle) != case["case_id"]:
        raise ValueError("source case digest differs")
    if bundle["source_task"]["cell"] != selection.cell:
        raise ValueError("selected source cell differs")
    if task["arm"] != "requirement_feedback" or task["cohort"] != "repair":
        raise ValueError(
            "this milestone requires an explicitly selected repaired model"
        )
    cell = _construction_bundle(construction_root, parent, bundle)
    baseline = diagnose_requirements(bundle, case["bindings"])
    if baseline != case["baseline"] or not baseline["requirement_gap"]:
        raise ValueError("source baseline diagnosis differs")
    path = source / "results" / selection.task_id / "state.json"
    sealed = sealed_read(path)
    state, records = requirements._state(source, plan, task)
    if (
        state["status"] != "complete"
        or state["stop_reason"] != "requirement_syntax_repaired"
    ):
        raise ValueError("source repair is not successfully complete")
    if (
        not state["attempts"]
        or len(state["attempts"]) > config.model_settings.attempts_per_step
    ):
        raise ValueError("invalid source attempt count")
    by_key = {record["request_hash"]: record for record in records}
    seen, evaluations, accepted = set(), [], None
    for index, event in enumerate(state["attempts"]):
        if (
            event["attempt"] != index
            or event["request_hash"] in seen
            or accepted is not None
        ):
            raise ValueError("invalid source attempt order")
        seen.add(event["request_hash"])
        record = by_key[event["request_hash"]]
        raw, result, feedback = None, None, None
        try:
            raw = visible_response(record)
            result = apply_requirement_repair(
                bundle,
                case["bindings"],
                raw,
                repair_policy=config.repair_policy,
            )
            if config.feedback_policy == STAGE_AWARE:
                result = with_repair_provenance(bundle, result)
        except (ValueError, TypeError, KeyError, ModelValidationError) as error:
            feedback = attempt_feedback(bundle, record, raw, error)
        if event["response"] != raw or event["accepted"] != (result is not None):
            raise ValueError("saved response or acceptance differs from replay")
        evaluations.append(
            {
                "attempt": index,
                "request_hash": event["request_hash"],
                "accepted": result is not None,
                "current_diagnosis": feedback,
                "historical_feedback": event["feedback"],
            }
        )
        if result is not None:
            accepted = result
    if seen != set(by_key):
        raise ValueError("unaccounted provider calls in selected episode")
    if (
        accepted is None
        or accepted != state["final"]
        or accepted["diagnosis"]["requirement_gap"]
    ):
        raise ValueError("saved accepted repair differs from revalidation")
    return {
        "task": task,
        "bundle": bundle,
        "cell": cell,
        "final": accepted,
        "source_artifact_sha256": sealed["artifact_sha256"],
        "attempt_evidence": evaluations,
        "repair_provenance": repair_provenance(bundle, accepted),
    }


def prepare_handoff(
    source: Path,
    construction_root: Path,
    public_root: Path,
    selection: HandoffSelection,
    root: Path,
) -> dict:
    """Freeze a verified repaired candidate and matching public development data."""
    _disjoint(root, source, construction_root, public_root)
    checked = verify_source(source, construction_root, selection)
    cell = checked["cell"]
    directory = public_root / "phase_b_v1" / selection.cell
    hashes = {}
    for name in ("manifest.json", "proposer_prompt.txt", "train.csv", "validation.csv"):
        hashes[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if name in cell["assets"] and hashes[name] != cell["assets"][name]:
            raise ValueError(f"public asset differs from source construction: {name}")
    dataset = construction.load_development(public_root, selection.cell)
    final = checked["final"]
    request = PublicFitRequest(
        base_candidate=final["initialization"]["base_candidate"],
        context=checked["bundle"]["context"],
        initialization_plan=final["initialization"]["plan"],
        parameter_guesses={},
        profile=selection.profile,
        random_seed=selection.random_seed,
        source={
            "stage": "requirement_repair",
            "task_id": selection.task_id,
            "artifact_sha256": checked["source_artifact_sha256"],
        },
    )
    payload = {
        "protocol": PROTOCOL,
        "selection": selection.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "accepted_candidate": final["candidate"],
        "accepted_candidate_sha256": content_sha256(final["candidate"]),
        "public_assets": hashes,
        "public_brief": checked["bundle"]["brief"],
        "source_task": checked["bundle"]["source_task"],
        "requirement_diagnosis": final["diagnosis"],
        "repair_provenance": checked["repair_provenance"],
        "attempt_evidence": checked["attempt_evidence"],
        "source_runtime_policy": (
            "verify sealed historical evidence; revalidate under new runtime"
        ),
        "live_llm_calls": 0,
        "test_data_opened": False,
        "private_reference_opened": False,
    }
    with _lock(root):
        existing = root / "handoff.json"
        if existing.exists() and sealed_read(existing) != {
            **payload,
            "artifact_sha256": content_hash(payload),
        }:
            raise ValueError("existing handoff differs")
        prepare_fit(request, dataset.train, dataset.validation, root / "fit")
        frozen = json.loads((root / "fit/freeze.json").read_text())
        if frozen["lowered_candidate"] != final["candidate"]:
            raise ValueError("fit lowering differs from accepted repaired candidate")
        sealed_write(existing, payload)
    return summarize_handoff(root)


def _verified_handoff(root: Path) -> tuple[dict, dict]:
    handoff = sealed_read(root / "handoff.json")
    inspection = inspect_fit(root / "fit")
    frozen = json.loads((root / "fit/freeze.json").read_text())
    if (
        handoff["protocol"] != PROTOCOL
        or frozen["request"] != handoff["request"]
        or frozen["lowered_candidate"] != handoff["accepted_candidate"]
        or content_sha256(frozen["lowered_candidate"])
        != handoff["accepted_candidate_sha256"]
    ):
        raise ValueError("fitter handoff lineage differs")
    return handoff, inspection


def summarize_handoff(root: Path) -> dict:
    """Produce candidate-bound numerical feedback without starting an optimizer."""
    handoff, inspection = _verified_handoff(root)
    report = {
        "protocol": PROTOCOL,
        "status": "prepared",
        "handoff_sha256": handoff["artifact_sha256"],
        "source_task": handoff["source_task"],
        "selected_repair_task": handoff["selection"]["task_id"],
        "candidate_sha256": handoff["accepted_candidate_sha256"],
        "capability": inspection,
        "repair_provenance": handoff["repair_provenance"],
        "attempt_evidence": handoff["attempt_evidence"],
        "live_llm_calls": 0,
        "scientific_judge_called": False,
        "test_data_opened": False,
        "private_reference_opened": False,
        "scientific_status": "not_certified",
        "limitation": (
            "One preselected candidate and one existing fitter budget. Finite scores "
            "do not certify scientific validity; failure does not prove structural "
            "infeasibility. No post-fit proposer revision has run."
        ),
    }
    if (root / "fit/result.json").exists():
        result = execute_fit(root / "fit")  # terminal-only, validated immutable replay
        route = {
            "complete": "model_review_with_fit_evidence",
            "fit_failed": "numerical_failure_review",
            "capability_unsupported": "fitter_capability",
            "interrupted": "execution_recovery_required",
        }[result.status]
        report.update(
            status=result.status,
            fit=result.model_dump(mode="json"),
            numerical_feedback={
                "schema_version": "public-fit-feedback-1",
                "candidate_sha256": result.lowered_candidate_sha256,
                "route": route,
                "training": result.training.model_dump(mode="json"),
                "validation": result.validation.model_dump(mode="json"),
                "native_optimizer_converged": result.native_optimizer_converged,
                "budget_exhausted": result.budget_exhausted,
                "independent_replay": result.independent_replay,
                "scientific_verdict": None,
                "structural_infeasibility_proven": False,
                "instruction": (
                    "Use the observed numerical evidence to assess the current "
                    "functions. Poor fit or an unavailable rollout does not identify "
                    "a faulty term or establish instability. Preserve valid unrelated "
                    "components when proposing an evidence-supported revision."
                ),
            },
        )
    return report


def run_handoff(root: Path) -> dict:
    """Consume the declared fit attempt once, then write its reviewable report."""
    with _lock(root):
        _verified_handoff(root)
        execute_fit(root / "fit")
        report = summarize_handoff(root)
        sealed_write(root / "summary.json", report)
    return report
