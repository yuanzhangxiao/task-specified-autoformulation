"""M42: separately budgeted boundary-search diagnostics for sealed M41 fits."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import nonlinear_practical_campaign as previous
from autoformalism.fitting import nonlinear_practical_profile as profiles
from autoformalism.fitting import practical_uncertainty_supplement as snapshot
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-nonlinear-practical-supplement-1"


class Policy(StrictSchema):
    """Freeze the distinct diagnostic allowance; do not alter historical policy."""

    protocol: Literal["nonlinear-practical-supplement-policy-1"] = (
        "nonlinear-practical-supplement-policy-1"
    )
    selection: Literal["incomplete_profiles"] = "incomplete_profiles"
    diagnostic: profiles.Policy = Field(
        default_factory=lambda: profiles.Policy(reference_boundary_cells=True)
    )

    @model_validator(mode="after")
    def boundary_search_required(self) -> Policy:
        """A supplement must opt into the corrected reference search explicitly."""
        if not self.diagnostic.reference_boundary_cells:
            raise ValueError("supplement requires boundary-cell reference refinement")
        return self


def _source_rows(source: Path) -> tuple[dict, dict, list[dict]]:
    """Validate historical stages without fitting, evaluation or source writes."""
    plan, data = previous.verify(source, runtime=False)
    rows = [previous.report_row(source, plan, data, task) for task in plan["tasks"]]
    for row in rows:
        backend = row.get("backend", {})
        parameters = backend.get("parameters")
        profile = row.get("profile", {})
        qualification = profile.get("diagnostic", {}).get("endpoint_qualification")
        if (
            row["status"] != "complete"
            or backend.get("status") != "complete"
            or backend.get("verification", {}).get("passed") is not True
            or not isinstance(parameters, dict)
            or set(parameters) != set(previous.fixtures.PARAMETERS)
            or any(
                isinstance(value, bool) or not math.isfinite(value)
                for value in parameters.values()
            )
            or type(backend.get("objective_value")) not in {float, int}
            or not math.isfinite(backend["objective_value"])
            or profile.get("status") not in {"partial", "complete"}
            or not qualification
            or qualification.get("numerically_qualified") is not True
            or qualification.get("retained_fit_inadequate") is not False
        ):
            raise ValueError("supplement requires every source endpoint to qualify")
    return plan, data, rows


def audit(source: Path) -> dict:
    """Select by diagnostic availability only, never by evaluator-only errors."""
    plan, _, rows = _source_rows(source)
    records = [
        {
            "task_id": row["task_id"],
            "backend_sha256": public.content_sha256(row["backend"]),
            "source_profile_sha256": public.content_sha256(row["profile"]),
            "source_profile_status": row["profile"]["status"],
        }
        for row in rows
    ]
    return {
        "status": "passed",
        "source_plan_sha256": public.content_sha256(plan),
        "expected": len(rows),
        "verified_fits": len(rows),
        "profile_status_counts": dict(
            Counter(record["source_profile_status"] for record in records)
        ),
        "selected_task_ids": [
            record["task_id"]
            for record in records
            if record["source_profile_status"] == "partial"
        ],
        "rows": records,
        "selection_uses_truth": False,
        "fitting_performed": False,
        "numerical_calls": 0,
    }


def prepare(root: Path, source: Path, policy: Policy | None = None) -> dict:
    """Freeze original bytes and a new diagnostic identity in a separate root."""
    policy = Policy() if policy is None else policy
    root, source = root.resolve(), source.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise ValueError("supplement and source must use separate directories")
    manifest = snapshot._files(source)
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, frozen = verify(root)
            if frozen != manifest or plan["policy"] != policy.model_dump(mode="json"):
                raise ValueError("supplement source/policy differs")
        else:
            source_audit = audit(source)
            source_plan = read_seal(source / "plan.json")
            snapshot._copy_snapshot(source, root / "source_snapshot", manifest)
            selected = set(source_audit["selected_task_ids"])
            plan = {
                "protocol": PROTOCOL,
                "policy": policy.model_dump(mode="json"),
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "inputs_sha256": public.content_sha256(manifest),
                "audit_sha256": public.content_sha256(source_audit),
                "source_plan_sha256": public.content_sha256(source_plan),
                "tasks": [
                    task for task in source_plan["tasks"] if task["task_id"] in selected
                ],
                "test_data_opened": False,
                "live_llm_calls": 0,
                "fitting_performed": False,
                "truth_used_for_selection": False,
            }
            seal(root / "inputs.json", manifest)
            seal(root / "portable_audit.json", source_audit)
            seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Check the portable snapshot and bind execution to the new source/runtime."""
    plan, manifest = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    frozen = read_seal(root / "portable_audit.json")
    source_plan = read_seal(root / "source_snapshot/plan.json")
    selected = set(frozen["selected_task_ids"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(manifest)
        or snapshot._files(root / "source_snapshot") != manifest
        or plan["audit_sha256"] != public.content_sha256(frozen)
        or plan["source_plan_sha256"] != public.content_sha256(source_plan)
        or plan["source_plan_sha256"] != frozen["source_plan_sha256"]
        or plan["tasks"]
        != [task for task in source_plan["tasks"] if task["task_id"] in selected]
        or frozen["status"] != "passed"
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["fitting_performed"] is not False
        or plan["truth_used_for_selection"] is not False
    ):
        raise ValueError("supplement snapshot/identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("supplement source/runtime differs")
    return plan, manifest


def _identity(plan: dict, task: dict, row: dict) -> dict:
    """Bind the full new allowance to the original fit and partial diagnosis."""
    return {
        "plan_sha256": public.content_sha256(plan),
        "task": task,
        "backend_sha256": public.content_sha256(row["backend"]),
        "source_profile_sha256": public.content_sha256(row["profile"]),
        "definition": {
            "protocol": profiles.PROTOCOL,
            "policy": plan["policy"]["diagnostic"],
        },
    }


def _completed(folder: Path, identity: dict) -> dict:
    """Verify a terminal receipt against the separate profile and its journal."""
    result = read_seal(folder / "result.json")
    profile = previous._read_profile(folder, identity)
    if (
        profile is None
        or profile["status"] == "unfinished"
        or result
        != {
            "identity": identity,
            "status": profile["status"],
            "profile_sha256": public.content_sha256(profile),
        }
    ):
        raise ValueError("supplement terminal identity differs")
    return profile


def run_task(root: Path, index: int) -> dict:
    """Re-diagnose one saved endpoint with no fitter or evaluator calls."""
    plan, _ = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown nonlinear practical supplement task index")
    source = root / "source_snapshot"
    source_plan, data = previous.verify(source, runtime=False)
    task = plan["tasks"][index]
    row = previous.report_row(source, source_plan, data, task)
    identity = _identity(plan, task, row)
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            profile = _completed(folder, identity)
        else:
            profile = previous._profile_once(
                folder,
                identity,
                previous.training_problem(data, task),
                row["backend"]["parameters"],
                Policy.model_validate(plan["policy"]).diagnostic,
            )
            seal(
                folder / "result.json",
                {
                    "identity": identity,
                    "status": profile["status"],
                    "profile_sha256": public.content_sha256(profile),
                },
            )
        return task | {"status": profile["status"], "profile": profile}


def report(root: Path) -> dict:
    """Report original and supplemental evidence with additive, separate costs."""
    plan, _ = verify(root, runtime=False)
    source = root / "source_snapshot"
    if audit(source) != read_seal(root / "portable_audit.json"):
        raise ValueError("supplement portable audit differs")
    _, _, source_rows = _source_rows(source)
    selected = {task["task_id"]: task for task in plan["tasks"]}
    effective_rows, compact, extension_rows = [], [], []
    for row in source_rows:
        extension = None
        if row["task_id"] in selected:
            identity = _identity(plan, selected[row["task_id"]], row)
            folder = root / "results" / row["task_id"]
            extension = (
                _completed(folder, identity)
                if (folder / "result.json").exists()
                else previous._read_profile(folder, identity)
            )
            extension_rows.append({"profile": extension or {"status": "missing"}})
        # New adverse evidence must remain visible; never substitute a favorable
        # historical diagnostic for an unqualified or failed extension.
        effective = row if extension is None else row | {"profile": extension}
        effective_rows.append(effective)
        compact.append(
            previous._compact(effective)
            | {
                "source_profile_status": row["profile"]["status"],
                "extension_profile_status": extension["status"] if extension else None,
                "fit_preserved": True,
            }
        )
    counts = dict(Counter(row["profile"]["status"] for row in extension_rows))
    recorded = sum(
        row["profile"]["status"] not in {"missing", "unfinished"}
        for row in extension_rows
    )
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(selected) else "incomplete",
        "expected": len(selected),
        "recorded": recorded,
        "status_counts": counts,
        "source_expected": len(source_rows),
        "source_verified_fits": len(source_rows),
        "source_profile_status_counts": dict(
            Counter(row["profile"]["status"] for row in source_rows)
        ),
        "combined_profile_status_counts": dict(
            Counter(row["profile"]["status"] for row in effective_rows)
        ),
        "rows": compact,
        "cost": {
            "source_fit_unchanged": previous._cost_summary(source_rows),
            "source_diagnostic": previous._diagnostic_cost(source_rows),
            "extension_diagnostic": previous._diagnostic_cost(extension_rows),
            "combined_diagnostic": previous._diagnostic_cost(
                source_rows + extension_rows
            ),
            "source_evaluation_unchanged": previous.stages._stage_cost(
                source_rows, "evaluation"
            ),
        },
        "fitting_performed": False,
        "test_data_opened": False,
        "truth_used_for_selection": False,
        "live_llm_calls": 0,
        "confidence_probability": None,
        "data_fault_attribution": None,
        "limitation": (
            "Only incomplete profiles receive a separately budgeted full diagnosis. "
            "Original fits, profiles and costs are immutable. Different reference "
            "searches share analytic responses and affine solves; agreement is not "
            "global proof. Supports remain sampled, not continuous intervals or "
            "calibrated confidence. Unknown costs remain unavailable; nested timing "
            "is not additive. No automatic diagnostic retries or model promotion."
        ),
    }
    with public._lock(root / "report-lock"):
        public._write(root / "summary.json", summary)
    return summary
