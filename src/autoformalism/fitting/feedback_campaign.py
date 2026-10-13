"""M31: replay saved training fits and qualify feedback on small synthetic cases."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import feedback_fixtures as fixtures
from autoformalism.fitting import fitting_feedback as feedback
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_integration_campaign as integration
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting.fitting_feedback import Policy as FeedbackPolicy
from autoformalism.fitting.rollout_adapter import FitResult
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-fitting-feedback-qualification-1"
INPUT_PROTOCOL = "phase-c-fitting-feedback-inputs-1"
ARCHIVED = (*integration.CONSTRUCTED, *("control_" + n for n in integration.CONTROLS))


class Policy(StrictSchema):
    """Diagnostic allowance is separate from the immutable historical fitting cost."""

    protocol: Literal["phase-c-fitting-feedback-qualification-1"] = PROTOCOL
    feedback: FeedbackPolicy = Field(default_factory=FeedbackPolicy)


def training_problem(value: dict) -> checks.TrainingProblem:
    """Restore the explicit piecewise opt-in without widening the input allowlist."""
    schema = (
        checks.PiecewiseTrainingProblem
        if value.get("allow_piecewise") is True
        else checks.TrainingProblem
    )
    return schema.model_validate(value)


class ArchivedInput(StrictSchema):
    """Only training problems and their frozen optimizer results cross this boundary."""

    task_id: str
    kind: Literal["constructed", "known_structure"]
    case: str
    source_status: str
    source_identity: dict
    problem: dict | None
    fit: FitResult | None
    backend: dict | None

    @model_validator(mode="after")
    def consistent(self):
        present = (
            self.problem is not None,
            self.fit is not None,
            self.backend is not None,
        )
        if any(present) and not all(present):
            raise ValueError("archived training problem, fit and backend must be joint")
        if self.problem is not None:
            problem = training_problem(self.problem)
            if self.fit.backend_sha256 != public.content_sha256(self.backend):
                raise ValueError("archived backend identity differs")
            if self.fit.identity.get("problem_sha256") != public.content_sha256(
                problem.model_dump(mode="json")
            ):
                raise ValueError("archived training problem identity differs")
        return self


class Inputs(StrictSchema):
    """Frozen archive denominator and fixture identities; no evaluation arrays."""

    protocol: Literal["phase-c-fitting-feedback-inputs-1"] = INPUT_PROTOCOL
    source_identity: dict
    records: list[ArchivedInput]
    fixtures: list[dict]
    test_data_opened: Literal[False] = False
    validation_used: Literal[False] = False
    reference_values_used: Literal[False] = False


def export_inputs(source: Path, output: Path) -> dict:
    """Read M30's output-aware arm without fitting or copying evaluator material.

    The source may be a local review extraction or the original Delta directory.
    No source locks, reports, verification operations or output writes are made.
    """
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("feedback export must be outside the historical source")
    parent = read_seal(source / "plan.json")
    child_root = source / "arms/output_aware"
    child = read_seal(child_root / "plan.json")
    if (
        parent["protocol"] != "phase-c-poor-fit-qualification-1"
        or child["protocol"] != integration.PROTOCOL
        or parent["test_data_opened"] is not False
        or child["test_data_opened"] is not False
        or parent["children"]["output_aware"] != public.content_sha256(child)
        or tuple(t["task_id"] for t in child["tasks"]) != ARCHIVED
    ):
        raise ValueError("expected the eleven-task frozen M30 output-aware arm")
    expected = [
        t
        | {
            "task_id": "output_aware__" + t["task_id"],
            "source_task_id": t["task_id"],
            "arm": "output_aware",
            "source_index": i,
        }
        for i, t in enumerate(child["tasks"])
    ]
    if [t for t in parent["tasks"] if t["arm"] == "output_aware"] != expected:
        raise ValueError("M30 parent/child roster differs")
    identity = {
        "protocol": parent["protocol"],
        "arm": "output_aware",
        "plan_sha256": public.content_sha256(parent),
        "child_plan_sha256": public.content_sha256(child),
        "inputs_sha256": child["inputs_sha256"],
        "source_sha256": child["source_sha256"],
        "runtime": child["runtime"],
    }
    records = []
    for task in child["tasks"]:
        folder = child_root / "results" / task["task_id"]
        row = (
            read_seal(folder / "result.json")
            if (folder / "result.json").exists()
            else None
        )
        problem, fit, backend = None, None, None
        provenance = {"source_task_id": task["task_id"], **identity}
        if row is not None:
            if row["identity"] != {
                "plan_sha256": public.content_sha256(child),
                "task": task,
            }:
                raise ValueError("M30 source result identity differs")
            provenance["result_sha256"] = public.content_sha256(row)
            if row["fit"] is not None:
                outer = bridge.completed(folder / "fit")
                if outer.model_dump(mode="json") != row["fit"]:
                    raise ValueError("M30 source fitted handoff differs")
                if outer.fit is not None:
                    problem = outer.preparation["value"]["problem"]
                    fit = outer.fit
                    backend = read_seal(folder / "fit/fit/backend.json")
                    provenance["problem_sha256"] = public.content_sha256(problem)
                    provenance["fit_sha256"] = public.content_sha256(
                        fit.model_dump(mode="json")
                    )
                    provenance["backend_sha256"] = public.content_sha256(backend)
        records.append(
            ArchivedInput(
                **task,
                source_status=row["status"] if row else "source_result_missing",
                source_identity=provenance,
                problem=problem,
                fit=fit,
                backend=backend,
            )
        )
    data = Inputs(source_identity=identity, records=records, fixtures=fixtures.roster())
    payload = data.model_dump(mode="json")
    roster(payload)
    seal(output, payload)
    return {
        "inputs_sha256": public.content_sha256(payload),
        "archived": len(records),
        "fixtures": len(data.fixtures),
        "tasks": len(records) + len(data.fixtures),
    }


def roster(data: dict) -> list[dict]:
    """Retain every unavailable source and synthetic control in the denominator."""
    inputs = Inputs.model_validate(data)
    if tuple(r.task_id for r in inputs.records) != ARCHIVED:
        raise ValueError("feedback archived roster differs")
    fixture_ids = [r["case_id"] for r in inputs.fixtures]
    if len(fixture_ids) != len(set(fixture_ids)) or any(
        not name.replace("_", "").isalnum() for name in fixture_ids
    ):
        raise ValueError("feedback fixture identities differ")
    return [
        {
            "task_id": "archive__" + r.task_id,
            "source_task_id": r.task_id,
            "kind": "archived",
            "case": r.case,
        }
        for r in inputs.records
    ] + [
        {
            "task_id": "fixture__" + r["case_id"],
            "source_task_id": r["case_id"],
            "kind": "synthetic_fixture",
            "case": r["case_id"],
        }
        for r in inputs.fixtures
    ]


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    """Freeze replay inputs, fixture roster, runtime and diagnostic allowance."""
    data = read_seal(inputs)
    if data["fixtures"] != fixtures.roster():
        raise ValueError("fixture roster differs from this source version")
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(data),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Changed inputs, policies, source code and runtime cannot silently resume."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(data)
        or plan["inputs_sha256"] != public.content_sha256(data)
    ):
        raise ValueError("feedback campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or data["fixtures"] != fixtures.roster()
    ):
        raise ValueError("feedback campaign source/runtime differs")
    return plan, data


def completed(folder: Path, identity: dict) -> dict:
    """Check the sealed packet and exact diagnostic inputs before returning a row."""
    row = read_seal(folder / "result.json")
    if row["identity"] != identity:
        raise ValueError("feedback task identity differs")
    if row["feedback"] is not None:
        if row["feedback"] != read_seal(folder / "feedback.json") or row[
            "feedback"
        ] != feedback.completed(folder / "diagnostics", runtime=False).model_dump(
            mode="json"
        ):
            raise ValueError("feedback packet differs")
        feedback.FeedbackPacket.model_validate(row["feedback"])
        if row["diagnostic_inputs_sha256"] != public.content_sha256(
            read_seal(folder / "diagnostic_inputs.json")
        ):
            raise ValueError("feedback diagnostic input identity differs")
    if row["evaluator_labels"] is not None and row["evaluator_labels"] != read_seal(
        folder / "evaluator_labels.json"
    ):
        raise ValueError("feedback evaluator labels differ")
    return row


def run_task(root: Path, index: int) -> dict:
    """Replay archived fits; only synthetic fixtures may perform parameter fitting."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown feedback task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity)
        record = next(
            (r for r in data["records"] if r["task_id"] == task["source_task_id"]), None
        )
        values = (
            {k: record[k] for k in ("problem", "fit", "backend")} if record else None
        )
        if task["kind"] == "synthetic_fixture":
            values = fixtures.fit_fixture(task["source_task_id"], folder / "fixture")
        packet, labels, inputs_hash = None, None, None
        if values["fit"] is not None:
            problem = training_problem(values["problem"])
            fit = FitResult.model_validate(values["fit"])
            seal(folder / "diagnostic_inputs.json", values)
            inputs_hash = public.content_sha256(values)
            packet = feedback.run(
                problem,
                fit,
                values["backend"],
                folder / "diagnostics",
                Policy.model_validate(plan["policy"]).feedback,
            ).model_dump(mode="json")
            seal(folder / "feedback.json", packet)
            if task["kind"] == "synthetic_fixture":
                labels = fixtures.make(task["source_task_id"])[2]
                seal(folder / "evaluator_labels.json", labels)
        row = task | {
            "identity": identity,
            "status": "complete" if packet else "source_unavailable",
            "source_status": record["source_status"] if record else "synthetic_fixture",
            "source_identity": record["source_identity"] if record else None,
            "feedback": packet,
            "diagnostic_inputs_sha256": inputs_hash,
            "fitting_performed": task["kind"] == "synthetic_fixture",
            "evaluator_labels": labels,
            "original_fit_cost": values["fit"]["cost"] if values["fit"] else None,
        }
        seal(folder / "result.json", row)
        return row


def report(root: Path) -> dict:
    """Report all outcomes and preserve missing/unknown cost values as unknown."""
    plan, _ = verify(root, runtime=False)
    rows = [
        completed(
            root / "results" / task["task_id"],
            {"plan_sha256": public.content_sha256(plan), "task": task},
        )
        if (root / "results" / task["task_id"] / "result.json").exists()
        else task | {"status": "missing", "feedback": None}
        for task in plan["tasks"]
    ]
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "incomplete"
        if any(r["status"] == "missing" for r in rows)
        else "complete",
        "expected": len(rows),
        "recorded": sum(r["status"] != "missing" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "diagnosis_counts": dict(
            Counter(r["feedback"]["diagnosis"] for r in rows if r["feedback"])
        ),
        "rows": rows,
        "diagnostic_costs": {
            r["task_id"]: r["feedback"]["cost"] if r["feedback"] else None for r in rows
        },
        "diagnostic_cost_totals": _cost_totals(rows),
        "fixture_fitting_cost_totals": _fixture_cost_totals(rows),
        **_qualification(rows),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Saved M30 fits are replayed, never refitted. Synthetic fits "
        "are separately labeled. Feedback uses training only; evaluator labels stay "
        "outside packets. Unknown diagnostic cost remains unknown. Numerical diagnosis "
        "does not establish scientific validity, global optimality or "
        "noisy-data calibration.",
    }
    public._write(root / "summary.json", result)
    return result


def _cost_totals(rows: list[dict]) -> dict:
    """Sum measured work separately; missing/unknown usage never becomes zero."""
    costs = [r["feedback"]["cost"] for r in rows if r["feedback"]]
    missing = sum(r["status"] == "missing" for r in rows)
    return _sum_costs(
        costs,
        missing,
        ("wall_seconds", "cpu_seconds", "budget_charge_seconds", "calls"),
    )


def _fixture_cost_totals(rows: list[dict]) -> dict:
    """Only synthetic fits are new fitting work; archive costs remain historical."""
    fixtures = [r for r in rows if r["kind"] == "synthetic_fixture"]
    costs = [r["original_fit_cost"] for r in fixtures if r.get("original_fit_cost")]
    return _sum_costs(
        costs,
        len(fixtures) - len(costs),
        (
            "wall_seconds",
            "cpu_seconds",
            "budget_charge_seconds",
            "search_calls_started",
            "search_calls_completed",
        ),
    )


def _sum_costs(costs: list[dict], missing: int, keys: tuple[str, ...]) -> dict:
    """Retain measured subtotals while withholding totals with unknown members."""
    result = {
        "accounting_complete": not missing
        and all(
            c["accounting_complete"] and all(c[k] is not None for k in keys)
            for c in costs
        ),
        "missing_tasks": missing,
    }
    for key in keys:
        known = [c[key] for c in costs if c[key] is not None]
        unknown = missing + sum(c[key] is None for c in costs)
        result[key] = None if unknown else sum(known)
        result["known_" + key] = sum(known)
        result["unknown_" + key + "_tasks"] = unknown
    return result


def _check_status(statuses: list[str]) -> str:
    """A known failure remains visible even when other checks are still pending."""
    return (
        "failed"
        if "failed" in statuses
        else "pending"
        if "pending" in statuses
        else "passed"
    )


def _qualification(rows: list[dict]) -> dict:
    """Compare evaluator-only expectations after packets exist; never feed them back."""
    fixture_checks, boundary_checks, diagnostic_checks = [], [], []
    for row in rows:
        packet = row["feedback"]
        boundary = {
            "task_id": row["task_id"],
            "specification_not_assessed": packet["specification_verdict"]
            == "not_assessed"
            if packet
            else None,
            "no_automatic_revision": packet["automatic_specification_revision_allowed"]
            is False
            if packet
            else None,
        }
        boundary["status"] = (
            "passed"
            if all(
                boundary[k]
                for k in ("specification_not_assessed", "no_automatic_revision")
            )
            else "failed"
            if packet
            else "pending"
            if row["status"] == "missing"
            else "not_applicable"
        )
        boundary_checks.append(boundary)
        diagnostic_status = packet["diagnostics"]["status"] if packet else None
        diagnostic_checks.append(
            {
                "task_id": row["task_id"],
                "diagnostic_status": diagnostic_status,
                "status": "pending"
                if row["status"] == "missing"
                else "not_applicable"
                if packet is None
                else "passed"
                if diagnostic_status == "complete"
                or (
                    row["kind"] == "archived"
                    and diagnostic_status == "skipped"
                    and packet["diagnosis"] == "rollout_unverified"
                )
                else "failed",
            }
        )
        if row["kind"] == "synthetic_fixture":
            expected = (row.get("evaluator_labels") or {}).get("expected_diagnosis")
            actual = packet["diagnosis"] if packet else None
            status = (
                "pending"
                if row["status"] == "missing"
                else (
                    "passed"
                    if expected is not None
                    and actual == expected
                    and diagnostic_status == "complete"
                    else "failed"
                )
            )
            fixture_checks.append(
                {
                    "task_id": row["task_id"],
                    "expected_diagnosis": expected,
                    "actual_diagnosis": actual,
                    "diagnosis_matches": None
                    if status == "pending"
                    else expected is not None and actual == expected,
                    "diagnostics_complete": diagnostic_status == "complete"
                    if packet
                    else None,
                    "status": status,
                }
            )
    fixture_status = _check_status([r["status"] for r in fixture_checks])
    boundary_status = _check_status([r["status"] for r in boundary_checks])
    diagnostic_status = _check_status([r["status"] for r in diagnostic_checks])
    accounting = (
        _cost_totals(rows)["accounting_complete"]
        and _fixture_cost_totals(rows)["accounting_complete"]
    )
    all_recorded = all(r["status"] != "missing" for r in rows)
    status = _check_status(
        [
            fixture_status,
            boundary_status,
            diagnostic_status,
            "passed"
            if all_recorded and accounting
            else "failed"
            if all_recorded
            else "pending",
        ]
    )
    return {
        "fixture_checks": fixture_checks,
        "fixture_check_status": fixture_status,
        "packet_boundary_checks": boundary_checks,
        "packet_boundary_check_status": boundary_status,
        "diagnostic_checks": diagnostic_checks,
        "diagnostic_check_status": diagnostic_status,
        "qualification_status": status,
        "qualification_gates": {
            "all_outcomes_recorded": all_recorded,
            "fixture_controls_pass": fixture_status,
            "packet_boundaries_preserved": boundary_status,
            "required_diagnostics_available": diagnostic_status,
            "new_work_accounting_complete": accounting,
        },
    }
