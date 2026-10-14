"""M32: fixed-vector replay of M31 with staged, training-only diagnostics."""

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import feedback_campaign as historical
from autoformalism.fitting import feedback_fixtures as historical_fixtures
from autoformalism.fitting import fitting_feedback as legacy_feedback
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import staged_feedback as feedback
from autoformalism.fitting import staged_feedback_fixtures as fixtures
from autoformalism.fitting.rollout_adapter import FitResult
from autoformalism.fitting.staged_feedback import Policy as FeedbackPolicy
from autoformalism.schemas.base import StrictSchema

PROTOCOL = "phase-c-staged-feedback-qualification-1"
INPUT_PROTOCOL = "phase-c-staged-feedback-inputs-1"
ARCHIVED = tuple("archive__" + name for name in historical.ARCHIVED) + tuple(
    "fixture__" + item["case_id"] for item in historical_fixtures.roster()
)


class Policy(StrictSchema):
    """Keep diagnostic allowances separate from frozen historical fitting work."""

    protocol: Literal["phase-c-staged-feedback-qualification-1"] = PROTOCOL
    feedback: FeedbackPolicy = Field(default_factory=FeedbackPolicy)


class ArchivedInput(StrictSchema):
    """Frozen M31 inputs and prior numerical evidence, without evaluator labels."""

    task_id: str
    source_task_id: str
    source_kind: Literal["archived", "synthetic_fixture"]
    case: str
    source_status: Literal["complete", "source_unavailable", "source_result_missing"]
    source_identity: dict
    problem: dict | None
    fit: FitResult | None
    backend: dict | None
    prior_feedback: legacy_feedback.FeedbackPacket | None

    @model_validator(mode="after")
    def consistent(self):
        """Bind each available problem/backend/packet to its exact original fit."""
        available = (
            self.problem is not None,
            self.fit is not None,
            self.backend is not None,
        )
        if any(available) and not all(available):
            raise ValueError("archived problem, fit and backend must be joint")
        if (self.source_status == "complete") != all(available):
            raise ValueError("archived source status and diagnostic inputs differ")
        if self.problem is not None:
            problem = historical.training_problem(self.problem)
            problem_hash = public.content_sha256(problem.model_dump(mode="json"))
            if self.fit.backend_sha256 != public.content_sha256(self.backend):
                raise ValueError("archived backend identity differs")
            if self.fit.identity.get("problem_sha256") != problem_hash:
                raise ValueError("archived training problem identity differs")
            if self.prior_feedback is None or (
                self.prior_feedback.identity.problem_sha256 != problem_hash
                or self.prior_feedback.identity.fit_sha256
                != public.content_sha256(self.fit.model_dump(mode="json"))
                or self.prior_feedback.identity.backend_sha256
                != self.fit.backend_sha256
            ):
                raise ValueError("archived feedback identity differs")
        elif self.prior_feedback is not None:
            raise ValueError("feedback without archived diagnostic inputs")
        return self


class Inputs(StrictSchema):
    """Keep all sixteen historical outcomes and separately named new controls."""

    protocol: Literal["phase-c-staged-feedback-inputs-1"] = INPUT_PROTOCOL
    source_identity: dict
    records: list[ArchivedInput]
    fixtures: list[dict]
    test_data_opened: Literal[False] = False
    validation_used: Literal[False] = False
    reference_values_used: Literal[False] = False


def export_inputs(source: Path, output: Path) -> dict:
    """Read the sealed M31 archive without fitting, mutating it or copying labels."""
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("staged feedback export must be outside the historical source")
    plan, _ = historical.verify(source, runtime=False)
    if tuple(task["task_id"] for task in plan["tasks"]) != ARCHIVED:
        raise ValueError("expected the sixteen-task frozen M31 roster")
    identity = {
        "protocol": plan["protocol"],
        "plan_sha256": public.content_sha256(plan),
        "inputs_sha256": plan["inputs_sha256"],
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
    }
    records = []
    for task in plan["tasks"]:
        folder = source / "results" / task["task_id"]
        row = (
            historical.completed(
                folder, {"plan_sha256": identity["plan_sha256"], "task": task}
            )
            if (folder / "result.json").exists()
            else None
        )
        values = dict.fromkeys(("problem", "fit", "backend"))
        provenance = {"source_task_id": task["task_id"], **identity}
        if row is not None:
            provenance["result_sha256"] = public.content_sha256(row)
            if row["feedback"] is not None:
                values = read_seal(folder / "diagnostic_inputs.json")
                if set(values) != {"problem", "fit", "backend"}:
                    raise ValueError(
                        "historical diagnostic inputs contain extra fields"
                    )
                if row["original_fit_cost"] != values["fit"]["cost"]:
                    raise ValueError("historical fitting cost differs")
                provenance["diagnostic_inputs_sha256"] = public.content_sha256(values)
        records.append(
            ArchivedInput(
                task_id=task["task_id"],
                source_task_id=task["source_task_id"],
                source_kind=task["kind"],
                case=task["case"],
                source_status=row["status"] if row else "source_result_missing",
                source_identity=provenance,
                prior_feedback=row["feedback"] if row else None,
                **values,
            )
        )
    data = Inputs(source_identity=identity, records=records, fixtures=fixtures.roster())
    payload = data.model_dump(mode="json")
    tasks = roster(payload)
    seal(output, payload)
    return {
        "inputs_sha256": public.content_sha256(payload),
        "archived": len(records),
        "frozen_synthetic_fits": sum(
            r.source_kind == "synthetic_fixture" for r in records
        ),
        "new_fixtures": len(data.fixtures),
        "tasks": len(tasks),
    }


def roster(data: dict) -> list[dict]:
    """Retain unavailable and skipped M31 entries without changing their identities."""
    inputs = Inputs.model_validate(data)
    if tuple(record.task_id for record in inputs.records) != ARCHIVED:
        raise ValueError("staged feedback archived roster differs")
    for record in inputs.records:
        prefix, source_id = record.task_id.split("__", 1)
        expected_kind = "synthetic_fixture" if prefix == "fixture" else "archived"
        if record.source_task_id != source_id or record.source_kind != expected_kind:
            raise ValueError("staged feedback archived task identity differs")
    if any(set(item) != {"case_id", "title"} for item in inputs.fixtures):
        raise ValueError("fixture roster contains undeclared fields")
    fixture_ids = [item["case_id"] for item in inputs.fixtures]
    if (
        len(fixture_ids) != len(set(fixture_ids))
        or any(not name.replace("_", "").isalnum() for name in fixture_ids)
        or any("fixture__" + name in ARCHIVED for name in fixture_ids)
    ):
        raise ValueError("staged feedback fixture identities differ")
    return [
        {
            "task_id": record.task_id,
            "source_task_id": record.source_task_id,
            "kind": "archived_fixture"
            if record.source_kind == "synthetic_fixture"
            else "archived",
            "case": record.case,
        }
        for record in inputs.records
    ] + [
        {
            "task_id": "fixture__" + name,
            "source_task_id": name,
            "kind": "synthetic_fixture",
            "case": name,
        }
        for name in fixture_ids
    ]


def prepare(root: Path, inputs: Path, policy: Policy) -> dict:
    """Freeze exact training inputs, policy, runtime and historical denominator."""
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
    """Reject changed inputs, policies, runtime or code on execution resume."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(data)
        or plan["inputs_sha256"] != public.content_sha256(data)
    ):
        raise ValueError("staged feedback campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
        or data["fixtures"] != fixtures.roster()
    ):
        raise ValueError("staged feedback campaign source/runtime differs")
    return plan, data


def completed(folder: Path, identity: dict, record: dict | None = None) -> dict:
    """Verify sealed packets, inputs, labels and stage artifacts before reuse."""
    row = read_seal(folder / "result.json")
    if row["identity"] != identity:
        raise ValueError("staged feedback task identity differs")
    if any(row.get(key) != value for key, value in identity["task"].items()):
        raise ValueError("staged feedback task fields differ")
    if (row["status"] == "complete") != (row["feedback"] is not None):
        raise ValueError("staged feedback task status differs")
    if record is not None and (row["feedback"] is not None) != (
        record["fit"] is not None
    ):
        raise ValueError("staged feedback source availability differs")
    if record is not None and (
        row["prior_feedback"] != record["prior_feedback"]
        or row["source_identity"] != record["source_identity"]
        or row["source_status"] != record["source_status"]
        or row["historical_fit_cost"] != (record["fit"] or {}).get("cost")
        or row["new_fixture_fit_cost"] is not None
        or row["fitting_performed"] is not False
    ):
        raise ValueError("staged feedback historical evidence differs")
    if row["feedback"] is not None:
        values = read_seal(folder / "diagnostic_inputs.json")
        if row["diagnostic_inputs_sha256"] != public.content_sha256(values):
            raise ValueError("staged feedback diagnostic input identity differs")
        if record is not None and values != {
            key: record[key] for key in ("problem", "fit", "backend")
        }:
            raise ValueError("staged feedback archived input differs")
        if (
            set(values) != {"problem", "fit", "backend"}
            or row["feedback"] != read_seal(folder / "feedback.json")
            or row["feedback"]
            != feedback.completed(folder / "diagnostics", runtime=False).model_dump(
                mode="json"
            )
        ):
            raise ValueError("staged feedback packet differs")
        feedback.FeedbackPacket.model_validate(row["feedback"])
        for key in ("problem", "fit", "backend"):
            if row["feedback"]["identity"][key + "_sha256"] != public.content_sha256(
                values[key]
            ):
                raise ValueError("staged feedback fitted input differs")
    if row["evaluator_labels"] is not None and row["evaluator_labels"] != read_seal(
        folder / "evaluator_labels.json"
    ):
        raise ValueError("staged feedback evaluator labels differ")
    return row


def run_task(root: Path, index: int) -> dict:
    """Never refit an M31 model; only new synthetic controls receive bounded fits."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown staged feedback task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    record = next(
        (row for row in data["records"] if row["task_id"] == task["task_id"]), None
    )
    with public._lock(folder):
        if (folder / "result.json").exists():
            return completed(folder, identity, record)
        values = (
            {key: record[key] for key in ("problem", "fit", "backend")}
            if record
            else fixtures.fit_fixture(task["source_task_id"], folder / "fixture")
        )
        packet, labels, inputs_hash = None, None, None
        if values["fit"] is not None:
            if set(values) != {"problem", "fit", "backend"}:
                raise ValueError("diagnostic input contains evaluator or extra fields")
            problem = historical.training_problem(values["problem"])
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
            if task["kind"] in {"synthetic_fixture", "archived_fixture"}:
                module = (
                    fixtures
                    if task["kind"] == "synthetic_fixture"
                    else historical_fixtures
                )
                labels = module.make(task["source_task_id"])[2]
                seal(folder / "evaluator_labels.json", labels)
        row = task | {
            "identity": identity,
            "status": "complete" if packet else "source_unavailable",
            "source_status": record["source_status"]
            if record
            else "new_synthetic_fixture",
            "source_identity": record["source_identity"] if record else None,
            "feedback": packet,
            "prior_feedback": record["prior_feedback"] if record else None,
            "diagnostic_inputs_sha256": inputs_hash,
            "fitting_performed": record is None,
            "evaluator_labels": labels,
            "historical_fit_cost": values["fit"]["cost"]
            if record and values["fit"]
            else None,
            "new_fixture_fit_cost": values["fit"]["cost"] if record is None else None,
        }
        seal(folder / "result.json", row)
        return row


def _cost_totals(rows: list[dict]) -> dict:
    """Charge only M32 diagnostic work, never M31 fits or its timed-out replay."""
    return historical._sum_costs(
        [row["feedback"]["cost"] for row in rows if row["feedback"]],
        sum(row["status"] == "missing" for row in rows),
        (
            "wall_seconds",
            "cpu_seconds",
            "budget_charge_seconds",
            "calls",
            "calls_completed",
        ),
    )


def _fit_cost_totals(rows: list[dict], *, historical_work: bool) -> dict:
    """Keep new control fits and old fitting charges in separate cost ledgers."""
    selected = [
        row for row in rows if (row["kind"] != "synthetic_fixture") == historical_work
    ]
    field = "historical_fit_cost" if historical_work else "new_fixture_fit_cost"
    return historical._sum_costs(
        [row[field] for row in selected if row.get(field)],
        0 if historical_work else sum(row["status"] == "missing" for row in selected),
        (
            "wall_seconds",
            "cpu_seconds",
            "budget_charge_seconds",
            "search_calls_started",
            "search_calls_completed",
        ),
    )


def _comparison(row: dict) -> dict:
    """Show action, output and derivative changes without reranking prior fits."""
    prior, current = row.get("prior_feedback"), row["feedback"]
    return {
        "task_id": row["task_id"],
        "status": row["status"],
        "m31_diagnosis": prior["diagnosis"] if prior else None,
        "m32_diagnosis": current["diagnosis"] if current else None,
        "m31_action": prior["recommended_action"] if prior else None,
        "m32_action": current["recommended_action"] if current else None,
        "action_changed": prior["recommended_action"] != current["recommended_action"]
        if prior and current
        else None,
        "m31_diagnostic_status": prior["diagnostics"]["status"] if prior else None,
        "m32_output_status": current["diagnostics"]["output_status"]
        if current
        else None,
        "m31_derivative_scope": prior["diagnostics"]["derivative_scope"]
        if prior
        else None,
        "m32_derivative_scope": current["diagnostics"]["derivative_scope"]
        if current
        else None,
        "piecewise_status": (
            current["diagnostics"].get("piecewise_assessment") or {}
        ).get("status")
        if current
        else None,
        "historical_diagnostic_cost": prior["cost"] if prior else None,
        "new_diagnostic_cost": current["cost"] if current else None,
    }


def _qualification(rows: list[dict]) -> dict:
    """Qualify outputs and controls, retaining negative derivative evidence."""
    controls, boundaries, outputs, accuracy = [], [], [], []
    for row in rows:
        packet = row["feedback"]
        pending = row["status"] == "missing"
        boundary = bool(
            packet
            and packet["specification_verdict"] == "not_assessed"
            and packet["automatic_specification_revision_allowed"] is False
        )
        boundaries.append(
            {
                "task_id": row["task_id"],
                "status": "pending"
                if pending
                else "not_applicable"
                if packet is None
                else "passed"
                if boundary
                else "failed",
            }
        )
        output_status = packet["diagnostics"]["output_status"] if packet else None
        output_pass = output_status == "verified" or (
            output_status == "skipped"
            and packet["diagnosis"] == "rollout_unverified"
            and row["kind"] == "archived"
        )
        outputs.append(
            {
                "task_id": row["task_id"],
                "output_status": output_status,
                "status": "pending"
                if pending
                else "not_applicable"
                if packet is None
                else "passed"
                if output_pass
                else "failed",
            }
        )
        prior = row.get("prior_feedback")
        if prior and prior["diagnosis"].startswith("accurate_"):
            accuracy.append(
                {
                    "task_id": row["task_id"],
                    "historical_diagnosis": prior["diagnosis"],
                    "current_diagnosis": packet["diagnosis"] if packet else None,
                    "status": "pending"
                    if pending
                    else "passed"
                    if packet
                    and packet["diagnosis"].startswith("accurate_")
                    and output_status == "verified"
                    else "failed",
                }
            )
        if row["kind"] in {"synthetic_fixture", "archived_fixture"}:
            expected = row.get("evaluator_labels") or {}
            diagnosis = packet["diagnosis"] if packet else None
            derivative = (
                (packet["diagnostics"].get("piecewise_assessment") or {}).get(
                    "status", packet["diagnostics"]["derivative_scope"]
                )
                if packet
                else None
            )
            matches = (
                expected.get("expected_diagnosis") == diagnosis
                and expected.get("expected_diagnosis") is not None
            )
            support = expected.get("expected_derivative_support")
            if support is not None:
                matches = matches and derivative == support
            controls.append(
                {
                    "task_id": row["task_id"],
                    "expected_diagnosis": expected.get("expected_diagnosis"),
                    "actual_diagnosis": diagnosis,
                    "expected_derivative_support": support,
                    "actual_derivative_support": derivative,
                    "status": "pending"
                    if pending
                    else "passed"
                    if matches and output_pass
                    else "failed",
                }
            )
    recorded = all(row["status"] != "missing" for row in rows)
    accounting = (
        _cost_totals(rows)["accounting_complete"]
        and _fit_cost_totals(rows, historical_work=False)["accounting_complete"]
    )
    statuses = {
        "controls": historical._check_status([item["status"] for item in controls]),
        "boundaries": historical._check_status([item["status"] for item in boundaries]),
        "outputs": historical._check_status([item["status"] for item in outputs]),
        "accuracy": historical._check_status([item["status"] for item in accuracy]),
    }
    return {
        "qualification_status": historical._check_status(
            [
                *statuses.values(),
                "passed"
                if recorded and accounting
                else "failed"
                if recorded
                else "pending",
            ]
        ),
        "qualification_gates": {
            "all_outcomes_recorded": recorded,
            "control_behavior": statuses["controls"],
            "packet_boundaries_preserved": statuses["boundaries"],
            "required_outputs_verified": statuses["outputs"],
            "verified_accuracy_preserved": statuses["accuracy"],
            "new_work_accounting_complete": accounting,
        },
        "control_checks": controls,
        "packet_boundary_checks": boundaries,
        "output_checks": outputs,
        "accuracy_preservation_checks": accuracy,
    }


def report(root: Path) -> dict:
    """Report coverage, changed actions and separate historical/new cost ledgers."""
    plan, data = verify(root, runtime=False)
    rows = []
    records = {row["task_id"]: row for row in data["records"]}
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        if (folder / "result.json").exists():
            row = completed(
                folder,
                {"plan_sha256": public.content_sha256(plan), "task": task},
                records.get(task["task_id"]),
            )
        else:
            record = records.get(task["task_id"], {})
            row = task | {
                "status": "missing",
                "feedback": None,
                "prior_feedback": record.get("prior_feedback"),
                "historical_fit_cost": (record.get("fit") or {}).get("cost"),
            }
        rows.append(row)
    old_costs = [
        row["prior_feedback"]["cost"] for row in rows if row.get("prior_feedback")
    ]
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "incomplete"
        if any(row["status"] == "missing" for row in rows)
        else "complete",
        "expected": len(rows),
        "recorded": sum(row["status"] != "missing" for row in rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "diagnosis_counts": dict(
            Counter(row["feedback"]["diagnosis"] for row in rows if row["feedback"])
        ),
        "rows": rows,
        "comparisons": [_comparison(row) for row in rows],
        "diagnostic_cost_totals": _cost_totals(rows),
        "new_fixture_fitting_cost_totals": _fit_cost_totals(
            rows, historical_work=False
        ),
        "historical_fitting_cost_totals": _fit_cost_totals(rows, historical_work=True),
        "historical_diagnostic_cost_totals": historical._sum_costs(
            old_costs,
            0,
            ("wall_seconds", "cpu_seconds", "budget_charge_seconds", "calls"),
        ),
        **_qualification(rows),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "All sixteen M31 outcomes remain frozen, including five "
        "synthetic fits. Only new controls are fitted. New diagnostics keep retained "
        "parameters fixed and use training only. Historical costs are excluded from "
        "new work. Piecewise support is local and may remain switching, weak or "
        "unavailable; neither qualification nor accurate prediction proves scientific "
        "validity or global identifiability.",
    }
    public._write(root / "summary.json", result)
    return result
