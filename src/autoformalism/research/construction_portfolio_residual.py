"""Warm root-profile continuation and separately qualified residual-only controls."""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_continuation as coordinates
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_portfolio_nonlinear_reduction as reduced
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_profile as previous
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-residual-1"
SOURCE_PLAN = "91aca141ae159930ef5d521759806346168cdd7b7f66999e3244c98eff68fcf3"
# Filled from the authenticated compact projection of the reviewed archive.
SOURCE_RECORDS = (
    "7f658f2ee360a24b4d0864a217020a029206c27bec9bc7765f23d35f58c5081a",
    "9ac4acfed26e51c58ab490ff307c5282645fb050f0db3c9457c9cd5a8913876b",
    "c480b584bce5e204e00539d72cf03b00f737d80742c9d7cffe623744a4289299",
    "669ce45e7b84e36d62491f32cd2750e892f8edde5aba6a2e7bc24f6ad25b7839",
)
SOURCE_TRAINING = "169e2b72a1ba4145ac4ec5911875e9d7e064a525f7a1ace92cc2b151efe70795"
SOURCE_NUMERICAL = "f09211586ea929f038b2319489a72266ccd39b749a50be581de7cf6d7d280ba0"
BOUNDARY_COORDINATE = "k_Uii_term_12_0_244d72574a"


class Policy(StrictSchema):
    """Explicit new reservations; no historical budget or optimizer state reset."""

    population: Literal["scaled-readout-4", "synthetic-smoke"] = "scaled-readout-4"
    prerequisite_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    search_seconds: FiniteFloat = Field(default=1800, gt=0, le=1800)
    maximum_calls: int = Field(default=2000, gt=0, le=2000)
    verification_seconds: FiniteFloat = Field(default=300, gt=0, le=300)
    boundary_seconds: FiniteFloat = Field(default=120, gt=0, le=120)
    improvement_absolute_tolerance: FiniteFloat = Field(default=1e-10, ge=0)
    improvement_relative_tolerance: FiniteFloat = Field(default=1e-8, ge=0)


def _wanted(task):
    return task["kind"] == "nonlinear_reduction" or (
        task["kind"] == "root_profile" and task["seed"] == "action_tau_10"
    )


def _source_inputs(source: Path) -> dict:
    """Authenticate every source terminal, then retain a compact runnable intake."""
    plan, data = previous.verify(source, runtime=False)
    p = previous.Policy.model_validate(plan["policy"])
    portfolio = data["source_inputs"]["source_inputs"]["portfolio"]
    settings = first.Policy.model_validate(
        data["source_inputs"]["source_inputs"]["original_policy"]
    ).numerical
    records = []
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        ops = previous._operations(folder, identity, previous._allowances(task, p))
        terminal = read_seal(folder / "result.json")
        reconciliation = reporting.reconcile(
            terminal, previous._assemble(identity, data, ops), ops
        )
        if _wanted(task):
            records.append(
                {
                    "source_task": task,
                    "problem": portfolio["models"][task["model_index"]]["problem"],
                    "terminal": terminal,
                    "operations": ops,
                    "reconciliation": reconciliation,
                    "terminal_sha256": public.content_sha256(terminal),
                    "boundary_coordinate": BOUNDARY_COORDINATE
                    if task["kind"] == "root_profile"
                    else None,
                }
            )
    return {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "training": portfolio["training"],
        "numerical": settings.model_dump(mode="json"),
        "records": records,
        "test_data_opened": False,
    }


def _tasks(data: dict, p: Policy) -> list[dict]:
    """Bind the compact intake to original plan, endpoints and operation hashes."""
    source = data["source_plan"]
    production = p.population == "scaled-readout-4"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or source["protocol"] != previous.PROTOCOL
    ):
        raise ValueError("source protocol differs")
    if production and public.content_sha256(source) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed profile campaign")
    if production and (
        public.content_sha256(data["training"]) != SOURCE_TRAINING
        or public.content_sha256(data["numerical"]) != SOURCE_NUMERICAL
    ):
        raise ValueError("reviewed training data or numerical settings differ")
    expected = [t for t in source["tasks"] if _wanted(t)]
    names = [t["candidate_id"] for t in expected]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names
    ):
        raise ValueError("unsafe or duplicate candidate path")
    if len(expected) != len(data["records"]):
        raise ValueError("source roster differs")
    if production and len(expected) != len(SOURCE_RECORDS):
        raise ValueError("reviewed source projections are not pinned")
    # Data are one shared public training split, rather than the full history.
    training = PublicSplit.model_validate(data["training"])
    if training.name != "train":
        raise ValueError("only public training data are permitted")
    kernel.NumericalPolicy.model_validate(data["numerical"])
    tasks = []
    for i, (task, record) in enumerate(zip(expected, data["records"], strict=True)):
        terminal, ops = record["terminal"], record["operations"]
        identity = {"plan_sha256": public.content_sha256(source), "task": task}
        problem = PrefixFitProblem.model_validate(record["problem"])
        if (
            record["source_task"] != task
            or terminal["identity"] != identity
            or public.content_sha256(terminal) != record["terminal_sha256"]
            or record["reconciliation"]["scientific_fields_exact"] is not True
            or public.content_sha256(record["problem"]) != task["problem_sha256"]
            or terminal["operation_receipts"]
            != {k: public.content_sha256(v) for k, v in ops.items()}
        ):
            raise ValueError("source endpoint or receipt authentication differs")
        if production and public.content_sha256(record) != SOURCE_RECORDS[i]:
            raise ValueError("source projection differs from reviewed evidence")
        retained = terminal.get("retained") or {}
        if not retained.get("original_equations_verified") or not np.isfinite(
            retained.get("training_nmse", np.nan)
        ):
            raise ValueError("source lacks a verified finite endpoint")
        if production and tuple(problem.context.targets) != ("Gp", "I", "U"):
            raise ValueError("reviewed target contract differs")
        if not production and source["policy"]["population"] != "synthetic-smoke":
            raise ValueError("synthetic policy cannot consume a production source")
        root_profile = task["kind"] == "root_profile"
        boundary = record["boundary_coordinate"]
        if root_profile and (not isinstance(boundary, str) or not boundary):
            raise ValueError("root control requires an explicit boundary coordinate")
        tasks.append(
            {
                "index": i,
                "record_index": i,
                "task_id": task["candidate_id"]
                + ("-profiled" if root_profile else "-residual"),
                "candidate_id": task["candidate_id"],
                "model_index": task["model_index"],
                "kind": "root_profile" if root_profile else "residual_search",
                "seed": "retained",
                "problem_sha256": task["problem_sha256"],
                "source_record_sha256": public.content_sha256(record),
                "boundary_coordinate": boundary,
            }
        )
    return tasks


def prepare(root: Path, source: Path, policy: Policy) -> dict:
    """Freeze new work separately without modifying historical receipts."""
    if (
        root.resolve() == source.resolve()
        or root.resolve().is_relative_to(source.resolve())
        or source.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError("new root must be separate from source")
    return prepare_inputs(root, _source_inputs(source), policy)


def prepare_inputs(root: Path, data: dict, policy: Policy) -> dict:
    """Publish immutable source projection and numerical policy."""
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": _tasks(data, policy),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
        "training_prefix_reestimated": False,
        "objective": "original_full_training_reconstruction",
    }
    if reservation_ceiling(plan) > 13200:
        raise ValueError("reservations exceed worker envelope")
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime=True) -> tuple[dict, dict]:
    """Recompute roster, provenance and optionally exact execution runtime."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    p = Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _tasks(data, p)
        or plan["objective"] != "original_full_training_reconstruction"
        or any(
            plan[k] is not False
            for k in (
                "test_data_opened",
                "automatic_revision",
                "automatic_promotion",
                "training_prefix_reestimated",
            )
        )
        or plan["live_llm_calls"] != 0
        or reservation_ceiling(plan) > 13200
    ):
        raise ValueError("residual campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("residual campaign code/runtime differs")
    return plan, data


def _allowances(task, p):
    stages = {
        "setup": 180.0,
        "start_verification": p.verification_seconds + 60,
        "qualification": p.prerequisite_seconds + 180,
        "search": p.search_seconds + 180,
        "endpoint_verification": p.verification_seconds + 60,
    }
    if task["kind"] == "root_profile":
        stages |= {
            "profiled_start_verification": p.verification_seconds + 60,
            "boundary_control": p.boundary_seconds + 120,
            "boundary_verification": p.verification_seconds + 60,
        }
    return stages


def reservation_ceiling(plan):
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def compare_endpoints(old: dict, new: dict, p: Policy) -> dict:
    """Distinguish numerical ties using a frozen guard and solver discrepancies."""
    a, b = float(old["training_nmse"]), float(new["training_nmse"])
    errors = [abs(float(x.get("solver_loss_discrepancy", 0))) for x in (old, new)]
    if not np.isfinite([a, b, *errors]).all():
        raise ValueError("nonfinite endpoint comparison")
    guard = max(
        p.improvement_absolute_tolerance,
        p.improvement_relative_tolerance * max(abs(a), abs(b)),
        4 * sum(errors),
    )
    delta = a - b
    return {
        "classification": "meaningful_improvement"
        if delta > guard
        else "numerically_tied"
        if abs(delta) <= guard
        else "worse",
        "old_training_nmse": a,
        "new_training_nmse": b,
        "absolute_improvement": delta,
        "comparison_tolerance": guard,
        "rule": "max(abs_tolerance, relative_tolerance*loss_scale, "
        "4*(old+new solver_loss_discrepancy))",
    }


def _selection(record, ops, p):
    selected = ("source", record["terminal"]["retained"])
    comparisons = {}
    if not first._verified(ops.get("start_verification")) or not first._value(
        ops.get("qualification")
    ).get("passed"):
        return (*selected, comparisons)
    for stage in (
        "profiled_start_verification",
        "endpoint_verification",
        "boundary_verification",
    ):
        if first._verified(ops.get(stage)):
            endpoint = first._value(ops[stage])
            check = compare_endpoints(selected[1], endpoint, p)
            comparisons[stage] = check
            if check["classification"] == "meaningful_improvement":
                selected = (stage, endpoint)
    return (*selected, comparisons)


def _termination(search):
    if not search:
        return "not_started"
    if search.get("stop_reason") == "call_limit" or search.get(
        "search_calls_attempted", 0
    ) >= search.get("maximum_calls", float("inf")):
        return "evaluation_limit"
    if search.get("search_calls_started", 0) >= search.get(
        "maximum_calls", float("inf")
    ):
        return "evaluation_limit"
    message = str(search.get("optimizer_message", "")) + str(
        search.get("stop_reason", "")
    )
    if "function evaluations" in message or "call allowance" in message:
        return "evaluation_limit"
    if search.get("budget_exhausted"):
        return "numerical_budget_exhausted"
    return (
        "optimizer_stopped"
        if search.get("optimizer_success")
        else search.get("status", "unresolved")
    )


def _assemble(identity, data, ops, policy):
    task = identity["task"]
    record = data["records"][task["record_index"]]
    origin, retained, comparisons = _selection(record, ops, policy)
    search = first._value(ops.get("search"), partial=True)
    status = (
        "not_started"
        if not ops
        else "setup_unavailable"
        if not first._value(ops.get("setup"))
        else "improved_verified"
        if origin != "source"
        else "search_no_meaningful_improvement"
        if search
        else "qualification_unavailable"
    )
    termination = (
        _termination(search | {"maximum_calls": policy.maximum_calls})
        if search
        else "not_started"
    )
    return {
        "status": "complete",
        "identity": identity,
        "candidate_id": task["candidate_id"],
        "kind": task["kind"],
        "seed": task["seed"],
        "search_status": status,
        "search_termination": termination,
        "optimizer_reported_success": search.get("optimizer_success"),
        "optimization_converged": False,
        "stationarity_certified": False,
        "retained_origin": origin,
        "retained": retained,
        "source_endpoint": record["terminal"]["retained"],
        "training_reconstruction_nmse": retained["training_nmse"],
        "per_target_normalized_mse": retained.get("per_target_normalized_mse"),
        "endpoint_comparisons": comparisons,
        "qualification": first._value(ops.get("qualification")) or None,
        "boundary_control": first._value(ops.get("boundary_control"), partial=True)
        or None,
        "search_budget_exhausted": search.get("budget_exhausted"),
        "search_calls_started": search.get("search_calls_started"),
        "search_calls_attempted": search.get("search_calls_attempted"),
        "failure_diagnostics": search.get("failure_diagnostics"),
        "search_failure_histogram": search.get("invalid_trial_histogram"),
        "search_failure_examples": search.get("invalid_trial_examples"),
        "operation_statuses": {k: v["status"] for k, v in ops.items()},
        "operation_receipts": {k: public.content_sha256(v) for k, v in ops.items()},
        "cost": reporting.canonical_cost(ops),
        "inherited_cost": record["terminal"]["cost"],
        "evaluation_requests_by_operation": {
            k: first._value(v, partial=True).get("evaluation_requests_by_view")
            for k, v in ops.items()
        },
        "backend_execution_by_operation": {
            k: first._value(v, partial=True).get("stage_backend_calls")
            for k, v in ops.items()
            if first._value(v, partial=True).get("stage_backend_calls") is not None
        },
        "profile_execution_by_operation": {
            k: first._value(v, partial=True).get("profile_execution")
            for k, v in ops.items()
            if first._value(v, partial=True).get("profile_execution") is not None
        },
        "call_accounting_scope": (
            "Logical calls per view can be nested; not disjoint physical solves."
        ),
        "gradient_qualified": False,
        "parameter_confidence_percentage": None,
        "global_minimum_certified": False,
        "training_prefix_reestimated": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "automatic_revision": False,
        "automatic_promotion": False,
    }


class _Run(previous._Run):
    """Reuse journal/verification mechanics with distinct numerical gates."""

    def __init__(self, root, plan, data, index):
        self.plan, self.data = plan, data
        self.task = plan["tasks"][index]
        self.record = data["records"][self.task["record_index"]]
        self.policy = Policy.model_validate(plan["policy"])
        self.problem = PrefixFitProblem.model_validate(self.record["problem"])
        self.training = PublicSplit.model_validate(data["training"])
        self.identity = {"plan_sha256": public.content_sha256(plan), "task": self.task}
        self.folder = root / "results" / self.task["task_id"]
        self.allowances = _allowances(self.task, self.policy)
        self.ops = _operations(self.folder, self.identity, self.allowances)
        self.original = self.oracle = self.start = None

    def setup(self, save):
        anchor = self.record["terminal"]["retained"]["parameters"]
        settings = kernel.NumericalPolicy.model_validate(
            self.record["operations"]["setup"]["value"]["audit"]["numerical_policy"]
        )
        factory = timescale.make_oracle if self.task["kind"] == "root_profile" else None
        if factory:
            self.original, self.oracle = factory(
                self.problem, self.training, settings, anchor
            )
        else:
            self.original, self.oracle = reduced.make_oracle(
                self.problem, self.training, anchor, numerical=settings
            )
        self.start = coordinates.to_effective(self.original, self.oracle, anchor)
        value = {
            "passed": True,
            "names": list(self.oracle.names),
            "start": self.start,
            "audit": self.oracle.portfolio_audit,
            "optimizer_coordinates": len(self.oracle.names),
        }
        save(value)
        return value

    def fit(self):
        from autoformalism.fitting import prefix_portfolio_residual_search as residual

        p = self.policy
        self.verify_endpoint(
            "start_verification",
            self.start,
            self.record["terminal"]["retained"]["training_nmse"],
        )
        if not first._verified(self.ops.get("start_verification")):
            return
        is_root = self.task["kind"] == "root_profile"
        qualified = self.perform(
            "qualification",
            lambda save: roots.root_profile_qualification(
                self.oracle, self.start, p.prerequisite_seconds, save
            )
            if is_root
            else residual.qualify_residual(
                self.original, self.oracle, self.start, p.prerequisite_seconds, save
            ),
        )
        if not qualified.get("passed"):
            return
        if is_root:
            self.verify_endpoint(
                "profiled_start_verification",
                qualified["parameters"],
                qualified.get("training_nmse"),
            )
            if not first._verified(self.ops.get("profiled_start_verification")):
                return
        self.perform(
            "search",
            lambda save: roots.root_profile_search(
                self.oracle,
                self.start,
                p.search_seconds,
                p.maximum_calls,
                save,
                failure_diagnostics=True,
            )
            if is_root
            else residual.search(
                self.oracle, self.start, p.search_seconds, p.maximum_calls, save
            ),
        )
        search = first._value(self.ops.get("search"), partial=True)
        if search.get("best"):
            self.verify_endpoint(
                "endpoint_verification",
                search["best"]["parameters"],
                search["best"].get("training_nmse"),
            )
        if is_root:
            _, selected, _ = _selection(self.record, self.ops, p)
            parameters = coordinates.to_effective(
                self.original, self.oracle, selected["parameters"]
            )
            boundary = self.perform(
                "boundary_control",
                lambda save: roots.root_boundary_control(
                    self.oracle,
                    parameters,
                    self.task["boundary_coordinate"],
                    p.boundary_seconds,
                    save,
                ),
            )
            if boundary.get("passed") and boundary.get(
                "conditional_root_optimality_checked"
            ):
                self.verify_endpoint(
                    "boundary_verification",
                    boundary["parameters"],
                    boundary.get("training_nmse"),
                )


def run_task(root: Path, index: int) -> dict:
    """Resume unspent stages only; never repeat interrupted search allowances."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown residual task index")
    folder = root / "results" / plan["tasks"][index]["task_id"]
    p = Policy.model_validate(plan["policy"])
    with public._lock(folder):
        run = _Run(root, plan, data, index)
        if (folder / "result.json").exists():
            saved = read_seal(folder / "result.json")
            reporting.reconcile(
                saved, _assemble(run.identity, data, run.ops, p), run.ops
            )
            return saved
        if run.perform("setup", run.setup):
            if run.oracle is None:
                run.setup(lambda _: None)
            run.fit()
        result = _assemble(run.identity, data, run.ops, p)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Reconstruct results without integrating, fitting, or replacing terminals."""
    plan, data = verify(root, runtime=False)
    p = Policy.model_validate(plan["policy"])
    rows, audits = [], {}
    for task in plan["tasks"]:
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = root / "results" / task["task_id"]
        ops = _operations(folder, identity, _allowances(task, p))
        row = _assemble(identity, data, ops, p)
        terminal = (folder / "result.json").exists()
        if terminal:
            audits[task["task_id"]] = reporting.reconcile(
                read_seal(folder / "result.json"), row, ops
            )
        rows.append(
            row
            | {"status": "complete" if terminal else "partial" if ops else "missing"}
        )
    recorded = sum(row["status"] == "complete" for row in rows)
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "rows": rows,
        "status_counts": dict(Counter(row["search_status"] for row in rows)),
        "terminal_reconciliations": audits,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitations": [
            "Training reconstruction only; no new causal-prefix forecasts.",
            "New warm continuations do not restore internal Powell directions.",
            "Residual-only qualification does not qualify derivatives or confidence.",
            "Conditional root optimality is not outer convergence.",
            "Numerical ties retain the source incumbent.",
            "Model03 remains unresolved and is not included.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction residual continuation",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Model | Outcome | Termination | Training NMSE |",
            "| --- | --- | --- | ---: |",
        ]
        lines += [
            f"| {row['candidate_id']} | {row['search_status']} | "
            f"{row['search_termination']} | "
            f"{row['training_reconstruction_nmse']} |"
            for row in rows
        ]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


def inspection(root: Path) -> Path:
    """Package compact provenance and immutable receipts for offline inspection."""
    report(root)
    paths = [
        root / n for n in ("plan.json", "inputs.json", "summary.json", "REPORT.md")
    ]
    paths += sorted((root / "results").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.json"))
    paths += sorted((root / "submissions").rglob("*.id"))
    paths += (
        [root / "launcher_execution.json"]
        if (root / "launcher_execution.json").exists()
        else []
    )
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(set(paths)):
            if path.is_symlink() or not path.is_file():
                raise ValueError("nonregular inspection input")
            info = archive.gettarinfo(str(path), arcname=str(path.relative_to(root)))
            info.mtime = 0
            with path.open("rb") as stream:
                archive.addfile(info, stream)
        logs = {}
        for path in sorted((root / "logs").glob("*")):
            if path.is_file() and not path.is_symlink():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 16384))
                    logs[path.name] = stream.read().decode(errors="replace")
        payload = json.dumps(logs, sort_keys=True).encode()
        info = tarfile.TarInfo("inspection_log_tails.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    output = root / "inspection.tar.gz"
    output.write_bytes(buffer.getvalue())
    return output
