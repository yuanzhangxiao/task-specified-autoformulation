"""Verified model-15 continuation and finite meal-memory controls."""

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

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as retention
from autoformalism.research import construction_portfolio_targeted as previous
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-meal-continuation-1"
SOURCE_PLAN = "970eac91f92c978ec6250e4e010f6271257acc2957481bc5de1c6b1d3f3b7067"
SOURCE_TASK = "candidate-15-saved-meal_flux"
SOURCE_RECORD = "19c274ab7ff3d2f322f5f9c7cdffd341bfa7992d0bb92b6f0b3a7f6e7dde68ab"
SOURCE_TRAINING = previous.SOURCE_TRAINING
SOURCE_NUMERICAL = previous.SOURCE_NUMERICAL


class Policy(previous.Policy):
    """New nonrenewable allowance for one authenticated model family."""

    population: Literal["scaled-readout-meal-1", "synthetic-smoke"] = (
        "scaled-readout-meal-1"
    )


def _selected(plan):
    """Select the reviewed endpoint explicitly, never by comparing new losses."""
    if plan["policy"]["population"] == "synthetic-smoke":
        return [task for task in plan["tasks"] if task["seed"] == "control"]
    return [task for task in plan["tasks"] if task["task_id"] == SOURCE_TASK]


def _source_inputs(source: Path) -> dict:
    """Authenticate checkpoint terminals and project their immediate evidence."""
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    records = []
    for task in _selected(plan):
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        folder = source / "results" / task["task_id"]
        ops = previous._operations(folder, identity, previous._allowances(task, policy))
        terminal = read_seal(folder / "result.json")
        reconciliation = reporting.reconcile(
            terminal, previous._assemble(identity, data, ops, policy), ops
        )
        origin = data["records"][task["record_index"]]
        records.append(
            {
                "source_task": task,
                "problem": origin["problem"],
                "terminal": terminal,
                "operations": ops,
                "reconciliation": reconciliation,
                "terminal_sha256": public.content_sha256(terminal),
                "boundary_coordinate": task["boundary_coordinate"],
            }
        )
    return {
        "protocol": PROTOCOL,
        "source_plan": plan,
        "training": data["training"],
        "numerical": data["numerical"],
        "records": records,
        "test_data_opened": False,
    }


def _variants(record: dict, training: PublicSplit, production: bool) -> list[dict]:
    """Predetermine fixed positive decay values using only the public horizon."""
    def variant(label, *, fixed=None, rate=None, zero=False, chart="meal_flux"):
        return {
            "label": label,
            "fixed": dict(fixed or {}),
            "coordinate_chart": chart,
            "meal_rate_start": rate,
            "zero_representative": zero,
            "public_probe_scales": scales,
        }

    scales = None
    if not production:
        boundary = record["boundary_coordinate"]
        return [
            variant("control", chart="original"),
            variant("fixed-zero", fixed={boundary: 0.0}, chart="original"),
        ]
    horizon = max(float(row.time[-1] - row.time[0]) for row in training.rows)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("positive public training duration required")
    scales = {
        "horizon": horizon,
        "rule": "maximum public training trajectory duration",
        "training_sha256": public.content_sha256(training),
    }
    return [variant("warm-flux")] + [
        variant(label, fixed={"k_abs": fraction / horizon}, rate=fraction / horizon)
        for label, fraction in (
            ("fixed-0p001", 0.001), ("fixed-0p01", 0.01),
            ("fixed-0p1", 0.1), ("fixed-1", 1.0),
        )
    ] + [variant(
        "exact-zero", fixed={"k_abs": 0.0, "k_abs_rate": 0.0},
        zero=True, chart="original",
    )]


def _tasks(data: dict, p: Policy) -> list[dict]:
    """Pin unchanged equations, training, numerical policy and verified incumbents."""
    source = data["source_plan"]
    production = p.population == "scaled-readout-meal-1"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or source["protocol"] != previous.PROTOCOL
    ):
        raise ValueError("meal continuation source protocol differs")
    if production and public.content_sha256(source) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed meal continuation campaign")
    if production and (
        public.content_sha256(data["training"]) != SOURCE_TRAINING
        or public.content_sha256(data["numerical"]) != SOURCE_NUMERICAL
    ):
        raise ValueError("reviewed training data or numerical settings differ")
    expected = _selected(source)
    if len(expected) != len(data["records"]) or len(expected) != 1:
        raise ValueError("source roster differs")
    if not production and source["policy"]["population"] != "synthetic-smoke":
        raise ValueError("synthetic policy cannot consume a production source")
    names = [t["candidate_id"] for t in expected]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names
    ):
        raise ValueError("unsafe or duplicate candidate path")
    training = PublicSplit.model_validate(data["training"])
    if training.name != "train":
        raise ValueError("only public training data are permitted")
    kernel.NumericalPolicy.model_validate(data["numerical"])
    tasks = []
    for index, (task, record) in enumerate(zip(expected, data["records"], strict=True)):
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
            != {key: public.content_sha256(value) for key, value in ops.items()}
        ):
            raise ValueError("source endpoint or receipt authentication differs")
        if production and public.content_sha256(record) != SOURCE_RECORD:
            raise ValueError("source projection differs from reviewed evidence")
        retained = terminal.get("retained") or {}
        if not retained.get("original_equations_verified") or not np.isfinite(
            retained.get("training_nmse", np.nan)
        ):
            raise ValueError("source lacks a verified finite endpoint")
        if production and tuple(problem.context.targets) != ("Gp", "I", "U"):
            raise ValueError("reviewed target contract differs")
        if task["kind"] != "root_profile":
            raise ValueError("unsupported continuation kind")
        boundary = record["boundary_coordinate"]
        if boundary != task["boundary_coordinate"] or (
            task["kind"] == "root_profile"
            and (not isinstance(boundary, str) or not boundary)
        ):
            raise ValueError("boundary coordinate differs")
        kernel.NumericalPolicy.model_validate(
            ops["setup"]["value"]["audit"]["numerical_policy"]
        )
        for experiment in _variants(record, training, production):
            tasks.append(
                {
                    "index": len(tasks),
                    "record_index": index,
                    "task_id": task["candidate_id"] + "-" + experiment["label"],
                    "candidate_id": task["candidate_id"],
                    "model_index": task["model_index"],
                    "kind": task["kind"],
                    "seed": experiment["label"],
                    "problem_sha256": task["problem_sha256"],
                    "source_record_sha256": public.content_sha256(record),
                    "boundary_coordinate": boundary,
                    "experiment": experiment,
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
        raise ValueError("meal continuation campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("meal continuation campaign code/runtime differs")
    return plan, data


def _allowances(task, p):
    """Use the tested root-profile reservations without changing old campaigns."""
    return previous._allowances(task, p)


def reservation_ceiling(plan):
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def _assemble(identity, data, ops, policy):
    """Expose qualified termination separately from the optimizer's raw flag."""
    result = previous._assemble(identity, data, ops, policy)
    search = first._value(ops.get("search"), partial=True)
    setup = first._value(ops.get("setup"))
    return result | {
        "optimizer_raw_success": search.get("optimizer_raw_success"),
        "optimizer_qualified_success": search.get("optimizer_success"),
        "optimizer_feasible_termination": (search.get("optimizer_terminal") or {}).get(
            "feasible_evaluation_verified"
        ),
        "optimizer_terminal": search.get("optimizer_terminal"),
        "zero_representative_certificate": setup.get("zero_representative_certificate"),
        "zero_initial_columns_scope": (
            "Original root profiler retained; zero meal-root output columns do not "
            "identify the hidden meal initial value."
            if identity["task"]["experiment"]["zero_representative"] else None
        ),
    }


class _Run(previous._Run):
    """Fit explicit diagnostic starts while preserving the historical incumbent."""

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
        self.source_start = self.chart = None

    def setup(self, save):
        """Compile original equations, then construct a domain-checked trial seed."""
        from autoformalism.fitting import prefix_meal_coordinates as meal

        value = retention._Run.setup(self, lambda _: None)
        self.source_start = dict(self.start)
        experiment = self.task["experiment"]
        if self.task["candidate_id"] == "candidate-15":
            self.chart = meal.certify(self.oracle, self.source_start)
            if experiment["meal_rate_start"] is not None:
                self.start = self.chart.reseed(
                    self.start, experiment["meal_rate_start"]
                )
        self.start = dict(self.start) | experiment["fixed"]
        zero_certificate = None
        if experiment["zero_representative"]:
            if self.chart is None:
                raise ValueError("zero meal control requires an exact certificate")
            zero = self.chart.zero_representative(self.start)
            self.start = zero["parameters"]
            zero_certificate = zero["certificate"]
            if zero_certificate["fixed_shared_coordinates"] != experiment["fixed"]:
                raise ValueError("zero control differs from its certified coordinates")
        self.oracle.vector(self.start)
        coordinate_start = None
        if self.chart is not None:
            coordinate_start = (
                self.chart.zero_stratum(self.start)
                if self.start[self.chart.rate] == 0
                else {
                    "flux_gain": self.start[self.chart.rate]
                    * self.start[self.chart.input_gain],
                    "scaled_initials": self.chart.scaled_initials(self.start),
                }
            )
        value |= {
            "source_start": self.source_start,
            "start": self.start,
            "experiment": experiment,
            "coordinate_audit": self.chart.audit if self.chart else None,
            "coordinate_start": coordinate_start,
            "zero_representative_certificate": zero_certificate,
        }
        save(value)
        return value

    def verify_endpoint(self, stage, parameters, expected=None):
        """Check held values as well as independent original-equation replay."""
        if stage != "start_verification":
            fixed = self.task["experiment"]["fixed"]
            if any(parameters.get(name) != value for name, value in fixed.items()):
                raise ValueError("diagnostic endpoint changed a held coordinate")
        return retention._Run.verify_endpoint(self, stage, parameters, expected)


def run_task(root: Path, index: int) -> dict:
    """Resume unspent stages only; never repeat interrupted search allowances."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown meal continuation task index")
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
            "Fixed-coordinate trial losses are upper bounds, not certified minima.",
            "No parameter region is excluded by an unsuccessful nuisance solve.",
            "Meal-flux coordinates apply only on the strictly positive rate branch.",
            "Exact zero absorption uses the original model, not its singular closure.",
            "At zero, the meal gain is fixed by an observed-output equivalence proof.",
            "Zero meal-root output columns do not identify hidden meal initials.",
            "Raw optimizer success is distinct from a paired finite terminal point.",
            "Routine progress counters can lag; operation reservations never renew.",
            "Overall-wall and numerical-active limits are separate.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction meal continuation",
            "",
            f"{recorded}/{len(rows)} terminal tasks.",
            "",
            "| Model / trial | Outcome | Termination | Trial NMSE | Retained NMSE |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        lines += [
            f"| {row['candidate_id']} / {row['seed']} | {row['search_status']} | "
            f"{row['search_termination']} | "
            f"{row['trial_training_nmse']} | "
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
