"""Targeted parameter profiles and certified meal-coordinate comparisons."""

from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_checkpoint as previous
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as retention
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-targeted-1"
SOURCE_PLAN = "49affc0c7b22ba6f5463b5e04febc829745f4598dabd19745034f96f4aae506a"
SOURCE_RECORDS = (
    "d22bd4084a494041811cd521aae2b79ad0fa308e57041c003f104a7a55cc37ad",
    "96bdcaa2a5b2c69f95e7df15ce58787f4c68878e68f1a906acda74526b0b3332",
    "75b26d946e27646ea8c685ba9c844a95a68181057937e2d3871b09a4a37b4da2",
    "2b421b595bf445e381d0152d9865eb6a62b59eb63a8859650a4169e5dbb70026",
)
SOURCE_TRAINING = previous.SOURCE_TRAINING
SOURCE_NUMERICAL = previous.SOURCE_NUMERICAL


class Policy(previous.Policy):
    """Matched positive-branch searches and shorter saturation controls."""

    diagnostic_seconds: FiniteFloat = Field(default=600, gt=0, le=600)
    diagnostic_calls: int = Field(default=250, gt=0, le=250)


def _source_inputs(source: Path) -> dict:
    """Authenticate checkpoint terminals and project their immediate evidence."""
    plan, data = previous.verify(source, runtime=False)
    policy = previous.Policy.model_validate(plan["policy"])
    records = []
    for task in plan["tasks"]:
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


def _training_scales(training: PublicSplit) -> dict:
    """Choose predetermined probe units from public training values only."""
    horizon = max(float(row.time[-1] - row.time[0]) for row in training.rows)
    glucose = np.concatenate(
        [np.abs(np.asarray(row.targets["Gp"], float)) for row in training.rows]
    )
    positive = glucose[np.isfinite(glucose) & (glucose > 0)]
    if not np.isfinite(horizon) or horizon <= 0 or not len(positive):
        raise ValueError("positive public training horizon and glucose scale required")
    return {
        "horizon": horizon,
        "glucose_scale": float(np.median(positive)),
        "rule": "max trajectory duration; median positive absolute observed Gp",
        "training_sha256": public.content_sha256(training),
    }


def _variants(record: dict, training: PublicSplit, production: bool) -> list[dict]:
    """Freeze probe values independently of fit outcomes or private coefficients."""
    name = record["source_task"]["candidate_id"]
    anchor = record["terminal"]["retained"]["parameters"]

    def variant(label, *, fixed=None, chart="original", rate=None, scales=None):
        return {
            "label": label,
            "fixed": dict(fixed or {}),
            "coordinate_chart": chart,
            "meal_rate_start": rate,
            "public_probe_scales": scales,
        }

    if not production:
        boundary = record["boundary_coordinate"]
        return [variant("control"), variant("fixed-zero", fixed={boundary: 0.0})]
    scales = _training_scales(training)
    horizon, glucose = scales["horizon"], scales["glucose_scale"]
    if name == "candidate-12":
        return [variant("free-control", scales=scales)] + [
            variant(label, fixed={"k_A_decay": value}, scales=scales)
            for label, value in (
                ("decay-zero", 0.0),
                ("decay-0p1", 0.1 / horizon),
                ("decay-1", 1.0 / horizon),
                ("decay-3", 3.0 / horizon),
            )
        ]
    if name == "candidate-15":
        result = [
            variant(label + "-" + chart, chart=chart, rate=rate, scales=scales)
            for label, rate in (
                ("saved", None),
                ("decay-0p1", 0.1 / horizon),
                ("decay-1", 1.0 / horizon),
            )
            for chart in ("original", "meal_flux")
        ]
        result.append(variant("exact-zero", fixed={"k_abs": 0.0}, scales=scales))
        return result
    if name in {"candidate-10", "candidate-14"}:
        return [
            variant(label, fixed={"Km_ins": value}, scales=scales)
            for label, value in (
                ("saturation-saved", anchor["Km_ins"]),
                ("saturation-0p1", 0.1 * glucose),
                ("saturation-1", glucose),
                ("saturation-10", 10.0 * glucose),
            )
        ]
    raise ValueError("unsupported targeted model")


def _tasks(data: dict, p: Policy) -> list[dict]:
    """Pin unchanged equations, training, numerical policy and verified incumbents."""
    source = data["source_plan"]
    production = p.population == "scaled-readout-4"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or source["protocol"] != previous.PROTOCOL
    ):
        raise ValueError("targeted source protocol differs")
    if production and public.content_sha256(source) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed targeted campaign")
    if production and (
        public.content_sha256(data["training"]) != SOURCE_TRAINING
        or public.content_sha256(data["numerical"]) != SOURCE_NUMERICAL
    ):
        raise ValueError("reviewed training data or numerical settings differ")
    expected = source["tasks"]
    if len(expected) != len(data["records"]) or (
        production and len(expected) != len(SOURCE_RECORDS)
    ):
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
        if production and public.content_sha256(record) != SOURCE_RECORDS[index]:
            raise ValueError("source projection differs from reviewed evidence")
        retained = terminal.get("retained") or {}
        if not retained.get("original_equations_verified") or not np.isfinite(
            retained.get("training_nmse", np.nan)
        ):
            raise ValueError("source lacks a verified finite endpoint")
        if production and tuple(problem.context.targets) != ("Gp", "I", "U"):
            raise ValueError("reviewed target contract differs")
        if task["kind"] not in {"root_profile", "residual_search"}:
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
        raise ValueError("targeted campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("targeted campaign code/runtime differs")
    return plan, data


def _allowances(task, p):
    seconds = (
        p.search_seconds if task["kind"] == "root_profile" else p.diagnostic_seconds
    )
    stages = {
        "setup": 180.0,
        "start_verification": p.verification_seconds + 60,
        "qualification": p.prerequisite_seconds + 180,
        "search": seconds + 180,
        "endpoint_verification": p.verification_seconds + 60,
    }
    if task["kind"] == "root_profile":
        stages["profiled_start_verification"] = p.verification_seconds + 60
    return stages


def reservation_ceiling(plan):
    p = Policy.model_validate(plan["policy"])
    return max((sum(_allowances(t, p).values()) for t in plan["tasks"]), default=0)


def _operations(folder, identity, allowances):
    return previous._operations(folder, identity, allowances)


def _termination(search: dict) -> str:
    """Prefer measured timeout reasons to broad budget flags or counter guesses."""
    if not search:
        return "not_started"
    if (search.get("timing") or {}).get("callback_failures", 0) or (
        search.get("checkpoint_persistence") or {}
    ).get("writes_failed", 0):
        return "checkpoint_failure"
    reason = (search.get("timing") or {}).get("timeout_reason")
    if reason == "overall_wall_limit":
        return "overall_wall_limit"
    if reason == "numerical_active_wall_limit":
        return "numerical_active_wall_limit"
    if search.get("stop_reason") == "call_limit":
        return "evaluation_limit"
    if search.get("search_calls_attempted", 0) >= search.get(
        "maximum_calls", float("inf")
    ):
        return "evaluation_limit"
    if search.get("budget_exhausted"):
        return "budget_limit_unclassified"
    if search.get("optimizer_success"):
        return "optimizer_stopped"
    return search.get("stop_reason") or search.get("status", "unresolved")


def _assemble(identity, data, ops, policy):
    """Keep incumbent retention separate from constrained-profile evidence."""
    task = identity["task"]
    calls = (
        policy.maximum_calls
        if task["kind"] == "root_profile"
        else policy.diagnostic_calls
    )
    result = previous._assemble(
        identity, data, ops, policy.model_copy(update={"maximum_calls": calls})
    )
    search = first._value(ops.get("search"), partial=True)
    timing = search.get("timing") or {}
    termination = (
        _termination(search | {"maximum_calls": calls}) if search else "not_started"
    )
    journal_status = (ops.get("search") or {}).get("status")
    if journal_status in {"interrupted", "unavailable"} and termination not in {
        "overall_wall_limit",
        "numerical_active_wall_limit",
        "checkpoint_failure",
    }:
        # A killed process may leave an earlier ordinary progress snapshot.
        termination = journal_status
    verified = [
        first._value(ops[stage])
        for stage in ("profiled_start_verification", "endpoint_verification")
        if first._verified(ops.get(stage))
    ]
    trial = (
        min(verified, key=lambda value: value["training_nmse"]) if verified else None
    )
    return result | {
        "experiment": task["experiment"],
        "trial_endpoint": trial,
        "trial_training_nmse": trial["training_nmse"] if trial else None,
        "trial_scope": (
            "independently verified feasible loss; upper bound on profile minimum"
        ),
        "profile_minimum_certified": False,
        "parameter_region_excluded": False,
        "coordinate_audit": first._value(ops.get("setup")).get("coordinate_audit"),
        "search_allowance": {
            "numerical_seconds": policy.search_seconds
            if task["kind"] == "root_profile"
            else policy.diagnostic_seconds,
            "objective_calls": calls,
        },
        "search_termination": termination,
        "search_timing": timing,
        "checkpoint_policy": {"interval_seconds": policy.checkpoint_interval_seconds},
        "checkpoint_persistence": search.get("checkpoint_persistence"),
        "compact_progress": search.get("compact_progress"),
        "progress_may_lag_counters": True,
        "partial_progress_does_not_renew_budget": True,
    }


class _Run(retention._Run):
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

        value = super().setup(lambda _: None)
        self.source_start = dict(self.start)
        experiment = self.task["experiment"]
        if self.task["candidate_id"] == "candidate-15":
            self.chart = meal.certify(self.oracle, self.source_start)
            if experiment["meal_rate_start"] is not None:
                self.start = self.chart.reseed(
                    self.start, experiment["meal_rate_start"]
                )
        self.start = dict(self.start) | experiment["fixed"]
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
        }
        save(value)
        return value

    def verify_endpoint(self, stage, parameters, expected=None):
        """Check held values as well as independent original-equation replay."""
        if stage != "start_verification":
            fixed = self.task["experiment"]["fixed"]
            if any(parameters.get(name) != value for name, value in fixed.items()):
                raise ValueError("diagnostic endpoint changed a held coordinate")
        return super().verify_endpoint(stage, parameters, expected)

    def search(self, save):
        from autoformalism.fitting import prefix_portfolio_residual_search as residual
        from autoformalism.fitting import prefix_portfolio_root_profile as roots
        from autoformalism.fitting.compact_search_checkpoint import (
            SearchCheckpointPolicy,
        )

        p = self.policy
        checkpoint_policy = SearchCheckpointPolicy(
            interval_seconds=p.checkpoint_interval_seconds
        )
        if self.task["kind"] == "root_profile":
            return roots.root_profile_search(
                self.oracle,
                self.start,
                p.search_seconds,
                p.maximum_calls,
                save,
                failure_diagnostics=True,
                checkpoint_policy=checkpoint_policy,
                fixed=self.task["experiment"]["fixed"],
                coordinate_chart=self.chart
                if self.task["experiment"]["coordinate_chart"] == "meal_flux"
                else None,
            )
        return residual.search(
            self.oracle,
            self.start,
            p.diagnostic_seconds,
            p.diagnostic_calls,
            save,
            checkpoint_policy=checkpoint_policy,
            fixed=self.task["experiment"]["fixed"],
        )

    def fit(self):
        """Qualify each trial separately; a failed seed never erases the incumbent."""
        from autoformalism.fitting import prefix_portfolio_residual_search as residual
        from autoformalism.fitting import prefix_portfolio_root_profile as roots

        p = self.policy
        self.verify_endpoint(
            "start_verification",
            self.source_start,
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
        self.perform("search", self.search)
        search = first._value(self.ops.get("search"), partial=True)
        if search.get("best"):
            self.verify_endpoint(
                "endpoint_verification",
                search["best"]["parameters"],
                search["best"].get("training_nmse"),
            )


def run_task(root: Path, index: int) -> dict:
    """Resume unspent stages only; never repeat interrupted search allowances."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown targeted task index")
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
            "Routine progress counters can lag; operation reservations never renew.",
            "Overall-wall and numerical-active limits are separate.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction targeted parameter study",
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
