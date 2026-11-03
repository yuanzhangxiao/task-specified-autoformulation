"""Continue verified construction fits with compact durable search progress."""

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
from autoformalism.research import construction_portfolio_fitting as first
from autoformalism.research import construction_portfolio_residual as previous
from autoformalism.research import continuation_report_cost as reporting
from autoformalism.schemas.base import FiniteFloat
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "construction-portfolio-checkpoint-1"
SOURCE_PLAN = "299fd4f5a0861fe4966a95b71f8501651bc872a21c75c24aecd41e5d0d2d5673"
SOURCE_RECORDS = (
    "7b359dc8447babddfbdf60e782d0ed89fa98b18945c5b66d33465be1b362fb97",
    "f4cecd679b3d39375bb067c66ce721808ece18e1dc1a5e4a97d9b4a2a42603f6",
    "70ea5eda6001d4ac1c16a4f0b3ae3d70039c6abea78b63a43bf455e0b5ef9fbd",
    "73bf201720eb3d2e902664c95dea5a067d969cc6670af042a9f9f1e20c639a8d",
)
SOURCE_TRAINING = previous.SOURCE_TRAINING
SOURCE_NUMERICAL = previous.SOURCE_NUMERICAL


class Policy(previous.Policy):
    """Preserve numerical reservations while bounding routine persistence frequency."""

    checkpoint_interval_seconds: FiniteFloat = Field(default=30, gt=0, le=60)


def _source_inputs(source: Path) -> dict:
    """Authenticate all residual terminals and project only their immediate evidence."""
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


def _tasks(data: dict, p: Policy) -> list[dict]:
    """Pin unchanged equations, training, numerical policy and verified incumbents."""
    source = data["source_plan"]
    production = p.population == "scaled-readout-4"
    if (
        data["protocol"] != PROTOCOL
        or data["test_data_opened"] is not False
        or source["protocol"] != previous.PROTOCOL
    ):
        raise ValueError("checkpoint source protocol differs")
    if production and public.content_sha256(source) != SOURCE_PLAN:
        raise ValueError("source is not the reviewed residual campaign")
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
        tasks.append(
            {
                "index": index,
                "record_index": index,
                "task_id": task["candidate_id"]
                + ("-profiled" if task["kind"] == "root_profile" else "-residual"),
                "candidate_id": task["candidate_id"],
                "model_index": task["model_index"],
                "kind": task["kind"],
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
        raise ValueError("checkpoint campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("checkpoint campaign code/runtime differs")
    return plan, data


def _allowances(task, p):
    return previous._allowances(task, p)


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
    """Reuse verified retention and expose precise stopping/persistence evidence."""
    result = previous._assemble(identity, data, ops, policy)
    search = first._value(ops.get("search"), partial=True)
    timing = search.get("timing") or {}
    termination = (
        _termination(search | {"maximum_calls": policy.maximum_calls})
        if search
        else "not_started"
    )
    journal_status = (ops.get("search") or {}).get("status")
    if journal_status in {"interrupted", "unavailable"} and termination not in {
        "overall_wall_limit",
        "numerical_active_wall_limit",
        "checkpoint_failure",
    }:
        # A killed process may leave an earlier ordinary progress snapshot.
        termination = journal_status
    return result | {
        "search_termination": termination,
        "search_timing": timing,
        "checkpoint_policy": {"interval_seconds": policy.checkpoint_interval_seconds},
        "checkpoint_persistence": search.get("checkpoint_persistence"),
        "compact_progress": search.get("compact_progress"),
        "progress_may_lag_counters": True,
        "partial_progress_does_not_renew_budget": True,
    }


class _Run(previous._Run):
    """Keep retention/qualification semantics while opting into compact progress."""

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
            )
        return residual.search(
            self.oracle,
            self.start,
            p.search_seconds,
            p.maximum_calls,
            save,
            checkpoint_policy=checkpoint_policy,
        )


def run_task(root: Path, index: int) -> dict:
    """Resume unspent stages only; never repeat interrupted search allowances."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown checkpoint task index")
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
            "Routine progress counters can lag; operation reservations never renew.",
            "Overall-wall and numerical-active limits are separate.",
        ],
    }
    with public._lock(root / "report"):
        public._write(root / "summary.json", result)
        lines = [
            "# Construction checkpoint continuation",
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
