"""Durable bounded comparison of the two finite-budget premise baselines.

Scientific checkpoints are deterministic. Separate attempt receipts record actual
elapsed time and distinguish logical search work from interrupted/replayed work.
An unfinished receipt leaves actual execution cost unknown, never implicitly zero.
"""

from __future__ import annotations

import copy
import decimal
import fcntl
import hashlib
import json
import math
import platform
import re
import time
from contextlib import suppress
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from autoformalism.research import premise_baseline as core
from autoformalism.research import premise_intervals as intervals
from autoformalism.research import premise_reference as persistence

PROTOCOL = "vega-premise-baseline-campaign-v1"
_COST_KEYS = ("boxes", "kernels", "point_tests")
_ATTEMPT_NAME = re.compile(r"attempt-(\d{3})-(\d{3})-(start|finish)\.json")


def _json_value(value: Any) -> Any:
    """Normalize tuples to their exact JSON representation before sealing."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _jobs() -> list[tuple[str, core.Problem, str]]:
    jobs = [
        (f"{problem.case_id}--{problem.contract}--{method}", problem, method)
        for problem in core.problems()
        for method in core.METHODS
    ]
    if len({job_id for job_id, _, _ in jobs}) != len(jobs):
        raise ValueError("comparison job identifiers must be unique")
    return jobs


def _roster() -> list[tuple[str, core.Problem, str, int]]:
    return [
        (job_id, problem, method, budget)
        for job_id, problem, method in _jobs()
        for budget in core.BUDGETS
    ]


def _plan() -> dict[str, Any]:
    """Bind public observations, fixed work budgets, source and numeric runtime."""
    sources = (
        Path(__file__),
        Path(core.__file__),
        Path(intervals.__file__),
        Path(persistence.__file__),
    )
    return {
        "protocol": PROTOCOL,
        "core_protocol": core.PROTOCOL,
        "problems": [_json_value(asdict(problem)) for problem in core.problems()],
        "jobs": [
            {
                "job_id": job_id,
                "problem": _json_value(asdict(problem)),
                "method": method,
            }
            for job_id, problem, method in _jobs()
        ],
        "budgets": list(core.BUDGETS),
        "schedule": "problem order, then method order, then increasing box budgets",
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "decimal": decimal.__version__,
            "libmpdec": decimal.__libmpdec_version__,
            "precision": intervals.PRECISION,
            "numpy": version("numpy"),
            "scipy": version("scipy"),
        },
        "cost_semantics": {
            "logical": "latest cumulative core cost per job, including preparation",
            "executed": (
                "sum observed attempt deltas; unknown if any receipt is unfinished"
            ),
            "timing": (
                "initialization, advance, summarization and checkpoint publication"
            ),
            "timing_excludes": (
                "startup, prior-evidence validation and read-only reporting"
            ),
        },
    }


def _cost(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_COST_KEYS):
        raise ValueError("cost must contain boxes, kernels and point_tests")
    if any(type(value[key]) is not int or value[key] < 0 for key in _COST_KEYS):
        raise ValueError("cost counts must be nonnegative integers")
    return dict(value)


def _state_identity(
    state: dict[str, Any], problem: core.Problem, method: str, budget: int
) -> dict[str, int]:
    if not isinstance(state, dict) or (
        state.get("protocol") != core.PROTOCOL
        or state.get("problem") != _json_value(asdict(problem))
        or state.get("method") != method
    ):
        raise ValueError("checkpoint state problem, method or protocol differs")
    cost = _cost(state.get("cost"))
    if cost["boxes"] > budget:
        raise ValueError("checkpoint exceeds its logical box budget")
    return cost


def _load(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a contiguous global chain and recompute reports without search."""
    rows = []
    previous = persistence._digest(plan)
    prior_cost: dict[str, dict[str, int]] = {}
    required = {"job_id", "budget", "previous_sha256", "state", "report"}
    for index, (job_id, problem, method, budget) in enumerate(_roster()):
        path = root / f"checkpoint-{index:03d}.json"
        if not path.exists():
            break
        row = persistence._read(path)
        if (
            not isinstance(row, dict)
            or set(row) != required
            or (
                row["job_id"] != job_id
                or type(row["budget"]) is not int
                or row["budget"] != budget
                or row["previous_sha256"] != previous
            )
        ):
            raise ValueError("checkpoint identity or predecessor differs")
        cost = _state_identity(row["state"], problem, method, budget)
        if job_id in prior_cost and any(
            cost[key] < prior_cost[job_id][key] for key in _COST_KEYS
        ):
            raise ValueError("cumulative checkpoint work decreases")
        computed = core.summarize(problem, copy.deepcopy(row["state"]))
        if row["report"] != computed:
            raise ValueError("checkpoint report differs from saved state")
        prior_cost[job_id] = cost
        previous = persistence._digest(row)
        rows.append(row)
    found = {path.name for path in root.glob("checkpoint-*.json")}
    if found != {f"checkpoint-{index:03d}.json" for index in range(len(rows))}:
        raise ValueError("noncontiguous or unexpected checkpoints")
    return rows


def _summary(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep missing rows explicit and avoid double-counting cumulative costs."""
    roster = _roster()
    latest = {row["job_id"]: row for row in rows}
    return {
        "protocol": PROTOCOL,
        "plan_sha256": persistence._digest(plan),
        "expected_jobs": len(_jobs()),
        "expected_checkpoints": len(roster),
        "recorded_checkpoints": len(rows),
        "complete": len(rows) == len(roster),
        "completed_jobs": [
            job_id
            for job_id, _, _ in _jobs()
            if job_id in latest and latest[job_id]["budget"] == core.BUDGETS[-1]
        ],
        "missing_checkpoints": [
            {"job_id": job_id, "budget": budget}
            for job_id, _, _, budget in roster[len(rows) :]
        ],
        "logical_cost": {
            key: sum(row["state"]["cost"][key] for row in latest.values())
            for key in _COST_KEYS
        },
        "rows": [
            {"job_id": row["job_id"], "budget": row["budget"], "report": row["report"]}
            for row in rows
        ],
    }


def _attempts(root: Path) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
    """Read separate runtime receipts; missing completion remains visible."""
    result: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for path in sorted((root / "runtime").glob("attempt-*.json")):
        match = _ATTEMPT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("malformed runtime receipt filename")
        checkpoint, attempt = int(match[1]), int(match[2])
        result.setdefault((checkpoint, attempt), {})[match[3]] = persistence._read(path)
    return result


def _execution_accounting(
    root: Path, plan: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    attempts = _attempts(root)
    roster = _roster()
    observed_seconds = 0.0
    observed_cost = dict.fromkeys(_COST_KEYS, 0)
    complete_indices: set[int] = set()
    unfinished, unknown_cost, failed = [], [], []
    per_checkpoint: dict[int, list[int]] = {}
    for (index, attempt), receipts in sorted(attempts.items()):
        if not 0 <= index < len(roster) or index > len(rows) or "start" not in receipts:
            raise ValueError("orphan or out-of-roster runtime receipt")
        per_checkpoint.setdefault(index, []).append(attempt)
        job_id, _, _, budget = roster[index]
        start = receipts["start"]
        if (
            not isinstance(start, dict)
            or set(start)
            != {
                "plan_sha256",
                "checkpoint_index",
                "attempt",
                "job_id",
                "budget",
                "previous_sha256",
                "cost_before",
            }
            or (
                start["plan_sha256"] != persistence._digest(plan)
                or type(start["checkpoint_index"]) is not int
                or type(start["attempt"]) is not int
                or type(start["budget"]) is not int
                or start["checkpoint_index"] != index
                or start["attempt"] != attempt
                or start["job_id"] != job_id
                or start["budget"] != budget
            )
        ):
            raise ValueError("runtime start identity differs")
        previous = persistence._digest(plan if index == 0 else rows[index - 1])
        if start["previous_sha256"] != previous:
            raise ValueError("runtime receipt predecessor differs")
        before = _cost(start["cost_before"])
        expected_before = (
            rows[index - 1]["state"]["cost"]
            if index > 0 and rows[index - 1]["job_id"] == job_id
            else dict.fromkeys(_COST_KEYS, 0)
        )
        if before != expected_before:
            raise ValueError("runtime starting cost differs from saved predecessor")
        name = f"{index:03d}:{attempt:03d}"
        if "finish" not in receipts:
            unfinished.append(name)
            unknown_cost.append(name)
            continue
        finish = receipts["finish"]
        if (
            not isinstance(finish, dict)
            or set(finish)
            != {
                "start_sha256",
                "status",
                "elapsed_seconds",
                "cost_after",
                "checkpoint_sha256",
            }
            or finish["start_sha256"] != persistence._digest(start)
        ):
            raise ValueError("runtime completion identity differs")
        elapsed = finish["elapsed_seconds"]
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("runtime elapsed time must be finite and nonnegative")
        observed_seconds += elapsed
        if finish["cost_after"] is None:
            unknown_cost.append(name)
        else:
            after = _cost(finish["cost_after"])
            if any(after[key] < before[key] for key in _COST_KEYS):
                raise ValueError("runtime attempt cost decreases")
            for key in _COST_KEYS:
                observed_cost[key] += after[key] - before[key]
        if finish["status"] == "complete":
            if index >= len(rows) or (
                finish["checkpoint_sha256"] != persistence._digest(rows[index])
                or finish["cost_after"] != rows[index]["state"]["cost"]
                or index in complete_indices
            ):
                raise ValueError(
                    "runtime completion disagrees with scientific checkpoint"
                )
            complete_indices.add(index)
        elif finish["status"] == "interrupted":
            if finish["checkpoint_sha256"] is not None:
                raise ValueError("interrupted receipt claims a completed checkpoint")
            failed.append(name)
        else:
            raise ValueError("unknown runtime completion status")
    if any(values != list(range(len(values))) for values in per_checkpoint.values()):
        raise ValueError("runtime attempt numbers are not contiguous")
    missing = sorted(set(range(len(rows))) - complete_indices)
    return {
        "attempts_started": len(attempts),
        "interrupted_attempts": failed,
        "unfinished_attempts": unfinished,
        "missing_completed_checkpoint_timings": missing,
        "observed_elapsed_seconds": observed_seconds,
        "actual_total_elapsed_seconds": None
        if unfinished or missing
        else observed_seconds,
        "observed_execution_cost": observed_cost,
        "actual_total_execution_cost": None
        if unknown_cost or missing
        else observed_cost,
        "unknown_cost_attempts": unknown_cost,
        "scope": plan["cost_semantics"]["timing"],
    }


def _report(
    root: Path, plan: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    summary = _summary(plan, rows)
    if (root / "summary.json").exists() and persistence._read(
        root / "summary.json"
    ) != summary:
        raise ValueError("saved summary differs from scientific checkpoints")
    return summary | {"execution_accounting": _execution_accounting(root, plan, rows)}


def report_campaign(root: Path) -> dict[str, Any]:
    """Validate saved evidence only, without initializing or advancing a search."""
    with (root / ".lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        plan = persistence._read(root / "plan.json")
        if plan != _plan():
            raise ValueError("plan source, runtime or scientific contract differs")
        return _report(root, plan, _load(root, plan))


def _record_attempt(
    root: Path, plan: dict[str, Any], rows: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    job_id, problem, method, budget = _roster()[index]
    last = rows[-1] if rows and rows[-1]["job_id"] == job_id else None
    state = None if last is None else copy.deepcopy(last["state"])
    before = dict.fromkeys(_COST_KEYS, 0) if state is None else _cost(state["cost"])
    previous = persistence._digest(plan if not rows else rows[-1])
    attempt = sum(i == index for i, _ in _attempts(root))
    directory = root / "runtime"
    directory.mkdir(exist_ok=True)
    prefix = directory / f"attempt-{index:03d}-{attempt:03d}"
    start = {
        "plan_sha256": persistence._digest(plan),
        "checkpoint_index": index,
        "attempt": attempt,
        "job_id": job_id,
        "budget": budget,
        "previous_sha256": previous,
        "cost_before": before,
    }
    persistence._seal(Path(f"{prefix}-start.json"), start)
    began = time.perf_counter()
    try:
        if state is None:
            state = core.initialize(problem, method)
        state = core.advance(problem, state, budget)
        _state_identity(state, problem, method, budget)
        row = {
            "job_id": job_id,
            "budget": budget,
            "previous_sha256": previous,
            "state": state,
            "report": core.summarize(problem, state),
        }
        persistence._seal(root / f"checkpoint-{index:03d}.json", row)
    except BaseException:
        after = None
        if state is not None:
            with suppress(ValueError):
                after = _cost(state.get("cost"))
        persistence._seal(
            Path(f"{prefix}-finish.json"),
            {
                "start_sha256": persistence._digest(start),
                "status": "interrupted",
                "elapsed_seconds": time.perf_counter() - began,
                "cost_after": after,
                "checkpoint_sha256": None,
            },
        )
        raise
    persistence._seal(
        Path(f"{prefix}-finish.json"),
        {
            "start_sha256": persistence._digest(start),
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - began,
            "cost_after": _cost(state["cost"]),
            "checkpoint_sha256": persistence._digest(row),
        },
    )
    return row


def run_campaign(
    root: Path, *, max_new_checkpoints: int | None = None
) -> dict[str, Any]:
    """Run or resume the fixed roster; publish deterministic scientific receipts."""
    if max_new_checkpoints is not None and (
        type(max_new_checkpoints) is not int or max_new_checkpoints < 0
    ):
        raise ValueError("max_new_checkpoints must be a nonnegative integer or None")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = _plan()
        persistence._seal(root / "plan.json", plan)
        rows = _load(root, plan)
        report = _report(root, plan, rows)
        if report["complete"]:
            if not (root / "summary.json").exists():
                persistence._seal(root / "summary.json", _summary(plan, rows))
            return report
        stop = (
            len(_roster())
            if max_new_checkpoints is None
            else min(len(_roster()), len(rows) + max_new_checkpoints)
        )
        for index in range(len(rows), stop):
            rows.append(_record_attempt(root, plan, rows, index))
        if len(rows) == len(_roster()):
            persistence._seal(root / "summary.json", _summary(plan, rows))
        return _report(root, plan, rows)
