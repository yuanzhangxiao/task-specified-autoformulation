"""Separate, bounded fits for namespace failures before numerical work began.

Original round-zero proposals, costs and terminal results remain historical.
Recovered fits are diagnostic sidecars, never substituted into later proposals.
"""

from __future__ import annotations

import fcntl
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.initialization_audit import audit_saved_initialization
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "unstarted-initialization-recovery-1"


@contextmanager
def _read_lock(directory: Path):
    """Share historical reads while excluding an original numerical worker."""
    with (directory / ".lock").open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("historical fit is still in use") from error
        yield


def _snapshot(source: Path, plan: dict, task: dict) -> dict:
    """Require a terminal supervisor failure and an untouched numerical attempt."""
    directory = io.round_path(source, task, 0)
    if not (directory / "fit/freeze.json").is_file():
        raise ValueError("saved initial fit freeze missing")
    # Use the original locks: unrelated lineages may continue during this read.
    with (
        _read_lock(directory / "supervisor"),
        _read_lock(directory),
        _read_lock(directory / "fit"),
    ):
        unexpected = sorted(
            p.name
            for p in (directory / "fit").iterdir()
            if p.name not in {"freeze.json", ".lock"}
        )
        if unexpected:
            raise ValueError(f"numerical attempt has existing artifacts: {unexpected}")
        proposal = sealed_read(directory / "proposal.json")
        result = sealed_read(directory / "result.json")
        marker = sealed_read(directory / "worker_started.json")
        if (
            marker.get("plan") != plan["artifact_sha256"]
            or marker.get("task") != task
            or marker.get("round") != 0
            or result.get("task") != task
            or result.get("round") != 0
            or result.get("status") != "worker_interrupted"
            or result.get("trial") is not None
            or result.get("selected") is not None
            or proposal.get("status") != "constructed"
            or proposal.get("task") != task
            or proposal.get("round") != 0
            or result.get("cost") != proposal.get("cost")
        ):
            raise ValueError("requires a bound, interrupted round-zero construction")
        bundle = proposal["bundle"]
        if result.get("construction_draft") != (
            proposal.get("construction_draft") or bundle
        ):
            raise ValueError("historical interrupted draft differs from its proposal")
        frozen = public._read(directory / "fit/freeze.json")
        cell = plan["cells"][task["cell"]]
        if (
            frozen["request"]
            != pipeline.request_for(bundle, plan, task, 0).model_dump(mode="json")
            or frozen["training"] != cell["training"]
            or frozen["validation"] != cell["validation"]
            or frozen["source_sha256"] != plan["source_sha256"]
        ):
            raise ValueError("historical fit differs from the saved campaign handoff")
        audit = audit_saved_initialization(directory / "fit")
        if (
            not audit["initial_observation_process_name_collisions"]
            or not audit["capability_supported"]
            or audit["source_matches"]
            or audit["boundary_comparison"]["status"] == "disagreement"
        ):
            raise ValueError(
                "requires a supported historical initializer name collision"
            )
        return {
            "task": task,
            "proposal": proposal,
            "result": result,
            "worker_started": marker,
            "fit_freeze": frozen,
            "audit": audit,
        }


def _request(snapshot: dict) -> tuple[PublicFitRequest, PublicSplit, PublicSplit]:
    """Keep the saved random seed, starting guesses, data and fitter profile."""
    frozen = snapshot["fit_freeze"]
    return (
        PublicFitRequest.model_validate(frozen["request"]),
        PublicSplit.model_validate(frozen["training"]),
        PublicSplit.model_validate(frozen["validation"]),
    )


def prepare(source: Path, root: Path, task_ids: list[str]) -> dict:
    """Freeze explicitly selected initial fits in a disjoint recovery directory."""
    source, root = source.resolve(), root.resolve()
    if source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("source and recovery roots must be disjoint")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("specify distinct task IDs")
    with io.execution_lease(source), public._lock(root):
        io.require_open(root)
        original = io.verify(source, execution=False)
        if original["protocol"] != io.ABLATION_PROTOCOL:
            raise ValueError("requires a component campaign")
        tasks = {t["task_id"]: t for t in original["tasks"]}
        if set(task_ids) - set(tasks):
            raise ValueError("task outside frozen campaign matrix")
        entries = [
            _snapshot(source, original, tasks[name]) for name in sorted(task_ids)
        ]
        for entry in entries:
            entry["recovery_fit_identity"] = public._bundle(*_request(entry))[
                "identity"
            ]
        value = {
            "protocol": PROTOCOL,
            "source_root": str(source),
            "source_plan": original,
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "entries": entries,
            "maximum_new_fit_attempts": len(entries),
            "historical_numerical_attempts_started": 0,
            "llm_calls": 0,
            "automatic_promotion": False,
            "automatic_followup": False,
            "test_data_opened": False,
        }
        if not (root / "plan.json").exists() and any(
            p.name != ".lock" for p in root.iterdir()
        ):
            raise ValueError("recovery root is nonempty without a frozen plan")
        plan = sealed_write(root / "plan.json", value)
        for entry in entries:
            public.prepare_fit(
                *_request(entry), root / "results" / entry["task"]["task_id"]
            )
        return {
            "identity": plan["artifact_sha256"],
            "tasks": [e["task"]["task_id"] for e in entries],
            "maximum_new_fit_attempts": len(entries),
            "llm_calls": 0,
            "campaign_changes": 0,
            "test_data_opened": False,
        }


def _verify(root: Path) -> dict:
    """Check exact snapshots and runtime without broadening the recovery scope."""
    plan = sealed_read(root / "plan.json")
    if (
        plan["protocol"] != PROTOCOL
        or plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("recovery source or runtime differs")
    io.require_open(root)
    source = Path(plan["source_root"])
    original = io.verify(source, execution=False)
    if original != plan["source_plan"]:
        raise ValueError("original campaign plan changed")
    for entry in plan["entries"]:
        expected = _snapshot(source, original, entry["task"])
        expected["recovery_fit_identity"] = public._bundle(*_request(expected))[
            "identity"
        ]
        if expected != entry:
            raise ValueError("historical recovery snapshot differs")
        directory = root / "results" / entry["task"]["task_id"]
        frozen, _, _, _ = public._load(directory)
        if frozen["identity"] != entry["recovery_fit_identity"]:
            raise ValueError("recovery fit handoff differs")
    return plan


def run(root: Path, task_index: int) -> dict:
    """Spend at most one original-profile numerical allowance for this saved model."""
    root = root.resolve()
    source = Path(sealed_read(root / "plan.json")["source_root"])
    with io.execution_lease(source), io.execution_lease(root):
        plan = _verify(root)
        if not 0 <= task_index < len(plan["entries"]):
            raise ValueError("task index outside recovery matrix")
        entry = plan["entries"][task_index]
        result = public.execute_fit(root / "results" / entry["task"]["task_id"])
        return {
            "task": entry["task"]["task_id"],
            "result": result.model_dump(mode="json"),
        }


def report(root: Path) -> dict:
    """Report terminal evidence without starting a fit or rewriting campaign results."""
    root = root.resolve()
    source = Path(sealed_read(root / "plan.json")["source_root"])
    with io.execution_lease(source), public._lock(root):
        plan = _verify(root)
        rows = []
        for entry in plan["entries"]:
            directory = root / "results" / entry["task"]["task_id"]
            started = (directory / "started.json").exists()
            if (directory / "result.json").exists():
                # execute_fit only validates/returns an existing immutable result.
                result = public.execute_fit(directory).model_dump(mode="json")
                row = {
                    k: result[k]
                    for k in ("status", "training", "validation", "message")
                }
            else:
                row = {"status": "started_without_result" if started else "pending"}
            rows.append(
                {
                    "task": entry["task"]["task_id"],
                    "historical_status": entry["result"]["status"],
                    "historical_cost": entry["result"]["cost"],
                    "historical_fit_identity": entry["fit_freeze"]["identity"],
                    "recovery_fit_identity": entry["recovery_fit_identity"],
                    "numerical_attempt_started": started,
                    **row,
                }
            )
        value = {
            "protocol": PROTOCOL,
            "identity": plan["artifact_sha256"],
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "rows": rows,
            "maximum_new_fit_attempts": plan["maximum_new_fit_attempts"],
            "numerical_attempts_started": sum(
                r["numerical_attempt_started"] for r in rows
            ),
            "historical_cost_scope": (
                "Saved round-zero proposal costs only; unchanged, not new usage."
            ),
            "llm_calls": 0,
            "campaign_changes": 0,
            "selection_changes": 0,
            "automatic_followup": False,
            "test_data_opened": False,
        }
        public._write(root / "summary.json", value)
        return value
