#!/usr/bin/env python3
"""Annotate a verified historical replay timeout without fitting or replaying it."""

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import transcription_campaign as campaign


def closeout(root: Path, index: int) -> dict:
    """Write separate closeout artifacts; original backend and summary stay frozen."""
    plan, inputs = campaign.verify(root, runtime=False)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    task = plan["tasks"][index]
    folder = root / "results" / task["task_id"]
    if (folder / "result.json").exists():
        raise ValueError("endpoint already finalized")
    backend = read_seal(folder / "backend.json")
    if backend["worker_payload_sha256"] != public.content_sha256(
        campaign.worker_payload(plan, inputs, task)
    ):
        raise ValueError("backend identity differs")
    manifest = public._read(root / "submission_manifest.json")
    if manifest["identity"]["plan_sha256"] != public.content_sha256(plan):
        raise ValueError("scheduler plan differs")
    job = manifest["jobs"]["fit"]
    if not str(job).isdecimal():
        raise ValueError("invalid fit job")
    log = root / f"logs/fit-{job}_{index}.err"
    error = log.read_text()
    if not all(
        s in error
        for s in (
            "checked = replay(",
            "fitting/qualification.py",
            "TimeoutError: fitting wall-clock limit reached",
        )
    ):
        raise ValueError("log does not establish the expected replay timeout")
    summary = public._read(root / "summary.json")
    if summary["plan_sha256"] != public.content_sha256(plan):
        raise ValueError("summary identity differs")
    old = summary["rows"][index]
    if old["task_id"] != task["task_id"] or old["status"] != "missing":
        raise ValueError("summary row is not the missing endpoint")
    result = {
        "protocol": "fitting-replay-timeout-closeout-1",
        "plan_sha256": public.content_sha256(plan),
        "backend_sha256": public.content_sha256(backend),
        "log_sha256": public.content_sha256(error),
        "source_summary_sha256": public.content_sha256(summary),
        "task_id": task["task_id"],
        "index": index,
        "status": "no_complete_replay",
        "evaluation_stop_reason": "recorded_replay_wall_budget_exhausted",
        "saved_training_screen_nmse": backend.get("training_nmse"),
        "saved_backend_seconds": backend.get("total_seconds"),
        "training_nmse": None,
        "validation_nmse": None,
        "independent_solver_agreement": None,
        "parameter_fitting_performed": False,
        "new_replay_performed": False,
        "test_data_opened": False,
        "note": "Post-hoc accounting only; no independent accuracy is inferred.",
    }
    with public._lock(root / "evaluation-closeout"):
        seal(root / "evaluation-closeout" / f"task-{index}.json", result)
        annotated = deepcopy(summary)
        annotated["rows"][index] = {**old, **result}
        counts = Counter(row["status"] for row in annotated["rows"])
        annotated.update(
            status="complete" if not counts["missing"] else "incomplete",
            status_counts=dict(counts),
            recorded=sum(row["status"] != "missing" for row in annotated["rows"]),
            closeout=result,
        )
        # Group scores are left exactly as published; do not impute missing metrics.
        public._write(root / "summary-closeout.json", annotated)
    return result


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(closeout(args.root, args.index), indent=2))
