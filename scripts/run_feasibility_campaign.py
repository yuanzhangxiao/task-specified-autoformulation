#!/usr/bin/env python3
"""Supervise fitter-feasibility experiments; summarize with the stdlib."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path


def write(path: Path, payload) -> None:
    """Atomically publish progress before importing the numerical runtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(output: Path) -> None:
    """Report stages and model features using saved files only; never run a solver."""
    import hashlib

    frozen = json.loads((output / "freeze.json").read_text())
    records, diagnostics = [], []
    lines = [
        "# Fitter feasibility and model-feature comparison",
        "",
        "All arms fit latent initialization. Equations, data and total phase "
        "budgets are paired.",
        "Legacy = previous transition; feasible = checkpoint recovery "
        "+ screened starts; "
        "branch_aware additionally targets branch coverage before collocation.",
        "Train/validation NMSE use observations; clean validation and recovery "
        "use independent "
        "synthetic references after fitting. Public candidates have no recovery label.",
        "",
        "| Case | Arm | Status | C success | C constraint max | Calls | Train NMSE | "
        "Validation NMSE | Clean validation NMSE | Recovered |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for task in frozen["tasks"]:
        case = frozen["cases"][task["case_index"]]
        directory = output / f"results/task_{task['index']:03d}"
        path = directory / "result.json"
        if path.exists():
            record = json.loads(path.read_text())
            expected = hashlib.sha256(
                json.dumps([frozen["identity"], task], sort_keys=True).encode()
            ).hexdigest()
            if record.get("identity") != expected:
                raise ValueError("saved result identity differs")
        else:
            supervisor = directory / "supervisor.json"
            status = (
                "worker_" + json.loads(supervisor.read_text())["status"]
                if supervisor.exists()
                else "missing"
            )
            record = {"task": task, "case": case, "status": status}
        records.append(record)
        fit = record.get("fit") or {}
        initial = fit.get("initializer") or {}
        progress = initial.get("progress") or {}
        if not progress and (directory / "collocation/progress.json").exists():
            progress = json.loads((directory / "collocation/progress.json").read_text())
        last = (progress.get("iterations") or [{}])[-1]
        refinement = fit.get("refinement") or {}
        clean = (
            record.get("replays", {})
            .get("val", {})
            .get("scores", {})
            .get("Radau", {})
            .get("clean_nmse")
        )
        lines.append(
            f"| {case['label']} | {task['arm']} | {record['status']} | "
            f"{initial.get('success')} | {last.get('constraint_maximum')} | "
            f"{refinement.get('actual_residual_calls')} | "
            f"{(fit.get('training') or {}).get('normalized_mse')} | "
            f"{(fit.get('validation') or {}).get('normalized_mse')} | "
            f"{clean} | {record.get('recovered')} |"
        )
        features_path = directory / "model_features.json"
        features = (
            json.loads(features_path.read_text()) if features_path.exists() else {}
        )
        diagnostics.append(
            {
                "case": case["label"],
                "arm": task["arm"],
                "status": record["status"],
                "features": features,
                "collocation_phase": progress.get("phase"),
                "collocation_last_iteration": last,
                "collocation_message": initial.get("message"),
                "collocation_construction_seconds": progress.get(
                    "construction_seconds"
                ),
                "collocation_checkpoints": progress.get("checkpoints", {}),
                "refinement_message": refinement.get("message"),
                "state_integration_succeeded": refinement.get(
                    "state_integration_succeeded"
                ),
                "augmented_integration_failed": refinement.get(
                    "augmented_integration_failed"
                ),
                "screens": refinement.get("screens", []),
                "failure_evidence": refinement.get("failure_evidence", []),
                "selected_parameters": fit.get("parameters"),
                "error": record.get("error"),
            }
        )
    statuses = dict(Counter(r["status"] for r in records))
    lines[1:1] = [f"Planned arms: {len(records)}. Status counts: {statuses}.", ""]
    write(
        output / "summary.json",
        {"identity": frozen["identity"], "statuses": statuses, "records": records},
    )
    write(output / "diagnostics.json", diagnostics)
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(statuses))
    print(output / "summary.md")


def supervised(output: Path, index: int) -> None:
    """Bound imports, native fitting and replay; kill all descendants on expiry."""
    root = output / f"results/task_{index:03d}"
    root.mkdir(parents=True, exist_ok=True)
    write(root / "stage.json", {"stage": "starting_worker"})
    with (root / "worker.log").open("a") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--output",
                str(output),
                "--task-index",
                str(index),
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=1500)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            write(
                root / "supervisor.json",
                {
                    "status": "timeout",
                    "resume": "rerun this task; saved fits/replays are retained",
                },
            )
            raise SystemExit(
                "Worker exceeded 1500 seconds; checkpoints retained"
            ) from None
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    write(
        root / "supervisor.json",
        {"status": "complete" if code == 0 else "failed", "exit_code": code},
    )
    if code:
        raise SystemExit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "run", "worker", "summarize", "smoke")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    if args.stage == "summarize":
        summarize(output)
        return
    if args.stage in {"worker", "run"}:
        frozen = json.loads((output / "freeze.json").read_text())
        if args.task_index is None or not 0 <= args.task_index < len(frozen["tasks"]):
            parser.error("valid task-index required")
        root = output / f"results/task_{args.task_index:03d}"
        root.mkdir(parents=True, exist_ok=True)
        if args.stage == "run":
            with (root / "supervisor.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                supervised(output, args.task_index)
            return
        write(root / "stage.json", {"stage": "importing_numerical_runtime"})
    from autoformalism.rebuttal.feasibility_campaign import (
        FeasibilityExperiment,
        execute,
        prepare,
    )

    if args.stage == "prepare":
        plan = (
            FeasibilityExperiment.model_validate(json.loads(args.config.read_text()))
            if args.config
            else FeasibilityExperiment()
        )
        with (output / "prepare.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            frozen = prepare(plan, output, args.source)
        print(
            json.dumps({"identity": frozen["identity"], "tasks": len(frozen["tasks"])})
        )
    elif args.stage == "smoke":
        from autoformalism.expressions import ValidationContext, compile_candidate
        from autoformalism.fitting.collocation_sensitivity import (
            CollocationSensitivityConfig,
            fit_collocation_forward_sensitivity,
        )
        from autoformalism.fitting.initialization import LatentInitializationPlan
        from autoformalism.rebuttal.initialization_campaign import synthetic_problem
        from autoformalism.rebuttal.piecewise_campaign import unpack_split
        from autoformalism.schemas import CandidateModel

        for kind in ("shared", "piecewise"):
            problem, _ = synthetic_problem(kind, 0.0, 0)
            model = compile_candidate(
                CandidateModel.model_validate(problem["candidate"]),
                ValidationContext.model_validate(problem["context"]),
            )
            result = fit_collocation_forward_sensitivity(
                model,
                unpack_split(problem["splits"]["train"]),
                unpack_split(problem["splits"]["val"]),
                CollocationSensitivityConfig(
                    initializer_seconds=40,
                    refinement_seconds=40,
                    maximum_function_evaluations=80,
                    collocation_node_start="rollout_or_observed",
                    recovery_policy="branch_aware",
                    collocation_diagnostics=True,
                    least_squares_ftol=None,
                ),
                output / "smoke" / kind,
                initial_parameters=problem["start"],
                initialization_plan=LatentInitializationPlan.model_validate(
                    problem["initialization_plan"]
                ),
            )
            write(output / "smoke" / f"{kind}.json", result)
            if (
                result["status"] != "complete"
                or result["validation"]["normalized_mse"] > 1e-5
            ):
                raise SystemExit(f"Failed feasibility smoke: {kind}")
        print("smooth and inactive-threshold recovery smoke passed")
    else:
        with (root / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "stage.json", {"stage": "fitting_or_replay"})
            print(execute(output, args.task_index)["status"])
            write(root / "stage.json", {"stage": "finished"})


if __name__ == "__main__":
    main()
