#!/usr/bin/env python3
"""Prepare and supervise initial-state fitting; summarize using only the stdlib."""

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
    """Read saved JSON without importing NumPy, CasADi or opening data tables."""
    import hashlib

    def identity(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True).encode()
        ).hexdigest()

    frozen = json.loads((output / "freeze.json").read_text())
    records = []
    lines = [
        "# Physical initialization comparison",
        "",
        "Same equations, data, numerical settings and fitting budgets; "
        "only latent initialization changes.",
        "Observed initial values are measured. Initializer parameters are trained "
        "once and frozen for validation.",
        "",
        "| Case | Arm | Status | C success | Initials optimized | "
        "Train NMSE | Validation NMSE | Recovered |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for task in frozen["tasks"]:
        case = frozen["cases"][task["case_index"]]
        path = output / f"results/task_{task['index']:03d}/result.json"
        record = (
            json.loads(path.read_text())
            if path.exists()
            else {"task": task, "case": case, "status": "missing"}
        )
        if not path.exists():
            supervisor = path.parent / "supervisor.json"
            if supervisor.exists():
                record["status"] = "worker_" + json.loads(
                    supervisor.read_text()
                )["status"]
        if path.exists() and record.get("identity") != identity(
            [frozen["identity"], task]
        ):
            raise ValueError("saved result identity differs")
        records.append(record)
        fit = record.get("fit", {})
        init = fit.get("initializer", {})
        lines.append(
            f"| {case['label']} | {task['arm']} | {record['status']} | "
            f"{init.get('success')} | {init.get('initial_conditions_optimized')} | "
            f"{(fit.get('training') or {}).get('normalized_mse')} | "
            f"{(fit.get('validation') or {}).get('normalized_mse')} | "
            f"{record.get('recovered')} |"
        )
    statuses = dict(Counter(r["status"] for r in records))
    write(
        output / "summary.json",
        {"identity": frozen["identity"], "statuses": statuses, "records": records},
    )
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
    from autoformalism.rebuttal.initialization_campaign import (
        InitializationExperiment,
        execute,
        prepare,
    )

    if args.stage == "prepare":
        plan = (
            InitializationExperiment.model_validate(json.loads(args.config.read_text()))
            if args.config
            else InitializationExperiment()
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

        for kind in ("shared", "causal_map"):
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
                or result["validation"]["normalized_mse"] > 1e-6
                or not result["initializer"]["success"]
            ):
                raise SystemExit(f"Failed recovery smoke: {kind}")
        print("shared and causal-map recovery smoke passed")
    else:
        with (root / "worker.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write(root / "stage.json", {"stage": "fitting_or_replay"})
            print(execute(output, args.task_index)["status"])
            write(root / "stage.json", {"stage": "finished"})


if __name__ == "__main__":
    main()
