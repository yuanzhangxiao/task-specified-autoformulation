#!/usr/bin/env python3
"""Isolated attainable/reference controls; report with the standard library only."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from pathlib import Path


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def summarize(output):
    """Retain missing/failing arms and separate oracle information from recovery."""
    frozen = json.loads((output / "freeze.json").read_text())
    rows, generation = [], []
    lines = [
        "# Fixed-skeleton attainability comparison",
        "",
        "Diagnostic only. Reference equations and known hidden boundaries are isolated "
        "from proposer/judge feedback. No test data. Native solver success, numerical "
        "verification and output recovery are separate outcomes.",
        "",
        "Full/mesh use all observations and the same input interpolation. Fixed arms "
        "optimize only state nodes; near arms are explicitly truth-assisted starts. "
        "Ordinary starts never use the generating parameters.",
        "",
    ]
    for case in frozen["cases"]:
        directory = output / f"generated/{case['index']:03d}"
        path = directory / "result.json"
        result = (
            json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        )
        if path.exists():
            expected = hashlib.sha256(
                json.dumps(
                    [frozen["identity"], "generate", case["index"]], sort_keys=True
                ).encode()
            ).hexdigest()
            if result.get("identity") != expected:
                raise ValueError("generation identity differs")
        elif (directory / "supervisor.json").exists():
            result["status"] = (
                "worker_"
                + json.loads((directory / "supervisor.json").read_text())["status"]
            )
        if (directory / "stage.json").exists():
            result["last_stage"] = json.loads((directory / "stage.json").read_text())
        generation.append({"case": case, **result})
        lines.append(
            f"Generation {case['label']}: {result['status']}; "
            f"activity={result.get('activity')}; error={result.get('error')}"
        )
        if result.get("sampled_input_truth_scores"):
            lines.append(
                "Reference parameters under sampled inputs: "
                + json.dumps(result["sampled_input_truth_scores"])
            )
        if result.get("shared_initialization_limit"):
            lines.append(
                "Shared-initialization training lower bound: "
                + json.dumps(result["shared_initialization_limit"])
            )
    lines += [
        "",
        "| Case | Data | Arm | Status | C success | Verified | Recovered | "
        "Train NMSE | Validation NMSE | Calls | Fixed-node NMSE | Constraint |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in frozen["tasks"]:
        case = frozen["cases"][task["case_index"]]
        root = output / f"results/task_{task['index']:03d}"
        path = root / "result.json"
        if path.exists():
            row = json.loads(path.read_text())
            expected = hashlib.sha256(
                json.dumps([frozen["identity"], task], sort_keys=True).encode()
            ).hexdigest()
            if row.get("identity") != expected:
                raise ValueError("saved result identity differs")
        else:
            supervisor = root / "supervisor.json"
            status = (
                "worker_" + json.loads(supervisor.read_text())["status"]
                if supervisor.exists()
                else "missing"
            )
            row = {"task": task, "case": case, "status": status}
        rows.append(row)
        fit = row.get("fit") or {}
        fixed = task["arm"].startswith("fixed")
        init = fit if fixed else fit.get("initializer") or {}
        last = (
            (fit.get("last_iteration") or {})
            if fixed
            else (((init.get("progress") or {}).get("iterations") or [{}])[-1])
        )
        ref = fit.get("refinement") or {}
        lines.append(
            f"| {case['label']} | {task['dataset']} | "
            f"{task['arm']} | {row['status']} | "
            f"{init.get('success')} | {row.get('verified')} | {row.get('recovered')} | "
            f"{(fit.get('training') or {}).get('normalized_mse')} | "
            f"{(fit.get('validation') or {}).get('normalized_mse')} | "
            f"{ref.get('actual_residual_calls')} | "
            f"{fit.get('collocation_training_nmse')} | "
            f"{last.get('constraint_maximum')} |"
        )
    counts = dict(Counter(r["status"] for r in rows))
    lines[1:1] = [f"Planned fitting arms: {len(rows)}. Status counts: {counts}.", ""]
    write(
        output / "summary.json",
        {
            "identity": frozen["identity"],
            "statuses": counts,
            "generation": generation,
            "records": rows,
            "proposer_access": False,
        },
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(counts))


def supervised(output, index, generation):
    """Kill native solvers and their process groups before the scheduler limit."""
    root = output / (
        f"generated/{index:03d}" if generation else f"results/task_{index:03d}"
    )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "worker.log").open("a") as log:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "generation-worker" if generation else "worker",
                "--output",
                str(output),
                "--task-index",
                str(index),
            ]
            child = subprocess.Popen(
                command, stdout=log, stderr=log, start_new_session=True
            )
            try:
                code = child.wait(timeout=1500 if generation else 2700)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                write(root / "supervisor.json", {"status": "timeout"})
                return
            write(
                root / "supervisor.json",
                {"status": "finished" if code == 0 else "failed", "exit_code": code},
            )


def smoke(output):
    """Verify generation, fixed-node fitting and joint fitting on a control."""
    from autoformalism.expressions import ValidationContext, compile_candidate
    from autoformalism.fitting.collocation_sensitivity import (
        CollocationSensitivityConfig,
        fit_collocation_forward_sensitivity,
    )
    from autoformalism.fitting.initialization import LatentInitializationPlan
    from autoformalism.rebuttal.attainability_controls import (
        fixed_state_fit,
        generated_problem,
        ordinary_start,
        simulate_split,
        system_for,
    )
    from autoformalism.rebuttal.initialization_campaign import synthetic_problem
    from autoformalism.rebuttal.piecewise_campaign import unpack_split
    from autoformalism.schemas import CandidateModel

    problem, _ = synthetic_problem("shared", 0.0, 1)
    truth = ordinary_start(problem)
    for name in system_for(problem)[0].initial_parameter_names:
        truth[name] = 2.0
    clean = {k: simulate_split(problem, truth, k)[0] for k in ("train", "val")}
    problem = generated_problem(problem, clean)
    fixed = fixed_state_fit(problem, truth, None, output / "smoke/fixed", 40)
    write(output / "smoke/fixed.json", fixed)
    if not fixed["success"] or fixed["collocation_training_nmse"] > 1e-5:
        raise RuntimeError("fixed-state attainable smoke failed")
    model = compile_candidate(
        CandidateModel.model_validate(problem["candidate"]),
        ValidationContext.model_validate(problem["context"]),
    )
    fit = fit_collocation_forward_sensitivity(
        model,
        unpack_split(problem["splits"]["train"]),
        unpack_split(problem["splits"]["val"]),
        CollocationSensitivityConfig(
            initializer_seconds=40,
            refinement_seconds=40,
            maximum_function_evaluations=80,
            collocation_assembly="mapped",
            collocation_node_start="rollout_or_observed",
            recovery_policy="feasible",
            recovery_handoff="best_valid",
            sensitivity_invalid_trials="reject",
            least_squares_ftol=None,
        ),
        output / "smoke/joint",
        initial_parameters=ordinary_start(problem),
        initialization_plan=LatentInitializationPlan.model_validate(
            problem["initialization_plan"]
        ),
    )
    write(output / "smoke/joint.json", fit)
    if fit["status"] != "complete" or fit["validation"]["normalized_mse"] > 1e-5:
        raise RuntimeError("joint attainable smoke failed")
    print("attainable generation, fixed-node and joint-fit smoke passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "export-reference",
            "prepare",
            "generate",
            "generation-worker",
            "run",
            "worker",
            "summarize",
            "smoke",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task-index", type=int, default=0)
    args = parser.parse_args()
    if args.action == "summarize":
        summarize(args.output)
        return
    if args.action in {"run", "generate"}:
        supervised(args.output, args.task_index, args.action == "generate")
        return
    from autoformalism.rebuttal.attainability_campaign import (
        AttainabilityPlan,
        execute,
        generate,
        prepare,
    )

    if args.action == "export-reference":
        from autoformalism.rebuttal.attainability_reference import export_reference

        print(export_reference(args.source, args.output)["identity"])
    elif args.action == "prepare":
        plan = AttainabilityPlan.model_validate(
            json.loads(args.config.read_text()) if args.config else {}
        )
        frozen = prepare(plan, args.output, args.source, args.reference)
        print(
            json.dumps(
                {
                    "identity": frozen["identity"],
                    "generation_tasks": len(frozen["cases"]),
                    "fit_tasks": len(frozen["tasks"]),
                }
            )
        )
    elif args.action == "smoke":
        smoke(args.output)
    else:
        print(
            (generate if args.action == "generation-worker" else execute)(
                args.output, args.task_index
            )["status"]
        )


if __name__ == "__main__":
    main()
