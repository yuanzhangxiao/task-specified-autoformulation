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
    expected_freeze = hashlib.sha256(
        json.dumps(
            {k: v for k, v in frozen.items() if k != "identity"}, sort_keys=True
        ).encode()
    ).hexdigest()
    if expected_freeze != frozen.get("identity"):
        raise ValueError("frozen manifest identity differs")
    rows, generation, diagnostics = [], [], []

    def retained_path(collection, index, default):
        relative = frozen.get(collection, {}).get(str(index))
        if relative is None:
            return default, frozen["identity"], False
        path = (output / relative).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != frozen["assets"][relative]
        ):
            raise ValueError("retained artifact path or hash differs")
        return path, frozen["recovery_of"], True

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
        "Strict recovery means both synthetic NMSEs <= 1e-4; missing this target "
        "does not imply an unusable fit. C accepted is the initializer acceptance "
        "flag, not refinement convergence. Constraint refers to the last C iterate.",
        "",
    ]
    for case in frozen["cases"]:
        directory = output / f"generated/{case['index']:03d}"
        path, identity, retained = retained_path(
            "retained_generations", case["index"], directory / "result.json"
        )
        result = (
            json.loads(path.read_text()) if path.exists() else {"status": "missing"}
        )
        if path.exists():
            expected = hashlib.sha256(
                json.dumps(
                    [identity, "generate", case["index"]], sort_keys=True
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
        generation.append({"case": case, **result, "retained": retained})
        lines.append(
            f"Generation {case['label']}: {result['status']}; "
            f"activity={result.get('activity')}; error={result.get('error')}"
        )
        audit = result.get("scaled_reference_audit")
        if audit:
            checks = [r for rows in audit["checks"].values() for r in rows]
            maximum = max(
                r["tight_solver_agreement"]["maximum_scaled_difference"] for r in checks
            )
            lines.append(
                f"Scaled reference audit: pass={audit['pass']}; "
                f"training scale={audit['training_output_scale']}; "
                f"worst tight-solver scaled difference="
                f"{maximum}"
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
        "| Case | Data | Arm | Status | C accepted | Verified | Strict recovery | "
        "Train NMSE | Validation NMSE | Calls | Fixed-node NMSE | Constraint |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in frozen["tasks"]:
        case = frozen["cases"][task["case_index"]]
        root = output / f"results/task_{task['index']:03d}"
        path, identity, retained = retained_path(
            "retained_results", task["index"], root / "result.json"
        )
        if path.exists():
            row = json.loads(path.read_text())
            expected = hashlib.sha256(
                json.dumps([identity, task], sort_keys=True).encode()
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
            prerequisite = generation[task["case_index"]]
            if (
                case["reference_skeleton"] or task["dataset"] == "synthetic"
            ) and prerequisite["status"] in {
                "generation_failed",
                "interrupted",
                "worker_failed",
                "worker_timeout",
            }:
                row.update(
                    status="blocked_by_generation",
                    worker_status=status,
                    error=prerequisite.get("error", prerequisite["status"]),
                )
        row["retained"] = retained
        if (
            row["status"] == "missing"
            and frozen["plan"]["protocol"] == "fitter-attainability-3"
            and not task["arm"].startswith("fixed")
        ):
            gate_path = output / "preflight.json"
            if gate_path.exists() and not json.loads(gate_path.read_text()).get("pass"):
                row["status"] = "blocked_by_preflight"
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
        diagnostics.append(
            {
                "task": task,
                "case": case["label"],
                "status": row["status"],
                "retained": retained,
                "error": row.get("error"),
                "initializer_accepted": init.get("success"),
                "initializer_message": init.get("message"),
                "initializer_seconds": init.get("seconds"),
                "last_constraint": last.get("constraint_maximum"),
                "refinement_message": ref.get("message"),
                "stages": [
                    {
                        "mode": stage.get("mode"),
                        "source": stage.get("source"),
                        "error": stage.get("error"),
                        "evaluation_limit_seconds": stage.get(
                            "evaluation_limit_seconds"
                        ),
                        **{
                            k: (stage.get("result") or {}).get(k)
                            for k in (
                                "message",
                                "optimizer_success",
                                "optimizer_status",
                                "optimality",
                                "optimizer_native_success",
                                "optimizer_verified_convergence",
                                "actual_residual_calls",
                            )
                        },
                        "selected_parameters_match": bool(fit.get("parameters"))
                        and (stage.get("result") or {}).get("parameters")
                        == fit["parameters"],
                    }
                    for stage in ref.get("stages", [])
                ],
            }
        )
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
    if frozen.get("recovery_of"):
        lines[3:3] = [
            f"Retained original arms: {sum(r['retained'] for r in rows)}; "
            f"selected new reference arms: {len(frozen['selected_tasks'])}. "
            "Original identities and results are preserved; "
            "timings are not a paired speed comparison.",
            "",
        ]
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
    write(
        output / "diagnostics.json",
        {"identity": frozen["identity"], "records": diagnostics},
    )
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
                extended = (
                    json.loads((output / "freeze.json").read_text())["plan"]["protocol"]
                    == "fitter-attainability-2"
                )
                code = child.wait(timeout=1500 if generation and not extended else 2700)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                write(root / "supervisor.json", {"status": "timeout"})
                return
            write(
                root / "supervisor.json",
                {"status": "finished" if code == 0 else "failed", "exit_code": code},
            )


def smoke(output, resolution=False):
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
            collocation_target_variables=30 if resolution else None,
            collocation_minimum_intervals=20 if resolution else 1,
            sensitivity_jacobian_format="sparse" if resolution else "dense",
            recovery_retry_policy="distinct" if resolution else "repeat_best",
            recovery_prioritize_initial=resolution,
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
            "prepare-reference-recovery",
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
    elif args.action in {"prepare", "prepare-reference-recovery"}:
        plan = AttainabilityPlan.model_validate(
            json.loads(args.config.read_text()) if args.config else {}
        )
        if args.action == "prepare-reference-recovery":
            from autoformalism.rebuttal.attainability_restart import (
                prepare_reference_recovery,
            )

            frozen = prepare_reference_recovery(args.source, args.output, plan)
        else:
            frozen = prepare(plan, args.output, args.source, args.reference)
        print(
            json.dumps(
                {
                    "identity": frozen["identity"],
                    "generation_tasks": len(frozen["cases"]),
                    "fit_tasks": len(frozen["tasks"]),
                    "selected_tasks": frozen.get("selected_tasks"),
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
