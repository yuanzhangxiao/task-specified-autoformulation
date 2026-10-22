#!/usr/bin/env python3
"""Check each LLM-SR ODE transcription against the program it transcribes.

For every sealed LLM-SR pick with a reviewed transcription, the program is
rolled out on the data grid exactly as its sealed validation score was, and
its transcription with the shared simulator under the mechanism tests'
settings (Radau, rtol 1e-7, atol 1e-9), on the training and the validation
trajectories. Reported per split: each one's normalized rollout error against
the data, and the normalized difference between the two (both scaled by the
training standard deviation, as the shared evaluator scales). For a pick of
the current values, the largest difference between the program's output and
the transcription's right-hand side on the training rows is reported too: it
should be rounding only.

Development data only: it changes nothing and opens no test file. The
programs run in the restricted worker, as the sealed rollout ran them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.expressions import baseline_validator, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal import llm_sr_transcriptions as transcriptions
from autoformalism.rebuttal.llm_sr_campaign import training_rows
from autoformalism.rebuttal.llm_sr_programs import (
    check_program,
    run_worker,
    training_outputs,
    trajectory_rows,
)
from autoformalism.rebuttal.mechanism_functional import ConstantForcing
from autoformalism.rebuttal.phase_c_vendored_campaign import load_development
from autoformalism.rebuttal.prefit_replay import sealed_read


def program_rollouts(programs, channels, context, split, step, seconds):
    """The program's grid rollout of every trajectory, as the sealed score ran it."""
    ordered = [programs[target] for target in context.targets]
    reply = run_worker(
        {
            "mode": "rollout",
            "sources": [program.source for program, _ in ordered],
            "names": [program.name for program, _ in ordered],
            "parameters": [list(parameters) for _, parameters in ordered],
            "trajectories": [
                trajectory_rows(t, channels, context.targets).tolist()
                for t in split.trajectories
            ],
            "target_columns": [list(channels).index(t) for t in context.targets],
            "step": step,
            "seconds_per_trajectory": seconds,
        },
        seconds=seconds * len(split.trajectories) + 60.0,
    )
    return [
        np.asarray(r["predictions"], dtype=float) if r["ok"] else None
        for r in reply["trajectories"]
    ]


def ode_rollout(model, parameters, trajectory, targets, seconds):
    """The transcription's free rollout, or ``None`` when it fails."""
    config = FitConfig(
        integration_backend="solve_ivp",
        integration_method="Radau",
        relative_tolerance=1e-7,
        absolute_tolerance=1e-9,
        maximum_wall_time_seconds=seconds,
        allow_derivative_regression=False,
    )
    try:
        result = simulate_trajectory(
            model,
            trajectory,
            parameters,
            {},
            config,
            deadline=time.monotonic() + seconds,
            reset_observed_states=False,
        )
    except TimeoutError:
        return None
    if not result.success:
        return None
    return np.column_stack(
        [result.states[model.state_names.index(t)] for t in targets]
    )


def normalized(first, second, scales):
    """Mean over targets of the mean squared difference in training SDs."""
    return float(
        np.mean(
            [
                np.mean(((first[:, j] - second[:, j]) / scale) ** 2)
                for j, scale in enumerate(scales)
            ]
        )
    )


def split_report(model, parameters, programs, channels, context, split, step,
                 scales, seconds):
    """Program, transcription and data, compared trajectory by trajectory."""
    program = program_rollouts(programs, channels, context, split, step, seconds)
    rows = {"program_vs_data": [], "ode_vs_data": [], "ode_vs_program": []}
    failures = {"program": 0, "ode": 0}
    for trajectory, predicted in zip(split.trajectories, program, strict=True):
        data = np.column_stack([trajectory.targets[t] for t in context.targets])
        ode = ode_rollout(model, parameters, trajectory, context.targets, seconds)
        if predicted is None:
            failures["program"] += 1
        else:
            rows["program_vs_data"].append(normalized(predicted, data, scales))
        if ode is None:
            failures["ode"] += 1
        else:
            rows["ode_vs_data"].append(normalized(ode, data, scales))
            if predicted is not None:
                rows["ode_vs_program"].append(normalized(ode, predicted, scales))
    return {
        **{
            key: float(np.mean(values)) if values else None
            for key, values in rows.items()
        },
        "worst_trajectory_ode_vs_program": max(rows["ode_vs_program"], default=None),
        "failed_rollouts": failures,
        "trajectories": len(split.trajectories),
    }


def pointwise(model, parameters, programs, channels, context, values):
    """For a pick of the current values: program output vs transcribed rhs."""
    (target,) = context.targets
    program, coefficients = programs[target]
    rows = len(values)
    outputs, _, _ = training_outputs(
        program, coefficients, values, np.zeros(rows), [rows]
    )
    largest = 0.0
    for row, output in zip(values, outputs, strict=True):
        environment = dict(zip(channels, row, strict=True))
        forcing = {k: v for k, v in environment.items() if k != target}
        rhs = model.rhs(
            0.0, [environment[target]], parameters, ConstantForcing(forcing)
        )[0]
        largest = max(largest, abs(rhs - output) / max(1.0, abs(output)))
    return largest


def check(sealed, root, reviewed, item, seconds):
    """One task's report."""
    index = item.task
    result = sealed_read(root / "results" / str(index) / "result.json")
    selection = sealed_read(root / "results" / str(index) / "native-selection.json")
    payload = selection["selection"]["selection_payload"]
    transcriptions.bind(item, payload["programs"])
    row = sealed["rows"][index]
    development, cell_context = load_development(sealed, row)
    context = transcriptions.context_for(cell_context)
    model = compile_candidate(
        transcriptions.candidate(item, context), context,
        validator=baseline_validator(),
    )
    channels = tuple(row["channels"])
    programs = {
        target: (
            check_program(record["function"], record["variables"]),
            tuple(record["parameters"]),
        )
        for target, record in payload["programs"].items()
    }
    scaling = TrainingScaler().fit(development.train).scales
    scales = [
        float(scaling[f"target:{t}"].standard_deviation) for t in context.targets
    ]
    report = {
        "task": index,
        "benchmark_id": row["benchmark_id"],
        "repetition": row["repetition"],
        "hidden_states": list(item.hidden_states),
        "sealed_validation_error": result.get("development_rollout_error"),
    }
    for name, split in (
        ("train", development.train),
        ("validation", development.validation),
    ):
        report[name] = split_report(
            model, item.parameters, programs, channels, context, split,
            float(result["grid_step"]), scales, seconds,
        )
    if not item.hidden_states:
        values, _ = training_rows(sealed["plan"], row, development.train, cell_context)
        report["largest_relative_rhs_difference"] = pointwise(
            model, item.parameters, programs, channels, context, values
        )
    return report


def main() -> None:
    """Check every transcription, or the chosen tasks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="the LLM-SR run, or a copy of its plan and results")
    parser.add_argument("--transcriptions", type=Path, required=True)
    parser.add_argument("--release", type=Path,
                        help="this machine's copy of the frozen release")
    parser.add_argument("--task", type=int, action="append",
                        help="a task to check; repeat for more (default: all)")
    parser.add_argument("--seconds", type=float, default=120.0,
                        help="time limit per trajectory and rollout")
    args = parser.parse_args()
    sealed = sealed_read(args.root / "plan.json")
    if args.release is not None:
        # The receipt and every cell's identity are still checked.
        sealed = {**sealed, "release": str(args.release)}
    reviewed = transcriptions.load(args.transcriptions)
    if reviewed.source_plan_sha256 != sealed["artifact_sha256"]:
        raise SystemExit("the transcriptions were made for another LLM-SR run")
    for item in reviewed.transcriptions:
        if args.task and item.task not in args.task:
            continue
        print(json.dumps(check(sealed, args.root, reviewed, item, args.seconds)),
              flush=True)


if __name__ == "__main__":
    main()
