#!/usr/bin/env python3
"""Diagnostic T1 curve export from saved round-12 and baseline models; no refit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal.final_evaluation import FrozenEvaluationSubject
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.candidate import CandidateModel
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

CELLS = (
    "phase_b_dalla_man_t1_canonical_named_easy",
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy",
)
TASKS = tuple(
    f"cell{cell:02d}_seed{seed}_{arm}"
    for cell, arms in enumerate((("full", "no_latent"), ("full", "no_spec")))
    for seed in (0, 1)
    for arm in arms
)
PLAN_HASH = "67108d6fe6b9b70b9d0a9e0e30a0aa6fbc39cccd21e3d8a8174ebeb25d6b0cf1"
BASELINE_HASH = "602191cd2fc6acdc6dd3d60f8debf07afbc0eb3a38b5a3d9e08dfd73735a6425"
SETTINGS = FitConfig(
    integration_method="Radau",
    relative_tolerance=1e-7,
    absolute_tolerance=1e-9,
    maximum_wall_time_seconds=120,
    allow_derivative_regression=False,
)


def archive_inputs(archive: Path, destination: Path, members: tuple[str, ...]) -> None:
    """Copy only expected bounded regular files, never extract links or paths."""
    with tarfile.open(archive, "r:gz") as stream:
        actual = stream.getmembers()
        if sorted(m.name for m in actual) != sorted(members):
            raise ValueError("archive must contain exactly the expected members")
        if any(not m.isfile() or m.size > 32_000_000 for m in actual):
            raise ValueError("expected bounded regular archive members")
        for member in actual:
            source = stream.extractfile(member)
            assert source is not None
            payload = source.read()
            path = destination / member.name
            if path.exists() and path.read_bytes() != payload:
                raise ValueError(f"saved input changed: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(payload)
            if member.name != "summary.json":
                sealed_read(path)


def baseline_job(row: dict, cell: dict, saved: dict) -> dict:
    """Bind an adapted frozen subject only to its recorded public data identity."""
    for name, key in (("train", "training"), ("validation", "validation")):
        if row["data_identity"][name] != cell[key]["fingerprint"]:
            raise ValueError(f"baseline {name} data fingerprint differs")
    subject = FrozenEvaluationSubject.model_validate(row["subject"])
    if subject.execution_semantics != "continuous_ode_free_rollout":
        raise ValueError("this diagnostic exports continuous ODE subjects only")
    if subject.parameterization.status not in {"available", "not_required"}:
        raise ValueError("incomplete frozen parameterization")
    variant = "named" if row["benchmark_id"] == CELLS[0] else "obfuscated"
    return {
        "model_id": f"{variant}_{row['source_kind']}_rep{row['repetition']}",
        "cell": row["benchmark_id"],
        "method": row["source_kind"],
        "seed": row["repetition"],
        "candidate": subject.candidate.model_dump(mode="json"),
        "context": subject.validation_context.model_dump(mode="json"),
        "parameters": subject.parameterization.global_parameters,
        "initials": subject.parameterization.global_initial_conditions,
        "saved_training_nmse": None,
        "saved_validation_nmse": saved.get("normalized_mse"),
        "source_provenance": subject.source_provenance.model_dump(mode="json"),
        "source_audit": row["audit"],
    }


def replay_one(model, trajectory, parameters, initials, target, scale, path: Path):
    """Persist a free rollout without observed-state resets or new estimation."""
    if path.exists():
        return sealed_read(path)
    marker = path.with_suffix(".started.json")
    if marker.exists():
        raise RuntimeError(f"interrupted replay needs inspection: {marker}")
    sealed_write(marker, {"trajectory_id": trajectory.trajectory_id})
    simulation = simulate_trajectory(
        model,
        trajectory,
        parameters,
        initials,
        SETTINGS,
        deadline=monotonic() + 120,
        reset_observed_states=False,
    )
    observed = trajectory.targets[target]
    predicted = simulation.predictions.get(target)
    success = bool(
        simulation.success
        and predicted is not None
        and predicted.shape == observed.shape
        and np.isfinite(predicted).all()
    )
    return sealed_write(
        path,
        {
            "trajectory_id": trajectory.trajectory_id,
            "success": success,
            "message": simulation.message,
            "time": trajectory.time.tolist(),
            "observed": observed.tolist(),
            "predicted": predicted.tolist() if success else None,
            "nmse": float(np.mean(((predicted - observed) / scale) ** 2))
            if success
            else None,
            "initial_state": dict(
                zip(model.state_names, simulation.states[:, 0], strict=True)
            )
            if simulation.states is not None
            else None,
        },
    )


def pooled_error(records: list[dict]) -> float | None:
    """Keep failures visible; pool samples rather than averaging unequal series."""
    if not records or any(not r["success"] for r in records):
        return None
    return sum(r["nmse"] * len(r["time"]) for r in records) / sum(
        len(r["time"]) for r in records
    )


def replay_model(arguments: tuple) -> dict:
    """One worker owns all checkpoints for one model."""
    job, cell, output = arguments
    model = compile_candidate(
        CandidateModel.model_validate(job["candidate"]),
        ValidationContext.model_validate(job["context"]),
    )
    if set(model.parameter_names) != set(job["parameters"]):
        raise ValueError("fitted parameters do not match the compiled model")
    targets = job["context"]["targets"]
    if len(targets) != 1 or job["context"]["lagged_targets"]:
        raise ValueError("expected one target and no measured target forcing")
    target = targets[0]
    splits = {
        name: public.unpack_split(PublicSplit.model_validate(cell[key]))
        for name, key in (("train", "training"), ("validation", "validation"))
    }
    scale = (
        TrainingScaler()
        .fit(splits["train"])
        .scales[f"target:{target}"]
        .standard_deviation
    )
    results = {}
    for name, split in splits.items():
        records = [
            replay_one(
                model,
                trajectory,
                job["parameters"],
                job["initials"],
                target,
                scale,
                output / "replays" / job["model_id"] / name / f"{index:03d}.json",
            )
            for index, trajectory in enumerate(split.trajectories)
        ]
        results[name] = {
            "nmse": pooled_error(records),
            "complete": all(r["success"] for r in records),
            "trajectories": len(records),
            "saved_nmse": job[f"saved_{'training' if name == 'train' else name}_nmse"],
        }
    return {
        "model_id": job["model_id"],
        "cell": job["cell"],
        "method": job["method"],
        "seed": job["seed"],
        "target": target,
        "training_scale": scale,
        "latent_states": [
            s.name for s in model.validated.candidate.states if s.kind == "latent"
        ],
        "forcing_symbols": sorted(model.validated.forcing_symbols),
        **results,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Export ordinary numeric tables for a separate plotting workflow."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(ours: Path, baselines: Path, output: Path, workers: int = 1) -> dict:
    """Replay every available T1 endpoint and repetition with immutable inputs."""
    with public._lock(output):
        archive_inputs(
            ours,
            output / "inputs/ours",
            (
                "plan.json",
                *(f"results/{task}/round_03/result.json" for task in TASKS),
            ),
        )
        archive_inputs(
            baselines, output / "inputs/baselines", ("plan.json", "summary.json")
        )
        plan = sealed_read(output / "inputs/ours/plan.json")
        baseline = sealed_read(output / "inputs/baselines/plan.json")
        historical = json.loads((output / "inputs/baselines/summary.json").read_text())
        if (
            plan["artifact_sha256"] != PLAN_HASH
            or baseline["artifact_sha256"] != BASELINE_HASH
            or historical["plan_sha256"] != BASELINE_HASH
            or plan["continuation"]["source_round"] != 9
        ):
            raise ValueError("unexpected historical plan")
        identity = sealed_write(
            output / "replay_identity.json",
            {
                "protocol": "t1-frozen-curve-replay-1",
                "plan_sha256": PLAN_HASH,
                "baseline_plan_sha256": BASELINE_HASH,
                "archives_sha256": {
                    k: hashlib.sha256(p.read_bytes()).hexdigest()
                    for k, p in (("ours", ours), ("baselines", baselines))
                },
                "source_sha256": public._source_identity(),
                "runtime": public._runtime(),
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "settings": SETTINGS.model_dump(mode="json"),
                "scope": (
                    "New diagnostic replay in the current runtime, "
                    "not a historical result overwrite."
                ),
                "parameter_fitting_performed": False,
                "live_llm_calls": 0,
                "test_data_opened": False,
                "private_reference_opened": False,
            },
        )
        jobs, unavailable = [], []
        for task in TASKS:
            result = sealed_read(
                output / "inputs/ours/results" / task / "round_03/result.json"
            )
            info, selected = result["task"], result["selected"]
            if (
                info["task_id"] != task
                or info["cell"] not in CELLS
                or result["round"] != 3
            ):
                raise ValueError("checkpoint task, cell or round differs")
            if selected is None:
                unavailable.append({"model_id": task, "reason": "no selected endpoint"})
                continue
            model, _, _ = public._lower(
                PublicFitRequest.model_validate(selected["request"])
            )
            candidate = model.validated.candidate.model_dump(mode="json")
            if (
                public.content_sha256(candidate)
                != selected["fit"]["lowered_candidate_sha256"]
            ):
                raise ValueError("lowering changed the fitted model")
            jobs.append(
                {
                    "model_id": task,
                    "cell": info["cell"],
                    "method": info["arm"],
                    "seed": info["seed"],
                    "candidate": candidate,
                    "context": model.validated.context.model_dump(mode="json"),
                    "parameters": selected["fit"]["parameters"],
                    "initials": {},
                    "saved_training_nmse": selected["fit"]["training"][
                        "normalized_mse"
                    ],
                    "saved_validation_nmse": selected["fit"]["validation"][
                        "normalized_mse"
                    ],
                    "source_result_sha256": result["artifact_sha256"],
                    "certificate": selected["certificate"],
                    "origin_round": selected["origin_round"],
                }
            )
        historical_rows = {r["index"]: r for r in historical["rows"]}
        for row in baseline["rows"]:
            if row["benchmark_id"] not in CELLS or row["tier"] != "easy":
                continue
            if row["status"] != "ready":
                unavailable.append(
                    {
                        "cell": row["benchmark_id"],
                        "method": row["source_kind"],
                        "repetition": row["repetition"],
                        "reason": "source unavailable",
                    }
                )
                continue
            jobs.append(
                baseline_job(
                    row,
                    plan["cells"][row["benchmark_id"]],
                    historical_rows[row["index"]],
                )
            )
        sealed_write(
            output / "models.json", {"models": jobs, "unavailable": unavailable}
        )
        arguments = [(job, plan["cells"][job["cell"]], output) for job in jobs]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for row in executor.map(replay_model, arguments):
                rows.append(row)
                print(json.dumps(row), flush=True)
        summary = sealed_write(
            output / "summary.json",
            {
                "identity_sha256": identity["artifact_sha256"],
                "models": rows,
                "unavailable": unavailable,
                "status": "complete",
                "limitation": (
                    "Diagnostic free rollouts at unchanged saved parameters. "
                    "Supplied auxiliaries remain available. Historical source files "
                    "are not in these archives. No refitting, test access or "
                    "scientific certification."
                ),
            },
        )
        curves, scores = [], []
        for job in jobs:
            for split in ("train", "validation"):
                for path in sorted(
                    (output / "replays" / job["model_id"] / split).glob(
                        "[0-9][0-9][0-9].json"
                    )
                ):
                    record = sealed_read(path)
                    common = {
                        "model_id": job["model_id"],
                        "cell": job["cell"],
                        "split": split,
                        "trajectory_id": record["trajectory_id"],
                    }
                    scores.append(
                        {**common, "success": record["success"], "nmse": record["nmse"]}
                    )
                    for i, time in enumerate(record["time"]):
                        curves.append(
                            {
                                **common,
                                "time_min": time,
                                "observed_Gp": record["observed"][i],
                                "predicted_Gp": record["predicted"][i]
                                if record["success"]
                                else None,
                            }
                        )
        write_csv(output / "curves.csv", curves)
        write_csv(output / "trajectory_scores.csv", scores)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    run(args.ours, args.baselines, args.output, args.workers)
