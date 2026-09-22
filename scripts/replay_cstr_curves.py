#!/usr/bin/env python3
"""Replay the saved CSTR round-12 archive without fitting or test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

CELL = "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy"
PLAN_HASH = "67108d6fe6b9b70b9d0a9e0e30a0aa6fbc39cccd21e3d8a8174ebeb25d6b0cf1"
TASKS = tuple(f"cell04_seed{s}_{a}" for s in (0, 1) for a in ("full", "no_latent"))
MEMBERS = (
    "plan.json",
    *tuple(f"results/{task}/round_03/result.json" for task in TASKS),
)
SETTINGS = FitConfig(
    integration_method="Radau",
    relative_tolerance=1e-7,
    absolute_tolerance=1e-9,
    allow_derivative_regression=False,
    maximum_wall_time_seconds=300,
)


def read_archive(archive: Path, destination: Path) -> dict:
    """Copy only five known regular members, preserving and checking their seals."""
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if sorted(m.name for m in members) != sorted(MEMBERS):
            raise ValueError("archive must contain exactly the five CSTR inputs")
        if any(not m.isfile() or m.size > 32_000_000 for m in members):
            raise ValueError("archive member must be a bounded regular file")
        for member in members:
            source = stream.extractfile(member)
            assert source is not None
            payload = source.read()
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != payload:
                raise ValueError(f"saved input changed: {path}")
            if not path.exists():
                path.write_bytes(payload)
            sealed_read(path)
    return sealed_read(destination / "plan.json")


def normalized_error(
    observed: np.ndarray, predicted: np.ndarray, scale: float
) -> float:
    """Use the pooled training scale, including the initial sample."""
    if observed.shape != predicted.shape or scale <= 0:
        raise ValueError("invalid error arrays or training scale")
    if not np.isfinite(predicted).all():
        raise ValueError("nonfinite predictions")
    return float(np.mean(((predicted - observed) / scale) ** 2))


def pooled_error(records: list[dict]) -> float | None:
    """Preserve all-trajectory completeness and sample-weighted scoring."""
    if not records or any(not r["success"] for r in records):
        return None
    return sum(r["nmse"] * len(r["time"]) for r in records) / sum(
        len(r["time"]) for r in records
    )


def replay_one(model, trajectory, parameters, scale, path: Path) -> dict:
    """Checkpoint one production free rollout; completed work is never rerun."""
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
        {},
        SETTINGS,
        reset_observed_states=False,
        deadline=monotonic() + 300,
    )
    observed = trajectory.targets["T"]
    predicted = simulation.predictions.get("T")
    return sealed_write(
        path,
        {
            "trajectory_id": trajectory.trajectory_id,
            "success": simulation.success,
            "message": simulation.message,
            "time": trajectory.time.tolist(),
            "observed": observed.tolist(),
            "predicted": None if predicted is None else predicted.tolist(),
            "nmse": None
            if predicted is None
            else normalized_error(observed, predicted, scale),
            "initial_state": None
            if simulation.states is None
            else dict(
                zip(model.state_names, simulation.states[:, 0].tolist(), strict=True)
            ),
        },
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write portable numeric records for a separate plotting workflow."""
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(archive: Path, output: Path) -> dict:
    """Replay exactly the retained endpoints, with immutable data and parameters."""
    with public._lock(output):
        plan = read_archive(archive, output / "inputs")
        if plan["artifact_sha256"] != PLAN_HASH:
            raise ValueError("unexpected source plan")
        if plan["continuation"]["source_round"] != 9:
            raise ValueError("global round 12 must correspond to local round 3")
        identity = {
            "protocol": "cstr-round12-curve-replay-1",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "plan_sha256": PLAN_HASH,
            "source_sha256": public._source_identity(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": public._runtime(),
            "settings": SETTINGS.model_dump(mode="json"),
        }
        sealed_write(output / "replay_identity.json", identity)
        cell = plan["cells"][CELL]
        splits = {
            name: public.unpack_split(PublicSplit.model_validate(cell[key]))
            for name, key in (("train", "training"), ("validation", "validation"))
        }
        scale = (
            TrainingScaler().fit(splits["train"]).scales["target:T"].standard_deviation
        )
        summary, curves, individual, forcings, equations = [], [], [], [], []
        for name, split in splits.items():
            for trajectory in split.trajectories:
                for role in ("external_inputs", "auxiliaries"):
                    for channel, values in getattr(trajectory, role).items():
                        for time, value in zip(trajectory.time, values, strict=True):
                            forcings.append(
                                {
                                    "split": name,
                                    "trajectory_id": trajectory.trajectory_id,
                                    "role": role,
                                    "channel": channel,
                                    "time": float(time),
                                    "value": float(value),
                                }
                            )
        for task in TASKS:
            result = sealed_read(
                output / "inputs/results" / task / "round_03/result.json"
            )
            task_info = result["task"]
            if (
                task_info["task_id"] != task
                or task_info["cell"] != CELL
                or result["round"] != 3
            ):
                raise ValueError("checkpoint task/round mismatch")
            selected = result["selected"]
            request = PublicFitRequest.model_validate(selected["request"])
            model, _, _ = public._lower(request)
            candidate = model.validated.candidate.model_dump(mode="json")
            if (
                public.content_sha256(candidate)
                != selected["fit"]["lowered_candidate_sha256"]
            ):
                raise ValueError("lowered candidate differs from fitted model")
            parameters = selected["fit"]["parameters"]
            if set(parameters) != set(model.parameter_names):
                raise ValueError("fitted vector does not cover the exact model")
            seed, arm = task_info["seed"], task_info["arm"]
            equations.append(
                {
                    "task_id": task,
                    "candidate": candidate,
                    "parameters": parameters,
                    "origin_round": selected["origin_round"],
                    "certificate": selected["certificate"],
                    "source_result_sha256": result["artifact_sha256"],
                }
            )
            for name, split in splits.items():
                records = []
                for index, trajectory in enumerate(split.trajectories):
                    record = replay_one(
                        model,
                        trajectory,
                        parameters,
                        scale,
                        output / "replays" / task / name / f"{index:03d}.json",
                    )
                    records.append(record)
                    individual.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "split": name,
                            "trajectory_id": trajectory.trajectory_id,
                            "success": record["success"],
                            "nmse": record["nmse"],
                            "message": record["message"],
                        }
                    )
                    if not record["success"]:
                        continue
                    for time, observed, predicted in zip(
                        record["time"],
                        record["observed"],
                        record["predicted"],
                        strict=True,
                    ):
                        curves.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "split": name,
                                "trajectory_id": trajectory.trajectory_id,
                                "target": "T",
                                "time": time,
                                "observed": observed,
                                "predicted": predicted,
                                "residual": predicted - observed,
                                "training_scale": scale,
                            }
                        )
                score = pooled_error(records)
                expected = selected["fit"][
                    "training" if name == "train" else "validation"
                ]["normalized_mse"]
                row = {
                    "task_id": task,
                    "seed": seed,
                    "arm": arm,
                    "split": name,
                    "expected_nmse": expected,
                    "replay_nmse": score,
                    "matches_saved": score is not None
                    and math.isclose(score, expected, rel_tol=1e-5, abs_tol=1e-8),
                    "trajectories": len(records),
                    "failed": sum(not r["success"] for r in records),
                }
                summary.append(row)
                print(json.dumps(row), flush=True)
        report = {
            "protocol": identity["protocol"],
            "training_scale": scale,
            "rows": summary,
            "all_scores_match": all(r["matches_saved"] for r in summary),
            "live_llm_calls": 0,
            "parameter_refit_applied": False,
            "test_data_opened": False,
            "unavailable_comparisons": {
                "no_spec": "not run for this CSTR cell in this campaign",
                "specification_rejected": "archive contains retained checkpoints only",
                "external_baselines": "model artifacts not included",
            },
        }
        sealed_write(output / "summary.json", report)
        sealed_write(output / "models.json", {"models": equations})
        write_csv(output / "trajectories.csv", curves)
        write_csv(output / "trajectory_metrics.csv", individual)
        write_csv(output / "inputs_auxiliaries.csv", forcings)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.archive, args.output)
    if not report["all_scores_match"]:
        raise SystemExit("replay did not reproduce every saved score; inspect summary")


if __name__ == "__main__":
    main()
