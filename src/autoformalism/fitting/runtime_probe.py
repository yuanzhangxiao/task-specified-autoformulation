"""Checkpointed fixed-vector rollout timing and derivative consistency probes."""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.fitting import fitter
from autoformalism.fitting.numerical import TrackedResidual, bounded_step
from autoformalism.fitting.simulation import RolloutProfile, simulate_trajectory
from autoformalism.fitting.stagnation import RolloutOracle, matrix_report
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.staged_topology import content_hash

METHODS = ("RK45", "DOP853", "Radau")


def save_array(path: Path, values: np.ndarray) -> str:
    """Atomically store a numerical checkpoint without pickle payloads."""
    buffer = io.BytesIO()
    np.savez_compressed(buffer, values=values)
    _write_bytes(path, buffer.getvalue())
    return sha256(path)


def read_array(path: Path, digest: str) -> np.ndarray:
    """Reject a changed numerical checkpoint before comparison or reuse."""
    if sha256(path) != digest:
        raise ValueError("numerical checkpoint digest differs")
    with np.load(path, allow_pickle=False) as data:
        return data["values"]


def residual_distance(left: np.ndarray, right: np.ndarray) -> dict:
    """Compare normalized prediction residuals, not just aggregate fit scores."""
    if left.shape != right.shape or not left.size:
        raise ValueError("residual comparison shapes differ or are empty")
    delta = left - right
    return {
        "rms": float(np.sqrt(np.mean(delta**2))),
        "maximum_absolute": float(np.max(np.abs(delta))),
    }


def measured_residual(
    model, training, parameters, scales, settings, deadline, profiles
):
    """Use production residual terms while recording each free-rollout phase."""
    pieces = []
    for trajectory in training.trajectories:
        measurement = RolloutProfile()
        try:
            simulation = simulate_trajectory(
                model,
                trajectory,
                parameters,
                {},
                settings,
                deadline=deadline,
                reset_observed_states=False,
                profile=measurement,
            )
        finally:
            profiles.append(
                {"trajectory_id": trajectory.trajectory_id, **asdict(measurement)}
            )
        if not simulation.success:
            raise ValueError(simulation.message)
        profiles[-1]["state_ranges"] = {
            name: {
                "minimum": float(np.min(simulation.states[index])),
                "maximum": float(np.max(simulation.states[index])),
            }
            for index, name in enumerate(model.state_names)
        }
        for channel in model.validated.context.targets:
            pieces.append(
                fitter._bounded_normalized_residual(
                    simulation.predictions[channel],
                    trajectory.targets[channel],
                    scales[channel],
                    settings.failure_penalty,
                )
            )
        soft = fitter._soft_constraint_residuals(model, simulation.states)
        if soft.size:
            pieces.append(np.sqrt(settings.soft_constraint_penalty_weight) * soft)
    return np.concatenate(pieces)


def rollout_case(
    root,
    identity,
    model,
    training,
    parameters,
    scales,
    settings,
    deadline,
    call_seconds,
):
    """Cache one physical repeat, including partial work on failure or timeout."""
    path = root / "result.json"
    if path.exists():
        record = read_json(path)
        if record["identity"] != identity:
            raise ValueError("rollout case identity differs")
        if record["status"] == "complete":
            read_array(root / "residual.npz", record["array_sha256"])
        return record
    profiles = []
    started = monotonic()
    record = {"identity": identity, "settings": settings.model_dump(mode="json")}
    try:
        if started >= deadline:
            raise TimeoutError("profile budget exhausted before this case")
        values = measured_residual(
            model,
            training,
            parameters,
            scales,
            settings,
            min(deadline, started + call_seconds),
            profiles,
        )
        compute_seconds = monotonic() - started
        checkpoint_started = monotonic()
        digest = save_array(root / "residual.npz", values)
        record.update(
            status="complete",
            array_sha256=digest,
            cost=float(0.5 * values @ values),
            compute_seconds=compute_seconds,
            array_checkpoint_seconds=monotonic() - checkpoint_started,
        )
    except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
        record.update(
            status="timeout" if isinstance(error, TimeoutError) else "failed",
            error=str(error),
        )
    record.update(seconds=monotonic() - started, trajectories=profiles)
    write_json(path, _finite_payload(record))
    return record


def interpreter_profile(
    root, identity, model, trajectory, parameters, settings, deadline, call_seconds
):
    """Separate intrusive Python profiling from the runtime comparison repeats."""
    path = root / "interpreter.json"
    if path.exists():
        record = read_json(path)
        if record["identity"] != identity:
            raise ValueError("interpreter profile identity differs")
        return record
    profiler = cProfile.Profile()
    started = monotonic()
    record = {
        "identity": identity,
        "trajectory_id": trajectory.trajectory_id,
        "timing_comparison_eligible": False,
    }
    try:
        if started >= deadline:
            raise TimeoutError("profile budget exhausted")
        profiler.enable()
        result = simulate_trajectory(
            model,
            trajectory,
            parameters,
            {},
            settings,
            deadline=min(deadline, started + call_seconds),
            reset_observed_states=False,
        )
        record.update(
            status="complete" if result.success else "failed", error=result.message
        )
    except (TimeoutError, ValueError, RuntimeError) as error:
        record.update(status="failed", error=str(error))
    finally:
        profiler.disable()
    statistics = pstats.Stats(profiler)
    record["seconds"] = monotonic() - started
    record["functions"] = [
        {
            "file": Path(key[0]).name,
            "line": key[1],
            "function": key[2],
            "primitive_calls": value[0],
            "calls": value[1],
            "self_seconds": value[2],
            "cumulative_seconds": value[3],
        }
        for key, value in sorted(
            statistics.stats.items(), key=lambda item: item[1][3], reverse=True
        )[:25]
    ]
    write_json(path, record)
    return record


def derivative_cases(
    root,
    identity,
    model,
    training,
    parameters,
    scales,
    settings,
    deadline,
    call_seconds,
    step_factor=1e-4,
    scale_floor=1.0,
    verification_settings=None,
    accuracy_rms=1e-6,
    accuracy_maximum=1e-5,
):
    """Compare relative/scaled probes and two scaled steps at the frozen vector."""
    root.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (root / f"attempt-{attempt}").exists():
        attempt += 1
    oracle = RolloutOracle(
        model, training, scales, settings, root / f"attempt-{attempt}", deadline
    )

    checked_points = {}
    checked_arrays = {}

    def cached(point):
        key = content_hash([identity, point.tolist()])
        folder = root / "points" / key
        if (folder / "result.json").exists():
            record = read_json(folder / "result.json")
            if record["identity"] != key:
                raise ValueError("derivative point identity differs")
            result = read_array(folder / "residual.npz", record["array_sha256"])
        else:
            # All physical evaluations remain bounded and logged.
            bounded = RolloutOracle(
                model,
                training,
                scales,
                settings,
                root / f"attempt-{attempt}" / key,
                min(deadline, monotonic() + call_seconds),
            )
            result = bounded(point)
            if bounded.failures.count:
                raise ValueError("derivative point integration failed")
            digest = save_array(folder / "residual.npz", result)
            write_json(
                folder / "result.json", {"identity": key, "array_sha256": digest}
            )
        if verification_settings is not None:
            record = rollout_case(
                folder / "verification",
                content_hash([key, "verification"]),
                model,
                training,
                dict(zip(model.parameter_names, point, strict=True)),
                scales,
                verification_settings,
                deadline,
                call_seconds,
            )
            if record["status"] != "complete":
                raise ValueError("derivative reference evaluation failed")
            other = read_array(
                folder / "verification/residual.npz", record["array_sha256"]
            )
            difference = residual_distance(result, other)
            checked_points[key] = difference
            checked_arrays[key] = other
            if (
                difference["rms"] > accuracy_rms
                or difference["maximum_absolute"] > accuracy_maximum
            ):
                raise ValueError("derivative point reference methods disagree")
        return result

    records = []
    matrices = {}
    x = oracle.vector(parameters)
    for name, policy, factor in (
        ("relative", "relative", step_factor),
        ("scaled", "scaled", step_factor),
        ("scaled_smaller", "scaled", step_factor / 10),
    ):
        target = root / f"{name}.json"
        expected = content_hash([identity, name])
        if target.exists():
            record = read_json(target)
            if record["identity"] != expected:
                raise ValueError("derivative case identity differs")
            if record["status"] == "complete":
                matrices[name] = read_array(
                    root / f"{name}.npz", record["array_sha256"]
                )
            records.append(record)
            continue
        record = {
            "identity": expected,
            "name": name,
            "policy": policy,
            "factor": factor,
        }
        try:
            if monotonic() >= deadline:
                raise TimeoutError("profile budget exhausted")
            function = TrackedResidual(cached, lambda: 0)
            base = function.at(x)
            columns = []
            verification_columns = []
            verification_base = checked_arrays.get(content_hash([identity, x.tolist()]))
            steps = []
            for index, value in enumerate(x):
                amount = (
                    factor * max(scale_floor, abs(value))
                    if policy == "scaled"
                    else factor * abs(value)
                    if value
                    else np.sqrt(np.finfo(float).eps)
                )
                step = bounded_step(
                    value, oracle.lower[index], oracle.upper[index], amount
                )
                point = x.copy()
                point[index] = np.clip(
                    value + step, oracle.lower[index], oracle.upper[index]
                )
                step = point[index] - value
                if step == 0:
                    raise ValueError("bounded derivative probe has zero displacement")
                columns.append((function(point) - base) / step)
                if verification_base is not None:
                    verification_columns.append(
                        (
                            checked_arrays[content_hash([identity, point.tolist()])]
                            - verification_base
                        )
                        / step
                    )
                steps.append(step)
            matrix = np.column_stack(columns)
            matrices[name] = matrix
            record.update(
                status="complete",
                actual_steps=steps,
                matrix=matrix_report(matrix, base),
                array_sha256=save_array(root / f"{name}.npz", matrix),
                point_reference_checks=dict(checked_points),
                jacobian_reference_relative_difference=(
                    float(
                        np.linalg.norm(matrix - np.column_stack(verification_columns))
                        / max(np.linalg.norm(matrix), 1e-300)
                    )
                    if verification_columns
                    else None
                ),
            )
        except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
            record.update(status="failed", error=str(error))
        write_json(target, _finite_payload(record))
        records.append(record)
    reference = matrices.get("scaled_smaller")
    if reference is not None:
        for record in records:
            if record["name"] in matrices:
                record["relative_difference_from_smaller_scaled"] = float(
                    np.linalg.norm(matrices[record["name"]] - reference)
                    / max(np.linalg.norm(reference), 1e-300)
                )
    return records
