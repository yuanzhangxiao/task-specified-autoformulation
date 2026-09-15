"""Read-only fixed-parameter training replay for a saved two-round public fit."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from time import monotonic, time

import numpy as np

from autoformalism.data import TrainingScaler
from autoformalism.fitting import fit_convergence as lineage
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.collocation_sensitivity import CollocationSensitivityConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.fit_convergence import ConvergenceSelection
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit
from autoformalism.schemas.residual_feedback import FeedbackSelection, ResidualSettings
from autoformalism.search.residual_evidence import (
    build_residual_evidence,
    validate_residual_evidence,
)


def _seed(
    parent_path: Path, continuation_path: Path, selection: FeedbackSelection
) -> dict:
    """Reuse historical verification only; reserve no convergence budget."""
    return lineage._historical_seed(
        parent_path,
        continuation_path,
        ConvergenceSelection(
            continuation_identity=selection.continuation_identity,
            continuation_backend_sha256=selection.continuation_backend_sha256,
        ),
    )


def prepare_residual_feedback(
    parent_path: Path,
    continuation_path: Path,
    directory: Path,
    selection: FeedbackSelection | None = None,
) -> dict:
    """Seal both historical points without executing the old campaign or fitting."""
    selection = selection or FeedbackSelection()
    parent_path, continuation_path, directory = (
        p.resolve() for p in (parent_path, continuation_path, directory)
    )
    for source in (parent_path.parent, continuation_path.parent):
        if directory.is_relative_to(source) or source.is_relative_to(directory):
            raise ValueError(
                "feedback output must be separate from historical experiments"
            )
    seed = _seed(parent_path, continuation_path, selection)
    payload = {
        "protocol": "residual-feedback-freeze-1",
        "selection": selection.model_dump(mode="json"),
        "settings": ResidualSettings().model_dump(mode="json"),
        "seed": seed,
        "parent_path": str(parent_path),
        "continuation_path": str(continuation_path),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
    }
    frozen = {**payload, "identity": public.content_sha256(payload)}
    with public._lock(directory):
        path = directory / "freeze.json"
        if path.exists():
            if public._read(path) != frozen:
                raise ValueError("existing feedback freeze differs")
        else:
            if any(p.name != ".lock" for p in directory.iterdir()):
                raise ValueError("feedback directory is not empty")
            public._write(path, frozen)
    return load_residual_feedback(directory)


def _load(directory: Path) -> dict:
    frozen = lineage._read_freeze(directory / "freeze.json")
    if (
        frozen["source_sha256"] != public._source_identity()
        or frozen["runtime"] != public._runtime()
    ):
        raise ValueError("feedback source or runtime changed")
    if (
        _seed(
            Path(frozen["parent_path"]),
            Path(frozen["continuation_path"]),
            FeedbackSelection.model_validate(frozen["selection"]),
        )
        != frozen["seed"]
    ):
        raise ValueError("feedback historical seed changed")
    return frozen


def _point_request(frozen: dict, name: str, parameters: dict) -> dict:
    return {
        "identity": frozen["identity"],
        "point": name,
        "parameter_sha256": public.content_sha256(parameters),
        "training_content_sha256": public.content_sha256(
            frozen["seed"]["parent"]["freeze"]["training"]
        ),
        "maximum_seconds": frozen["selection"]["replay_seconds_per_point"],
    }


def _point(
    directory: Path,
    request: dict,
    model,
    training,
    parameters,
    settings,
    scales,
    *,
    allow_new: bool = True,
) -> dict:
    """Checkpoint one bounded replay; an interrupted point never gains a fresh clock."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "result.json"
    if path.exists():
        result = lineage._read_sealed(path)["result"]
        if result["request"] != request:
            raise ValueError("replay point lineage differs")
        return result
    if (directory / "started.json").exists() or not allow_new:
        if (directory / "started.json").exists() and public._read(
            directory / "started.json"
        )["request"] != request:
            raise ValueError("replay started identity differs")
        result = {
            "request": request,
            "status": "interrupted",
            "seconds": None,
            "predictions": {},
            "normalized_mse": None,
            "error": "Fixed replay interrupted; no automatic fresh budget",
        }
    else:
        public._write(
            directory / "started.json", {"request": request, "utc_seconds": time()}
        )
        started = monotonic()
        predictions, squared, error = {}, [], None
        try:
            for index, trajectory in enumerate(training.trajectories):
                simulation = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    settings,
                    deadline=started + request["maximum_seconds"],
                )
                if not simulation.success:
                    raise ValueError(
                        f"{trajectory.trajectory_id}: {simulation.message}"
                    )
                record = {
                    "time": simulation.time.tolist(),
                    "predictions": {
                        k: v.tolist() for k, v in simulation.predictions.items()
                    },
                }
                # The simulator computes internal states, but they are deliberately
                # absent from every exported artifact and the provider packet.
                predictions[trajectory.trajectory_id] = record
                public._write(
                    directory / f"trajectory-{index:03d}.json",
                    {"trajectory_id": trajectory.trajectory_id, **record},
                )
                for target in model.validated.context.targets:
                    residual = (
                        simulation.predictions[target] - trajectory.targets[target]
                    ) / scales[target]
                    squared.append(residual**2)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        result = {
            "request": request,
            "status": "complete" if error is None else "failed",
            "seconds": monotonic() - started,
            "predictions": predictions,
            "normalized_mse": None
            if error
            else float(np.mean(np.concatenate(squared))),
            "error": error,
        }
    public._write(path, lineage._seal(result))
    return result


def _agreement(point: dict, expected: float) -> bool:
    return point["status"] == "complete" and math.isclose(
        point["normalized_mse"], expected, rel_tol=1e-5, abs_tol=1e-8
    )


def _result(directory: Path, frozen: dict) -> dict:
    path = directory / "result.json"
    if not path.exists():
        return {
            "identity": frozen["identity"],
            "status": "prepared",
            "packet": None,
            "seed": frozen["seed"],
        }
    result = lineage._read_sealed(path)["result"]
    if result["identity"] != frozen["identity"]:
        raise ValueError("feedback result lineage differs")
    for name, digest in result.get("point_sha256", {}).items():
        point = lineage._read_sealed(directory / "points" / name / "result.json")
        if digest != point["sha256"]:
            raise ValueError("feedback replay point digest differs")
    if result["packet"] is not None:
        parent = frozen["seed"]["parent"]["freeze"]
        validate_residual_evidence(
            result["packet"],
            candidate_sha256=public.content_sha256(parent["lowered_candidate"]),
            parameters=frozen["seed"]["state"]["parameters"],
            training_content_sha256=public.content_sha256(parent["training"]),
        )
    return {**result, "seed": frozen["seed"]}


def load_residual_feedback(directory: Path) -> dict:
    """Verify the frozen request and completed packet without running numerical work."""
    return _result(directory, _load(directory))


def run_residual_feedback(directory: Path) -> dict:
    """Replay the retained and preceding fits on training only; never optimize."""
    with public._lock(directory):
        frozen = _load(directory)
        if (directory / "result.json").exists():
            return _result(directory, frozen)
        resuming = (directory / "started.json").exists()
        if resuming:
            # Recover completed point publication only, never repeat an interrupted
            # or not-yet-started numerical point after a killed exporter.
            if (
                public._read(directory / "started.json")["identity"]
                != frozen["identity"]
            ):
                raise ValueError("feedback started identity differs")
        else:
            public._write(
                directory / "started.json",
                {"identity": frozen["identity"], "utc_seconds": time()},
            )
        parent = frozen["seed"]["parent"]
        state = frozen["seed"]["state"]
        request = PublicFitRequest.model_validate(parent["freeze"]["request"])
        model, _, _ = public._lower(request)
        split = PublicSplit.model_validate(parent["freeze"]["training"])
        # unpack_split intentionally replaces the descriptive fingerprint. Restore
        # its sealed original here so repacking hashes the exact original PublicSplit.
        training = replace(public.unpack_split(split), fingerprint=split.fingerprint)
        settings = CollocationSensitivityConfig.model_validate(
            parent["freeze"]["settings"]
        ).fit_config()
        scaler = TrainingScaler().fit(training)
        scales = {
            k: scaler.scales[f"target:{k}"].standard_deviation
            for k in request.context.targets
        }
        points = {}
        for name, params in (
            ("current", state["parameters"]),
            ("previous", parent["result"]["result"]["parameters"]),
        ):
            points[name] = _point(
                directory / "points" / name,
                _point_request(frozen, name, params),
                model,
                training,
                params,
                settings,
                scales,
                allow_new=not resuming,
            )
        current, previous = points["current"], points["previous"]
        agrees = _agreement(current, state["training"]["normalized_mse"])
        previous_agrees = _agreement(
            previous, parent["result"]["result"]["training"]["normalized_mse"]
        )
        packet = None
        if agrees:
            terminal = frozen["seed"]["continuation"]["result"]["result"]
            raw = frozen["seed"]["continuation"]["backend"]
            old_cost = raw["start_check"]["cost"]
            packet = build_residual_evidence(
                training,
                request.context,
                current["predictions"],
                candidate_sha256=public.content_sha256(
                    parent["freeze"]["lowered_candidate"]
                ),
                parameters=state["parameters"],
                previous=previous["predictions"] if previous_agrees else None,
                settings=ResidualSettings.model_validate(frozen["settings"]),
                numerical_status={
                    "feedback_status": terminal["feedback_status"],
                    "native_optimizer_converged": terminal[
                        "native_optimizer_converged"
                    ],
                    "budget_exhausted": terminal["extension_budget_exhausted"],
                    "residual_calls": terminal["cumulative_residual_calls"],
                    "numerical_seconds": terminal["extension_numerical_seconds"],
                    "previous_residual_calls": parent["result"]["result"][
                        "actual_residual_calls"
                    ],
                    "recent_training_costs": [
                        v["cost"] for v in raw["optimizer"].get("iterations", [])[-6:]
                    ],
                    "relative_training_cost_drop": (old_cost - state["training_cost"])
                    / max(old_cost, 1e-30),
                },
            ).model_dump(mode="json")
        result = {
            "identity": frozen["identity"],
            "status": "ready"
            if agrees
            else ("interrupted" if resuming else "replay_failed"),
            "packet": packet,
            "current_score_agrees": agrees,
            "previous_comparison_available": previous_agrees,
            "point_sha256": {k: public.content_sha256(v) for k, v in points.items()},
            "replay_seconds": {k: v["seconds"] for k, v in points.items()},
            "errors": {k: v["error"] for k, v in points.items() if v["error"]},
            "optimization_calls": 0,
            "test_data_opened": False,
            "independent_solver_replay": False,
        }
        public._write(directory / "result.json", lineage._seal(result))
        return _result(directory, frozen)
