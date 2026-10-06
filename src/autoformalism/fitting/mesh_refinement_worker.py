"""M11 training-only mesh solves and polynomial transfer; no evaluator inputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from autoformalism.fitting import adaptive_mesh as mesh
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.collocation_mesh import plan_meshes
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicFitRequest


def grids(payload: dict, target: int | None, minimum: int) -> tuple[dict, dict]:
    """Resolve a fixed resolution without dropping observations or input corners."""
    training = public.unpack_split(
        TrainingOnlySplit.model_validate(payload["training"])
    )
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(payload["request"]))[0],
        allow_piecewise=True,
    )
    planned, audit = plan_meshes(system, training, target, minimum_intervals=minimum)
    return {
        row.trajectory_id: grid.time.tolist()
        for row, grid in zip(training.trajectories, planned, strict=True)
    }, audit


def transfer(detail: dict, new_meshes: dict) -> dict:
    """Prolong the complete old Radau polynomial, including its internal node.

    No new integration or observation values seed transferred latent trajectories.
    This transfers primal guesses only, never dual variables or optimizer state.
    """
    if set(detail["trajectories"]) != set(new_meshes):
        raise ValueError("transfer trajectory identities differ")
    nodes, inner = {}, {}
    for key, new in new_meshes.items():
        old = detail["trajectories"][key]
        time = mesh.validate_mesh(old["mesh"], new)
        previous = np.asarray(old["mesh"], float)
        if not set(previous).issubset(set(time)):
            raise ValueError("refinement must retain previous boundaries")
        x, z = np.asarray(old["nodes"], float), np.asarray(old["inner_nodes"], float)
        if (
            x.ndim != 2
            or len(x) != len(previous)
            or z.shape != (len(previous) - 1, x.shape[1])
            or not np.isfinite(x).all()
            or not np.isfinite(z).all()
        ):
            raise ValueError("invalid transfer nodes")

        def at(t, previous=previous, x=x, z=z):
            indices = np.clip(
                np.searchsorted(previous, t, side="right") - 1, 0, len(previous) - 2
            )
            fractions = (t - previous[indices]) / np.diff(previous)[indices]
            return mesh.polynomial(
                x[indices], z[indices], x[indices + 1], fractions[:, None]
            ).tolist()

        nodes[key] = at(time)
        inner[key] = at(time[:-1] + np.diff(time) / 3)
    return {"nodes": nodes, "inner_nodes": inner}


def native_payload(base: dict, parameters: dict, meshes: dict, guesses: dict) -> dict:
    """Explicit allowlist keeps validation and reference labels out of the NLP."""
    TrainingOnlySplit.model_validate(base["training"])
    return {k: deepcopy(base[k]) for k in ("request", "training", "coordinates")} | {
        "start": deepcopy(parameters),
        "meshes": meshes,
        **guesses,
        "method": "collocation",
        "tolerance": 1e-7,
        "hessian_approximation": "exact",
        "collocation_dense_output": True,
        "checkpoint_mode": "compact",
        "retain_inner_nodes": True,
    }


def run(mode: str, folder: Path) -> None:
    """Existing numerical implementations under the M11 process supervisor."""
    payload = public._read(folder / "payload.json")
    TrainingOnlySplit.model_validate(payload["training"])
    if mode == "point":
        from autoformalism.fitting.bounded_screening import evaluate

        evaluate(payload, folder)
    elif mode == "nodes":
        from autoformalism.fitting.screening_diagnostic import node_worker

        node_worker(payload, folder)
    elif mode == "native":
        from autoformalism.fitting.transcription_solver import solve

        solve(payload, folder)
    elif mode in {"recovery_rollout", "recovery_check"}:
        from autoformalism.fitting.recovery_worker import certify, rollout

        (rollout if mode == "recovery_rollout" else certify)(payload, folder)
    else:
        raise ValueError("unknown mesh worker")


if __name__ == "__main__":
    import sys

    run(sys.argv[1], Path(sys.argv[2]))
