"""Deterministic, domain-respecting start generation without reference values."""

from __future__ import annotations

import numpy as np

from autoformalism.fitting import fitter
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.identifiable_campaign import SETTINGS
from autoformalism.schemas.public_fitting import PublicFitRequest


def domain(base: dict) -> tuple[tuple, np.ndarray, np.ndarray, np.ndarray]:
    """Use exactly the rollout fitter's bounds and fixed numerical units."""
    train = public.unpack_split(TrainingOnlySplit.model_validate(base["training"]))
    model = public._lower(PublicFitRequest.model_validate(base["request"]))[0]
    variables = fitter._training_variables(model, train, SETTINGS)
    if any(not v.name.startswith("parameter:") for v in variables):
        raise ValueError("portfolio requires shared initialization parameters")
    names = tuple(v.name.removeprefix("parameter:") for v in variables)
    _, scales = NumericalCoordinates.model_validate(base["coordinates"]).arrays(
        "parameters", names
    )
    return (
        names,
        np.array([v.lower for v in variables]),
        np.array([v.upper for v in variables]),
        scales,
    )


def valid(parameters: dict, bounds: tuple) -> bool:
    """A saved point must satisfy the same finite physical parameter domain."""
    names, lower, upper, _ = bounds
    if set(parameters) != set(names):
        return False
    vector = np.array([parameters[n] for n in names], dtype=float)
    return bool(
        np.isfinite(vector).all()
        and np.all(vector >= lower)
        and np.all(vector <= upper)
    )


def generate(base: dict, spread: float) -> list[dict]:
    """Keep the generic pair and two seeded perturbations, never fitted assistance.

    Positive parameters with nonnegative lower bounds receive multiplicative
    perturbations. Signed parameters/initials receive additive coordinate noise.
    The same generic vector produces the same candidates in both portfolio arms.
    """
    bounds = domain(base)
    names, lower, upper, scales = bounds
    if not valid(base["start"], bounds):
        raise ValueError("generic pair violates the fitting domain")
    seed = int(public.content_sha256(base["start"])[:16], 16)
    random = np.random.default_rng(seed)
    original = np.array([base["start"][n] for n in names])
    points = [{"source": "generic_start", "parameters": dict(base["start"])}]
    for index in range(2):
        noise = random.normal(0, spread, len(names))
        vector = original + scales * noise
        positive = (lower >= 0) & (original > 0)
        vector[positive] = original[positive] * np.exp(noise[positive])
        vector = np.clip(vector, lower, upper)
        parameters = dict(zip(names, vector.tolist(), strict=True))
        if not valid(parameters, bounds):
            raise ValueError("nonfinite generated start")
        points.append(
            {"source": f"generic_perturbation_{index}", "parameters": parameters}
        )
    return points


def choose(candidates: list[dict]) -> dict | None:
    """Continue the lowest-loss eligible trial, breaking ties by progress/cost.

    This only applies after every candidate's initial trial. Retired candidates
    can still own the deployable incumbent; retirement never erases evidence.
    """
    eligible = [c for c in candidates if not c.get("retired") and c.get("best")]
    return (
        min(
            eligible,
            key=lambda c: (
                c["best"]["training_nmse"],
                -c.get("progress_per_second", 0),
                c["source"],
            ),
        )
        if eligible
        else None
    )
