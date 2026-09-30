"""Opt-in affine optimizer coordinates; model equations remain in physical units."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import OptimizeResult, least_squares

from autoformalism.data import DatasetSplit, SplitName
from autoformalism.schemas.base import FiniteFloat, StrictSchema


class AffineAxis(StrictSchema):
    """The invertible map physical = center + scale * optimizer_coordinate."""

    center: FiniteFloat
    scale: FiniteFloat = Field(gt=0)


class NumericalCoordinates(StrictSchema):
    """Frozen transforms, distinct from physical domains and observation weights."""

    policy: Literal["affine-training-coordinates-1"] = "affine-training-coordinates-1"
    parameters: Mapping[str, AffineAxis]
    states: Mapping[str, AffineAxis]
    provenance: Mapping[str, str]

    @model_validator(mode="after")
    def nonempty(self):
        if not self.parameters or not self.states:
            raise ValueError("coordinates require parameters and states")
        return self

    def arrays(self, kind: str, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Align a complete transform to the runtime's exact symbol ordering."""
        axes = self.parameters if kind == "parameters" else self.states
        if kind not in {"parameters", "states"} or set(names) != set(axes):
            raise ValueError("coordinate symbols differ from model")
        return (
            np.array([axes[n].center for n in names]),
            np.array([axes[n].scale for n in names]),
        )


def training_coordinates(
    model,
    training: DatasetSplit,
    start: Mapping[str, float],
    *,
    state_channels: Mapping[str, str] | None = None,
) -> NumericalCoordinates:
    """Use training-channel units or starting initials, never hidden labels.

    Explicit channel proxies are unit assumptions supplied by the experiment,
    not inferred physical truth. Otherwise use a direct observed channel or
    the candidate's initial values. Parameter scales use the numerical start.
    """
    from autoformalism.fitting.simulation import trajectory_initial_state

    if training.name is not SplitName.TRAIN:
        raise ValueError("coordinate design requires training only")
    if (
        set(start) != set(model.parameter_names)
        or not np.isfinite(list(start.values())).all()
    ):
        raise ValueError("coordinate start must be complete and finite")
    proxies = dict(state_channels or {})
    if set(proxies) - set(model.state_names):
        raise ValueError("unknown state channel proxy")
    initial = np.array(
        [
            trajectory_initial_state(model, row, {}, parameters=start)
            for row in training.trajectories
        ]
    )
    states, provenance = {}, {}
    for i, state in enumerate(model.state_names):
        channel = proxies.get(state, model.direct_state_observation_channels.get(state))
        if channel:
            values = np.concatenate(
                [
                    {**r.targets, **r.auxiliaries, **r.external_inputs}[channel]
                    for r in training.trajectories
                ]
            )
            scale = max(float(np.std(values)), 1.0)
            provenance[state] = f"training channel {channel}; std floor 1 in its units"
        else:
            values = initial[:, i]
            scale = max(float(np.std(values)), abs(float(np.mean(values))), 1.0)
            provenance[state] = "candidate initial guesses; no hidden trajectory"
        states[state] = AffineAxis(center=float(np.mean(values)), scale=scale)
    return NumericalCoordinates(
        parameters={
            n: AffineAxis(center=v, scale=max(abs(v), 1.0)) for n, v in start.items()
        },
        states=states,
        provenance=provenance,
    )


def coordinate_least_squares(fun, x, *, coordinates, names, **options):
    """Optimize transformed parameters while keeping oracle/callback units physical."""
    center, scale = coordinates.arrays("parameters", names)
    lower, upper = options.pop("bounds")
    jac = options.pop("jac")
    if not callable(jac):
        raise ValueError("coordinate refinement requires an explicit Jacobian")
    callback = options.pop("callback", None)

    def physical(z):
        return center + scale * z

    def observe(intermediate_result):
        if callback is not None:
            translated = OptimizeResult(intermediate_result)
            translated.x = physical(intermediate_result.x)
            callback(translated)

    result = least_squares(
        lambda z: fun(physical(z)),
        (x - center) / scale,
        jac=lambda z: jac(physical(z)) * scale[np.newaxis, :],
        bounds=((lower - center) / scale, (upper - center) / scale),
        callback=observe,
        **options,
    )
    result.x = physical(result.x)
    result.jac = result.jac / scale[np.newaxis, :]
    result.grad = result.grad / scale
    # Native optimality is intentionally retained in the coordinates optimized.
    return result
