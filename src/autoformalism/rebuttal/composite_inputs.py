"""Reactor inputs that sum several shapes on each input, for held-out suites.

The reactor's Phase B protocols draw every input from one shape: a constant, a
step, a sine, or one sine per input (``phase_b_generation._vector_input``). A
held-out suite that puts staircases, ramps, pulse trains, multisines and random
binary sequences on one trajectory needs inputs that are sums of shapes.
``value`` evaluates such an input and ``boundaries`` lists its jumps and kinks.
``extended`` lets the generator's own reactor simulation read the composite
kind for the duration of a block. The generator itself is not edited: the
intervention studies pin its source by hash.

A composite specification is ``{"kind": "composite", "channels": [Cf, Tf, Tjf]}``
in the training steps' units, where each channel is a list of components:

* ``{"shape": "step", "start": a, "end": b, "value": v}``: ``v`` on ``[a, b)``,
  the convention of the generator's own steps;
* ``{"shape": "ramp", "points": [[t0, v0], [t1, v1], ...]}``: linear between
  increasing times, zero outside them;
* ``{"shape": "sine", "period": p, "amplitude": a}``: ``a sin(2 pi t / p)``
  from time zero, as the generator's own sines.

An optional ``initial_shift`` is read by the simulator as for any protocol.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from autoformalism.benchmarks import phase_b_generation as generator

KIND = "composite"
#: The reactor's inputs, in the order of the generator's input vector.
CHANNELS = ("Cf", "Tf", "Tjf")
_FIELDS = {
    "step": {"shape", "start", "end", "value"},
    "ramp": {"shape", "points"},
    "sine": {"shape", "period", "amplitude"},
}


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite")
    return float(value)


def validate(spec: Mapping[str, Any], duration: float) -> None:
    """Refuse anything but a well-formed composite input on the horizon."""
    if spec.get("kind") != KIND or set(spec) - {"kind", "channels", "initial_shift"}:
        raise ValueError("not a composite reactor input")
    shift = spec.get("initial_shift")
    if shift is not None:
        if not isinstance(shift, list) or len(shift) != len(CHANNELS):
            raise ValueError("initial_shift must give one shift per reactor state")
        for item in shift:
            _finite(item, "initial shift")
    channels = spec.get("channels")
    if not isinstance(channels, list) or len(channels) != len(CHANNELS):
        raise ValueError("a composite input gives one component list per input")
    for name, components in zip(CHANNELS, channels, strict=True):
        if not isinstance(components, list):
            raise ValueError(f"{name}: components must be a list")
        for item in components:
            shape = item.get("shape") if isinstance(item, Mapping) else None
            if shape not in _FIELDS or set(item) != _FIELDS[shape]:
                raise ValueError(f"{name}: unknown or malformed component {item!r}")
            if shape == "step":
                start = _finite(item["start"], "step start")
                end = _finite(item["end"], "step end")
                _finite(item["value"], "step value")
                if not 0 <= start < end <= duration:
                    raise ValueError(f"{name}: step outside the horizon or empty")
            elif shape == "ramp":
                points = item["points"]
                if not isinstance(points, list) or len(points) < 2:
                    raise ValueError(f"{name}: a ramp needs at least two points")
                times = []
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError(f"{name}: ramp points are [time, value]")
                    times.append(_finite(point[0], "ramp time"))
                    _finite(point[1], "ramp value")
                if times[0] < 0 or times[-1] > duration or np.any(np.diff(times) <= 0):
                    raise ValueError(f"{name}: ramp times must increase on the horizon")
            else:
                if _finite(item["period"], "sine period") <= 0:
                    raise ValueError(f"{name}: a sine period must be positive")
                _finite(item["amplitude"], "sine amplitude")


def _component(time: float, item: Mapping[str, Any]) -> float:
    shape = item["shape"]
    if shape == "step":
        return float(item["value"]) if item["start"] <= time < item["end"] else 0.0
    if shape == "ramp":
        times, values = zip(*item["points"], strict=True)
        if not times[0] <= time <= times[-1]:
            return 0.0
        return float(np.interp(time, times, values))
    return float(item["amplitude"]) * math.sin(2 * math.pi * time / item["period"])


def value(time: float, spec: Mapping[str, Any]) -> np.ndarray:
    """The input vector in training-step units at one time."""
    return np.asarray(
        [
            sum(_component(time, item) for item in channel)
            for channel in spec["channels"]
        ],
        dtype=float,
    )


def boundaries(spec: Mapping[str, Any]) -> tuple[float, ...]:
    """Every declared jump or kink: step edges and ramp points."""
    times = []
    for channel in spec["channels"]:
        for item in channel:
            if item["shape"] == "step":
                times += [item["start"], item["end"]]
            elif item["shape"] == "ramp":
                times += [point[0] for point in item["points"]]
    return tuple(sorted(set(map(float, times))))


@contextmanager
def extended() -> Iterator[None]:
    """Let the generator's reactor simulation read composite inputs.

    Every other kind still goes to the generator's own functions, unchanged.
    """
    vector, bounds = generator._vector_input, generator.forcing_boundaries

    def composite_vector(time, spec, width):
        if spec["kind"] != KIND:
            return vector(time, spec, width)
        if width != len(CHANNELS):
            raise ValueError("composite inputs are defined for the reactor only")
        return value(time, spec)

    def composite_bounds(protocol):
        spec = protocol.specification
        if protocol.family != "cstr" or spec.get("kind") != KIND:
            return bounds(protocol)
        validate(spec, protocol.duration)
        return boundaries(spec)

    generator._vector_input = composite_vector
    generator.forcing_boundaries = composite_bounds
    try:
        yield
    finally:
        generator._vector_input = vector
        generator.forcing_boundaries = bounds
