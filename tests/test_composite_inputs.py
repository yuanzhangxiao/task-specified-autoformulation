"""Composite reactor inputs: shapes, declared jumps, and the generator's own path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autoformalism.benchmarks import phase_b_generation as generator
from autoformalism.benchmarks.phase_c_protocols import phase_c_protocols
from autoformalism.rebuttal import composite_inputs as ci

SPEC = {
    "kind": "composite",
    "channels": [
        [{"shape": "step", "start": 2.0, "end": 8.0, "value": 1.0}],
        [{"shape": "ramp", "points": [[1.0, 0.0], [8.0, 3.0], [22.0, -3.0]]}],
        [
            {"shape": "sine", "period": 4.0, "amplitude": 0.5},
            {"shape": "step", "start": 5.0, "end": 5.3, "value": 1.0},
        ],
    ],
}


def _protocol(spec):
    base = next(r for r in phase_c_protocols("cstr") if r.protocol_id == "train_zero")
    return base.model_copy(
        update={"protocol_id": "test_probe_toy", "split": "test", "specification": spec}
    )


def test_each_shape_follows_the_generator_conventions():
    assert ci.value(1.9, SPEC)[0] == 0.0
    assert ci.value(2.0, SPEC)[0] == 1.0  # a step holds on [start, end)
    assert ci.value(8.0, SPEC)[0] == 0.0
    assert ci.value(0.5, SPEC)[1] == 0.0  # a ramp is zero outside its points
    assert ci.value(4.5, SPEC)[1] == pytest.approx(1.5)
    assert ci.value(15.0, SPEC)[1] == pytest.approx(0.0)
    assert ci.value(23.0, SPEC)[1] == 0.0
    assert ci.value(1.0, SPEC)[2] == pytest.approx(0.5)  # sines add to steps
    assert ci.value(5.1, SPEC)[2] == pytest.approx(
        1.0 + 0.5 * np.sin(2 * np.pi * 5.1 / 4)
    )


def test_boundaries_are_the_step_edges_and_ramp_points():
    assert ci.boundaries(SPEC) == (1.0, 2.0, 5.0, 5.3, 8.0, 22.0)


@pytest.mark.parametrize(
    "channel",
    [
        [{"shape": "pulse", "start": 1.0, "end": 2.0, "value": 1.0}],
        [{"shape": "step", "start": 1.0, "end": 2.0, "value": 1.0, "extra": 0}],
        [{"shape": "step", "start": 2.0, "end": 2.0, "value": 1.0}],
        [{"shape": "step", "start": 1.0, "end": 31.0, "value": 1.0}],
        [{"shape": "step", "start": 1.0, "end": 2.0, "value": float("nan")}],
        [{"shape": "step", "start": 1.0, "end": 2.0, "value": True}],
        [{"shape": "ramp", "points": [[1.0, 0.0], [1.0, 2.0]]}],
        [{"shape": "ramp", "points": [[1.0, 0.0]]}],
        [{"shape": "sine", "period": 0.0, "amplitude": 1.0}],
    ],
)
def test_malformed_components_are_refused(channel):
    with pytest.raises(ValueError):
        ci.validate({"kind": "composite", "channels": [channel, [], []]}, 30.0)


def test_malformed_specifications_are_refused():
    with pytest.raises(ValueError, match="one component list per input"):
        ci.validate({"kind": "composite", "channels": [[], []]}, 30.0)
    with pytest.raises(ValueError, match="not a composite"):
        ci.validate({**SPEC, "value": [0, 0, 0]}, 30.0)
    with pytest.raises(ValueError, match="initial_shift"):
        ci.validate({**SPEC, "initial_shift": [0.0, 5.0]}, 30.0)
    ci.validate({**SPEC, "initial_shift": [0.0, -10.0, 0.0]}, 30.0)


def test_the_generator_reads_composites_only_inside_the_block():
    step = {"kind": "step", "start": 5.0, "end": 25.0, "value": [1.0, 0.0, 0.0]}
    original = generator._vector_input
    with pytest.raises(ValueError):
        generator._vector_input(3.0, SPEC, 3)
    with ci.extended():
        assert np.array_equal(
            generator._vector_input(5.1, SPEC, 3), ci.value(5.1, SPEC)
        )
        assert generator.forcing_boundaries(_protocol(SPEC)) == ci.boundaries(SPEC)
        for t in (4.9, 5.0, 24.9, 25.0):
            assert np.array_equal(
                generator._vector_input(t, step, 3), original(t, step, 3)
            )
        assert generator.forcing_boundaries(_protocol(step)) == (5.0, 25.0)
        with pytest.raises(ValueError, match="horizon"):
            bad = {
                "kind": "composite",
                "channels": [
                    [],
                    [],
                    [{"shape": "step", "start": 1.0, "end": 40.0, "value": 1.0}],
                ],
            }
            generator.forcing_boundaries(_protocol(bad))
    assert generator._vector_input is original


def test_a_composite_step_equals_the_generator_step_at_every_knot():
    step = {"kind": "step", "start": 5.0, "end": 25.0, "value": [1.0, 0.0, -0.5]}
    same = {
        "kind": "composite",
        "channels": [
            [{"shape": "step", "start": 5.0, "end": 25.0, "value": 1.0}],
            [],
            [{"shape": "step", "start": 5.0, "end": 25.0, "value": -0.5}],
        ],
    }
    for t in np.round(np.arange(0.0, 30.05, 0.1), 10):
        assert np.array_equal(ci.value(t, same), generator._vector_input(t, step, 3))
    with ci.extended():
        assert generator.forcing_boundaries(
            _protocol(same)
        ) == generator.forcing_boundaries(_protocol(step))


@pytest.mark.skipif(
    not (Path("data_raw") / "benchmark5_anonymous_nonlinear_process").exists(),
    reason="needs the private reactor specification",
)
def test_a_composite_protocol_simulates_like_the_same_native_protocol():
    native = next(
        r for r in phase_c_protocols("cstr") if r.protocol_id == "train_tf_tjf_pair"
    )
    spec = native.specification
    same = {
        "kind": "composite",
        "channels": [
            [{"shape": "step", "start": spec["start"], "end": spec["end"], "value": v}]
            for v in spec["value"]
        ],
    }
    with ci.extended():
        a = generator.simulate_phase_b(native)
        b = generator.simulate_phase_b(
            native.model_copy(update={"specification": same})
        )
    assert np.array_equal(a.time, b.time)
    assert np.array_equal(a.states, b.states)
