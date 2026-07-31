"""Safe numerical compiler and forcing interpolation tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from autoformalism.expressions import (
    PiecewiseLinearForcing,
    RuntimeExpressionError,
    ValidationContext,
    compile_candidate,
)
from autoformalism.schemas import CandidateModel
from tests.test_candidate_validation import candidate_payload


def _compiled():
    candidate = CandidateModel.model_validate(candidate_payload())
    context = ValidationContext(
        targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
        fixed_covariates=("covariate",),
    )
    return compile_candidate(candidate, context), context


def test_piecewise_linear_forcing_interpolates_and_checks_support() -> None:
    forcing = PiecewiseLinearForcing(
        [0.0, 1.0, 2.0],
        {"u": [0.0, 2.0, 4.0]},
        allowed_channels=frozenset({"u"}),
    )

    assert forcing.value("u", 0.0) == 0.0
    assert forcing.value("u", 0.5) == 1.0
    assert forcing.value("u", 2.0) == 4.0
    with pytest.raises(RuntimeExpressionError, match="outside"):
        forcing.value("u", 2.1)
    with pytest.raises(RuntimeExpressionError, match="not allowed"):
        forcing.value("other", 1.0)


@pytest.mark.parametrize(
    ("time", "channels", "message"),
    [
        ([0.0, 0.0], {"u": [1.0, 2.0]}, "strictly increasing"),
        ([0.0, 1.0], {"u": [1.0]}, "align"),
        ([0.0, 1.0], {"u": [1.0, float("nan")]}, "nonfinite"),
        ([0.0, 1.0], {"secret": [1.0, 2.0]}, "unavailable"),
        ([0.0, 1.0], {"u": ["schedule", "schedule"]}, "numeric"),
    ],
)
def test_forcing_rejects_invalid_data(
    time: list[float],
    channels: dict[str, list[object]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        PiecewiseLinearForcing(
            time,
            channels,
            allowed_channels=frozenset({"u"}),
        )


def test_compiled_rhs_processes_and_observation_are_correct() -> None:
    model, context = _compiled()
    forcing = PiecewiseLinearForcing(
        [0.0, 1.0],
        {
            "aux": [1.0, 3.0],
            "input_u": [2.0, 4.0],
            "covariate": [7.0, 7.0],
        },
        allowed_channels=context.forcing_channels,
    )
    parameters = {"gain": 2.0, "offset": 2.0, "decay": 0.5}
    state = np.asarray([0.0, 4.0])

    rhs = model.rhs(0.5, state, parameters, forcing)
    observations = model.observe(0.5, state, parameters, forcing)

    expected_flux = 2.0 * math.log(2.0) / 4.0
    assert model.state_names == ("x", "y")
    assert model.parameter_names == ("gain", "offset", "decay")
    assert rhs == pytest.approx([3.0, expected_flux - 2.0])
    assert observations == {"target": 4.0}


def test_compiler_supports_all_approved_functions() -> None:
    payload = candidate_payload()
    payload["processes"][0]["expression"] = (
        "exp(0) + log(gain) + tanh(x) + sqrt(abs(x)) + "
        "min(gain, offset) + max(gain, offset) + sigmoid(x) + softplus(x)"
    )
    payload["state_equations"][1]["rhs"] = "flux - decay * y"
    candidate = CandidateModel.model_validate(payload)
    context = ValidationContext(
        targets=("target",),
        auxiliaries=("aux",),
        external_inputs=("input_u",),
    )
    model = compile_candidate(candidate, context)
    forcing = PiecewiseLinearForcing(
        [0.0, 1.0],
        {"input_u": [0.0, 0.0]},
        allowed_channels=context.forcing_channels,
    )

    result = model.rhs(
        0.0,
        [0.0, 0.0],
        {"gain": 1.0, "offset": 2.0, "decay": 0.5},
        forcing,
    )

    assert result[1] == pytest.approx(1 + 0 + 0 + 0 + 1 + 2 + 0.5 + math.log(2))


def test_compiled_model_rejects_runtime_shape_parameters_and_missing_forcing() -> None:
    model, context = _compiled()
    forcing = PiecewiseLinearForcing(
        [0.0, 1.0],
        {"aux": [1.0, 1.0]},
        allowed_channels=context.forcing_channels,
    )
    parameters = {"gain": 1.0, "offset": 2.0, "decay": 0.5}

    with pytest.raises(RuntimeExpressionError, match="state shape"):
        model.rhs(0.0, [1.0], parameters, forcing)
    with pytest.raises(RuntimeExpressionError, match="nonnegative"):
        model.rhs(0.0, [1.0, -1.0], parameters, forcing)
    with pytest.raises(RuntimeExpressionError, match="parameter mismatch"):
        model.rhs(0.0, [1.0, 1.0], {"gain": 1.0}, forcing)
    with pytest.raises(RuntimeExpressionError, match="outside"):
        model.rhs(
            0.0,
            [1.0, 1.0],
            {"gain": 100.0, "offset": 2.0, "decay": 0.5},
            forcing,
        )
    with pytest.raises(RuntimeExpressionError, match="missing"):
        model.rhs(0.0, [1.0, 1.0], parameters, forcing)


def test_source_contains_no_dynamic_code_execution_calls() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src/autoformalism"
    compact_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_root.rglob("*.py"))
    ).replace(" ", "")

    forbidden_call_names = ("ev" + "al(", "ex" + "ec(")
    for call_name in forbidden_call_names:
        assert call_name not in compact_sources
