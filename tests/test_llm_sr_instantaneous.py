"""Private baseline expression semantics without dynamic Python execution."""

import numpy as np
import pytest

from autoformalism.expressions import RuntimeExpressionError, ValidationContext
from autoformalism.expressions.parser import APPROVED_FUNCTION_ARITY
from autoformalism.fitting import llm_sr_instantaneous as replay
from autoformalism.fitting import llm_sr_numerical_control as control
from autoformalism.schemas.public_fitting import PublicSplit
from tests.test_llm_sr_numerical_control import example


def value(source, **environment):
    return replay.evaluate(replay.parse(source, set(environment)).tree, environment)


def test_real_powers_clip_and_equalities_preserve_native_rules():
    assert value("x ** p", x=4, p=1.5) == 8
    assert value("max(x, 0) ** p", x=-1, p=0.5) == 0
    assert value("clip(x, -2, 3)", x=10) == 3
    assert value("heaviside(x, 1)", x=0) == 1
    assert value("heaviside(x, 0)", x=0) == 0
    source = "dh + heaviside(-h, 1) * max(-dh, 0)"
    for h, dh, expected in ((0, -1, 0), (0, 0, 0), (0, 1, 1), (1, -1, -1)):
        assert value(source, h=h, dh=dh) == expected
    source = "dh + heaviside(-(h + dh), 0) * (-h - dh)"
    assert value(source, h=1, dh=-1) == -1
    assert value(source, h=1, dh=-2) == -1
    assert "heaviside" not in APPROVED_FUNCTION_ARITY
    assert "clip" not in APPROVED_FUNCTION_ARITY


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "x[0]",
        "x.__class__",
        "(lambda: 1)()",
        "[x for x in [1]]",
        "exp(x=1)",
        "heaviside(x)",
        "x if x else 0",
        "True",
        "clip(x, 0, 1, 2)",
        "x + unknown",
        "def rhs(x): return x",
    ],
)
def test_untrusted_syntax_is_rejected(source):
    with pytest.raises(ValueError):
        replay.parse(source, {"x"})


@pytest.mark.parametrize(
    "source,env",
    [
        ("x ** p", {"x": -1, "p": 0.5}),
        ("1 / x", {"x": 0}),
        ("exp(x)", {"x": 1000}),
        ("min(1, x)", {"x": float("nan")}),
    ],
)
def test_domains_and_nonfinite_values_fail_instead_of_clamping(source, env):
    with pytest.raises(RuntimeExpressionError):
        value(source, **env)


def test_expression_size_and_depth_are_bounded():
    with pytest.raises(ValueError):
        replay.parse("x + " * 2000 + "x", {"x"})
    with pytest.raises(ValueError):
        replay.parse("-" * 80 + "x", {"x"})


def test_later_targets_change_scores_but_never_drive_current_value_rollout():
    """Exercise nonlinear real powers and forcing through the actual adapter."""
    item, _, _, training = example()
    item = control.Transcription.model_validate(
        item.model_dump(mode="json")
        | {
            "hidden_states": {},
            "processes": {"drive": "clip(u, 0, 1)", "loss": "a*max(x, 0)**b"},
            "targets": {"x": "drive-loss"},
            "rules": ["direct"],
        }
    )
    context = ValidationContext(targets=("x",), external_inputs=("u",))
    raw = training.model_dump(mode="json")
    row = raw["rows"][0]
    row["external_inputs"] = {"u": (0.2 + 0.1 * np.sin(row["time"])).tolist()}
    original = control.replay_training(
        item,
        context,
        PublicSplit.model_validate(raw),
        control.ReplaySettings(seconds=5),
        model_factory=replay.compile_current_value,
    )
    row["targets"]["x"][1:] = [1000 + i for i in range(len(row["time"]) - 1)]
    changed = control.replay_training(
        item,
        context,
        PublicSplit.model_validate(raw),
        control.ReplaySettings(seconds=5),
        model_factory=replay.compile_current_value,
    )
    assert original["status"] == changed["status"] == "complete"
    first = original["trajectories"][0]["predictions"]["x"]
    second = changed["trajectories"][0]["predictions"]["x"]
    np.testing.assert_array_equal(first, second)
    assert np.ptp(first) > 0.1
    assert original["normalized_mse"] != changed["normalized_mse"]


@pytest.mark.parametrize("case", ["cycle", "forcing_alias", "function_alias"])
def test_current_value_process_cycles_and_symbol_ambiguity_are_rejected(case):
    item, _, _, _ = example()
    raw = item.model_dump(mode="json") | {
        "hidden_states": {},
        "targets": {"x": "-a*x"},
        "rules": ["direct"],
    }
    context = ValidationContext(targets=("x",))
    if case == "cycle":
        raw["processes"] = {"first": "second+x", "second": "first-x"}
        match = "cyclic current-value processes"
    elif case == "forcing_alias":
        context = ValidationContext(targets=("x",), external_inputs=("a",))
        match = "ambiguous current-value symbol"
    else:
        raw["parameters"]["exp"] = 1.0
        match = "ambiguous current-value symbol"
    with pytest.raises(ValueError, match=match):
        replay.compile_current_value(control.Transcription.model_validate(raw), context)
