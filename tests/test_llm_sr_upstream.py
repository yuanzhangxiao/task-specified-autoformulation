"""Converting an LLM-SR program into the grammar the evaluator can score.

LLM-SR evolves the body of a Python function, not an expression. Some bodies
have an exact expression equivalent and some genuinely do not; a loop over
timesteps is a real discovery our evaluator cannot represent. What matters is
that the second kind is refused by name and counted, never silently dropped or
approximated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.expressions.parser import RestrictedParser, baseline_parser
from autoformalism.rebuttal.llm_sr_upstream import (
    InexpressibleProgram,
    convert_program,
    sample_yield,
)

PARAMS = {index: float(index + 1) / 2 for index in range(10)}
INPUTS = {"x": "G", "v": "I"}


def _converted(body: str) -> str:
    """Convert, and require the result to survive the restricted parser."""
    result = convert_program(body, INPUTS, PARAMS)
    RestrictedParser().parse(result.expression, location="test")
    return result.expression


def test_the_published_skeleton_converts_with_its_fitted_values() -> None:
    """This is the shape every LLM-SR specification starts from."""
    assert _converted(
        "dv = params[0] * x + params[1] * v + params[2]\nreturn dv"
    ) == "0.5 * G + 1.0 * I + 1.5"


def test_intermediate_assignments_are_inlined() -> None:
    """A body that names subexpressions is still one expression underneath."""
    assert _converted(
        "a = np.exp(params[0] * x)\nb = params[1] / (1.0 + a)\nreturn b"
    ) == "1.0 / (1.0 + exp(0.5 * G))"


def test_numpy_spellings_map_onto_approved_functions() -> None:
    assert _converted('"""doc."""\nreturn np.maximum(params[0] * x, params[1])') == (
        "max(0.5 * G, 1.0)"
    )
    assert _converted("return np.absolute(v) + np.sqrt(params[3] * x)") == (
        "abs(I) + sqrt(2.0 * G)"
    )


def test_which_parameters_and_channels_were_used_is_reported() -> None:
    """The caller needs to know what the model actually depends on."""
    result = convert_program("return params[2] * v", INPUTS, PARAMS)
    assert result.used_parameters == (2,)
    assert result.used_inputs == ("v",)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # a loop over timesteps: a real discovery, but not an equation
        ("out = 0\nfor i in range(len(x)):\n    out = out + x[i]\nreturn out",
         "a range over the rows"),
        ("out = 0\nfor value in x:\n    out = out + value\nreturn out",
         "a loop over the rows"),
        ("out = x\nwhile out.max() > 1:\n    out = out / 2\nreturn out", "a loop"),
        ("if x > 0:\n    return x\nreturn v", "a branch on the data"),
        ("return np.besselj(params[0] * x)", "not approved"),
        ("return np.convolve(x, np.exp(-params[0] * v))", "not approved"),
        ("return params[0] * z", "unknown name"),
        ("return x.T * params[0]", "attribute access"),
        ("return x.mean() * params[0]", "attribute access"),
        ("return np.mean(x) * params[0]", "a reduction over the rows"),
        ("return params[int(v[0])] * x", "indexes the data rows"),
        ("return x[1:] - x[:-1]", "indexes the data rows"),
        ("return [params[0] * y for y in x]", "comprehension over the rows"),
        ("return params[0] * x if x > 0 else params[1]", "conditional"),
        ("return np.where(x > params[0], x, 0.0)", "a comparison on the data"),
        ("try:\n    y = x\nexcept Exception:\n    y = v\nreturn y",
         "exception handling"),
        ("import scipy\nreturn x", "other than numpy or math"),
        ("def f(n):\n    return f(n)\nreturn f(1)", "nested too deeply"),
        # beyond the vector their specification declares
        ("return params[99] * x", "beyond the declared vector"),
        ("x = (", "does not parse"),
        ("pass", "never returns"),
        ("a = params[0] * x", "never returns"),
        ("return [x, v]", "several values"),
        ("return params[0] * np.inf", "not finite"),
    ],
)
def test_a_program_outside_the_grammar_is_refused_by_name(
    body: str, expected: str
) -> None:
    with pytest.raises(InexpressibleProgram, match=expected):
        convert_program(body, INPUTS, PARAMS)


def test_nothing_is_executed_during_conversion(monkeypatch) -> None:
    """The point of parsing rather than running is that nothing runs.

    LLM-SR's own evaluator exec()s these bodies; our conversion must not, or
    the frozen evaluation would be executing proposer-generated code too.
    """
    import builtins

    # `compile` is excluded deliberately: ast.parse uses it with PyCF_ONLY_AST
    # to build a tree, which runs none of the parsed code.
    for name in ("eval", "exec"):
        monkeypatch.setattr(
            builtins, name,
            lambda *a, _n=name, **k: pytest.fail(f"{_n} called during conversion"),
        )
    assert _converted("return params[0] * x + params[1]")


# --- recovering a model without executing it ------------------------------


def test_a_symbolic_conversion_keeps_coefficients_fittable() -> None:
    """Their evaluate fits the coefficients and then returns only the loss.

    So a recovered program arrives with its structure but not its values, and
    the conversion has to leave them as symbols until they are refitted.
    """
    from autoformalism.rebuttal.llm_sr_upstream import convert_program

    result = convert_program("return params[0] * x + params[2]", INPUTS)
    assert result.expression == "p_0 * G + p_2"
    assert result.used_parameters == (0, 2)


def test_the_discarded_coefficients_are_recovered_from_train_data() -> None:
    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import (
        convert_program,
        evaluate_expression,
        refit_program,
    )

    body = "dv = params[0] * x + params[1] * v + params[2]\nreturn dv"
    rng = np.random.default_rng(0)
    channels = {"G": rng.normal(size=200), "I": rng.normal(size=200)}
    target = -0.7 * channels["G"] + 0.3 * channels["I"] + 1.4

    fitted = refit_program(body, INPUTS, channels, target)
    assert fitted is not None
    # their evaluate fits the whole vector; an unused entry stays at its start
    assert sorted(fitted) == list(range(10)) and fitted[9] == 1.0
    final = convert_program(body, INPUTS, fitted)
    predicted = evaluate_expression(final.expression, channels)
    assert float(np.max(np.abs(predicted - target))) < 1e-5


def test_evaluation_never_reaches_eval_or_exec(monkeypatch) -> None:
    """Their sandbox execs these bodies; recovering a model must not."""
    import builtins

    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import evaluate_expression

    for name in ("eval", "exec"):
        monkeypatch.setattr(
            builtins, name,
            lambda *a, _n=name, **k: pytest.fail(f"{_n} called"),
        )
    value = evaluate_expression(
        "exp(G) + max(I, 0.5) - p_0", {"G": np.zeros(3), "I": np.ones(3), "p_0": 1.0}
    )
    assert value.shape == (3,)


def test_the_best_sample_is_the_highest_score(tmp_path: Path) -> None:
    """Their score is negative MSE, so larger is better."""
    import json

    from autoformalism.rebuttal.llm_sr_upstream import best_sample

    samples = tmp_path / "samples"
    samples.mkdir()
    for order, score in ((1, -9.0), (2, -0.5), (3, None), (4, -3.0)):
        (samples / f"samples_{order}.json").write_text(
            json.dumps({"sample_order": order, "function": f"f{order}", "score": score})
        )
    (samples / "samples_5.json").write_text("{ not json")
    best = best_sample(tmp_path)
    assert best["sample_order"] == 2 and best["score"] == -0.5
    assert best_sample(tmp_path / "absent") is None


def test_the_rendered_specification_is_valid_and_round_trips() -> None:
    """It is an input to LLM-SR, so it must satisfy their own runner."""
    import ast

    from autoformalism.rebuttal.llm_sr_upstream import (
        build_specification,
        convert_program,
        equation_body,
    )

    specification, mapping = build_specification(
        "Recover the appearance rate.", ("G", "I"), "G"
    )
    tree = ast.parse(specification)
    functions = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert functions == ["evaluate", "equation"]
    assert "@evaluate.run" in specification and "@equation.evolve" in specification
    # the seed skeleton is itself expressible, so a run that never improves
    # still yields something the evaluator can score
    body = equation_body(specification.split("@equation.evolve")[1])
    assert convert_program(body, mapping).expression == "p_0 * G + p_1 * I + p_2"


def test_a_target_outside_the_channels_is_refused() -> None:
    from autoformalism.rebuttal.llm_sr_upstream import build_specification

    with pytest.raises(ValueError, match="not among the channels"):
        build_specification("x", ("G", "I"), "missing")


def test_channel_names_that_collide_as_identifiers_are_refused() -> None:
    """Silently merging two channels would mislabel what the model reads."""
    from autoformalism.rebuttal.llm_sr_upstream import build_specification

    with pytest.raises(ValueError, match="collide as identifiers"):
        build_specification("x", ("a b", "a-b"), "a b")


def test_more_channels_than_the_starting_parameters_cover_is_refused() -> None:
    """Supplied inputs widen the signature; the skeleton must still be valid."""
    from autoformalism.rebuttal.llm_sr_upstream import build_specification

    channels = tuple(f"c{index}" for index in range(10))
    with pytest.raises(ValueError, match="starting parameters"):
        build_specification("A task.", channels, "c0")
    nine = channels[:9]
    specification, _ = build_specification("A task.", nine, "c0")
    assert "params[9]" in specification


def test_a_real_power_law_with_named_constants_is_recovered() -> None:
    """`params[0] * G ** params[1]` is a power law; its exponent is refitted."""
    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import refit_program

    body = "return params[0] * x ** params[1] + np.cos(np.pi * v) + np.arctan(v)"
    symbolic = convert_program(body, INPUTS)
    assert "3.141592653589793" in symbolic.expression
    x, v = np.linspace(1.0, 3.0, 60), np.linspace(0.0, 1.0, 60)
    target = -0.5 * x**0.7 + np.cos(np.pi * v) + np.arctan(v)
    fitted = refit_program(body, INPUTS, {"G": x, "I": v}, target)
    assert fitted[1] == pytest.approx(0.7, abs=1e-4)
    expression = convert_program(body, INPUTS, fitted).expression
    baseline_parser().parse(expression, location="test")
    with pytest.raises(ModelValidationError):
        RestrictedParser().parse(expression, location="ours")


def test_the_yield_counts_only_samples_the_model_wrote(tmp_path: Path) -> None:
    """Sample 0 is the specification's own program, which always scores.

    Counting it would let a search whose every reply was unreadable look as if
    something had scored, which is how the first budget pilot looked complete.
    """
    import json

    samples = tmp_path / "llmsr-G" / "samples"
    samples.mkdir(parents=True)
    for order, score in ((0, -1.2), (2, -0.5), (3, None), (4, float("nan"))):
        (samples / f"samples_{order}.json").write_text(
            json.dumps({"sample_order": order, "function": "f", "score": score})
        )
    (samples / "samples_5.json").write_text("{torn")
    assert sample_yield(tmp_path / "llmsr-G") == {
        "model_samples": 3,
        "model_samples_scored": 1,
    }
    assert sample_yield(tmp_path / "absent") == {
        "model_samples": 0,
        "model_samples_scored": 0,
    }


# --- the guards a scored program carries ----------------------------------------
#
# Bodies the model wrote in the budget pilot guard their reading of `params`
# against a vector shorter than ten, or check the inputs' shapes. Their
# specification always passes ten entries and arrays of one shape, so each
# guard has one outcome, known without the data, and the body is the equation
# the guard protects.


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # a conditional on the vector's length
        ("a = params[0] if len(params) > 0 else 500.0\nreturn a * x", "0.5 * G"),
        ("a = float(params[1]) if params.size > 1 else 1.0\nreturn a * v",
         "1.0 * I"),
        # a helper that reads a coefficient with a default
        ("def _p(i: int, default: float) -> float:\n"
         "    return float(params[i]) if i < len(params) else default\n"
         "return _p(0, 1.0) * x + _p(12, 2.0)", "0.5 * G + 2.0"),
        # a padded copy, then unpacking
        ("p = np.ones(MAX_NPARAMS)\nn = min(len(params), 3)\np[:n] = params[:n]\n"
         "k1, k2, k3 = p[:3]\nreturn k1 * x + k2 * v - k3",
         "0.5 * G + 1.0 * I - 1.5"),
        ("a, b = params[:1].tolist() + [2.0]\nreturn a * x + b", "0.5 * G + 2.0"),
        # guards on the inputs' shapes and on the vector
        ("if not x.shape == v.shape:\n    raise ValueError('shapes differ')\n"
         "if params is None or params.shape[0] > MAX_NPARAMS:\n"
         "    params = params[:MAX_NPARAMS]\nreturn params[0] * x", "0.5 * G"),
        # conversions that change no value
        ("q = np.asarray(params, dtype=float)\nreturn np.float64(q[0]) * x",
         "0.5 * G"),
        ("eps = np.finfo(float).eps\nreturn x / (params[0] + eps)",
         "G / 0.5000000000000002"),
        # a loop over coefficients, not over rows
        ("out = 0.0\nfor i in range(3):\n    out += params[i] * x ** i\nreturn out",
         "0.0 + 0.5 * G ** 0 + 1.0 * G ** 1 + 1.5 * G ** 2"),
        # a one-sided clip is a max
        ("return np.clip(params[0] * x, 0.0, np.inf)", "max(0.5 * G, 0.0)"),
        ("return np.clip(x, params[0], params[1])", "clip(G, 0.5, 1.0)"),
    ],
)
def test_a_guard_with_one_known_outcome_reduces_to_the_equation(
    body: str, expected: str
) -> None:
    expression = convert_program(body, INPUTS, PARAMS).expression
    baseline_parser().parse(expression, location="test")
    assert expression == expected


def test_a_compound_local_used_twice_is_inlined_whole() -> None:
    """Reusing a named subexpression once failed as 'unknown name: p_1'."""
    body = "d = x - params[1]\nreturn params[0] * d / (1.0 + params[2] * d)"
    assert convert_program(body, INPUTS).expression == (
        "p_0 * (G - p_1) / (1.0 + p_2 * (G - p_1))"
    )
    assert _converted(body) == "0.5 * (G - 1.0) / (1.0 + 1.5 * (G - 1.0))"


def test_a_name_rebound_in_terms_of_itself_reads_its_previous_value() -> None:
    """`k = abs(k)` once recursed without end, which would have crashed sealing."""
    body = "k = params[0]\nk = abs(k)\nk = k + 1.0\nreturn k * x"
    assert convert_program(body, INPUTS).expression == "(abs(p_0) + 1.0) * G"
    assert _converted(body) == "1.5 * G"


def test_a_negative_coefficient_keeps_its_sign_under_a_power() -> None:
    """`(-0.5) ** G` must not be written as `-0.5 ** G`, which is -(0.5 ** G)."""
    negative = {**PARAMS, 0: -0.5}
    assert convert_program("return params[0] ** x", INPUTS, negative).expression == (
        "(-0.5) ** G"
    )


def test_a_branch_on_a_coefficient_is_resolved_by_the_fitted_value() -> None:
    """A guard on a coefficient has one outcome once the coefficient is fitted."""
    from autoformalism.rebuttal.llm_sr_upstream import CoefficientBranch

    body = "K = params[1]\nif K == 0:\n    K = 1e-6\nreturn params[0] * x / (K + x)"
    with pytest.raises(CoefficientBranch, match="fitted coefficient"):
        convert_program(body, INPUTS)
    assert _converted(body) == "0.5 * G / (1.0 + G)"
    zero = {**PARAMS, 1: 0.0}
    assert convert_program(body, INPUTS, zero).expression == "0.5 * G / (1e-06 + G)"


def test_the_refit_follows_a_branch_on_a_coefficient() -> None:
    """Their BFGS reads the body at each trial vector, and so does the refit."""
    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import refit_program, training_error

    rng = np.random.default_rng(1)
    channels = {"G": rng.uniform(1.0, 3.0, 300), "I": rng.uniform(0.0, 1.0, 300)}
    target = 2.0 * channels["G"] / (0.5 + channels["G"]) - 0.3 * channels["I"]
    body = (
        "K = params[1]\nif K == 0:\n    K = 1e-6\n"
        "return params[0] * x / (K + x) + params[2] * v"
    )
    fitted = refit_program(body, INPUTS, channels, target)
    assert fitted[0] == pytest.approx(2.0, abs=1e-4)
    assert fitted[1] == pytest.approx(0.5, abs=1e-4)
    expression = convert_program(body, INPUTS, fitted).expression
    assert training_error(expression, channels, target) < 1e-10


def test_a_body_that_is_not_an_equation_has_no_refit() -> None:
    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import refit_program

    channels = {"G": np.ones(5), "I": np.ones(5)}
    assert refit_program("return np.cumsum(x) * params[0]", INPUTS, channels,
                         np.ones(5)) is None


def test_the_refit_executes_nothing(monkeypatch) -> None:
    import builtins

    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import refit_program

    for name in ("eval", "exec"):
        monkeypatch.setattr(
            builtins, name,
            lambda *a, _n=name, **k: pytest.fail(f"{_n} called during the refit"),
        )
    grid = np.arange(5.0)
    fitted = refit_program(
        "return params[0] * x", INPUTS, {"G": grid, "I": np.zeros(5)}, 2.0 * grid
    )
    assert fitted[0] == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("a = x\n" + "a = a + a\n" * 20 + "return a", "too large"),
        ("a = 0\nfor i in range(1000):\n    for j in range(1000):\n"
         "        a = a + 1\nreturn a * x", "too long to read"),
        ("return " + "-" * 2000 + "x", "nested too deeply|too large|does not parse"),
    ],
)
def test_no_body_can_make_the_reader_run_away(body: str, expected: str) -> None:
    """Every body ends in an expression or a refusal, never a crash or a hang."""
    with pytest.raises(InexpressibleProgram, match=expected):
        convert_program(body, INPUTS, PARAMS)


def test_the_starting_variable_is_named_after_the_target() -> None:
    """Upstream names it after the derivative it computes, as in its `dv`."""
    from autoformalism.rebuttal.llm_sr_upstream import build_specification

    specification, _ = build_specification("A task.", ("G", "I"), "I")
    assert (
        "dI = params[0] * G + params[1] * I + params[2]\n    return dI"
        in specification
    )
    assert "dG" not in specification


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # guards that leave every finite value as it is, as nan_to_num does
        ("d = params[0] * x\nreturn np.where(np.isnan(d), 0.0, d)", "0.5 * G"),
        ("d = params[0] * x\nreturn np.where(np.isnan(d) | np.isinf(d), 0.0, d)",
         "0.5 * G"),
        ("d = params[0] * x\nreturn np.where(np.isfinite(d), d, 0.0)", "0.5 * G"),
        # an index read with a mode, and a closeness test of a coefficient
        ("return np.take(params, 12, mode='clip') * x", "5.0 * G"),
        ("k = params[1]\nif np.allclose(k, 0.0):\n    k = 1.0\nreturn k * x",
         "1.0 * G"),
    ],
)
def test_a_guard_that_keeps_finite_values_is_read_through(
    body: str, expected: str
) -> None:
    assert _converted(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        # a guard that changes values where the data are exactly zero, as a
        # meal rate is on most rows
        "d = x * v\nreturn np.where(d == 0, 1e-12, d)",
        "d = params[0] * x\nreturn np.where(np.isnan(d), 0.0, v)",
    ],
)
def test_a_guard_that_changes_values_is_refused(body: str) -> None:
    with pytest.raises(InexpressibleProgram, match="on the data"):
        convert_program(body, INPUTS, PARAMS)


def test_a_refit_past_its_time_limit_is_refused() -> None:
    import numpy as np

    from autoformalism.rebuttal.llm_sr_upstream import refit_program

    channels = {"G": np.arange(5.0), "I": np.zeros(5)}
    with pytest.raises(InexpressibleProgram, match="time limit"):
        refit_program("return params[0] * x", INPUTS, channels, np.ones(5),
                      seconds=-1.0)
