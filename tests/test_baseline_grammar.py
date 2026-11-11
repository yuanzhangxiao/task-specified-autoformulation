"""External baselines are read with a wider grammar; our method is not.

A baseline is scored on the model it produced. Refusing a real exponent or a
function its own vocabulary offered would measure our grammar rather than the
method, so the baseline grammar admits them. Our method's grammar must not
move at all, and the wider one must still never execute text.
"""

from __future__ import annotations

import ast
import math

import pytest

from autoformalism.expressions import (
    CandidateValidator,
    ModelValidationError,
    RuntimeExpressionError,
    ValidationContext,
    baseline_validator,
    compile_candidate,
)
from autoformalism.expressions.baseline_functions import BASELINE_FUNCTIONS
from autoformalism.expressions.compiler import _evaluate
from autoformalism.expressions.parser import (
    APPROVED_FUNCTION_ARITY,
    RestrictedParser,
    baseline_parser,
)
from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

CONTEXT = ValidationContext(
    targets=("G",), auxiliaries=("E",), external_inputs=(), fixed_covariates=()
)

BASELINE_ONLY = (
    "G**0.734",
    "G**E",
    "cos(G)",
    "Abs(G)",
    "arctan2(G, E)",
    "1.3e14*G",
)


@pytest.mark.parametrize("source", BASELINE_ONLY)
def test_our_grammar_still_refuses_what_only_baselines_may_use(source) -> None:
    with pytest.raises(ModelValidationError):
        RestrictedParser().parse(source, location="ours")
    baseline_parser().parse(source, location="baseline")


def test_our_function_set_is_unchanged() -> None:
    assert set(RestrictedParser().functions) == set(APPROVED_FUNCTION_ARITY)
    assert set(APPROVED_FUNCTION_ARITY) == {
        "abs", "exp", "log", "max", "min", "sigmoid", "sin", "softplus",
        "sqrt", "tanh",
    }


def test_the_wider_grammar_still_refuses_code() -> None:
    """Wider in vocabulary, never in kind: no attributes, lambdas or calls out."""
    for source in (
        "__import__('os').system('true')",
        "G.real",
        "(lambda x: x)(G)",
        "G if E else 1.0",
        "G < E",
        "np.cos(G)",
        "undefined_function(G)",
    ):
        with pytest.raises(ModelValidationError):
            baseline_parser().parse(source, location="baseline")


def _sample_arguments(name: str, count: int) -> list[float]:
    """A point inside every function's real domain."""
    if name in {"acosh", "arccosh"}:
        return [1.5]
    if name == "clip":
        return [0.4, 0.0, 1.0]
    return [0.4 + 0.1 * index for index in range(count)]


@pytest.mark.parametrize("name", sorted(BASELINE_FUNCTIONS))
def test_every_baseline_function_evaluates_through_the_compiler(name) -> None:
    (minimum, _), _ = BASELINE_FUNCTIONS[name]
    arguments = _sample_arguments(name, minimum)
    symbols = [f"a{index}" for index in range(len(arguments))]
    parsed = baseline_parser().parse(
        f"{name}({', '.join(symbols)})", location="baseline"
    )
    value = _evaluate(parsed, dict(zip(symbols, arguments, strict=True)))
    assert math.isfinite(value)


def test_power_call_and_operator_agree() -> None:
    parser = baseline_parser()
    environment = {"G": 2.0, "E": 0.73}
    assert _evaluate(parser.parse("pow(G, E)", location="a"), environment) == (
        _evaluate(parser.parse("G**E", location="b"), environment)
    )
    assert _evaluate(parser.parse("G**E", location="c"), environment) == 2.0**0.73


def test_a_real_power_of_a_negative_base_fails_when_evaluated() -> None:
    """Where numpy would return NaN, the rollout fails rather than going complex."""
    parsed = baseline_parser().parse("G**0.5", location="baseline")
    with pytest.raises(RuntimeExpressionError):
        _evaluate(parsed, {"G": -1.0})


def test_integer_powers_keep_their_old_semantics() -> None:
    """Our grammar's integer path, including its guarded zero base, is unchanged."""
    ours = RestrictedParser().parse("G**-2", location="ours")
    theirs = baseline_parser().parse("G**-2", location="baseline")
    for value in (0.0, -3.0, 2.5):
        assert _evaluate(ours, {"G": value}) == _evaluate(theirs, {"G": value})


def test_a_baseline_model_with_domain_risks_is_compiled_and_flagged() -> None:
    """log(G) may leave its domain; for a baseline that is decided at rollout."""
    candidate = equation_candidate(
        "llm_ode", {"G": "-0.2*log(G) + sqrt(E)*G**0.73 + cos(E)"}, CONTEXT
    )
    with pytest.raises(ModelValidationError, match="UNSUPPORTED"):
        compile_candidate(candidate, CONTEXT)
    compiled = compile_candidate(candidate, CONTEXT, validator=baseline_validator())
    codes = {item.code for item in compiled.validated.warnings}
    assert {"DOMAIN_LOG_NONPOSITIVE", "DOMAIN_SQRT_NEGATIVE"} <= codes


def test_our_validator_still_refuses_a_domain_risk() -> None:
    candidate = equation_candidate("ours", {"G": "-log(G)"}, CONTEXT)
    with pytest.raises(ModelValidationError, match="DOMAIN_LOG_NONPOSITIVE"):
        CandidateValidator().validate(candidate, CONTEXT)
    baseline_validator().validate(candidate, CONTEXT)


def test_a_computed_exponent_does_not_break_interval_analysis() -> None:
    candidate = equation_candidate("llm_sr", {"G": "-G**(E + 0.5)"}, CONTEXT)
    validated = baseline_validator().validate(candidate, CONTEXT)
    assert isinstance(validated.equation_expressions["G"].tree.body, ast.UnaryOp)


def _toy_split(name):
    import numpy as np

    from autoformalism.data import DatasetSplit, SplitName, Trajectory

    time = np.linspace(0.0, 3.0, 31)
    starts = (1.0, 2.0) if name is SplitName.TRAIN else (1.5,)
    return DatasetSplit(
        name,
        tuple(
            Trajectory(
                trajectory_id=f"{name.value}_{index}",
                time=time,
                targets={"G": start * np.exp(-time)},
                auxiliaries={"E": np.ones_like(time)},
                external_inputs={},
                fixed_covariates={},
                derivatives={},
            )
            for index, start in enumerate(starts)
        ),
        name.value,
    )


def test_a_symbolic_regression_model_is_evaluated_with_the_baseline_grammar() -> None:
    """SINDy and PySR models reach evaluation through evaluate_equations."""
    from autoformalism.baselines.core import evaluate_equations
    from autoformalism.data import SplitName

    metrics = evaluate_equations(
        {"G": "-(G**0.5)**2.0 * cos(0.0 * E) + 1.3e14 * (E - 1.0)"},
        CONTEXT,
        _toy_split(SplitName.TRAIN),
        {"G": 1.0},
        identifier="pysr_toy",
    )
    assert metrics.normalized_mse < 1e-4


def test_d3_sine_is_a_sine_and_an_unknown_name_is_an_error() -> None:
    """An unlisted name used to fall through to max(), so sin(x) returned x."""
    import numpy as np

    from autoformalism.baselines.d3_native import NativeD3Error, _evaluate
    from autoformalism.baselines.d3_rollout import _NUMPY

    environment = {"x": np.array([0.5, 1.0])}
    tree = baseline_parser().parse("sin(x)", location="d3").tree.body
    np.testing.assert_allclose(_evaluate(tree, environment, _NUMPY), np.sin([0.5, 1.0]))
    tree = baseline_parser().parse("x**0.5 + arctan(x)", location="d3").tree.body
    np.testing.assert_allclose(
        _evaluate(tree, environment, _NUMPY),
        np.sqrt([0.5, 1.0]) + np.arctan([0.5, 1.0]),
    )
    with pytest.raises(NativeD3Error, match="unsupported function"):
        _evaluate(ast.parse("besselj(x)", mode="eval").body, environment, _NUMPY)


def test_every_array_table_covers_exactly_the_baseline_vocabulary() -> None:
    from autoformalism.baselines.d3_rollout import _NUMPY
    from autoformalism.expressions.parser import BASELINE_FUNCTION_ARITY
    from autoformalism.rebuttal.llm_sr_upstream import _NUMPY_FUNCTIONS

    assert set(_NUMPY.array_functions) == set(BASELINE_FUNCTION_ARITY)
    assert set(_NUMPY_FUNCTIONS) == set(BASELINE_FUNCTION_ARITY)


@pytest.mark.parametrize("name", sorted(BASELINE_FUNCTIONS))
def test_every_function_agrees_between_scalar_and_array_evaluation(name) -> None:
    """Selection, D3's rollout and LLM-SR's refit must read a model alike."""
    import numpy as np

    from autoformalism.baselines.d3_rollout import _NUMPY

    (minimum, _), _ = BASELINE_FUNCTIONS[name]
    arguments = _sample_arguments(name, minimum)
    symbols = [f"a{index}" for index in range(len(arguments))]
    parsed = baseline_parser().parse(
        f"{name}({', '.join(symbols)})", location="baseline"
    )
    scalar = _evaluate(parsed, dict(zip(symbols, arguments, strict=True)))
    array = _NUMPY.array_functions[name](*[np.asarray(item) for item in arguments])
    assert float(array) == pytest.approx(scalar, rel=1e-12)


def test_the_torch_table_covers_the_baseline_vocabulary() -> None:
    torch = pytest.importorskip("torch")
    from autoformalism.baselines.d3_native import _torch_functions
    from autoformalism.expressions.parser import BASELINE_FUNCTION_ARITY

    table = _torch_functions(torch)
    assert set(table) == set(BASELINE_FUNCTION_ARITY)
    for name in BASELINE_FUNCTIONS:
        (minimum, _), _ = BASELINE_FUNCTIONS[name]
        arguments = [
            torch.tensor(value, dtype=torch.float64)
            for value in _sample_arguments(name, minimum)
        ]
        assert math.isfinite(float(table[name](*arguments)))


def test_a_baseline_vocabulary_name_is_still_a_usable_symbol() -> None:
    """A D3 or Sol model may call a rate `gamma`; only a call is the function."""
    parsed = baseline_parser().parse("-gamma*G + gamma(E)", location="baseline")
    assert parsed.symbols == frozenset({"gamma", "G", "E"})
    assert RestrictedParser().parse("-gamma*G", location="ours").symbols == {
        "gamma",
        "G",
    }
    with pytest.raises(ModelValidationError, match="FUNCTION_AS_VALUE"):
        baseline_parser().parse("-sin*G", location="baseline")
