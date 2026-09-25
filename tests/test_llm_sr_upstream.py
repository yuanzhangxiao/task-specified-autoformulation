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

from autoformalism.expressions.parser import RestrictedParser
from autoformalism.rebuttal.llm_sr_upstream import (
    InexpressibleProgram,
    convert_program,
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
        ("out = 0\nfor i in range(2):\n    out = out + x\nreturn out",
         "no expression equivalent"),
        ("if x > 0:\n    return x\nreturn v", "no expression equivalent"),
        ("return np.arctan(params[0] * x)", "not approved"),
        ("return params[0] * z", "unknown name"),
        ("return x.T * params[0]", "attribute access"),
        ("return x.mean() * params[0]", "attribute access"),
        ("return params[int(v[0])] * x", "not a literal"),
        ("return [params[0] * y for y in x]", "comprehension"),
        ("return params[0] * x if x > 0 else params[1]", "conditional"),
        # beyond the vector their specification declares
        ("return params[99] * x", "beyond the declared vector"),
        ("x = (", "does not parse"),
        ("pass", "no expression equivalent"),
        ("a = params[0] * x", "never returns"),
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
        refit_parameters,
    )

    body = "dv = params[0] * x + params[1] * v + params[2]\nreturn dv"
    symbolic = convert_program(body, INPUTS)
    rng = np.random.default_rng(0)
    channels = {"G": rng.normal(size=200), "I": rng.normal(size=200)}
    target = -0.7 * channels["G"] + 0.3 * channels["I"] + 1.4

    fitted = refit_parameters(
        symbolic.expression, channels, target, symbolic.used_parameters
    )
    assert fitted is not None
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
