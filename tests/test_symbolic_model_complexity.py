"""Complexity of the symbolic models a frozen evaluation actually scored."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "complexity",
    Path(__file__).resolve().parent.parent / "scripts" / "symbolic_model_complexity.py",
)
complexity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(complexity)


def test_the_counts_are_the_projects_own_not_a_second_definition() -> None:
    """Two numbers called complexity in one paper would be one too many.

    pruning.process_aware already decides whether one model is smaller than
    another, and this reuses its term decomposition and node count so a
    baseline's complexity is measured the same way the pruner measures one.
    """
    import ast

    from autoformalism.expressions import ValidationContext
    from autoformalism.pruning.process_aware import definitions, terms
    from autoformalism.rebuttal.final_evaluation_adapters import equation_candidate

    equations = {"G": "0.5*G + 1.0*I + 1.5", "I": "-0.2*I"}
    measured = complexity.model_complexity(equations)

    candidate = equation_candidate(
        "complexity", equations, ValidationContext(targets=tuple(equations))
    )
    expressions = (
        list(definitions(candidate).values())
        + [item.expression for item in candidate.observation_mappings]
        + [item.expression for item in candidate.initial_conditions if item.expression]
    )
    assert measured["terms"] == len(terms(candidate))
    assert measured["expression_nodes"] == sum(
        sum(1 for _ in ast.walk(ast.parse(item, mode="eval"))) for item in expressions
    )
    assert measured["states"] == 2


def test_a_model_is_all_of_its_equations() -> None:
    """A two-state model counts both right-hand sides."""
    one = complexity.model_complexity({"G": "0.5*G + 1.0*I"})
    two = complexity.model_complexity({"G": "0.5*G + 1.0*I", "I": "-0.2*I"})
    assert two["states"] == 2 and one["states"] == 1
    assert two["expression_nodes"] > one["expression_nodes"]


def test_an_unparsable_equation_makes_the_model_unmeasured_not_zero() -> None:
    """Scoring a broken model as complexity zero would rank it as simplest."""
    assert complexity.model_complexity({"G": "0.5*(G +"}) is None
    assert complexity.model_complexity({}) is None


def test_quartiles_are_inclusive_and_carry_their_count() -> None:
    """Per-cell counts are small, so the exclusive method discards too much."""
    assert complexity.quartiles([1, 2, 3, 4, 5, 6, 7, 8, 9]) == {
        "n": 9, "median": 5, "q1": 3, "q3": 7, "iqr": 4
    }
    single = complexity.quartiles([4.0])
    assert single["n"] == 1 and single["iqr"] == 0.0
    assert complexity.quartiles([])["median"] is None


def test_only_the_named_sources_are_measured(tmp_path: Path) -> None:
    """The models measured must be the models the evaluation scored.

    Scanning a development directory instead would include candidates that
    were never selected, and quietly change what the number describes.
    """
    scored = tmp_path / "scored.json"
    scored.write_text(json.dumps({
        "method": "pysr", "benchmark_id": "cell_a", "tier": "easy", "seed": 0,
        "equations": {"G": "0.5*G + 1.0*I"},
    }))
    unscored = tmp_path / "not_requested.json"
    unscored.write_text(json.dumps({
        "method": "pysr", "equations": {"G": "1.0*G + 2.0*I + 3.0*G*I + 4.0"},
    }))
    requests = tmp_path / "source_adapter_requests.jsonl"
    requests.write_text("\n".join([
        json.dumps({"source_kind": "pysr", "source_path": str(scored)}),
        # an unrelated method, and a source that no longer exists
        json.dumps({"source_kind": "autoformalism", "source_path": str(unscored)}),
        json.dumps(
            {"source_kind": "sindy", "source_path": str(tmp_path / "gone.json")}
        ),
    ]))

    value = complexity.collect(requests)
    assert list(value["summary"]) == ["pysr"]
    assert value["summary"]["pysr"]["terms"]["n"] == 1
    assert value["summary"]["pysr"]["terms"]["median"] == 2
    assert "expression_nodes" in value["summary"]["pysr"]
    assert value["unreadable"] == {"sindy:missing_source": 1}
    assert value["test_data_opened"] is False


@pytest.mark.parametrize(
    "payload", [{"equations": {}}, {"equations": "not a dict"}, {}]
)
def test_a_source_without_equations_is_counted_not_dropped(
    tmp_path: Path, payload: dict
) -> None:
    """Silently skipping sources would shrink the denominator invisibly."""
    source = tmp_path / "model.json"
    source.write_text(json.dumps({"method": "sindy", **payload}))
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps({"source_kind": "sindy", "source_path": str(source)})
    )
    value = complexity.collect(requests)
    assert value["unreadable"] == {"sindy:no_equations": 1}
    assert value["summary"] == {}
