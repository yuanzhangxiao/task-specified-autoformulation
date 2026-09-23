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


def test_the_three_measures_separate_size_from_structure() -> None:
    """Node count, term count and operator count answer different questions."""
    assert complexity.expression_complexity("G") == {
        "nodes": 1, "terms": 1, "operators": 0
    }
    # a + b + c is three terms; a * b is one however large
    assert complexity.expression_complexity("0.5*G + 1.0*I + 1.5")["terms"] == 3
    assert complexity.expression_complexity("0.5*G*I*7.0")["terms"] == 1
    # subtraction is still additive structure
    assert complexity.expression_complexity("G - I")["terms"] == 2


def test_a_model_is_all_of_its_equations() -> None:
    measured = complexity.model_complexity({"G": "0.5*G + 1.0*I", "I": "-0.2*I"})
    assert measured["equations"] == 2
    assert measured["terms"] == 3
    assert measured["nodes"] == 16


def test_an_unparsable_equation_makes_the_model_unmeasured_not_zero() -> None:
    """Scoring a broken model as complexity zero would rank it as simplest."""
    assert complexity.expression_complexity("0.5*(G +") is None
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
