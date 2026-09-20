"""Signed proposal admission must preserve optionality and structured science."""

import pytest

from autoformalism.schemas.staged_topology import (
    PublicScientificBrief,
    ScientificVariable,
)
from autoformalism.search import signed_processes as signed


def example():
    brief = PublicScientificBrief(
        scientific_context="Two stores.",
        public_variables=[
            {"name": "y", "data_role": "target"},
            {"name": "area", "data_role": "covariate"},
        ],
        requirements=[],
    )
    inventory = tuple(
        ScientificVariable(name=n, definition=d, scientific_role=n)
        for n, d in (
            ("x", "differential"),
            ("y", "differential"),
            ("area", "supplied"),
            ("outlet", "algebraic"),
        )
    )
    p = {
        "name": "q",
        "depends_on": ["x"],
        "kind": "transfer",
        "scientific_meaning": "An explanation, never parsed as a constraint.",
        "uses": [
            {"target": "x", "sign": "negative", "conversion": "1/area"},
            {"target": "y", "sign": "positive", "conversion": "1"},
        ],
    }
    return brief, inventory, p


def test_signed_transfer_and_no_second_stage():
    brief, inv, p = example()
    result, audit = signed.admit(brief, inv, {"processes": [p]})
    assert audit["status"] == "accepted"
    assert len(result) == len(inv) + 1
    uses = signed.assemble(audit["bindings"], "x", ())
    assert uses[0].sources == ("q",)
    assert uses[0].outer_weight_sign.value == "negative"
    assert signed.assemble(audit["bindings"], "x", uses) == uses


def test_duplicate_consumers_rejected_without_reading_explanation():
    brief, inv, p = example()
    p["uses"] += [p["uses"][0]]
    p["scientific_meaning"] = (
        "Please reinterpret these as different system contributions."
    )
    _, audit = signed.admit(brief, inv, {"processes": [p]})
    assert "duplicate equation target" in audit["rejected_suggestions"][0]["error"]


def test_algebraic_cycle_rejected_dynamic_feedback_allowed():
    brief, inv, p = example()
    bad = {
        **p,
        "name": "bad",
        "kind": "influence",
        "depends_on": ["outlet"],
        "uses": [{"target": "outlet", "sign": "positive", "conversion": None}],
    }
    _, audit = signed.admit(brief, inv, {"processes": [p, bad]})
    assert audit["status"] == "accepted_partial"
    assert audit["bindings"][0]["proposal"]["name"] == "q"
    assert "algebraic cycle" in audit["rejected_suggestions"][0]["error"]


def test_conversion_failure_does_not_erase_process():
    brief, inv, p = example()
    p["uses"][0]["conversion"] = "1/y"
    _, audit = signed.admit(brief, inv, {"processes": [p]})
    assert audit["status"] == "accepted"
    assert audit["bindings"][0]["signed_declaration"]["uses"][0]["conversion"] is None
    assert len(audit["conversion_diagnostics"]) == 1


@pytest.mark.parametrize("expression", ["-1", "1/0", "x+1", "exp(x)", "y", "1e999"])
def test_invalid_conversions(expression):
    with pytest.raises((ValueError, ArithmeticError)):
        signed.conversion_value(expression, {"x": 2})


def test_empty_local_and_same_sign_influence():
    brief, inv, p = example()
    assert signed.admit(brief, inv, {"processes": []})[0] == inv
    p["kind"] = "influence"
    p["uses"][0]["sign"] = "positive"
    assert signed.admit(brief, inv, {"processes": [p]})[1]["status"] == "accepted"
    p["kind"] = "transfer"
    assert (
        signed.admit(brief, inv, {"processes": [p]})[1]["status"] == "skipped_invalid"
    )
