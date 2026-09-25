"""Fixed outer signs, isolated scope, public evidence and declaration preservation."""

import copy
import json

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.public_fitting import PublicFitRequest
from autoformalism.search import sign_review as review
from scripts.smoke_dalla_sign_repair import fixture


def example(tmp_path):
    source, _, _ = fixture(tmp_path)
    packet = public._read(source)
    return (
        PublicFitRequest.model_validate(
            packet["models"][0]["result"]["selected_request"]
        ),
        packet["cells"]["toy"]["brief"],
    )


def decision(request, sign="negative", **extra):
    return {
        "decisions": [
            {
                "slot_id": review.slots(request)[0]["slot_id"],
                "outer_weight_sign": sign,
                "basis": "public_task",
                "public_quote": "The output has clearance.",
                "rationale": "Clearance removes the modeled output.",
                **extra,
            }
        ]
    }


@pytest.mark.parametrize("sign", ["positive", "negative", "unrestricted"])
def test_exact_scope_and_domains(tmp_path, sign):
    request, brief = example(tmp_path)
    before = request.model_dump(mode="json")
    result = review.apply(request, brief, decision(request, sign))
    child = PublicFitRequest.model_validate(result["request"])
    assert request.model_dump(mode="json") == before
    assert child.initialization_plan == request.initialization_plan
    assert child.base_candidate.processes == request.base_candidate.processes
    assert (
        child.base_candidate.state_equations[1:]
        == request.base_candidate.state_equations[1:]
    )
    assert [p for p in child.base_candidate.parameters if p.name != "c"] == [
        p for p in request.base_candidate.parameters if p.name != "c"
    ]
    if sign == "unrestricted":
        assert child == request and not result["changed"]
    else:
        parameter = next(p for p in child.base_candidate.parameters if p.name == "c")
        assert parameter.domain.value == "nonnegative"
        assert "c" not in child.parameter_guesses
        model, _, _ = public._lower(child)
        from autoformalism.fitting.fitter import _parameter_variable
        from autoformalism.fitting.models import FitConfig

        assert _parameter_variable(model, parameter, FitConfig()).lower == 0


@pytest.mark.parametrize("expression", ["-c*x", "-(c*x)", "c*(-x)", "-(-c*x)"])
def test_literal_minus_is_applied_exactly_once(tmp_path, expression):
    request, brief = example(tmp_path)
    raw = request.model_dump(mode="json")
    raw["base_candidate"]["state_equations"][0]["rhs"] = expression
    request = PublicFitRequest.model_validate(raw)
    result = review.apply(request, brief, decision(request))
    assert (
        result["request"]["base_candidate"]["state_equations"][0]["rhs"] == "-(c * x)"
    )


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "unknown", "quote", "undetermined", "extra_edit"],
)
def test_invalid_decisions_cannot_modify_model(tmp_path, mutation):
    request, brief = example(tmp_path)
    raw = decision(request)
    if mutation == "duplicate":
        raw["decisions"] *= 2
    elif mutation == "missing":
        raw["decisions"] = []
    elif mutation == "unknown":
        raw["decisions"][0]["slot_id"] = "invented"
    elif mutation == "quote":
        raw["decisions"][0]["public_quote"] = "Hidden reference sign"
    elif mutation == "undetermined":
        raw["decisions"][0]["basis"] = "undetermined"
    else:
        raw["equations"] = [{"state": "x", "rhs": "0"}]
    with pytest.raises(ValueError):
        review.apply(request, brief, raw)


@pytest.mark.parametrize("expression", ["c*x+c*z", "tanh(c*x)", "x/c", "c*eps*x"])
def test_shared_internal_denominator_and_ambiguous_gains_are_protected(
    tmp_path, expression
):
    request, _ = example(tmp_path)
    raw = request.model_dump(mode="json")
    raw["base_candidate"]["state_equations"][0]["rhs"] = expression
    assert review.slots(PublicFitRequest.model_validate(raw)) == []


def test_public_context_has_no_fit_values_scores_or_arrays(tmp_path):
    request, brief = example(tmp_path)
    brief = {**brief, "validation": {"secret": 998877}, "private_reference": "SECRET"}
    payload = review.context(request, brief)
    text = json.dumps(payload)
    assert (
        "998877" not in text
        and "SECRET" not in text
        and "parameter_guesses" not in text
    )
    assert "training" not in payload and "validation" not in payload
    assert payload["scientific_context"] == brief["scientific_context"]


def test_bounds_and_boundary_references_are_not_reinterpreted(tmp_path):
    request, _ = example(tmp_path)
    raw = request.model_dump(mode="json")
    other = copy.deepcopy(raw)
    next(p for p in other["base_candidate"]["parameters"] if p["name"] == "c")[
        "bounds"
    ] = {"lower": -1, "upper": 1}
    assert review.slots(PublicFitRequest.model_validate(other)) == []
