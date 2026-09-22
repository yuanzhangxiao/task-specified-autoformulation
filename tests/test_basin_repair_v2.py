"""Saved repair interface regressions and conservative parameter-free witnesses."""

from copy import deepcopy

import pytest

from autoformalism.expressions import ModelValidationError
from autoformalism.rebuttal import basin_static_checks as static
from autoformalism.search import basin_model_repair as legacy
from autoformalism.search import basin_model_repair_v2 as repair
from tests.test_basin_equation_checks import CONTEXT, SURVEY, candidate
from tests.test_basin_model_repair import bundle


def patch(**kwargs):
    return {"hypothesis": "Explicit public-model revision.", **kwargs}


def test_acceptance_flag_with_edits_and_last_call_need_no_extra_confirmation():
    parent = bundle(True)
    raw = patch(
        accept_displayed=True,
        equations=[
            {
                "component": "h_down",
                "expression": "inflow_down/area_down - d*max(0,h_down-crest_down)",
            }
        ],
    )
    result = repair.transition(parent, parent, raw, "independent", [SURVEY])
    assert result["status"] == "eligible_for_fit"
    assert result["transaction"]["legacy_accept_flag_ignored_with_edits"]
    # Repeating the exact patch does not require another confirmation either.
    again = repair.transition(result["bundle"], parent, raw, "independent", [SURVEY])
    assert again["status"] == "eligible_for_fit"
    assert not again["transaction"]["changed"]
    with pytest.raises(ValueError, match="cannot accompany"):
        legacy.ModelRepair.model_validate(raw)


def test_new_shared_parameter_resolves_with_inherited_uses_in_same_transaction():
    parent = bundle()
    raw = patch(
        new_parameters=[{"name": "shared", "role": "coefficient"}],
        parameter_bindings=[
            {"parameter": "k", "same_as": "shared"},
            {"parameter": "d", "same_as": "shared"},
        ],
    )
    result = repair.apply(parent, raw, "coupled")
    assert {p["name"] for p in result["bundle"]["candidate"]["parameters"]} == {
        "shared"
    }
    assert result["parameter_replacements"] == {"k": "shared", "d": "shared"}
    again = repair.apply(result["bundle"], raw, "coupled")
    assert set(again["repeated_bindings"]) == {"k", "d"}
    assert not again["changed"]
    # Old RHS spellings are also propagated without reintroducing eliminated gains.
    edit = repair.apply(
        result["bundle"],
        patch(
            equations=[{"component": "o", "expression": "d*max(0,h_down-crest_down)"}]
        ),
        "coupled",
    )
    assert set(edit["parameter_aliases"]) == {"k", "d"}
    assert not edit["changed"]


def test_new_parameter_can_be_fixed_in_same_reply_and_domain_is_checked():
    raw = patch(
        new_parameters=[{"name": "c", "role": "rate"}],
        equations=[{"component": "o", "expression": "c*max(0,h_down-crest_down)"}],
        parameter_bindings=[{"parameter": "c", "value": 1}],
    )
    result = repair.apply(bundle(True), raw, "independent")
    assert "c" not in {p["name"] for p in result["bundle"]["candidate"]["parameters"]}
    raw["parameter_bindings"][0]["value"] = -1
    with pytest.raises(ValueError, match="domain"):
        repair.apply(bundle(True), raw, "independent")


@pytest.mark.parametrize(
    "bindings",
    [
        [{"parameter": "k", "same_as": "area_up"}],
        [{"parameter": "k", "value": 1, "same_as": "d"}],
        [{"parameter": "k", "value": None, "same_as": None}],
        [{"parameter": "k", "same_as": "d"}, {"parameter": "d", "same_as": "k"}],
    ],
)
def test_ambiguities_still_rejected_atomically(bindings):
    parent = bundle()
    saved = deepcopy(parent)
    with pytest.raises(ValueError):
        repair.apply(parent, patch(parameter_bindings=bindings), "coupled")
    assert parent == saved


def test_conflicting_alias_and_incompatible_role_are_rejected():
    result = repair.apply(
        bundle(),
        patch(parameter_bindings=[{"parameter": "k", "same_as": "d"}]),
        "coupled",
    )
    with pytest.raises(ValueError, match="conflicting eliminated"):
        repair.apply(
            result["bundle"],
            patch(parameter_bindings=[{"parameter": "k", "value": 1}]),
            "coupled",
        )
    with pytest.raises(ValueError, match="same role"):
        repair.apply(
            bundle(),
            patch(
                new_parameters=[{"name": "positive", "role": "rate"}],
                parameter_bindings=[{"parameter": "k", "same_as": "positive"}],
            ),
            "coupled",
        )


def test_known_kind_inferred_unknown_component_precise_and_new_state_forbidden():
    result = repair.apply(
        bundle(),
        patch(
            equations=[
                {
                    "component": "h_down",
                    "kind": "algebraic",
                    "expression": "(inflow_down+q-o)/area_down",
                }
            ]
        ),
        "coupled",
    )
    assert result["component_kinds_normalized"] == [
        {"component": "h_down", "received": "algebraic", "retained": "dynamic"}
    ]
    with pytest.raises(
        ValueError, match=r"unknown component 'state_equations'.*h_down"
    ):
        repair.apply(
            bundle(),
            patch(
                equations=[
                    {
                        "component": "state_equations",
                        "kind": "dynamic",
                        "expression": "h_down",
                    }
                ]
            ),
            "coupled",
        )


def test_unknown_and_unsafe_symbols_and_process_cycles_still_fail():
    for expression in ("__import__('os')", "unknown", "o"):
        with pytest.raises((ValueError, ModelValidationError)):
            repair.apply(
                bundle(True),
                patch(equations=[{"component": "o", "expression": expression}]),
                "independent",
            )


def threshold(expression):
    model = candidate(independent=True, down=expression)
    return next(
        c
        for c in static.assess(model, CONTEXT, "independent", [SURVEY])["checks"]
        if c["code"] == "free_outlet_threshold_probes"
    )


def test_wrong_threshold_parameter_independent_zero_is_a_counterexample():
    check = threshold("inflow_down/area_down-d*max(0,h_down-warning_depth)")
    assert check["status"] == "fail"
    assert check["evidence"]["counterexamples"][0]["depth_derivatives_m_per_min"] == [
        0,
        0,
        0,
        0,
    ]
    model = candidate(
        independent=True, down="inflow_down/area_down-d*max(0,h_down-warning_depth)"
    )
    assert (
        next(
            c
            for c in legacy.checks.assess(
                model, CONTEXT, "independent", [SURVEY], None
            )["checks"]
            if c["code"] == "free_outlet_threshold_probes"
        )["status"]
        == "unverified"
    )


def test_correct_unknown_gain_is_unknown_not_static_pass():
    check = threshold("inflow_down/area_down-d*max(0,h_down-crest_down)")
    assert check["status"] == "unverified"
    assert check["evidence"]["probes"][0]["depth_derivatives_m_per_min"] == [
        0,
        0,
        0,
        None,
    ]
    assert not check["evidence"]["counterexamples"]
    assert (
        threshold("inflow_down/area_down-max(0,h_down-crest_down)")["status"] == "pass"
    )


def test_partial_evaluation_does_not_mask_undefined_unknown_or_overflow():
    import ast

    assert static.partial_value(ast.parse("0*exp(d)", mode="eval").body, {}, {"d"}) == (
        None,
        False,
    )
    assert static.partial_value(
        ast.parse("0*unmapped", mode="eval").body, {}, {"d"}
    ) == (None, False)
    assert static.partial_value(ast.parse("0*d", mode="eval").body, {}, {"d"}) == (
        0,
        True,
    )


def test_static_failure_draft_preserved_and_payload_current():
    parent = bundle(True)
    result = repair.transition(
        parent,
        parent,
        patch(
            accept_displayed=True,
            equations=[
                {
                    "component": "h_down",
                    "expression": "inflow_down/area_down-d*max(0,h_down-warning_depth)",
                }
            ],
        ),
        "independent",
        [SURVEY],
    )
    assert result["status"] == "static_repair_required"
    assert result["transaction"]["changed"]
    historic = repair.assessment(parent, "independent", [SURVEY])
    payload = repair.payload(
        result["bundle"], historic, result["static_assessment"], 1, None
    )
    assert payload["historical_evidence"]["candidate_matches_current"] is False
    assert not payload["explicit_patch_needs_separate_confirmation"]
    assert "historical_findings" not in payload
    assert any(c["status"] == "fail" for c in payload["current_unresolved_findings"])


def test_initial_historical_evidence_identifies_the_same_candidate():
    parent = bundle(True)
    historic = repair.assessment(parent, "independent", [SURVEY])
    value = repair.payload(parent, historic, historic, 3, None)
    assert value["historical_evidence"]["candidate_matches_current"]
    assert "historical fitted parameters" in value["historical_evidence"]["scope"]


def test_new_process_cannot_repurpose_an_eliminated_parameter_name():
    tx = repair.apply(
        bundle(),
        patch(parameter_bindings=[{"parameter": "k", "same_as": "d"}]),
        "coupled",
    )
    with pytest.raises(ValueError, match="collides with a recorded parameter alias"):
        repair.apply(
            tx["bundle"],
            patch(
                equations=[
                    {"component": "k", "kind": "algebraic", "expression": "h_down"}
                ]
            ),
            "coupled",
        )
