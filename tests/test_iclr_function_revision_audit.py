"""Separate equation revision, refitting, added coordinates and final survival."""

from copy import deepcopy

import pytest

from autoformalism.expressions.diagnostics import ModelValidationError
from scripts import iclr_function_revision_audit as audit
from tests.test_iclr_latent_revision_audit import model


def fitted(rhs, parameters=None, processes=None):
    m = model({"Gp": rhs}, [])
    m["parameters"] = parameters or {}
    m["candidate"]["parameters"] = [
        {"name": n, "scope": "global"} for n in m["parameters"]
    ]
    m["candidate"]["processes"] = processes or []
    return m


def family(rhs, parameters=None, *, fitted_coefficients=False, processes=None):
    return audit.Families(
        fitted(rhs, parameters, processes), fitted=fitted_coefficients
    ).equations()["Gp"]


def campaign(parent, child, final=None):
    return {
        "plan_sha256": "example",
        "model_settings": {"model": "20b"},
        "models": {"parent": parent, "child": child, "final": final or child},
        "runs": [
            {
                "task": {
                    "task_id": "s0",
                    "cell": "case",
                    "seed": 0,
                    "treatment": "full",
                },
                "final_status": "complete",
                "final_model": "final",
                "latest_model": "child",
                "visits": [
                    {"round": 0, "selected_model": "parent"},
                    {
                        "round": 1,
                        "selected_model": "child",
                        "attempts": [
                            {
                                "accepted": True,
                                "raw": {"hypothesis": "Change feedback."},
                            }
                        ],
                    },
                ],
            }
        ],
    }


def test_coefficient_values_names_signs_and_refits_do_not_count():
    assert family("a*Gp+b*u", {"a": 2, "b": 3}) == family("u/c-d*Gp", {"c": 5, "d": 6})
    rows, events, errors = audit.audit_campaign(
        "fixture", campaign(fitted("a*Gp", {"a": 2}), fitted("a*Gp", {"a": 3}))
    )
    assert not errors and not events
    assert all(r["retained_equation_revision_count"] == 0 for r in rows)


def test_polynomial_reordering_factoring_and_exact_cancellation_are_normalized():
    assert family("(Gp+u)**2") == family("u*u+2*Gp*u+Gp**2")
    assert family("(a+b)*Gp-a*Gp-b*Gp", {"a": 1, "b": 2}) == ()
    assert family("Gp/Gp") == family("1")


def test_process_names_and_decomposition_are_inlined():
    assert family(
        "a*P", {"a": 1}, processes=[{"name": "P", "expression": "Gp*u"}]
    ) == family("Gp*u")


def test_sigmoid_spelling_is_normalized_and_is_not_affine():
    assert family("sigmoid(Gp)") == family("1/(1+exp(-Gp))")
    assert family("sigmoid(Gp)") != family("Gp")
    assert any(audit.nonlinear(term) for term in family("sigmoid(Gp)"))


def test_same_class_kernels_do_not_cancel_before_coefficients_are_stripped():
    assert family("exp(Gp)-exp(-Gp)")
    assert family("1/(Gp+1)-1/(Gp+2)")
    assert family("exp(a*Gp)-exp(b*Gp)", {"a": 1, "b": 1})
    assert not family("exp(a*Gp)-exp(b*Gp)", {"a": 1, "b": 1}, fitted_coefficients=True)
    assert family("exp(a*Gp)-exp(b*Gp)", {"a": 1, "b": 2}, fitted_coefficients=True)


def test_nonlinear_form_survives_when_an_unrelated_term_is_pruned():
    c = campaign(fitted("u-Gp"), fitted("u-Gp*Gp"), fitted("-Gp*Gp"))
    rows, events, errors = audit.audit_campaign("fixture", c)
    assert not errors
    r = next(r for r in rows if r["snapshot"] == "final")
    assert r["retained_nonlinear_revision_count"] == 1
    assert r["retained_new_nonlinearity_count"] == 1
    assert not events[0]["entire_revised_family_retained"]


def test_pruning_or_zero_fitted_gain_removes_nonlinear_claim():
    parent = fitted("u-Gp", {"a": 1})
    child = fitted("u-Gp+a*Gp**2", {"a": 1})
    final = deepcopy(child)
    final["parameters"]["a"] = 0
    rows, _, errors = audit.audit_campaign("fixture", campaign(parent, child, final))
    assert not errors
    assert (
        next(r for r in rows if r["snapshot"] == "final")[
            "retained_nonlinear_revision_count"
        ]
        == 0
    )


def test_joint_state_addition_and_shape_revision_are_counted_separately():
    parent = fitted("u-Gp")
    child = model({"Gp": "M-Gp**2", "M": "u-M"}, [])
    rows, events, errors = audit.audit_campaign("fixture", campaign(parent, child))
    assert not errors and len(events) == 2
    assert rows[0]["retained_equation_revision_count"] == 0
    assert rows[0]["retained_shape_revision_count"] == 0
    assert rows[0]["retained_shape_revision_any_roster_count"] == 1
    assert all(not e["fixed_state_roster"] for e in events)


def test_adding_linear_memory_alone_is_not_a_functional_class_revision():
    parent = fitted("u-Gp")
    child = model({"Gp": "M-Gp", "M": "u-M"}, [])
    rows, _, errors = audit.audit_campaign("fixture", campaign(parent, child))
    assert not errors
    assert rows[0]["retained_equation_revision_any_roster_count"] == 1
    assert rows[0]["retained_shape_revision_any_roster_count"] == 0


def test_affine_dependency_change_is_broad_revision_not_nonlinear_form():
    rows, _, errors = audit.audit_campaign(
        "fixture", campaign(fitted("-Gp"), fitted("u-Gp"))
    )
    assert not errors
    assert rows[0]["retained_equation_revision_count"] == 1
    assert rows[0]["retained_nonlinear_revision_count"] == 0


def test_second_bilinear_path_is_not_a_new_nonlinear_law_class():
    parent = fitted("Gp*u")
    parent["context"]["auxiliaries"] = ["a"]
    child = deepcopy(parent)
    child["candidate"]["state_equations"][0]["rhs"] = "Gp*u + Gp*a"
    rows, _, errors = audit.audit_campaign("fixture", campaign(parent, child))
    assert not errors
    assert rows[0]["retained_nonlinear_revision_count"] == 1
    assert rows[0]["retained_shape_revision_count"] == 0


def test_polynomial_degree_and_saturation_are_distinct_shape_classes():
    rows, _, errors = audit.audit_campaign(
        "fixture", campaign(fitted("u-Gp"), fitted("u-Gp**2"))
    )
    assert not errors and rows[0]["retained_shape_revision_count"] == 1
    assert audit.shape(family("Gp*u")[0]) == audit.shape(family("Gp*t")[0])
    assert audit.shape(family("Gp**2")[0]) != audit.shape(family("Gp*u")[0])


def test_missing_fit_or_missing_history_is_not_a_zero_revision_success():
    s = audit.summarize(
        [{"available": False}, {"available": True, "history_audit_errors": 1}]
    )
    assert s["assigned"] == 2 and s["available"] == 1 and s["assessed"] == 0
    assert s["retained_equation_revision_count"]["histogram"] == {}


def test_untrusted_expression_and_unbounded_expansion_are_rejected(monkeypatch):
    with pytest.raises(ModelValidationError):
        family("__import__('os').system('x')")
    monkeypatch.setattr(audit, "MAX_TERMS", 10)
    with pytest.raises(ValueError, match="bound"):
        family("(Gp+u+Gp*u+1)**4")
    with pytest.raises(ValueError, match="zero denominator"):
        family("Gp/(a-a)", {"a": 1})
