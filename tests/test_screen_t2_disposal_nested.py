"""Screening must preserve the parent and detect a reversed action dependence."""

import numpy as np
import pytest

from scripts import screen_t2_disposal_nested as nested


def test_each_revision_nests_parent_and_reuses_common_adjustable_coordinates():
    rhs = "afcoef11+afcoef13*I_act+Uii"
    families = nested.menu({"formula": {"equations": {"U": rhs}}})
    assert len(families) == 5
    for f in families[2:]:
        assert f.rhs.startswith(rhs + "+disp_tissue*")
        assert f.coefficients == ("afcoef11", "afcoef13", "disp_tissue")


def test_minimum_slope_checks_reversed_effect_at_upper_and_lower_bound():
    x = np.array([55, 100, 150, 250])
    assert (
        nested.minimum_slope("add_tissue", {"afcoef13": 2, "disp_tissue": -100}, x) == 2
    )
    for k in (-5, 4):
        params = {"afcoef13": 0.2, "disp_tissue": k}
        expected = 0.2 + k * (x - 100) / 100
        assert nested.minimum_slope("add_tissue_action", params, x) == min(expected)
        expected = 0.2 + k * (x / (100 + x) - 0.5)
        assert nested.minimum_slope("add_saturating_tissue_action", params, x) == min(
            expected
        )
    with pytest.raises(ValueError, match="finite"):
        nested.minimum_slope("add_tissue", {}, np.array([np.nan]))


def test_train_selected_candidate_must_outperform_both_controls_with_positive_slope():
    def result(loss, slope=1):
        scores = {
            "pooled_nmse": loss,
            "per_target_nmse": {"U": loss},
            "min_U_minus_Uii": 0.2,
            "minima": {"Gp": 1},
        }
        return {
            "training": scores,
            "held_training": scores,
            "action": {"full_min_slope": 1, "held_min_slope": slope},
        }

    base = {
        "incumbent": result(3),
        "refit_existing": result(2),
        "main": result(2.5),
        "bad": result(1, slope=-0.1),
    }
    selected, fails = nested.select(base)
    assert selected == "incumbent"
    assert "held_training_nonpositive_action_slope" in fails["bad"]
    assert "training_U_vs_refit_existing" in fails["main"]
    assert nested.select(base | {"good": result(1.5)})[0] == "good"
