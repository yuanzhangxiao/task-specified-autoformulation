"""The control must recover a declared family and retain ambiguities."""

from dataclasses import replace
from fractions import Fraction as Q
from itertools import product

import pytest

from scripts.check_anchored_latent_functions import recover, run, simulate


def test_crossed_inputs_and_known_balances_determine_three_functions(tmp_path):
    summary = run(tmp_path)
    assert run(tmp_path) == summary
    case = summary["cases"]["crossed"]
    assert case["events"] == 18
    maps = case["functions"]
    assert all(v["rank"] == 6 for v in maps.values())
    assert maps["w=h(x,u)"]["coefficients"] == ["2", "0", "-1", "1", "0", "0"]
    assert maps["z'=f(x,u)"]["coefficients"] == ["1", "0", "0", "0", "1", "0"]
    assert maps["y'-1=H(z,x)"]["coefficients"] == ["0", "0", "0", "1", "2", "0"]


def test_unknown_gain_and_correlated_inputs_have_distinct_ambiguities(tmp_path):
    cases = run(tmp_path)["cases"]
    unknown = cases["unknown_gain"]
    assert unknown["identical_events_after_scaling_z_by_2"] == unknown["events"]
    assert unknown["functions"]["y'-1=H(z,x)"]["status"] == "unknown_latent_scale"
    same = cases["same_input"]["functions"]
    assert same["z'=f(x,u)"] == {"rank": 3, "status": "insufficient_variation"}
    assert same["y'-1=H(z,x)"]["status"] == "unique_in_declared_family"


def test_incompatible_rows_cannot_yield_unique_recovery():
    rows = [
        simulate(Q(x), Q(u), Q(z)) for x, u, z in product(range(3), range(3), (1, 3))
    ]
    rows[0] = replace(rows[0], s=rows[0].s + 1)
    assert recover(rows, calibrated=True)["w=h(x,u)"]["status"] == "family_excluded"
    with pytest.raises(ValueError, match="one measured interval"):
        recover([], calibrated=True)
