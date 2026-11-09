"""Regression checks for equation attribution and calibration of frozen trajectories."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import iclr_latent_review as review


def test_fitted_sign_cancellation_and_reachability():
    part = SimpleNamespace
    c = part(
        observation_mappings=[part(channel="Gp", expression="Gp")],
        parameters=[part(name="k")],
        processes=[part(name="sink", expression="-I * D")],
        state_equations=[
            part(state="Gp", rhs="meal_event_g - k * sink + X - X"),
            part(state="X", rhs="I-X"),
            part(state="D", rhs="I-D"),
            part(state="I", rhs="insulin_pmol_per_kg_min-I"),
        ],
    )
    eq = review.fitted_equations(c, {"k": 0.25})
    assert eq["Gp"][(("D", 1), ("I", 1))] == Fraction(1, 4)
    assert "X" not in review.ancestors(eq, {"Gp"})
    terms = review.components(eq)
    (insulin,) = [r for r in terms if r["group"] == "insulin_linked"]
    (meal,) = [r for r in terms if r["group"] == "meal_only"]
    frozen = {"I": [2.0, 4.0], "latent.D": [3.0, 2.0]}
    reference = {"time": [0.0, 1.0], "meal_event_g": [0.0, 4.0], "I": [1e20, 1e20]}
    np.testing.assert_equal(review.term_series(insulin, frozen, reference), [1.5, 2.0])
    np.testing.assert_equal(review.term_series(meal, frozen, reference), [0, 4])
    with pytest.raises(ValueError, match="coefficient"):
        review.fitted_equations(c, {})


def data():
    curves = {}
    refs = {}
    schedules = {}
    for i, k in enumerate(["train_a", "train_b", "train_c", "train_d", "pulse"]):
        x = np.arange(12.0) + i
        y = np.sin(np.arange(12.0) + i)
        curves[k] = {"latent.active": x, "latent.orphan": y, "I": x - y}
        refs[k] = {"Uid": y.copy()}
        schedules[k] = {
            "role": "train_calibration" if i < 4 else "contrast",
            "protocol": {"split": "train" if i < 4 else "test"},
        }
    return curves, refs, schedules


@pytest.mark.parametrize("method", review.METHODS)
def test_no_contrast_labels_or_coordinates_enter_calibration(method):
    curves, refs, schedules = data()
    args = (curves, refs, schedules, {"active"}, method, "Uid")
    before = review.calibration(*args)
    curves["pulse"] = {"latent.active": [-1e9], "latent.orphan": [2e10], "I": [22]}
    refs["pulse"] = {"Uid": [100.0]}
    assert review.calibration(*args) == before
    if method == "all_latent_affine":
        assert before["name"] == "latent.orphan"
    if method == "glucose_reachable_affine":
        assert before["name"] == "latent.active"
    if method == "all_latent_ridge":
        assert before["alpha"] in review.RIDGE_ALPHAS
        assert before["calibration_mse"] < 1e-6
    schedules["train_a"]["protocol"]["split"] = "test"
    with pytest.raises(ValueError, match="unverified"):
        review.calibration(*args)


def test_reference_event_controls_and_solver_failure():
    t = np.arange(4.0)
    truth = {"a": {"time": t, "Uid": t}, "b": {"time": t, "Uid": np.zeros(4)}}
    p = {"a": t, "b": np.zeros(4)}
    item = {"control": "b", "window": [0, 3]}
    assert review.score_paired(truth, "a", item, "Uid", p, p, 1)["effect_rse"] == 0
    bad = {"a": 2 * t, "b": np.zeros(4)}
    assert (
        review.score_paired(truth, "a", item, "Uid", p, bad, 1)["status"]
        == "effect_solver_disagreement"
    )
    truth["a"]["Uid"] = np.zeros(4)
    assert (
        review.score_paired(truth, "a", item, "Uid", p, p, 1)["status"]
        == "negligible_reference_effect"
    )


def test_unique_pairs_include_ineligible_ones():
    roster = [
        {"source_id": f"src{n}", "source_fit_sha256": str(n), "validation_nmse": 0.5}
        for n in range(3)
    ]
    plan = {
        "roster": roster,
        "pairs": [
            {"left": "0", "right": "1", "eligible": True, "reasons": []},
            {"left": "0", "right": "2", "eligible": False, "reasons": ["complexity"]},
        ],
    }
    rows = [
        {
            "subject": i,
            "case": "pulse",
            "quantity": q,
            "readout": "target",
            "status": "complete",
            "effect_rse": i,
        }
        for i in range(3)
        for q in ["Gp", "I"]
    ]
    pairs, effects = review.pair_tables(plan, rows)
    assert len(pairs) == 2
    assert [p["outcome_rows"] for p in pairs] == [2, 0]
    assert {r["pair_id"] for r in effects} == {"src0--src1"}


def test_wrong_archive_is_refused_before_creating_outputs(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"wrong payload")
    out = tmp_path / "report"
    with pytest.raises(ValueError, match="authenticated"):
        review.run(path, path, out)
    assert not out.exists()


def test_paired_product_separates_current_and_mediator_changes():
    term = {"powers": (("I", 1), ("D", 1)), "coefficient": 2}
    curves = {
        "pulse": {"I": [4, 6], "latent.D": [2, 5]},
        "base": {"I": [3, 2], "latent.D": [2, 4]},
    }
    refs = {k: {"time": [0, 1]} for k in curves}
    change = review.product_changes(term, curves, refs, "pulse", "base")
    np.testing.assert_equal(change["change in I alone"], [4, 32])
    np.testing.assert_equal(change["change in D alone"], [0, 4])
    np.testing.assert_equal(change["product interaction"], [0, 8])
    np.testing.assert_equal(sum(change.values()), [4, 44])
