"""Initialization, retention and witness checks for retrospective T2 refits."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.special import expit

from scripts import refit_t2_20b_revisions as refit
from scripts import review_t2_20b_refits as review


def test_graph_comparison_ignores_only_overridden_root_guesses():
    def graph(guess, scope="trajectory"):
        return {
            "rules": {
                "X": {
                    "initial": {
                        "mode": "value",
                        "scope": scope,
                        "role": "nonnegative_coefficient",
                        "guess": guess,
                    }
                }
            }
        }

    a, b = graph(1.5), graph(0.1)
    assert refit.initialization_policy(a) == refit.initialization_policy(b)
    assert a["rules"]["X"]["initial"]["guess"] == 1.5
    assert refit.initialization_policy(a) != refit.initialization_policy(
        graph(1.5, "global")
    )
    assert refit.initialization_policy(
        graph(0.1, "global")
    ) != refit.initialization_policy(graph(1.5, "global"))


def test_alternate_start_changes_the_same_parent_coordinate_and_keeps_roots():
    parent = {"k_prod": 0.02, "trajectory:train_000:A": 1.4}
    for added in (
        {},
        {"k_I_feedback": 1e-8, "k_X_decay_glucose": 1e-8},
        {
            "af_process_gain_6_0": 0.0,
            "k_glucose_action_thr": 0.02,
            "Gp_thr_action": 300.0,
        },
    ):
        warm = parent | added
        model = SimpleNamespace(
            warm=warm,
            channels=["Gp"],
            observed=np.array([[[1.0], [3.0]]]),
            reduce=lambda p: None,
        )
        alt = refit.start_vector(model, "activated")
        assert refit.start_vector(model, "supplied") == warm
        assert alt["k_prod"] == pytest.approx(0.016)
        assert alt["trajectory:train_000:A"] == 1.4
        assert all(alt[n] != warm[n] for n in added)
        with pytest.raises(ValueError, match="unknown start"):
            refit.start_vector(model, "unplanned")


def test_distinguish_recovery_shape_from_absolute_level_and_match_smoke(tmp_path):
    t = np.arange(301.0)
    obs = np.arange(301.0) / 60.0
    got = review.changes(t, obs, obs + 100.0)
    assert got["observed_change"] == got["fitted_change"] == 1
    assert got["absolute_change_error"] == 0
    assert got["rmse_240_300"] == 100
    reversed_change = review.changes(t, obs, -obs)
    assert reversed_change["absolute_change_error"] == 2
    smoke = {"pooled_training_nmse": 0.1, "I_300_minus_240_pmol_L": 1.0}
    refit.check_start({"training_nmse": 0.1, "late_insulin_change": 1.0}, smoke)
    with pytest.raises(ValueError, match="fixed-start"):
        refit.check_start({"training_nmse": 0.1, "late_insulin_change": -1.0}, smoke)
    packet = tmp_path / "handoff.zip"
    packet.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="different"):
        refit.load_packet(packet)


def test_fit_retains_domain_passing_original_checkpoints_and_resumes(
    tmp_path, monkeypatch
):
    p = {"k_prod": 0.1}
    mock = SimpleNamespace(warm=p, names=("k_prod",), reduce=lambda point: None)
    monkeypatch.setattr(
        refit,
        "load_packet",
        lambda path: {
            "models": {
                "child": {
                    "smoke": {
                        "pooled_training_nmse": 0.1,
                        "I_300_minus_240_pmol_L": 60.0,
                    }
                }
            }
        },
    )
    monkeypatch.setattr(refit, "model_for", lambda *args: mock)
    evaluations = []

    def search(model, seeds, *, physical, control, seconds, calls, save):
        evaluations.append(calls)
        return {
            "status": "solver_returned",
            "search_calls_started": 8,
            "search_calls_completed": 7,
            "best": {
                "numerical": {"parameters": {"k_prod": 0.01}},
                "admissible": {"parameters": {"k_prod": 0.05}},
            },
        }

    def verify(model, point):
        return {
            "parameter_sha256": refit.public.content_sha256(point),
            "training_nmse": point["k_prod"],
            "sampled_domain_passed": point["k_prod"] >= 0.05,
            "late_insulin_change": 60.0,
            "parameters": point,
        }

    monkeypatch.setattr(refit.fit.fitting, "search", search)
    monkeypatch.setattr(refit, "verify_point", verify)
    plan = {"starts": {"child/supplied": refit.public.content_sha256(p)}}

    def fit():
        return refit.run_one(tmp_path / "packet", tmp_path, "child", "supplied", plan)

    row = fit()
    assert row["status"] == "complete"
    assert row["best"]["endpoint"]["training_nmse"] == 0.05
    assert row["numerical"]["endpoint"]["training_nmse"] == 0.01
    assert fit() == row
    assert evaluations == [1000, 1000, 2000, 2000]
    plan["starts"]["child/supplied"] = "different"
    with pytest.raises(ValueError, match="differs"):
        fit()


@pytest.mark.parametrize("branch,sign", [(refit.MODELS[1], 1), (refit.MODELS[2], -1)])
def test_late_balance_validates_the_sign_on_the_fitted_trajectory(
    monkeypatch, branch, sign
):
    p = {"Gp_thr": 18.0, "k_I_feedback": 0.0001, "k_X_decay_glucose": 0.001}
    coefficients = {
        "max_secretion": 0.4,
        "sigmoid_slope": 0.1,
        "sigmoid_midpoint": 18.0,
        "clearance": 0.04,
        "insulin_to_X": 0.001,
        "X_decay": 0.03,
        "appearance_rate": 0.05,
    }

    def rhs(t, state, params=None, forcing=None):
        g, ins, action, reservoir, basal = state
        product = (g - 18) * action
        return np.array(
            [
                0.05 * reservoir - basal - action,
                0.4 * expit(0.1 * (g - 18)) - 0.04 * ins + 0.0001 * product,
                0.001 * ins - 0.03 * action + sign * 0.001 * product,
                -0.05 * reservoir,
                0.0,
            ]
        )

    times = np.linspace(240, 300, 481)
    states = solve_ivp(
        rhs,
        (240, 300),
        [10.0, 3.0, 0.4, 4.0, 0.02],
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
    ).y.T
    model = SimpleNamespace(
        branch=branch,
        model=None,
        inputs=(),
        shared=tuple(p),
        equations=SimpleNamespace(rhs=rhs),
        states=("Gp", "I", "X_insulin_action", "A", "af_internal_aux_1"),
    )
    row = SimpleNamespace(
        time=times[::8],
        targets={"Gp": states[::8, 0], "U": states[::8, 2] + states[::8, 4]},
    )
    monkeypatch.setattr(review.insulin, "coefficients", lambda point: coefficients)
    monkeypatch.setattr(review, "trajectory_forcing", lambda *args: None)
    evidence = review.late_balance(model, p, row, times, states)
    x = evidence["balances"]["X_insulin_action"]
    assert x["change"] == pytest.approx(x["integral"], abs=2e-5)
    assert np.sign(x["integrated_terms"]["new_feedback"]) == -sign
    model.branch = refit.MODELS[2] if sign == 1 else refit.MODELS[1]
    with pytest.raises(AssertionError):
        review.late_balance(model, p, row, times, states)
