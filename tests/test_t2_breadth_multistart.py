"""The four starts are independent and only qualified training endpoints advance."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_t2_breadth_multistart as ms


def model(branch):
    return SimpleNamespace(
        branch=branch,
        names=("tau_M", "k_prod"),
        channels=["I", "U", "Gp"],
        scales=np.array([10.0, 2.0, 50.0]),
        reduce=lambda p: p,
    )


def recipes(_):
    return {
        "finite_meal_memory": {"tau_M": 30.0, "gain_M": 1.0, "k_M_rise": 2.0},
        "glucose_scaled_sigmoid": {"k_prod": 0.02, "Gp_thr": 170.0, "prod_max": 3.0},
    }


def test_pairs_vary_only_the_free_shape_coordinate(monkeypatch):
    monkeypatch.setattr(ms.campaign.fitting, "starts", recipes)
    a, b, c, d = [
        ms.start_parameters(model(branch), t) for t, (branch, _, _) in ms.TASKS.items()
    ]
    assert [a["tau_M"], b["tau_M"], c["k_prod"], d["k_prod"]] == [10, 120, 0.005, 0.08]
    assert {n for n in a if a[n] != b[n]} == {"tau_M"}
    assert {n for n in c if c[n] != d[n]} == {"k_prod"}
    with pytest.raises(ValueError, match="free coordinates"):
        ms.start_parameters(model("c15-2"), "short-memory")


def diagnostic(point, loss, *, feasible=True):
    shifts = [
        {
            "trajectory": "train_" + tid,
            "predicted_insulin_peak_min": i + 5,
            "observed_insulin_peak_min": i,
            "peak_time_error_min": 5,
            "predicted_peak_unique": True,
            "observed_peak_unique": True,
        }
        for tid, i in (("003", 101), ("006", 161), ("007", 221))
    ]
    return {
        "two_original_solvers_verified": True,
        "parameters": point,
        "parameter_sha256": ms.campaign.public.content_sha256(point),
        "training_nmse": loss,
        "sampled_domain_passed": feasible,
        "target_nmse": {"I": loss},
        "domain": {"I": {"minimum": 1 if feasible else -1}},
        "shifted_90g": shifts,
        "extension_150g": {"I300": 25.0, "observed_I300": 22.5},
        "maximum_scaled_solver_difference": 1e-9,
    }


def test_measurements_use_the_same_three_schedules_and_measured_endpoint():
    r = diagnostic(recipes(None)["finite_meal_memory"], 0.2)
    r["shifted_90g"].reverse()
    got = ms.measurements(r, "c15-3")
    assert got["peak_minutes"] == [106, 166, 226] and got["I300_error"] == 2.5
    r["shifted_90g"].pop()
    with pytest.raises(ValueError, match="shifted"):
        ms.measurements(r, "c15-3")


def test_distinct_optimizers_selection_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ms.campaign.fitting, "starts", recipes)
    monkeypatch.setattr(ms.campaign, "_model", lambda b, d: model(b))
    data = {
        "archive_sha256": "archive",
        "training_sha256": "train",
        "models": {b: {"model_sha256": b} for b in ("c15-2", "c15-3")},
        "boundary": {"trajectory": "train_005"},
    }
    plan = ms.prepare(tmp_path, data)
    seen = []

    def search(m, seeds, *, physical, control, seconds, calls: int, save):
        assert all(p["tau_M"] == seeds["start"]["tau_M"] for p in seeds.values())
        assert control is False
        seen.append((seeds["start"]["tau_M"], physical, calls))
        reduced = seeds["start"] | {"gain_M": -1.0}
        feasible = seeds["start"] | {"gain_M": 0.5}
        ret = {
            "status": "solver_returned",
            "search_calls_started": 5,
            "search_calls_completed": 5,
            "best": {
                "admissible": {"parameters": feasible},
                "numerical": {"parameters": reduced},
            },
        }
        save(ret)
        return ret

    monkeypatch.setattr(ms.campaign.fitting, "search", search)
    monkeypatch.setattr(
        ms.campaign,
        "_verified",
        lambda m, p, d, seconds: diagnostic(
            p, 1 + p["gain_M"], feasible=p["gain_M"] >= 0
        ),
    )
    for name in ("short-memory", "long-memory"):
        row = ms.fit_start(tmp_path, name, data, plan)
        assert row["status"] == "complete"
        assert row["best_numerical"]["training_nmse"] == 0
        assert row["best_admissible"]["training_nmse"] == 1.5
        assert ms.fit_start(tmp_path, name, data, plan) == row
    assert [tau for tau, _, _ in seen] == [10.0] * 4 + [120.0] * 4
    assert [limit for _, _, limit in seen] == [1000, 1000, 2000, 2000] * 2
    broken = dict(
        plan, starts=plan["starts"] | {"short-memory": plan["starts"]["long-memory"]}
    )
    with pytest.raises(ValueError, match="frozen start"):
        ms.fit_start(tmp_path, "short-memory", data, broken)
