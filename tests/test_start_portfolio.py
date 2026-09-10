"""Initializer iterate retention, training-only selection, budgets and staged resume."""

import json
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting import start_portfolio as portfolio
from autoformalism.fitting.matching_probe import bounded_latent_start, latent_start
from autoformalism.rebuttal import fitter_methods as campaign
from autoformalism.rebuttal.fitter_diagnostic import read_json, write_json
from tests.test_fitter_methods import affine_training, small_plan


def test_iteration_limit_retains_diagnostics_and_finite_parameters(tmp_path):
    system, train = affine_training(small_plan())
    result = latent_start(
        system,
        train,
        np.array([0, 0, 0, -np.inf]),
        np.full(4, np.inf),
        np.array([2.0, 0.2, 1.2, 0.3]),
        1.0,
        small_plan().settings(),
        "collocation_init",
        15,
        tmp_path,
        record_progress=True,
        maximum_iterations=1,
    )
    assert not result["success"] and result["parameters"] is None
    assert "Maximum_Iterations_Exceeded" in result["message"]
    assert len(result["progress"]) >= 2
    saved = result["last_finite_iterate"]
    assert saved["finite_in_domain"] and not saved["rollout_verified"]
    assert saved["objective"] >= 0 and saved["constraint_maximum"] >= 0
    assert all(np.isfinite(list(saved["parameters"].values())))
    assert read_json(tmp_path / "last_finite.json") == saved
    assert json.loads((tmp_path / "iterations.json").read_text()) == result["progress"]


def test_hard_timeout_cannot_reuse_a_stale_saved_iterate(tmp_path):
    system, train = affine_training(small_plan())
    write_json(tmp_path / "last_finite.json", {"parameters": {"stale": 1}})
    write_json(tmp_path / "iterations.json", [{"stale": True}])
    result = bounded_latent_start(
        system,
        training=train,
        lower=np.array([0, 0, 0, -np.inf]),
        upper=np.full(4, np.inf),
        start=np.ones(4),
        scale=1,
        settings=small_plan().settings(),
        method="collocation_init",
        seconds=0.001,
        directory=tmp_path,
        record_progress=True,
    )
    assert not result["success"]
    assert result.get("last_finite_iterate") is None
    assert not (tmp_path / "last_finite.json").exists()


def minimal_training(name=SplitName.TRAIN):
    return DatasetSplit(
        name,
        (
            Trajectory(
                "t",
                np.array([0.0, 1.0, 2.0]),
                {"v01": np.array([0.0, 1.0, 0.0])},
                {},
                {},
                {},
                {},
            ),
        ),
        "frozen-training",
    )


def test_portfolio_chooses_progress_and_charges_every_stage(tmp_path, monkeypatch):
    calls = []
    bests = {"ordinary_pilot": 1.0, "collocation_pilot": 2.0, "continuation": 0.5}

    def stage(
        system,
        training,
        scale,
        settings,
        start,
        root,
        identity,
        seconds,
        count,
        deadline,
    ):
        name = root.name
        calls.append((name, seconds, count, start))
        # Initial loss favors collocation; the pilot result favors the ordinary start.
        return {
            "best": {"cost": bests[name], "parameters": {"x": bests[name]}},
            "report": {"optimizer_success": False},
            "seconds": 2,
            "calls": 4,
            "residual_seconds": 1,
            "integration_failures": 0,
            "initial_cost": 100 if name == "ordinary_pilot" else 0.1,
            "initial_rollout_valid": True,
        }

    monkeypatch.setattr(portfolio, "sensitivity_stage", stage)
    result = portfolio.portfolio_fit(
        None,
        minimal_training(),
        1,
        small_plan().settings(),
        {"x": 9},
        {"success": True, "parameters": {"x": 8}, "seconds": 5},
        tmp_path,
        "id",
        total_seconds=90,
        pilot_seconds=10,
        pilot_calls=4,
    )
    assert result["portfolio"]["pilot_winner"] == "ordinary_pilot"
    assert result["fit"]["parameters"] == {"x": 0.5}
    assert result["total_fit_seconds"] == 11
    assert result["fit"]["actual_residual_calls"] == 12
    assert calls[2][1:3] == (81, 42)  # 90 - init5 - pilots4; max50 - calls8.
    assert calls[2][3] == {"x": 1}
    assert not result["portfolio"]["validation_used"]
    assert not result["portfolio"]["reference_used"]


def test_selection_rejects_invalid_points_and_has_deterministic_ties():
    assert portfolio.choose_stage({"a": {"best": None}}) is None
    assert portfolio.choose_stage({"a": {"best": {"cost": float("nan")}}}) is None
    assert (
        portfolio.choose_stage(
            {
                "ordinary": {"best": {"cost": 1.0}},
                "collocation": {"best": {"cost": 1.0 + 1e-14}},
            }
        )
        == "ordinary"
    )


def test_native_sensitivity_stage_resume_and_total_call_limit(tmp_path, monkeypatch):
    system, train = affine_training(small_plan())
    arguments = {
        "system": system,
        "training": train,
        "scale": 1,
        "settings": small_plan().settings(),
        "start": {"a": 2, "b": 0.2, "c": 1.2, "d": 0.3},
        "root": tmp_path,
        "identity": "stage",
        "seconds": 10,
        "calls": 2,
        "outer_deadline": monotonic() + 15,
    }
    result = portfolio.sensitivity_stage(**arguments)
    assert result["initial_rollout_valid"] and result["best"] is not None
    assert result["calls"] == 2
    assert result["seconds"] < 10
    monkeypatch.setattr(
        portfolio, "instrumented_fit", lambda *a, **kw: pytest.fail("cache missed")
    )
    assert portfolio.sensitivity_stage(**arguments) == result
    with pytest.raises(ValueError, match="identity"):
        portfolio.sensitivity_stage(**{**arguments, "calls": 3})


def test_portfolio_rejects_validation_before_any_solver_call(tmp_path):
    with pytest.raises(ValueError, match="training split"):
        portfolio.portfolio_fit(
            None,
            minimal_training(SplitName.VALIDATION),
            1,
            small_plan().settings(),
            {},
            {},
            tmp_path,
            "id",
            total_seconds=90,
            pilot_seconds=10,
            pilot_calls=4,
        )


@pytest.mark.parametrize("method", ["start_portfolio", "start_portfolio_long"])
def test_v3_end_to_end_salvages_iterate_and_reuses_pilot_checkpoints(tmp_path, method):
    payload = small_plan().model_dump(mode="json")
    payload.update(
        protocol="fitter-methods-3",
        initializer_seconds=20,
        fit_seconds=60,
        pilot_seconds=4,
        pilot_calls=4,
        long_pilot_seconds=6,
        long_pilot_calls=6,
        initializer_iterations=1,
    )
    plan = campaign.MethodsPlan.model_validate(payload)
    frozen = campaign.prepare_methods(plan, tmp_path)
    assert len(frozen["tasks"]) == 6
    assert campaign.execute_methods(tmp_path, 0)["status"] == "complete"
    init = campaign.execute_methods(tmp_path, 1)
    assert init["status"] == "initializer_failed", init
    assert init["initializer"]["last_finite_iterate"]
    index = next(i for i, t in enumerate(frozen["tasks"]) if t.get("method") == method)
    result = campaign.execute_methods(tmp_path, index)
    assert result["status"] == "complete", result
    assert result["portfolio"]["unfinished_iterate_usable"]
    assert result["portfolio"]["pilot_calls"] == (
        4 if method == "start_portfolio" else 6
    )
    assert result["fit"]["actual_residual_calls"] <= 50
    assert result["total_fit_seconds"] <= 60
    assert result["clean_signal_nmse"]["train"] < 1e-4
    assert campaign.execute_methods(tmp_path, index) == result
    # Remove only the final envelopes, retaining the independently cached stages.
    task_root = tmp_path / "results" / frozen["tasks"][index]["name"]
    (task_root / "result.json").unlink()
    (task_root / "fit.json").unlink()
    resumed = campaign.execute_methods(tmp_path, index)
    assert resumed["status"] == "complete"
    assert resumed["fit"] == result["fit"]
    assert resumed["total_fit_seconds"] == result["total_fit_seconds"]
    summary = campaign.summarize_methods(tmp_path)
    diagnostic = next(
        d
        for d in summary["paired_summary"]["portfolio_diagnostics"]
        if d["method"] == method
    )
    assert diagnostic["unfinished_iterate_usable"]
    assert "Portfolio decisions" in (tmp_path / "summary.md").read_text()


def test_v3_matrix_and_budget_validation(tmp_path):
    path = Path(__file__).resolve().parents[1] / "configs/fitter_methods_v3.json"
    payload = read_json(path)
    frozen = campaign.prepare_methods(
        campaign.MethodsPlan.model_validate(payload), tmp_path
    )
    assert len(frozen["tasks"]) == 62
    assert {t["method"] for t in frozen["tasks"] if t["kind"] == "fit"} == set(
        campaign.PORTFOLIO
    )
    with pytest.raises(ValueError, match="reserve time"):
        campaign.MethodsPlan.model_validate({**payload, "fit_seconds": 150})
