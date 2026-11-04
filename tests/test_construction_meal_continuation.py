"""Meal continuations retain certified source endpoints and bounded profile evidence."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_meal_continuation as c
from autoformalism.research import construction_portfolio_targeted as previous
from autoformalism.schemas.public_fitting import PublicSplit
from tests import test_construction_portfolio_targeted as fixtures


def policy():
    return c.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        search_seconds=2,
        maximum_calls=10,
        verification_seconds=10,
        boundary_seconds=10,
        checkpoint_interval_seconds=30,
        diagnostic_seconds=2,
        diagnostic_calls=10,
    )


@pytest.fixture
def source(monkeypatch, tmp_path):
    origin = fixtures.source.__wrapped__(monkeypatch, tmp_path)
    root = tmp_path / "targeted-source"
    previous.prepare(root, origin, fixtures.policy().model_copy(
        update={"maximum_calls": 5}
    ))
    previous.run_task(root, 0)
    return root


@pytest.fixture
def ready(source, tmp_path):
    root = tmp_path / "meal"
    c.prepare(root, source, policy())
    return root


def test_actual_ode_controls_resume_and_inspection_without_recomputing(
    ready, monkeypatch
):
    for index in (0, 1):
        result = c.run_task(ready, index)
        assert result["qualification"]["passed"]
        assert result["retained"]["original_equations_verified"]
        assert result["trial_endpoint"]["original_equations_verified"]
        assert not result["optimization_converged"]
        assert not result["profile_minimum_certified"]
        assert not result["parameter_region_excluded"]
        assert result["search_allowance"] == {
            "numerical_seconds": 2.0, "objective_calls": 10
        }
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 1) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["status"] == "complete"
    assert summary["recorded"] == summary["expected"] == 2
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_fixed_trial_is_independently_verified_separately_from_better_incumbent(
    ready,
):
    result = c.run_task(ready, 1)
    assert result["experiment"]["fixed"] == {"read_gain": 0.0}
    assert result["trial_endpoint"]["effective_parameters"]["read_gain"] == 0.0
    assert result["trial_training_nmse"] != result["source_endpoint"]["training_nmse"]
    assert result["training_reconstruction_nmse"] <= (
        result["source_endpoint"]["training_nmse"] + 1e-10
    )
    assert result["trial_training_nmse"] >= result["training_reconstruction_nmse"]
    assert "upper bound" in result["trial_scope"]
    assert not result["profile_minimum_certified"]


def test_perturbed_trial_qualification_does_not_use_inherited_loss_gate(
    ready, monkeypatch
):
    requested = []
    original = c._Run.verify_endpoint

    def verify(self, stage, parameters, expected=None):
        requested.append((stage, dict(parameters), expected))
        return original(self, stage, parameters, expected)

    monkeypatch.setattr(c._Run, "verify_endpoint", verify)
    result = c.run_task(ready, 1)
    source = requested[0]
    trial = next(row for row in requested if row[0] == "profiled_start_verification")
    assert source[0] == "start_verification"
    assert source[2] == result["source_endpoint"]["training_nmse"]
    assert source[1]["read_gain"] != trial[1]["read_gain"]
    assert trial[1]["read_gain"] == 0
    assert trial[2] != source[2]
    assert result["qualification"]["passed"]


def test_failed_trial_qualification_preserves_source_without_search(
    ready, monkeypatch
):
    monkeypatch.setattr(
        roots, "root_profile_qualification", lambda *a: {"passed": False}
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **kw: pytest.fail("searched")
    )
    result = c.run_task(ready, 1)
    assert result["retained"] == result["source_endpoint"]
    assert result["retained_origin"] == "source"
    assert result["trial_training_nmse"] is None
    assert result["search_termination"] == "not_started"


def test_failed_trial_setup_preserves_authenticated_incumbent(ready, monkeypatch):
    def broken(*args):
        raise ValueError("synthetic setup failure")

    monkeypatch.setattr(c._Run, "setup", broken)
    result = c.run_task(ready, 1)
    assert result["retained"] == result["source_endpoint"]
    assert result["trial_endpoint"] is None
    assert result["search_termination"] == "not_started"


def test_wrong_held_coordinate_cannot_be_verified(ready):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 1)
    run.setup(lambda _: None)
    with pytest.raises(ValueError, match="held coordinate"):
        run.verify_endpoint("endpoint_verification", run.start | {"read_gain": 1.0})


def test_interrupted_search_keeps_verified_trial_without_new_allowance(
    ready, monkeypatch
):
    calls = []

    def interrupt(oracle, start, seconds, maximum_calls, checkpoint, **kwargs):
        calls.append((seconds, maximum_calls, kwargs["fixed"]))
        qualified = roots.root_profile_qualification(oracle, start, 10, lambda _: None)
        checkpoint({
            "best": qualified,
            "search_calls_attempted": 1,
            "search_calls_started": 1,
            "search_calls_completed": 1,
            "stop_reason": "interrupted",
        })
        raise KeyboardInterrupt

    monkeypatch.setattr(roots, "root_profile_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 1)
    result = c.run_task(ready, 1)
    assert calls == [(2.0, 10, {"read_gain": 0.0})]
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["search_termination"] == "interrupted"
    assert result["retained"]["original_equations_verified"]
    assert result["trial_endpoint"]["original_equations_verified"]


def test_hard_interruption_never_renews_the_search_allowance(ready, monkeypatch):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 1)
    folder = run.folder / "operations/search"
    folder.mkdir(parents=True)
    seal(folder / "started.json", {
        "campaign": run.identity,
        "stage": "search",
        "allowance_seconds": run.allowances["search"],
    })
    public._write(folder / "progress.json", {
        "best": None, "search_calls_attempted": 3, "stop_reason": "in_progress"
    })
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **kw: pytest.fail("retried")
    )
    result = c.run_task(ready, 1)
    receipt = read_seal(folder / "result.json")
    assert receipt["budget_charge_seconds"] == run.allowances["search"]
    assert receipt["accounting_complete"] is False
    assert result["search_termination"] == "interrupted"


def test_source_history_unchanged_and_compact_projection(source, tmp_path):
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    data = c._source_inputs(source)
    assert "source_inputs" not in data
    assert "source_inputs" not in data["records"][0]
    c.prepare_inputs(tmp_path / "new", data, policy())
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}


@pytest.mark.parametrize("field", ["terminal", "problem", "operations"])
def test_tampered_source_evidence_rejected(source, tmp_path, field):
    data = c._source_inputs(source)
    if field == "terminal":
        data["records"][0][field]["retained"]["training_nmse"] += 1
    elif field == "problem":
        data["records"][0][field]["parameter_guesses"]["read_gain"] += 1
    else:
        data["records"][0][field]["setup"]["value"]["audit"]["numerical_policy"][
            "rtol"
        ] = 1e-4
    with pytest.raises(ValueError, match="authentication"):
        c.prepare_inputs(tmp_path / "bad", data, policy())


def test_only_training_and_reviewed_population(source, tmp_path):
    data = c._source_inputs(source)
    wrong = deepcopy(data)
    wrong["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        c.prepare_inputs(tmp_path / "validation", wrong, policy())
    with pytest.raises(ValueError, match="reviewed"):
        c.prepare_inputs(tmp_path / "prod", data, c.Policy())
    wrong = deepcopy(data)
    wrong["source_plan"]["policy"]["population"] = "scaled-readout-4"
    with pytest.raises(ValueError):
        c.prepare_inputs(tmp_path / "bypass", wrong, policy())


def test_resealed_terminal_tampering_cannot_change_scientific_claim(ready):
    c.run_task(ready, 0)
    path = ready / "results/toy-control/result.json"
    value = read_seal(path)
    value["profile_minimum_certified"] = True
    public._write(path, {"sha256": public.content_sha256(value), "value": value})
    with pytest.raises(ValueError):
        c.report(ready)


def test_six_controls_use_public_horizon_and_separate_zero_branch():
    train = PublicSplit.model_validate({
        "name": "train", "fingerprint": "synthetic-public-scales",
        "rows": [{
            "trajectory_id": "train0", "time": [0, 300],
            "targets": {"Gp": [100, 300]},
        }],
    })
    variants = c._variants({}, train, True)
    assert len(variants) == 6
    assert variants[0]["fixed"] == {}
    assert variants[0]["meal_rate_start"] is None
    for value, row in zip((0.001, 0.01, 0.1, 1.0), variants[1:5], strict=True):
        assert row["fixed"] == {"k_abs": value / 300}
        assert row["meal_rate_start"] == value / 300
        assert row["coordinate_chart"] == "meal_flux"
        assert row["zero_representative"] is False
    assert variants[-1]["fixed"] == {"k_abs": 0.0, "k_abs_rate": 0.0}
    assert variants[-1]["coordinate_chart"] == "original"
    assert variants[-1]["zero_representative"] is True
    assert variants[0]["public_probe_scales"]["horizon"] == 300


def test_roots_must_be_separate_and_limits_fit_worker(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path, tmp_path / "old", c.Policy())
    p = c.Policy()
    plan = {
        "policy": p.model_dump(mode="json"),
        "tasks": [{"kind": "root_profile"}],
    }
    assert c.reservation_ceiling(plan) <= 13200
    assert p.diagnostic_seconds < p.search_seconds
    assert p.diagnostic_calls < p.maximum_calls


def _meal_run(monkeypatch, experiment):
    """Exercise campaign setup with an independently certified small meal ODE."""
    from autoformalism.fitting import prefix_meal_coordinates as meal
    from tests import test_prefix_meal_coordinates as fixtures_meal

    original, oracle = fixtures_meal.fixture()
    chart = fixtures_meal.certified(oracle)
    run = c._Run.__new__(c._Run)
    run.task = {"candidate_id": "candidate-15", "experiment": experiment}
    run.chart = None

    def setup(self, save):
        self.original, self.oracle = original, oracle
        self.start = dict(oracle.start)
        return {"passed": True}

    monkeypatch.setattr(c.retention._Run, "setup", setup)
    monkeypatch.setattr(meal, "certify", lambda *args: chart)
    return run, chart


def test_positive_fixed_start_preserves_flux_gain_and_all_scaled_initials(monkeypatch):
    run, chart = _meal_run(monkeypatch, {
        "fixed": {"k": 0.025}, "meal_rate_start": 0.025,
        "coordinate_chart": "meal_flux", "zero_representative": False,
    })
    value = run.setup(lambda _: None)
    assert run.start["k"] == 0.025
    assert run.start["k"] * run.start["q"] == pytest.approx(
        run.source_start["k"] * run.source_start["q"]
    )
    assert chart.scaled_initials(run.start) == pytest.approx(
        chart.scaled_initials(run.source_start)
    )
    assert value["zero_representative_certificate"] is None
    assert run.source_start["k"] == 0.1


def test_exact_zero_setup_uses_proven_output_equivalent_representative(monkeypatch):
    run, chart = _meal_run(monkeypatch, {
        "fixed": {"k": 0.0, "q": 0.0}, "meal_rate_start": None,
        "coordinate_chart": "original", "zero_representative": True,
    })
    value = run.setup(lambda _: None)
    assert run.start["k"] == run.start["q"] == 0
    assert all(run.start[name] == 0 for name in chart.root_names)
    assert value["zero_representative_certificate"]["fixed_shared_coordinates"] == {
        "k": 0.0, "q": 0.0
    }
    assert value["coordinate_start"]["flux_gain"] == 0
    assert run.source_start["k"] > 0
    assert run.source_start["q"] > 0


def test_zero_control_requires_the_exact_certified_fixed_coordinates(monkeypatch):
    run, _ = _meal_run(monkeypatch, {
        "fixed": {"k": 0.0}, "meal_rate_start": None,
        "coordinate_chart": "original", "zero_representative": True,
    })
    with pytest.raises(ValueError, match="certified coordinates"):
        run.setup(lambda _: None)


@pytest.mark.parametrize("raw,feasible", [(True, False), (False, True)])
def test_report_separates_raw_success_from_qualified_finite_termination(
    ready, monkeypatch, raw, feasible
):
    def invalid_success(oracle, start, seconds, calls, checkpoint, **kwargs):
        fitted = roots.root_profile_qualification(oracle, start, 10, lambda _: None)
        return {
            "best": fitted,
            "optimizer_raw_success": raw,
            "optimizer_success": raw and feasible,
            "optimizer_terminal": {
                "finite_objective": feasible,
                "objective": fitted["training_nmse"] if feasible else None,
                "pairing_status": "verified" if feasible else "unavailable",
                "feasible_evaluation_verified": feasible,
            },
            "stop_reason": "optimizer_terminal_unqualified",
            "search_calls_attempted": 1,
            "search_calls_started": 1,
            "search_calls_completed": 1,
        }

    monkeypatch.setattr(roots, "root_profile_search", invalid_success)
    result = c.run_task(ready, 0)
    assert result["optimizer_raw_success"] is raw
    assert result["optimizer_qualified_success"] is (raw and feasible)
    assert result["optimizer_feasible_termination"] is feasible
    assert result["optimizer_terminal"]["finite_objective"] is feasible
    assert not result["optimization_converged"]
    assert not result["stationarity_certified"]
    assert result["retained"]["original_equations_verified"]
    summary = c.report(ready)
    assert summary["rows"][0]["optimizer_terminal"] == result["optimizer_terminal"]
