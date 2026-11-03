"""Targeted controls retain authenticated incumbents and conditional-fit limits."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_checkpoint as previous
from autoformalism.research import construction_portfolio_targeted as c
from autoformalism.schemas.public_fitting import PublicSplit
from tests import test_construction_portfolio_checkpoint as fixtures


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
    root = tmp_path / "checkpoint-source"
    previous.prepare(root, origin, fixtures.policy().model_copy(
        update={"maximum_calls": 5}
    ))
    previous.run_task(root, 0)
    return root


@pytest.fixture
def ready(source, tmp_path):
    root = tmp_path / "targeted"
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
    with pytest.raises(ValueError, match="synthetic"):
        c.prepare_inputs(tmp_path / "bypass", wrong, policy())


def test_resealed_terminal_tampering_cannot_change_scientific_claim(ready):
    c.run_task(ready, 0)
    path = ready / "results/toy-control/result.json"
    value = read_seal(path)
    value["profile_minimum_certified"] = True
    public._write(path, {"sha256": public.content_sha256(value), "value": value})
    with pytest.raises(ValueError):
        c.report(ready)


def test_production_roster_is_twenty_predetermined_training_scaled_controls():
    train = PublicSplit.model_validate({
        "name": "train", "fingerprint": "synthetic-public-scales",
        "rows": [{
            "trajectory_id": "train0", "time": [0, 300],
            "targets": {"Gp": [100, 300]},
        }],
    })
    experiments = {}
    for name in ("candidate-10", "candidate-12", "candidate-14", "candidate-15"):
        record = {
            "source_task": {"candidate_id": name},
            "terminal": {"retained": {"parameters": {"Km_ins": 1e-5}}},
        }
        experiments[name] = c._variants(record, train, True)
    assert sum(map(len, experiments.values())) == 20
    for name in ("candidate-10", "candidate-14"):
        assert [v["fixed"]["Km_ins"] for v in experiments[name]] == [
            1e-5, 20.0, 200.0, 2000.0
        ]
        assert all(v["fixed"]["Km_ins"] > 0 for v in experiments[name])
    decay = experiments["candidate-12"]
    assert decay[0]["fixed"] == {}
    assert [v["fixed"]["k_A_decay"] for v in decay[1:]] == [
        0.0, 0.1 / 300, 1 / 300, 3 / 300
    ]
    meal = experiments["candidate-15"]
    assert len(meal) == 7
    for i in range(0, 6, 2):
        a, b = meal[i : i + 2]
        assert a["meal_rate_start"] == b["meal_rate_start"]
        assert a["coordinate_chart"] == "original"
        assert b["coordinate_chart"] == "meal_flux"
    assert meal[-1]["fixed"] == {"k_abs": 0.0}
    assert meal[-1]["coordinate_chart"] == "original"


def test_training_scale_rejects_zero_glucose():
    train = PublicSplit.model_validate({
        "name": "train", "fingerprint": "zero-scale",
        "rows": [{
            "trajectory_id": "train0", "time": [0, 300], "targets": {"Gp": [0, 0]}
        }],
    })
    with pytest.raises(ValueError, match="glucose scale"):
        c._training_scales(train)


def test_targeted_roots_must_be_separate_and_limits_fit_worker(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path, tmp_path / "old", c.Policy())
    p = c.Policy()
    plan = {
        "policy": p.model_dump(mode="json"),
        "tasks": [{"kind": "root_profile"}, {"kind": "residual_search"}],
    }
    assert c.reservation_ceiling(plan) <= 13200
    assert p.diagnostic_seconds < p.search_seconds
    assert p.diagnostic_calls < p.maximum_calls
