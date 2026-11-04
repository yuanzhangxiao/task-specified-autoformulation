"""Scale recovery preserves saved rate trials, the global incumbent and budgets."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_meal_continuation as previous
from autoformalism.research import construction_meal_scale_recovery as c
from tests import test_construction_meal_continuation as fixtures


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
        persistence_headroom_seconds=5,
    )


@pytest.fixture
def source(monkeypatch, tmp_path):
    origin = fixtures.source.__wrapped__(monkeypatch, tmp_path)
    root = tmp_path / "meal-source"
    previous.prepare(
        root, origin, fixtures.policy().model_copy(update={"maximum_calls": 5})
    )
    for i in (0, 1):
        previous.run_task(root, i)
    return root


@pytest.fixture
def ready(source, tmp_path):
    root = tmp_path / "scale-recovery"
    c.prepare(root, source, policy())
    return root


def test_actual_ode_resume_and_inspection_preserve_all_receipts(ready, monkeypatch):
    for index in (0, 1):
        result = c.run_task(ready, index)
        assert result["qualification"]["passed"]
        assert result["retained"]["original_equations_verified"]
        assert result["trial_endpoint"]["original_equations_verified"]
        assert not result["optimization_converged"]
        assert not result["profile_minimum_certified"]
        assert not result["parameter_region_excluded"]
        assert result["search_allowance"] == {
            "numerical_seconds": 2.0,
            "objective_calls": 10,
            "overall_wall_seconds": 7.0,
            "persistence_headroom_seconds": 5.0,
        }
        assert result["durable_improvements_preserved"]
        assert result["operation_statuses"]["trial_start_verification"] == "complete"
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 1) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["status"] == "complete"
    assert summary["recorded"] == summary["expected"] == 2
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_trial_seed_does_not_use_unconstrained_retained_point(ready):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 1)
    setup = run.setup(lambda _: None)
    trial = data["records"][1]["terminal"]["trial_endpoint"]
    assert setup["start"] == trial["effective_parameters"]
    assert setup["source_start"]["read_gain"] != setup["start"]["read_gain"]
    assert setup["start"]["read_gain"] == 0
    assert setup["source_trial_sha256"] == public.content_sha256(trial)


def test_failed_setup_retains_global_and_same_rate_checkpoints(ready, monkeypatch):
    monkeypatch.setattr(
        c._Run, "setup", lambda *a: (_ for _ in ()).throw(ValueError("bad"))
    )
    result = c.run_task(ready, 1)
    assert result["retained"] == result["source_endpoint"]
    assert result["trial_endpoint"] == result["source_trial_endpoint"]
    assert result["trial_origin"] == "source_verified_trial"
    assert result["trial_training_nmse"] > result["training_reconstruction_nmse"]
    assert result["search_termination"] == "not_started"


def test_failed_qualification_keeps_previously_verified_trial(ready, monkeypatch):
    monkeypatch.setattr(
        roots, "root_profile_qualification", lambda *a: {"passed": False}
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **k: pytest.fail("fit")
    )
    result = c.run_task(ready, 1)
    assert result["retained"] == result["source_endpoint"]
    assert result["trial_endpoint"] == result["source_trial_endpoint"]


def test_verifications_use_global_and_trial_losses_independently(ready, monkeypatch):
    calls = []
    original = c._Run.verify_endpoint

    def verify(self, stage, parameters, expected=None):
        calls.append((stage, dict(parameters), expected))
        return original(self, stage, parameters, expected)

    monkeypatch.setattr(c._Run, "verify_endpoint", verify)
    result = c.run_task(ready, 1)
    assert calls[0][0] == "start_verification"
    assert calls[0][2] == result["source_endpoint"]["training_nmse"]
    assert calls[1][0] == "trial_start_verification"
    assert calls[1][2] == result["source_trial_endpoint"]["training_nmse"]
    assert calls[0][1]["read_gain"] != calls[1][1]["read_gain"]


def test_search_gets_separate_wall_headroom_and_checkpoint_policy(ready, monkeypatch):
    received = {}

    def search(oracle, start, seconds, maximum_calls, checkpoint, **kwargs):
        received.update(kwargs)
        assert seconds == 2 and maximum_calls == 10
        return {"stop_reason": "synthetic_stop"}

    monkeypatch.setattr(roots, "root_profile_search", search)
    result = c.run_task(ready, 1)
    assert received["overall_wall_seconds"] == 7
    assert received["checkpoint_policy"].interval_seconds == 30
    assert received["fixed"] == {"read_gain": 0}
    assert result["trial_origin"] == "source_verified_trial"


def test_interrupted_search_does_not_renew_allowance(ready, monkeypatch):
    calls = []

    def interrupt(*args, **kwargs):
        calls.append(kwargs["overall_wall_seconds"])
        args[4]({"best": None, "stop_reason": "interrupted"})
        raise KeyboardInterrupt

    monkeypatch.setattr(roots, "root_profile_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 1)
    result = c.run_task(ready, 1)
    assert calls == [7]
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["trial_endpoint"] == result["source_trial_endpoint"]


def test_reservations_explicitly_include_headroom_and_trial_check():
    p = c.Policy()
    stages = c._allowances({"kind": "root_profile"}, p)
    assert stages["search"] == 2760
    assert stages["trial_start_verification"] == 360
    assert (
        c.reservation_ceiling(
            {"policy": p.model_dump(mode="json"), "tasks": [{"kind": "root_profile"}]}
        )
        <= 13200
    )
    for value in (-1, 901, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            c.Policy(persistence_headroom_seconds=value)


@pytest.mark.parametrize("field", ["terminal", "problem", "operations"])
def test_source_tampering_rejected(source, tmp_path, field):
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


def test_source_files_are_never_rewritten(source, tmp_path):
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    c.prepare(tmp_path / "new", source, policy())
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}


def test_only_training_and_pinned_population(source, tmp_path):
    data = c._source_inputs(source)
    wrong = deepcopy(data)
    wrong["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        c.prepare_inputs(tmp_path / "validation", wrong, policy())
    with pytest.raises(ValueError, match="reviewed"):
        c.prepare_inputs(tmp_path / "prod", data, c.Policy())
    wrong = deepcopy(data)
    wrong["source_plan"]["policy"]["population"] = "scaled-readout-meal-1"
    with pytest.raises(ValueError, match=r"synthetic|roster"):
        c.prepare_inputs(tmp_path / "bypass", wrong, policy())


def test_resealed_terminal_cannot_claim_convergence(ready):
    c.run_task(ready, 0)
    path = ready / "results/toy-control/result.json"
    value = read_seal(path)
    value["optimization_converged"] = True
    public._write(path, {"sha256": public.content_sha256(value), "value": value})
    with pytest.raises(ValueError):
        c.report(ready)


def test_nested_and_identical_roots_rejected(tmp_path):
    for root, source in ((tmp_path, tmp_path), (tmp_path, tmp_path / "old")):
        with pytest.raises(ValueError, match="separate"):
            c.prepare(root, source, c.Policy())


def test_changed_source_trial_rate_rejected(source, tmp_path):
    data = c._source_inputs(source)
    terminal = data["records"][1]["terminal"]
    terminal["trial_endpoint"]["effective_parameters"]["read_gain"] = 1
    data["records"][1]["terminal_sha256"] = public.content_sha256(terminal)
    with pytest.raises(ValueError, match="fixed rate"):
        c.prepare_inputs(tmp_path / "bad-rate", data, policy())


def test_source_paths_must_be_safe_before_projection(tmp_path):
    # The source verifier rejects altered plans before any task is executed.
    seal(tmp_path / "plan.json", {"protocol": "foreign"})
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        c._source_inputs(tmp_path)


def test_constructed_policy_cannot_bypass_budget_validation(source, tmp_path):
    data = c._source_inputs(source)
    for changed in (
        policy().model_copy(update={"search_seconds": 3600}),
        policy().model_copy(update={"persistence_headroom_seconds": 4000}),
    ):
        root = tmp_path / "oversized"
        with pytest.raises(ValueError):
            c.prepare_inputs(root, data, changed)
        assert not (root / "plan.json").exists()


def test_numerically_tied_trial_does_not_replace_saved_trial(ready, monkeypatch):
    plan, data = c.verify(ready)
    task = plan["tasks"][1]
    inherited = data["records"][1]["terminal"]["trial_endpoint"]
    derived = previous._assemble(
        {"plan_sha256": public.content_sha256(plan), "task": task},
        c._retention_data(data),
        {},
        policy(),
    )
    derived["trial_endpoint"] = inherited | {
        "training_nmse": inherited["training_nmse"] - 1e-12,
    }
    monkeypatch.setattr(previous, "_assemble", lambda *a: derived)
    result = c._assemble(
        {"plan_sha256": public.content_sha256(plan), "task": task}, data, {}, policy()
    )
    assert result["trial_origin"] == "source_verified_trial"
    assert result["trial_endpoint"] == inherited
    assert result["inherited_trial_vs_new_trial_comparison"]["classification"] == (
        "numerically_tied"
    )
