"""Matched joint profiling retains evidence and never renews spent allowances."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_meal_joint_profile as c
from autoformalism.research import construction_meal_scale_recovery as previous
from tests import test_construction_meal_scale_recovery as fixtures


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
        endpoint_probe_seconds=2,
    )


@pytest.fixture(scope="module")
def historical_source(tmp_path_factory):
    """Build the immutable numerical provenance chain once, not once per assertion."""
    tmp_path = tmp_path_factory.mktemp("joint-historical-source")
    with pytest.MonkeyPatch.context() as monkeypatch:
        origin = fixtures.source.__wrapped__(monkeypatch, tmp_path)
        root = tmp_path / "scale-source"
        previous.prepare(
            root, origin, fixtures.policy().model_copy(update={"maximum_calls": 5})
        )
        for i in (0, 1):
            previous.run_task(root, i)
    return root


@pytest.fixture
def source(historical_source, monkeypatch):
    from autoformalism.fitting import prefix_portfolio_timescale as timescale

    factory = timescale.make_oracle
    monkeypatch.setattr(
        timescale,
        "make_oracle",
        lambda *a, **kw: factory(*a, **(kw | {"states": ("x", "z")})),
    )
    monkeypatch.setattr(public, "_source_identity", lambda: "synthetic-source")
    return historical_source


@pytest.fixture
def ready(source, tmp_path, monkeypatch):
    # Campaign tests route both arms through a real small affine profiler.
    # The independent joint-profiler suite covers the actual meal certificate.
    from autoformalism.fitting import prefix_meal_joint_profile as joint

    monkeypatch.setattr(joint, "JointMealProfile", roots.RootProfile)
    root = tmp_path / "joint-profile"
    c.prepare(root, source, policy())
    return root


def test_actual_ode_resume_and_inspection_preserve_all_receipts(ready, monkeypatch):
    for index in (0, 1, 2, 3):
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
    assert c.run_task(ready, 3) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["status"] == "complete"
    assert summary["recorded"] == summary["expected"] == 4
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_trial_seed_does_not_use_unconstrained_retained_point(ready):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 2)
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
    result = c.run_task(ready, 2)
    assert (
        result["retained"]["training_nmse"]
        <= result["source_endpoint"]["training_nmse"]
    )
    assert (
        result["retained"]["training_nmse"]
        <= result["source_trial_endpoint"]["training_nmse"]
    )
    assert result["trial_endpoint"] == result["source_trial_endpoint"]
    assert result["trial_origin"] == "source_verified_trial"
    assert result["trial_training_nmse"] >= result["training_reconstruction_nmse"]
    assert result["search_termination"] == "not_started"


def test_failed_qualification_keeps_previously_verified_trial(ready, monkeypatch):
    monkeypatch.setattr(
        roots, "root_profile_qualification", lambda *a, **k: {"passed": False}
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **k: pytest.fail("fit")
    )
    result = c.run_task(ready, 2)
    assert (
        result["retained"]["training_nmse"]
        <= result["source_endpoint"]["training_nmse"]
    )
    assert (
        result["retained"]["training_nmse"]
        <= result["source_trial_endpoint"]["training_nmse"]
    )
    assert result["trial_endpoint"] == result["source_trial_endpoint"]


def test_verifications_use_global_and_trial_losses_independently(ready, monkeypatch):
    calls = []
    original = c._Run.verify_endpoint

    def verify(self, stage, parameters, expected=None):
        calls.append((stage, dict(parameters), expected))
        return original(self, stage, parameters, expected)

    monkeypatch.setattr(c._Run, "verify_endpoint", verify)
    result = c.run_task(ready, 2)
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
    result = c.run_task(ready, 2)
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
        c.run_task(ready, 2)
    result = c.run_task(ready, 2)
    assert calls == [7]
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["trial_endpoint"] == result["source_trial_endpoint"]


def test_reservations_explicitly_include_headroom_and_trial_check():
    p = c.Policy()
    stages = c._allowances({"kind": "root_profile"}, p)
    assert stages["search"] == 2760
    assert stages["trial_start_verification"] == 360
    assert stages["endpoint_probe"] == 360
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
    wrong["source_plan"]["policy"]["population"] = "scaled-readout-meal-scale-1"
    with pytest.raises(ValueError, match=r"synthetic|roster"):
        c.prepare_inputs(tmp_path / "bypass", wrong, policy())


def test_resealed_terminal_cannot_claim_convergence(ready):
    c.run_task(ready, 0)
    path = ready / "results/toy-control-root_only/result.json"
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
    task = plan["tasks"][2]
    inherited = data["records"][1]["terminal"]["trial_endpoint"]
    derived = c.base._assemble(
        {"plan_sha256": public.content_sha256(plan), "task": task},
        c._retention_data(data),
        {},
        policy(),
    )
    derived["trial_endpoint"] = inherited | {
        "training_nmse": inherited["training_nmse"] - 1e-12,
    }
    monkeypatch.setattr(c.base, "_assemble", lambda *a: derived)
    result = c._assemble(
        {"plan_sha256": public.content_sha256(plan), "task": task}, data, {}, policy()
    )
    assert result["trial_origin"] == "source_verified_trial"
    assert result["trial_endpoint"] == inherited
    assert result["inherited_trial_vs_new_trial_comparison"]["classification"] == (
        "numerically_tied"
    )


def test_paired_tasks_share_exact_start_and_new_allowances(ready):
    plan, data = c.verify(ready)
    assert len(plan["tasks"]) == 4
    for i in (0, 2):
        a, b = plan["tasks"][i : i + 2]
        assert (a["arm"], b["arm"]) == c.ARMS
        assert a["source_trial_sha256"] == b["source_trial_sha256"]
        assert a["source_record_sha256"] == b["source_record_sha256"]
        assert a["record_index"] == b["record_index"]
        assert a["experiment"] == b["experiment"]
        assert a["task_id"] != b["task_id"]
        assert c._allowances(a, policy()) == c._allowances(b, policy())
    assert len(data["records"]) == 2


def test_joint_certificate_failure_never_falls_back_to_root_only(ready, monkeypatch):
    from autoformalism.fitting import prefix_meal_joint_profile as joint

    def reject(*a, **k):
        raise ValueError("unsupported joint coefficient certificate")

    monkeypatch.setattr(joint, "JointMealProfile", reject)
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **k: pytest.fail("fit")
    )
    result = c.run_task(ready, 1)
    assert result["trial_origin"] == "source_verified_trial"
    assert result["search_termination"] == "not_started"
    assert result["arm"] == "joint"
    assert result["endpoint_probe_status"] == "not_started"


def test_report_pairs_only_independently_replayed_projections(ready):
    c.run_task(ready, 0)
    summary = c.report(ready)
    assert summary["paired_projection_comparisons"][0]["status"] == "unavailable"
    c.run_task(ready, 1)
    summary = c.report(ready)
    pair = summary["paired_projection_comparisons"][0]
    assert pair["status"] == "verified"
    assert (
        pair["source_trial_sha256"]["root_only"] == pair["source_trial_sha256"]["joint"]
    )
    assert not pair["outer_convergence_certified"]
    assert not pair["parameter_confidence_certified"]
    assert pair["comparison"]["classification"] == "numerically_tied"


def test_endpoint_probe_is_separate_and_cannot_change_retention(ready, monkeypatch):
    from autoformalism.fitting import profile_endpoint_probe as probe

    calls = []

    def diagnostics(oracle, point, seconds, checkpoint, **kwargs):
        calls.append((point, seconds, kwargs))
        assert kwargs["fixed"] == {"read_gain": 0}
        assert point["read_gain"] == 0
        result = {
            "status": "feasible_descent_detected",
            "calls_started": 3,
            "calls_completed": 3,
            "calls_failed": 0,
            "planned_calls": 5,
            "best_probe": {"training_nmse": 0, "parameters": {"forged": 1}},
            "stationarity_certified": False,
        }
        checkpoint(result)
        return result

    monkeypatch.setattr(probe, "profile_endpoint_probe", diagnostics)
    result = c.run_task(ready, 2)
    assert len(calls) == 1 and calls[0][1] == 2
    assert result["endpoint_probe_status"] == "feasible_descent_detected"
    assert result["endpoint_probe_calls"]["calls_completed"] == 3
    assert result["trial_training_nmse"] > 0
    assert "forged" not in result["retained"]["parameters"]
    assert not result["optimization_converged"]
    assert not result["endpoint_probe"]["subject"]["retention_changes_permitted"]
    c.run_task(ready, 2)
    assert len(calls) == 1


def test_interrupted_endpoint_probe_never_repeats_allowance(ready, monkeypatch):
    from autoformalism.fitting import profile_endpoint_probe as probe

    calls = []

    def diagnostics(oracle, point, seconds, checkpoint, **kwargs):
        calls.append(seconds)
        checkpoint({"status": "interrupted", "calls_started": 1})
        raise KeyboardInterrupt

    monkeypatch.setattr(probe, "profile_endpoint_probe", diagnostics)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert calls == [2]
    assert result["operation_statuses"]["endpoint_probe"] == "interrupted"
    assert result["endpoint_probe_status"] == "interrupted"
    assert result["retained"]["original_equations_verified"]


def test_projection_pair_rejects_different_frozen_starts():
    rows = [
        {
            "seed": "same-label",
            "arm": arm,
            "identity": {"task": {"source_trial_sha256": arm}},
            "profiled_start_endpoint": None,
        }
        for arm in c.ARMS
    ]
    with pytest.raises(ValueError, match="different source starts"):
        c._projection_pairs(rows, policy())
