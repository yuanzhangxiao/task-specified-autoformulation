"""Released-rate restarts and diagnostic candidates preserve authenticated evidence."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_meal_joint_profile as previous
from autoformalism.research import construction_meal_reliability as c
from tests import test_construction_meal_joint_profile as fixtures


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
        curvature_seconds=2,
        curvature_maximum_calls=12,
    )


@pytest.fixture(scope="module")
def historical_source(tmp_path_factory):
    """Build the existing immutable provenance chain once for all campaign tests."""
    origin = fixtures.historical_source.__wrapped__(tmp_path_factory)
    root = tmp_path_factory.mktemp("joint-source")
    with pytest.MonkeyPatch.context() as monkeypatch:
        from autoformalism.fitting import prefix_meal_joint_profile as joint

        source = fixtures.source.__wrapped__(origin, monkeypatch)
        monkeypatch.setattr(joint, "JointMealProfile", roots.RootProfile)
        previous.prepare(root, source, fixtures.policy())
        for index in (1, 3):
            previous.run_task(root, index)
    return root


@pytest.fixture
def source(historical_source, monkeypatch):
    from autoformalism.fitting import prefix_meal_joint_profile as joint
    from autoformalism.fitting import prefix_portfolio_timescale as timescale

    factory = timescale.make_oracle
    monkeypatch.setattr(
        timescale,
        "make_oracle",
        lambda *a, **kw: factory(*a, **(kw | {"states": ("x", "z")})),
    )
    monkeypatch.setattr(public, "_source_identity", lambda: "synthetic-source")
    monkeypatch.setattr(joint, "JointMealProfile", roots.RootProfile)
    return historical_source


@pytest.fixture
def ready(source, tmp_path):
    root = tmp_path / "reliability"
    c.prepare(root, source, policy())
    return root


def test_actual_ode_resume_and_reports_preserve_receipts(ready, monkeypatch):
    for index in (0, 1):
        result = c.run_task(ready, index)
        assert result["qualification"]["passed"]
        assert result["retained"]["original_equations_verified"]
        assert result["trial_endpoint"]["original_equations_verified"]
        assert not result["optimization_converged"]
        assert not result["global_minimum_certified"]
        assert not result["parameter_region_excluded"]
        assert result["operation_statuses"]["curvature_audit"] == "complete"
    before = {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 1) == result
    summary = c.report(ready)
    assert summary == c.report(ready)
    assert summary["status"] == "complete"
    assert summary["expected"] == summary["recorded"] == 2
    assert summary["scheduled_searches"] == 1
    assert summary["scheduled_audits"] == 2
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in (ready / "results").rglob("*.json")}


def test_warm_audit_never_runs_or_reserves_powell(ready, monkeypatch):
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **k: pytest.fail("fit")
    )
    result = c.run_task(ready, 0)
    assert result["mode"] == "audit_only"
    assert result["search_status"] == result["search_termination"] == "not_scheduled"
    assert result["search_allowance"]["objective_calls"] == 0
    assert result["search_allowance"]["numerical_seconds"] == 0
    assert not result["search_allowance"]["scheduled"]
    assert "search" not in result["operation_statuses"]
    assert "endpoint_verification" not in result["operation_statuses"]
    assert result["curvature_audit"]["subject"]["fixed_coordinates"] == {}


def test_released_start_is_exact_saved_trial_not_global_warm(ready):
    plan, data = c.verify(ready)
    task = plan["tasks"][1]
    run = c._Run(ready, plan, data, 1)
    setup = run.setup(lambda _: None)
    trial = data["records"][1]["terminal"]["trial_endpoint"]
    # Original coefficients are pinned exactly; reducing their floating-point
    # lift can differ by one ULP from the earlier saved effective coordinates.
    assert setup["source_trial_sha256"] == public.content_sha256(trial)
    assert setup["start"] == pytest.approx(
        trial["effective_parameters"], rel=1e-14, abs=1e-15
    )
    assert setup["source_start"]["read_gain"] != setup["start"]["read_gain"]
    assert task["mode"] == "released_rate_restart"
    assert task["experiment"]["fixed"] == {}
    assert task["experiment"]["source_fixed_coordinates"] == {"read_gain": 0}
    assert task["experiment"]["released_coordinates"] == ["read_gain"]
    assert data["records"][1]["source_task"]["experiment"]["fixed"] == {"read_gain": 0}


def test_released_search_passes_empty_fixed_map_and_new_wall_headroom(
    ready, monkeypatch
):
    calls = []

    def search(oracle, point, seconds, maximum_calls, save, **kwargs):
        calls.append((point, seconds, maximum_calls, kwargs))
        return {"stop_reason": "synthetic_stop"}

    monkeypatch.setattr(roots, "root_profile_search", search)
    result = c.run_task(ready, 1)
    assert len(calls) == 1
    point, seconds, maximum_calls, kwargs = calls[0]
    assert point["read_gain"] == 0
    assert seconds == 2 and maximum_calls == 10
    assert kwargs["fixed"] == {}
    assert kwargs["overall_wall_seconds"] == 7
    assert kwargs["checkpoint_policy"].interval_seconds == 30
    assert result["source_fixed_coordinates"] == {"read_gain": 0}
    assert result["search_allowance"]["scheduled"]


def test_search_interruption_never_renews_allowance(ready, monkeypatch):
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
    assert result["curvature_audit_status"] != "not_started"


def test_audit_interruption_never_renews_allowance(ready, monkeypatch):
    calls = []

    def interrupt(self, save):
        calls.append(self.policy.curvature_seconds)
        save({"status": "interrupted", "calls_started": 1})
        raise KeyboardInterrupt

    monkeypatch.setattr(c._Run, "audit_endpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert calls == [2]
    assert result["operation_statuses"]["curvature_audit"] == "interrupted"
    assert result["curvature_audit_status"] == "interrupted"
    assert result["retained"]["original_equations_verified"]


def test_audit_claim_requires_independent_verification_before_retention(
    ready, monkeypatch
):
    def fabricated(self, save):
        result = {
            "status": "descent_detected",
            "material_improvement": True,
            "best_candidate": {"parameters": dict(self.start), "training_nmse": -1},
        }
        save(result)
        return result

    monkeypatch.setattr(c._Run, "audit_endpoint", fabricated)
    result = c.run_task(ready, 0)
    assert "curvature_verification" in result["operation_statuses"]
    assert result["retained"]["training_nmse"] >= 0
    assert result["trial_origin"] != "curvature_verification"
    assert result["curvature_candidate_comparison"] is None


def test_verified_audit_candidate_updates_trial_and_global_comparisons(
    ready, monkeypatch
):
    plan, data = c.verify(ready)
    identity = {"plan_sha256": public.content_sha256(plan), "task": plan["tasks"][1]}
    derived = c.base._assemble(identity, c._retention_data(data), {}, policy())
    monkeypatch.setattr(c.base, "_assemble", lambda *a: deepcopy(derived))
    before = data["records"][1]["terminal"]["trial_endpoint"]
    better = before | {"training_nmse": 0.0, "solver_loss_discrepancy": 0.0}
    result = c._assemble(
        identity,
        data,
        {
            "curvature_verification": {"status": "complete", "value": better},
        },
        policy(),
    )
    assert result["trial_endpoint"] == better
    assert result["retained"] == better
    assert result["trial_before_audit"] == before
    assert (
        result["trial_origin"] == result["retained_origin"] == "curvature_verification"
    )
    assert (
        result["curvature_candidate_comparison"]["classification"]
        == "meaningful_improvement"
    )
    comparison = result["retained_vs_selected_trial_comparison"]
    assert comparison["old_training_nmse"] == comparison["new_training_nmse"] == 0
    assert result["inherited_trial_vs_final_trial_comparison"]["new_training_nmse"] == 0


def test_failed_setup_retains_both_authenticated_incumbents(ready, monkeypatch):
    monkeypatch.setattr(
        c._Run, "setup", lambda *a: (_ for _ in ()).throw(ValueError("bad"))
    )
    result = c.run_task(ready, 1)
    assert (
        result["retained"]["training_nmse"]
        <= result["source_endpoint"]["training_nmse"]
    )
    assert (
        result["retained"]["training_nmse"]
        <= result["source_trial_endpoint"]["training_nmse"]
    )
    assert result["trial_endpoint"] == result["source_trial_endpoint"]
    assert result["curvature_audit_status"] == "not_started"


def test_joint_certificate_failure_does_not_fall_back_or_search(ready, monkeypatch):
    from autoformalism.fitting import prefix_meal_joint_profile as joint

    def reject(*a, **k):
        raise ValueError("unsupported certificate")

    monkeypatch.setattr(joint, "JointMealProfile", reject)
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **k: pytest.fail("fit")
    )
    result = c.run_task(ready, 1)
    assert result["search_termination"] == "not_started"
    assert result["curvature_audit_status"] == "not_started"


def test_source_files_remain_unchanged(source, tmp_path):
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    c.prepare(tmp_path / "new", source, policy())
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}


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


def test_changed_original_fixed_rate_is_not_hidden_by_release(source, tmp_path):
    data = c._source_inputs(source)
    record = data["records"][1]
    record["terminal"]["trial_endpoint"]["effective_parameters"]["read_gain"] = 1
    record["terminal_sha256"] = public.content_sha256(record["terminal"])
    with pytest.raises(ValueError, match="fixed rate"):
        c.prepare_inputs(tmp_path / "bad", data, policy())


def test_training_separation_and_production_pins(source, tmp_path):
    data = c._source_inputs(source)
    wrong = deepcopy(data)
    wrong["training"]["name"] = "val"
    with pytest.raises(ValueError, match="training"):
        c.prepare_inputs(tmp_path / "val", wrong, policy())
    with pytest.raises(ValueError, match="reviewed"):
        c.prepare_inputs(tmp_path / "prod", data, c.Policy())
    wrong = deepcopy(data)
    wrong["source_plan"]["policy"]["population"] = "scaled-readout-meal-joint-1"
    with pytest.raises(ValueError, match="synthetic"):
        c.prepare_inputs(tmp_path / "bypass", wrong, policy())


def test_runtime_roster_and_summary_tampering_rejected(ready, monkeypatch):
    c.run_task(ready, 0)
    plan, _ = c.verify(ready)
    file = ready / "results" / plan["tasks"][0]["task_id"] / "result.json"
    value = read_seal(file)
    value["optimization_converged"] = True
    public._write(file, {"sha256": public.content_sha256(value), "value": value})
    with pytest.raises(ValueError):
        c.report(ready)
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="code/runtime"):
        c.verify(ready)


def test_reservations_bound_search_and_separate_curvature_work():
    p = c.Policy()
    restart = c._allowances(
        {"kind": "root_profile", "mode": "released_rate_restart"}, p
    )
    warm = c._allowances({"kind": "root_profile", "mode": "audit_only"}, p)
    assert restart["search"] == 2760
    assert restart["curvature_audit"] == 660
    assert restart["curvature_verification"] == 360
    assert sum(restart.values()) == 6180
    assert sum(warm.values()) == 3060
    assert "search" not in warm and "endpoint_verification" not in warm
    for field, value in [
        ("curvature_seconds", 361),
        ("curvature_maximum_calls", 221),
        ("persistence_headroom_seconds", 901),
        ("curvature_seconds", float("nan")),
    ]:
        with pytest.raises(ValueError):
            c.Policy(**{field: value})


def test_forged_policy_cannot_bypass_prepare_validation(source, tmp_path):
    data = c._source_inputs(source)
    for change in ({"search_seconds": 3600}, {"curvature_maximum_calls": 1000}):
        with pytest.raises(ValueError):
            c.prepare_inputs(tmp_path / "bad", data, policy().model_copy(update=change))
    assert not (tmp_path / "bad" / "plan.json").exists()


def test_nested_roots_rejected(tmp_path):
    for root, source in ((tmp_path, tmp_path), (tmp_path, tmp_path / "child")):
        with pytest.raises(ValueError, match="separate"):
            c.prepare(root, source, c.Policy())


def test_source_paths_reject_foreign_plans(tmp_path):
    seal(tmp_path / "plan.json", {"protocol": "foreign"})
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        c._source_inputs(tmp_path)


def test_partial_audit_candidate_is_replayed_without_repeating_audit(
    ready, monkeypatch
):
    calls = []

    def interrupt(self, save):
        calls.append(1)
        selected = self.best_verified_trial()
        save(
            {
                "status": "interrupted",
                "material_improvement": True,
                "best_candidate": {
                    "parameters": selected["effective_parameters"],
                    "training_nmse": selected["training_nmse"],
                },
            }
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(c._Run, "audit_endpoint", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert calls == [1]
    assert result["operation_statuses"]["curvature_audit"] == "interrupted"
    assert result["operation_statuses"]["curvature_verification"] == "complete"
    assert (
        result["curvature_candidate_comparison"]["classification"] == "numerically_tied"
    )
    assert result["trial_origin"] != "curvature_verification"


def test_report_missing_tasks_preserves_source_but_does_not_claim_finished(ready):
    report = c.report(ready)
    assert report["status"] == "incomplete"
    assert report["recorded"] == 0
    assert report["audit_status_counts"] == {"not_started": 2}
    assert all(r["status"] == "missing" for r in report["rows"])
    assert all(
        r["trial_endpoint"]["original_equations_verified"] for r in report["rows"]
    )
