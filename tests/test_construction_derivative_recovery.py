"""Derivative recovery repeats neither old budgets nor verification-only probes."""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("casadi")
pytest.importorskip("mpmath")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_derivative_recovery as c
from tests.test_construction_fit_reliability import Oracle, endpoint
from tests.test_construction_fit_reliability import (
    source as earlier_source,  # noqa: F401
)
from tests.test_construction_local_recovery import fake_local
from tests.test_construction_local_recovery import (
    source as reliability_source,  # noqa: F401
)


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        diagnostic_seconds=10,
        verification_seconds=10,
        maximum_calls=60,
    )


def test_campaign_diagnostic_budget_accepts_delivered_config_without_changing_parent():
    config = Path(__file__).resolve().parents[1] / (
        "configs/construction_derivative_recovery_v1.json"
    )
    delivered = c.Policy.model_validate(c.public._read(config))
    assert delivered.diagnostic_seconds == c.Policy().diagnostic_seconds == 600
    assert sum(c._allowances(delivered).values()) == 2670
    with pytest.raises(ValueError):
        c.Policy(diagnostic_seconds=601)
    assert c.previous.Policy().diagnostic_seconds == 300
    with pytest.raises(ValueError):
        c.previous.Policy(diagnostic_seconds=600)


@pytest.fixture
def source(reliability_source, tmp_path, monkeypatch):  # noqa: F811
    path = tmp_path / "local-inputs.json"
    c.previous.export_inputs(reliability_source, path)
    root = tmp_path / "local"
    c.previous.prepare(
        root, read_seal(path), c.previous.Policy(**policy().model_dump())
    )
    with monkeypatch.context() as patch:
        fake_local(patch, improve=False)
        c.previous.run_task(root, 0)
        c.previous.run_task(root, 1)
    return root


def inputs(source, tmp_path):
    path = tmp_path / "derivative-inputs.json"
    c.export_inputs(source, path)
    return read_seal(path)


def fail(*args, **kwargs):
    pytest.fail("unexpected numerical work")


def fake_recovery(
    monkeypatch,
    *,
    qualify=True,
    source_valid=True,
    candidate_valid=True,
    improve=True,
    qualification_status="complete",
):
    events = []

    def make(problem, training, numerical, anchor, *, precision_mode):
        del problem, training, numerical
        events.append(("build", anchor["a"], precision_mode))
        oracle = Oracle()
        oracle.affine_audit = {"backend": precision_mode, "calls": 0}
        oracle.recovery_audit = {"anchor": deepcopy(anchor), "inactive": ["fake"]}
        return oracle, oracle

    def verify(original, oracle, p, seconds, save, *, tolerances):
        assert tolerances == ((1e-11, 1e-13),)
        original.affine_audit["calls"] += 1
        events.append(("verify", p["a"], original.affine_audit["backend"]))
        valid = candidate_valid if p["a"] == 0.1 else source_valid
        value = endpoint(p, p["a"], valid)
        save(value)
        return value

    def qualification(
        problem, training, numerical, retained, seconds, save, *, adequacy_policy
    ):
        del problem, training, numerical, seconds
        assert adequacy_policy.local_adaptive and adequacy_policy.precision_aware
        assert adequacy_policy.local_bounds_aware
        events.append(("qualify", retained["a"]))
        value = {
            "passed": qualify,
            "status": qualification_status,
            "precision_mode": "multiprecision",
            "passes": [{"audit": {"backend": "multiprecision"}}],
        }
        save(value)
        return value

    def search(oracle, start, seconds, calls, save, *, reject_invalid_trials):
        assert reject_invalid_trials
        events.append(("search", start["a"], oracle.affine_audit["backend"]))
        point = start | {"a": 0.1 if improve else start["a"]}
        value = {
            "best": {"parameters": point, "training_nmse": point["a"]},
            "stop_reason": "step_or_gradient_converged",
            "calls": 1,
            "search_calls_started": 1,
        }
        save(value)
        return value

    def diagnose(original, oracle, p, checked, fit, seconds, save, *, adequacy_policy):
        assert adequacy_policy.local_bounds_aware
        events.append(("diagnose", p["a"], original.affine_audit["backend"]))
        value = {"assessment": {"classification": "locally_converged_poor_fit"}}
        save(value)
        return value

    monkeypatch.setattr(c.recovery, "make_oracle", make)
    monkeypatch.setattr(c.recovery, "qualification", qualification)
    monkeypatch.setattr(c.recovery, "diagnose", diagnose)
    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)
    monkeypatch.setattr(c.numerical, "search", search)
    monkeypatch.setattr(c.numerical, "predetermined_starts", fail)
    return events


def test_export_authenticates_parent_replays_without_exporting_their_endpoints(
    source, tmp_path, monkeypatch
):
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(c.public, "_runtime", fail)
    monkeypatch.setattr(c.public, "_source_identity", fail)
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    data = inputs(source, tmp_path)
    assert len(data["models"]) == 1 and len(data["parent_tasks"]) == 2
    assert data["models"][0]["endpoint"]["parameters"]["a"] == 0.25
    assert data["parent_tasks"][1]["kind"] == "verification_only"
    assert "endpoint" not in data["parent_tasks"][1]
    assert len(data["earlier_cost_by_model"]) == 1
    assert data["earlier_cost_by_model"][0]["arms"][1]["original_cost"] is None
    assert "source_inputs" not in data["training_inputs"]
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "bad.json")


@pytest.mark.parametrize("artifact", ["progress", "terminal", "reservation", "replay"])
def test_parent_authentication_rejects_altered_receipts(source, tmp_path, artifact):
    plan = read_seal(source / "plan.json")
    task = plan["tasks"][1 if artifact == "replay" else 0]
    folder = c.previous._folder(source, task)
    if artifact == "progress":
        path = folder / "operations/source_original_verification/progress.json"
        value = c.public._read(path)
        value["parameters"]["a"] += 1
        c.public._write(path, value)
    else:
        path = folder / (
            "operations/qualification/started.json"
            if artifact == "reservation"
            else "result.json"
        )
        value = read_seal(path)
        value["stage" if artifact == "reservation" else "status"] = "changed"
        path.unlink()
        seal(path, value)
    with pytest.raises(ValueError):
        c.export_inputs(source, tmp_path / "invalid.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "nested",
        "endpoint",
        "model",
        "origin",
        "parent_order",
        "replay_endpoint",
        "cost",
        "unknown_cost",
        "missing_history",
        "channels",
    ],
)
def test_strict_inputs_reject_identity_changes_and_unallowed_data(
    source, tmp_path, mutation
):
    data = inputs(source, tmp_path)
    if mutation == "heldout":
        data["validation"] = data["training_inputs"]["training"]
    elif mutation == "nested":
        data["training_inputs"]["validation"] = data["training_inputs"]["training"]
    elif mutation == "endpoint":
        data["models"][0]["endpoint"]["parameters"]["a"] += 1
    elif mutation == "model":
        data["models"][0]["problem_sha256"] = "a" * 64
    elif mutation == "origin":
        data["models"][0]["endpoint"]["origin"] = "disputed_verification"
    elif mutation == "parent_order":
        data["parent_tasks"].reverse()
    elif mutation == "replay_endpoint":
        data["parent_tasks"][1]["endpoint"] = data["models"][0]["endpoint"]
    elif mutation == "cost":
        data["parent_tasks"][0]["cost"]["test_nmse"] = 1.0
    elif mutation == "unknown_cost":
        data["earlier_cost_by_model"][0]["arms"][1]["original_cost_known"] = True
    elif mutation == "missing_history":
        data["earlier_cost_by_model"] = []
    else:
        data["models"][0]["endpoint"]["per_target_normalized_mse"]["hidden"] = 1.0
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "invalid", data, policy())


def test_selected_endpoint_reverified_then_chosen_precision_and_anchor_are_reused(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake_recovery(monkeypatch)
    value = c.run_task(root, 0)
    assert events == [
        ("qualify", 0.25),
        ("build", 0.25, "multiprecision"),
        ("verify", 0.25, "multiprecision"),
        ("build", 0.25, "multiprecision"),
        ("search", 0.25, "multiprecision"),
        ("build", 0.25, "multiprecision"),
        ("verify", 0.1, "multiprecision"),
        ("build", 0.25, "multiprecision"),
        ("diagnose", 0.1, "multiprecision"),
    ]
    assert value["source_incumbent"]["training_nmse"] == 0.25
    assert value["retained"]["training_nmse"] == 0.1
    assert value["search_started"] and value["search_not_started_reason"] is None
    assert value["optimization_status"] == "locally_converged_poor_fit"
    assert value["historical_verification_is_current"] is False
    assert (
        value["operations"]["continuation_verification"]["value"]["affine_execution"][
            "calls"
        ]
        == 1
    )
    assert value["operations"]["qualification"]["value"]["passes"]
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    assert c.run_task(root, 0) == value
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}
    with pytest.raises(ValueError, match="unknown derivative recovery"):
        c.run_task(root, 1)


@pytest.mark.parametrize(
    "failure", ["source", "qualification", "candidate", "incomplete"]
)
def test_no_false_certification_or_lost_historical_incumbent(
    source, tmp_path, monkeypatch, failure
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_recovery(
        monkeypatch,
        source_valid=failure != "source",
        qualify=failure != "qualification",
        candidate_valid=failure != "candidate",
        qualification_status="incomplete" if failure == "incomplete" else "complete",
    )
    value = c.run_task(root, 0)
    assert value["source_incumbent"]["original_equations_verified"]
    assert value["fitting_diagnosis"] is None
    assert value["optimization_status"] == "optimization_unresolved"
    if failure == "source":
        assert value["retained"] is None and value["selected_start"] is None
        assert (
            value["search_not_started_reason"]
            == "incumbent_current_verification_unavailable"
        )
    else:
        assert value["retained"]["training_nmse"] == 0.25
    if failure != "candidate":
        assert not value["search_started"]
        assert value["search_stop_reason"] == "not_started"


def test_interrupted_qualification_cannot_renew_budget(source, tmp_path, monkeypatch):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    plan, _ = c.verify(root)
    task = plan["tasks"][0]
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": task}
    directory = c._folder(root, task) / "operations/qualification"
    seal(
        directory / "started.json",
        {
            "campaign": identity,
            "stage": "qualification",
            "allowance_seconds": c._allowances(policy())["qualification"],
        },
    )
    fake_recovery(monkeypatch)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    monkeypatch.setattr(c.numerical, "search", fail)
    value = c.run_task(root, 0)
    assert value["operations"]["qualification"]["status"] == "interrupted"
    assert not value["cost"]["accounting_complete"]
    assert value["retained"]["training_nmse"] == 0.25
    assert value["search_not_started_reason"] == "qualification_interrupted"


def test_equal_retained_vector_may_inherit_own_search_diagnosis(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_recovery(monkeypatch, improve=False)
    value = c.run_task(root, 0)
    assert value["retained_origin"] == "incumbent_verification"
    assert value["fitting_diagnosis"] is not None


def test_report_preserves_old_cost_once_and_performs_no_fitting(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_recovery(monkeypatch)
    c.run_task(root, 0)
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    result = c.report(root)
    assert result["expected"] == result["recorded"] == 1
    assert len(result["earlier_cost_by_model"]) == 1
    assert len(result["parent_cost_by_task"]) == 2
    assert result["verification_only_tasks_repeated"] == 0
    assert all("original_cost" not in row for row in result["rows"])
    assert Path(c.inspect(root)["archive"]).is_file()


def test_multiprecision_runtime_change_blocks_execution_but_not_audit(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    plan, _ = c.verify(root)
    assert plan["runtime"]["precision_packages"]["mpmath"] == c.version("mpmath")
    altered = deepcopy(plan["runtime"])
    altered["precision_packages"]["mpmath"] = "different"
    monkeypatch.setattr(c, "_runtime", lambda: altered)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    with pytest.raises(ValueError, match="source/runtime differs"):
        c.run_task(root, 0)
    assert c.verify(root, runtime=False)[0] == plan


@pytest.mark.parametrize("artifact", ["terminal", "start"])
def test_resume_rejects_changed_retention(source, tmp_path, monkeypatch, artifact):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_recovery(monkeypatch)
    c.run_task(root, 0)
    path = (
        root
        / "results/task_00"
        / ("result.json" if artifact == "terminal" else "selected_start.json")
    )
    value = read_seal(path)
    value["status" if artifact == "terminal" else "training_nmse"] = "changed"
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError):
        c.run_task(root, 0)


def test_real_small_reduced_continuation(source, tmp_path, monkeypatch):
    root = tmp_path / "real"
    c.prepare(root, inputs(source, tmp_path), policy())
    result = c.run_task(root, 0)
    assert result["retained"] is not None, result["operations"]
    assert result["retained"]["original_equations_verified"]
    assert result["qualification_passed"], result["operations"]
    assert result["search_started"], result["operations"]
    assert result["retained"]["training_nmse"] < 1e-6
    assert result["cost"]["search_calls_started_total"] > 0
    assert not result["validation_data_opened"]
    assert not result["original_parameter_identifiability_certified"]
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    assert c.run_task(root, 0) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_diagnostic_cost_counts_precision_passes_without_top_level_duplicate():
    operations = {
        "qualification": {
            "status": "complete",
            "wall_seconds": 12,
            "value": {
                "derivative_check": {"calls_started": 7, "calls_completed": 7},
                "passes": [
                    {
                        "status": "complete",
                        "derivative_check": {
                            "calls_started": 5,
                            "calls_completed": 5,
                        },
                    },
                    {
                        "status": "complete",
                        "derivative_check": {
                            "calls_started": 7,
                            "calls_completed": 7,
                        },
                    },
                ],
            },
        },
        "continuation_diagnosis": {
            "status": "complete",
            "wall_seconds": 8,
            "value": {"derivative_check": {"calls_started": 3, "calls_completed": 3}},
        },
    }
    cost = c._diagnostic_cost(operations)
    assert cost["derivative_calls_started_observed"] == 15
    assert cost["derivative_calls_completed_observed"] == 15
    assert cost["stage_wall_seconds_observed"] == 20
    assert cost["precision_passes_observed"] == 2
    assert cost["complete"]
    first = operations["qualification"]["value"]["passes"][0]
    first["status"] = "incomplete"
    first["derivative_check"]["calls_completed"] = 4
    cost = c._diagnostic_cost(operations)
    assert cost["derivative_calls_started_observed"] == 15
    assert cost["derivative_calls_completed_observed"] == 14
    assert cost["stage_wall_seconds_observed"] == 20
    assert not cost["complete"]


def test_different_vector_cannot_inherit_retained_diagnosis():
    identity = {"test": "synthetic"}
    incumbent = endpoint({"a": 0.2}, 0.2)
    candidate = endpoint({"a": 0.3}, 0.2)
    operations = {
        "incumbent_verification": {"status": "complete", "value": incumbent},
        "continuation_verification": {"status": "complete", "value": candidate},
        "continuation_diagnosis": {
            "status": "complete",
            "value": {
                "assessment": {"classification": "locally_converged_poor_fit"},
            },
        },
    }
    for operation in operations.values():
        operation.update(accounting_complete=True, budget_charge_seconds=0)
    value = c._assemble(identity, {"endpoint": incumbent}, operations)
    assert value["retained"]["parameters"] == incumbent["parameters"]
    assert value["fitting_diagnosis"] is None
