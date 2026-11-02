"""Frozen source, conditional profiling, accounting, and deterministic resume."""

from contextlib import suppress
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.fitting import prefix_portfolio_followup as followup
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.research import construction_portfolio_profile as campaign
from autoformalism.research import construction_portfolio_search as source
from tests.test_construction_portfolio_continuation import continued  # noqa: F401
from tests.test_construction_portfolio_fitting import completed  # noqa: F401
from tests.test_construction_portfolio_recovery import recovered  # noqa: F401
from tests.test_construction_portfolio_search import (
    install_profile_controls,
    prepared,  # noqa: F401
)


def policy():
    return campaign.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        coordinate_seconds=10,
        search_seconds=2,
        maximum_calls=30,
        verification_seconds=10,
        diagnostic_seconds=10,
    )


@pytest.fixture
def ready(prepared, tmp_path, monkeypatch):  # noqa: F811
    install_profile_controls(monkeypatch)
    monkeypatch.setattr(source._Run, "endpoints", lambda *args: None)
    for j in range(2):
        source.run_task(prepared, j)
    root = tmp_path / "profile"
    before = {p: p.read_bytes() for p in prepared.rglob("*.json")}
    campaign.prepare(root, prepared, policy())
    assert before == {p: p.read_bytes() for p in prepared.rglob("*.json")}
    return root, prepared


def install_root_controls(monkeypatch):
    """Use actual independent rollout with a synthetic orchestration-only profiler."""
    monkeypatch.setattr(
        roots,
        "root_profile_qualification",
        lambda oracle, parameters, seconds, save: {
            "parameters": parameters,
            "conditional_root_optimality_checked": True,
            "passed": True,
        },
    )

    def search(oracle, parameters, seconds, calls, checkpoint):
        from time import monotonic

        residual, _ = oracle.evaluate(
            oracle.vector(parameters), monotonic() + 10, jacobian=False
        )
        result = {
            "best": {
                "parameters": parameters,
                "training_nmse": float(np.mean(residual**2)),
            },
            "calls_started": 1,
            "calls_completed": 1,
            "calls_failed": 0,
            "calls_attempted": 1,
            "status": "complete",
            "budget_exhausted": False,
        }
        checkpoint(result)
        return result

    monkeypatch.setattr(roots, "root_profile_search", search)
    monkeypatch.setattr(
        followup, "meal_precision", lambda *args: {"passed": False, "toy_control": True}
    )


def test_roster_profile_verification_resume_and_report(ready, monkeypatch):
    root, _ = ready
    install_root_controls(monkeypatch)
    plan, _ = campaign.verify(root)
    assert len(plan["tasks"]) == 4
    assert len({t["task_id"] for t in plan["tasks"]}) == 4
    assert {t["seed"] for t in plan["tasks"]} == {"incumbent", "action_tau_10"}
    results = [campaign.run_task(root, i) for i in range(4)]
    assert all(r["status"] == "complete" for r in results)
    assert all(r["qualification"]["passed"] for r in results)
    assert all(r["retained"]["original_equations_verified"] for r in results)
    assert all(r["cost"]["search_calls_started_total"] == 1 for r in results)
    assert not any(r["training_prefix_reestimated"] for r in results)
    before = {p: p.read_bytes() for p in (root / "results").rglob("*.json")}
    monkeypatch.setattr(campaign._Run, "setup", lambda *args: pytest.fail("recompiled"))
    assert campaign.run_task(root, 0) == results[0]
    assert campaign.report(root) == campaign.report(root)
    assert campaign.inspection(root).is_file()
    assert before == {p: p.read_bytes() for p in (root / "results").rglob("*.json")}


def test_failed_certificate_blocks_search_preserves_incumbent(ready, monkeypatch):
    root, _ = ready
    monkeypatch.setattr(
        roots,
        "RootProfile",
        lambda *args: (_ for _ in ()).throw(ValueError("nonaffine root map")),
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *args: pytest.fail("searched")
    )
    result = campaign.run_task(root, 0)
    assert result["retained"] == result["source_endpoint"]
    assert not result["qualification"].get("passed")
    assert "search" not in result["operation_statuses"]


def test_interrupted_search_is_not_repeated(ready, monkeypatch):
    root, _ = ready
    install_root_controls(monkeypatch)
    old = roots.root_profile_search
    calls = []

    def interrupt(*args):
        calls.append(1)
        old(*args)
        raise KeyboardInterrupt

    monkeypatch.setattr(roots, "root_profile_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_task(root, 0)
    result = campaign.run_task(root, 0)
    assert len(calls) == 1
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["cost"]["search_calls_started_total"] is None
    assert result["cost"]["search_calls_started_observed"] == 1


def test_source_and_terminal_tampering_rejected(ready, tmp_path):
    root, old = ready
    with pytest.raises(ValueError, match="separate"):
        campaign.prepare(old / "nested", old, policy())
    plan, data = campaign.verify(root)
    changed = deepcopy(data)
    changed["records"][0]["terminal"]["training_reconstruction_nmse"] += 1
    with pytest.raises(ValueError, match="authentication"):
        campaign.prepare_inputs(tmp_path / "bad", changed, policy())
    with pytest.raises(ValueError, match="reviewed"):
        campaign.prepare_inputs(tmp_path / "production", data, campaign.Policy())
    assert campaign.reservation_ceiling(plan) < 13200


@pytest.mark.parametrize("same", [True, False])
def test_identical_endpoint_diagnosis_reused_only_at_same_point(monkeypatch, same):
    selected = {
        "parameters": {"a": 1},
        "training_nmse": 1,
        "original_equations_verified": True,
    }
    run = object.__new__(campaign._Run)
    run.record = {
        "terminal": {"retained": selected},
        "source_task": {"kind": "nonlinear_search"},
    }
    run.ops = {}
    run.original = object()
    run.oracle = SimpleNamespace(portfolio_audit={"model": "frozen"})
    run.policy = policy()
    calls = []

    def perform(stage, work):
        value = work(lambda _: None)
        run.ops[stage] = {"status": "complete", "value": value}
        return value

    run.perform = perform
    run.verify_endpoint = lambda stage, parameters: selected | {
        "parameters": parameters
    }
    monkeypatch.setattr(campaign.coordinates, "to_effective", lambda o, c, p: p)

    def diagnose(
        original, oracle, parameters, verification, search, seconds, checkpoint
    ):
        calls.append(parameters)
        return {
            "parameters": parameters,
            "matched": parameters == search["last_accepted_parameters"],
        }

    monkeypatch.setattr(campaign.numerical, "diagnose", diagnose)
    run.diagnose({"last_accepted_parameters": {"a": 1 if same else 2}})
    assert len(calls) == (1 if same else 2)
    assert run.ops["endpoint_alias"]["value"]["same_parameters"] is same


def test_counted_nested_views_are_separate_and_restored():
    class Oracle:
        def evaluate(self, x, deadline, *, jacobian=True, method=None):
            if x < 0:
                raise ValueError("invalid")
            return [x], None

    original, reduced = Oracle(), Oracle()
    saved = []

    def work(checkpoint):
        original.evaluate(1, 2, jacobian=False, method="Radau")
        reduced.evaluate(1, 2)
        with suppress(ValueError):
            reduced.evaluate(-1, 2)
        checkpoint({"passed": True})
        return {"passed": True}

    value = followup.counted(work, saved.append, original, reduced)
    assert value["evaluation_requests_by_view"]["optimizer"] == {
        "started": 2,
        "completed": 1,
        "failed": 1,
        "jacobian": 2,
    }
    assert value["evaluation_requests_by_view"]["original"]["started"] == 1
    assert "evaluate" not in vars(original) and "evaluate" not in vars(reduced)
    with pytest.raises(KeyboardInterrupt):
        followup.counted(
            lambda save: (_ for _ in ()).throw(KeyboardInterrupt()),
            saved.append,
            original,
            reduced,
        )
    assert "evaluate" not in vars(original) and "evaluate" not in vars(reduced)


def test_counter_aliases_do_not_mutate_source_and_are_not_added_twice():
    old = {"calls_started": 120, "calls_completed": 120, "calls_failed": 0}
    new = followup.normalise_counts(old, search=True)
    assert new["search_calls_started"] == 120
    assert "search_calls_started" not in old
    assert followup.normalise_counts(new, search=True) == new


def test_interruption_receipt_retains_view_ledger():
    class Oracle:
        def evaluate(self, vector, deadline, **kwargs):
            return np.array([vector]), None

    oracle = Oracle()

    def work(save):
        oracle.evaluate(1, 2, method="BDF")
        error = KeyboardInterrupt()
        error.execution_receipt = {"status": "interrupted"}
        raise error

    with pytest.raises(KeyboardInterrupt) as raised:
        followup.counted(work, lambda _: None, oracle, oracle)
    receipt = raised.value.execution_receipt
    assert receipt["evaluation_requests_by_view"]["original"]["completed"] == 1
    assert "evaluate" not in vars(oracle)


def test_verified_inner_checkpoint_survives_unstarted_outer_search():
    def op(value):
        return {"status": "complete", "value": value}

    old = {"training_nmse": 2, "original_equations_verified": True}
    better = old | {"training_nmse": 1}
    record = {
        "terminal": {"retained": old},
        "source_task": {"kind": "timescale_profiles"},
    }
    operations = {
        "start_verification": op(old),
        "qualification": op({"passed": True}),
        "profiled_start_verification": op(better),
        "search": {"status": "interrupted", "observed_progress": {}},
    }
    assert campaign._selection(record, operations) == (
        "profiled_start_verification",
        better,
    )
    operations["start_verification"] = op({"original_equations_verified": False})
    assert campaign._selection(record, operations) == ("source", old)


def test_nonlinear_endpoint_cannot_bypass_reduction_certificate():
    old = {"training_nmse": 2, "original_equations_verified": True}
    new = old | {"training_nmse": 1}
    record = {
        "terminal": {"retained": old},
        "source_task": {"kind": "nonlinear_search"},
    }
    ops = {
        name: {"status": "complete", "value": value}
        for name, value in (
            ("qualification", {"passed": True}),
            ("start_verification", old),
            ("endpoint_verification", new),
        )
    }
    assert campaign._selection(record, ops) == ("source", old)


def test_unavailable_nonlinear_qualification_is_not_repeated_as_diagnosis(monkeypatch):
    from autoformalism.fitting import prefix_portfolio_nonlinear_reduction as reduced

    run = object.__new__(campaign._Run)
    run.task = {"kind": "nonlinear_reduction", "seed": "incumbent"}
    run.policy = policy()
    run.start = {"a": 1}
    run.original = run.oracle = None
    run.ops = {}
    endpoint = {
        "parameters": run.start,
        "training_nmse": 1,
        "original_equations_verified": True,
    }
    run.record = {"terminal": {"retained": endpoint}}

    def perform(stage, work):
        result = work(lambda _: None)
        run.ops[stage] = {"status": "complete", "value": result}
        return result

    run.perform = perform
    run.verify_endpoint = lambda stage, *args: perform(stage, lambda save: endpoint)
    run.diagnose = lambda *args: pytest.fail("duplicated failed prerequisite")
    monkeypatch.setattr(reduced, "chart_equivalence", lambda *args: {"passed": True})
    monkeypatch.setattr(
        campaign.numerical, "qualification", lambda *args: {"passed": False}
    )
    monkeypatch.setattr(
        campaign.numerical, "search", lambda *args: pytest.fail("unqualified search")
    )
    run.fit()
    assert "search" not in run.ops


def test_precision_reuse_requires_exact_vector_and_prerequisites(monkeypatch):
    monkeypatch.setattr(
        followup.nonlinear, "_precision_identity", lambda oracle: {"frozen": True}
    )
    params = {"a": 1.0}
    residual = np.array([0.3])
    value = {
        "identity": {"frozen": True},
        "kind": "residual_probe",
        "method": "BDF",
        "vector": params,
        "vector_sha256": campaign.public.content_sha256(params),
        "status": "complete",
        "passed": True,
        "residual": residual.tolist(),
        "residual_sha256": followup.adequacy._digest(residual),
    }
    source_probes = {"probe_00": {"status": "complete", "value": value}}
    assert (
        followup.reuse_precision_probe(None, source_probes, "probe_00", params) == value
    )
    assert (
        followup.reuse_precision_probe(None, source_probes, "probe_01", params) is None
    )
    with pytest.raises(ValueError, match="identity"):
        followup.reuse_precision_probe(None, source_probes, "probe_00", {"a": 2})
    changed = deepcopy(source_probes)
    changed["probe_00"]["value"]["residual"] = [9]
    with pytest.raises(ValueError, match="digest"):
        followup.reuse_precision_probe(None, changed, "probe_00", params)
