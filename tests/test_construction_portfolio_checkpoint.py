"""Authenticated continuation, compact recovery and precise stopping reasons."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_portfolio_root_profile as roots
from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_portfolio_checkpoint as c
from autoformalism.research import construction_portfolio_residual as previous
from tests import test_construction_portfolio_residual as fixtures
from tests.test_construction_portfolio_residual import policy as source_policy


def policy():
    return c.Policy(
        population="synthetic-smoke",
        prerequisite_seconds=10,
        search_seconds=2,
        maximum_calls=30,
        verification_seconds=10,
        boundary_seconds=10,
        checkpoint_interval_seconds=30,
    )


@pytest.fixture
def source(monkeypatch, tmp_path):
    intake = fixtures.intake.__wrapped__(monkeypatch)
    root = tmp_path / "source"
    p = source_policy().model_copy(update={"maximum_calls": 5})
    previous.prepare_inputs(root, intake, p)
    previous.run_task(root, 0)
    return root


@pytest.fixture
def ready(source, tmp_path):
    root = tmp_path / "continued"
    c.prepare(root, source, policy())
    return root


def test_compact_campaign_actual_ode_and_idempotent_receipt_resume(ready, monkeypatch):
    result = c.run_task(ready, 0)
    assert result["qualification"]["passed"]
    assert result["retained"]["original_equations_verified"]
    assert not result["optimization_converged"]
    assert result["checkpoint_policy"]["interval_seconds"] == 30
    assert result["checkpoint_persistence"]["writes_completed"] > 0
    folder = ready / "results/toy-profiled"
    progress = public._read(folder / "operations/search/progress.json")
    assert progress["best"]["parameters"]
    assert progress["compact_progress"]["omitted_static_sha256"]
    assert "root_profile_certificate" not in progress
    search = read_seal(folder / "operations/search/result.json")["value"]
    assert "root_profile_certificate" in search
    before = {p: p.read_bytes() for p in folder.rglob("*.json")}
    monkeypatch.setattr(c._Run, "setup", lambda *a: pytest.fail("recompiled"))
    assert c.run_task(ready, 0) == result
    assert c.report(ready) == c.report(ready)
    assert c.inspection(ready).is_file()
    assert before == {p: p.read_bytes() for p in folder.rglob("*.json")}


def test_source_history_unchanged_and_nonnested_projection(source, tmp_path):
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    data = c._source_inputs(source)
    assert "source_inputs" not in data
    assert "source_inputs" not in data["records"][0]
    c.prepare_inputs(tmp_path / "new", data, policy())
    assert before == {p: p.read_bytes() for p in source.rglob("*.json")}
    assert data["records"][0]["reconciliation"]["scientific_fields_exact"]
    assert (
        data["records"][0]["terminal"]["retained"]
        == read_seal(source / "results/toy-profiled/result.json")["retained"]
    )


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


def test_only_training_and_reviewed_source(source, tmp_path):
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


@pytest.mark.parametrize(
    "search,expected",
    [
        ({}, "not_started"),
        (
            {
                "timing": {"timeout_reason": "overall_wall_limit"},
                "budget_exhausted": True,
            },
            "overall_wall_limit",
        ),
        (
            {
                "timing": {"timeout_reason": "numerical_active_wall_limit"},
                "budget_exhausted": True,
            },
            "numerical_active_wall_limit",
        ),
        ({"stop_reason": "call_limit", "budget_exhausted": True}, "evaluation_limit"),
        ({"search_calls_attempted": 20, "maximum_calls": 20}, "evaluation_limit"),
        ({"budget_exhausted": True}, "budget_limit_unclassified"),
        ({"optimizer_success": True}, "optimizer_stopped"),
        ({"stop_reason": "interrupted"}, "interrupted"),
        ({"timing": {"callback_failures": 1}}, "checkpoint_failure"),
        ({"checkpoint_persistence": {"writes_failed": 1}}, "checkpoint_failure"),
        (
            {
                "timing": {"timeout_reason": "overall_wall_limit"},
                "search_calls_attempted": 20,
                "maximum_calls": 20,
            },
            "overall_wall_limit",
        ),
    ],
)
def test_stopping_reason_uses_measured_evidence(search, expected):
    assert c._termination(search) == expected


def test_interrupted_compact_checkpoint_retained_without_search_retry(
    ready, monkeypatch
):
    calls = []

    def interrupt(oracle, start, seconds, maximum_calls, checkpoint, **kwargs):
        calls.append(kwargs["checkpoint_policy"].interval_seconds)
        qualified = roots.root_profile_qualification(oracle, start, 10, lambda _: None)
        checkpoint(
            {
                "best": qualified,
                "search_calls_attempted": 1,
                "search_calls_started": 1,
                "search_calls_completed": 1,
                "stop_reason": "interrupted",
                "compact_progress": {"omitted_static_sha256": {}},
            }
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(roots, "root_profile_search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(ready, 0)
    result = c.run_task(ready, 0)
    assert calls == [30]
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["retained"]["original_equations_verified"]
    assert result["search_termination"] == "interrupted"
    assert result["partial_progress_does_not_renew_budget"]


def test_hard_interruption_does_not_spend_fresh_search_budget(ready, monkeypatch):
    plan, data = c.verify(ready)
    run = c._Run(ready, plan, data, 0)
    folder = run.folder / "operations/search"
    folder.mkdir(parents=True)
    bound = {
        "campaign": run.identity,
        "stage": "search",
        "allowance_seconds": run.allowances["search"],
    }
    seal(folder / "started.json", bound)
    public._write(
        folder / "progress.json",
        {
            "best": None,
            "search_calls_attempted": 3,
            "stop_reason": "in_progress",
        },
    )
    monkeypatch.setattr(
        roots, "root_profile_search", lambda *a, **kw: pytest.fail("retried")
    )
    result = c.run_task(ready, 0)
    receipt = read_seal(folder / "result.json")
    assert receipt["budget_charge_seconds"] == run.allowances["search"]
    assert receipt["accounting_complete"] is False
    assert result["operation_statuses"]["search"] == "interrupted"
    assert result["search_termination"] == "interrupted"
    assert result["retained"]["original_equations_verified"]


def test_runtime_and_resealed_terminal_tampering_rejected(ready, monkeypatch):
    c.run_task(ready, 0)
    path = ready / "results/toy-profiled/result.json"
    value = read_seal(path)
    value["search_termination"] = "forged_converged"
    public._write(path, {"sha256": public.content_sha256(value), "value": value})
    with pytest.raises(ValueError):
        c.report(ready)
    monkeypatch.setattr(public, "_source_identity", lambda: "changed")
    with pytest.raises(ValueError, match="code/runtime"):
        c.run_task(ready, 0)


def test_default_numerical_allowances_and_disjoint_roots(tmp_path):
    p = c.Policy()
    assert p.search_seconds == 1800 and p.maximum_calls == 2000
    plan = {"policy": p.model_dump(mode="json"), "tasks": [{"kind": "root_profile"}]}
    assert c.reservation_ceiling(plan) == 4620
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path, tmp_path / "old", p)
    with pytest.raises(ValueError, match="separate"):
        c.prepare(tmp_path / "old/new", tmp_path / "old", p)
