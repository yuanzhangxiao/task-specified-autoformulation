"""Matched inputs, authenticated reuse, and nonrenewable continuation budgets."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_continuation_campaign as c
from tests.test_llm_sr_detention_recovery_campaign import (
    exported,
)
from tests.test_llm_sr_detention_recovery_campaign import (
    fake_numerics as recovery_numerics,
)
from tests.test_llm_sr_detention_recovery_campaign import (
    prepared as recovery_prepared,
)

__all__ = ["exported", "recovery_prepared"]


@pytest.fixture
def prepared(recovery_prepared, monkeypatch, tmp_path):
    oldroot, _, _ = recovery_prepared
    recovery_numerics(monkeypatch)
    for index in range(6):
        c.source.run_task(oldroot, index)
    (oldroot / "report").mkdir()
    public._write(oldroot / "report/summary.json", c.source.report(oldroot))
    destination = tmp_path / "continuation-inputs.json"
    c.export_inputs(oldroot, destination)
    inputs = read_seal(destination)
    monkeypatch.setattr(
        c.kernel, "domain", lambda *args, **kw: {"names": ["a"], "mode": kw["mode"]}
    )
    root = tmp_path / "continuation"
    c.prepare(
        root,
        inputs,
        c.Policy(
            optimization_seconds=1,
            release_seconds=1,
            direction_seconds=1,
            verification_seconds=1,
            aggregation_seconds=1,
        ),
    )
    return root, inputs, oldroot


def fake_numerics(
    monkeypatch, *, interrupt=False, qualified=True, improve=True, interrupt_fit=False
):
    calls = []

    def domain(*args, **kwargs):
        return {"names": ["a"], "mode": kwargs["mode"]}

    def imported(item, context, training, parameters, rows, **kwargs):
        calls.append(("import", kwargs["mode"], parameters.copy()))
        assert all(r["parameters"] == parameters for r in rows)
        return deepcopy(rows)

    def direction(item, context, training, parameters, index, **kwargs):
        calls.append(("direction", kwargs["mode"], parameters.copy()))
        kwargs["checkpoint"]({"status": "incomplete"})
        if interrupt:
            raise KeyboardInterrupt()
        return {"parameters": parameters, "index": index, "precision": "tight"}

    def qualification(item, context, training, parameters, rows, **kwargs):
        assert all(r["parameters"] == parameters for r in rows)
        result = {
            "parameters": parameters,
            "directions": rows,
            "mode": kwargs["mode"],
            "search_qualified": bool(rows) and qualified,
            "complete": bool(rows),
        }
        result["qualification_sha256"] = public.content_sha256(result)
        return result

    def validate(item, context, training, parameters, rows, **kwargs):
        expected = qualification(
            item, context, training, parameters, rows, mode=kwargs["mode"]
        )
        if expected != kwargs["qualification"]:
            raise ValueError("qualification differs")

    def fit(item, context, training, parameters, **kwargs):
        calls.append(("fit", kwargs["mode"], parameters.copy(), kwargs["seconds"]))
        if not kwargs["qualification"]["search_qualified"]:
            return {"calls": 0, "stop_reason": "derivatives_unqualified", "best": None}
        a = parameters["a"] / 2 if improve else parameters["a"] + 10
        result = {
            "calls": 1,
            "stop_reason": "budget_limited",
            "best": {"parameters": {"a": a}, "training_nmse": a * a},
        }
        kwargs["checkpoint"](result)
        if interrupt_fit:
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(c.kernel, "domain", domain)
    monkeypatch.setattr(c.kernel, "import_original_directions", imported)
    monkeypatch.setattr(c.kernel, "direction", direction)
    monkeypatch.setattr(c.kernel, "qualification", qualification)
    monkeypatch.setattr(c.kernel, "validate_qualification", validate)
    monkeypatch.setattr(c.kernel, "fit", fit)
    monkeypatch.setattr(
        c,
        "_enriched",
        lambda problem, parameters, raw, policy: raw
        | {
            "stationarity_assessment": {"status": "reliable_descent_remains"},
            "rank_assessment": {"original_uniqueness": "unclaimed"},
        },
    )
    return calls


def test_roster_pairs_and_history_are_preserved(prepared, monkeypatch):
    root, inputs, oldroot = prepared
    fake_numerics(monkeypatch)
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    plan, _, _ = c.verify(root)
    assert [x["source_index"] for x in plan["tasks"]] == [3, 3, 4, 4, 5, 5, 2]
    assert [x["mode"] for x in plan["tasks"]] == ["original", "reduced_model15"] * 3 + [
        "subset_model14"
    ]
    for task in plan["tasks"]:
        assert (
            task["parameters"]
            == inputs["parents"][task["source_index"]]["terminal"]["retained"][
                "parameters"
            ]
        )
    report = c.report(root)
    assert report["recorded"] == 0 and len(report["historical_costs"]) == 6
    assert all(p["same_start"] for p in report["paired_comparisons"])
    assert before == {p: p.read_bytes() for p in before}


def test_original_reuses_authenticated_checks_without_duplicate_work(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    result = c.run_task(root, 0)
    assert any(x[:2] == ("import", "original") for x in calls)
    assert not any(x[0] == "direction" for x in calls)
    assert result["retained_origin"] == "endpoint_verification"
    assert (
        result["diagnosis"]["stationarity_assessment"]["status"]
        == "reliable_descent_remains"
    )
    assert result["operations"]["import_directions"]["value"]["new_integrations"] == 0
    before = deepcopy(calls)
    assert c.run_task(root, 0) == result
    assert [x for x in calls if x[0] != "import"] == [
        x for x in before if x[0] != "import"
    ]
    assert c.report(root)["recorded"] == 1


def test_reduced_arm_uses_fresh_chart_checks_and_same_budget(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    a = c.run_task(root, 0)
    b = c.run_task(root, 1)
    assert any(x[:2] == ("direction", "reduced_model15") for x in calls)
    fits = [x for x in calls if x[0] == "fit"]
    assert fits[0][2:] == fits[1][2:]
    assert a["cost"]["operations"] < b["cost"]["operations"] + 2
    assert c.report(root)["paired_comparisons"][0]["same_start"]


def test_subset_full_recheck_precedes_conditional_release(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    result = c.run_task(root, 6)
    fits = [x for x in calls if x[0] == "fit"]
    assert [x[1] for x in fits] == ["subset_model14", "original"]
    assert fits[1][2]["a"] == fits[0][2]["a"] / 2
    assert result["retained_origin"] == "release_verification"
    assert "full-recheck-00" in result["operations"]
    assert "diagnosis-00" in result["operations"]
    assert result["release_stop_reason"] == "budget_limited"


def test_unqualified_full_recheck_cannot_release(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, qualified=False)
    result = c.run_task(root, 6)
    assert [x[1] for x in calls if x[0] == "fit"] == ["subset_model14"]
    assert result["retained_origin"] == "start_verification"
    assert result["release_stop_reason"] == "not_requested"
    assert "diagnosis-00" not in result["operations"]
    assert "full-recheck-00" in result["operations"]


def test_interrupted_direction_has_no_renewed_budget(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 1)
    result = c.run_task(root, 1)
    assert sum(x[0] == "direction" for x in calls) == 1
    assert result["operations"]["search-00"]["status"] == "interrupted"
    assert result["optimizer_stop_reason"] == "derivatives_unqualified"
    assert result["retained_origin"] == "start_verification"


def test_worse_candidates_preserve_own_incumbent(prepared, monkeypatch):
    root, inputs, _ = prepared
    fake_numerics(monkeypatch, improve=False)
    result = c.run_task(root, 0)
    assert result["retained_origin"] == "start_verification"
    assert (
        result["retained"]["parameters"]
        == inputs["parents"][3]["terminal"]["retained"]["parameters"]
    )


def test_plan_and_terminal_tampering_are_rejected(prepared, monkeypatch):
    root, _, _ = prepared
    fake_numerics(monkeypatch)
    c.run_task(root, 0)
    path = root / "tasks/0/result.json"
    raw = read_seal(path)
    raw["optimizer_stop_reason"] = "invented_convergence"
    path.unlink()
    seal(path, raw)
    with pytest.raises(ValueError, match="terminal differs"):
        c.report(root)


def test_history_cost_tampering_rejected(prepared):
    _, inputs, _ = prepared
    broken = deepcopy(inputs)
    broken["parents"][0]["terminal"]["cost"]["operations"] += 1
    with pytest.raises(ValueError, match="terminal differs"):
        c.Inputs.model_validate(broken)


def test_scheduler_allowance_covers_every_reserved_stage(prepared, monkeypatch):
    root, _, _ = prepared
    fake_numerics(monkeypatch)
    plan, _, _ = c.verify(root)
    for task in plan["tasks"]:
        assert sum(c._allowances(task, c.Policy()).values()) < 7200
    with pytest.raises(ValueError):
        c.Policy(optimization_seconds=1801)


def test_failed_fresh_verification_preserves_parent_without_new_conclusions(
    prepared, monkeypatch
):
    root, inputs, _ = prepared
    calls = fake_numerics(monkeypatch)
    monkeypatch.setattr(
        c.original,
        "verify",
        lambda *args, **kw: {
            "parameters": args[3],
            "usable_for_retention": False,
            "status": "unavailable",
        },
    )
    result = c.run_task(root, 0)
    assert not calls
    assert result["retained_origin"] == "parent_verified_checkpoint"
    assert result["retained"] == inputs["parents"][3]["terminal"]["retained"]
    assert not result["fresh_start_verification_usable"]
    assert result["diagnosis"] is None and result["qualification"] == {}


def test_resealed_import_cannot_change_its_authenticated_source(prepared, monkeypatch):
    root, _, _ = prepared
    fake_numerics(monkeypatch)
    c.run_task(root, 0)
    folder = root / "tasks/0/operations/import_directions"
    operation = read_seal(folder / "result.json")
    operation["value"]["new_integrations"] = 1
    public._write(folder / "progress.json", operation["value"])
    operation["progress_sha256"] = public.content_sha256(operation["value"])
    (folder / "result.json").unlink()
    seal(folder / "result.json", operation)
    with pytest.raises(ValueError, match="imported derivative evidence"):
        c.report(root)


def test_interrupted_search_retains_verified_checkpoint_without_restarting(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt_fit=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    result = c.run_task(root, 0)
    assert sum(x[0] == "fit" for x in calls) == 1
    assert result["operations"]["optimization"]["status"] == "interrupted"
    assert result["retained_origin"] == "endpoint_verification"
    assert (
        result["retained"]["training_nmse"]
        < result["identity"]["task"]["parent_verified_checkpoint"]["training_nmse"]
    )
    assert result["cost"]["accounting_complete"] is False
