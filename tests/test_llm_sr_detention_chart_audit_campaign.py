"""Fixed-point chart audits preserve history, identities, and consumed budgets."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_chart_audit_campaign as c
from tests.test_llm_sr_detention_continuation_campaign import (
    exported,
    recovery_prepared,
)
from tests.test_llm_sr_detention_continuation_campaign import (
    fake_numerics as continuation_numerics,
)
from tests.test_llm_sr_detention_continuation_campaign import (
    prepared as continuation_prepared,
)
from tests.test_llm_sr_detention_refit import example

__all__ = ["continuation_prepared", "exported", "recovery_prepared"]


@pytest.fixture
def prepared(continuation_prepared, monkeypatch, tmp_path):
    oldroot, _, _ = continuation_prepared
    continuation_numerics(monkeypatch, qualified=False)
    for index in range(7):
        c.source.run_task(oldroot, index)
    (oldroot / "report").mkdir()
    public._write(oldroot / "report/summary.json", c.source.report(oldroot))
    destination = tmp_path / "audit-inputs.json"
    c.export_inputs(oldroot, destination)
    inputs = read_seal(destination)
    old_domain, old_qualification = c.kernel.domain, c.kernel.qualification

    def domain(*args, **kwargs):
        if kwargs["mode"] != c.MODE:
            return old_domain(*args, **kwargs)
        return {"names": [f"x{i}" for i in range(8)], "mode": c.MODE}

    def qualification(item, context, training, parameters, records, **kwargs):
        if kwargs["mode"] != c.MODE:
            return old_qualification(
                item, context, training, parameters, records, **kwargs
            )
        assert all(r["parameters"] == parameters for r in records)
        assert all(r["mode"] == c.MODE for r in records)
        indices = [r["index"] for r in records]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate coordinate")
        rows = [
            {
                "index": r["index"],
                "name": f"x{r['index']}",
                "status": r.get("precision_status", "passed"),
                "search_qualified": r.get("qualified", True),
            }
            for r in sorted(records, key=lambda row: row["index"])
        ]
        gate = {
            "complete": len(records) == 8,
            "search_qualified": len(records) == 8
            and all(r["search_qualified"] for r in rows),
            "rows": rows,
            "coordinate_names": [f"x{i}" for i in range(8)],
            "missing_indices": [i for i in range(8) if i not in indices],
        }
        return gate | {"qualification_sha256": public.content_sha256(gate)}

    monkeypatch.setattr(c.kernel, "domain", domain)
    monkeypatch.setattr(c.kernel, "qualification", qualification)
    root = tmp_path / "audit"
    c.prepare(root, inputs, c.Policy(direction_seconds=1))
    return root, inputs, oldroot


def fake_direction(monkeypatch, *, interrupt=False, unavailable=False, blocked=()):
    calls = []

    def work(item, context, training, parameters, index, **kwargs):
        calls.append((parameters.copy(), index, kwargs["mode"]))
        kwargs["checkpoint"]({"incomplete": True})
        if interrupt:
            raise KeyboardInterrupt()
        if unavailable:
            raise RuntimeError("integration unavailable")
        return {
            "status": "complete",
            "mode": kwargs["mode"],
            "parameters": parameters,
            "index": index,
            "precision_status": "inconclusive" if index in blocked else "passed",
            "qualified": index not in blocked,
        }

    monkeypatch.setattr(c.kernel, "direction", work)
    monkeypatch.setattr(c.kernel, "fit", lambda *a, **k: pytest.fail("fit was called"))
    return calls


def test_frozen_roster_and_complete_history(prepared):
    root, inputs, oldroot = prepared
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    plan, _, _ = c.verify(root)
    assert len(plan["tasks"]) == 24
    assert [t["source_index"] for t in plan["tasks"]] == [1] * 8 + [3] * 8 + [5] * 8
    for task in plan["tasks"]:
        assert (
            task["parameters"]
            == inputs["source_plan"]["tasks"][task["source_index"]]["parameters"]
        )
    report = c.report(root)
    assert report["recorded"] == 0
    assert len(report["historical_costs"]) == 7
    assert all(len(g["missing_coordinates"]) == 8 for g in report["groups"])
    assert before == {p: p.read_bytes() for p in before}


def test_resume_reuses_coordinate_and_incomplete_group_cannot_qualify(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_direction(monkeypatch)
    a = c.run_task(root, 0)
    assert c.run_task(root, 0) == a
    assert len(calls) == 1
    assert a["coordinate_search_qualified"]
    assert not a["full_chart_qualification_claimed"]
    report = c.report(root)
    assert report["recorded"] == 1
    assert not report["groups"][0]["qualification"]["search_qualified"]
    assert len(report["groups"][0]["missing_coordinates"]) == 7


def test_all_coordinates_aggregate_and_inconclusive_excess_error_is_listed(
    prepared, monkeypatch
):
    root, _, _ = prepared
    fake_direction(monkeypatch, blocked=(2,))
    for index in range(8):
        c.run_task(root, index)
    group = c.report(root)["groups"][0]
    assert group["qualification"]["complete"]
    assert group["missing_coordinates"] == []
    assert group["blocked_coordinates"] == ["x2"]
    assert not group["qualification"]["search_qualified"]
    assert not group["convergence_or_parameter_confidence_claimed"]


def test_interruption_keeps_charge_and_does_not_repeat_work(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_direction(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    result = c.run_task(root, 0)
    assert len(calls) == 1
    assert result["operations"]["direction"]["status"] == "interrupted"
    assert result["cost"]["budget_charge_seconds"] > 0
    assert not result["coordinate_search_qualified"]


def test_failed_operation_is_reported_without_search_or_retry(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_direction(monkeypatch, unavailable=True)
    result = c.run_task(root, 0)
    assert result["status"] == "complete"
    assert result["operations"]["direction"]["status"] == "unavailable"
    assert not result["coordinate_search_qualified"]
    assert c.run_task(root, 0) == result and len(calls) == 1


def test_source_history_tampering_rejected(prepared):
    _, inputs, _ = prepared
    broken = deepcopy(inputs)
    broken["source_report"]["tasks"][0]["cost"]["operations"] += 1
    broken["source_report_sha256"] = public.content_sha256(broken["source_report"])
    with pytest.raises(ValueError, match="terminal differs"):
        c.Inputs.model_validate(broken)


def test_current_terminal_and_plan_tampering_rejected(prepared, monkeypatch):
    root, _, _ = prepared
    fake_direction(monkeypatch)
    c.run_task(root, 0)
    path = root / "tasks/0/result.json"
    value = read_seal(path)
    value["parameters_changed"] = True
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError, match="terminal differs"):
        c.report(root)
    path = root / "plan.json"
    value = read_seal(path)
    value["tasks"][0]["coordinate_index"] = 7
    path.unlink()
    seal(path, value)
    with pytest.raises(ValueError, match="plan differs"):
        c.verify(root)


def test_reporting_does_not_require_original_runtime_and_limits_are_bounded(
    prepared, monkeypatch
):
    root, _, _ = prepared
    monkeypatch.setattr(c.public, "_runtime", lambda: {"different": True})
    assert c.report(root)["status"] == "incomplete"
    with pytest.raises(ValueError, match="runtime differs"):
        c.run_task(root, 0)
    with pytest.raises(ValueError):
        c.Policy(direction_seconds=121)


@pytest.mark.parametrize("tamper", ["endpoint", "mode", "units", "index"])
def test_real_coordinate_receipt_authentication_rejects_mixed_evidence(
    monkeypatch, tamper
):
    problem = example(15)
    parameters = problem[0].parameters

    def residual(oracle, point, deadline, method="Radau"):
        return np.asarray([oracle.parameters(point)["beta"] - 0.3, 0.1])

    monkeypatch.setattr(c.kernel.original.DetentionOracle, "residual", residual)
    numerical = c.source.diagnostic.Policy()
    record = c.kernel.direction(
        *problem, parameters, 1, mode=c.MODE, policy=numerical, seconds=10
    )
    monkeypatch.setattr(c, "_problem", lambda *a: problem)
    monkeypatch.setattr(c, "_numerical", lambda *a: numerical)
    identity = {"task": {"coordinate_index": 1, "parameters": parameters}}
    operation = {
        "status": "complete",
        "value": record,
        "accounting_complete": True,
        "budget_charge_seconds": 0.0,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
    }
    valid = c._assemble(identity, {"direction": operation}, None, terminal=True)
    assert valid["derivative_status"] == "complete"
    broken = deepcopy(record)
    if tamper == "endpoint":
        broken["original_parameters"]["beta"] += 0.1
    elif tamper == "mode":
        broken["identity"]["mode"] = "reduced_model15"
    elif tamper == "units":
        broken["coordinate_units"][1] *= 2
    else:
        broken["index"] = 2
    broken["evidence_sha256"] = public.content_sha256(
        {k: v for k, v in broken.items() if k != "evidence_sha256"}
    )
    with pytest.raises(ValueError):
        c._assemble(
            identity, {"direction": operation | {"value": broken}}, None, terminal=True
        )
