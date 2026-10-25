"""Authenticated historical controls and resumable per-coordinate recovery."""

from copy import deepcopy
from math import nextafter

import pytest

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_recovery_campaign as c
from tests.test_llm_sr_detention_refit_campaign import exported, fake_contract

__all__ = ["exported"]


def old_numerics(monkeypatch):
    calls = []

    def verify(item, context, training, parameters, **kwargs):
        result = {
            "parameters": parameters,
            "training_nmse": parameters["a"] ** 2,
            "alternate_training_nmse": parameters["a"] ** 2,
            "solver_loss_discrepancy": 0.0,
            "solver_prediction_discrepancy": 0.0,
            "usable_for_retention": True,
        }
        kwargs["checkpoint"](result)
        return result

    def fit(item, context, training, parameters, **kwargs):
        index = len(calls)
        calls.append(index)
        blocked = index in (1, 6, 7, 9)
        value = parameters["a"] if blocked else parameters["a"] - 0.1
        result = {
            "calls": 0 if blocked else 1,
            "stop_reason": "derivatives_unqualified" if blocked else "budget_limited",
            "best": {"parameters": {"a": value}, "training_nmse": value**2},
        }
        kwargs["checkpoint"](result)
        return result

    monkeypatch.setattr(c.source.kernel, "verify", verify)
    monkeypatch.setattr(c.source.kernel, "fit", fit)
    monkeypatch.setattr(
        c.source.kernel, "diagnose", lambda *args, **kw: {"status": "unavailable"}
    )


@pytest.fixture
def prepared(exported, monkeypatch, tmp_path):
    _, data = exported
    fake_contract(monkeypatch)
    monkeypatch.setattr(
        c.source.kernel,
        "domain_policy",
        lambda *args: {"names": ["a"], "bounds": "fixed"},
    )
    old_numerics(monkeypatch)
    oldroot = tmp_path / "old-refit"
    c.source.prepare(
        oldroot,
        data,
        c.source.Policy(
            optimization_seconds=1, verification_seconds=1, diagnostic_seconds=1
        ),
    )
    for index in range(10):
        c.source.run_task(oldroot, index)
    (oldroot / "report").mkdir()
    public._write(oldroot / "report/summary.json", c.source.report(oldroot))
    destination = tmp_path / "delivery/recovery-inputs.json"
    c.export_inputs(oldroot, destination)
    inputs = read_seal(destination)
    root = tmp_path / "recovery"
    c.prepare(
        root,
        inputs,
        c.Policy(
            direction_seconds=1,
            optimization_seconds=1,
            verification_seconds=1,
            aggregation_seconds=1,
        ),
    )
    return root, inputs, oldroot


def fake_numerics(monkeypatch, *, tight_only=False, interrupt=None, verified=True):
    calls = []

    def verify(item, context, training, parameters, **kwargs):
        calls.append(("verify", parameters.copy()))
        result = {
            "parameters": parameters,
            "training_nmse": parameters["a"] ** 2,
            "alternate_training_nmse": parameters["a"] ** 2,
            "solver_loss_discrepancy": 0.0,
            "solver_prediction_discrepancy": 0.0,
            "usable_for_retention": verified,
        }
        kwargs["checkpoint"](result)
        return result

    def direction(item, context, training, parameters, index, **kwargs):
        key = (kwargs["purpose"], kwargs["precision"], index)
        calls.append(key)
        result = {
            "status": "complete",
            "identity": {"parameters_sha256": public.content_sha256(parameters)},
            "parameters": parameters,
            "index": index,
            "purpose": kwargs["purpose"],
            "precision": kwargs["precision"],
        }
        kwargs["checkpoint"]({"status": "incomplete"})
        if key == interrupt:
            raise KeyboardInterrupt()
        return result

    def qualification(item, context, training, parameters, rows, **kwargs):
        assert all(row["parameters"] == parameters for row in rows)
        assert all(row["precision"] == kwargs["precision"] for row in rows)
        result = {
            "parameters": parameters,
            "directions": rows,
            "precision": kwargs["precision"],
            "complete": len(rows) == 1,
            "search_qualified": len(rows) == 1
            and (not tight_only or kwargs["precision"] == "tight"),
        }
        result["qualification_sha256"] = public.content_sha256(result)
        return result

    def fit(item, context, training, parameters, **kwargs):
        calls.append(("fit", parameters.copy()))
        assert kwargs["qualification"]["search_qualified"]
        result = {
            "best": {"parameters": {"a": 0.01}, "training_nmse": 0.0001},
            "calls": 1,
            "stop_reason": "budget_limited",
        }
        kwargs["checkpoint"](result)
        return result

    def diagnose(item, context, training, parameters, **kwargs):
        assert kwargs["verification"]["parameters"] == parameters
        return {
            "status": "complete" if len(kwargs["directions"]) == 1 else "unavailable",
            "parameters": parameters,
        }

    def validate_qualification(item, context, training, parameters, rows, **kwargs):
        expected = qualification(
            item, context, training, parameters, rows, precision=kwargs["precision"]
        )
        if kwargs["qualification"] != expected:
            raise ValueError("qualification differs from coordinate evidence")

    def validate_diagnosis(item, context, training, parameters, **kwargs):
        expected = diagnose(
            item,
            context,
            training,
            parameters,
            directions=kwargs["directions"],
            verification=kwargs["verification"],
        )
        if kwargs["diagnosis"] != expected:
            raise ValueError("diagnosis differs from coordinate evidence")

    monkeypatch.setattr(c.kernel, "validate_qualification", validate_qualification)
    monkeypatch.setattr(c.kernel, "validate_diagnosis", validate_diagnosis)
    monkeypatch.setattr(c.original, "verify", verify)
    monkeypatch.setattr(c.kernel, "direction", direction)
    monkeypatch.setattr(c.kernel, "qualification", qualification)
    monkeypatch.setattr(c.kernel, "fit", fit)
    monkeypatch.setattr(c.kernel, "diagnose", diagnose)
    return calls


def test_frozen_roster_history_and_exact_starts(prepared, monkeypatch):
    root, inputs, oldroot = prepared
    fake_numerics(monkeypatch)
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    plan, _, _ = c.verify(root)
    assert [t["source_index"] for t in plan["tasks"]] == [4, 8, 1, 6, 7, 9]
    assert [t["mode"] for t in plan["tasks"]] == ["diagnosis_only"] * 2 + [
        "recover_start"
    ] * 4
    for task in plan["tasks"]:
        index = task["source_index"]
        expected = (
            inputs["parents"][index]["terminal"]["retained"]["parameters"]
            if task["mode"] == "diagnosis_only"
            else inputs["source_plan"]["tasks"][index]["parameters"]
        )
        assert task["parameters"] == expected
    result = c.report(root)
    assert result["recorded"] == 0 and len(result["historical_costs"]) == 10
    assert len(result["historical_outcomes"]) == 10
    assert before == {p: p.read_bytes() for p in before}


def test_endpoint_only_diagnoses_without_search_and_no_duplicate_work(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    result = c.run_task(root, 0)
    assert not any(call[0] in {"fit", "search"} for call in calls)
    assert result["diagnosis_complete"]
    before = deepcopy(calls)
    assert c.run_task(root, 0) == result
    assert calls == before
    assert c.report(root)["recorded"] == 1


def test_tighter_retry_has_own_reservations_and_same_original_start(
    prepared, monkeypatch
):
    root, inputs, _ = prepared
    calls = fake_numerics(monkeypatch, tight_only=True)
    result = c.run_task(root, 2)
    expected = inputs["source_plan"]["tasks"][1]["parameters"]
    assert ("fit", expected) in calls
    assert ("search", "standard", 0) in calls and ("search", "tight", 0) in calls
    assert result["retained"]["parameters"] == {"a": 0.01}
    assert result["retained_origin"] == "endpoint_verification"
    assert result["diagnosis"]["parameters"] == {"a": 0.01}
    assert (
        result["optimization"]["qualification_sha256"]
        == result["qualification"]["tight"]["qualification_sha256"]
    )
    assert result["historical_allowances_renewed"] is False


def test_interrupted_coordinate_not_repeated_or_silently_qualified(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt=("search", "standard", 0))
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 2)
    result = c.run_task(root, 2)
    assert calls.count(("search", "standard", 0)) == 1
    assert calls.count(("search", "tight", 0)) == 1
    assert result["operations"]["search-standard-00"]["status"] == "interrupted"
    assert not result["qualification"]["standard"]["search_qualified"]
    assert result["qualification"]["tight"]["search_qualified"]


def test_failed_verification_never_certifies_diagnosis(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, verified=False)
    result = c.run_task(root, 2)
    assert result["retained"] is None
    assert not result["diagnosis_complete"]
    assert not any(call[0] == "diagnose" for call in calls)


def test_history_tampering_rejected_even_with_rehashed_outer_packet(prepared):
    _, data, _ = prepared
    changed = deepcopy(data)
    changed["parents"][1]["operations"]["optimization"]["value"]["calls"] = 1
    with pytest.raises(ValueError, match="historical terminal"):
        c.Inputs.model_validate(changed)
    changed = deepcopy(data)
    changed["source_report"]["recorded"] = 9
    changed["source_report_sha256"] = public.content_sha256(changed["source_report"])
    with pytest.raises(ValueError, match="historical report"):
        c.Inputs.model_validate(changed)


def test_rounding_tolerance_only_for_generic_parameters():
    original = [{"index": 0, "start_kind": "generic", "parameters": {"a": 1.0}}]
    varied = deepcopy(original)
    varied[0]["parameters"]["a"] = nextafter(1.0, 2.0)
    assert len(c._saved_starts(original, varied)) == 1
    varied[0]["parameters"]["a"] = 1.000001
    with pytest.raises(ValueError, match="starting parameter"):
        c._saved_starts(original, varied)
    varied = deepcopy(original)
    varied[0]["index"] = 1
    with pytest.raises(ValueError, match="metadata"):
        c._saved_starts(original, varied)
    original[0]["start_kind"] = "perturbed"
    varied = deepcopy(original)
    varied[0]["parameters"]["a"] = nextafter(1.0, 2.0)
    with pytest.raises(ValueError, match="starting parameter"):
        c._saved_starts(original, varied)


def test_aggregate_diagnosis_must_match_coordinate_receipts(prepared, monkeypatch):
    root, _, _ = prepared
    fake_numerics(monkeypatch)
    row = c.run_task(root, 0)
    plan, inputs, policy = c.verify(root)
    changed = deepcopy(row["operations"])
    changed["retained_diagnosis"]["value"]["parameters"] = {"a": 99.0}
    with pytest.raises(ValueError, match="coordinate evidence"):
        c._assemble(
            row["identity"],
            changed,
            policy,
            c._problem(inputs, plan["tasks"][0]),
            terminal=True,
        )


def test_unfinished_stage_reservation_charges_budget_without_reexecution(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    plan, _, policy = c.verify(root)
    identity = {"plan_sha256": public.content_sha256(plan), "task": plan["tasks"][0]}
    from autoformalism.benchmarks.audited_release import seal

    seal(
        root / "tasks/0/operations/diagnosis-tight-00/started.json",
        {
            "campaign": identity,
            "stage": "diagnosis-tight-00",
            "allowance_seconds": policy.direction_seconds,
        },
    )
    row = c.run_task(root, 0)
    assert ("diagnosis", "tight", 0) not in calls
    assert not row["diagnosis_complete"]
    assert (
        row["operations"]["diagnosis-tight-00"]["budget_charge_seconds"]
        == policy.direction_seconds
    )
    assert row["operations"]["diagnosis-tight-00"]["accounting_complete"] is False
