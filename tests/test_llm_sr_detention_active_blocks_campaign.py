"""Adaptive sets retain verified improvements and cannot renew consumed budgets."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_active_blocks_campaign as c
from tests.test_llm_sr_detention_chart_audit_campaign import (
    continuation_prepared,
    exported,
    fake_direction,
    recovery_prepared,
)
from tests.test_llm_sr_detention_chart_audit_campaign import prepared as audit_prepared

__all__ = ["audit_prepared", "continuation_prepared", "exported", "recovery_prepared"]


@pytest.fixture
def prepared(audit_prepared, monkeypatch, tmp_path):
    oldroot, _, _ = audit_prepared
    fake_direction(monkeypatch, blocked=(2,))
    for index in (*range(8), *range(16, 24)):
        c.audit.run_task(oldroot, index)
    fake_direction(monkeypatch)
    for index in range(8, 16):
        c.audit.run_task(oldroot, index)
    (oldroot / "report").mkdir()
    public._write(oldroot / "report/summary.json", c.audit.report(oldroot))
    path = tmp_path / "adaptive-input.json"
    c.export_inputs(oldroot, path)
    inputs = read_seal(path)

    def select(item, context, training, parameters, *, qualification, **kwargs):
        digest = qualification["qualification_sha256"]
        assert digest == public.content_sha256(
            {k: v for k, v in qualification.items() if k != "qualification_sha256"}
        )
        return tuple(r["name"] for r in qualification["rows"] if r["search_qualified"])

    monkeypatch.setattr(c.active, "select_active", select)
    root = tmp_path / "active"
    c.prepare(
        root,
        inputs,
        c.Policy(
            optimization_seconds=1,
            direction_seconds=1,
            precision_seconds=1,
            verification_seconds=1,
            aggregation_seconds=1,
        ),
    )
    return root, inputs, oldroot


def fake_numerics(
    monkeypatch,
    *,
    improve=True,
    reject=False,
    interrupt=False,
    noactive=False,
    fail_recheck=False,
):
    calls = []
    point_counts = {}

    def verify(item, context, training, parameters, **kwargs):
        calls.append(("verify", parameters.copy()))
        value = parameters["a"] ** 2
        result = {
            "parameters": parameters,
            "training_nmse": value,
            "alternate_training_nmse": value,
            "solver_loss_discrepancy": 0.0,
            "solver_prediction_discrepancy": 0.0,
            "usable_for_retention": not (
                reject and len([x for x in calls if x[0] == "fit"]) > 0
            ),
        }
        kwargs["checkpoint"](result)
        return result

    def fit(item, context, training, parameters, **kwargs):
        calls.append(
            ("fit", parameters.copy(), kwargs["active_names"], kwargs["seconds"])
        )
        assert kwargs["active_names"]
        value = parameters["a"] / 2 if improve else parameters["a"] + 10
        result = {
            "best": {"parameters": {"a": value}, "training_nmse": value * value},
            "stop_reason": "budget_limited",
            "active_names": kwargs["active_names"],
            "held_names": [
                n
                for n in kwargs["qualification"]["coordinate_names"]
                if n not in kwargs["active_names"]
            ],
            "qualification_sha256": kwargs["qualification"]["qualification_sha256"],
        }
        kwargs["checkpoint"](result)
        if interrupt and len([x for x in calls if x[0] == "fit"]) == 1:
            raise KeyboardInterrupt()
        return result

    def direction(item, context, training, parameters, index, **kwargs):
        calls.append(
            (
                "direction",
                parameters.copy(),
                index,
                kwargs["policy"].model_dump(mode="json"),
            )
        )
        if fail_recheck and index == 2:
            raise RuntimeError("missing coordinate")
        key = parameters["a"]
        if key not in point_counts:
            point_counts[key] = len(point_counts)
        # Release x2 after the first improvement; hold x1 after the second.
        block = 1 if point_counts[key] == 1 else None
        return {
            "status": "complete",
            "mode": kwargs["mode"],
            "parameters": parameters,
            "index": index,
            "precision_status": "inconclusive" if index == block else "passed",
            "qualified": not noactive and index != block,
        }

    monkeypatch.setattr(c.original, "verify", verify)
    monkeypatch.setattr(c.active, "fit_active", fit)
    monkeypatch.setattr(c.kernel, "direction", direction)
    return calls


def test_five_tasks_preserve_all_history_and_do_not_fit_in_prepare(prepared):
    root, inputs, oldroot = prepared
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    plan, _, policy = c.verify(root)
    assert [t["kind"] for t in plan["tasks"]] == ["fit"] * 3 + ["precision_check"] * 2
    assert [t["source_index"] for t in plan["tasks"]] == [1, 3, 5, 1, 5]
    assert [
        sum(r["search_qualified"] for r in g["qualification"]["rows"])
        for g in inputs["audit_report"]["groups"]
    ] == [7, 8, 7]
    assert policy.phases == 3
    report = c.report(root)
    assert report["recorded"] == 0 and len(report["historical_audit_costs"]) == 24
    assert len(report["historical_continuation_costs"]) == 7
    assert len(report["historical_original_arm_endpoints"]) == 3
    assert before == {p: p.read_bytes() for p in before}


def test_verified_progress_rechecks_all_coordinates_releases_and_reholds(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    row = c.run_task(root, 0)
    fits = [x for x in calls if x[0] == "fit"]
    assert len(fits) == 3
    assert [len(x[2]) for x in fits] == [7, 8, 7]
    assert "x2" not in fits[0][2] and "x2" in fits[1][2]
    assert "x1" not in fits[2][2]
    assert sum(x[0] == "direction" for x in calls) == 24
    assert row["phases_completed"] == 3
    assert row["phases"][1]["released_names"] == ["x2"]
    assert row["phases"][2]["newly_held_names"] == ["x1"]
    assert row["retained"]["training_nmse"] < row["phases"][0]["start"]["training_nmse"]
    assert row["stop_reason"] == "phase_budget_exhausted"
    assert not row["full_model_convergence_claimed"]
    before = deepcopy(calls)
    assert c.run_task(root, 0) == row and calls == before
    assert c.report(root)["tasks"][0] == row


def test_no_improvement_reuses_evidence_and_leaves_budget_unused(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, improve=False)
    row = c.run_task(root, 1)
    assert sum(x[0] == "fit" for x in calls) == 1
    assert not any(x[0] == "direction" for x in calls)
    assert row["stop_reason"] == "no_verified_progress_same_point"
    assert row["optimization_unused_allowance_seconds"] == 2
    assert (
        row["phases"][0]["recheck_source"] == "identical_point_matching_evidence_reused"
    )


def test_rejected_candidate_does_not_replace_incumbent(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, reject=True)
    row = c.run_task(root, 0)
    assert row["retained"]["parameters"] == row["identity"]["task"]["parameters"]
    assert row["stop_reason"] == "no_verified_progress_same_point"
    assert sum(x[0] == "fit" for x in calls) == 1


def test_interrupted_search_keeps_best_verifies_it_and_never_retries(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    row = c.run_task(root, 0)
    assert row["operations"]["phase-0-search"]["status"] == "interrupted"
    assert sum(x[0] == "fit" for x in calls) == 3
    assert row["phases"][0]["verified_improvement"]
    assert row["cost"]["budget_charge_seconds"] > 0
    assert not row["historical_allowances_renewed"]


def test_incomplete_full_recheck_stops_search_and_preserves_improvement(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, fail_recheck=True)
    row = c.run_task(root, 0)
    assert row["stop_reason"] == "full_coordinate_recheck_incomplete"
    assert sum(x[0] == "fit" for x in calls) == 1
    assert row["active_names"] == []
    assert row["retained_origin"] == "phase-0-verification"
    assert not row["full_coordinate_search_qualified"]


def test_no_qualified_direction_stops_without_fabricated_convergence(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, noactive=True)
    row = c.run_task(root, 0)
    assert row["stop_reason"] == "no_qualified_active_coordinates"
    assert sum(x[0] == "fit" for x in calls) == 1
    assert not row["full_model_convergence_claimed"]


def test_precision_side_task_is_independent_and_never_fits(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    row = c.run_task(root, 3)
    assert len(calls) == 1 and calls[0][0] == "direction"
    assert calls[0][2] == 2
    assert not row["parameters_changed"] and not row["optimization_performed"]
    assert row["coordinate_result"]["index"] == 2
    assert not row["full_model_convergence_claimed"]


def test_source_receipt_tampering_rejected_even_when_outer_hash_updated(prepared):
    _, inputs, _ = prepared
    broken = deepcopy(inputs)
    broken["audit_report"]["tasks"][0]["cost"]["operations"] += 1
    broken["audit_report_sha256"] = public.content_sha256(broken["audit_report"])
    with pytest.raises(ValueError, match="terminal differs"):
        c.Inputs.model_validate(broken)


def test_selection_and_terminal_tampering_rejected(prepared, monkeypatch):
    root, _, _ = prepared
    fake_numerics(monkeypatch)
    c.run_task(root, 0)
    path = root / "tasks/0/result.json"
    broken = read_seal(path)
    broken["retained"]["training_nmse"] = 0
    path.unlink()
    seal(path, broken)
    with pytest.raises(ValueError, match="terminal differs"):
        c.report(root)


def test_runtime_mismatch_blocks_execution_but_not_report(prepared, monkeypatch):
    root, _, _ = prepared
    monkeypatch.setattr(c.public, "_runtime", lambda: {"different": True})
    assert c.report(root)["status"] == "incomplete"
    with pytest.raises(ValueError, match="source/runtime differs"):
        c.run_task(root, 0)
    for kwargs in (
        {"phases": 4},
        {"optimization_seconds": 601},
        {"precision_seconds": 241},
    ):
        with pytest.raises(ValueError):
            c.Policy(**kwargs)


def test_search_interruption_before_checkpoint_stops_without_new_phase(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    starts = []

    def interrupt(*args, **kwargs):
        starts.append(1)
        raise KeyboardInterrupt()

    monkeypatch.setattr(c.active, "fit_active", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    row = c.run_task(root, 0)
    assert starts == [1]
    assert row["stop_reason"] == "search_checkpoint_unavailable"
    assert row["optimization_reserved_seconds"] == 1
    assert row["optimization_unused_allowance_seconds"] == 2
    assert row["retained"]["parameters"] == row["identity"]["task"]["parameters"]
    assert not any(x[0] == "direction" for x in calls)
    assert c.report(root)["tasks"][0] == row


def test_orphaned_selection_cannot_start_an_unbound_search(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    plan, _, policy = c.verify(root)
    identity = {"plan_sha256": public.content_sha256(plan), "task": plan["tasks"][0]}
    seal(
        root / "tasks/0/operations/phase-0-selection/started.json",
        {
            "campaign": identity,
            "stage": "phase-0-selection",
            "allowance_seconds": policy.aggregation_seconds,
        },
    )
    row = c.run_task(root, 0)
    assert not any(x[0] == "fit" for x in calls)
    assert row["stop_reason"] == "selection_unavailable"
    assert row["operations"]["phase-0-selection"]["budget_charge_seconds"] == 1
    assert row["optimization_unused_allowance_seconds"] == 3
    assert c.report(root)["tasks"][0] == row
