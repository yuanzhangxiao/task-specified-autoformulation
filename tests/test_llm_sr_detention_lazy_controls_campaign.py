"""Shared evidence, matched controls, missing receipts, and nonrenewable budgets."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_lazy_controls_campaign as c
from tests.test_llm_sr_detention_active_blocks_campaign import (
    audit_prepared,
    continuation_prepared,
    exported,
    recovery_prepared,
)
from tests.test_llm_sr_detention_active_blocks_campaign import (
    fake_numerics as active_numerics,
)
from tests.test_llm_sr_detention_active_blocks_campaign import (
    prepared as active_prepared,
)

__all__ = [
    "active_prepared",
    "audit_prepared",
    "continuation_prepared",
    "exported",
    "recovery_prepared",
]


@pytest.fixture
def prepared(active_prepared, monkeypatch, tmp_path):
    oldroot, _, _ = active_prepared
    active_numerics(monkeypatch)
    for index in (0, 1, 3, 4):
        c.source.run_task(oldroot, index)
    active_numerics(monkeypatch)
    old_direction = c.kernel.direction
    count = 0

    def interrupt(*args, **kwargs):
        nonlocal count
        if args[4] == 5:
            count += 1
            if count == 3:
                kwargs["checkpoint"]({"incomplete": True})
                raise KeyboardInterrupt()
        return old_direction(*args, **kwargs)

    monkeypatch.setattr(c.kernel, "direction", interrupt)
    with pytest.raises(KeyboardInterrupt):
        c.source.run_task(oldroot, 2)
    # Simulate the scheduler killing the operation after its reservation/progress.
    folder = oldroot / "tasks/2/operations/phase-2-recheck-5"
    (folder / "result.json").unlink()
    (folder / "result.sha256").unlink(missing_ok=True)
    report = c.source.report(oldroot)
    (oldroot / "report").mkdir(exist_ok=True)
    public._write(oldroot / "report/summary.json", report)
    monkeypatch.setattr(
        c, "EXPECTED_SOURCE_REPORT_SHA256", public.content_sha256(report)
    )
    destination = tmp_path / "lazy-input.json"
    c.export_inputs(oldroot, destination)
    inputs = read_seal(destination)
    root = tmp_path / "lazy"
    c.prepare(
        root,
        inputs,
        c.Policy(
            optimization_seconds=3,
            direction_seconds=1,
            precision_seconds=2,
            verification_seconds=1,
        ),
    )
    return root, inputs, oldroot


def fake_numerics(
    monkeypatch,
    *,
    blocked_base=(),
    blocked_tight=(),
    blocked_endpoint=(),
    improve=True,
    interrupt_fit=False,
    unavailable_direction=False,
    reject=False,
):
    calls = []
    base_policy = None

    def gate(inputs, task, parameters, records, numerical):
        names = task.get("coordinate_domain", {}).get(
            "names", [f"x{i}" for i in range(8)]
        )
        assert all(r["parameters"] == parameters for r in records)
        assert all(r["mode"] == task["mode"] for r in records)
        indices = [r["index"] for r in records]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate coordinate")
        rows = [
            {
                "index": r["index"],
                "name": names[r["index"]],
                "search_qualified": r.get("qualified", True),
            }
            for r in records
        ]
        result = {
            "parameters": parameters,
            "directions": records,
            "coordinate_names": names,
            "complete": len(rows) == len(names),
            "rows": rows,
            "missing_indices": [i for i in range(len(names)) if i not in indices],
            "search_qualified": len(rows) == len(names)
            and all(r["search_qualified"] for r in rows),
        }
        return result | {"qualification_sha256": public.content_sha256(result)}

    def verify(item, context, training, parameters, **kwargs):
        calls.append(("verify", parameters.copy()))
        value = parameters["a"] ** 2
        result = {
            "parameters": parameters,
            "training_nmse": value,
            "alternate_training_nmse": value,
            "solver_loss_discrepancy": 0.0,
            "solver_prediction_discrepancy": 0.0,
            "usable_for_retention": not (reject and any(x[0] == "fit" for x in calls)),
        }
        kwargs["checkpoint"](result)
        return result

    def direction(item, context, training, parameters, index, **kwargs):
        nonlocal base_policy
        numerical = kwargs["policy"].numerical.tight_relative_tolerance
        if base_policy is None:
            base_policy = numerical
        tight = numerical < base_policy
        blocked = (
            blocked_endpoint
            if any(x[0] == "fit" for x in calls)
            else blocked_tight
            if tight
            else blocked_base
        )
        calls.append(("direction", parameters.copy(), index, tight, kwargs["seconds"]))
        if unavailable_direction:
            raise RuntimeError("no derivative")
        return {
            "parameters": parameters,
            "mode": kwargs["mode"],
            "index": index,
            "qualified": index not in blocked,
        }

    def fit(item, context, training, parameters, **kwargs):
        calls.append(("fit", parameters.copy(), deepcopy(kwargs)))
        value = parameters["a"] / 2 if improve else parameters["a"] + 10
        result = {
            "best": {"parameters": {"a": value}, "training_nmse": value * value},
            "stop_reason": "budget_limited",
            "search_calls_started": 1,
            "search_calls_completed": 1,
            "search_calls_failed": 0,
        }
        kwargs["checkpoint"](result)
        if interrupt_fit and len([x for x in calls if x[0] == "fit"]) == 1:
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(c, "_gate", gate)
    monkeypatch.setattr(c.original, "verify", verify)
    monkeypatch.setattr(c.kernel, "direction", direction)
    monkeypatch.setattr(c.controls, "fit_control", fit)
    return calls


def test_roster_matches_starts_allowances_and_preserves_partial_history(
    prepared, monkeypatch
):
    root, inputs, oldroot = prepared
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    fake_numerics(monkeypatch)
    plan, _, policy = c.verify(root)
    tasks = plan["tasks"]
    assert [t["kind"] for t in tasks] == ["prequalification"] * 6 + ["fit"] * 18 + [
        "diagnostic_supplement"
    ] * 3
    for start in range(3):
        arms = tasks[6 + 6 * start : 12 + 6 * start]
        assert [t["arm"] for t in arms] == [a[0] for a in c.ARMS]
        assert all(
            t["parameters"] == inputs["source_plan"]["tasks"][start]["parameters"]
            for t in arms
        )
        assert all(
            sum(
                seconds
                for name, seconds in c._allowances(t, policy).items()
                if name.endswith("-search")
            )
            == 3
            for t in arms
        )
    assert [t["coordinate_index"] for t in tasks[24:]] == [5, 6, 7]
    assert (
        inputs["incomplete_reservations"]["2/phase-2-recheck-5"]["allowance_seconds"]
        == 1
    )
    report = c.report(root)
    assert report["recorded"] == 0
    assert report["historical_generic_endpoint_supplement"]["missing_indices"] == [
        5,
        6,
        7,
    ]
    assert before == {p: p.read_bytes() for p in before}


def test_prequalification_escalation_shared_verbatim_by_eager_lazy(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, blocked_base=(2,))
    shared = c.run_task(root, 1)
    assert shared["selected_level"] == "escalated"
    assert len([x for x in calls if x[0] == "direction"]) == 16
    eager, lazy = c.run_task(root, 8), c.run_task(root, 9)
    fits = [x for x in calls if x[0] == "fit"]
    assert [x[2]["lazy_jacobian"] for x in fits] == [False, True]
    assert [x[2]["seconds"] for x in fits] == [3, 3]
    assert fits[0][1] == fits[1][1]
    assert fits[0][2]["qualification"] == fits[1][2]["qualification"]
    assert fits[0][2]["policy"] == fits[1][2]["policy"]
    assert eager["shared_qualification_sha256"] == lazy["shared_qualification_sha256"]
    assert shared["invocations"][0]["history_authentications"] == 1
    assert eager["invocations"][0]["startup_wall_seconds"] >= 0
    count = len(calls)
    assert c.run_task(root, 8) == eager
    assert len(calls) == count
    assert c.report(root)["tasks"][8] == eager


def test_segmented_search_rechecks_and_holds_only_adaptive(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, blocked_endpoint=(2,))
    c.run_task(root, 1)
    allrow = c.run_task(root, 10)
    assert len(allrow["phases"]) == 2
    assert allrow["stop_reason"] == "initial_or_endpoint_qualification_blocked"
    assert allrow["optimization_reserved_seconds"] == 1
    calls.clear()
    adaptive = c.run_task(root, 11)
    fits = [x for x in calls if x[0] == "fit"]
    assert [x[2]["seconds"] for x in fits] == [1, 1, 1]
    assert [len(x[2]["active_names"]) for x in fits] == [8, 7, 7]
    assert adaptive["optimization_reserved_seconds"] == 3
    assert adaptive["held_names"] == ["x2"]
    assert len([x for x in calls if x[0] == "direction"]) == 24
    assert not adaptive["full_model_convergence_claimed"]


def test_missing_dependency_never_spends_fit_allowance(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    with pytest.raises(ValueError, match="dependency is not terminal"):
        c.run_task(root, 8)
    assert calls == []
    assert not (root / "tasks/8/operations").exists()


def test_unresolved_initial_coordinates_block_all_but_allow_adaptive(
    prepared, monkeypatch
):
    root, _, _ = prepared
    fake_numerics(monkeypatch, blocked_base=(2,), blocked_tight=(2,))
    shared = c.run_task(root, 1)
    assert not shared["qualification"]["search_qualified"]
    row = c.run_task(root, 8)
    assert row["optimization_reserved_seconds"] == 0
    assert row["optimization_unused_allowance_seconds"] == 3
    assert not row["optimization_performed"]
    adaptive = c.run_task(root, 11)
    assert adaptive["phases"][0]["selection"]["held_names"] == ["x2"]


def test_interrupted_search_checkpoint_verifies_without_renewal(prepared, monkeypatch):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch, interrupt_fit=True)
    c.run_task(root, 1)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 8)
    result = c.run_task(root, 8)
    assert len([x for x in calls if x[0] == "fit"]) == 1
    assert result["operations"]["phase-0-search"]["status"] == "interrupted"
    assert result["operations"]["phase-0-search"]["budget_restarted"] is False
    assert (
        result["retained"]["training_nmse"]
        < result["phases"][0]["start"]["training_nmse"]
    )
    assert len(result["invocations"]) == 2


def test_diagnostic_only_fills_missing_coordinates_with_new_cost(prepared, monkeypatch):
    root, inputs, oldroot = prepared
    before = {p: p.read_bytes() for p in oldroot.rglob("*.json")}
    calls = fake_numerics(monkeypatch)
    for index in (24, 25, 26):
        row = c.run_task(root, index)
        assert not row["optimization_performed"]
        assert not row["parameters_changed"]
        assert row["new_allowance_seconds"] == 1
    assert [x[2] for x in calls] == [5, 6, 7]
    report = c.report(root)
    assert report["historical_generic_endpoint_supplement"]["complete"]
    assert (
        report["historical_costs"][2]["cost"]
        == inputs["source_report"]["tasks"][2]["cost"]
    )
    assert (
        report["tasks"][24]["historical_operation_receipt"]["status"]
        == "running_or_interrupted"
    )
    assert before == {p: p.read_bytes() for p in before}


def test_input_tampering_and_resealed_selection_are_rejected(prepared, monkeypatch):
    root, inputs, _ = prepared
    changed = deepcopy(inputs)
    changed["source_report"]["tasks"][2]["retained"]["training_nmse"] /= 2
    changed["source_report_sha256"] = public.content_sha256(changed["source_report"])
    with pytest.raises(ValueError, match="identity differs"):
        c.Inputs.model_validate(changed)
    fake_numerics(monkeypatch)
    c.run_task(root, 1)
    c.run_task(root, 8)
    path = root / "tasks/8/result.json"
    result = read_seal(path)
    result["phases"][0]["selection"]["active_names"] = []
    path.unlink()
    seal(path, result)
    with pytest.raises(ValueError, match="terminal differs"):
        c.run_task(root, 8)


def test_all_twenty_seven_mocked_tasks_complete_and_report_without_work(
    prepared, monkeypatch
):
    root, _, _ = prepared
    calls = fake_numerics(monkeypatch)
    for index in range(27):
        c.run_task(root, index)
    count = len(calls)
    result = c.report(root)
    assert result["status"] == "complete"
    assert result["recorded"] == 27
    assert len(result["controlled_comparisons"]) == 18
    assert all(row["same_original_start"] for row in result["controlled_comparisons"])
    assert all(
        row["same_initial_qualification"]
        for row in result["controlled_comparisons"]
        if "eager_vs_lazy" in row["contrast"]
    )
    assert len([x for x in calls if x[0] == "fit"]) == 30
    assert len(calls) == count
    assert not result["test_data_opened"]
    assert not result["validation_data_opened"]
    assert result["llm_calls"] == 0


def test_launcher_context_authenticates_once_and_includes_earlier_start(
    prepared, monkeypatch
):
    from dataclasses import replace

    root, _, _ = prepared
    fake_numerics(monkeypatch)
    old_verify = c.verify
    counts = []

    def verify(*args, **kwargs):
        counts.append(1)
        return old_verify(*args, **kwargs)

    monkeypatch.setattr(c, "verify", verify)
    context = c.load_run(root)
    context = replace(context, started_wall=context.started_wall - 2)
    result = c.run_task(root, 24, context=context)
    assert len(counts) == 1
    assert result["invocations"][0]["startup_wall_seconds"] >= 2
    assert result["invocations"][0]["started_at_utc"] == context.started_utc
    with pytest.raises(ValueError, match="invocation context differs"):
        c.run_task(root, 25, context=replace(context, root=root / "other"))
    with pytest.raises(ValueError, match="invocation context differs"):
        c.run_task(root, 25, context={"plan": context.plan})
