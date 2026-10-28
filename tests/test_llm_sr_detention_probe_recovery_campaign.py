"""Compact provenance, exact evidence reuse, and bounded incumbent recovery."""

from copy import deepcopy

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.research import llm_sr_detention_probe_recovery_campaign as c
from tests.test_llm_sr_detention_refit import example


def verified(parameters, *, usable=True):
    value = parameters["a"] ** 2
    return {
        "parameters": parameters,
        "training_nmse": value,
        "alternate_training_nmse": value,
        "solver_loss_discrepancy": 0.0,
        "solver_prediction_discrepancy": 0.0,
        "usable_for_retention": usable,
    }


def fake_gate(inputs, task, parameters, records):
    names = task["coordinate_names"]
    indices = [r["index"] for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate coordinate")
    assert all(r["parameters"] == parameters for r in records)
    valid = [r for r in records if r["status"] == "complete"]
    rows = [
        {
            "index": r["index"],
            "name": names[r["index"]],
            "search_qualified": r["qualified"],
        }
        for r in valid
    ]
    missing = [i for i in range(len(names)) if i not in [r["index"] for r in valid]]
    failed = [r["index"] for r in valid if not r["qualified"]]
    result = {
        "directions": records,
        "parameters": parameters,
        "coordinate_names": names,
        "rows": rows,
        "missing_indices": missing,
        "failed_indices": failed,
        "complete": not missing,
        "search_qualified": not missing and not failed,
    }
    return result | {"qualification_sha256": public.content_sha256(result)}


def direction(parameters, index, *, qualified=True, status="complete"):
    return {
        "index": index,
        "parameters": parameters,
        "status": status,
        "qualified": qualified,
        "budget_exhausted": status != "complete",
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    item, context, training = example(15)
    monkeypatch.setattr(c, "_gate", fake_gate)
    monkeypatch.setattr(c.public, "_source_identity", lambda: "a" * 64)
    monkeypatch.setattr(c.public, "_runtime", lambda: {"testing": True})
    cases = []
    for i, source_indices in enumerate(([7], [23], [18, 19], [17], [1], [])):
        point = {"a": 2.0}
        seeds = [
            direction(
                point,
                j,
                qualified=not (i == 1 and j == 1),
                status="unavailable" if i > 1 and j == 1 else "complete",
            )
            for j in range(2)
        ]
        cases.append(
            {
                "index": i,
                "kind": "continuation" if i < 2 else "diagnostic",
                "source_indices": source_indices,
                "mode": "original",
                "parameters": point,
                "parent_verified_checkpoint": verified(point),
                "coordinate_names": ["a", "b"],
                "seed_directions": seeds,
                "historical_costs": [{"wall_seconds": 100}],
                "source_receipts": [{"row_sha256": "b" * 64}],
                "numerical_policy": c.kernel.Policy().model_dump(mode="json"),
            }
        )
    raw = {
        "protocol": "llm-sr-detention-probe-recovery-input-1",
        "model": item.model_dump(mode="json"),
        "context": context.model_dump(mode="json"),
        "training": training.model_dump(mode="json"),
        "cases": cases,
        "provenance": {"source_report_sha256": c.EXPECTED_SOURCE_REPORT_SHA256},
        "historical_costs": {"source": 1000},
        "validation_data_opened": False,
        "test_data_opened": False,
        "llm_calls": 0,
    }
    monkeypatch.setattr(c, "EXPECTED_INPUTS_SHA256", public.content_sha256(raw))
    root = tmp_path / "recovery"
    c.prepare(root, raw)
    return root, raw


def mock_numerics(
    monkeypatch,
    *,
    improve=True,
    reject=False,
    fail_indices=(),
    unavailable_indices=(),
    interrupt=False,
):
    calls = []

    def probe(item, context, training, parameters, index, **kwargs):
        calls.append(
            (
                "direction",
                index,
                deepcopy(parameters),
                kwargs["cache_root"],
                len(kwargs["seed_directions"]),
                kwargs["seconds"],
            )
        )
        result = direction(
            parameters,
            index,
            qualified=index not in fail_indices,
            status="unavailable" if index in unavailable_indices else "complete",
        )
        kwargs["checkpoint"](result)
        return result

    def fit(item, context, training, parameters, **kwargs):
        calls.append(("fit", deepcopy(parameters), kwargs["active_names"]))
        point = {"a": parameters["a"] / 2 if improve else parameters["a"]}
        result = {"best": {"parameters": point, "training_nmse": point["a"] ** 2}}
        kwargs["checkpoint"](result)
        if interrupt:
            raise KeyboardInterrupt()
        return result

    def verify(item, context, training, parameters, **kwargs):
        calls.append(("verify", deepcopy(parameters)))
        result = verified(parameters, usable=not reject)
        kwargs["checkpoint"](result)
        return result

    monkeypatch.setattr(c.recovery, "direction", probe)
    monkeypatch.setattr(c.controls, "fit_control", fit)
    monkeypatch.setattr(c.original, "verify", verify)
    return calls


def test_compact_roster_passed_reuse_and_three_bounded_searches(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert row["status"] == "complete"
    assert row["retained"]["training_nmse"] == 0.0625
    assert row["stop_reason"] == "search_allowance_exhausted"
    assert sum(x[0] == "fit" for x in calls) == 3
    # Initial audit is fully reused; only three new endpoint audits are executed.
    probes = [x for x in calls if x[0] == "direction"]
    assert len(probes) == 6
    assert all(x[-1] == 360 for x in probes)
    assert probes[0][3] == probes[1][3] != probes[2][3]
    assert row["cost"]["callback_wall_seconds"] is None
    assert row["parameter_confidence_claimed"] is False
    before = len(calls)
    assert c.run_task(root, 0) == row
    assert len(calls) == before
    assert c.report(root)["recorded"] == 1


def test_failed_row_explicitly_rechecked_then_held(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch, fail_indices=(1,), improve=False)
    row = c.run_task(root, 1)
    assert row["reused_direction_indices"] == [0]
    assert row["explicitly_reevaluated_direction_indices"] == [1]
    assert row["initial_qualification"]["complete"]
    assert row["initial_qualification"]["failed_indices"] == [1]
    assert next(x for x in calls if x[0] == "fit")[-1] == ["a"]
    assert row["retained"]["training_nmse"] == 4
    assert row["diagnostic_status"] == "failed"
    assert row["optimization_status"] == "searched"


def test_incomplete_audit_never_enables_active_subset(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch, unavailable_indices=(1,))
    row = c.run_task(root, 1)
    assert row["status"] == "complete"
    assert row["optimization_status"] == "blocked_before_search"
    assert row["diagnostic_status"] == "timed_out"
    assert not any(x[0] == "fit" for x in calls)
    assert row["retained"]["training_nmse"] == 4


def test_duplicate_endpoint_diagnosed_once_without_fit(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    row = c.run_task(root, 2)
    assert row["identity"]["task"]["source_indices"] == [18, 19]
    assert row["optimization_status"] == "not_requested"
    assert [(x[0], x[1]) for x in calls] == [("direction", 1)]
    assert calls[0][-2] == 2  # all authenticated seeds passed to raw-sample cache
    c.run_task(root, 2)
    assert len(calls) == 1


def test_unusable_verification_keeps_old_incumbent(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch, reject=True)
    row = c.run_task(root, 0)
    assert row["retained"]["training_nmse"] == 4
    assert row["stop_reason"] == "no_verified_progress_same_point"
    assert sum(x[0] == "fit" for x in calls) == 1
    assert not any(x[0] == "direction" for x in calls)


def test_interrupt_retains_checkpoint_without_renewing_search(prepared, monkeypatch):
    root, _ = prepared
    calls = mock_numerics(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    assert sum(x[0] == "fit" for x in calls) == 1
    receipt = read_seal(root / "tasks/0/operations/phase-0-search/result.json")
    assert receipt["status"] == "interrupted"
    # Resume verifies the saved checkpoint; later phases fail to improve.
    subsequent = mock_numerics(monkeypatch, improve=False)
    row = c.run_task(root, 0)
    assert row["retained"]["training_nmse"] == 1
    assert sum(x[0] == "fit" for x in subsequent) == 1
    assert read_seal(root / "tasks/0/operations/phase-0-search/result.json") == receipt
    assert (
        row["operations"]["phase-0-search"]["budget_charge_seconds"]
        == receipt["budget_charge_seconds"]
    )
    assert row["stop_reason"] == "no_verified_progress_same_point"


def test_reservation_without_receipt_consumed_once(prepared, monkeypatch):
    root, _ = prepared
    plan, _, policy = c.verify(root)
    task = plan["tasks"][2]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "tasks/2/operations/initial-direction-1"
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": "initial-direction-1",
            "allowance_seconds": policy.direction_seconds,
        },
    )
    calls = mock_numerics(monkeypatch)
    row = c.run_task(root, 2)
    assert not calls
    assert row["operations"]["initial-direction-1"]["status"] == "interrupted"
    assert row["cost"]["budget_charge_seconds"] == 360
    assert row["cost"]["wall_seconds"] is None
    assert row["status"] == "complete"
    assert row["final_qualification"]["missing_indices"] == [1]


@pytest.mark.parametrize("field", ["model", "training", "policy", "source", "seed"])
def test_payload_pins_cover_all_portable_fields(prepared, field):
    _, raw = prepared
    changed = deepcopy(raw)
    if field == "model":
        changed["model"]["parameters"]["C"] = 0.5
    elif field == "training":
        changed["training"]["rows"][0]["targets"]["h_down"][0] += 0.5
    elif field == "policy":
        changed["cases"][0]["numerical_policy"]["numerical"][
            "tight_relative_tolerance"
        ] *= 2
    elif field == "source":
        changed["provenance"]["source_report_sha256"] = "f" * 64
    else:
        changed["cases"][0]["seed_directions"][0]["qualified"] = False
    with pytest.raises(ValueError, match="source extraction differs"):
        c.Inputs.model_validate(changed)


def test_receipt_mutation_and_extra_operations_rejected(prepared, monkeypatch):
    root, _ = prepared
    mock_numerics(monkeypatch)
    c.run_task(root, 2)
    (root / "tasks/2/operations/rogue").mkdir()
    with pytest.raises(ValueError, match="unknown probe-recovery operation"):
        c.report(root)


def test_unknown_timing_not_fabricated_as_zero():
    result = c._cost(
        {"missing": {"budget_charge_seconds": 360, "accounting_complete": False}}
    )
    assert result["wall_seconds"] is None
    assert result["cpu_seconds"] is None
    assert not result["accounting_complete"]


def test_runtime_pin_rejected_before_any_allowance(prepared, monkeypatch):
    root, _ = prepared
    monkeypatch.setattr(c.public, "_source_identity", lambda: "d" * 64)
    with pytest.raises(ValueError, match="source/runtime differs"):
        c.run_task(root, 0)
    assert not (root / "tasks").exists()


def test_complete_but_imprecise_check_is_not_reported_passed(prepared, monkeypatch):
    root, _ = prepared
    plan, inputs, policy = c.verify(root)
    task = plan["tasks"][2]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    gate = fake_gate(
        inputs,
        task,
        task["parameters"],
        [direction(task["parameters"], 0), direction(task["parameters"], 1)],
    )
    gate["search_qualified"] = False
    gate["rows"][1]["search_qualified"] = False
    gate["failed_indices"] = []
    monkeypatch.setattr(
        c,
        "_state",
        lambda *args: {
            "stop_reason": "diagnostic_allowance_completed",
            "final_qualification": gate,
            "retained": task["parent_verified_checkpoint"],
            "phases": [],
        },
    )
    row = c._assemble(identity, {}, inputs, policy, terminal=True)
    assert row["diagnostic_status"] == "search_precision_unqualified"
    assert not row["parameter_confidence_claimed"]


def test_search_reservation_does_not_prove_optimization_executed(prepared, monkeypatch):
    root, _ = prepared
    plan, _, policy = c.verify(root)
    identity = {"plan_sha256": public.content_sha256(plan), "task": plan["tasks"][0]}
    folder = root / "tasks/0/operations/phase-0-search"
    seal(
        folder / "started.json",
        {
            "campaign": identity,
            "stage": "phase-0-search",
            "allowance_seconds": policy.segment_seconds,
        },
    )
    calls = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert not calls
    assert row["search_reserved"]
    assert row["optimization_performed"] is None
    assert row["optimization_status"] == "search_reserved_execution_unconfirmed"
    assert row["stop_reason"] == "search_checkpoint_unavailable"


def test_recheck_interruption_keeps_improvement_without_repeating_search(
    prepared, monkeypatch
):
    root, _ = prepared
    calls = mock_numerics(monkeypatch)
    original = c.recovery.direction

    def interrupted(*args, **kwargs):
        if args[4] == 1:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(c.recovery, "direction", interrupted)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    assert c.report(root)["tasks"][0]["retained"]["training_nmse"] == 1
    later = mock_numerics(monkeypatch)
    row = c.run_task(root, 0)
    assert not later
    assert sum(x[0] == "fit" for x in calls) == 1
    assert row["retained"]["training_nmse"] == 1
    assert row["stop_reason"] == "full_coordinate_recheck_incomplete"
    assert row["final_qualification"]["missing_indices"] == [1]
