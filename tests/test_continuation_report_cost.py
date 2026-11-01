"""Only authenticated derived-time addition roundoff can be reconciled."""

from copy import deepcopy
from functools import reduce

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.research import construction_fit_qualification as journal
from autoformalism.research import continuation_report_cost as costs


def operations():
    return {
        name: {
            "status": "complete",
            "accounting_complete": True,
            "budget_charge_seconds": value,
            "wall_seconds": value,
            "cpu_seconds": value,
            "value": {
                "verification_calls_started": 1,
                "verification_calls_completed": 1,
                "timing": {
                    "callback_wall_seconds": value,
                    "callback_cpu_seconds": value,
                    "numerical_active_wall_seconds": value,
                },
            },
        }
        for name, value in zip(
            ("setup", "baseline", "jacobian"), (0.1, 0.2, 0.3), strict=True
        )
    }


def terminal(ops, cost):
    return {
        "candidate_id": "candidate-03",
        "training_nmse": 0.23,
        "parameters": {"a": 0.1},
        "status": "complete",
        "cost": cost,
        "operation_receipts": {
            name: public.content_sha256(value) for name, value in ops.items()
        },
    }


def views(ops=None):
    ops = operations() if ops is None else ops
    saved = terminal(ops, journal._cost(ops))
    canonical = terminal(ops, costs.canonical_cost(ops))
    return ops, saved, canonical


def test_reproduced_roundoff_is_audited_and_originals_unchanged():
    ops, saved, canonical = views()
    old = deepcopy((ops, saved, canonical))
    assert saved["cost"]["callback_cpu_seconds_total"] == 0.6000000000000001
    assert canonical["cost"]["callback_cpu_seconds_total"] == 0.6
    audit = costs.reconcile(saved, canonical, ops)
    assert audit["status"] == "derived_time_roundoff_only"
    assert audit["scientific_fields_exact"]
    assert not audit["original_terminal_modified"]
    assert all(
        row["absolute_difference"] <= row["addition_roundoff_bound"]
        for row in audit["differences"]
    )
    assert (ops, saved, canonical) == old


def test_canonical_sum_is_independent_of_insertion_order():
    ops = operations()
    expected = costs.canonical_cost(ops)
    assert costs.canonical_cost(dict(reversed(list(ops.items())))) == expected
    assert costs.canonical_cost(dict(sorted(ops.items()))) == expected
    assert (
        costs.reconcile(terminal(ops, expected), terminal(ops, expected), ops)["status"]
        == "exact"
    )


def test_python311_left_fold_totals_are_supported():
    ops, saved, canonical = views()
    inputs = costs._time_inputs(ops)
    for name, values in inputs.items():
        if values is not None:
            saved["cost"][name] = reduce(
                lambda total, value: total + value, sorted(values), 0
            )
    assert (
        costs.reconcile(saved, canonical, ops)["status"] == "derived_time_roundoff_only"
    )


@pytest.mark.parametrize(
    "field", ["training_nmse", "parameters", "status", "operation_receipts"]
)
def test_scientific_or_identity_mismatch_is_never_tolerated(field):
    ops, saved, canonical = views()
    if field == "training_nmse":
        saved[field] += 1e-15
    elif field == "parameters":
        saved[field]["a"] += 1e-15
    elif field == "status":
        saved[field] = "partial"
    else:
        saved[field]["setup"] = "different"
    with pytest.raises(ValueError, match=r"differs|differ"):
        costs.reconcile(saved, canonical, ops)


@pytest.mark.parametrize(
    "field",
    ["verification_calls_started_total", "operations", "accounting_complete", "scope"],
)
def test_nontiming_cost_fields_are_exact(field):
    ops, saved, canonical = views()
    saved["cost"][field] = "changed"
    with pytest.raises(ValueError, match="outside derived timing"):
        costs.reconcile(saved, canonical, ops)


@pytest.mark.parametrize(
    "field",
    ["budget_charge_seconds", "wall_seconds_total", "callback_cpu_seconds_observed"],
)
def test_material_timing_change_is_rejected(field):
    ops, saved, canonical = views()
    saved["cost"][field] += 1e-9
    with pytest.raises(ValueError, match="beyond roundoff"):
        costs.reconcile(saved, canonical, ops)


def test_interruption_retains_unknown_totals_and_observed_lower_bounds():
    ops = operations()
    op = ops["baseline"]
    op.update(
        status="interrupted",
        accounting_complete=False,
        wall_seconds=None,
        cpu_seconds=None,
        value=None,
        observed_progress={
            "timing": {"elapsed_wall_seconds": 0.4, "elapsed_cpu_seconds": 0.3}
        },
    )
    ops, saved, canonical = views(ops)
    assert canonical["cost"]["wall_seconds_total"] is None
    assert canonical["cost"]["callback_cpu_seconds_total"] is None
    assert canonical["cost"]["wall_seconds_observed"] == 0.8
    costs.reconcile(saved, canonical, ops)
    saved["cost"]["wall_seconds_total"] = 0.8
    with pytest.raises(ValueError, match="unknown timing"):
        costs.reconcile(saved, canonical, ops)


@pytest.mark.parametrize("bad", [True, -0.1, float("nan"), float("inf"), "0.1"])
def test_malformed_operand_is_rejected(bad):
    ops = operations()
    ops["setup"]["value"]["timing"]["callback_cpu_seconds"] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        costs.canonical_cost(ops)


def test_recomputed_cost_and_changed_operands_cannot_silently_supply_new_bounds():
    ops, saved, canonical = views()
    corrupted = deepcopy(canonical)
    corrupted["cost"]["wall_seconds_observed"] += 1e-10
    with pytest.raises(ValueError, match="not canonical"):
        costs.reconcile(saved, corrupted, ops)
    ops["setup"]["wall_seconds"] += 1
    with pytest.raises(ValueError, match="receipts differ"):
        costs.reconcile(saved, canonical, ops)


def test_historical_verification_timing_precedence_is_preserved():
    ops = operations()
    ops["setup"]["value"]["verification_timing"] = {"callback_cpu_seconds": 0.5}
    ops["setup"]["value"]["timing"]["callback_cpu_seconds"] = 100
    assert costs.canonical_cost(ops)["callback_cpu_seconds_total"] == 1


def test_zero_totals_cannot_accept_nonzero_material_changes_or_booleans():
    ops = operations()
    for value in ops.values():
        value["value"]["timing"]["callback_cpu_seconds"] = 0
    ops, saved, canonical = views(ops)
    saved["cost"]["callback_cpu_seconds_total"] = 1e-300
    with pytest.raises(ValueError, match="beyond roundoff"):
        costs.reconcile(saved, canonical, ops)
    saved["cost"]["callback_cpu_seconds_total"] = False
    with pytest.raises(ValueError, match="finite and nonnegative"):
        costs.reconcile(saved, canonical, ops)
