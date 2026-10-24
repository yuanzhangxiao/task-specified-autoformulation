"""Frozen timescale slices preserve signed domains, provenance and operation caps."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import seal
from autoformalism.research import construction_timescale_controls as c
from tests.test_construction_fit_reliability import endpoint


def source_data():
    """Small train-only fake used to isolate journal semantics from ODE runtime."""
    names = list(c.OUTER_ROLES.values()) + [f"inner_{i}" for i in range(21)]
    values = dict.fromkeys(names, 1.0) | {c.RATE: -2363901.0}
    point = endpoint(values, 1.0, True)
    point["effective_parameters"] = values
    point["reduced_vector"] = list(values.values())
    model = {"task_id": "frozen-model-six", "problem": {}}
    return {
        "endpoint": point,
        "profile_outer_names": names[:6],
        "profile_inner_names": names[6:],
        "historical_inputs": {
            "models": [model] * 8,
            "historical_inputs": {
                "training_inputs": {
                    "models": [model] * 8,
                    "training": {"name": "train", "rows": [{"time": [0, 1, 300]}]},
                    "numerical_policy": {},
                }
            },
        },
        "parent_tasks": [],
    }


def test_roster_is_frozen_from_times_only_with_latest_anchor():
    data = source_data()
    tasks = c._tasks(data, c.Policy())
    assert len(tasks) == 10
    assert tasks[0]["arm"] == "saved_continuation"
    assert tasks[0]["start_overrides"] == {}
    rates = [t["frozen_kappa"] for t in tasks[1:7]]
    np.testing.assert_allclose(rates, [1 / 300, 1 / np.sqrt(300), 1, 10, 1000, 2363901])
    assert all(t["free_outer_coordinates"] == 5 for t in tasks[1:7])
    assert all(t["inner_coordinates"] == 21 for t in tasks)
    starts = tasks[7:]
    assert len({tuple(t["start_overrides"].values()) for t in starts}) == 3
    assert all(set(t["start_overrides"]) == set(c.OUTER_ROLES.values()) for t in starts)
    assert all(t["frozen_kappa"] is None for t in starts)
    changed = deepcopy(data)
    changed["historical_inputs"]["historical_inputs"]["training_inputs"]["training"][
        "rows"
    ][0]["targets"] = {"secret_truth": [-1e9]}
    assert c._tasks(changed, c.Policy()) == tasks
    assert c._contract()["signed_family_bounds_preserved"]
    assert c._contract()["global_lower_bound"] is False
    assert sum(c._allowances(c.Policy()).values()) == 2670
    with pytest.raises(ValueError):
        c.Policy(search_seconds=601)


@pytest.mark.parametrize("times", [[], [1], [1, 1], [2, 1], [0, float("inf")]])
def test_invalid_time_grid_is_rejected(times):
    with pytest.raises(ValueError, match="observation grids"):
        c.time_grid({"rows": [{"time": times}]})


def test_slice_removes_only_fixed_outer_direction_without_changing_domains():
    names = ("a", c.RATE, "b", "inner")
    lower, upper = np.full(4, -np.inf), np.full(4, np.inf)
    oracle = SimpleNamespace(names=names, lower=lower, upper=upper)
    calls = []

    def evaluate(point, deadline, *, jacobian):
        calls.append(point.copy())
        point[3] = 99.0
        return (
            point,
            np.array([2.0]),
            np.array([[10, 20, 30]]) if jacobian else None,
            {},
        )

    profile = SimpleNamespace(outer=(0, 1, 2), inner=(3,), audit={}, evaluate=evaluate)
    sliced = c.FrozenRateProfile(profile, oracle, 3.0)
    value = sliced.evaluate(np.array([1.0, -3.0, 2.0, 0.0]), 100.0)
    assert sliced.outer == (0, 2) and sliced.inner == (3,)
    np.testing.assert_array_equal(value[2], [[10, 30]])
    np.testing.assert_array_equal(value[0], [1, -3, 2, 99])
    assert lower[1] == -np.inf and upper[1] == np.inf
    assert sliced.audit["global_lower_bound"] is False
    with pytest.raises(ValueError, match="frozen rate changed"):
        sliced.evaluate(np.array([1.0, -4.0, 2.0, 0.0]), 100.0)
    assert len(calls) == 1


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    data = source_data()
    monkeypatch.setattr(c, "_validate", lambda data: None)
    monkeypatch.setattr(c.public, "_source_identity", lambda: "a" * 64)
    monkeypatch.setattr(c.pilot, "_runtime", lambda: {"runtime": "fixed"})
    root = tmp_path / "controls"
    c.prepare(
        root,
        data,
        c.Policy(
            search_seconds=10,
            diagnostic_seconds=10,
            verification_seconds=10,
            maximum_calls=20,
        ),
    )
    return root, data


def install_fake(monkeypatch, *, qualified=True, valid=True, interrupt=False):
    events = []
    monkeypatch.setattr(
        c, "PrefixFitProblem", SimpleNamespace(model_validate=lambda p: p)
    )
    monkeypatch.setattr(c, "PublicSplit", SimpleNamespace(model_validate=lambda p: p))
    monkeypatch.setattr(
        c.kernel, "NumericalPolicy", SimpleNamespace(model_validate=lambda p: p)
    )

    def make(problem, training, settings, anchor):
        names = tuple(c.OUTER_ROLES.values()) + tuple(f"inner_{i}" for i in range(21))
        oracle = SimpleNamespace(
            names=names,
            vector=lambda p: np.array([p[n] for n in names]),
            parameters=lambda v: dict(zip(names, v.tolist(), strict=True)),
            affine_audit={},
            recovery_audit={},
            units=np.ones(len(names)),
        )
        oracle.original = oracle
        return (
            oracle,
            oracle,
            SimpleNamespace(outer=tuple(range(6)), inner=tuple(range(6, 27)), audit={}),
        )

    monkeypatch.setattr(c.profiling, "make_oracle", make)

    def qualify(*args, coordinate_oracle, coordinate_parameters):
        assert coordinate_oracle.names == tuple(c.OUTER_ROLES.values()) + tuple(
            f"inner_{i}" for i in range(21)
        )
        assert coordinate_parameters == args[3]
        events.append("qualify")
        value = {"status": "complete", "passed": qualified, "precision_mode": "guarded"}
        args[-1](value)
        return value

    monkeypatch.setattr(c.pilot, "qualify_profile", qualify)

    def verify(o, a, parameters, seconds, save, **kwargs):
        events.append("verify")
        value = endpoint(parameters, parameters["inner_0"], valid)
        value["effective_parameters"] = parameters
        value["reduced_vector"] = a.vector(parameters).tolist()
        save(value)
        return value

    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)

    def search(oracle, profile, start, seconds, calls, save, *, preserve_full_history):
        events.append("search")
        assert preserve_full_history
        value = {
            "best": {"parameters": start | {"inner_0": 0.1}, "training_nmse": 0.1},
            "stop_reason": "budget",
            "search_calls_started": 1,
            "accepted_history_protocol": "full-coordinate-accepted-steps-1",
        }
        save(value)
        if interrupt:
            raise KeyboardInterrupt()
        return value

    monkeypatch.setattr(c.pilot, "_profile_search", search)

    def diagnose(o, a, parameters, verification, fit, seconds, save):
        events.append("diagnose")
        assert parameters == verification["parameters"]
        value = {
            "assessment": {"classification": "optimization_unresolved"},
            "full_coordinate_history_available": False,
            "search_history_matches_retained": False,
        }
        save(value)
        return value

    monkeypatch.setattr(c.pilot, "diagnose_profile", diagnose)
    return events


def test_own_endpoint_verification_diagnosis_and_no_renewal(prepared, monkeypatch):
    root, data = prepared
    events = install_fake(monkeypatch)
    value = c.run_task(root, 1)
    assert value["status"] == "complete"
    assert value["retained"]["training_nmse"] == 0.1
    assert value["retained"]["effective_parameters"][c.RATE] == -1 / 300
    assert data["endpoint"]["effective_parameters"][c.RATE] == -2363901
    assert value["conditional_rate_slice"]
    assert value["rate_profile_is_global_lower_bound"] is False
    assert value["constrained_fitting_adequacy"] == "not_assessed"
    diagnosis = value["operations"]["retained_diagnosis"]["value"]
    assert diagnosis["continuation_endpoint_matches_retained"] is True
    assert diagnosis["full_coordinate_history_available"] is False
    assert diagnosis["search_history_matches_retained"] is False
    before = events.copy()
    assert c.run_task(root, 1) == value
    assert events == before
    report = c.report(root)
    assert report["expected"] == 10 and report["recorded"] == 1
    assert report["rows"][1]["currently_verified"]


def test_failed_qualification_keeps_own_verified_start_without_search(
    prepared, monkeypatch
):
    root, _ = prepared
    events = install_fake(monkeypatch, qualified=False)
    value = c.run_task(root, 7)
    assert "search" not in events and "diagnose" in events
    assert value["retained"]["effective_parameters"][c.RATE] == -1 / 300
    assert value["retained_origin"] == "incumbent_verification"


def test_failed_verification_cannot_start_search(prepared, monkeypatch):
    root, _ = prepared
    events = install_fake(monkeypatch, valid=False)
    value = c.run_task(root, 0)
    assert "search" not in events and "diagnose" not in events
    assert value["retained"] is None


def test_interrupted_search_never_renews_allowance(prepared, monkeypatch):
    root, _ = prepared
    events = install_fake(monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 0)
    value = c.run_task(root, 0)
    assert events.count("search") == 1
    assert value["operations"]["continuation_search"]["status"] == "interrupted"
    assert value["old_allowances_renewed"] is False


def test_stale_retained_diagnosis_is_rejected(prepared, monkeypatch):
    root, data = prepared
    install_fake(monkeypatch)
    value = c.run_task(root, 1)
    operations = deepcopy(value["operations"])
    operations["retained_diagnosis"]["value"]["diagnosed_parameters_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="diagnosis differs"):
        c._assemble(value["identity"], data, operations)


def test_source_change_blocks_fit_but_preserves_readonly_reporting(
    prepared, monkeypatch
):
    root, _ = prepared
    monkeypatch.setattr(c.public, "_source_identity", lambda: "b" * 64)
    with pytest.raises(ValueError, match="source/runtime"):
        c.run_task(root, 0)
    assert c.report(root)["expected"] == 10


def test_tampered_terminal_and_unreserved_operation_fail(prepared, monkeypatch):
    root, _ = prepared
    install_fake(monkeypatch)
    value = c.run_task(root, 0)
    path = root / "results/task_00/result.json"
    path.unlink()
    seal(path, value | {"status": "invented"})
    with pytest.raises(ValueError, match="terminal differs"):
        c.report(root)
    (root / "results/task_01/operations/extra").mkdir(parents=True)
    with pytest.raises(ValueError, match="unknown timescale"):
        c.run_task(root, 1)


def test_schema_rejects_unknown_input_fields():
    with pytest.raises(ValueError):
        c.Inputs.model_validate({"protocol": c.INPUT_PROTOCOL, "test_truth": [1]})


def test_export_authenticates_all_eight_and_does_not_change_source(
    tmp_path, monkeypatch
):
    """Even the final precision-only receipt is mandatory before exporting model six."""
    from autoformalism.benchmarks.audited_release import read_seal

    data = source_data()
    old = data["historical_inputs"]
    tasks = c.pilot._tasks(old, c.pilot.Policy())
    plan = {"policy": c.pilot.Policy().model_dump(mode="json"), "tasks": tasks}
    source = tmp_path / "source"
    called = []
    point = data["endpoint"]
    profile = {
        "outer_names": data["profile_outer_names"],
        "inner_names": data["profile_inner_names"],
    }
    operations = {
        "continuation_search": {
            "status": "complete",
            "value": {"profile_audit": profile},
        },
        "continuation_verification": {"status": "complete", "value": point},
    }

    def assemble(identity, model, ops, *, terminal):
        called.append(identity["task"]["index"])
        return {
            "retained": point,
            "retained_origin": "continuation_verification",
            "cost": {},
        }

    for task in tasks:
        folder = c._folder(source, task)
        seal(
            folder / "result.json",
            {
                "retained": point,
                "retained_origin": "continuation_verification",
                "cost": {},
            },
        )
        seal(
            folder / "operations/continuation_verification/result.json",
            operations["continuation_verification"],
        )
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    monkeypatch.setattr(c.pilot, "verify", lambda *args, **kwargs: (plan, old))
    monkeypatch.setattr(c.pilot, "_operations", lambda *args: operations)
    monkeypatch.setattr(c.pilot, "_assemble", assemble)
    monkeypatch.setattr(c.pilot.history, "_receipts", lambda *args: {})
    monkeypatch.setattr(c, "_validate", lambda data: None)
    target = tmp_path / "export.json"
    summary = c.export_inputs(source, target)
    assert summary["authenticated_parent_tasks"] == 8
    assert called == list(range(8))
    saved = read_seal(target)
    assert len(saved["parent_tasks"]) == 8
    assert saved["historical_inputs"] == old
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "invalid.json")
    last = c._folder(source, tasks[-1]) / "result.json"
    last.unlink()
    seal(last, {"retained": None})
    with pytest.raises(ValueError, match="terminal differs"):
        c.export_inputs(source, tmp_path / "bad.json")
