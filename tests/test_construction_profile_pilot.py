"""Matched profiling budgets, immutable precision probes, and authenticated resume."""

from copy import deepcopy
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("mpmath")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_profile_pilot as c
from tests.test_construction_fit_reliability import endpoint
from tests.test_construction_precision_reduction import (
    fake_recovery as fake_parent,
)
from tests.test_construction_precision_reduction import (
    policy as parent_policy,
)
from tests.test_construction_precision_reduction import (
    prepared_history as derivative_history,
)


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        fixed_inner_seconds=2,
        diagnostic_seconds=10,
        verification_seconds=10,
        precision_seconds=10,
        maximum_calls=20,
    )


def fail(*args, **kwargs):
    pytest.fail("unexpected numerical operation")


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(c.public, "_source_identity", lambda: "d" * 64)


@pytest.fixture(scope="module")
def prepared_history(tmp_path_factory):
    previous = derivative_history.__wrapped__(tmp_path_factory)
    directory = tmp_path_factory.mktemp("profile-history")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c.public, "_source_identity", lambda: "d" * 64)
        path = directory / "precision-inputs.json"
        c.previous.export_inputs(previous, path)
        root = directory / "precision"
        c.previous.prepare(root, read_seal(path), parent_policy())
        fake_parent(patch, candidate=0.2)
        c.previous.run_task(root, 0)
    return root


@pytest.fixture
def source(prepared_history, tmp_path):
    return copytree(prepared_history, tmp_path / "source")


def inputs(source, tmp_path):
    path = tmp_path / "inputs.json"
    c.export_inputs(source, path)
    return read_seal(path)


def fake(
    monkeypatch,
    *,
    qualified=True,
    inner=0.1,
    candidate=0.05,
    candidate_valid=True,
    source_valid=True,
    interrupt=None,
):
    events = []

    def make(
        problem, training, numerical, anchor, *, precision_mode="guarded", **kwargs
    ):
        names = tuple(anchor)
        oracle = SimpleNamespace(
            names=names,
            vector=lambda p: np.array([p[n] for n in names]),
            parameters=lambda p: dict(zip(names, p.tolist(), strict=True)),
            affine_audit={"mode": precision_mode},
            recovery_audit={"anchor": anchor},
            lower=np.full(len(names), -np.inf),
            upper=np.full(len(names), np.inf),
            units=np.ones(len(names)),
        )
        oracle.original = oracle
        return oracle, oracle, SimpleNamespace(audit={})

    def qualify(*args):
        events.append(("qualify", args[-2]))
        value = {"passed": qualified, "precision_mode": "guarded", "status": "complete"}
        args[-1](value)
        return value

    def verify(original, oracle, p, seconds, save, *, tolerances):
        assert tolerances == ((1e-11, 1e-13),)
        events.append(("verify", p["a"], seconds))
        valid = source_valid if p["a"] == 0.2 else candidate_valid
        value = endpoint(p, p["a"], valid)
        value["reduced_vector"] = oracle.vector(p).tolist()
        save(value)
        return value

    def inner_solve(oracle, profile, start, seconds, save):
        events.append(("inner", seconds))
        value = {
            "best": {"parameters": start | {"a": inner}, "training_nmse": inner},
            "inner_evaluations_started": 1,
        }
        save(value)
        if interrupt == "inner":
            raise KeyboardInterrupt()
        return value

    def search(oracle, start, seconds, calls, save, **kwargs):
        events.append(("joint", seconds))
        value = {
            "best": {
                "parameters": start | {"a": candidate},
                "training_nmse": candidate,
            },
            "calls": 1,
            "search_calls_started": 1,
            "stop_reason": "step_or_gradient_converged",
            "message": "`xtol` termination condition is satisfied.",
        }
        save(value)
        return value

    def profiled(oracle, profile, start, seconds, calls, save):
        events.append(("profile", seconds))
        return search(oracle, start, seconds, calls, save)

    def diagnose(original, oracle, p, verification, context, seconds, save, **kwargs):
        assert p == verification["parameters"]
        events.append(("diagnose", p["a"], context["stop_reason"]))
        value = {"assessment": {"classification": "optimization_unresolved"}}
        save(value)
        return value

    def compare(oracle, p, seconds, save, *, indices, steps):
        events.append(("precision", deepcopy(p), indices))
        value = {
            "status": "complete",
            "fixed_parameters": deepcopy(p),
            "indices": list(indices),
        }
        save(value)
        return value

    monkeypatch.setattr(c.profiling, "make_oracle", make)
    monkeypatch.setattr(c.recovery, "make_oracle", lambda *a, **k: make(*a, **k)[:2])
    monkeypatch.setattr(c, "_qualification", qualify)
    monkeypatch.setattr(c, "_fixed_inner", inner_solve)
    monkeypatch.setattr(c, "_profile_search", profiled)
    monkeypatch.setattr(c.numerical, "search", search)
    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)
    monkeypatch.setattr(c.recovery, "diagnose", diagnose)
    monkeypatch.setattr(c.precision, "compare", compare)
    return events


def test_budget_matching_resource_ceiling_and_roles():
    p = c.Policy.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "configs/construction_profile_pilot_v1.json"
        ).read_text()
    )
    data = {"models": [{"task_id": str(i)} for i in range(8)]}
    tasks = c._tasks(data, p)
    assert len(tasks) == 8
    assert [(t["model"], t["arm"]) for t in tasks[:2]] == [
        (6, "joint"),
        (6, "profiled"),
    ]
    assert [t["model"] for t in tasks[2:]] == [0, 3, 4, 5, 6, 7]
    assert (
        p.fixed_inner_seconds == 120 and p.search_seconds - p.fixed_inner_seconds == 480
    )
    assert sum(c._allowances(p, tasks[0]).values()) == 2670
    assert sum(c._allowances(p, tasks[1]).values()) == 3000
    assert sum(c._allowances(p, tasks[2]).values()) == 690
    with pytest.raises(ValueError):
        c.Policy(search_seconds=100, fixed_inner_seconds=120)
    with pytest.raises(ValueError):
        c.Policy(precision_seconds=601)


def test_export_authenticates_every_parent_and_preserves_history(source, tmp_path):
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    data = inputs(source, tmp_path)
    assert len(data["models"]) == len(data["parent_tasks"]) == 1
    assert len(data["historical_inputs"]["local_recovery_tasks"]) == 2
    assert data["models"][0]["endpoint"]["parameters"]["a"] == 0.2
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "bad.json")
    terminal = source / "results/task_00/result.json"
    altered = read_seal(terminal)
    altered["retained"]["parameters"]["a"] = 0.123
    terminal.unlink()
    seal(terminal, altered)
    with pytest.raises(ValueError, match="terminal differs"):
        c.export_inputs(source, tmp_path / "bad.json")


@pytest.mark.parametrize(
    "change",
    ["heldout", "problem", "order", "vector", "index", "history", "unverified"],
)
def test_bad_inputs_fail_before_numerical_work(source, tmp_path, monkeypatch, change):
    data = inputs(source, tmp_path)
    if change == "heldout":
        data["validation"] = {"rows": []}
    elif change == "problem":
        data["models"][0]["problem_sha256"] = "0" * 64
    elif change == "order":
        data["models"][0]["model"] = 1
    elif change == "vector":
        data["models"][0]["fixed_vector"][0] += 1
    elif change == "index":
        data["models"][0]["probe_indices"] = [999]
    elif change == "history":
        data["historical_inputs"]["local_recovery_tasks"] = []
    else:
        data["models"][0]["endpoint"]["original_equations_verified"] = False
    monkeypatch.setattr(c.profiling, "make_oracle", fail)
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "run", data, policy())


def test_precision_task_never_optimizes_selects_or_renews(
    source, tmp_path, monkeypatch
):
    data = inputs(source, tmp_path)
    root = tmp_path / "run"
    c.prepare(root, data, policy())
    events = fake(monkeypatch)
    monkeypatch.setattr(c, "_qualification", fail)
    monkeypatch.setattr(c, "_fixed_inner", fail)
    monkeypatch.setattr(c.numerical, "search", fail)
    value = c.run_task(root, 2)
    assert not value["search_started"] and not value["selection_performed"]
    assert value["retained"] is None
    assert events == [
        (
            "precision",
            data["models"][0]["fixed_parameters"],
            tuple(data["models"][0]["probe_indices"]),
        )
    ]
    assert c.run_task(root, 2) == value and len(events) == 1
    summary = c.report(root)
    assert len(summary["historical_costs"]["local_recovery_by_task"]) == 2
    assert summary["rows"][2]["training_nmse"] is None


@pytest.mark.parametrize(
    "qualified, candidate, expected",
    [(True, 0.05, 0.05), (True, 0.3, 0.1), (False, 0.05, 0.1)],
)
def test_profile_keeps_verified_inner_even_if_outer_blocked_or_worse(
    source, tmp_path, monkeypatch, qualified, candidate, expected
):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake(monkeypatch, qualified=qualified, candidate=candidate)
    value = c.run_task(root, 1)
    assert value["retained"]["parameters"]["a"] == expected
    assert ("inner", 2) in events
    assert (("profile", 8) in events) is qualified
    assert events[-1][0:2] == ("diagnose", expected)
    assert not value["operations"]["retained_diagnosis"]["value"][
        "full_coordinate_history_available"
    ]
    assert c.run_task(root, 1) == value


def test_joint_uses_full_budget_and_diagnoses_retained(source, tmp_path, monkeypatch):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake(monkeypatch)
    value = c.run_task(root, 0)
    assert ("joint", 10) in events and not any(e[0] == "inner" for e in events)
    assert value["search_stop_reason"] == "xtol"
    assert value["operations"]["retained_diagnosis"]["value"][
        "full_coordinate_history_available"
    ]


def test_failed_new_verification_never_displaces_incumbent(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake(monkeypatch, candidate_valid=False)
    value = c.run_task(root, 1)
    assert value["retained"]["parameters"]["a"] == 0.2
    assert value["retained_origin"] == "incumbent_verification"


def test_failed_source_verification_blocks_all_search_and_inner(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake(monkeypatch, source_valid=False)
    value = c.run_task(root, 1)
    assert value["retained"] is None
    assert not any(e[0] in {"joint", "profile", "inner", "diagnose"} for e in events)


def test_interruption_does_not_repeat_fixed_inner(source, tmp_path, monkeypatch):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake(monkeypatch, interrupt="inner")
    with pytest.raises(KeyboardInterrupt):
        c.run_task(root, 1)
    events = fake(monkeypatch)
    value = c.run_task(root, 1)
    assert not any(e[0] == "inner" for e in events)
    assert value["operations"]["fixed_inner"]["status"] == "interrupted"
    assert value["cost"]["budget_charge_seconds"] > 0


def test_stale_diagnosis_rejected(source, tmp_path, monkeypatch):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake(monkeypatch)
    value = c.run_task(root, 0)
    ops = deepcopy(value["operations"])
    ops["retained_diagnosis"]["value"]["diagnosed_parameters_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact retained"):
        c._assemble(
            value["identity"], read_seal(root / "inputs.json")["models"][0], ops
        )


@pytest.mark.parametrize("change", ["runtime", "source"])
def test_runtime_source_change_blocks_execution_but_allows_offline_report(
    source, tmp_path, monkeypatch, change
):
    root = tmp_path / "run"
    c.prepare(root, inputs(source, tmp_path), policy())
    if change == "runtime":
        monkeypatch.setattr(c, "_runtime", lambda: {"changed": True})
    else:
        monkeypatch.setattr(c.public, "_source_identity", lambda: "e" * 64)
    with pytest.raises(ValueError, match="source/runtime"):
        c.run_task(root, 0)
    assert c.report(root)["recorded"] == 0


def test_tiny_numerical_inner_profile_and_original_verification_smoke():
    from tests.test_prefix_multioutput_profile import profile_case

    original, oracle, profile = c.profiling.make_oracle(*profile_case())
    start = dict(oracle.start)
    outer = oracle.names[profile.outer[0]]
    start[outer] *= 1.02
    snapshots = []
    inner = c._fixed_inner(oracle, profile, start, 10, snapshots.append)
    assert inner["status"] == "complete" and inner["best"]
    assert inner["projection_audit"]["status"] == "conditional_optimum"
    value = c._profile_search(
        oracle, profile, inner["best"]["parameters"], 15, 20, snapshots.append
    )
    assert value["profile_derivative_check"]["search_qualified"]
    assert value["search_calls_started"] > 0 and value["best"]
    verified = c.numerical.verify_endpoint_ladder(
        original,
        oracle,
        value["best"]["parameters"],
        10,
        snapshots.append,
        tolerances=((1e-11, 1e-13),),
    )
    assert verified["original_equations_verified"]
    assert verified["training_nmse"] < inner["best"]["training_nmse"]
    assert (
        c.previous._verified_coordinates(original, oracle, verified)
        == value["best"]["parameters"]
    )


def test_saved_probe_steps_and_failed_columns_are_frozen_in_order():
    check = {
        "failed_indices": [3],
        "inconclusive_indices": [1],
        "directions": [
            {"index": 0, "projected_gradient_error_bound": 10},
            {"index": 2, "projected_gradient_error_bound": 20},
        ],
    }
    assert c._probe_indices(check, 7) == [1, 3, 2, 0, 4, 5]


@pytest.mark.parametrize("pair", [[0, 0], [1, 0.7], [float("inf"), 1], [1]])
def test_invalid_saved_local_step_pairs_are_rejected(source, tmp_path, pair):
    data = inputs(source, tmp_path)
    i = data["models"][0]["probe_indices"][0]
    data["models"][0]["probe_steps"][str(i)] = pair
    with pytest.raises(ValueError, match=r"precision pair|finite number"):
        c.prepare(tmp_path / "run", data, policy())


def test_unavailable_profile_derivative_preserves_structured_reason():
    from tests.test_prefix_multioutput_profile import profile_case

    _, oracle, profile = c.profiling.make_oracle(*profile_case())

    def unavailable(*args, **kwargs):
        raise c.profiling.ProfileUnavailable(
            "projection design rank unresolved",
            {"status": "rank_unavailable", "design_rank": 1},
        )

    profile.evaluate = unavailable
    value = c._profile_search(oracle, profile, oracle.start, 5, 5, lambda _: None)
    assert value["search_calls_started"] == 0
    assert value["projection_failures_observed"] >= 1
    assert value["projection_failures"][0]["phase"] == "outer_derivative"
    assert value["projection_failures"][0]["audit"]["status"] == "rank_unavailable"
    for _ in range(12):
        c._record_projection_failure(
            value,
            c.profiling.ProfileUnavailable("x", {"status": "rank_unavailable"}),
            "test",
        )
    assert len(value["projection_failures"]) == 8
    assert value["projection_failures_observed"] >= 13


def test_full_profile_history_binds_units_vectors_and_exact_endpoint():
    from tests.test_recovery_numerics import QuadraticOracle

    oracle = QuadraticOracle()
    prior, point = np.array([1.0, 3.0]), np.array([1.5, 4.5])
    parameters = oracle.parameters(point)
    fit = {
        "accepted_history_protocol": "full-coordinate-accepted-steps-1",
        "history_coordinate_names": list(oracle.names),
        "history_coordinate_units": oracle.units.tolist(),
        "last_accepted_parameters": parameters,
        "best": {"parameters": parameters, "training_nmse": 0.25},
        "accepted_steps": [
            {
                "training_nmse": 0.25,
                "evaluation_count": 2,
                "coordinate_step_norm": float(
                    np.linalg.norm((point - prior) / oracle.units)
                ),
                "previous_reduced_vector": prior.tolist(),
                "reduced_vector": point.tolist(),
                "parameters_sha256": c.public.content_sha256(parameters),
            }
        ],
    }
    history, matched = c._bound_full_history(oracle, parameters, fit)
    assert matched and len(history) == 1
    for change in ("units", "vector", "norm", "endpoint", "loss"):
        bad = deepcopy(fit)
        if change == "units":
            bad["history_coordinate_units"][0] = 99
        elif change == "vector":
            bad["accepted_steps"][0]["reduced_vector"][1] += 1
        elif change == "norm":
            bad["accepted_steps"][0]["coordinate_step_norm"] = 0.25
        elif change == "endpoint":
            bad["last_accepted_parameters"] = {"a": 1.6, "initial": 4.8}
        else:
            bad["best"]["training_nmse"] = 0.2
        assert c._bound_full_history(oracle, parameters, bad) == ([], False)


def test_full_residual_fallback_uses_same_budget_and_policy(monkeypatch):
    from time import monotonic

    policy = c.previous._adequacy_policy(diagnostic=True)
    guarded = SimpleNamespace(
        search_qualified=False,
        status="inconclusive",
        error=None,
        gradient_precision_qualified=False,
        parameter_information_qualified=False,
    )
    monkeypatch.setattr(c.recovery, "_derivative", lambda *a: guarded)
    monkeypatch.setattr(c.public, "_jsonable", lambda value: {"status": value.status})
    observed = []

    def full(oracle, vector, deadline, **kwargs):
        observed.append(deadline)
        assert kwargs["adequacy_policy"] == policy
        kwargs["audit"]["calls_started"] = 10
        return SimpleNamespace(status="passed")

    monkeypatch.setattr(c.precision, "derivative_check", full)
    budget = c.prefix_execution._budget(1, 2)
    state, snapshots = {}, []
    check = c._full_residual_fallback(
        None, np.ones(1), budget, 1, policy, state, snapshots.append
    )
    assert check.status == "passed"
    assert monotonic() < observed[0] <= budget.wall + 1.01
    assert state["full_residual_precision_audit"]["calls_started"] == 10
    assert [p["backend"] for p in state["derivative_passes"]] == [
        "guarded",
        "full-residual-multiprecision",
    ]
    assert snapshots


def test_new_diagnostic_cost_counts_both_passes_without_final_check_duplicates():
    def stage(guarded, full):
        return {
            "status": "complete",
            "wall_seconds": 12,
            "value": {
                "status": "complete",
                "derivative_passes": [
                    {
                        "backend": "guarded",
                        "derivative_check": {
                            "calls_started": guarded,
                            "calls_completed": guarded,
                        },
                    },
                    {
                        "backend": "full-residual-multiprecision",
                        "derivative_check": {
                            "calls_started": full,
                            "calls_completed": full,
                        },
                    },
                ],
                "derivative_check": {"calls_started": full, "calls_completed": full},
                "full_residual_precision_audit": {
                    "calls_started": full,
                    "calls_completed": full,
                },
            },
        }

    operations = {
        "qualification": stage(3, 7),
        "retained_diagnosis": stage(5, 11),
        "continuation_search": {
            "value": {
                "profile_derivative_check": {
                    "calls_started": 13,
                    "calls_completed": 13,
                    "elapsed_seconds": 2,
                }
            },
        },
    }
    before = deepcopy(operations)
    cost = c._diagnostic_cost(operations)
    assert cost["derivative_calls_started_observed"] == 26
    assert cost["derivative_calls_completed_observed"] == 26
    assert cost["stage_wall_seconds_observed"] == 24
    assert cost["precision_passes_observed"] == 2
    assert cost["profile_derivative_calls_started_observed"] == 13
    assert cost["complete"]
    assert operations == before


@pytest.mark.parametrize("stage", ["qualification", "retained_diagnosis"])
def test_new_diagnostic_cost_keeps_incomplete_mp_counters_without_check(stage):
    operations = {
        stage: {
            "status": "interrupted",
            "wall_seconds": 4,
            "observed_progress": {
                "derivative_passes": [
                    {
                        "backend": "guarded",
                        "derivative_check": {
                            "calls_started": 3,
                            "calls_completed": 3,
                        },
                    }
                ],
                "full_residual_precision_audit": {
                    "calls_started": 7,
                    "calls_completed": 5,
                    "calls_failed": 1,
                },
            },
        },
    }
    cost = c._diagnostic_cost(operations)
    assert cost["derivative_calls_started_observed"] == 10
    assert cost["derivative_calls_completed_observed"] == 8
    assert cost["stage_wall_seconds_observed"] == 4
    assert not cost["complete"]


def test_legacy_diagnostic_cost_is_unchanged():
    operations = {
        "qualification": {
            "status": "complete",
            "wall_seconds": 4,
            "value": {
                "passes": [
                    {
                        "status": "complete",
                        "derivative_check": {
                            "calls_started": 3,
                            "calls_completed": 3,
                        },
                    }
                ]
            },
        },
        "retained_diagnosis": {
            "status": "complete",
            "wall_seconds": 5,
            "value": {"derivative_check": {"calls_started": 7, "calls_completed": 7}},
        },
    }
    legacy = c.previous.previous._diagnostic_cost(operations)
    actual = c._diagnostic_cost(operations)
    assert {key: actual[key] for key in legacy} == legacy
    assert set(actual) - set(legacy) == {
        "profile_derivative_calls_started_observed",
        "profile_derivative_calls_completed_observed",
        "profile_derivative_seconds_observed",
        "profile_derivative_charge",
    }
