"""Cross-arm recovery authenticates history, current verification and budgets."""

from pathlib import Path

import pytest

pytest.importorskip("casadi")
pytest.importorskip("mpmath")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_local_recovery as c
from tests.test_construction_fit_reliability import (
    Oracle,
    endpoint,
    fake_numerics,
)
from tests.test_construction_fit_reliability import (
    source as earlier_source,  # noqa: F401
)


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        diagnostic_seconds=10,
        verification_seconds=10,
        maximum_calls=60,
    )


@pytest.fixture
def source(earlier_source, tmp_path, monkeypatch):  # noqa: F811
    old_path = tmp_path / "old-inputs.json"
    c.previous.export_inputs(earlier_source, old_path)
    root = tmp_path / "reliability"
    c.previous.prepare(
        root, read_seal(old_path), c.previous.Policy(population="synthetic-smoke")
    )
    with monkeypatch.context() as patch:
        fake_numerics(patch)
        c.previous.run_task(root, 0)

        def search(a, start, seconds, calls, save, **kwargs):
            del a, seconds, calls, kwargs
            value = {
                "best": {"parameters": start | {"a": 0.4}, "training_nmse": 0.4},
                "stop_reason": "budget_limited",
                "calls": 1,
            }
            save(value)
            return value

        patch.setattr(c.numerical, "search", search)
        c.previous.run_task(root, 1)
    return root


def inputs(source, tmp_path):
    path = tmp_path / "local-inputs.json"
    c.export_inputs(source, path)
    return read_seal(path)


def fail(*args, **kwargs):
    pytest.fail("unexpected numerical work")


def fake_local(
    monkeypatch, *, qualify=True, source_valid=True, candidate_valid=True, improve=True
):
    events = []
    oracle = Oracle()
    oracle.affine_audit = {"backend": "fake-test", "calls": 0}
    monkeypatch.setattr(c.numerical, "make_oracle", lambda *a, **kw: (oracle, oracle))

    def verify(original, oracle, p, seconds, save, *, tolerances):
        assert tolerances == ((1e-11, 1e-13),)
        oracle.affine_audit["calls"] += 1
        events.append(("verify", p["a"]))
        valid = candidate_valid if p["a"] == 0.1 else source_valid
        value = endpoint(p, p["a"], valid)
        save(value)
        return value

    def qualification(original, oracle, p, seconds, save, *, adequacy_policy):
        assert adequacy_policy.local_adaptive and adequacy_policy.precision_aware
        events.append(("qualify", p["a"]))
        value = {"passed": qualify}
        save(value)
        return value

    def search(oracle, start, seconds, calls, save, *, reject_invalid_trials):
        assert reject_invalid_trials
        events.append(("search", start["a"]))
        point = start | {"a": 0.1 if improve else start["a"]}
        value = {
            "best": {"parameters": point, "training_nmse": point["a"]},
            "stop_reason": "step_or_gradient_converged",
            "calls": 1,
            "search_calls_started": 1,
        }
        save(value)
        return value

    def diagnose(oracle, p, checked, fit, seconds, save, *, adequacy_policy):
        assert adequacy_policy.local_adaptive
        events.append(("diagnose", p["a"]))
        value = {"assessment": {"classification": "locally_converged_poor_fit"}}
        save(value)
        return value

    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)
    monkeypatch.setattr(c.numerical, "qualification", qualification)
    monkeypatch.setattr(c.numerical, "search", search)
    monkeypatch.setattr(c.numerical, "diagnose", diagnose)
    monkeypatch.setattr(c.numerical, "predetermined_starts", fail)
    return events


def test_export_checks_full_history_without_source_mutation(
    source, tmp_path, monkeypatch
):
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(c.public, "_runtime", fail)
    monkeypatch.setattr(c.public, "_source_identity", fail)
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    data = inputs(source, tmp_path)
    assert len(data["models"]) == 1 and len(data["replays"]) == 1
    assert [a["endpoint"]["parameters"]["a"] for a in data["models"][0]["arms"]] == [
        0.25,
        0.4,
    ]
    assert data["models"][0]["arms"][1]["original_cost"] is None
    assert data["models"][0]["arms"][1]["original_cost_known"] is False
    assert "source_inputs" not in data["training_inputs"]
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "new.json")


@pytest.mark.parametrize("artifact", ["progress", "terminal", "reservation"])
def test_source_authentication_fails_for_altered_receipts(source, tmp_path, artifact):
    folder = source / "results/control/affine_original"
    if artifact == "progress":
        path = folder / "operations/continuation_verification/progress.json"
        value = c.public._read(path)
        value["parameters"]["a"] += 1
        c.public._write(path, value)
    else:
        path = folder / (
            "result.json"
            if artifact == "terminal"
            else "operations/qualification/started.json"
        )
        value = read_seal(path)
        value["status" if artifact == "terminal" else "stage"] = "changed"
        path.unlink()
        seal(path, value)
    with pytest.raises(ValueError):
        c.export_inputs(source, tmp_path / "invalid.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "nested",
        "endpoint",
        "channels",
        "model",
        "order",
        "cost",
        "duplicate_replay",
        "replay_model",
        "replay_verified",
    ],
)
def test_strict_export_rejects_wrong_model_or_hidden_data(source, tmp_path, mutation):
    data = inputs(source, tmp_path)
    arm = data["models"][0]["arms"][0]
    if mutation == "heldout":
        data["validation"] = data["training_inputs"]["training"]
    elif mutation == "nested":
        data["training_inputs"]["validation"] = data["training_inputs"]["training"]
    elif mutation == "endpoint":
        arm["endpoint"]["parameters"]["a"] += 1
    elif mutation == "channels":
        arm["endpoint"]["per_target_normalized_mse"]["unknown"] = 1.0
    elif mutation == "model":
        data["models"][0]["problem_sha256"] = "a" * 64
    elif mutation == "order":
        data["models"][0]["arms"].reverse()
    elif mutation == "cost":
        arm["reliability_cost"]["test_nmse"] = 0.1
    elif mutation == "duplicate_replay":
        data["replays"] *= 2
    elif mutation == "replay_model":
        data["replays"][0]["model"] = 1
    else:
        data["replays"][0]["endpoint"]["original_equations_verified"] = True
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "bad", data, policy())


def test_cross_arm_best_current_verification_precedes_reduced_search(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake_local(monkeypatch)
    value = c.run_task(root, 0)
    assert events == [
        ("verify", 0.25),
        ("verify", 0.4),
        ("qualify", 0.25),
        ("search", 0.25),
        ("verify", 0.1),
        ("diagnose", 0.1),
    ]
    assert value["selected_start"]["source_arm"] == "affine_original"
    assert value["retained"]["training_nmse"] == 0.1
    assert value["search_started"] and value["search_not_started_reason"] is None
    assert value["optimization_status"] == "locally_converged_poor_fit"
    assert all(
        op["value"]["affine_execution"]["backend"] == "fake-test"
        for op in value["operations"].values()
    )
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    assert c.run_task(root, 0) == value
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_disputed_replay_is_independent_and_never_fits(source, tmp_path, monkeypatch):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake_local(monkeypatch)
    monkeypatch.setattr(c.numerical, "search", fail)
    monkeypatch.setattr(c.numerical, "qualification", fail)
    value = c.run_task(root, 1)
    assert events == [("verify", 0.5)]
    assert value["disputed_endpoint"]["original_equations_verified"]
    assert value["retained"] is None and value["selected_start"] is None
    assert value["search_not_started_reason"] == "verification_only_by_design"
    assert set(value["operations"]) == {"disputed_verification"}
    report = c.report(root)
    assert report["recorded"] == 1 and report["expected"] == 2
    assert len(report["historical_cost_by_model"]) == 1
    assert all("original_cost" not in row for row in report["rows"])
    assert Path(c.inspect(root)["archive"]).is_file()


@pytest.mark.parametrize("failure", ["source", "qualification", "candidate"])
def test_no_false_certification_or_lost_incumbent(
    source, tmp_path, monkeypatch, failure
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_local(
        monkeypatch,
        source_valid=failure != "source",
        qualify=failure != "qualification",
        candidate_valid=failure != "candidate",
    )
    value = c.run_task(root, 0)
    if failure == "source":
        assert value["retained"] is None
        assert (
            value["search_not_started_reason"]
            == "no_currently_verified_source_endpoint"
        )
        assert value["historical_endpoints"][0]["endpoint"][
            "original_equations_verified"
        ]
    else:
        assert value["retained"]["training_nmse"] == 0.25
        assert value["retained_origin"] == "source_original_verification"
    assert value["fitting_diagnosis"] is None
    assert value["optimization_status"] == "optimization_unresolved"


def test_interrupted_qualification_cannot_renew_budget(source, tmp_path, monkeypatch):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    plan, _ = c.verify(root)
    task = plan["tasks"][0]
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": task}
    directory = c._folder(root, task) / "operations/qualification"
    seal(
        directory / "started.json",
        {
            "campaign": identity,
            "stage": "qualification",
            "allowance_seconds": c._allowances(policy(), task)["qualification"],
        },
    )
    fake_local(monkeypatch)
    monkeypatch.setattr(c.numerical, "qualification", fail)
    monkeypatch.setattr(c.numerical, "search", fail)
    value = c.run_task(root, 0)
    assert value["operations"]["qualification"]["status"] == "interrupted"
    assert not value["cost"]["accounting_complete"]
    assert value["retained"]["training_nmse"] == 0.25
    assert value["search_not_started_reason"] == "qualification_interrupted"


def test_equal_retained_vector_can_inherit_own_search_diagnosis(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_local(monkeypatch, improve=False)
    value = c.run_task(root, 0)
    assert value["retained_origin"] == "source_original_verification"
    assert value["fitting_diagnosis"] is not None


def test_real_small_reduced_continuation(source, tmp_path):
    root = tmp_path / "real"
    c.prepare(root, inputs(source, tmp_path), policy())
    result = c.run_task(root, 0)
    assert result["retained"] is not None, result["operations"]
    assert result["retained"]["original_equations_verified"]
    assert result["qualification_passed"], result["operations"]
    assert result["search_started"], result["operations"]
    assert result["retained"]["training_nmse"] < 1e-6
    assert (
        result["operations"]["qualification"]["value"]["optimizer_coordinates"]
        < result["operations"]["qualification"]["value"]["original_coordinates"]
    )
    assert result["cost"]["search_calls_started_total"] > 0
    assert not result["validation_data_opened"]
