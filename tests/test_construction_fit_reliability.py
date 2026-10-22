"""Reliability keeps source history, replay ordering and new allowances separate."""

from copy import deepcopy
from pathlib import Path

import pytest

pytest.importorskip("casadi")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_fit_reliability as c
from tests.test_construction_fit_continuation import inputs


def policy(**changes):
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        diagnostic_seconds=10,
        verification_seconds=10,
        replay_seconds=10,
        maximum_calls=30,
        **changes,
    )


def endpoint(parameters, loss, verified=True):
    return {
        "status": "complete",
        "parameters": deepcopy(parameters),
        "effective_parameters": deepcopy(parameters),
        "original_equations_verified": verified,
        "training_nmse": loss,
        "alternate_training_nmse": loss,
        "solver_loss_discrepancy": 0.0,
        "solver_prediction_discrepancy": 0.0,
        "per_target_normalized_mse": {"y": loss},
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    data = inputs()
    settings = c.previous.Policy(population="synthetic-smoke")
    c.previous.prepare(root, data, settings)
    plan, _ = c.previous.verify(root)
    task = plan["tasks"][0]
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"] / task["arm"]
    base = data["models"][0]["parameters"]
    values = {
        "qualification": {"passed": True},
        "retained_control_verification": endpoint(base, 4.0),
        "inherited_verification": endpoint(base | {"a": 0.5}, 0.5, False),
        "declared_verification": endpoint(base | {"a": 2.0}, 2.0),
    }
    allowances = c.previous._allowances(settings)
    for name, value in values.items():
        c.journal._operation(
            folder,
            identity,
            name,
            allowances[name],
            lambda save, v=value: (save(v), v)[1],
        )
    ops = c.previous._operations(folder, identity, settings)
    seal(folder / "result.json", c.previous._assemble(identity, ops, terminal=True))
    return root


def exported(source, tmp_path):
    path = tmp_path / "bundle.json"
    c.export_inputs(source, path)
    return read_seal(path)


def fail(*args, **kwargs):
    pytest.fail("unexpected numerical work")


def test_export_authenticates_old_receipts_without_runtime_or_writes(
    source, tmp_path, monkeypatch
):
    before = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(c.public, "_runtime", fail)
    monkeypatch.setattr(c.public, "_source_identity", fail)
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    data = exported(source, tmp_path)
    arm, missing = data["arms"]
    assert arm["incumbent"]["origin"] == "declared_verification"
    assert arm["incumbent"]["parameters"]["a"] == 2.0
    assert arm["failed_endpoints"][0]["parameters"]["a"] == 0.5
    assert arm["source_cost_known"] and arm["source_cost"]["operations"] == 4
    assert missing["source_status"] == "source_missing"
    assert missing["source_cost"] is None and not missing["source_cost_known"]
    assert (
        missing["incumbent"]["parameters"]
        == data["source_inputs"]["models"][0]["parameters"]
    )
    assert before == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "new.json")


@pytest.mark.parametrize("artifact", ["progress", "terminal", "reservation"])
def test_source_authentication_rejects_mismatched_receipts(source, tmp_path, artifact):
    folder = source / "results/control/affine_original"
    if artifact == "progress":
        path = folder / "operations/inherited_verification/progress.json"
        value = c.public._read(path)
        value["parameters"]["a"] = 123.0
        c.public._write(path, value)
    else:
        path = (
            folder / "result.json"
            if artifact == "terminal"
            else folder / "operations/qualification/started.json"
        )
        value = read_seal(path)
        value["status" if artifact == "terminal" else "stage"] = "changed"
        path.unlink()
        seal(path, value)
    with pytest.raises(ValueError):
        c.export_inputs(source, tmp_path / "bad.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "nested",
        "endpoint",
        "provenance",
        "cost",
        "missing",
        "order",
        "duplicate",
        "targets",
    ],
)
def test_strict_input_boundary_rejects_hidden_payloads_and_identity_changes(
    source, tmp_path, mutation
):
    data = exported(source, tmp_path)
    arm = data["arms"][0]
    if mutation == "heldout":
        data["validation"] = data["source_inputs"]["training"]
    elif mutation == "nested":
        data["source_inputs"]["validation"] = data["source_inputs"]["training"]
    elif mutation == "endpoint":
        arm["incumbent"]["parameters"]["a"] += 1
    elif mutation == "provenance":
        arm["operations"]["qualification"]["validation_nmse"] = 1.0
    elif mutation == "cost":
        arm["source_cost"]["test_nmse"] = 1.0
    elif mutation == "missing":
        data["arms"][1]["source_cost_known"] = True
    elif mutation == "order":
        data["arms"].reverse()
    elif mutation == "duplicate":
        arm["failed_endpoints"] *= 2
    else:
        arm["incumbent"]["per_target_normalized_mse"]["validation"] = 1.0
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "new", data, policy())


class Oracle:
    def vector(self, values):
        return values

    def parameters(self, values):
        return deepcopy(values)


def fake_numerics(
    monkeypatch, *, qualified=True, replay_verified=True, search_verified=True
):
    events = []
    oracle = Oracle()
    monkeypatch.setattr(c.numerical, "make_oracle", lambda *args: (oracle, oracle))

    def verify(o, a, p, seconds, save):
        events.append(("verify", p["a"]))
        valid = (
            replay_verified
            if p["a"] == 0.5
            else search_verified
            if p["a"] == 0.25
            else True
        )
        value = endpoint(p, p["a"], valid)
        save(value)
        return value

    def qualify(o, a, p, seconds, save, *, adequacy_policy):
        assert adequacy_policy.precision_aware
        events.append(("qualify", p["a"]))
        value = {"passed": qualified}
        save(value)
        return value

    def search(a, start, seconds, calls, save, *, reject_invalid_trials):
        assert reject_invalid_trials
        events.append(("search", start["a"]))
        value = {
            "best": {"parameters": start | {"a": 0.25}, "training_nmse": 0.25},
            "stop_reason": "step_or_gradient_converged",
            "calls": 1,
        }
        save(value)
        return value

    def diagnose(a, p, v, fit, seconds, save, *, adequacy_policy):
        assert adequacy_policy.precision_aware
        events.append(("diagnose", p["a"]))
        value = {"assessment": {"classification": "locally_converged_poor_fit"}}
        save(value)
        return value

    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)
    monkeypatch.setattr(c.numerical, "qualification", qualify)
    monkeypatch.setattr(c.numerical, "search", search)
    monkeypatch.setattr(c.numerical, "diagnose", diagnose)
    monkeypatch.setattr(c.numerical, "predetermined_starts", fail)
    return events


def test_replays_before_start_choice_then_one_continuation_and_no_old_starts(
    source, tmp_path, monkeypatch
):
    data = exported(source, tmp_path)
    root = tmp_path / "new"
    c.prepare(root, data, policy())
    events = fake_numerics(monkeypatch)
    result = c.run_task(root, 0)
    assert events == [
        ("verify", 2.0),
        ("verify", 0.5),
        ("qualify", 0.5),
        ("search", 0.5),
        ("verify", 0.25),
        ("diagnose", 0.25),
    ]
    assert result["selected_start"]["origin"] == "replay_0_verification"
    assert result["retained"]["parameters"]["a"] == 0.25
    assert result["source_incumbent"]["parameters"]["a"] == 2.0
    assert result["before_after"] == {
        "source_training_nmse": 2.0,
        "selected_start_training_nmse": 0.5,
        "retained_training_nmse": 0.25,
    }
    assert result["source_cost"] == data["arms"][0]["source_cost"]
    assert result["cost"]["operations"] == 6
    assert not result["old_allowances_renewed"]
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    assert c.run_task(root, 0) == result
    assert c.report(root)["recorded"] == 1
    assert c.report(root)["expected"] == 2
    assert Path(c.inspect(root)["archive"]).is_file()


def test_unresolved_replay_and_qualification_keep_exact_incumbent(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    data = exported(source, tmp_path)
    c.prepare(root, data, policy())
    events = fake_numerics(monkeypatch, qualified=False, replay_verified=False)
    result = c.run_task(root, 0)
    assert events == [("verify", 2.0), ("verify", 0.5), ("qualify", 2.0)]
    assert (
        result["retained"]["parameters"] == data["arms"][0]["incumbent"]["parameters"]
    )
    assert result["optimization_status"] == "optimization_unresolved"


def test_failed_new_verification_keeps_selected_replayed_endpoint(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, exported(source, tmp_path), policy())
    events = fake_numerics(monkeypatch, search_verified=False)
    result = c.run_task(root, 0)
    assert result["retained"]["parameters"]["a"] == 0.5
    assert result["optimization_status"] == "optimization_unresolved"
    assert not any(e[0] == "diagnose" for e in events)


def test_missing_arm_has_new_budget_but_no_known_or_zero_source_cost(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    data = exported(source, tmp_path)
    c.prepare(root, data, policy())
    fake_numerics(monkeypatch, qualified=False)
    result = c.run_task(root, 1)
    assert result["source_status"] == "source_missing"
    assert result["source_cost"] is None
    assert not result["source_cost_known"]
    assert result["source_incumbent"]["origin"] == "original_saved_vector"
    assert (
        result["source_incumbent"]["parameters"]
        == data["source_inputs"]["models"][0]["parameters"]
    )
    assert result["cost"]["operations"] == 2
    report = c.report(root)
    assert report["source_missing"] == [1] and not report["source_cost_complete"]


def test_failed_original_replay_preserves_historical_incumbent_without_search(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    data = exported(source, tmp_path)
    c.prepare(root, data, policy())
    fake_numerics(monkeypatch)
    monkeypatch.setattr(
        c.numerical,
        "verify_endpoint_ladder",
        lambda *args: {"original_equations_verified": False},
    )
    monkeypatch.setattr(c.numerical, "search", fail)
    result = c.run_task(root, 0)
    assert result["retained"] == data["arms"][0]["incumbent"]
    assert result["retained_origin"] == "source_incumbent"
    assert not result["selected_start"]["currently_verified"]


def test_interrupted_new_search_recovers_best_without_renewing_budget(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    data = exported(source, tmp_path)
    c.prepare(root, data, policy())
    plan, _ = c.verify(root)
    task, arm = plan["tasks"][0], data["arms"][0]
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": task}
    location = root / "results/control/affine_original/operations/continuation_search"
    bound = {
        "campaign": identity,
        "stage": "continuation_search",
        "allowance_seconds": c._allowances(policy(), arm)["continuation_search"],
    }
    seal(location / "started.json", bound)
    c.public._write(
        location / "progress.json",
        {
            "best": {
                "parameters": arm["incumbent"]["parameters"] | {"a": 0.25},
                "training_nmse": 0.25,
            }
        },
    )
    events = fake_numerics(monkeypatch)
    monkeypatch.setattr(c.numerical, "search", fail)

    # The real diagnostic treats interrupted search as unresolved; assert it gets
    # the interruption rather than manufacturing a fresh optimizer termination.
    def diagnosis(a, p, v, fit, seconds, save, **kwargs):
        assert fit["stop_reason"] == "interrupted" and fit["budget_exhausted"]
        return {"assessment": {"classification": "optimization_unresolved"}}

    monkeypatch.setattr(c.numerical, "diagnose", diagnosis)
    result = c.run_task(root, 0)
    assert result["retained"]["parameters"]["a"] == 0.25
    assert result["operations"]["continuation_search"]["status"] == "interrupted"
    assert not result["cost"]["accounting_complete"]
    assert not any(e[0] == "search" for e in events)
    assert result["optimization_status"] == "optimization_unresolved"


def test_schemas_only_prepare_and_read_only_reports(source, tmp_path, monkeypatch):
    data = exported(source, tmp_path)
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    monkeypatch.setattr(c.numerical, "search", fail)
    root = tmp_path / "new"
    c.prepare(root, data, policy())
    report = c.report(root)
    assert report["expected"] == 2 and report["recorded"] == 0
    assert report["rows"][1]["source_cost"] is None
    assert Path(c.inspect(root)["archive"]).is_file()
    with pytest.raises(ValueError, match="unknown reliability"):
        c.run_task(root, 2)


def test_real_numerical_smoke_preserves_original_model_and_scoped_roots(tmp_path):
    source = tmp_path / "source"
    data = inputs()
    c.previous.prepare(
        source, data, c.previous.Policy(population="synthetic-smoke", restart_count=0)
    )
    portable = exported(source, tmp_path)
    root = tmp_path / "new"
    c.prepare(root, portable, policy())
    result = c.run_task(root, 1)
    assert result["qualification_passed"], result["operations"].get("qualification")
    assert result["retained"]["original_equations_verified"]
    assert result["retained"]["training_nmse"] < 1e-7
    assert set(result["retained"]["parameters"]) == set(data["models"][0]["parameters"])
    assert result["source_cost"] is None
    assert read_seal(source / "inputs.json") == data
    assert (
        result["operations"]["continuation_search"]["value"]["solver_scaling"][
            "objective_multiplier"
        ]
        == 1.0
    )


@pytest.mark.parametrize("same_parameters", [True, False])
def test_equal_score_diagnosis_binds_exact_retained_parameters(
    source, tmp_path, monkeypatch, same_parameters
):
    root = tmp_path / "new"
    c.prepare(root, exported(source, tmp_path), policy())
    fake_numerics(monkeypatch, replay_verified=False)

    def search(a, start, seconds, calls, save, **kwargs):
        return {
            "best": {
                "parameters": start if same_parameters else start | {"g": 9.0},
                "training_nmse": 2.0,
            },
            "stop_reason": "step_or_gradient_converged",
        }

    monkeypatch.setattr(c.numerical, "search", search)
    result = c.run_task(root, 0)
    assert result["retained_origin"] == "incumbent_verification"
    assert result["optimization_status"] == (
        "locally_converged_poor_fit" if same_parameters else "optimization_unresolved"
    )
    assert result["retained"]["parameters"]["g"] != 9.0


def test_selected_start_seal_must_match_presearch_replays(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, exported(source, tmp_path), policy())
    fake_numerics(monkeypatch)
    c.run_task(root, 0)
    path = root / "results/control/affine_original/selected_start.json"
    value = read_seal(path)
    value["parameters"]["a"] = 4.0
    path.unlink()
    seal(path, value)
    monkeypatch.setattr(c.numerical, "make_oracle", fail)
    with pytest.raises(ValueError, match="selected start"):
        c.report(root)
    with pytest.raises(ValueError, match="selected start"):
        c.run_task(root, 0)


@pytest.mark.parametrize(
    "mutation", ["missing_verified", "unsealed_endpoint", "bad_discrepancy"]
)
def test_provenance_cannot_certify_missing_or_unsealed_or_disagreeing_endpoint(
    source, tmp_path, mutation
):
    data = exported(source, tmp_path)
    if mutation == "missing_verified":
        data["arms"][1]["incumbent"]["original_equations_verified"] = True
    elif mutation == "unsealed_endpoint":
        data["arms"][0]["operations"]["declared_verification"]["receipt_sha256"] = None
    else:
        data["arms"][0]["incumbent"]["solver_prediction_discrepancy"] = 1.0
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "bad", data, policy())
