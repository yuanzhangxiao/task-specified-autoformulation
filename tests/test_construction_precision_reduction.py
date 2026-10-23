"""Certified continuation preserves provenance and diagnoses retained endpoints."""

from copy import deepcopy
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("mpmath")

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.research import construction_precision_reduction as c
from tests.test_construction_derivative_recovery import fake_recovery as fake_previous
from tests.test_construction_derivative_recovery import (
    source as derivative_source,
)
from tests.test_construction_fit_reliability import endpoint
from tests.test_construction_fit_reliability import (
    source as earlier_source,
)
from tests.test_construction_local_recovery import (
    source as reliability_source,
)


def policy():
    return c.Policy(
        population="synthetic-smoke",
        search_seconds=10,
        diagnostic_seconds=10,
        verification_seconds=10,
        maximum_calls=60,
    )


def fail(*args, **kwargs):
    pytest.fail("unexpected numerical work")


@pytest.fixture(autouse=True)
def stable_synthetic_source_identity(monkeypatch):
    """Unit fixtures use a fixed source identity; explicit mutation is tested below."""
    monkeypatch.setattr(c.public, "_source_identity", lambda: "d" * 64)


@pytest.fixture(scope="module")
def prepared_history(tmp_path_factory):
    """Build the authenticated synthetic lineage once, then isolate every test copy."""
    directory = tmp_path_factory.mktemp("precision-history")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(c.public, "_source_identity", lambda: "d" * 64)
        earlier = earlier_source.__wrapped__(directory)
        reliability = reliability_source.__wrapped__(earlier, directory, patch)
        local = derivative_source.__wrapped__(reliability, directory, patch)
        path = directory / "derivative-inputs.json"
        c.previous.export_inputs(local, path)
        root = directory / "derivative"
        c.previous.prepare(root, read_seal(path), policy())
        fake_previous(patch, improve=False)
        c.previous.run_task(root, 0)
    return root


@pytest.fixture
def source(prepared_history, tmp_path):
    return copytree(prepared_history, tmp_path / "source")


def inputs(source, tmp_path):
    path = tmp_path / "precision-inputs.json"
    c.export_inputs(source, path)
    return read_seal(path)


def fake_recovery(
    monkeypatch,
    *,
    qualify=True,
    source_valid=True,
    candidate_valid=True,
    candidate=0.1,
    candidate_loss=None,
    qualification_status="complete",
):
    events = []

    def make(
        problem, training, numerical, anchor, *, precision_mode, structural_reduction
    ):
        assert structural_reduction
        events.append(("build", anchor["a"], precision_mode))
        names = tuple(anchor)
        oracle = SimpleNamespace(
            vector=lambda values: np.asarray([values[name] for name in names]),
            parameters=lambda values: dict(zip(names, values.tolist(), strict=True)),
        )
        oracle.affine_audit = {"backend": precision_mode, "calls": 0}
        oracle.recovery_audit = {"anchor": deepcopy(anchor), "certified": True}
        return oracle, oracle

    def verify(original, oracle, p, seconds, save, *, tolerances):
        assert tolerances == ((1e-11, 1e-13),)
        events.append(("verify", p["a"], original.affine_audit["backend"]))
        is_candidate = any(event[0] == "search" for event in events)
        valid = candidate_valid if is_candidate else source_valid
        loss = candidate_loss if is_candidate and candidate_loss is not None else p["a"]
        value = endpoint(p, loss, valid)
        value["reduced_vector"] = oracle.vector(p).tolist()
        save(value)
        return value

    def qualification(
        problem,
        training,
        numerical,
        anchor,
        seconds,
        save,
        *,
        adequacy_policy,
        structural_reduction,
    ):
        assert structural_reduction
        assert adequacy_policy.local_max_scales == 6
        assert not adequacy_policy.diagnostic_precision
        events.append(("qualify", anchor["a"]))
        value = {
            "passed": qualify,
            "status": qualification_status,
            "precision_mode": "multiprecision",
            "passes": [
                {
                    "status": "complete",
                    "derivative_check": {"calls_started": 2, "calls_completed": 2},
                }
            ],
        }
        save(value)
        return value

    def search(oracle, start, seconds, calls, save, *, reject_invalid_trials):
        assert reject_invalid_trials
        events.append(("search", start["a"], oracle.affine_audit["backend"]))
        point = start | {"a": candidate}
        value = {
            "best": {"parameters": point, "training_nmse": candidate},
            "stop_reason": "step_or_gradient_converged",
            "message": "`xtol` termination condition is satisfied.",
            "calls": 1,
            "search_calls_started": 1,
        }
        save(value)
        return value

    def diagnose(original, oracle, p, checked, fit, seconds, save, *, adequacy_policy):
        assert adequacy_policy.diagnostic_precision
        assert adequacy_policy.local_max_scales == 12
        assert p == checked["parameters"]
        events.append(
            ("diagnose", p["a"], original.affine_audit["backend"], fit["stop_reason"])
        )
        value = {
            "assessment": {"classification": "optimization_unresolved"},
            "derivative_check": {"calls_started": 3, "calls_completed": 3},
        }
        save(value)
        return value

    monkeypatch.setattr(c.recovery, "make_oracle", make)
    monkeypatch.setattr(c.recovery, "qualification", qualification)
    monkeypatch.setattr(c.recovery, "diagnose", diagnose)
    monkeypatch.setattr(c.numerical, "verify_endpoint_ladder", verify)
    monkeypatch.setattr(c.numerical, "search", search)
    monkeypatch.setattr(c.numerical, "predetermined_starts", fail)
    return events


def test_policy_separates_qualification_and_diagnostic_precision():
    root = Path(__file__).resolve().parents[1]
    p = c.Policy.model_validate_json(
        (root / "configs/construction_precision_reduction_v1.json").read_text()
    )
    assert p.search_seconds == p.diagnostic_seconds == 600
    assert p.verification_seconds == 180
    assert sum(c._allowances(p).values()) == 2670
    assert c._adequacy_policy().local_max_scales == 6
    assert c._adequacy_policy(diagnostic=True).local_max_scales == 12
    assert c._adequacy_policy(diagnostic=True).diagnostic_precision
    with pytest.raises(ValueError):
        c.Policy(diagnostic_seconds=601)


def test_export_preserves_each_history_once_without_writing_source(
    source, tmp_path, monkeypatch
):
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    data = inputs(source, tmp_path)
    assert len(data["models"]) == len(data["parent_tasks"]) == 1
    assert len(data["local_recovery_tasks"]) == 2
    assert len(data["earlier_cost_by_model"]) == 1
    assert data["models"][0]["endpoint"]["parameters"]["a"] == 0.25
    assert "endpoint" not in data["local_recovery_tasks"][1]
    assert not data["earlier_cost_by_model"][0]["arms"][1]["original_cost_known"]
    assert before == {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in source.rglob("*")
        if p.is_file()
    }
    with pytest.raises(ValueError, match="source campaign"):
        c.export_inputs(source, source / "bad.json")


@pytest.mark.parametrize("artifact", ["progress", "terminal", "reservation"])
def test_export_rejects_corrupted_parent_receipts(source, tmp_path, artifact):
    folder = source / "results/task_00"
    if artifact == "progress":
        path = folder / "operations/incumbent_verification/progress.json"
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
        c.export_inputs(source, tmp_path / "bad.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "heldout",
        "nested",
        "endpoint",
        "identity",
        "history",
        "cost",
        "origin",
        "replay",
    ],
)
def test_strict_inputs_preserve_roster_split_and_cost_schema(
    source, tmp_path, mutation
):
    data = inputs(source, tmp_path)
    if mutation == "heldout":
        data["validation"] = {}
    elif mutation == "nested":
        data["training_inputs"]["validation"] = {}
    elif mutation == "endpoint":
        data["models"][0]["endpoint"]["parameters"]["a"] += 1
    elif mutation == "identity":
        data["models"][0]["problem_sha256"] = "a" * 64
    elif mutation == "history":
        data["local_recovery_tasks"] = []
    elif mutation == "cost":
        data["parent_tasks"][0]["cost"]["test_nmse"] = 0
    elif mutation == "origin":
        data["models"][0]["endpoint"]["origin"] = "qualification"
    else:
        data["local_recovery_tasks"][1]["endpoint"] = data["models"][0]["endpoint"]
    with pytest.raises(ValueError):
        c.prepare(tmp_path / "bad", data, policy())


@pytest.mark.parametrize(
    "case", ["improved", "tied_different", "worse", "invalid_candidate", "blocked"]
)
def test_diagnoses_exact_retained_point_including_unchanged_incumbent(
    source, tmp_path, monkeypatch, case
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    args = {
        "qualify": case != "blocked",
        "candidate_valid": case != "invalid_candidate",
        "candidate": 0.1 if case in {"improved", "invalid_candidate"} else 0.4,
        "candidate_loss": 0.25 if case == "tied_different" else None,
    }
    events = fake_recovery(monkeypatch, **args)
    result = c.run_task(root, 0)
    expected = 0.1 if case == "improved" else 0.25
    assert result["retained"]["parameters"]["a"] == expected
    diagnosed = [e for e in events if e[0] == "diagnose"]
    assert len(diagnosed) == 1 and diagnosed[0][1] == expected
    receipt = result["operations"]["retained_diagnosis"]["value"]
    assert receipt["diagnosed_parameters_sha256"] == c.public.content_sha256(
        result["retained"]["parameters"]
    )
    assert receipt["verification_sha256"] == c.public.content_sha256(result["retained"])
    assert receipt["search_history_matches_retained"] == (case == "improved")
    assert result["fitting_diagnosis"] is not None
    assert result["optimizer_termination"] == (
        "not_started" if case == "blocked" else "xtol"
    )
    assert all(
        e[-1] == "multiprecision"
        for e in events
        if e[0] in {"build", "verify", "search"}
    )
    assert all(e[1] == 0.25 for e in events if e[0] == "build")
    assert result["diagnostic_cost"]["derivative_calls_started_observed"] == 5
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    assert c.run_task(root, 0) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}


def test_failed_current_verification_preserves_history_without_search_or_diagnosis(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    events = fake_recovery(monkeypatch, source_valid=False)
    result = c.run_task(root, 0)
    assert result["retained"] is None
    assert result["source_incumbent"]["original_equations_verified"]
    assert result["fitting_diagnosis"] is None
    assert not any(e[0] in {"search", "diagnose"} for e in events)


def test_interrupted_stage_cannot_renew_budget(source, tmp_path, monkeypatch):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    plan, _ = c.verify(root)
    folder = root / "results/task_00"
    identity = {"plan_sha256": c.public.content_sha256(plan), "task": plan["tasks"][0]}
    seal(
        folder / "operations/qualification/started.json",
        {
            "campaign": identity,
            "stage": "qualification",
            "allowance_seconds": c._allowances(policy())["qualification"],
        },
    )
    fake_recovery(monkeypatch)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    monkeypatch.setattr(c.numerical, "search", fail)
    value = c.run_task(root, 0)
    assert value["operations"]["qualification"]["status"] == "interrupted"
    assert not value["cost"]["accounting_complete"]
    assert value["retained"] and value["fitting_diagnosis"]


def test_diagnosis_cannot_be_reassigned_to_another_retained_vector():
    point = endpoint({"a": 0.25}, 0.25)
    ops = {
        "incumbent_verification": {"status": "complete", "value": point},
        "retained_diagnosis": {
            "status": "complete",
            "value": {
                "diagnosed_parameters_sha256": c.public.content_sha256({"a": 0.1}),
                "verification_sha256": c.public.content_sha256(point),
                "assessment": {"classification": "locally_converged_poor_fit"},
            },
        },
    }
    with pytest.raises(ValueError, match="exact retained endpoint"):
        c._assemble({}, {"endpoint": point}, ops)


def test_verified_vector_is_reused_without_nonexact_reduce_lift_round_trip(monkeypatch):
    point = {
        "effective_parameters": {"decay": 0.3},
        "reduced_vector": [0.3],
        "parameters": {"tau": 1 / 0.3},
    }
    oracle = SimpleNamespace(vector=lambda values: np.asarray([values["decay"]]))
    monkeypatch.setattr(c.numerical, "to_effective", fail)
    monkeypatch.setattr(c.numerical, "to_original", lambda *a: point["parameters"])
    assert c._verified_coordinates(None, oracle, point) == {"decay": 0.3}
    changed = deepcopy(point)
    changed["reduced_vector"][0] = np.nextafter(0.3, np.inf)
    with pytest.raises(ValueError, match="coordinate vector"):
        c._verified_coordinates(None, oracle, changed)
    monkeypatch.setattr(c.numerical, "to_original", lambda *a: {"tau": 2.0})
    with pytest.raises(ValueError, match="coordinate lift"):
        c._verified_coordinates(None, oracle, point)


def test_report_is_read_only_numerically_and_retains_separate_costs(
    source, tmp_path, monkeypatch
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    fake_recovery(monkeypatch)
    c.run_task(root, 0)
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    result = c.report(root)
    assert result["expected"] == result["recorded"] == 1
    assert len(result["parent_cost_by_task"]) == 1
    assert len(result["local_recovery_cost_by_task"]) == 2
    assert len(result["earlier_cost_by_model"]) == 1
    assert result["verification_only_tasks_repeated"] == 0
    assert Path(c.inspect(root)["archive"]).is_file()


@pytest.mark.parametrize("changed", ["runtime", "source"])
def test_changed_runtime_blocks_execution_not_report(
    source, tmp_path, monkeypatch, changed
):
    root = tmp_path / "new"
    c.prepare(root, inputs(source, tmp_path), policy())
    if changed == "runtime":
        monkeypatch.setattr(c, "_runtime", lambda: {"different": True})
    else:
        monkeypatch.setattr(c.public, "_source_identity", lambda: "c" * 64)
    with pytest.raises(ValueError, match="source/runtime"):
        c.run_task(root, 0)
    assert c.verify(root, runtime=False)[0]["protocol"] == c.PROTOCOL


def test_real_small_certified_continuation_and_resume(source, tmp_path, monkeypatch):
    root = tmp_path / "real"
    c.prepare(root, inputs(source, tmp_path), policy())
    result = c.run_task(root, 0)
    assert result["retained"] and result["retained"]["original_equations_verified"]
    assert result["qualification_passed"] and result["search_started"]
    assert result["retained"]["training_nmse"] < 1e-6
    assert result["operations"]["retained_diagnosis"]["value"][
        "diagnosed_parameters_sha256"
    ] == c.public.content_sha256(result["retained"]["parameters"])
    before = {p: p.read_bytes() for p in root.rglob("*.json")}
    monkeypatch.setattr(c.recovery, "make_oracle", fail)
    monkeypatch.setattr(c.recovery, "qualification", fail)
    assert c.run_task(root, 0) == result
    assert before == {p: p.read_bytes() for p in root.rglob("*.json")}
