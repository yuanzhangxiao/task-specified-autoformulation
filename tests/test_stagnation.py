"""M25 derivative gates, scaling instrumentation, frozen inputs and CPU submission."""

from copy import deepcopy
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting import stagnation_campaign as campaign
from autoformalism.fitting.stagnation_audit import AuditPolicy, audit
from tests.test_incumbent_continuation import bundle_path as warm_bundle_fixture
from tests.test_nonlinear_comparison import checked
from tests.test_nonlinear_comparison import exported as exported_fixture
from tests.test_recovery_numerics import QuadraticOracle

exported = exported_fixture
warm_bundle = warm_bundle_fixture


class Profile:
    outer = (0, 1)

    def evaluate(self, vector, end):
        return vector, vector - [2, 3], np.eye(2), {"active_mask": [0]}


def refined_policy():
    return campaign.Policy.model_validate_json(
        (Path(__file__).resolve().parents[1] / "configs/phase_c_stagnation_v2.json")
        .read_text()
    )


def test_refined_steps_resolve_truncation_without_accepting_wrong_derivatives():
    class Curved(Profile):
        def evaluate(self, vector, end):
            phase = 4000 * (vector[0] - 1) + 0.2
            residual = np.array([np.sin(phase), vector[1] - 3])
            jacobian = np.diag([4000 * np.cos(phase), 1])
            return vector, residual, jacobian, {"active_mask": [0]}

    class Wrong(Curved):
        def evaluate(self, vector, end):
            v, r, j, info = super().evaluate(vector, end)
            return v, r, 1.01 * j, info

    oracle, parameters = QuadraticOracle(), {"a": 1.0, "initial": 0.0}
    end, save = monotonic() + 10, lambda _: None
    coarse = audit(oracle, Curved(), parameters, AuditPolicy(), end, save)
    fine = audit(oracle, Curved(), parameters, refined_policy().audit, end, save)
    wrong = audit(oracle, Wrong(), parameters, refined_policy().audit, end, save)
    assert coarse["status"] == "failed"
    assert fine["status"] == "passed" and fine["calls"] == 25
    assert wrong["status"] == "failed"


def test_derivative_audit_checks_four_directions_three_scales():
    progress = []
    result = audit(
        QuadraticOracle(),
        Profile(),
        {"a": 1, "initial": 0},
        AuditPolicy(),
        monotonic() + 10,
        progress.append,
    )
    assert result["status"] == "passed"
    assert result["calls"] == result["completed_calls"] == 25
    assert len(result["directions"]) == 4
    assert progress[-1]["completed_calls"] == 25


@pytest.mark.parametrize("policy", [AuditPolicy(), refined_policy().audit])
def test_wrong_derivative_fails_and_active_transitions_are_inconclusive(policy):
    class Wrong(Profile):
        def evaluate(self, vector, end):
            v, r, j, info = super().evaluate(vector, end)
            return v, r, j * 2, info

    class Transition(Profile):
        def evaluate(self, vector, end):
            v, r, j, _ = super().evaluate(vector, end)
            return v, r, j, {"active_mask": [int(vector[0] > 1)]}

    args = ({"a": 1, "initial": 0}, policy, monotonic() + 10, lambda _: None)
    assert audit(QuadraticOracle(), Wrong(), *args)["status"] == "failed"
    result = audit(QuadraticOracle(), Transition(), *args)
    assert result["status"] == "inconclusive"
    assert any(
        s["status"] == "active_set_transition"
        for d in result["directions"]
        for s in d["steps"]
    )
    bound = audit(
        QuadraticOracle(),
        Profile(),
        {"a": 0.01, "initial": 0},
        policy,
        monotonic() + 10,
        lambda _: None,
    )
    assert bound["status"] == "inconclusive"
    assert any(
        s["status"] == "outer_bound_limited"
        for d in bound["directions"]
        for s in d["steps"]
    )


@pytest.mark.parametrize("scaling", campaign.METHODS)
def test_scaling_search_accepted_steps_and_prediction(scaling):
    r = numerical.search(
        QuadraticOracle(),
        {"a": 1.0, "initial": 0.0},
        monotonic() + 10,
        lambda _: None,
        calls=40,
        target=1e-20,
        telemetry=True,
        record_steps=True,
        optimizer_scaling=scaling,
    )
    assert r["stop_reason"] == "numerical_target_reached"
    assert r["accepted_steps"]
    assert r["best"]["parameters"] == pytest.approx({"a": 2, "initial": 3})
    assert r["calls"] == r["completed_calls"] <= 40
    for step in r["accepted_steps"]:
        assert step["coordinate_step_norm"] > 0
        assert step["actual_reduction"] > 0
        assert step["reduction_ratio"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="scaling"):
        numerical.search(
            QuadraticOracle(),
            {},
            monotonic() + 1,
            lambda _: None,
            calls=3,
            target=1e-20,
            optimizer_scaling="unknown",
        )


@pytest.fixture
def bundle_path(warm_bundle, tmp_path, monkeypatch):
    inputs = read_seal(warm_bundle)
    policy = campaign.source.Policy().model_dump(mode="json")
    plan = {"policy": policy, "inputs_sha256": public.content_sha256(inputs)}
    monkeypatch.setattr(campaign, "SOURCE_PLAN", public.content_sha256(plan))
    parents = {}
    for name, problem in campaign.source.validate_inputs(inputs).items():
        shared = campaign.source.warm_record(
            inputs["parents"][name]["backend"], problem
        )
        point = checked(shared["parameters"], 0.1)
        backend = {
            "identity": {
                "problem_sha256": public.content_sha256(
                    problem.model_dump(mode="json")
                ),
                "method": "best_rollout",
                "continuation_order": "incumbent_first",
                "policy": policy,
                "shared_warm": shared["parameters"],
            },
            "after_incumbent_continuation": point
            | {"origin": "incumbent-continuation/check"},
            "operations": [
                {
                    "operation": "incumbent-continuation/rollout",
                    "status": "complete",
                    "identity": {"point": {"start": shared["parameters"]}},
                    "value": {"best": {"parameters": point["parameters"]}},
                },
                {
                    "operation": "incumbent-continuation/check",
                    "status": "complete",
                    "value": point,
                },
            ],
        }
        parents[name] = {
            "backend": backend,
            "backend_sha256": public.content_sha256(backend),
        }
    bundle = {
        "protocol": campaign.PROTOCOL,
        "source_plan": plan,
        "source_inputs": inputs,
        "parents": parents,
    }
    path = tmp_path / "m25-inputs.json"
    seal(path, bundle)
    return path


@pytest.fixture
def mock_fit(monkeypatch):
    seen = []
    monkeypatch.setattr(
        campaign.rollout, "build", lambda _: (QuadraticOracle(), Profile(), {})
    )
    monkeypatch.setattr(campaign, "audit", lambda *a: {"status": "passed"})

    def run(problem, policy, method, folder, **kw):
        assert set(problem.model_dump()) == {
            "request",
            "training",
            "coordinates",
            "start",
            "incumbent",
        }
        seen.append((kw, policy.continuation_calls))
        result = dict.fromkeys(
            (
                "search_calls_started",
                "search_calls_completed",
                "additional_wall_seconds",
                "additional_cpu_seconds",
                "additional_budget_charge_seconds",
            ),
            0,
        )
        return result | {
            "status": "complete",
            "assessment": {},
            "trajectory_assessment": {},
            "operations": [],
            "retained_parameters": kw["shared_warm"],
            "cost_complete": True,
            "search_call_accounting_complete": True,
        }

    monkeypatch.setattr(campaign.fitting, "run", run)

    def evaluate(folder, *args):
        b = read_seal(folder.parent / "backend.json")
        assert set(b["arms"]) == set(campaign.METHODS)
        return {"status": "complete", "training_nmse": 0.1}

    monkeypatch.setattr(campaign.screening_replay, "_evaluation", evaluate)
    return seen


def test_matched_scaling_gate_resume_and_retrospective_boundary(
    bundle_path, mock_fit, tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.prepare(root, bundle_path, campaign.Policy())
    assert campaign.report(root)["recorded"] == 0
    row = campaign.run_task(root, 0)
    assert row["status"] == "complete"
    assert len(mock_fit) == 2
    assert mock_fit[0][0]["shared_warm"] == mock_fit[1][0]["shared_warm"]
    assert mock_fit[0][1] == mock_fit[1][1] == 120
    assert [r[0]["optimizer_scaling"] for r in mock_fit] == list(campaign.METHODS)
    monkeypatch.setattr(
        campaign.fitting, "run", lambda *a, **kw: pytest.fail("repeated fit")
    )
    assert campaign.run_task(root, 0) == row
    assert campaign.report(root)["recorded"] == 1
    with pytest.raises(ValueError, match="index"):
        campaign.run_task(root, 3)


@pytest.mark.parametrize("status", ["failed", "inconclusive", "interrupted"])
def test_failed_or_interrupted_audit_never_launches_fitting(
    bundle_path, mock_fit, tmp_path, monkeypatch, status
):
    root = tmp_path / status
    campaign.prepare(root, bundle_path, campaign.Policy())
    plan, _ = campaign.verify(root)
    task = plan["tasks"][0]
    if status == "interrupted":
        seal(
            root / "results" / task["task_id"] / "derivatives/started.json",
            {
                "plan_sha256": public.content_sha256(plan),
                "task": task,
                "allowance_seconds": 1800.0,
            },
        )
        monkeypatch.setattr(
            campaign, "audit", lambda *a: pytest.fail("restarted audit")
        )
    else:
        monkeypatch.setattr(campaign, "audit", lambda *a: {"status": status})
    result = campaign.run_task(root, 0)
    assert result["status"] == "diagnostic_blocked" and not mock_fit
    if status == "interrupted":
        assert result["derivative_audit"]["budget_charge_seconds"] == 1800
        assert not result["cost_complete"]


def test_source_tamper_and_endpoint_substitution_rejected(bundle_path):
    bundle = read_seal(bundle_path)
    for kind in ("missing", "digest", "endpoint"):
        bad = deepcopy(bundle)
        name = next(iter(bad["parents"]))
        if kind == "missing":
            del bad["parents"][name]
        elif kind == "digest":
            bad["parents"][name]["backend_sha256"] = "wrong"
        else:
            p = bad["parents"][name]
            p["backend"]["after_incumbent_continuation"]["origin"] = "trial-2/check"
            p["backend_sha256"] = public.content_sha256(p["backend"])
        with pytest.raises(ValueError):
            campaign.validate_inputs(bad)


def test_refined_campaign_cannot_overwrite_coarse_results(bundle_path, tmp_path):
    original = tmp_path / "original"
    campaign.prepare(original, bundle_path, campaign.Policy())
    before = read_seal(original / "plan.json")
    with pytest.raises(ValueError, match="sealed artifact differs"):
        campaign.prepare(original, bundle_path, refined_policy())
    assert read_seal(original / "plan.json") == before
    fresh = tmp_path / "refined"
    campaign.prepare(fresh, bundle_path, refined_policy())
    after, _ = campaign.verify(fresh)
    assert before["inputs_sha256"] == after["inputs_sha256"]
    assert before["tasks"] == after["tasks"]
    old_policy, new_policy = deepcopy(before["policy"]), deepcopy(after["policy"])
    assert old_policy["audit"].pop("steps") != new_policy["audit"].pop("steps")
    assert old_policy == new_policy


def test_export_freezes_predeclared_points(bundle_path, tmp_path, monkeypatch):
    bundle = read_seal(bundle_path)
    root = tmp_path / "source"
    monkeypatch.setattr(
        campaign.source,
        "verify",
        lambda *a, **kw: (bundle["source_plan"], bundle["source_inputs"]),
    )
    for name, parent in bundle["parents"].items():
        task = {
            "common": name,
            "method": "incumbent_first",
            "task_id": name + "__incumbent_first",
        }
        folder = root / "results" / task["task_id"]
        seal(folder / "backend.json", parent["backend"])
        seal(
            folder / "result.json",
            {
                "identity": {"plan_sha256": campaign.SOURCE_PLAN, "task": task},
                "backend_sha256": parent["backend_sha256"],
                "evaluations": {"validation_nmse": "must not be imported"},
            },
        )
    output = tmp_path / "export.json"
    assert campaign.export_inputs(root, output)["endpoints"] == 3
    assert "must not be imported" not in output.read_text()
    assert len(campaign.validate_inputs(read_seal(output))) == 3
    monkeypatch.setattr(campaign, "SOURCE_PLAN", "wrong")
    with pytest.raises(ValueError, match="source plan"):
        campaign.export_inputs(root, tmp_path / "wrong.json")


def test_scaling_identity_and_unchanged_source_incumbent(monkeypatch, tmp_path):
    from tests.test_nonlinear_comparison import problem

    monkeypatch.setattr(
        campaign.fitting.rollout, "build", lambda _: (QuadraticOracle(), Profile(), {})
    )
    monkeypatch.setattr(
        campaign.fitting.rollout, "verify", lambda o, p, *a, **kw: checked(p, 0.1)
    )
    monkeypatch.setattr(
        numerical,
        "search",
        lambda *a, **kw: {
            "best": None,
            "calls": 2,
            "completed_calls": 2,
            "stop_reason": "numerical_failure",
        },
    )
    policy = campaign.Policy(continuation_calls=2)
    p = problem()
    kw = {"shared_warm": p.incumbent, "continuation_order": "incumbent_first"}
    result = campaign.fitting.run(
        p, policy, "best_rollout", tmp_path, optimizer_scaling="jacobian", **kw
    )
    assert result["retained_parameters"] == p.incumbent
    assert result["search_calls_started"] == 2
    with pytest.raises(ValueError, match="identity"):
        campaign.fitting.run(
            p,
            policy,
            "best_rollout",
            tmp_path,
            optimizer_scaling="fixed_coordinates",
            **kw,
        )


def test_cpu_submission_idempotence(bundle_path, tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from scripts import submit_phase_c_fitting_stagnation as submitter

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        submitter.scheduler.subprocess, "check_output", lambda *a, **kw: "fixture\n"
    )
    config = tmp_path / "policy.json"
    config.write_text(campaign.Policy().model_dump_json())
    calls = []

    def submit(argv, **kw):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", submit)
    args = {"account": "bibo-delta-cpu", "concurrency": 3, "inputs": bundle_path}
    root = tmp_path / "submission"
    result = submitter.submit(root, config, **args)
    assert result["array_tasks"] == 3 and result["gpus"] == 0
    assert "--array=0-2%3" in calls[1] and "--time=02:30:00" in calls[1]
    assert submitter.submit(root, config, **args) == result and len(calls) == 3
