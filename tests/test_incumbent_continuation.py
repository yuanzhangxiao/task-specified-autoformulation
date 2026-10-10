"""M24 shared warm identity, bounded continuation, retention and real telemetry."""

from copy import deepcopy
from time import monotonic

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import incumbent_campaign as campaign
from autoformalism.fitting import nonlinear_comparison as fit
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as numerical
from tests.test_nonlinear_comparison import checked, problem
from tests.test_nonlinear_comparison import exported as exported_fixture
from tests.test_nonlinear_comparison import mocked as mocked_fixture
from tests.test_recovery_numerics import QuadraticOracle

# Register the established fixtures without overriding their names.
exported = exported_fixture
mocked = mocked_fixture


@pytest.mark.usefixtures("mocked")
def test_matched_shared_vector_and_restart_starts_different_continuation(
    monkeypatch, tmp_path
):
    history = []

    def search(o, start, end, save, **kw):
        history.append((start, kw["calls"], kw["telemetry"]))
        save({"calls": 1})
        return {
            "best": {"parameters": start, "training_nmse": start["a"]},
            "calls": 1,
            "completed_calls": 1,
            "stop_reason": "budget_limited",
        }

    monkeypatch.setattr(fit.numerical, "search", search)
    starts = [{"a": float(i), "initial": 0.0} for i in (3, 2, 4)]
    monkeypatch.setattr(fit.numerical, "diverse_starts", lambda *a: starts)
    shared = {"a": 0.1, "initial": 0.0}
    results = [
        fit.run(
            problem(),
            campaign.Policy(),
            "best_rollout",
            tmp_path / m,
            shared_warm=shared,
            continuation_order=m,
        )
        for m in campaign.METHODS
    ]
    assert history[0] == (shared, 60, True)
    assert [h[0] for h in history[1:4]] == [h[0] for h in history[4:7]] == starts
    assert history[7] == (
        starts[1],
        60,
        True,
    )  # M23 comparator intentionally excludes incumbent.
    assert all(
        r["retained_parameters"] == shared and r["trial_count"] == 3 for r in results
    )
    assert all(
        r["search_calls_started"] == r["search_calls_completed"] == 4 for r in results
    )
    assert all(
        not any(o["operation"] == "warm/rollout" for o in r["operations"])
        for r in results
    )
    for method, result in zip(campaign.METHODS, results, strict=True):
        assert (
            fit.run(
                problem(),
                campaign.Policy(),
                "best_rollout",
                tmp_path / method,
                shared_warm=shared,
                continuation_order=method,
            )
            == result
        )
    with pytest.raises(ValueError, match="identity differs"):
        fit.run(
            problem(),
            campaign.Policy(),
            "best_rollout",
            tmp_path / campaign.METHODS[0],
            shared_warm=shared | {"a": 0.2},
            continuation_order=campaign.METHODS[0],
        )


@pytest.mark.usefixtures("mocked")
def test_incumbent_success_early_stop_and_interrupted_allowance_not_reset(
    monkeypatch, tmp_path
):
    def search(o, start, end, save, **kw):
        save({"calls": 2})
        return {
            "best": {"parameters": {"a": 1e-15, "initial": 0.0}},
            "calls": 2,
            "completed_calls": 2,
            "stop_reason": "numerical_target_reached",
        }

    monkeypatch.setattr(fit.numerical, "search", search)
    warm = {"a": 0.1, "initial": 0.0}
    args = (problem(), campaign.Policy(), "best_rollout")
    kw = {"shared_warm": warm, "continuation_order": "incumbent_first"}
    result = fit.run(*args, tmp_path / "success", **kw)
    assert result["incumbent_continuation_run"] and result["trial_count"] == 0
    assert result["selected"]["training_nmse"] == 1e-15
    interrupted = tmp_path / "interrupted"
    saved = read_seal(tmp_path / "success/incumbent-continuation/rollout/started.json")
    seal(interrupted / "incumbent-continuation/rollout/started.json", saved)
    monkeypatch.setattr(fit.numerical, "diverse_starts", lambda *a: [])
    monkeypatch.setattr(
        fit.numerical, "search", lambda *a, **kw: pytest.fail("repeated search")
    )
    result = fit.run(*args, interrupted, **kw)
    assert result["retained_parameters"] == warm
    assert not result["cost_complete"] and not result["search_call_accounting_complete"]
    assert result["search_calls_started"] is None
    assert result["search_calls_completed"] is None
    op = next(o for o in result["operations"] if o["operation"].endswith("/rollout"))
    assert op["status"] == "interrupted" and op["budget_charge_seconds"] == 1200


@pytest.mark.usefixtures("mocked")
def test_failed_shared_recheck_preserves_imported_vector(monkeypatch, tmp_path):
    def fail(*a, **kw):
        raise TimeoutError("verification unavailable on this worker")

    monkeypatch.setattr(fit.rollout, "verify", fail)
    monkeypatch.setattr(fit.numerical, "search", lambda *a, **kw: pytest.fail("search"))
    warm = {"a": 0.1, "initial": 0.0}
    for method in campaign.METHODS:
        result = fit.run(
            problem(),
            campaign.Policy(),
            "best_rollout",
            tmp_path / method,
            shared_warm=warm,
            continuation_order=method,
        )
        assert result["retained_parameters"] == warm and result["selected"] is None
        assert result["status"] == "retained_unverified" and result["trial_count"] == 0
        assert result["search_calls_started"] == result["search_calls_completed"] == 0


def test_real_telemetry_completed_attempts_projected_bounds_and_cache(monkeypatch):
    oracle, logs = QuadraticOracle(), []
    result = numerical.search(
        oracle,
        {"a": 1.0, "initial": 0.0},
        monotonic() + 10,
        logs.append,
        calls=100,
        target=1e-20,
        telemetry=True,
    )
    assert result["completed_calls"] == result["calls"] == len(result["evaluations"])
    assert logs[-1]["completed_calls"] == result["calls"]
    assert result["evaluations"][-1]["best_training_nmse"] <= 1e-20

    def interrupted(v, end):
        raise TimeoutError("integration deadline")

    monkeypatch.setattr(oracle, "evaluate", interrupted)
    r = numerical.search(
        oracle,
        {"a": 1.0, "initial": 0.0},
        monotonic() + 10,
        lambda _: None,
        calls=2,
        target=1e-20,
        telemetry=True,
    )
    assert r["calls"] == 1 and r["completed_calls"] == 0
    monkeypatch.setattr(oracle, "evaluate", lambda v, end: (v - [-1, 3], np.eye(2)))
    r = numerical.search(
        oracle,
        {"a": 0.01, "initial": 3.0},
        monotonic() + 10,
        lambda _: None,
        calls=2,
        target=1e-20,
        telemetry=True,
    )
    assert r["evaluations"][0]["active_outer_bounds"] == {"a": "lower"}
    assert r["evaluations"][0]["projected_gradient_inf"] < 1e-8


@pytest.fixture
def bundle_path(exported, monkeypatch, tmp_path):
    data = read_seal(exported)
    policy = fit.ComparisonPolicy(allocation="measured").model_dump(mode="json")
    plan = {"inputs_sha256": public.content_sha256(data), "policy": policy}
    monkeypatch.setattr(campaign, "SOURCE_PLAN", public.content_sha256(plan))
    parents = {}
    for name, p in campaign.source.bases(data).items():
        parameters = p.incumbent
        point = checked(parameters, 0.1)
        backend = {
            "identity": {
                "problem_sha256": public.content_sha256(p.model_dump(mode="json")),
                "method": "best_rollout",
                "policy": policy,
            },
            "after_warm": point | {"origin": "warm/check"},
            "setup": [{"wall_seconds": 1, "cpu_seconds": 1}],
            "operations": [
                {
                    "operation": "warm/rollout",
                    "status": "complete",
                    "wall_seconds": 2,
                    "cpu_seconds": 2,
                    "identity": {"point": {"start": parameters}},
                    "value": {"best": {"parameters": parameters}},
                },
                {
                    "operation": "warm/check",
                    "status": "complete",
                    "value": point,
                    "wall_seconds": 3,
                    "cpu_seconds": 3,
                },
            ],
        }
        parents[name] = {
            "backend": backend,
            "backend_sha256": public.content_sha256(backend),
        }
    path = tmp_path / "bundle.json"
    seal(
        path,
        {
            "protocol": campaign.PROTOCOL,
            "source_plan": plan,
            "data": data,
            "parents": parents,
        },
    )
    return path


def test_sealed_six_task_handoff_training_only_and_postfit_evaluator(
    bundle_path, tmp_path, monkeypatch
):
    root = tmp_path / "campaign"
    campaign.prepare(root, bundle_path, campaign.Policy())
    assert len(campaign.verify(root)[0]["tasks"]) == 6
    assert campaign.report(root)["recorded"] == 0
    seen = []

    def run(p, policy, method, folder, **kw):
        assert set(p.model_dump()) == {
            "request",
            "training",
            "coordinates",
            "start",
            "incumbent",
        }
        seen.append(kw["shared_warm"])
        point = checked(kw["shared_warm"], 0.1)
        result = dict.fromkeys(
            (
                "trial_count",
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
            "after_warm": point,
            "selected": point,
            "assessment": {},
            "trajectory_assessment": {},
            "routing": {},
            "operations": [],
            "continuation_run": False,
            "incumbent_continuation_run": True,
            "cost_complete": True,
            "search_call_accounting_complete": True,
        }

    def evaluate(folder, *args):
        assert (folder.parent / "backend.json").exists()
        return {"training_nmse": 0.1, "validation_nmse": 0.2}

    monkeypatch.setattr(campaign.fitting, "run", run)
    monkeypatch.setattr(campaign.screening_replay, "_evaluation", evaluate)
    a, b = campaign.run_task(root, 0), campaign.run_task(root, 1)
    assert (
        seen[0] == seen[1]
        and a["shared_parameters_sha256"] == b["shared_parameters_sha256"]
    )
    monkeypatch.setattr(
        campaign.fitting, "run", lambda *a, **kw: pytest.fail("repeated fit")
    )
    assert campaign.run_task(root, 0) == a
    report = campaign.report(root)
    assert report["recorded"] == 2 and report["status"] == "incomplete"
    assert all(
        c["wall_seconds"] == 6
        for c in report["shared_warm_cost_once_per_seed"].values()
    )
    with pytest.raises(ValueError, match="task index"):
        campaign.run_task(root, 6)


def test_tampered_or_missing_source_rejected(bundle_path):
    bundle = read_seal(bundle_path)
    for change, message in [
        ("remove", "all original"),
        ("digest", "digest"),
        ("origin", "checkpoint"),
    ]:
        bad = deepcopy(bundle)
        name = next(iter(bad["parents"]))
        if change == "remove":
            del bad["parents"][name]
        elif change == "digest":
            bad["parents"][name]["backend_sha256"] = "wrong"
        else:
            parent = bad["parents"][name]
            parent["backend"]["after_warm"]["origin"] = "continuation/check"
            parent["backend_sha256"] = public.content_sha256(parent["backend"])
        with pytest.raises(ValueError, match=message):
            campaign.validate_inputs(bad)


def test_export_preserves_all_warms_and_rejects_source_identity(
    bundle_path, tmp_path, monkeypatch
):
    bundle = read_seal(bundle_path)
    source = tmp_path / "source"
    monkeypatch.setattr(
        campaign.source,
        "verify",
        lambda *a, **kw: (bundle["source_plan"], bundle["data"]),
    )
    for name, parent in bundle["parents"].items():
        task = {
            "task_id": name + "__best_rollout",
            "common": name,
            "method": "best_rollout",
        }
        folder = source / "results" / task["task_id"]
        seal(folder / "backend.json", parent["backend"])
        seal(
            folder / "result.json",
            {
                "backend_sha256": parent["backend_sha256"],
                "identity": {"plan_sha256": campaign.SOURCE_PLAN, "task": task},
            },
        )
    output = tmp_path / "export.json"
    assert campaign.export_inputs(source, output)["shared_warm_fits"] == 3
    assert len(campaign.validate_inputs(read_seal(output))) == 3
    monkeypatch.setattr(campaign, "SOURCE_PLAN", "wrong")
    with pytest.raises(ValueError, match="source plan"):
        campaign.export_inputs(source, tmp_path / "wrong.json")


def test_submission_cpu_array_idempotence_and_uncertain_reply(
    bundle_path, tmp_path, monkeypatch
):
    import subprocess
    import sys
    from types import SimpleNamespace

    from scripts import submit_phase_c_incumbent_continuation as submitter

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
    root = tmp_path / "submission"
    args = {"account": "bibo-delta-cpu", "concurrency": 3, "inputs": bundle_path}
    result = submitter.submit(root, config, **args)
    assert result["array_tasks"] == 6 and result["gpus"] == 0
    assert "--array=0-5%3" in calls[1] and "--time=01:30:00" in calls[1]
    assert "--dependency=afterok:101" in calls[1]
    assert submitter.submit(root, config, **args) == result and len(calls) == 3

    def timeout(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 45)

    monkeypatch.setattr(submitter.scheduler.subprocess, "run", timeout)
    other = tmp_path / "uncertain"
    with pytest.raises(ValueError, match="outcome unknown"):
        submitter.submit(other, config, **args)
    monkeypatch.setattr(
        submitter.scheduler.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("duplicate submission"),
    )
    with pytest.raises(ValueError, match="Unconfirmed prepare"):
        submitter.submit(other, config, **args)
