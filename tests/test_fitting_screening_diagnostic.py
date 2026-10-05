"""M9 hard screening ceilings, assistance provenance and fixed/released NLPs."""

import sys
from copy import deepcopy
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from autoformalism.benchmarks.audited_release import seal
from autoformalism.fitting import bounded_screening as b
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_diagnostic as d
from autoformalism.fitting import transcription_campaign as c
from autoformalism.fitting import transcription_solver as solver
from tests.test_fitting_strategies import small_payload


@pytest.fixture(scope="module")
def inputs():
    return controls.make_inputs()


def payload(inputs):
    p = small_payload(inputs, "collocation")
    p["screening"] = {"method": "Radau", "point_seconds": 10}
    return p


def test_screen_failed_point_preserves_incumbent_and_does_not_reset(
    inputs, tmp_path, monkeypatch
):
    p = payload(inputs)
    best = {"parameters": p["start"], "training_nmse": 0.1}
    calls = []

    def timeout(request, folder, seconds):
        calls.append(seconds)
        assert (
            public._read(tmp_path / "screening.json")["attempts"][-1]["status"]
            == "started"
        )
        return {"wall_timeout": True, "elapsed_seconds": seconds}

    monkeypatch.setattr(b, "invoke_point", timeout)
    points = [{"parameters": p["start"]}] * 2
    found, count = b.screen(
        p, tmp_path, "ordinary", points, deadline=monotonic() + 60, maximum=2, best=best
    )
    assert found == best and count == 2 and calls == [10, 10]
    assert b.screen(
        p, tmp_path, "ordinary", points, deadline=monotonic() + 60, maximum=2, best=best
    ) == (best, 2)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="points"):
        b.screen(
            p, tmp_path, "ordinary", [], deadline=monotonic() + 60, maximum=2, best=best
        )


def test_interruption_and_distinct_phase_journals(inputs, tmp_path, monkeypatch):
    p = payload(inputs)
    points = [{"parameters": p["start"]}]
    monkeypatch.setattr(
        b, "invoke_point", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        b.screen(
            p,
            tmp_path,
            "ordinary",
            points,
            deadline=monotonic() + 60,
            maximum=2,
            best=None,
        )
    monkeypatch.setattr(b, "invoke_point", lambda *a: {"wall_timeout": True})
    b.screen(
        p, tmp_path, "ordinary", points, deadline=monotonic() + 60, maximum=2, best=None
    )
    b.screen(
        p,
        tmp_path,
        "after-nlp",
        points,
        deadline=monotonic() + 60,
        maximum=2,
        best=None,
    )
    attempts = public._read(tmp_path / "screening.json")["attempts"]
    assert [a["status"] for a in attempts] == ["interrupted", "timeout"]
    assert [a["phase"] for a in attempts] == ["ordinary", "after-nlp"]


def test_real_point_deadline_covers_startup(tmp_path):
    result = b.invoke_point({}, tmp_path, 0.001)
    assert result["wall_timeout"] and result["returncode"] is not None
    assert result["elapsed_seconds"] < 3


def test_score_requires_complete_rollout_and_training_only(
    inputs, tmp_path, monkeypatch
):
    p = payload(inputs) | {
        "parameters": controls.TRUTHS["linear"] | {"init_z_value": 0.4}
    }
    result = b.evaluate(p, tmp_path)
    assert result["status"] == "complete" and np.isfinite(result["training_nmse"])
    p["training"]["name"] = "val"
    with pytest.raises(ValueError):
        b.evaluate(p, tmp_path)
    p["training"]["name"] = "train"
    monkeypatch.setattr(b.SymbolicOracle, "__call__", lambda *a: np.zeros(21))
    assert b.evaluate(p, tmp_path)["status"] == "unavailable"


def test_fixed_parameters_reduce_variables_and_nodes_are_rollout_generated(
    inputs, tmp_path
):
    p = payload(inputs) | {"checkpoint_mode": "compact", "retain_inner_nodes": True}
    node_dir = tmp_path / "nodes"
    node_dir.mkdir()
    generated = d.node_worker(p, node_dir)
    p.update({k: generated[k] for k in ("nodes", "inner_nodes")})
    released = solver.build_problem(p, tmp_path / "released")
    fixed = solver.build_problem(p | {"fixed_parameters": True}, tmp_path / "fixed")
    assert released.opti.nx - fixed.opti.nx == len(p["start"])
    path = tmp_path / "solve"
    path.mkdir()
    result = solver.solve(p | {"fixed_parameters": True}, path)
    assert result["native_success"]
    detail = public._read(path / "final_checkpoint_diagnostics.json")
    assert detail["parameters"] == pytest.approx(p["start"])
    assert all("inner_nodes" in v for v in detail["trajectories"].values())
    wrong = deepcopy(p)
    wrong["inner_nodes"][next(iter(p["inner_nodes"]))] = [[0]]
    with pytest.raises(ValueError, match="inner node"):
        solver.build_problem(wrong, tmp_path / "wrong")


def test_assistance_eligibility_ignores_reference_and_validation():
    common = {"case": "linear", "seed": 0, "request": {"a": 1}}
    row = {
        "request_sha256": public.content_sha256(common["request"]),
        "training_nmse": 1e-10,
        "parameters": {"a": 2},
        "training_verified": True,
    }
    data = {
        "assisted_starts": {"linear_s0": row},
        "reference_parameters": {"a": 999},
        "validation_nmse": 999,
    }
    assert d.bind_start(common, data, d.DiagnosticPolicy())["eligible"]
    row["training_nmse"] = 0.1
    assert not d.bind_start(common, data, d.DiagnosticPolicy())["eligible"]
    assert d.bind_start(common, data, d.DiagnosticPolicy())["parameters"] is None
    row["request_sha256"] = "wrong"
    with pytest.raises(ValueError, match="request"):
        d.bind_start(common, data, d.DiagnosticPolicy())


def configured(inputs, tmp_path, monkeypatch):
    data = deepcopy(inputs)
    request = controls.request("linear", 0).model_dump(mode="json")
    data["assisted_starts"] = {
        "linear_s0": {
            "request_sha256": public.content_sha256(request),
            "training_nmse": 0.2,
            "parameters": None,
            "training_verified": False,
        }
    }
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(data))
    config = c.CampaignConfig(
        starts=1,
        include_cstr=False,
        cases=("linear",),
        numerical_diagnostic={},
        screening_diagnostic={},
    )
    c.prepare(tmp_path, config)
    return c.verify(tmp_path)


def test_campaign_pairs_and_ineligible_rows(inputs, tmp_path, monkeypatch):
    plan, data = configured(inputs, tmp_path, monkeypatch)
    assert plan["protocol"] == d.PROTOCOL and len(plan["tasks"]) == 8
    for i in (0, 2, 4):
        a, z = [c.worker_payload(plan, data, t) for t in plan["tasks"][i : i + 2]]
        assert a["screening"].pop("method") == "RK45"
        assert z["screening"].pop("method") == "Radau"
        assert a == z
        assert not {"assisted_start", "validation", "reference_parameters"} & set(a)
    seal(
        tmp_path / "qualification/result.json",
        {"passed": True, "plan_sha256": public.content_sha256(plan)},
    )
    monkeypatch.setattr(c, "run", lambda *a: pytest.fail("ineligible source launched"))
    assert c.run_task(tmp_path, 6)["status"] == "assisted_source_ineligible"
    assert c.run_task(tmp_path, 6)["status"] == "assisted_source_ineligible"
    assert c.report(tmp_path)["recorded"] == 1


def test_export_checks_training_payload_without_reading_results(
    inputs, tmp_path, monkeypatch
):
    plan, data = configured(inputs, tmp_path / "campaign", monkeypatch)
    plan["protocol"] = c.numerical.PROTOCOL
    plan["config"].pop("screening_diagnostic")
    plan["tasks"] = [
        {
            "task_id": "linear_s0_rollout_continue",
            "common": "linear_s0",
            "arm": "rollout_continue",
        }
    ]
    task = plan["tasks"][0]
    backend = {
        "worker_payload_sha256": public.content_sha256(
            c.worker_payload(plan, data, task)
        ),
        "selection": "best_complete_training_rollout_across_both_phases",
        "reference_values_used": False,
        "validation_used_for_fitting": False,
        "training_nmse": 1e-12,
        "parameters": {"a": 2},
        "total_seconds": 15,
        "actual_residual_calls": 8,
    }
    seal(tmp_path / "results" / task["task_id"] / "backend.json", backend)
    monkeypatch.setattr(c, "verify", lambda *a, **k: (plan, data))
    exported = d.export_starts(tmp_path, tmp_path / "export.json")
    assert exported["source_starts"] == 1
    out = public._read(tmp_path / "export.json")["value"]["assisted_starts"][
        "linear_s0"
    ]
    assert out["source_seconds"] == 15 and out["parameters"] == {"a": 2}
    backend["worker_payload_sha256"] = "wrong"
    (tmp_path / "results" / task["task_id"] / "backend.json").unlink()
    seal(tmp_path / "results" / task["task_id"] / "backend.json", backend)
    with pytest.raises(ValueError, match="payload"):
        d.export_starts(tmp_path, tmp_path / "other.json")


def test_scheduler_cpu_and_idempotent_receipts(inputs, tmp_path, monkeypatch):
    from scripts import submit_phase_c_fitting_strategies as submit

    plan, data = configured(inputs, tmp_path / "fixture", monkeypatch)
    monkeypatch.setattr(submit.q, "prepare", lambda *a: None)
    monkeypatch.setattr(submit.q, "verify", lambda *a: (plan, data))
    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(submit.subprocess, "check_output", lambda *a, **k: "revision\n")
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        assert "--partition=cpu" in argv and "--mem=16G" in argv
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(submit.subprocess, "run", sbatch)
    root = tmp_path / "submit"
    root.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}")
    result = submit.submit(root, config, account="test", concurrency=2)
    assert "--array=0-7%2" in calls[1]
    assert submit.submit(root, config, account="test", concurrency=2) == result
    assert len(calls) == 3


def test_frozen_ordinary_points_survive_generator_roundoff(
    inputs, tmp_path, monkeypatch
):
    plan, data = configured(inputs, tmp_path / "first", monkeypatch)
    common = deepcopy(plan["commons"]["linear_s0"])
    common.pop("assisted_start")
    data["cases"] = deepcopy(inputs["cases"])
    data["frozen_start_commons"] = {"linear_s0": common}
    monkeypatch.setattr(c.controls, "make_inputs", lambda: deepcopy(data))
    request = c.case_request

    def different_guess(*args):
        req = request(*args)
        return req.model_copy(
            update={
                "parameter_guesses": {
                    k: np.nextafter(v, np.inf) for k, v in req.parameter_guesses.items()
                }
            }
        )

    monkeypatch.setattr(c, "case_request", different_guess)
    c.prepare(tmp_path / "second", c.CampaignConfig.model_validate(plan["config"]))
    new, _ = c.verify(tmp_path / "second")
    actual = dict(new["commons"]["linear_s0"])
    actual.pop("assisted_start")
    assert actual == common
