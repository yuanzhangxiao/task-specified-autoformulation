"""Saved-profile recovery selection, provenance, failed checks and exact resume."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_campaign as old
from autoformalism.fitting import profile_recovery as fit
from autoformalism.fitting import profile_recovery_campaign as campaign
from autoformalism.fitting import public_fitting as public


class Oracle:
    names = ("a", "initial")
    units = np.ones(2)
    lower, upper = np.array([-10.0, -10.0]), np.array([10.0, 10.0])

    def __init__(self, *args):
        self.audit = {"test": True}

    def vector(self, p):
        return np.array([p[n] for n in self.names])

    def parameters(self, x):
        return dict(zip(self.names, x.tolist(), strict=True))

    def evaluate(self, x, deadline, **kwargs):
        return x - [2.0, 3.0], np.eye(2)


def fake_problem():
    return SimpleNamespace(
        incumbent={"a": 1.0, "initial": 2.0},
        start={"a": 0.0, "initial": 0.0},
        anchor={"a": 1.0, "initial": 2.0},
        alternatives=[],
        profiles=[
            fit.SavedProfile(
                parameter="a",
                value=1.01,
                offset=0.01,
                scale=1.0,
                in_domain=True,
                candidate={"a": 1.01, "initial": 2.0},
            )
        ],
        model_dump=lambda **_: {"fixture": "profile-recovery"},
    )


def test_training_only_shortlist_cap_and_witness_skip():
    center = {
        "parameters": {"a": 1.0, "b": 1.0},
        "training_nmse": 0.0,
        "alternate_training_nmse": 0.0,
    }

    def point(name, loss):
        return {
            "parameter": name,
            "value": 1.1,
            "scale": 1.0,
            "in_domain": True,
            "checked": {
                "training_nmse": loss,
                "alternate_training_nmse": loss,
                "solver_loss_discrepancy": 0.0,
                "solver_prediction_discrepancy": 0.0,
            },
        }

    ps = [point("a", 1e-14), point("b", 1e-6), point("b", 1e-5)]
    assert fit.shortlist(ps, center, 1e-12, 4) == [1]
    assert fit.shortlist(ps, center, 1e-12, 0) == []


def test_bounded_controller_refines_and_reuses(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    problem, policy = fake_problem(), fit.RecoveryPolicy()
    result = fit.run(problem, policy, tmp_path)
    assert (
        result["status"] == "complete" and result["selected"]["training_nmse"] < 1e-18
    )
    assert result["profile_refit_indices"] == [0] and result["rescue_starts"] == 2
    assert result["profiles"][0]["refit"]["best"]["parameters"]["a"] == 1.01
    assert result["confidence"]["probability"] is None
    assert not result["confidence"]["local_exclusion_certified"]
    assert all(o["calls"] is not None for o in result["operations"])
    monkeypatch.setattr(
        fit, "AffineRollouts", lambda *_: pytest.fail("repeated fitting")
    )
    assert fit.run(problem, policy, tmp_path) == result
    with pytest.raises(ValueError, match="identity"):
        fit.run(problem, policy.model_copy(update={"rescue_calls": 500}), tmp_path)


def test_failed_verification_never_promotes_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    monkeypatch.setattr(
        fit,
        "verify_outputs",
        lambda *a: (_ for _ in ()).throw(TimeoutError("check timeout")),
    )
    result = fit.run(fake_problem(), fit.RecoveryPolicy(), tmp_path)
    assert result["status"] == "no_verified_training_rollout"
    assert result["selected"] is None and not result["profile_refit_indices"]
    assert (
        result["rescue_starts"] == 0 and result["confidence"]["level"] == "unavailable"
    )


def test_saved_fixed_coordinate_must_match(monkeypatch, tmp_path):
    monkeypatch.setattr(fit, "AffineRollouts", Oracle)
    problem = fake_problem()
    problem.profiles[0].candidate["a"] = 1.02
    with pytest.raises(ValueError, match="changed fixed coordinate"):
        fit.run(problem, fit.RecoveryPolicy(), tmp_path)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    from autoformalism.fitting import larger_coupled_inputs as controls

    root = tmp_path_factory.mktemp("profile-source")
    controls.export_inputs(root / "linear.json")
    data = read_seal(root / "linear.json")
    patch = pytest.MonkeyPatch()
    patch.setattr(old, "SOURCE_INPUTS", public.content_sha256(data))
    endpoints = [
        {"task_id": f"{k}_{arm}", "common": k, "arm": arm, "parameters": base["start"]}
        for k, base in data["commons"].items()
        for arm in ("rollout_only", "coupled_profiled_rollout")
    ]
    seal(
        root / "m19-inputs.json",
        {
            "protocol": old.INPUT_PROTOCOL,
            "data": data,
            "endpoints": endpoints,
            "test_data_opened": False,
        },
    )
    old.prepare(root / "m19", root / "m19-inputs.json", old.ConfidencePolicy())
    plan, inputs = old.verify(root / "m19")
    patch.setattr(campaign, "SOURCE_PLAN", public.content_sha256(plan))
    for i, task in enumerate(plan["tasks"]):
        folder = root / "m19/results" / task["task_id"]
        problem = old.training_problem(inputs, endpoints[i])
        point = {
            "parameter": "a0",
            "value": problem.incumbent["a0"] * 1.1,
            "offset": 0.1,
            "scale": problem.incumbent["a0"],
            "in_domain": True,
        }
        backend = {
            "identity": {
                "problem_sha256": public.content_sha256(
                    problem.model_dump(mode="json")
                ),
                "policy": plan["policy"],
            },
            "selected": {"parameters": problem.incumbent},
            "profiles": [point],
            "operations": [],
            "additional_wall_seconds": 1.0,
            "additional_cpu_seconds": 1.0,
            "rollout_calls_observed": 2,
            "cost_complete": True,
        }
        seal(folder / "backend.json", backend)
        seal(folder / "fit/finished.json", backend)
        grid = {"anchor": problem.incumbent, "points": [point]}
        seal(folder / "fit/profile-grid.json", grid)
        profile = folder / "fit/profiles/000"
        seal(
            profile / "started.json",
            {"parameters": {"anchor": problem.incumbent, "point": point}},
        )
        public._write(
            profile / "progress.json",
            {"best": {"parameters": problem.incumbent | {"a0": point["value"]}}},
        )
        seal(
            folder / "result.json",
            {
                "identity": {"plan_sha256": campaign.SOURCE_PLAN, "task": task},
                "backend_sha256": public.content_sha256(backend),
                "status": "complete",
                "evaluations": {"selected": {"status": "complete"}},
            },
        )
    campaign.export(root / "m19", root / "export.json")
    yield root / "export.json"
    patch.undo()


def test_export_roster_and_training_boundary(exported, tmp_path):
    data = read_seal(exported)
    assert len(data["endpoints"]) == 24
    p = data["endpoints"][0]["problem"]
    assert p["profiles"][0]["progress_sha256"]
    with pytest.raises(ValidationError):
        fit.RecoveryInput.model_validate(p | {"validation": {}})
    campaign.prepare(tmp_path, exported, fit.RecoveryPolicy())
    assert campaign.report(tmp_path)["recorded"] == 0
    bad = deepcopy(data)
    bad["endpoints"][0]["problem"]["training"]["rows"][0]["targets"]["y"][0] += 1
    with pytest.raises(ValueError, match="training problem differs"):
        campaign.validate_inputs(bad)


def test_retrospective_scoring_only_after_seal(exported, tmp_path, monkeypatch):
    campaign.prepare(tmp_path, exported, fit.RecoveryPolicy())

    def backend(problem, policy, folder):
        point = {"parameters": problem.incumbent}
        return {
            "status": "complete",
            "after_saved": point,
            "selected": point,
            "confidence": {"level": "search_incomplete"},
            "profiles": [],
            "profile_refit_indices": [],
            "rescue_starts": 0,
            "stage_costs": {},
            "additional_wall_seconds": 0.0,
            "additional_cpu_seconds": 0.0,
            "cost_complete": True,
            "rollout_calls_observed": 0,
        }

    monkeypatch.setattr(fit, "run", backend)

    def score(folder, *args):
        assert (folder.parent / "backend.json").is_file()
        return {"status": "complete"}

    monkeypatch.setattr(campaign.screening_replay, "_evaluation", score)
    result = campaign.run_task(tmp_path, 0)
    assert campaign.run_task(tmp_path, 0) == result
    assert campaign.report(tmp_path)["recorded"] == 1


def test_submission_is_idempotent_and_cpu_only(exported, tmp_path, monkeypatch):
    import sys

    from scripts import submit_phase_c_generic_recovery as scheduler
    from scripts import submit_phase_c_profile_recovery as submit

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    config = tmp_path / "policy.json"
    config.write_text(fit.RecoveryPolicy().model_dump_json())
    kwargs = {"account": "bibo-delta-cpu", "concurrency": 6, "inputs": exported}
    value = submit.submit(tmp_path / "run", config, **kwargs)
    assert submit.submit(tmp_path / "run", config, **kwargs) == value
    assert len(calls) == 3 and value["gpus"] == 0 and value["array_tasks"] == 24
    assert "--time=01:00:00" in calls[1] and "--array=0-23%6" in calls[1]
    assert "--dependency=afterok:101" in calls[1]


def test_launch_scripts_parse():
    import subprocess

    for kind in ("submit", "run", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_profile_recovery_delta.sh"],
            check=True,
        )
