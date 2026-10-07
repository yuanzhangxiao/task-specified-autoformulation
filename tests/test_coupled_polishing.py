"""M17 repeats stay separate, polishing shares budgets and evaluation is post-fit."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import coupled_campaign as coupled
from autoformalism.fitting import coupled_polishing as campaign
from autoformalism.fitting import generic_recovery as common
from autoformalism.fitting import polishing_fit
from autoformalism.fitting import public_fitting as public
from tests.test_polishing_fit import backend


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = tmp_path_factory.mktemp("coupled-polishing") / "source.json"
    coupled.export_inputs(path)
    return path


@pytest.fixture
def source(generated, monkeypatch):
    monkeypatch.setattr(
        campaign, "SOURCE_INPUT_SHA256", public.content_sha256(read_seal(generated))
    )
    return generated


def policy():
    return campaign.PolishingPolicy.model_validate_json(
        Path("configs/phase_c_coupled_polishing_v1.json").read_text()
    )


def test_exact_inputs_roster_and_original_denominators(source, tmp_path):
    root = tmp_path / "campaign"
    assert campaign.prepare(root, source, policy())["tasks"] == 14
    plan, data = campaign.verify(root)
    repeated = [t for t in plan["tasks"] if t["arm"].startswith("repeat")]
    assert len(repeated) == 2
    assert {t["common"] for t in repeated} == {"coupled_fast_slow_s2"}
    summary = campaign.report(root)
    assert summary["expected"] == 14 and summary["recorded"] == 0
    assert [summary["groups"][a]["expected"] for a in campaign.ARMS] == [1, 1, 6, 6]
    assert summary["original_m16"]["original_unavailable"] == 2
    assert not summary["original_m16"]["repeats_replace_original_results"]
    changed = deepcopy(data)
    changed["config"]["initial_absolute"] = 0.01
    seal(tmp_path / "changed.json", changed)
    with pytest.raises(ValueError, match="exact M16"):
        campaign.prepare(tmp_path / "bad", tmp_path / "changed.json", policy())
    before = coupled.bases(data)
    for case in data["cases"].values():
        case["validation"] = {"poison": True}
        case["reference_parameters"] = {"poison": True}
    assert before == coupled.bases(data)


@pytest.mark.parametrize("arm", campaign.ARMS)
def test_budgets_modes_and_no_automatic_retry(source, tmp_path, monkeypatch, arm):
    base = coupled.bases(read_seal(source))["coupled_fast_slow_s2"]
    now, calls = [0.0], []

    def fit(b, method, settings, folder, *, rollout_mode, diagnostic_timing):
        calls.append(deepcopy((b, method, settings)))
        assert diagnostic_timing
        assert rollout_mode == (
            "recovery_rollout" if arm.endswith("joint") else "coupled_profiled_rollout"
        )
        assert set(b) == {"request", "training", "coordinates", "nodes", "start"}
        now[0] += 30
        if len(calls) == 1:
            return backend(b["start"], error=8e-9, calls=9)
        assert settings["seconds"] == 270 and settings["maximum_rollout_calls"] == 291
        assert settings["training_nmse"] == 1e-11
        assert settings["trajectory_nmse"] == 1e-10
        return backend(b["start"], error=1e-12, calls=3)

    monkeypatch.setattr(polishing_fit, "monotonic", lambda: now[0])
    monkeypatch.setattr(campaign.recovery_fit, "fit", fit)
    result = campaign.fit(base, arm, policy().model_dump(mode="json"), tmp_path)
    assert len(calls) == (1 if arm.startswith("repeat") else 2)
    if arm.startswith("polish"):
        assert result["actual_residual_calls"] == 12 and result["fit_seconds"] == 60
        assert result["strict_prediction_certified"]


def test_post_fit_comparison_and_resume(source, tmp_path, monkeypatch):
    campaign.prepare(tmp_path, source, policy())
    plan, _ = campaign.verify(tmp_path)
    calls = []

    def fit(base, arm, settings, folder):
        first = base["start"]
        final = {key: value + 0.01 for key, value in first.items()}
        return backend(final) | {
            "first_prediction": {"selected": {"parameters": first}}
        }

    def evaluate(folder, data, entry, parameters, seconds):
        assert (folder.parent / "backend.json").exists()
        assert data["case_name"] == entry["case"]
        if folder.name == "first-evaluation":
            assert (folder.parent / "result.json").exists()
        calls.append(folder.name)
        return {"status": "complete", "accuracy_passed": True}

    monkeypatch.setattr(campaign, "fit", fit)
    monkeypatch.setattr(common.replay, "_evaluation", evaluate)
    for i in range(14):
        row = campaign.run_task(tmp_path, i)
        assert campaign.run_task(tmp_path, i) == row
    assert len(calls) == 26  # 14 final scores + 12 first vectors, no retry scores.
    summary = campaign.report(tmp_path)
    assert summary["status"] == "complete" and summary["expected"] == 14
    assert summary["original_m16"]["denominator_per_method"] == 6
    index = next(
        i for i, task in enumerate(plan["tasks"]) if task["arm"] == "polish_joint"
    )
    folder = tmp_path / "results" / plan["tasks"][index]["task_id"]
    (folder / "comparison.json").unlink()
    assert campaign.report(tmp_path)["status"] == "incomplete"


def test_timing_missing_and_tamper_detection(tmp_path):
    operation = tmp_path / "fit/operation"
    seal(operation / "started.json", {})
    assert campaign.timing_records(tmp_path)[0]["timing"] is None
    data = {"entry_monotonic": 12}
    public._write(operation / "timing.json", data)
    public._write(operation / "launch.json", {"monotonic": 10})
    seal(
        operation / "process.json",
        {
            "status": "complete",
            "termination_confirmed": True,
            "artifacts_sha256": {"timing.json": public.content_sha256(data)},
        },
    )
    assert campaign.timing_records(tmp_path)[0]["launch_to_entry_seconds"] == 2
    public._write(operation / "timing.json", {})
    with pytest.raises(ValueError, match="timing artifact"):
        campaign.timing_records(tmp_path)


def test_cpu_submission_idempotent(source, tmp_path, monkeypatch):
    from scripts import submit_phase_c_coupled_polishing as submit
    from scripts import submit_phase_c_generic_recovery as scheduler

    monkeypatch.setenv("AF_PYTHON", sys.executable)
    monkeypatch.setattr(
        scheduler.subprocess, "check_output", lambda *a, **k: "revision\n"
    )
    calls = []

    def sbatch(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(100 + len(calls)), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", sbatch)
    kwargs = {"account": "test", "concurrency": 6, "inputs": source}
    config = Path("configs/phase_c_coupled_polishing_v1.json")
    result = submit.submit(tmp_path, config, **kwargs)
    assert "--array=0-13%6" in calls[1] and "--partition=cpu" in calls[1]
    assert "--cpus-per-task=1" in calls[1]
    assert submit.submit(tmp_path, config, **kwargs) == result and len(calls) == 3


def test_shells():
    for kind in ("run", "submit", "inspect"):
        subprocess.run(
            ["bash", "-n", f"scripts/hpc/{kind}_phase_c_coupled_polishing_delta.sh"],
            check=True,
        )
