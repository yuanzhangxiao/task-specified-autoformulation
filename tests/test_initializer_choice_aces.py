"""The separate ACES diagnostic cannot reuse old caches or hide unavailable output."""

import subprocess
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import initializer_choice_aces as adapter
from scripts import submit_initializer_choice_aces as scheduler


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adapter.harmony_vocab,
        "verify",
        lambda path: {"protocol": "fixture-harmony", "sha256": "fixed-vocabulary"},
    )
    plan = {
        "protocol": adapter.PROTOCOL,
        "execution_platform": adapter.PLATFORM,
        "artifact_sha256": "new-diagnostic-plan",
        "cases": {"seed1": {}, "seed2": {}},
        "parent_plan": {
            "settings": {
                "model": adapter.MODEL,
                "model_revision": adapter.REVISION,
                "max_output_tokens": 8192,
            }
        },
        "budgets": deepcopy(adapter.BUDGETS),
        "automatic_followup": False,
        "test_data_opened": False,
    }
    monkeypatch.setattr(adapter.inputs, "verify", lambda root: deepcopy(plan))
    root, image = tmp_path / "output", tmp_path / "image.sif"
    image.write_bytes(b"fixed image")
    adapter.prepare(tmp_path / "inputs", root, image)
    return plan, root, image


@pytest.mark.parametrize(
    "mutation",
    [
        "old_protocol",
        "platform",
        "model",
        "revision",
        "output_budget",
        "calls_budget",
        "token_budget",
        "cases",
        "followup",
        "test",
    ],
)
def test_incompatible_or_expanded_diagnostic_is_rejected(setup, mutation):
    plan, root, _ = setup
    if mutation == "old_protocol":
        plan["protocol"] = "saved-topology-initialization-6"
    elif mutation == "platform":
        plan["execution_platform"] = "jetstream-h100"
    elif mutation in {"model", "revision", "output_budget"}:
        key = {
            "model": "model",
            "revision": "model_revision",
            "output_budget": "max_output_tokens",
        }[mutation]
        plan["parent_plan"]["settings"][key] = "changed"
    elif mutation in {"calls_budget", "token_budget"}:
        key = "maximum_calls" if mutation == "calls_budget" else "maximum_tokens"
        plan["budgets"][key] += 1
    elif mutation == "cases":
        plan["cases"]["third"] = {}
    elif mutation == "followup":
        plan["automatic_followup"] = True
    else:
        plan["test_data_opened"] = True
    with pytest.raises(ValueError, match="bounded ACES"):
        adapter.verify_inputs(root)


def test_execution_is_frozen_and_old_results_are_not_imported(setup, tmp_path):
    _, root, image = setup
    old = sealed_read(root / "execution.json")
    assert adapter.prepare(tmp_path, root, image) == old
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        adapter.prepare(tmp_path, root, image)
    foreign = tmp_path / "historical"
    (foreign / "calls").mkdir(parents=True)
    with pytest.raises(ValueError, match="lack diagnostic execution"):
        adapter.prepare(tmp_path, foreign, image)


def test_run_delegates_all_units_once_without_a_provider_failure_shortcut(
    setup, tmp_path, monkeypatch
):
    plan, root, _ = setup
    bound, called = [], []
    summary = {
        "status": "complete",
        "availability_status": "unavailable",
        "status_counts": {"provider_unavailable": 1, "assessed": 3},
    }
    monkeypatch.setattr(adapter, "bind_server", lambda *args: bound.append(args))

    def run(directory, current, *, base_url, can_start):
        assert current == plan
        assert can_start()
        called.append((directory, base_url))
        return summary

    monkeypatch.setattr(adapter.diagnostic, "run", run)
    monkeypatch.setattr(adapter.diagnostic, "report", lambda *args: summary)
    assert (
        adapter.run(root, "http://127.0.0.1:8000", tmp_path / "runtime.json") == summary
    )
    assert len(bound) == len(called) == 1
    assert adapter.run_exit_code(summary) == 1


def test_run_reports_after_error_without_resetting_the_plan(
    setup, tmp_path, monkeypatch
):
    plan, root, _ = setup
    monkeypatch.setattr(adapter, "bind_server", lambda *args: None)
    reports = []

    def fail(*args, **kwargs):
        raise RuntimeError("transport interrupted")

    monkeypatch.setattr(adapter.diagnostic, "run", fail)
    monkeypatch.setattr(
        adapter.diagnostic, "report", lambda *args: reports.append(args)
    )
    with pytest.raises(RuntimeError, match="transport interrupted"):
        adapter.run(root, "http://127.0.0.1:8000", tmp_path / "runtime.json")
    assert reports == [(root, plan)]


@pytest.mark.parametrize(
    "scientific",
    [
        "repair_required",
        "topology_change_needed",
        "insufficient_information",
        "checks_clear",
    ],
)
def test_scientific_outcomes_do_not_cause_operational_failure(scientific):
    assert (
        adapter.run_exit_code(
            {
                "status": "complete",
                "availability_status": "available",
                "scientific_status": scientific,
            }
        )
        == 0
    )
    assert (
        adapter.run_exit_code({"status": "pending", "availability_status": "pending"})
        == 1
    )
    assert (
        adapter.run_exit_code(
            {
                "status": "complete",
                "availability_status": "available",
                "status_counts": {"provider_unavailable": 1},
            }
        )
        == 1
    )


def scheduler_setup(setup, monkeypatch):
    _, root, image = setup
    for key, value in {
        "AF_PYTHON": "/python",
        "AF_REPO_ROOT": str(adapter.REPO),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": "/cache",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(scheduler, "check_storage", lambda *args, **kwargs: {})
    return root


def test_submission_is_idempotent_and_keeps_separate_six_call_limit(setup, monkeypatch):
    root = scheduler_setup(setup, monkeypatch)
    submitted = []

    def queue(directory, key, options, worker, stage, index):
        submitted.append((options, worker))
        return "1234"

    monkeypatch.setattr(scheduler, "submit_job", queue)
    value = scheduler.submit(root, "choice-1")
    assert scheduler.submit(root, "choice-1") == value
    assert len(submitted) == 1
    assert value["maximum_generation_requests"] == 6
    assert value["maximum_charged_tokens"] == 196608
    assert value["maximum_cases"] == 2
    assert "--gres=gpu:h100:1" in submitted[0][0]
    assert "--time=01:30:00" in submitted[0][0]
    assert submitted[0][1].name == "run_initializer_choice_aces.sh"


def test_uncertain_submission_does_not_retry(setup, monkeypatch):
    root = scheduler_setup(setup, monkeypatch)
    directory = root / "submissions" / "choice-1"
    directory.mkdir(parents=True)
    (directory / "propose.intent.json").write_text("{}")
    monkeypatch.setattr(scheduler, "submit_job", lambda *args: pytest.fail("no retry"))
    with pytest.raises(ValueError, match="outcome needs inspection"):
        scheduler.submit(root, "choice-1")


def test_inspection_includes_probe_calls_and_preserves_unavailable_status(
    setup, monkeypatch
):
    _, root, _ = setup
    (root / "calls" / "probe0").mkdir(parents=True)
    (root / "calls" / "probe0" / "response.json").write_text('{"status":"unavailable"}')
    (root / "results").mkdir()
    (root / "results" / "choice.json").write_text("{}")
    summary = {
        "status": "complete",
        "availability_status": "unavailable",
        "status_counts": {"provider_unavailable": 1},
    }
    monkeypatch.setattr(adapter.diagnostic, "report", lambda *args: summary)
    result = adapter.inspect(root)
    assert result["availability_status"] == "unavailable"
    with tarfile.open(result["archive"]) as archive:
        names = archive.getnames()
    assert "calls/probe0/response.json" in names
    assert "results/choice.json" in names
    assert "execution.json" in names


def test_shell_launchers_parse():
    for name in adapter.ENTRYPOINTS:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(adapter.REPO / name)], check=True)
