"""ACES retests preserve bounded plans, backend identity and uncertain submissions."""

import io
import json
import subprocess
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import saved_initialization_aces as adapter
from scripts import submit_saved_initialization_aces as scheduler


@pytest.fixture
def setup(tmp_path, monkeypatch):
    plan = {
        "protocol": adapter.PROTOCOL,
        "execution_platform": adapter.PLATFORM,
        "artifact_sha256": "frozen-plan",
        "tasks": [{"task_id": f"case-{i}"} for i in range(3)],
        "settings": {
            "model": adapter.MODEL,
            "model_revision": adapter.REVISION,
            "maximum_requests": 10,
            "maximum_total_tokens": 262144,
        },
        "automatic_followup": False,
    }
    monkeypatch.setattr(adapter.campaign, "verify", lambda root: deepcopy(plan))
    root, image = tmp_path / "output", tmp_path / "image.sif"
    image.write_bytes(b"fixed test image")
    adapter.prepare(tmp_path / "inputs", root, image)
    return plan, root, image


@pytest.mark.parametrize(
    "mutation",
    ["platform", "model", "revision", "roster", "duplicate", "followup", "budget"],
)
def test_reject_expanded_or_incompatible_plan(setup, mutation):
    plan, root, _ = setup
    if mutation == "platform":
        plan["execution_platform"] = "jetstream-h100"
    elif mutation == "model":
        plan["settings"]["model"] = "other"
    elif mutation == "revision":
        plan["settings"]["model_revision"] = "other"
    elif mutation == "roster":
        plan["tasks"].append({"task_id": "case-3"})
    elif mutation == "duplicate":
        plan["tasks"][1] = plan["tasks"][0]
    elif mutation == "followup":
        plan["automatic_followup"] = True
    else:
        plan["settings"]["maximum_requests"] = 20
    with pytest.raises(ValueError, match="bounded ACES"):
        adapter.verify_inputs(root)


def test_backend_identity_is_immutable_and_old_caches_are_not_imported(setup, tmp_path):
    _, root, image = setup
    before = sealed_read(root / "execution.json")
    assert adapter.prepare(tmp_path, root, image) == before
    image.write_bytes(b"different image")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        adapter.prepare(tmp_path, root, image)
    foreign = tmp_path / "foreign"
    (foreign / "results").mkdir(parents=True)
    with pytest.raises(ValueError, match="lack ACES"):
        adapter.prepare(tmp_path, foreign, image)


def runtime_record(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps(
            {
                "visible_gpus": 1,
                "cuda_kernel_passed": True,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "vllm": "0.27.1",
                "driver": "fixed-driver",
            }
        )
    )
    return path


def models(monkeypatch, value=None):
    monkeypatch.setattr(
        adapter,
        "urlopen",
        lambda *args, **kwargs: io.StringIO(
            json.dumps({"data": [{"id": value or adapter.MODEL}]})
        ),
    )


def test_server_binding_requires_tested_h100_local_model_and_same_resume_runtime(
    setup, tmp_path, monkeypatch
):
    _, root, _ = setup
    path = runtime_record(tmp_path)
    models(monkeypatch)
    adapter.bind_server(root, path, "http://127.0.0.1:8123")
    adapter.bind_server(root, path, "http://127.0.0.1:9999")
    with pytest.raises(ValueError, match="loopback"):
        adapter.bind_server(root, path, "https://external.example")
    models(monkeypatch, "other-model")
    with pytest.raises(ValueError, match="model differs"):
        adapter.bind_server(root, path, "http://127.0.0.1:8123")
    models(monkeypatch)
    record = json.loads(path.read_text())
    record["driver"] = "new-driver"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frozen artifact differs"):
        adapter.bind_server(root, path, "http://127.0.0.1:8123")
    record["cuda_kernel_passed"] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="working H100"):
        adapter.bind_server(root, path, "http://127.0.0.1:8123")


def test_run_preserves_plan_caps_and_reports_on_deferral(setup, tmp_path, monkeypatch):
    plan, root, _ = setup
    path = runtime_record(tmp_path)
    models(monkeypatch)
    called = []

    def episode(directory, current, task, **kwargs):
        assert current == plan
        assert kwargs["can_start"]()
        called.append(task)
        if len(called) == 2:
            raise adapter.DeferredCall("stop safely")

    monkeypatch.setattr(adapter.campaign, "run_episode", episode)
    monkeypatch.setattr(adapter.campaign, "report", lambda *args: {"reported": True})
    assert adapter.run(root, "http://127.0.0.1:8000", path) == {"reported": True}
    assert called == plan["tasks"][:2]


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


def test_submission_is_once_only_and_resources_are_bounded(setup, monkeypatch):
    root = scheduler_setup(setup, monkeypatch)
    submitted = []

    def queue(directory, key, options, worker, stage, index):
        submitted.append(options)
        return "1234"

    monkeypatch.setattr(scheduler, "submit_job", queue)
    first = scheduler.submit(root, "focused-1")
    assert scheduler.submit(root, "focused-1") == first
    assert len(submitted) == 1
    assert first["maximum_generation_requests"] == 30
    assert "--gres=gpu:h100:1" in submitted[0]
    assert not any("--array" in option for option in submitted[0])


def test_uncertain_submission_is_not_retried(setup, monkeypatch):
    root = scheduler_setup(setup, monkeypatch)
    directory = root / "submissions" / "focused-1"
    directory.mkdir(parents=True)
    (directory / "propose.intent.json").write_text("{}")
    monkeypatch.setattr(
        scheduler, "submit_job", lambda *args: pytest.fail("must not resubmit")
    )
    with pytest.raises(ValueError, match="outcome needs inspection"):
        scheduler.submit(root, "focused-1")


def test_shell_launchers_parse():
    for name in adapter.ENTRYPOINTS:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(adapter.REPO / name)], check=True)


def test_inspection_includes_report_and_actual_group_counts(setup, monkeypatch):
    _, root, _ = setup
    groups = {"focused": {"assessed": 3}}
    (root / "REPORT.html").write_text("<p>Saved episode results</p>")
    monkeypatch.setattr(adapter.campaign, "report", lambda *args: {"groups": groups})
    result = adapter.inspect(root)
    assert result["groups"] == groups
    with tarfile.open(result["archive"]) as archive:
        assert "REPORT.html" in archive.getnames()
        assert archive.extractfile("REPORT.html").read() == (
            b"<p>Saved episode results</p>"
        )
