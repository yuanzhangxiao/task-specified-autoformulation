"""ACES retests preserve bounded plans, backend identity and uncertain submissions."""

import io
import json
import os
import subprocess
import tarfile
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import saved_initialization_aces as adapter
from scripts import submit_saved_initialization_aces as scheduler


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adapter.harmony_vocab,
        "verify",
        lambda path: {
            "protocol": adapter.harmony_vocab.PROTOCOL,
            "filename": adapter.harmony_vocab.VOCAB_FILENAME,
            "sha256": adapter.harmony_vocab.VOCAB_SHA256,
        },
    )
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
                "harmony": {
                    "protocol": adapter.harmony_vocab.PROTOCOL,
                    "filename": adapter.harmony_vocab.VOCAB_FILENAME,
                    "sha256": adapter.harmony_vocab.VOCAB_SHA256,
                    "openai_harmony": "fixed-version",
                    "rendered_tokens": 8,
                    "harmony_render_passed": True,
                    "generation_requests": 0,
                },
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("sha256", "wrong"),
        ("rendered_tokens", 0),
        ("harmony_render_passed", False),
        ("generation_requests", 1),
    ],
)
def test_missing_or_invalid_harmony_receipt_blocks_generation(
    setup, tmp_path, monkeypatch, field, value
):
    _, root, _ = setup
    path = runtime_record(tmp_path)
    record = json.loads(path.read_text())
    record["harmony"][field] = value
    path.write_text(json.dumps(record))
    monkeypatch.setattr(adapter, "urlopen", lambda *a, **k: pytest.fail("no HTTP"))
    with pytest.raises(ValueError, match="Harmony render preflight"):
        adapter.bind_server(root, path, "http://127.0.0.1:8000")


def test_asset_verification_failure_precedes_plan_write(setup, tmp_path, monkeypatch):
    _, _, image = setup

    def missing(path):
        raise ValueError("Missing bundled Harmony vocabulary")

    monkeypatch.setattr(adapter.harmony_vocab, "verify", missing)
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="Missing bundled"):
        adapter.prepare(tmp_path, fresh, image)
    assert not fresh.exists()


def test_recovery_cannot_switch_serving_image(setup, tmp_path, monkeypatch):
    plan, _, image = setup
    plan["provider_recovery_policy"] = "aces-provider-recovery-1"
    plan["recovery_source"] = {"execution": {"image_sha256": adapter.file_hash(image)}}
    monkeypatch.setattr(adapter, "verify_inputs", lambda _: plan)
    fresh = tmp_path / "recovery"
    image.write_bytes(b"a different image")
    with pytest.raises(ValueError, match="original serving image"):
        adapter.prepare(tmp_path, fresh, image)
    assert not fresh.exists()


def test_provider_failure_stops_other_cases_without_generation(
    setup, tmp_path, monkeypatch
):
    plan, root, _ = setup
    path = runtime_record(tmp_path)
    models(monkeypatch)
    called = []

    def episode(directory, current, task, **kwargs):
        called.append(task)
        return {"status": "provider_unavailable"}

    monkeypatch.setattr(adapter.campaign, "run_episode", episode)
    monkeypatch.setattr(adapter.campaign, "report", lambda *args: {"stopped": True})
    assert adapter.run(root, "http://127.0.0.1:8000", path) == {"stopped": True}
    assert called == plan["tasks"][:1]


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


def test_recovery_submission_advertises_only_remaining_calls(setup, monkeypatch):
    plan, _, _ = setup
    root = scheduler_setup(setup, monkeypatch)
    execution = sealed_read(root / "execution.json")
    plan["inherited_provider_outcomes"] = {
        task["task_id"]: {"physical_request_offset": 3} for task in plan["tasks"]
    }
    monkeypatch.setattr(scheduler.campaign, "verify", lambda _: (plan, execution))
    monkeypatch.setattr(scheduler, "submit_job", lambda *args: "1234")
    assert scheduler.submit(root, "recovery-1")["maximum_generation_requests"] == 21


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


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("health", ["healthy", "repairable", "broken", "module_failed"])
def test_launcher_checks_real_imports_and_stops_if_module_cannot_repair(
    tmp_path, worker, health
):
    """A numpy-only probe may pass even when pandas cannot import ctypes/libffi."""
    python = tmp_path / "python"
    python.write_text(
        "#!/bin/bash\n"
        'printf "python %s\\n" "$*" >> "$AF_TEST_LOG"\n'
        '[[ "$1" != -c ]] || exit 0\n'
        '[[ "$PYTHONPATH" == "$AF_REPO_ROOT:$AF_REPO_ROOT/src" ]] || exit 91\n'
        'if [[ "$2" == --help ]]; then\n'
        '  [[ "${AF_TEST_READY:-}" != 1 ]] || exit 0\n'
        "  echo 'ImportError: libffi.so.8' >&2\n"
        "  exit 1\n"
        "fi\n"
        # The batch test deliberately stops at verify, before touching any GPU.
        '[[ "$2" != verify ]] || exit 73\n'
        "exit 92\n"
    )
    python.chmod(0o755)
    modules = tmp_path / "modules.sh"
    modules.write_text(
        "module() {\n"
        '  printf "module %s\\n" "$*" >> "$AF_TEST_LOG"\n'
        '  [[ "$AF_TEST_HEALTH" != module_failed ]] || return 1\n'
        '  if [[ "$AF_TEST_HEALTH" == repairable ]]; then\n'
        "    export AF_TEST_READY=1\n"
        "  fi\n"
        "}\n"
    )
    squeue = tmp_path / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    squeue.chmod(0o755)
    log = tmp_path / "calls.log"
    output = tmp_path / "results"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "BASH_ENV": str(modules),
        "AF_TEST_LOG": str(log),
        "AF_TEST_HEALTH": health,
        "AF_TEST_READY": "1" if health == "healthy" else "0",
        "AF_PYTHON": str(python),
        "AF_GCCCORE_MODULE": "GCCcore/test-version",
        "AF_PYTHON_MODULE": "Python/test-version",
        "AF_REPO_ROOT": str(adapter.REPO),
        "AF_GROUP": str(tmp_path),
        "AF_OUTPUT_ROOT": str(output),
        "AF_VLLM_IMAGE": "/unused/image.sif",
        "AF_HF_HOME": "/unused/cache",
        "SLURM_JOB_ID": "test-job",
    }
    script = "run" if worker else "start"
    result = subprocess.run(
        [
            "bash",
            str(adapter.REPO / f"scripts/hpc/{script}_saved_initialization_aces.sh"),
            "propose" if worker else "status",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = log.read_text().splitlines()
    assert calls[0].endswith("saved_initialization_aces.py --help")
    if health == "healthy":
        assert not any(c.startswith("module ") for c in calls)
    else:
        assert calls[1] == "module load GCCcore/test-version Python/test-version"
    if health in {"healthy", "repairable"}:
        assert result.returncode == (73 if worker else 0), result.stderr
        if worker:
            assert " verify --root " in calls[-1]
    else:
        assert result.returncode == 1
        assert not any(" verify " in c or " prepare " in c for c in calls)
        if health == "broken":
            assert "ImportError: libffi.so.8" in result.stderr
    assert not output.exists()


def test_equality_adapter_admits_two_cases_but_not_three(setup):
    plan, root, _ = setup
    plan["protocol"] = adapter.EQUALITY_PROTOCOL
    plan["tasks"] = plan["tasks"][:2]
    assert adapter.verify_inputs(root)["protocol"] == adapter.EQUALITY_PROTOCOL
    plan["tasks"].append({"task_id": "third-case"})
    with pytest.raises(ValueError, match="bounded ACES"):
        adapter.verify_inputs(root)
