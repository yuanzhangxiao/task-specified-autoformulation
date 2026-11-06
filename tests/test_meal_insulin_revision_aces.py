"""Offline one-call ACES submission, source checks and shell dispatch."""

import io
import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest

from scripts import submit_meal_insulin_revision_aces as submission
from scripts import submit_structural_revision_aces as shared

REAL_REPO = submission.REPO


def _snapshot(cache):
    path = (
        cache
        / "hub/models--openai--gpt-oss-120b/snapshots"
        / submission.MODELS["120b"]["revision"]
    )
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors"}})
    )
    (path / "model-1.safetensors").write_bytes(b"offline test weights")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    code = tmp_path / "code"
    for name in submission.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test source {name}\n")
    monkeypatch.setattr(submission, "REPO", code)
    monkeypatch.setattr(
        submission.harmony_vocab,
        "verify",
        lambda path: {
            "protocol": submission.harmony_vocab.PROTOCOL,
            "filename": submission.harmony_vocab.VOCAB_FILENAME,
            "sha256": submission.harmony_vocab.VOCAB_SHA256,
        },
    )
    image, cache = tmp_path / "image.sif", tmp_path / "cache"
    image.write_bytes(b"offline test image")
    _snapshot(cache)
    for key, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
    }.items():
        monkeypatch.setenv(key, value)
    plan = {
        "protocol": submission.PROTOCOL,
        "artifact_sha256": "a" * 64,
        "model_size": "120b",
        "settings": dict(submission.SETTINGS),
        "seed": 0,
        "optimizer_calls": 0,
        "test_data_opened": False,
        "automatic_followup": False,
    }
    monkeypatch.setattr(submission, "_proposer_load", lambda root: (deepcopy(plan), {}))
    return tmp_path, plan


def test_one_call_resources_and_idempotent_submission(setup, monkeypatch):
    tmp, plan = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "1234;aces\n", "")

    monkeypatch.setattr(shared.subprocess, "run", scheduler)
    root = tmp / "revision"
    first = submission.submit(root)
    assert submission.submit(root) == first
    (root / "attempt.json").write_text("in-flight generation reservation")
    assert submission.submit(root) == first
    loaded, execution = submission.verify(root, check_assets=True)
    assert loaded == plan
    assert len(calls) == 1
    assert {
        "--gres=gpu:h100:2",
        "--cpus-per-task=16",
        "--mem=128G",
        "--time=01:00:00",
    } <= set(calls[0])
    assert first["job"] == "1234"
    assert first["maximum_generation_requests"] == 1
    assert execution["model_revision"] == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    assert execution["maximum_generation_requests"] == 1
    assert execution["maximum_output_tokens"] == 4096
    assert execution["maximum_charged_tokens"] == 32768
    assert execution["served_context_tokens"] == 32768
    assert execution["scheduler_walltime_seconds"] == 3600
    assert execution["optimizer_calls"] == 0
    assert not execution["automatic_followup"]
    assert shared._read(root / "submission/manifest.json") == first


@pytest.mark.parametrize("mode", ["timeout", "empty_success", "failure", "os_error"])
def test_uncertain_submission_never_duplicates(setup, monkeypatch, mode):
    tmp, _ = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        if mode == "os_error":
            raise OSError("scheduler interrupted")
        return subprocess.CompletedProcess(
            argv, 1 if mode == "failure" else 0, "\n", "Socket timed out"
        )

    monkeypatch.setattr(shared.subprocess, "run", scheduler)
    root = tmp / "revision"
    with pytest.raises((ValueError, OSError)):
        submission.submit(root)
    with pytest.raises(ValueError, match="existing submission intent"):
        submission.submit(root)
    assert len(calls) == 1
    assert (root / "submission/propose.intent.json").is_file()
    if mode != "os_error":
        assert (root / "submission/propose.reply.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "openai/gpt-oss-20b"),
        ("maximum_requests", 2),
        ("attempts_per_step", 2),
        ("max_output_tokens", 8192),
        ("maximum_total_tokens", 65536),
        ("reasoning_effort", "high"),
        ("model_revision", "unverified"),
    ],
)
def test_generation_drift_is_rejected_before_slurm(setup, monkeypatch, field, value):
    tmp, plan = setup
    plan["settings"][field] = value
    monkeypatch.setattr(
        submission, "_submit_job", lambda *a: pytest.fail("must not submit")
    )
    with pytest.raises(ValueError, match="requires one 120B call"):
        submission.submit(tmp / "revision")


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_calls", 1),
        ("test_data_opened", True),
        ("automatic_followup", True),
        ("model_size", "20b"),
    ],
)
def test_scientific_scope_drift_is_rejected(setup, field, value):
    tmp, plan = setup
    plan[field] = value
    with pytest.raises(ValueError, match="requires one 120B call"):
        submission.submit(tmp / "revision")


def test_source_verifier_failure_stops_before_assets(setup, monkeypatch):
    tmp, _ = setup

    def invalid(root):
        raise ValueError("frozen proposer plan or source differs")

    monkeypatch.setattr(submission, "_proposer_load", invalid)
    monkeypatch.setattr(submission, "_assets", lambda: pytest.fail("no assets"))
    with pytest.raises(ValueError, match="source differs"):
        submission.submit(tmp / "revision")


@pytest.mark.parametrize("asset", ["model-1.safetensors", "tokenizer.json"])
def test_missing_offline_assets_are_rejected(setup, asset):
    tmp, _ = setup
    snapshot = (
        tmp
        / "cache/hub/models--openai--gpt-oss-120b/snapshots"
        / submission.MODELS["120b"]["revision"]
    )
    (snapshot / asset).unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        submission.submit(tmp / "revision")


def test_changed_assets_entrypoints_and_manifest_are_rejected(setup, monkeypatch):
    tmp, _ = setup
    monkeypatch.setattr(submission, "_submit_job", lambda *a: "1234")
    root = tmp / "revision"
    submission.submit(root)
    (tmp / "image.sif").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(root, check_assets=True)
    (tmp / "image.sif").write_bytes(b"offline test image")
    entrypoint = tmp / "code" / submission.ENTRYPOINTS[0]
    original = entrypoint.read_text()
    entrypoint.write_text("changed code")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(root)
    entrypoint.write_text(original)
    manifest = root / "submission/manifest.json"
    value = json.loads(manifest.read_text())
    value["job"] = "9999"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest differs"):
        submission.submit(root)


def test_terminal_receipt_does_not_require_assets_or_another_gpu(setup, monkeypatch):
    tmp, _ = setup
    root = tmp / "revision"
    root.mkdir()
    (root / "result.json").write_text("verified by owner")
    monkeypatch.setattr(
        submission, "_proposer_status", lambda root: {"status": "eligible_for_refit"}
    )
    monkeypatch.setattr(submission, "_assets", lambda: pytest.fail("no assets"))
    assert submission.submit(root) == {
        "status": "eligible_for_refit",
        "already_complete": True,
    }


def test_existing_generation_reservation_never_allocates_again(setup, monkeypatch):
    tmp, _ = setup
    root = tmp / "revision"
    root.mkdir()
    (root / "attempt.json").write_text("existing reservation")
    monkeypatch.setattr(submission, "_assets", lambda: pytest.fail("no assets"))
    with pytest.raises(ValueError, match="generation already reserved"):
        submission.submit(root)


@pytest.fixture
def serving(setup, monkeypatch):
    tmp, _ = setup
    monkeypatch.setattr(submission, "_submit_job", lambda *a: "1234")
    root = tmp / "revision"
    submission.submit(root)
    _, execution = submission.verify(root)
    runtime = {
        "visible_gpus": 2,
        "gpu_names": ["NVIDIA H100 PCIe"] * 2,
        "gpu_memory_bytes": [80 * 1024**3] * 2,
        "gpu_driver": ["580.178.04"],
        "cuda_kernel_passed": True,
        "torch": "2.13.0+cu130",
        "cuda": "13.0",
        "vllm": "0.27.1",
        "harmony": {
            **execution["harmony_vocabulary"],
            "harmony_render_passed": True,
            "openai_harmony": "0.0.8",
            "rendered_tokens": 10,
            "generation_requests": 0,
        },
        "serving_identity": {
            key: execution[key]
            for key in (
                "artifact_sha256",
                "model_size",
                "model",
                "model_revision",
                "image_sha256",
                "served_context_tokens",
                "tensor_parallel_size",
                "maximum_sequences",
            )
        },
    }
    record = root / "runtime.json"
    submission._seal(record, runtime)
    monkeypatch.setattr(
        submission,
        "urlopen",
        lambda *a, **kw: io.StringIO(
            json.dumps(
                {
                    "data": [{"id": execution["model"], "max_model_len": 32768}],
                }
            )
        ),
    )
    return root, record, runtime


def test_runtime_binding_is_immutable_but_allows_port_change(serving):
    root, record, runtime = serving
    first = submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert submission.bind_server(root, record, "http://127.0.0.1:5678") == first
    record.unlink()
    runtime["gpu_driver"] = ["different"]
    submission._seal(record, runtime)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.bind_server(root, record, "http://127.0.0.1:1234")


@pytest.mark.parametrize(
    "defect",
    ["count", "memory", "gpu", "harmony", "revision", "version", "alias", "seal"],
)
def test_incompatible_runtime_cannot_bind(serving, monkeypatch, defect):
    root, record, runtime = serving
    record.unlink()
    if defect == "count":
        runtime["visible_gpus"] = 1
    elif defect == "memory":
        runtime["gpu_memory_bytes"] = [40 * 1024**3] * 2
    elif defect == "gpu":
        runtime["gpu_names"] = ["NVIDIA A100"] * 2
    elif defect == "harmony":
        runtime["harmony"]["harmony_render_passed"] = False
    elif defect == "revision":
        runtime["serving_identity"]["model_revision"] = "other"
    elif defect == "version":
        runtime["vllm"] = "0.26.0"
    elif defect == "alias":
        monkeypatch.setattr(
            submission,
            "urlopen",
            lambda *a, **kw: io.StringIO(
                json.dumps({"data": [{"id": "other", "max_model_len": 32768}]})
            ),
        )
    submission._seal(record, runtime)
    if defect == "seal":
        record.write_text(record.read_text().replace('"0.27.1"', '"changed"'))
    with pytest.raises(ValueError):
        submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert not (root / "serving_runtime.json").exists()


@pytest.mark.parametrize(
    "endpoint",
    ["https://example.com", "http://localhost:1234", "http://user@127.0.0.1"],
)
def test_binding_requires_loopback(serving, endpoint):
    root, record, _ = serving
    with pytest.raises(ValueError, match="loopback HTTP"):
        submission.bind_server(root, record, endpoint)


def test_shell_syntax_and_offline_worker_contract():
    for name in (
        "start_meal_insulin_revision_aces.sh",
        "run_meal_insulin_revision_aces.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(REAL_REPO / "scripts/hpc" / name)], check=True
        )
    worker = (REAL_REPO / "scripts/hpc/run_meal_insulin_revision_aces.sh").read_text()
    assert "BASH_SOURCE" not in worker  # Slurm executes a spooled script.
    assert "${AF_REPO_ROOT:?}" in worker
    assert 'setsid "$AF_CONTAINER"' in worker
    assert 'signal_group KILL "$AF_SERVER_PID"' in worker
    assert worker.index("--verify-only --check-assets") < worker.index("--nv")
    assert '--tensor-parallel-size "$AF_GPUS"' in worker
    assert '--runtime-record "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json"' in worker
    assert 'mktemp -d "/tmp/af-insulin-revision-' in worker
    assert 'rm -rf -- "$AF_JOB_TMP"' in worker
    assert "vllm.__version__ == '0.27.1'" in worker
    assert "HF_HUB_OFFLINE=1" in worker and "TRANSFORMERS_OFFLINE=1" in worker
    assert "--host 127.0.0.1" in worker
    assert '"$AF_CLI" inspect --root "$AF_OUTPUT_ROOT"' in worker
    assert "--seconds" not in worker


@pytest.mark.parametrize("command", ["run", "status", "inspect", "preview"])
@pytest.mark.parametrize("load_modules", [False, True])
def test_shell_dispatch_uses_actual_checkout_and_task_root(
    tmp_path, command, load_modules
):
    group = tmp_path / "group scratch"
    group.mkdir()
    log = tmp_path / "python-calls.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "marker = Path(os.environ['AF_TEST_LOG'] + '.probe')\n"
        "if os.environ.get('AF_TEST_MODULES') and not marker.exists():\n"
        "    marker.touch()\n"
        "    sys.exit(1)\n"
        "with open(os.environ['AF_TEST_LOG'], 'a') as f:\n"
        "    value = {'argv': sys.argv[1:], 'root': os.environ['AF_OUTPUT_ROOT'],\n"
        "             'repo': os.environ['AF_REPO_ROOT'],\n"
        "             'pythonpath': os.environ['PYTHONPATH']}\n"
        "    f.write(json.dumps(value) + '\\n')\n"
    )
    stub.chmod(0o755)
    shell_env = tmp_path / "bash-env"
    shell_env.write_text(
        "module() { export PYTHONPATH=/module/dependencies; }\nsqueue() { return 0; }\n"
    )
    env = {
        **os.environ,
        "AF_GROUP": str(group),
        "AF_OUTPUT_ROOT": str(group / "old-fitting"),
        "AF_REPO_ROOT": str(tmp_path / "wrong-code"),
        "PYTHONPATH": "/another/campaign",
        "AF_PYTHON": str(stub),
        "AF_TEST_LOG": str(log),
        "BASH_ENV": str(shell_env),
    }
    if load_modules:
        env["AF_TEST_MODULES"] = "yes"
    env.pop("AF_MEAL_INSULIN_REVISION_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            str(REAL_REPO / "scripts/hpc/start_meal_insulin_revision_aces.sh"),
            command,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    expected = str(group / "phase-c-insulin-revision-120b-v1")
    assert calls and all(call["root"] == expected for call in calls)
    assert all(call["repo"] == str(REAL_REPO) for call in calls)
    assert all(
        call["pythonpath"].startswith(f"{REAL_REPO}:{REAL_REPO}/src") for call in calls
    )
    if load_modules:
        assert all("/module/dependencies" in call["pythonpath"] for call in calls)
    if command == "run":
        prepare = next(c["argv"] for c in calls if "prepare" in c["argv"])
        assert prepare[prepare.index("--source") + 1] == str(
            REAL_REPO / "inputs/revision-intake.json"
        )
        assert prepare[prepare.index("--model-size") + 1] == "120b"
        assert calls[-1]["argv"][0].endswith("submit_meal_insulin_revision_aces.py")
    else:
        assert not any(
            "submit_meal_insulin_revision_aces.py" in c["argv"][0] for c in calls
        )
        assert command in calls[-1]["argv"]
