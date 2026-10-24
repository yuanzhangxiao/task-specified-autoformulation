"""Matched resource/identity gates and offline ACES launcher dispatch."""

import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import submit_structural_revision_aces as submission

REAL_REPO = submission.REPO


def _snapshot(cache, size):
    path = (
        cache
        / f"hub/models--openai--gpt-oss-{size}/snapshots"
        / submission.MODELS[size]["revision"]
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
    for size in submission.MODELS:
        _snapshot(cache, size)
    for key, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
    }.items():
        monkeypatch.setenv(key, value)
    plans = {}
    for size, model in submission.MODELS.items():
        plans[size] = {
            "protocol": submission.PROTOCOL,
            "artifact_sha256": size + "a" * 64,
            "model_size": size,
            "cases": [
                {"name": f"P{parent:02d}-seed{seed}"}
                for parent in range(6)
                for seed in range(3)
            ],
            "settings": {
                **submission.MATCHED_SETTINGS,
                "model": model["model"],
                "model_revision": model["revision"],
            },
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
    monkeypatch.setattr(
        submission, "_campaign_verify", lambda root: deepcopy(plans[root.name])
    )
    return tmp_path, plans


def test_matched_budgets_and_distinct_resources(setup, monkeypatch):
    tmp, plans = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, str(100 + len(calls)), "")

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    records = []
    for size in ("20b", "120b"):
        root = tmp / size
        first = submission.submit(root)
        assert submission.submit(root) == first
        assert first["maximum_generation_requests"] == 54
        plan, execution = submission.verify(root, check_assets=True)
        assert plan == plans[size]
        assert execution["worker_seconds"] == 5400
        assert execution["maximum_charged_tokens"] == 18 * 98304
        records.append(execution)
    assert len(calls) == 2
    assert {"--gres=gpu:h100:1", "--mem=64G", "--cpus-per-task=8"} <= set(calls[0])
    assert {"--gres=gpu:h100:2", "--mem=128G", "--cpus-per-task=16"} <= set(calls[1])
    for key in (
        "case_ids",
        "generation_settings",
        "served_context_tokens",
        "maximum_generation_requests",
        "maximum_charged_tokens",
        "worker_seconds",
    ):
        assert records[0][key] == records[1][key]
    assert records[0]["model_revision"] != records[1]["model_revision"]


@pytest.mark.parametrize("mode", ["timeout", "empty_success", "failure"])
def test_ambiguous_or_failed_scheduler_intent_cannot_duplicate(
    setup, monkeypatch, mode
):
    tmp, _ = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(
            argv, 1 if mode == "failure" else 0, "\n", "Socket timed out"
        )

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    with pytest.raises(ValueError, match=r"uncertain|unconfirmed"):
        submission.submit(tmp / "20b")
    with pytest.raises(ValueError, match="existing submission intent"):
        submission.submit(tmp / "20b")
    assert len(calls) == 1
    assert (tmp / "20b/submission/propose.intent.json").is_file()
    assert (tmp / "20b/submission/propose.reply.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("maximum_requests", 4),
        ("max_output_tokens", 8193),
        ("reasoning_effort", "high"),
    ],
)
def test_generation_drift_rejected_before_slurm(setup, monkeypatch, field, value):
    tmp, plans = setup
    plans["20b"]["settings"][field] = value
    monkeypatch.setattr(
        submission.subprocess, "run", lambda *a, **kw: pytest.fail("no submit")
    )
    with pytest.raises(ValueError, match="requires six"):
        submission.submit(tmp / "20b")


def test_missing_weight_shard_and_tokenizer_rejected(setup):
    tmp, _ = setup
    snapshot = (
        tmp
        / "cache/hub/models--openai--gpt-oss-120b/snapshots"
        / submission.MODELS["120b"]["revision"]
    )
    (snapshot / "model-1.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        submission.submit(tmp / "120b")
    (snapshot / "model-1.safetensors").write_bytes(b"restored")
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match=r"tokenizer\.json"):
        submission.submit(tmp / "120b")


def test_actual_image_and_source_rechecked_before_gpu(setup, monkeypatch):
    tmp, _ = setup
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "123", ""),
    )
    submission.submit(tmp / "20b")
    (tmp / "image.sif").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(tmp / "20b", check_assets=True)
    (tmp / "image.sif").write_bytes(b"offline test image")
    (tmp / "code" / submission.ENTRYPOINTS[0]).write_text("changed source")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(tmp / "20b")


def test_shell_syntax_and_sealed_server_arguments():
    for name in (
        "start_structural_revision_aces.sh",
        "run_structural_revision_aces.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(REAL_REPO / "scripts/hpc" / name)], check=True
        )
    worker = (REAL_REPO / "scripts/hpc/run_structural_revision_aces.sh").read_text()
    assert "--verify-only --check-assets" in worker
    assert '--tensor-parallel-size "$AF_GPUS"' in worker
    assert '--runtime-record "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json"' in worker
    assert 'mktemp -d "/tmp/af-structural-' in worker
    assert "HF_HUB_OFFLINE=1" in worker
    assert "TRANSFORMERS_OFFLINE=1" in worker
    assert "gpu_names" in worker and "vllm.__version__ == '0.27.1'" in worker


@pytest.fixture
def serving(setup, monkeypatch):
    tmp, plans = setup
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "1234", ""),
    )

    def make(size):
        root = tmp / size
        submission.submit(root)
        _, execution = submission.verify(root)
        count = execution["tensor_parallel_size"]
        runtime = {
            "visible_gpus": count,
            "gpu_names": ["NVIDIA H100 PCIe"] * count,
            "gpu_memory_bytes": [80 * 1024**3] * count,
            "gpu_driver": ["580.178.04"] * count,
            "cuda_kernel_passed": True,
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "vllm": "0.27.1",
            "harmony": {
                **execution["harmony_vocabulary"],
                "harmony_render_passed": True,
                "openai_harmony": "test-version",
                "rendered_tokens": 12,
                "generation_requests": 0,
            },
            "serving_identity": {
                k: execution[k]
                for k in (
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
            lambda url, **kw: io.StringIO(
                json.dumps({"data": [{"id": plans[size]["settings"]["model"]}]})
            ),
        )
        return root, record, runtime

    return make


@pytest.mark.parametrize("size", ["20b", "120b"])
def test_actual_serving_identity_bound_for_both_arms(serving, size):
    root, record, runtime = serving(size)
    first = submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert first == submission.bind_server(root, record, "http://127.0.0.1:4567")
    assert first["serving_identity"]["model_size"] == size
    assert (root / "serving_runtime.json").is_file()
    record.unlink()
    runtime["gpu_driver"] = ["595.99.02"] * runtime["visible_gpus"]
    submission._seal(record, runtime)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.bind_server(root, record, "http://127.0.0.1:5678")


@pytest.mark.parametrize(
    "defect", ["count", "memory", "gpu", "harmony", "arm", "alias", "seal"]
)
def test_incompatible_runtime_rejected_before_binding(serving, monkeypatch, defect):
    root, record, runtime = serving("120b")
    record.unlink()
    if defect == "count":
        runtime["visible_gpus"] = 1
    elif defect == "memory":
        runtime["gpu_memory_bytes"] = [40 * 1024**3] * 2
    elif defect == "gpu":
        runtime["gpu_names"] = ["NVIDIA A100"] * 2
    elif defect == "harmony":
        runtime["harmony"]["harmony_render_passed"] = False
    elif defect == "arm":
        runtime["serving_identity"]["model_size"] = "20b"
    elif defect == "alias":
        monkeypatch.setattr(
            submission,
            "urlopen",
            lambda *a, **kw: io.StringIO(
                json.dumps({"data": [{"id": "openai/gpt-oss-20b"}]})
            ),
        )
    submission._seal(record, runtime)
    if defect == "seal":
        changed = json.loads(record.read_text())
        changed["vllm"] = "changed"
        record.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert not (root / "serving_runtime.json").exists()


def test_server_binding_rejects_nonlocal_endpoint(serving):
    root, record, _ = serving("20b")
    with pytest.raises(ValueError, match="loopback HTTP"):
        submission.bind_server(root, record, "https://example.com")


@pytest.mark.parametrize("size", ["20b", "120b"])
@pytest.mark.parametrize("command", ["run", "inspect", "report", "preview"])
def test_real_shell_dispatch_ignores_old_generic_root(tmp_path, size, command):
    group, cache = tmp_path / "group scratch", tmp_path / "cache"
    group.mkdir()
    _snapshot(cache, size)
    image = tmp_path / "image.sif"
    image.write_bytes(b"test image")
    log = tmp_path / "python-calls.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['AF_TEST_LOG'], 'a') as f:\n"
        "    value = {'argv': sys.argv[1:], 'root': os.environ['AF_OUTPUT_ROOT']}\n"
        "    f.write(json.dumps(value) + '\\n')\n"
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "AF_GROUP": str(group),
        "AF_OUTPUT_ROOT": str(group / "old-fitting"),
        "AF_PYTHON": str(stub),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
        "AF_TEST_LOG": str(log),
    }
    env.pop("AF_STRUCTURAL_REVISION_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            str(REAL_REPO / "scripts/hpc/start_structural_revision_aces.sh"),
            command,
            size,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    expected = str(group / f"phase-c-structural-revision-{size}-v1")
    assert calls and all(call["root"] == expected for call in calls)
    if command == "run":
        prepare = next(c["argv"] for c in calls if "prepare" in c["argv"])
        assert prepare[prepare.index("--model-size") + 1] == size
        assert prepare[prepare.index("--source") + 1].endswith(
            "inputs/saved-source.tar.gz"
        )
        assert Path(calls[-1]["argv"][0]).name == "submit_structural_revision_aces.py"
