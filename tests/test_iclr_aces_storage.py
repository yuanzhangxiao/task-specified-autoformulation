"""Exercise the ACES worker's storage paths before any model calls."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

WORKER = (
    Path(__file__).resolve().parents[1]
    / "scripts/hpc/run_iclr_controlled_search_aces.sh"
)


@pytest.fixture
def worker_env(tmp_path):
    if not all(shutil.which(cmd) for cmd in ("bash", "jq", "sha256sum")):
        pytest.skip("Slurm worker smoke test needs Bash, jq and sha256sum")
    scratch = tmp_path / "campaign"
    node = tmp_path / "node"
    commands = tmp_path / "bin"
    for directory in (scratch, node, commands):
        directory.mkdir()
    image = tmp_path / "fixture.sif"
    image.write_bytes(b"image")
    (scratch / "plan.json").write_text(
        json.dumps(
            {
                "protocol": "iclr-controlled-search-1",
                "critic_enabled": False,
                "model_settings": {"model": "fixture", "model_revision": "pinned"},
                "serving_image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "artifact_sha256": "fixture-plan",
            }
        )
    )
    for name, body in {
        "module": "exit 0",
        "nvidia-smi": "echo '0, NVIDIA H100 80GB HBM3'",
        "apptainer": (
            'printf "%s\\n" "$@" > "$AF_SEARCH_ROOT/server-args"\nexec sleep 30'
        ),
        "curl": (
            'test -s "$AF_SEARCH_ROOT/server-args" || exit 1\n'
            """printf '%s\\n' '{"data":[{"id":"fixture"}]}'"""
        ),
        "python": (
            'if [ "$1" = -c ]; then echo 43210; exit 0; fi\n'
            'if [ "$2" = verify ]; then exit 0; fi\n'
            'test "$2" = scripts.iclr_controlled_search && test "$3" = run'
        ),
    }.items():
        path = commands / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    return {
        "PATH": f"{commands}:{os.environ['PATH']}",
        "TMPDIR": str(node),
        "AF_PYTHON": str(commands / "python"),
        "AF_REPO_ROOT": str(tmp_path),
        "AF_SEARCH_ROOT": str(scratch),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(tmp_path),
        "AF_HARMONY_ASSETS": str(tmp_path / "harmony"),
        "SLURM_JOB_ID": "425",
        "SLURM_ARRAY_TASK_ID": "0",
    }


@pytest.mark.parametrize(
    ("shared_cache", "parallel"), [(False, 1), (True, 1), (False, 2)]
)
def test_compute_cache_and_durable_receipt(
    tmp_path, worker_env, shared_cache, parallel
):
    scratch = Path(worker_env["AF_SEARCH_ROOT"])
    if parallel == 2:
        plan = scratch / "plan.json"
        plan.write_text(
            json.dumps(
                {**json.loads(plan.read_text()), "proposer_tensor_parallel_size": 2}
            )
        )
    if shared_cache:
        worker_env["AF_COMPUTE_CACHE_ROOT"] = str(tmp_path / "shared")
    completed = subprocess.run(
        ["bash", str(WORKER)],
        env=worker_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Stage: gpu-health" in completed.stderr
    assert "Stage: controlled-search-worker" in completed.stderr
    assert "Failed during stage" not in completed.stderr
    receipt = json.loads((scratch / "runtime/service-425.json").read_text())
    cache = Path(receipt["compute_cache"])
    args = (scratch / "server-args").read_text().splitlines()
    assert receipt["tensor_parallel_size"] == parallel
    assert args[args.index("--tensor-parallel-size") + 1] == str(parallel)
    for name, subdir in (
        ("TRITON_CACHE_DIR", "triton"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        assert f"{name}={cache / subdir}" in args
    mounts = [
        m
        for i, arg in enumerate(args)
        if arg == "--bind"
        for m in args[i + 1].split(",")
    ]
    assert mounts.count(f"{cache}:{cache}") == 1
    assert not list(scratch.rglob(".write-check.*"))
    assert not list(Path(worker_env["TMPDIR"]).iterdir())
    if shared_cache:
        assert cache == tmp_path / "shared/425" and cache.is_dir()
    else:
        assert cache.parent == Path(worker_env["TMPDIR"]) and not cache.exists()


def test_checkpoint_quota_failure_precedes_serving(tmp_path, worker_env):
    mktemp = tmp_path / "bin/mktemp"
    mktemp.write_text('#!/bin/sh\necho "Disk quota exceeded" >&2\nexit 1\n')
    mktemp.chmod(0o755)
    result = subprocess.run(
        ["bash", str(WORKER)], env=worker_env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "Cannot create a durable checkpoint" in result.stderr
    assert "Failed during stage: checkpoint-storage" in result.stderr
    scratch = Path(worker_env["AF_SEARCH_ROOT"])
    assert not (scratch / "server-args").exists()
    assert not list((scratch / "runtime").iterdir())


@pytest.mark.parametrize("exit_code", [0, 1])
def test_gpu_health_failure_precedes_image_and_serving(tmp_path, worker_env, exit_code):
    probe = tmp_path / "bin/nvidia-smi"
    probe.write_text(
        "#!/bin/sh\necho 'Failed to get device handle for GPU0: Unknown Error'\n"
        f"exit {exit_code}\n"
    )
    result = subprocess.run(
        ["bash", str(WORKER)], env=worker_env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "GPU health check failed" in result.stderr
    assert "Unknown Error" in result.stderr
    assert "Failed during stage: gpu-health" in result.stderr
    assert "Stage: image-verification" not in result.stderr
    assert not (Path(worker_env["AF_SEARCH_ROOT"]) / "server-args").exists()
    assert not list(Path(worker_env["TMPDIR"]).iterdir())


@pytest.mark.parametrize("missing", [False, True])
def test_serving_image_failure_is_explicit(worker_env, missing):
    image = Path(worker_env["AF_VLLM_IMAGE"])
    if missing:
        image.unlink()
    else:
        image.write_bytes(b"different image")
    result = subprocess.run(
        ["bash", str(WORKER)], env=worker_env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Failed during stage: image-verification" in result.stderr
    assert (
        "Cannot read or hash the serving image"
        if missing
        else "Serving image SHA-256 differs from the frozen plan"
    ) in result.stderr
    assert not (Path(worker_env["AF_SEARCH_ROOT"]) / "server-args").exists()
    assert not list(Path(worker_env["TMPDIR"]).iterdir())


def test_vllm_failure_shows_bounded_redacted_tail(tmp_path, worker_env):
    runtime = tmp_path / "bin/apptainer"
    runtime.write_text(
        "#!/bin/sh\n"
        "i=0\nwhile [ $i -lt 100 ]; do echo \"startup line $i\"; i=$((i + 1)); done\n"
        "echo 'api_key=fixture-credential-not-for-logs'\n"
        "echo 'pynvml.NVMLError_Unknown: Unknown Error'\nexit 1\n"
    )
    (tmp_path / "bin/curl").write_text("#!/bin/sh\nexit 1\n")
    result = subprocess.run(
        ["bash", str(WORKER)],
        env=worker_env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1
    assert "Failed during stage: vllm-startup" in result.stderr
    assert "pynvml.NVMLError_Unknown: Unknown Error" in result.stderr
    assert "startup line 0\n" not in result.stderr
    assert "fixture-credential-not-for-logs" not in result.stderr
    assert "[credential-related log line omitted]" in result.stderr
    assert "Stage: controlled-search-worker" not in result.stderr
    assert not list(Path(worker_env["TMPDIR"]).iterdir())


def test_worker_failure_reports_stage_without_echoing_credential(tmp_path, worker_env):
    interpreter = tmp_path / "bin/python"
    interpreter.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = -c ]; then echo 43210; exit 0; fi\n'
        'if [ "$2" = verify ]; then exit 0; fi\n'
        "exit 7\n"
    )
    worker_env["AF_JETSTREAM_API_KEY"] = "fixture-key-not-for-logs"
    result = subprocess.run(
        ["bash", str(WORKER)], env=worker_env, capture_output=True, text=True
    )
    assert result.returncode == 7
    assert "Failed during stage: controlled-search-worker (exit 7)" in result.stderr
    assert "fixture-key-not-for-logs" not in result.stdout + result.stderr
    assert "Server log tail" not in result.stderr
    assert not list(Path(worker_env["TMPDIR"]).iterdir())
