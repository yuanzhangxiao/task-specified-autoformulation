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
    subprocess.run(
        ["bash", str(WORKER)], env=worker_env, check=True, capture_output=True
    )
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
    scratch = Path(worker_env["AF_SEARCH_ROOT"])
    assert not (scratch / "server-args").exists()
    assert not list((scratch / "runtime").iterdir())
