"""Tools for serving gpt-oss on a Jetstream2 GPU VM.

The GPU, Docker and the network are not available here, so what is under test
is the logic that decides: whether downloaded weights match the published
hashes, how a throughput measurement becomes an SU projection, and that the
shell entry points refuse bad arguments before touching anything.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.jetstream import verify_weights as weights
from scripts.jetstream import vllm_throughput as throughput

REPO = Path(__file__).resolve().parents[1]
JETSTREAM = REPO / "scripts" / "jetstream"

# `printf 'hello\n' | git hash-object --stdin` and `| shasum -a 256`.
HELLO_GIT_BLOB = "ce013625030ba8dba906f756967f9e9ca394464a"
HELLO_SHA256 = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
CONFIG_GIT_BLOB = "cb5b2f69babc53f9145b45ad20b9a075d48717ef"  # '{"a": 1}\n'
SKIPPED_FILE = {"blobId": "x", "lfs": {"sha256": "y", "size": 1}}


def _snapshot(tmp_path: Path) -> Path:
    snapshot = weights.snapshot_path(tmp_path / "hf", "openai/gpt-oss-20b", "abc123")
    snapshot.mkdir(parents=True)
    (snapshot / "model-00000-of-00001.safetensors").write_bytes(b"hello\n")
    (snapshot / "config.json").write_bytes(b'{"a": 1}\n')
    return snapshot


EXPECTED = {
    "model-00000-of-00001.safetensors": {"sha256": HELLO_SHA256, "size": 6},
    "config.json": {"git_blob": CONFIG_GIT_BLOB, "size": 9},
}


def test_a_file_is_hashed_the_way_git_and_lfs_name_it(tmp_path: Path) -> None:
    path = tmp_path / "hello"
    path.write_bytes(b"hello\n")
    assert weights.local_hashes(path) == {
        "sha256": HELLO_SHA256,
        "git_blob": HELLO_GIT_BLOB,
        "size": 6,
    }


def test_the_listing_skips_the_folders_vllm_does_not_load() -> None:
    siblings = [
        {"rfilename": "config.json", "blobId": CONFIG_GIT_BLOB, "size": 9},
        {
            "rfilename": "model-00000-of-00001.safetensors",
            "blobId": "pointer",
            "lfs": {"sha256": HELLO_SHA256, "size": 6, "pointerSize": 130},
        },
        {"rfilename": "original/model.safetensors", **SKIPPED_FILE},
        {"rfilename": "metal/model.bin", **SKIPPED_FILE},
    ]
    assert weights.expected_hashes(siblings) == EXPECTED


def test_a_matching_snapshot_verifies(tmp_path: Path) -> None:
    assert weights.compare(_snapshot(tmp_path), EXPECTED) == []


def test_a_changed_missing_or_extra_file_is_named(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    (snapshot / "model-00000-of-00001.safetensors").write_bytes(b"hellO\n")
    (snapshot / "config.json").unlink()
    (snapshot / "stray.bin").write_bytes(b"")
    problems = weights.compare(snapshot, EXPECTED)
    assert problems[0] == "unexpected file stray.bin"
    assert "missing config.json" in problems
    assert any(
        p.startswith("model-00000-of-00001.safetensors: sha256") for p in problems
    )


def test_a_cached_file_is_followed_through_its_link(tmp_path: Path) -> None:
    """huggingface_hub keeps snapshot files as links into a blob store."""
    snapshot = _snapshot(tmp_path)
    blob = tmp_path / "blob"
    blob.write_bytes(b'{"a": 1}\n')
    (snapshot / "config.json").unlink()
    (snapshot / "config.json").symlink_to(blob)
    assert weights.compare(snapshot, EXPECTED) == []


def test_a_measurement_reports_generated_tokens_per_second() -> None:
    result = throughput.summarize(
        "llm_sr", 4, [(10.0, 2000), (12.0, 1800)], elapsed=20.0, errors=[]
    )
    assert result["tokens_per_second"] == 190.0
    assert result["tokens_per_request"] == 1900.0
    assert result["mean_latency_seconds"] == 11.0
    assert result["first_error"] is None
    empty = throughput.summarize("llm_ode", 1, [], elapsed=5.0, errors=["URLError: x"])
    assert empty["tokens_per_request"] is None and empty["errors"] == 1


def test_the_projection_uses_the_best_rate_for_each_shape() -> None:
    results = [
        throughput.summarize("llm_sr", 1, [(1.0, 2000)], elapsed=2.0, errors=[]),
        throughput.summarize("llm_sr", 16, [(1.0, 2000)] * 8, elapsed=2.0, errors=[]),
        throughput.summarize("llm_ode", 16, [], elapsed=2.0, errors=["refused"]),
    ]
    projection = throughput.project(results, su_per_hour=128)
    # 55,000 requests x 2,000 tokens at 8,000 tokens/s is 13,750 s.
    assert projection["llm_sr"]["at_concurrency"] == 16
    assert projection["llm_sr"]["generated_tokens"] == 110_000_000
    assert projection["llm_sr"]["generation_hours"] == round(13_750 / 3600, 1)
    assert projection["llm_sr"]["su"] == round(13_750 / 3600 * 128)
    assert "llm_ode" not in projection  # nothing was measured for it


def test_the_llm_sr_shape_is_the_request_its_plans_send() -> None:
    """A chat request at the declared limit; raw text at 512 was a bug."""
    import json

    path, body, generated = throughput._shape("llm_sr", "m")
    plan = json.loads(
        (REPO / "configs" / "phase_c_llm_sr_budget_pilot_v2.json").read_text()
    )
    assert path == "/v1/chat/completions"
    assert body["messages"][0]["role"] == "user" and body["n"] == 4
    assert body["max_tokens"] == plan["reasoning_model_adaptation"]["max_new_tokens"]
    assert generated({"completion_tokens": 9000}) == 9000


def test_an_unknown_request_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown request shape"):
        throughput.measure("http://127.0.0.1:1", "m", "llm_d3", 1, 0.0)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("script", sorted(JETSTREAM.glob("*.sh")), ids=lambda p: p.name)
def test_every_vm_script_parses(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("script", "arguments", "message"),
    [
        ("gpu_server.sh", [], "Serve gpt-oss"),
        ("gpu_server.sh", ["start", "30b", "4"], "unknown model '30b'"),
        ("gpu_server.sh", ["start", "20b", "zero"], "usage: gpu_server.sh start"),
        ("run_phase_c_tasks.sh", ["sindy", "configs/x.json", "run", "4"], "METHOD"),
        (
            "run_phase_c_tasks.sh",
            ["d3", "configs/phase_c_d3_smoke_vm_20b_v2.json", "run", "0"],
            "METHOD",
        ),
        ("run_phase_c_tasks.sh", ["llm_sr", "configs/no.json", "run", "4"], "METHOD"),
        (
            "run_phase_c_hosted.sh",
            ["llm_sr", "configs/phase_c_d3_smoke_v2.json", "run"],
            "METHOD",
        ),
        ("run_phase_c_hosted.sh", ["d3", "configs/no.json", "run"], "METHOD"),
        (
            "run_phase_c_hosted.sh",
            ["d3", "configs/phase_c_d3_smoke_v2.json", "run", "0"],
            "METHOD",
        ),
        ("run_phase_c_classical.sh", [], "CONFIG"),
        ("run_phase_c_classical.sh", ["configs/no.json", "run"], "CONFIG"),
        (
            "run_phase_c_classical.sh",
            ["configs/phase_c_public_baseline_jetstream_cpu_v1.json", "a run"],
            "CONFIG",
        ),
        (
            "run_phase_c_classical.sh",
            ["configs/phase_c_public_baseline_jetstream_cpu_v1.json", "run"],
            "setup_classical.sh first",
        ),
    ],
)
def test_bad_arguments_are_refused_before_anything_runs(
    script: str, arguments: list[str], message: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        ["bash", str(JETSTREAM / script), *arguments],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        check=False,
    )
    assert result.returncode == 2
    assert message in result.stdout + result.stderr
    assert not (tmp_path / "af").exists()
