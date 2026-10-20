"""ACES function construction binds resources, local assets, and scheduler intent."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import submit_construction_functions_aces as submission


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Build local fake assets; all scheduler and provider calls remain mocked."""
    root = tmp_path / "campaign"
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": submission.PROTOCOL,
            "cases": [{"case_id": f"proposal-{index}"} for index in range(8)],
            "settings": {
                "model": submission.MODEL,
                "model_revision": submission.REVISION,
                "max_output_tokens": 8192,
                "maximum_requests": 96,
                "maximum_total_tokens": 2097152,
            },
            "budgets": deepcopy(submission.BUDGETS),
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )
    monkeypatch.setattr(submission.campaign, "verify", lambda _: deepcopy(plan))
    monkeypatch.setattr(
        submission.harmony_vocab,
        "verify",
        lambda path: {"protocol": "fixture-harmony", "sha256": "fixed-vocabulary"},
    )
    image = tmp_path / "image.sif"
    image.write_bytes(b"test image")
    cache = tmp_path / "cache"
    snapshot = (
        cache
        / "hub/models--openai--gpt-oss-20b/snapshots"
        / submission.REVISION
        / "config.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    for name, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
        "AF_REPO_ROOT": str(submission.REPO),
        "AF_OUTPUT_ROOT": str(root),
    }.items():
        monkeypatch.setenv(name, value)
    return root, plan, image, cache


def test_confirmed_submission_is_idempotent_and_has_fixed_resources(setup, monkeypatch):
    root, plan, _, _ = setup
    calls = []

    def queue(directory, key, options, worker, stage, index):
        calls.append(options)
        assert key == stage == "propose"
        assert index == 0
        assert Path(worker).name == "run_construction_functions_aces.sh"
        return "1234"

    monkeypatch.setattr(submission, "submit_job", queue)
    result = submission.submit(root)
    assert submission.submit(root) == result
    assert len(calls) == 1
    assert result["job"] == "1234"
    assert result["maximum_cases"] == 8
    assert result["maximum_generation_requests"] == 768
    assert result["maximum_charged_tokens"] == 8 * 2097152
    assert result["automatic_followup"] is False
    assert {
        "--gres=gpu:h100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=02:00:00",
        "--signal=B:TERM@300",
    } <= set(calls[0])
    assert not any(item.startswith("--dependency") for item in calls[0])
    current, execution = submission.verify(root)
    assert current == plan
    assert execution["protocol"] == submission.EXECUTION_PROTOCOL
    assert execution["worker_seconds"] == 5400
    assert execution["optimizer_calls"] == 0
    assert execution["test_data_opened"] is False
    assert execution["plan_sha256"] == plan["artifact_sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "cases",
        "model",
        "revision",
        "calls",
        "tokens",
        "budget",
        "fit",
        "test",
    ],
)
def test_expanded_or_incompatible_plan_never_reaches_scheduler(
    setup, monkeypatch, mutation
):
    root, plan, _, _ = setup
    if mutation == "protocol":
        plan["protocol"] = "saved-construction-initialization-repair-2"
    elif mutation == "cases":
        plan["cases"].append({"case_id": "extra"})
    elif mutation == "model":
        plan["settings"]["model"] = "different"
    elif mutation == "revision":
        plan["settings"]["model_revision"] = "different"
    elif mutation == "calls":
        plan["settings"]["maximum_requests"] += 1
    elif mutation == "tokens":
        plan["settings"]["maximum_total_tokens"] += 1
    elif mutation == "budget":
        plan["budgets"]["attempts_per_step"] += 1
    elif mutation == "fit":
        plan["optimizer_calls"] = 1
    else:
        plan["test_data_opened"] = True
    monkeypatch.setattr(
        submission, "submit_job", lambda *args: pytest.fail("no submission")
    )
    with pytest.raises(ValueError, match="bounded ACES"):
        submission.submit(root)
    assert not (root / "execution.json").exists()


def test_scheduler_timeout_preserves_intent_and_never_retries(setup, monkeypatch):
    root, _, _, _ = setup
    calls = []

    def timeout(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, 60)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ValueError, match="uncertain scheduler timeout"):
        submission.submit(root)
    folder = root / "submissions/functions-1"
    assert json.loads((folder / "propose.intent.json").read_text())["argv"] == calls[0]
    assert json.loads((folder / "propose.reply.json").read_text()) == {
        "status": "uncertain_timeout"
    }
    with pytest.raises(ValueError, match="inspect uncertain submission"):
        submission.submit(root)
    assert len(calls) == 1
    assert not (folder / "manifest.json").exists()


def test_changed_image_or_code_cannot_reuse_execution(setup, monkeypatch):
    root, _, image, _ = setup
    monkeypatch.setattr(submission, "submit_job", lambda *args: "1234")
    submission.submit(root)
    image.write_bytes(b"changed image")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.submit(root, "manual-resume")
    original = submission.file_hash
    monkeypatch.setattr(
        submission,
        "file_hash",
        lambda path: "changed"
        if path.name == "run_construction_functions_aces.sh"
        else original(path),
    )
    with pytest.raises(ValueError, match="execution identity differs"):
        submission.verify(root)


def test_corrupt_execution_and_missing_model_are_rejected(setup, monkeypatch):
    root, _, _, cache = setup
    monkeypatch.setattr(submission, "submit_job", lambda *args: "1234")
    submission.submit(root)
    path = root / "execution.json"
    execution = sealed_read(path)
    execution["worker_seconds"] = 7200
    path.write_text(json.dumps(execution))
    with pytest.raises(ValueError, match="artifact digest differs"):
        submission.verify(root)
    next(cache.rglob("config.json")).unlink()
    with pytest.raises(ValueError, match="pinned offline model snapshot"):
        submission.submit(root, "manual-resume")


@pytest.mark.parametrize("wave", ["../bad", "bad/wave", "bad wave", ""])
def test_unsafe_wave_never_creates_submission(setup, wave):
    root, _, _, _ = setup
    with pytest.raises(ValueError, match="simple identifier"):
        submission.submit(root, wave)
    assert not (root / "submissions").exists()


def _shell_environment(tmp_path: Path) -> dict[str, str]:
    """Keep inherited campaign configuration out of isolated shell tests."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("AF_")
    }
    environment.update(AF_GROUP=str(tmp_path), AF_REPO_ROOT=str(submission.REPO))
    return environment


def test_start_wrapper_uses_new_root_source_and_manual_single_wave(setup, tmp_path):
    _, _, image, cache = setup
    recorder = tmp_path / "record-python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    recorder.chmod(0o755)
    arguments = tmp_path / "arguments.txt"
    environment = _shell_environment(tmp_path)
    environment.update(
        AF_PYTHON=str(recorder),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(cache),
        AF_ARGUMENT_LOG=str(arguments),
    )
    wrapper = submission.REPO / "scripts/hpc/start_construction_functions_aces.sh"
    subprocess.run(["bash", str(wrapper), "run"], check=True, env=environment)
    lines = arguments.read_text().splitlines()
    assert lines[lines.index("--source") + 1] == str(
        submission.REPO / "inputs/source-inspection.tar.gz"
    )
    assert lines[lines.index("--wave") + 1] == "functions-1"
    roots = [lines[index + 1] for index, value in enumerate(lines) if value == "--root"]
    assert roots == [str(tmp_path / "phase-c-functions-v1")] * 2
    assert (
        sum(item.endswith("submit_construction_functions_aces.py") for item in lines)
        == 1
    )
    assert not (tmp_path / "phase-c-functions-v1").exists()


def test_batch_stops_before_cuda_when_execution_verification_fails(setup, tmp_path):
    root, _, image, cache = setup
    recorder = tmp_path / "record-python"
    recorder.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n'
        '[[ "${2:-}" != --verify-only ]] || exit 17\n'
    )
    recorder.chmod(0o755)
    container = tmp_path / "apptainer"
    container.write_text('#!/bin/bash\ntouch "$AF_CONTAINER_STARTED"\n')
    container.chmod(0o755)
    arguments = tmp_path / "arguments.txt"
    marker = tmp_path / "container-started"
    environment = _shell_environment(tmp_path)
    environment.update(
        AF_PYTHON=str(recorder),
        AF_OUTPUT_ROOT=str(root),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(cache),
        AF_ARGUMENT_LOG=str(arguments),
        AF_CONTAINER_STARTED=str(marker),
        SLURM_JOB_ID="fixture",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
    )
    wrapper = submission.REPO / "scripts/hpc/run_construction_functions_aces.sh"
    result = subprocess.run(["bash", str(wrapper)], check=False, env=environment)
    assert result.returncode == 17
    assert "--verify-only" in arguments.read_text().splitlines()
    assert not marker.exists()
    assert not (root / "runtime").exists()


def _executable(path: Path, body: str) -> Path:
    """Install a fake local command for a network-free batch smoke test."""
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("worker_status", [0, 23])
def test_batch_uses_frozen_server_limits_and_preserves_worker_exit(
    setup, tmp_path, monkeypatch, worker_status
):
    root, _, image, cache = setup
    monkeypatch.setattr(submission, "submit_job", lambda *args: "1234")
    submission.submit(root)
    commands = tmp_path / "commands"
    commands.mkdir()
    recorder = _executable(
        commands / "python",
        """import json, os, subprocess, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["AF_ARGUMENT_LOG"], "a") as stream:
    stream.write(json.dumps({"command": "python", "args": args}) + "\\n")
if args[0] == "-c":
    print(os.environ["AF_IMAGE_SHA"] if "image_sha256" in args[1] else "8765")
elif args[0] == "-":
    result = subprocess.run([sys.executable, *args], input=sys.stdin.read(), text=True)
    raise SystemExit(result.returncode)
elif len(args) > 1 and args[1] == "run":
    raise SystemExit(int(os.environ["AF_WORKER_STATUS"]))
""",
    )
    _executable(
        commands / "apptainer",
        """import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["AF_ARGUMENT_LOG"], "a") as stream:
    stream.write(json.dumps({"command": "container", "args": args}) + "\\n")
if "serve" in args:
    Path(os.environ["AF_SERVER_STARTED"]).touch()
    time.sleep(60)
elif any(item.endswith("harmony_vocab.py") for item in args):
    print(json.dumps({"harmony_render_passed": True}))
else:
    print(json.dumps({"visible_gpus": 1, "gpu_name": "H100", "vllm": "0.27.1",
                     "cuda_kernel_passed": True}))
""",
    )
    _executable(
        commands / "curl",
        """import os
from pathlib import Path
raise SystemExit(0 if Path(os.environ["AF_SERVER_STARTED"]).exists() else 22)
""",
    )
    _executable(
        commands / "sha256sum",
        """import os, sys
print(os.environ["AF_IMAGE_SHA"] + "  " + sys.argv[1])
""",
    )
    arguments = tmp_path / "arguments.jsonl"
    environment = _shell_environment(tmp_path)
    environment.update(
        AF_PYTHON=str(recorder),
        AF_OUTPUT_ROOT=str(root),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(cache),
        AF_ARGUMENT_LOG=str(arguments),
        AF_IMAGE_SHA=submission.file_hash(image),
        AF_SERVER_STARTED=str(tmp_path / "server-started"),
        AF_WORKER_STATUS=str(worker_status),
        SLURM_JOB_ID="fixture",
        PATH=str(commands) + os.pathsep + os.environ["PATH"],
    )
    wrapper = submission.REPO / "scripts/hpc/run_construction_functions_aces.sh"
    result = subprocess.run(
        ["bash", str(wrapper)], check=False, env=environment, timeout=15
    )
    assert result.returncode == worker_status
    calls = [json.loads(line) for line in arguments.read_text().splitlines()]
    servers = [row for row in calls if "serve" in row["args"]]
    workers = [row for row in calls if "run" in row["args"]]
    assert len(servers) == len(workers) == 1
    server = servers[0]["args"]
    assert server[server.index("--revision") + 1] == submission.REVISION
    assert server[server.index("--host") + 1] == "127.0.0.1"
    assert server[server.index("--max-model-len") + 1] == "32768"
    assert server[server.index("--tensor-parallel-size") + 1] == "1"
    assert "HF_HUB_OFFLINE=1" in server
    assert "TRANSFORMERS_OFFLINE=1" in server
    temporary = next(
        value.split("=", 1)[1] for value in server if value.startswith("TMPDIR=")
    )
    assert temporary.startswith("/tmp/af-functions-fixture.")
    assert not Path(temporary).exists()
    worker = workers[0]["args"]
    assert worker[worker.index("--seconds") + 1] == "5400"
    assert worker[worker.index("--base-url") + 1] == "http://127.0.0.1:8765"
    runtime = json.loads((root / "runtime/probe-fixture.json").read_text())
    assert runtime["serving_identity"]["model_revision"] == submission.REVISION
    assert runtime["serving_identity"]["image_sha256"] == submission.file_hash(image)
    assert (
        root / "runtime/worker-exit-fixture.txt"
    ).read_text() == f"{worker_status}\n"


def test_shell_launchers_parse():
    for name in submission.ENTRYPOINTS:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(submission.REPO / name)], check=True)
