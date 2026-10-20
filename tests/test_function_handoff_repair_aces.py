"""Initializer retry launches one bounded server without regenerating dynamics."""

import json
import os
import subprocess
import sys
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import submit_function_handoff_repair_aces as submission
from tests.test_construction_functions_aces import _executable, _shell_environment


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Use fake serving assets and a sealed eight-case/three-retry plan."""
    root = tmp_path / "campaign"
    cases = [
        {
            "case_id": f"proposal-{index}",
            "action": "retry_failed_initializer"
            if index < 3
            else "migrate_saved_initializer",
        }
        for index in range(8)
    ]
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": submission.PROTOCOL,
            "cases": cases,
            "retry_case_ids": [case["case_id"] for case in cases[:3]],
            "settings": deepcopy(submission.SETTINGS),
            "source_archive_sha256": "a" * 64,
            "runtime_source_sha256": "b" * 64,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
            "dynamics_generation_calls": 0,
        },
    )
    monkeypatch.setattr(submission.campaign, "verify", lambda _: deepcopy(plan))
    monkeypatch.setattr(
        submission.harmony_vocab,
        "verify",
        lambda path: {"protocol": "fixture-harmony", "sha256": "fixed-vocabulary"},
    )
    image, cache = tmp_path / "image.sif", tmp_path / "cache"
    image.write_bytes(b"test image")
    snapshot = (
        cache
        / "hub/models--openai--gpt-oss-20b/snapshots"
        / submission.REVISION
        / "config.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    for key, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
        "AF_REPO_ROOT": str(submission.REPO),
        "AF_OUTPUT_ROOT": str(root),
    }.items():
        monkeypatch.setenv(key, value)
    return root, plan, image, cache


def test_single_submission_freezes_history_and_six_request_limits(setup, monkeypatch):
    root, plan, _, _ = setup
    seen = []

    def queue(directory, key, options, worker, stage, index):
        seen.append(options)
        assert key == stage == "propose" and index == 0
        assert (
            worker
            == submission.REPO / "scripts/hpc/run_function_handoff_repair_aces.sh"
        )
        return "1234"

    monkeypatch.setattr(submission, "submit_job", queue)
    result = submission.submit(root)
    assert submission.submit(root) == result
    assert len(seen) == 1
    assert result["maximum_cases"] == 8
    assert result["maximum_initializer_cases"] == 3
    assert result["maximum_generation_requests"] == 6
    assert result["maximum_charged_tokens"] == 196608
    assert result["automatic_followup"] is False
    assert {
        "--gres=gpu:h100:1",
        "--cpus-per-task=8",
        "--mem=64G",
        "--time=01:00:00",
        "--signal=B:TERM@300",
    } <= set(seen[0])
    assert not any(item.startswith("--dependency") for item in seen[0])
    current, execution = submission.verify(root)
    assert current == plan
    assert execution["plan_sha256"] == plan["artifact_sha256"]
    assert execution["source_archive_sha256"] == plan["source_archive_sha256"]
    assert execution["runtime_source_sha256"] == plan["runtime_source_sha256"]
    assert execution["worker_seconds"] == 1800
    assert execution["dynamics_generation_calls"] == execution["optimizer_calls"] == 0
    assert set(execution["entrypoint_sha256"]) == set(submission.ENTRYPOINTS)
    with pytest.raises(ValueError, match="one-job bound"):
        submission.submit(root, "another-wave")
    assert len(seen) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "protocol",
        "calls",
        "attempts",
        "output",
        "tokens",
        "model",
        "revision",
        "too_many",
        "duplicate",
        "wrong_action",
        "source",
        "source_null",
        "runtime",
        "fit",
        "dynamics",
        "test",
        "followup",
    ],
)
def test_invalid_or_expanded_plan_never_reaches_scheduler(setup, monkeypatch, mutation):
    root, plan, _, _ = setup
    settings = {
        "calls": "maximum_requests",
        "attempts": "attempts_per_step",
        "output": "max_output_tokens",
        "tokens": "maximum_total_tokens",
        "model": "model",
        "revision": "model_revision",
    }
    if mutation in settings:
        key = settings[mutation]
        value = plan["settings"][key]
        plan["settings"][key] = value + 1 if type(value) is int else "changed"
    elif mutation == "too_many":
        plan["cases"][3]["action"] = "retry_failed_initializer"
        plan["retry_case_ids"].append("proposal-3")
    elif mutation == "duplicate":
        plan["retry_case_ids"][1] = plan["retry_case_ids"][0]
    elif mutation == "wrong_action":
        plan["cases"][0]["action"] = "migrate_saved_initializer"
    elif mutation in {"source", "source_null", "runtime"}:
        plan[
            "runtime_source_sha256"
            if mutation == "runtime"
            else "source_archive_sha256"
        ] = None if mutation == "source_null" else "invalid"
    else:
        key = {
            "protocol": "protocol",
            "fit": "optimizer_calls",
            "dynamics": "dynamics_generation_calls",
            "test": "test_data_opened",
            "followup": "automatic_followup",
        }[mutation]
        plan[key] = "changed" if mutation == "protocol" else True
    monkeypatch.setattr(
        submission, "submit_job", lambda *args: pytest.fail("no scheduler")
    )
    with pytest.raises(ValueError):
        submission.submit(root)
    assert not (root / "execution.json").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_reply_cannot_be_retried_under_any_wave(setup, monkeypatch, timeout):
    root, _, _, _ = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if timeout:
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(
            argv,
            0,
            "\n",
            "sbatch: error: Batch job submission failed: Socket timed out",
        )

    monkeypatch.setattr(subprocess, "run", scheduler)
    with pytest.raises(ValueError, match=r"uncertain scheduler timeout|unconfirmed"):
        submission.submit(root)
    folder = root / "submissions/initializer-retry-1"
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    assert set(before) == {"propose.intent.json", "propose.reply.json"}
    for wave in ("initializer-retry-1", "alternate"):
        with pytest.raises(ValueError, match="one-job bound"):
            submission.submit(root, wave)
    assert len(calls) == 1
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before
    assert not (folder / "manifest.json").exists()


def test_image_source_and_execution_changes_fail_before_submission(setup, monkeypatch):
    root, plan, image, _ = setup
    monkeypatch.setattr(submission, "submit_job", lambda *args: "1234")
    submission.submit(root)
    plan["source_archive_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.submit(root)
    plan["source_archive_sha256"] = "a" * 64
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.submit(root)
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
    value = sealed_read(root / "execution.json")
    value["worker_seconds"] += 1
    (root / "execution.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact digest differs"):
        submission.verify(root)


@pytest.mark.parametrize("wave", ["../bad", "bad/wave", "bad wave", ""])
def test_invalid_wave_never_creates_submission(setup, wave):
    with pytest.raises(ValueError, match="simple identifier"):
        submission.submit(setup[0], wave)
    assert not (setup[0] / "submissions").exists()


def test_start_launcher_uses_new_root_source_and_explicit_single_wave(setup, tmp_path):
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
    wrapper = submission.REPO / "scripts/hpc/start_function_handoff_repair_aces.sh"
    subprocess.run(["bash", str(wrapper), "run"], check=True, env=environment)
    lines = arguments.read_text().splitlines()
    assert lines[lines.index("--source") + 1] == str(
        submission.REPO / "inputs/function-inspection.tar.gz"
    )
    assert lines[lines.index("--wave") + 1] == "initializer-retry-1"
    roots = [lines[i + 1] for i, item in enumerate(lines) if item == "--root"]
    assert roots == [str(tmp_path / "phase-c-function-handoff-repair-v1")] * 2
    assert (
        sum(item.endswith("submit_function_handoff_repair_aces.py") for item in lines)
        == 1
    )
    assert not any(item.endswith("construction_functions.py") for item in lines)


@pytest.mark.parametrize("action", ["report", "inspect"])
def test_read_only_launcher_actions_never_submit(setup, tmp_path, action):
    recorder = tmp_path / "record-python"
    recorder.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n')
    recorder.chmod(0o755)
    log = tmp_path / "arguments.txt"
    env = _shell_environment(tmp_path)
    env.update(AF_PYTHON=str(recorder), AF_ARGUMENT_LOG=str(log))
    subprocess.run(
        [
            "bash",
            str(submission.REPO / "scripts/hpc/start_function_handoff_repair_aces.sh"),
            action,
        ],
        check=True,
        env=env,
    )
    lines = log.read_text().splitlines()
    assert action in lines
    assert not any("submit_" in item for item in lines)


def test_batch_verifies_new_cli_before_starting_cuda(setup, tmp_path):
    root, _, image, cache = setup
    recorder = tmp_path / "record-python"
    recorder.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" >> "$AF_ARGUMENT_LOG"\n'
        '[[ "${2:-}" != --verify-only ]] || exit 17\n'
    )
    recorder.chmod(0o755)
    marker, arguments = tmp_path / "container-started", tmp_path / "arguments.txt"
    container = tmp_path / "apptainer"
    container.write_text('#!/bin/bash\ntouch "$AF_CONTAINER_STARTED"\n')
    container.chmod(0o755)
    env = _shell_environment(tmp_path)
    env.update(
        AF_PYTHON=str(recorder),
        AF_OUTPUT_ROOT=str(root),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(cache),
        AF_ARGUMENT_LOG=str(arguments),
        AF_CONTAINER_STARTED=str(marker),
        SLURM_JOB_ID="fixture",
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        AF_FUNCTION_CLI="wrong",
        AF_FUNCTION_SUBMIT_CLI="wrong",
        AF_FUNCTION_WORKER_SECONDS="99999",
    )
    wrapper = submission.REPO / "scripts/hpc/run_function_handoff_repair_aces.sh"
    result = subprocess.run(["bash", str(wrapper)], check=False, env=env)
    assert result.returncode == 17
    lines = arguments.read_text().splitlines()
    assert str(submission.REPO / "scripts/repair_function_handoff.py") in lines
    assert (
        str(submission.REPO / "scripts/submit_function_handoff_repair_aces.py") in lines
    )
    assert "wrong" not in lines
    assert not marker.exists()


@pytest.mark.parametrize("worker_status", [0, 23])
def test_batch_reuses_server_and_runs_only_bounded_repair_worker(
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
args = sys.argv[1:]
with open(os.environ['AF_ARGUMENT_LOG'], 'a') as stream:
    stream.write(json.dumps({'command':'python', 'args':args}) + '\\n')
if args[0] == '-c':
    print(os.environ['AF_IMAGE_SHA'] if 'image_sha256' in args[1] else '8765')
elif args[0] == '-':
    result = subprocess.run([sys.executable, *args], input=sys.stdin.read(), text=True)
    raise SystemExit(result.returncode)
elif len(args) > 1 and args[1] == 'run':
    raise SystemExit(int(os.environ['AF_WORKER_STATUS']))
""",
    )
    _executable(
        commands / "apptainer",
        """import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['AF_ARGUMENT_LOG'], 'a') as stream:
    stream.write(json.dumps({'command':'container', 'args':args}) + '\\n')
if 'serve' in args:
    Path(os.environ['AF_SERVER_STARTED']).touch()
    time.sleep(60)
elif any(item.endswith('harmony_vocab.py') for item in args):
    print(json.dumps({'harmony_render_passed':True}))
else:
    print(json.dumps({'visible_gpus':1,'gpu_name':'H100','vllm':'0.27.1',
                     'cuda_kernel_passed':True}))
""",
    )
    _executable(
        commands / "curl",
        """import os
from pathlib import Path
raise SystemExit(0 if Path(os.environ['AF_SERVER_STARTED']).exists() else 22)
""",
    )
    _executable(
        commands / "sha256sum",
        "import os,sys\nprint(os.environ['AF_IMAGE_SHA'] + '  ' + sys.argv[1])\n",
    )
    log = tmp_path / "arguments.jsonl"
    env = _shell_environment(tmp_path)
    env.update(
        AF_PYTHON=str(recorder),
        AF_OUTPUT_ROOT=str(root),
        AF_VLLM_IMAGE=str(image),
        AF_HF_HOME=str(cache),
        AF_ARGUMENT_LOG=str(log),
        AF_IMAGE_SHA=submission.file_hash(image),
        AF_SERVER_STARTED=str(tmp_path / "server-started"),
        AF_WORKER_STATUS=str(worker_status),
        SLURM_JOB_ID="fixture",
        PATH=str(commands) + os.pathsep + os.environ["PATH"],
        AF_FUNCTION_WORKER_SECONDS="99999",
    )
    wrapper = submission.REPO / "scripts/hpc/run_function_handoff_repair_aces.sh"
    result = subprocess.run(["bash", str(wrapper)], check=False, env=env, timeout=15)
    assert result.returncode == worker_status
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    workers = [row["args"] for row in calls if "run" in row["args"]]
    servers = [row["args"] for row in calls if "serve" in row["args"]]
    assert len(workers) == len(servers) == 1
    worker, server = workers[0], servers[0]
    assert worker[0] == str(submission.REPO / "scripts/repair_function_handoff.py")
    assert worker[worker.index("--seconds") + 1] == "1800"
    assert worker[worker.index("--runtime-record") + 1] == str(
        root / "runtime/probe-fixture.json"
    )
    assert worker[worker.index("--base-url") + 1] == "http://127.0.0.1:8765"
    assert server[server.index("--revision") + 1] == submission.REVISION
    assert server[server.index("--host") + 1] == "127.0.0.1"
    assert server[server.index("--max-model-len") + 1] == "32768"
    assert server[server.index("--tensor-parallel-size") + 1] == "1"
    runtime = json.loads((root / "runtime/probe-fixture.json").read_text())
    assert runtime["serving_identity"]["image_sha256"] == submission.file_hash(image)
    assert (
        root / "runtime/worker-exit-fixture.txt"
    ).read_text() == f"{worker_status}\n"


def test_shell_launchers_parse():
    for name in submission.ENTRYPOINTS:
        if name.endswith(".sh"):
            subprocess.run(["bash", "-n", str(submission.REPO / name)], check=True)
