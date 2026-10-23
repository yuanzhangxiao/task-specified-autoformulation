"""The revision launcher requests one bounded H100 allocation with no fit jobs."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import submit_construction_revision_aces as submission


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "campaign"
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": submission.PROTOCOL,
            "cases": [
                {"case_id": f"parent_{i}__revision_seed{s}"}
                for i in range(8)
                for s in range(3)
            ],
            "settings": deepcopy(submission.SETTINGS),
            "input_bundle_sha256": "a" * 64,
            "runtime_source_sha256": "b" * 64,
            "automatic_followup": False,
            "test_data_opened": False,
            "optimizer_calls": 0,
        },
    )
    monkeypatch.setattr(submission.campaign, "verify", lambda root: deepcopy(plan))
    monkeypatch.setattr(
        submission.harmony_vocab, "verify", lambda path: {"sha256": "test"}
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
    }.items():
        monkeypatch.setenv(key, value)
    return root, plan


def test_one_bounded_allocation_idempotent(setup, monkeypatch):
    root, plan = setup
    seen = []

    def queue(directory, key, options, worker, stage, index):
        seen.append(options)
        assert worker.name == "run_construction_revision_aces.sh"
        assert key == stage == "propose" and index == 0
        return "12345"

    monkeypatch.setattr(submission, "submit_job", queue)
    first = submission.submit(root)
    assert submission.submit(root) == first
    assert len(seen) == 1
    assert first["maximum_generation_requests"] == 72
    assert first["maximum_charged_tokens"] == 24 * 98304
    assert {"--gres=gpu:h100:1", "--time=02:00:00", "--mem=64G"} <= set(seen[0])
    current, execution = submission.verify(root)
    assert current == plan
    assert execution["worker_seconds"] == 5400
    assert execution["optimizer_calls"] == 0
    assert execution["automatic_followup"] is False
    with pytest.raises(ValueError, match="one-job bound"):
        submission.submit(root, "another-wave")


@pytest.mark.parametrize("timeout", [False, True])
def test_uncertain_submission_not_automatically_repeated(setup, monkeypatch, timeout):
    root, _ = setup
    from scripts import submit_review_continuation

    seen = []

    def scheduler(argv, **kwargs):
        seen.append(argv)
        if timeout:
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(argv, 0, "\n", "Socket timed out")

    monkeypatch.setattr(submit_review_continuation.subprocess, "run", scheduler)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        submission.submit(root)
    with pytest.raises(ValueError):
        submission.submit(root, "retry")
    assert len(seen) == 1
    assert (root / "submissions/revision-1/propose.intent.json").exists()


def test_expanded_requests_are_rejected(setup, monkeypatch):
    root, plan = setup
    plan["settings"]["maximum_requests"] = 4
    monkeypatch.setattr(submission, "submit_job", lambda *a: pytest.fail("no submit"))
    with pytest.raises(ValueError):
        submission.submit(root)


def test_shell_entrypoints_validate_and_use_shared_server():
    for name in (
        "start_construction_revision_aces.sh",
        "run_construction_revision_aces.sh",
    ):
        path = submission.REPO / "scripts/hpc" / name
        subprocess.run(["bash", "-n", str(path)], check=True)
    worker = (
        submission.REPO / "scripts/hpc/run_construction_revision_aces.sh"
    ).read_text()
    assert "run_construction_functions_aces.sh" in worker
    assert "AF_FUNCTION_WORKER_SECONDS=5400" in worker
    start = (
        submission.REPO / "scripts/hpc/start_construction_revision_aces.sh"
    ).read_text()
    assert "phase-c-construction-revision-v1" in start
    assert "inputs/revision-inputs.json" in start
    assert '"${AF_GCCCORE_MODULE:-GCCcore/13.2.0}"' in start


@pytest.fixture
def launcher(tmp_path):
    """Run the real shell dispatch with local assets and a recording Python stub."""
    group = tmp_path / "group scratch"
    old_root = group / "phase-c-construction-fitting-v1"
    old_root.mkdir(parents=True)
    old_inputs = old_root / "inputs.json"
    old_inputs.write_bytes(b'{"sha256":"old-fitting-seal","value":{}}\n')
    original = old_inputs.read_bytes()
    image, cache = tmp_path / "image.sif", tmp_path / "cache"
    image.write_bytes(b"local test image")
    snapshot = (
        cache
        / "hub/models--openai--gpt-oss-20b/snapshots"
        / submission.REVISION
        / "config.json"
    )
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("{}")
    log = tmp_path / "python-calls.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "record = {'argv': sys.argv[1:], "
        "'output_root': os.environ.get('AF_OUTPUT_ROOT'), "
        "'revision_root': os.environ.get('AF_REVISION_ROOT')}\n"
        "with open(os.environ['AF_TEST_CALL_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(record) + '\\n')\n"
        "if (Path(sys.argv[1]).name == 'construction_feedback_revision.py' "
        "and sys.argv[2] == 'prepare'):\n"
        "    sys.exit(int(os.environ.get('AF_TEST_PREPARE_EXIT', '0')))\n"
    )
    stub.chmod(0o755)

    def invoke(command, *, revision_root=None, prepare_exit=0, protocol=None):
        log.write_text("")
        env = {
            **os.environ,
            "AF_GROUP": str(group),
            "AF_OUTPUT_ROOT": str(old_root),
            "AF_PYTHON": str(stub),
            "AF_VLLM_IMAGE": str(image),
            "AF_HF_HOME": str(cache),
            "AF_TEST_CALL_LOG": str(log),
            "AF_TEST_PREPARE_EXIT": str(prepare_exit),
        }
        env.pop("AF_REVISION_ROOT", None)
        env.pop("AF_REVISION_PROTOCOL", None)
        if protocol is not None:
            env["AF_REVISION_PROTOCOL"] = protocol
        if revision_root is not None:
            env["AF_REVISION_ROOT"] = str(revision_root)
        result = subprocess.run(
            [
                "bash",
                str(
                    submission.REPO / "scripts/hpc/start_construction_revision_aces.sh"
                ),
                command,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert old_inputs.read_bytes() == original
        return result, [json.loads(line) for line in log.read_text().splitlines()]

    return invoke, group


def test_v2_launcher_uses_fresh_root_and_authenticated_evidence(launcher):
    invoke, group = launcher
    result, calls = invoke("run", protocol=submission.campaign.V2_PROTOCOL)
    assert result.returncode == 0, result.stderr
    expected = str(group / "phase-c-construction-revision-checks-v2")
    assert all(call["revision_root"] == expected for call in calls)
    prepare = next(c["argv"] for c in calls if "prepare" in c["argv"])
    assert prepare[prepare.index("--protocol") + 1] == submission.campaign.V2_PROTOCOL
    assert prepare[prepare.index("--consistency-evidence") + 1].endswith(
        "/inputs/consistency-evidence.json"
    )


def test_v2_execution_freezes_acceptance_policy(setup, monkeypatch):
    root, plan = setup
    plan.update(
        protocol=submission.campaign.V2_PROTOCOL,
        policies=deepcopy(submission.campaign.V2_POLICIES),
        consistency_evidence_sha256="c" * 64,
    )
    seen = []

    def queue(directory, key, options, worker, stage, index):
        seen.extend(options)
        return "12345"

    monkeypatch.setattr(submission, "submit_job", queue)
    submission.submit(root)
    _, execution = submission.verify(root)
    assert execution["protocol"] == "construction-revision-aces-execution-2"
    assert execution["acceptance_policies"] == plan["policies"]
    assert execution["consistency_evidence_sha256"] == "c" * 64
    assert "--job-name=phasec-revision-checks" in seen


def test_v3_launcher_keeps_science_run_separate(launcher):
    invoke, group = launcher
    result, calls = invoke("run", protocol=submission.campaign.V3_PROTOCOL)
    assert result.returncode == 0, result.stderr
    expected = str(group / "phase-c-function-science-checks-v1")
    assert all(call["revision_root"] == expected for call in calls)
    prepare = next(c["argv"] for c in calls if "prepare" in c["argv"])
    assert prepare[prepare.index("--protocol") + 1] == submission.campaign.V3_PROTOCOL
    assert "--consistency-evidence" in prepare


def test_v3_execution_freezes_scientific_acceptance_policy(setup, monkeypatch):
    root, plan = setup
    plan.update(
        protocol=submission.campaign.V3_PROTOCOL,
        policies=deepcopy(submission.campaign.V3_POLICIES),
        consistency_evidence_sha256="c" * 64,
    )
    seen = []

    def queue(directory, key, options, worker, stage, index):
        seen.extend(options)
        return "12345"

    monkeypatch.setattr(submission, "submit_job", queue)
    submission.submit(root)
    _, execution = submission.verify(root)
    assert execution["protocol"] == "construction-revision-aces-execution-3"
    assert execution["acceptance_policies"] == plan["policies"]
    assert execution["acceptance_policies"]["endpoint_scores_in_prompt"] is False
    assert "--job-name=phasec-function-checks" in seen
    assert execution["optimizer_calls"] == 0


@pytest.mark.parametrize("command", ["run", "report", "inspect"])
@pytest.mark.parametrize("explicit_root", [False, True])
def test_launcher_isolates_revision_root_from_stale_generic_root(
    launcher, command, explicit_root
):
    invoke, group = launcher
    override = group / "custom revision" if explicit_root else None
    expected = str(override or group / "phase-c-construction-revision-v1")
    result, calls = invoke(command, revision_root=override)
    assert result.returncode == 0, result.stderr
    assert calls
    assert all(call["output_root"] == expected for call in calls)
    assert all(call["revision_root"] == expected for call in calls)
    rooted = [call["argv"] for call in calls if "--root" in call["argv"]]
    assert len(rooted) == (2 if command == "run" else 1)
    assert all(args[args.index("--root") + 1] == expected for args in rooted)
    if command == "run":
        assert Path(rooted[0][0]).name == "construction_feedback_revision.py"
        assert rooted[0][1] == "prepare"
        assert Path(rooted[1][0]).name == "submit_construction_revision_aces.py"
    else:
        assert Path(rooted[0][0]).name == "construction_feedback_revision.py"
        assert rooted[0][1] == command


def test_launcher_prepare_failure_never_reaches_submission(launcher):
    invoke, group = launcher
    result, calls = invoke("run", prepare_exit=37)
    assert result.returncode == 37
    assert [Path(call["argv"][0]).name for call in calls] == [
        "construction_feedback_revision.py",
        "harmony_vocab.py",
        "construction_feedback_revision.py",
    ]
    assert calls[-1]["argv"][1] == "prepare"
    assert calls[-1]["output_root"] == str(group / "phase-c-construction-revision-v1")


def test_execution_identity_cannot_be_changed(setup, monkeypatch):
    root, _ = setup
    monkeypatch.setattr(submission, "submit_job", lambda *a: "12345")
    submission.submit(root)
    path = root / "execution.json"
    value = json.loads(path.read_text())
    value["maximum_sequences"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        submission.verify(root)


def test_v4_fixed_six_saved_models_three_seeds_bound(setup, monkeypatch):
    root, plan = setup
    plan.update(
        protocol=submission.campaign.V4_PROTOCOL,
        policies=deepcopy(submission.campaign.V4_POLICIES),
        consistency_evidence_sha256="c" * 64,
    )
    plan["cases"] = plan["cases"][:18]
    monkeypatch.setattr(submission, "submit_job", lambda *args: "12345")
    outcome = submission.submit(root)
    assert outcome["maximum_generation_requests"] == 54
    assert outcome["maximum_charged_tokens"] == 18 * 98304
    _, execution = submission.verify(root)
    assert execution["protocol"] == "construction-revision-aces-execution-4"
    assert execution["acceptance_policies"] == submission.campaign.V4_POLICIES
    plan["cases"].append({"case_id": "extra"})
    with pytest.raises(ValueError):
        submission.verify_inputs(root)
