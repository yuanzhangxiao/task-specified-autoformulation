"""CPU-only submission cannot duplicate allocations or alter historical output."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_prefit_numerical_sibling_submission import launcher  # noqa: F401


@pytest.fixture
def submitter(request):
    _, log, root, env = request.getfixturevalue("launcher")
    source = Path(env["AF_SOURCE_ROOT"])
    (source / "results").mkdir()
    (source / "results/state.json").write_text("{}")
    script = (
        Path(env["AF_REPO_ROOT"]) / "scripts/hpc/submit_prefit_parameter_replay_aces.sh"
    )

    def run(**overrides):
        return subprocess.run(
            ["bash", str(script)],
            env={**env, **overrides},
            text=True,
            capture_output=True,
        )

    return run, log, root, source


def test_replay_and_fit_are_separate_cpu_allocations(submitter):
    run, log, root, _ = submitter
    assert run(AF_REPLAY_ACTION="fit").returncode != 0
    result = run()
    assert result.returncode == 0, result.stderr
    assert run().stdout == result.stdout
    calls = json.loads(log.read_text())
    assert len(calls) == 1 and "--partition=cpu" in calls[0]
    assert not any("gpu" in word for word in calls[0])
    (root / "plan.json").write_text("{}")
    assert run(AF_REPLAY_ACTION="fit").returncode == 0
    assert run(AF_REPLAY_ACTION="fit").returncode == 0
    assert len(json.loads(log.read_text())) == 2


def test_uncertain_submission_does_not_submit_again(submitter):
    run, log, _, _ = submitter
    assert run(FAIL_STAGE="worker").returncode != 0
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == 1


def test_changed_state_does_not_reuse_submission_identity(submitter):
    run, log, _, source = submitter
    assert run().returncode == 0
    (source / "results/state.json").write_text('{"changed":true}')
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == 1


def test_overlap_fails_before_creating_any_files(submitter):
    run, log, _, source = submitter
    output = source / "new-output"
    assert run(AF_OUTPUT_ROOT=str(output)).returncode != 0
    assert not output.exists() and not log.exists()
