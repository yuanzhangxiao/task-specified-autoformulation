"""A pinned historical executor and one isolated CPU allocation, never an H100."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.test_prefit_numerical_sibling_submission import launcher  # noqa: F401


@pytest.fixture
def submitter(request, tmp_path):
    _, log, root, env = request.getfixturevalue("launcher")
    source = Path(env["AF_SOURCE_ROOT"])
    history = tmp_path / "history"
    history.mkdir()
    (history / "plan.json").write_text(
        json.dumps(
            {
                "paths": {
                    "source": str(source),
                    "construction": env["AF_CONSTRUCTION_ROOT"],
                    "parent": env["AF_PARENT_FIT"],
                    "continuation": env["AF_CONTINUATION_FIT"],
                }
            }
        )
    )
    (source / "plan.json").write_text(json.dumps({"source_root": str(history)}))
    (source / "child_fit").mkdir()
    for name in ("freeze.json", "result.json", "backend_result.json"):
        (source / "child_fit" / name).write_text("{}")
    env.update(AF_FIT_REPO=env["AF_REPO_ROOT"], AF_FIT_COMMIT="abc123")
    script = (
        Path(env["AF_REPO_ROOT"]) / "scripts/hpc/submit_prefit_matched_refit_aces.sh"
    )

    def run(worker=False, **overrides):
        return subprocess.run(
            ["bash", str(script), *(["worker"] if worker else [])],
            env={**env, **overrides},
            text=True,
            capture_output=True,
        )

    return run, log, root, source


def test_cpu_submission_is_idempotent(submitter):
    run, log, _, _ = submitter
    first = run()
    assert first.returncode == 0, first.stderr
    assert run().stdout == first.stdout
    calls = json.loads(log.read_text())
    assert len(calls) == 1
    assert "--partition=cpu" in calls[0] and "--cpus-per-task=1" in calls[0]
    assert not any("gpu" in word for word in calls[0])


def test_wrong_pinned_executor_prevents_submission(submitter):
    run, log, _, _ = submitter
    result = run(AF_FIT_COMMIT="different")
    assert result.returncode != 0 and "fitter commit differs" in result.stderr
    assert not log.exists()


def test_uncertain_submission_cannot_be_duplicated(submitter):
    run, log, _, _ = submitter
    assert run(FAIL_STAGE="worker").returncode != 0
    assert run().returncode != 0
    assert len(json.loads(log.read_text())) == 1


def test_queued_child_change_rejected_before_work(submitter):
    run, _, root, source = submitter
    assert run().returncode == 0
    (source / "child_fit/result.json").write_text('{"changed":true}')
    result = run(worker=True)
    assert result.returncode != 0 and "Queued identity changed" in result.stderr
    assert not (root / "tmp").exists()


def test_source_overlap_fails_before_logs(submitter):
    run, log, _, source = submitter
    destination = source / "nested-output"
    result = run(AF_OUTPUT_ROOT=str(destination))
    assert result.returncode != 0 and "separate" in result.stderr
    assert not destination.exists() and not log.exists()
