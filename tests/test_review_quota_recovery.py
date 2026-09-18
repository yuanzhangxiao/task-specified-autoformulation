"""Quota recovery replays saved points, preserves costs, and never refits."""

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoformalism.data import TrainingScaler
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.rebuttal import review_continuation as continuation
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import (
    PublicFitRequest,
    PublicFitResult,
    PublicSplit,
)
from scripts import recover_review_quota as recovery
from tests.test_review_continuation import source_fixture


def fixture(source: Path, *, point=True):
    """Use the existing synthetic campaign and a deliberately worse incumbent."""
    plan = source_fixture(source)
    (source / "execution.lock").touch()
    task = plan["tasks"][0]
    parent = io.read_round(source, task, 2)
    selected = parent["selected"]
    # Start with complete round 1 and missing round 2 for this task only.
    for other in plan["tasks"]:
        value = io.read_round(source, other, 2)
        sealed_write(
            io.round_path(source, other, 1) / "result.json",
            {
                **{k: v for k, v in value.items() if k != "artifact_sha256"},
                "round": 1,
            },
        )
    directory = io.round_path(source, task, 2)
    (directory / "result.json").unlink()
    request = pipeline.request_for(selected["bundle"], plan, task, 2)
    cell = plan["cells"][task["cell"]]
    training, validation = (
        PublicSplit.model_validate(cell[n]) for n in ("training", "validation")
    )
    parent = io.read_round(source, task, 1)
    proposal = sealed_write(
        directory / "proposal.json",
        {
            "task": task,
            "round": 2,
            "parent_sha256": parent["artifact_sha256"],
            "bundle": selected["bundle"],
            "status": "committed",
            "cost": {"physical_requests": 0},
        },
    )
    fit = directory / "fit"
    frozen = sibling_fit.prepare_child_fit(
        PublicFitRequest.model_validate(selected["request"]),
        request,
        selected["fit"]["parameters"],
        training,
        validation,
        fit,
        lineage={
            "campaign": plan["artifact_sha256"],
            "proposal": proposal["artifact_sha256"],
        },
        allow_initialization_changes=True,
    )
    public._write(fit / "started.json", {"identity": frozen["identity"]})
    public._write(directory / "worker_started.json", {"interrupted": True})
    model, _, _ = public._lower(request)
    train = public.unpack_split(training)
    scalers = TrainingScaler().fit(train).scales
    scales = {
        k: scalers[f"target:{k}"].standard_deviation for k in request.context.targets
    }
    params = selected["fit"]["parameters"]
    tr = recovery.fixed_replay(model, train, params, scales)
    best = fit / "backend/recovery/augmented_0/best_evaluated.json"
    best.parent.mkdir(parents=True)
    if point:
        public._write(best, {"parameters": params, "cost": tr["cost"], "call": 7})
    return plan, best


def forbid_optimization(*args, **kwargs):
    pytest.fail("quota recovery must never invoke optimization")


def test_complete_recovery_reuses_checkpoint_and_preserves_original(
    tmp_path, monkeypatch
):
    source, root = tmp_path / "source", tmp_path / "recovery"
    plan, _ = fixture(source)
    original = recovery.inventory(source)
    monkeypatch.setattr(public, "_run_backend", forbid_optimization)
    monkeypatch.setattr(pipeline, "fit_one", forbid_optimization)
    manifest = recovery.prepare(source, root, 2, [0])
    assert recovery.prepare(source, root, 2, [0]) == manifest
    # Exercise real production replay, including learned causal initial maps.
    result = recovery.replay_inner(root, 0)
    assert result["status"] == "replayed"
    assert result["replayed_cost"] == manifest["entries"][0]["point"]["cost"]
    trial = result["trial"]
    assert trial["fit"]["native_optimizer_converged"] is None
    assert trial["fit"]["actual_residual_calls"] is None
    assert (
        trial["packet"]["numerical_status"]["feedback_status"]
        == "numerical_failure_unresolved"
    )
    assert trial["fit"]["parameters"] == manifest["entries"][0]["point"]["parameters"]
    receipt = {
        **result,
        "identity": {
            "manifest_sha256": manifest["artifact_sha256"],
            "entry": manifest["entries"][0],
        },
        "optimizer_calls": 0,
        "test_data_opened": False,
    }
    sealed_write(root / "quota_replays/0/result.json", receipt)
    summary = recovery.finalize(root)
    assert recovery.finalize(root) == summary
    assert recovery.inventory(source) == original
    assert recovery.worker(root, 0)["status"] == "replayed"
    saved = io.read_round(root, plan["tasks"][0], 2)
    assert saved["cost"] == {"physical_requests": 0}
    assert saved["recovery"]["optimizer_calls"] == 0
    assert saved["selected"] == min(
        [io.read_round(root, plan["tasks"][0], 1)["selected"], trial],
        key=pipeline.selection_key,
    )
    # Standard importer accepts recovery, so no custom numerical protocol is needed.
    new = continuation.prepare(
        root, tmp_path / "next", source_round=2, visits=1, protocol=io.REVISION_PROTOCOL
    )
    assert new["continuation"]["additional_visits"] == 1
    assert new["tasks"] == plan["tasks"]
    fit = sibling_fit.inspect_child_fit(
        io.round_path(root, plan["tasks"][0], 2) / "fit"
    )
    PublicFitResult.model_validate(fit["result"])


def test_no_point_retains_parent_without_replay_or_new_budget(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    plan, _ = fixture(source, point=False)
    recovery.prepare(source, root, 2, [0])
    monkeypatch.setattr(recovery.subprocess, "run", forbid_optimization)
    value = recovery.worker(root, 0)
    assert value["status"] == "no_saved_point"
    recovery.finalize(root)
    value = io.read_round(root, plan["tasks"][0], 2)
    assert value["status"] == "worker_interrupted" and value["trial"] is None
    assert value["selected"] == io.read_round(root, plan["tasks"][0], 1)["selected"]
    assert not (io.round_path(root, plan["tasks"][0], 2) / "fit/result.json").exists()


def test_started_replay_is_consumed_and_resume_never_calls_optimizer(
    tmp_path, monkeypatch
):
    source, root = tmp_path / "source", tmp_path / "recovery"
    fixture(source)
    manifest = recovery.prepare(source, root, 2, [0])
    identity = {
        "manifest_sha256": manifest["artifact_sha256"],
        "entry": manifest["entries"][0],
    }
    sealed_write(root / "quota_replays/0/started.json", {"identity": identity})
    monkeypatch.setattr(recovery.subprocess, "run", forbid_optimization)
    assert recovery.worker(root, 0)["status"] == "replay_interrupted"
    assert recovery.worker(root, 0)["fresh_budget_on_resume"] is False


def test_child_receipt_roundtrip_and_hard_timeout(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    fixture(source)
    manifest = recovery.prepare(source, root, 2, [0])
    identity = {
        "manifest_sha256": manifest["artifact_sha256"],
        "entry": manifest["entries"][0],
    }

    def fake_child(*args, **kwargs):
        assert kwargs["timeout"] == recovery.WORKER_SECONDS
        sealed_write(
            root / "quota_replays/0/inner.json",
            {"status": "training_cost_disagrees", "identity": identity},
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(recovery.subprocess, "run", fake_child)
    value = recovery.worker(root, 0)
    assert sealed_read(root / "quota_replays/0/result.json") == value
    assert recovery.worker(root, 0) == value


def test_completed_child_is_retained_after_supervisor_interruption(
    tmp_path, monkeypatch
):
    source, root = tmp_path / "source", tmp_path / "recovery"
    fixture(source)
    manifest = recovery.prepare(source, root, 2, [0])
    identity = {
        "manifest_sha256": manifest["artifact_sha256"],
        "entry": manifest["entries"][0],
    }
    sealed_write(root / "quota_replays/0/started.json", {"identity": identity})
    sealed_write(
        root / "quota_replays/0/inner.json",
        {
            **recovery.replay_inner(root, 0),
            "identity": identity,
        },
    )
    monkeypatch.setattr(recovery.subprocess, "run", forbid_optimization)
    result = recovery.worker(root, 0)
    assert result["status"] == "replayed"
    assert recovery.worker(root, 0) == result


def test_hard_timeout_is_terminal_and_never_repeated(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    fixture(source)
    recovery.prepare(source, root, 2, [0])
    calls = []

    def timeout(cmd, **kwargs):
        calls.append(cmd)
        raise recovery.subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(recovery.subprocess, "run", timeout)
    first = recovery.worker(root, 0)
    assert first["status"] == "replay_timeout"
    assert recovery.worker(root, 0) == first
    assert len(calls) == 1


def test_training_cost_mismatch_never_reads_validation(tmp_path, monkeypatch):
    source, root = tmp_path / "source", tmp_path / "recovery"
    _, best = fixture(source)
    changed = public._read(best)
    changed["cost"] += 100
    public._write(best, changed)
    recovery.prepare(source, root, 2, [0])
    replay = recovery.fixed_replay
    calls = []

    def checked(model, split, *args):
        calls.append(split.name.value)
        assert split.name.value == "train"
        return replay(model, split, *args)

    monkeypatch.setattr(recovery, "fixed_replay", checked)
    assert recovery.replay_inner(root, 0)["status"] == "training_cost_disagrees"
    assert calls == ["train"]


@pytest.mark.parametrize(
    "kind", ["nonfinite", "incomplete", "out_of_bounds", "malformed"]
)
def test_corrupt_or_ineligible_checkpoint_not_selected(tmp_path, kind):
    _, best = fixture(tmp_path)
    value = public._read(best)
    name = next(iter(value["parameters"]))
    if kind == "nonfinite":
        value["parameters"][name] = float("nan")
    elif kind == "incomplete":
        value["parameters"].pop(name)
    elif kind == "out_of_bounds":
        value["parameters"]["rate"] = -1
    else:
        value = {}
    best.write_text(json.dumps(value))
    points, rejected = recovery.point_candidates(best.parents[3])
    assert not points and len(rejected) == 1


def test_saved_cost_order_and_collocation_point_exclusion(tmp_path):
    _, best = fixture(tmp_path)
    fit = best.parents[3]
    value = public._read(best)
    another = fit / "backend/recovery/primal_screen/best_evaluated.json"
    another.parent.mkdir()
    public._write(another, {**value, "cost": value["cost"] + 1})
    public._write(
        fit / "backend/initializer.json",
        {"parameters": value["parameters"], "success": True},
    )
    points, _ = recovery.point_candidates(fit)
    assert len(points) == 2 and points[0]["path"].endswith(
        "augmented_0/best_evaluated.json"
    )


def test_tampering_and_wrong_missing_set_fail_closed(tmp_path):
    source, root = tmp_path / "source", tmp_path / "recovery"
    _, best = fixture(source)
    with pytest.raises(ValueError, match="indices differ"):
        recovery.prepare(source, root, 2, [1])
    recovery.prepare(source, root, 2, [0])
    copied = root / best.relative_to(source)
    raw = copy.deepcopy(public._read(copied))
    raw["cost"] += 1
    public._write(copied, raw)
    with pytest.raises(ValueError, match="copied source changed"):
        recovery.verify_snapshot(root)


def test_finalize_requires_all_replay_receipts(tmp_path):
    source, root = tmp_path / "source", tmp_path / "recovery"
    fixture(source)
    recovery.prepare(source, root, 2, [0])
    with pytest.raises(FileNotFoundError):
        recovery.finalize(root)


def test_pending_later_round_prevents_competing_continuation(tmp_path):
    source, root = tmp_path / "source", tmp_path / "recovery"
    plan, _ = fixture(source)
    changed = {k: v for k, v in plan.items() if k != "artifact_sha256"}
    changed["config"] = {**changed["config"], "rounds": 4}
    (source / "plan.json").unlink()
    sealed_write(source / "plan.json", changed)
    (source / "submission-intent/round-3").mkdir(parents=True)
    with pytest.raises(ValueError, match="later round submission exists"):
        recovery.prepare(source, root, 2, [0])


@pytest.mark.parametrize("bad_reply", [False, True])
def test_submission_receipts_prevent_duplicate_jobs(tmp_path, bad_reply):
    """Exercise the real shell orchestration with a synthetic scheduler."""
    tools = tmp_path / "bin"
    tools.mkdir()
    root = tmp_path / "recovery"
    root.mkdir()
    for name, body in {
        "module": "exit 0",
        "git": "echo dfc6f81a613e186ddffdd0b2406feca583a58944",
        "squeue": "exit 0",
        "sbatch": (
            'echo "$*" >> "$AF_RECOVERY_ROOT/submissions.txt"\n'
            + ("echo ambiguous" if bad_reply else "echo 1234567")
        ),
    }.items():
        path = tools / name
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "AF_REPO_ROOT": str(tmp_path),
        "AF_PYTHON": "/usr/bin/true",
        "AF_SOURCE_ROOT": str(tmp_path / "source"),
        "AF_RECOVERY_ROOT": str(root),
        "AF_RECOVERY_CODE": str(tmp_path),
        "USER": "synthetic-user",
    }
    script = (
        Path(__file__).parents[1] / "scripts/hpc/submit_review_quota_recovery_aces.sh"
    )
    first = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert first.returncode == (2 if bad_reply else 0), first.stderr
    requests = (root / "submissions.txt").read_text().splitlines()
    assert len(requests) == (1 if bad_reply else 2)
    assert "--array=0-7%8" in requests[0] and "--cpus-per-task=1" in requests[0]
    assert (root / "quota_submission/replay.out").exists()
    again = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert again.returncode == 2
    assert (root / "submissions.txt").read_text().splitlines() == requests
