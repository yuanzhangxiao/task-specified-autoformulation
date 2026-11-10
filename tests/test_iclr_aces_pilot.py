"""Operational evidence, scope and scheduling checks for the separate ACES pilot."""

import copy
from pathlib import Path

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import review_deadline_io as io
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from scripts import iclr_aces_pilot as pilot
from scripts.iclr_anchor_fidelity import verify_controlled_bundle


def initial_pair(root: Path, monkeypatch, *, retained: bool = True) -> dict:
    """Prepare the same immutable results and shared-start contract as a worker."""
    block = {
        "cell": "acquisition_00",
        "seed": 97,
        "block": 0,
        "tasks": [
            {"task_id": arm, "treatment": arm, "critic": True}
            for arm in pilot.PILOT_ARMS
        ],
    }
    full, fixed = block["tasks"]
    first = io.round_path(root, full, 0)
    fit = {"parameters": {"k": 2.0}, "training": {"normalized_mse": 0.02}}
    selected = (
        {
            "origin_task": "full",
            "bundle": {"candidate": {}, "initialization": {}},
            "fit": fit,
        }
        if retained
        else None
    )
    sealed_write(
        first / "result.json",
        {
            "task": full,
            "round": 0,
            "status": "complete" if retained else "fit_failed",
            "selected": selected,
            "trial": None,
            "cost": {},
            "proposal_sha256": "proposal",
        },
    )
    if retained:
        (first / "fit").mkdir()
        public._write(
            first / "fit/result.json",
            {"sha256": public.content_sha256(fit), "result": fit},
        )
        sealed_write(
            first / "critic.json", {"status": "reviewed", "proposal_sha256": "proposal"}
        )
    monkeypatch.setattr(pilot.controls, "gamma", lambda _: {"example": True})
    pilot.study.share_initial(root, full, fixed)
    return block


def test_shared_start_records_verified_fit_and_detects_changes(tmp_path, monkeypatch):
    block = initial_pair(tmp_path, monkeypatch)
    result = pilot.shared_start(tmp_path, block)
    assert result["available"]
    assert result["parameters_sha256"] == public.content_sha256({"k": 2.0})
    fixed_path = io.round_path(tmp_path, block["tasks"][1], 0) / "fit/result.json"
    public._write(fixed_path, {"result": {}})
    with pytest.raises(ValueError, match="fitter receipt differs"):
        pilot.shared_start(tmp_path, block)


def test_shared_initial_failure_retains_both_receipts(tmp_path, monkeypatch):
    failed = pilot.shared_start(
        tmp_path, initial_pair(tmp_path, monkeypatch, retained=False)
    )
    assert failed["available"] is False
    assert failed["full_round_sha256"] and failed["fixed_round_sha256"]


def test_scientific_evaluators_exclude_engineering_pilot(tmp_path):
    sealed_write(tmp_path / "plan.json", {"development_pilot": True})
    with pytest.raises(ValueError, match="excluded from scientific evaluation"):
        verify_controlled_bundle(tmp_path / "frozen_models.json", {})


def test_pilot_finish_requires_engineering_plan(tmp_path):
    sealed_write(tmp_path / "plan.json", {"protocol": "not-the-pilot"})
    with pytest.raises(ValueError, match="declared two-visit"):
        pilot.finish(tmp_path)


@pytest.fixture
def cached_visit(tmp_path, monkeypatch):
    block = initial_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(public, "_source_identity", lambda: "source")
    monkeypatch.setattr(public, "_runtime", lambda: {})
    plan = sealed_write(
        tmp_path / "plan.json",
        {
            "protocol": pilot.study.PROTOCOL,
            "development_pilot": True,
            "arms": pilot.PILOT_ARMS,
            "visits": 2,
            "critic_enabled": True,
            "pruning_enabled": True,
            "blocks": [block],
            "source_sha256": "source",
            "runtime": {},
        },
    )
    sealed_write(
        tmp_path / "critic_authorization.json",
        {
            "identity": plan["artifact_sha256"],
            "known_case_gate_passed": True,
        },
    )
    full, fixed = block["tasks"]
    sealed_write(
        io.round_path(tmp_path, full, 1) / "result.json", {"task": full, "round": 1}
    )
    directory = io.round_path(tmp_path, fixed, 1)
    proposal = sealed_write(directory / "proposal.json", {"task": fixed, "round": 1})
    sealed_write(
        directory / "critic.json",
        {
            "task": fixed,
            "round": 1,
            "identity": plan["artifact_sha256"],
            "proposal_sha256": proposal["artifact_sha256"],
        },
    )
    return directory


@pytest.mark.parametrize("problem", ["missing", "mismatched", "quota"])
def test_cached_resume_refuses_missing_receipts_or_unwritable_fit(
    tmp_path, cached_visit, monkeypatch, problem
):
    path = cached_visit / "critic.json"
    if problem == "quota":
        real = pilot.tempfile.NamedTemporaryFile

        def guarded(**kwargs):
            if kwargs["dir"] == cached_visit / "fit":
                raise OSError(122, "Disk quota exceeded")
            return real(**kwargs)

        monkeypatch.setattr(pilot.tempfile, "NamedTemporaryFile", guarded)
        with pytest.raises(OSError, match="Disk quota exceeded"):
            pilot.check_cached_pilot(tmp_path)
    else:
        review = sealed_read(path)
        path.unlink()
        if problem == "mismatched":
            review.pop("artifact_sha256")
            sealed_write(path, {**review, "proposal_sha256": "other"})
        with pytest.raises(ValueError, match=r"missing saved|lineage differs"):
            pilot.check_cached_pilot(tmp_path)
    assert not list(tmp_path.rglob(".resume-write-*"))


def test_cached_worker_blocks_proposer_and_uses_cpu_accounting(
    tmp_path, cached_visit, monkeypatch
):
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "fixture")
    submission = []

    def fake_submit(directory, key, options, worker, stage, index):
        submission.append((options, worker.name))
        (directory / "array.id").write_text("1234\n")
        return "1234"

    monkeypatch.setattr(pilot, "submit_job", fake_submit)
    monkeypatch.setattr(pilot, "pending_ids", lambda root: [0])
    assert pilot.submit(tmp_path, "cpu-wave", "test", 1, cached_pilot=True) == "1234"
    assert pilot.submit(tmp_path, "cpu-wave", "test", 1, cached_pilot=True) == "1234"
    assert len(submission) == 1
    assert "--partition=cpu" in submission[0][0] and "--array=0%1" in submission[0][0]
    assert submission[0][1] == "run_iclr_cached_pilot_aces.sh"

    def run_block(root, number, url, seconds, *, transport, token_transport):
        for send in (transport, token_transport):
            with pytest.raises(RuntimeError, match="new proposer call"):
                send()
        return {"status": "complete", "block": number}

    monkeypatch.setattr(pilot.study, "run_block", run_block)
    assert pilot.run_cached_pilot(tmp_path, 30)["status"] == "complete"
    monkeypatch.delenv("AF_JETSTREAM_API_KEY")
    with pytest.raises(ValueError, match="critic credential"):
        pilot.run_cached_pilot(tmp_path, 30)


def test_cached_preflight_waits_for_active_worker(tmp_path, cached_visit):
    with (
        io.execution_lease(tmp_path),
        pytest.raises(ValueError, match="active campaign"),
    ):
        pilot.check_cached_pilot(tmp_path)
    assert not (tmp_path / "pruning").exists()


def test_submission_resume_and_uncertain_slurm_reply(tmp_path, monkeypatch):
    plan = sealed_write(tmp_path / "plan.json", {"protocol": pilot.study.PROTOCOL})
    sealed_write(
        tmp_path / "critic_authorization.json",
        {
            "identity": plan["artifact_sha256"],
            "known_case_gate_passed": True,
        },
    )
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "fixture")
    monkeypatch.setenv("USER", "fixture")
    monkeypatch.setattr(pilot, "pending_ids", lambda _: [0, 3])
    submitted = []

    def submit_job(directory, key, args, worker, stage, index):
        submitted.append(copy.copy(args))
        (directory / "array.id").write_text("1234\n")
        return "1234"

    monkeypatch.setattr(pilot, "submit_job", submit_job)
    assert pilot.submit(tmp_path, "wave1", "12345", 2) == "1234"
    assert pilot.submit(tmp_path, "wave1", "12345", 2) == "1234"
    assert len(submitted) == 1 and "--array=0,3%2" in submitted[0]
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda *a, **kw: "1234\n")
    with pytest.raises(ValueError, match="active"):
        pilot.submit(tmp_path, "wave2", "12345", 2)
    absent = tmp_path / "submissions/interrupted"
    absent.mkdir()
    public._write(absent / "array.intent.json", {"argv": []})
    with pytest.raises(ValueError, match="uncertain Slurm"):
        pilot.submit(tmp_path, "wave3", "12345", 2)


def test_fake_provider_data_cannot_qualify_live_pilot(tmp_path, monkeypatch):
    block = initial_pair(tmp_path, monkeypatch)
    plan = sealed_write(
        tmp_path / "plan.json",
        {
            "protocol": pilot.study.PROTOCOL,
            "development_pilot": True,
            "arms": pilot.PILOT_ARMS,
            "visits": 2,
            "critic_enabled": True,
            "pruning_enabled": True,
            "blocks": [block],
            "model_settings": {"model_revision": "demo"},
            "runtime": {},
            "source_sha256": "source",
            "serving_image_sha256": "image",
        },
    )
    sealed_write(
        tmp_path / "critic_authorization.json",
        {
            "identity": plan["artifact_sha256"],
            "known_case_gate_passed": True,
        },
    )
    for task in block["tasks"]:
        sealed_write(
            tmp_path / "pruning" / task["task_id"] / "result.json",
            {"status": "complete"},
        )
    calls = io.round_path(tmp_path, block["tasks"][1], 1) / "calls"
    calls.mkdir(parents=True)
    public._write(
        calls / "request.json",
        {
            "status": "responded",
            "endpoint": "http://fixture",
            "request": {
                "body": {"frozen_dynamic_representation": {}},
                "settings": {"model_revision": "demo"},
            },
        },
    )
    monkeypatch.setattr(pilot, "pending_ids", lambda _: [])
    monkeypatch.setattr(pilot.study, "run_block", lambda *a, **kw: None)
    monkeypatch.setattr(
        pilot.study,
        "report",
        lambda *a: {"artifact_sha256": "models", "rows": [{"cost": {}}]},
    )
    monkeypatch.setattr(
        pilot,
        "pair_receipts",
        lambda _: {"artifact_sha256": "pair", "pairs": [{"available": True}]},
    )
    result = pilot.finish(tmp_path)
    assert not result["qualified"] and result["proposer_responses"] == 0
    assert not result["checks"]["complete_live_critic_review"]
    assert sealed_read(tmp_path / "pilot_receipt.json") == pilot.finish(tmp_path)


def test_reprepared_design_checks_data_and_all_scientific_fields(tmp_path):
    folder = tmp_path / "run"
    original = tmp_path / "original.json"
    roster = tmp_path / "roster.json"
    files = {"train.json": "training", "val.json": "validation"}
    cell = sealed_write(folder / "cells/test.json", {"files": files})
    plan = {
        "cells": {"test": cell["artifact_sha256"]},
        "arms": ["full"],
        "visits": 15,
        "runtime": {"python": "site"},
        "source_sha256": "release",
    }
    sealed_write(folder / "plan.json", plan)
    sealed_write(
        original, {**plan, "runtime": {"python": "mac"}, "source_sha256": "original"}
    )
    public._write(roster, {"public_files": {"test": files}})
    assert pilot.verify_design(folder, original, roster)["identical_scientific_design"]
    public._write(roster, {"public_files": {"test": {"train.json": "different"}}})
    with pytest.raises(ValueError, match="public inputs"):
        pilot.verify_design(folder, original, roster)
    sealed_write(tmp_path / "wrong.json", {**plan, "visits": 14})
    with pytest.raises(ValueError, match="scientific design"):
        pilot.verify_design(folder, tmp_path / "wrong.json", roster)


@pytest.mark.parametrize("job_local", [True, False])
def test_historical_proposer_inherits_its_casadi_and_project_paths(tmp_path, job_local):
    """Exercise path propagation across both native proposer launchers."""
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import tempfile

    if not all(shutil.which(cmd) for cmd in ("bash", "jq", "sha256sum")):
        pytest.skip("native Slurm launchers require Bash, jq and sha256sum")
    code = tmp_path / "code"
    code.mkdir()
    (code / "scripts").symlink_to(io.REPO / "scripts", target_is_directory=True)
    commit = "1" * 40
    (code / "SOURCE_COMMIT").write_text(commit)
    jobs = tmp_path / "campaign"
    jobs.mkdir()
    image = tmp_path / "test.sif"
    image.write_bytes(b"smoke")
    (jobs / "plan.json").write_text(
        json.dumps(
            {
                "config": {
                    "protocol": "final-component-campaign-1",
                    "platform": "aces-h100x1",
                    "model_settings": {"model": "fixture", "model_revision": "unit"},
                    "served_context_tokens": 10,
                    "wall_seconds": 10,
                    "serving_image_sha256": hashlib.sha256(
                        image.read_bytes()
                    ).hexdigest(),
                }
            }
        )
    )
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "module").write_text("#!/bin/sh\nexit 0\n")
    python = fake / "python"
    python.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "-c" ]; then echo 43210; exit 0; fi\n'
        'if [ "$2" = "verify" ]; then\n'
        '  printf "%s\\n" "$PYTHONPATH" >> "$AF_TEST_VERIFY"\n'
        '  if [ "$(wc -l < "$AF_TEST_VERIFY")" -ge 2 ]; then exit 41; fi\n'
        "  exit 0\nfi\nexit 2\n"
    )
    python.chmod(0o755)
    (fake / "module").chmod(0o755)
    verification = tmp_path / "verify.log"
    casadi = tmp_path / "casadi"
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="ipc-") as ipc:
        env = os.environ | {
            "PATH": str(fake) + ":" + os.environ["PATH"],
            "AF_TEST_VERIFY": str(verification),
            "AF_CASADI_ROOT": str(casadi),
            "AF_REPO_ROOT": str(code),
            "AF_SITE": "aces",
            "AF_PYTHON": str(python),
            "AF_OUTPUT_ROOT": str(jobs),
            "AF_COMMIT": commit,
            "AF_HF_HOME": str(tmp_path),
            "TMPDIR": str(tmp_path / "node"),
            "AF_COMPUTE_CACHE_ROOT": "job-local"
            if job_local
            else str(tmp_path / "cache"),
            "AF_IPC_TMP_ROOT": ipc,
            "SLURM_JOB_ID": "999",
            "AF_VLLM_IMAGE": str(image),
        }
        proc = subprocess.run(
            [
                "bash",
                str(code / "scripts/hpc/run_component_campaign.sh"),
                "propose",
                "0",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    assert proc.returncode == 41, proc.stderr
    assert verification.read_text().splitlines() == [f"{code}/src:{code}:{casadi}"] * 2
    assert (tmp_path / ("node" if job_local else "cache") / "999").is_dir()
