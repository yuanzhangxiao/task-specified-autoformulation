"""A packaged numerical smoke and fail-closed recovery of unspent fit attempts."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import component_critic as critic
from autoformalism.rebuttal import review_deadline_pipeline as pipeline
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit
from scripts import iclr_controlled_receipts as receipts
from scripts import package_iclr_targeted_perturbations as package
from scripts import recover_iclr_targeted_perturbations as recovery
from tests.test_iclr_targeted_perturbations import inputs, prepared  # noqa: F401


@pytest.fixture
def failed(request, monkeypatch):
    root, *_ = request.getfixturevalue("prepared")
    plan = recovery.campaign.verify(root)
    (root / "judge_cache").mkdir()
    (root / "runtime").mkdir()
    (root / "runtime/service-fixture.json").write_text('{"source":"fixture"}')

    def missing(*args):
        raise FileNotFoundError(
            2, "No such file or directory", "/old/" + recovery.MISSING_RUNNER
        )

    monkeypatch.setattr(public, "_run_backend", missing)
    rows = []
    for block in plan["blocks"]:
        task = block["tasks"][0]
        raw = sealed_read(root / "cells" / f"{task['cell']}.json")
        target = raw["context"]["targets"][0]
        candidate = {
            "candidate_id": "test_parent",
            "parent_candidate_id": None,
            "states": [{"name": "q", "kind": "latent"}],
            "state_equations": [{"state": "q", "rhs": "-rate*q"}],
            "observation_mappings": [{"channel": target, "expression": "q"}],
            "parameters": [{"name": "rate", "scope": "global", "role": "rate"}],
            "initial_conditions": [{"state": "q", "scope": "global", "fixed_value": 1}],
        }
        bundle = {
            "candidate": candidate,
            "context": raw["context"],
            "initialization": {
                "base_candidate": candidate,
                "plan": {"rules": {}},
                "guesses": {"rate": 1},
                "context": raw["context"],
            },
        }
        directory = recovery.campaign.search.io.round_path(root, task, 0)
        proposal = sealed_write(
            directory / "proposal.json",
            {
                "task": task,
                "round": 0,
                "parent_sha256": None,
                "bundle": bundle,
                "status": "constructed",
                "cost": {"requests": 1},
            },
        )
        sealed_write(
            directory / "critic.json",
            {
                "identity": plan["artifact_sha256"],
                "task": task,
                "round": 0,
                "proposal_sha256": proposal["artifact_sha256"],
                "status": "reviewed",
            },
        )
        cell = recovery.campaign.search.controls.treatment(raw, "full")
        current = recovery.campaign.search.view(plan, cell)
        client = pipeline._client(
            root,
            current,
            task,
            0,
            "http://fixture.invalid",
            lambda: True,
            transport=lambda *args, candidate=candidate: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(candidate)},
                    }
                ],
                "usage": {"total_tokens": 12},
            },
        )
        cached = client.call(
            system="Return the fixture candidate.",
            user="Use the supplied public channels.",
            response_model=CandidateModel,
            step="fixture_construction",
            attempt=0,
        )
        public._write(
            directory / "construction.json", {"request_hash": cached["request_hash"]}
        )
        request = pipeline.request_for(bundle, current, task, 0)
        public.prepare_fit(
            request,
            PublicSplit.model_validate(raw["training"]),
            PublicSplit.model_validate(raw["validation"]),
            directory / "fit",
        )
        fit = public.execute_fit(directory / "fit")
        assert fit.status == "fit_failed" and "FileNotFoundError" in fit.message
        env = public._read(directory / "fit/result.json")
        result = sealed_write(
            directory / "result.json",
            {
                "task": task,
                "round": 0,
                "status": "fit_failed",
                "selected": None,
                "proposal_sha256": proposal["artifact_sha256"],
                "trial": {
                    "fit": env["result"],
                    "fit_result_sha256": public.content_sha256(env),
                },
            },
        )
        for i in range(1, 15):
            sealed_write(
                recovery.campaign.search.io.round_path(root, task, i) / "result.json",
                {
                    "task": task,
                    "round": i,
                    "status": "shared_initial_unavailable",
                    "shared_initial_sha256": result["artifact_sha256"],
                },
            )
        rows.append({"status": "no_retained_model", "source_id": task["task_id"]})
    freeze = sealed_write(
        root / "evaluation_freeze.json", {"plan_sha256": plan["artifact_sha256"]}
    )
    sealed_write(
        root / "frozen_models.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "selection_freeze_sha256": freeze["artifact_sha256"],
            "rows": rows,
        },
    )
    return root


def test_recovery_preserves_inputs_and_reuses_both_llm_receipts(failed, tmp_path):
    before = {
        str(p.relative_to(failed)): p.read_bytes()
        for p in failed.rglob("*")
        if p.is_file()
    }
    destination = tmp_path / "recovered"
    receipt = recovery.recover(failed, destination)
    assert receipt == recovery.recover(failed, destination)
    assert len(receipt["failed_attempts"]) == 6
    assert not (destination / "evaluation_freeze.json").exists()
    plan = recovery.campaign.verify(destination)
    for block in plan["blocks"]:
        task = block["tasks"][0]
        folder = recovery.campaign.search.io.round_path(destination, task, 0)
        assert (folder / "fit/freeze.json").exists()
        assert not (folder / "fit/started.json").exists()
        assert not (folder / "fit/result.json").exists()
        assert not (folder / "result.json").exists()
        (journal,) = (folder / "calls").glob("*.json")
        assert public._read(journal)["request_hash"] == journal.stem
        assert pipeline.propose_one(destination, {}, task, 0, None) == sealed_read(
            folder / "proposal.json"
        )
        assert critic.review_one(destination, plan, task, 0, None) == sealed_read(
            folder / "critic.json"
        )
    assert all((failed / name).read_bytes() == value for name, value in before.items())


def test_recovered_block_reuses_cached_calls_before_one_numerical_attempt(
    failed, tmp_path, monkeypatch
):
    """Exercise the real worker path with no provider or numerical solver calls."""
    destination = tmp_path / "recovered"
    receipt = recovery.recover(failed, destination)
    plan = recovery.campaign.verify(destination)
    task = plan["blocks"][0]["tasks"][0]
    directory = recovery.campaign.search.io.round_path(destination, task, 0)
    preserved = {
        p: p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file() and p.name != ".lock"
    }
    attempts = []

    def no_provider(*args, **kwargs):
        pytest.fail("recovered initial visit attempted a provider call")

    def numerical_result(request, model, train, val, guesses, settings, folder):
        attempts.append(folder)
        assert folder == directory / "fit"
        assert (
            request.model_dump(mode="json")
            == public._read(folder / "freeze.json")["request"]
        )
        # A numerical failure must preserve the campaign's fitted-start rule.
        return {"parameters": None, "training": {}, "validation": {}}

    monkeypatch.delenv("AF_JETSTREAM_API_KEY", raising=False)
    monkeypatch.setattr(critic.judge, "perform_review", no_provider)
    monkeypatch.setattr(critic.jetstream, "JetstreamTransport", no_provider)
    monkeypatch.setattr(public, "_run_backend", numerical_result)

    def run():
        return recovery.campaign.search.run_block(
            destination,
            0,
            "http://127.0.0.1:1",
            60,
            transport=no_provider,
            token_transport=no_provider,
        )

    assert run() == {"status": "complete", "block": 0}
    assert attempts == [directory / "fit"]
    result = sealed_read(directory / "result.json")
    assert result["status"] == "fit_failed" and result["selected"] is None
    assert result["trial"]["fit"]["backend_result_sha256"] is not None
    for index in range(1, plan["visits"]):
        later = recovery.campaign.search.io.round_path(destination, task, index)
        assert sealed_read(later / "result.json")["status"] == (
            "shared_initial_unavailable"
        )
        assert not (later / "calls").exists()
    assert (
        sealed_read(destination / "pruning" / task["task_id"] / "result.json")["status"]
        == "skipped_no_finite_parent"
    )
    assert all(path.read_bytes() == value for path, value in preserved.items())
    completed = {
        p: p.read_bytes()
        for name in ("results", "pruning")
        for p in (destination / name).rglob("*.json")
    }
    assert recovery.recover(failed, destination) == receipt
    assert run() == {"status": "complete", "block": 0}
    assert attempts == [directory / "fit"]
    assert all(path.read_bytes() == value for path, value in completed.items())


@pytest.mark.parametrize("defect", ["namespace", "missing_receipt"])
def test_recovered_block_rejects_invalid_cached_calls_before_fitting(
    failed, tmp_path, monkeypatch, defect
):
    """The worker must validate journal provenance before reusing a proposal."""
    destination = tmp_path / "recovered"
    recovery.recover(failed, destination)
    plan = recovery.campaign.verify(destination)
    task = plan["blocks"][0]["tasks"][0]
    directory = recovery.campaign.search.io.round_path(destination, task, 0)
    (journal,) = (directory / "calls").glob("*.json")
    if defect == "namespace":
        cached = public._read(journal)
        cached["request"]["namespace"] = "wrong-campaign"
        cached["request_hash"] = pipeline.content_hash(cached["request"])
        journal.unlink()
        public._write(journal.with_name(cached["request_hash"] + ".json"), cached)
    else:
        journal.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("invalid cached provenance reached provider or numerical work")

    monkeypatch.setattr(public, "_run_backend", forbidden)
    with pytest.raises(ValueError, match=r"provider (request provenance|cache)"):
        recovery.campaign.search.run_block(
            destination,
            0,
            "http://127.0.0.1:1",
            60,
            transport=forbidden,
            token_transport=forbidden,
        )
    assert not (directory / "fit/started.json").exists()
    assert not (directory / "result.json").exists()


@pytest.mark.parametrize(
    "defect",
    [
        "other_error",
        "backend_work",
        "later_calls",
        "changed_copy",
        "symlink",
        "same_root",
    ],
)
def test_recovery_refuses_ambiguous_failure_or_changed_evidence(
    failed, tmp_path, defect
):
    directory = failed / "results/block0000_full/round_00"
    if defect == "other_error":
        env = public._read(directory / "fit/result.json")
        env["result"]["message"] = "ValueError: numerical failure"
        env["sha256"] = public.content_sha256(env["result"])
        public._write(directory / "fit/result.json", env)
    elif defect == "backend_work":
        (directory / "fit/backend").mkdir()
        (directory / "fit/backend/initializer_started.json").write_text("{}")
    elif defect == "later_calls":
        (directory.parent / "round_01/extra.json").write_text("{}")
    elif defect == "symlink":
        (directory / "link").symlink_to(tmp_path)
    destination = failed if defect == "same_root" else tmp_path / "recovered"
    if defect == "changed_copy":
        recovery.recover(failed, destination)
        (destination / "results/block0000_full/round_00/proposal.json").write_text("{}")
    with pytest.raises(ValueError):
        recovery.recover(failed, destination)


def test_fit_receipt_exports_the_actionable_failure_message(failed):
    folder = failed / "results/block0000_full/round_00/fit"
    result = receipts.fit_receipt(folder, {}, None)
    assert result["fit_status"] == "fit_failed"
    assert recovery.MISSING_RUNNER in result["fit_message"]
    assert result["actual_residual_calls"] is None


def test_partial_recovery_resumes_without_replacing_finished_work(
    failed, tmp_path, monkeypatch
):
    destination = tmp_path / "recovered"
    original = recovery.shutil.copyfile

    def interrupted(source, target):
        target.write_bytes(b"partial copy")
        raise OSError("simulated interrupted filesystem write")

    monkeypatch.setattr(recovery.shutil, "copyfile", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        recovery.recover(failed, destination)
    assert (destination / "packaging_recovery_intent.json").exists()
    assert not (destination / "packaging_recovery.json").exists()
    monkeypatch.setattr(recovery.shutil, "copyfile", original)
    result = recovery.recover(failed, destination)
    assert len(result["failed_attempts"]) == 6
    # New numerical results must survive subsequent preparation, too.
    fit = destination / "results/block0000_full/round_00/fit/result.json"
    fit.write_text('{"new_numerical_attempt":"preserve"}')
    assert recovery.recover(failed, destination) == result
    assert json.loads(fit.read_text())["new_numerical_attempt"] == "preserve"


def test_isolated_delivery_identity_and_real_collocation_fit(tmp_path):
    """Use only delivered imports/resources, not the development checkout."""
    pytest.importorskip("casadi")
    bundle = tmp_path / "bundle"
    shutil.copytree(
        package.REPO / "src",
        bundle / "src",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    paths = package.script_dependencies()
    assert recovery.MISSING_RUNNER in paths
    assert "scripts/fitting_timed_worker.py" in paths
    for name in paths:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package.REPO / name, target)
    env = {
        **os.environ,
        "PYTHONPATH": f"{bundle / 'src'}:{bundle}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    identity_command = [
        sys.executable,
        "-c",
        "from autoformalism.rebuttal.fitter_diagnostic import runtime_identity; "
        "runtime_identity()",
    ]
    missing = bundle / recovery.MISSING_RUNNER
    content = missing.read_bytes()
    missing.unlink()
    error = subprocess.run(
        identity_command,
        cwd=bundle,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert error.returncode and "FileNotFoundError" in error.stderr
    missing.write_bytes(content)
    subprocess.run(
        identity_command,
        cwd=bundle,
        env=env,
        check=True,
        capture_output=True,
        timeout=30,
    )
    driver = bundle / "check.py"
    driver.write_text("""from pathlib import Path
import json
from autoformalism.fitting import public_fitting as public
from scripts.smoke_public_fitting import control
if __name__ == "__main__":
    assert Path(public.__file__).resolve().is_relative_to(Path.cwd())
    request, train, val = control("collocation-multi-target-v1", multiple_targets=True)
    output = Path("fit")
    public.prepare_fit(request, train, val, output)
    result = public.execute_fit(output)
    assert result.status == "complete", result.message
    assert result.backend_result_sha256 is not None
    assert result.training.normalized_mse < 1e-8
    assert result.validation.normalized_mse < 1e-8
    before = (output / "result.json").read_bytes()
    assert public.execute_fit(output) == result
    assert (output / "result.json").read_bytes() == before
    print(json.dumps(result.model_dump(mode="json")))
""")
    result = subprocess.run(
        [sys.executable, str(driver)],
        cwd=bundle,
        env=env,
        capture_output=True,
        text=True,
        timeout=370,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1])["status"] == "complete"
