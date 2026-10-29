"""Matched resource/identity gates and offline ACES launcher dispatch."""

import io
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import submit_construction_search_aces as submission

REAL_REPO = submission.REPO


def _snapshot(cache, size):
    path = (
        cache
        / f"hub/models--openai--gpt-oss-{size}/snapshots"
        / submission.MODELS[size]["revision"]
    )
    path.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors"}})
    )
    (path / "model-1.safetensors").write_bytes(b"offline test weights")
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    code = tmp_path / "code"
    for name in submission.ENTRYPOINTS:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test source {name}\n")
    monkeypatch.setattr(submission, "REPO", code)
    monkeypatch.setattr(
        submission.harmony_vocab,
        "verify",
        lambda path: {
            "protocol": submission.harmony_vocab.PROTOCOL,
            "filename": submission.harmony_vocab.VOCAB_FILENAME,
            "sha256": submission.harmony_vocab.VOCAB_SHA256,
        },
    )
    image, cache = tmp_path / "image.sif", tmp_path / "cache"
    image.write_bytes(b"offline test image")
    for size in submission.MODELS:
        _snapshot(cache, size)
    for key, value in {
        "AF_PYTHON": sys.executable,
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
    }.items():
        monkeypatch.setenv(key, value)
    plans = {}
    for size, model in submission.MODELS.items():
        plans[size] = {
            "protocol": submission.PROTOCOL,
            "profile": "three-policy",
            "policies": sorted(submission.POLICIES),
            "required_options_in_first_call": 3,
            "maximum_options_in_first_call": 3,
            "maximum_calls_per_episode": 3,
            "artifact_sha256": size + "a" * 64,
            "model_size": size,
            "cases": [
                {
                    "name": f"P{parent:02d}-seed{seed}-{policy}",
                    "case_id": f"P{parent:02d}-seed{seed}-{policy}",
                    "parent_id": f"P{parent:02d}",
                    "seed": seed,
                    "policy": policy,
                }
                for parent in range(6)
                for seed in range(3)
                for policy in sorted(submission.POLICIES)
            ],
            "settings": {
                **submission.MATCHED_SETTINGS,
                "model": model["model"],
                "model_revision": model["revision"],
            },
            "optimizer_calls": 0,
            "test_data_opened": False,
            "automatic_followup": False,
        }
    monkeypatch.setattr(
        submission, "_campaign_verify", lambda root: deepcopy(plans[root.name])
    )
    monkeypatch.setattr(
        submission,
        "_campaign_report",
        lambda root: {
            "status": "incomplete",
            "smoke_complete": False,
        },
    )
    return tmp_path, plans


def test_matched_budgets_and_distinct_resources(setup, monkeypatch):
    tmp, plans = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, str(100 + len(calls)), "")

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    records = []
    for size in ("20b", "120b"):
        root = tmp / size
        first = submission.submit(root)
        assert submission.submit(root) == first
        assert first["maximum_generation_requests"] == 9
        plan, execution = submission.verify(root, check_assets=True)
        assert plan == plans[size]
        assert execution["worker_seconds"] == 12600
        assert execution["maximum_charged_tokens"] == 54 * 98304
        records.append(execution)
    assert len(calls) == 2
    assert {"--gres=gpu:h100:1", "--mem=64G", "--cpus-per-task=8"} <= set(calls[0])
    assert {"--gres=gpu:h100:2", "--mem=128G", "--cpus-per-task=16"} <= set(calls[1])
    for key in (
        "case_ids",
        "generation_settings",
        "served_context_tokens",
        "maximum_generation_requests",
        "maximum_charged_tokens",
        "worker_seconds",
    ):
        assert records[0][key] == records[1][key]
    assert records[0]["model_revision"] != records[1]["model_revision"]
    assert records[0]["maximum_generation_requests"] == 162
    assert len(records[0]["smoke_case_ids"]) == 3
    assert {"--time=04:00:00", "--signal=B:TERM@300"} <= set(calls[0])


def test_full_requires_completed_smoke_before_any_submission(setup, monkeypatch):
    tmp, _ = setup
    monkeypatch.setattr(
        submission.subprocess, "run", lambda *a, **kw: pytest.fail("no submit")
    )
    with pytest.raises(ValueError, match="smoke phase is not complete"):
        submission.submit(tmp / "20b", "full")
    assert not list((tmp / "20b").glob("submission/*/*/propose.intent.json"))


def test_smoke_full_and_explicit_resume_share_budget_and_execution(setup, monkeypatch):
    tmp, _ = setup
    calls = []
    summary = {"status": "incomplete", "smoke_complete": False}
    monkeypatch.setattr(submission, "_campaign_report", lambda root: summary)

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, str(100 + len(calls)), "")

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    root = tmp / "20b"
    smoke = submission.submit(root)
    execution_bytes = (root / "execution.json").read_bytes()
    assert submission.submit(root) == smoke
    summary["smoke_complete"] = True
    full = submission.submit(root, "full")
    resumed = submission.submit(root, "full", "wave2")
    assert full["execution_sha256"] == smoke["execution_sha256"]
    assert resumed["execution_sha256"] == smoke["execution_sha256"]
    assert resumed["campaign_maximum_generation_requests"] == 162
    assert resumed["budget_reset"] is False
    assert (root / "execution.json").read_bytes() == execution_bytes
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 3
    assert submission.submit(root, "smoke", "wave2")["status"] == "already_complete"
    summary["status"] = "complete"
    assert submission.submit(root, "full", "wave3")["status"] == "already_complete"
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 3


@pytest.mark.parametrize("phase,wave", [("smoke", "wave2"), ("full", "wave1")])
def test_active_prior_allocation_blocks_other_wave_or_phase(
    setup, monkeypatch, phase, wave
):
    tmp, _ = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv, 0, "123|RUNNING\n" if argv[0] == "squeue" else "123", ""
        )

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    submission.submit(tmp / "20b")
    with pytest.raises(ValueError, match="prior allocation still active"):
        submission.submit(tmp / "20b", phase, wave)
    assert len([argv for argv in calls if argv[0] == "sbatch"]) == 1


@pytest.mark.parametrize("reply", ["bad reply", None, "nonzero"])
def test_unknown_prior_job_state_blocks_resume(setup, monkeypatch, reply):
    tmp, _ = setup

    def scheduler(argv, **kwargs):
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(argv, 0, "123", "")
        if reply is None:
            raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(
            argv, 1 if reply == "nonzero" else 0, reply, ""
        )

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    submission.submit(tmp / "20b")
    with pytest.raises(ValueError, match="no submission"):
        submission.submit(tmp / "20b", "smoke", "wave2")


@pytest.mark.parametrize("phase,wave", [("other", "wave1"), ("smoke", "../x")])
def test_invalid_phase_or_wave_has_no_side_effect(setup, phase, wave):
    tmp, _ = setup
    with pytest.raises(ValueError, match="invalid phase or wave"):
        submission.submit(tmp / "20b", phase, wave)
    assert not (tmp / "20b").exists()


@pytest.mark.parametrize("mode", ["timeout", "empty_success", "failure"])
def test_ambiguous_or_failed_scheduler_intent_cannot_duplicate(
    setup, monkeypatch, mode
):
    tmp, _ = setup
    calls = []

    def scheduler(argv, **kwargs):
        calls.append(argv)
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, 60)
        return subprocess.CompletedProcess(
            argv, 1 if mode == "failure" else 0, "\n", "Socket timed out"
        )

    monkeypatch.setattr(submission.subprocess, "run", scheduler)
    with pytest.raises(ValueError, match=r"uncertain|unconfirmed"):
        submission.submit(tmp / "20b")
    with pytest.raises(ValueError, match="existing submission intent"):
        submission.submit(tmp / "20b")
    with pytest.raises(ValueError, match="existing submission intent"):
        submission.submit(tmp / "20b", "full", "wave2")
    assert len(calls) == 1
    assert (tmp / "20b/submission/smoke/wave1/propose.intent.json").is_file()
    assert (tmp / "20b/submission/smoke/wave1/propose.reply.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("maximum_requests", 4),
        ("max_output_tokens", 8193),
        ("reasoning_effort", "high"),
    ],
)
def test_generation_drift_rejected_before_slurm(setup, monkeypatch, field, value):
    tmp, plans = setup
    plans["20b"]["settings"][field] = value
    monkeypatch.setattr(
        submission.subprocess, "run", lambda *a, **kw: pytest.fail("no submit")
    )
    with pytest.raises(ValueError, match="requires six"):
        submission.submit(tmp / "20b")


def test_missing_weight_shard_and_tokenizer_rejected(setup):
    tmp, _ = setup
    snapshot = (
        tmp
        / "cache/hub/models--openai--gpt-oss-120b/snapshots"
        / submission.MODELS["120b"]["revision"]
    )
    (snapshot / "model-1.safetensors").unlink()
    with pytest.raises(ValueError, match="missing or empty"):
        submission.submit(tmp / "120b")
    (snapshot / "model-1.safetensors").write_bytes(b"restored")
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match=r"tokenizer\.json"):
        submission.submit(tmp / "120b")


def test_actual_image_and_source_rechecked_before_gpu(setup, monkeypatch):
    tmp, _ = setup
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "123", ""),
    )
    submission.submit(tmp / "20b")
    (tmp / "image.sif").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(tmp / "20b", check_assets=True)
    (tmp / "image.sif").write_bytes(b"offline test image")
    (tmp / "code" / submission.ENTRYPOINTS[0]).write_text("changed source")
    with pytest.raises(ValueError, match="identity differs"):
        submission.verify(tmp / "20b")


def test_shell_syntax_and_sealed_server_arguments():
    for name in (
        "start_construction_search_aces.sh",
        "run_construction_search_aces.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(REAL_REPO / "scripts/hpc" / name)], check=True
        )
    worker = (REAL_REPO / "scripts/hpc/run_construction_search_aces.sh").read_text()
    assert "--verify-only --check-assets" in worker
    assert '--tensor-parallel-size "$AF_GPUS"' in worker
    assert '--runtime-record "$AF_RUNTIME/probe-${SLURM_JOB_ID}.json"' in worker
    assert 'mktemp -d "/tmp/af-search-' in worker
    assert '--phase "$AF_SEARCH_PHASE"' in worker
    assert "HF_HUB_OFFLINE=1" in worker
    assert "TRANSFORMERS_OFFLINE=1" in worker
    assert "gpu_names" in worker and "vllm.__version__ == '0.27.1'" in worker


@pytest.fixture
def serving(setup, monkeypatch):
    tmp, plans = setup
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0, "1234", ""),
    )

    def make(size):
        root = tmp / size
        submission.submit(root)
        _, execution = submission.verify(root)
        count = execution["tensor_parallel_size"]
        runtime = {
            "visible_gpus": count,
            "gpu_names": ["NVIDIA H100 PCIe"] * count,
            "gpu_memory_bytes": [80 * 1024**3] * count,
            "gpu_driver": ["580.178.04"] * count,
            "cuda_kernel_passed": True,
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "vllm": "0.27.1",
            "harmony": {
                **execution["harmony_vocabulary"],
                "harmony_render_passed": True,
                "openai_harmony": "test-version",
                "rendered_tokens": 12,
                "generation_requests": 0,
            },
            "serving_identity": {
                k: execution[k]
                for k in (
                    "artifact_sha256",
                    "model_size",
                    "model",
                    "model_revision",
                    "image_sha256",
                    "served_context_tokens",
                    "tensor_parallel_size",
                    "maximum_sequences",
                )
            },
        }
        record = root / "runtime.json"
        submission._seal(record, runtime)
        monkeypatch.setattr(
            submission,
            "urlopen",
            lambda url, **kw: io.StringIO(
                json.dumps({"data": [{"id": plans[size]["settings"]["model"]}]})
            ),
        )
        return root, record, runtime

    return make


@pytest.mark.parametrize("size", ["20b", "120b"])
def test_actual_serving_identity_bound_for_both_arms(serving, size):
    root, record, runtime = serving(size)
    first = submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert first == submission.bind_server(root, record, "http://127.0.0.1:4567")
    assert first["serving_identity"]["model_size"] == size
    assert (root / "serving_runtime.json").is_file()
    record.unlink()
    runtime["gpu_driver"] = ["595.99.02"] * runtime["visible_gpus"]
    submission._seal(record, runtime)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        submission.bind_server(root, record, "http://127.0.0.1:5678")


@pytest.mark.parametrize(
    "defect", ["count", "memory", "gpu", "harmony", "arm", "alias", "seal"]
)
def test_incompatible_runtime_rejected_before_binding(serving, monkeypatch, defect):
    root, record, runtime = serving("120b")
    record.unlink()
    if defect == "count":
        runtime["visible_gpus"] = 1
    elif defect == "memory":
        runtime["gpu_memory_bytes"] = [40 * 1024**3] * 2
    elif defect == "gpu":
        runtime["gpu_names"] = ["NVIDIA A100"] * 2
    elif defect == "harmony":
        runtime["harmony"]["harmony_render_passed"] = False
    elif defect == "arm":
        runtime["serving_identity"]["model_size"] = "20b"
    elif defect == "alias":
        monkeypatch.setattr(
            submission,
            "urlopen",
            lambda *a, **kw: io.StringIO(
                json.dumps({"data": [{"id": "openai/gpt-oss-20b"}]})
            ),
        )
    submission._seal(record, runtime)
    if defect == "seal":
        changed = json.loads(record.read_text())
        changed["vllm"] = "changed"
        record.write_text(json.dumps(changed))
    with pytest.raises(ValueError):
        submission.bind_server(root, record, "http://127.0.0.1:1234")
    assert not (root / "serving_runtime.json").exists()


def test_server_binding_rejects_nonlocal_endpoint(serving):
    root, record, _ = serving("20b")
    with pytest.raises(ValueError, match="loopback HTTP"):
        submission.bind_server(root, record, "https://example.com")


@pytest.mark.parametrize(
    "profile",
    [
        "three-policy",
        "breadth-five",
        "focused-retest",
        "action-feedback",
        "repair-contract",
        "admission-guard",
    ],
)
@pytest.mark.parametrize("size", ["20b", "120b"])
@pytest.mark.parametrize("command", ["run", "inspect", "report", "preview"])
@pytest.mark.parametrize("phase,wave", [("smoke", "wave1"), ("full", "wave2")])
def test_real_shell_dispatch_ignores_old_generic_root(
    tmp_path, size, command, phase, wave, profile
):
    group, cache = tmp_path / "group scratch", tmp_path / "cache"
    group.mkdir()
    _snapshot(cache, size)
    image = tmp_path / "image.sif"
    image.write_bytes(b"test image")
    log = tmp_path / "python-calls.jsonl"
    stub = tmp_path / "python-stub"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['AF_TEST_LOG'], 'a') as f:\n"
        "    value = {'argv': sys.argv[1:], 'root': os.environ['AF_OUTPUT_ROOT']}\n"
        "    f.write(json.dumps(value) + '\\n')\n"
    )
    stub.chmod(0o755)
    env = {
        **os.environ,
        "AF_GROUP": str(group),
        "AF_OUTPUT_ROOT": str(group / "old-fitting"),
        "AF_PYTHON": str(stub),
        "AF_VLLM_IMAGE": str(image),
        "AF_HF_HOME": str(cache),
        "AF_TEST_LOG": str(log),
    }
    env.pop("AF_CONSTRUCTION_SEARCH_ROOT", None)
    result = subprocess.run(
        [
            "bash",
            str(REAL_REPO / "scripts/hpc/start_construction_search_aces.sh"),
            command,
            size,
            phase,
            wave,
            profile,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if profile == "action-feedback" and size == "120b":
        assert result.returncode == 2
        assert "20B-only" in result.stderr
        assert not log.exists()
        return
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    version = {
        "focused-retest": "v4",
        "action-feedback": "v5",
        "repair-contract": "v6",
        "admission-guard": "v7",
    }.get(profile, "v3")
    expected = str(group / f"phase-c-search-{profile}-{size}-{version}")
    assert calls and all(call["root"] == expected for call in calls)
    if command == "run":
        prepare = next(c["argv"] for c in calls if "prepare" in c["argv"])
        assert prepare[prepare.index("--model-size") + 1] == size
        assert prepare[prepare.index("--profile") + 1] == profile
        assert prepare[prepare.index("--source") + 1].endswith(
            "inputs/saved-source.tar.gz"
        )
        assert Path(calls[-1]["argv"][0]).name == "submit_construction_search_aces.py"
        submit_args = calls[-1]["argv"]
        assert submit_args[submit_args.index("--phase") + 1] == phase
        assert submit_args[submit_args.index("--wave") + 1] == wave


@pytest.mark.parametrize(
    "profile,count",
    [
        ("breadth-five", 18),
        ("focused-retest", 3),
        ("action-feedback", 3),
        ("repair-contract", 3),
        ("admission-guard", 3),
    ],
)
def test_breadth_submission_seals_larger_allowance_and_smaller_roster(
    setup, monkeypatch, profile, count
):
    tmp, plans = setup
    for plan in plans.values():
        plan.update(
            profile=profile,
            policies=["options_shallow"],
            required_options_in_first_call=5,
            maximum_options_in_first_call=5,
            maximum_calls_per_episode=5,
        )
        plan["cases"] = [c for c in plan["cases"] if c["policy"] == "options_shallow"]
        plan["settings"].update(submission.PROFILES[profile]["settings"])
        if profile in {
            "focused-retest", "action-feedback", "repair-contract", "admission-guard"
        }:
            plan["cases"] = [c for c in plan["cases"] if c["parent_id"] == "P00"]
            plan["engine_options"] = dict(
                submission.PROFILES[profile]["engine_options"]
            )
            plan["smoke_required_before_full"] = False
        if profile in {"action-feedback", "repair-contract", "admission-guard"}:
            plan["allowed_model_sizes"] = list(
                submission.PROFILES[profile]["allowed_model_sizes"]
            )
            plan["action_feedback_policy"] = submission.PROFILES[profile][
                "action_feedback_policy"
            ]
        if profile in {"repair-contract", "admission-guard"}:
            for key in ("repair_contract_policy", "identical_action_policy"):
                plan[key] = submission.PROFILES[profile][key]
        if profile == "admission-guard":
            plan["admission_policy"] = submission.PROFILES[profile]["admission_policy"]
            plan["gate_policy"] = {
                "defined_zero_forcing": submission.PROFILES[profile][
                    "defined_zero_forcing_policy"
                ]
            }
    monkeypatch.setattr(
        submission.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "1234", ""),
    )
    for size in plans:
        phase = (
            "full"
            if profile in {
                "focused-retest", "action-feedback",
                "repair-contract", "admission-guard"
            }
            else "smoke"
        )
        if profile == "action-feedback" and size == "120b":
            with pytest.raises(ValueError, match="20B-only"):
                submission.submit(tmp / size, phase)
            assert not (tmp / size / "submission").exists()
            continue
        result = submission.submit(tmp / size, phase)
        expected_cases = count if phase == "full" else 1
        assert result["maximum_cases"] == expected_cases
        assert result["maximum_generation_requests"] == expected_cases * 5
        assert result["campaign_maximum_generation_requests"] == count * 5
        _, execution = submission.verify(tmp / size)
        assert execution["maximum_cases"] == count
        assert execution["maximum_output_tokens"] == 8192
        assert execution["maximum_charged_tokens"] == count * 163840
        assert execution["required_options_in_first_call"] == 5


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "unlimited"),
        ("admission_policy", "construction-admission-guard-1"),
        (
            "gate_policy",
            {"defined_zero_forcing": "defined-common-zero-forcing-affine-1"},
        ),
        ("repair_contract_policy", "explicit-in-scope-option-repair-contract-1"),
        ("identical_action_policy", "identical-pending-action-normalization-1"),
        ("action_feedback_policy", "typed-branch-action-feedback-1"),
        ("allowed_model_sizes", ["20b"]),
        ("smoke_required_before_full", False),
        ("engine_options", {"branch_delta": True, "common_zero_forcing": True}),
        ("required_options_in_first_call", 5),
        ("maximum_options_in_first_call", 5),
        ("maximum_calls_per_episode", 5),
        ("policies", ["options_shallow"]),
    ],
)
def test_profile_drift_rejected_before_submission(setup, monkeypatch, field, value):
    tmp, plans = setup
    plans["20b"][field] = value
    monkeypatch.setattr(
        submission.subprocess, "run", lambda *a, **kw: pytest.fail("no submit")
    )
    with pytest.raises(ValueError, match=r"profile|requires six"):
        submission.submit(tmp / "20b")
