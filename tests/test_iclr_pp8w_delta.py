"""Delta resource requests, transport isolation and resumable terminal checks."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_write
from scripts import iclr_anytime_cost_export as costs
from scripts import iclr_pp8w_delta as delta


def test_assets_reject_changed_image_and_missing_cached_weights(tmp_path, monkeypatch):
    python = tmp_path / "python"
    python.write_text("fixture")
    image = tmp_path / "image.sif"
    image.write_bytes(b"fixture image")
    expected = hashlib.sha256(image.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    for size, revision in (
        ("20b", delta.PROPOSER_REVISION),
        ("120b", delta.local_judge.REVISION),
    ):
        model = (
            cache / "hub" / f"models--openai--gpt-oss-{size}" / "snapshots" / revision
        )
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        (model / "tokenizer.json").write_text("{}")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "part.safetensors"}})
        )
        (model / "part.safetensors").write_bytes(b"fixture weights")
    monkeypatch.setattr(
        delta,
        "paths",
        lambda: {
            "python": python,
            "image": image,
            "cache": cache,
        },
    )
    monkeypatch.setattr(delta, "IMAGE_SHA", expected)
    monkeypatch.setattr(delta.harmony_vocab, "verify", lambda _: {"verified": True})
    assert delta.assets()["image_sha256"] == expected
    image.write_bytes(b"different image")
    with pytest.raises(ValueError, match="pinned Delta image"):
        delta.assets()
    image.write_bytes(b"fixture image")
    (model / "part.safetensors").unlink()
    with pytest.raises(ValueError, match="weight shards"):
        delta.assets()


def test_local_worker_environment_excludes_hosted_credentials(monkeypatch):
    monkeypatch.setenv("AF_JETSTREAM_API_KEY", "synthetic")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    env = delta.env()
    assert "AF_JETSTREAM_API_KEY" not in env and "OPENAI_API_KEY" not in env
    assert env["OPENBLAS_NUM_THREADS"] == "1"


def test_submission_records_separate_dedicated_gpu_resources(tmp_path, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=str(123 + len(calls)) + "\n")

    monkeypatch.setattr(delta.subprocess, "run", submit)
    assert delta.submit_job(tmp_path, critic=True) == "124"
    assert delta.submit_job(tmp_path, critic=False, indices=[0, 4]) == "125"
    assert "--gpus-per-node=4" in calls[0][0]
    assert "--gpus-per-node=1" in calls[1][0]
    assert "--array=0,4%4" in calls[1][0]
    assert calls[1][0][-2:] == ["search-worker", str(tmp_path)]
    assert json.loads((tmp_path / "submissions/125.json").read_text())["indices"] == [
        0,
        4,
    ]
    with pytest.raises(ValueError):
        delta.submit_job(tmp_path, critic=False, indices=[])


def test_pruning_without_final_critic_is_not_a_complete_block(tmp_path):
    plan = sealed_write(
        tmp_path / "search/plan.json",
        {"blocks": [{"block": 0, "tasks": [{"task_id": "a"}]}]},
    )
    assert delta.pending(tmp_path) == [0]
    directory = tmp_path / "search/pruning/a"
    sealed_write(
        directory / "result.json",
        {"identity": plan["artifact_sha256"], "status": "complete"},
    )
    sealed_write(directory / "cost.json", {"pruning_seconds": 1})
    assert delta.pending(tmp_path) == [0]
    sealed_write(directory / "critic.json", {"identity": plan["artifact_sha256"]})
    assert delta.pending(tmp_path) == []


def test_cost_export_uses_observed_receipts_and_flags_missing_times(tmp_path):
    sealed_write(
        tmp_path / "plan.json",
        {
            "visits": 2,
            "blocks": [
                {
                    "tasks": [
                        {"task_id": "a", "cell": "cell", "seed": 0, "treatment": "full"}
                    ]
                }
            ],
        },
    )
    for index in (0, 1):
        folder = tmp_path / "results/a" / f"round_{index:02d}"
        sealed_write(
            folder / "result.json",
            {"cost": {"observed_tokens": 10, "physical_requests": 1}},
        )
        if index == 0:
            sealed_write(folder / "cost.json", {"fit_seconds": 2, "visit_seconds": 3})
    sealed_write(tmp_path / "pruning/a/cost.json", {"pruning_seconds": 4})
    result = costs.export(tmp_path)
    assert result["rows"][-1]["proposer_tokens"] == 20
    assert result["rows"][-1]["total_seconds"] == 7
    assert result["rows"][-1]["unknown_visit_costs"] == 1
    assert not result["test_data_opened"]


def test_completed_campaign_submission_starts_no_jobs(tmp_path, monkeypatch, capsys):
    sealed_write(tmp_path / "deployment.json", {"status": "prepared"})
    monkeypatch.setattr(delta.os, "umask", lambda _: None)
    monkeypatch.setattr(delta, "pending", lambda _: [])
    monkeypatch.setattr(
        delta.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="")
    )
    exports = []
    monkeypatch.setattr(delta, "finish_campaign", exports.append)
    monkeypatch.setattr(
        delta, "submit_job", lambda *a, **k: pytest.fail("new job after completion")
    )
    monkeypatch.setattr(sys, "argv", ["launcher", "submit", "--root", str(tmp_path)])
    delta.main()
    assert json.loads(capsys.readouterr().out)["new_jobs"] == 0
    assert exports == [tmp_path]


def test_worker_refuses_expired_critic_before_starting_any_llm(tmp_path, monkeypatch):
    plan = sealed_write(tmp_path / "search/plan.json", {"blocks": []})
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "local-critic-key").write_text("synthetic-service-credential")
    (runtime / "critic-service.json").write_text(
        json.dumps(
            {
                "search_plan": plan["artifact_sha256"],
                "revision": delta.local_judge.REVISION,
                "image_sha256": delta.IMAGE_SHA,
                "endpoint": "http://127.0.0.1:19000",
                "expires_at": 1,
            }
        )
    )
    monkeypatch.setattr(delta, "ready", lambda *args: True)
    monkeypatch.setattr(
        delta, "serve", lambda *a, **k: pytest.fail("LLM started with expired critic")
    )
    with pytest.raises(RuntimeError, match="allocation is ending"):
        delta.search_worker(tmp_path, 0)
