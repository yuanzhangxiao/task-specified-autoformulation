"""Exact Phase-B identities, native discovery, accounting, and safe resume."""

import json
from pathlib import Path

import pytest

from autoformalism.baselines import d3
from autoformalism.baselines.d3_native import NativeD3Fit
from autoformalism.data import DevelopmentDataset
from autoformalism.data.models import TierRoles
from autoformalism.expressions import ValidationContext
from autoformalism.llm.mock import MockLLMClient
from autoformalism.rebuttal import phase_b_d3 as campaign
from autoformalism.rebuttal.prefit_replay import sealed_read
from tests.test_d3_rollout import candidate, splits

BENCHMARK = "phase_b_dalla_man_t1_canonical_named_easy"


def fixture(tmp_path, monkeypatch, generations=1):
    train, val = splits((1, 0.5, 0.25))
    data = DevelopmentDataset(BENCHMARK, "easy", TierRoles(targets=("x",)), train, val)
    context = ValidationContext(targets=("x",))
    prompt = tmp_path / "proposer_prompt.txt"
    prompt.write_text("Frozen public prompt")
    identity = {
        "train": "train",
        "validation": "val",
        "prompt": campaign.file_hash(prompt),
    }
    monkeypatch.setattr(campaign, "load_public", lambda *a: (data, context, identity))
    monkeypatch.setattr(campaign, "_prompt_path", lambda *a: prompt)
    monkeypatch.setattr(campaign, "environment_identity", lambda: {"code": "fixed"})
    monkeypatch.setattr(
        d3,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": -0.5}, 1, 0, 10, {"x": 1}),
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "benchmark_id": BENCHMARK,
                        "tier": "easy",
                        "public_prompt_sha256": identity["prompt"],
                    }
                ],
                "generations": generations,
            }
        )
    )
    root = tmp_path / "run"
    plan = campaign.prepare(config, tmp_path, root, "explicit-model")
    return root, plan, identity, config


def test_discovery_selected_native_and_exact_completed_resume(tmp_path, monkeypatch):
    root, plan, _, _ = fixture(tmp_path, monkeypatch)
    client = MockLLMClient(proposer_responses=[candidate()])
    result = campaign.run(root, 0, client=client)
    assert result["status"] == "complete"
    assert result["evaluation"]["phase_b_rollout"]["normalized_mse"] == 0
    assert result["evaluation"]["bounds_violations"]
    assert campaign.run(root, 0, client=MockLLMClient()) == result
    assert len(client.calls) == 1
    assert "no dt multiplier" in client.calls[0]["system_prompt"]
    assert "test" not in json.loads(client.calls[0]["user_prompt"])["previous_results"]
    assert plan["maximum_logical_calls"] == 1
    summary = campaign.report(root)
    assert summary["counts"] == {"complete": 1}
    assert summary["metrics"]["phase_b_rollout"]["finite"] == 1
    assert summary["test_data_opened"] is False


def test_completed_generation_resume_after_interruption(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch, generations=2)
    calls = []

    def fit(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return NativeD3Fit({"k": -0.5}, 1, 0, 10, {"x": 1})

    monkeypatch.setattr(d3, "fit_native_d3", fit)
    with pytest.raises(KeyboardInterrupt):
        campaign.run(
            root, 0, client=MockLLMClient(proposer_responses=[candidate(), candidate()])
        )
    assert campaign.report(root)["counts"] == {"pending": 1}
    client = MockLLMClient(proposer_responses=[candidate()])
    assert campaign.run(root, 0, client=client)["status"] == "complete"
    assert len(client.calls) == 1
    assert json.loads(client.calls[0]["user_prompt"])["generation"] == 1
    assert len(calls) == 3  # completed generation 0 is never refitted


@pytest.mark.parametrize("field", ["train", "validation", "prompt"])
def test_development_drift_refused_before_provider(tmp_path, monkeypatch, field):
    root, _, identity, _ = fixture(tmp_path, monkeypatch)
    identity[field] = "changed"
    with pytest.raises(ValueError, match="input drift"):
        campaign.run(root, 0, client=MockLLMClient())


def test_code_drift_and_model_change_refused(tmp_path, monkeypatch):
    root, _, _, config = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="frozen artifact differs"):
        campaign.prepare(config, tmp_path, root, "different-model")
    monkeypatch.setattr(campaign, "environment_identity", lambda: {"code": "different"})
    with pytest.raises(ValueError, match="code or dependencies"):
        campaign.run(root, 0, client=MockLLMClient())


def test_prompt_hash_mismatch_refused(tmp_path, monkeypatch):
    _, _, identity, config = fixture(tmp_path, monkeypatch)
    identity["prompt"] = "changed"
    with pytest.raises(ValueError, match="public prompt differs"):
        campaign.prepare(config, tmp_path, tmp_path / "other", "explicit-model")


def test_invalid_candidate_recorded_without_fitting(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)

    def forbidden(*a, **k):
        raise AssertionError("unsafe model must not reach the fitter")

    monkeypatch.setattr(d3, "fit_native_d3", forbidden)
    result = campaign.run(
        root,
        0,
        client=MockLLMClient(proposer_responses=[candidate("__import__('os')")]),
    )
    assert result["status"] == "discovery_failed"
    assert campaign.run(root, 0, client=MockLLMClient()) == result
    assert campaign.report(root)["metrics"]["phase_b_rollout"]["finite"] == 0


def test_native_mismatch_not_treated_as_success(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        d3,
        "fit_native_d3",
        lambda *a, **k: NativeD3Fit({"k": -0.5}, 1, 9, 10, {"x": 1}),
    )
    result = campaign.run(
        root, 0, client=MockLLMClient(proposer_responses=[candidate()])
    )
    assert result["status"] == "native_check_failed"
    assert campaign.report(root)["metrics"]["phase_b_rollout"]["finite"] == 0


def test_saved_native_mutation_refused(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)
    campaign.run(root, 0, client=MockLLMClient(proposer_responses=[candidate()]))
    path = root / "results/0/d3_checkpoint.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="native source changed"):
        campaign.run(root, 0, client=MockLLMClient())


def test_no_provider_calls_on_prepare_report_and_config_rosters():
    base = Path(__file__).resolve().parents[1] / "configs"
    original = json.loads(
        (base / "phase_b_public_baseline_full_delta_cpu_v1.json").read_text()
    )
    hashes = {c["benchmark_id"]: c["public_prompt_sha256"] for c in original["cells"]}
    for scope, expected in (("pilot", 6), ("full", 120)):
        config = campaign.Campaign.model_validate_json(
            (base / f"phase_b_d3_{scope}_v1.json").read_text()
        )
        assert len(config.cells) * len(config.repetitions) == expected
        assert all(
            hashes[c.benchmark_id] == c.public_prompt_sha256 for c in config.cells
        )


@pytest.mark.parametrize(
    "mutation",
    [
        {"repetitions": []},
        {"repetitions": [0, 0]},
        {"repetitions": [-1]},
        {"test_data_opened": True},
        {"generations": 6, "patience": 5},
    ],
)
def test_invalid_config_refused(mutation):
    with pytest.raises(ValueError):
        campaign.Campaign.model_validate(
            {
                "cells": [
                    {
                        "benchmark_id": BENCHMARK,
                        "tier": "easy",
                        "public_prompt_sha256": "a" * 64,
                    }
                ],
                **mutation,
            }
        )


def test_accounting_includes_failed_calls_not_cache_hits(tmp_path):
    path = tmp_path / "calls.jsonl"
    events = [
        {"event": "llm_failure", "provider_attempts": 1, "usage": None},
        {"event": "llm_failure", "provider_attempts": 1, "usage": {"total_tokens": 3}},
        {"event": "llm_response", "provider_attempts": 1, "usage": {"total_tokens": 7}},
        {
            "event": "llm_response",
            "provider_attempts": 0,
            "cache_hit": True,
            "usage": {"total_tokens": 7},
        },
    ]
    path.write_text("\n".join(json.dumps(e) for e in events))
    assert campaign.accounting(path) == {
        "physical_requests": 3,
        "observed_tokens": 10,
        "unknown_usage_requests": 1,
        "cache_hits": 1,
    }


def test_sealed_plan_detects_tampering(tmp_path, monkeypatch):
    root, _, _, _ = fixture(tmp_path, monkeypatch)
    plan = sealed_read(root / "plan.json")
    plan["model"] = "other"
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="artifact digest differs"):
        campaign.run(root, 0, client=MockLLMClient())
