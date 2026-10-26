"""Pinned saved-patch rechecks preserve history, cost and bounded offline resume."""

import hashlib
import json

import pytest

from autoformalism.llm.staged_topology import atomic_json
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.staged_topology import content_hash
from scripts import replay_search_policy_results as cli
from tests.test_construction_search_policy import option, parent, patch  # noqa: F401


@pytest.fixture
def saved(parent, tmp_path, monkeypatch):  # noqa: F811
    """A minimal historical one-option episode with real seals and request hashes."""
    root = tmp_path / "source"
    root.mkdir()
    (root / "saved-source.tar.gz").write_bytes(b"opaque pinned historical source")
    selected = sealed_write(
        root / "selected_sources.json",
        {
            "parents": {"p": parent},
            "inputs": {},
            "consistency_source": {"evidence": {}},
        },
    )
    monkeypatch.setattr(cli.base.old, "_validate_inputs", lambda _: None)
    settings = cli.base.settings_for("20b").model_dump(mode="json")
    entry = {
        "case_id": "p__seed0__options_shallow",
        "parent_id": "p",
        "seed": 0,
        "policy": "options_shallow",
        "input_sha256": content_hash(parent),
        "namespace": "historical-synthetic-1",
    }
    plan = sealed_write(
        root / "plan.json",
        {
            "protocol": "construction-search-policy-1",
            "model_size": "20b",
            "selected_sources_sha256": selected["artifact_sha256"],
            "source_archive_sha256": hashlib.sha256(
                (root / "saved-source.tar.gz").read_bytes()
            ).hexdigest(),
            "test_data_opened": False,
            "cases": [entry],
            "settings": settings,
            "comparison_sha256": "synthetic-comparison",
        },
    )
    sealed_write(
        root / "controls/p.json",
        {
            "problem": parent["problem"],
            "problem_sha256": content_hash(parent["problem"]),
            "control_unchanged": True,
        },
    )
    raw = patch()
    old = cli.engine.assess_candidate(parent, raw, candidate_id="call0-option0")
    assert old["status"] == "eligible"
    request = {
        "namespace": entry["namespace"],
        "settings": settings,
        "body": {
            "seed": int(content_hash([0, "search_policy", 0])[:8], 16) % 2**31,
            "messages": [
                {"role": "system", "content": "Historical system"},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "current_model": {
                                "problem_sha256": content_hash(parent["problem"])
                            },
                            "public_task": parent["public_brief"],
                        }
                    ),
                },
            ],
        },
    }
    record = {
        "request": request,
        "request_hash": content_hash(request),
        "attempt": 0,
        "step": "search_policy",
        "budget_charge": 31,
        "observed_total_tokens": 31,
        "status": "responded",
        "latency_seconds": 0.2,
        "raw_response": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "options": [option(0, raw)],
                                "fewer_options_reason": "One was allowed historically.",
                            }
                        ),
                    },
                }
            ]
        },
    }
    directory = root / "results" / entry["case_id"]
    atomic_json(directory / "calls" / f"{record['request_hash']}.json", record)
    result = {
        "status": "eligible_found",
        "generation_calls": 1,
        "candidate_count": 1,
        "candidates": [old],
        "attempts": [
            {
                "attempt": 0,
                "request_hash": record["request_hash"],
                "mode": "options",
                "parse_error": None,
                "candidate_ids": ["call0-option0"],
            }
        ],
    }
    atomic_json(directory / "result.json", result)
    sealed_write(
        directory / "completion.json",
        {
            "plan_sha256": plan["artifact_sha256"],
            "input_sha256": entry["input_sha256"],
            "calls_sha256": content_hash([record]),
            "tokenizations_sha256": content_hash([]),
            "status": "eligible_found",
            "files": {
                "result.json": hashlib.sha256(
                    (directory / "result.json").read_bytes()
                ).hexdigest(),
            },
        },
    )
    monkeypatch.setitem(
        cli.SOURCE_PINS,
        "20b",
        {
            "plan_sha256": plan["artifact_sha256"],
            "preserved_tree_sha256": content_hash(cli.tree_manifest(root)),
            "original_archive_sha256": "provenance-only",
        },
    )
    return root


def forbidden(*args, **kwargs):
    pytest.fail("No provider, fitter or re-assessment of a completed checkpoint")


def test_replay_keeps_historical_short_option_batch_separate(
    saved, tmp_path, monkeypatch
):
    from autoformalism.fitting import prefix_initialization

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    monkeypatch.setattr(cli.base.ConstructionClient, "call", forbidden)
    before = cli.tree_manifest(saved)
    root = tmp_path / "recheck"
    result = cli.replay({"20b": saved}, root)
    arm = result["arms"]["20b"]
    assert arm["candidates"] == arm["eligible_after"] == 1
    assert arm["historical_batches_short_of_new_rule"] == 1
    assert arm["historical_cost"]["observed_tokens"] == 31
    assert (
        result["llm_calls"]
        == result["optimizer_calls"]
        == result["fitting_exports"]
        == 0
    )
    assert not result["test_data_opened"]
    monkeypatch.setattr(cli.engine, "assess_candidate", forbidden)
    assert cli.replay({"20b": saved}, root) == result
    assert cli.tree_manifest(saved) == before
    assert (root / "PROMPTS.html").exists()
    assert len(list((root / "rows").glob("*.html"))) == 1
    row = next((root / "rows").glob("*.json"))
    value = json.loads(row.read_text())
    value["candidate"]["status"] = "invented"
    atomic_json(row, value)
    with pytest.raises(ValueError, match="digest"):
        cli.replay({"20b": saved}, root)


def test_tree_pin_and_links_fail_closed(saved, tmp_path):
    (saved / "unexpected.txt").write_text("drift")
    with pytest.raises(ValueError, match="tree identity"):
        cli.load_source(saved, "20b")
    link = tmp_path / "link"
    link.symlink_to(saved, target_is_directory=True)
    with pytest.raises(ValueError, match="regular source"):
        cli.tree_manifest(link)
    (saved / "bad-link").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="link or special"):
        cli.tree_manifest(saved)


def test_parent_control_rechecked_beyond_tree_pin(saved, monkeypatch):
    value = sealed_read(saved / "controls/p.json")
    value.pop("artifact_sha256")
    value["control_unchanged"] = False
    atomic_json(
        saved / "controls/p.json", {**value, "artifact_sha256": content_hash(value)}
    )
    monkeypatch.setitem(
        cli.SOURCE_PINS["20b"],
        "preserved_tree_sha256",
        content_hash(cli.tree_manifest(saved)),
    )
    with pytest.raises(ValueError, match="parent control"):
        cli.load_source(saved, "20b")


def test_source_and_output_must_not_overlap(saved):
    for root in (saved, saved / "output", saved.parent):
        with pytest.raises(ValueError, match="separate"):
            cli.replay({"20b": saved}, root)


def test_runtime_drift_prevents_resume(saved, tmp_path, monkeypatch):
    root = tmp_path / "recheck"
    cli.replay({"20b": saved}, root)
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: "changed")
    monkeypatch.setattr(cli.engine, "assess_candidate", forbidden)
    with pytest.raises(ValueError, match="frozen artifact"):
        cli.replay({"20b": saved}, root)


def test_request_hash_rechecked_beyond_tree_pin(saved, monkeypatch):
    path = next(saved.glob("results/*/calls/*.json"))
    record = json.loads(path.read_text())
    record["request"]["body"]["messages"][0]["content"] = "different request"
    atomic_json(path, record)
    monkeypatch.setitem(
        cli.SOURCE_PINS["20b"],
        "preserved_tree_sha256",
        content_hash(cli.tree_manifest(saved)),
    )
    with pytest.raises(ValueError, match="namespace or digest"):
        cli.load_source(saved, "20b")


def test_raw_patch_must_match_provider_reply_even_with_resealed_result(
    saved, monkeypatch
):
    path = next(saved.glob("results/*/result.json"))
    result = json.loads(path.read_text())
    result["candidates"][0]["raw_reply"]["hypothesis"] = "Invented after the call"
    atomic_json(path, result)
    receipt_path = path.parent / "completion.json"
    receipt = sealed_read(receipt_path)
    receipt.pop("artifact_sha256")
    receipt["files"]["result.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(receipt_path, {**receipt, "artifact_sha256": content_hash(receipt)})
    monkeypatch.setitem(
        cli.SOURCE_PINS["20b"],
        "preserved_tree_sha256",
        content_hash(cli.tree_manifest(saved)),
    )
    with pytest.raises(ValueError, match="raw patch differs"):
        cli.load_source(saved, "20b")
