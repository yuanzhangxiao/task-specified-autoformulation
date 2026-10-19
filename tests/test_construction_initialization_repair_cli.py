"""Fixed-model multiseed repair preserves costs, source identities and bounds."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts import construction_initialization_graph as legacy
from scripts import construction_initialization_repair as cli
from tests.test_construction_initialization_graph_cli import (
    historical_plan,
    reply,
    tokenizer,
    write_archive,
)


def old_choice():
    return {
        "records": [
            {
                "state": "z",
                "rule": {"mode": "value", "scope": "shared_training"},
                "justification": "Shared initial preparation.",
            }
        ]
    }


def trajectory_choice():
    return {
        "decision": "revise",
        "records": [
            {
                "state": "z",
                "rule": {"mode": "value", "scope": "trajectory"},
                "justification": "Unknown trajectory preparation inferred from prefix.",
            }
        ],
        "explanation": "Independent hidden initial preparation can vary.",
    }


@pytest.fixture
def source(tmp_path):
    ancestor, previous = tmp_path / "ancestor.tar.gz", tmp_path / "previous"
    write_archive(ancestor, historical_plan())
    legacy.prepare(ancestor, previous)
    legacy.run(
        previous,
        "http://127.0.0.1:8000",
        transport=lambda *args: reply(old_choice()),
        token_transport=tokenizer,
    )
    return Path(legacy.inspect(previous)["archive"])


@pytest.fixture
def prepared(source, tmp_path):
    root = tmp_path / "repair"
    cli.prepare(source, root)
    return source, root


def test_prepare_five_seeds_and_single_history_per_original_model(prepared):
    source, root = prepared
    plan = cli.verify(root)
    assert plan["proposal_seeds"] == [0, 1, 2, 3, 4]
    assert len(plan["cases"]) == 10
    assert len({entry["namespace"] for entry in plan["cases"]}) == 10
    assert len(list((root / "inputs").glob("*.json"))) == 2
    assert (root / "historical/source.tar.gz").read_bytes() == source.read_bytes()
    assert plan["settings"]["maximum_requests"] == 3
    assert plan["settings"]["attempts_per_step"] == 3
    assert plan["settings"]["maximum_total_tokens"] == 393216
    for name in plan["source_models"]:
        assert {
            row["seed"] for row in plan["cases"] if row["source_model_id"] == name
        } == set(range(5))
        model = sealed_read(root / "inputs" / f"{name}.json")
        assert (
            model["initial_handoff"]["records"][0]["rule"]["scope"] == "shared_training"
        )
    summary = cli.report(root)
    assert summary["maximum_calls"] == 30
    assert summary["cost"]["physical_requests"] == 0
    assert summary["historical_source_diagnostic_cost"]["physical_requests"] == 2
    assert len(summary["historical_by_source_model"]) == 2
    original = (root / "plan.json").read_bytes()
    cli.prepare(source, root)
    assert (root / "plan.json").read_bytes() == original
    assert not (root / "results").exists()


@pytest.mark.parametrize("feedback_policy", ["legacy-1", "scope-aware-2"])
def test_run_resume_and_previews_match_actual_requests(
    source, tmp_path, feedback_policy
):
    root = tmp_path / "repair"
    cli.prepare(source, root, feedback_policy=feedback_policy)
    initial = source.read_bytes()
    requests = []

    def transport(url, body, timeout):
        requests.append(body)
        return reply(trajectory_choice())

    result = cli.run(
        root, "http://127.0.0.1:8000", transport=transport, token_transport=tokenizer
    )
    assert len(requests) == 10
    assert result["accepted_episodes"] == 10
    assert result["structurally_valid_episodes"] == 10
    assert result["compatibility_status_counts"] == {"deferred": 10}
    assert result["cost"]["physical_requests"] == 10
    assert result["cost"]["observed_tokens"] == 310
    assert not result["scientific_correctness_established"]
    for entry, request in zip(cli.verify(root)["cases"], requests, strict=True):
        preview = json.loads(
            (root / "preview" / f"{entry['case_id']}.json").read_text()
        )
        assert request["messages"] == [
            {"role": "system", "content": preview["system"]},
            {"role": "user", "content": preview["user"]},
        ]
        assert (
            request["response_format"]["json_schema"]["schema"]
            == preview["response_schema"]
        )

    def blocked(*args):
        pytest.fail("terminal resume must not call the provider or tokenizer")

    assert (
        cli.run(
            root, "http://127.0.0.1:8001", transport=blocked, token_transport=blocked
        )
        == result
    )
    assert source.read_bytes() == initial


def test_scope_policy_is_sealed_and_cannot_overwrite_legacy_history(prepared, tmp_path):
    source, legacy_root = prepared
    root = tmp_path / "scope-repair"
    cli.prepare(source, root, feedback_policy="scope-aware-2")
    old = cli.verify(legacy_root)
    new = cli.verify(root)
    assert "feedback_policy" not in old
    assert new["feedback_policy"] == "scope-aware-2"
    assert old["protocol"] == cli.PROTOCOL
    assert new["protocol"] == "saved-construction-initialization-repair-2"
    assert new["proposal_seeds"] == old["proposal_seeds"]
    assert new["source_archive_sha256"] == old["source_archive_sha256"]
    assert new["source_models"] == old["source_models"]
    assert new["settings"] == old["settings"]
    assert new["budgets"] == old["budgets"]
    assert {case["namespace"] for case in new["cases"]}.isdisjoint(
        case["namespace"] for case in old["cases"]
    )
    assert (
        "scripts/hpc/start_initialization_scope_repair_aces.sh"
        in new["entrypoint_sha256"]
    )
    assert (
        "scripts/hpc/start_initialization_scope_repair_aces.sh"
        not in old["entrypoint_sha256"]
    )
    legacy_plan = (legacy_root / "plan.json").read_bytes()
    with pytest.raises(ValueError, match="fresh root"):
        cli.prepare(source, legacy_root, feedback_policy="scope-aware-2")
    assert (legacy_root / "plan.json").read_bytes() == legacy_plan
    assert cli.report(root)["feedback_policy"] == "scope-aware-2"
    assert cli.report(root)["protocol"] == new["protocol"]
    with pytest.raises(ValueError, match=r"unknown.*policy"):
        cli.prepare(source, tmp_path / "bad", feedback_policy="guess")
    assert not (tmp_path / "bad").exists()


def test_resealed_policy_downgrade_cannot_reuse_scope_identity(source, tmp_path):
    root = tmp_path / "scope-repair"
    cli.prepare(source, root, feedback_policy="scope-aware-2")
    plan = sealed_read(root / "plan.json")
    plan["feedback_policy"] = "legacy-1"
    plan["artifact_sha256"] = content_hash(
        {key: value for key, value in plan.items() if key != "artifact_sha256"}
    )
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="identity"):
        cli.verify(root)


def test_conflicting_replies_are_never_accepted_and_cap_is_three(prepared):
    _, root = prepared
    body = {
        "decision": "revise",
        "explanation": "Keep the shared preparation.",
        **old_choice(),
    }
    result = cli.run(
        root,
        "http://127.0.0.1:8000",
        transport=lambda *args: reply(body),
        token_transport=tokenizer,
    )
    assert result["cost"]["physical_requests"] == 30
    assert result["status_counts"] == {"repair_exhausted": 10}
    assert result["accepted_episodes"] == 0
    assert result["structurally_valid_episodes"] == 10
    assert result["compatibility_status_counts"] == {"conflict": 10}
    assert all(row["conflict_count"] >= 1 for row in result["rows"])
    assert result["historical_source_diagnostic_cost"]["physical_requests"] == 2


def test_topology_escalation_finishes_unaccepted_without_followup(prepared):
    _, root = prepared
    result = cli.run(
        root,
        "http://127.0.0.1:8000",
        token_transport=tokenizer,
        transport=lambda *args: reply(
            {
                "decision": "needs_topology_revision",
                "records": [],
                "explanation": "The frozen readout omits another causal state.",
            }
        ),
    )
    assert result["status_counts"] == {"needs_topology_revision": 10}
    assert result["accepted_episodes"] == 0
    assert result["cost"]["physical_requests"] == 10
    assert not result["automatic_followup"]


def test_provider_failure_remains_charged_without_retry(prepared):
    _, root = prepared
    calls = []

    def fail(*args):
        calls.append(args)
        raise OSError("synthetic provider unavailable")

    result = cli.run(
        root, "http://127.0.0.1:8000", transport=fail, token_transport=tokenizer
    )
    assert result["status_counts"] == {"unavailable": 10}
    assert result["cost"]["requests_with_unknown_usage"] == 10
    assert (
        cli.run(
            root, "http://127.0.0.1:8000", transport=fail, token_transport=tokenizer
        )
        == result
    )
    assert len(calls) == 10


def rewrite_archive(source, target, mutate):
    with tarfile.open(source) as original, tarfile.open(target, "w:gz") as output:
        for member in original.getmembers():
            if not member.isfile():
                continue
            blob = original.extractfile(member).read()
            blob = mutate(member.name, blob)
            item = tarfile.TarInfo(member.name)
            item.size = len(blob)
            output.addfile(item, io.BytesIO(blob))


@pytest.mark.parametrize("kind", ["cost", "source_reply", "input", "heldout"])
def test_forged_source_is_rejected_before_new_requests(source, tmp_path, kind):
    def mutate(name, blob):
        if name == "summary.json" and kind in {"cost", "heldout"}:
            value = json.loads(blob)
            if kind == "cost":
                value["cost"]["physical_requests"] = 999
            else:
                value["test_data_opened"] = True
            return json.dumps(value).encode()
        if "/calls/" in name and kind == "source_reply":
            value = json.loads(blob)
            value["raw_response"] = reply({"records": []})
            return json.dumps(value).encode()
        if name.startswith("inputs/") and kind == "input":
            value = json.loads(blob)
            value["scientific_context"] = "Changed public context."
            value["artifact_sha256"] = content_hash(
                {k: v for k, v in value.items() if k != "artifact_sha256"}
            )
            return json.dumps(value).encode()
        return blob

    broken = tmp_path / "broken.tar.gz"
    rewrite_archive(source, broken, mutate)
    with pytest.raises(ValueError):
        cli.prepare(broken, tmp_path / "bad")


def test_seed_changes_do_not_reuse_a_plan(prepared):
    _, root = prepared
    plan = sealed_read(root / "plan.json")
    plan["cases"][0]["seed"] = 99
    plan["artifact_sha256"] = content_hash(
        {k: v for k, v in plan.items() if k != "artifact_sha256"}
    )
    (root / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="identity"):
        cli.verify(root)


def test_inspection_is_offline_and_cli_help_standalone(prepared):
    source, root = prepared
    package = cli.inspect(root)
    with tarfile.open(package["archive"]) as archive:
        assert (
            archive.extractfile("historical/source.tar.gz").read()
            == source.read_bytes()
        )
        assert "PROMPTS.html" in archive.getnames()
    subprocess.run(
        [sys.executable, cli.__file__, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )


def test_interrupted_request_is_not_sent_again(prepared):
    _, root = prepared

    def interrupt(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        cli.run(
            root,
            "http://127.0.0.1:8000",
            transport=interrupt,
            token_transport=tokenizer,
        )
    before = cli.report(root)
    assert before["cost"]["physical_requests"] == 1
    called = []

    def respond(*args):
        called.append(args)
        return reply(trajectory_choice())

    result = cli.run(
        root, "http://127.0.0.1:8000", transport=respond, token_transport=tokenizer
    )
    assert len(called) == 9
    assert result["status_counts"] == {"unavailable": 1, "complete": 9}
    assert result["cost"]["physical_requests"] == 10
    assert result["cost"]["requests_with_unknown_usage"] == 1
    assert result["cost"]["budget_charge"] >= before["cost"]["budget_charge"]


def test_execution_record_cannot_rebind_to_a_different_plan(prepared):
    from autoformalism.rebuttal.prefit_replay import sealed_write

    _, root = prepared
    sealed_write(
        root / "execution.json",
        {
            "protocol": "initialization-repair-aces-execution-1",
            "plan_sha256": "0" * 64,
            "launcher_sha256": cli.legacy._file_hash(
                cli.REPO / "scripts/hpc/run_initializer_choice_aces.sh"
            ),
            "automatic_followup": False,
        },
    )
    with pytest.raises(ValueError, match="execution plan"):
        cli.verify(root)


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://example.com"])
def test_nonlocal_or_unverified_endpoints_do_not_start_calls(prepared, url):
    _, root = prepared
    with pytest.raises(ValueError, match="loopback"):
        cli.run(root, url)
    assert not (root / "results").exists()
