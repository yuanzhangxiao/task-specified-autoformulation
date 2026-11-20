"""Offline transport, checkpoint and scheduler tests for matched topology repair."""

import json
import tarfile

import pytest

from autoformalism.staged_topology import content_hash
from scripts import submit_t2_breadth as submitter
from scripts import topology_repair_20b as cli
from tests.test_construction_initialization_graph_cli import reply, tokenizer


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "case.json"
    source.write_text(
        json.dumps({"shared": ["left", "right"], "case_sha256": "example"})
    )
    monkeypatch.setattr(cli.trial, "verify_case", lambda _: None)
    monkeypatch.setattr(
        cli.trial, "messages", lambda case: ("one edit", json.dumps(case["shared"]))
    )
    monkeypatch.setattr(
        cli.trial,
        "assess_reply",
        lambda _, edit: {
            "status": "distinguishes",
            "edit_sha256": content_hash(edit.edit.model_dump()),
        },
    )
    root = tmp_path / "experiment"
    cli.prepare(root, source)
    return root


def test_five_identical_evidence_draws_and_separate_failure_counts(prepared):
    seen = []

    def transport(url, body, timeout):
        seen.append(body)
        if len(seen) == 1:
            raise TimeoutError("missing response")
        if len(seen) == 2:
            return reply({"edit": "broken"})
        return reply(
            {
                "reason": "The parent lacks this source.",
                "edit": {"kind": "add_dependency", "target": "x", "source": "y"},
            }
        )

    first = cli.run(
        prepared, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert first["status_counts"] == {
        "provider_failure": 1,
        "invalid": 1,
        "distinguishes": 3,
    }
    assert first["public_packet_sha256"] == content_hash(["left", "right"])
    assert first["distinct_witness_breaking_edits"] == 1
    assert first["physical_generation_requests"] == 5 and first["unknown_usage"] == 1
    assert len({b["seed"] for b in seen}) == 5
    assert all(
        b["messages"]
        == [
            {"role": "system", "content": "one edit"},
            {"role": "user", "content": '["left", "right"]'},
        ]
        for b in seen
    )
    assert (
        cli.run(
            prepared, "http://127.0.0.1", transport=transport, token_transport=tokenizer
        )
        == first
    )
    assert len(seen) == 5
    archive = cli.inspect(prepared)
    with tarfile.open(archive["archive"]) as tar:
        assert tar.extractfile("results/seed-100.json").read()
        assert json.load(tar.extractfile("summary.json"))["recorded"] == 5
    receipt = next((prepared / "steps/seed-102/calls").glob("*.json"))
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="changed provider record"):
        cli.status(prepared)


def test_submission_resumes_with_20b_resources(prepared, monkeypatch, tmp_path):
    image = tmp_path / "vllm.sif"
    image.write_bytes(b"offline fixture")
    monkeypatch.setattr(submitter.aces, "_assets", lambda size: (image, {"size": size}))
    monkeypatch.setattr(submitter.aces.harmony_vocab, "verify", lambda _: {})
    calls = []

    def submit(path, argv):
        calls.append(argv)
        return str(1000 + len(calls))

    monkeypatch.setattr(submitter, "_submit_job", submit)
    one = submitter.submit(prepared)
    assert one == submitter.submit(prepared)
    assert calls[0][-1] == "topology-repair" and "--gres=gpu:h100:1" in calls[0]
    assert (
        "--time=01:00:00" in calls[0] and "--job-name=phasec-topo-edit-20b" in calls[0]
    )
    identity = submitter.verify(prepared, check_assets=True)
    assert (
        identity["maximum_generation_requests"] == 5 and identity["model_size"] == "20b"
    )
    monkeypatch.setattr(submitter, "_finished", lambda job: True)
    assert submitter.submit(prepared, resume=True)["job"] == "1002"
