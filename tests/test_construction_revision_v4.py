"""The saved-case pilot preserves provenance and exact durable replay."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research import construction_saved_revision_sources as sources
from scripts import construction_feedback_revision as cli
from tests.test_construction_feedback_revision import (  # noqa: F401
    prepared,
    prepared_v2,
    prepared_v3,
    science_reply,
)
from tests.test_construction_initialization_graph_cli import reply as response
from tests.test_construction_initialization_graph_cli import tokenizer
from tests.test_construction_revision_edits import problem


@pytest.fixture
def prepared_v4(prepared_v3, tmp_path, monkeypatch):  # noqa: F811
    """Only the separately tested archival loader is replaced with synthetic data."""
    root, _ = prepared_v3
    inputs = sealed_read(root / "inputs.json")
    inputs.pop("artifact_sha256")
    sidecar = sealed_read(root / "consistency_evidence.json")
    selected = {
        "inputs": inputs,
        "consistency_source": sidecar,
        "parents": dict(list(inputs["parents"].items())[:6]),
        "selection": [],
        "provenance": {"synthetic": True},
    }
    monkeypatch.setattr(sources, "load", lambda path: deepcopy(selected))
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"synthetic fixture; archive loader tested separately")
    target = tmp_path / "v4"
    cli.prepare(archive, target, protocol=cli.V4_PROTOCOL)
    return target


def test_v4_retains_source_and_18_seeds_and_exact_resume(prepared_v4):
    root = prepared_v4
    plan = cli.verify(root)
    assert len(plan["cases"]) == 18
    assert len(sealed_read(root / "inputs.json")["parents"]) == 8
    assert len(sealed_read(root / "selected_sources.json")["parents"]) == 6
    cli.preview(root)
    calls = []

    def respond(url, body, timeout):
        calls.append(body)
        packet = json.loads(body["messages"][1]["content"])
        assert "initial_family_consistency" in packet["training_consistency"]
        assert "affine_family_consistency" in packet["training_consistency"]
        return response(science_reply())

    summary = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=tokenizer
    )
    assert summary["unchanged_controls"] == 6
    assert not summary["status_counts"].get("pending")
    assert summary["cost"]["physical_requests"] == len(calls)

    def forbidden(*args):
        pytest.fail("completed calls must replay")

    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == summary
    )
    assert cli.inspect(root)["status"] == "complete"


def test_v4_repeated_patch_despite_changed_prose_and_sources(prepared_v4):
    root = prepared_v4
    parent = cli._prepared_parent(
        root, sealed_read(root / "inputs.json"), "parent_0", cli.verify(root)
    )
    packets = []

    class Client:
        def call(self, **kwargs):
            packets.append(json.loads(kwargs["user"]))
            bad = science_reply(
                edits=[
                    {
                        "component": "I",
                        "kind": "dynamic",
                        "sources": ["I"] if len(packets) == 2 else [],
                        "expression": "1",
                    }
                ],
                hypothesis=f"new wording {len(packets)}",
            )
            return {
                "status": "responded",
                "request_hash": str(len(packets)),
                "raw_response": response(bad),
            }

    result = cli._episode(parent, Client())
    assert result["status"] == "rejected"
    feedback = packets[2]["repair_feedback"]
    assert not feedback["repeated_reply"]
    assert feedback["repeated_equation_patch"]
    assert feedback["repeated_patch_attempts"] == [0]
    assert parent["problem"] == problem().model_dump(mode="json")


def test_v4_alias_does_not_bypass_model_data_gate(prepared_v4):
    root = prepared_v4
    parent = cli._prepared_parent(
        root, sealed_read(root / "inputs.json"), "parent_0", cli.verify(root)
    )
    reply = science_reply(
        edits=[{"component": "I", "kind": "dynamic", "sources": [], "expression": "1"}]
    )
    reply["evidence_ids"] = ["function_consistency"]
    with pytest.raises(ValueError, match="MODEL_DATA_CONFLICT"):
        cli._apply(parent, reply)


def test_v4_interrupted_replay_keeps_original_budget(prepared_v4):
    root = prepared_v4
    calls = []

    def interrupted(url, body, timeout):
        calls.append(body)
        if len(calls) == 2:
            raise DeferredCall("stop")
        return response(
            science_reply(
                edits=[
                    {
                        "component": "I",
                        "kind": "dynamic",
                        "sources": [],
                        "expression": "1",
                    }
                ]
            )
        )

    cli.run(root, "http://127.0.0.1", transport=interrupted, token_transport=tokenizer)
    summary = cli.run(
        root,
        "http://127.0.0.1",
        transport=lambda *a: response(science_reply()),
        token_transport=tokenizer,
    )
    assert not summary["status_counts"].get("pending")
    assert summary["cost"]["physical_requests"] <= 55
    assert summary["cost"]["requests_with_unknown_usage"] == 1


@pytest.mark.parametrize("raw", [None, [], "text"])
def test_v4_non_object_reply_remains_a_bounded_schema_failure(prepared_v4, raw):
    root = prepared_v4
    parent = cli._prepared_parent(
        root, sealed_read(root / "inputs.json"), "parent_0", cli.verify(root)
    )
    packets = []

    class Client:
        def call(self, **kwargs):
            packets.append(json.loads(kwargs["user"]))
            return {
                "status": "responded",
                "request_hash": str(len(packets)),
                "raw_response": response(raw),
            }

    result = cli._episode(parent, Client())
    assert result["status"] == "rejected"
    assert len(packets) == 3
    assert all(
        a["diagnostic"]["code"] == "RESPONSE_SCHEMA_OR_AVAILABILITY"
        for a in result["attempts"]
    )
