"""Historical actions require sealed source, calls and actual rejection evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile

import pytest

from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
)
from autoformalism.search.training_evidence import build_training_evidence
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as original
from scripts import repair_function_handoff as repair
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_staged_function_graph_initialization import fixture
from tests.test_training_evidence import training


def _json(path):
    return json.loads(path.read_text())


def _write(path, value, seal=None):
    if seal is not None:
        value = {key: item for key, item in value.items() if key != seal}
        value[seal] = content_hash(value)
    path.write_text(json.dumps(value))


def _archive(root):
    """Exercise the real bounded archive reader without extracting or networking."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(root)))
    return stream.getvalue()


@pytest.fixture
def historical(tmp_path, monkeypatch):
    """Run both original terminal schemas using only deterministic fake responses."""
    brief, context, source, contract, _, client, _ = fixture(tmp_path / "toy")
    graph = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z",
                    "rule": {
                        "mode": "map",
                        "state_initials": [],
                        "public_initials": ["x"],
                    },
                    "justification": "Recorded preparation identifies latent initial",
                }
            ]
        }
    )
    source.update(
        initialization_graph_contract=contract.model_dump(mode="json"),
        initialization_graph_handoff=graph.model_dump(mode="json"),
    )
    evidence = build_training_evidence(training(), context).model_dump(mode="json")
    cases = {
        f"toy__proposal_seed{seed}": {
            "entry": {"source_model_id": "toy", "seed": seed},
            "result": {
                "result_sha256": content_hash(["accepted", seed]),
                "compatibility_status": "deferred",
            },
            "brief": brief.model_dump(mode="json"),
            "case": {
                "context": context.model_dump(mode="json"),
                "training_evidence": {},
            },
            "function_source": copy.deepcopy(source),
            "training_evidence": evidence,
        }
        for seed in range(2)
    }
    loaded = {
        "source_plan": {
            "artifact_sha256": content_hash("source"),
            "settings": client.settings.model_dump(mode="json"),
        },
        "source_summary": {"cost": {"physical_requests": 7}},
        "roster": [{"case_id": name, "accepted": True} for name in cases],
        "cases": cases,
    }
    for module in (original, repair):
        monkeypatch.setattr(
            module, "load_sources", lambda source: copy.deepcopy(loaded)
        )
    source_path = tmp_path / "source.tar.gz"
    source_path.write_bytes(b"verified synthetic source")
    root = tmp_path / "historical"
    original.prepare(source_path, root)
    graph_requests = 0

    def transport(url, body, timeout):
        nonlocal graph_requests
        packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        if "selected_initializer_dependency" in packet:
            graph_requests += 1
            return reply(
                {
                    "expression": (
                        "public_1" if graph_requests == 1 else "z(0)=public_1"
                    ),
                    "parameters": [],
                }
            )
        terms = packet.get("selected_equation", {}).get("terms")
        if terms is not None:
            return reply(
                {
                    "functions": [
                        {
                            "expression": "+".join(term["sources"]) or "1",
                            "parameters": [],
                        }
                        for term in terms
                    ]
                }
            )
        return reply(
            {
                "expression": "+".join(packet["selected_term"]["sources"]) or "1",
                "parameters": [],
            }
        )

    summary = original.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert summary["status_counts"] == {"complete": 1, "failed": 1}
    return root


def _failed(root):
    return root / "results/toy__proposal_seed1"


def _refresh_receipt(directory):
    receipt = _json(directory / "completion.json")
    receipt["files"] = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in receipt["files"]
    }
    receipt["calls_sha256"] = content_hash(
        [_json(path) for path in sorted((directory / "calls").glob("*.json"))]
    )
    _write(directory / "completion.json", receipt, "artifact_sha256")


def test_actual_legacy_schemas_and_enriched_public_brief(historical):
    history = repair.load_history(_archive(historical))
    success = history["cases"]["toy__proposal_seed0"]
    failure = history["cases"]["toy__proposal_seed1"]
    assert success["action"] == "migrate_saved_initializer"
    assert "status" not in success["result"]["initialization"]
    assert success["rejected_initializer_choices"] == []
    assert failure["action"] == "retry_failed_initializer"
    assert failure["result"]["initialization"] is None
    assert failure["initializer_checkpoint"]["status"] == "running"
    assert failure["rejected_initializer_choices"] == [
        {"expression": "z(0)=public_1", "parameters": []},
        {"expression": "z(0)=public_1", "parameters": []},
    ]
    assert history["cost"]["physical_requests"] > 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("short", "was not exhausted"),
        ("accepted", "attempt provenance differs"),
        ("duplicate", "attempt provenance differs"),
        ("state", "attempt provenance differs"),
        ("error", "rejection differs"),
        ("choices", "failed initializer graph differs"),
    ],
)
def test_checkpoint_rejections_require_matching_calls(historical, mutation, message):
    path = _failed(historical) / "initialization_graph/state.json"
    checkpoint = _json(path)
    if mutation == "short":
        checkpoint["attempts"].pop()
    elif mutation == "accepted":
        checkpoint["attempts"][-1]["accepted"] = True
    elif mutation == "duplicate":
        checkpoint["attempts"][-1]["request_hash"] = checkpoint["attempts"][0][
            "request_hash"
        ]
    elif mutation == "state":
        checkpoint["attempts"][-1]["state"] = "other"
    elif mutation == "error":
        checkpoint["attempts"][-1]["error"] = "unverified failure"
    else:
        checkpoint["choices"] = {"z": {"expression": "public_1", "parameters": []}}
    _write(path, checkpoint, "state_sha256")
    with pytest.raises(ValueError, match=message):
        repair.load_history(_archive(historical))


def test_valid_rhs_cannot_be_reclassified_as_historical_syntax_failure(historical):
    directory = _failed(historical)
    checkpoint = _json(directory / "initialization_graph/state.json")
    path = directory / "calls" / f"{checkpoint['attempts'][-1]['request_hash']}.json"
    record = _json(path)
    record["raw_response"]["choices"][0]["message"]["content"] = json.dumps(
        {"expression": "public_1", "parameters": []}
    )
    _write(path, record)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="not a syntax failure"):
        repair.load_history(_archive(historical))


def test_request_must_use_exact_selected_source_graph(historical):
    directory = _failed(historical)
    checkpoint_path = directory / "initialization_graph/state.json"
    checkpoint = _json(checkpoint_path)
    path = directory / "calls" / f"{checkpoint['attempts'][-1]['request_hash']}.json"
    record = _json(path)
    message = record["request"]["body"]["messages"][1]
    prefix, text = message["content"].split("\n", 1)
    packet = json.loads(text)
    packet["selected_initializer_dependency"]["rule"]["public_initials"] = []
    message["content"] = prefix + "\n" + json.dumps(packet)
    record["request_hash"] = content_hash(record["request"])
    path.unlink()
    _write(directory / "calls" / f"{record['request_hash']}.json", record)
    checkpoint["attempts"][-1]["request_hash"] = record["request_hash"]
    _write(checkpoint_path, checkpoint, "state_sha256")
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="request graph differs"):
        repair.load_history(_archive(historical))


def test_historical_accounting_cannot_be_reduced(historical):
    directory = _failed(historical)
    path = directory / "result.json"
    result = _json(path)
    result["budget_charge"] -= 1
    _write(path, result)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="cost or token ceiling differs"):
        repair.load_history(_archive(historical))


def test_agreeing_ledger_and_result_still_enforce_historical_ceiling(historical):
    directory = _failed(historical)
    path = next((directory / "calls").glob("*.json"))
    record = _json(path)
    plan = _json(historical / "plan.json")
    record["budget_charge"] = plan["settings"]["maximum_total_tokens"] + 1
    _write(path, record)
    result_path = directory / "result.json"
    result = _json(result_path)
    result["budget_charge"] = sum(
        _json(path)["budget_charge"] for path in (directory / "calls").glob("*.json")
    )
    _write(result_path, result)
    _refresh_receipt(directory)
    with pytest.raises(ValueError, match="cost or token ceiling differs"):
        repair.load_history(_archive(historical))


def test_complete_checkpoint_must_match_verified_initializer(historical):
    path = historical / "results/toy__proposal_seed0/initialization_graph/state.json"
    checkpoint = _json(path)
    checkpoint["choices"] = {}
    _write(path, checkpoint, "state_sha256")
    with pytest.raises(ValueError, match="complete initializer checkpoint differs"):
        repair.load_history(_archive(historical))
