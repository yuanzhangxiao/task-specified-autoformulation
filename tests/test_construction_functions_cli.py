"""Saved formulas preserve source identity, scoped export and restart budgets."""

from __future__ import annotations

import copy
import json

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as cli
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_staged_function_graph_initialization import fixture


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Use the real function runner on an explicit toy graph, without source IO."""
    parts = fixture(tmp_path / "toy", dependent=True)
    brief, context, source, contract, handoff, _, _ = parts
    source.update(
        initialization_graph_contract=contract.model_dump(mode="json"),
        initialization_graph_handoff=handoff.model_dump(mode="json"),
    )
    cases = {}
    for seed in range(2):
        cases[f"toy__proposal_seed{seed}"] = {
            "entry": {"source_model_id": "toy", "seed": seed},
            "result": {
                "result_sha256": content_hash(["accepted", seed]),
                "compatibility_status": "deferred",
            },
            "brief": brief.model_dump(mode="json"),
            "case": {
                "context": context.model_dump(mode="json"),
                "training_evidence": {"split": "train", "deferred": True},
            },
            "function_source": source,
            "training_evidence": None,
        }
    loaded = {
        "source_plan": {
            "artifact_sha256": content_hash("source"),
            "settings": parts[5].settings.model_dump(mode="json"),
        },
        "source_summary": {"cost": {"physical_requests": 10}},
        "roster": [{"case_id": name, "accepted": True} for name in cases]
        + [{"case_id": "blocked", "accepted": False}],
        "cases": cases,
    }
    monkeypatch.setattr(cli, "load_sources", lambda source: copy.deepcopy(loaded))
    original = tmp_path / "source.tar.gz"
    original.write_bytes(b"verified synthetic source")
    root = tmp_path / "campaign"
    cli.prepare(original, root)
    return root


def transport(url, body, timeout):
    """Supply valid toy function batches, exact aliases and an independent root."""
    packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
    if "selected_initializer_dependency" in packet:
        if packet["selected_initializer_dependency"]["rule"]["mode"] == "value":
            return reply({"role": "coefficient"})
        return reply({"expression": "state_1+public_1", "parameters": []})
    selected = packet.get("selected_equation", {}).get("terms")
    if selected is not None:
        return reply(
            {
                "functions": [
                    {"expression": "+".join(item["sources"]) or "1", "parameters": []}
                    for item in selected
                ]
            }
        )
    return reply(
        {
            "expression": "+".join(packet["selected_term"]["sources"]) or "1",
            "parameters": [],
        }
    )


def test_complete_export_resume_and_no_selection(prepared, monkeypatch):
    from autoformalism.fitting import prefix_initialization

    def forbidden(*args, **kwargs):
        pytest.fail("numerical fitting/forecasting is not authorized")

    for name in ("fit_training", "infer_prefix", "forecast", "score_forecast"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    result = cli.run(
        prepared,
        "http://127.0.0.1:8000",
        transport=transport,
        token_transport=tokenizer,
    )
    assert result["status_counts"] == {"complete": 2}
    assert result["cost"]["physical_requests"] == 10
    assert len(result["source_roster"]) == 3
    assert all(row["fitting_exported"] for row in result["rows"])
    assert result["optimizer_calls"] == result["solver_rollouts"] == 0
    assert not result["test_data_opened"]
    again = cli.run(
        prepared,
        "http://127.0.0.1:8000",
        transport=forbidden,
        token_transport=forbidden,
    )
    assert again == result
    for path in (prepared / "results").glob("*/fit_handoff.json"):
        export = sealed_read(path)
        assert (
            export["prefix_fit_problem"]["graph"]["rules"]["z"]["initial"]["scope"]
            == "trajectory"
        )
    assert cli.inspect(prepared)["status"] == "complete"


def test_preview_is_exact_first_request_and_starts_no_generation(prepared):
    assert cli.preview(prepared)["llm_calls"] == 0
    first = json.loads((prepared / "preview/toy__proposal_seed0.json").read_text())
    sent = []

    def capture(url, body, timeout):
        sent.append(body["messages"])
        return transport(url, body, timeout)

    cli.run(prepared, "http://127.0.0.1", transport=capture, token_transport=tokenizer)
    assert sent[0] == [
        {"role": "system", "content": first["system"]},
        {"role": "user", "content": first["user"]},
    ]


def test_deferred_work_resumes_without_marking_failure(prepared, monkeypatch):
    original = cli._construct

    def defer(*args):
        raise DeferredCall("allocation draining")

    monkeypatch.setattr(cli, "_construct", defer)
    result = cli.run(
        prepared, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert result["status_counts"] == {"pending": 2}
    monkeypatch.setattr(cli, "_construct", original)
    result = cli.run(
        prepared, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert result["status_counts"] == {"complete": 2}


def test_changed_completed_result_is_rejected(prepared):
    cli.run(
        prepared, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    path = next((prepared / "results").glob("*/result.json"))
    value = json.loads(path.read_text())
    value["complete_model"] = False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact differs"):
        cli.report(prepared)


def test_changed_input_and_source_are_rejected(prepared):
    path = next((prepared / "inputs").glob("*.json"))
    saved = path.read_bytes()
    value = json.loads(saved)
    value["entry"]["seed"] = 999
    value["artifact_sha256"] = content_hash(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="input differs"):
        cli.verify(prepared)
    path.write_bytes(saved)
    (prepared / "historical/source.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="plan differs"):
        cli.verify(prepared)


def test_provider_failure_records_terminal_unavailability_and_cost(prepared):
    def fail(*args):
        raise TimeoutError("synthetic timeout")

    result = cli.run(
        prepared, "http://127.0.0.1", transport=fail, token_transport=tokenizer
    )
    assert result["status"] == "complete"
    assert result["availability_status"] == "unavailable"
    assert not any(row["fitting_exported"] for row in result["rows"])
    assert result["cost"]["physical_requests"] == 4
    assert result["cost"]["requests_with_unknown_usage"] == 4


def test_available_invalid_functions_are_not_transport_failure(prepared):
    result = cli.run(
        prepared,
        "http://127.0.0.1",
        transport=lambda *args: reply({"functions": []}),
        token_transport=tokenizer,
    )
    assert result["status_counts"] == {"failed": 2}
    assert result["availability_status"] == "available"


def test_tokenizer_failure_is_unavailable_without_generation(prepared):
    def fail(*args):
        raise TimeoutError("tokenizer timed out")

    def forbidden(*args):
        pytest.fail("generation must not start without token admission")

    result = cli.run(
        prepared, "http://127.0.0.1", transport=forbidden, token_transport=fail
    )
    assert result["availability_status"] == "unavailable"
    assert not any(row["fitting_exported"] for row in result["rows"])
    assert result["cost"]["physical_requests"] == 0
    assert cli.run(
        prepared, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
    ) == result
    receipt = next((prepared / "results").glob("*/calls/tokenization/*.json"))
    value = json.loads(receipt.read_text())
    value["error"] = "changed"
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completion provenance differs"):
        cli.report(prepared)


def test_live_run_requires_serving_receipt(prepared):
    with pytest.raises(ValueError, match="verified runtime"):
        cli.run(prepared, "http://127.0.0.1")
    with pytest.raises(ValueError, match="loopback"):
        cli.run(prepared, "http://remote-host", transport=transport)


@pytest.mark.parametrize("completed_calls", [2, 4])
def test_partial_generation_drain_preserves_calls_and_resumes(
    prepared, monkeypatch, completed_calls
):
    """Drain during functions or initialization without resetting incurred costs."""
    from autoformalism.fitting import prefix_initialization

    def forbidden(*args, **kwargs):
        pytest.fail("numerical fitting/forecasting is not authorized")

    for name in ("fit_training", "infer_prefix", "forecast", "score_forecast"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    construct = cli._construct
    generation_calls = []

    def counted_transport(url, body, timeout):
        generation_calls.append(content_hash(body))
        return transport(url, body, timeout)

    def drain_after_calls(value, client, directory):
        permitted = client.can_start
        client.can_start = (
            lambda: len(client.records) < completed_calls and permitted()
        )
        return construct(value, client, directory)

    monkeypatch.setattr(cli, "_construct", drain_after_calls)
    partial = cli.run(
        prepared,
        "http://127.0.0.1",
        transport=counted_transport,
        token_transport=tokenizer,
    )
    assert partial["status_counts"] == {"pending": 2}
    assert partial["availability_status"] == "pending"
    assert partial["cost"]["physical_requests"] == completed_calls
    assert len(generation_calls) == completed_calls
    saved = {
        path: path.read_bytes()
        for path in (prepared / "results").glob("*/calls/*.json")
    }
    assert len(saved) == completed_calls
    assert not list((prepared / "results").glob("*/completion.json"))
    if completed_calls == 4:
        assert list(
            (prepared / "results").glob("*/initialization_graph/state.json")
        )

    monkeypatch.setattr(cli, "_construct", construct)
    resumed = cli.run(
        prepared,
        "http://127.0.0.1",
        transport=counted_transport,
        token_transport=tokenizer,
    )
    assert resumed["status_counts"] == {"complete": 2}
    assert resumed["availability_status"] == "available"
    assert resumed["cost"]["physical_requests"] == 10
    assert len(generation_calls) == len(set(generation_calls)) == 10
    assert all(path.read_bytes() == blob for path, blob in saved.items())
    assert all(row["fitting_exported"] for row in resumed["rows"])
    assert resumed["optimizer_calls"] == resumed["solver_rollouts"] == 0
    assert resumed["test_data_opened"] is False
