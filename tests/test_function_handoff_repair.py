"""Saved initializer retries preserve successful graphs, dynamics and finite budgets."""

import copy
import json

import pytest

from autoformalism.fitting import prefix_initialization
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
)
from autoformalism.staged_topology import content_hash
from scripts import construction_functions as old
from scripts import repair_function_handoff as repair
from tests.test_construction_initialization_graph_cli import reply, tokenizer
from tests.test_staged_function_graph_initialization import fixture


def valid(url, body, timeout):
    packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
    if "selected_initializer_dependency" in packet:
        aliases = packet["dependency_aliases"]
        names = sorted(aliases["state_initials"]) + sorted(aliases["public_initials"])
        return reply({"expression": "+".join(names), "parameters": []})
    if "selected_equation" in packet:
        return reply(
            {
                "functions": [
                    {"expression": "+".join(t["sources"]) or "1", "parameters": []}
                    for t in packet["selected_equation"]["terms"]
                ]
            }
        )
    return reply(
        {"expression": "+".join(packet["selected_term"]["sources"]), "parameters": []}
    )


def invalid(url, body, timeout):
    return reply({"expression": "z(0) = public_1", "parameters": []})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    brief, context, source, contract, _, client, _ = fixture(tmp_path / "toy")
    handoff = GraphDependencyHandoff.model_validate(
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
        initialization_graph_handoff=handoff.model_dump(mode="json"),
    )
    values = {}
    for seed in range(2):
        values[f"toy__proposal_seed{seed}"] = {
            "entry": {"source_model_id": "toy", "seed": seed},
            "result": {
                "result_sha256": content_hash(seed),
                "compatibility_status": "deferred",
            },
            "brief": brief.model_dump(mode="json"),
            "case": {
                "context": context.model_dump(mode="json"),
                "training_evidence": {},
            },
            "function_source": copy.deepcopy(source),
            "training_evidence": None,
        }
    loaded = {
        "source_plan": {
            "artifact_sha256": content_hash("source"),
            "settings": client.settings.model_dump(mode="json"),
        },
        "source_summary": {"cost": {"physical_requests": 7}},
        "roster": [{"case_id": name, "accepted": True} for name in values],
        "cases": values,
    }
    monkeypatch.setattr(old, "load_sources", lambda source: copy.deepcopy(loaded))
    source_path = tmp_path / "source.tar.gz"
    source_path.write_bytes(b"sealed test source")
    historical = tmp_path / "old"
    old.prepare(source_path, historical)
    old.run(historical, "http://127.0.0.1", transport=valid, token_transport=tokenizer)
    historical_plan = old.verify(historical)
    cases = {}
    for name, value in values.items():
        result = json.loads((historical / "results" / name / "result.json").read_text())
        assert result["status"] == "complete", result.get("error")
        action = "migrate_saved_initializer"
        if value["entry"]["seed"] == 1:
            result.update(status="failed", complete_model=False, initialization=None)
            action = "retry_failed_initializer"
        cases[name] = {"source": value, "result": result, "action": action}
    history = {
        "plan": historical_plan,
        "cases": cases,
        "cost": {"physical_requests": 8},
    }
    monkeypatch.setattr(repair, "load_history", lambda source: copy.deepcopy(history))
    root = tmp_path / "repair"
    repair.prepare(source_path, root)
    return root, history


def forbidden(*args, **kwargs):
    pytest.fail("not authorized")


def test_offline_migration_then_map_only_live_resume(prepared, monkeypatch):
    root, history = prepared
    for name in ("fit_training", "infer_prefix", "forecast", "score_forecast"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    summary = repair.report(root)
    assert summary["status_counts"] == {"complete": 1, "pending": 1}
    assert summary["cost"]["physical_requests"] == 0
    assert summary["maximum_calls"] == 2
    assert summary["historical_function_cost"] == history["cost"]
    seen = []

    def capture(url, body, timeout):
        packet = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        assert "selected_initializer_dependency" in packet
        seen.append(body["messages"])
        return valid(url, body, timeout)

    repair.preview(root)
    preview = json.loads((root / "preview/toy__proposal_seed1.json").read_text())
    result = repair.run(
        root, "http://127.0.0.1", transport=capture, token_transport=tokenizer
    )
    assert result["status_counts"] == {"complete": 2}
    assert result["cost"]["physical_requests"] == 1
    assert seen[0] == [
        {"role": "system", "content": preview["system"]},
        {"role": "user", "content": preview["user"]},
    ]
    assert all(row["fitting_exported"] for row in result["rows"])
    assert not result["optimizer_calls"] and not result["test_data_opened"]
    assert (
        repair.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )
    assert repair.inspect(root)["status"] == "complete"


def test_two_failed_attempts_terminal_feedback_and_no_budget_reset(prepared):
    root, _ = prepared
    seen = []

    def fail(url, body, timeout):
        seen.append(json.loads(body["messages"][1]["content"].split("\n", 1)[1]))
        return invalid(url, body, timeout)

    result = repair.run(
        root, "http://127.0.0.1", transport=fail, token_transport=tokenizer
    )
    assert result["status_counts"] == {"complete": 1, "failed": 1}
    assert result["cost"]["physical_requests"] == 2
    assert result["availability_status"] == "available"
    assert (
        seen[1]["diagnostics"][0]["repair_feedback"]["code"] == "INITIALIZER_RHS_ONLY"
    )
    assert (
        repair.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )


def test_drain_then_resume_and_unknown_usage(prepared):
    root, _ = prepared
    result = repair.run(
        root,
        "http://127.0.0.1",
        seconds=1,
        transport=forbidden,
        token_transport=tokenizer,
    )
    assert result["status_counts"] == {"complete": 1, "pending": 1}

    def timeout(*args):
        raise TimeoutError("provider interruption")

    result = repair.run(
        root, "http://127.0.0.1", transport=timeout, token_transport=tokenizer
    )
    assert result["availability_status"] == "unavailable"
    assert result["cost"]["physical_requests"] <= 2
    assert result["cost"]["requests_with_unknown_usage"] > 0
    assert (
        repair.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )


def test_completed_receipts_and_inputs_cannot_be_silently_changed(prepared):
    root, _ = prepared
    path = root / "results/toy__proposal_seed0/fit_handoff.json"
    saved = path.read_bytes()
    value = json.loads(saved)
    value["readiness"] = {}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact differs"):
        repair.report(root)
    path.write_bytes(saved)
    path = root / "inputs/toy__proposal_seed1.json"
    value = sealed_read(path)
    value["action"] = "migrate_saved_initializer"
    value["artifact_sha256"] = content_hash(
        {k: v for k, v in value.items() if k != "artifact_sha256"}
    )
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="input differs"):
        repair.verify(root)


def test_no_unverified_live_endpoint(prepared):
    root, _ = prepared
    with pytest.raises(ValueError, match="loopback"):
        repair.run(root, "http://example.invalid", transport=valid)
    with pytest.raises(ValueError, match="runtime record"):
        repair.run(root, "http://127.0.0.1")


def test_choices_must_match_migration_and_actual_cached_reply(prepared):
    root, _ = prepared
    repair.run(root, "http://127.0.0.1", transport=valid, token_transport=tokenizer)
    plan = repair.verify(root)
    for entry in plan["cases"]:
        directory = root / "results" / entry["case_id"]
        value = sealed_read(root / "inputs" / f"{entry['case_id']}.json")
        lowered = repair._lower(value)
        result = json.loads((directory / "result.json").read_text())
        result["choices"]["z"]["expression"] = "2*public_1"
        result["result_sha256"] = content_hash(
            {k: v for k, v in result.items() if k != "result_sha256"}
        )
        if entry["action"] == "retry_failed_initializer":
            path = directory / "initializer/state.json"
            state = json.loads(path.read_text())
            state.update(result=result, choices=result["choices"])
            state["state_sha256"] = content_hash(
                {k: v for k, v in state.items() if k != "state_sha256"}
            )
            path.write_text(json.dumps(state))
        with pytest.raises(
            ValueError, match=r"differs from saved|differ from accepted"
        ):
            repair._verify_result(
                directory, entry, value, lowered, result, old._calls(root, entry, plan)
            )
