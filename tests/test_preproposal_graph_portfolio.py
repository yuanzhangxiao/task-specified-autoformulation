"""End-to-end check of all four architecture paths, ID selection and resume."""

import json

import pytest

from scripts import preproposal_graph_portfolio as portfolio
from scripts import preproposal_selection_20b as cli
from scripts import submit_t2_breadth as submit
from tests import test_preproposal_selection_20b as previous
from tests.test_construction_initialization_graph_cli import reply, tokenizer

prepared = previous.prepared


def test_matched_graph_selection_and_resume(prepared, tmp_path, monkeypatch):
    _, original = prepared
    source = tmp_path / "with-events.json"
    source.write_text(
        json.dumps({**json.loads(original.read_text()), "variable_exits": {}})
    )
    root = tmp_path / "portfolio"
    monkeypatch.setattr(portfolio, "SEEDS", (100,))
    monkeypatch.setattr(portfolio, "MAX_GRAPHS", 4)
    monkeypatch.setattr(
        portfolio,
        "portfolio_cases",
        lambda frozen, events: {
            k: dict(v, check_all_measured=True, allow_topology_latents=True)
            for k, v in frozen.items()
        },
    )
    plan = cli.prepare(root, source, complete_graphs=True)
    assert plan == cli.prepare(root, source, complete_graphs=True)
    assert submit._trial(plan["protocol"]) is cli
    preflight = cli.preflight(root)
    assert all(r["eligible_graphs"] for r in preflight["arms"])
    previews = cli.preview(root)
    assert all("rank_graphs" in ex and "graph_first" in ex for ex in previews)
    assert "rank_sources" not in previews[1]
    assert cli.status(root)["status"] == "incomplete"
    calls = []

    def provider(url, body, timeout):
        req = json.loads(body["messages"][1]["content"])
        calls.append(req)
        if "graphs" in req:
            return reply({"graph_ids": [r["graph_id"] for r in req["graphs"][:3]]})
        if "candidates" in req:
            return reply({"source_ids": [req["candidates"][0]["candidate_id"]]})
        return reply(
            {
                "hypothesis": "Storage transmits forcing to level.",
                "sources": {
                    "level": ["upstream", "rain", "spill"],
                    "upstream": ["spill"],
                },
                "latent_initializers": [],
                "new_latents": [],
            }
        )

    summary = cli.run(
        root, "http://localhost", transport=provider, token_transport=tokenizer
    )
    assert summary["status"] == "complete"
    assert summary["completed_active_draws"] == 2
    assert (
        summary["physical_generation_requests"]
        == len(calls)
        <= plan["maximum_generation_requests"]
    )
    for group in summary["cases"]:
        audit = group["complete_graphs"]
        ids = {r["candidate_id"] for r in audit["candidates"]}
        draw = audit["draws"][0]
        assert draw["scientific"]["status"] == "selected"
        assert set(draw["scientific"]["graph_ids"]) <= ids
        assert set(draw["uniform_graph_ids"]) <= ids
    assert (
        cli.run(root, "http://localhost", transport=provider, token_transport=tokenizer)
        == summary
    )
    assert len(calls) == summary["physical_generation_requests"]
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="inputs differ"):
        cli.prepare(root, source, complete_graphs=True)
