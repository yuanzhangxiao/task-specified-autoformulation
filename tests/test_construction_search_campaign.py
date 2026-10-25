"""Search policies share frozen budgets and survive smoke/full deterministic resume."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.prefit_replay import sealed_read
from scripts import construction_search_campaign as cli
from tests.test_construction_feedback_revision import (  # noqa: F401
    prepared,
    prepared_v2,
    prepared_v3,
)
from tests.test_construction_initialization_graph_cli import reply as response
from tests.test_construction_initialization_graph_cli import tokenizer
from tests.test_structural_revision_campaign import (  # noqa: F401
    revision,
    structural_source,
)


@pytest.fixture
def search_roots(structural_source, tmp_path, monkeypatch):  # noqa: F811
    source, _, _ = structural_source
    runtime, entrypoints = cli.runtime_source_hash(), cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: runtime)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    roots = {size: tmp_path / f"search-{size}" for size in cli.base.MODEL_SPECS}
    for size, root in roots.items():
        cli.prepare(source, root, size)
    return roots


def forbidden(*args, **kwargs):
    pytest.fail("No numerical work, provider spend or changed-cache replay permitted")


def respond(url, body, timeout):
    value = revision()
    if body["response_format"]["json_schema"]["name"] == "OptionsReply":
        value = {
            "options": [
                {
                    "option_id": "nonlinear",
                    "affected_target": "Gp",
                    "premise_changed": "Include nonlinear glucose-memory coupling.",
                    "patch": value,
                }
            ],
            "fewer_options_reason": "One distinct proposal in this fixture.",
        }
    return response(value)


def test_matching_rosters_budgets_and_preview(search_roots):
    plans = {size: cli.verify(root) for size, root in search_roots.items()}
    assert plans["20b"]["comparison_sha256"] == plans["120b"]["comparison_sha256"]
    for size, root in search_roots.items():
        plan = plans[size]
        assert len(plan["cases"]) == 54
        assert len(cli.smoke_case_ids(plan)) == 3
        assert len({e["namespace"] for e in plan["cases"]}) == 54
        assert plan["settings"]["maximum_requests"] == 3
        cli.preview(root)
        assert (root / "PROMPTS.html").exists()
        assert not (root / "results").exists()
    assert (
        search_roots["20b"] / "preview/parent_0-options_shallow.json"
    ).read_bytes() == (
        search_roots["120b"] / "preview/parent_0-options_shallow.json"
    ).read_bytes()


@pytest.mark.parametrize("size", ["20b", "120b"])
def test_smoke_full_exact_resume_and_all_eligible_handoffs(
    search_roots, size, monkeypatch
):
    root = search_roots[size]
    from autoformalism.fitting import prefix_initialization

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    with pytest.raises(ValueError, match="smoke"):
        cli.run(root, "http://127.0.0.1", phase="full", transport=forbidden)
    smoke = cli.run(
        root,
        "http://127.0.0.1",
        phase="smoke",
        transport=respond,
        token_transport=tokenizer,
    )
    assert smoke["smoke_complete"] and smoke["status"] == "incomplete"
    assert sum(r["completed"] for r in smoke["rows"]) == 3
    assert smoke["cost"]["physical_requests"] <= 9
    saved = {
        str(p.relative_to(root)): p.read_bytes()
        for p in (root / "results").rglob("*.json")
    }
    assert (
        cli.run(
            root,
            "http://127.0.0.1",
            phase="smoke",
            transport=forbidden,
            token_transport=forbidden,
        )
        == smoke
    )
    full = cli.run(
        root,
        "http://127.0.0.1",
        phase="full",
        transport=respond,
        token_transport=tokenizer,
    )
    assert full["status"] == "complete"
    assert full["cost"]["physical_requests"] <= 162
    assert sum(r["completed"] for r in full["rows"]) == 54
    assert all(v["eligible_episodes"] == 18 for v in full["by_policy"].values())
    assert all((root / p).read_bytes() == blob for p, blob in saved.items())
    assert (
        cli.run(
            root,
            "http://127.0.0.1",
            phase="full",
            transport=forbidden,
            token_transport=forbidden,
        )
        == full
    )
    packet = sealed_read(next((root / "results").glob("*/fit-*.json")))
    assert not packet["test_data_opened"] and not packet["fitting_launched"]
    assert packet["training_consistency"]["status"] != "conflict"
    assert cli.inspect(root)["archive"].endswith("inspection.tar.gz")


def test_tampering_and_settings_drift_prevent_resume(search_roots, monkeypatch):
    root = search_roots["20b"]
    cli.run(root, "http://127.0.0.1", transport=respond, token_transport=tokenizer)
    path = next((root / "results").glob("*/result.json"))
    value = json.loads(path.read_text())
    value["status"] = "invented"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt"):
        cli.report(root)
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: "changed")
    with pytest.raises(ValueError, match="source"):
        cli.verify(root)


def test_off_loopback_and_unverified_live_fail_closed(search_roots):
    root = search_roots["20b"]
    with pytest.raises(ValueError, match="loopback"):
        cli.run(root, "https://remote.invalid")
    with pytest.raises(ValueError, match="serving"):
        cli.run(root, "http://127.0.0.1")


def test_excessive_provider_charge_blocks_without_renewing_budget(search_roots):
    root = search_roots["20b"]

    def oversized_usage(*args):
        value = respond(*args)
        value["usage"]["total_tokens"] = 98304
        return value

    value = cli.run(
        root, "http://127.0.0.1", transport=oversized_usage, token_transport=tokenizer
    )
    assert value["status_counts"]["budget_blocked"] == 1
    assert not value["smoke_complete"]
    assert value["cost"]["physical_requests"] == 3
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == value
    )
