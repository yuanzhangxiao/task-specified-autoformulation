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
    schema = body["response_format"]["json_schema"]
    if schema["name"] in {"OptionsReply", "FiveOptionsReply"}:
        count = schema["schema"]["properties"]["options"]["minItems"]
        value = {
            "options": [
                {
                    "option_id": f"nonlinear_{index}",
                    "affected_target": "Gp",
                    "premise_changed": "Include nonlinear glucose-memory coupling.",
                    "patch": value,
                }
                for index in range(count)
            ],
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
        assert plan["gate_policy"]["combined_symbolic_initials"] == (
            "combined-symbolic-map-readout-initials-1"
        )
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


def test_breadth_profile_binds_budget_roster_schema_and_exact_resume(
    structural_source, tmp_path, monkeypatch  # noqa: F811
):
    source, _, _ = structural_source
    runtime, entrypoints = cli.runtime_source_hash(), cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: runtime)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    root = tmp_path / "breadth-five"
    prepared_result = cli.prepare(source, root, "20b", "breadth-five")
    assert prepared_result["maximum_generation_requests"] == 90
    plan = cli.verify(root)
    assert len(plan["cases"]) == 18
    assert len(cli.smoke_case_ids(plan)) == 1
    assert plan["settings"]["maximum_requests"] == 5
    assert plan["settings"]["maximum_total_tokens"] == 163840
    assert plan["settings"]["max_output_tokens"] == 8192
    assert plan["required_options_in_first_call"] == 5
    assert plan["gate_policy"]["combined_symbolic_initials"] == (
        "combined-symbolic-map-readout-initials-1"
    )
    assert plan["policies"] == ["options_shallow"]
    assert "FiveOptionsReply" in plan["schema_sha256"]
    smoke = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=tokenizer
    )
    assert smoke["smoke_complete"]
    assert sum(row["completed"] for row in smoke["rows"]) == 1
    assert smoke["cost"]["physical_requests"] == 1
    assert smoke["rows"][0]["eligible_candidates"] == 5
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == smoke
    )
    cli.preview(root)
    preview = json.loads((root / "preview/parent_0-options_shallow.json").read_text())
    assert preview["response_schema"]["properties"]["options"]["minItems"] == 5
    assert "exactly five" in preview["user"]
    full = cli.run(
        root,
        "http://127.0.0.1",
        phase="full",
        transport=respond,
        token_transport=tokenizer,
    )
    assert full["status"] == "complete"
    assert len(full["rows"]) == 18
    assert full["profile"] == "breadth-five"
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
    before = (root / "plan.json").read_bytes()
    with pytest.raises(ValueError, match=r"frozen|differs"):
        cli.prepare(source, root, "20b", "three-policy")
    assert (root / "plan.json").read_bytes() == before


def test_unknown_profile_is_rejected_before_writing(
    structural_source, tmp_path  # noqa: F811
):
    source, _, _ = structural_source
    root = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unknown construction search profile"):
        cli.prepare(source, root, "20b", "unlimited")
    assert not root.exists()
