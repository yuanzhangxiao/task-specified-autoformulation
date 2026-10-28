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


@pytest.mark.parametrize(
    "profile,count",
    [("breadth-five", 18), ("focused-retest", 3), ("action-feedback", 3)],
)
def test_breadth_profile_binds_budget_roster_schema_and_exact_resume(
    structural_source,  # noqa: F811
    tmp_path,
    monkeypatch,
    profile,
    count,
):
    source, _, _ = structural_source
    runtime, entrypoints = cli.runtime_source_hash(), cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: runtime)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    root = tmp_path / profile
    prepared_result = cli.prepare(source, root, "20b", profile)
    assert prepared_result["maximum_generation_requests"] == count * 5
    plan = cli.verify(root)
    assert len(plan["cases"]) == count
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
    if profile in {"focused-retest", "action-feedback"}:
        assert {row["parent_id"] for row in plan["cases"]} == {"parent_0"}
        assert {row["seed"] for row in plan["cases"]} == {0, 1, 2}
        assert plan["engine_options"] == {
            "branch_delta": True,
            "common_zero_forcing": True,
            **({"action_feedback": True} if profile == "action-feedback" else {}),
        }
        if profile == "action-feedback":
            assert plan["action_feedback_policy"] == "typed-branch-action-feedback-1"
            assert plan["allowed_model_sizes"] == ["20b"]
        else:
            assert "action_feedback_policy" not in plan
        assert plan["branch_edit_policy"] == "explicit-branch-patch-delta-1"
        assert plan["gate_policy"]["common_zero_forcing"] == (
            "certified-common-zero-forcing-affine-1"
        )
        assert "BranchRepairReply" in plan["schema_sha256"]
        assert plan["smoke_required_before_full"] is False
    else:
        assert "engine_options" not in plan
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
    assert len(full["rows"]) == count
    assert full["profile"] == profile
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
    with pytest.raises(ValueError, match=r"frozen|differ"):
        cli.prepare(source, root, "20b", "three-policy")
    assert (root / "plan.json").read_bytes() == before


def test_unknown_profile_is_rejected_before_writing(
    structural_source,  # noqa: F811
    tmp_path,
):
    source, _, _ = structural_source
    root = tmp_path / "unknown"
    with pytest.raises(ValueError, match="unknown construction search profile"):
        cli.prepare(source, root, "20b", "unlimited")
    assert not root.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("branch_delta", False),
        ("common_zero_forcing", False),
        ("smoke_required_before_full", True),
    ],
)
def test_focused_profile_rejects_flag_drift_before_cached_resume(
    structural_source,  # noqa: F811
    tmp_path,
    monkeypatch,
    field,
    value,
):
    source, _, _ = structural_source
    root = tmp_path / "focused"
    cli.prepare(source, root, "20b", "focused-retest")
    before = (root / "plan.json").read_bytes()
    monkeypatch.setitem(cli.PROFILES["focused-retest"], field, value)
    with pytest.raises(ValueError, match=r"frozen|differ"):
        cli.verify(root)
    assert (root / "plan.json").read_bytes() == before


@pytest.mark.parametrize("profile", ["focused-retest", "action-feedback"])
def test_focused_branch_delta_followup_and_cache_replay_are_identical(
    structural_source,  # noqa: F811
    tmp_path,
    monkeypatch,
    profile,
):
    source, _, _ = structural_source
    runtime, entrypoints = cli.runtime_source_hash(), cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: runtime)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    root = tmp_path / "focused-delta"
    cli.prepare(source, root, "20b", profile)
    schemas = []

    def transport(url, body, timeout):
        schema = body["response_format"]["json_schema"]["name"]
        schemas.append(schema)
        if schema == "FiveOptionsReply":
            raw = revision()
            raw["evidence_ids"] = ["unsupported_citation"]
            return response(
                {
                    "options": [
                        {
                            "option_id": f"candidate_{index}",
                            "affected_target": "Gp",
                            "premise_changed": "Restore nonlinear coupling.",
                            "patch": raw,
                        }
                        for index in range(5)
                    ]
                }
            )
        assert schema == "BranchRepairReply"
        packet = json.loads(body["messages"][1]["content"])
        ref = packet["repair_feedback"]["branch_reference"]
        assert packet["repair_feedback"]["scientific_assessment"] == (
            "trial_not_scientifically_assessed"
        )
        return response(
            {
                "mode": "delta",
                **ref,
                "patch": {"evidence_ids": revision()["evidence_ids"]},
                "drop_actions": [],
            }
        )

    result = cli.run(
        root, "http://127.0.0.1", transport=transport, token_transport=tokenizer
    )
    assert schemas == ["FiveOptionsReply", "BranchRepairReply"]
    if profile == "action-feedback":
        assert result["action_feedback_policy"] == "typed-branch-action-feedback-1"
    else:
        assert "action_feedback_policy" not in result
    assert result["smoke_complete"]
    assert result["cost"]["physical_requests"] == 2
    assert result["smoke_eligible_episodes"] == 1
    candidate = json.loads(next((root / "results").glob("*/result.json")).read_text())[
        "candidates"
    ][-1]
    assert candidate["submitted_reply"]["mode"] == "delta"
    assert candidate["composed_patch"]["edits"] == revision()["edits"]
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == result
    )


@pytest.mark.parametrize("profile", ["focused-retest", "action-feedback"])
def test_focused_full_can_start_without_smoke_but_is_only_three_episodes(
    structural_source,  # noqa: F811
    tmp_path,
    monkeypatch,
    profile,
):
    source, _, _ = structural_source
    runtime, entrypoints = cli.runtime_source_hash(), cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: runtime)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    root = tmp_path / "direct-focused"
    cli.prepare(source, root, "20b", profile)
    result = cli.run(
        root,
        "http://127.0.0.1",
        phase="full",
        transport=respond,
        token_transport=tokenizer,
    )
    assert result["status"] == "complete"
    assert len(result["rows"]) == 3
    assert result["cost"]["physical_requests"] == 3
    assert (
        cli.run(
            root,
            "http://127.0.0.1",
            phase="full",
            transport=forbidden,
            token_transport=forbidden,
        )
        == result
    )


def test_action_feedback_roster_contract_and_only_20b(
    structural_source,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    source, _, _ = structural_source
    root = tmp_path / "action"
    with pytest.raises(ValueError, match="20B-only"):
        cli.prepare(source, root, "120b", "action-feedback")
    assert not root.exists()
    selected = cli.base.sources.load(source)
    prior = cli._plan(selected, "20b", "focused-retest")
    changed = cli._plan(selected, "20b", "action-feedback")
    for key in ("settings", "system_sha256", "schema_sha256", "gate_policy"):
        assert changed[key] == prior[key]
    assert [(c["parent_id"], c["seed"]) for c in changed["cases"]] == [
        (c["parent_id"], c["seed"]) for c in prior["cases"]
    ]
    assert {c["namespace"] for c in changed["cases"]}.isdisjoint(
        {c["namespace"] for c in prior["cases"]}
    )
    cli.prepare(source, root, "20b", "action-feedback")
    before = (root / "plan.json").read_bytes()
    monkeypatch.setitem(cli.PROFILES["action-feedback"], "action_feedback", False)
    with pytest.raises(ValueError, match="differ"):
        cli.verify(root)
    assert (root / "plan.json").read_bytes() == before
