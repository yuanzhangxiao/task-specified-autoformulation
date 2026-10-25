"""Matched cross-stage revision arms preserve evidence, budget and replay gates."""

import json
from copy import deepcopy

import pytest

from autoformalism.llm.staged_topology import DeferredCall
from autoformalism.rebuttal.prefit_replay import sealed_read
from autoformalism.research.construction_revision_edits import RevisionValidationError
from autoformalism.staged_topology import content_hash
from scripts import structural_revision_campaign as cli
from tests.test_construction_feedback_revision import (  # noqa: F401
    prepared,
    prepared_v2,
    prepared_v3,
)
from tests.test_construction_initialization_graph_cli import reply as response
from tests.test_construction_initialization_graph_cli import tokenizer


def revision(**changes):
    """An explicit nonlinear repair, optionally adding a complete latent state."""
    return {
        "decision": "revise",
        "hypothesis": "Coupled glucose dependence may explain different responses.",
        "evidence_ids": ["public_task"],
        "expected_observable_effect": "Allow glucose-dependent insulin effects.",
        "edits": [
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["mem", "Gp"],
                "expression": "a*mem*Gp",
            }
        ],
        **changes,
    }


@pytest.fixture
def structural_source(prepared_v3, tmp_path, monkeypatch):  # noqa: F811
    """Use real static/evidence gates with a fixed executable producer identity."""
    prior, _ = prepared_v3
    inputs = sealed_read(prior / "inputs.json")
    inputs.pop("artifact_sha256")
    evidence = sealed_read(prior / "consistency_evidence.json")
    selected = {
        "inputs": inputs,
        "consistency_source": evidence,
        "parents": dict(list(inputs["parents"].items())[:6]),
        "selection": [],
        "provenance": {"synthetic_producer": True},
    }
    source = tmp_path / "source-inspection.tar.gz"
    source.write_bytes(b"synthetic authenticated producer fixture")

    def load(path):
        if path.read_bytes() != b"synthetic authenticated producer fixture":
            raise ValueError("synthetic source archive digest differs")
        return deepcopy(selected)

    monkeypatch.setattr(cli.sources, "load", load)
    # Version identities are deterministic fixture inputs, so concurrent edits
    # in a shared checkout do not race these orchestration tests. The drift test
    # below changes each identity independently and requires fail-closed replay.
    executable = cli.runtime_source_hash()
    entrypoints = cli._entrypoints()
    monkeypatch.setattr(cli, "runtime_source_hash", lambda: executable)
    monkeypatch.setattr(cli, "_entrypoints", lambda: deepcopy(entrypoints))
    roots = {size: tmp_path / f"structural-{size}" for size in cli.MODEL_SPECS}
    for size, root in roots.items():
        cli.prepare(source, root, size)
    return source, selected, roots


def parent_from(root):
    selected = sealed_read(root / "selected_sources.json")
    return cli._parent(selected, "parent_0")


def forbidden(*args, **kwargs):
    pytest.fail("no fitting, simulation, repeated live calls or invalid-cache spending")


def test_matched_arms_have_identical_payloads_budgets_and_comparison_identity(
    structural_source,
):
    _, _, roots = structural_source
    plans = {size: cli.verify(root) for size, root in roots.items()}
    assert plans["20b"]["comparison_sha256"] == plans["120b"]["comparison_sha256"]
    assert plans["20b"]["generation_policy"] == plans["120b"]["generation_policy"]
    assert len(plans["20b"]["cases"]) == len(plans["120b"]["cases"]) == 18
    for left, right in zip(plans["20b"]["cases"], plans["120b"]["cases"], strict=True):
        assert {k: v for k, v in left.items() if k != "namespace"} == {
            k: v for k, v in right.items() if k != "namespace"
        }
        assert left["namespace"] != right["namespace"]
    requests = {}
    for size, root in roots.items():
        cli.preview(root)
        requests[size] = json.loads((root / "preview/parent_0.json").read_text())
        assert not (root / "results").exists()
    assert requests["20b"] == requests["120b"]


@pytest.mark.parametrize("model_size", ["20b", "120b"])
def test_run_and_exact_resume_preserve_controls_and_fit_boundary(
    structural_source, model_size, monkeypatch
):
    _, _, roots = structural_source
    root = roots[model_size]
    from autoformalism.fitting import prefix_initialization

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    originals = {p.name: p.read_bytes() for p in (root / "controls").glob("*.json")}
    selected_before = (root / "selected_sources.json").read_bytes()
    calls = []

    def respond(url, body, timeout):
        calls.append(body)
        assert body["model"] == cli.MODEL_SPECS[model_size][0]
        packet = json.loads(body["messages"][1]["content"])
        assert "test" not in packet and "validation" not in packet
        return response(
            revision(
                new_states=[
                    {
                        "name": "new_memory",
                        "description": "Additional latent history",
                        "rhs": "I-new_memory",
                        "sources": ["I", "new_memory"],
                        "initial": {"mode": "value", "scope": "trajectory"},
                    }
                ]
            )
        )

    cli.preview(root)
    preview = json.loads((root / "preview/parent_0.json").read_text())
    summary = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=tokenizer
    )
    assert summary["status_counts"] == {"revised": 18}
    assert summary["cost"]["physical_requests"] == len(calls) == 18
    assert not summary["test_data_opened"] and not summary["automatic_followup"]
    assert summary["optimizer_calls"] == summary["solver_rollouts"] == 0
    assert calls[0]["messages"] == [
        {"role": "system", "content": preview["system"]},
        {"role": "user", "content": preview["user"]},
    ]
    handoff = sealed_read(next((root / "results").glob("*/fit_handoff.json")))
    assert (
        handoff["prefix_fit_problem"]["graph"]["rules"]["new_memory"]["initial"][
            "scope"
        ]
        == "trajectory"
    )
    assert not handoff["fitting_launched"] and not handoff["validation_used"]
    assert handoff["training_consistency"]["status"] != "conflict"
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == summary
    )
    assert originals == {
        p.name: p.read_bytes() for p in (root / "controls").glob("*.json")
    }
    assert selected_before == (root / "selected_sources.json").read_bytes()
    assert cli.inspect(root)["status"] == "complete"


def test_qualified_conflicted_parent_cannot_escape_by_retaining(structural_source):
    parent = parent_from(structural_source[2]["20b"])
    parent["problem"]["base_candidate"]["state_equations"][0]["rhs"] = "1"
    with pytest.raises(
        RevisionValidationError, match="QUALIFIED_MODEL_DATA_CONFLICT"
    ) as error:
        cli._apply(parent, revision(decision="retain", edits=[]))
    assert error.value.details["assessment"]["status"] == "conflict"


def test_unrelated_new_latent_state_cannot_erase_qualified_conflict(structural_source):
    parent = parent_from(structural_source[2]["20b"])
    parent["problem"]["base_candidate"]["state_equations"][0]["rhs"] = "1"
    with pytest.raises(RevisionValidationError, match="QUALIFIED_MODEL_DATA_CONFLICT"):
        cli._apply(
            parent,
            revision(
                edits=[],
                new_states=[
                    {
                        "name": "unrelated",
                        "description": "A disconnected latent root",
                        "rhs": "-unrelated",
                        "initial": {"mode": "value", "scope": "trajectory"},
                    }
                ],
            ),
        )


def test_requires_wider_revision_terminates_without_export(structural_source):
    root = structural_source[2]["20b"]
    count = 0

    def respond(*args):
        nonlocal count
        count += 1
        return response(revision(decision="requires_wider_revision", edits=[]))

    summary = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=tokenizer
    )
    assert summary["status_counts"] == {"requires_wider_revision": 18}
    assert count == 18  # One call per episode, not the three-call allowance.
    assert not list((root / "results").glob("*/fit_handoff.json"))
    assert not any(r["fitting_exported"] for r in summary["rows"])
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == summary
    )


def test_preflight_block_is_recorded_and_remaining_cases_continue(structural_source):
    root = structural_source[2]["20b"]
    token_calls = 0
    generation_calls = 0

    def blocked_first(*args):
        nonlocal token_calls
        token_calls += 1
        if token_calls == 1:
            raise ValueError("synthetic first-case tokenizer refusal")
        return tokenizer(*args)

    def respond(*args):
        nonlocal generation_calls
        generation_calls += 1
        return response(revision())

    summary = cli.run(
        root, "http://127.0.0.1", transport=respond, token_transport=blocked_first
    )
    assert summary["status"] == "incomplete"
    assert summary["status_counts"] == {"preflight_blocked": 1, "revised": 17}
    assert generation_calls == summary["cost"]["physical_requests"] == 17
    blocked = next((root / "results").glob("*/preflight_blocked.json"))
    marker = sealed_read(blocked)
    assert marker["problem"] is None
    assert "tokenizer refusal" in marker["error"]
    assert not (blocked.parent / "fit_handoff.json").exists()
    assert len(list((root / "results").glob("*/fit_handoff.json"))) == 17
    assert (
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )
        == summary
    )


def test_identical_qualified_checks_are_memoized_without_shared_mutable_results(
    structural_source, monkeypatch
):
    parent = parent_from(structural_source[2]["20b"])
    from autoformalism.schemas.prefix_initialization import PrefixFitProblem

    problem = PrefixFitProblem.model_validate(parent["problem"])
    original = cli.old._consistency
    calls = 0

    def counting(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(cli.old, "_consistency", counting)
    first = cli._assessment(parent, problem)
    first["status"] = "deliberately_mutated_return_value"
    assert cli._assessment(parent, problem)["status"] != first["status"]
    assert calls == 1
    changed = problem.model_dump(mode="json")
    changed["base_candidate"]["state_equations"][0]["rhs"] = "1"
    assert (
        cli._assessment(parent, PrefixFitProblem.model_validate(changed))["status"]
        == "conflict"
    )
    assert calls == 2


def test_property_citations_require_displayed_candidate_evidence(
    structural_source, monkeypatch
):
    parent = parent_from(structural_source[2]["20b"])
    monkeypatch.setattr(
        cli.feedback, "consistency_view",
        lambda *args: {"conflicts": [{"property": "displayed_property"}]},
    )
    raw = revision(evidence_ids=["displayed_property", "invented_property"])
    normalized, changes = cli._normalize_displayed_citations(parent, raw)
    assert normalized["evidence_ids"] == ["training_consistency", "invented_property"]
    assert changes == [{"from": "displayed_property", "to": "training_consistency"}]
    assert raw["evidence_ids"] == ["displayed_property", "invented_property"]
    with pytest.raises(RevisionValidationError, match="UNKNOWN_EVIDENCE"):
        cli._apply(parent, raw)


def test_v2_explicitly_enables_symbolic_gate_and_owns_new_state_fields(
    structural_source,
):
    root = structural_source[2]["20b"]
    plan = cli.verify(root)
    assert plan["protocol"] == "construction-structural-revision-2"
    assert plan["gate_policy"]["symbolic_initials"] == cli.SYMBOLIC_INITIAL_POLICY
    packet = json.loads(cli.request(parent_from(root)))
    assert "new_states/new_processes" in packet["editing_scope"]["edits"]
    assert "new_states" in packet["editing_scope"]["initialization_edits"]


def test_repeated_scientific_patch_is_detected_despite_prose_changes(structural_source):
    parent = parent_from(structural_source[2]["20b"])
    packets = []

    class Client:
        def call(self, **kwargs):
            packets.append(json.loads(kwargs["user"]))
            raw = revision(
                hypothesis=f"Changed explanation {len(packets)}",
                edits=[
                    {
                        "component": "I",
                        "kind": "dynamic",
                        "sources": [],
                        "expression": "1",
                    }
                ],
            )
            return {
                "status": "responded",
                "request_hash": str(len(packets)),
                "raw_response": response(raw),
            }

    result = cli._episode(parent, Client(), cli.settings_for("20b"))
    assert result["status"] == "rejected"
    repair = packets[2]["repair_feedback"]
    assert repair["repeated_equation_patch"]
    assert repair["repeated_patch_attempts"] == [0]
    assert packets[0]["current_model"] == packets[2]["current_model"]
    assert "unchanged" in repair["status"]


def test_interrupted_episode_resumes_with_original_charges(structural_source):
    root = structural_source[2]["20b"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise DeferredCall("synthetic interruption")
        return response(
            revision(
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

    partial = cli.run(
        root, "http://127.0.0.1", transport=interrupted, token_transport=tokenizer
    )
    assert partial["status_counts"] == {"pending": 18}
    resumed = cli.run(
        root,
        "http://127.0.0.1",
        transport=lambda *args: response(revision()),
        token_transport=tokenizer,
    )
    assert resumed["status_counts"] == {"revised": 18}
    assert resumed["cost"]["physical_requests"] == 20
    assert resumed["cost"]["requests_with_unknown_usage"] == 1


@pytest.mark.parametrize("mutation", ["settings", "protocol", "extra"])
def test_pending_cache_corruption_fails_before_new_requests(
    structural_source, mutation
):
    root = structural_source[2]["20b"]
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise DeferredCall("synthetic interruption")
        return response(
            revision(
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
    path = next((root / "results").glob("*/calls/*.json"))
    value = json.loads(path.read_text())
    if mutation == "settings":
        value["request"]["settings"]["temperature"] = 0.7
    else:
        value["request"][mutation] = "tampered"
        value["request_hash"] = content_hash(value["request"])
        destination = path.with_name(value["request_hash"] + ".json")
        path.rename(destination)
        path = destination
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )


@pytest.mark.parametrize("mutation", ["source", "plan", "selected", "control"])
def test_source_plan_and_control_drift_fail_closed(structural_source, mutation):
    root = structural_source[2]["20b"]
    if mutation == "source":
        (root / "saved-source.tar.gz").write_bytes(b"changed source")
    else:
        path = {
            "plan": root / "plan.json",
            "selected": root / "selected_sources.json",
            "control": root / "controls/parent_0.json",
        }[mutation]
        value = json.loads(path.read_text())
        value["tampered"] = True
        value["artifact_sha256"] = content_hash(
            {k: v for k, v in value.items() if k != "artifact_sha256"}
        )
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )


@pytest.mark.parametrize("identity", ["runtime", "entrypoints"])
def test_executable_identity_drift_prevents_replay(
    structural_source, monkeypatch, identity
):
    root = structural_source[2]["20b"]
    if identity == "runtime":
        monkeypatch.setattr(cli, "runtime_source_hash", lambda: "changed-source")
    else:
        monkeypatch.setattr(
            cli, "_entrypoints", lambda: {"unexpected": "changed-script"}
        )
    with pytest.raises(ValueError, match="executable source"):
        cli.run(
            root, "http://127.0.0.1", transport=forbidden, token_transport=forbidden
        )


def test_held_out_payload_cannot_enter_source_projection(
    structural_source, tmp_path, monkeypatch
):
    source, selected, _ = structural_source
    changed = deepcopy(selected)
    changed["inputs"]["parents"]["parent_0"]["test_metrics"] = {"nmse": 0.1}
    monkeypatch.setattr(cli.sources, "load", lambda path: deepcopy(changed))
    with pytest.raises(ValueError, match="held-out"):
        cli.prepare(source, tmp_path / "held-out-root", "20b")


def test_live_calls_require_local_verified_server(structural_source):
    root = structural_source[2]["20b"]
    with pytest.raises(ValueError, match="serving identity"):
        cli.run(root, "http://127.0.0.1")
    with pytest.raises(ValueError, match="loopback"):
        cli.run(root, "https://example.com", transport=forbidden)
