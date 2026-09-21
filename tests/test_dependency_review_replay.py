"""Cold and persisted process reviews must produce identical new edit histories."""

import json
from copy import deepcopy

import pytest

from autoformalism.rebuttal.process_assembly_audit import _contexts
from autoformalism.search import function_dependencies as dep
from autoformalism.staged_topology import content_hash
from scripts.smoke_function_dependencies import fixture


def linked_fixture():
    """Reproduce the shared objects created by signed.admit and topology assembly."""
    brief, context, source = fixture()
    bindings = source["shared_process_contract"]["bindings"]
    source["process_review"] = {
        "bindings": list(bindings),
        "suggestions": [b["proposal"] for b in bindings],
    }
    return brief, context, source


def history(source, brief, context, *, legacy=False):
    current, events = source, []
    for identifier, expression in (
        ("term_0_0", "max(0,x-crest)"),
        ("term_1_0", "u/area"),
    ):
        current, event = dep.prepare(
            brief,
            context,
            current,
            identifier,
            dep.DependencyFunctionReply(
                expression=expression, parameters=[], revise_dependencies=True
            ),
            _legacy_review_aliases=legacy,
        )
        events.append(event)
    return {
        "dependency_policy": dep.POLICY,
        "dependency_revisions": events,
        "source_topology_result_sha256": content_hash(source),
        "effective_source": current,
        "batch_term_audits": [
            {"interaction_id": f"term_{i}_{j}"}
            for i, equation in enumerate(source["equations"])
            for j, _ in enumerate(equation["terms"])
        ],
    }


def test_new_edits_are_identical_before_and_after_json_and_leave_review_intact():
    brief, context, source = linked_fixture()
    original = deepcopy(source)
    cold = history(source, brief, context)
    warm = history(json.loads(json.dumps(source)), brief, context)
    assert cold == warm
    assert cold["effective_source"]["process_review"] == source["process_review"]
    assert source == original
    assert cold["effective_source"]["shared_process_contract"] != source[
        "shared_process_contract"
    ]
    diagnostics = {}
    dep.replay(brief, context, source, cold, diagnostics=diagnostics)
    assert diagnostics["mode"] == "isolated_review"


def test_historical_cold_aliases_replayed_only_with_exact_hashes_and_final_state():
    brief, context, source = linked_fixture()
    old = history(source, brief, context, legacy=True)
    stored_source, stored = json.loads(json.dumps([source, old]))
    before = deepcopy([stored_source, stored])
    # Reproduce the reported failure: only the post-edit hash differs.
    ordinary = history(stored_source, brief, context)
    event = old["dependency_revisions"][0]
    expected = ordinary["dependency_revisions"][0]
    assert [k for k in event if event[k] != expected[k]] == ["after_sha256"]
    diagnostics = {}
    final, states = dep.replay(
        brief, context, stored_source, stored, diagnostics=diagnostics
    )
    assert final == stored["effective_source"]
    assert diagnostics == {
        "mode": "historical_shared_review_aliases",
        "exact_ledger_and_final_state_verified": True,
        "dependency_edits": 2,
    }
    assert content_hash(states["term_1_0"]) == event["after_sha256"]
    contexts = _contexts(brief, context, stored_source, stored)
    assert contexts["term_1_0"] == states["term_1_0"]
    assert contexts["term_1_1"] == final
    assert before == [stored_source, stored]


@pytest.mark.parametrize("damage", ["hash", "reply", "sources", "final", "layout"])
def test_compatibility_does_not_bypass_integrity(damage):
    brief, context, source = linked_fixture()
    old = history(source, brief, context, legacy=True)
    source, old = json.loads(json.dumps([source, old]))
    if damage == "hash":
        old["dependency_revisions"][0]["after_sha256"] = "0" * 64
    elif damage == "reply":
        old["dependency_revisions"][0]["reply"]["expression"] = "x"
    elif damage == "sources":
        old["dependency_revisions"][0]["after_sources"] = ["x"]
    elif damage == "final":
        suggestion = old["effective_source"]["process_review"]["suggestions"][0]
        suggestion["depends_on"] = ["x"]
    else:
        source["process_review"]["suggestions"][0]["depends_on"] = ["x"]
        old["source_topology_result_sha256"] = content_hash(source)
    with pytest.raises(ValueError):
        dep.replay(brief, context, source, old)


def test_historical_warm_run_does_not_need_compatibility():
    brief, context, source = linked_fixture()
    source = json.loads(json.dumps(source))
    old = history(source, brief, context, legacy=True)
    diagnostics = {}
    dep.replay(brief, context, source, old, diagnostics=diagnostics)
    assert diagnostics["mode"] == "isolated_review"
