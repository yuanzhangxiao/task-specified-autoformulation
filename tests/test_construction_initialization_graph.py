"""Topology initial graphs keep runtime provenance and dependency scope explicit."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from autoformalism.expressions import ValidationContext
from autoformalism.fitting.initialization import KnownInitialValue
from autoformalism.fitting.prefix_initialization import PrefixPolicy
from autoformalism.search.construction_initialization_graph import (
    GraphConstructionContract,
    GraphDependencyHandoff,
    GraphMapDependency,
    build_contract,
    dependency_summary,
    initializer_aliases,
    validate_contract,
    validate_handoff,
)
from autoformalism.search.construction_ledger import Draft


def fixture(*, trajectory=True, window=True):
    """Return reviewed identities with shared, known, observed and dependent nodes."""
    context = ValidationContext(
        targets=("signal",),
        auxiliaries=("aux",),
        external_inputs=("u",),
        fixed_covariates=("dose",),
    )
    draft = Draft.model_validate(
        {
            "variables": [
                {
                    "name": name,
                    "definition": "algebraic" if name == "law" else "differential",
                    "scientific_role": f"Reviewed coordinate {name}",
                }
                for name in ("y", "root", "local", "known", "middle", "end", "law")
            ]
        }
    )
    contract = build_contract(
        draft,
        context,
        scientific_context="Measured signal is y, in the same units.",
        permitted_public_initials=("signal", "aux", "u", "dose", "t"),
        observed_initial_channels={"y": "signal"},
        known_initials={
            "known": KnownInitialValue(
                value=0, justification="Prepared empty reservoir"
            )
        },
        trajectory_initialization="observation_prefix" if trajectory else "forbidden",
        prefix_policy=PrefixPolicy(duration=1, future_inputs=("u",))
        if trajectory and window
        else None,
        provenance="Reviewed public identity and preparation protocol, revision 1",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": state,
                    "rule": rule,
                    "justification": f"Initialization hypothesis for {state}",
                }
                for state, rule in {
                    "root": {"mode": "value", "scope": "shared_training"},
                    "local": {
                        "mode": "value",
                        "scope": "trajectory" if trajectory else "shared_training",
                    },
                    "known": {"mode": "known"},
                    "middle": {
                        "mode": "map",
                        "state_initials": ["root", "y", "known"],
                        "public_initials": ["aux"],
                    },
                    "end": {
                        "mode": "map",
                        "state_initials": ["middle", "local"],
                        "public_initials": ["u", "t"],
                    },
                }.items()
            ]
        }
    )
    return draft, context, contract, handoff


def changed_rule(handoff, state, rule):
    """Replace one complete dependency without modifying its retained explanation."""
    raw = handoff.model_dump(mode="json")
    next(record for record in raw["records"] if record["state"] == state)["rule"] = rule
    return GraphDependencyHandoff.model_validate(raw)


def test_complete_graph_and_transitive_scopes_preserve_runtime_provenance():
    draft, context, contract, handoff = fixture()
    before = deepcopy(contract.model_dump(mode="json"))
    assert validate_handoff(handoff, contract, draft=draft, context=context) == handoff
    summary = dependency_summary(handoff, contract)
    assert summary["end"] == {
        "state_initials": ["known", "local", "middle", "root", "y"],
        "public_initials": ["aux", "signal", "t", "u"],
        "shared_training_roots": ["root"],
        "trajectory_roots": ["local"],
        "known_initials": ["known"],
    }
    assert summary["y"]["public_initials"] == ["signal"]
    assert summary["known"]["known_initials"] == ["known"]
    assert contract.known_initials["known"].justification == "Prepared empty reservoir"
    assert contract.model_dump(mode="json") == before
    assert (
        GraphConstructionContract.model_validate_json(contract.model_dump_json())
        == contract
    )


def test_trajectory_capability_can_precede_fitting_window_selection():
    draft, context, contract, handoff = fixture(window=False)
    assert contract.prefix_policy is None
    assert validate_handoff(handoff, contract, draft=draft, context=context) == handoff
    assert dependency_summary(handoff, contract)["end"]["trajectory_roots"] == ["local"]


def test_trajectory_root_and_policy_require_runtime_authorization():
    _, _, contract, handoff = fixture(trajectory=False)
    changed = changed_rule(handoff, "local", {"mode": "value", "scope": "trajectory"})
    with pytest.raises(ValueError, match="observation_prefix authorization"):
        validate_handoff(changed, contract)
    raw = contract.model_dump(mode="json")
    raw["prefix_policy"] = {"duration": 1}
    with pytest.raises(ValueError, match="observation_prefix authorization"):
        GraphConstructionContract.model_validate(raw)


@pytest.mark.parametrize(
    "state,rule,match",
    [
        ("root", {"mode": "known"}, "not established by runtime"),
        ("known", {"mode": "value", "scope": "shared_training"}, "runtime known"),
        ("middle", {"mode": "map", "state_initials": ["law"]}, "not differential"),
        ("middle", {"mode": "map", "state_initials": ["absent"]}, "not differential"),
        ("middle", {"mode": "map", "public_initials": ["root"]}, "unavailable public"),
        (
            "middle",
            {"mode": "map", "public_initials": ["future"]},
            "unavailable public",
        ),
        (
            "middle",
            {"mode": "map", "public_initials": ["trajectory_id"]},
            "unavailable public",
        ),
        ("middle", {"mode": "map", "state_initials": ["middle"]}, "cyclic"),
        ("middle", {"mode": "map", "state_initials": ["end"]}, "cyclic"),
    ],
)
def test_invalid_dependency_graph_is_rejected(state, rule, match):
    _, _, contract, handoff = fixture()
    with pytest.raises(ValueError, match=match):
        validate_handoff(changed_rule(handoff, state, rule), contract)


@pytest.mark.parametrize("state", ["y", "law", "absent"])
def test_observed_algebraic_and_unknown_state_records_are_rejected(state):
    _, _, contract, handoff = fixture()
    raw = handoff.model_dump(mode="json")
    raw["records"].append(
        {
            "state": state,
            "rule": {"mode": "value", "scope": "shared_training"},
            "justification": "Unsupported replacement",
        }
    )
    with pytest.raises(ValueError, match=r"observed state|not a differential state"):
        validate_handoff(GraphDependencyHandoff.model_validate(raw), contract)


def test_missing_and_duplicate_latent_records_are_rejected():
    _, _, contract, handoff = fixture()
    raw = handoff.model_dump(mode="json")
    raw["records"].pop()
    with pytest.raises(ValueError, match="missing latent"):
        validate_handoff(GraphDependencyHandoff.model_validate(raw), contract)
    raw["records"].append(raw["records"][0])
    with pytest.raises(ValueError, match="duplicate initializer state"):
        GraphDependencyHandoff.model_validate(raw)


@pytest.mark.parametrize(
    "rule",
    [
        {"mode": "map"},
        {"mode": "map", "state_initials": ["root", "root"]},
        {"mode": "map", "public_initials": ["signal", "signal"]},
        {"mode": "map", "public_initials": ["signal"], "expression": "a*signal"},
        {"mode": "map", "public_initials": ["signal"], "parameters": []},
        {"mode": "value"},
        {"mode": "value", "scope": "shared_training", "guess": 1},
        {"mode": "value", "scope": "per_sample"},
        {"mode": "known", "value": 0},
        {"mode": "known", "justification": "Proposer claims preparation"},
    ],
)
def test_topology_schema_has_no_formulas_parameters_guesses_or_known_values(rule):
    _, _, _, handoff = fixture()
    with pytest.raises(ValidationError):
        changed_rule(handoff, "middle", rule)


def test_mechanistic_explanations_are_required_and_retained_without_interpretation():
    _, _, _, handoff = fixture()
    raw = handoff.model_dump(mode="json")
    del raw["records"][0]["justification"]
    with pytest.raises(ValidationError):
        GraphDependencyHandoff.model_validate(raw)
    raw["records"][0]["justification"] = "A hypothesis requiring later verification"
    assert GraphDependencyHandoff.model_validate(raw).records[0].justification == (
        "A hypothesis requiring later verification"
    )


def test_alias_translation_changes_only_state_names_not_public_channels():
    _, context, contract, handoff = fixture()
    aliases = {state: f"x_{state}" for state in contract.state_declaration_sha256}
    translated = validate_handoff(
        handoff,
        contract,
        state_names=tuple(aliases.values()),
        context=context.model_copy(update={"fitted_initialization": True}),
        state_aliases=aliases,
    )
    middle = next(record for record in translated.records if record.state == "x_middle")
    assert middle.rule.state_initials == ("x_root", "x_y", "x_known")
    assert middle.rule.public_initials == ("aux",)
    assert middle.justification == "Initialization hypothesis for middle"


@pytest.mark.parametrize(
    "aliases",
    [
        {"root": "y"},
        {"root": "same", "local": "same"},
        {"root": "local", "local": "root"},
        {"unknown": "x"},
        {"root": "root.__class__"},
    ],
)
def test_ambiguous_or_unknown_aliases_fail(aliases):
    _, _, contract, handoff = fixture()
    with pytest.raises(ValueError):
        validate_handoff(handoff, contract, state_aliases=aliases)


def test_deterministic_map_aliases_keep_overlapping_names_in_separate_namespaces():
    rule = GraphMapDependency(
        state_initials=("z", "signal"), public_initials=("u", "signal")
    )
    assert initializer_aliases(rule) == {
        "state_initials": {"state_1": "signal", "state_2": "z"},
        "public_initials": {"public_1": "signal", "public_2": "u"},
    }
    assert initializer_aliases(rule) == initializer_aliases(
        GraphMapDependency(
            state_initials=("signal", "z"), public_initials=("signal", "u")
        )
    )


@pytest.mark.parametrize("change", ["role", "definition", "removed", "added"])
def test_source_declaration_changes_invalidate_reviewed_contract(change):
    draft, _, contract, _ = fixture()
    raw = draft.model_dump(mode="json")
    if change == "role":
        raw["variables"][1]["scientific_role"] = "Another physical coordinate"
    elif change == "definition":
        raw["variables"][1]["definition"] = "algebraic"
    elif change == "removed":
        raw["variables"].pop(1)
    else:
        raw["variables"].append(
            {
                "name": "new_state",
                "definition": "differential",
                "scientific_role": "New",
            }
        )
    with pytest.raises(ValueError, match="stale initializer state"):
        validate_contract(contract, draft=Draft.model_validate(raw))


def test_runtime_context_and_public_text_are_bound():
    _, context, contract, _ = fixture()
    with pytest.raises(ValueError, match="public channel context"):
        validate_contract(
            contract, context=context.model_copy(update={"auxiliaries": ()})
        )
    with pytest.raises(ValueError, match="scientific context"):
        validate_contract(contract, scientific_context="Changed description")
    with pytest.raises(ValueError, match="state inventory"):
        validate_contract(contract, state_names=("y",))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"permitted_public_initials": ("signal", "unknown")}, "unknown public"),
        ({"permitted_public_initials": ("signal", "signal")}, "duplicate permitted"),
        ({"observed_initial_channels": {"absent": "signal"}}, "unavailable states"),
        ({"observed_initial_channels": {"y": "u"}}, "not a measured channel"),
        ({"observed_initial_channels": {"y": "missing"}}, "not permitted"),
        (
            {"prefix_policy": PrefixPolicy(duration=1, future_inputs=("aux",))},
            "not external",
        ),
        (
            {
                "known_initials": {
                    "y": {"mode": "known", "value": 0, "justification": "fixed"}
                }
            },
            "conflict",
        ),
    ],
)
def test_runtime_contract_rejects_unavailable_channels_and_conflicting_fixed_rules(
    change, match
):
    draft, context, contract, _ = fixture()
    raw = contract.model_dump(mode="json") | change
    with pytest.raises(ValueError, match=match):
        validate_contract(
            GraphConstructionContract.model_validate(raw), draft=draft, context=context
        )


def test_explicit_coordinate_contract_never_infers_observed_identity_from_name():
    context = ValidationContext(targets=("signal",))
    contract = build_contract(
        context=context,
        state_names=("signal",),
        permitted_public_initials=("signal",),
        provenance="Explicit coordinates only",
    )
    assert not contract.observed_initial_channels
    with pytest.raises(ValueError, match="missing latent"):
        validate_handoff(GraphDependencyHandoff(), contract)
    with pytest.raises(ValueError, match="duplicate differential"):
        build_contract(
            context=context,
            state_names=("signal", "signal"),
            permitted_public_initials=("signal",),
            provenance="Duplicate",
        )


def test_mutated_nested_schema_content_is_revalidated_at_usage_boundary():
    _, _, contract, handoff = fixture()
    contract.known_initials["ghost"] = KnownInitialValue(
        value=0, justification="tampered"
    )
    with pytest.raises(ValueError, match="unavailable states"):
        validate_handoff(handoff, contract)
