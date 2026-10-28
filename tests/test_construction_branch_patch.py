"""Explicit pending-patch edits preserve identity without scientific inference."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_branch_patch as branch


def candidate(**changes):
    patch = {
        "decision": "revise",
        "hypothesis": "Repair the branch.",
        "evidence_ids": ["public_task"],
        "expected_observable_effect": "A different dependence.",
        "edits": [
            {
                "component": "U",
                "kind": "algebraic",
                "expression": "a*X",
                "sources": ["X"],
            }
        ],
        "new_parameters": [
            {"name": "a", "role": "coefficient"},
            {"name": "b", "role": "rate"},
        ],
        "new_states": [
            {
                "name": "X",
                "description": "History",
                "rhs": "b*I-X",
                "initial": {"mode": "value", "scope": "trajectory"},
            }
        ],
        **changes,
    }
    return {"id": "call0-option0", "branch_id": "call0-option0", "raw_reply": patch}


def delta(base, patch=None, **changes):
    return {
        "mode": "delta",
        **branch.reference(base),
        "patch": None if changes.get("mode") == "withdraw" else patch or {},
        **changes,
    }


def test_delta_preserves_exact_omitted_parameters_states_and_initializer():
    base = candidate()
    before = deepcopy(base)
    raw = delta(
        base,
        {
            "edits": [
                {
                    "component": "U",
                    "kind": "algebraic",
                    "expression": "a*I*X",
                    "sources": [],
                }
            ]
        },
    )
    composed, provenance = branch.compose(raw, base)
    assert composed["new_parameters"] == before["raw_reply"]["new_parameters"]
    assert composed["new_states"] == before["raw_reply"]["new_states"]
    assert composed["edits"][0]["expression"] == "a*I*X"
    assert len(provenance["retained_actions"]) == 3
    assert not provenance["model_adopted"] and provenance["parent_unchanged"]
    assert base == before


def test_empty_lists_do_not_clear_but_explicit_drop_removes_intention():
    base = candidate()
    kept, _ = branch.compose(delta(base, {"new_parameters": []}), base)
    assert kept["new_parameters"] == base["raw_reply"]["new_parameters"]
    removed, provenance = branch.compose(
        delta(base, drop_actions=[{"field": "new_parameters", "key": "a"}]), base
    )
    assert removed["new_parameters"] == [{"name": "b", "role": "rate"}]
    assert removed["edits"] == base["raw_reply"]["edits"]  # no inferred cleanup
    assert provenance["dropped_actions"] == [{"field": "new_parameters", "key": "a"}]


def test_whole_replacement_and_withdrawal_are_explicit():
    base = candidate()
    replaced, _ = branch.compose(
        delta(base, {"decision": "retain"}, mode="replace"), base
    )
    assert replaced == {"decision": "retain"}  # full validation is the caller's job
    withdrawn, provenance = branch.compose(
        delta(base, mode="withdraw", patch=None), base
    )
    assert withdrawn is None and provenance["mode"] == "withdraw"


@pytest.mark.parametrize(
    "field,value",
    [("base_candidate_id", "other-option"), ("base_patch_sha256", "0" * 64)],
)
def test_wrong_branch_or_stale_reference_rejected(field, value):
    base = candidate()
    with pytest.raises(ValueError, match="branch reference differs"):
        branch.compose(delta(base, **{field: value}), base)


def test_duplicate_conflicting_upserts_rejected_without_partial_composition():
    base = candidate()
    before = deepcopy(base)
    raw = delta(
        base,
        {
            "new_parameters": [
                {"name": "a", "role": "rate"},
                {"name": "a", "role": "coefficient"},
            ]
        },
    )
    with pytest.raises(ValueError, match="ambiguous repeated"):
        branch.compose(raw, base)
    assert base == before


def test_invalid_stored_duplicate_can_only_be_replaced_or_explicitly_dropped():
    base = candidate(
        new_parameters=[
            {"name": "a", "role": "rate"},
            {"name": "a", "role": "coefficient"},
        ]
    )
    with pytest.raises(ValueError, match="ambiguous repeated"):
        branch.compose(delta(base), base)
    fixed, _ = branch.compose(
        delta(base, drop_actions=[{"field": "new_parameters", "key": "a"}]), base
    )
    assert fixed["new_parameters"] == []


@pytest.mark.parametrize(
    "drops",
    [
        [{"field": "new_parameters", "key": "absent"}],
        [{"field": "new_parameters", "key": "a"}] * 2,
    ],
)
def test_drop_requires_unambiguous_existing_intention(drops):
    base = candidate()
    with pytest.raises(ValueError):
        branch.compose(delta(base, drop_actions=drops), base)


def test_upsert_and_drop_same_identity_is_rejected():
    base = candidate()
    with pytest.raises(ValueError, match="both drop and replace"):
        branch.compose(
            delta(
                base,
                {"new_parameters": [{"name": "a", "role": "rate"}]},
                drop_actions=[{"field": "new_parameters", "key": "a"}],
            ),
            base,
        )


def test_size_bounds_apply_to_composed_union():
    base = candidate(
        new_parameters=[{"name": f"p{i}", "role": "coefficient"} for i in range(64)]
    )
    with pytest.raises(ValueError, match="bounded list limit"):
        branch.compose(
            delta(base, {"new_parameters": [{"name": "extra", "role": "coefficient"}]}),
            base,
        )


def test_initializer_replacement_is_whole_entry_not_recursive_guess():
    base = candidate(
        initialization_edits=[
            {"state": "old", "initial": {"mode": "value", "scope": "shared_training"}}
        ]
    )
    new = {
        "state": "old",
        "initial": {
            "mode": "map",
            "expression": "y0",
            "public_initials": [{"alias": "y0", "channel": "I"}],
        },
    }
    result, _ = branch.compose(delta(base, {"initialization_edits": [new]}), base)
    assert result["initialization_edits"] == [new]
    assert "scope" not in result["initialization_edits"][0]["initial"]


def test_literal_history_preserves_rhs_and_initial_map():
    base = candidate()
    literal = branch.literal_changes(base)
    assert literal["edits"][0]["expression"] == "a*X"
    assert literal["new_states"][0]["rhs"] == "b*I-X"
    assert literal["new_states"][0]["initial"]["scope"] == "trajectory"


@pytest.mark.parametrize("mode", ["delta", "replace"])
def test_opt_in_exact_repeated_actions_normalize_with_provenance(mode):
    base = candidate()
    row = {"name": "c", "role": "rate"}
    raw = delta(base, {"new_parameters": [row, deepcopy(row)]}, mode=mode)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match="ambiguous repeated"):
        branch.compose(raw, base)
    result, proof = branch.compose(raw, base, repair_contract=True)
    assert result["new_parameters"].count(row) == 1
    record = proof["identical_action_normalization"]
    assert record["policy"] == branch.IDENTICAL_ACTION_POLICY
    assert record["removed_entries"] == [
        {
            "location": "submitted_patch",
            "field": "new_parameters",
            "key": "c",
            "retained_index": 0,
            "removed_index": 1,
            "entry_sha256": branch.content_hash(row),
        }
    ]
    assert record["submitted_reply_sha256"] == branch.content_hash(raw)
    assert raw == before


def test_opt_in_normalizes_identical_stored_entries_without_mutating_saved_intention():
    row = {"name": "a", "role": "rate"}
    base = candidate(new_parameters=[row, deepcopy(row)])
    before = deepcopy(base)
    result, proof = branch.compose(delta(base), base, repair_contract=True)
    assert result["new_parameters"] == [row]
    assert proof["identical_action_normalization"]["removed_entries"][0][
        "location"
    ] == ("stored_patch_after_explicit_drops")
    assert base == before


def test_new_contract_does_not_merge_records_equal_only_after_schema_defaults():
    base = candidate()
    row = {"component": "U", "kind": "algebraic", "expression": "a*I"}
    raw = delta(base, {"edits": [row, {**row, "sources": []}]})
    with pytest.raises(ValueError, match="ambiguous repeated"):
        branch.compose(raw, base, repair_contract=True)


def test_opt_in_identical_duplicates_still_obey_original_list_bounds():
    base = candidate()
    row = {"name": "c", "role": "rate"}
    with pytest.raises(ValueError, match="excessive"):
        branch.compose(
            delta(base, {"new_parameters": [row] * 65}), base, repair_contract=True
        )


@pytest.mark.parametrize("stored", [False, True])
def test_opt_in_conflicting_duplicates_are_still_rejected(stored):
    rows = [{"name": "a", "role": "rate"}, {"name": "a", "role": "coefficient"}]
    base = candidate(**({"new_parameters": rows} if stored else {}))
    raw = delta(base, {} if stored else {"new_parameters": rows})
    with pytest.raises(ValueError, match="ambiguous repeated"):
        branch.compose(raw, base, repair_contract=True)


def test_opt_in_normalization_never_bypasses_branch_hash():
    base = candidate()
    row = {"name": "c", "role": "rate"}
    with pytest.raises(ValueError, match="branch reference differs"):
        branch.compose(
            delta(base, {"new_parameters": [row, row]}, base_patch_sha256="0" * 64),
            base,
            repair_contract=True,
        )
