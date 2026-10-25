"""Repair families ignore private names, never changed dynamical assumptions."""

from copy import deepcopy

import pytest

from autoformalism.research import construction_repair_identity as identity
from autoformalism.research.construction_structural_revision import (
    apply_structural_revision,
)
from tests.test_construction_revision_edits import problem
from tests.test_construction_structural_revision import reply


def child(
    name="X", parameter="tau", *, nonlinear=False, scope="trajectory", second=False
):
    """One or two filters contribute to an existing observed state."""
    source = problem()
    names = [name, f"{name}_other"] if second else [name]
    parameters = [parameter, f"{parameter}_other"] if second else [parameter]
    payload = reply(
        edits=[
            {
                "component": "I",
                "kind": "dynamic",
                "sources": [],
                "expression": "-q+u+" + "+".join(names),
            }
        ],
        new_states=[
            {
                "name": state,
                "description": "A private latent filter",
                "rhs": f"-{state}/{tau}+u" + ("**2" if nonlinear else ""),
                "initial": {"mode": "value", "scope": scope},
            }
            for state, tau in zip(names, parameters, strict=True)
        ],
        new_parameters=[{"name": tau, "role": "time_constant"} for tau in parameters],
    )
    return apply_structural_revision(source, payload, evidence_ids=["training:I"])[
        "problem"
    ]


def signature(candidate):
    result = identity.family_identity(problem(), candidate)
    assert result["status"] == "identified", result
    return result["sha256"]


def test_renamed_filter_and_fitted_time_constant_are_one_family():
    first, renamed = child(), child("Z", "decay_time")
    assert signature(first) == signature(renamed)
    assert first != renamed


def test_new_process_names_and_list_order_do_not_change_family():
    source = problem()

    def make(process):
        return apply_structural_revision(
            source,
            reply(
                edits=[
                    {
                        "component": "I",
                        "kind": "dynamic",
                        "sources": [],
                        "expression": f"-q+u+{process}",
                    }
                ],
                new_processes=[
                    {
                        "name": process,
                        "description": "Added effect",
                        "expression": "a*Gp",
                    }
                ],
            ),
            evidence_ids=["training:I"],
        )["problem"]

    first, renamed = make("law"), make("other_law").model_dump(mode="json")
    for field in (
        "states",
        "state_equations",
        "processes",
        "parameters",
        "initial_conditions",
        "observation_mappings",
    ):
        renamed["base_candidate"][field].reverse()
    assert signature(first) == signature(renamed)


@pytest.mark.parametrize(
    "changes", [{"nonlinear": True}, {"scope": "shared_training"}, {"second": True}]
)
def test_same_topology_or_additional_independent_state_is_not_equivalence(changes):
    assert signature(child()) != signature(child(**changes))


def test_guess_prose_and_candidate_ids_do_not_define_a_family():
    first = child()
    altered = first.model_dump(mode="json")
    altered["parameter_guesses"]["tau"] = 900.0
    altered["graph"]["rules"]["X"]["initial"]["guess"] = 500.0
    altered["base_candidate"]["candidate_id"] = "different_receipt"
    altered["base_candidate"]["change_summary"] = "Different explanation"
    altered["base_candidate"]["states"][-1]["description"] = "New explanation"
    assert signature(first) == signature(altered)


def test_dynamics_parameter_sharing_is_preserved():
    separate = child(second=True)
    shared = separate.model_dump(mode="json")
    shared["base_candidate"]["state_equations"][-1]["rhs"] = "-X_other/tau+u"
    shared["base_candidate"]["parameters"] = [
        p for p in shared["base_candidate"]["parameters"] if p["name"] != "tau_other"
    ]
    shared["parameter_guesses"].pop("tau_other")
    assert signature(separate) != signature(shared)


def test_domain_and_bounds_are_preserved():
    first = child()
    bounded = first.model_dump(mode="json")
    param = next(
        p for p in bounded["base_candidate"]["parameters"] if p["name"] == "tau"
    )
    param["bounds"] = {"lower": 0.1, "upper": 10.0}
    assert signature(first) != signature(bounded)
    role = first.model_dump(mode="json")
    param = next(p for p in role["base_candidate"]["parameters"] if p["name"] == "tau")
    param.update(role="coefficient", domain="real")
    assert signature(first) != signature(role)


def test_existing_parent_names_remain_fixed():
    first = problem().model_dump(mode="json")
    renamed = deepcopy(first)
    for equation in renamed["base_candidate"]["state_equations"]:
        if equation["state"] == "mem":
            equation["state"] = "new_mem"
        equation["rhs"] = equation["rhs"].replace("mem", "new_mem")
    for state in renamed["base_candidate"]["states"]:
        if state["name"] == "mem":
            state["name"] = "new_mem"
    for initial in renamed["base_candidate"]["initial_conditions"]:
        if initial["state"] == "mem":
            initial["state"] = "new_mem"
    renamed["base_candidate"]["processes"][0]["expression"] = "a*new_mem"
    renamed["graph"]["rules"]["new_mem"] = renamed["graph"]["rules"].pop("mem")
    renamed["graph"]["rules"]["dep"]["initial"]["state_initials"]["first"] = "new_mem"
    assert signature(first) != signature(renamed)


def map_child(state="X", local="gain", alias="first", *, two_parameters=False):
    value = child(state).model_dump(mode="json")
    value["graph"]["rules"][state]["initial"] = {
        "mode": "map",
        "expression": f"{local}*{alias}"
        + ("+other*public" if two_parameters else "+public"),
        "parameters": [{"name": local, "role": "coefficient", "guess": 1.0}]
        + (
            [{"name": "other", "role": "coefficient", "guess": 1.0}]
            if two_parameters
            else []
        ),
        "state_initials": {alias: "mem"},
        "public_initials": {"public": "aux"},
    }
    return value


def test_initial_map_aliases_local_coefficients_and_owner_rename_consistently():
    assert signature(map_child()) == signature(map_child("Z", "alpha", "prior"))
    assert signature(map_child()) != signature(map_child(two_parameters=True))


def test_initial_map_dependencies_and_known_values_are_significant():
    original = map_child()
    changed = deepcopy(original)
    changed["graph"]["rules"]["X"]["initial"]["state_initials"]["first"] = "I"
    assert signature(original) != signature(changed)
    for value in (0.0, 1.0):
        changed["graph"]["rules"]["X"]["initial"] = {
            "mode": "known",
            "value": value,
            "justification": "Established externally",
        }
        if value == 0:
            zero = signature(changed)
        else:
            assert signature(changed) != zero


def test_unknown_or_unsafe_expression_has_no_identity():
    candidate = child().model_dump(mode="json")
    candidate["base_candidate"]["state_equations"][-1]["rhs"] = (
        "__import__('os').system('x')"
    )
    value = identity.family_identity(problem(), candidate)
    assert value["status"] == "deferred" and value["sha256"] is None


def test_private_symbol_limit_defers_without_fabricating_equality(monkeypatch):
    monkeypatch.setitem(identity.LIMITS, "private_symbols", 1)
    result = identity.family_identity(problem(), child())
    assert result["status"] == "deferred" and result["sha256"] is None
    assert "bound" in result["reason"]


def test_symmetry_limit_and_exact_small_graph_canonicalization(monkeypatch):
    nodes = {f"p{i}": {"type": "parameter", "role": "coefficient"} for i in range(4)}
    canonical, _, permutations = identity._canonical({"fixed": 1}, nodes)
    assert permutations == 24
    renamed = {
        f"renamed{i}": value for i, value in enumerate(reversed(list(nodes.values())))
    }
    assert identity._canonical({"fixed": 1}, renamed)[0] == canonical
    monkeypatch.setitem(identity.LIMITS, "permutations", 23)
    with pytest.raises(ValueError, match="symmetry"):
        identity._canonical({"fixed": 1}, nodes)


def test_diagnostic_identity_retains_witnesses_but_ignores_known_provenance():
    first = {
        "code": "QUALIFIED_MODEL_DATA_CONFLICT",
        "details": {
            "child_problem_sha256": "old",
            "target": "I",
            "observed": [2, 4],
            "expression": "X/tau",
        },
    }
    renamed = {
        "code": "QUALIFIED_MODEL_DATA_CONFLICT",
        "details": {
            "child_problem_sha256": "new",
            "target": "I",
            "observed": [2, 4],
            "expression": "Z/decay",
        },
    }
    left = identity.family_identity(problem(), child())
    right = identity.family_identity(problem(), child("Z", "decay"))
    assert identity.diagnostic_identity(
        first, private_symbol_mapping=left["private_symbol_mapping"]
    ) == identity.diagnostic_identity(
        renamed, private_symbol_mapping=right["private_symbol_mapping"]
    )
    renamed["details"]["observed"] = [2, 5]
    assert identity.diagnostic_identity(
        first, private_symbol_mapping=left["private_symbol_mapping"]
    ) != identity.diagnostic_identity(
        renamed, private_symbol_mapping=right["private_symbol_mapping"]
    )
    assert identity.diagnostic_identity({"code": "A"}) != identity.diagnostic_identity(
        {"code": "B"}
    )


def test_state_count_and_public_forcing_never_disappear_from_identity():
    first = child()
    altered = first.model_dump(mode="json")
    altered["base_candidate"]["state_equations"][-1]["rhs"] = "-X/tau+aux"
    assert signature(first) != signature(altered)
    assert signature(child("Z", "decay", second=True)) != signature(first)


def test_whitespace_parentheses_normalize_but_algebra_is_not_claimed():
    original = child()
    spaced = original.model_dump(mode="json")
    spaced["base_candidate"]["state_equations"][-1]["rhs"] = " -(X) / tau + u "
    # Existing parser strips no leading indent; equivalent legal AST spacing does.
    spaced["base_candidate"]["state_equations"][-1]["rhs"] = spaced["base_candidate"][
        "state_equations"
    ][-1]["rhs"].strip()
    assert signature(original) == signature(spaced)
    spaced["base_candidate"]["state_equations"][-1]["rhs"] = "u-X/tau"
    assert signature(original) != signature(spaced)


def test_diagnostic_symbol_sets_ignore_rename_order_but_witness_vectors_do_not():
    first = {
        "code": "QUALIFIED_MODEL_DATA_CONFLICT",
        "details": {
            "closed_states": ["I", "X", "Z"],
            "initialized_states": ["I", "X", "Z"],
            "model_relations": {"X": "-X/tau", "Z": "X-Z"},
            "trajectories": ["train1", "train2"],
            "observations": [1, 2],
        },
    }
    renamed = {
        "code": "QUALIFIED_MODEL_DATA_CONFLICT",
        "details": {
            "closed_states": ["A", "I", "Y"],
            "initialized_states": ["A", "I", "Y"],
            "model_relations": {"Y": "-Y/time_scale", "A": "Y-A"},
            "trajectories": ["train1", "train2"],
            "observations": [1, 2],
        },
    }
    left = {"X": "$private:0$", "Z": "$private:1$", "tau": "$private:2$"}
    right = {"Y": "$private:0$", "A": "$private:1$", "time_scale": "$private:2$"}
    expected = identity.diagnostic_identity(first, private_symbol_mapping=left)
    assert expected == identity.diagnostic_identity(
        renamed, private_symbol_mapping=right
    )
    renamed["details"]["observations"].reverse()
    assert expected != identity.diagnostic_identity(
        renamed, private_symbol_mapping=right
    )
    renamed["details"]["observations"].reverse()
    renamed["details"]["trajectories"].reverse()
    assert expected != identity.diagnostic_identity(
        renamed, private_symbol_mapping=right
    )
