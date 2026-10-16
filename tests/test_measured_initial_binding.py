"""Reviewed measured initials cannot turn into unconstrained fitted boundaries."""

import copy

import pytest

from autoformalism.search.causal_initialization import (
    compile_initialization_result,
    construct_initializers,
)
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.topology_initialization import (
    InitializerDependencyHandoff,
    MeasuredInitialInformationContract,
    derived_evidence,
    inspect,
    rejected_handoff_feedback,
    request,
    validate_information_contract,
    validate_public_bindings,
)
from autoformalism.staged_topology import content_hash
from tests.test_causal_initialization_construction import setup
from tests.test_topology_initialization import choice, fixture

PUBLIC = "Initial quantity b is a direct reading of z in the same units."


def runtime(model, info, *, state="z", source="b"):
    variable = next(row for row in model.variables if row.name == state)
    return MeasuredInitialInformationContract.model_validate(
        {
            **info.model_dump(mode="json"),
            "protocol": "recorded-t0-initial-information-2",
            "measured_initial_bindings": [
                {
                    "state": state,
                    "source": source,
                    "public_context_sha256": content_hash(PUBLIC),
                    "public_excerpt": PUBLIC,
                    "variable_declaration_sha256": content_hash(
                        variable.model_dump(mode="json")
                    ),
                    "provenance": "Explicit synthetic reviewed identity",
                }
            ],
        }
    )


def test_v1_contract_is_byte_identical_and_has_no_inferred_binding():
    model, evidence, info = fixture()
    before = info.model_dump_json()
    parsed = validate_information_contract(info)
    assert parsed.model_dump_json() == before
    payload = request(model, evidence, parsed)
    assert "measured_initial_values" not in payload
    assert payload["unresolved_states"] == ["z"]


def test_reviewed_measurement_fixes_initial_dependency_without_proposer_record():
    model, evidence, info = fixture(extra_initial=True)
    bound = runtime(model, info)
    validate_public_bindings(model, PUBLIC, bound)
    payload = request(model, evidence, bound)
    assert payload["unresolved_states"] == []
    assert payload["editable_states"] == []
    assert payload["fixed_runtime_rules"]["z"] == {
        "mode": "common_map",
        "arguments": ["b"],
    }
    assert payload["measured_initial_values"][0]["initial_equation"] == "z(t0) = b"
    assert "for editable initializers" in payload["instruction"]
    assert "runtime identity formulas are already fixed" in payload["instruction"]
    derived = derived_evidence(model, evidence, bound)
    assert derived.contract.initial_rules["z"].arguments == ("b",)
    assert evidence.contract.initial_rules.get("z") is None
    assert inspect(model, evidence, bound)["unresolved_states"] == []


@pytest.mark.parametrize(
    "handoff", [choice("shared_parameter", ()), choice(arguments=("a",))]
)
def test_measured_value_cannot_be_overridden_by_shared_or_different_map(handoff):
    model, evidence, info = fixture()
    with pytest.raises(ValueError, match="runtime initialization is fixed for z"):
        inspect(model, evidence, runtime(model, info), handoff)


@pytest.mark.parametrize("change", ["remove", "retype", "rename", "role"])
def test_changed_scientific_declaration_requires_explicit_contract_review(change):
    model, evidence, info = fixture()
    bound = runtime(model, info)
    raw = model.model_dump(mode="json")
    variable = next(row for row in raw["variables"] if row["name"] == "z")
    if change == "remove":
        raw["variables"].remove(variable)
    elif change == "retype":
        variable["definition"] = "algebraic"
    elif change == "rename":
        variable["name"] = "new_z"
    else:
        variable["scientific_role"] = "different scientific interpretation"
    changed = type(model).model_validate(raw)
    with pytest.raises(ValueError, match="MEASURED_INITIAL_STATE_CHANGED"):
        inspect(changed, evidence, bound)
    with pytest.raises(ValueError, match="MEASURED_INITIAL_STATE_CHANGED"):
        validate_public_bindings(changed, PUBLIC, bound)


def test_changed_or_forged_public_citation_is_rejected():
    model, _, info = fixture()
    bound = runtime(model, info)
    with pytest.raises(ValueError, match="MEASURED_INITIAL_PUBLIC_SOURCE"):
        validate_public_bindings(model, "changed " + PUBLIC, bound)
    raw = bound.model_dump(mode="json")
    raw["measured_initial_bindings"][0]["public_excerpt"] = "not present"
    with pytest.raises(ValueError, match="MEASURED_INITIAL_PUBLIC_SOURCE"):
        validate_public_bindings(
            model, PUBLIC, MeasuredInitialInformationContract.model_validate(raw)
        )


def test_unavailable_and_competing_bindings_are_rejected():
    model, evidence, info = fixture()
    with pytest.raises(ValueError, match="MEASURED_INITIAL_SOURCE_UNAVAILABLE"):
        inspect(model, evidence, runtime(model, info, source="hidden"))
    raw = runtime(model, info).model_dump(mode="json")
    raw["measured_initial_bindings"] *= 2
    with pytest.raises(ValueError, match="distinct states"):
        MeasuredInitialInformationContract.model_validate(raw)
    with pytest.raises(ValueError, match="MEASURED_INITIAL_FIXED_CONFLICT"):
        inspect(model, evidence, runtime(model, info, state="a"))


def test_rejected_reply_preserves_valid_intent_without_partial_commit():
    model, evidence, info = fixture()
    attempted = InitializerDependencyHandoff(
        records=choice(arguments=("b",)).records
        + choice("shared_parameter", (), "a").records
    )
    before = attempted.model_dump(mode="json")
    result = rejected_handoff_feedback(model, evidence, info, attempted)
    assert result["transaction_committed"] is False
    assert result["valid_intended_editable_records"] == [before["records"][0]]
    assert result["invalid_records"][0]["state"] == "a"
    assert "fixed" in result["invalid_records"][0]["error"]
    assert attempted.model_dump(mode="json") == before
    assert request(model, evidence, info)["retained_handoff"]["records"] == []


def test_rejected_new_state_receipt_distinguishes_retained_and_intended_topology():
    model, evidence, info = fixture()
    attempted = choice("shared_parameter", (), "new_state")
    result = rejected_handoff_feedback(model, evidence, info, attempted)
    assert result["invalid_records"][0]["state"] == "new_state"
    assert not result["transaction_committed"]
    assert "isolated checks use the retained topology" in result["instruction"]
    assert "resubmitted together with its initializer" in result["instruction"]
    assert "does not by itself" in result["instruction"]


def test_measured_identity_is_compiled_without_calls_and_rechecked(
    tmp_path, monkeypatch
):
    from tests.test_initializer_handoff_consumption import contract, handoff

    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "synthetic-runtime",
    )
    candidate, context, client, requests, _ = setup(tmp_path)
    model, _, _ = fixture()
    # Compiler-boundary validation uses selected names and exact runtime identity;
    # declaration/public attestations were checked at the topology boundary.
    raw = runtime(model, contract(context), source="v01").model_dump(mode="json")
    raw["measured_initial_bindings"][0]["state"] = "m"
    bound = MeasuredInitialInformationContract.model_validate(raw)
    kwargs = {
        "initial_information_contract": bound,
        "initializer_handoff": handoff(),
    }
    result = construct_initializers(
        candidate, context, {}, client, tmp_path / "result", **kwargs
    )
    assert requests == result["attempts"] == []
    assert result["plan"]["rules"]["m"]["initial"]["expression"] == "v01"
    assert result["plan"]["rules"]["m"]["initial"]["parameters"] == []
    assert (
        construct_initializers(
            candidate, context, {}, client, tmp_path / "result", **kwargs
        )
        == result
    )
    assert compile_initialization_result(candidate, context, result)
    changed = copy.deepcopy(result)
    changed["plan"]["rules"]["m"]["initial"]["expression"] = "2*v01"
    with pytest.raises(ValueError, match="MEASURED_INITIAL_IDENTITY"):
        compile_initialization_result(candidate, context, changed)


def test_measured_binding_cannot_disappear_at_formula_boundary(tmp_path):
    from tests.test_initializer_handoff_consumption import contract, handoff

    candidate, context, client, requests, _ = setup(tmp_path)
    model, _, _ = fixture()
    raw = runtime(model, contract(context), source="v01").model_dump(mode="json")
    raw["measured_initial_bindings"][0]["state"] = "m"
    bound = MeasuredInitialInformationContract.model_validate(raw)
    with pytest.raises(ValueError, match="MEASURED_INITIAL_DEPENDENCY"):
        construct_initializers(
            candidate,
            context,
            {},
            client,
            tmp_path / "result",
            initial_information_contract=bound,
            initializer_handoff=handoff(mode="shared_parameter", arguments=()),
        )
    assert not requests


@pytest.mark.parametrize("alias", [False, True])
def test_interaction_source_consumes_identity_including_public_aliases(
    tmp_path, monkeypatch, alias
):
    from tests.test_initializer_handoff_consumption import runner_setup

    monkeypatch.setattr(
        "autoformalism.rebuttal.staged_topology_campaign.runtime_source_hash",
        lambda: "synthetic-runtime",
    )
    args, requests = runner_setup(tmp_path, alias=alias)
    source = args[2]
    model, _, _ = fixture()
    bound = runtime(
        model,
        validate_information_contract(source["initial_information_contract"]),
        source="z" if alias else "x",
    )
    source["initial_information_contract"] = bound.model_dump(mode="json")
    result = run_staged_functions(*args)
    assert result["complete_model"], result["error"]
    assert len(requests) == 2
    assert not result["initialization"]["attempts"]
    rules = result["initialization"]["plan"]["rules"]
    assert len(rules) == 1
    state, rule = next(iter(rules.items()))
    assert state.startswith("af_internal_aux_") if alias else state == "z"
    assert rule["initial"]["expression"] == ("z" if alias else "x")
    assert rule["initial"]["parameters"] == []
    assert run_staged_functions(*args) == result
    assert len(requests) == 2
