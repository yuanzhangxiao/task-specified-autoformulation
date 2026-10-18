"""Stable advisory identities retain provenance without certifying a response."""

from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.research.construction_training import (
    HISTORICAL_QUESTION_IDENTITY,
    MATHEMATICAL_QUESTION_IDENTITY,
    SEMANTIC_QUESTION_IDENTITY,
    compatibility_feedback,
)
from autoformalism.search import construction_evidence
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    build_evidence,
    diagnose,
)
from autoformalism.staged_topology import content_hash
from tests.test_construction_realizations import public


def fixture():
    """Complete synthetic evidence, unrelated to any benchmark representation."""
    context = ValidationContext(targets=("a",), external_inputs=("u", "v"))
    trajectories = tuple(
        Trajectory(
            trajectory_id=f"record{i}",
            time=np.asarray([0, 1, 2], float),
            targets={"a": np.asarray(values, float)},
            auxiliaries={},
            external_inputs={"u": np.zeros(3), "v": np.zeros(3)},
            fixed_covariates={},
            derivatives={},
        )
        for i, values in enumerate(((0, 1, 0), (0, 3, 0)))
    )
    train = DatasetSplit(SplitName.TRAIN, trajectories, "synthetic")
    contract = EvidenceContract(
        provenance="Fixture public runtime assumptions, not sampled inference",
        dependencies_exhaustive=True,
        shared_parameters=True,
        open_loop=True,
        reset_impulse_free=True,
        uniqueness_assumed=True,
        interpolation={"u": "linear_complete_knots", "v": "linear_complete_knots"},
        absolute_error_bounds={"a": 0, "u": 0, "v": 0},
        initial_rules={"a": InitialRule(mode="observed_initial")},
    )
    model = Draft.model_validate(
        {
            "variables": [
                {"name": "a", "definition": "differential", "scientific_role": "output"}
            ],
            "processes": [
                {
                    "name": "P",
                    "depends_on": ["a", "u"],
                    "kind": "influence",
                    "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
                    "scientific_meaning": "the selected response process",
                }
            ],
        }
    )
    return model, context, train, contract


def feedback(model, evidence, identity_policy=SEMANTIC_QUESTION_IDENTITY):
    return compatibility_feedback(
        model, evidence, identity_policy=identity_policy
    )


def ids(value):
    return {q["property"]: q["question_id"] for q in value["advisory_questions"]}


def test_historical_identity_and_response_fields_remain_unchanged():
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    historical = compatibility_feedback(model, evidence)
    assert historical == compatibility_feedback(
        model, evidence, identity_policy=HISTORICAL_QUESTION_IDENTITY
    )
    report = diagnose(model, evidence)
    expected = {
        row.property: content_hash(
            {
                "evidence_sha256": report.evidence_sha256,
                "topology_sha256": report.topology_sha256,
                "diagnostic": row.model_dump(mode="json"),
            }
        )[:20]
        for row in report.diagnostics
        if row.status == "conditional_conflict"
    }
    assert ids(historical) == expected
    assert set(historical) == {
        "protocol",
        "evidence_sha256",
        "topology_sha256",
        "acceptance_gate",
        "advisory_questions",
        "counts",
        "full_report",
    }
    assert all(
        "semantic_topology_sha256" not in q for q in historical["advisory_questions"]
    )
    assert set(ids(historical).values()).isdisjoint(
        ids(feedback(model, evidence)).values()
    )
    with pytest.raises(ValueError, match="identity policy"):
        compatibility_feedback(model, evidence, identity_policy="unknown")


@pytest.mark.parametrize(
    "identity_policy", [SEMANTIC_QUESTION_IDENTITY, MATHEMATICAL_QUESTION_IDENTITY]
)
def test_prose_and_bookkeeping_edits_preserve_ids_but_not_full_provenance(
    identity_policy,
):
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    before = feedback(model, evidence, identity_policy)
    raw = model.model_dump(mode="json")
    raw["variables"][0]["scientific_role"] = "revised explanation of the same output"
    raw["processes"][0]["scientific_meaning"] = "clearer explanation, same declaration"
    raw["feedback_bindings"] = [{"target": "a", "states": ["a"]}]
    revised = Draft.model_validate(raw)
    after = feedback(revised, evidence, identity_policy)
    assert before["topology_sha256"] != after["topology_sha256"]
    assert ids(before) == ids(after)
    assert before["advisory_questions"] == after["advisory_questions"]
    assert after["question_identity_policy"] == identity_policy
    assert after["full_report"]["topology_sha256"] == content_hash(raw)
    assert ids(compatibility_feedback(model, evidence)) != ids(
        compatibility_feedback(revised, evidence)
    )


def test_valid_independent_subsystem_and_declaration_order_do_not_reask():
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    before = feedback(model, evidence)
    raw = model.model_dump(mode="json")
    raw["variables"].insert(
        0, {"name": "z", "definition": "differential", "scientific_role": "independent"}
    )
    raw["equations"] = [
        {
            "name": "z",
            "terms": [
                {
                    "sources": ["z", "v"],
                    "outer_weight_sign": "negative",
                    "scientific_role": "independent evolution",
                }
            ],
        }
    ]
    raw["processes"][0]["depends_on"].reverse()
    after = feedback(Draft.model_validate(raw), evidence)
    assert ids(after) == ids(before)
    assert after["topology_sha256"] != before["topology_sha256"]
    # Invalid independent topology still defers the diagnostic; identity stability
    # does not bypass the existing whole-draft well-formedness requirements.
    raw["equations"][0]["terms"][0]["sources"] = ["unknown"]
    assert not feedback(Draft.model_validate(raw), evidence)["advisory_questions"]


@pytest.mark.parametrize(
    "identity_policy", [SEMANTIC_QUESTION_IDENTITY, MATHEMATICAL_QUESTION_IDENTITY]
)
@pytest.mark.parametrize(
    "edit", ["dependency", "kind", "sign", "conversion", "multiplicity"]
)
def test_scientific_process_edits_reask_even_when_the_conflict_remains(
    edit, identity_policy
):
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    before = feedback(model, evidence, identity_policy)
    raw = model.model_dump(mode="json")
    process = raw["processes"][0]
    if edit == "dependency":
        process["depends_on"] = ["a", "v"]
    elif edit == "kind":
        process["kind"] = "transfer"
    elif edit == "sign":
        process["uses"][0]["sign"] = "negative"
    elif edit == "conversion":
        process["uses"][0]["conversion"] = "2"
    else:
        process["uses"].append(dict(process["uses"][0]))
    after = feedback(Draft.model_validate(raw), evidence, identity_policy)
    assert ids(before).keys() == ids(after).keys()
    assert set(ids(before).values()).isdisjoint(ids(after).values())


@pytest.mark.parametrize(
    "identity_policy", [SEMANTIC_QUESTION_IDENTITY, MATHEMATICAL_QUESTION_IDENTITY]
)
def test_relevant_variable_type_and_ordinary_dependencies_reask(identity_policy):
    model, context, train, contract = fixture()
    raw = model.model_dump(mode="json")
    raw["processes"][0]["depends_on"] = ["a", "z"]
    raw["variables"].append(
        {"name": "z", "definition": "algebraic", "scientific_role": "mediator"}
    )
    raw["equations"] = [
        {
            "name": "z",
            "terms": [
                {
                    "sources": ["u"],
                    "outer_weight_sign": "positive",
                    "scientific_role": "mediated response",
                }
            ],
        }
    ]
    contract = contract.model_copy(
        update={
            "initial_rules": {
                **contract.initial_rules,
                "z": InitialRule(mode="shared_parameter"),
            }
        }
    )
    evidence = build_evidence(train, context, contract)
    before = feedback(Draft.model_validate(raw), evidence, identity_policy)
    raw["equations"][0]["terms"][0]["scientific_role"] = "better prose"
    assert ids(feedback(Draft.model_validate(raw), evidence, identity_policy)) == ids(
        before
    )
    raw["equations"][0]["terms"][0]["sources"] = ["v"]
    dependency = feedback(Draft.model_validate(raw), evidence, identity_policy)
    assert ids(dependency)["closed_subsystem"] != ids(before)["closed_subsystem"]
    raw["variables"][1]["definition"] = "differential"
    typed = feedback(Draft.model_validate(raw), evidence, identity_policy)
    assert ids(typed)["closed_subsystem"] != ids(dependency)["closed_subsystem"]
    assert "scalar_autonomous" not in ids(typed)


def test_changed_evidence_and_conditioning_hypotheses_reask_or_defer():
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    before = feedback(model, evidence)
    trajectories = (
        replace(train.trajectories[0], targets={"a": np.array([0, 2, 0])}),
        train.trajectories[1],
    )
    altered = DatasetSplit(SplitName.TRAIN, trajectories, "altered")
    after = feedback(model, build_evidence(altered, context, contract))
    assert set(ids(before).values()).isdisjoint(ids(after).values())
    modified = contract.model_copy(
        update={
            "initial_rules": {"a": InitialRule(mode="common_map", arguments=("a",))}
        }
    )
    conditioned = feedback(model, build_evidence(train, context, modified))
    assert set(ids(before).values()).isdisjoint(ids(conditioned).values())
    unsupported = contract.model_copy(update={"uniqueness_assumed": None})
    assert not feedback(model, build_evidence(train, context, unsupported))[
        "advisory_questions"
    ]


@pytest.mark.parametrize("disposition", ["retain_approximation", "dispute_assumption"])
def test_answering_does_not_clear_the_conflict_or_verify_science(disposition):
    model, context, train, contract = fixture()
    evidence = build_evidence(train, context, contract)
    before = feedback(model, evidence)
    draft = construction_evidence.Draft.model_validate(
        model.model_dump(mode="json", exclude={"mechanism_bindings"})
    )
    patch = construction_evidence.DraftPatch(
        stage_complete=True,
        compatibility_decisions=[
            {
                "question_id": before["advisory_questions"][0]["question_id"],
                "disposition": disposition,
                "explanation": "Proposer explanation, not new verified evidence.",
            }
        ],
    )
    after = construction_evidence.apply_patch(public(), draft, patch)
    assert after == draft
    reassessed = feedback(after, evidence)
    assert ids(reassessed) == ids(before)
    assert reassessed["counts"] == before["counts"]
    assert reassessed["counts"]["conditional_conflict"] == 2
    assert reassessed["acceptance_gate"] is False
