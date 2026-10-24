"""Cross-stage repairs remain atomic and preserve the frozen public problem."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JSONValidationError

from autoformalism.llm.staged_topology import strict_provider_schema
from autoformalism.research.construction_revision_consistency import assess
from autoformalism.research.construction_revision_edits import RevisionValidationError
from autoformalism.research.construction_structural_revision import (
    StructuralRevisionReply,
    apply_structural_revision,
    editable_initial_states,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from tests.test_construction_revision_consistency import evidence, row
from tests.test_construction_revision_consistency import problem as observed_problem
from tests.test_construction_revision_edits import brief, problem


def reply(**changes):
    """A scientific structural edit, never fitted numbers or a scoring request."""
    return {
        "decision": "revise",
        "hypothesis": "An independent hidden initial condition may explain variation.",
        "evidence_ids": ["training:I"],
        "expected_observable_effect": "Permit different responses under equal inputs.",
        "edits": [],
        **changes,
    }


def state(name="extra", **changes):
    return {
        "name": name,
        "description": "Additional latent history",
        "rhs": "I-extra",
        "sources": ["I", "extra"],
        "initial": {"mode": "value", "scope": "trajectory"},
        **changes,
    }


def apply(source, payload, **options):
    return apply_structural_revision(
        source, payload, evidence_ids=["training:I"], **options
    )


def test_new_state_shared_process_and_dependent_initialization_are_atomic(monkeypatch):
    source = problem()
    before = source.model_dump(mode="json")
    from autoformalism.fitting import prefix_initialization

    def forbidden(*args, **kwargs):
        pytest.fail("structural revision must not fit or simulate")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, forbidden)
    payload = reply(
        edits=[
            {
                "component": "I",
                "kind": "dynamic",
                "sources": [],
                "expression": "d(I)/dt = -q+u+r",
            },
            {"component": "Gp", "kind": "dynamic", "sources": [], "expression": "q-r"},
        ],
        new_states=[state()],
        new_processes=[
            {"name": "r", "description": "New shared exchange", "expression": "b*extra"}
        ],
        new_parameters=[{"name": "b", "role": "rate"}],
        initialization_edits=[
            {
                "state": "dep",
                "initial": {
                    "mode": "map",
                    "expression": "gain*root+measured",
                    "parameters": [{"name": "gain", "role": "coefficient"}],
                    "state_initials": [{"alias": "root", "state": "extra"}],
                    "public_initials": [{"alias": "measured", "channel": "aux"}],
                },
            }
        ],
    )
    result = apply(source, payload, brief=brief())
    child = result["problem"]
    assert result["status"] == "revised"
    assert child.context == source.context
    assert (
        child.base_candidate.observation_mappings
        == source.base_candidate.observation_mappings
    )
    assert child.base_candidate.constraints == source.base_candidate.constraints
    assert child.graph.rules["extra"].initial.scope == "trajectory"
    assert child.graph.rules["dep"].initial.state_initials == {"root": "extra"}
    assert child.parameter_guesses["a"] == source.parameter_guesses["a"]
    assert "b" in child.parameter_guesses
    assert source.model_dump(mode="json") == before
    assert apply(source, payload, brief=brief()) == result
    assert result["audit"]["candidate_specific_data_checks_required"]
    assert len(result["audit"]["normalization"]["changes"]) >= 2


def test_new_algebraic_process_can_have_one_consumer():
    result = apply(
        problem(),
        reply(
            edits=[
                {
                    "component": "I",
                    "kind": "dynamic",
                    "sources": [],
                    "expression": "-q+u+r",
                }
            ],
            new_processes=[
                {"name": "r", "description": "Local forcing", "expression": "aux"}
            ],
        ),
    )
    law = next(r for r in result["audit"]["shared_laws_after"] if r["process"] == "r")
    assert law["direct_consumers"] == ["I"]


def test_explicit_removal_updates_all_consumers_and_graph():
    source = problem()
    result = apply(
        source,
        reply(
            remove_states=["dep"],
            remove_processes=["q"],
            edits=[
                {
                    "component": "I",
                    "kind": "dynamic",
                    "sources": [],
                    "expression": "-a*mem+u",
                },
                {
                    "component": "Gp",
                    "kind": "dynamic",
                    "sources": [],
                    "expression": "a*mem",
                },
            ],
        ),
        brief=brief(),
    )
    assert set(result["problem"].graph.rules) == {"mem"}
    assert not result["problem"].base_candidate.processes
    assert source.graph.rules.get("dep")


@pytest.mark.parametrize(
    "payload,code",
    [
        (reply(remove_processes=["q"]), "DANGLING_REFERENCE"),
        (reply(remove_states=["mem"]), "DANGLING_REFERENCE"),
        (reply(remove_states=["I"]), "REMOVAL_SCOPE"),
        (reply(remove_processes=["absent"]), "REMOVAL_SCOPE"),
        (reply(new_states=[state("I")]), "DECLARATION_IDENTITY"),
        (
            reply(
                new_processes=[{"name": "u", "description": "input", "expression": "1"}]
            ),
            "DECLARATION_IDENTITY",
        ),
        (
            reply(new_parameters=[{"name": "a", "role": "coefficient"}]),
            "DECLARATION_IDENTITY",
        ),
        (
            reply(new_parameters=[{"name": "orphan", "role": "rate"}]),
            "UNUSED_NEW_PARAMETER",
        ),
        (
            reply(
                new_states=[state()],
                initialization_edits=[
                    {
                        "state": "extra",
                        "initial": {"mode": "value", "scope": "shared_training"},
                    }
                ],
            ),
            "INITIALIZATION_IDENTITY",
        ),
        (
            reply(
                initialization_edits=[
                    {"state": "I", "initial": {"mode": "value", "scope": "trajectory"}}
                ]
            ),
            "INITIALIZATION_IDENTITY",
        ),
        (
            reply(
                remove_states=["dep"],
                new_processes=[
                    {
                        "name": "dep",
                        "description": "Convert to algebraic",
                        "expression": "mem",
                    }
                ],
            ),
            "DECLARATION_IDENTITY",
        ),
        (
            reply(
                edits=[
                    {
                        "component": "I",
                        "kind": "algebraic",
                        "sources": [],
                        "expression": "u",
                    }
                ]
            ),
            "COMPONENT_IDENTITY",
        ),
        (reply(new_states=[state(rhs="__import__('os')")]), "MODEL_VALIDATION"),
        (reply(new_states=[state(rhs="unknown+I")]), "UNKNOWN_FORMULA_SYMBOL"),
        (reply(context={}), "REVISION_SCHEMA"),
        (
            reply(evidence_ids=["test:metric"], remove_states=["dep"]),
            "UNKNOWN_EVIDENCE",
        ),
    ],
)
def test_rejected_structural_patches_preserve_exact_parent(payload, code):
    source = problem()
    before = deepcopy(source.model_dump(mode="json"))
    with pytest.raises(RevisionValidationError) as error:
        apply(source, payload)
    assert error.value.code == code
    assert source.model_dump(mode="json") == before


def test_initializer_dependency_cycle_and_unused_alias_rejected():
    source = problem()
    for expression, aliases in [
        ("self", [{"alias": "self", "state": "dep"}]),
        ("1", [{"alias": "unused", "state": "mem"}]),
    ]:
        with pytest.raises(RevisionValidationError, match="REVISION_VALIDATION"):
            apply(
                source,
                reply(
                    initialization_edits=[
                        {
                            "state": "dep",
                            "initial": {
                                "mode": "map",
                                "expression": expression,
                                "state_initials": aliases,
                            },
                        }
                    ]
                ),
            )


def test_public_alias_and_known_initial_rules_are_not_editable():
    source = problem()
    assert editable_initial_states(source, {"aux": "mem"}) == ["dep"]
    with pytest.raises(RevisionValidationError, match="INITIALIZATION_IDENTITY"):
        apply(
            source,
            reply(
                initialization_edits=[
                    {
                        "state": "mem",
                        "initial": {"mode": "value", "scope": "trajectory"},
                    }
                ]
            ),
            scientific_aliases={"aux": "mem"},
        )
    raw = source.model_dump(mode="json")
    raw["graph"]["rules"]["mem"] = {
        "initial": {"mode": "known", "value": 0, "justification": "Task preparation"}
    }
    source = PrefixFitProblem.model_validate(raw)
    assert editable_initial_states(source) == ["dep"]


def test_trajectory_scope_requires_explicit_experiment_support():
    with pytest.raises(RevisionValidationError, match="INITIALIZATION_SCOPE"):
        apply(problem(), reply(new_states=[state()]), allow_trajectory_roots=False)


def test_maps_can_depend_on_new_shared_initial_constants():
    result = apply(
        problem(),
        reply(
            new_states=[state(initial={"mode": "value", "scope": "shared_training"})],
            initialization_edits=[
                {
                    "state": "dep",
                    "initial": {
                        "mode": "map",
                        "expression": "root+measured",
                        "state_initials": [{"alias": "root", "state": "extra"}],
                        "public_initials": [{"alias": "measured", "channel": "aux"}],
                    },
                }
            ],
        ),
    )
    audit = result["audit"]["initialization_audit"]
    assert "init_extra_value" in audit["shared_parameter_names"]
    assert "extra" not in audit["trajectory_root_parameters"]
    assert audit["topological_order"].index("extra") < audit["topological_order"].index(
        "dep"
    )


@pytest.mark.parametrize(
    "decision,status",
    [("retain", "retained"), ("requires_wider_revision", "requires_wider_revision")],
)
def test_distinct_no_adoption_decision_is_not_a_scientific_pass(decision, status):
    source = problem()
    result = apply(source, reply(decision=decision))
    assert result["problem"] == (source if decision == "retain" else None)
    assert result["status"] == status
    assert result["audit"]["static_eligibility"] != "eligible"
    with pytest.raises(RevisionValidationError, match="REVISION_SCHEMA") as error:
        apply(source, reply(decision=decision, new_states=[state()]))
    assert "must not contain edits" in str(error.value.details)


def test_retained_model_can_pass_static_paths_but_still_requires_data_gate():
    result = apply(problem(), reply(decision="retain"), brief=brief())
    assert result["audit"]["static_eligibility"] == "eligible"
    assert result["audit"]["candidate_specific_data_checks_required"]
    assert not result["audit"]["scientific_correctness_established"]


def test_duplicate_declarations_produce_specific_bounded_schema_error():
    with pytest.raises(RevisionValidationError, match="REVISION_SCHEMA") as error:
        apply(problem(), reply(new_states=[state(), state()]))
    assert "duplicate new component" in str(error.value.details)
    assert StructuralRevisionReply.model_json_schema()["properties"]["decision"][
        "enum"
    ] == ["revise", "retain", "requires_wider_revision"]


def test_public_path_regression_still_blocks_broad_repair():
    with pytest.raises(RevisionValidationError, match="PUBLIC_PATH_REGRESSION"):
        apply(
            problem(),
            reply(
                edits=[
                    {
                        "component": "mem",
                        "kind": "dynamic",
                        "sources": [],
                        "expression": "-mem",
                    }
                ]
            ),
            brief=brief(),
        )


def test_existing_parameter_roles_and_constraints_remain_unchanged():
    source = problem()
    result = apply(source, reply(new_states=[state()]))
    assert (
        result["problem"].base_candidate.parameters == source.base_candidate.parameters
    )
    assert (
        result["problem"].base_candidate.constraints
        == source.base_candidate.constraints
    )


def test_initializer_repair_defers_equality_without_claiming_scientific_success():
    source = observed_problem(rhs="-a*x+z", root="shared_training")
    data = evidence(source, row("a", (1, 1, 1)), row("b", (1, 3, 4)))
    assert assess(source, data)["status"] == "conflict"
    result = apply(
        source,
        reply(
            initialization_edits=[
                {
                    "state": "z",
                    "initial": {"mode": "value", "scope": "trajectory"},
                }
            ]
        ),
    )
    check = assess(result["problem"], data)
    assert not check["conflicts"]
    assert check["deferred"]
    assert not result["audit"]["scientific_correctness_established"]


def test_adding_unrelated_root_cannot_remove_existing_data_conflict():
    source = observed_problem(root="shared_training")
    data = evidence(source, row("a", (1, 1, 1)), row("b", (1, 3, 4)))
    result = apply(source, reply(new_states=[state(rhs="-extra", sources=["extra"])]))
    assert assess(result["problem"], data)["status"] == "conflict"
    assert result["audit"]["candidate_specific_data_checks_required"]


def test_observed_initials_cannot_be_freed_to_escape_witness():
    source = observed_problem(root="shared_training")
    with pytest.raises(RevisionValidationError, match="INITIALIZATION_IDENTITY"):
        apply(
            source,
            reply(
                initialization_edits=[
                    {"state": "x", "initial": {"mode": "value", "scope": "trajectory"}}
                ]
            ),
        )


def test_strict_provider_schema_accepts_nonempty_dependent_initial_map():
    """The exact provider schema must permit aliases, not seal them out as extras."""
    raw = reply(
        new_states=[state(initial={"mode": "value", "scope": "shared_training"})],
        initialization_edits=[
            {
                "state": "dep",
                "initial": {
                    "mode": "map",
                    "expression": "gain*root+measured",
                    "parameters": [{"name": "gain", "role": "coefficient"}],
                    "state_initials": [{"alias": "root", "state": "extra"}],
                    "public_initials": [{"alias": "measured", "channel": "aux"}],
                },
            }
        ],
    )
    payload = StructuralRevisionReply.model_validate(raw).model_dump(mode="json")
    schema = strict_provider_schema(StructuralRevisionReply.model_json_schema())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(payload)
    graph = apply(problem(), payload)["problem"].graph
    assert graph.rules["dep"].initial.state_initials == {"root": "extra"}
    assert graph.rules["dep"].initial.public_initials == {"measured": "aux"}

    for field, old_mapping in (
        ("state_initials", {"root": "extra"}),
        ("public_initials", {"measured": "aux"}),
    ):
        invalid = deepcopy(payload)
        invalid["initialization_edits"][0]["initial"][field] = old_mapping
        with pytest.raises(JSONValidationError):
            validator.validate(invalid)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "state_initials": [
                {"alias": "root", "state": "mem"},
                {"alias": "root", "state": "dep"},
            ]
        },
        {
            "public_initials": [
                {"alias": "reading", "channel": "aux"},
                {"alias": "reading", "channel": "I"},
            ]
        },
        {
            "state_initials": [{"alias": "both", "state": "mem"}],
            "public_initials": [{"alias": "both", "channel": "aux"}],
        },
        {
            "parameters": [{"name": "gain", "role": "coefficient"}],
            "state_initials": [{"alias": "gain", "state": "mem"}],
        },
    ],
)
def test_initializer_alias_lists_cannot_overwrite_a_duplicate(changes):
    with pytest.raises(RevisionValidationError, match="REVISION_SCHEMA") as error:
        apply(
            problem(),
            reply(
                initialization_edits=[
                    {
                        "state": "dep",
                        "initial": {"mode": "map", "expression": "0", **changes},
                    }
                ]
            ),
        )
    assert "duplicate or ambiguous initializer alias/parameter" in str(
        error.value.details
    )
