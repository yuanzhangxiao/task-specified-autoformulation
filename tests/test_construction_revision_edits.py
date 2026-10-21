"""Construction revisions are atomic, scoped, explicit and training-only."""

from copy import deepcopy

import pytest

from autoformalism.research.construction_revision_edits import (
    RevisionReply,
    RevisionValidationError,
    apply_revision,
    model_view,
    public_path_checks,
)
from autoformalism.schemas.prefix_initialization import PrefixFitProblem
from autoformalism.schemas.staged_topology import PublicScientificBrief


def problem(*, trajectory=True):
    """Two consumers share a law; one latent map depends on a scoped root."""
    return PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "parent",
                "parent_candidate_id": None,
                "states": [
                    {
                        "name": name,
                        "kind": "observed" if name in {"I", "Gp"} else "latent",
                    }
                    for name in ("I", "Gp", "mem", "dep")
                ],
                "processes": [{"name": "q", "expression": "a*mem"}],
                "state_equations": [
                    {"state": "I", "rhs": "-q+u"},
                    {"state": "Gp", "rhs": "q"},
                    {"state": "mem", "rhs": "I-mem"},
                    {"state": "dep", "rhs": "-dep"},
                ],
                "parameters": [{"name": "a", "scope": "global", "role": "rate"}],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("I", "Gp", "mem", "dep")
                ],
                "observation_mappings": [
                    {"channel": name, "expression": name} for name in ("I", "Gp")
                ],
            },
            "context": {
                "targets": ["I", "Gp"],
                "external_inputs": ["u"],
                "auxiliaries": ["aux"],
            },
            "graph": {
                "rules": {
                    "mem": {
                        "initial": {
                            "mode": "value",
                            "scope": "trajectory" if trajectory else "shared_training",
                            "role": "nonnegative_coefficient",
                            "guess": 2.0,
                        }
                    },
                    "dep": {
                        "initial": {
                            "mode": "map",
                            "expression": "gain*first+measured",
                            "parameters": [{"name": "gain", "guess": 1.0}],
                            "state_initials": {"first": "mem"},
                            "public_initials": {"measured": "aux"},
                        }
                    },
                }
            },
            "parameter_guesses": {"a": 1.0},
        }
    )


def reply(**changes):
    """A justified formula and dependency revision without fitted values."""
    return {
        "hypothesis": "Glucose dependence may explain the training pattern.",
        "evidence_ids": ["training:I"],
        "expected_observable_effect": "Change responses across glucose levels.",
        "edits": [
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["mem", "Gp"],
                "expression": "a*mem*Gp",
            }
        ],
        "new_parameters": [],
        **changes,
    }


def brief(*, memory=True, auxiliary=False):
    """Only public dependency obligations enter path validation."""
    return PublicScientificBrief.model_validate(
        {
            "scientific_context": "Represent the delayed action of insulin on glucose.",
            "public_variables": [
                {"name": name, "data_role": role}
                for name, role in (
                    ("I", "target"),
                    ("Gp", "target"),
                    ("u", "external_input"),
                    ("aux", "auxiliary"),
                    ("t", "time"),
                )
            ],
            "requirements": [
                {
                    "id": "delayed_action",
                    "public_requirement": "A delayed insulin pathway",
                    "drivers": ["I"],
                    "targets": ["Gp"],
                    "requires_dynamic_memory": memory,
                }
            ],
            "target_dependencies": [
                {
                    "target": "Gp",
                    "acceptable_sources": ["aux"],
                    "public_requirement": "Include the auxiliary contribution.",
                }
            ]
            if auxiliary
            else [],
        }
    )


@pytest.mark.parametrize("trajectory", [True, False])
def test_joint_source_revision_preserves_all_scopes_and_consumers(
    trajectory, monkeypatch
):
    source = problem(trajectory=trajectory)
    before = source.model_dump(mode="json")
    from autoformalism.fitting import prefix_initialization

    def prohibited(*args, **kwargs):
        pytest.fail("construction must not run fitting or simulation")

    for name in ("fit_training", "infer_prefix", "forecast", "symbolic_rollout"):
        monkeypatch.setattr(prefix_initialization, name, prohibited)
    result = apply_revision(source, reply(), evidence_ids=["training:I"], brief=brief())
    child = result["problem"]
    assert result["status"] == "revised"
    assert result["audit"]["static_eligibility"] == "eligible"
    assert child.base_candidate.processes[0].expression == "a*mem*Gp"
    assert child.base_candidate.state_equations == source.base_candidate.state_equations
    assert child.graph == source.graph
    assert child.context == source.context
    assert child.parameter_guesses == source.parameter_guesses
    for field in (
        "states",
        "initial_conditions",
        "constraints",
        "observation_mappings",
        "parameters",
    ):
        assert getattr(child.base_candidate, field) == getattr(
            source.base_candidate, field
        )
    assert result["audit"]["shared_laws_after"][0]["direct_consumers"] == ["Gp", "I"]
    assert source.model_dump(mode="json") == before
    assert (
        apply_revision(source, reply(), evidence_ids=["training:I"], brief=brief())
        == result
    )


def test_linear_offset_allowed_and_unused_old_parameter_removed():
    source = problem()
    raw = reply(
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["mem"],
                "expression": "mem+b",
            }
        ],
        new_parameters=[{"name": "b", "role": "offset"}],
    )
    result = apply_revision(source, raw, evidence_ids=["training:I"])
    assert result["audit"]["removed_unused_parameters"] == ["a"]
    assert [p.name for p in result["problem"].base_candidate.parameters] == ["b"]
    assert set(result["problem"].parameter_guesses) == {"b"}
    assert result["problem"].graph == source.graph


def test_noop_retains_exact_problem_and_reports_existing_path_gap():
    source = problem()
    result = apply_revision(
        source,
        reply(edits=[]),
        evidence_ids=["training:I"],
        brief=brief(auxiliary=True),
        scientific_aliases={},
    )
    assert result["status"] == "retained"
    assert result["problem"] == source
    assert result["audit"]["static_eligibility"] == "repair_required"
    view = model_view(source)
    assert view["initialization_graph"] == source.graph.model_dump(mode="json")
    assert view["equations"][-1]["sources"] == ["mem"]
    assert view["equations"][0]["display"] == "d(I)/dt = -q+u"
    assert view["equations"][-1]["display"] == "q = a*mem"
    assert view["equations"][-1]["scientific_role"] == "unspecified"
    assert view["parameters"] == [{"name": "a", "role": "rate", "domain": "positive"}]
    assert view["states"][0] == {"name": "I", "kind": "observed"}
    assert "law" not in view["shared_law_relationships"][0]
    assert "retained_fitted_parameters" not in view


@pytest.mark.parametrize(
    "mutation,code",
    [
        ({"evidence_ids": ["validation:I"]}, "UNKNOWN_EVIDENCE"),
        (
            {"new_parameters": [{"name": "a", "role": "coefficient"}]},
            "PARAMETER_IDENTITY",
        ),
        (
            {"new_parameters": [{"name": "unused", "role": "rate"}]},
            "UNUSED_NEW_PARAMETER",
        ),
        ({"new_parameters": [{"name": "Gp", "role": "rate"}]}, "PARAMETER_IDENTITY"),
        (
            {"new_parameters": [{"name": "init_mem_value", "role": "rate"}]},
            "PARAMETER_IDENTITY",
        ),
        ({"graph": {}}, "REVISION_SCHEMA"),
        (
            {
                "new_parameters": [
                    {"name": "b", "role": "rate", "bounds": {"lower": 0, "upper": 1}}
                ]
            },
            "REVISION_SCHEMA",
        ),
    ],
)
def test_rejected_reply_never_changes_parent(mutation, code):
    source = problem()
    before = deepcopy(source.model_dump(mode="json"))
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(source, reply(**mutation), evidence_ids=["training:I"])
    assert error.value.code == code
    assert error.value.as_dict()["code"] == code
    assert source.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "edit,code",
    [
        (
            {
                "component": "new_state",
                "kind": "dynamic",
                "sources": ["I"],
                "expression": "I",
            },
            "COMPONENT_IDENTITY",
        ),
        (
            {"component": "q", "kind": "dynamic", "sources": ["I"], "expression": "I"},
            "COMPONENT_IDENTITY",
        ),
        (
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["mem"],
                "expression": "a*mem*Gp",
            },
            "SOURCE_MISMATCH",
        ),
        (
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["future_I"],
                "expression": "future_I",
            },
            "UNAVAILABLE_SOURCE",
        ),
        (
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["q"],
                "expression": "q",
            },
            "MODEL_VALIDATION",
        ),
        (
            {
                "component": "q",
                "kind": "algebraic",
                "sources": [],
                "expression": "__import__('os')",
            },
            "MODEL_VALIDATION",
        ),
    ],
)
def test_dependency_formula_and_cycle_failures_are_atomic(edit, code):
    source = problem()
    before = source.model_dump(mode="json")
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(source, reply(edits=[edit]), evidence_ids=["training:I"])
    assert error.value.code == code
    assert source.model_dump(mode="json") == before


def test_changed_shared_law_cannot_lose_consumer_in_coordinated_patch():
    raw = reply()
    raw["edits"].append(
        {"component": "I", "kind": "dynamic", "sources": ["u"], "expression": "u"}
    )
    with pytest.raises(RevisionValidationError, match="SHARED_CONSUMER_REMOVED"):
        apply_revision(problem(), raw, evidence_ids=["training:I"])


def test_public_memory_path_cannot_be_replaced_by_source_self_memory():
    raw = reply(
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["I"],
                "expression": "a*I",
            },
            {"component": "Gp", "kind": "dynamic", "sources": ["q"], "expression": "q"},
        ]
    )
    # Gp itself remains a real downstream dynamic state and therefore qualifies
    # as declared memory. The source state I never serves as its own witness.
    result = apply_revision(problem(), raw, evidence_ids=["training:I"], brief=brief())
    assert (
        result["audit"]["public_path_checks_after"][0]["witnesses"][0]["memory_state"]
        == "Gp"
    )
    raw = reply(
        edits=[
            {
                "component": "mem",
                "kind": "dynamic",
                "sources": ["mem"],
                "expression": "-mem",
            }
        ]
    )
    with pytest.raises(RevisionValidationError, match="PUBLIC_PATH_REGRESSION"):
        apply_revision(problem(), raw, evidence_ids=["training:I"], brief=brief())


def test_initial_only_auxiliary_path_is_reported_without_dynamic_pass():
    raw = reply(
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["mem", "dep"],
                "expression": "a*mem+dep",
            }
        ]
    )
    result = apply_revision(
        problem(),
        raw,
        evidence_ids=["training:I"],
        brief=brief(auxiliary=True),
        scientific_aliases={},
    )
    row = result["audit"]["public_path_checks_after"][-1]
    assert row["status"] == "initialization_only"
    assert row["initialization_only_paths"][0]["path"][0] == "aux"
    assert result["audit"]["static_eligibility"] == "repair_required"


def test_schema_is_list_based_and_public_brief_must_match_context():
    schema = RevisionReply.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["edits"]["type"] == "array"
    invalid = brief().model_dump(mode="json")
    invalid["public_variables"].append(
        {"name": "secret_channel", "data_role": "auxiliary"}
    )
    with pytest.raises(RevisionValidationError, match="PUBLIC_BRIEF_CONTEXT"):
        public_path_checks(problem(), PublicScientificBrief.model_validate(invalid))


def modeled_auxiliary_problem():
    """Mirror a compiler-renamed generated auxiliary with its exact t0 identity."""
    raw = problem().model_dump(mode="json")
    candidate = raw["base_candidate"]
    candidate["states"].append(
        {
            "name": "af_internal_aux_1",
            "kind": "latent",
            "description": "Modeled auxiliary contribution",
        }
    )
    candidate["state_equations"].append({"state": "af_internal_aux_1", "rhs": "0"})
    candidate["initial_conditions"].append(
        {
            "state": "af_internal_aux_1",
            "scope": "global",
            "fixed_value": 0,
        }
    )
    candidate["state_equations"][1]["rhs"] = "q+af_internal_aux_1"
    raw["graph"]["rules"]["af_internal_aux_1"] = {
        "initial": {
            "mode": "map",
            "expression": "public_1",
            "parameters": [],
            "state_initials": {},
            "public_initials": {"public_1": "aux"},
        }
    }
    return PrefixFitProblem.model_validate(raw)


def test_verified_modeled_auxiliary_alias_is_a_semantic_source_not_forcing_edge():
    source = modeled_auxiliary_problem()
    aliases = {"I": "I", "Gp": "Gp", "aux": "af_internal_aux_1"}
    result = apply_revision(
        source,
        reply(edits=[]),
        evidence_ids=["training:I"],
        brief=brief(auxiliary=True),
        scientific_aliases=aliases,
    )
    assert result["status"] == "retained"
    assert result["problem"] == source
    assert result["audit"]["static_eligibility"] == "eligible"
    witness = result["audit"]["public_path_checks_after"][-1]["witnesses"][0]
    assert witness["source"] == "aux"
    assert witness["actual_source"] == "af_internal_aux_1"
    assert witness["source_mode"] == "modeled_alias"
    assert witness["path"][0] == "af_internal_aux_1"
    assert "aux" not in witness["path"]
    assert (
        model_view(source, scientific_aliases=aliases)["scientific_aliases"] == aliases
    )


def test_unknown_alias_metadata_is_unassessed_not_a_missing_mechanism_claim():
    source = modeled_auxiliary_problem()
    result = apply_revision(
        source,
        reply(edits=[]),
        evidence_ids=["training:I"],
        brief=brief(auxiliary=True),
    )
    assert result["audit"]["static_eligibility"] == "unassessed"
    assert (
        result["audit"]["public_path_checks_after"][-1]["status"]
        == "unresolved_identity"
    )
    explicit_absence = public_path_checks(
        source, brief(auxiliary=True), scientific_aliases={}
    )
    assert explicit_absence[-1]["status"] == "initialization_only"


def test_explicit_raw_auxiliary_dependency_remains_permitted_with_modeled_alias():
    source = modeled_auxiliary_problem()
    result = apply_revision(
        source,
        reply(
            edits=[
                {
                    "component": "Gp",
                    "kind": "dynamic",
                    "sources": ["q", "aux"],
                    "expression": "q+aux",
                }
            ]
        ),
        evidence_ids=["training:I"],
        brief=brief(auxiliary=True),
        scientific_aliases={"aux": "af_internal_aux_1"},
    )
    witness = result["audit"]["public_path_checks_after"][-1]["witnesses"][0]
    assert witness["source_mode"] == "direct"
    assert witness["actual_source"] == "aux"
    assert result["audit"]["static_eligibility"] == "eligible"


@pytest.mark.parametrize(
    "aliases",
    [
        {"u": "mem"},
        {"aux": "unknown"},
        {"aux": "I"},
        {"Gp": "mem"},
        {"I": "Gp", "Gp": "I"},
        {"I": "mem", "aux": "mem"},
        {"aux": 1},
    ],
)
def test_invalid_or_borrowed_aliases_rejected_atomically(aliases):
    source = modeled_auxiliary_problem()
    before = source.model_dump(mode="json")
    with pytest.raises(RevisionValidationError, match="SCIENTIFIC_ALIASES"):
        apply_revision(
            source,
            reply(edits=[]),
            evidence_ids=["training:I"],
            brief=brief(auxiliary=True),
            scientific_aliases=aliases,
        )
    assert source.model_dump(mode="json") == before
