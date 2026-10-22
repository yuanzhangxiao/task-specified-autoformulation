"""Construction revisions are atomic, scoped, explicit and training-only."""

from copy import deepcopy

import pytest

from autoformalism.research.construction_revision_edits import (
    FORMATTING_POLICY,
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


@pytest.mark.parametrize(
    "component,kind,expression,expected",
    [
        ("I", "dynamic", "d(I)/dt = -q+u-I", "-q+u-I"),
        ("I", "dynamic", " I' = -q+u-I ", "-q+u-I"),
        ("I", "dynamic", "I\u2032 = -q+u-I", "-q+u-I"),
        ("I", "dynamic", "d ( I ) / d t = -q+u-I", "-q+u-I"),
        ("q", "algebraic", " q = a*mem+Gp ", "a*mem+Gp"),
    ],
)
def test_v2_matching_labels_and_missing_dependencies_are_audited(
    component, kind, expression, expected
):
    source = problem()
    before = source.model_dump(mode="json")
    raw = reply(
        edits=[
            {
                "component": component,
                "kind": kind,
                "sources": [],
                "expression": expression,
            }
        ]
    )
    original = deepcopy(raw)
    result = apply_revision(
        source, raw, evidence_ids=["training:I"], tolerant_formatting=True
    )
    audit = result["audit"]["normalization"]
    assert audit["policy"] == FORMATTING_POLICY
    assert audit["raw_reply"] == original
    assert audit["canonical_reply"] == result["audit"]["reply"]
    normalized = audit["canonical_reply"]["edits"][0]
    assert normalized["expression"] == expected
    assert normalized["sources"] == (
        ["I", "q", "u"] if kind == "dynamic" else ["Gp", "mem"]
    )
    assert [row["field"] for row in audit["changes"]] == ["expression", "sources"]
    assert audit["changes"][0]["before"] == expression
    assert result["problem"].graph == source.graph
    assert result["problem"].context == source.context
    assert result["problem"].base_candidate.initial_conditions == (
        source.base_candidate.initial_conditions
    )
    assert source.model_dump(mode="json") == before
    assert raw == original


def test_v2_parameter_metadata_duplicates_and_unused_sources_are_normalized():
    source = problem()
    raw = reply(
        edits=[
            {
                "component": "q",
                "kind": "algebraic",
                "sources": ["a", "mem", "b", "mem", "unused_metadata"],
                "expression": "q = a*mem+b",
            }
        ],
        new_parameters=[{"name": "b", "role": "offset"}],
    )
    original = deepcopy(raw)
    result = apply_revision(
        source, raw, evidence_ids=["training:I"], tolerant_formatting=True
    )
    canonical = result["audit"]["reply"]["edits"][0]
    assert canonical == {
        "component": "q",
        "kind": "algebraic",
        "sources": ["mem"],
        "expression": "a*mem+b",
    }
    assert result["audit"]["new_parameters"] == ["b"]
    assert result["audit"]["normalization"]["raw_reply"] == original
    assert raw == original
    assert result["problem"].graph == source.graph
    assert result["audit"]["shared_laws_after"][0]["direct_consumers"] == ["Gp", "I"]


@pytest.mark.parametrize(
    "component,kind,expression",
    [
        ("I", "dynamic", "Gp' = -I"),
        ("I", "dynamic", "I = -I"),
        ("I", "dynamic", "d(I)/ds = -I"),
        ("I", "dynamic", "d(Gp)/dt = -I"),
        ("I", "dynamic", "d(I)/dt = I = 0"),
        ("I", "dynamic", "I' == -I"),
        ("I", "dynamic", "I' ="),
        ("q", "algebraic", "d(q)/dt = a*mem"),
        ("q", "algebraic", "q' = a*mem"),
        ("q", "algebraic", "Gp = a*mem"),
        ("q", "algebraic", "q <= a*mem"),
    ],
)
def test_v2_ambiguous_or_mismatched_labels_are_rejected_without_guessing(
    component, kind, expression
):
    source = problem()
    before = source.model_dump(mode="json")
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            source,
            reply(
                edits=[
                    {
                        "component": component,
                        "kind": kind,
                        "sources": [],
                        "expression": expression,
                    }
                ]
            ),
            evidence_ids=["training:I"],
            tolerant_formatting=True,
        )
    assert error.value.code == "RHS_LABEL"
    assert error.value.details["expected_kind"] == kind
    assert error.value.details["syntax_examples"]
    assert source.model_dump(mode="json") == before


def test_v2_component_kind_feedback_explains_allowed_self_dependency():
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            problem(),
            reply(
                edits=[
                    {
                        "component": "I",
                        "kind": "algebraic",
                        "sources": ["I"],
                        "expression": "I = -I",
                    }
                ]
            ),
            evidence_ids=["training:I"],
            tolerant_formatting=True,
        )
    assert error.value.code == "COMPONENT_IDENTITY"
    assert error.value.details["expected_kind"] == "dynamic"
    assert error.value.details["existing_components"]["q"] == "algebraic"
    assert "may reference its own state" in str(error.value)


def test_v2_unknown_formula_symbol_is_not_invented_from_source_metadata():
    raw = reply()
    raw["edits"][0].update(
        sources=["undeclared_rate", "mem"], expression="undeclared_rate*mem"
    )
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
        )
    assert error.value.code == "UNKNOWN_FORMULA_SYMBOL"
    details = error.value.details
    assert details["unknown"] == ["undeclared_rate"]
    assert details["declared_parameters"] == ["a"]
    assert "mem" in details["allowed_variable_sources"]
    assert "new_parameters" in details["guidance"]
    assert "cannot add states or processes" in details["guidance"]


def test_v2_time_indexed_sources_keep_existing_restricted_parser_semantics():
    raw = reply()
    raw["edits"][0].update(sources=[], expression="a*mem+aux(t)")
    result = apply_revision(
        problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
    )
    normalized = result["audit"]["reply"]["edits"][0]
    assert normalized["expression"] == "a*mem+aux(t)"
    assert normalized["sources"] == ["aux", "mem"]


@pytest.mark.parametrize(
    "expression",
    [
        "mem" + "*1" * 1000,
        "+".join(["(" + "+".join(["mem"] * 30) + ")"] * 10),
        "-" * 2000 + "mem",
    ],
)
def test_v2_expression_complexity_rejected_before_recursive_parser(
    expression, monkeypatch
):
    from autoformalism.research import construction_revision_edits as module

    def forbidden(*args, **kwargs):
        pytest.fail("recursive restricted parser ran before AST bounds")

    monkeypatch.setattr(module.RestrictedParser, "parse", forbidden)
    raw = reply()
    raw["edits"][0].update(expression=expression)
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
        )
    assert error.value.code == "EXPRESSION_COMPLEXITY"
    assert error.value.details["limits"]["max_depth"] == 64
    assert error.value.details["syntax_examples"] == ["0", "q = 0"]


@pytest.mark.parametrize("expression", ["a*", "(a*mem", "a mem"])
def test_v2_syntax_feedback_contains_component_kind_and_rhs_examples(expression):
    raw = reply()
    raw["edits"][0]["expression"] = expression
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
        )
    assert error.value.code == "RHS_SYNTAX"
    assert error.value.details["expected_kind"] == "algebraic"
    assert error.value.details["syntax_examples"] == ["0", "q = 0"]


@pytest.mark.parametrize(
    "expression", ["__import__('os')", "mem.__class__", "[mem][0]"]
)
def test_v2_still_uses_restricted_expression_grammar(expression):
    raw = reply()
    raw["edits"][0].update(sources=[], expression=expression)
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
        )
    assert error.value.code == "MODEL_VALIDATION"
    assert error.value.details["diagnostics"]
    assert "restricted scalar RHS" in error.value.details["guidance"]


@pytest.mark.parametrize("gate", ["shared_law", "public_path", "evidence", "schema"])
def test_v2_normalization_preserves_existing_scientific_and_scope_gates(gate):
    source = problem()
    before = source.model_dump(mode="json")
    raw = reply()
    raw["edits"][0]["expression"] = "q = a*mem*Gp"
    if gate == "shared_law":
        raw["edits"].append(
            {"component": "I", "kind": "dynamic", "sources": [], "expression": "I' = u"}
        )
        code = "SHARED_CONSUMER_REMOVED"
    elif gate == "public_path":
        raw["edits"] = [
            {
                "component": "mem",
                "kind": "dynamic",
                "sources": [],
                "expression": "mem' = -mem",
            }
        ]
        code = "PUBLIC_PATH_REGRESSION"
    elif gate == "evidence":
        raw["evidence_ids"] = ["validation:I"]
        code = "UNKNOWN_EVIDENCE"
    else:
        raw["graph"] = {}
        code = "REVISION_SCHEMA"
    with pytest.raises(RevisionValidationError) as error:
        apply_revision(
            source,
            raw,
            evidence_ids=["training:I"],
            brief=brief(),
            tolerant_formatting=True,
        )
    assert error.value.code == code
    assert source.model_dump(mode="json") == before


def test_v2_metadata_tolerance_does_not_bypass_schema_bounds_or_item_checks():
    for sources in (["mem"] * 65, ["mem", "illegal-name"], ["mem", 17]):
        raw = reply()
        raw["edits"][0]["sources"] = sources
        with pytest.raises(RevisionValidationError, match="REVISION_SCHEMA"):
            apply_revision(
                problem(), raw, evidence_ids=["training:I"], tolerant_formatting=True
            )


def test_v1_defaults_remain_strict_and_v2_noop_preserves_exact_parent():
    source = problem()
    strict = apply_revision(source, reply(), evidence_ids=["training:I"])
    assert "normalization" not in strict["audit"]
    assert strict == apply_revision(
        source, reply(), evidence_ids=["training:I"], tolerant_formatting=False
    )
    raw = reply()
    raw["edits"][0].update(sources=["a", "mem"], expression="q = a*mem")
    with pytest.raises(RevisionValidationError, match="UNAVAILABLE_SOURCE"):
        apply_revision(source, raw, evidence_ids=["training:I"])
    raw["edits"][0]["sources"] = ["mem"]
    with pytest.raises(RevisionValidationError, match="MODEL_VALIDATION"):
        apply_revision(source, raw, evidence_ids=["training:I"])
    result = apply_revision(
        source, raw, evidence_ids=["training:I"], tolerant_formatting=True
    )
    assert result["status"] == "retained"
    assert result["problem"] == source
    assert result["audit"]["normalization"]["raw_reply"] == raw
