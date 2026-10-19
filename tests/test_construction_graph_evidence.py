"""Conditional DAG equality, scoped prefix deferrals and durable saved evidence."""

import copy

import pytest

from autoformalism.fitting.prefix_initialization import PrefixPolicy
from autoformalism.search.construction_graph_evidence import (
    PREFIX_REASON,
    inspect_graph_initialization,
)
from autoformalism.search.construction_initialization_graph import (
    GraphDependencyHandoff,
    build_contract,
)
from autoformalism.search.topology_compatibility import TrainingCompatibilityEvidence
from autoformalism.staged_topology import content_hash
from tests.test_topology_compatibility import draft, evidence, trajectory
from tests.test_topology_initialization import prefix_fixture


def graph_fixture(*, scope="shared_training", different_input=False, window=True):
    """A generic readout whose initial DAG has two transitive latent maps."""
    model = draft(
        "algebraic",
        ("z2",),
        extras=[(name, "differential", ()) for name in ("z0", "z1", "z2")],
    )
    packet = evidence(
        trajectory("left", (1, 2, 3)),
        trajectory("right", (2, 3, 4), (1, 1, 1) if different_input else (0, 0, 0)),
    )
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        trajectory_initialization="observation_prefix"
        if scope == "trajectory"
        else "forbidden",
        prefix_policy=PrefixPolicy(duration=1)
        if scope == "trajectory" and window
        else None,
        provenance="Synthetic public initialization boundary.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z0",
                    "rule": {"mode": "value", "scope": scope},
                    "justification": "Independent latent preparation.",
                },
                {
                    "state": "z1",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["z0"],
                        "public_initials": ["u"],
                    },
                    "justification": "Shared map of preparation and initial input.",
                },
                {
                    "state": "z2",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["z1"],
                        "public_initials": [],
                    },
                    "justification": "Dependent latent response.",
                },
            ]
        }
    )
    return model, packet, runtime, handoff


def diagnostic(result, property_="closed_subsystem"):
    return next(
        row
        for row in result["declared_report"]["diagnostics"]
        if row["property"] == property_
    )


def saved_question(index=0):
    return {
        "question_id": "historical-matched-pair",
        "property": "closed_subsystem",
        "target": "a",
        "status": "conditional_conflict",
        "reason": "Historical shared-initialization qualification.",
        "witnesses": [
            {
                "trajectories": ["left", "right"],
                "sample_indices": [index, index],
                "times": [index, index],
                "observations": [-100, -100],
            }
        ],
    }


def test_shared_root_maps_use_transitive_public_inputs_without_formulas():
    model, packet, runtime, handoff = graph_fixture()
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    dependencies = result["initialization_dependency_summary"]["z2"]
    assert dependencies["public_initials"] == ["u"]
    assert dependencies["shared_training_roots"] == ["z0"]
    assert dependencies["trajectory_roots"] == []
    assert diagnostic(result)["status"] == "conditional_conflict"
    assert (
        result["initial_time_report"]["diagnostics"][0]["status"]
        == "conditional_conflict"
    )
    assert result["qualification"] == "structural_only"
    assert not result["executable_initializers_selected"]
    assert not result["scientific_pass"]


def test_different_transitive_map_inputs_do_not_assert_equal_outputs():
    model, packet, runtime, handoff = graph_fixture(different_input=True)
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert diagnostic(result)["status"] != "conditional_conflict"
    assert not result["initial_time_report"]["diagnostics"]
    check = result["initial_time_report"]["checks"][0]
    assert check["status"] == "conditioning_distinction"
    assert not check["scientific_repair_established"]


@pytest.mark.parametrize("window", (True, False))
def test_transitive_prefix_root_defers_t0_and_dynamic_equality(window):
    model, packet, runtime, handoff = graph_fixture(scope="trajectory", window=window)
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert diagnostic(result)["status"] == "deferred"
    assert PREFIX_REASON in diagnostic(result)["unavailable"]
    assert diagnostic(result)["prefix_inferred_roots"] == ["z0"]
    assert not result["initial_time_report"]["diagnostics"]
    check = result["initial_time_report"]["checks"][0]
    assert check["status"] == "deferred" and check["retained_unresolved"]
    assert check["reason"] == PREFIX_REASON
    if window:
        assert result["prefix_policy"]["duration"] == 1
    else:
        assert result["prefix_policy"] is None
        assert "observation-window policy unresolved" in check["reasons"]
    assert not result["scientific_pass"]


def test_equal_observed_prefixes_do_not_prove_equal_inferred_roots():
    model, _, runtime, handoff = graph_fixture(scope="trajectory")
    packet = evidence(trajectory("left", (0, 0, 1)), trajectory("right", (0, 0, 2)))
    result = inspect_graph_initialization(
        model, packet, runtime, handoff, previous_questions=[saved_question(2)]
    )
    check = result["retained_witness_report"]["checks"][0]
    assert check["status"] == "deferred"
    assert check["recomputed_witness"]["observations"] == [1, 2]
    assert (
        "Equal prefixes do not establish root equality"
        in result["qualification_reason"]
    )


@pytest.mark.parametrize("scope", ("shared_training", "trajectory"))
def test_saved_status_reason_and_witness_remain_intact_and_values_are_recomputed(scope):
    model, packet, runtime, handoff = graph_fixture(scope=scope)
    question = saved_question()
    original = copy.deepcopy(question)
    result = inspect_graph_initialization(
        model, packet, runtime, handoff, previous_questions=[question]
    )
    report = result["retained_witness_report"]
    assert report["original_questions"] == [original]
    assert question == original
    check = report["checks"][0]
    assert check["original_witness"] == original["witnesses"][0]
    assert check["recomputed_witness"]["observations"] == [1, 2]
    assert check["status"] == (
        "preserved" if scope == "shared_training" else "deferred"
    )
    assert not check["scientific_repair_established"]


def test_invalid_saved_coordinates_are_retained_unresolved():
    model, packet, runtime, handoff = graph_fixture(scope="trajectory")
    question = saved_question(100)
    result = inspect_graph_initialization(
        model, packet, runtime, handoff, previous_questions=[question]
    )
    check = result["retained_witness_report"]["checks"][0]
    assert check["status"] == "invalid_saved_witness"
    assert check["retained_unresolved"]
    assert "recomputed_witness" not in check


def test_saved_unrelated_question_is_retained_without_claiming_repair():
    model, packet, runtime, handoff = graph_fixture()
    question = {**saved_question(), "property": "scalar_autonomous"}
    result = inspect_graph_initialization(
        model, packet, runtime, handoff, previous_questions=[question]
    )
    assert result["retained_witness_report"]["original_questions"] == [question]
    assert (
        result["retained_witness_report"]["checks"][0]["status"]
        == "retained_unresolved"
    )


@pytest.mark.parametrize(
    "kind,property_",
    (("differential", "scalar_autonomous"), ("algebraic", "algebraic_map")),
)
def test_unrelated_prefix_root_does_not_clear_independent_topology_conflict(
    kind, property_
):
    model = draft(kind, ("u",), extras=[("z", "differential", ())])
    packet = evidence(trajectory("left", (0, 1, 0)))
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        observed_initial_channels={"a": "a"} if kind == "differential" else {},
        trajectory_initialization="observation_prefix",
        provenance="Synthetic independent latent preparation.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z",
                    "rule": {"mode": "value", "scope": "trajectory"},
                    "justification": "Independent latent preparation.",
                }
            ]
        }
    )
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert diagnostic(result, property_)["status"] == "conditional_conflict"
    assert PREFIX_REASON not in diagnostic(result)["unavailable"]


def test_t0_readout_stops_at_states_without_following_prefix_dependent_rhs():
    model, packet, runtime, handoff = graph_fixture(scope="trajectory")
    model = draft(
        "algebraic",
        ("z2",),
        extras=[
            ("z0", "differential", ()),
            ("z1", "differential", ()),
            ("z2", "differential", ("z0",)),
        ],
    )
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        trajectory_initialization="observation_prefix",
        provenance="Synthetic boundary.",
    )
    raw = handoff.model_dump(mode="json")
    raw["records"][2]["rule"] = {"mode": "value", "scope": "shared_training"}
    handoff = GraphDependencyHandoff.model_validate(raw)
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert diagnostic(result)["status"] == "deferred"
    assert (
        result["initial_time_report"]["diagnostics"][0]["status"]
        == "conditional_conflict"
    )
    assert result["initial_time_report"]["checks"][0]["status"] == "preserved"


@pytest.mark.parametrize("uncertain_input", (True, False))
def test_uncertain_public_map_input_or_unknown_dynamics_remain_unavailable(
    uncertain_input,
):
    model, packet, runtime, handoff = graph_fixture()
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    if uncertain_input:
        raw["contract"]["absolute_error_bounds"]["u"] = 0.1
    else:
        raw["contract"]["uniqueness_assumed"] = None
    packet = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert diagnostic(result)["status"] == "deferred"
    assert not result["initial_time_report"]["diagnostics"]
    assert result["initial_time_report"]["checks"][0]["retained_unresolved"]


@pytest.mark.parametrize("early_forcing", (True, False))
def test_saved_pair_rechecks_only_forcing_through_the_observation(early_forcing):
    model, packet, question = prefix_fixture(early_forcing=early_forcing)
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "v", "t"),
        observed_initial_channels={"a": "a"},
        provenance="Synthetic matched forcing prefix.",
    )
    result = inspect_graph_initialization(
        model, packet, runtime, GraphDependencyHandoff(), [question]
    )
    check = result["retained_witness_report"]["checks"][0]
    assert check["status"] == (
        "conditioning_distinction" if early_forcing else "preserved"
    )
    assert check["recomputed_witness"]["observations"] == [1, 2]
    assert check["conditioning_end_time"] == 1
    assert result["retained_witness_report"]["original_questions"] == [question]


def test_inspection_preserves_sources_and_is_deterministic_without_fitting():
    model, packet, runtime, handoff = graph_fixture()
    originals = [value.model_dump_json() for value in (model, packet, runtime, handoff)]
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert result == inspect_graph_initialization(model, packet, runtime, handoff)
    assert originals == [
        value.model_dump_json() for value in (model, packet, runtime, handoff)
    ]
    assert result["original_evidence_sha256"] == packet.sha256
    assert result["derived_evidence_sha256"] != packet.sha256
    assert packet.sha256 in result["derived_evidence_provenance"]
    assert result["optimizer_calls"] == result["llm_calls"] == 0
    assert not result["test_data_opened"]


def test_forged_source_seal_is_rejected():
    model, packet, runtime, handoff = graph_fixture()
    with pytest.raises(ValueError, match="digest"):
        inspect_graph_initialization(
            model, packet.model_copy(update={"sha256": "0" * 64}), runtime, handoff
        )


def test_existing_observed_state_cannot_be_freed_by_new_graph_contract():
    model = draft()
    packet = evidence(trajectory("left"))
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        provenance="Synthetic boundary cannot contradict fixed observed identity.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "a",
                    "rule": {"mode": "value", "scope": "shared_training"},
                    "justification": "Invalid free observed initial value.",
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="observed"):
        inspect_graph_initialization(model, packet, runtime, handoff)


def test_observed_binding_can_be_used_transitively_without_copying_hidden_truth():
    model = draft(sources=("z",), extras=[("z", "differential", ())])
    packet = evidence(trajectory("left", (0, 1, 2)), trajectory("right", (0, 2, 3)))
    runtime = build_contract(
        model,
        packet.context,
        permitted_public_initials=("a", "u", "t"),
        observed_initial_channels={"a": "a"},
        provenance="Synthetic measured identity.",
    )
    handoff = GraphDependencyHandoff.model_validate(
        {
            "records": [
                {
                    "state": "z",
                    "rule": {
                        "mode": "map",
                        "state_initials": ["a"],
                        "public_initials": [],
                    },
                    "justification": "Shared map of measured initial state.",
                }
            ]
        }
    )
    result = inspect_graph_initialization(model, packet, runtime, handoff)
    assert result["initialization_dependency_summary"]["z"]["public_initials"] == ["a"]
    assert diagnostic(result)["status"] == "conditional_conflict"
