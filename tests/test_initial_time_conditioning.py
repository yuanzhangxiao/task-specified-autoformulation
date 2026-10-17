"""Initial algebraic readouts depend on initializers, not differential RHSs."""

from copy import deepcopy

import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.search import topology_initialization as initial
from autoformalism.search.topology_compatibility import (
    InitialRule,
    TrainingCompatibilityEvidence,
)
from autoformalism.staged_topology import content_hash
from tests.test_topology_compatibility import contract, draft, evidence, trajectory


def fixture(*, z_rule=None, origin=0, shared_process=False):
    """A readout a(z,u), z'=f(i), i'=g(b); only b differs initially."""
    context = ValidationContext(
        targets=("a",), external_inputs=("u",), auxiliaries=("b",)
    )
    packet = evidence(
        trajectory(
            "left",
            (1, 2, 3),
            aux={"b": (3, 3, 3)},
            time=(origin, origin + 1, origin + 2),
        ),
        trajectory(
            "right",
            (2, 4, 6),
            aux={"b": (4, 4, 4)},
            time=(origin, origin + 1, origin + 2),
        ),
        context=context,
        config=contract(
            interpolation={"u": "linear_complete_knots", "b": "linear_complete_knots"},
            absolute_error_bounds={"a": 0, "u": 0, "b": 0},
            initial_rules={
                "z": z_rule or InitialRule(mode="shared_parameter"),
                "i": InitialRule(mode="common_map", arguments=("b",)),
            },
        ),
    )
    processes = (
        [
            {
                "name": "p",
                "depends_on": ["r", "u"],
                "kind": "influence",
                "scientific_meaning": "readout contribution",
                "uses": [{"target": "a", "sign": "positive", "conversion": "1"}],
            }
        ]
        if shared_process
        else []
    )
    model = draft(
        "algebraic",
        () if shared_process else ("z", "u"),
        extras=[("z", "differential", ("i",)), ("i", "differential", ("b",))]
        + ([("r", "algebraic", ("z",))] if shared_process else []),
        processes=processes,
    )
    question = {
        "question_id": "saved-initial-comparison",
        "property": "closed_subsystem",
        "target": "a",
        "witnesses": [
            {
                "trajectories": ["left", "right"],
                "sample_indices": [0, 0],
                "times": [origin, origin],
                "observations": [-100, -100],
            }
        ],
    }
    return model, packet, question


def recheck(model, packet, question):
    return initial.recheck_closed_witnesses(
        model, packet, [question], initial_time_policy=initial.INITIAL_TIME_POLICY
    )


def change_contract(packet, **changes):
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    raw["contract"].update(changes)
    value = TrainingCompatibilityEvidence.model_validate({**raw, "sha256": "0" * 64})
    return value.model_copy(
        update={
            "sha256": content_hash(value.model_dump(mode="json", exclude={"sha256"}))
        }
    )


@pytest.mark.parametrize("shared_process", [False, True])
@pytest.mark.parametrize("origin", [0, 5])
def test_different_rhs_ancestor_cannot_explain_initial_readout(shared_process, origin):
    model, packet, question = fixture(shared_process=shared_process, origin=origin)
    historical = initial.recheck_closed_witnesses(model, packet, [question])
    assert historical.checks[0]["status"] == "conditioning_distinction"
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "preserved"
    assert result.checks[0]["initialized_states"] == ["z"]
    assert result.checks[0]["instantaneous_supplied_sources"] == ["u"]
    assert "i" not in result.checks[0]["conditioning_scope"]
    assert "b" not in result.diagnostics[0].scope
    if shared_process:
        assert result.diagnostics[0].scope == ("a", "p", "r", "u", "z")
    assert result.diagnostics[0].witnesses[0].observations == (1, 2)
    assert result.diagnostics[0].witnesses[0].pair_squared_error_lower_bound == 0.5
    assert initial.recheck_closed_witnesses(model, packet, [question]) == historical


def test_different_declared_initializer_arguments_allow_but_do_not_prove_difference():
    model, packet, question = fixture(
        z_rule=InitialRule(mode="common_map", arguments=("b",))
    )
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "conditioning_distinction"
    assert not result.diagnostics


def test_equal_declared_initializer_arguments_preserve_initial_contrast():
    model, packet, question = fixture(
        z_rule=InitialRule(mode="common_map", arguments=("u",))
    )
    assert recheck(model, packet, question).checks[0]["status"] == "preserved"


def test_unknown_initialization_is_not_assumed_equal():
    model, packet, question = fixture(z_rule=InitialRule(mode="unknown"))
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "unavailable"
    assert "initialization of z is unavailable" in result.checks[0]["reasons"]
    assert not result.diagnostics


@pytest.mark.parametrize("channel", ["u", "b"])
def test_uncertain_initial_reading_or_initializer_argument_defers(channel):
    model, packet, question = fixture(
        z_rule=InitialRule(mode="common_map", arguments=("b",))
    )
    packet = change_contract(
        packet, absolute_error_bounds={"a": 0, "u": 0, "b": 0, channel: 0.1}
    )
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "unavailable"
    assert not result.diagnostics


def test_t0_does_not_require_interpolation_but_later_comparisons_still_do():
    model, packet, question = fixture()
    packet = change_contract(packet, interpolation={})
    assert recheck(model, packet, question).checks[0]["status"] == "preserved"
    later = deepcopy(question)
    later["witnesses"][0].update(sample_indices=[1, 1], times=[1, 1])
    result = recheck(model, packet, later)
    historical = initial.recheck_closed_witnesses(model, packet, [later])
    assert result == historical
    assert result.checks[0]["status"] == "unavailable"


def test_rhs_ancestor_can_distinguish_later_readouts():
    model, packet, question = fixture()
    question["witnesses"][0].update(sample_indices=[1, 1], times=[1, 1])
    result = recheck(model, packet, question)
    assert result == initial.recheck_closed_witnesses(model, packet, [question])
    assert result.checks[0]["status"] == "conditioning_distinction"


def test_uncertain_target_does_not_become_supported_initial_conflict():
    model, packet, question = fixture()
    packet = change_contract(packet, absolute_error_bounds={"a": 1, "u": 0, "b": 0})
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "uncertainty_limited"
    assert not result.diagnostics


@pytest.mark.parametrize(
    "field",
    [
        "dependencies_exhaustive",
        "shared_parameters",
        "open_loop",
        "reset_impulse_free",
        "uniqueness_assumed",
    ],
)
def test_initial_time_policy_preserves_existing_class_qualifications(field):
    model, packet, question = fixture()
    result = recheck(model, change_contract(packet, **{field: None}), question)
    assert result.checks[0]["status"] == "unavailable"
    assert not result.diagnostics


def test_invalid_algebraic_cycle_is_not_hidden_by_initial_state_boundary():
    _, packet, question = fixture()
    model = draft(
        "algebraic",
        ("r",),
        extras=[("r", "algebraic", ("a", "z")), ("z", "differential", ("u",))],
    )
    result = recheck(model, packet, question)
    assert result.checks[0]["status"] == "unavailable"
    assert "algebraic cycle" in result.checks[0]["reasons"]


def test_current_initial_pairs_can_be_discovered_without_historical_question():
    model, packet, _ = fixture()
    before = packet.model_dump(mode="json")
    questions = initial.initial_time_questions(model, packet)
    assert len(questions) == len(questions[0]["witnesses"]) == 1
    result = recheck(model, packet, questions[0])
    assert result.checks[0]["status"] == "preserved"
    assert packet.model_dump(mode="json") == before
    assert initial.initial_time_questions(model, packet) == questions


def test_initial_pair_discovery_skips_differential_or_pure_algebraic_targets():
    _, packet, _ = fixture()
    assert initial.initial_time_questions(draft(), packet) == []
    assert initial.initial_time_questions(draft("algebraic", ("u",)), packet) == []


def test_unknown_initial_time_policy_is_rejected():
    model, packet, question = fixture()
    with pytest.raises(ValueError, match="unknown initial-time"):
        initial.recheck_closed_witnesses(
            model, packet, [question], initial_time_policy="unreviewed"
        )
