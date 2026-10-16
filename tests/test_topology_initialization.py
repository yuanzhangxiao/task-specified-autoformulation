"""Explicit initialization dependencies, conservative family tests and stale records."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.expressions import ValidationContext
from autoformalism.search.topology_compatibility import (
    InitialRule,
    TrainingCompatibilityEvidence,
)
from autoformalism.search.topology_initialization import (
    InitialInformationContract,
    InitializerDependencyHandoff,
    InitializerDependencyRecord,
    derived_evidence,
    inspect,
    recheck_closed_witnesses,
    request,
)
from autoformalism.staged_topology import content_hash
from tests.test_topology_compatibility import contract, draft, evidence, trajectory


def information(**changes):
    return InitialInformationContract(
        **(
            {
                "provenance": "explicit synthetic recorded-t0 prediction contract",
                "scope": "recorded_t0",
                "shared_deterministic": True,
                "complete": True,
                "permitted_symbols": ("a", "u", "t"),
            }
            | changes
        )
    )


def fixture(*, extra_initial=False, extra_forcing=False, noisy=False):
    context = ValidationContext(
        targets=("a",), external_inputs=("u",), auxiliaries=("b",)
    )
    packet = evidence(
        trajectory("first", (0, 1, 0), aux={"b": (0, 0, 0)}),
        trajectory(
            "second",
            (0, 2, 0),
            (0, 1, 0) if extra_forcing else (0, 0, 0),
            aux={"b": (1, 0, 0) if extra_initial else (0, 0, 0)},
        ),
        config=contract(
            absolute_error_bounds={"a": 0, "b": 0.1 if noisy else 0, "u": 0}
        ),
        context=context,
    )
    model = draft(sources=("z", "u"), extras=[("z", "differential", ())])
    return model, packet, information(permitted_symbols=("a", "b", "u", "t"))


def choice(mode="common_map", arguments=("a",), state="z"):
    return InitializerDependencyHandoff(
        records=(
            InitializerDependencyRecord(
                state=state, rule=InitialRule(mode=mode, arguments=arguments)
            ),
        )
    )


def closed(report):
    return next(
        row for row in report["diagnostics"] if row["property"] == "closed_subsystem"
    )


def test_unknown_latent_preserves_witness_for_explicit_maximal_initial_family():
    model, packet, info = fixture()
    original = copy.deepcopy(packet.model_dump(mode="json"))
    result = inspect(model, packet, info)
    assert result["unresolved_states"] == ["z"]
    assert closed(result["declared_report"])["status"] == "deferred"
    assert closed(result["family_qualified_report"])["status"] == "conditional_conflict"
    assert result["handoff"]["records"] == []
    assert not result["executable_initializers_selected"]
    assert result["original_evidence_sha256"] == packet.sha256
    assert (
        len(
            {
                packet.sha256,
                result["declared_evidence_sha256"],
                result["family_evidence_sha256"],
            }
        )
        == 3
    )
    assert packet.model_dump(mode="json") == original
    assert inspect(model, packet, info) == result


@pytest.mark.parametrize(
    "mode,args", [("shared_parameter", ()), ("common_map", ("a", "b"))]
)
def test_explicit_declarations_are_diagnosed_without_selecting_functions(mode, args):
    model, packet, info = fixture()
    handoff = choice(mode, args)
    result = inspect(model, packet, info, handoff)
    assert result["unresolved_states"] == []
    assert closed(result["declared_report"])["status"] == "conditional_conflict"
    assert result["handoff"] == handoff.model_dump(mode="json")
    derived = derived_evidence(model, packet, info, handoff)
    assert derived.contract.initial_rules["z"] == handoff.records[0].rule
    assert packet.sha256 in derived.contract.provenance


@pytest.mark.parametrize("case", ["extra_initial", "extra_forcing", "noisy"])
def test_maximal_family_does_not_invent_initial_or_forcing_equality(case):
    model, packet, info = fixture(**{case: True})
    result = inspect(model, packet, info)
    row = closed(result["family_qualified_report"])
    assert row["status"] != "conditional_conflict"
    assert row["witnesses"] == []
    if case == "noisy":
        assert row["status"] == "deferred"
        assert "exact initial argument b is not established" in row["unavailable"]


@pytest.mark.parametrize(
    "changes",
    [
        {"scope": "unknown"},
        {"scope": "history_or_unrecorded"},
        {"shared_deterministic": None},
        {"complete": None},
        {"complete": False},
    ],
)
def test_unqualified_runtime_initial_information_abstains(changes):
    model, packet, _ = fixture()
    info = information(**changes)
    result = inspect(model, packet, info)
    assert result["family_qualified_report"] is None
    assert result["family_qualification"]["unavailable"]
    with pytest.raises(ValueError, match="family qualification unavailable"):
        derived_evidence(model, packet, info, family=True)


def test_proposer_declaration_cannot_assert_unknown_runtime_scope():
    model, packet, _ = fixture()
    result = inspect(model, packet, information(scope="unknown"), choice())
    assert result["unresolved_states"] == ["z"]
    assert result["declaration_unavailable"]
    assert closed(result["declared_report"])["status"] == "deferred"


@pytest.mark.parametrize("symbol", ["future", "trajectory_id", "z", "a_0", "hidden"])
def test_unavailable_arguments_and_runtime_symbols_are_rejected(symbol):
    model, packet, info = fixture()
    with pytest.raises(ValueError, match="unavailable initializer arguments"):
        inspect(model, packet, info, choice(arguments=(symbol,)))
    with pytest.raises(ValueError, match="unavailable initial information"):
        inspect(model, packet, information(permitted_symbols=(symbol,)))


@pytest.mark.parametrize("symbol", ["a[1]", "a(t+1)"])
def test_nonidentifier_future_syntax_is_rejected(symbol):
    with pytest.raises(ValueError):
        choice(arguments=(symbol,))


def test_duplicate_records_arguments_and_unused_arguments_are_rejected():
    with pytest.raises(ValueError, match="distinct states"):
        InitializerDependencyHandoff(records=choice().records * 2)
    with pytest.raises(ValueError, match="arguments must be distinct"):
        choice(arguments=("a", "a"))
    with pytest.raises(ValueError, match="requires explicit"):
        choice(arguments=())
    with pytest.raises(ValueError, match="only common_map"):
        choice(mode="shared_parameter")
    with pytest.raises(ValueError, match="symbols must be distinct"):
        information(permitted_symbols=("a", "a"))


def test_observed_rules_are_fixed_and_unknown_runtime_rules_can_be_filled():
    model, packet, info = fixture()
    with pytest.raises(ValueError, match="runtime initialization is fixed"):
        inspect(model, packet, info, choice("shared_parameter", (), "a"))
    with pytest.raises(ValueError, match="frozen runtime rule"):
        inspect(model, packet, info, choice("observed_initial", (), "z"))
    retained = choice("observed_initial", (), "a")
    assert (
        request(model, packet, info, retained)["fixed_runtime_rules"]["a"]["mode"]
        == "observed_initial"
    )
    unknown = evidence(
        trajectory("first"),
        trajectory("second", (0, 2, 0)),
        config=contract(
            initial_rules={
                "a": InitialRule(mode="observed_initial"),
                "z": InitialRule(mode="unknown"),
            }
        ),
    )
    assert inspect(model, unknown, information(), choice())["unresolved_states"] == []


@pytest.mark.parametrize(
    "changed", [draft(), draft(extras=[("z", "algebraic", ("u",))])]
)
def test_durable_records_fail_after_state_removal_or_type_change(changed):
    _, packet, info = fixture()
    with pytest.raises(ValueError, match="stale initializer record"):
        inspect(changed, packet, info, choice())


def test_unknown_record_remains_pending_and_request_is_readable():
    model, packet, info = fixture()
    handoff = choice("unknown", ())
    payload = request(model, packet, info, handoff)
    assert payload["unresolved_states"] == ["z"]
    assert payload["retained_handoff"] == handoff.model_dump(mode="json")
    assert "t0" in payload["instruction"]
    assert "InitialInformationContract" not in payload["response_schema"].get(
        "$defs", {}
    )
    result = inspect(model, packet, info, handoff, qualify_family=False)
    assert result["family_qualification"]["status"] == "off"


def test_runtime_no_initial_features_has_only_shared_constant_family():
    model, packet, _ = fixture()
    result = inspect(model, packet, information(permitted_symbols=()))
    assert closed(result["family_qualified_report"])["status"] == "conditional_conflict"
    derived = derived_evidence(
        model, packet, information(permitted_symbols=()), family=True
    )
    assert derived.contract.initial_rules["z"].mode == "shared_parameter"


def test_unknown_dynamics_sharing_still_defers_family_diagnostic():
    model, _, _ = fixture()
    packet = evidence(
        trajectory("first"),
        trajectory("second", (0, 2, 0)),
        config=contract(shared_parameters=None),
    )
    result = inspect(model, packet, information())
    assert closed(result["family_qualified_report"])["status"] == "deferred"


def test_forged_evidence_and_model_copy_records_are_revalidated():
    model, packet, info = fixture()
    with pytest.raises(ValueError, match="digest differs"):
        inspect(model, packet.model_copy(update={"sha256": "0" * 64}), info)
    broken = choice().model_copy(update={"records": choice().records * 2})
    with pytest.raises(ValueError, match="distinct states"):
        inspect(model, packet, info, broken)


def prefix_fixture(*, early_forcing=False):
    """A genuine old discrepancy before a newly admitted forcing diverges."""
    context = ValidationContext(targets=("a",), external_inputs=("u", "v"))
    rows = [
        trajectory("left", (0, 1, 1, 1), (0, 0, 0, 0)),
        trajectory("right", (0, 2, 1, 1), (0, 0, 0, 0)),
    ]
    rows = [
        replace(
            row,
            external_inputs={
                **row.external_inputs,
                "v": np.asarray((0, 1, 1, 1) if early_forcing else (0, 0, 0, 1), float)
                * index,
            },
        )
        for index, row in enumerate(rows)
    ]
    packet = evidence(
        *rows,
        context=context,
        config=contract(
            interpolation={"u": "linear_complete_knots", "v": "linear_complete_knots"},
            absolute_error_bounds={"a": 0, "u": 0, "v": 0},
        ),
    )
    question = {
        "question_id": "saved-pair",
        "property": "closed_subsystem",
        "target": "a",
        "witnesses": [
            {
                "trajectories": ["left", "right"],
                "sample_indices": [1, 1],
                "times": [1, 1],
                "observations": [-100, -100],
            }
        ],
    }
    return draft(sources=("u", "v")), packet, question


def test_old_witness_uses_prefix_and_recomputes_untrusted_observations():
    model, packet, question = prefix_fixture()
    result = recheck_closed_witnesses(model, packet, [question])
    assert result.checks[0]["status"] == "preserved"
    assert result.diagnostics[0].witnesses[0].observations == (1, 2)
    assert result.checks[0]["conditioning_end_time"] == 1
    assert result.evidence_sha256 == packet.sha256


def test_new_forcing_distinction_before_witness_does_not_preserve_equality():
    model, packet, question = prefix_fixture(early_forcing=True)
    result = recheck_closed_witnesses(model, packet, [question])
    assert result.diagnostics == ()
    assert result.checks[0]["status"] == "conditioning_distinction"


@pytest.mark.parametrize(
    "change",
    [
        {"sample_indices": [-1, 1]},
        {"times": [2, 2]},
        {"trajectories": ["missing", "right"]},
        {"observations": [1]},
    ],
)
def test_invalid_old_witness_coordinates_never_establish_a_conflict(change):
    model, packet, question = prefix_fixture()
    question["witnesses"][0].update(change)
    result = recheck_closed_witnesses(model, packet, [question])
    assert not result.diagnostics
    assert result.checks[0]["status"] == "invalid_saved_witness"


def test_unknown_latent_or_broken_graph_abstains_in_prefix_recheck():
    _, packet, question = prefix_fixture()
    for model in [
        draft(sources=("z",), extras=[("z", "differential", ())]),
        draft(sources=("unknown",)),
    ]:
        result = recheck_closed_witnesses(model, packet, [question])
        assert not result.diagnostics
        assert result.checks[0]["status"] == "unavailable"


def test_family_qualified_prefix_recheck_does_not_create_a_declared_initializer():
    _, packet, question = prefix_fixture()
    model = draft(sources=("z", "u", "v"), extras=[("z", "differential", ())])
    info = information(permitted_symbols=("a", "u", "v", "t"))
    derived = derived_evidence(model, packet, info, family=True)
    result = recheck_closed_witnesses(model, derived, [question, question])
    assert len(result.diagnostics) == len(result.diagnostics[0].witnesses) == 1
    assert "z" not in packet.contract.initial_rules


def test_recheck_unknown_model_class_assumption_is_unavailable():
    model, packet, question = prefix_fixture()
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["uniqueness_assumed"] = None
    changed = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    result = recheck_closed_witnesses(model, changed, [question])
    assert not result.diagnostics
    assert result.checks[0]["status"] == "unavailable"


@pytest.mark.parametrize("channel", ["a", "v"])
def test_uncertain_initial_or_forcing_values_prevent_preserving_equality(channel):
    model, packet, question = prefix_fixture()
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["absolute_error_bounds"][channel] = 0.1
    changed = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    result = recheck_closed_witnesses(model, changed, [question])
    assert not result.diagnostics
    assert result.checks[0]["status"] == "unavailable"


def test_uncertain_output_is_not_promoted_to_supported_preserved_witness():
    _, packet, question = prefix_fixture()
    model = draft("algebraic", ("z",), extras=[("z", "differential", ("u", "v"))])
    raw = packet.model_dump(mode="json", exclude={"sha256"})
    raw["contract"]["absolute_error_bounds"]["a"] = 1.0
    raw["contract"]["initial_rules"]["z"] = {
        "mode": "shared_parameter",
        "arguments": [],
    }
    changed = TrainingCompatibilityEvidence.model_validate(
        {**raw, "sha256": content_hash(raw)}
    )
    result = recheck_closed_witnesses(model, changed, [question])
    assert not result.diagnostics
    assert result.checks[0]["status"] == "uncertainty_limited"
