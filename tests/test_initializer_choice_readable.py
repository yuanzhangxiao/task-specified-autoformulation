"""Readable decisions retain qualified evidence and add only permitted t0 data."""

from copy import deepcopy

import pytest

from autoformalism.research import initializer_choice_prompt as choice
from autoformalism.research import initializer_choice_readable as readable
from autoformalism.search.construction_readable import model_view
from tests.test_initialization_evidence_feedback import feedback, fixture, sealed


def scenario(*, initial=True):
    """Build a real structured choice with neutral synthetic model quantities."""
    full, draft, evidence = fixture(initial=initial)
    full = feedback.project(full, draft, evidence)
    for question in full["compatibility_questions"]:
        question["assumptions"] = [readable._PROPAGATION_ASSUMPTION]
    for card in full["model_data_conflicts"]:
        card["assumptions"] = [readable._PROPAGATION_ASSUMPTION]
    full["model_view"] = model_view(draft)
    full["public_brief"].update(
        scientific_context="Shared causal laws generate all targets. Preserve memory.",
        requirements=[{"id": "memory", "public_requirement": "Preserve memory."}],
        target_dependencies=[],
    )
    full["initialization"]["runtime_initial_information"].update(
        complete=True, shared_deterministic=True
    )
    return choice.build_packet(full, "z"), evidence


def reseal(packet, evidence, change):
    """Maintain explicit source identity while testing changed recorded evidence."""
    raw = evidence.model_dump(mode="json")
    change(raw)
    source = sealed(raw)
    packet["initialization"]["original_evidence_sha256"] = source.sha256
    for card in packet["model_data_conflicts"]:
        if "paired_recorded_values" in card:
            card["paired_recorded_values"]["evidence_sha256"] = source.sha256
    return source


@pytest.mark.parametrize("initial", [True, False])
def test_full_model_context_initial_choices_and_witness_are_shown_once(initial):
    packet, evidence = scenario(initial=initial)
    before = deepcopy(packet)
    source = evidence.model_dump(mode="json")
    text = readable.render(packet, evidence)
    card = packet["model_data_conflicts"][0]
    assert text.count(card["model_relation"]) == 1
    assert text.count(card["equality_argument"]) == 1
    assert "compatibility_questions" not in text
    assert "CURRENT EDITABLE INITIALIZER" in text
    assert "FIXED RUNTIME INITIALIZERS (cannot be edited)" in text
    assert "Null means NO CHANGE" in text
    assert (
        "numeric value unavailable" in text
        if initial
        else "initial value unavailable" in text
    )
    assert "Shared causal laws generate all targets. Preserve memory." in text
    for row in packet["model_view"]["balances"]:
        assert text.count(row["topology"]) == 1
    assert "| y | target | 1.2 | 1.9 | 0.0 |" in text
    assert "| x | target | 3.0 | 3.0 | 0.0 |" in text
    assert "| u | auxiliary | 7.0 | 7.0 | 0.0 |" in text
    assert "| drive | external_input | 0.0 | 0.0 | 0.0 |" in text
    assert "| t | time | 0.0 | 0.0 | not an observation channel |" in text
    assert packet == before
    assert evidence.model_dump(mode="json") == source
    assert text == readable.render(packet, source)


def test_initial_readout_does_not_require_ode_uniqueness_or_display_future_data():
    packet, evidence = scenario()

    def change(raw):
        for row in raw["trajectories"]:
            row["targets"]["y"][8] = 987654.0
            row["inputs"]["drive"][8] = 876543.0
            row["auxiliaries"]["u"][8] = 765432.0

    evidence = reseal(packet, evidence, change)
    text = readable.render(packet, evidence)
    assert "no trajectory is propagated" in text
    assert "unique solutions" not in text
    assert packet["model_data_conflicts"][0]["assumptions"][0] not in text
    assert "987654" not in text and "876543" not in text and "765432" not in text
    assert "sample index 8" not in text
    assert "train_a: y(0.0)=1.2 (sample index 0)" in text


def test_later_subsystem_explains_vector_and_checks_all_forcing_knots():
    packet, evidence = scenario(initial=False)
    text = readable.render(packet, evidence)
    assert "F is vector-valued" in text
    assert "one potentially different component law" in text
    assert "Saved assumptions for this propagation" in text
    assert "train_a: x(8.0)=1.2 (sample index 8)" in text
    assert "z(t0) = h_z(x(t0)) [common_map]" in text
    assert "drive: linear_complete_knots" in text
    assert "| x | target | 3.0 | 3.0 |" in text
    # An undisplayed interior forcing knot must still invalidate the saved match.
    evidence = reseal(
        packet,
        evidence,
        lambda raw: raw["trajectories"][1]["inputs"]["drive"].__setitem__(5, 999),
    )
    with pytest.raises(ValueError, match="forcing match"):
        readable.render(packet, evidence)


def test_fixed_covariates_clock_and_unavailable_uncertainty_are_explicit():
    packet, evidence = scenario()
    packet["public_brief"]["public_variables"].append(
        {"name": "mass", "data_role": "covariate", "meaning": "Known fixed quantity"}
    )
    packet["initialization"]["runtime_initial_information"]["permitted_symbols"].append(
        "mass"
    )

    def change(raw):
        raw["context"]["fixed_covariates"] = ["mass"]
        raw["trajectories"][0]["covariates"] = {"mass": 4.0}
        raw["trajectories"][1]["covariates"] = {"mass": 6.0}

    evidence = reseal(packet, evidence, change)
    text = readable.render(packet, evidence)
    assert "| mass | covariate | 4.0 | 6.0 | unavailable / unspecified |" in text
    assert "Known fixed quantity" in text


@pytest.mark.parametrize(
    "change",
    [
        "seal",
        "packet_hash",
        "split",
        "missing_channel",
        "unknown_id",
        "witness_time",
        "witness_index",
        "witness_value",
        "numeric_card",
        "unknown_permitted",
        "duplicate_permitted",
        "context",
        "fixed_state",
        "ambiguous_state",
        "missing_equations",
        "missing_card",
        "reference",
        "unqualified",
        "missing_context",
        "missing_assumptions",
        "overlapping_errors",
        "unknown_rule",
        "rule_arguments",
        "missing_archive",
        "extra_packet_key",
    ],
)
def test_malformed_or_mismatched_sources_fail_closed(change):
    packet, evidence = scenario()
    witness = packet["compatibility_questions"][0]["witnesses"][0]
    runtime = packet["initialization"]["runtime_initial_information"]
    if change == "seal":
        evidence = evidence.model_copy(update={"sha256": "0" * 64})
    elif change == "packet_hash":
        packet["initialization"]["original_evidence_sha256"] = "0" * 64
    elif change == "split":
        evidence = evidence.model_dump(mode="json") | {"split": "test"}
    elif change == "missing_channel":
        evidence = reseal(
            packet, evidence, lambda raw: raw["trajectories"][0]["targets"].pop("y")
        )
    elif change == "unknown_id":
        witness["trajectories"][0] = "absent"
    elif change == "witness_time":
        witness["times"] = [1.0, 1.0]
    elif change == "witness_index":
        witness["sample_indices"] = [10, 10]
    elif change == "witness_value":
        witness["observations"][0] = 99.0
    elif change == "numeric_card":
        packet["model_data_conflicts"][0]["paired_recorded_values"]["observed_output"][
            0
        ]["value"] = 99.0
    elif change == "unknown_permitted":
        runtime["permitted_symbols"].append("hidden")
    elif change == "duplicate_permitted":
        runtime["permitted_symbols"].append("x")
    elif change == "context":
        packet["public_brief"]["public_variables"][0]["data_role"] = "auxiliary"
    elif change == "fixed_state":
        packet["initialization"]["fixed_runtime_rules"]["z"] = {
            "mode": "observed_initial",
            "arguments": [],
        }
    elif change == "ambiguous_state":
        packet["initialization"]["editable_states"].append("x")
    elif change == "missing_equations":
        packet["model_view"].pop("balances")
    elif change == "missing_card":
        packet["model_data_conflicts"] = []
    elif change == "reference":
        packet["model_data_conflicts"][0]["evidence_reference"] = "unknown"
    elif change == "unqualified":
        packet["compatibility_questions"][0]["initialization_qualification"][
            "family_available"
        ] = False
    elif change == "missing_context":
        packet["public_brief"]["scientific_context"] = ""
    elif change == "missing_assumptions":
        packet["model_data_conflicts"][0]["assumptions"] = []
    elif change == "overlapping_errors":
        evidence = reseal(
            packet,
            evidence,
            lambda raw: raw["contract"]["absolute_error_bounds"].update(y=100.0),
        )
    elif change == "unknown_rule":
        packet["initialization"]["retained_handoff"]["records"][0]["rule"]["mode"] = (
            "unknown"
        )
    elif change == "rule_arguments":
        packet["initialization"]["retained_handoff"]["records"][0]["rule"][
            "arguments"
        ] = ["x"]
    elif change == "missing_archive":
        packet["projection_scope"]["full_source_retained_by_runner"] = False
    else:
        packet["additional_instructions"] = "Unsupported data"
    with pytest.raises(ValueError):
        readable.render(packet, evidence)


def test_previous_null_does_not_delete_retained_rule_and_feedback_is_bounded():
    packet, evidence = scenario()
    packet["previous_feedback"] = {
        "status": "assessed",
        "reply": {
            "decision": "insufficient_information",
            "rule": None,
            "explanation": "x" * 4000,
        },
        "error": None,
    }
    text = readable.render(packet, evidence)
    assert "z(t0) = one shared unknown constant" in text
    assert "Null means NO CHANGE" in text
    assert "complete receipt retained externally" in text
    assert "x" * 1601 not in text
    assert packet["previous_feedback"]["reply"]["explanation"] == "x" * 4000


def test_complete_scientific_context_is_never_silently_truncated():
    packet, evidence = scenario()
    packet["public_brief"]["scientific_context"] = "scientific evidence " * 2000
    assert packet["public_brief"]["scientific_context"] in readable.render(
        packet, evidence
    )


def test_initial_readout_preserves_other_assumptions_without_propagation_premise():
    packet, evidence = scenario()
    premise = "Exact reproduction is conditional on the displayed error bounds."
    packet["compatibility_questions"][0]["assumptions"].append(premise)
    packet["model_data_conflicts"][0]["assumptions"].append(premise)
    text = readable.render(packet, evidence)
    assert text.count(premise) == 1
    assert "unique solutions" not in text


@pytest.mark.parametrize(
    "field", ["initial_equalities", "pair_squared_error_lower_bound"]
)
def test_saved_derived_numbers_and_initial_equalities_cannot_disagree(field):
    packet, evidence = scenario()
    if field == "initial_equalities":
        packet["model_data_conflicts"][0][field] = ["z(t0) = measured z(t0)"]
    else:
        packet["compatibility_questions"][0]["witnesses"][0][field] = 999.0
    with pytest.raises(ValueError):
        readable.render(packet, evidence)


def test_all_qualified_witnesses_survive_with_one_initial_table_for_the_pair():
    packet, evidence = scenario(initial=False)

    def change(raw):
        raw["trajectories"][0]["targets"]["x"][7] = 1.5
        raw["trajectories"][1]["targets"]["x"][7] = 2.5

    evidence = reseal(packet, evidence, change)
    packet["compatibility_questions"][0]["witnesses"].append(
        {
            "trajectories": ["train_a", "train_b"],
            "sample_indices": [7, 7],
            "times": [7.0, 7.0],
            "observations": [1.5, 2.5],
            "error_intervals_disjoint": True,
        }
    )
    text = readable.render(packet, evidence)
    assert text.count("Observed training witness:") == 2
    assert "x(7.0)=1.5" in text
    assert "x(8.0)=1.2" in text
    assert text.count("| Permitted quantity |") == 1
