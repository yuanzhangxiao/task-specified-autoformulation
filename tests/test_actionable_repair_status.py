"""Answer receipts, noisy witnesses, and unsupported disappearance are not repairs."""

from copy import deepcopy

import pytest

from autoformalism.research.actionable_repair_status import build_state


def report(
    *,
    conflict=True,
    robust=True,
    property_="closed_subsystem",
    scope=("a", "u"),
    status=None,
    comparisons=2,
    unavailable=(),
    question_id="q1",
):
    """Model the qualified diagnostic API without using any benchmark identity."""
    diagnostic = {
        "target": "a",
        "property": property_,
        "scope": list(scope),
        "status": status
        or ("conditional_conflict" if conflict else "no_conflict_found"),
        "comparisons": comparisons,
        "unavailable": list(unavailable),
    }
    question = {
        "question_id": question_id,
        "target": "a",
        "property": property_,
        "scope": list(scope),
        "witnesses": [{"error_intervals_disjoint": robust}],
    }
    return {
        "protocol": "construction-training-advice-1",
        "evidence_sha256": "fixed-public-evidence",
        "topology_sha256": "model-hash",
        "advisory_questions": [question] if conflict else [],
        "full_report": {"diagnostics": [diagnostic]},
        "acceptance_gate": False,
    }


@pytest.mark.parametrize(
    "disposition", ["revise", "retain_approximation", "dispute_assumption"]
)
def test_answers_do_not_clear_a_current_qualified_conflict(disposition):
    current = report()
    decision = {
        "question_id": "q1",
        "disposition": disposition,
        "explanation": "A claim, not a changed model or verified evidence.",
        "supporting_evidence": ["unverified prose is not a changed assumption"],
    }
    state = build_state(current, current, True, {"q1": decision})
    assert state["repair_status"] == "repair_required"
    assert not state["completion_allowed"]
    assert state["pending_questions"][0]["answer_resolves_conflict"] is False
    assert state["decisions"] == [decision]
    assert state["scientific_correctness"] == "unverified"


@pytest.mark.parametrize("bound", [False, None])
def test_exact_sample_conflicts_inside_unknown_or_overlapping_noise_are_limited(bound):
    current = report(robust=bound)
    state = build_state(current, current, True)
    assert not state["actionable_questions"]
    assert len(state["uncertain_questions"]) == 1
    assert state["repair_status"] == "review_deferred"
    assert not state["completion_allowed"]


def test_invalid_model_cannot_remove_an_old_conflict():
    current = report(conflict=False, status="deferred", unavailable=["algebraic cycle"])
    state = build_state(current, report(), False)
    assert state["repair_status"] == "structurally_blocked"
    assert state["baseline_outcomes"][0]["status"] == "structurally_blocked"
    assert not state["completion_allowed"]


@pytest.mark.parametrize(
    "reason",
    [
        "initialization of z is unavailable",
        "complete interpolation for v is unavailable",
        "unique-solution candidate-class hypothesis is unavailable",
        "unparsed conversion for a",
        "new diagnostic reason must abstain",
    ],
)
def test_unavailable_assumptions_after_revision_do_not_resolve(reason):
    current = report(
        conflict=False,
        status="deferred",
        scope=("a", "u", "z"),
        comparisons=0,
        unavailable=[reason],
    )
    state = build_state(current, report(), True)
    assert state["repair_status"] == "review_deferred"
    assert state["baseline_outcomes"][0]["status"] == "deferred_after_revision"
    assert not state["completion_allowed"]


def test_partially_supported_no_conflict_does_not_resolve_missing_old_comparison():
    current = report(
        conflict=False,
        unavailable=["initialization of z is unavailable"],
    )
    state = build_state(current, report(), True)
    assert state["repair_status"] == "review_deferred"


def test_performed_supported_recheck_clears_only_the_tracked_necessary_condition():
    state = build_state(report(conflict=False), report(), True)
    assert state["repair_status"] == "checks_clear"
    assert state["completion_allowed"]
    assert state["baseline_outcomes"][0]["status"] == "no_conflict_found"
    assert state["scientific_correctness"] == "unverified"


@pytest.mark.parametrize("property_", ["closed_subsystem", "scalar_autonomous"])
def test_verified_changed_conditioning_is_distinct_from_unknown_conditioning(property_):
    previous = report(property_=property_)
    current = report(
        property_=property_,
        conflict=False,
        status="deferred",
        scope=("a", "u", "v"),
        comparisons=0,
        unavailable=["no qualifying observed comparison"],
    )
    state = build_state(current, previous, True)
    assert state["completion_allowed"]
    assert (
        state["baseline_outcomes"][0]["status"] == "conditioning_distinction_verified"
    )
    current["full_report"]["diagnostics"][0]["unavailable"].append(
        "initialization of z is unavailable"
    )
    assert not build_state(current, previous, True)["completion_allowed"]


def test_no_qualifying_comparison_without_a_changed_scope_does_not_clear():
    state = build_state(
        report(
            conflict=False,
            status="deferred",
            comparisons=0,
            unavailable=["no qualifying observed comparison"],
        ),
        report(),
        True,
    )
    assert not state["completion_allowed"]


def test_explicit_time_removes_only_autonomous_restriction():
    previous = report(property_="scalar_autonomous")
    current = report(
        property_="scalar_autonomous",
        conflict=False,
        status="deferred",
        scope=("a", "u", "t"),
        unavailable=["explicit time dependence is not autonomous"],
        comparisons=0,
    )
    state = build_state(current, previous, True)
    assert state["baseline_outcomes"][0]["status"] == "autonomous_restriction_removed"
    assert state["completion_allowed"]
    assert not build_state(current, previous, False)["completion_allowed"]


def test_new_conflicts_are_carried_forward_and_changed_question_id_does_not_erase():
    initial = build_state(report(conflict=False), None, True)
    introduced = build_state(report(question_id="introduced"), initial, True)
    changed = build_state(report(question_id="changed"), introduced, True)
    assert changed["baseline_outcomes"][0]["previous_question_id"] == "introduced"
    assert changed["baseline_outcomes"][0]["current_question_id"] == "changed"
    unknown = report(
        conflict=False, status="deferred", unavailable=["initialization unavailable"]
    )
    deferred = build_state(unknown, changed, True)
    assert not deferred["completion_allowed"]
    assert deferred["tracked_conflicts"][0]["question_id"] == "introduced"


def test_missing_diagnostic_is_not_success():
    current = report(conflict=False)
    current["full_report"]["diagnostics"] = []
    state = build_state(current, report(), True)
    assert not state["completion_allowed"]


def test_evidence_is_frozen_and_inputs_are_never_mutated():
    current, baseline = report(), report()
    originals = deepcopy((current, baseline))
    build_state(current, baseline, True)
    assert (current, baseline) == originals
    current["evidence_sha256"] = "changed"
    with pytest.raises(ValueError, match="evidence changed"):
        build_state(current, baseline, True)


def test_actual_diagnostics_distinguish_new_known_forcing_from_unknown_latent_initial():
    from autoformalism.expressions import ValidationContext
    from autoformalism.research.construction_training import compatibility_feedback
    from tests.test_topology_compatibility import contract, draft, evidence, trajectory

    context = ValidationContext(
        targets=("a",), external_inputs=("u",), auxiliaries=("b",)
    )
    packet = evidence(
        trajectory("first", (0, 1, 2), aux={"b": (1, 1, 1)}),
        trajectory("second", (0, 2, 4), aux={"b": (2, 2, 2)}),
        context=context,
        config=contract(
            interpolation={"u": "linear_complete_knots", "b": "linear_complete_knots"},
            absolute_error_bounds={"a": 0, "u": 0, "b": 0},
        ),
    )
    previous = compatibility_feedback(draft(), packet)
    assert len(previous["advisory_questions"]) == 1
    distinguished = compatibility_feedback(draft(sources=("a", "u", "b")), packet)
    state = build_state(distinguished, previous, True)
    assert state["completion_allowed"]
    assert (
        state["baseline_outcomes"][0]["status"] == "conditioning_distinction_verified"
    )

    hidden = draft(sources=("a", "z"), extras=[("z", "differential", ("u",))])
    unsupported = compatibility_feedback(hidden, packet)
    state = build_state(unsupported, previous, True)
    assert not state["completion_allowed"]
    assert state["baseline_outcomes"][0]["status"] == "deferred_after_revision"
