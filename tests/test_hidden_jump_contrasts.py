"""Verify the conditional additive constraint using public observations only."""

from __future__ import annotations

import pytest

from autoformalism.research.hidden_jump_contrasts import (
    LINEAR,
    STEP,
    screen_input_jumps,
)
from scripts.check_hidden_jump_controls import controls, event, run


def screen(name, **updates):
    rows, kwargs = controls()[name]
    return screen_input_jumps(rows, **(kwargs | updates))


def test_exact_levels_and_degree_are_relative_to_reference():
    result = screen("exact")
    assert result["status"] == "identified_on_levels"
    assert [v["exact_relative_value"] for v in result["components"][0]["levels"]] == [
        "0",
        "1",
        "4",
        "9",
    ]
    assert result["affine"] == {"status": "excluded"}
    assert result["anchored_polynomial_interpolant"] == {
        "degree": 2,
        "unique_within_degree_at_most": 3,
    }
    rows, kwargs = controls()["exact"]
    assert screen_input_jumps(rows[::-1], **kwargs) == result


def test_bounded_noisy_hidden_motion():
    result = screen("bounded")
    assert result["status"] == "bounded_on_levels"
    for value in result["components"][0]["levels"]:
        assert value["lower"] <= value["level"] ** 2 <= value["upper"]
    assert result["affine"]["status"] == "excluded"
    assert "anchored_polynomial_interpolant" not in result


def test_interaction_cycle_falsifies_additive_assumptions():
    result = screen("interaction")
    assert result["status"] == "incompatible"
    assert len(result["cycle"]) == 4
    assert result["minimum_cycle_violation"] == 17
    assert "components" not in result
    for current, following in zip(
        result["cycle"], result["cycle"][1:] + result["cycle"][:1], strict=True
    ):
        assert current["to"] == following["from"]
        assert {current["from"], current["to"]} == set(
            result["witnesses"][current["witness"]]["levels"]
        )


def test_disconnected_components_have_independent_offsets_and_need_a_bridge():
    result = screen("disconnected")
    assert result["status"] == "disconnected_levels"
    assert [c["reference_level"] for c in result["components"]] == [0, 2]
    assert [c["levels"][1]["exact_relative_value"] for c in result["components"]] == [
        "1",
        "5",
    ]


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        ("continuous_input", "no_level_jump_contract"),
        ("uncertified_rates", "unbounded_rate_error"),
        ("marked_resets", "no_isolated_jump_witness"),
        ("simultaneous", "no_isolated_jump_witness"),
    ],
)
def test_unavailable_evidence(name, reason):
    result = screen(name)
    assert (result["status"], result["reason"]) == ("abstained", reason)
    assert not result["witnesses"]


def test_unverified_continuity_or_unknown_other_schedule():
    assert (
        screen("exact", continuous_nuisance=False)["reason"] == "unverified_continuity"
    )
    assert (
        screen("simultaneous", interpolation={"u": STEP})["reason"]
        == "unknown_forcing_schedule"
    )
    # A second continuously interpolated input stays continuous at the switch.
    assert (
        screen("simultaneous", interpolation={"u": STEP, "u2": LINEAR})["status"]
        == "identified_on_levels"
    )


def test_one_sided_terminal_level_does_not_gain_a_value_and_affine_feasible():
    row = event(0, [0, 1, 2])
    _, kwargs = controls()["exact"]
    result = screen_input_jumps([row], **kwargs)
    assert result["status"] == "disconnected_levels"
    assert result["skipped"] == {"no_right_sample": 1}
    assert result["affine"] == {
        "status": "compatible",
        "slope_lower": 1,
        "slope_upper": 1,
    }


def test_reject_bad_bound_or_split_and_invalid_reset_indices():
    rows, kwargs = controls()["exact"]
    for bad_bound in (float("nan"), -0.01):
        with pytest.raises(ValueError):
            screen("exact", observation_error=bad_bound)
    with pytest.raises(ValueError):
        screen_input_jumps([{**rows[0], "trajectory_id": "validation_000"}], **kwargs)
    with pytest.raises(ValueError):
        screen("exact", reset_knots={"train_000": [-1]})


def test_deterministic_resume(tmp_path):
    first = run(tmp_path)
    assert run(tmp_path) == first
    assert len(first["cases"]) == 8
