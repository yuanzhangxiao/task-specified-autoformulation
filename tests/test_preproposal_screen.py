"""The pre-proposal screen uses only permitted information and observed targets."""

from itertools import combinations

import pytest

from autoformalism.search.preproposal_screen import (
    assess_options,
    assess_realized_roots,
    screen_target,
)
from tests.test_topology_repair_generator import evidence, row


def complementary():
    # Shared law dx/dt = rain + 2*spill; three scalar training trajectories.
    return [
        row("a", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        row("b", (0, 1, 2), (0, 0, 0), (1, 1, 1), (0, 0, 0)),
        row("c", (0, 2, 4), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ]


def test_complementary_signals_and_equal_signature_alternatives_remain_distinct():
    packet = evidence(complementary())
    screen = screen_target(
        packet, "level", histories=("spill", "rain"), initials=("upstream",)
    )
    assert screen.status == "screened"
    assert {c.options for c in screen.candidates} == {
        ("history:rain", "history:spill"),
        ("history:rain", "initial:upstream"),
    }
    assert screen.pair_count == screen.remaining_witness_count == 3
    assert screen.subsets_examined == 8
    assert screen.subset_counts == {"screened": 3, "contradicted": 5, "unresolved": 0}
    by_id = {w.id: w for w in screen.witnesses}
    for candidate in screen.candidates:
        assert assess_options(screen, candidate.options).status == "screened"
        for option, witness_id in candidate.private_witnesses.items():
            assert set(candidate.options).intersection(by_id[witness_id].different) == {
                option
            }
    baseline = assess_options(screen, ())
    assert baseline.conflicting_pairs == 3 and baseline.max_pair_error_floor == 1
    assert screen == screen_target(
        evidence(complementary()[::-1]),
        "level",
        histories=("rain", "spill"),
        initials=("upstream",),
    )


def test_independent_exhaustive_cover_check_matches_bitset_search():
    screen = screen_target(
        evidence(complementary()),
        "level",
        histories=("rain", "spill"),
        initials=("spill", "rain", "upstream"),
        required_histories=("rain",),
    )
    assert screen.remaining_witness_count == 1
    passing = set()
    counts = dict.fromkeys(screen.subset_counts, 0)
    for size in range(len(screen.optional_options) + 1):
        for subset in combinations(screen.optional_options, size):
            result = assess_options(screen, subset)
            counts[result.status] += 1
            if result.status == "screened":
                passing.add(frozenset(subset))
    assert counts == screen.subset_counts
    assert {frozenset(c.options) for c in screen.candidates} == {
        s for s in passing if not any(other < s for other in passing)
    }


def test_measured_initial_suffices_with_no_new_required_dependency():
    records = [
        row("a", (2, 3, 4), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        row("b", (1, 1, 1), (9, 9, 9), (9, 9, 9), (0, 0, 0)),
    ]
    screen = screen_target(
        evidence(records), "level", histories=("rain",), initials=("upstream", "level")
    )
    assert screen.remaining_witness_count == 0 and len(screen.candidates) == 1
    assert screen.candidates[0].options == ()
    assert screen.witnesses[0].time == 0 and screen.witnesses[0].different == (
        "initial:level",
    )
    assert screen.required_options == ("initial:level",)


def test_realized_roots_check_one_support_with_the_same_witness_logic():
    records = complementary()
    packet = evidence(records)
    args = {"histories": ("rain",), "initials": ("upstream", "rain")}
    original = screen_target(packet, "level", **args)
    realized, assessment = assess_realized_roots(packet, "level", **args)
    assert assessment == assess_options(original, original.optional_options)
    assert assessment.status == "screened" and realized.subsets_examined == 1
    assert realized.optional_options == () and len(realized.required_options) == 4
    _, without_other_root = assess_realized_roots(
        packet, "level", histories=("rain",), initials=()
    )
    assert without_other_root.status == "contradicted"
    _, uncertain = assess_realized_roots(
        evidence(records, absolute_error_bounds={"level": 0}),
        "level",
        histories=("rain",),
        initials=("upstream",),
    )
    assert uncertain.status == "unresolved"
    _, single_trajectory = assess_realized_roots(
        evidence(records[:1]), "level", histories=(), initials=()
    )
    assert single_trajectory.status == "unresolved"


def test_held_terminal_jump_cannot_distinguish_continuous_target_but_linear_ramp_can():
    records = [
        row("a", (0, 1, 1), (0, 0, 0), (0, 1, 1), (0, 0, 0)),
        row("b", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ]

    def probe(mode):
        return screen_target(
            evidence(records, interpolation=mode), "level", histories=("rain",)
        )

    assert probe("previous_complete_knots").status == "contradicted"
    linear = probe("linear_complete_knots")
    assert linear.status == "screened"
    (w,) = linear.witnesses
    assert (w.trajectories, w.time, w.observations, w.different) == (
        ("a", "b"),
        1,
        (1, 0),
        ("history:rain",),
    )


@pytest.mark.parametrize("amplitude,status", [(0.5, "unresolved"), (3, "screened")])
def test_uncertain_interleaved_knot_is_unknown_or_certifiably_different(
    amplitude, status
):
    records = [
        row("a", (0, 0, 1), (0, 0, 0), (0, amplitude, 0), (0, 0, 0), time=(0, 0.5, 2)),
        row("b", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ]
    packet = evidence(records, absolute_error_bounds={"level": 0, "rain": 0.5})
    screen = screen_target(packet, "level", histories=("rain",))
    assert screen.status == status
    assert assess_options(screen, ("history:rain",)).status == status
    assert assess_options(screen, ()).status == "contradicted"


def test_unavailable_contract_and_unknown_pairs_remain_separate_from_conflict():
    records = complementary()[:2]
    screen = screen_target(
        evidence(records, shared_parameters=None), "level", histories=()
    )
    assert (
        screen.status == "unresolved"
        and screen.candidates == ()
        and screen.subsets_examined == 0
    )
    assert assess_options(screen, ()).unresolved_pairs == 1
    screen = screen_target(evidence(records, initial_rules={}), "level", histories=())
    assert "observed time-zero reading" in screen.unavailable[0]
    screen = screen_target(
        evidence(records, interpolation="unknown"), "level", histories=("rain",)
    )
    assert screen.witnesses[0].unknown == ("history:rain",)
    assert assess_options(screen, ()).status == "contradicted"
    assert assess_options(screen, ("history:rain",)).unresolved_pairs == 1


def test_different_origins_do_not_create_vacuous_empty_support():
    records = [
        row("a", (0, 1, 2), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
        row("b", (4, 4, 4), (0, 0, 0), (0, 0, 0), (0, 0, 0), time=(1, 2, 3)),
    ]
    screen = screen_target(evidence(records), "level", histories=("rain",))
    assert screen.status == "unresolved" and screen.candidates == ()
    assert screen.noncomparable_pairs == (("a", "b"),)


def test_materiality_uses_earliest_qualified_gap_with_unchanged_source_precision():
    records = [
        row("a", (0, 0.001, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0)),
        row("b", (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    ]
    packet = evidence(records)
    assert screen_target(packet, "level", histories=("rain",)).status == "contradicted"
    s = screen_target(packet, "level", histories=("rain",), minimum_target_gap=0.1)
    assert s.minimum_target_gap == 0.1
    assert s.witnesses[0].time == 2 and s.witnesses[0].different == ("history:rain",)
    assert assess_options(s, ()).status == "contradicted"
    assert s.candidates[0].options == ("history:rain",)
    no_w = screen_target(packet, "level", histories=("rain",), minimum_target_gap=1.0)
    assert no_w.witnesses == () and no_w.candidates[0].options == ()
    for invalid in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="nonnegative"):
            screen_target(packet, "level", histories=(), minimum_target_gap=invalid)


def test_reject_invalid_sources_unbounded_enumeration_and_data_tampering():
    packet = evidence(complementary())
    for kw in (
        {"histories": ("upstream",)},
        {"histories": ("level",)},
        {"histories": ("trajectory_id",)},
        {"histories": ("rain", "rain")},
        {"histories": (), "initials": ("hidden",)},
        {"histories": (), "required_histories": ("rain",)},
    ):
        with pytest.raises(ValueError, match="permitted information"):
            screen_target(packet, "level", **kw)
    with pytest.raises(ValueError, match="4 subsets"):
        screen_target(packet, "level", histories=("rain", "spill"), max_subsets=3)
    with pytest.raises(ValueError, match="measured public target"):
        screen_target(packet, "unsupported", histories=())
    with pytest.raises(ValueError, match="digest"):
        screen_target(
            packet.model_copy(update={"sha256": "0" * 64}), "level", histories=()
        )
    with pytest.raises(ValueError, match="split"):
        screen_target(
            packet.model_copy(update={"split": "test"}), "level", histories=()
        )
    screen = screen_target(packet, "level", histories=("rain", "spill"))
    with pytest.raises(ValueError, match="Select unique options"):
        assess_options(screen, ("initial:level",))
