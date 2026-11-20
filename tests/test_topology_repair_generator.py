"""Train-only, prefix-sensitive edits for graph contradictions on arbitrary names."""

from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.search.construction_ledger import Draft
from autoformalism.search.topology_compatibility import (
    EvidenceContract,
    InitialRule,
    build_evidence,
)
from autoformalism.search.topology_repair_generator import (
    TopologyEdit,
    assess_edit,
    generate_repairs,
)


def row(name, level, upstream, rain, spill, *, time=(0, 1, 2)):
    return Trajectory(
        trajectory_id=name,
        time=np.array(time, dtype=float),
        targets={"level": np.array(level), "upstream": np.array(upstream)},
        external_inputs={"rain": np.array(rain), "spill": np.array(spill)},
        auxiliaries={},
        fixed_covariates={},
        derivatives={},
    )


def evidence(
    rows,
    *,
    interpolation="linear_complete_knots",
    forbid=False,
    lagged_targets=(),
    **changes,
):
    context = ValidationContext(
        targets=("level", "upstream"),
        external_inputs=("rain", "spill"),
        forbid_latent_states=forbid,
        lagged_targets=lagged_targets,
    )
    fields = {
        "provenance": "Synthetic public linear reservoir records",
        "dependencies_exhaustive": True,
        "shared_parameters": True,
        "open_loop": True,
        "reset_impulse_free": True,
        "uniqueness_assumed": True,
        "initial_rules": {
            n: InitialRule(mode="observed_initial") for n in ("level", "upstream")
        },
        "absolute_error_bounds": dict.fromkeys(
            ("level", "upstream", "rain", "spill"), 0
        ),
        "interpolation": dict.fromkeys(("rain", "spill"), interpolation),
    }
    return build_evidence(
        DatasetSplit(SplitName.TRAIN, tuple(rows), "fixture"),
        context,
        EvidenceContract(**(fields | changes)),
    )


def draft(upstream=("upstream",), *, extra=None):
    sources = {"level": ("level",), "upstream": upstream}
    if extra:
        sources.update(extra)
    return Draft.model_validate(
        {
            "variables": [
                {"name": n, "definition": "differential", "scientific_role": n}
                for n in sources
            ],
            "equations": [
                {
                    "name": n,
                    "terms": [
                        {
                            "sources": s,
                            "outer_weight_sign": "unrestricted",
                            "scientific_role": "stored topology",
                        }
                    ],
                }
                for n, s in sources.items()
            ],
        }
    )


PAIR = ("event_a", "event_b")
CHANNELS = ("level", "upstream", "rain", "spill")


def options(model, packet, *, time=2, sources=CHANNELS):
    results = generate_repairs(
        model, packet, "level", allowed_sources=sources, pair=PAIR, time=time
    )
    return {(r.edit.kind, r.edit.source): r for r in results}


def rows():
    return [
        row("event_a", (0, 2, 3), (0, 0, 0), (0, 1, 0), (0, 0, 0)),
        row("event_b", (0, 0, 1), (1, 1, 1), (0, 0, 0), (0, 0, 1)),
        row("event_c", (0, 4, 5), (0, 0, 0), (0, 1, 0), (0, 0, 0)),
    ]


def test_direct_and_mediated_edits_preserve_parents_and_check_every_pair():
    model, packet = draft(), evidence(rows())
    original = model.model_dump_json()
    answer = options(model, packet)
    rain = answer["add_dependency", "rain"]
    assert rain.status == "distinguishes" and rain.added_dependencies == (
        ("rain", "level"),
    )
    assert rain.distinctions == ("rain forcing differs",)
    assert [c.trajectories for c in rain.remaining_conflicts] == [
        ("event_a", "event_c")
    ]
    assert rain.remaining_conflicts[0].time == 1
    memory = answer["via_new_memory", "rain"]
    z = memory.memory_state
    assert memory.added_dependencies == (("rain", z), (z, z), (z, "level"))
    assert memory.remaining_conflicts == rain.remaining_conflicts
    upstream = answer["add_dependency", "upstream"]
    assert upstream.distinctions == ("upstream initial information differs",)
    assert model.model_dump_json() == original
    assert answer["add_dependency", "level"].status == "invalid"
    assert answer["via_new_memory", "level"].status == "still_forced_equal"
    assert (
        assess_edit(
            model, packet, memory.edit, allowed_sources=CHANNELS, pair=PAIR, time=2
        )
        == memory
    )
    assert options(model, evidence(rows()[::-1]), sources=CHANNELS[::-1]) == answer


def test_last_sample_jump_cannot_move_continuous_state_at_that_instant():
    before = options(draft(), evidence(rows(), interpolation="previous_complete_knots"))
    assert before["add_dependency", "spill"].status == "still_forced_equal"
    assert before["add_dependency", "rain"].status == "distinguishes"
    assert before["add_dependency", "rain"].remaining_conflicts[0].time == 1
    # Piecewise-linear ramps to the endpoint act over the preceding interval.
    after = options(draft(), evidence(rows()))
    assert after["add_dependency", "spill"].status == "distinguishes"
    assert (
        options(draft(), evidence(rows()), time=1)["add_dependency", "spill"].status
        == "still_forced_equal"
    )


def test_generated_measurement_difference_does_not_drive_predicted_state():
    samples = rows()
    samples[1] = replace(
        samples[1],
        targets={"level": np.array((0, 0, 1)), "upstream": np.array((0, 9, 9))},
    )
    answer = options(draft(), evidence(samples))
    assert answer["add_dependency", "upstream"].status == "still_forced_equal"
    # Unknown hidden information defers a comparison, but a known differing
    # source in the same closure still breaks the equality argument.
    model = draft(upstream=("hidden", "spill"), extra={"hidden": ("hidden",)})
    answer = options(model, evidence(samples), sources=("hidden", "upstream"))
    assert answer["add_dependency", "hidden"].status == "unresolved"
    assert answer["add_dependency", "upstream"].status == "distinguishes"
    assert answer["add_dependency", "upstream"].unavailable == ("hidden",)


def test_unavailable_source_policy_and_latent_prohibition_are_enforced():
    model, packet = draft(), evidence(rows(), forbid=True)
    assert options(model, packet)["via_new_memory", "rain"].status == "invalid"
    edit = TopologyEdit(kind="add_dependency", source="spill", target="level")
    assert (
        assess_edit(
            model, packet, edit, allowed_sources=("rain",), pair=PAIR, time=2
        ).status
        == "invalid"
    )
    with pytest.raises(ValueError, match="allowed sources"):
        options(model, packet, sources=("undeclared",))
    alternate = options(draft(extra={"repair_memory": ()}), evidence(rows()))
    assert alternate["via_new_memory", "rain"].memory_state == "repair_memory_2"


def test_parent_requires_robust_and_certified_witness():
    model = draft()
    with pytest.raises(ValueError, match="disjoint"):
        options(model, evidence(rows()), time=0)
    with pytest.raises(ValueError, match="dependencies_exhaustive"):
        options(model, evidence(rows(), dependencies_exhaustive=None))
    with pytest.raises(ValueError, match="does not certify"):
        options(model, evidence(rows(), initial_rules={}))
    with pytest.raises(ValueError, match="digest"):
        options(model, evidence(rows()).model_copy(update={"sha256": "0" * 64}))


def test_inconsistent_clocks_and_structural_ownership_errors_abstain():
    samples = rows()
    samples[1] = replace(samples[1], time=np.array((1, 2, 3)))
    with pytest.raises(ValueError, match="matched training"):
        options(draft(), evidence(samples))
    m = draft()
    variables = tuple(
        v.model_copy(update={"definition": "algebraic"}) for v in m.variables
    )
    with pytest.raises(ValueError, match="algebraic cycle"):
        options(m.model_copy(update={"variables": variables}), evidence(rows()))


def test_distinguishing_input_on_interleaved_grids_uses_whole_prefix():
    # Only the unshared middle knot carries a pulse, with equal end observations
    # of the forcing on both records. The preceding signal still reaches level.
    a = row("event_a", (0, 0, 3), (0, 0, 0), (0, 1, 0), (0, 0, 0), time=(0, 0.25, 2))
    b = row("event_b", (0, 1), (0, 0), (0, 0), (0, 0), time=(0, 2))
    for mode in ("previous_complete_knots", "linear_complete_knots"):
        answer = options(draft(), evidence([a, b], interpolation=mode))
        assert answer["add_dependency", "rain"].status == "distinguishes"
        assert answer["add_dependency", "spill"].status == "still_forced_equal"


def test_target_difference_within_declared_error_bands_does_not_trigger_generation():
    initial = {"level": InitialRule(mode="shared_parameter")}
    bounds = {"level": 2.0, "upstream": 0, "rain": 0, "spill": 0}
    with pytest.raises(ValueError, match="disjoint"):
        options(
            draft(),
            evidence(rows(), absolute_error_bounds=bounds, initial_rules=initial),
        )


def test_lagged_target_context_abstains_before_comparing_forcing():
    packet = evidence(rows(), lagged_targets=("upstream",))
    with pytest.raises(ValueError, match="history/observation-update"):
        options(draft(), packet)


def test_ode_edit_cannot_correct_incompatible_target_initialization():
    # The common target initializer sees identical spill(0). Neither a
    # different upstream initial nor differing rain(0) changes its arguments.
    a = row("event_a", (1, 2, 3), (1, 1, 1), (1, 1, 1), (0, 0, 0))
    b = row("event_b", (0, 0, 0), (2, 2, 2), (4, 4, 4), (0, 0, 0))
    rules = {
        "level": InitialRule(mode="common_map", arguments=("spill",)),
        "upstream": InitialRule(mode="observed_initial"),
    }
    packet = evidence([a, b], initial_rules=rules)
    for source in ("rain", "upstream"):
        at_start = options(draft(), packet, time=0)["add_dependency", source]
        assert at_start.status == "still_forced_equal"
        later = options(draft(), packet, time=2)["add_dependency", source]
        assert later.status == "distinguishes"
        assert later.remaining_conflicts[0].time == 0
    # With different *permitted initializer* arguments, the parent itself
    # already has a way to differ. A structural contradiction is not certified.
    rules["level"] = InitialRule(mode="common_map", arguments=("rain",))
    with pytest.raises(ValueError, match="does not certify"):
        options(draft(), evidence([a, b], initial_rules=rules), time=0)
