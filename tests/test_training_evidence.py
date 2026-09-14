"""Train-only descriptive evidence, bounded provenance and constructor wiring."""

import json
from dataclasses import replace

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.llm.staged_topology import StagedModelSettings, StagedTopologyClient
from autoformalism.rebuttal.staged_topology_campaign import diagnostic_task
from autoformalism.schemas.staged_topology import ModelingLimits, PublicScientificBrief
from autoformalism.search.staged_function_runner import run_staged_functions
from autoformalism.search.staged_topology_runner import run_staged_topology
from autoformalism.search.training_evidence import (
    EvidenceSettings,
    build_training_evidence,
    freeze_training_evidence,
    validate_training_evidence,
)
from tests.test_staged_topology_runner import provider_response


def training(values=(0, 1, 2, 1, 0, 1, 2), inputs=None, count=1):
    """Use descriptive arrays without benchmark files or derivative references."""
    return DatasetSplit(
        SplitName.TRAIN,
        tuple(
            Trajectory(
                trajectory_id=f"train_{i}",
                time=np.arange(len(values), dtype=float),
                targets={"x": np.array(values, dtype=float) + i},
                auxiliaries={},
                external_inputs={
                    "u": np.array(
                        inputs if inputs is not None else [0] * len(values), dtype=float
                    )
                },
                fixed_covariates={},
                derivatives={},
            )
            for i in range(count)
        ),
        "untrusted-fingerprint",
    )


CONTEXT = ValidationContext(targets=("x",), external_inputs=("u",))


def test_measured_turning_points_and_zero_input_are_precise():
    packet = build_training_evidence(training(), CONTEXT)
    trajectory = packet.trajectories[0]
    assert [
        (p.index, p.time, p.value) for p in trajectory.targets["x"].turning_points
    ] == [(2, 2, 2), (4, 4, 0)]
    assert trajectory.zero_sampled_inputs_with_changing_targets == ("x",)
    assert trajectory.inputs["u"].change_count == 0
    validate_training_evidence(packet, CONTEXT)


@pytest.mark.parametrize("name", [SplitName.VALIDATION, SplitName.TEST])
def test_heldout_rejected_before_reading_trajectories(name):
    with pytest.raises(ValueError, match="train split only"):
        build_training_evidence(DatasetSplit(name, None, "heldout"), CONTEXT)


@pytest.mark.parametrize("values", [(2,), (2, 2, 2), (2, 2.000000001, 2)])
def test_short_constant_and_below_tolerance_noise(values):
    packet = build_training_evidence(training(values), CONTEXT)
    item = packet.trajectories[0]
    assert not item.targets["x"].turning_points
    assert not item.zero_sampled_inputs_with_changing_targets
    assert (item.minimum_time_step is None) == (len(values) == 1)


def test_mixed_schedules_bounds_and_initial_variability():
    data = training(inputs=(0, 0, 1, 1, -1, 2, 0), count=8)
    packet = build_training_evidence(
        data, CONTEXT, EvidenceSettings(maximum_trajectories=3, maximum_events=2)
    )
    assert packet.omitted_trajectories == 5
    assert packet.initial_target_ranges == {"x": (0, 7)}
    input_ = packet.trajectories[0].inputs["u"]
    assert input_.change_count == 4 and input_.omitted_changes == 2
    assert [(p.before.index, p.after.index) for p in input_.changes] == [(1, 2), (5, 6)]
    assert input_.changes[0].target_change_in_same_bracket == {"x": 1}
    assert not packet.trajectories[0].zero_sampled_inputs_with_changing_targets
    assert packet == build_training_evidence(
        replace(data, trajectories=tuple(reversed(data.trajectories))),
        CONTEXT,
        packet.settings,
    )


def test_stale_data_and_tampering_fail_even_with_same_split_fingerprint(tmp_path):
    packet = build_training_evidence(training(), CONTEXT)
    path = tmp_path / "packet.json"
    freeze_training_evidence(path, packet)
    freeze_training_evidence(path, packet)
    changed = build_training_evidence(training((0, 2, 2, 1, 0, 1, 2)), CONTEXT)
    assert changed.source_sha256 != packet.source_sha256
    with pytest.raises(ValueError, match="frozen training evidence differs"):
        freeze_training_evidence(path, changed)
    with pytest.raises(ValueError, match="digest or public context"):
        validate_training_evidence(
            packet.model_copy(update={"omitted_trajectories": 99}), CONTEXT
        )
    with pytest.raises(ValueError, match="digest or public context"):
        validate_training_evidence(
            packet, CONTEXT.model_copy(update={"targets": ("other",)})
        )


@pytest.mark.parametrize("values", [(0, float("nan")), (0, float("inf"))])
def test_nonfinite_data_fails_explicitly(values):
    with pytest.raises(ValueError, match="finite and aligned"):
        build_training_evidence(training(values), CONTEXT)


def test_packet_reaches_every_construction_stage_and_cannot_change_on_resume(tmp_path):
    fixture = diagnostic_task("driven_memory", ModelingLimits())
    brief = PublicScientificBrief.model_validate(fixture["brief"])
    context = ValidationContext.model_validate(fixture["context"])
    packet = build_training_evidence(training(), context)
    calls = []

    def transport(url, body, timeout):
        payload = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        calls.append(payload)
        assert payload["public_brief"]["scientific_context"] == brief.scientific_context
        assert (
            payload["public_brief"]["training_observations"]["packet_sha256"]
            == packet.packet_sha256
        )
        if "selected_state" in payload:
            return provider_response(
                {"initial": {"mode": "shared_value", "role": "coefficient"}}
            )
        if "selected_term" in payload:
            return provider_response(
                {
                    "expression": "z-x"
                    if payload["selected_term"]["lhs"] == "x"
                    else "u-z",
                    "parameters": [],
                }
            )
        if "selected_lhs" in payload:
            return provider_response(
                {
                    "terms": [
                        {
                            "sources": ["z", "x"]
                            if payload["selected_lhs"]["name"] == "x"
                            else ["u", "z"],
                            "outer_weight_sign": "positive",
                            "scientific_role": "response",
                        }
                    ],
                    "inventory_revision": None,
                }
            )
        return provider_response({"variables": fixture["initial_inventory"]})

    client = StagedTopologyClient(
        settings=StagedModelSettings(),
        base_url="http://unused",
        directory=tmp_path / "calls",
        namespace="evidence",
        seed=0,
        transport=transport,
    )
    source = run_staged_topology(
        brief, context, client, tmp_path / "topology", training_evidence=packet
    )
    assert source["complete_topology"]
    result = run_staged_functions(
        brief,
        context,
        source,
        client,
        tmp_path / "functions",
        initialization_policy="causal_training",
        training_evidence=packet,
    )
    assert result["complete_model"], result.get("error")
    assert any("agenda" in c for c in calls)
    assert any("selected_lhs" in c for c in calls)
    assert any("selected_term" in c for c in calls)
    assert any("selected_state" in c for c in calls)
    before = len(calls)
    assert (
        run_staged_functions(
            brief,
            context,
            source,
            client,
            tmp_path / "functions",
            initialization_policy="causal_training",
            training_evidence=packet,
        )
        == result
    )
    assert len(calls) == before
    with pytest.raises(ValueError, match="contract differs"):
        run_staged_topology(brief, context, client, tmp_path / "topology")
    with pytest.raises(ValueError, match="contract differs"):
        run_staged_functions(
            brief,
            context,
            source,
            client,
            tmp_path / "functions",
            initialization_policy="causal_training",
        )
