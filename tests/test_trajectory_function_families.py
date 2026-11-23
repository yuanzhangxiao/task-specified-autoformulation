"""Function-family screening compares whole training events under a fixed graph."""

from copy import deepcopy

import numpy as np
import pytest

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.expressions import ValidationContext
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.trajectory_function_families import build_shortlist
from autoformalism.search.topology_compatibility import EvidenceContract, build_evidence


def packet(rows, context, case="synthetic", **contract):
    train = DatasetSplit(SplitName.TRAIN, tuple(rows), "unit-test")
    evidence = build_evidence(
        train, context, EvidenceContract(provenance="synthetic", **contract)
    )
    return {"case_id": case, "evidence": evidence.model_dump(mode="json")}


def event(name, time, targets, inputs, fixed=None):
    return Trajectory(
        trajectory_id=name,
        time=np.asarray(time),
        targets=targets,
        external_inputs=inputs,
        auxiliaries={},
        derivatives={},
        fixed_covariates=fixed or {},
    )


@pytest.mark.parametrize("kind", ["algebraic", "differential"])
def test_ranks_joint_dependence_from_values_and_from_intervals(kind):
    rows = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        t = np.cumsum(rng.uniform(0.5, 2, size=30))
        a, b = rng.uniform(-2, 2, (2, len(t)))
        f = 2 + a - b + 3 * a * b
        y = (
            f
            if kind == "algebraic"
            else np.r_[1, 1 + np.cumsum((f[1:] + f[:-1]) * np.diff(t) / 2)]
        )
        rows.append(event(f"train_{seed}", t, {"y": y}, {"a": a, "b": b}))
    src = packet(rows, ValidationContext(targets=("y",), external_inputs=("a", "b")))
    graph = {
        "sources": {"y": ["a", "b"]},
        "variable_types": {"y": kind},
        "initializers": {},
    }
    if kind == "differential":
        graph["initializers"]["y"] = {"mode": "observed_initial", "channel": "y"}
    original = deepcopy(graph)
    result = build_shortlist(graph, src)
    assert result["candidate_families"][0]["y"] == "interaction a b"
    info = result["evidence"]["equations"]["y"]
    assert info["family_rmse"]["interaction a b"] < 1e-4
    assert info["status"] == "train_trajectory_cross_validation"
    assert info["trajectories"] == 6
    assert graph == original
    assert len(result["candidates"]) == 2
    for model in result["candidates"]:
        symbols = (
            RestrictedParser().parse(model["equations"]["y"], location="y").symbols
        )
        assert symbols - model["parameters"].keys() == {"a", "b"}
        assert not model["initializers"]


def test_saturation_can_beat_affine_and_heldout_data_rejected():
    a = np.linspace(-3, 3, 201)
    y = 2 + 3 * np.tanh((a - np.mean(a)) / np.std(a))
    rows = [
        event(f"train_{i}", np.arange(len(a)), {"y": y}, {"a": a}) for i in range(3)
    ]
    src = packet(rows, ValidationContext(targets=("y",), external_inputs=("a",)))
    graph = {
        "sources": {"y": ["a"]},
        "variable_types": {"y": "algebraic"},
        "initializers": {},
    }
    result = build_shortlist(graph, src)
    assert "tanh" in result["candidates"][0]["equations"]["y"]
    assert result["evidence"]["equations"]["y"]["family_rmse"]["saturating a"] < 1e-4
    src["evidence"]["split"] = "validation"
    with pytest.raises(ValueError, match="training evidence"):
        build_shortlist(graph, src)
    src["evidence"]["split"] = "train"
    src["evidence"]["trajectories"][0]["targets"]["y"][0] += 1
    with pytest.raises(ValueError, match="digest"):
        build_shortlist(graph, src)


def test_step_forcing_and_causal_latent_initializers():
    y = np.array([1, 1, 6, 4, 6.0])
    a = np.array([0, 5, -2, 2, 10.0])
    rows = [event(f"train_{i}", np.arange(5), {"y": y}, {"a": a}) for i in range(3)]
    src = packet(
        rows,
        ValidationContext(targets=("y",), external_inputs=("a",)),
        interpolation={"a": "previous_complete_knots"},
    )
    graph = {
        "sources": {"y": ["a"], "h": ["y"], "v": ["h"]},
        "variable_types": dict.fromkeys(("y", "h", "v"), "differential"),
        "initializers": {
            "y": {"mode": "observed_initial", "channel": "y"},
            "h": {"mode": "value", "scope": "shared_training"},
            "v": {"mode": "map", "public_initials": ["a"], "state_initials": ["h"]},
        },
    }
    out = build_shortlist(graph, src)
    rows = out["evidence"]["equations"]
    assert rows["y"]["family_rmse"]["affine"] < 1e-4
    assert rows["h"]["unobserved"] == ["h"]
    assert rows["v"]["status"] == "whole_system_rollout_required"
    for model in out["candidates"]:
        assert set(model["initializers"]) == {"h", "v"}
        refs = (
            RestrictedParser().parse(model["initializers"]["v"], location="v").symbols
        )
        assert refs - model["parameters"].keys() == {"a", "state0_h"}


def test_isolate_train_only_basin_law_and_report_missing_edges():
    cov = {
        "area_up": 800,
        "area_down": 1200,
        "initial_up": 0.35,
        "crest_up": 0.35,
        "crest_down": 0.12,
    }
    rows = []
    for idx, (size, head) in enumerate([(20, 1.0), (8, 0.25), (4, 0.6)]):
        rows.append(
            event(
                f"train_{idx}",
                np.arange(size),
                {"h_down": np.full(size, 0.12 + head)},
                {
                    "inflow_down": np.full(size, 3 * head**1.5),
                    "inflow_up": np.zeros(size),
                },
                cov,
            )
        )
    ctx = ValidationContext(
        targets=("h_down",),
        external_inputs=("inflow_down", "inflow_up"),
        fixed_covariates=tuple(cov),
    )
    src = packet(
        rows,
        ctx,
        case="basin_seed0_downstream",
        interpolation=dict.fromkeys(ctx.external_inputs, "linear_complete_knots"),
    )
    graph = {
        "sources": {"h_down": ["h_up"], "h_up": ["inflow_up", "h_up"]},
        "variable_types": {"h_down": "differential", "h_up": "differential"},
        "initializers": {
            "h_down": {"mode": "observed_initial", "channel": "h_down"},
            "h_up": {"mode": "observed_initial", "channel": "initial_up"},
        },
    }
    out = build_shortlist(graph, src)
    evidence = out["evidence"]["isolated_outlet"]
    assert evidence["fitting_trajectory"] == "train_0"
    assert evidence["scores"][0]["power"] == 1.5
    assert evidence["scores"][0]["flow_rmse_m3_per_min"] < 1e-9
    assert "h_down" in evidence["missing_direct_sources_for_depth_balances"]["h_down"]
    for model in out["candidates"]:
        assert "crest_down" not in model["equations"]["h_down"]
