"""Conditional weight fitting, scale covariance, and fixed-boundary invariants."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("casadi")

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting.conditional_collocation import (
    ConditionalCollocation,
    common_initialization,
)
from autoformalism.fitting.conditional_optimizer import load_point, optimize
from autoformalism.fitting.conditional_scaling import (
    algebraic_scale_audit,
    partition_and_exponents,
    system_for,
)
from autoformalism.fitting.models import FitConfig


def fixture():
    candidate = {
        "candidate_id": "scale_test",
        "parent_candidate_id": None,
        "states": [
            {"name": "x", "kind": "latent"},
            {"name": "v01", "kind": "observed"},
        ],
        "state_equations": [
            {"state": "x", "rhs": "-r*x + g*u01"},
            {"state": "v01", "rhs": "h*tanh(s*x) - d*v01"},
        ],
        "observation_mappings": [{"channel": "v01", "expression": "v01"}],
        "parameters": [
            {"name": n, "scope": "global", "role": "rate"}
            for n in ("r", "g", "h", "s", "d")
        ],
        "initial_conditions": [
            {"state": "x", "scope": "global", "expression": "known_x"},
            {"state": "v01", "scope": "global", "expression": "v01"},
        ],
    }
    raw = {
        "candidate": candidate,
        "context": {
            "targets": ["v01"],
            "external_inputs": ["u01"],
            "fixed_covariates": ["known_x"],
            "fitted_initialization": True,
        },
        "initialization_plan": {"rules": {}},
    }
    system = system_for(raw)
    t = np.linspace(0, 2, 9)
    row = Trajectory(
        "train_anchor",
        t,
        {"v01": 1 - np.exp(-t)},
        {},
        {"u01": np.ones(len(t))},
        {"known_x": 0.5},
        {},
    )
    train = DatasetSplit(SplitName.TRAIN, (row,), "scale-test")
    start = np.ones(len(system.names))
    partition = partition_and_exponents(system)
    common, z0 = common_initialization(
        system,
        train,
        start,
        FitConfig(integration_method="Radau"),
        partition,
        target_variables=100,
        minimum_intervals=4,
        warmup_seconds=0.001,
    )
    return raw, system, train, common, z0


def test_scale_affinity_boundaries_and_units():
    _, system, train, common, z0 = fixture()
    p = common["partition"]
    assert p["shapes"] == ["s"]
    assert dict(zip(system.names, p["parameter_exponents"], strict=True)) == {
        "r": 0,
        "g": 1,
        "h": 0,
        "s": -1,
        "d": 0,
    }
    audit = algebraic_scale_audit(system, train, np.ones(5), p)
    assert audit["pass"] and not audit["common_scale_symmetry_with_fixed_boundaries"]
    first = ConditionalCollocation(system, train, common, z0, 1e4)
    scaled = ConditionalCollocation(system, train, common, z0, 1e4, unit_factor=2)
    z = z0 + np.random.default_rng(2).normal(0, 0.1, size=len(z0))
    q = first.q0.copy()
    np.testing.assert_allclose(
        first.residual(z, q), scaled.residual(z, q), rtol=1e-11, atol=1e-11
    )
    # At arbitrary nodes/shapes, the residual is affine in the weight block.
    a, r = first.linear_design(z, q)
    changed = q.copy()
    changed[first.wi] *= 1.3
    np.testing.assert_allclose(
        first.residual(z, changed),
        r + a @ (changed[first.wi] - q[first.wi]),
        atol=1e-10,
    )
    proposal, report = first.linear_step(z, q, np.full(5, 1e-12), np.full(5, 1e12))
    assert report["accepted"] and first.loss(z, proposal) <= first.loss(z, q)
    np.testing.assert_array_equal(proposal[first.si], q[first.si])


@pytest.mark.parametrize("arm", ["joint", "alternating"])
def test_real_block_optimizer_and_corruption(tmp_path, arm):
    _, system, train, common, z0 = fixture()
    problem = ConditionalCollocation(system, train, common, z0, 1e4)
    result = optimize(
        problem,
        np.full(5, 1e-12),
        np.full(5, 1e12),
        arm=arm,
        seconds=15,
        maximum_iterations=12,
        cycles=2,
        block_iterations=6,
        root=tmp_path,
        identity="test",
    )
    assert result["objective"] <= result["initial_objective"]
    record, z, q = load_point(tmp_path, "test")
    assert record["known_initials_fixed"] and len(z) == len(z0) and len(q) == 5
    with pytest.raises(ValueError, match="identity"):
        load_point(tmp_path, "changed")
    (tmp_path / record["array"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_point(tmp_path, "test")


def test_nonlinear_weight_dependence_and_wrong_split_rejected():
    raw, _, train, common, z0 = fixture()
    raw = deepcopy(raw)
    raw["candidate"]["state_equations"][0]["rhs"] = "-r*x + g**2*u01"
    with pytest.raises(ValueError, match=r"affine|integer"):
        partition_and_exponents(system_for(raw))
    _, system, _, _, _ = fixture()
    validation = DatasetSplit(
        SplitName.VALIDATION, train.trajectories, train.fingerprint
    )
    with pytest.raises(ValueError, match="training"):
        ConditionalCollocation(system, validation, common, z0, 1e4)


def test_tanh_products_biases_input_shapes_and_tied_skew():
    raw, _, training, _, _ = fixture()
    raw["candidate"]["states"].insert(1, {"name": "w", "kind": "latent"})
    equations = {
        "x": "-r*x+k*w+a*tanh(s*w+b)+v*tanh(p*x)*tanh(q*w)+g*tanh(kappa*u01)",
        "w": "-r2*w-k*x",
        "v01": "h*tanh(s_out*x)-d*v01",
    }
    names = (
        "r",
        "k",
        "a",
        "s",
        "b",
        "v",
        "p",
        "q",
        "g",
        "kappa",
        "r2",
        "h",
        "s_out",
        "d",
    )
    raw["candidate"]["state_equations"] = [
        {"state": n, "rhs": v} for n, v in equations.items()
    ]
    raw["candidate"]["parameters"] = [
        {"name": n, "scope": "global", "role": "coefficient"} for n in names
    ]
    raw["candidate"]["initial_conditions"].insert(
        1, {"state": "w", "scope": "global", "expression": "0"}
    )
    system = system_for(raw)
    partition = partition_and_exponents(system)
    dims = dict(zip(system.names, partition["parameter_exponents"], strict=True))
    assert all(dims[n] == 1 for n in ("a", "v", "g"))
    assert all(dims[n] == -1 for n in ("s", "p", "q", "s_out"))
    assert all(dims[n] == 0 for n in ("k", "r", "r2", "b", "kappa", "h", "d"))
    audit = algebraic_scale_audit(system, training, np.ones(len(names)), partition)
    assert audit["pass"]


def test_sparse_and_dense_augmented_rollouts_agree():
    from time import monotonic

    from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout

    _, sparse, training, _, _ = fixture()
    dense = SymbolicODE(sparse.model, solver_jacobian_format="dense")
    values = []
    for system in (sparse, dense):
        values.append(
            symbolic_rollout(
                system,
                training.trajectories[0],
                np.ones(5),
                FitConfig(integration_method="Radau"),
                monotonic() + 10,
                sensitivities=True,
            )
        )
    for i in (0, 1, 2):
        np.testing.assert_allclose(values[0][i], values[1][i], rtol=1e-7, atol=1e-9)


def test_invalid_trial_rejection_preserves_real_incumbent(tmp_path, monkeypatch):
    from time import monotonic

    from autoformalism.fitting import sensitivity_probe
    from autoformalism.fitting.feasibility import (
        EvaluationBudget,
        GuardedOracle,
        SensitivityUnavailable,
    )

    _, system, training, _, _ = fixture()
    budget = EvaluationBudget(monotonic() + 10, 5)
    oracle = GuardedOracle(
        system,
        training,
        1.0,
        FitConfig(integration_method="Radau"),
        tmp_path,
        budget.deadline,
        sensitivities=True,
        budget=budget,
        point_seconds=5,
        reject_invalid_trials=True,
    )
    assert np.isfinite(oracle(np.ones(5))).all()
    best = deepcopy(oracle.best)

    def failed(*args, **kwargs):
        raise ValueError("deliberate failed trial")

    monkeypatch.setattr(sensitivity_probe, "symbolic_rollout", failed)
    assert np.isinf(oracle(np.full(5, 2.0))).all()
    assert (
        oracle.best == best and oracle.last_jac is None and oracle.rejected_trials == 1
    )
    with pytest.raises(SensitivityUnavailable):
        oracle.jacobian(np.full(5, 2.0))
