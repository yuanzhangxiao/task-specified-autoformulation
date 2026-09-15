"""Preflight: common node freeze, dimensional covariance, and a small live solve."""

from __future__ import annotations

import io
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from autoformalism.data import DatasetSplit, SplitName, Trajectory
from autoformalism.fitting.conditional_collocation import (
    ConditionalCollocation,
    common_initialization,
)
from autoformalism.fitting.conditional_optimizer import optimize
from autoformalism.fitting.conditional_scaling import (
    algebraic_scale_audit,
    partition_and_exponents,
    system_for,
)
from autoformalism.fitting.feasibility import EvaluationBudget, GuardedOracle
from autoformalism.fitting.sensitivity_probe import symbolic_rollout
from autoformalism.rebuttal.fitter_diagnostic import (
    _finite_payload,
    _write_bytes,
    read_json,
    sha256,
    write_json,
)
from autoformalism.rebuttal.piecewise_campaign import pack_split, unpack_split
from autoformalism.rebuttal.scaled_alternating import (
    ScaledAlternatingPlan,
    checked,
    verify,
)
from autoformalism.staged_topology import content_hash


def small_problem():
    """Independent two-state control: pass only output samples to the fitter."""
    candidate = {
        "candidate_id": "conditional_plumbing_control",
        "parent_candidate_id": None,
        "states": [
            {"name": "x", "kind": "latent"},
            {"name": "v01", "kind": "observed"},
        ],
        "state_equations": [
            {"state": "x", "rhs": "-r*x+g*u01"},
            {"state": "v01", "rhs": "h*tanh(s*x)-d*v01"},
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
    problem = {
        "candidate": candidate,
        "context": {
            "targets": ["v01"],
            "external_inputs": ["u01"],
            "fixed_covariates": ["known_x"],
            "fitted_initialization": True,
        },
        "initialization_plan": {"rules": {}},
        "start": {"r": 1.0, "g": 1.0, "h": 1.0, "s": 1.0, "d": 1.0},
    }
    splits = {}
    t = np.linspace(0, 4, 25)
    for name, initials in (("train", (0.0, 0.5, 1.0)), ("val", (0.25,))):
        rows = []
        for i, initial in enumerate(initials):
            u = 0.7 + 0.2 * np.sin(t * (i + 1))

            def rhs(tt, x, u=u):
                return [
                    -0.7 * x[0] + 1.2 * np.interp(tt, t, u),
                    2 * np.tanh(0.8 * x[0]) - 0.4 * x[1],
                ]

            solved = solve_ivp(
                rhs,
                (t[0], t[-1]),
                [initial, 0.0],
                t_eval=t,
                method="DOP853",
                rtol=1e-11,
                atol=1e-13,
                max_step=0.03,
            )
            if not solved.success:
                raise ValueError("independent smoke generator failed")
            rows.append(
                Trajectory(
                    f"{name}_{i}",
                    t.copy(),
                    targets={"v01": solved.y[1]},
                    auxiliaries={},
                    external_inputs={"u01": u},
                    fixed_covariates={"known_x": initial},
                    derivatives={},
                )
            )
        data = DatasetSplit(SplitName(name), tuple(rows), "conditional-smoke-" + name)
        splits[name] = pack_split(data)
    problem["splits"] = splits
    return problem


def smoke(root: Path) -> dict:
    """Exercise both native optimizers followed by physical sensitivity fitting."""
    problem = small_problem()
    identity = content_hash(problem)
    if (root / "result.json").exists():
        return checked(root / "result.json", identity)
    if (root / "started.json").exists():
        raise ValueError("interrupted smoke; no silent fresh attempt in this output")
    write_json(root / "started.json", {"identity": identity}, immutable=True)
    system = system_for(problem)
    train = unpack_split(problem["splits"]["train"])
    validation = unpack_split(problem["splits"]["val"])
    plan = ScaledAlternatingPlan(
        initializer_seconds=15,
        screen_seconds=5,
        refinement_seconds=30,
        initializer_iterations=30,
        alternating_cycles=3,
        block_iterations=10,
        target_variables=300,
        minimum_intervals=8,
        warmup_seconds=0.01,
    )
    partition = partition_and_exponents(system)
    start = np.array([problem["start"][n] for n in system.names])
    common, z = common_initialization(
        system,
        train,
        start,
        plan.settings(),
        partition,
        target_variables=plan.target_variables,
        minimum_intervals=plan.minimum_intervals,
        warmup_seconds=0.01,
    )
    graph = ConditionalCollocation(system, train, common, z, plan.penalty)
    report = {}
    for arm in ("joint", "alternating"):
        directory = root / arm
        budget = EvaluationBudget(monotonic() + 45, 100)
        oracle = GuardedOracle(
            system,
            train,
            common["output_scale"],
            plan.settings(),
            directory / "calls",
            budget.deadline,
            sensitivities=True,
            budget=budget,
            point_seconds=15,
            reject_invalid_trials=True,
        )
        init = optimize(
            graph,
            oracle.lower,
            oracle.upper,
            arm=arm,
            seconds=15,
            maximum_iterations=30,
            cycles=3,
            block_iterations=10,
            root=directory / "initializer",
            identity=arm,
        )
        least_squares(
            oracle,
            oracle.vector(init["parameters"]),
            jac=oracle.jacobian,
            bounds=(oracle.lower, oracle.upper),
            ftol=None,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=80,
        )
        if not oracle.best:
            raise ValueError("smoke refinement found no valid parameters")
        theta = oracle.vector(oracle.best["parameters"])
        scores = {}
        agreement = {}
        for data in (train, validation):
            predictions = []
            for method in ("Radau", "BDF"):
                predicted = []
                for row in data.trajectories:
                    yy, _, _, _ = symbolic_rollout(
                        system,
                        row,
                        theta,
                        plan.settings(tight=True, method=method),
                        monotonic() + 15,
                        sensitivities=False,
                    )
                    predicted.extend(yy[:, 0])
                predictions.append(np.array(predicted))
            y = np.concatenate([r.targets["v01"] for r in data.trajectories])
            scores[data.name.value] = float(
                np.mean(((predictions[0] - y) / common["output_scale"]) ** 2)
            )
            agreement[data.name.value] = float(
                np.max(abs(predictions[0] - predictions[1])) / common["output_scale"]
            )
        report[arm] = {
            "scores": scores,
            "replay_difference": agreement,
            "initializer_objective": init["objective"],
            "pass": max(scores.values()) <= 1e-4 and max(agreement.values()) <= 1e-4,
        }
    result = {
        "identity": identity,
        "pass": all(v["pass"] for v in report.values()),
        "arms": report,
        "hidden_trajectory_labels_used": False,
    }
    write_json(root / "result.json", result, immutable=True)
    return result


def scale_direction(system, training, start, partition, settings, scale):
    """Diagnostic only: keep physical boundaries fixed along the scale direction."""
    e = np.asarray(partition["parameter_exponents"])
    h = 0.01
    records = []
    arrays = []
    for factor in (np.exp(-h), np.exp(h)):
        deadline = monotonic() + 60
        values = []
        point = start * factor**e
        try:
            for row in training.trajectories:
                prediction, _, _, _ = symbolic_rollout(
                    system, row, point, settings, deadline, sensitivities=False
                )
                values.extend(prediction[:, 0] / scale)
            arrays.append(np.array(values))
            records.append({"factor": float(factor), "status": "complete"})
        except (ValueError, RuntimeError, TimeoutError, ArithmeticError) as error:
            arrays.append(None)
            records.append(
                {
                    "factor": float(factor),
                    "status": "unavailable",
                    "message": str(error),
                }
            )
    rms = (
        None
        if any(x is None for x in arrays)
        else float(np.sqrt(np.mean(((arrays[1] - arrays[0]) / (2 * h)) ** 2)))
    )
    return {
        "points": records,
        "scaled_output_derivative_rms": rms,
        "step_log_scale": h,
        "fixed_physical_boundaries": True,
        "used_for_selection": False,
    }


def gate(output: Path):
    frozen = verify(output)
    root = output / "gate"
    result_path = root / "result.json"
    if result_path.exists():
        return checked(result_path, frozen["identity"])
    problem = read_json(output / "problem.json")
    system = system_for(problem)
    train = unpack_split(problem["splits"]["train"])
    plan = ScaledAlternatingPlan.model_validate(frozen["plan"])
    partition = partition_and_exponents(system)
    if len(partition["weights"]) != 26 or len(partition["shapes"]) != 22:
        raise ValueError("reference partition must be 26 weights and 22 shapes")
    start = np.array([problem["start"][n] for n in system.names])
    if (output / "common.json").exists():
        common = read_json(output / "common.json")
        nodes = np.load(output / "nodes.npy", allow_pickle=False)
        if (
            common["identity"]
            != content_hash({k: v for k, v in common.items() if k != "identity"})
            or content_hash(nodes.tolist()) != common["nodes_identity"]
        ):
            raise ValueError("saved common initialization changed")
    else:
        if (root / "common.started.json").exists():
            raise ValueError(
                "interrupted common freeze; retain output and use a new explicit run"
            )
        write_json(
            root / "common.started.json",
            {"identity": frozen["identity"]},
            immutable=True,
        )
        common, nodes = common_initialization(
            system,
            train,
            start,
            plan.settings(),
            partition,
            target_variables=plan.target_variables,
            minimum_intervals=plan.minimum_intervals,
            warmup_seconds=plan.warmup_seconds,
        )
        common.pop("identity")
        common["parameter_names"] = list(system.names)
        common["identity"] = content_hash(common)
        buf = io.BytesIO()
        np.save(buf, nodes, allow_pickle=False)
        _write_bytes(output / "nodes.npy", buf.getvalue(), immutable=True)
        write_json(output / "common.json", common, immutable=True)
    algebra = algebraic_scale_audit(system, train, start, partition)
    base = ConditionalCollocation(system, train, common, nodes, plan.penalty)
    rng = np.random.default_rng(20260916)
    z = nodes + rng.normal(0, 0.01, len(nodes))
    q = base.q0.copy()
    r = np.asarray(base.residual(z, q)).ravel()
    invariance = {}
    for factor in (0.5, 2.0):
        transformed = ConditionalCollocation(
            system, train, common, nodes, plan.penalty, unit_factor=factor
        )
        rr = np.asarray(transformed.residual(z, q)).ravel()
        invariance[str(factor)] = float(np.max(abs(rr - r)) / max(1, np.max(abs(r))))
        del transformed
    a, rr = base.linear_design(z, q)
    new = q.copy()
    new[base.wi] *= 1.03
    predicted = np.asarray(rr).ravel() + np.asarray(a) @ (new[base.wi] - q[base.wi])
    affinity = float(
        np.max(abs(np.asarray(base.residual(z, new)).ravel() - predicted))
        / max(1, np.max(abs(r)))
    )
    del base, a
    direction = scale_direction(
        system, train, start, partition, plan.settings(), common["output_scale"]
    )
    live = smoke(root / "smoke")
    result = _finite_payload(
        {
            "identity": frozen["identity"],
            "common_identity": common["identity"],
            "nodes_sha256": sha256(output / "nodes.npy"),
            "partition": partition,
            "algebraic_scale_audit": algebra,
            "dimensionless_residual_unit_difference": invariance,
            "conditional_affinity_difference": affinity,
            "anchored_scale_direction": direction,
            "smoke": live,
            "pass": algebra["pass"]
            and max(invariance.values()) < 1e-10
            and affinity < 1e-10
            and live["pass"],
            "gauge_fixed": False,
            "test_data_opened": False,
            "llm_calls": 0,
        }
    )
    write_json(result_path, result, immutable=True)
    return result
