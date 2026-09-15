"""Restricted conditional-linearity and common-latent-unit audits.

This diagnostic accepts expanded polynomial/tanh equations, identity v01 output,
and fixed per-trajectory boundaries. It never evaluates expression text as code.
"""

from __future__ import annotations

import ast

import casadi as ca
import numpy as np

from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas import CandidateModel


def system_for(problem: dict) -> SymbolicODE:
    """Compile the frozen equations with sparse augmented solver Jacobians."""
    if problem.get("initialization_plan", {}).get("rules"):
        raise ValueError("this reference diagnostic requires fixed known initials")
    system = SymbolicODE(
        compile_candidate(
            CandidateModel.model_validate(problem["candidate"]),
            ValidationContext.model_validate(problem["context"]),
        ),
        solver_jacobian_format="sparse",
    )
    candidate = system.model.validated.candidate
    if (
        candidate.processes
        or system.has_piecewise
        or system.channels != ("v01",)
        or system.initial_parameter_names
        or len(candidate.observation_mappings) != 1
        or candidate.observation_mappings[0].expression != "v01"
    ):
        raise ValueError("requires expanded smooth equations and identity v01")
    return system


def partition_and_exponents(system: SymbolicODE) -> dict:
    """Infer unit exponents and certify affinity in parameters outside functions."""
    names = system.names
    p = len(names)
    state_dims = np.array([float(n != "v01") for n in system.model.state_names])
    env = {
        n: np.r_[np.zeros(p), d]
        for n, d in zip(system.model.state_names, state_dims, strict=True)
    }
    env.update({n: np.zeros(p + 1) for n in system.inputs})
    env[system.model.validated.context.time_symbol] = np.zeros(p + 1)
    for i, name in enumerate(names):
        env[name] = np.r_[np.eye(p)[i], 0.0]
    constraints, shape = [], set()

    def dimension(node):
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
            return np.zeros(p + 1)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return dimension(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id != "tanh" or len(node.args) != 1:
                raise ValueError("scale audit supports tanh calls only")
            shape.update(
                n.id
                for n in ast.walk(node.args[0])
                if isinstance(n, ast.Name) and n.id in names
            )
            constraints.append(dimension(node.args[0]))
            return np.zeros(p + 1)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or not isinstance(
                    node.right.value, (int, float)
                ):
                    raise ValueError("scale audit requires a literal exponent")
                return dimension(node.left) * node.right.value
            left, right = dimension(node.left), dimension(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                constraints.append(left - right)
                return left
            if isinstance(node.op, ast.Mult):
                return left + right
            if isinstance(node.op, ast.Div):
                return left - right
        raise ValueError("unsupported expression in scale audit")

    for equation in system.model.validated.candidate.state_equations:
        constraints.append(
            dimension(ast.parse(equation.rhs, mode="eval").body) - env[equation.state]
        )
    matrix = np.asarray(constraints)
    exponents, _, rank, _ = np.linalg.lstsq(matrix[:, :p], -matrix[:, p], rcond=None)
    if rank != p or np.max(abs(matrix[:, :p] @ exponents + matrix[:, p])) > 1e-10:
        raise ValueError("common-scale exponents are inconsistent or ambiguous")
    if np.max(abs(exponents - np.rint(exponents))) > 1e-10:
        raise ValueError("diagnostic expects integer common-scale exponents")
    exponents = np.rint(exponents)
    wi = [i for i, n in enumerate(names) if n not in shape]
    si = [i for i, n in enumerate(names) if n in shape]
    if not wi or not si:
        raise ValueError("requires both outer weights and nonlinear shapes")
    x, theta, t, u = (
        ca.SX.sym("x", system.state_count),
        ca.SX.sym("p", p),
        ca.SX.sym("t"),
        ca.SX.sym("u", len(system.inputs)),
    )
    derivative = ca.jacobian(system.rhs(t, x, theta, u), theta)[:, wi]
    if any(ca.depends_on(derivative, theta[i]) for i in wi):
        raise ValueError("outer-weight block is not conditionally affine")
    return {
        "weights": [names[i] for i in wi],
        "shapes": [names[i] for i in si],
        "weight_indices": wi,
        "shape_indices": si,
        "parameter_exponents": exponents.tolist(),
        "state_exponents": state_dims.tolist(),
        "conditional_affinity_certified": True,
    }


def algebraic_scale_audit(system, training, start, partition) -> dict:
    """Check unit covariance at finite probes, and expose fixed boundary anchors."""
    e = np.asarray(partition["parameter_exponents"])
    d = np.asarray(partition["state_exponents"])
    rng = np.random.default_rng(20260916)
    probes = [rng.normal(size=system.state_count) for _ in range(4)]
    worst = 0.0
    for factor in (0.5, 2.0):
        for x in probes:
            u = rng.normal(size=len(system.inputs))
            base = np.asarray(system.rhs(0.37, x, start, u)).ravel()
            scaled = np.asarray(
                system.rhs(0.37, x * factor**d, start * factor**e, u)
            ).ravel()
            error = np.max(abs(scaled - base * factor**d)) / max(1, np.max(abs(base)))
            worst = max(worst, float(error))
    anchors = []
    for row in training.trajectories:
        initial = system.initial_for(row, start)
        values = {
            n: float(x)
            for n, x, dim in zip(system.model.state_names, initial, d, strict=True)
            if dim and x != 0
        }
        if values:
            anchors.append(
                {"trajectory": row.trajectory_id, "fixed_hidden_initials": values}
            )
    return {
        "pass": worst < 1e-10,
        "maximum_scaled_rhs_difference": worst,
        "factors": [0.5, 2.0],
        "fixed_boundary_anchors": anchors,
        "common_scale_symmetry_with_fixed_boundaries": not bool(anchors),
        "initials_must_transform_for_equivalence": True,
        "training_only": True,
        "hidden_trajectory_labels_used": False,
    }
