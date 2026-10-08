"""Training-only uncertainty diagnostics; no statistical coverage is asserted."""

from __future__ import annotations

from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.optimize import least_squares

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import fitter
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.coordinates import NumericalCoordinates
from autoformalism.fitting.identifiable_campaign import scale_for
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.sensitivity_probe import SymbolicODE, symbolic_rollout
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest


class ConfidencePolicy(StrictSchema):
    """Finite, predeclared search; tolerances are numerical, not noise estimates."""

    restart_seconds: float = Field(default=120, ge=1, le=180)
    restart_calls: int = Field(default=160, ge=2, le=300)
    profile_seconds: float = Field(default=25, ge=1, le=40)
    profile_calls: int = Field(default=40, ge=2, le=100)
    check_seconds: float = Field(default=60, ge=1, le=90)
    replay_seconds: float = Field(default=180, ge=1, le=240)
    offsets: tuple[float, float] = (0.01, 0.1)
    loss_tolerances: tuple[float, float, float] = (1e-12, 1e-10, 1e-8)
    relative_tolerance: float = Field(default=1e-10, gt=0, le=1e-8)
    absolute_tolerance: float = Field(default=1e-12, gt=0, le=1e-10)
    residual_multiplier: float = Field(default=1e6, ge=1, le=1e8)
    stop_nmse: float = Field(default=1e-18, gt=0, le=1e-14)
    mode: Literal["noiseless_tolerance_profiles"] = "noiseless_tolerance_profiles"

    @model_validator(mode="after")
    def ordered(self):
        for values in (self.offsets, self.loss_tolerances):
            if any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("positive finite tolerances required")
            if list(values) != sorted(set(values)):
                raise ValueError("tolerances must be strictly increasing")
        if self.offsets[-1] >= 1:
            raise ValueError("profile fractions must be smaller than one")
        return self


class TrainingProblem(StrictSchema):
    """Allowlist prevents evaluator arrays and reference parameters entering fits."""

    request: PublicFitRequest
    training: TrainingOnlySplit
    coordinates: NumericalCoordinates
    start: dict[str, float]
    incumbent: dict[str, float]


class Rollouts:
    """Complete normalized residual/Jacobian, including fitted initial values."""

    def __init__(self, problem: TrainingProblem, policy: ConfidencePolicy):
        self.model = public._lower(problem.request)[0]
        self.system = SymbolicODE(self.model)
        self.names = self.system.names
        self.train = public.unpack_split(problem.training)
        self.scales = scale_for(self.train)
        self.settings = FitConfig(
            integration_method="DOP853",
            relative_tolerance=policy.relative_tolerance,
            absolute_tolerance=policy.absolute_tolerance,
        )
        variables = fitter._training_variables(self.model, self.train, self.settings)
        if [v.name for v in variables] != [f"parameter:{n}" for n in self.names]:
            raise ValueError("requires global coefficients and explicit initializers")
        self.lower = np.array([v.lower for v in variables])
        self.upper = np.array([v.upper for v in variables])
        self.units = problem.coordinates.arrays("parameters", self.names)[1]
        for point in (problem.start, problem.incumbent):
            self.vector(point)

    def vector(self, values: dict) -> np.ndarray:
        if set(values) != set(self.names):
            raise ValueError("incomplete parameter vector")
        x = np.array([values[n] for n in self.names])
        if not np.isfinite(x).all() or np.any(x < self.lower) or np.any(x > self.upper):
            raise ValueError("parameter vector outside declared domain")
        return x

    def parameters(self, x: np.ndarray) -> dict:
        return dict(zip(self.names, x.tolist(), strict=True))

    def evaluate(self, x, deadline, *, jacobian=True, method=None):
        """A failed trajectory is unavailable, never a residual penalty or zero J."""
        self.vector(self.parameters(x))
        settings = (
            self.settings.model_copy(update={"integration_method": method})
            if method
            else self.settings
        )
        residuals, jacobians = [], []
        scales = np.array([self.scales[c] for c in self.system.channels])
        for row in self.train.trajectories:
            y, j, _, _ = symbolic_rollout(
                self.system, row, x, settings, deadline, sensitivities=jacobian
            )
            observed = np.column_stack([row.targets[c] for c in self.system.channels])
            residuals.append(((y - observed) / scales).ravel())
            if jacobian:
                jacobians.append((j / scales[None, :, None]).reshape(-1, len(x)))
        return np.concatenate(residuals), np.vstack(jacobians) if jacobian else None


def operation(folder: Path, identity: dict, seconds: float, function) -> dict:
    """Resume each completed operation; interrupted operations never restart."""
    with public._lock(folder):
        start, result = folder / "started.json", folder / "result.json"
        identity = identity | {"allowance_seconds": seconds}
        if start.exists() and read_seal(start) != identity:
            raise ValueError("diagnostic operation identity differs")
        if result.exists():
            saved = read_seal(result)
            if saved["identity"] != identity:
                raise ValueError("diagnostic result identity differs")
            return saved
        if start.exists():
            value = {
                "status": "interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "calls": None,
                "accounting_complete": False,
            }
        else:
            seal(start, identity)
            begun, cpu = monotonic(), process_time()
            progress = {}

            def checkpoint(value):
                progress.update(value)
                public._write(folder / "progress.json", progress)

            try:
                value = {
                    "status": "complete",
                    "value": function(begun + seconds, checkpoint),
                }
            except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
                value = {
                    "status": "unavailable",
                    "value": None,
                    "error": str(error)[-1000:],
                }
            elapsed = monotonic() - begun
            value.update(
                wall_seconds=elapsed,
                cpu_seconds=process_time() - cpu,
                budget_charge_seconds=elapsed,
                calls=progress.get("calls", 0),
                accounting_complete=True,
            )
        saved = value | {"identity": identity, "budget_restarted": False}
        seal(result, saved)
        return saved


def optimize(oracle, points, policy, deadline, checkpoint, *, fixed=None, maximum=None):
    """Scaled joint TRF; optional fixed coefficient profiles all other unknowns."""
    fixed = fixed or {}
    free = [i for i, n in enumerate(oracle.names) if n not in fixed]
    maximum = maximum or policy.restart_calls
    best, calls, attempts = None, 0, []

    class Accurate(Exception):
        pass

    for number, point in enumerate(points):
        anchor = oracle.vector(point).copy()
        for name, value in fixed.items():
            anchor[oracle.names.index(name)] = value
        oracle.vector(oracle.parameters(anchor))
        cached = None
        limit = monotonic() + max(0, deadline - monotonic()) / (len(points) - number)
        call_limit = calls + max(1, (maximum - calls) // (len(points) - number))

        def physical(q, anchor=anchor):
            x = anchor.copy()
            x[free] += oracle.units[free] * q
            return x

        def evaluate(q, limit=limit, call_limit=call_limit):
            nonlocal best, calls, cached
            x = physical(q)
            if cached is not None and np.array_equal(x, cached[0]):
                return cached[1:]
            if calls >= call_limit or monotonic() >= limit:
                raise TimeoutError("optimization allowance exhausted")
            calls += 1
            checkpoint({"calls": calls, "best": best})
            r, j = oracle.evaluate(x, limit)
            loss = float(np.mean(r**2))
            if best is None or loss < best["training_nmse"]:
                best = {"parameters": oracle.parameters(x), "training_nmse": loss}
            checkpoint({"calls": calls, "best": best})
            cached = (x, r, j)
            if loss <= policy.stop_nmse:
                raise Accurate
            return r, j

        try:
            result = least_squares(
                lambda q: evaluate(q)[0] * policy.residual_multiplier,
                np.zeros(len(free)),
                jac=lambda q: evaluate(q)[1][:, free]
                * oracle.units[free]
                * policy.residual_multiplier,
                bounds=(
                    (oracle.lower[free] - anchor[free]) / oracle.units[free],
                    (oracle.upper[free] - anchor[free]) / oracle.units[free],
                ),
                x_scale="jac",
                ftol=None,
                xtol=1e-12,
                gtol=1e-10,
                max_nfev=maximum,
            )
            attempts.append(
                {
                    "status": "converged" if result.success else "limited",
                    "message": result.message,
                    "optimality": float(result.optimality),
                }
            )
        except Accurate:
            attempts.append({"status": "numerical_target_reached"})
        except (TimeoutError, ValueError, RuntimeError, ArithmeticError) as error:
            attempts.append({"status": "limited", "message": str(error)[-500:]})
        checkpoint({"calls": calls, "best": best, "attempts": attempts})
    return {
        "best": best,
        "calls": calls,
        "attempts": attempts,
        "fixed": fixed,
        "all_attempts_converged": all(a["status"] != "limited" for a in attempts),
    }


def sensitivity(jacobian: np.ndarray, units: np.ndarray, names: tuple) -> dict:
    """SVD in explicit optimizer units; a rank check is not an accuracy guarantee."""
    j = jacobian * units / np.sqrt(len(jacobian))
    _, singular, vt = np.linalg.svd(j, full_matrices=False)
    rank = int(np.sum(singular > max(singular[0] * 1e-8, 1e-15)))
    return {
        "parameters": list(names),
        "units": units.tolist(),
        "singular_values": singular.tolist(),
        "rank": rank,
        "parameter_count": len(names),
        "condition_number": float(singular[0] / singular[-1])
        if singular[-1] > 0
        else None,
        "column_norms": np.linalg.norm(j, axis=0).tolist(),
        "weak_direction": dict(zip(names, vt[-1].tolist(), strict=True)),
        "initials_included": True,
        "global_identifiability_certified": False,
    }


def inspect_point(oracle, parameters, deadline, checkpoint):
    """Independent algorithms check numerical accuracy before interpreting J."""
    x = oracle.vector(parameters)
    checkpoint({"calls": 1})
    r, j = oracle.evaluate(x, deadline)
    checkpoint({"calls": 2})
    other, _ = oracle.evaluate(x, deadline, jacobian=False, method="Radau")
    return {
        "parameters": parameters,
        "training_nmse": float(np.mean(r**2)),
        "alternate_training_nmse": float(np.mean(other**2)),
        "solver_loss_discrepancy": abs(float(np.mean(r**2) - np.mean(other**2))),
        "solver_prediction_discrepancy": float(np.max(np.abs(r - other))),
        "sensitivity": sensitivity(j, oracle.units, oracle.names),
    }


def profile_grid(oracle, center, offsets):
    """Both sides at 1%/10% of explicit scales; clipping is never a new sample."""
    x = oracle.vector(center)
    for i, name in enumerate(oracle.names):
        scale = (
            max(abs(x[i]), 0.01)
            if name not in oracle.system.initial_parameter_names
            else oracle.units[i]
        )
        for offset in offsets:
            for direction in (-1, 1):
                value = x[i] + direction * offset * scale
                yield {
                    "parameter": name,
                    "offset": direction * offset,
                    "scale": float(scale),
                    "value": float(value),
                    "in_domain": bool(oracle.lower[i] <= value <= oracle.upper[i]),
                }


def confidence_report(
    selected, inspected, profiles, anchor, policy, reliability_complete
):
    """Feasible alternatives are evidence; high local losses are not proofs."""
    result = {
        "probability": None,
        "statistical_coverage": None,
        "interpretation": "conditional numerical evidence; no confidence probability",
        "profile_center_changed": selected != anchor,
        "tolerances": [],
    }
    if inspected is None:
        return result | {"level": "unavailable"}
    noise = inspected["solver_loss_discrepancy"]
    sensitivity_rank = inspected.get("sensitivity", {})
    rank_deficient = sensitivity_rank.get("rank", 0) < sensitivity_rank.get(
        "parameter_count", 0
    )
    result["joint_sensitivity_rank_deficient"] = rank_deficient
    all_complete = (
        bool(profiles)
        and reliability_complete
        and all(p.get("optimizer_converged", False) for p in profiles)
    )
    for tolerance in policy.loss_tolerances:
        witnesses = []
        for p in profiles:
            check = p.get("checked")
            if (
                not check
                or check["solver_loss_discrepancy"] > tolerance / 10
                or check.get("solver_prediction_discrepancy", 0)
                > np.sqrt(tolerance) / 10
            ):
                continue
            delta = check["training_nmse"] - inspected["training_nmse"]
            if (
                delta <= tolerance
                and abs(p["value"] - selected[p["parameter"]]) >= 0.009 * p["scale"]
            ):
                witnesses.append(
                    {
                        "parameter": p["parameter"],
                        "value": p["value"],
                        "loss_increase": delta,
                    }
                )
        level = (
            "numerically_unresolved"
            if noise > tolerance / 10
            or inspected.get("solver_prediction_discrepancy", 0)
            > np.sqrt(tolerance) / 10
            else (
                "weakly_constrained"
                if witnesses
                else (
                    "search_incomplete"
                    if rank_deficient or not all_complete or selected != anchor
                    else "locally_supported_on_tested_grid"
                )
            )
        )
        result["tolerances"].append(
            {
                "loss_tolerance": tolerance,
                "level": level,
                "alternative_witnesses": witnesses,
            }
        )
    result["level"] = result["tolerances"][0]["level"]
    return result
