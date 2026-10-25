"""Training-only joint refits of two reviewed instantaneous detention models.

The branch-preserving working boxes are numerical experiment declarations, not
scientific priors. Native Python, held-out observations and latent initials are
never used. Searches share the existing bounded recovery/scaling machinery.
"""

from __future__ import annotations

from collections import OrderedDict
from time import monotonic, process_time

import numpy as np
from pydantic import Field, model_validator

from autoformalism.data import TrainingScaler
from autoformalism.fitting import llm_sr_numerical_control as supplied
from autoformalism.fitting import poor_fit
from autoformalism.fitting import prefix_fit_adequacy as adequacy
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import recovery_numerics as recovery
from autoformalism.fitting.llm_sr_instantaneous import compile_current_value
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.optimizer_scaling import RelativeScaling
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.base import FiniteFloat, StrictSchema

PROTOCOL = "llm-sr-detention-refit-1"
NAMES = {
    14: ("k0", "p0", "kw", "pw", "beta", "h_max", "gamma_c", "k_spill"),
    15: ("C", "beta", "eps_s", "h_max", "Q_max", "dt", "k_loss", "k_gw", "kappa"),
}


class Policy(StrictSchema):
    """Frozen accuracy, local probe and guarded-scaling settings."""

    relative_tolerance: FiniteFloat = Field(default=1e-9, gt=0, le=1e-7)
    absolute_tolerance: FiniteFloat = Field(default=1e-11, gt=0, le=1e-9)
    tight_relative_tolerance: FiniteFloat = Field(default=1e-11, gt=0, le=1e-9)
    tight_absolute_tolerance: FiniteFloat = Field(default=1e-13, gt=0, le=1e-11)
    finite_difference_relative_step: FiniteFloat = Field(default=2e-5, gt=0, le=1e-4)
    qualification_seconds: FiniteFloat = Field(default=240, gt=0, le=300)
    maximum_calls: int = Field(default=1000, ge=1, le=10000)
    target_nmse: FiniteFloat = Field(default=1e-8, gt=0, le=1e-4)
    scaling: RelativeScaling = Field(default_factory=RelativeScaling)
    stopping: poor_fit.Policy = Field(default_factory=poor_fit.Policy)
    derivative: adequacy.Policy = Field(
        default_factory=lambda: adequacy.Policy(
            precision_aware=True,
            local_adaptive=True,
            local_bounds_aware=True,
            diagnostic_precision=True,
            local_max_scales=6,
        )
    )

    @model_validator(mode="after")
    def independent_precision(self):
        if self.tight_relative_tolerance >= self.relative_tolerance or (
            self.tight_absolute_tolerance >= self.absolute_tolerance
        ):
            raise ValueError("verification must tighten both integration tolerances")
        if not (self.derivative.precision_aware and self.derivative.local_bounds_aware):
            raise ValueError(
                "refitting requires precision-aware local derivative checks"
            )
        return self


class IntegrationSettings(StrictSchema):
    relative_tolerance: FiniteFloat
    absolute_tolerance: FiniteFloat


def domain_policy(item, context, training) -> dict:
    """Declare branch-preserving boxes from the unchanged public training scales."""
    packed, _ = supplied.checked_training(training, context)
    if item.task not in NAMES or set(item.parameters) != set(NAMES[item.task]):
        raise ValueError("only the reviewed eight/nine-coordinate detention models")
    if context.targets != ("h_down",) or item.hidden_states:
        raise ValueError(
            "detention controls require observed h_down and no hidden states"
        )
    if any(row.fixed_covariates.get("area_down", 0) <= 1 for row in packed.rows):
        raise ValueError("dropped native area guard requires area_down > 1")
    height = max(abs(y) for row in packed.rows for y in row.targets["h_down"])
    span = max(row.time[-1] - row.time[0] for row in packed.rows)
    interval = min(
        b - a
        for row in packed.rows
        for a, b in zip(row.time[:-1], row.time[1:], strict=True)
    )
    rate = max(
        height / span,
        max(
            abs(q) / row.fixed_covariates["area_down"]
            for row in packed.rows
            for q in row.external_inputs["inflow_down"]
        ),
    )
    if height <= 0 or rate <= 0:
        raise ValueError("positive public height and rate scales are required")
    h, r = height, rate
    if item.task == 14:
        bounds = {name: [-100 * r, 100 * r] for name in ("k0", "kw", "k_spill")}
        bounds |= {name: [1e-3, 8.0] for name in ("p0", "pw", "gamma_c")}
        bounds |= {"beta": [-100 * r / h, 100 * r / h], "h_max": [1e-3 * h, 1e3 * h]}
        units = {
            "k0": r,
            "p0": 1.0,
            "kw": r,
            "pw": 1.0,
            "beta": r / h,
            "h_max": h,
            "gamma_c": 1.0,
            "k_spill": r,
        }
        fixed = {"native_eps_index7": 1.0, "unused_native_index9": 1.0}
    else:
        bounds = {name: [0.0, 100 * r] for name in ("C", "Q_max", "k_loss")}
        bounds |= {
            "k_gw": [-100 * r, 100 * r],
            "kappa": [0.0, 100 / h],
            "beta": [1e-3, 8.0],
            "eps_s": [1e-3 * h, 1e3 * h],
            "h_max": [1e-3 * h, 1e3 * h],
            "dt": [1e-3 * interval, 100 * interval],
        }
        units = {
            "C": r,
            "beta": 1.0,
            "eps_s": h,
            "h_max": h,
            "Q_max": r,
            "dt": 1.0,
            "k_loss": r,
            "k_gw": r,
            "kappa": 1 / h,
        }
        fixed = {
            "effective_warning_boost": 0.0,
            "absent_sigma": True,
            "absent_upstream_coupling": True,
        }
    if any(
        not bounds[n][0] <= item.parameters[n] <= bounds[n][1] for n in NAMES[item.task]
    ):
        raise ValueError("supplied coefficients lie outside the declared working box")
    return {
        "protocol": "llm-sr-detention-branch-domains-1",
        "task": item.task,
        "names": list(NAMES[item.task]),
        "bounds": bounds,
        "units": units,
        "public_anchors": {
            "height": h,
            "rate": r,
            "sampling_interval": interval,
            "span": span,
        },
        "frozen_native_branches": fixed,
        "scientific_bounds": False,
        "scope": (
            "Numerical working box inside the reviewed native branch; "
            "not the full native family."
        ),
        "dt_is_integration_step": False,
    }


def deterministic_starts(item, context, training, *, seed=20261025) -> list[dict]:
    """Freeze supplied, two perturbed and two public-scale generic starts."""
    domain = domain_policy(item, context, training)
    names = domain["names"]
    source = np.asarray([item.parameters[n] for n in names])
    units = np.asarray([domain["units"][n] for n in names])
    lower, upper = np.asarray([domain["bounds"][n] for n in names]).T
    records = [
        {
            "start_id": "supplied",
            "start_kind": "supplied",
            "parameters": dict(item.parameters),
        }
    ]
    for index, sign in enumerate((-1, 1)):
        point = source + sign * 0.25 * np.where(
            np.arange(len(names)) % 2, -1, 1
        ) * np.maximum(np.abs(source), units)
        point = np.clip(
            point, lower + (upper - lower) * 1e-9, upper - (upper - lower) * 1e-9
        )
        records.append(
            {
                "start_id": f"perturbed_{index}",
                "start_kind": "perturbed",
                "parameters": dict(zip(names, point.tolist(), strict=True)),
            }
        )
    h, r = (domain["public_anchors"][n] for n in ("height", "rate"))
    generic = (
        {
            "k0": r / h,
            "p0": 1.0,
            "kw": r / h,
            "pw": 2.0,
            "beta": 0.0,
            "h_max": 2 * h,
            "gamma_c": 1.0,
            "k_spill": 0.0,
        }
        if item.task == 14
        else {
            "C": r / h,
            "beta": 1.0,
            "eps_s": h,
            "h_max": 2 * h,
            "Q_max": r,
            "dt": 1.0,
            "k_loss": r / 2,
            "k_gw": r / 2,
            "kappa": 1 / h,
        }
    )
    rng = np.random.default_rng(seed + item.task)
    for index in range(2):
        point = np.asarray([generic[n] for n in names])
        point *= np.exp(rng.uniform(-0.4, 0.4, len(names)))
        zero = point == 0
        point[zero] = rng.uniform(-0.25, 0.25, int(zero.sum())) * units[zero]
        point = np.clip(
            point, lower + (upper - lower) * 1e-9, upper - (upper - lower) * 1e-9
        )
        records.append(
            {
                "start_id": f"generic_{index}",
                "start_kind": "generic",
                "parameters": dict(zip(names, point.tolist(), strict=True)),
            }
        )
    return records


class DetentionOracle:
    """Accurate free rollout residuals with bounded local numerical derivatives."""

    def __init__(self, item, context, training, policy=None, checkpoint=None):
        self.began, self.cpu, self.role = monotonic(), process_time(), "search"
        self.policy = Policy.model_validate(
            (policy or Policy()).model_dump(mode="json")
        )
        self.domain = domain_policy(item, context, training)
        self.item, self.context = item, context
        self.training, self.split = supplied.checked_training(training, context)
        self.model = compile_current_value(item, context)
        self.names = tuple(self.domain["names"])
        self.start = dict(item.parameters)
        self.units = np.asarray([self.domain["units"][n] for n in self.names])
        self.lower, self.upper = np.asarray(
            [self.domain["bounds"][n] for n in self.names]
        ).T
        self.scale = (
            TrainingScaler().fit(self.split).scales["target:h_down"].standard_deviation
        )
        self.settings = IntegrationSettings(
            relative_tolerance=self.policy.relative_tolerance,
            absolute_tolerance=self.policy.absolute_tolerance,
        )
        self.checkpoint = checkpoint or (lambda _: None)
        self.cost = {
            "residual_calls_started": 0,
            "residual_calls_completed": 0,
            "residual_calls_failed": 0,
            "residual_cache_hits": 0,
            "jacobian_calls": 0,
            "trajectory_integrations": 0,
        }
        self.cache, self.jacobian_cache = OrderedDict(), OrderedDict()
        self.best_residual = None
        self.retain_base_checkpoints = False

    def _progress(self, extra=None):
        self.checkpoint(
            _accounting(self, self.began, self.cpu, self.role)
            | {"numerical_cost": dict(self.cost)}
            | (extra or {})
        )

    def vector(self, parameters):
        if set(parameters) != set(self.names):
            raise ValueError("parameter roster differs from the reviewed control")
        point = np.asarray([parameters[n] for n in self.names], dtype=float)
        if (
            not np.isfinite(point).all()
            or np.any(point < self.lower)
            or np.any(point > self.upper)
        ):
            raise ValueError("parameter vector outside the declared working box")
        return point

    def parameters(self, vector):
        point = np.asarray(vector, dtype=float)
        if point.shape != self.units.shape:
            raise ValueError("parameter vector shape differs")
        result = dict(zip(self.names, point.tolist(), strict=True))
        self.vector(result)
        return result

    def local_coordinate_scales(self, vector):
        point = self.vector(self.parameters(vector))
        scales = np.maximum(np.abs(point), self.units)
        for name in ("h_max", "eps_s", "dt"):
            if name in self.names:
                i = self.names.index(name)
                scales[i] = min(scales[i], abs(point[i]))
        return scales

    def tighten(self):
        self.settings = IntegrationSettings(
            relative_tolerance=self.policy.tight_relative_tolerance,
            absolute_tolerance=self.policy.tight_absolute_tolerance,
        )

    def _key(self, vector, method):
        return (
            vector.tobytes(),
            method,
            self.settings.relative_tolerance,
            self.settings.absolute_tolerance,
        )

    def residual(self, vector, deadline, method="Radau"):
        if monotonic() >= deadline:
            raise TimeoutError("detention replay deadline exhausted")
        parameters = self.parameters(vector)
        vector = self.vector(parameters)
        key = self._key(vector, method)
        if key in self.cache:
            self.cost["residual_cache_hits"] += 1
            return self.cache[key].copy()
        self.cost["residual_calls_started"] += 1
        self._progress()
        values = []
        try:
            config = FitConfig(
                integration_backend="solve_ivp",
                integration_method=method,
                relative_tolerance=self.settings.relative_tolerance,
                absolute_tolerance=self.settings.absolute_tolerance,
                maximum_wall_time_seconds=max(0.001, deadline - monotonic()),
                allow_derivative_regression=False,
            )
            for row in self.split.trajectories:
                self.cost["trajectory_integrations"] += 1
                outcome = simulate_trajectory(
                    self.model,
                    row,
                    parameters,
                    {},
                    config,
                    deadline=deadline,
                    reset_observed_states=False,
                )
                if monotonic() >= deadline:
                    raise TimeoutError("detention replay deadline exhausted")
                if not outcome.success:
                    raise recovery.NumericalInvalidTrial(
                        outcome.message or "rollout failed"
                    )
                values.append(
                    (outcome.predictions["h_down"] - row.targets["h_down"]) / self.scale
                )
            residual = np.concatenate(values)
            if not np.isfinite(residual).all():
                raise recovery.NumericalInvalidTrial("nonfinite detention residual")
            with np.errstate(over="ignore", invalid="ignore"):
                loss = float(np.mean(residual**2))
            if not np.isfinite(loss):
                raise recovery.NumericalInvalidTrial("nonfinite detention objective")
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError):
            self.cost["residual_calls_failed"] += 1
            self._progress()
            raise
        self.cost["residual_calls_completed"] += 1
        self.cache[key] = residual.copy()
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        self._progress()
        return residual

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        point = self.vector(self.parameters(vector))
        method = method or "Radau"
        residual = self.residual(point, deadline, method)
        if not jacobian:
            return residual, None
        # Preserve search trial base points before derivatives can time out.
        # Residual-only finite-difference/qualification probes never enter here.
        if self.retain_base_checkpoints and method == "Radau":
            loss = float(np.mean(residual**2))
            if self.best_residual is None or loss < self.best_residual["training_nmse"]:
                self.best_residual = {
                    "parameters": self.parameters(point),
                    "training_nmse": loss,
                    "origin": "search_base_rollout_before_derivative_completion",
                    "independent_verification_required": True,
                }
                self._progress({"best_rollout_checkpoint": self.best_residual})
        key = self._key(point, method)
        if key in self.jacobian_cache:
            return residual, self.jacobian_cache[key].copy()
        self.cost["jacobian_calls"] += 1
        columns = []
        scales = self.local_coordinate_scales(point)
        for i in range(len(point)):
            step = min(
                self.policy.finite_difference_relative_step * scales[i],
                (self.upper[i] - self.lower[i]) / 4,
            )
            left, right = point[i] - self.lower[i], self.upper[i] - point[i]
            if min(left, right) >= step:
                plus, minus = point.copy(), point.copy()
                plus[i], minus[i] = point[i] + step, point[i] - step
                column = (
                    self.residual(plus, deadline, method)
                    - self.residual(minus, deadline, method)
                ) / (plus[i] - minus[i])
            else:
                sign = 1 if right >= left else -1
                step = min(step, max(left, right) / 2)
                one, two = point.copy(), point.copy()
                one[i], two[i] = point[i] + sign * step, point[i] + sign * 2 * step
                if one[i] == point[i] or two[i] == one[i]:
                    raise recovery.NumericalInvalidTrial(
                        "unrepresentable local derivative step"
                    )
                column = (
                    sign
                    * (
                        -3 * residual
                        + 4 * self.residual(one, deadline, method)
                        - self.residual(two, deadline, method)
                    )
                    / (2 * step)
                )
            columns.append(column)
        result = np.column_stack(columns)
        if not np.isfinite(result).all():
            raise recovery.NumericalInvalidTrial("nonfinite detention derivative")
        self.jacobian_cache[key] = result.copy()
        if len(self.jacobian_cache) > 8:
            self.jacobian_cache.popitem(last=False)
        return residual, result


def _qualify(oracle, vector, deadline):
    """Check a distinct shrinking probe ladder against the optimizer FD Jacobian."""
    return adequacy.derivative_check(
        oracle.evaluate,
        vector,
        oracle.lower,
        oracle.upper,
        deadline,
        coordinate_units=oracle.units,
        policy=oracle.policy.derivative,
        residual_only=lambda x, d: oracle.evaluate(x, d, jacobian=False)[0],
        local_coordinate_scales=oracle.local_coordinate_scales(vector),
    )


def _clock(seconds):
    if not np.isfinite(seconds) or seconds <= 0 or seconds > 3600:
        raise ValueError("bounded positive stage seconds required")
    start, cpu = monotonic(), process_time()
    return start, cpu, start + seconds


def _accounting(oracle, began, cpu, role):
    """Expose standard journal counts; each call means one full training rollout."""
    cost = oracle.cost
    return {
        f"{role}_calls_attempted": cost["residual_calls_started"],
        f"{role}_calls_started": cost["residual_calls_started"],
        f"{role}_calls_completed": cost["residual_calls_completed"],
        f"{role}_calls_failed": cost["residual_calls_failed"],
        "timing": {
            "elapsed_wall_seconds": monotonic() - began,
            "elapsed_cpu_seconds": process_time() - cpu,
        },
        "call_scope": "full training residual integrations including derivative probes",
    }


def fit(item, context, training, start, *, seconds=600, checkpoint=None, policy=None):
    """Qualify local derivatives then run the existing guarded bounded TRF search."""
    began, cpu, deadline = _clock(seconds)
    state = {}

    def save(value):
        state.update(value)
        candidate = value.get("best_rollout_checkpoint") or (
            value.get("search") or {}
        ).get("best")
        if candidate and (
            state.get("best") is None
            or candidate["training_nmse"] < state["best"]["training_nmse"]
        ):
            state["best"] = candidate
        if checkpoint:
            checkpoint(dict(state))

    oracle = DetentionOracle(item, context, training, policy, save)
    vector = oracle.vector(start)
    result, checks = {}, []
    qualification_end = min(deadline, monotonic() + oracle.policy.qualification_seconds)
    try:
        residual = oracle.evaluate(vector, deadline, jacobian=False)[0]
        initial = {
            "parameters": dict(start),
            "training_nmse": float(np.mean(residual**2)),
        }
        save({"phase": "qualification", "best": initial})
        check = _qualify(oracle, vector, qualification_end)
        checks.append(public._jsonable(check))
        if not check.search_qualified and monotonic() < qualification_end:
            oracle.tighten()
            check = _qualify(oracle, vector, qualification_end)
            checks.append(public._jsonable(check))
        save({"derivative_qualification": checks})
        if check.search_qualified:
            save({"phase": "search"})
            oracle.retain_base_checkpoints = True
            result = recovery.search(
                oracle,
                start,
                deadline,
                lambda value: save({"search": value}),
                calls=oracle.policy.maximum_calls,
                target=oracle.policy.target_nmse,
                optimizer_scaling="guarded_relative",
                scaling_policy=oracle.policy.scaling,
                normalized_objective=True,
                poor_fit=oracle.policy.stopping,
                reject_invalid_trials=True,
                telemetry=True,
                record_steps=True,
                record_full_steps=True,
            )
            result["best"] = result.get("best") or initial
        else:
            result = {
                "best": initial,
                "stop_reason": "derivatives_unqualified",
                "calls": 0,
            }
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result = {
            "best": state.get("best"),
            "stop_reason": "budget_limited"
            if isinstance(error, TimeoutError)
            else "numerical_failure",
            "error": f"{type(error).__name__}: {error}",
        }
    if state.get("best") and (
        result.get("best") is None
        or state["best"]["training_nmse"] < result["best"]["training_nmse"]
    ):
        result["best"] = state["best"]
    return (
        result
        | _accounting(oracle, began, cpu, "search")
        | {
            "protocol": PROTOCOL,
            "derivative_qualification": checks,
            "domain": oracle.domain,
            "policy": oracle.policy.model_dump(mode="json"),
            "integration_settings": oracle.settings.model_dump(mode="json"),
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "cpu_seconds": process_time() - cpu,
            "budget_exhausted": monotonic() >= deadline,
            "independent_verification_required": True,
            "global_convergence_certified": False,
            "validation_or_test_access": False,
            "native_program_executed": False,
            "initial_values_fitted": False,
        }
    )


def verify(
    item, context, training, parameters, *, seconds=120, checkpoint=None, policy=None
):
    """Independently replay both accurate ODE solvers at one frozen endpoint."""
    began, cpu, deadline = _clock(seconds)
    oracle = DetentionOracle(item, context, training, policy, checkpoint)
    oracle.role = "verification"
    oracle.tighten()
    point = oracle.vector(parameters)
    try:
        first = oracle.evaluate(point, deadline, jacobian=False)[0]
        second = oracle.evaluate(point, deadline, jacobian=False, method="DOP853")[0]
        result = {
            "status": "complete",
            "parameters": dict(parameters),
            "training_nmse": float(np.mean(first**2)),
            "alternate_training_nmse": float(np.mean(second**2)),
            "solver_loss_discrepancy": abs(
                float(np.mean(first**2) - np.mean(second**2))
            ),
            "solver_prediction_discrepancy": float(np.max(np.abs(first - second))),
            "verification": "Radau_vs_DOP853_training_outputs_only",
        }
        result["usable_for_retention"] = recovery.usable(result)
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result = {
            "status": "unavailable",
            "parameters": dict(parameters),
            "usable_for_retention": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return (
        result
        | _accounting(oracle, began, cpu, "verification")
        | {
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "cpu_seconds": process_time() - cpu,
            "budget_exhausted": monotonic() >= deadline,
            "initial_values_fitted": False,
            "validation_or_test_access": False,
        }
    )


def diagnose(
    item, context, training, parameters, *, seconds=120, checkpoint=None, policy=None
):
    """Report local weak combinations, precision/bound caveats and the exact gauge."""
    began, cpu, deadline = _clock(seconds)
    oracle = DetentionOracle(item, context, training, policy, checkpoint)
    oracle.role = "verification"
    oracle.tighten()
    point = oracle.vector(parameters)
    result = {
        "parameters": dict(parameters),
        "coordinate_names": list(oracle.names),
        "coordinate_units": oracle.units.tolist(),
        "global_identifiability_certified": False,
        "coefficient_accuracy_claimed": False,
        "confidence_probability": None,
    }
    if item.task == 15:
        g, loss, kappa, hmax, dt = (
            parameters[n] for n in ("k_gw", "k_loss", "kappa", "h_max", "dt")
        )
        result["exact_ambiguity"] = {
            "rank_ceiling": 8,
            "declared_coordinates": 9,
            "combinations": {"b": g - loss, "d": g * kappa, "M": hmax + dt * g},
            "definitions": {
                "b": "k_gw-k_loss",
                "d": "k_gw*kappa",
                "M": "h_max+dt*k_gw",
            },
            "scope": (
                "Exact within the preserved native branch; individual "
                "four-coordinate recovery is not unique."
            ),
        }
    try:
        check = _qualify(oracle, point, deadline)
        result["derivative_qualification"] = public._jsonable(check)
        residual, jacobian = oracle.evaluate(point, deadline)
        scaled = jacobian * oracle.units
        _, singular, right = np.linalg.svd(scaled, full_matrices=False)
        active = np.minimum(
            point - oracle.lower, oracle.upper - point
        ) <= 1e-8 * np.maximum(oracle.units, np.abs(point))
        result |= {
            "status": "complete",
            "training_nmse": float(np.mean(residual**2)),
            "singular_values": singular.tolist(),
            "sampled_numeric_rank": int(
                np.sum(
                    singular
                    > oracle.policy.derivative.rank_relative_tolerance * singular[0]
                )
            ),
            "active_bounds": [
                n for n, flag in zip(oracle.names, active, strict=True) if flag
            ],
            "parameter_information_qualified": bool(
                check.parameter_information_qualified
                and not np.any(active)
                and item.task != 15
            ),
            "weak_combinations": [
                {
                    "singular_value": float(singular[i]),
                    "dimensionless_direction": dict(
                        zip(oracle.names, right[i].tolist(), strict=True)
                    ),
                }
                for i in range(max(0, len(singular) - 2), len(singular))
            ],
            "direction_interpretation": (
                "delta(parameter_i) = epsilon * coordinate_unit_i * direction_i"
            ),
            "scope": (
                "Local conditional derivative information, not global confidence; "
                "active bounds, unresolved kinks or numerical precision "
                "withhold stronger rank claims."
            ),
        }
        if item.task == 15 and abs(g) > np.finfo(float).tiny:
            tangent = np.zeros(len(point))
            for name, value in {
                "k_gw": 1.0,
                "k_loss": 1.0,
                "kappa": -kappa / g,
                "h_max": -dt,
            }.items():
                tangent[oracle.names.index(name)] = value
            result["exact_ambiguity"]["sampled_tangent_residual_rms"] = float(
                np.sqrt(np.mean((jacobian @ tangent) ** 2))
            )
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        result |= {
            "status": "unavailable",
            "parameter_information_qualified": False,
            "error": f"{type(error).__name__}: {error}",
        }
    return (
        result
        | _accounting(oracle, began, cpu, "verification")
        | {
            "numerical_cost": dict(oracle.cost),
            "wall_seconds": monotonic() - began,
            "cpu_seconds": process_time() - cpu,
            "budget_exhausted": monotonic() >= deadline,
            "validation_or_test_access": False,
        }
    )
