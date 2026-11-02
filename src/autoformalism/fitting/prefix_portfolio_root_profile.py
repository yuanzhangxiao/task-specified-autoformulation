"""Certified affine initial-root profiling with bounded shared-parameter search.

Only trajectory roots are projected. Shared parameters remain outer coordinates;
the original initialization DAG, readouts, sample weights and parameter domains
are preserved. Conditional least-squares optimality does not certify outer
optimization, uniqueness, biological validity or held-out forecasting accuracy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from fractions import Fraction
from time import monotonic

import casadi as ca
import numpy as np
from scipy.optimize import minimize

from autoformalism.fitting import prefix_affine as affine
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_multioutput_profile as projection
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_timescale as timescale
from autoformalism.fitting.affine_precision import TransitionEvaluator
from autoformalism.fitting.public_fitting import content_sha256
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial
from autoformalism.fitting.simulation import trajectory_forcing

PROTOCOL = "portfolio-affine-initial-root-profile-1"


def _check(deadline: float) -> None:
    if not np.isfinite(deadline):
        raise ValueError("finite root-profile deadline required")
    if monotonic() >= deadline:
        raise TimeoutError("root-profile numerical deadline reached")


def _finite(value, label: str) -> np.ndarray:
    value = np.asarray(value, float)
    if not np.isfinite(value).all():
        raise NumericalInvalidTrial(f"nonfinite root-profile {label}")
    return value


def _root_chart(oracle, names: tuple[str, ...]) -> None:
    """Certify that existing coordinate lifts leave every original root alone."""
    reduction = oracle.reduction
    if isinstance(reduction, timescale.TimescaleReduction):
        if any(n in reduction.renamed for n in names):
            raise ValueError("timescale coordinate overlaps a trajectory root")
        reduction = reduction.base
    substitutions = timescale._lift_polynomials(reduction)
    roots = set(names)
    for name, polynomial in substitutions.items():
        used = {n for term in polynomial for n, _ in term}
        if name in roots:
            if polynomial != {((name, 1),): Fraction(1)}:
                raise ValueError("trajectory root lift is not the identity")
        elif used & roots:
            raise ValueError("shared coordinate lift depends on trajectory roots")


class RootProfile:
    """Exact affine basis and bounded LS for independent training initial roots."""

    def __init__(self, oracle):
        self.oracle, self.original = oracle, oracle.original
        original, system = self.original, self.original.system
        candidate = system.model.validated.candidate
        if candidate.constraints or original.problem.base_candidate.constraints:
            raise ValueError("root profiling does not support additional constraints")
        if not original.roots or not original.rows:
            raise ValueError("nonempty independent trajectory roots required")
        self.root_names = tuple(
            f"trajectory:{row.trajectory_id}:{name}"
            for row in original.rows
            for name in original.roots
        )
        _root_chart(oracle, self.root_names)
        if set(self.root_names) - set(oracle.names):
            raise ValueError("coordinate chart omitted an independent root")
        self.inner = tuple(oracle.names.index(n) for n in self.root_names)
        self.outer = tuple(i for i in range(len(oracle.names)) if i not in self.inner)
        original_indices = [original.names.index(n) for n in self.root_names]
        if not (
            np.array_equal(
                oracle.lower[list(self.inner)], original.lower[original_indices]
            )
            and np.array_equal(
                oracle.upper[list(self.inner)], original.upper[original_indices]
            )
        ):
            raise ValueError("root domains differ from original model")
        n, m, p = system.state_count, len(system.inputs), len(system.names)
        theta, x, u, t = (
            ca.SX.sym("root_profile_parameters", p),
            ca.SX.sym("root_profile_states", n),
            ca.SX.sym("root_profile_inputs", m),
            ca.SX.sym("root_profile_time"),
        )
        local_roots = tuple(system.names.index(name) for name in original.roots)
        gamma = theta[list(local_roots)]
        rhs = system.rhs(t, x, theta, u)
        readout = system.observe(t, x, theta, u)
        if ca.depends_on(ca.vertcat(rhs, readout), gamma):
            raise ValueError("trajectory roots must occur only in initialization")
        h = ca.jacobian(readout, x)
        if ca.depends_on(h, x):
            raise ValueError("readouts must be affine in the states")
        self.readout = ca.Function(
            "root_profile_readout",
            [t, theta, u],
            [h, ca.substitute(readout, x, ca.SX.zeros(n))],
        )
        self.initializers = []
        for i, row in enumerate(original.rows):
            initial = system.initial_symbolic(row, theta)
            derivative = ca.jacobian(initial, gamma)
            if ca.depends_on(derivative, gamma):
                raise ValueError("initialization must be jointly affine in roots")
            basis = ca.horzcat(
                ca.substitute(initial, gamma, ca.SX.zeros(len(local_roots))), derivative
            )
            self.initializers.append(
                ca.Function(f"root_profile_initial_{i}", [theta], [basis])
            )
        self.propagation = affine._Propagation(
            system, (), precision_policy=original.precision_policy
        )
        self.transitions = TransitionEvaluator(original.precision_policy)
        self.audit = {
            "protocol": PROTOCOL,
            "problem_sha256": content_sha256(original.problem),
            "training_sha256": content_sha256(original.training),
            "reduction_sha256": oracle.reduction.receipt["sha256"],
            "root_names": list(self.root_names),
            "outer_names": [oracle.names[i] for i in self.outer],
            "target_scales": dict(original.scales),
            "affine_dynamics": self.propagation.audit,
            "initializers_jointly_affine_in_roots": True,
            "readouts_affine_in_states_and_root_independent": True,
            "root_domain_and_scope_preserved": True,
            "sample_mask": "all supplied training samples and target channels",
            "linear_solver": "column-scaled BVLS with normalized KKT check",
            "derivatives_used_for_outer_search": False,
            "global_minimum_certified": False,
            "held_out_data_accessed": False,
        }
        self.audit["sha256"] = content_sha256(self.audit)
        self.counts = dict.fromkeys(
            (
                "profile_calls_started",
                "profile_calls_completed",
                "profile_calls_failed",
                "basis_calls_started",
                "basis_calls_completed",
                "basis_calls_failed",
                "pairing_calls_started",
                "pairing_calls_completed",
                "pairing_calls_failed",
            ),
            0,
        )

    def design(self, parameters: Mapping[str, float], deadline: float) -> list[dict]:
        """Build row-separable residual = design @ roots + offset exactly in form."""
        _check(deadline)
        vector = self.oracle.reduction.lift(self.oracle.vector(parameters))
        self.original.vector(self.original.parameters(vector))
        original, system = self.original, self.original.system
        n, k, m = system.state_count, len(original.roots), len(system.inputs)
        scales = np.asarray([original.scales[c] for c in system.channels])
        blocks, cache = [], {}
        for row, initializer in zip(original.rows, self.initializers, strict=True):
            _check(deadline)
            local = original.local_vector(vector, row)
            generator = _finite(self.propagation.generator(local), "generator")
            state = _finite(initializer(local), "initial basis")
            forcing = trajectory_forcing(system.model, row)
            inputs = np.asarray(
                [[forcing.value(name, t) for name in system.inputs] for t in row.time]
            ).reshape(len(row.time), m)
            outputs = []
            for i, time in enumerate(row.time):
                _check(deadline)
                if i:
                    dt = float(time - row.time[i - 1])
                    key = generator.tobytes(), dt
                    if key not in cache:
                        cache[key] = self.transitions(generator * dt, [], deadline)[0]
                    extended = np.zeros((generator.shape[0], k + 1))
                    extended[:n] = state
                    extended[n : n + m, 0] = inputs[i - 1]
                    extended[n + m : n + 2 * m, 0] = (inputs[i] - inputs[i - 1]) / dt
                    extended[-1, 0] = 1.0
                    state = _finite((cache[key] @ extended)[:n], "propagated basis")
                h, offset = self.readout(time, local, inputs[i])
                values = _finite(np.asarray(h) @ state, "readout basis")
                values[:, 0] += np.asarray(offset).ravel()
                outputs.append(values)
            values = _finite(outputs, "weighted outputs")
            observed = np.column_stack([row.targets[c] for c in system.channels])
            names = [f"trajectory:{row.trajectory_id}:{r}" for r in original.roots]
            indices = [self.oracle.names.index(name) for name in names]
            blocks.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "indices": indices,
                    "design": (values[:, :, 1:] / scales[None, :, None]).reshape(-1, k),
                    "offset": ((values[:, :, 0] - observed) / scales).ravel(),
                    "lower": self.oracle.lower[indices],
                    "upper": self.oracle.upper[indices],
                }
            )
        _check(deadline)
        return blocks

    def _call(self, kind: str, function):
        self.counts[f"{kind}_calls_started"] += 1
        try:
            value = function()
        except BaseException:
            self.counts[f"{kind}_calls_failed"] += 1
            raise
        self.counts[f"{kind}_calls_completed"] += 1
        return value

    def profile(self, parameters: Mapping[str, float], deadline: float) -> dict:
        """Fit roots, pair with full rollout, and return admissible parameters."""
        return self._call("profile", lambda: self._profile(parameters, deadline))

    def _profile(self, parameters: Mapping[str, float], deadline: float) -> dict:
        vector = self.oracle.vector(parameters).copy()
        blocks = self._call("basis", lambda: self.design(parameters, deadline))
        residuals, audits = [], []
        for block in blocks:
            _check(deadline)
            a, b = block["design"], block["offset"]
            lower, upper = block["lower"], block["upper"]
            fixed = lower == upper
            fitted = lower.copy()
            if np.any(~fixed):
                fitted[~fixed], residual, _, audit = projection._project(
                    a[:, ~fixed],
                    b + a[:, fixed] @ lower[fixed],
                    None,
                    None,
                    lower[~fixed],
                    upper[~fixed],
                    deadline,
                )
            else:
                residual = a @ fitted + b
                audit = {"status": "all_roots_fixed", "bounds_satisfied": True}
            vector[block["indices"]] = fitted
            residuals.append(residual)
            audits.append({"trajectory_id": block["trajectory_id"], **audit})
        fitted_parameters = self.oracle.parameters(vector)
        self.oracle.vector(fitted_parameters)
        predicted = _finite(np.concatenate(residuals), "projected residual")
        actual, _ = self._call(
            "pairing", lambda: self.oracle.evaluate(vector, deadline, jacobian=False)
        )
        _check(deadline)
        discrepancy = float(np.max(np.abs(predicted - actual), initial=0))
        predicted_loss, actual_loss = (
            float(np.mean(predicted**2)),
            float(np.mean(actual**2)),
        )
        if (
            not np.isfinite([discrepancy, predicted_loss, actual_loss]).all()
            or discrepancy > portfolio.PAIRING_TOLERANCE
            or abs(predicted_loss - actual_loss) > max(1e-12, 1e-8 * actual_loss)
        ):
            raise NumericalInvalidTrial("profile basis/full-rollout pairing differs")
        return {
            "parameters": fitted_parameters,
            "original_parameters": continuation.to_original(
                self.original, self.oracle, fitted_parameters
            ),
            "training_nmse": actual_loss,
            "projected_training_nmse": predicted_loss,
            "maximum_pairing_discrepancy": discrepancy,
            "root_solve": audits,
            "certificate_sha256": self.audit["sha256"],
            "conditional_root_optimality_checked": True,
            "outer_convergence_certified": False,
            "independent_original_verification_required": True,
        }


def root_profile_qualification(
    oracle,
    parameters: Mapping[str, float],
    seconds: float,
    checkpoint: Callable[[dict], None],
) -> dict:
    """Bound certification, conditional solve and paired rollout in one receipt."""
    budget = execution._budget(seconds, min(seconds + 120, 3600))
    profiler, interrupted = None, None
    result = {
        "protocol": PROTOCOL,
        "passed": False,
        "parameters_sha256": content_sha256(parameters),
        "verification_calls_attempted": 0,
        "verification_calls_started": 0,
        "verification_calls_completed": 0,
        "verification_calls_failed": 0,
        "global_minimum_certified": False,
        "independent_original_verification_required": True,
    }
    budget.callback(lambda: checkpoint(deepcopy(result)))
    try:
        profiler = budget.numerical(lambda _: RootProfile(oracle))
        result["root_profile_certificate"] = deepcopy(profiler.audit)
        budget.callback(lambda: checkpoint(deepcopy(result)))
        endpoint = budget.numerical(lambda d: profiler.profile(parameters, d))
        budget.check()
        result.update(endpoint)
        result["passed"] = endpoint["conditional_root_optimality_checked"]
    except Exception as error:
        continuation._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "qualification")
    if profiler is not None:
        result["profile_execution"] = deepcopy(profiler.counts)
        for name in ("started", "completed", "failed"):
            result[f"verification_calls_{name}"] = sum(
                profiler.counts[f"{scope}_calls_{name}"]
                for scope in ("basis", "pairing")
            )
        result["verification_calls_attempted"] = result["verification_calls_started"]
    return continuation._finish(
        result, budget, checkpoint, "qualification", interrupted
    )


def root_profile_search(
    oracle,
    parameters: Mapping[str, float],
    seconds: float,
    maximum_calls: int,
    checkpoint: Callable[[dict], None],
    *,
    fixed: Mapping[str, float] | None = None,
) -> dict:
    """Profile initial roots at each bounded Powell shared-parameter trial.

    The caller must journal this operation with a nonrenewable allowance. Each
    improved feasible checkpoint is saved immediately; interrupted work is not
    automatically replayed. Two different seeds receive independent equal limits.
    """
    if (
        not 0 < seconds <= 3600
        or type(maximum_calls) is not int
        or not 1 <= maximum_calls <= 10000
    ):
        raise ValueError("root-profile search needs <=3600 seconds and <=10000 calls")
    budget = execution._budget(seconds, min(seconds + 120, 3600))
    result = {
        "protocol": PROTOCOL,
        "best": None,
        "unoptimized_control": None,
        "initial_root_profile": None,
        "search_calls_attempted": 0,
        "search_calls_started": 0,
        "search_calls_completed": 0,
        "search_calls_failed": 0,
        "control_calls_started": 0,
        "control_calls_completed": 0,
        "control_calls_failed": 0,
        "invalid_trials": 0,
        "budget_exhausted": False,
        "stop_reason": "not_started",
        "derivatives_used": False,
        "optimizer": "Powell over shared coordinates; bounded LS over roots",
        "global_minimum_certified": False,
        "independent_original_verification_required": True,
        "initial_sweep_completed": False,
        "accepted_shared_iteration_history": [],
        "outer_coordinates_varied": [],
    }
    profiler, interrupted = None, None

    def save():
        if profiler is not None:
            result["profile_execution"] = deepcopy(profiler.counts)
        budget.callback(
            lambda: checkpoint(deepcopy(result) | {"timing": budget.snapshot()})
        )

    try:
        profiler = budget.numerical(lambda _: RootProfile(oracle))
        result["root_profile_certificate"] = deepcopy(profiler.audit)
        fixed = dict(fixed or {})
        if set(fixed) - {oracle.names[i] for i in profiler.outer}:
            raise ValueError("fixed coordinates must be existing shared coordinates")
        start = dict(parameters) | fixed
        initial = oracle.vector(start)
        free = np.asarray(
            [
                i
                for i in profiler.outer
                if oracle.names[i] not in fixed and oracle.lower[i] < oracle.upper[i]
            ],
            int,
        )
        units = np.maximum(abs(initial), oracle.units)[free]
        taus = {b.tau for b in getattr(oracle.reduction, "blocks", ())}
        logarithmic = np.asarray([oracle.names[i] in taus for i in free], bool)
        bounds = []
        for j, i in enumerate(free):
            if logarithmic[j]:
                lo, hi = oracle.lower[i], oracle.upper[i]
                bounds.append(
                    (
                        np.log(lo) - np.log(initial[i]) if lo > 0 else -np.inf,
                        np.log(hi) - np.log(initial[i]) if np.isfinite(hi) else np.inf,
                    )
                )
            else:
                bounds.append(
                    (
                        (oracle.lower[i] - initial[i]) / units[j],
                        (oracle.upper[i] - initial[i]) / units[j],
                    )
                )

        def point_for(q):
            point = initial.copy()
            q = np.asarray(q)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                for j, i in enumerate(free):
                    point[i] = (
                        np.exp(np.log(initial[i]) + q[j])
                        if logarithmic[j]
                        else initial[i] + q[j] * units[j]
                    )
            # The exact seed is retained without an exp(log(tau)) round-trip.
            point[free[q == 0]] = initial[free[q == 0]]
            if not np.isfinite(point).all() or np.any(point[free[logarithmic]] <= 0):
                raise NumericalInvalidTrial("nonrepresentable logarithmic timescale")
            return point

        result.update(
            optimizer_start=start,
            fixed_coordinates=fixed,
            outer_coordinates=[oracle.names[i] for i in free],
            outer_coordinate_chart={
                "log_timescales": [
                    oracle.names[i] for j, i in enumerate(free) if logarithmic[j]
                ],
                "rule": ("log(tau/tau_seed); other coordinates scaled at frozen seed"),
                "units": units.tolist(),
                "positive_timescale_floor_added": False,
                "initial_sweep": "each available coordinate at +/-1 before Powell",
            },
        )
        result["control_calls_started"] += 1
        save()
        try:
            control, _ = budget.numerical(
                lambda d: oracle.evaluate(initial, d, jacobian=False)
            )
            result["control_calls_completed"] += 1
        except BaseException:
            result["control_calls_failed"] += 1
            raise
        budget.check()
        with np.errstate(over="ignore", invalid="ignore"):
            control_loss = float(np.mean(control**2))
        if not np.isfinite(control_loss):
            raise NumericalInvalidTrial("nonfinite unoptimized control objective")
        result["unoptimized_control"] = {
            "parameters": start,
            "training_nmse": control_loss,
        }
        result["best"] = deepcopy(result["unoptimized_control"])
        save()

        best_q = np.zeros(len(free))

        def objective(q):
            nonlocal best_q
            budget.check()
            if result["search_calls_started"] >= maximum_calls:
                raise TimeoutError("root-profile call allowance exhausted")
            result["search_calls_attempted"] += 1
            result["search_calls_started"] += 1
            save()
            varied = {oracle.names[i] for j, i in enumerate(free) if q[j] != 0}
            result["outer_coordinates_varied"] = sorted(
                set(result["outer_coordinates_varied"]) | varied
            )
            try:
                point = point_for(q)
                endpoint = budget.numerical(
                    lambda d: profiler.profile(oracle.parameters(point), d)
                )
                budget.check()
            except (ValueError, NumericalInvalidTrial) as error:
                result["search_calls_failed"] += 1
                result["invalid_trials"] += 1
                result["last_invalid_trial"] = f"{type(error).__name__}: {error}"[-500:]
                save()
                return float("inf")
            except BaseException:
                result["search_calls_failed"] += 1
                save()
                raise
            result["search_calls_completed"] += 1
            if result["initial_root_profile"] is None:
                result["initial_root_profile"] = deepcopy(endpoint)
            if endpoint["training_nmse"] < result["best"]["training_nmse"]:
                result["best"] = deepcopy(endpoint)
                best_q = np.asarray(q).copy()
            save()
            return endpoint["training_nmse"]

        if not np.isfinite(objective(np.zeros(len(free)))):
            raise ValueError("initial root profile is unavailable")
        if len(free):
            for j, (lo, hi) in enumerate(bounds):
                for direction in (-1.0, 1.0):
                    q = np.zeros(len(free))
                    q[j] = np.clip(direction, lo, hi)
                    if q[j] != 0:
                        objective(q)
            result["initial_sweep_completed"] = True
            save()

            def accepted(q):
                point = point_for(q)
                result["accepted_shared_iteration_history"].append(
                    {
                        "shared_parameters": {
                            oracle.names[i]: float(point[i]) for i in free
                        },
                        "scope": "shared coordinates only; no root/stationarity claim",
                    }
                )
                save()

            fit = minimize(
                objective,
                best_q,
                method="Powell",
                bounds=bounds,
                callback=accepted,
                options={
                    "maxfev": max(1, maximum_calls - result["search_calls_started"]),
                    "xtol": 1e-8,
                    "ftol": 1e-10,
                },
            )
            result.update(
                stop_reason="optimizer_stopped",
                optimizer_success=bool(fit.success),
                optimizer_message=str(fit.message),
                optimizer_iterations=int(fit.nit),
                budget_exhausted=bool(result["search_calls_started"] >= maximum_calls),
            )
        else:
            result.update(
                stop_reason="no_free_shared_coordinates", optimizer_success=True
            )
    except Exception as error:
        continuation._invalidate(result, error, "search")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "search")
    if profiler is not None:
        result["profile_execution"] = deepcopy(profiler.counts)
    return continuation._finish(result, budget, checkpoint, "search", interrupted)
