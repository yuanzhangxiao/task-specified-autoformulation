"""Conditional stable-branch coordinates and bounded nuisance profiles.

These coordinates preserve the saved equations and their original-domain lift.
A positive self-relaxation time is a diagnostic stratum, not an extra scientific
constraint: zero and unstable self-coefficients remain in the original family.
Profiles use residual-only optimization, never borrowed derivative evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import minimize

from autoformalism.fitting import prefix_construction_recovery as recovery
from autoformalism.fitting import prefix_continuation as continuation
from autoformalism.fitting import prefix_execution as execution
from autoformalism.fitting import prefix_gain_reduction as gains
from autoformalism.fitting import prefix_portfolio as portfolio
from autoformalism.fitting import prefix_portfolio_polish as polish
from autoformalism.fitting import prefix_structural_reduction as structural
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.prefix_additive_reduction import AdditiveReduction
from autoformalism.fitting.recovery_numerics import NumericalInvalidTrial

PROTOCOL = "prefix-conditional-timescale-profile-1"
DEFAULT_STATES = ("I", "X_insulin_action")


def _constant(value):
    return {(): Fraction.from_float(float(value))} if value else {}


def _variable(name):
    return {((name, 1),): Fraction(1)}


def _substitute(poly, replacements):
    """Compose bounded restricted polynomials without evaluating generated text."""
    result = {}
    for powers, coefficient in poly.items():
        term = {(): coefficient}
        for name, power in powers:
            replacement = replacements.get(name, _variable(name))
            if power < 0:
                if len(replacement) != 1:
                    raise ValueError("nonmonomial denominator in timescale proof")
                ((monomial, factor),) = replacement.items()
                if not factor:
                    raise ValueError("zero denominator in timescale proof")
                term = structural._product(
                    term,
                    {tuple((n, p * power) for n, p in monomial): factor**power},
                )
            else:
                for _ in range(power):
                    term = structural._product(term, replacement)
        result = structural._sum(result, term)
    return result


def _lift_polynomials(chart):
    """Prove the known constant sections algebraically, rather than by sampling."""
    if isinstance(chart, gains.GainReduction):
        fixed = {product.gain for product in chart.products}
        return {
            n: _constant(1) if n in fixed else _variable(n)
            for n in chart.original_names
        }
    if isinstance(chart, recovery.ActiveReduction):
        base = chart.gain_reduction
    elif isinstance(chart, (structural.StructuralReduction, AdditiveReduction)):
        if isinstance(chart, structural.StructuralReduction) and chart.decays:
            raise ValueError("nonlinear prior section is unsupported")
        base = chart.base
    else:
        raise ValueError("uncertified prior coordinate chart")
    replacements = {
        n: _variable(n) if n in chart.names else _constant(chart.anchor[i])
        for i, n in enumerate(base.names)
    }
    if isinstance(chart, AdditiveReduction):
        for group in chart.groups:
            weights = dict(zip(group.members, group.weights, strict=True))
            for name, value in group.constants:
                replacements[name] = _constant(value)
            pivot = structural._sum(
                _variable(group.coordinate),
                _constant(0),
            )
            offset = sum(
                Fraction.from_float(float(v)) * weights[n] for n, v in group.constants
            )
            pivot = structural._sum(pivot, {(): -offset} if offset else {})
            replacements[group.pivot] = {
                key: value / weights[group.pivot] for key, value in pivot.items()
            }
    return {
        name: _substitute(poly, replacements)
        for name, poly in _lift_polynomials(base).items()
    }


@dataclass(frozen=True)
class Coefficient:
    """An exact coefficient offset + multiplier * one prior coordinate."""

    coordinate: str
    multiplier: Fraction
    offset: Fraction
    driver: str


@dataclass(frozen=True)
class Relaxation:
    """One state self-coefficient and independent conditional forcing gains."""

    state: str
    decay: Coefficient
    drivers: tuple[Coefficient, ...]
    fixed_terms: tuple[tuple[str, str], ...]
    physical_forcing: tuple[Coefficient, ...] = ()

    @property
    def tau(self):
        return f"timescale:{self.state}"

    def gain(self, driver):
        return f"equilibrium_gain:{self.state}:{driver.driver}"


def _blocks(original, reduction, states, physical_forcing=()):
    trees = gains._expressions(original)
    processes = {
        k.removeprefix("process:"): v
        for k, v in trees.items()
        if k.startswith("process:")
    }
    substitutions = _lift_polynomials(reduction)
    parameter_names = {
        p.name for p in original.system.model.validated.candidate.parameters
    }
    used, blocks = set(), []
    for state in states:
        if f"rhs:{state}" not in trees:
            raise ValueError(f"missing declared timescale state: {state}")
        polynomial = _substitute(
            structural._polynomial(trees[f"rhs:{state}"], processes, parameter_names),
            substitutions,
        )
        coefficients = {}
        for powers, value in polynomial.items():
            params = tuple((n, p) for n, p in powers if n in reduction.names)
            forcing = tuple((n, p) for n, p in powers if n not in reduction.names)
            if len(forcing) > 1 or (forcing and forcing[0][1] != 1):
                raise ValueError("timescale block is not affine in its drivers")
            driver = forcing[0][0] if forcing else "1"
            coefficients[driver] = structural._sum(
                coefficients.get(driver, {}), {params: value}
            )
        declared, fixed, unscaled = {}, [], []
        for driver, poly in coefficients.items():
            variable = {key: value for key, value in poly.items() if key}
            if not variable:
                fixed.append((driver, str(poly.get((), Fraction(0)))))
                continue
            if len(variable) != 1:
                raise ValueError("coefficient is not a single exact affine coordinate")
            ((powers, multiplier),) = variable.items()
            if len(powers) != 1 or powers[0][1] != 1:
                raise ValueError("coefficient depends nonlinearly on parameters")
            name = powers[0][0]
            index = reduction.names.index(name)
            coordinate = Coefficient(
                name, multiplier, poly.get((), Fraction(0)), driver
            )
            if name in physical_forcing:
                if (
                    driver not in original.problem.context.external_inputs
                    or name in used
                    or multiplier != 1
                    or coordinate.offset != 0
                ):
                    raise ValueError(
                        "physical forcing requires a pure external-input gain"
                    )
                used.add(name)
                unscaled.append(coordinate)
                continue
            if (reduction.lower[index], reduction.upper[index]) != (-np.inf, np.inf):
                raise ValueError(
                    "timescale pilot requires unrestricted coefficient domains"
                )
            if name in used:
                raise ValueError("overlapping timescale coefficient coordinates")
            used.add(name)
            declared[driver] = coordinate
        if state not in declared:
            raise ValueError("no variable self-relaxation coefficient")
        decay = declared.pop(state)
        blocks.append(
            Relaxation(
                state, decay, tuple(declared.values()), tuple(fixed), tuple(unscaled)
            )
        )
    if set(physical_forcing) != {
        d.coordinate for b in blocks for d in b.physical_forcing
    }:
        raise ValueError("missing physical forcing coordinate")
    return tuple(blocks)


class TimescaleReduction:
    """Invertible timescale/gain chart on the negative-self-coefficient stratum."""

    def __init__(self, original, base, states, *, physical_forcing=()):
        if not states or len(set(states)) != len(states):
            raise ValueError("distinct nonempty timescale states required")
        self.base, self.blocks = base, _blocks(original, base, states, physical_forcing)
        self.renamed = {}
        for block in self.blocks:
            self.renamed[block.decay.coordinate] = block.tau
            self.renamed.update({d.coordinate: block.gain(d) for d in block.drivers})
        self.names = tuple(self.renamed.get(n, n) for n in base.names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("timescale coordinate name collision")
        taus = {b.tau for b in self.blocks}
        self.lower = np.asarray(
            [
                0.0 if self.renamed.get(n) in taus else lo
                for n, lo in zip(base.names, base.lower, strict=True)
            ]
        )
        self.upper = base.upper.copy()

    @property
    def receipt(self):
        record = {
            "protocol": PROTOCOL,
            "base_reduction": self.base.receipt,
            "effective_names": list(self.names),
            "blocks": [
                {
                    "state": b.state,
                    "timescale": b.tau,
                    "self_coefficient": {
                        "coordinate": b.decay.coordinate,
                        "multiplier": str(b.decay.multiplier),
                        "offset": str(b.decay.offset),
                    },
                    "gains": [
                        {
                            "coordinate": b.gain(d),
                            "driver": d.driver,
                            "original_coordinate": d.coordinate,
                            "multiplier": str(d.multiplier),
                            "offset": str(d.offset),
                        }
                        for d in b.drivers
                    ],
                    "unchanged_fixed_forcing_terms": dict(b.fixed_terms),
                    **(
                        {
                            "physical_forcing_coordinates": {
                                d.driver: d.coordinate for d in b.physical_forcing
                            }
                        }
                        if b.physical_forcing
                        else {}
                    ),
                }
                for b in self.blocks
            ],
            "lift": (
                "self=-1/tau; driver_coefficient=gain/tau; "
                "invert exact affine coefficients"
            ),
            "domain": "tau>0, no arbitrary lower bound; original lift domains checked",
            "coefficient_domain_policy": (
                "selected effective coefficients must be unrestricted; "
                "bounded or sign-constrained coefficients defer"
            ),
            "scope": "conditional negative-self-coefficient stratum only",
            "zero_and_unstable_branches": (
                "retained in original family, unassessed here"
            ),
            "coupled_system_stability_certified": False,
            "interpretation": (
                "self-relaxation at fixed drivers, not full-system eigenmode timescales"
            ),
            "original_equations_unchanged": True,
        }
        return record | {"sha256": public.content_sha256(record)}

    def vector(self, parameters):
        return self._checked(gains._mapping_vector(parameters, self.names))

    def _checked(self, vector):
        point = gains._checked_vector(vector, self.names, self.lower, self.upper)
        if any(point[self.names.index(b.tau)] <= 0 for b in self.blocks):
            raise ValueError("strictly positive diagnostic timescale required")
        return point

    def reduce(self, original_vector):
        values = dict(
            zip(self.base.names, self.base.reduce(original_vector), strict=True)
        )
        result = {self.renamed.get(n, n): v for n, v in values.items()}
        for block in self.blocks:
            decay = (
                float(block.decay.offset)
                + float(block.decay.multiplier) * values[block.decay.coordinate]
            )
            if not np.isfinite(decay) or decay >= 0:
                raise ValueError(
                    "endpoint outside conditional stable self-coefficient branch"
                )
            tau = -1 / decay
            result[block.tau] = tau
            for driver in block.drivers:
                result[block.gain(driver)] = tau * (
                    float(driver.offset)
                    + float(driver.multiplier) * values[driver.coordinate]
                )
        return self.vector(result)

    def section(self, vector, *, jacobian=True):
        point = self._checked(vector)
        values = dict(zip(self.base.names, point, strict=True))
        derivative = np.eye(len(point)) if jacobian else None
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            for block in self.blocks:
                i = self.names.index(block.tau)
                tau = point[i]
                slope = float(block.decay.multiplier)
                values[block.decay.coordinate] = (
                    -1 / tau - float(block.decay.offset)
                ) / slope
                realized_self = (
                    float(block.decay.offset) + slope * values[block.decay.coordinate]
                )
                if not realized_self < 0:
                    raise NumericalInvalidTrial(
                        "timescale reciprocal lost at original coefficient resolution"
                    )
                if jacobian:
                    derivative[i, i] = (1 / tau) ** 2 / slope
                for driver in block.drivers:
                    j = self.names.index(block.gain(driver))
                    gain, slope = point[j], float(driver.multiplier)
                    values[driver.coordinate] = (
                        gain / tau - float(driver.offset)
                    ) / slope
                    if jacobian:
                        derivative[j, j] = 1 / (tau * slope)
                        derivative[j, i] = -(gain / tau) / (tau * slope)
        if not np.isfinite(list(values.values())).all() or (
            jacobian and not np.isfinite(derivative).all()
        ):
            raise NumericalInvalidTrial("nonrepresentable timescale lift")
        return self.base.vector(values), derivative

    def lift(self, vector):
        return self.base.lift(self.section(vector, jacobian=False)[0])

    def lift_jacobian(self, vector):
        point, derivative = self.section(vector)
        return self.base.lift_jacobian(point) @ derivative


class _TimescaleOracle:
    """Compose the exact chart with the certified original-equation affine oracle."""

    def __init__(self, base, states, *, physical_forcing=()):
        self.base_oracle, self.original = base, base.original
        self.reduction = TimescaleReduction(
            self.original, base.reduction, states, physical_forcing=physical_forcing
        )
        self.names = self.reduction.names
        self.lower, self.upper = self.reduction.lower, self.reduction.upper
        self.incumbent_original_parameters = deepcopy(
            base.incumbent_original_parameters
        )
        self.start = self.parameters(
            self.reduction.reduce(
                self.original.vector(self.incumbent_original_parameters)
            )
        )
        self.units = np.maximum(np.abs(self.vector(self.start)), 1.0)
        self.sensitive_names = tuple(b.tau for b in self.reduction.blocks)
        for name in self.sensitive_names:
            self.units[self.names.index(name)] = self.start[name]

    def __getattr__(self, name):
        return getattr(self.base_oracle, name)

    def vector(self, parameters):
        return self.reduction.vector(parameters)

    def parameters(self, vector):
        return dict(
            zip(self.names, self.reduction._checked(vector).tolist(), strict=True)
        )

    def local_coordinate_scales(self, vector):
        point = self.reduction._checked(vector)
        scales = np.maximum(np.abs(point), self.units)
        for name in self.sensitive_names:
            index = self.names.index(name)
            scales[index] = min(abs(point[index]), self.units[index])
        return scales

    def evaluate(self, vector, deadline, *, jacobian=True, method=None):
        point, chain = self.reduction.section(vector, jacobian=jacobian)
        residual, derivative = self.base_oracle.evaluate(
            point, deadline, jacobian=jacobian, method=method
        )
        return residual, derivative @ chain if jacobian else None


def make_oracle(
    problem,
    training,
    numerical,
    retained,
    *,
    states=DEFAULT_STATES,
    physical_forcing=(),
):
    """Create a diagnostic chart after existing certified gain/additive reductions."""
    original, base = polish.make_oracle(problem, training, numerical, retained)
    chart = _TimescaleOracle(
        base.oracle, tuple(states), physical_forcing=physical_forcing
    )
    audit = base.portfolio_audit | {
        "protocol": PROTOCOL,
        "reduction": chart.reduction.receipt,
        "optimizer_units": chart.units.tolist(),
        "profile_selection_policy": "diagnostic_only_no_automatic_incumbent_promotion",
    }
    return original, portfolio.PortfolioOracle(chart, audit)


def chart_equivalence(original, oracle, seconds, checkpoint):
    """Check the saved source, chart lift and independent original Radau rollout."""
    budget = execution._budget(seconds, seconds + 60)
    result = {
        "passed": False,
        "no_search_performed": True,
        "chart_sha256": oracle.reduction.receipt["sha256"],
        "parameters_sha256": public.content_sha256(oracle.start),
        "jacobian_qualification_claimed": False,
        "calls_started": 0,
        "calls_completed": 0,
    }
    interrupted = None

    def call(function):
        result["calls_started"] += 1
        budget.callback(lambda: checkpoint(deepcopy(result)))
        value = budget.numerical(function)
        budget.check()
        result["calls_completed"] += 1
        return value[0]

    try:
        point = oracle.vector(oracle.start)
        lifted = oracle.reduction.lift(point)
        source = call(
            lambda d: original.evaluate(
                original.vector(oracle.incumbent_original_parameters), d, jacobian=False
            )
        )
        chart = call(lambda d: oracle.evaluate(point, d, jacobian=False))
        reference = call(
            lambda d: original.evaluate(lifted, d, jacobian=False, method="Radau")
        )
        source_gap = float(np.max(np.abs(source - chart), initial=0))
        ode_gap = float(np.max(np.abs(reference - chart), initial=0))
        loss, other = float(np.mean(chart**2)), float(np.mean(reference**2))
        result.update(
            source_lift_prediction_discrepancy=source_gap,
            original_Radau_prediction_discrepancy=ode_gap,
            training_nmse=loss,
            alternate_training_nmse=other,
            passed=bool(
                np.isfinite([source_gap, ode_gap, loss, other]).all()
                and max(source_gap, ode_gap) <= portfolio.PAIRING_TOLERANCE
                and abs(loss - other) <= max(1e-12, 1e-8 * max(loss, other))
            ),
        )
    except Exception as error:
        continuation._invalidate(result, error, "qualification")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "qualification")
    result["original_equations_verified"] = result["passed"]
    return continuation._finish(
        result, budget, checkpoint, "qualification", interrupted
    )


def profile_grid(oracle):
    """Return frozen native-time grids; every point restarts from the same endpoint."""
    return [
        {
            "id": f"{block.state}-{label}",
            "state": block.state,
            "coordinate": block.tau,
            "tau": float(tau),
            "role": label,
        }
        for block in oracle.reduction.blocks
        for label, tau in (
            ("anchor", oracle.start[block.tau]),
            ("0.001", 0.001),
            ("0.1", 0.1),
            ("1", 1.0),
            ("10", 10.0),
        )
    ]


def profile_search(oracle, profile_spec, seconds, maximum_calls, checkpoint):
    """Bounded residual-only nuisance optimization, with an unoptimized paired control.

    All nonfixed chart parameters and trajectory initials are free. This yields
    a finite-budget upper envelope, not a converged profile minimum, likelihood
    interval or evidence that other branches cannot fit. Every retained point
    still requires independent original-ODE verification by the caller.
    """
    if not 0 < seconds <= 120 or not 1 <= maximum_calls <= 120:
        raise ValueError("profiles require <=120 seconds and <=120 calls")
    allowed = profile_grid(oracle)
    if dict(profile_spec) not in allowed:
        raise ValueError("profile must be a predetermined grid member")
    coordinate, tau = profile_spec["coordinate"], profile_spec["tau"]
    start = oracle.start | {coordinate: tau}
    initial = oracle.vector(start)
    fixed = oracle.names.index(coordinate)
    free = np.asarray(
        [
            i
            for i in range(len(initial))
            if i != fixed and oracle.lower[i] < oracle.upper[i]
        ]
    )
    units = oracle.local_coordinate_scales(initial)[free]
    budget = execution._budget(seconds, seconds + 60)
    result = {
        "protocol": PROTOCOL,
        "profile": deepcopy(profile_spec),
        "optimizer": "Powell",
        "optimizer_start": start,
        "best": None,
        "unoptimized_control": None,
        "calls_attempted": 0,
        "calls_started": 0,
        "calls_completed": 0,
        "calls_failed": 0,
        "invalid_trials": 0,
        "stop_reason": "not_started",
        "budget_exhausted": False,
        "nuisance_coordinates": [oracle.names[i] for i in free],
        "nuisance_optimization_performed": False,
        "nuisance_optimization_attempted": False,
        "nuisance_coordinates_varied": [],
        "derivatives_used": False,
        "scope": (
            "finite-budget conditional upper envelope, not certified profile minimum"
        ),
        "global_minimum_certified": False,
        "automatic_promotion": False,
        "independent_original_verification_required": True,
        "accepted_iteration_history": [],
    }
    interrupted = None

    def save():
        budget.callback(
            lambda: checkpoint(deepcopy(result) | {"timing": budget.snapshot()})
        )

    def objective(q):
        budget.check()
        if result["calls_started"] >= maximum_calls:
            raise TimeoutError("profile call allowance exhausted")
        point = initial.copy()
        point[free] += np.asarray(q) * units
        result["calls_attempted"] += 1
        save()

        def evaluate(deadline):
            result["calls_started"] += 1
            varied = {oracle.names[i] for i in free if point[i] != initial[i]}
            result["nuisance_optimization_performed"] |= bool(varied)
            result["nuisance_coordinates_varied"] = sorted(
                set(result["nuisance_coordinates_varied"]) | varied
            )
            try:
                value = oracle.evaluate(point, deadline, jacobian=False)
            except BaseException:
                result["calls_failed"] += 1
                raise
            result["calls_completed"] += 1
            return value

        try:
            residual, _ = budget.numerical(evaluate)
            budget.check()
            loss = float(np.mean(residual**2))
            if not np.isfinite(loss):
                raise NumericalInvalidTrial("nonfinite profile objective")
        except (ValueError, NumericalInvalidTrial) as error:
            result["invalid_trials"] += 1
            result["last_invalid_trial"] = f"{type(error).__name__}: {error}"[-500:]
            save()
            return float("inf")
        endpoint = {"parameters": oracle.parameters(point), "training_nmse": loss}
        if result["unoptimized_control"] is None:
            result["unoptimized_control"] = deepcopy(endpoint)
        if result["best"] is None or loss < result["best"]["training_nmse"]:
            result["best"] = endpoint
        save()
        return loss

    def accepted(q):
        point = initial.copy()
        point[free] += np.asarray(q) * units
        result["accepted_iteration_history"].append(
            {"parameters": oracle.parameters(point)}
        )
        save()

    try:
        initial_loss = objective(np.zeros(len(free)))
        if not np.isfinite(initial_loss):
            raise ValueError("invalid deterministic fixed-timescale start")
        result["nuisance_optimization_attempted"] = bool(len(free))
        bounds = list(
            zip(
                (oracle.lower[free] - initial[free]) / units,
                (oracle.upper[free] - initial[free]) / units,
                strict=True,
            )
        )
        fit = minimize(
            objective,
            np.zeros(len(free)),
            method="Powell",
            bounds=bounds,
            callback=accepted,
            options={"maxfev": maximum_calls - 1, "xtol": 1e-8, "ftol": 1e-10},
        )
        result.update(
            stop_reason="optimizer_stopped",
            optimizer_success=bool(fit.success),
            optimizer_message=str(fit.message),
            optimizer_iterations=int(fit.nit),
            budget_exhausted=bool(result["calls_started"] >= maximum_calls),
        )
    except Exception as error:
        continuation._invalidate(result, error, "search")
    except BaseException as error:
        interrupted = error
        continuation._invalidate(result, error, "search")
    result["portfolio_execution"] = oracle.execution_receipt()
    return continuation._finish(result, budget, checkpoint, "search", interrupted)
