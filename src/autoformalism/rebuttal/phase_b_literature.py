"""Literature models on the nine-case Phase B comparison.

docs/PHASE_B_LITERATURE_MODELS.md states the models and the protocol. Each
model is written once, as a candidate in the restricted grammar, and compiled
by the shared validator. The fit evaluates the validated expressions over all
of a split's trajectories at once; the shared simulator then scores the fitted
candidate one trajectory at a time, as it scores the compared methods, with a
Radau retry. A start counts only if the two give the same training NMSE.

Only the nine cells' public training and validation tables are read. Nothing
here opens test data, a private reference, or a generator's code.
"""

from __future__ import annotations

import ast
import json
import math
import time as clock
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root

from autoformalism.baselines.core import target_scales
from autoformalism.data import DatasetSplit, Trajectory
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON, CompiledModel
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal import phase_b_nine_case as nine
from autoformalism.schemas import CandidateModel

PROTOCOL = "phase-b-literature-models-1"
MEAL = "meal_event_g"
INSULIN = "insulin_pmol_per_kg_min"
#: The last meal's amount in mg, a causal transform of the public meal channel.
DOSE = "meal_dose_mg"
#: Before the first meal the stomach is empty; k_empt is then k_max.
DOSE_FLOOR_MG = 1.0
STARTS = (0, 1, 2)
PERTURBATION = 0.3
BOUND_FACTOR = 10.0
FRACTION_CEILING = 1.0 - 1e-6
#: Relative finite-difference step: well above the solver's 1e-7 tolerance.
DIFF_STEP = 1e-3
MAX_NFEV = 200
FAILURE_RESIDUAL = 1e3
#: Fast and shared training NMSE must agree to this relative difference, or
#: to the absolute floor where both sit at the solvers' round-off.
REPLAY_TOLERANCE = 1e-4
REPLAY_FLOOR = 1e-9
#: The reference meal of the published model (BioModels BIOMD0000000379).
REFERENCE_DOSE_MG = 78000.0

#: Dalla Man, Rizza and Cobelli (2007), normal subject, as listed in
#: BioModels BIOMD0000000379; Fcns is the constant Uii.
DM07 = {
    "k1": 0.065, "k2": 0.079, "m1": 0.190, "m2": 0.484, "m4": 0.194,
    "m5": 0.0304, "m6": 0.6471, "kmax": 0.0558, "kmin": 0.0080,
    "kabs": 0.057, "kgri": 0.0558, "f": 0.90, "b": 0.82, "d": 0.010,
    "kp1": 2.70, "kp2": 0.0021, "kp3": 0.009, "kp4": 0.0618, "ki": 0.0079,
    "Fcns": 1.0, "Vm0": 2.50, "Vmx": 0.047, "Km0": 225.59, "p2U": 0.0331,
    "K": 2.30, "alpha": 0.050, "beta": 0.11, "gamma": 0.5, "ke1": 0.0005,
    "ke2": 339.0, "Gb": 95.0, "Ib": 25.0, "Sb": 1.8,
}
#: Held at the published values: distribution volumes and body weight.
VG, VI, BW = 1.88, 0.05, 78.0
DM07_FRACTIONS = frozenset({"f", "b", "d", "m6"})
#: Generic starts for the extended minimal model: typical normal-subject
#: magnitudes from Bergman et al. (1979, 1981), t_max from Hovorka et al.
#: (2004), f from Dalla Man et al. (2007). Not fitted to these data.
BERGMAN = {
    "SG": 0.025, "Gpb": 95.0 * VG, "p2": 0.025, "p3": 2.0e-6, "Ib": 25.0,
    "n": 0.14, "gammaB": 0.5, "h": 95.0, "tmax": 40.0, "f": 0.9,
}
#: Generic CSTR starts; r_ref's start comes from the training data's scale.
CSTR_STARTS = {"q": 1.0, "h": 1.0, "ER": 10000.0}

T1_EASY = "phase_b_dalla_man_t1_canonical_named_easy"
T1_EASY_PERTURBED = "phase_b_dalla_man_t1_perturbed_named_easy"
T1_HARD = "phase_b_dalla_man_t1_canonical_named_hard"
T2_EASY = "phase_b_dalla_man_t2_canonical_named_easy"
T2_HARD = "phase_b_dalla_man_t2_canonical_named_hard"
CSTR = "phase_b_cstr_controlled_reactor_mechanism_canonical_named_easy"
NOT_APPLICABLE = {
    "phase_b_anonymous_system_t1_canonical_obfuscated_easy": (
        "not applicable: no names, so no literature can be chosen"
    ),
    "phase_b_anonymous_system_t1_perturbed_obfuscated_easy": (
        "not applicable: no names, so no literature can be chosen"
    ),
    "phase_b_alien_device_unknown_device_mechanism_canonical_functional_easy": (
        "not applicable: no literature exists"
    ),
}
UPPER_BOUND = "generator's published structure (upper bound)"


@dataclass(frozen=True)
class Pair:
    """One literature model fitted to one case."""

    key: str
    benchmark_id: str
    model: str
    label: str
    unfitted: bool


PAIRS = (
    Pair("dm07_t1_easy_canonical", T1_EASY, "dm07_t1_easy", UPPER_BOUND, True),
    Pair(
        "dm07_t1_easy_perturbed",
        T1_EASY_PERTURBED,
        "dm07_t1_easy",
        "published structure under changed exchange laws",
        True,
    ),
    Pair("dm07_t1_hard", T1_HARD, "dm07_t1_hard", UPPER_BOUND, True),
    Pair("dm07_t2_easy", T2_EASY, "dm07_t2_easy", UPPER_BOUND, True),
    Pair("dm07_t2_hard", T2_HARD, "dm07_t2_hard", UPPER_BOUND, True),
    Pair("bergman_t2_hard", T2_HARD, "bergman", "classic expert model", False),
    Pair(
        "cstr_balance",
        CSTR,
        "cstr",
        "textbook structure on the affected historical export; not an upper bound",
        False,
    ),
)
PAIR_BY_KEY = {pair.key: pair for pair in PAIRS}


class SimulationFailed(RuntimeError):
    """A rollout that did not reach the last sample with finite values."""


class OutOfTime(RuntimeError):
    """The start's wall-clock limit was reached during the fit."""


# --------------------------------------------------------------------------
# Model structures, in the restricted grammar.


@dataclass(frozen=True)
class Structure:
    """One literature model's equations as grammar text."""

    observed: Mapping[str, str]
    latent: tuple[str, ...]
    processes: tuple[tuple[str, str], ...]
    equations: Mapping[str, str]
    observations: Mapping[str, str]
    starts: Mapping[str, float]
    fractions: frozenset[str]
    needs_dose: bool

    @property
    def states(self) -> tuple[str, ...]:
        return (*self.observed, *self.latent)

    def symbols(self) -> frozenset[str]:
        """Every identifier the equations, processes and observations read.

        Function names, such as ``exp``, are not symbols.
        """
        texts = (
            *(text for _, text in self.processes),
            *self.equations.values(),
            *self.observations.values(),
        )
        names: set[str] = set()
        for text in texts:
            tree = ast.parse(text, mode="eval")
            functions = {
                id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
            }
            names |= {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and id(node) not in functions
            }
        return frozenset(names)


DM07_SUPPLIED = {
    "dm07_t1_easy": frozenset({"EGP", "Uii", "E", "Gt"}),
    "dm07_t1_hard": frozenset({"Gt"}),
    "dm07_t2_easy": frozenset({"EGP", "Uii", "E", "Gt"}),
    "dm07_t2_hard": frozenset({"Uii"}),
}


def dm07(variant: str) -> Structure:
    """The Dalla Man (2007) model, keeping what generates the case's targets.

    A supplied auxiliary enters as given and its own equation is dropped.
    Insulin is written in plasma concentration, I = Ip/VI, which is the same
    model in other units.
    """
    supplied = DM07_SUPPLIED[variant]
    insulin = variant != "dm07_t1_easy"
    insulin_observed = variant in {"dm07_t2_easy", "dm07_t2_hard"}
    production = "EGP" not in supplied
    remote = variant in {"dm07_t2_easy", "dm07_t2_hard"}
    insulin_input = variant in {"dm07_t2_easy", "dm07_t2_hard"}

    processes: list[tuple[str, str]] = []
    if insulin:
        processes.append(("G", f"Gp / {VG!r}"))
    processes += [
        ("Qsto", "Qsto1 + Qsto2"),
        ("Dm", f"max({DOSE}, {DOSE_FLOOR_MG!r})"),
        (
            "kempt",
            "kmin + (kmax - kmin) / 2"
            " * (tanh(5 / (2 * Dm * (1 - b)) * (Qsto - b * Dm))"
            " - tanh(5 / (2 * Dm * d) * (Qsto - d * Dm)) + 2)",
        ),
        ("Ra", f"f * kabs * Qgut / {BW!r}"),
    ]
    if production:
        processes += [
            ("EGP", "max(kp1 - kp2 * Gp - kp3 * Id - kp4 * Ipo, 0)"),
            ("E", "ke1 * max(Gp - ke2, 0)"),
        ]
    if "Uii" not in supplied:
        processes.append(("Uii", "Fcns"))
    if remote:
        processes.append(("Uid", "(Vm0 + Vmx * X) * Gt / (Km0 + Gt)"))
    processes.append(("dGp", "EGP + Ra - Uii - E - k1 * Gp + k2 * Gt"))
    if insulin:
        processes += [
            ("S", "gamma * Ipo"),
            ("HE", "m6 - m5 * S"),
            ("m3", "HE * m1 / (1 - HE)"),
            ("Spo", f"Y + K * max(dGp / {VG!r}, 0) + Sb"),
        ]

    equations = {"Gp": "dGp"}
    observed = {"Gp": "Gp"}
    latent: list[str] = []
    if "Gt" not in supplied:
        latent.append("Gt")
        equations["Gt"] = "-Uid + k1 * Gp - k2 * Gt"
    latent += ["Qsto1", "Qsto2", "Qgut"]
    equations["Qsto1"] = f"-kgri * Qsto1 + 1000 * {MEAL}"
    equations["Qsto2"] = "kgri * Qsto1 - kempt * Qsto2"
    equations["Qgut"] = "kempt * Qsto2 - kabs * Qgut"
    if insulin:
        if insulin_observed:
            observed["I"] = "I"
        else:
            latent.append("I")
        latent.append("Il")
        equations["Il"] = f"-(m1 + m3) * Il + m2 * {VI!r} * I + S"
        forcing = f"(m1 * Il + {INSULIN}) / {VI!r}" if insulin_input else (
            f"m1 * Il / {VI!r}"
        )
        equations["I"] = f"-(m2 + m4) * I + {forcing}"
        if production:
            latent += ["I1", "Id"]
            equations["I1"] = "-ki * (I1 - I)"
            equations["Id"] = "-ki * (Id - I1)"
        if remote:
            latent.append("X")
            equations["X"] = "-p2U * X + p2U * (I - Ib)"
        latent += ["Ipo", "Y"]
        equations["Ipo"] = "-gamma * Ipo + Spo"
        equations["Y"] = "-alpha * (Y - beta * (G - Gb))"

    observations = {channel: state for state, channel in observed.items()}
    if variant == "dm07_t2_easy":
        observations["U"] = "Uii + Uid"
    structure = Structure(
        observed=observed,
        latent=tuple(latent),
        processes=tuple(processes),
        equations=equations,
        observations=observations,
        starts={},
        fractions=frozenset(),
        needs_dose=True,
    )
    names = structure.symbols() & set(DM07)
    return replace(
        structure,
        starts={name: DM07[name] for name in DM07 if name in names},
        fractions=DM07_FRACTIONS & names,
    )


def bergman() -> Structure:
    """The extended minimal model with two-compartment absorption."""
    return Structure(
        observed={"Gp": "Gp", "I": "I"},
        latent=("X", "Q1", "Q2"),
        processes=(("Ra", f"f * Q2 / (tmax * {BW!r})"),),
        equations={
            "Gp": "-(SG + X) * Gp + SG * Gpb + Ra",
            "I": f"-n * (I - Ib) + gammaB * max(Gp / {VG!r} - h, 0)"
            f" + {INSULIN} / {VI!r}",
            "X": "-p2 * X + p3 * (I - Ib)",
            "Q1": f"-Q1 / tmax + 1000 * {MEAL}",
            "Q2": "(Q1 - Q2) / tmax",
        },
        observations={"Gp": "Gp", "I": "I"},
        starts=dict(BERGMAN),
        fractions=frozenset({"f"}),
        needs_dose=False,
    )


def cstr(r_ref: float) -> Structure:
    """The textbook reactor-temperature balance about 350 K."""
    return Structure(
        observed={"T": "T"},
        latent=(),
        processes=(),
        equations={
            "T": "q * (Tf - T) + r_ref * exp(-ER * (1 / T - 1 / 350)) * C"
            " - h * (T - Tj)",
        },
        observations={"T": "T"},
        starts={**CSTR_STARTS, "r_ref": r_ref},
        fractions=frozenset(),
        needs_dose=False,
    )


# --------------------------------------------------------------------------
# Public data: the derived dose channel, contexts and stacked arrays.


def dose_series(meal: NDArray[np.float64]) -> NDArray[np.float64]:
    """The last meal's amount in mg at each sample, zero before any meal."""
    values = np.asarray(meal, dtype=float)
    dose = np.zeros_like(values)
    last = 0.0
    for index, amount in enumerate(values):
        if amount > 0.0:
            last = 1000.0 * amount
        dose[index] = last
    return dose


def with_dose(split: DatasetSplit) -> DatasetSplit:
    """Add the causal dose channel to every trajectory of a split."""
    trajectories = tuple(
        replace(
            trajectory,
            external_inputs={
                **trajectory.external_inputs,
                DOSE: dose_series(trajectory.external_inputs[MEAL]),
            },
        )
        for trajectory in split.trajectories
    )
    return replace(split, trajectories=trajectories)


def model_context(
    context: ValidationContext, splits: Sequence[DatasetSplit], needs_dose: bool
) -> ValidationContext:
    """The case's shared context, with fitted initialization and the dose."""
    payload = context.model_dump()
    payload["fitted_initialization"] = True
    if needs_dose:
        top = max(
            float(np.max(trajectory.external_inputs[DOSE]))
            for split in splits
            for trajectory in split.trajectories
        )
        payload["external_inputs"] = (*context.external_inputs, DOSE)
        payload["forcing_bounds"] = {
            **payload["forcing_bounds"],
            DOSE: (0.0, max(top, DOSE_FLOOR_MG)),
        }
    return ValidationContext.model_validate(payload)


@dataclass(frozen=True)
class Stack:
    """A split's trajectories on their shared time grid, one row each."""

    trajectory_ids: tuple[str, ...]
    time: NDArray[np.float64]
    channels: Mapping[str, NDArray[np.float64]]

    @classmethod
    def of(cls, split: DatasetSplit) -> Stack:
        first = split.trajectories[0]
        for trajectory in split.trajectories:
            if not np.array_equal(trajectory.time, first.time):
                raise ValueError("trajectories of a split must share one time grid")
        channels: dict[str, NDArray[np.float64]] = {}
        for group in ("targets", "auxiliaries", "external_inputs"):
            for name in getattr(first, group):
                channels[name] = np.asarray(
                    [getattr(t, group)[name] for t in split.trajectories], dtype=float
                )
        return cls(
            tuple(t.trajectory_id for t in split.trajectories),
            np.asarray(first.time, dtype=float),
            channels,
        )

    def breaks(self, forcing: Sequence[str]) -> NDArray[np.intp]:
        """Every sample where a read channel changes slope in any trajectory."""
        boundaries = np.zeros(len(self.time), dtype=bool)
        boundaries[[0, -1]] = True
        steps = np.diff(self.time)
        for name in forcing:
            slopes = np.diff(self.channels[name], axis=1) / steps
            boundaries[1:-1] |= (slopes[:, 1:] != slopes[:, :-1]).any(axis=0)
        return np.flatnonzero(boundaries)

    def at(self, name: str, time: float) -> NDArray[np.float64]:
        """Linear interpolation of one channel, as the shared forcing does."""
        grid = self.time
        index = int(
            np.clip(np.searchsorted(grid, time, side="right") - 1, 0, len(grid) - 2)
        )
        weight = (time - grid[index]) / (grid[index + 1] - grid[index])
        values = self.channels[name]
        return values[:, index] + weight * (values[:, index + 1] - values[:, index])


# --------------------------------------------------------------------------
# Vectorized evaluation of the validated expressions.

Vector = Callable[[Mapping[str, Any]], Any]


def _guard(value: Any) -> Any:
    """The shared evaluator's denominator guard, elementwise."""
    value = np.asarray(value, dtype=float)
    safe = np.where(value >= 0.0, SAFE_DIVISION_EPSILON, -SAFE_DIVISION_EPSILON)
    return np.where(np.abs(value) >= SAFE_DIVISION_EPSILON, value, safe)


def _sigmoid(value: Any) -> Any:
    small = np.exp(-np.abs(value))
    return np.where(value >= 0.0, 1.0 / (1.0 + small), small / (1.0 + small))


def _softplus(value: Any) -> Any:
    return np.maximum(value, 0.0) + np.log1p(np.exp(-np.abs(value)))


_UNARY: dict[str, Callable[[Any], Any]] = {
    "exp": np.exp,
    "log": np.log,
    "tanh": np.tanh,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sigmoid": _sigmoid,
    "softplus": _softplus,
}


def vectorize(node: ast.AST) -> Vector:
    """Compile a validated expression tree to an elementwise numpy closure."""
    if isinstance(node, ast.Expression):
        return vectorize(node.body)
    if isinstance(node, ast.Constant):
        constant = float(node.value)
        return lambda env: constant
    if isinstance(node, ast.Name):
        name = node.id
        return lambda env: env[name]
    if isinstance(node, ast.UnaryOp):
        operand = vectorize(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        return lambda env: -operand(env)
    if isinstance(node, ast.BinOp):
        left = vectorize(node.left)
        right = vectorize(node.right)
        if isinstance(node.op, ast.Add):
            return lambda env: left(env) + right(env)
        if isinstance(node.op, ast.Sub):
            return lambda env: left(env) - right(env)
        if isinstance(node.op, ast.Mult):
            return lambda env: left(env) * right(env)
        if isinstance(node.op, ast.Div):
            return lambda env: left(env) / _guard(right(env))
        raise ValueError("the literature models use no powers")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        arguments = [vectorize(argument) for argument in node.args]
        name = node.func.id
        if name in ("max", "min"):
            reduce = np.maximum if name == "max" else np.minimum

            def extreme(env: Mapping[str, Any]) -> Any:
                result = arguments[0](env)
                for argument in arguments[1:]:
                    result = reduce(result, argument(env))
                return result

            return extreme
        if name in _UNARY and len(arguments) == 1:
            function = _UNARY[name]
            (argument,) = arguments
            return lambda env: function(argument(env))
    raise ValueError(f"unsupported expression node {ast.dump(node)[:80]}")


@dataclass(frozen=True)
class FastModel:
    """A compiled candidate's expressions, evaluated over stacked trajectories."""

    states: tuple[str, ...]
    observed: Mapping[str, str]
    processes: tuple[tuple[str, Vector], ...]
    equations: tuple[Vector, ...]
    observations: Mapping[str, Vector]
    initials: Mapping[str, Vector]
    forcing: tuple[str, ...]
    time_symbol: str

    @classmethod
    def of(cls, compiled: CompiledModel) -> FastModel:
        validated = compiled.validated
        observed = dict(compiled.direct_state_observation_channels)
        return cls(
            states=compiled.state_names,
            observed=observed,
            processes=tuple(
                (name, vectorize(validated.process_expressions[name].tree))
                for name in validated.process_order
            ),
            equations=tuple(
                vectorize(validated.equation_expressions[state].tree)
                for state in compiled.state_names
            ),
            observations={
                channel: vectorize(expression.tree)
                for channel, expression in validated.observation_expressions.items()
            },
            initials={
                state: vectorize(validated.initial_condition_expressions[state].tree)
                for state in compiled.state_names
                if state not in observed
            },
            forcing=tuple(sorted(validated.forcing_symbols)),
            time_symbol=validated.context.time_symbol,
        )

    def _environment(
        self,
        time: Any,
        states: Mapping[str, Any],
        channels: Mapping[str, Any],
        parameters: Mapping[str, float],
    ) -> dict[str, Any]:
        env: dict[str, Any] = {
            self.time_symbol: time,
            **parameters,
            **channels,
            **states,
        }
        for name, process in self.processes:
            env[name] = process(env)
        return env

    def derivatives(
        self,
        time: Any,
        states: Mapping[str, Any],
        channels: Mapping[str, Any],
        parameters: Mapping[str, float],
        count: int,
    ) -> NDArray[np.float64]:
        """State derivatives, one row per state and one column per trajectory."""
        env = self._environment(time, states, channels, parameters)
        return np.asarray(
            [np.broadcast_to(equation(env), (count,)) for equation in self.equations],
            dtype=float,
        )

    def initial_state(
        self, stack: Stack, parameters: Mapping[str, float]
    ) -> NDArray[np.float64]:
        """States at the first sample: observed ones read, latent ones mapped."""
        count = len(stack.trajectory_ids)
        env = {
            self.time_symbol: float(stack.time[0]),
            **parameters,
            **{name: values[:, 0] for name, values in stack.channels.items()},
        }
        rows = []
        for state in self.states:
            if state in self.observed:
                rows.append(stack.channels[self.observed[state]][:, 0])
            else:
                rows.append(np.broadcast_to(self.initials[state](env), (count,)))
        initial = np.asarray(rows, dtype=float)
        if not np.isfinite(initial).all():
            raise SimulationFailed("initial states contain nonfinite values")
        return initial

    def rollout(
        self,
        stack: Stack,
        parameters: Mapping[str, float],
        *,
        max_step: float,
        method: str = "RK45",
        initial: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Free rollout of every trajectory, restarting at slope changes."""
        count = len(stack.trajectory_ids)
        shape = (len(self.states), count)
        start_state = (
            self.initial_state(stack, parameters) if initial is None else initial
        )
        states = np.empty((*shape, len(stack.time)))
        states[:, :, 0] = start_state

        def rhs(time: float, flat: NDArray[np.float64]) -> NDArray[np.float64]:
            derivative = self.derivatives(
                time,
                dict(zip(self.states, flat.reshape(shape), strict=True)),
                {name: stack.at(name, time) for name in self.forcing},
                parameters,
                count,
            )
            if not np.isfinite(derivative).all():
                raise SimulationFailed("right-hand side produced nonfinite values")
            return derivative.ravel()

        with np.errstate(all="ignore"):
            for start, end in pairwise(stack.breaks(self.forcing)):
                solution = solve_ivp(
                    rhs,
                    (float(stack.time[start]), float(stack.time[end])),
                    states[:, :, start].ravel(),
                    t_eval=stack.time[start : end + 1],
                    method=method,
                    rtol=1e-7,
                    atol=1e-9,
                    max_step=max_step,
                )
                if not solution.success:
                    raise SimulationFailed(f"integration failed: {solution.message}")
                states[:, :, start : end + 1] = solution.y.reshape(
                    *shape, end - start + 1
                )
            if not np.isfinite(states).all():
                raise SimulationFailed("integration produced nonfinite states")
            env = self._environment(
                stack.time[None, :],
                dict(zip(self.states, states, strict=True)),
                {name: stack.channels[name] for name in self.forcing},
                parameters,
            )
            predictions = {
                channel: np.broadcast_to(expression(env), (count, len(stack.time)))
                for channel, expression in self.observations.items()
            }
        if any(not np.isfinite(values).all() for values in predictions.values()):
            raise SimulationFailed("observation produced nonfinite values")
        return {"states": states, **predictions}

    def simulate(
        self, stack: Stack, parameters: Mapping[str, float], *, max_step: float
    ) -> dict[str, NDArray[np.float64]]:
        """RK45, then Radau if RK45 fails."""
        try:
            return self.rollout(stack, parameters, max_step=max_step)
        except SimulationFailed:
            return self.rollout(stack, parameters, max_step=max_step, method="Radau")


# --------------------------------------------------------------------------
# Basal states, map scales and parameter vectors.


def _single_stack(
    duration: float, step: float, channels: Mapping[str, float]
) -> Stack:
    time = np.arange(0.0, duration + step / 2, step)
    return Stack(
        ("reference",),
        time,
        {name: np.full((1, len(time)), value) for name, value in channels.items()},
    )


def _bare_context(channels: Mapping[str, str]) -> ValidationContext:
    """A context naming only the given channels, by role."""
    groups: dict[str, list[str]] = {
        "targets": [],
        "auxiliaries": [],
        "external_inputs": [],
    }
    for name, group in channels.items():
        groups[group].append(name)
    return ValidationContext.model_validate(
        {
            **{key: tuple(value) for key, value in groups.items()},
            "fitted_initialization": True,
        }
    )


def _compile(
    structure: Structure,
    context: ValidationContext,
    *,
    initials: Mapping[str, str],
    weights: Sequence[str] = (),
    identifier: str = "literature_model",
) -> CompiledModel:
    """Build and compile the candidate for a structure."""
    parameters = []
    for name, start in structure.starts.items():
        lower, upper = bounds(name, start, structure.fractions)
        parameters.append(
            {
                "name": name,
                "scope": "global",
                "role": "scale",
                "bounds": {"lower": lower, "upper": upper},
            }
        )
    parameters += [
        {"name": name, "scope": "global", "role": "coefficient"} for name in weights
    ]
    candidate = CandidateModel.model_validate(
        {
            "candidate_id": identifier,
            "parent_candidate_id": None,
            "change_summary": "Literature model.",
            "states": [
                {
                    "name": state,
                    "kind": "observed" if state in structure.observed else "latent",
                }
                for state in structure.states
            ],
            "processes": [
                {"name": name, "expression": text} for name, text in structure.processes
            ],
            "state_equations": [
                {"state": state, "rhs": structure.equations[state]}
                for state in structure.states
            ],
            "observation_mappings": [
                {"channel": channel, "expression": text}
                for channel, text in structure.observations.items()
            ],
            "parameters": parameters,
            "initial_conditions": [
                {
                    "state": state,
                    "scope": "global",
                    "expression": structure.observed[state],
                }
                for state in structure.observed
            ]
            + [
                {"state": state, "scope": "global", "expression": initials[state]}
                for state in structure.latent
            ],
        }
    )
    return compile_candidate(candidate, context)


def bounds(name: str, start: float, fractions: frozenset[str]) -> tuple[float, float]:
    """A factor of ten either side of the start; fractions stay below one."""
    lower, upper = start / BOUND_FACTOR, start * BOUND_FACTOR
    if name in fractions:
        upper = min(upper, FRACTION_CEILING)
    return lower, upper


@cache
def dm07_basal() -> dict[str, float]:
    """The published model's fasting steady state, and its readings there."""
    structure = dm07("dm07_t2_hard")
    channels = {
        "Gp": "targets",
        "I": "targets",
        "Uii": "auxiliaries",
        MEAL: "external_inputs",
        INSULIN: "external_inputs",
        DOSE: "external_inputs",
    }
    compiled = _compile(
        structure,
        _bare_context(channels),
        initials=dict.fromkeys(structure.latent, "0"),
    )
    fast = FastModel.of(compiled)
    inputs = {"Uii": 1.0, MEAL: 0.0, INSULIN: 0.0, DOSE: 0.0}
    # BioModels BIOMD0000000379's initial state.
    guess = {
        "Gp": 178.0, "I": 25.0, "Gt": 135.0, "Il": 4.5, "I1": 25.0,
        "Id": 25.0, "X": 0.0, "Ipo": 3.6, "Y": 0.0,
    }
    gut = {"Qsto1", "Qsto2", "Qgut"}
    unknown = [state for state in fast.states if state not in gut]

    def environment(values: NDArray[np.float64]) -> dict[str, Any]:
        states = dict.fromkeys(gut, 0.0) | dict(zip(unknown, values, strict=True))
        return fast._environment(0.0, states, inputs, structure.starts)

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        states = dict.fromkeys(gut, 0.0) | dict(zip(unknown, values, strict=True))
        derivative = fast.derivatives(0.0, states, inputs, structure.starts, 1)
        by_state = dict(zip(fast.states, derivative[:, 0], strict=True))
        return np.asarray([by_state[state] for state in unknown])

    solution = root(residual, [guess[state] for state in unknown], tol=1e-12)
    if not solution.success or np.max(np.abs(residual(solution.x))) > 1e-8:
        raise RuntimeError(f"no published steady state: {solution.message}")
    env = environment(solution.x)
    basal = {state: float(env[state]) for state in fast.states}
    for name in ("EGP", "E", "Uid"):
        basal[name] = float(env[name])
    basal["Uii"] = 1.0
    basal["U"] = basal["Uii"] + basal["Uid"]
    return basal


def latent_basal(pair: Pair, structure: Structure) -> dict[str, float]:
    """Each latent state's basal value: the published steady state."""
    if pair.model == "bergman":
        return {"X": 0.0, "Q1": 0.0, "Q2": 0.0}
    basal = dm07_basal()
    return {state: basal[state] for state in structure.latent}


def reading_basal(pair: Pair, channel: str) -> float:
    """A reading's basal value; public inputs are zero at basal."""
    if channel in (MEAL, INSULIN):
        return 0.0
    if pair.model == "bergman":
        return {"Gp": BERGMAN["Gpb"], "I": BERGMAN["Ib"]}[channel]
    return dm07_basal()[channel]


def varying_readings(split: DatasetSplit) -> dict[str, float]:
    """Public channels whose first readings vary over the training set.

    Returns each one's standard deviation over the training trajectories'
    first samples, used to scale the initial-value map.
    """
    first = split.trajectories[0]
    readings: dict[str, float] = {}
    for group in ("targets", "auxiliaries", "external_inputs"):
        for name in getattr(first, group):
            values = np.asarray(
                [float(getattr(t, group)[name][0]) for t in split.trajectories]
            )
            if np.ptp(values) > 0.0:
                readings[name] = float(np.std(values))
    return readings


def reference_scales(
    pair: Pair, structure: Structure, basal: Mapping[str, float]
) -> dict[str, float]:
    """Each latent state's magnitude under the published reference meal.

    The published (or generic) model starts at basal with the reference meal
    in the stomach and runs 300 minutes; the largest absolute value a latent
    state reaches, or its basal value if larger, scales its initial-value map.
    """
    if not structure.latent:
        return {}
    dose_state = "Q1" if pair.model == "bergman" else "Qsto1"
    channels = {MEAL: 0.0, INSULIN: 0.0, DOSE: REFERENCE_DOSE_MG}
    if pair.model != "bergman":
        published = dm07_basal()
        for name in DM07_SUPPLIED[pair.model]:
            channels[name] = published[name]
        observed = {state: published[state] for state in structure.observed}
    else:
        observed = {"Gp": BERGMAN["Gpb"], "I": BERGMAN["Ib"]}
    inputs = (MEAL, INSULIN, DOSE)
    groups = {
        **{name: "auxiliaries" for name in channels if name not in inputs},
        **dict.fromkeys(inputs, "external_inputs"),
        **dict.fromkeys(structure.observations, "targets"),
    }
    compiled = _compile(
        structure,
        _bare_context(groups),
        initials=dict.fromkeys(structure.latent, "0"),
    )
    fast = FastModel.of(compiled)
    readings = {
        channel: observed[state] for state, channel in structure.observed.items()
    }
    stack = _single_stack(300.0, 1.0, {**channels, **readings})
    initial = np.asarray(
        [
            [observed[state]]
            if state in structure.observed
            else [REFERENCE_DOSE_MG if state == dose_state else basal[state]]
            for state in fast.states
        ]
    )
    rollout = fast.rollout(stack, structure.starts, max_step=1.0, initial=initial)
    states = rollout["states"]
    return {
        state: max(float(np.max(np.abs(states[index]))), abs(basal[state]), 1e-9)
        for index, state in enumerate(fast.states)
        if state in structure.latent
    }


def weight_name(state: str, channel: str) -> str:
    return f"w_{state}_{channel}"


def initial_expressions(
    structure: Structure,
    basal: Mapping[str, float],
    scales: Mapping[str, float],
    readings: Mapping[str, float],
    reading_basals: Mapping[str, float],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Basal plus a fitted linear map of the readings' scaled deviations."""
    expressions: dict[str, str] = {}
    weights: list[str] = []
    for state in structure.latent:
        terms = []
        for channel, spread in readings.items():
            name = weight_name(state, channel)
            weights.append(name)
            terms.append(
                f"{name} * ({channel} - ({reading_basals[channel]!r})) / {spread!r}"
            )
        shift = f" + {scales[state]!r} * ({' + '.join(terms)})" if terms else ""
        expressions[state] = f"({basal[state]!r}){shift}"
    return expressions, tuple(weights)


@dataclass(frozen=True)
class Prepared:
    """Everything one pair's fit and scoring need."""

    pair: Pair
    structure: Structure
    compiled: CompiledModel
    fast: FastModel
    train: DatasetSplit
    validation: DatasetSplit
    train_stack: Stack
    validation_stack: Stack
    targets: tuple[str, ...]
    scales: Mapping[str, float]
    weights: tuple[str, ...]
    max_step: float
    description: Mapping[str, Any]


def cstr_reference_rate(train: DatasetSplit) -> float:
    """r_ref's start: the least-squares rate at the other generic starts."""
    numerator = denominator = 0.0
    for trajectory in train.trajectories:
        temperature = trajectory.targets["T"]
        slope = np.gradient(temperature, trajectory.time)
        drive = np.exp(-CSTR_STARTS["ER"] * (1 / temperature - 1 / 350)) * (
            trajectory.auxiliaries["C"]
        )
        rest = slope - CSTR_STARTS["q"] * (
            trajectory.external_inputs["Tf"] - temperature
        ) + CSTR_STARTS["h"] * (temperature - trajectory.auxiliaries["Tj"])
        numerator += float(np.sum(drive * rest))
        denominator += float(np.sum(drive * drive))
    rate = numerator / denominator if denominator > 0 else 0.0
    if rate <= 0.0:
        raise ValueError("no positive reaction-rate scale in the training data")
    return rate


def prepare(inputs: Path, key: str) -> Prepared:
    """Load a pair's public cell and build its compiled candidate."""
    pair = PAIR_BY_KEY[key]
    cell = nine.load_cell(inputs, pair.benchmark_id, context="baseline")
    return assemble(pair, cell.dataset.train, cell.dataset.validation, cell.context)


def assemble(
    pair: Pair,
    train: DatasetSplit,
    validation: DatasetSplit,
    context: ValidationContext,
) -> Prepared:
    """Build a pair's compiled candidate from its public splits and context."""
    key = pair.key
    readings = varying_readings(train)
    if pair.model.startswith("dm07"):
        structure = dm07(pair.model)
    elif pair.model == "bergman":
        structure = bergman()
    else:
        structure = cstr(cstr_reference_rate(train))
    if structure.needs_dose:
        train, validation = with_dose(train), with_dose(validation)
    context = model_context(context, (train, validation), structure.needs_dose)
    basal = latent_basal(pair, structure)
    if not structure.latent:
        readings = {}
    reading_basals = {channel: reading_basal(pair, channel) for channel in readings}
    scales_of_latent = reference_scales(pair, structure, basal)
    initials, weights = initial_expressions(
        structure, basal, scales_of_latent, readings, reading_basals
    )
    compiled = _compile(
        structure, context, initials=initials, weights=weights, identifier=key
    )
    targets = tuple(context.targets)
    if set(structure.observations) != set(targets):
        raise ValueError(f"{key}: the model must predict exactly {targets}")
    step = float(np.diff(train.trajectories[0].time)[0])
    description = {
        "key": key,
        "benchmark_id": pair.benchmark_id,
        "model": pair.model,
        "label": pair.label,
        "states": {
            "observed": dict(structure.observed),
            "latent": list(structure.latent),
        },
        "processes": [list(item) for item in structure.processes],
        "equations": dict(structure.equations),
        "observations": dict(structure.observations),
        "starts": dict(structure.starts),
        "bounds": {
            name: list(bounds(name, start, structure.fractions))
            for name, start in structure.starts.items()
        },
        "fractions": sorted(structure.fractions),
        "held": {"VG": VG, "VI": VI, "BW": BW},
        "latent_basal": basal,
        "latent_scales": scales_of_latent,
        "readings": {
            channel: {"basal": reading_basals[channel], "training_spread": spread}
            for channel, spread in readings.items()
        },
        "initial_conditions": initials,
        "max_step": step,
    }
    return Prepared(
        pair=pair,
        structure=structure,
        compiled=compiled,
        fast=FastModel.of(compiled),
        train=train,
        validation=validation,
        train_stack=Stack.of(train),
        validation_stack=Stack.of(validation),
        targets=targets,
        scales=target_scales(train, targets),
        weights=weights,
        max_step=step,
        description=description,
    )


@dataclass(frozen=True)
class Vectorizer:
    """Log model parameters, then linear map weights."""

    names: tuple[str, ...]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    weights: tuple[str, ...]

    @classmethod
    def of(cls, prepared: Prepared) -> Vectorizer:
        structure = prepared.structure
        names = tuple(structure.starts)
        limits = [
            bounds(name, structure.starts[name], structure.fractions) for name in names
        ]
        return cls(
            names,
            np.asarray([low for low, _ in limits]),
            np.asarray([high for _, high in limits]),
            prepared.weights,
        )

    def start(self, starts: Mapping[str, float], index: int) -> NDArray[np.float64]:
        """Start 0 at the starts; others log-uniformly perturbed by +-0.3."""
        logs = np.log([starts[name] for name in self.names])
        if index:
            rng = np.random.default_rng(index)
            logs = logs + rng.uniform(-PERTURBATION, PERTURBATION, len(logs))
        logs = np.clip(logs, np.log(self.lower), np.log(self.upper))
        return np.concatenate([logs, np.zeros(len(self.weights))])

    def lower_upper(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        count = len(self.weights)
        return (
            np.concatenate([np.log(self.lower), np.full(count, -np.inf)]),
            np.concatenate([np.log(self.upper), np.full(count, np.inf)]),
        )

    def values(self, x: NDArray[np.float64]) -> dict[str, float]:
        count = len(self.names)
        model = np.clip(np.exp(x[:count]), self.lower, self.upper)
        pairs = (
            *zip(self.names, model, strict=True),
            *zip(self.weights, x[count:], strict=True),
        )
        return {name: float(value) for name, value in pairs}


def normalized_errors(
    prepared: Prepared, predictions: Mapping[str, NDArray[np.float64]], stack: Stack
) -> NDArray[np.float64]:
    """Errors over every sample, each target scaled by its training spread."""
    return np.concatenate(
        [
            (
                (predictions[target] - stack.channels[target]) / prepared.scales[target]
            ).ravel()
            for target in prepared.targets
        ]
    )


def fast_nmse(
    prepared: Prepared, parameters: Mapping[str, float], stack: Stack
) -> float:
    """The fit's own NMSE of a split: every trajectory rolled out at once."""
    try:
        predictions = prepared.fast.simulate(
            stack, parameters, max_step=prepared.max_step
        )
    except SimulationFailed:
        return math.inf
    return float(np.mean(normalized_errors(prepared, predictions, stack) ** 2))


def score(
    prepared: Prepared, split: DatasetSplit, parameters: Mapping[str, float]
) -> dict[str, Any]:
    """The shared evaluator's free-rollout NMSE, one trajectory at a time.

    Each target's squared errors over every sample, divided by its training
    variance, pooled as the shared evaluator pools them; RK45 first, then
    Radau for a trajectory RK45 cannot finish. A trajectory neither finishes
    makes the score infinite.
    """
    squared: dict[str, list[NDArray[np.float64]]] = {
        target: [] for target in prepared.targets
    }
    failed: list[str] = []
    retried: list[str] = []

    def rollout(trajectory: Trajectory, method: str):
        config = FitConfig(
            integration_method=method, maximum_integration_step=prepared.max_step
        )
        return simulate_trajectory(
            prepared.compiled, trajectory, parameters, {}, config
        )

    for trajectory in split.trajectories:
        result = rollout(trajectory, "RK45")
        if not result.success:
            retried.append(trajectory.trajectory_id)
            result = rollout(trajectory, "Radau")
        if not result.success:
            failed.append(trajectory.trajectory_id)
            continue
        for target in prepared.targets:
            error = result.predictions[target] - trajectory.targets[target]
            squared[target].append((error / prepared.scales[target]) ** 2)
    if failed:
        return {
            "normalized_mse": math.inf,
            "per_target": dict.fromkeys(prepared.targets, math.inf),
            "failed_trajectories": failed,
            "radau_retried": retried,
        }
    pooled = np.concatenate([np.concatenate(values) for values in squared.values()])
    return {
        "normalized_mse": float(np.mean(pooled)),
        "per_target": {
            target: float(np.mean(np.concatenate(values)))
            for target, values in squared.items()
        },
        "failed_trajectories": [],
        "radau_retried": retried,
    }


def fit(prepared: Prepared, start: int, *, wall_seconds: float) -> dict[str, Any]:
    """One start: least squares on the training rollout, then scoring."""
    vectorizer = Vectorizer.of(prepared)
    x0 = vectorizer.start(prepared.structure.starts, start)
    deadline = clock.monotonic() + wall_seconds
    stack = prepared.train_stack
    count = len(prepared.targets) * len(stack.trajectory_ids) * len(stack.time)
    best = {"cost": math.inf, "x": x0.copy()}
    evaluations = {"count": 0, "failed": 0}

    def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
        if clock.monotonic() > deadline:
            raise OutOfTime
        evaluations["count"] += 1
        try:
            predictions = prepared.fast.simulate(
                stack, vectorizer.values(x), max_step=prepared.max_step
            )
        except SimulationFailed:
            evaluations["failed"] += 1
            return np.full(count, FAILURE_RESIDUAL)
        errors = normalized_errors(prepared, predictions, stack)
        cost = float(errors @ errors)
        if cost < best["cost"]:
            best["cost"], best["x"] = cost, x.copy()
        return errors

    started = clock.monotonic()
    lower, upper = vectorizer.lower_upper()
    outcome: dict[str, Any]
    try:
        solution = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            diff_step=DIFF_STEP,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-8,
            max_nfev=MAX_NFEV,
        )
        outcome = {
            "status": int(solution.status),
            "message": str(solution.message),
            "nfev": int(solution.nfev),
            "njev": int(solution.njev) if solution.njev is not None else None,
        }
    except OutOfTime:
        outcome = {"status": "wall_clock_limit", "message": "kept the best start seen"}
    x = best["x"]
    parameters = vectorizer.values(x)
    seconds = clock.monotonic() - started
    fast_train = fast_nmse(prepared, parameters, stack)
    train = score(prepared, prepared.train, parameters)
    validation = score(prepared, prepared.validation, parameters)
    shared_train = train["normalized_mse"]
    if math.isfinite(shared_train) and math.isfinite(fast_train):
        difference = abs(fast_train - shared_train)
        matches = difference <= max(REPLAY_TOLERANCE * shared_train, REPLAY_FLOOR)
    else:
        difference = None
        matches = math.isinf(shared_train) and math.isinf(fast_train)
    return {
        "key": prepared.pair.key,
        "start": start,
        "seed": start or None,
        "starting_values": vectorizer.values(x0),
        "parameters": parameters,
        "least_squares": outcome,
        "residual_evaluations": evaluations,
        "fit_seconds": seconds,
        "train": train,
        "validation": validation,
        "replay": {
            "fast_train_nmse": fast_train,
            "shared_train_nmse": shared_train,
            "absolute_difference": difference,
            "matches": bool(matches),
        },
    }


def unfitted(prepared: Prepared) -> dict[str, Any]:
    """The published values with no fitting: hidden states start at basal."""
    parameters = {
        **dict(prepared.structure.starts),
        **dict.fromkeys(prepared.weights, 0.0),
    }
    return {
        "key": prepared.pair.key,
        "parameters": parameters,
        "train": score(prepared, prepared.train, parameters),
        "validation": score(prepared, prepared.validation, parameters),
    }


# --------------------------------------------------------------------------
# Campaign: plan, resumable jobs, report.


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    temporary.replace(path)


def plan(inputs: Path, *, source_commit: str, wall_seconds: float) -> dict[str, Any]:
    """Freeze what every job will do, from the verified staged cells."""
    return {
        "schema_version": "phase-b-literature-models-plan-1",
        "protocol": PROTOCOL,
        "inputs_receipt_sha256": nine.verify(inputs),
        "source_commit": source_commit,
        "starts": list(STARTS),
        "perturbation": PERTURBATION,
        "bound_factor": BOUND_FACTOR,
        "least_squares": {
            "method": "trf",
            "x_scale": "jac",
            "diff_step": DIFF_STEP,
            "ftol": 1e-6,
            "xtol": 1e-6,
            "gtol": 1e-8,
            "max_nfev": MAX_NFEV,
        },
        "wall_seconds_per_start": wall_seconds,
        "replay_tolerance": {"relative": REPLAY_TOLERANCE, "floor": REPLAY_FLOOR},
        "pairs": [prepare(inputs, pair.key).description for pair in PAIRS],
        "not_applicable": NOT_APPLICABLE,
        "test_data_opened": False,
        "private_reference_opened": False,
    }


def jobs(output: Path) -> list[tuple[str, str]]:
    """Every job not yet sealed: (key, "start_<k>" or "unfitted")."""
    pending = []
    for pair in PAIRS:
        names = [f"start_{start}" for start in STARTS]
        if pair.unfitted:
            names.append("unfitted")
        for name in names:
            if not (output / "results" / pair.key / f"{name}.json").exists():
                pending.append((pair.key, name))
    return pending


def run_job(
    inputs: Path, output: Path, key: str, name: str, wall_seconds: float
) -> str:
    """Run and seal one job; a sealed job is never repeated."""
    path = output / "results" / key / f"{name}.json"
    if path.exists():
        return "sealed"
    prepared = prepare(inputs, key)
    if name == "unfitted":
        result = unfitted(prepared)
    else:
        start = int(name.removeprefix("start_"))
        result = fit(prepared, start, wall_seconds=wall_seconds)
    _write(path, result)
    return "complete"


def report(output: Path) -> dict[str, Any]:
    """Select each pair's start on validation and summarize every row."""
    rows = []
    for pair in PAIRS:
        directory = output / "results" / pair.key
        starts = [
            json.loads((directory / f"start_{start}.json").read_text())
            for start in STARTS
            if (directory / f"start_{start}.json").exists()
        ]
        eligible = [item for item in starts if item["replay"]["matches"]]
        selected = min(
            eligible,
            key=lambda item: item["validation"]["normalized_mse"],
            default=None,
        )
        unfitted_path = directory / "unfitted.json"
        unfitted_result = (
            json.loads(unfitted_path.read_text()) if unfitted_path.exists() else None
        )
        rows.append(
            {
                "key": pair.key,
                "benchmark_id": pair.benchmark_id,
                "model": pair.model,
                "label": pair.label,
                "starts": {
                    item["start"]: {
                        "train_nmse": item["train"]["normalized_mse"],
                        "validation_nmse": item["validation"]["normalized_mse"],
                        "replay_matches": item["replay"]["matches"],
                        "least_squares_status": item["least_squares"]["status"],
                        "fit_seconds": item["fit_seconds"],
                    }
                    for item in starts
                },
                "selected_start": None if selected is None else selected["start"],
                "validation_nmse": (
                    None
                    if selected is None
                    else selected["validation"]["normalized_mse"]
                ),
                "validation_per_target": (
                    None if selected is None else selected["validation"]["per_target"]
                ),
                "unfitted_validation_nmse": (
                    None
                    if unfitted_result is None
                    else unfitted_result["validation"]["normalized_mse"]
                ),
                "complete": len(starts) == len(STARTS)
                and (not pair.unfitted or unfitted_result is not None),
            }
        )
    summary = {
        "protocol": PROTOCOL,
        "status": "complete" if all(row["complete"] for row in rows) else "pending",
        "selection": "lowest validation NMSE among starts whose replay matches",
        "rows": rows,
        "not_applicable": NOT_APPLICABLE,
        "test_data_opened": False,
    }
    _write(output / "summary.json", summary)
    return summary


__all__ = [
    "PAIRS",
    "PROTOCOL",
    "FastModel",
    "Stack",
    "assemble",
    "bergman",
    "cstr",
    "dm07",
    "dose_series",
    "fit",
    "plan",
    "prepare",
    "report",
    "run_job",
    "score",
    "unfitted",
    "vectorize",
]
