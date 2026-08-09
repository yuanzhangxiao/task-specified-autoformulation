"""Trusted private Dalla Man B1 simulator for post-selection interventions.

This is a side-effect-free extraction of the two benchmark generator notebooks.
Only the original model and the level-1 plasma/tissue exchange perturbation used
by the registered B1 benchmarks are included here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

STATE_NAMES = (
    "Gp",
    "Gt",
    "Qsto1",
    "Qsto2",
    "Qgut",
    "Ip",
    "Il",
    "Ipo",
    "Y",
    "I1",
    "Id",
    "X",
)
STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}


@dataclass(frozen=True)
class DallaManParameters:
    """Frozen parameters shared by the original and B1-perturbed generators."""

    BW: float = 78.0
    VG: float = 1.88
    k1: float = 0.065
    k2: float = 0.079
    VI: float = 0.05
    m1: float = 0.190
    m2: float = 0.484
    m4: float = 0.194
    m5: float = 0.0304
    m6: float = 0.6471
    HEb: float = 0.60
    kmax: float = 0.0558
    kmin: float = 0.0080
    kabs: float = 0.057
    kgri: float = 0.0558
    f: float = 0.90
    b: float = 0.82
    d: float = 0.010
    kp1: float = 2.70
    kp2: float = 0.0021
    kp3: float = 0.009
    kp4: float = 0.0618
    ki: float = 0.0079
    Fcns: float = 1.0
    Vm0: float = 2.50
    Vmx: float = 0.047
    Km0: float = 225.59
    p2U: float = 0.0331
    K: float = 2.30
    alpha: float = 0.050
    beta: float = 0.11
    gamma: float = 0.50
    ke1: float = 0.0005
    ke2: float = 339.0
    he_min: float = 1e-6
    he_max: float = 0.99
    exchange_eta_p: float = 0.35
    exchange_eta_t: float = -0.25
    exchange_scale_p_fraction: float = 0.20
    exchange_scale_t_fraction: float = 0.20


@dataclass(frozen=True)
class DallaManBasal:
    """Numerically self-consistent basal quantities and initial state."""

    Gb: float
    Ib: float
    EGPb: float
    Sb: float
    h: float
    initial_state: np.ndarray


@dataclass(frozen=True)
class DallaManTrajectory:
    """Private full-state and derived trajectory."""

    time: np.ndarray
    states: np.ndarray
    meal_event_g: np.ndarray
    derived: dict[str, np.ndarray]


@dataclass(frozen=True)
class DallaManExternalForcing:
    """Piecewise-constant forcing in native derivative units.

    Glucose segments use ``mg kg^-1 min^-1`` and enter ``dGp/dt``. Insulin
    segments use ``pmol kg^-1 min^-1`` and enter ``dIp/dt``.
    """

    glucose_mg_per_kg_min: tuple[tuple[float, float, float], ...] = ()
    insulin_pmol_per_kg_min: tuple[tuple[float, float, float], ...] = ()

    def values_at(self, time: float) -> tuple[float, float]:
        """Return the glucose and insulin forcing active at ``time``."""

        return (
            _piecewise_forcing_value(time, self.glucose_mg_per_kg_min),
            _piecewise_forcing_value(time, self.insulin_pmol_per_kg_min),
        )


def compute_dalla_man_basal(p: DallaManParameters) -> DallaManBasal:
    """Reproduce the generator's rounded-parameter steady-state solve."""

    secretion = (p.m6 - p.HEb) / p.m5
    portal = secretion / p.gamma
    hepatic_rate = p.HEb * p.m1 / (1.0 - p.HEb)
    insulin_coefficient = (p.m1 + hepatic_rate) * (p.m2 + p.m4) / p.m1 - p.m2
    plasma_insulin = secretion / insulin_coefficient
    liver_insulin = (p.m2 + p.m4) * plasma_insulin / p.m1
    basal_insulin = plasma_insulin / p.VI

    def residual(plasma_glucose: float) -> float:
        production = (
            p.kp1 - p.kp2 * plasma_glucose - p.kp3 * basal_insulin - p.kp4 * portal
        )
        utilization = production - p.Fcns
        if utilization <= 0.0 or utilization >= p.Vm0:
            return np.nan
        tissue_glucose = utilization * p.Km0 / (p.Vm0 - utilization)
        return p.k1 * plasma_glucose - p.k2 * tissue_glucose - utilization

    grid = np.linspace(80.0, 400.0, 2000)
    values = np.asarray([residual(value) for value in grid])
    bracket = next(
        (
            (grid[index], grid[index + 1])
            for index in range(len(grid) - 1)
            if np.isfinite(values[index])
            and np.isfinite(values[index + 1])
            and values[index] * values[index + 1] <= 0.0
        ),
        None,
    )
    if bracket is None:
        raise RuntimeError("could not bracket Dalla Man basal glucose")
    plasma_glucose = float(brentq(residual, *bracket))
    production = p.kp1 - p.kp2 * plasma_glucose - p.kp3 * basal_insulin - p.kp4 * portal
    utilization = production - p.Fcns
    tissue_glucose = utilization * p.Km0 / (p.Vm0 - utilization)
    initial = np.asarray(
        [
            plasma_glucose,
            tissue_glucose,
            0.0,
            0.0,
            0.0,
            plasma_insulin,
            liver_insulin,
            portal,
            0.0,
            basal_insulin,
            basal_insulin,
            0.0,
        ],
        dtype=float,
    )
    return DallaManBasal(
        Gb=plasma_glucose / p.VG,
        Ib=basal_insulin,
        EGPb=production,
        Sb=secretion,
        h=plasma_glucose / p.VG,
        initial_state=initial,
    )


def _gastric_emptying_rate(
    stomach_glucose: float, meal_reference_mg: float, p: DallaManParameters
) -> float:
    if meal_reference_mg <= 0.0:
        return p.kmax
    alpha = 5.0 / (2.0 * meal_reference_mg * (1.0 - p.b))
    beta = 5.0 / (2.0 * meal_reference_mg * p.d)
    return p.kmin + (p.kmax - p.kmin) / 2.0 * (
        np.tanh(alpha * (stomach_glucose - p.b * meal_reference_mg))
        - np.tanh(beta * (stomach_glucose - p.d * meal_reference_mg))
        + 2.0
    )


def _rhs_and_derived(
    state: np.ndarray,
    p: DallaManParameters,
    basal: DallaManBasal,
    meal_reference_mg: float,
    variant: Literal["original", "perturbed_b1"],
    external_forcing: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, dict[str, float]]:
    gp, gt, q1, q2, qgut, ip, il, ipo, y, i1, insulin_delay, action = state
    gp0, gt0 = max(gp, 0.0), max(gt, 0.0)
    q1, q2, qgut = max(q1, 0.0), max(q2, 0.0), max(qgut, 0.0)
    ip0, ipo0 = max(ip, 0.0), max(ipo, 0.0)
    glucose = gp0 / p.VG
    insulin = ip0 / p.VI
    stomach = q1 + q2
    emptying = _gastric_emptying_rate(stomach, meal_reference_mg, p)
    appearance = p.f * p.kabs * qgut / p.BW
    production = max(
        p.kp1 - p.kp2 * gp0 - p.kp3 * insulin_delay - p.kp4 * ipo0,
        0.0,
    )
    independent_use = p.Fcns
    dependent_use = max((p.Vm0 + p.Vmx * action) * gt0 / (p.Km0 + gt0), 0.0)
    excretion = p.ke1 * max(gp0 - p.ke2, 0.0)
    secretion = p.gamma * ipo0
    extraction = float(np.clip(-p.m5 * secretion + p.m6, p.he_min, p.he_max))
    hepatic_rate = extraction * p.m1 / (1.0 - extraction)
    if variant == "perturbed_b1":
        gp_b = float(basal.initial_state[STATE_INDEX["Gp"]])
        gt_b = float(basal.initial_state[STATE_INDEX["Gt"]])
        plasma_to_tissue = (
            p.k1
            * gp0
            * (
                1.0
                + p.exchange_eta_p
                * np.tanh((gp0 - gp_b) / max(p.exchange_scale_p_fraction * gp_b, 1e-12))
            )
        )
        tissue_to_plasma = (
            p.k2
            * gt0
            * (
                1.0
                + p.exchange_eta_t
                * np.tanh((gt0 - gt_b) / max(p.exchange_scale_t_fraction * gt_b, 1e-12))
            )
        )
    else:
        plasma_to_tissue = p.k1 * gp0
        tissue_to_plasma = p.k2 * gt0
    glucose_forcing, insulin_forcing = external_forcing
    dgp = (
        production
        + appearance
        - independent_use
        - excretion
        - plasma_to_tissue
        + tissue_to_plasma
        + glucose_forcing
    )
    dgt = -dependent_use + plasma_to_tissue - tissue_to_plasma
    dglucose = dgp / p.VG
    portal_output = y + p.K * max(dglucose, 0.0) + basal.Sb
    beta_signal = p.beta * (glucose - basal.h)
    dy = (
        -p.alpha * (y - beta_signal)
        if beta_signal >= -basal.Sb
        else -p.alpha * y - p.alpha * basal.Sb
    )
    derivative = np.asarray(
        [
            dgp,
            dgt,
            -p.kgri * q1,
            -emptying * q2 + p.kgri * q1,
            -p.kabs * qgut + emptying * q2,
            -(p.m2 + p.m4) * ip0 + p.m1 * il + insulin_forcing,
            -(p.m1 + hepatic_rate) * il + p.m2 * ip0 + secretion,
            -p.gamma * ipo0 + portal_output,
            dy,
            -p.ki * (i1 - insulin),
            -p.ki * (insulin_delay - i1),
            -p.p2U * action + p.p2U * (insulin - basal.Ib),
        ],
        dtype=float,
    )
    return derivative, {
        "G": glucose,
        "I": insulin,
        "Ra": appearance,
        "EGP": production,
        "Uii": independent_use,
        "Uid": dependent_use,
        "U": independent_use + dependent_use,
        "S": portal_output,
        "E": excretion,
        "J_p_to_t": plasma_to_tissue,
        "J_t_to_p": tissue_to_plasma,
        "exchange_net": plasma_to_tissue - tissue_to_plasma,
        "glucose_forcing": glucose_forcing,
        "insulin_forcing": insulin_forcing,
    }


def simulate_dalla_man(
    *,
    meals: tuple[tuple[float, float], ...],
    duration: float,
    dt: float,
    variant: Literal["original", "perturbed_b1"],
    parameters: DallaManParameters | None = None,
    parameter_multipliers: dict[str, float] | None = None,
    initial_state: tuple[float, ...] | None = None,
    external_forcing: DallaManExternalForcing | None = None,
    basal_reference: DallaManBasal | None = None,
) -> DallaManTrajectory:
    """Simulate meals as exact stomach-state jumps on a fixed output grid."""

    p = parameters or DallaManParameters()
    forcing = external_forcing or DallaManExternalForcing()
    _validate_external_forcing(forcing, duration)
    multipliers = parameter_multipliers or {}
    unknown = set(multipliers).difference(p.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown Dalla Man parameter multipliers: {sorted(unknown)}")
    if any(not np.isfinite(value) or value <= 0.0 for value in multipliers.values()):
        raise ValueError("Dalla Man parameter multipliers must be finite and positive")
    if multipliers:
        p = replace(
            p,
            **{
                name: float(getattr(p, name)) * float(multiplier)
                for name, multiplier in multipliers.items()
            },
        )
    basal = basal_reference or compute_dalla_man_basal(p)
    events: dict[float, float] = {}
    for event_time, grams in meals:
        if event_time < 0.0 or event_time > duration or grams <= 0.0:
            raise ValueError("meal must have positive mass within the duration")
        events[float(event_time)] = events.get(float(event_time), 0.0) + 1000.0 * grams
    grid = np.arange(0.0, duration + 0.5 * dt, dt)
    state = (
        basal.initial_state.copy()
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    if state.shape != (len(STATE_NAMES),) or not np.all(np.isfinite(state)):
        raise ValueError("Dalla Man initial state has the wrong shape or values")
    meal_reference = 0.0
    if 0.0 in events:
        meal_reference = events[0.0]
        state[STATE_INDEX["Qsto1"]] += meal_reference
    output_time: list[np.ndarray] = []
    output_states: list[np.ndarray] = []
    current = 0.0
    forcing_boundaries = {
        boundary
        for segments in (
            forcing.glucose_mg_per_kg_min,
            forcing.insulin_pmol_per_kg_min,
        )
        for start, end, _value in segments
        for boundary in (start, end)
        if 0.0 < boundary < duration
    }
    boundaries = sorted(
        {
            *(time for time in events if time > 0.0),
            *forcing_boundaries,
            duration,
        }
    )
    for boundary in boundaries:
        evaluation = grid[(grid >= current - 1e-12) & (grid <= boundary + 1e-12)]
        if (
            output_time
            and len(evaluation)
            and np.isclose(evaluation[0], output_time[-1][-1])
        ):
            evaluation = evaluation[1:]
        if boundary > current:
            interval_meal_reference = meal_reference

            def interval_rhs(
                _time: float,
                values: np.ndarray,
                reference: float = interval_meal_reference,
            ) -> np.ndarray:
                return _rhs_and_derived(
                    values,
                    p,
                    basal,
                    reference,
                    variant,
                    forcing.values_at(_time),
                )[0]

            solution = solve_ivp(
                interval_rhs,
                (current, boundary),
                state,
                method="LSODA",
                t_eval=evaluation if len(evaluation) else None,
                rtol=1e-8,
                atol=1e-10,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            if len(evaluation):
                output_time.append(solution.t)
                output_states.append(solution.y.T)
            state = solution.y[:, -1]
        current = boundary
        if boundary in events and boundary < duration + 1e-12:
            amount = events[boundary]
            meal_reference = (
                max(
                    state[STATE_INDEX["Qsto1"]] + state[STATE_INDEX["Qsto2"]],
                    0.0,
                )
                + amount
            )
            state[STATE_INDEX["Qsto1"]] += amount
    time = np.concatenate(output_time)
    states = np.concatenate(output_states)
    meal_event = np.zeros_like(time)
    for event_time, amount in events.items():
        matches = np.flatnonzero(np.isclose(time, event_time, atol=1e-9))
        if len(matches):
            meal_event[matches[0]] += amount / 1000.0
    derived_rows = [
        _rhs_and_derived(
            row,
            p,
            basal,
            _meal_reference_at(t, events),
            variant,
            forcing.values_at(float(t)),
        )[1]
        for t, row in zip(time, states, strict=True)
    ]
    derived = {
        name: np.asarray([row[name] for row in derived_rows], dtype=float)
        for name in derived_rows[0]
    }
    return DallaManTrajectory(time, states, meal_event, derived)


def dalla_man_hidden_trajectory(
    states: np.ndarray,
    *,
    variant: Literal["original", "perturbed_b1"],
    parameter_multipliers: dict[str, float] | None = None,
) -> np.ndarray:
    """Return the benchmark-declared private B1 hidden mechanisms."""

    p = DallaManParameters()
    multipliers = parameter_multipliers or {}
    if multipliers:
        p = replace(
            p,
            **{
                name: float(getattr(p, name)) * float(multiplier)
                for name, multiplier in multipliers.items()
            },
        )
    basal = compute_dalla_man_basal(p)
    values = np.asarray(states, dtype=float)
    gp = np.maximum(values[:, STATE_INDEX["Gp"]], 0.0)
    gt = np.maximum(values[:, STATE_INDEX["Gt"]], 0.0)
    qgut = np.maximum(values[:, STATE_INDEX["Qgut"]], 0.0)
    appearance = p.f * p.kabs * qgut / p.BW
    base = [
        appearance,
        values[:, STATE_INDEX["Qsto1"]],
        values[:, STATE_INDEX["Qsto2"]],
        qgut,
    ]
    if variant == "original":
        return np.column_stack(base)
    gp_b = float(basal.initial_state[STATE_INDEX["Gp"]])
    gt_b = float(basal.initial_state[STATE_INDEX["Gt"]])
    plasma_to_tissue = (
        p.k1
        * gp
        * (
            1.0
            + p.exchange_eta_p
            * np.tanh((gp - gp_b) / (p.exchange_scale_p_fraction * gp_b))
        )
    )
    tissue_to_plasma = (
        p.k2
        * gt
        * (
            1.0
            + p.exchange_eta_t
            * np.tanh((gt - gt_b) / (p.exchange_scale_t_fraction * gt_b))
        )
    )
    return np.column_stack(
        [
            *base,
            gt,
            plasma_to_tissue,
            tissue_to_plasma,
            plasma_to_tissue - tissue_to_plasma,
        ]
    )


def _meal_reference_at(time: float, events: dict[float, float]) -> float:
    """Return the most recent meal magnitude for derived gastric quantities."""

    eligible = [event for event in events if event <= time + 1e-12]
    return events[max(eligible)] if eligible else 0.0


def _piecewise_forcing_value(
    time: float, segments: tuple[tuple[float, float, float], ...]
) -> float:
    """Evaluate half-open piecewise-constant forcing segments."""

    return float(sum(value for start, end, value in segments if start <= time < end))


def _validate_external_forcing(
    forcing: DallaManExternalForcing, duration: float
) -> None:
    """Reject malformed or out-of-window forcing segments."""

    for segments in (
        forcing.glucose_mg_per_kg_min,
        forcing.insulin_pmol_per_kg_min,
    ):
        for start, end, value in segments:
            values = np.asarray((start, end, value), dtype=float)
            if (
                not np.all(np.isfinite(values))
                or start < 0.0
                or end <= start
                or end > duration
            ):
                raise ValueError("invalid Dalla Man external-forcing segment")
