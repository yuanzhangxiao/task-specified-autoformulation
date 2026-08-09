"""Private post-selection empirical observability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from autoformalism.rebuttal.dalla_man import (
    STATE_INDEX,
    DallaManParameters,
    DallaManTrajectory,
    compute_dalla_man_basal,
    simulate_dalla_man,
)


@dataclass(frozen=True)
class ObservabilityResult:
    """Scaled singular spectrum for a declared hidden-state subset."""

    task: str
    protocol: str
    outputs: tuple[str, ...]
    hidden_states: tuple[str, ...]
    singular_values: tuple[float, ...]
    rank_at_1e3: int
    rank_at_1e6: int
    stable_rank: float
    condition_number: float


@dataclass(frozen=True)
class ParameterSensitivityResult:
    """Scaled local output/flux sensitivity spectrum for task parameters."""

    task: str
    protocol: str
    quantity_kind: str
    quantities: tuple[str, ...]
    parameters: tuple[str, ...]
    singular_values: tuple[float, ...]
    rank_at_1e3: int
    rank_at_1e6: int
    stable_rank: float
    condition_number: float


TASK_HIDDEN_STATES: dict[str, tuple[str, ...]] = {
    "T1": ("Qsto1", "Qsto2", "Qgut"),
    "T2": ("Qsto1", "Qsto2", "Qgut", "X"),
    "T3": ("Qsto1", "Qsto2", "Qgut", "X", "I1", "Id"),
    "T4": ("Qsto1", "Qsto2", "Qgut", "X", "Ipo", "Y", "I1", "Id"),
}

TASK_OUTPUTS: dict[str, tuple[str, ...]] = {
    "T1": ("Gp",),
    "T2": ("Gp", "I"),
    "T3": ("Gp", "I"),
    "T4": ("Gp", "I"),
}

TASK_PARAMETERS: dict[str, tuple[str, ...]] = {
    "T1": ("kgri", "kmax", "kmin", "kabs", "f"),
    "T2": ("kgri", "kabs", "f", "p2U", "Vm0", "Vmx", "Km0"),
    "T3": (
        "kgri",
        "kabs",
        "f",
        "p2U",
        "Vmx",
        "kp1",
        "kp2",
        "kp3",
        "kp4",
        "ki",
    ),
    "T4": (
        "kgri",
        "kabs",
        "f",
        "p2U",
        "Vmx",
        "kp1",
        "kp2",
        "kp3",
        "kp4",
        "ki",
        "alpha",
        "beta",
        "gamma",
        "m1",
        "m2",
        "m4",
        "m5",
        "m6",
    ),
}

TASK_FLUXES: dict[str, tuple[str, ...]] = {
    "T1": ("Ra",),
    "T2": ("Ra", "U"),
    "T3": ("Ra", "U", "EGP"),
    "T4": ("Ra", "U", "EGP", "S"),
}


def empirical_dalla_observability(
    task: str,
    *,
    meals: tuple[tuple[float, float], ...],
    duration: float = 300.0,
    dt: float = 1.0,
    protocol: str = "single_meal",
    perturbation_fraction: float = 1e-4,
    initial_state: tuple[float, ...] | None = None,
) -> ObservabilityResult:
    """Estimate local task-state observability from scaled output sensitivities.

    This private diagnostic perturbs each task-relevant initial hidden coordinate
    and observes the resulting output trajectory. It does not alter or inform a
    discovered model. Numerical rank is reported at two relative thresholds to
    expose conditioning rather than imply a binary identifiability theorem.
    """

    if task not in TASK_HIDDEN_STATES:
        raise ValueError(f"unknown Dalla Man task {task!r}")
    if perturbation_fraction <= 0.0:
        raise ValueError("perturbation_fraction must be positive")
    basal = compute_dalla_man_basal(DallaManParameters())
    nominal_initial = (
        basal.initial_state.copy()
        if initial_state is None
        else np.asarray(initial_state, dtype=float).copy()
    )
    base = simulate_dalla_man(
        meals=meals,
        duration=duration,
        dt=dt,
        variant="original",
        initial_state=tuple(nominal_initial),
    )
    outputs = TASK_OUTPUTS[task]
    hidden = TASK_HIDDEN_STATES[task]
    base_output = _outputs(base, outputs)
    output_scale = np.maximum(np.std(base_output, axis=0), 1e-8)
    columns: list[NDArray[np.float64]] = []
    for state_name in hidden:
        state_index = STATE_INDEX[state_name]
        characteristic = max(
            abs(float(nominal_initial[state_index])),
            float(np.max(np.abs(base.states[:, state_index]))),
            1.0,
        )
        initial = nominal_initial.copy()
        initial[state_index] += perturbation_fraction * characteristic
        shifted = simulate_dalla_man(
            meals=meals,
            duration=duration,
            dt=dt,
            variant="original",
            initial_state=tuple(initial),
        )
        normalized_sensitivity = (
            (_outputs(shifted, outputs) - base_output)
            / output_scale[np.newaxis, :]
            / perturbation_fraction
        )
        columns.append(normalized_sensitivity.reshape(-1))
    singular, rank_1e3, rank_1e6, stable_rank, condition = _spectrum(
        np.column_stack(columns)
    )
    return ObservabilityResult(
        task=task,
        protocol=protocol,
        outputs=outputs,
        hidden_states=hidden,
        singular_values=tuple(float(value) for value in singular),
        rank_at_1e3=rank_1e3,
        rank_at_1e6=rank_1e6,
        stable_rank=stable_rank,
        condition_number=condition,
    )


def empirical_dalla_parameter_sensitivity(
    task: str,
    *,
    meals: tuple[tuple[float, float], ...],
    duration: float = 300.0,
    dt: float = 1.0,
    protocol: str = "single_meal",
    quantity_kind: str = "outputs",
    perturbation_fraction: float = 1e-4,
) -> ParameterSensitivityResult:
    """Estimate practical local sensitivity of outputs or private fluxes."""

    if task not in TASK_PARAMETERS:
        raise ValueError(f"unknown Dalla Man task {task!r}")
    if quantity_kind not in {"outputs", "fluxes"}:
        raise ValueError("quantity_kind must be 'outputs' or 'fluxes'")
    parameters = TASK_PARAMETERS[task]
    quantities = TASK_OUTPUTS[task] if quantity_kind == "outputs" else TASK_FLUXES[task]
    basal = compute_dalla_man_basal(DallaManParameters())
    nominal_initial = tuple(basal.initial_state)
    base = simulate_dalla_man(
        meals=meals,
        duration=duration,
        dt=dt,
        variant="original",
        initial_state=nominal_initial,
    )
    base_values = (
        _outputs(base, quantities)
        if quantity_kind == "outputs"
        else _derived(base, quantities)
    )
    scales = np.maximum(np.std(base_values, axis=0), 1e-8)
    columns: list[NDArray[np.float64]] = []
    for parameter in parameters:
        shifted = simulate_dalla_man(
            meals=meals,
            duration=duration,
            dt=dt,
            variant="original",
            parameter_multipliers={parameter: 1.0 + perturbation_fraction},
            initial_state=nominal_initial,
        )
        shifted_values = (
            _outputs(shifted, quantities)
            if quantity_kind == "outputs"
            else _derived(shifted, quantities)
        )
        columns.append(
            (
                (shifted_values - base_values) / scales / perturbation_fraction
            ).reshape(-1)
        )
    singular, rank_1e3, rank_1e6, stable_rank, condition = _spectrum(
        np.column_stack(columns)
    )
    return ParameterSensitivityResult(
        task=task,
        protocol=protocol,
        quantity_kind=quantity_kind,
        quantities=quantities,
        parameters=parameters,
        singular_values=tuple(float(value) for value in singular),
        rank_at_1e3=rank_1e3,
        rank_at_1e6=rank_1e6,
        stable_rank=stable_rank,
        condition_number=condition,
    )


def _spectrum(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int, int, float, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    leading = max(float(singular[0]), 1e-15)
    relative = singular / leading
    smallest = float(singular[-1])
    return (
        singular,
        int(np.count_nonzero(relative >= 1e-3)),
        int(np.count_nonzero(relative >= 1e-6)),
        float(np.sum(singular**2) / max(float(singular[0] ** 2), 1e-30)),
        leading / smallest if smallest > 0.0 else float("inf"),
    )


def _outputs(
    trajectory: DallaManTrajectory,
    names: tuple[str, ...],
) -> NDArray[np.float64]:
    values: list[NDArray[np.float64]] = []
    for name in names:
        if name == "I":
            values.append(trajectory.derived["I"])
        elif name in STATE_INDEX:
            values.append(trajectory.states[:, STATE_INDEX[name]])
        else:
            raise ValueError(f"unsupported observability output {name!r}")
    return np.column_stack(values)


def _derived(
    trajectory: DallaManTrajectory,
    names: tuple[str, ...],
) -> NDArray[np.float64]:
    return np.column_stack([trajectory.derived[name] for name in names])
