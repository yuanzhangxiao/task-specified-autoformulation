"""Task-specific pre-release identifiability gates for Phase-B benchmarks."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from autoformalism.benchmarks.phase_b_generation import (
    BasicGateReport,
    Family,
    PhaseBProtocol,
    PrivateTrajectory,
    audit_basic_gates,
    phase_b_protocols,
    simulate_phase_b,
)

Tier = Literal["easy", "hard"]


class MechanismGateDefinition(BaseModel):
    """Frozen private mechanism subspace claimed by one benchmark cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Family
    task: str
    tier: Tier
    mechanisms: tuple[str, ...] = Field(min_length=1)
    sensitivity_outputs: tuple[str, ...] = Field(min_length=1)
    ablation_outputs: tuple[str, ...] = Field(min_length=1)
    claimed_dimension: int = Field(gt=0)

    @model_validator(mode="after")
    def dimension_is_supported(self) -> MechanismGateDefinition:
        if self.claimed_dimension > len(self.mechanisms):
            raise ValueError("claimed dimension exceeds mechanism directions")
        return self


class MechanismAblationResult(BaseModel):
    """Normalized validation discrepancy caused by one private ablation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mechanism: str
    normalized_discrepancy: float = Field(ge=0.0)
    threshold: float = Field(ge=0.0)
    passed: bool


class TaskGateReport(BaseModel):
    """Complete pre-release report for one family/task/tier cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Family
    task: str
    tier: Tier
    dynamics: Literal["canonical", "perturbed"]
    mechanisms: tuple[str, ...]
    sensitivity_outputs: tuple[str, ...]
    singular_values: tuple[float, ...]
    relative_singular_values: tuple[float, ...]
    claimed_dimension: int
    rank_at_1e3: int
    claimed_subspace_condition_number: float
    stable_rank: float
    rank_pass: bool
    condition_pass: bool
    stable_rank_pass: bool
    ablations: tuple[MechanismAblationResult, ...]
    ablation_pass: bool
    basic: BasicGateReport
    release_ready: bool


def mechanism_gate_definition(
    family: Family,
    tier: Tier,
    *,
    task: str | None = None,
    data_root: Path = Path("data_raw"),
) -> MechanismGateDefinition:
    """Return the prespecified mechanism directions and public outputs."""

    if family == "dalla_man":
        return _dalla_definition(task or "T1", tier)
    if task is not None:
        raise ValueError("task is only valid for Dalla Man")
    if family == "cstr":
        if tier == "easy":
            return MechanismGateDefinition(
                family=family,
                task="controlled_reactor_mechanism",
                tier=tier,
                mechanisms=(
                    "feed_transport",
                    "reaction_heat",
                    "jacket_exchange",
                ),
                sensitivity_outputs=("T", "C", "Tj"),
                ablation_outputs=("T",),
                claimed_dimension=2,
            )
        return MechanismGateDefinition(
            family=family,
            task="controlled_reactor_mechanism",
            tier=tier,
            mechanisms=(
                "reactant_reaction",
                "feed_transport",
                "reaction_heat",
                "jacket_exchange",
                "jacket_dynamics",
            ),
            sensitivity_outputs=("T",),
            ablation_outputs=("T",),
            claimed_dimension=2,
        )
    auxiliaries = alien_sensitivity_selected_auxiliaries(data_root)
    if tier == "easy":
        return MechanismGateDefinition(
            family=family,
            task="unknown_device_mechanism",
            tier=tier,
            mechanisms=("input_drive", "output_generation"),
            sensitivity_outputs=("y", *auxiliaries),
            ablation_outputs=("y",),
            claimed_dimension=2,
        )
    return MechanismGateDefinition(
        family=family,
        task="unknown_device_mechanism",
        tier=tier,
        mechanisms=(
            "input_drive",
            "persistent_coupling",
            "nonlinear_feedback",
            "output_generation",
        ),
        sensitivity_outputs=("y",),
        ablation_outputs=("y",),
        claimed_dimension=3,
    )


def audit_task_gates(
    definition: MechanismGateDefinition,
    *,
    dynamics: Literal["canonical", "perturbed"] = "canonical",
    data_root: Path = Path("data_raw"),
    sensitivity_fraction: float = 1e-3,
    ablation_scale: float = 1e-6,
) -> TaskGateReport:
    """Run sensitivity and ablation gates without discovery-method outputs."""

    if sensitivity_fraction <= 0.0:
        raise ValueError("sensitivity_fraction must be positive")
    if not 0.0 < ablation_scale < 1.0:
        raise ValueError("ablation_scale must lie in (0, 1)")
    task = definition.task if definition.family == "dalla_man" else None
    protocols = phase_b_protocols(definition.family, task=task)
    training = tuple(item for item in protocols if item.split == "train")
    validation = tuple(item for item in protocols if item.split == "validation")
    nominal_training = _simulate_all(training, dynamics, data_root)
    nominal_validation = _simulate_all(validation, dynamics, data_root)
    scales = _output_scales(nominal_training, definition.sensitivity_outputs)
    columns: list[NDArray[np.float64]] = []
    for mechanism in definition.mechanisms:
        shifted = tuple(
            simulate_phase_b(
                protocol,
                dynamics=dynamics,
                data_root=data_root,
                private_mechanism_scales={mechanism: 1.0 + sensitivity_fraction},
            )
            for protocol in training
        )
        columns.append(
            _stacked_normalized_difference(
                shifted,
                nominal_training,
                definition.sensitivity_outputs,
                scales,
            )
            / sensitivity_fraction
        )
    singular = np.linalg.svd(np.column_stack(columns), compute_uv=False)
    leading = max(float(singular[0]), 1e-15)
    relative = singular / leading
    rank = int(np.count_nonzero(relative >= 1e-3))
    claimed_index = definition.claimed_dimension - 1
    claimed_condition = leading / max(float(singular[claimed_index]), 1e-15)
    stable_rank = float(np.sum(singular**2) / max(leading**2, 1e-30))
    condition_threshold = 1_000.0 if definition.tier == "easy" else 5_000.0
    stable_threshold = (
        0.50 if definition.tier == "easy" else 0.35
    ) * definition.claimed_dimension
    ablation_threshold = 0.20 if definition.tier == "easy" else 0.15
    ablation_scales = _output_scales(nominal_training, definition.ablation_outputs)
    ablations: list[MechanismAblationResult] = []
    for mechanism in definition.mechanisms:
        ablated = tuple(
            simulate_phase_b(
                protocol,
                dynamics=dynamics,
                data_root=data_root,
                private_mechanism_scales={mechanism: ablation_scale},
            )
            for protocol in validation
        )
        discrepancy = _normalized_mse(
            ablated,
            nominal_validation,
            definition.ablation_outputs,
            ablation_scales,
        )
        ablations.append(
            MechanismAblationResult(
                mechanism=mechanism,
                normalized_discrepancy=discrepancy,
                threshold=ablation_threshold,
                passed=discrepancy >= ablation_threshold,
            )
        )
    target_name = definition.ablation_outputs[0]
    basic = audit_basic_gates(
        definition.family,
        nominal_training,
        target_name=target_name,
    )
    rank_pass = rank >= definition.claimed_dimension
    condition_pass = claimed_condition <= condition_threshold
    stable_pass = stable_rank >= stable_threshold
    ablation_pass = all(item.passed for item in ablations)
    basic_pass = (
        basic.finite_rollouts_pass
        and basic.input_design_pass
        and basic.persistence_pass
    )
    return TaskGateReport(
        family=definition.family,
        task=definition.task,
        tier=definition.tier,
        dynamics=dynamics,
        mechanisms=definition.mechanisms,
        sensitivity_outputs=definition.sensitivity_outputs,
        singular_values=tuple(float(value) for value in singular),
        relative_singular_values=tuple(float(value) for value in relative),
        claimed_dimension=definition.claimed_dimension,
        rank_at_1e3=rank,
        claimed_subspace_condition_number=claimed_condition,
        stable_rank=stable_rank,
        rank_pass=rank_pass,
        condition_pass=condition_pass,
        stable_rank_pass=stable_pass,
        ablations=tuple(ablations),
        ablation_pass=ablation_pass,
        basic=basic,
        release_ready=(
            rank_pass
            and condition_pass
            and stable_pass
            and ablation_pass
            and basic_pass
        ),
    )


@cache
def alien_sensitivity_selected_auxiliaries(
    data_root: Path = Path("data_raw"),
) -> tuple[str, str]:
    """Select telemetry by empirical influence of latent initial states on ``y``."""

    path = data_root / "benchmark6_alien_device/private/selected_system_spec.json"
    count = int(json.loads(path.read_text(encoding="utf-8"))["n_latent"])
    protocols = tuple(
        item for item in phase_b_protocols("alien_device") if item.split == "train"
    )
    nominal = _simulate_all(protocols, "canonical", data_root)
    scale = _output_scales(nominal, ("y",))
    scores = np.zeros(count, dtype=float)
    perturbation = 1e-2
    for index in range(count):
        shifted = tuple(
            simulate_phase_b(
                protocol,
                data_root=data_root,
                private_initial_offsets={f"z{index + 1}": perturbation},
            )
            for protocol in protocols
        )
        sensitivity = (
            _stacked_normalized_difference(shifted, nominal, ("y",), scale)
            / perturbation
        )
        scores[index] = float(np.linalg.norm(sensitivity))
    selected = np.argsort(-scores, kind="stable")[:2]
    return tuple(f"z{int(index) + 1}" for index in selected)  # type: ignore[return-value]


def _dalla_definition(task: str, tier: Tier) -> MechanismGateDefinition:
    outputs = {
        "T1": {
            "easy": ("Gp", "EGP", "Uii", "E", "Gt"),
            "hard": ("Gp", "Gt"),
        },
        "T2": {
            "easy": ("Gp", "I", "U", "EGP", "Uii", "E", "Gt"),
            "hard": ("Gp", "I", "Uii"),
        },
        "T3": {
            "easy": ("Gp", "I", "EGP", "U", "Uii", "E", "Gt", "Ipo"),
            "hard": ("Gp", "I", "EGP"),
        },
        "T4": {
            "easy": ("Gp", "I", "Uii", "E", "Gt", "Ipo"),
            "hard": ("Gp", "I"),
        },
    }
    mechanisms = {
        "T1": {
            "easy": ("gastric_memory", "meal_appearance"),
            "hard": ("gastric_memory", "meal_appearance", "glucose_exchange"),
        },
        "T2": {
            "easy": ("delayed_disposal",),
            "hard": ("meal_appearance", "delayed_disposal", "glucose_exchange"),
        },
        "T3": {
            "easy": ("hepatic_regulation",),
            "hard": ("meal_appearance", "delayed_disposal", "hepatic_regulation"),
        },
        "T4": {
            "easy": (
                "meal_appearance",
                "delayed_disposal",
                "hepatic_regulation",
                "insulin_secretion",
            ),
            "hard": (
                "meal_appearance",
                "delayed_disposal",
                "hepatic_regulation",
                "insulin_secretion",
            ),
        },
    }
    if task not in outputs:
        raise ValueError(f"unknown Dalla Man task {task!r}")
    selected = mechanisms[task][tier]
    claimed_dimension = 3 if task == "T4" and tier == "easy" else len(selected)
    ablation_outputs = {
        "T1": ("Gp",),
        "T2": ("Gp", "I", "U") if tier == "easy" else ("Gp", "I"),
        "T3": ("Gp", "I", "EGP"),
        "T4": ("Gp", "I"),
    }[task]
    return MechanismGateDefinition(
        family="dalla_man",
        task=task,
        tier=tier,
        mechanisms=selected,
        sensitivity_outputs=outputs[task][tier],
        ablation_outputs=ablation_outputs,
        claimed_dimension=claimed_dimension,
    )


def _simulate_all(
    protocols: tuple[PhaseBProtocol, ...],
    dynamics: Literal["canonical", "perturbed"],
    data_root: Path,
) -> tuple[PrivateTrajectory, ...]:
    return tuple(
        simulate_phase_b(
            protocol,
            dynamics=dynamics,
            data_root=data_root,
        )
        for protocol in protocols
    )


def _values(
    trajectory: PrivateTrajectory, outputs: tuple[str, ...]
) -> NDArray[np.float64]:
    columns: list[NDArray[np.float64]] = []
    for name in outputs:
        if name in trajectory.state_names:
            columns.append(trajectory.states[:, trajectory.state_names.index(name)])
        elif name in trajectory.derived:
            columns.append(trajectory.derived[name])
        else:
            raise ValueError(f"private output {name!r} is unavailable")
    return np.column_stack(columns)


def _output_scales(
    trajectories: tuple[PrivateTrajectory, ...], outputs: tuple[str, ...]
) -> NDArray[np.float64]:
    values = np.concatenate([_values(item, outputs) for item in trajectories])
    return np.maximum(np.std(values, axis=0), 1e-8)


def _stacked_normalized_difference(
    shifted: tuple[PrivateTrajectory, ...],
    nominal: tuple[PrivateTrajectory, ...],
    outputs: tuple[str, ...],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.concatenate(
        [
            ((_values(left, outputs) - _values(right, outputs)) / scales).ravel()
            for left, right in zip(shifted, nominal, strict=True)
        ]
    )


def _normalized_mse(
    shifted: tuple[PrivateTrajectory, ...],
    nominal: tuple[PrivateTrajectory, ...],
    outputs: tuple[str, ...],
    scales: NDArray[np.float64],
) -> float:
    residual = _stacked_normalized_difference(shifted, nominal, outputs, scales)
    return float(np.mean(residual**2))
