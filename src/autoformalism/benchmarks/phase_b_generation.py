"""Private Phase-B trajectory generation and pre-release diagnostics.

Nothing in this module registers generated data as a production benchmark.
It executes the frozen protocol so that private reference trajectories can be
audited before public assets or prompts are created.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import solve_ivp

from autoformalism.rebuttal.dalla_man import (
    STATE_INDEX,
    DallaManExternalForcing,
    DallaManParameters,
    compute_dalla_man_basal,
    simulate_dalla_man,
)

Family = Literal["dalla_man", "cstr", "alien_device"]
Split = Literal["train", "validation", "test"]


class PhaseBProtocol(BaseModel):
    """One immutable simulator protocol in the frozen Phase-B design."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_id: str = Field(pattern=r"^(train|validation|test)_[a-z0-9_]+$")
    family: Family
    split: Split
    duration: float = Field(gt=0.0)
    dt: float = Field(gt=0.0)
    input_names: tuple[str, ...] = Field(min_length=1)
    specification: dict[str, Any]

    @model_validator(mode="after")
    def id_matches_split(self) -> PhaseBProtocol:
        """Keep split membership explicit in both ID and typed field."""

        if not self.protocol_id.startswith(f"{self.split}_"):
            raise ValueError("protocol ID must begin with its split")
        if self.dt > self.duration:
            raise ValueError("dt must not exceed duration")
        return self


class PrivateTrajectory(BaseModel):
    """Full private reference returned by a trusted numerical simulator."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    protocol_id: str
    family: Family
    time: NDArray[np.float64]
    state_names: tuple[str, ...]
    states: NDArray[np.float64]
    input_names: tuple[str, ...]
    inputs: NDArray[np.float64]
    derived: dict[str, NDArray[np.float64]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def arrays_are_aligned(self) -> PrivateTrajectory:
        """Reject nonfinite or dimensionally inconsistent simulations."""

        if self.states.shape != (len(self.time), len(self.state_names)):
            raise ValueError("state array does not match time/state names")
        if self.inputs.shape != (len(self.time), len(self.input_names)):
            raise ValueError("input array does not match time/input names")
        arrays = [self.time, self.states, self.inputs, *self.derived.values()]
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("private trajectory contains nonfinite values")
        if len(self.time) < 2 or not np.all(np.diff(self.time) > 0.0):
            raise ValueError("trajectory time must be strictly increasing")
        return self


class BasicGateReport(BaseModel):
    """Simulator-only gates available before task-specific sensitivity audits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Family
    finite_rollout_fraction: float = Field(ge=0.0, le=1.0)
    input_gram_eigenvalue_ratio: float = Field(ge=0.0)
    persistence_nmse: float = Field(ge=0.0)
    finite_rollouts_pass: bool
    input_design_pass: bool
    persistence_pass: bool
    standalone_release_ready: Literal[False] = False
    pending_gates: tuple[str, ...] = (
        "task_scaled_sensitivity_rank",
        "claimed_subspace_condition_number",
        "stable_rank",
        "required_mechanism_ablation_separability",
    )


def phase_b_protocols(
    family: Family, *, task: str | None = None
) -> tuple[PhaseBProtocol, ...]:
    """Construct the 16/4/6 frozen protocols for one family."""

    if family == "dalla_man":
        return _dalla_protocols(task or "T1")
    if task is not None:
        raise ValueError("task is only used for Dalla Man protocols")
    if family == "cstr":
        return _cstr_protocols()
    return _alien_protocols()


def simulate_phase_b(
    protocol: PhaseBProtocol,
    *,
    dynamics: Literal["canonical", "perturbed"] = "canonical",
    data_root: Path = Path("data_raw"),
    private_mechanism_scales: Mapping[str, float] | None = None,
    private_initial_offsets: Mapping[str, float] | None = None,
) -> PrivateTrajectory:
    """Execute one private Phase-B protocol without exposing private truth.

    ``private_mechanism_scales`` is reserved for pre-release sensitivity and
    ablation audits. It must never be serialized into a public benchmark.
    """

    scales = private_mechanism_scales or {}
    offsets = private_initial_offsets or {}
    if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
        raise ValueError("private mechanism scales must be finite and positive")
    if any(not np.isfinite(value) for value in offsets.values()):
        raise ValueError("private initial offsets must be finite")

    if protocol.family == "dalla_man":
        return _simulate_dalla(protocol, dynamics, scales, offsets)
    if dynamics != "canonical":
        raise ValueError("only Dalla Man defines a perturbed dynamics condition")
    if protocol.family == "cstr":
        spec = _load_json(
            data_root
            / "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
        )
        return _simulate_cstr(protocol, _scale_cstr_spec(spec, scales), offsets)
    spec = _load_json(
        data_root / "benchmark6_alien_device/private/selected_system_spec.json"
    )
    return _simulate_alien(protocol, _scale_alien_spec(spec, scales), offsets)


def audit_basic_gates(
    family: Family,
    trajectories: tuple[PrivateTrajectory, ...],
    *,
    target_name: str,
) -> BasicGateReport:
    """Evaluate finite rollout, excitation, and persistence gates on training data."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if any(item.family != family for item in trajectories):
        raise ValueError("all trajectories must belong to the audited family")
    finite_fraction = float(
        np.mean(
            [
                np.all(np.isfinite(item.states)) and np.all(np.isfinite(item.inputs))
                for item in trajectories
            ]
        )
    )
    input_matrix = np.concatenate(
        [_standardize_columns(item.inputs) for item in trajectories], axis=0
    )
    gram = input_matrix.T @ input_matrix / max(len(input_matrix), 1)
    eigenvalues = np.linalg.eigvalsh(gram)
    gram_ratio = float(max(eigenvalues[0], 0.0) / max(eigenvalues[-1], 1e-15))
    persistence = _persistence_nmse(trajectories, target_name, family)
    persistence_threshold = 0.25 if family == "dalla_man" else 0.5
    return BasicGateReport(
        family=family,
        finite_rollout_fraction=finite_fraction,
        input_gram_eigenvalue_ratio=gram_ratio,
        persistence_nmse=persistence,
        finite_rollouts_pass=finite_fraction == 1.0,
        input_design_pass=gram_ratio >= 1e-3,
        persistence_pass=persistence >= persistence_threshold,
    )


def write_private_bundle(
    output_root: Path,
    protocols: tuple[PhaseBProtocol, ...],
    trajectories: tuple[PrivateTrajectory, ...],
) -> None:
    """Write a private, versioned NPZ bundle; never a public benchmark asset."""

    if tuple(item.protocol_id for item in protocols) != tuple(
        item.protocol_id for item in trajectories
    ):
        raise ValueError("protocol and trajectory order must match")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "phase_b_private_bundle_v1",
        "private_reference": True,
        "available_to_discovery_methods": False,
        "protocols": [item.model_dump(mode="json") for item in protocols],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    for item in trajectories:
        np.savez_compressed(
            output_root / f"{item.protocol_id}.npz",
            time=item.time,
            states=item.states,
            inputs=item.inputs,
            state_names=np.asarray(item.state_names),
            input_names=np.asarray(item.input_names),
            **{f"derived__{name}": value for name, value in item.derived.items()},
        )


def _protocol(
    family: Family,
    split: Split,
    name: str,
    duration: float,
    dt: float,
    inputs: tuple[str, ...],
    **specification: Any,
) -> PhaseBProtocol:
    return PhaseBProtocol(
        protocol_id=f"{split}_{name}",
        family=family,
        split=split,
        duration=duration,
        dt=dt,
        input_names=inputs,
        specification=specification,
    )


def _dalla_protocols(task: str) -> tuple[PhaseBProtocol, ...]:
    if task not in {"T1", "T2", "T3", "T4"}:
        raise ValueError(f"unknown Dalla Man task {task!r}")
    base: list[tuple[str, dict[str, Any]]] = [
        ("basal", {"meals": []}),
        *[
            (f"meal_{grams}g_t0", {"meals": [[0, grams]]})
            for grams in (30, 60, 90, 120, 150)
        ],
        ("meal_90g_t60", {"meals": [[60, 90]]}),
        ("meal_90g_t120", {"meals": [[120, 90]]}),
        ("meals_60_30", {"meals": [[0, 60], [90, 30]]}),
        ("meals_45_45", {"meals": [[0, 45], [60, 45]]}),
        ("meals_30_60", {"meals": [[0, 30], [120, 60]]}),
        (
            "initial_gp_minus10",
            {"meals": [], "initial_shift": {"Gp": 0.90}},
        ),
        ("initial_gp_plus10", {"meals": [], "initial_shift": {"Gp": 1.10}}),
        (
            "initial_insulin_minus15",
            {"meals": [], "initial_shift": {"Ip": 0.85, "Il": 0.85}},
        ),
        (
            "initial_insulin_plus15",
            {"meals": [], "initial_shift": {"Ip": 1.15, "Il": 1.15}},
        ),
    ]
    if task == "T1":
        base.append(
            ("three_separated_meals", {"meals": [[0, 30], [90, 30], [180, 30]]})
        )
    elif task == "T2":
        base.append(
            ("insulin_pulse", {"meals": [[0, 90]], "insulin": [[30, 45, 0.35]]})
        )
    else:
        base.append(
            (
                "insulin_glucose_clamp",
                {
                    "meals": [[0, 90]],
                    "insulin": [[30, 60, 0.35]],
                    "glucose": [[45, 90, 0.8]],
                },
            )
        )
    validation = [
        ("meal_75g_t30", {"meals": [[30, 75]]}),
        ("meal_105g_t90", {"meals": [[90, 105]]}),
        ("meals_30_75", {"meals": [[0, 30], [90, 75]]}),
        (
            "initial_gp_plus5_insulin_minus7p5",
            {
                "meals": [],
                "initial_shift": {"Gp": 1.05, "Ip": 0.925, "Il": 0.925},
            },
        ),
    ]
    strong = (
        {"meals": [[0, 30], [60, 30], [120, 30], [180, 30]]}
        if task == "T1"
        else {
            "meals": [[0, 90]],
            "insulin": [[20, 65, 0.55]],
            "glucose": [[40, 100, 1.2]] if task in {"T3", "T4"} else [],
        }
    )
    test = [
        ("meal_20g_t0", {"meals": [[0, 20]]}),
        ("meal_180g_t0", {"meals": [[0, 180]]}),
        ("meal_90g_t180", {"meals": [[180, 90]]}),
        ("three_30g_meals", {"meals": [[0, 30], [60, 30], [120, 30]]}),
        (
            "initial_gp_minus12_insulin_plus20",
            {
                "meals": [],
                "initial_shift": {"Gp": 0.88, "Ip": 1.20, "Il": 1.20},
            },
        ),
        ("strong_orthogonal", strong),
    ]
    inputs = {
        "T1": ("meal_event_g",),
        "T2": ("meal_event_g", "insulin_pmol_per_kg_min"),
        "T3": (
            "meal_event_g",
            "glucose_mg_per_kg_min",
            "insulin_pmol_per_kg_min",
        ),
        "T4": (
            "meal_event_g",
            "glucose_mg_per_kg_min",
            "insulin_pmol_per_kg_min",
        ),
    }[task]
    return tuple(
        _protocol("dalla_man", split, name, 300.0, 1.0, inputs, **spec)
        for split, rows in (("train", base), ("validation", validation), ("test", test))
        for name, spec in rows
    )


def _cstr_protocols() -> tuple[PhaseBProtocol, ...]:
    zeros = [0.0, 0.0, 0.0]
    train = [("zero", {"kind": "constant", "value": zeros})]
    for index, name in enumerate(("cf", "tf", "tjf")):
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            value = zeros.copy()
            value[index] = sign
            train.append(
                (
                    f"{name}_{label}_step",
                    {"kind": "step", "start": 5, "end": 25, "value": value},
                )
            )
        value = zeros.copy()
        value[index] = 1.0
        train.append(
            (f"{name}_pulse", {"kind": "step", "start": 8, "end": 12, "value": value})
        )
        amplitude = zeros.copy()
        amplitude[index] = 0.75
        train.append(
            (f"{name}_sine", {"kind": "sine", "period": 12, "amplitude": amplitude})
        )
    train += [
        (
            "cf_tf_pair",
            {"kind": "step", "start": 5, "end": 25, "value": [0.7, -0.7, 0]},
        ),
        (
            "tf_tjf_pair",
            {"kind": "step", "start": 5, "end": 25, "value": [0, 0.7, -0.7]},
        ),
        (
            "initial_dominant_plus",
            {"kind": "constant", "value": zeros, "initial_shift": [0.08, 5.0, 3.0]},
        ),
    ]
    validation = [
        ("cf_mid_step", {"kind": "step", "start": 6, "end": 22, "value": [0.5, 0, 0]}),
        (
            "tf_mid_pulse",
            {"kind": "step", "start": 10, "end": 15, "value": [0, -0.5, 0]},
        ),
        ("tjf_mid_sine", {"kind": "sine", "period": 16, "amplitude": [0, 0, 0.5]}),
        ("paired_mid", {"kind": "step", "start": 7, "end": 20, "value": [0.4, 0.4, 0]}),
    ]
    test = [
        (
            "cf_extrapolation",
            {"kind": "step", "start": 5, "end": 25, "value": [1.2, 0, 0]},
        ),
        (
            "fast_multisine",
            {"kind": "multisine", "periods": [4, 5, 6], "amplitude": [0.6, 0.6, 0.6]},
        ),
        ("delayed_pulse", {"kind": "step", "start": 20, "end": 24, "value": [0, 0, 1]}),
        (
            "opposing_tf_tjf",
            {"kind": "step", "start": 5, "end": 25, "value": [0, 1.2, -1.2]},
        ),
        (
            "initial_extrapolation",
            {"kind": "constant", "value": zeros, "initial_shift": [0.12, 8, 5]},
        ),
        (
            "combined_shift",
            {
                "kind": "step",
                "start": 8,
                "end": 20,
                "value": [0.8, -0.8, 0.8],
                "initial_shift": [-0.08, -6, -4],
            },
        ),
    ]
    return tuple(
        _protocol("cstr", split, name, 30.0, 0.1, ("Cf", "Tf", "Tjf"), **spec)
        for split, rows in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        for name, spec in rows
    )


def _alien_protocols() -> tuple[PhaseBProtocol, ...]:
    train = [("zero", {"kind": "constant", "value": 0.0})]
    train += [
        (f"pulse_{label}", {"kind": "pulse", "start": 8, "end": 12, "amplitude": value})
        for label, value in (("m1", -1), ("m0p5", -0.5), ("p0p5", 0.5), ("p1", 1))
    ]
    train += [
        (f"step_{label}", {"kind": "step", "start": 8, "end": 60, "amplitude": value})
        for label, value in (("m0p75", -0.75), ("p0p5", 0.5), ("p1", 1))
    ]
    train += [
        (f"sine_p{period}", {"kind": "sine", "period": period, "amplitude": 0.8})
        for period in (8, 16, 32)
    ]
    train += [
        (
            "chirp_32_to_8",
            {"kind": "chirp", "period_start": 32, "period_end": 8, "amplitude": 0.8},
        ),
        ("multi_pulse_a", {"kind": "pulses", "pulses": [[8, 12, 1], [32, 36, -0.7]]}),
        ("multi_pulse_b", {"kind": "pulses", "pulses": [[15, 20, -1], [42, 48, 0.8]]}),
        (
            "initial_shift_a",
            {"kind": "constant", "value": 0.0, "initial_shift": [0.5, 0, 0, 0, 0, 0]},
        ),
        (
            "initial_shift_b",
            {"kind": "constant", "value": 0.0, "initial_shift": [0, 0.5, 0, 0, 0, 0]},
        ),
    ]
    validation = [
        ("pulse_mid", {"kind": "pulse", "start": 10, "end": 15, "amplitude": 0.75}),
        (
            "pulse_negative_mid",
            {"kind": "pulse", "start": 20, "end": 26, "amplitude": -0.65},
        ),
        ("sine_p12", {"kind": "sine", "period": 12, "amplitude": 0.7}),
        ("sine_p24", {"kind": "sine", "period": 24, "amplitude": -0.7}),
    ]
    test = [
        (
            "amplitude_extrapolation",
            {"kind": "pulse", "start": 8, "end": 12, "amplitude": 1.2},
        ),
        ("fast_frequency", {"kind": "sine", "period": 5, "amplitude": 0.8}),
        (
            "unseen_chirp",
            {"kind": "chirp", "period_start": 20, "period_end": 5, "amplitude": 0.9},
        ),
        (
            "delayed_multi_pulse",
            {"kind": "pulses", "pulses": [[35, 39, 1], [50, 55, -1]]},
        ),
        (
            "initial_extrapolation",
            {"kind": "constant", "value": 0.0, "initial_shift": [0.8, 0, 0, 0, 0, 0]},
        ),
        (
            "combined_shift",
            {
                "kind": "pulse",
                "start": 20,
                "end": 28,
                "amplitude": -1.1,
                "initial_shift": [0, -0.8, 0, 0, 0, 0],
            },
        ),
    ]
    return tuple(
        _protocol("alien_device", split, name, 60.0, 0.1, ("u",), **spec)
        for split, rows in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        for name, spec in rows
    )


def _simulate_dalla(
    protocol: PhaseBProtocol,
    dynamics: str,
    mechanism_scales: Mapping[str, float],
    initial_offsets: Mapping[str, float],
) -> PrivateTrajectory:
    spec = protocol.specification
    basal = compute_dalla_man_basal(DallaManParameters())
    initial = basal.initial_state.copy()
    shifts = spec.get("initial_shift", {})
    if "Gt" in shifts:
        raise ValueError("Dalla Gt shifts are derived by glucose-mass conservation")
    if "Gp" in shifts:
        original_gp = float(initial[STATE_INDEX["Gp"]])
        shifted_gp = original_gp * float(shifts["Gp"])
        initial[STATE_INDEX["Gp"]] = shifted_gp
        initial[STATE_INDEX["Gt"]] += original_gp - shifted_gp
    for name, multiplier in shifts.items():
        if name == "Gp":
            continue
        initial[STATE_INDEX[name]] *= float(multiplier)
    unknown_offsets = set(initial_offsets).difference(STATE_INDEX)
    if unknown_offsets:
        raise ValueError(f"unknown Dalla initial offsets: {sorted(unknown_offsets)}")
    for name, offset in initial_offsets.items():
        initial[STATE_INDEX[name]] += float(offset)
    forcing = DallaManExternalForcing(
        glucose_mg_per_kg_min=tuple(
            tuple(map(float, row)) for row in spec.get("glucose", [])
        ),
        insulin_pmol_per_kg_min=tuple(
            tuple(map(float, row)) for row in spec.get("insulin", [])
        ),
    )
    result = simulate_dalla_man(
        meals=tuple(tuple(map(float, row)) for row in spec.get("meals", [])),
        duration=protocol.duration,
        dt=protocol.dt,
        variant="perturbed_b1" if dynamics == "perturbed" else "original",
        initial_state=tuple(initial),
        external_forcing=forcing,
        parameter_multipliers=_dalla_parameter_multipliers(mechanism_scales),
        basal_reference=basal,
    )
    available_inputs = {
        "meal_event_g": result.meal_event_g,
        "glucose_mg_per_kg_min": result.derived["glucose_forcing"],
        "insulin_pmol_per_kg_min": result.derived["insulin_forcing"],
    }
    inputs = np.column_stack([available_inputs[name] for name in protocol.input_names])
    return PrivateTrajectory(
        protocol_id=protocol.protocol_id,
        family="dalla_man",
        time=result.time,
        state_names=tuple(STATE_INDEX),
        states=result.states,
        input_names=protocol.input_names,
        inputs=inputs,
        derived=result.derived,
    )


def _simulate_cstr(
    protocol: PhaseBProtocol,
    spec: Mapping[str, Any],
    initial_offsets: Mapping[str, float],
) -> PrivateTrajectory:
    p = {name: float(value) for name, value in spec["parameters"].items()}
    equilibrium = np.asarray(
        [spec["equilibrium"][name] for name in ("C", "T", "Tj")], dtype=float
    )
    initial = equilibrium + np.asarray(
        protocol.specification.get("initial_shift", [0, 0, 0]), dtype=float
    )
    state_index = {name: index for index, name in enumerate(("C", "T", "Tj"))}
    unknown_offsets = set(initial_offsets).difference(state_index)
    if unknown_offsets:
        raise ValueError(f"unknown CSTR initial offsets: {sorted(unknown_offsets)}")
    for name, offset in initial_offsets.items():
        initial[state_index[name]] += float(offset)
    time = _time_grid(protocol)

    def physical_inputs(value: float) -> NDArray[np.float64]:
        normalized = _vector_input(value, protocol.specification, 3)
        return np.asarray(
            [p["C_feed_base"], p["T_feed_base"], p["T_secondary_feed_base"]]
        ) + normalized * np.asarray([0.25, 10.0, 12.0])

    def rhs(value: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        concentration, temperature, jacket = state
        cf, tf, tjf = physical_inputs(value)
        reaction = (
            p["k0"]
            * np.exp(-p["E_over_R"] / max(float(temperature), 250.0))
            * max(float(concentration), 0.0)
        )
        return np.asarray(
            [
                p["flow_rate"] * (cf - concentration) - reaction,
                p["flow_rate"] * (tf - temperature)
                + p["source_gain"] * reaction
                - p["exchange_rate"] * (temperature - jacket),
                p["secondary_flow_rate"] * (tjf - jacket)
                + p["secondary_exchange_rate"] * (temperature - jacket),
            ]
        )

    states = _integrate(rhs, time, initial)
    inputs = np.vstack([physical_inputs(float(value)) for value in time])
    return PrivateTrajectory(
        protocol_id=protocol.protocol_id,
        family="cstr",
        time=time,
        state_names=("C", "T", "Tj"),
        states=states,
        input_names=protocol.input_names,
        inputs=inputs,
    )


def _simulate_alien(
    protocol: PhaseBProtocol,
    spec: Mapping[str, Any],
    initial_offsets: Mapping[str, float],
) -> PrivateTrajectory:
    count = int(spec["n_latent"])
    decay = np.asarray(spec["decay"], dtype=float)
    skew = np.asarray(spec["skew"], dtype=float)
    initial = np.asarray(
        protocol.specification.get("initial_shift", [0.0] * (count + 1)), dtype=float
    )
    state_index = {
        **{f"z{index + 1}": index for index in range(count)},
        "y": count,
    }
    unknown_offsets = set(initial_offsets).difference(state_index)
    if unknown_offsets:
        raise ValueError(f"unknown alien initial offsets: {sorted(unknown_offsets)}")
    for name, offset in initial_offsets.items():
        initial[state_index[name]] += float(offset)
    time = _time_grid(protocol)

    def rhs(value: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        latent = state[:count]
        derivative = -decay * latent + skew @ latent
        for target, terms in enumerate(spec["tanh_terms"]):
            for term in terms:
                derivative[target] += float(term["coefficient"]) * np.tanh(
                    float(term["scale"]) * latent[int(term["source"])]
                    + float(term["bias"])
                )
        for target, terms in enumerate(spec["product_terms"]):
            for term in terms:
                derivative[target] += (
                    float(term["coefficient"])
                    * np.tanh(float(term["scale_1"]) * latent[int(term["source_1"])])
                    * np.tanh(float(term["scale_2"]) * latent[int(term["source_2"])])
                )
        forcing = _scalar_input(value, protocol.specification)
        derivative += np.asarray(spec["input_vector"]) * np.tanh(
            float(spec["input_scale"]) * forcing
        )
        output = -float(spec["output_decay"]) * state[count]
        for term in spec["output_terms"]:
            output += float(term["coefficient"]) * np.tanh(
                float(term["scale"]) * latent[int(term["source"])]
            )
        for term in spec["output_product_terms"]:
            output += (
                float(term["coefficient"])
                * np.tanh(float(term["scale_1"]) * latent[int(term["source_1"])])
                * np.tanh(float(term["scale_2"]) * latent[int(term["source_2"])])
            )
        return np.concatenate([derivative, [output]])

    states = _integrate(rhs, time, initial, rtol=1e-6, atol=1e-8)
    inputs = np.asarray(
        [[_scalar_input(float(value), protocol.specification)] for value in time]
    )
    return PrivateTrajectory(
        protocol_id=protocol.protocol_id,
        family="alien_device",
        time=time,
        state_names=(*(f"z{index + 1}" for index in range(count)), "y"),
        states=states,
        input_names=protocol.input_names,
        inputs=inputs,
    )


def _scalar_input(time: float, spec: Mapping[str, Any]) -> float:
    kind = spec["kind"]
    if kind == "constant":
        return float(spec["value"])
    if kind in {"pulse", "step"}:
        return (
            float(spec["amplitude"])
            if float(spec["start"]) <= time < float(spec["end"])
            else 0.0
        )
    if kind == "pulses":
        return float(
            sum(
                amplitude
                for start, end, amplitude in spec["pulses"]
                if start <= time < end
            )
        )
    if kind == "sine":
        return float(spec["amplitude"]) * np.sin(
            2 * np.pi * time / float(spec["period"])
        )
    if kind == "chirp":
        duration = 60.0
        f0, f1 = 1 / float(spec["period_start"]), 1 / float(spec["period_end"])
        phase = 2 * np.pi * (f0 * time + 0.5 * (f1 - f0) * time**2 / duration)
        return float(spec["amplitude"]) * np.sin(phase)
    raise ValueError(f"unsupported scalar input kind {kind!r}")


def _vector_input(
    time: float, spec: Mapping[str, Any], width: int
) -> NDArray[np.float64]:
    kind = spec["kind"]
    if kind == "constant":
        return np.asarray(spec["value"], dtype=float)
    if kind == "step":
        return (
            np.asarray(spec["value"], dtype=float)
            if float(spec["start"]) <= time < float(spec["end"])
            else np.zeros(width)
        )
    if kind == "sine":
        return np.asarray(spec["amplitude"], dtype=float) * np.sin(
            2 * np.pi * time / float(spec["period"])
        )
    if kind == "multisine":
        return np.asarray(
            [
                amplitude * np.sin(2 * np.pi * time / period)
                for amplitude, period in zip(
                    spec["amplitude"], spec["periods"], strict=True
                )
            ]
        )
    raise ValueError(f"unsupported vector input kind {kind!r}")


def _integrate(
    rhs: Any,
    time: NDArray[np.float64],
    initial: NDArray[np.float64],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> NDArray[np.float64]:
    solution = solve_ivp(
        rhs,
        (float(time[0]), float(time[-1])),
        initial,
        method="LSODA",
        t_eval=time,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"private reference simulation failed: {solution.message}")
    return solution.y.T


def _time_grid(protocol: PhaseBProtocol) -> NDArray[np.float64]:
    return np.arange(0.0, protocol.duration + 0.5 * protocol.dt, protocol.dt)


def _standardize_columns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - np.mean(values, axis=0)
    return centered / np.maximum(np.std(centered, axis=0), 1e-12)


def _persistence_nmse(
    trajectories: tuple[PrivateTrajectory, ...], target_name: str, family: Family
) -> float:
    values = [
        item.states[:, item.state_names.index(target_name)] for item in trajectories
    ]
    scale = max(float(np.std(np.concatenate(values))), 1e-12)
    errors: list[float] = []
    for _item, target in zip(trajectories, values, strict=True):
        if family == "dalla_man":
            horizon = min(30, len(target) - 1)
            residual = target[horizon:] - target[:-horizon]
        else:
            residual = target[1:] - target[0]
        errors.append(float(np.mean(residual**2) / scale**2))
    return float(np.mean(errors))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dalla_parameter_multipliers(
    mechanism_scales: Mapping[str, float],
) -> dict[str, float]:
    groups = {
        "gastric_memory": ("kgri", "kmax", "kmin"),
        "meal_appearance": ("kabs", "f"),
        "delayed_disposal": ("p2U", "Vmx"),
        "hepatic_regulation": ("kp2", "kp3", "kp4", "ki"),
        "glucose_exchange": ("k1", "k2"),
        "insulin_secretion": ("alpha", "beta", "gamma"),
    }
    unknown = set(mechanism_scales).difference(groups)
    if unknown:
        raise ValueError(f"unknown Dalla mechanism scales: {sorted(unknown)}")
    result: dict[str, float] = {}
    for mechanism, scale in mechanism_scales.items():
        for parameter in groups[mechanism]:
            if parameter in result:
                raise ValueError(f"Dalla parameter {parameter} belongs to two scales")
            result[parameter] = float(scale)
    return result


def _scale_cstr_spec(
    source: Mapping[str, Any], scales: Mapping[str, float]
) -> dict[str, Any]:
    groups = {
        "reactant_reaction": ("k0",),
        "feed_transport": ("flow_rate",),
        "reaction_heat": ("source_gain",),
        "jacket_exchange": ("exchange_rate",),
        "jacket_dynamics": ("secondary_flow_rate", "secondary_exchange_rate"),
    }
    unknown = set(scales).difference(groups)
    if unknown:
        raise ValueError(f"unknown CSTR mechanism scales: {sorted(unknown)}")
    spec = json.loads(json.dumps(source))
    for mechanism, scale in scales.items():
        for parameter in groups[mechanism]:
            spec["parameters"][parameter] *= float(scale)
    return spec


def _scale_alien_spec(
    source: Mapping[str, Any], scales: Mapping[str, float]
) -> dict[str, Any]:
    allowed = {
        "input_drive",
        "persistent_coupling",
        "nonlinear_feedback",
        "output_generation",
    }
    unknown = set(scales).difference(allowed)
    if unknown:
        raise ValueError(f"unknown alien mechanism scales: {sorted(unknown)}")
    spec = json.loads(json.dumps(source))
    if "input_drive" in scales:
        factor = float(scales["input_drive"])
        spec["input_vector"] = [factor * value for value in spec["input_vector"]]
    if "persistent_coupling" in scales:
        factor = float(scales["persistent_coupling"])
        spec["skew"] = [[factor * value for value in row] for row in spec["skew"]]
    if "nonlinear_feedback" in scales:
        factor = float(scales["nonlinear_feedback"])
        for collection in (spec["tanh_terms"], spec["product_terms"]):
            for terms in collection:
                for term in terms:
                    term["coefficient"] *= factor
    if "output_generation" in scales:
        factor = float(scales["output_generation"])
        for key in ("output_terms", "output_product_terms"):
            for term in spec[key]:
                term["coefficient"] *= factor
    return spec
