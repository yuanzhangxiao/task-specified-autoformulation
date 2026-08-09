"""Private post-selection intervention-suite schemas and reference simulators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import solve_ivp

from autoformalism.rebuttal.dalla_man import simulate_dalla_man

BenchmarkId = Literal[
    "original_b1",
    "perturbed_b1",
    "obfuscated_original_case01",
    "obfuscated_perturbed_case01",
    "benchmark5",
    "benchmark6",
]


class InterventionCase(BaseModel):
    """One prespecified intervention evaluated only after model selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    benchmark_id: BenchmarkId
    shift_types: tuple[str, ...]
    protocol: dict[str, Any]
    duration: float = Field(gt=0.0)
    dt: float = Field(gt=0.0)
    initial_state: tuple[float, ...]
    parameter_multipliers: dict[str, float] = Field(default_factory=dict)
    observation_stride: int = Field(default=1, ge=1)
    noise_fraction: float = Field(default=0.0, ge=0.0)
    noise_seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> InterventionCase:
        expected = {
            "benchmark5": 3,
            "benchmark6": 6,
            "original_b1": 12,
            "perturbed_b1": 12,
            "obfuscated_original_case01": 12,
            "obfuscated_perturbed_case01": 12,
        }[self.benchmark_id]
        if len(self.initial_state) != expected:
            raise ValueError(
                f"{self.benchmark_id} requires {expected} initial-state values"
            )
        if self.dt > self.duration:
            raise ValueError("dt must not exceed duration")
        if not self.shift_types:
            raise ValueError("at least one shift type is required")
        return self


class InterventionSuite(BaseModel):
    """Frozen intervention definition with explicit leakage controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    suite_id: str
    frozen_before_evaluation: Literal[True]
    uses_private_reference: Literal[True]
    available_to_proposal_fit_or_selection: Literal[False]
    cases: tuple[InterventionCase, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> InterventionSuite:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("intervention case IDs must be unique")
        if not self.cases:
            raise ValueError("intervention suite must contain at least one case")
        return self

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the complete frozen suite definition."""

        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ReferenceTrajectory(BaseModel):
    """Ground-truth trajectory generated from a private simulator."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    case_id: str
    benchmark_id: BenchmarkId
    time: tuple[float, ...]
    forcing: tuple[tuple[float, ...], ...]
    states_clean: tuple[tuple[float, ...], ...]
    states_observed: tuple[tuple[float, ...], ...]


def load_intervention_suite(path: Path) -> InterventionSuite:
    """Load and validate a frozen suite definition."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") == "template-1":
        cases = []
        for benchmark_id in payload["benchmark_ids"]:
            for template in payload["case_templates"]:
                case = json.loads(json.dumps(template))
                case["benchmark_id"] = benchmark_id
                case["case_id"] = f"{benchmark_id}_{case['case_id']}"
                cases.append(case)
        payload = {
            "schema_version": "1",
            "suite_id": payload["suite_id"],
            "frozen_before_evaluation": payload["frozen_before_evaluation"],
            "uses_private_reference": payload["uses_private_reference"],
            "available_to_proposal_fit_or_selection": payload[
                "available_to_proposal_fit_or_selection"
            ],
            "cases": cases,
        }
    return InterventionSuite.model_validate(payload)


def file_sha256(path: Path) -> str:
    """Hash a source artifact used to generate private references."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulate_reference(
    case: InterventionCase,
    *,
    system_spec: Mapping[str, Any],
) -> ReferenceTrajectory:
    """Simulate one trusted private reference without evaluating text equations."""

    if case.benchmark_id == "benchmark5":
        time, states, forcing = _simulate_benchmark5(case, system_spec)
    elif case.benchmark_id == "benchmark6":
        time, states, forcing = _simulate_benchmark6(case, system_spec)
    else:
        perturbed = case.benchmark_id in {
            "perturbed_b1",
            "obfuscated_perturbed_case01",
        }
        simulation = simulate_dalla_man(
            meals=tuple(
                (float(item["time"]), float(item["grams"]))
                for item in case.protocol["meals"]
            ),
            duration=case.duration,
            dt=case.dt,
            variant="perturbed_b1" if perturbed else "original",
            parameter_multipliers=case.parameter_multipliers,
            initial_state=case.initial_state,
        )
        time = simulation.time
        states = simulation.states
        forcing = simulation.meal_event_g[:, None]

    indices = np.arange(0, len(time), case.observation_stride, dtype=int)
    time = time[indices]
    states = states[indices]
    forcing = forcing[indices]
    observed = states.copy()
    if case.noise_fraction > 0.0:
        rng = np.random.default_rng(case.noise_seed)
        scale = np.maximum(np.ptp(states, axis=0), 1e-12)
        observed += rng.normal(size=states.shape) * scale * case.noise_fraction

    return ReferenceTrajectory(
        case_id=case.case_id,
        benchmark_id=case.benchmark_id,
        time=tuple(float(value) for value in time),
        forcing=tuple(tuple(float(value) for value in row) for row in forcing),
        states_clean=tuple(tuple(float(value) for value in row) for row in states),
        states_observed=tuple(tuple(float(value) for value in row) for row in observed),
    )


def _times(case: InterventionCase) -> np.ndarray:
    return np.arange(0.0, case.duration + 0.5 * case.dt, case.dt, dtype=float)


def _scaled_parameters(
    base: Mapping[str, Any], multipliers: Mapping[str, float]
) -> dict[str, float]:
    parameters = {name: float(value) for name, value in base.items()}
    unknown = set(multipliers).difference(parameters)
    if unknown:
        raise ValueError(f"unknown parameter multipliers: {sorted(unknown)}")
    for name, multiplier in multipliers.items():
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"parameter multiplier for {name} must be positive")
        parameters[name] *= multiplier
    return parameters


def _benchmark5_input(
    time: float, protocol: Mapping[str, Any], parameters: Mapping[str, float]
) -> np.ndarray:
    baseline = np.array(
        [
            parameters["C_feed_base"],
            parameters["T_feed_base"],
            parameters["T_secondary_feed_base"],
        ],
        dtype=float,
    )
    kind = protocol["kind"]
    if kind == "piecewise":
        delta = np.zeros(3, dtype=float)
        for segment in protocol["segments"]:
            if float(segment["start"]) <= time < float(segment["end"]):
                delta += np.asarray(segment["delta"], dtype=float)
        return baseline + delta
    if kind in {"sine", "chirp"}:
        start = float(protocol["start"])
        if time < start:
            return baseline
        tau = time - start
        if kind == "sine":
            phase = np.asarray(protocol["omega"], dtype=float) * tau
        else:
            phase = (
                np.asarray(protocol["omega0"], dtype=float) * tau
                + 0.5 * np.asarray(protocol["chirp_rate"], dtype=float) * tau**2
            )
        phase += np.asarray(protocol.get("phase", [0.0, 0.0, 0.0]), dtype=float)
        return baseline + np.asarray(protocol["amplitude"], dtype=float) * np.sin(phase)
    raise ValueError(f"unsupported benchmark5 protocol kind: {kind}")


def _simulate_benchmark5(
    case: InterventionCase, system_spec: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = _scaled_parameters(system_spec["parameters"], case.parameter_multipliers)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        concentration, temperature, jacket_temperature = state
        feed_concentration, feed_temperature, jacket_feed_temperature = (
            _benchmark5_input(time, case.protocol, p)
        )
        source = (
            p["k0"]
            * np.exp(-p["E_over_R"] / max(float(temperature), 250.0))
            * max(float(concentration), 0.0)
        )
        return np.array(
            [
                p["flow_rate"] * (feed_concentration - concentration) - source,
                p["flow_rate"] * (feed_temperature - temperature)
                + p["source_gain"] * source
                - p["exchange_rate"] * (temperature - jacket_temperature),
                p["secondary_flow_rate"]
                * (jacket_feed_temperature - jacket_temperature)
                + p["secondary_exchange_rate"] * (temperature - jacket_temperature),
            ],
            dtype=float,
        )

    time = _times(case)
    solution = solve_ivp(
        rhs,
        (0.0, case.duration),
        np.asarray(case.initial_state, dtype=float),
        method="LSODA",
        t_eval=time,
        rtol=1e-8,
        atol=1e-10,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(
            f"benchmark5 reference simulation failed: {solution.message}"
        )
    forcing = np.vstack([_benchmark5_input(float(t), case.protocol, p) for t in time])
    return time, solution.y.T, forcing


def _benchmark6_input(time: float, protocol: Mapping[str, Any]) -> float:
    kind = protocol["kind"]
    if kind == "pulses":
        return float(
            sum(
                float(pulse["amplitude"])
                for pulse in protocol["pulses"]
                if float(pulse["start"]) <= time < float(pulse["end"])
            )
        )
    if kind == "step":
        return (
            float(protocol["amplitude"])
            if float(protocol["start"]) <= time < float(protocol["end"])
            else 0.0
        )
    if kind in {"sine", "chirp"}:
        start = float(protocol["start"])
        if time < start:
            return 0.0
        tau = time - start
        if kind == "sine":
            phase = float(protocol["omega"]) * tau
        else:
            phase = (
                float(protocol["omega0"]) * tau
                + 0.5 * float(protocol["chirp_rate"]) * tau**2
            )
        return float(protocol["amplitude"]) * np.sin(
            phase + float(protocol.get("phase", 0.0))
        )
    raise ValueError(f"unsupported benchmark6 protocol kind: {kind}")


def _multiply_spec_parameters(
    system_spec: Mapping[str, Any], multipliers: Mapping[str, float]
) -> dict[str, Any]:
    spec = json.loads(json.dumps(system_spec))
    allowed = {"input_scale", "output_decay"}
    unknown = set(multipliers).difference(allowed)
    if unknown:
        raise ValueError(f"unknown benchmark6 parameter multipliers: {sorted(unknown)}")
    for name, multiplier in multipliers.items():
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"parameter multiplier for {name} must be positive")
        spec[name] = float(spec[name]) * multiplier
    return spec


def _simulate_benchmark6(
    case: InterventionCase, system_spec: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = _multiply_spec_parameters(system_spec, case.parameter_multipliers)
    number_latent = int(spec["n_latent"])
    decay = np.asarray(spec["decay"], dtype=float)
    skew = np.asarray(spec["skew"], dtype=float)
    input_vector = np.asarray(spec["input_vector"], dtype=float)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        latent = state[:number_latent]
        output = float(state[number_latent])
        derivative = -decay * latent + skew @ latent
        for target, terms in enumerate(spec["tanh_terms"]):
            for term in terms:
                source = int(term["source"])
                derivative[target] += float(term["coefficient"]) * np.tanh(
                    float(term["scale"]) * latent[source] + float(term["bias"])
                )
        for target, terms in enumerate(spec["product_terms"]):
            for term in terms:
                derivative[target] += (
                    float(term["coefficient"])
                    * np.tanh(float(term["scale_1"]) * latent[int(term["source_1"])])
                    * np.tanh(float(term["scale_2"]) * latent[int(term["source_2"])])
                )
        forcing = _benchmark6_input(time, case.protocol)
        derivative += input_vector * np.tanh(float(spec["input_scale"]) * forcing)
        output_derivative = -float(spec["output_decay"]) * output
        for term in spec["output_terms"]:
            output_derivative += float(term["coefficient"]) * np.tanh(
                float(term["scale"]) * latent[int(term["source"])]
            )
        for term in spec["output_product_terms"]:
            output_derivative += (
                float(term["coefficient"])
                * np.tanh(float(term["scale_1"]) * latent[int(term["source_1"])])
                * np.tanh(float(term["scale_2"]) * latent[int(term["source_2"])])
            )
        return np.concatenate([derivative, np.array([output_derivative])])

    time = _times(case)
    solution = solve_ivp(
        rhs,
        (0.0, case.duration),
        np.asarray(case.initial_state, dtype=float),
        method="LSODA",
        t_eval=time,
        rtol=1e-6,
        atol=1e-8,
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(
            f"benchmark6 reference simulation failed: {solution.message}"
        )
    forcing = np.array([[_benchmark6_input(float(t), case.protocol)] for t in time])
    return time, solution.y.T, forcing


def load_system_spec(data_root: Path, benchmark_id: str) -> tuple[Path, dict[str, Any]]:
    """Load the trusted private simulator specification for a supported benchmark."""

    relative = {
        "original_b1": Path("benchmark1_original_dalla_man/manifest.json"),
        "perturbed_b1": Path(
            "benchmark2_perturbed_dalla_man/B1_meal_appearance/manifest.json"
        ),
        "obfuscated_original_case01": Path(
            "benchmark3_obfuscated_dalla_man/private/case_01/secret_mapping.json"
        ),
        "obfuscated_perturbed_case01": Path(
            "benchmark4_obfuscated_perturbed_dalla_man/private/case_01/"
            "secret_mapping.json"
        ),
        "benchmark5": Path(
            "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
        ),
        "benchmark6": Path("benchmark6_alien_device/private/selected_system_spec.json"),
    }[benchmark_id]
    path = data_root / relative
    return path, json.loads(path.read_text(encoding="utf-8"))


def suite_benchmarks(cases: Sequence[InterventionCase]) -> tuple[str, ...]:
    """Return supported benchmark IDs in deterministic order."""

    return tuple(sorted({case.benchmark_id for case in cases}))
