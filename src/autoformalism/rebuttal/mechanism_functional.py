"""Frozen-model local response tests, with explicit finite-test semantics.

No names/tags are used to identify latent mechanisms. Generated driver outputs
must map to an explicit state. For a balance-rate readout the output storage is
held fixed, preventing ordinary output integration from masquerading as delay.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from autoformalism.baselines.d3_rollout import predict
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import FitConfig, simulate_trajectory
from autoformalism.rebuttal.basin_algebra import expanded
from autoformalism.rebuttal.mechanism_checks import effective_trees
from autoformalism.rebuttal.mechanism_probes import _native
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.base import StrictSchema
from autoformalism.search.requirement_feedback import _ancestors
from autoformalism.staged_topology import content_hash


class Settings(StrictSchema):
    """Fixed resource and detection settings, identical for every method."""

    @field_validator("operating_fractions")
    @classmethod
    def valid_fractions(cls, value):
        if not value or len(value) > 10 or any(not 0 <= x <= 1 for x in value):
            raise ValueError("require one to ten fractions in [0,1]")
        return value

    maximum_training_trajectories: int = Field(default=3, ge=1, le=10)
    operating_fractions: tuple[float, ...] = (0.0, 0.5, 1.0)
    trajectory_seconds: float = Field(default=120, gt=0, allow_inf_nan=False)
    rtol: float = Field(default=1e-7, gt=0, allow_inf_nan=False)
    atol: float = Field(default=1e-9, gt=0, allow_inf_nan=False)
    state_agreement_tolerance: float = Field(default=1e-4, gt=0, allow_inf_nan=False)
    difference_step: float = Field(default=2e-5, gt=0, allow_inf_nan=False)
    difference_agreement_rtol: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    detection_tolerance: float = Field(default=1e-8, gt=0, allow_inf_nan=False)


class Rule(StrictSchema):
    """Public obligation and explicit operational interpretation, never model tags."""

    id: str
    public_requirement: str
    kind: Literal["causal_response", "memory", "delayed_action", "thermal_balance"]
    target: str
    driver: str | None = None
    expected_sign: Literal[-1, 0, 1] = 0
    readout: Literal["output", "balance_rate"] = "output"
    feed: str | None = None
    jacket: str | None = None
    reactant: str | None = None
    interpretation: str
    sign_basis: str | None = None
    identification_limit: str | None = None

    @model_validator(mode="after")
    def require_channels(self):
        needed = (
            (self.feed, self.jacket, self.reactant)
            if self.kind == "thermal_balance"
            else (self.driver,)
        )
        if not all(needed):
            raise ValueError("rule is missing required public channels")
        return self


@dataclass
class ConstantForcing:
    values: dict

    def value(self, channel: str, time: float) -> float:
        return float(self.values[channel])


class Runtime:
    """Use production ODE expressions or the native D3 increment interpreter."""

    def __init__(self, row: dict):
        self.row = row
        if row["semantics"] not in {"native_increment", "continuous_time"}:
            raise ValueError("unsupported execution semantics")
        self.context = ValidationContext.model_validate(row["context"])
        self.candidate = CandidateModel.model_validate(row["candidate"])
        self.native = _native(row) if row["semantics"] == "native_increment" else None
        self.compiled = (
            None if self.native else compile_candidate(self.candidate, self.context)
        )
        self.names = self.native.states if self.native else self.compiled.state_names
        self.dynamic = tuple(
            i
            for i, n in enumerate(self.names)
            if not self.native or n not in self.context.auxiliaries
        )
        self.parameters = row["parameters"]
        if len(self.names) > 128:
            raise ValueError("local Jacobian resource budget exceeds 128 states")

    def coordinate(self, channel: str) -> int:
        """Require a known public coordinate, not a latent-name interpretation."""
        if self.native:
            return self.names.index(channel)
        mapping = next(
            o.expression
            for o in self.candidate.observation_mappings
            if o.channel == channel
        )
        tree = expanded(
            mapping, {p.name: p.expression for p in self.candidate.processes}
        )
        if isinstance(tree, ast.Name) and tree.id in self.names:
            return self.names.index(tree.id)
        raise ValueError(f"nonidentity public coordinate {channel} is unresolved")

    def rhs(self, x, forcing, time):
        if self.native:
            env = {**dict(zip(self.names, x, strict=True)), **forcing, "t": time}
            return self.native.increment(env)
        return self.compiled.rhs(time, x, self.parameters, ConstantForcing(forcing))

    def output(self, x, forcing, time, target):
        if self.native:
            return float(x[self.coordinate(target)])
        return float(
            self.compiled.observe(time, x, self.parameters, ConstantForcing(forcing))[
                target
            ]
        )

    def trajectory(self, trajectory, solver, settings):
        if self.native:
            values = predict(
                self.native,
                trajectory,
                teacher_forced=False,
                seconds=settings.trajectory_seconds,
            ).T.copy()
            for name in self.context.auxiliaries:
                values[self.names.index(name)] = trajectory.auxiliaries[name]
            return values
        result = simulate_trajectory(
            self.compiled,
            trajectory,
            self.parameters,
            self.row["initials"],
            FitConfig(
                integration_backend="solve_ivp",
                integration_method=solver,
                relative_tolerance=settings.rtol,
                absolute_tolerance=settings.atol,
                maximum_wall_time_seconds=settings.trajectory_seconds,
                allow_derivative_regression=False,
            ),
            deadline=monotonic() + settings.trajectory_seconds,
            reset_observed_states=False,
        )
        if not result.success:
            raise ValueError(result.message or "baseline rollout failed")
        return result.states


def baseline(runtime, trajectory, solver, settings, directory, data_identity):
    """Checkpoint each physical rollout; interruption never resets its budget."""
    key = content_hash(
        [
            runtime.row,
            data_identity,
            trajectory.trajectory_id,
            solver,
            settings.model_dump(mode="json"),
        ]
    )
    path = directory / f"{key}.json"
    if path.exists():
        saved = sealed_read(path)
        if saved["identity"] != key:
            raise ValueError("rollout cache identity differs")
        return saved
    start = path.with_suffix(".started.json")
    if start.exists():
        if sealed_read(start)["identity"] != key:
            raise ValueError("rollout start identity differs")
        return sealed_write(
            path,
            {
                "identity": key,
                "status": "unresolved",
                "reason": "interrupted rollout; budget retained",
            },
        )
    sealed_write(start, {"identity": key})
    begin = monotonic()
    try:
        states = runtime.trajectory(trajectory, solver, settings)
        if states.shape != (len(runtime.names), len(trajectory.time)):
            raise ValueError("incomplete or misaligned state rollout")
        if not np.isfinite(states).all():
            raise ValueError("nonfinite saved states")
        result = {"status": "complete", "states": states.tolist()}
    except Exception as exc:
        result = {"status": "unresolved", "reason": f"{type(exc).__name__}: {exc}"}
    return sealed_write(
        path, {"identity": key, "seconds": monotonic() - begin, **result}
    )


def local_response(runtime, rule, x, forcing, time, scales, time_unit, settings):
    """Probe a local transfer after holding specified public coordinates fixed.

    Taylor/Markov coefficients C A^k B distinguish effective dynamic coupling
    from direct feedthrough and cancel parallel paths. These are local finite
    numerical tests, not an assertion of global monotonicity or identifiability.
    """
    driver, target = rule["driver"], rule["target"]
    coordinate = (
        runtime.coordinate(driver) if driver in runtime.context.targets else None
    )
    balance = rule.get("readout") == "balance_rate"
    target_index = runtime.coordinate(target) if balance else None
    held = {i for i in (coordinate, target_index) if i is not None}
    free = [i for i in runtime.dynamic if i not in held]
    dynamic_required = rule["kind"] in {"memory", "delayed_action"}
    if dynamic_required:
        trees = effective_trees(runtime.candidate, runtime.parameters)
        graph = {
            n: {v.id for v in ast.walk(t) if isinstance(v, ast.Name)}
            for n, t in trees.items()
        }
        readout = f"output:{target}"
        if balance:
            graph["probe:balance"] = set(graph[runtime.names[target_index]])
            readout = "probe:balance"
        for i in held:
            graph.pop(runtime.names[i], None)
        reachable = _ancestors(readout, graph)
        origin = runtime.names[coordinate] if coordinate is not None else driver
        if not any(
            runtime.names[i] in reachable
            and origin in _ancestors(runtime.names[i], graph)
            for i in free
        ):
            return {
                "status": "fail",
                "reason": "no effective dynamic mediation after clamping",
                "held_states": [runtime.names[i] for i in sorted(held)],
                "generated_driver_uses_model_state": coordinate is not None,
            }
    u = x[coordinate] if coordinate is not None else forcing[driver]
    zscale = np.maximum(np.abs(x[free]), 1.0)
    uscale, yscale = scales[driver], scales[target]
    dt = 1.0 if runtime.native else time_unit

    def evaluate(z, du):
        state, inputs = x.copy(), dict(forcing)
        state[free] += z * zscale
        if coordinate is None:
            inputs[driver] = u + du * uscale
        else:
            state[coordinate] = u + du * uscale
        f = runtime.rhs(state, inputs, time)
        q = (
            f[target_index] * dt
            if balance
            else runtime.output(state, inputs, time, target)
        )
        return np.r_[f[free] * dt / zscale, q / yscale]

    def jacobian(h):
        zero = np.zeros(len(free))
        columns = []
        for j in range(len(free)):
            delta = zero.copy()
            delta[j] = h
            columns.append((evaluate(delta, 0) - evaluate(-delta, 0)) / (2 * h))
        # A one-sided derivative avoids negative meal/insulin near zero.
        direction = (
            (evaluate(zero, h) - evaluate(zero, 0)) / h
            if u - h * uscale < 0
            else (evaluate(zero, h) - evaluate(zero, -h)) / (2 * h)
        )
        return np.column_stack([*columns, direction])

    coarse, fine = (
        jacobian(settings.difference_step),
        jacobian(settings.difference_step / 2),
    )
    if not np.isfinite(fine).all() or not np.isfinite(coarse).all():
        raise ValueError("nonfinite local response derivatives")
    error = float(np.max(np.abs(fine - coarse)))
    if error > settings.difference_agreement_rtol * max(
        1.0, float(np.max(np.abs(fine)))
    ):
        return {
            "status": "unresolved",
            "reason": "finite difference scales disagree",
            "derivative_disagreement": error,
        }
    carry = np.eye(len(free)) if runtime.native else np.zeros((len(free), len(free)))
    norm = (
        max(
            1.0,
            *(
                float(np.linalg.norm(j[:-1, :-1] + carry, ord=np.inf))
                for j in (fine, coarse)
            ),
        )
        if free
        else 1.0
    )

    def markov(jac):
        matrix = (jac[:-1, :-1] + carry) / norm
        vector, values = jac[:-1, -1], []
        for _ in free:
            values.append(float(jac[-1, :-1] @ vector))
            vector = matrix @ vector
        return np.array(values)

    fine_terms, coarse_terms = markov(fine), markov(coarse)
    if not np.isfinite(fine_terms).all() or not np.isfinite(coarse_terms).all():
        raise ValueError("nonfinite Markov response")
    coefficients = fine_terms.tolist()
    d = float(fine[-1, -1])
    gain_error = abs(d - float(coarse[-1, -1]))
    markov_error = float(np.max(np.abs(fine_terms - coarse_terms))) if free else 0.0
    tol = max(settings.detection_tolerance, 10 * markov_error)
    active = [(k, v) for k, v in enumerate(coefficients) if abs(v) > tol]
    effect = active[0][1] if active else None
    if not dynamic_required and abs(d) > max(
        settings.detection_tolerance, 10 * gain_error
    ):
        effect = d
    expected = rule.get("expected_sign", 0)
    status = (
        "inactive"
        if effect is None
        else ("fail" if expected and expected * effect < 0 else "pass")
    )
    return {
        "status": status,
        "direct_gain": d,
        "dynamic_markov_coefficients": coefficients,
        "first_dynamic_order": active[0][0] if active else None,
        "first_effect": effect,
        "expected_sign": expected,
        "detection_tolerance": tol,
        "derivative_disagreement": error,
        "markov_disagreement": markov_error,
        "direct_gain_disagreement": gain_error,
        "dynamic_matrix_positive_scaling": norm,
        "held_states": [runtime.names[i] for i in sorted(held)],
        "generated_driver_uses_model_state": coordinate is not None,
        "reason": "opposite local response direction" if status == "fail" else None,
    }


def thermal_balance(runtime, rule, x, forcing, time, scales, time_unit, settings):
    """Finite heat-flow/boundary tests in an explicit public temperature coordinate."""
    target = rule["target"]
    index = runtime.coordinate(target)
    # An unlabelled extra energy/concentration coordinate needs semantic mapping.
    tree = effective_trees(runtime.candidate, runtime.parameters)
    dependencies = {
        n: {v.id for v in ast.walk(t) if isinstance(v, ast.Name)}
        for n, t in tree.items()
    }
    ancestors = _ancestors(runtime.names[index], dependencies)
    others = {runtime.names[i] for i in runtime.dynamic if i != index}
    if ancestors & others:
        return {
            "status": "unresolved",
            "reason": "additional internal thermal coordinates",
            "coordinates": sorted(ancestors & others),
        }
    feed, jacket, reactant = rule["feed"], rule["jacket"], rule["reactant"]
    if not {feed, jacket, reactant} <= ancestors:
        return {
            "status": "unresolved",
            "reason": "alternative/missing public thermal channels",
        }
    dt = 1.0 if runtime.native else time_unit
    temperature = float(x[index])
    c_value = float(forcing.get(reactant, 0.0))
    if c_value <= 0:
        return {
            "status": "inactive",
            "reason": "no positive supplied reactant at this point",
        }

    def value(tf, tj, concentration):
        values = {**forcing, feed: tf, jacket: tj, reactant: concentration}
        state = x.copy()
        for n, v in values.items():
            if runtime.native and n in runtime.names:
                state[runtime.names.index(n)] = v
        result = float(runtime.rhs(state, values, time)[index]) * dt / scales[target]
        if not np.isfinite(result):
            raise ValueError("nonfinite thermal probe; cannot establish a verdict")
        return result

    h = settings.difference_step * max(abs(temperature), 1.0)
    cold = value(temperature, temperature, 0.0)
    reaction = value(temperature, temperature, c_value)
    feed_gain = (value(temperature + h, temperature, 0.0) - cold) / h
    jacket_gain = (value(temperature, temperature + h, 0.0) - cold) / h
    half_feed = (value(temperature + h / 2, temperature, 0.0) - cold) / (h / 2)
    half_jacket = (value(temperature, temperature + h / 2, 0.0) - cold) / (h / 2)
    error = max(abs(feed_gain - half_feed), abs(jacket_gain - half_jacket))
    if error > settings.difference_agreement_rtol * max(
        1.0, abs(feed_gain), abs(jacket_gain)
    ):
        return {"status": "unresolved", "reason": "thermal derivative scales disagree"}
    tol = max(settings.detection_tolerance, 10 * error)
    # Conditional additivity distinguishes the three public heat channels.
    # Mixed finite increments permit algebraically factored but additive laws.
    point = np.array([temperature, temperature, c_value])
    steps = 0.05 * np.array([scales.get(feed, 1.0), scales.get(jacket, 1.0), c_value])
    base = value(*point)
    mixed, mixed_limits = [], []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        a, b = point.copy(), point.copy()
        a[first] += steps[first]
        b[second] += steps[second]
        both = a.copy()
        both[second] += steps[second]
        va, vb, vab = value(*a), value(*b), value(*both)
        mixed.append(vab - va - vb + base)
        mixed_limits.append(
            max(
                settings.detection_tolerance,
                1e-6 * max(1.0, abs(base), abs(va), abs(vb), abs(vab)),
            )
        )
    checks = {
        "additive_public_heat_channels": "pass"
        if all(abs(v) <= e for v, e in zip(mixed, mixed_limits, strict=True))
        else "fail",
        "equal_temperature_zero_reactant_balance": "pass"
        if abs(cold) <= tol
        else "fail",
        "feed_heats_in_its_direction": "pass"
        if half_feed > tol
        else ("fail" if half_feed < -tol else "unresolved"),
        "jacket_heats_in_its_direction": "pass"
        if half_jacket > tol
        else ("fail" if half_jacket < -tol else "unresolved"),
        "reaction_heat_generation": "pass"
        if reaction > tol
        else ("fail" if reaction < -tol else "unresolved"),
    }
    return {
        "status": conjunctive(list(checks.values())),
        "checks": checks,
        "mixed_channel_increments": mixed,
        "mixed_channel_tolerances": mixed_limits,
        "zero_boundary_residual": cold,
        "reaction_heat": reaction,
        "feed_gain": half_feed,
        "jacket_gain": half_jacket,
        "tolerance": tol,
        "temperature": temperature,
        "reactant": c_value,
        "scope": "conditional_public_channel_thermal_balance",
    }


def conjunctive(statuses: list[str]) -> str:
    """One counterexample fails; missing evidence prevents a confirmed pass."""
    if "fail" in statuses:
        return "fail"
    if not statuses or "unresolved" in statuses or "pass" not in statuses:
        return "unresolved"
    return "pass"


def structural_check(row, rule):
    """Absent effective dependencies are definitive; presence alone never passes."""
    if rule["kind"] == "thermal_balance":
        return None
    trees = effective_trees(
        CandidateModel.model_validate(row["candidate"]), row["parameters"]
    )
    deps = {
        n: {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        for n, t in trees.items()
    }
    driver = rule["driver"]
    context = ValidationContext.model_validate(row["context"])
    if driver in context.targets:
        candidate = CandidateModel.model_validate(row["candidate"])
        mapping = next(
            o.expression for o in candidate.observation_mappings if o.channel == driver
        )
        tree = expanded(mapping, {p.name: p.expression for p in candidate.processes})
        if not isinstance(tree, ast.Name) or tree.id not in {
            s.name for s in candidate.states
        }:
            return {
                "status": "unresolved",
                "reason": "generated driver has no explicit public state coordinate",
            }
        driver = tree.id
    if driver not in _ancestors(f"output:{rule['target']}", deps):
        return {"status": "fail", "reason": "no effective fitted driver-to-target path"}
    return None


def assess(row, rules, train, settings, directory):
    """Evaluate fixed training operating points without held-out outputs or refits."""
    directory.mkdir(parents=True, exist_ok=True)
    findings = [
        {
            "id": r["id"],
            "public_requirement": r["public_requirement"],
            "points": [],
            "structural": structural_check(row, r),
        }
        for r in rules
    ]
    if all(f["structural"] for f in findings):
        return [{**f, **f["structural"]} for f in findings]
    runtime = Runtime(row)
    names = {
        *runtime.context.targets,
        *runtime.context.external_inputs,
        *runtime.context.auxiliaries,
    }
    scales = {}
    for name in names:
        values = np.concatenate(
            [
                {**t.targets, **t.external_inputs, **t.auxiliaries}[name]
                for t in train.trajectories
            ]
        )
        scales[name] = max(float(np.std(values)), 1e-6)
    selected = sorted(train.trajectories, key=lambda t: t.trajectory_id)[
        : settings.maximum_training_trajectories
    ]
    for trajectory in selected:
        solvers = ["native"] if runtime.native else ["Radau", "BDF"]
        runs = [
            baseline(
                runtime, trajectory, solver, settings, directory, train.fingerprint
            )
            for solver in solvers
        ]
        if any(r["status"] != "complete" for r in runs):
            for f in findings:
                f["points"].append(
                    {
                        "trajectory": trajectory.trajectory_id,
                        "status": "unresolved",
                        "reason": "baseline rollout unavailable",
                    }
                )
            continue
        states = np.asarray(runs[0]["states"])
        other = np.asarray(runs[-1]["states"])
        state_scales = np.maximum(1.0, np.max(np.abs(states), axis=1))
        disagreement = float(np.max(np.abs(states - other) / state_scales[:, None]))
        if disagreement > settings.state_agreement_tolerance:
            for f in findings:
                f["points"].append(
                    {
                        "trajectory": trajectory.trajectory_id,
                        "status": "unresolved",
                        "reason": "state replay disagreement",
                    }
                )
            continue
        points = sorted(
            {
                round(v * (len(trajectory.time) - 1))
                for v in settings.operating_fractions
            }
        )
        dt = float(np.median(np.diff(trajectory.time)))
        for k in points:
            forcing = {
                **trajectory.fixed_covariates,
                **{
                    n: float(v[k])
                    for n, v in {
                        **trajectory.external_inputs,
                        **trajectory.auxiliaries,
                    }.items()
                },
            }
            for rule, finding in zip(rules, findings, strict=True):
                if finding["structural"]:
                    continue
                try:
                    function = (
                        thermal_balance
                        if rule["kind"] == "thermal_balance"
                        else local_response
                    )
                    value = function(
                        runtime,
                        rule,
                        states[:, k],
                        forcing,
                        float(trajectory.time[k]),
                        scales,
                        dt,
                        settings,
                    )
                    if not runtime.native:
                        secondary = function(
                            runtime,
                            rule,
                            other[:, k],
                            forcing,
                            float(trajectory.time[k]),
                            scales,
                            dt,
                            settings,
                        )
                        value["secondary_solver_status"] = secondary["status"]
                        if value["status"] != secondary["status"]:
                            value.update(
                                status="unresolved",
                                reason="solver-point tests disagree",
                            )
                except Exception as exc:
                    value = {
                        "status": "unresolved",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                finding["points"].append(
                    {
                        "trajectory": trajectory.trajectory_id,
                        "sample_index": k,
                        "time": float(trajectory.time[k]),
                        "state_replay_disagreement": disagreement,
                        **value,
                    }
                )
    for rule, finding in zip(rules, findings, strict=True):
        finding["status"] = (
            finding["structural"]["status"]
            if finding["structural"]
            else conjunctive([p["status"] for p in finding["points"]])
        )
        if rule.get("identification_limit"):
            finding["proxy_test_status"] = finding["status"]
            if finding["status"] == "pass":
                finding["status"] = "unresolved"
            finding["identification_limit"] = rule["identification_limit"]
    return findings
