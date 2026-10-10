"""Training-only directional checks of the exact profiled residual derivative."""

from time import monotonic

import numpy as np
from pydantic import Field, model_validator

from autoformalism.schemas.base import StrictSchema


class AuditPolicy(StrictSchema):
    """Numerical agreement tolerances, not scientific confidence thresholds."""

    steps: tuple[float, ...] = (1e-3, 1e-4, 1e-5)
    relative_tolerance: float = Field(default=5e-4, gt=0, le=0.01)
    absolute_tolerance: float = Field(default=2e-7, gt=0, le=1e-4)
    seconds: float = Field(default=1800, ge=1, le=2400)

    @model_validator(mode="after")
    def ordered(self):
        if (
            len(self.steps) != 3
            or list(self.steps) != sorted(set(self.steps), reverse=True)
            or any(not np.isfinite(h) or not 0 < h <= 0.01 for h in self.steps)
        ):
            raise ValueError("require three decreasing positive finite audit steps")
        return self


def directions(jacobian: np.ndarray, residual: np.ndarray) -> list[np.ndarray]:
    """Largest-gradient coordinate, gradient and two fixed-seed random directions."""
    n = jacobian.shape[1]
    gradient = jacobian.T @ residual
    axis = np.eye(n)[int(np.argmax(np.abs(gradient)))]
    gradient = gradient / np.linalg.norm(gradient) if np.linalg.norm(gradient) else axis
    rng = np.random.default_rng(2501)
    vectors = [axis, gradient, rng.normal(size=n), rng.normal(size=n)]
    return [d / np.linalg.norm(d) for d in vectors]


def audit(oracle, profile, parameters: dict, policy: AuditPolicy, deadline, save):
    """Check four directions at three scales; ambiguous active sets never pass."""
    if profile is None:
        raise ValueError("M25 requires a certified exact profile")
    vector = np.asarray(oracle.vector(parameters), dtype=float)
    outer = np.asarray(profile.outer)
    units = oracle.units[outer]
    count, completed, rows = 0, 0, []

    def evaluate(v):
        nonlocal count, completed
        if monotonic() >= deadline:
            raise TimeoutError("derivative audit allowance exhausted")
        count += 1
        save({"calls": count, "completed_calls": completed, "directions": rows})
        oracle.vector(oracle.parameters(v))
        _, r, j, info = profile.evaluate(v, deadline)
        if not np.isfinite(r).all() or not np.isfinite(j).all():
            raise ValueError("nonfinite derivative audit")
        completed += 1
        return r, j, info["active_mask"]

    r, j, active = evaluate(vector)
    scaled = j * units
    for i, d in enumerate(directions(scaled, r)):
        expected = scaled @ d
        evidence = []
        for h in policy.steps:
            plus, minus = vector.copy(), vector.copy()
            plus[outer] += h * units * d
            minus[outer] -= h * units * d
            if any(
                np.any(v < oracle.lower) or np.any(v > oracle.upper)
                for v in (plus, minus)
            ):
                evidence.append({"step": h, "status": "outer_bound_limited"})
                continue
            a, _, ap = evaluate(plus)
            b, _, am = evaluate(minus)
            fd = (a - b) / (2 * h)
            def rms(x):
                return float(np.sqrt(np.mean(x * x)))
            error = rms(fd - expected)
            scale = max(rms(fd), rms(expected))
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * scale
            stable = ap == active == am
            evidence.append(
                {
                    "step": h,
                    "status": "same_active_set" if stable else "active_set_transition",
                    "base_active_mask": active,
                    "plus_active_mask": ap,
                    "minus_active_mask": am,
                    "derivative_rms": rms(expected),
                    "finite_difference_rms": rms(fd),
                    "error_rms": error,
                    "tolerance": tolerance,
                    "agrees": error <= tolerance,
                }
            )
        stable = [e for e in evidence if e["status"] == "same_active_set"]
        # Two agreeing scales guard against a single accidental FD match. A stable
        # discrepancy at both finest available scales blocks optimizer comparison.
        finest = stable[-2:]
        status = (
            "passed"
            if len(finest) == 2 and all(e["agrees"] for e in finest)
            else "failed"
            if len(finest) == 2 and all(not e["agrees"] for e in finest)
            else "inconclusive"
        )
        rows.append(
            {"index": i, "direction": d.tolist(), "status": status, "steps": evidence}
        )
        save({"calls": count, "completed_calls": completed, "directions": rows})
    statuses = [row["status"] for row in rows]
    return {
        "status": "passed"
        if all(s == "passed" for s in statuses)
        else "failed"
        if "failed" in statuses
        else "inconclusive",
        "calls": count,
        "completed_calls": completed,
        "outer_names": [oracle.names[i] for i in outer],
        "coordinate_units": units.tolist(),
        "base_training_nmse": float(np.mean(r * r)),
        "directions": rows,
        "scope": "Four local profiled-residual directions; no global certificate.",
    }
