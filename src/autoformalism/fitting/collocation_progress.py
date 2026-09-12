"""Bounded, atomic evidence from native collocation iterations.

Saved parameters are restart suggestions, never certified physical rollouts.
The native IPOPT state is not resumable from these records.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.rebuttal.fitter_diagnostic import _finite_payload, write_json


class CollocationProgress:
    """Preserve the latest, least-infeasible, and best feasible parameter points."""

    def __init__(self, opti, theta, names, lower, upper, directory: Path, started):
        self.opti, self.theta, self.names = opti, theta, names
        self.lower, self.upper = lower, upper
        self.path, self.started = directory / "progress.json", started
        self.history: list[dict] = []
        self.saved: dict[str, dict] = {}
        self.dual = (
            ca.gradient(opti.f, opti.x) + ca.jacobian(opti.g, opti.x).T @ opti.lam_g
        )
        self.metadata = {
            "decision_variables": int(opti.nx),
            "constraints": int(opti.ng),
            "construction_seconds": monotonic() - started,
            "physical_rollout_verified": False,
            "native_solver_state_resumable": False,
        }
        self.save("solver_called")

    def save(self, phase: str) -> None:
        write_json(
            self.path,
            _finite_payload(
                {
                    **self.metadata,
                    "phase": phase,
                    "seconds": monotonic() - self.started,
                    "iterations": self.history,
                    "checkpoints": self.saved,
                }
            ),
        )

    def record(self, iteration: int) -> None:
        """Observe an iterate without changing native termination or constraints."""
        row = {"iteration": int(iteration), "seconds": monotonic() - self.started}
        try:
            value = self.opti.debug.value
            p = np.asarray(value(self.theta)).reshape(-1)
            g = np.asarray(value(self.opti.g)).reshape(-1)
            lo = np.asarray(value(self.opti.lbg)).reshape(-1)
            hi = np.asarray(value(self.opti.ubg)).reshape(-1)
            violation = float(np.max(np.maximum(lo - g, g - hi), initial=0))
            objective = float(value(self.opti.f))
            finite = bool(
                np.isfinite(p).all() and np.isfinite(g).all() and np.isfinite(objective)
            )
            in_domain = bool(np.all(p >= self.lower) and np.all(p <= self.upper))
            row.update(
                objective=objective,
                constraint_maximum=violation,
                finite=finite,
                in_domain=in_domain,
                decision_maximum=float(np.max(np.abs(value(self.opti.x)))),
            )
            try:
                row["lagrangian_gradient_maximum"] = float(
                    np.max(np.abs(value(self.dual)), initial=0)
                )
            except (RuntimeError, ValueError):
                row["lagrangian_gradient_maximum"] = None
            if finite and in_domain:
                point = {
                    **row,
                    "parameters": dict(zip(self.names, p, strict=True)),
                    "physical_rollout_verified": False,
                }
                self.saved["latest"] = point
                least = self.saved.get("least_violation")
                if least is None or (violation, objective) < (
                    least["constraint_maximum"],
                    least["objective"],
                ):
                    self.saved["least_violation"] = point
                best = self.saved.get("best_feasible")
                if violation <= 1e-6 and (
                    best is None or objective < best["objective"]
                ):
                    self.saved["best_feasible"] = point
        except (RuntimeError, ValueError, ArithmeticError) as error:
            row["diagnostic_error"] = str(error)[-400:]
        self.history.append(row)
        self.save("iteration")
