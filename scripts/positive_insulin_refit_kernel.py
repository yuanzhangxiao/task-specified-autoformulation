"""Train-only least squares on the four frozen T2 intake graphs."""

from __future__ import annotations

from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.expressions.compiler import SAFE_DIVISION_EPSILON
from autoformalism.expressions.parser import RestrictedParser
from autoformalism.research.trajectory_function_fitting import _Rollout, _rows
from scripts.transfer_positive_insulin_law import TARGETS, production


class FitProblem:
    """Reuse restricted RK4 flows, knot-crossing meshes and their derivatives."""

    def __init__(self, model: dict, training_rows: list[dict], *, substeps: int = 4):
        if substeps < 1:
            raise ValueError("RK4 requires positive substeps")
        rows = _rows(training_rows, "train")
        self.engine = _Rollout(model["graph"], model["formula"], TARGETS, substeps)
        self.names = tuple(self.engine.parameters)
        self.targets = tuple(TARGETS)
        self.scales = np.array([model["scales"][n] for n in self.targets])
        if not np.isfinite(self.scales).all() or min(self.scales) <= 0:
            raise ValueError("Target scales must be finite and positive")
        self.substeps = substeps
        self.positive = "pos_tau" in self.names
        if self.positive != bool(model.get("knots")):
            raise ValueError("Positive-production parameters require fixed basis knots")
        parsed = RestrictedParser().parse(
            model["formula"]["equations"]["I"], location="insulin law"
        )
        self.insulin_parameters = frozenset(parsed.symbols.intersection(self.names))
        self.records = []
        self.jacobians = {}
        for raw, row in zip(training_rows, rows, strict=True):
            if set(row.targets) != set(self.targets):
                raise ValueError("Expected exactly the three training targets")
            t = np.asarray(row.time)
            if not np.all(np.diff(t) > 0):
                raise ValueError("Time knots must increase")
            grid = production.grid(raw, model["knots"])[1] if self.positive else t
            _, (z, u, _) = self.engine.record(row)
            forcing = np.array([np.interp(grid, t, values) for values in u])
            samples = np.searchsorted(grid, t)
            data = np.array([row.targets[n] for n in self.targets]).T
            args = (z, forcing, np.diff(grid).reshape(1, -1))
            self.records.append((len(grid), args, samples, data))

    def _vector(self, parameters: dict) -> np.ndarray:
        if set(parameters) != set(self.names):
            raise ValueError("Fitting requires the full preserved parameter inventory")
        p = np.array([parameters[n] for n in self.names], dtype=float)
        if not np.isfinite(p).all():
            raise ValueError("Starting parameters must be finite")
        if self.positive and (
            parameters["pos_tau"] < 2 * SAFE_DIVISION_EPSILON
            or any(v < 0 for k, v in parameters.items() if k.startswith("pos_"))
        ):
            raise ValueError("Positive insulin law requires admissible gains and tau")
        return p

    def residual(self, p: np.ndarray) -> np.ndarray:
        """Return scaled prediction errors in event, sample, target order."""
        blocks = []
        for size, args, samples, observed in self.records:
            y, x = self.engine._flow(size)(p, *args)
            state = np.asarray(x)
            error = (np.asarray(y)[:, samples].T - observed) / self.scales
            if (
                not np.isfinite(error).all()
                or not np.isfinite(state).all()
                or np.max(abs(state)) > 1e18
                or np.max(abs(np.asarray(y))) > 1e18
            ):
                raise ArithmeticError("Unresolved free rollout")
            blocks.append(error.ravel())
        return np.concatenate(blocks)

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        """Differentiate the discrete free rollout, including initializer maps."""
        ca = self.engine.ca
        blocks = []
        for size, args, samples, _ in self.records:
            if size not in self.jacobians:
                flow = self.engine._flow(size)
                symbols = [flow.mx_in(i) for i in range(flow.n_in())]
                y = flow(*symbols)[0]
                j = ca.jacobian(ca.vec(y), symbols[0])
                self.jacobians[size] = ca.Function("prediction_jac", symbols, [j])
            j = np.asarray(self.jacobians[size](p, *args))
            m = len(self.targets)
            scaled = j.reshape(size, m, -1)[samples] / self.scales[None, :, None]
            blocks.append(scaled.reshape(-1, len(self.names)))
        result = np.vstack(blocks)
        if not np.isfinite(result).all():
            raise ArithmeticError("Nonfinite trajectory sensitivities")
        return result

    def score(self, parameters: dict) -> dict:
        """Pool normalized SSE across samples, then equally across targets."""
        resid = self.residual(self._vector(parameters)).reshape(-1, len(self.targets))
        target = np.mean(resid**2, axis=0)
        return {
            "pooled_nmse": float(np.mean(target)),
            "per_target_nmse": dict(zip(self.targets, target.tolist(), strict=True)),
        }

    def fit_stage(
        self, stage: str, starting_parameters: dict, *, max_nfev: int = 80
    ) -> dict:
        """Keep the lowest training loss reached within the matched search budget."""
        if stage not in {"fixed_insulin", "joint"} or max_nfev < 1:
            raise ValueError("Expected a fitting stage and positive search budget")
        started = monotonic()
        base = self._vector(starting_parameters)
        active = np.array(
            [
                i
                for i, n in enumerate(self.names)
                if stage == "joint" or n not in self.insulin_parameters
            ],
            dtype=int,
        )
        names = [self.names[i] for i in active]
        lo = np.array(
            [
                2 * SAFE_DIVISION_EPSILON
                if n == "pos_tau"
                else 0
                if self.positive and n.startswith("pos_")
                else -np.inf
                for n in names
            ]
        )
        initial = self.residual(base)
        best, best_loss = base.copy(), float(np.mean(initial**2))
        start_loss = best_loss
        invalid = 0

        def expand(v):
            p = base.copy()
            p[active] = v
            return p

        def residual(v):
            nonlocal best, best_loss, invalid
            p = expand(v)
            try:
                r = initial if np.array_equal(p, base) else self.residual(p)
            except ArithmeticError:
                invalid += 1
                return np.full_like(initial, 1e30)
            loss = float(np.mean(r**2))
            if loss < best_loss:
                best, best_loss = p, loss
            return r

        def jac(v):
            return self.jacobian(expand(v))[:, active]

        fit = least_squares(
            residual,
            base[active],
            jac=jac,
            bounds=(lo, np.inf),
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-9,
        )
        selected_terminal = np.array_equal(best[active], fit.x)
        # Include endpoint optimality only if it pertains to the retained vector.
        return {
            "parameters": dict(zip(self.names, best.tolist(), strict=True)),
            "training_nmse": best_loss,
            "diagnostics": {
                "stage": stage,
                "active_parameters": names,
                "frozen_parameters": [n for n in self.names if n not in names],
                "start_nmse": start_loss,
                "budget": max_nfev,
                "nfev": fit.nfev,
                "njev": fit.njev,
                "elapsed_seconds": monotonic() - started,
                "invalid_trials": invalid,
                "solver_status": fit.status,
                "termination": fit.message,
                "solver_success": bool(fit.success),
                "retained_solver_endpoint": bool(selected_terminal),
                "scipy_optimality": float(fit.optimality)
                if selected_terminal
                else None,
                "substeps": self.substeps,
            },
        }
