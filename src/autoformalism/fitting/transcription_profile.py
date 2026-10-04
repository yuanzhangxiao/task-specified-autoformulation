"""Persist bounded native derivative diagnostics before potentially long calls."""

from pathlib import Path
from time import monotonic

import casadi as ca
import numpy as np

from autoformalism.fitting import public_fitting as public


def profile(problem, directory: Path, *, hessian: str) -> dict:
    """Measure one initial-point evaluation; partial phase survives process kills.

    This is diagnostic overhead charged to the native budget, not an estimate of
    all optimizer work. Limited-memory solves do not build an exact Hessian.
    """
    opti = problem.opti
    x0 = opti.debug.value(opti.x, opti.initial())
    record = {"scope": "one_initial_point_including_graph_generation", "phases": {}}
    for name in ("objective", "gradient", "jacobian", "hessian"):
        if name == "hessian" and hessian == "limited-memory":
            record["phases"][name] = {"status": "not_requested"}
            public._write(directory / "derivative_profile.json", record)
            continue
        if monotonic() >= problem.deadline:
            break
        begun = monotonic()
        phase = {"status": "building"}
        record["phases"][name] = phase
        public._write(directory / "derivative_profile.json", record)
        try:
            lam = ca.MX.sym("multipliers", opti.ng)
            arguments, values = [opti.x], [x0]
            if name == "objective":
                expression = problem.objective
            elif name == "gradient":
                expression = ca.gradient(problem.objective, opti.x)
            elif name == "jacobian":
                expression = ca.jacobian(opti.g, opti.x)
            else:
                expression = ca.hessian(
                    problem.objective + ca.dot(lam, opti.g), opti.x
                )[0]
                arguments.append(lam)
                values.append(np.ones(opti.ng))
            function = ca.Function(f"profile_{name}", arguments, [expression])
            phase.update(
                build_seconds=monotonic() - begun,
                structural_nonzeros=int(expression.nnz()),
                status="evaluating",
            )
            public._write(directory / "derivative_profile.json", record)
            evaluated = monotonic()
            result = function(*values)
            phase.update(
                evaluation_seconds=monotonic() - evaluated,
                status="complete"
                if np.isfinite(result.nonzeros()).all()
                else "nonfinite",
            )
        except (RuntimeError, ValueError) as error:
            phase.update(status="failed", message=str(error)[-1000:])
        phase["seconds"] = monotonic() - begun
        public._write(directory / "derivative_profile.json", record)
    return record
