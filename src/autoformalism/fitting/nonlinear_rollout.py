"""Strongest symbolically applicable rollout kernel for the M22 comparison.

Both arms share this implementation. Terminal output profiling remains enabled
inside nonlinear systems; absence of full-system affinity does not disable it.
"""

from time import monotonic

import numpy as np

from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import output_activity
from autoformalism.fitting import recovery_numerics as numerical
from autoformalism.fitting.affine_profile import AffineProfile
from autoformalism.fitting.affine_propagation import AffineRollouts
from autoformalism.fitting.profiled_output import ProfiledOutput, project


class TerminalProfile:
    """Adapt exact terminal-output projection to the common recovery interface."""

    def __init__(self, oracle, *, active_parameters=None):
        self.oracle = oracle
        self.kernel = ProfiledOutput(oracle.system, active_parameters=active_parameters)
        self.outer = tuple(self.kernel.outer_indices)
        self.audit = self.kernel.audit

    def evaluate(self, vector, deadline):
        o, k = self.oracle, self.kernel
        designs, offsets, derivatives, offset_derivatives = [], [], [], []
        for row in o.train.trajectories:
            a, b, da, db, _ = k.trajectory(row, vector, o.settings, deadline)
            a, b, da, db = o.profile_transform(
                row,
                a,
                b - row.targets[o.system.channels[0]],
                da,
                db,
            )
            designs.append(a)
            offsets.append(b)
            derivatives.append(da)
            offset_derivatives.append(db)
        inner = project(
            np.concatenate(designs),
            np.concatenate(offsets),
            np.concatenate(derivatives),
            np.concatenate(offset_derivatives),
            o.lower[k.gain_indices],
            o.upper[k.gain_indices],
        )
        fitted = vector.copy()
        fitted[k.gain_indices] = inner["gains"]
        o.vector(o.parameters(fitted))
        return fitted, inner["residual"], inner["jacobian"], inner["audit"]


def build(problem, *, reduce_inactive=False):
    """Use affine-block or terminal-output profiling when certified, else joint."""
    settings = checks.ConfidencePolicy()
    rejected = []
    try:
        oracle = AffineRollouts(problem, settings)
    except ValueError as error:
        rejected.append({"kernel": "affine_rollout", "reason": str(error)})
        oracle = checks.Rollouts(problem, settings)
    activity = output_activity.certificate(oracle.system) if reduce_inactive else None
    active = frozenset(activity["active_parameters"]) if activity else None

    def result(profile, route):
        if activity:
            oracle.active_indices = tuple(
                i for i, n in enumerate(oracle.names) if n in active
            )
            route["output_activity"] = activity
            if profile is not None:
                profile = output_activity.ActiveProfile(profile, oracle.names, active)
                route["certificate"] = profile.audit
        return oracle, profile, route

    if isinstance(oracle, AffineRollouts):
        try:
            profile = AffineProfile(oracle, active_parameters=active)
            return result(
                profile,
                {
                    "selected": "affine_profile",
                    "certificate": profile.audit,
                    "inapplicable": rejected,
                },
            )
        except ValueError as error:
            rejected.append({"kernel": "affine_profile", "reason": str(error)})
    try:
        profile = TerminalProfile(oracle, active_parameters=active)
        return result(
            profile,
            {
                "selected": "terminal_output_profile",
                "certificate": profile.audit,
                "inapplicable": rejected,
            },
        )
    except ValueError as error:
        rejected.append({"kernel": "terminal_output_profile", "reason": str(error)})
    return result(
        None,
        {
            "selected": "joint_rollout",
            "inapplicable": rejected,
            "reason": "No exact linear rollout block was symbolically certified.",
        },
    )


def trajectory_losses(oracle, residual):
    """Retain every output/sample and expose worst-trajectory error separately."""
    sizes = [
        len(r.time) * len(oracle.system.channels) for r in oracle.train.trajectories
    ]
    if len(residual) != sum(sizes):
        raise ValueError("rollout residual layout differs from training")
    return [float(np.mean(r**2)) for r in np.split(residual, np.cumsum(sizes)[:-1])]


def verify(oracle, parameters, deadline, checkpoint, *, tight=False, timing=False):
    """Independent DOP853/Radau checks of original equations, never node fits."""
    previous = oracle.settings
    vector = oracle.vector(parameters)
    if tight:
        oracle.settings = previous.model_copy(
            update={"relative_tolerance": 1e-12, "absolute_tolerance": 1e-14}
        )
    try:
        checkpoint({"calls": 1})
        begun = monotonic()
        a, _ = oracle.evaluate(vector, deadline, jacobian=False, method="DOP853")
        dop_seconds = monotonic() - begun
        checkpoint({"calls": 2})
        begun = monotonic()
        b, _ = oracle.evaluate(vector, deadline, jacobian=False, method="Radau")
        radau_seconds = monotonic() - begun
    finally:
        oracle.settings = previous
    if monotonic() >= deadline or not np.isfinite(np.r_[a, b]).all():
        raise TimeoutError("independent verification unavailable within allowance")
    result = {
        "parameters": parameters,
        "training_nmse": float(np.mean(a**2)),
        "alternate_training_nmse": float(np.mean(b**2)),
        "solver_loss_discrepancy": abs(float(np.mean(a**2) - np.mean(b**2))),
        "solver_prediction_discrepancy": float(np.max(np.abs(a - b))),
        "maximum_trajectory_nmse": max(
            *trajectory_losses(oracle, a), *trajectory_losses(oracle, b)
        ),
        "verification": "original_equations_DOP853_vs_Radau_outputs_only",
        "tight_verification": tight,
    }
    if timing:
        result["integrator_seconds"] = {"DOP853": dop_seconds, "Radau": radau_seconds}
    return result | {"usable_for_retention": numerical.usable(result)}
