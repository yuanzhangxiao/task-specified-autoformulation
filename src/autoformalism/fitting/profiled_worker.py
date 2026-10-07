"""Training-only profiled rollout optimization under the journaled supervisor."""

from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.bounded_screening import TrainingOnlySplit
from autoformalism.fitting.feasibility import EvaluationBudget
from autoformalism.fitting.identifiable_campaign import SETTINGS, scale_for
from autoformalism.fitting.identifiable_refinement import RefinementPolicy
from autoformalism.fitting.operation_timing import timed
from autoformalism.fitting.portfolio_starts import domain, valid
from autoformalism.fitting.profiled_output import ProfiledOutput, project
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicFitRequest


class _Stop(Exception):
    pass


def rollout(payload: dict, folder: Path, *, profiler=ProfiledOutput) -> dict:
    """One original start, exact inner bounded LS, outer scaled TRF, no restarts."""
    expected = {
        "request",
        "training",
        "coordinates",
        "points",
        "seconds",
        "maximum_calls",
        "training_nmse",
        "trajectory_nmse",
    }
    if set(payload) != expected or len(payload["points"]) != 1:
        raise ValueError("profiled worker requires one training-only starting pair")
    launched = public._read(folder / "launch.json")["monotonic"]
    budget = EvaluationBudget(
        launched + payload["seconds"] - 1, payload["maximum_calls"]
    )
    training = public.unpack_split(
        TrainingOnlySplit.model_validate(payload["training"])
    )
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(payload["request"]))[0],
        allow_piecewise=True,
    )
    profiled = profiler(system)
    bounds = domain(payload)
    names, lo, hi, units = bounds
    if names != system.names or not valid(payload["points"][0]["parameters"], bounds):
        raise ValueError("profiled starting pair violates original fitting domain")
    anchor = np.array([payload["points"][0]["parameters"][n] for n in names])
    outer, gains = profiled.outer_indices, profiled.gain_indices
    scales = scale_for(training)
    policy = RefinementPolicy()
    target = system.channels[0]
    sizes = [len(row.time) for row in training.trajectories]
    directory = folder / "refinement"
    directory.mkdir(parents=True, exist_ok=True)
    best, last_q, last_jac, previous = None, None, None, None
    history, stalled = [], 0
    totals = {
        "nfev": 0,
        "segments": 0,
        "complete_trajectory_integrations": 0,
        "bounded_linear_solves": 0,
    }
    public._write(directory / "structure.json", profiled.audit)

    @timed("residual_point")
    def evaluate(q):
        nonlocal best, last_q, last_jac
        budget.take()
        started = monotonic()
        deadline = min(budget.deadline, started + 30)
        vector = anchor.copy()
        vector[outer] += units[outer] * q
        record = {
            "call": budget.calls,
            "outer_parameters": dict(
                zip(profiled.outer_names, vector[outer].tolist(), strict=True)
            ),
            "complete": False,
        }
        try:
            designs, offsets, derivatives, offset_derivatives = [], [], [], []
            for i, row in enumerate(training.trajectories):
                public._write(
                    directory / "progress.json",
                    {
                        "call": budget.calls,
                        "trajectory": row.trajectory_id,
                        "complete_trajectories": i,
                        "expected_trajectories": len(sizes),
                    },
                )
                a, b, da, db, counts = profiled.trajectory(
                    row, vector, SETTINGS, deadline
                )
                totals["complete_trajectory_integrations"] += 1
                for key in counts:
                    totals[key] += counts[key]
                designs.append(a / scales[target])
                offsets.append((b - row.targets[target]) / scales[target])
                derivatives.append(da / scales[target])
                offset_derivatives.append(db / scales[target])
            if monotonic() >= deadline:
                raise TimeoutError("profiled point deadline before linear solve")
            a, b = np.concatenate(designs), np.concatenate(offsets)
            inner = project(
                a,
                b,
                np.concatenate(derivatives),
                np.concatenate(offset_derivatives),
                lo[gains],
                hi[gains],
            )
            totals["bounded_linear_solves"] += 1
            vector[gains] = inner["gains"]
            residual = inner["residual"]
            if (
                not np.isfinite(residual).all()
                or np.max(abs(residual)) >= SETTINGS.failure_penalty
            ):
                raise ValueError(
                    "profiled residual reaches production clipping threshold"
                )
            last_q, last_jac = q.copy(), inner["jacobian"] * units[outer]
            nmse = float(np.mean(residual**2))
            worst = max(
                float(np.mean(r**2)) for r in np.split(residual, np.cumsum(sizes)[:-1])
            )
            parameters = dict(zip(names, vector.tolist(), strict=True))
            if not valid(parameters, bounds):
                raise ValueError("profiled vector violates original bounds")
            record.update(
                complete=True,
                training_nmse=nmse,
                maximum_trajectory_nmse=worst,
                parameters=parameters,
                inner=inner["audit"],
                original_gain_training_nmse=float(
                    np.mean((a @ anchor[gains] + b) ** 2)
                ),
            )
            if best is None or nmse < best["training_nmse"]:
                best = {
                    "parameters": parameters,
                    "training_nmse": nmse,
                    "maximum_trajectory_nmse": worst,
                    "call": budget.calls,
                    "inner": inner["audit"],
                }
                public._write(directory / "best.json", best)
            if nmse <= payload["training_nmse"] and worst <= payload["trajectory_nmse"]:
                raise _Stop("training_accuracy_reached_pending_replay")
            return residual
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            record["error"] = str(error)[-1000:]
            raise
        finally:
            record["seconds"] = monotonic() - started
            history.append(record)
            public._write(directory / "training_history.json", history)
            public._write(
                directory / "accounting.json",
                totals | {"attempted_calls": budget.calls},
            )

    def jacobian(q):
        if last_q is None or not np.array_equal(q, last_q):
            raise ValueError(
                "profiled Jacobian requested without its completed residual"
            )
        return last_jac

    def callback(intermediate_result):
        nonlocal previous, stalled
        current = (float(intermediate_result.cost), intermediate_result.x.copy())
        if previous is not None:
            improvement = (previous[0] - current[0]) / max(abs(previous[0]), 1e-30)
            step = np.linalg.norm(current[1] - previous[1])
            stalled = (
                stalled + 1
                if (
                    improvement <= policy.relative_progress
                    and step <= policy.scaled_step
                )
                else 0
            )
            if stalled >= policy.stall_steps:
                raise _Stop("stalled")
        previous = current

    reason, native = "optimizer_terminated", None
    try:
        solved = least_squares(
            evaluate,
            np.zeros(len(outer)),
            jac=jacobian,
            bounds=(
                (lo[outer] - anchor[outer]) / units[outer],
                (hi[outer] - anchor[outer]) / units[outer],
            ),
            method="trf",
            ftol=None,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=payload["maximum_calls"],
            callback=callback,
        )
        native = {
            "success": bool(solved.success),
            "status": int(solved.status),
            "message": solved.message,
        }
    except _Stop as error:
        reason = str(error)
    except TimeoutError:
        reason = (
            "budget_exhausted"
            if monotonic() >= budget.deadline
            else "point_allowance_exhausted"
        )
    except (ValueError, RuntimeError, ArithmeticError) as error:
        reason, native = "numerical_failure", {"error": str(error)[-1000:]}
    result = {
        "payload_sha256": public.content_sha256(payload),
        "parameters": best["parameters"] if best else None,
        "training_nmse": best["training_nmse"] if best else None,
        "stop_reason": reason,
        "native": native,
        "actual_residual_calls": budget.calls,
        "accounting": totals,
        "profiled_output": profiled.audit,
        "selected_inner": best["inner"] if best else None,
        "seconds": monotonic() - launched,
        "budget_exhausted": budget.calls >= budget.maximum
        or monotonic() >= budget.deadline,
    }
    public._write(folder / "result.json", result)
    return result
