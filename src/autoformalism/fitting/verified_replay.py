"""Fixed-parameter evaluator replay with one charged numerical retry."""

from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import qualification
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


class ReplayPolicy(StrictSchema):
    """Tolerances are pinned; retries tighten accuracy, never weaken agreement."""

    protocol: Literal["verified-fixed-replay-1"] = "verified-fixed-replay-1"
    initial_seconds: float = Field(default=300, ge=1, le=600, allow_inf_nan=False)
    retry_seconds: float = Field(default=300, ge=1, le=600, allow_inf_nan=False)
    agreement_limit: float = Field(default=1e-4, gt=0, le=1e-4, allow_inf_nan=False)


def agrees(value: dict | None, limit: float) -> bool:
    """Missing/nonfinite results cannot pass numerical verification."""
    if not value or not value.get("complete"):
        return False
    difference = value.get("maximum_solver_difference")
    scores = value.get("metrics", {})
    return bool(
        difference is not None
        and np.isfinite(difference)
        and 0 <= difference <= limit
        and set(scores) == {"train", "val"}
        and all(v is not None and np.isfinite(v) and v >= 0 for v in scores.values())
    )


def run(
    request: PublicFitRequest,
    parameters: dict[str, float],
    training: PublicSplit,
    validation: PublicSplit,
    policy: ReplayPolicy,
    folder: Path,
) -> dict:
    """Retry an unavailable/disagreeing replay once, without any parameter fitting.

    This evaluator is outside the optimizer interface. Both attempts and their
    original errors survive; an interrupted attempt consumes its full allowance.
    """
    identity = {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "request_sha256": public.content_sha256(request.model_dump(mode="json")),
        "parameters_sha256": public.content_sha256(parameters),
        "training_sha256": public.content_sha256(training.model_dump(mode="json")),
        "validation_sha256": public.content_sha256(validation.model_dump(mode="json")),
        "policy": policy.model_dump(mode="json"),
    }
    with public._lock(folder):
        seal(folder / "identity.json", identity)
        finished = folder / "result.json"
        if finished.exists():
            saved = read_seal(finished)
            if saved["identity"] != identity:
                raise ValueError("verified replay identity differs")
            return saved
        attempts = []
        for name, seconds, rtol, atol in (
            ("initial", policy.initial_seconds, 1e-9, 1e-11),
            ("tight", policy.retry_seconds, 1e-12, 1e-14),
        ):

            def work(end, save, name=name, rtol=rtol, atol=atol):
                save({"fixed_parameters_sha256": identity["parameters_sha256"]})
                return qualification.replay(
                    request,
                    parameters,
                    training,
                    validation,
                    max(0, end - monotonic()),
                    journal=folder / name / "replay_progress.json",
                    relative_tolerance=rtol,
                    absolute_tolerance=atol,
                )

            op = checks.operation(
                folder / name,
                identity | {"rtol": rtol, "atol": atol},
                seconds,
                work,
            )
            attempts.append({"attempt": name, "rtol": rtol, "atol": atol, **op})
            if agrees(op.get("value"), policy.agreement_limit):
                break
        last = attempts[-1].get("value")
        passed = agrees(last, policy.agreement_limit)
        result = {
            "identity": identity,
            "status": "verified" if passed else "unverified",
            "solver_agreement_passed": passed,
            "retry_used": len(attempts) == 2,
            "attempts": attempts,
            "selected_attempt": attempts[-1]["attempt"] if passed else None,
            "metrics": last.get("metrics") if last else None,
            "maximum_solver_difference": last.get("maximum_solver_difference")
            if last
            else None,
            "wall_seconds": sum(a["wall_seconds"] or 0 for a in attempts),
            "cpu_seconds": sum(a["cpu_seconds"] or 0 for a in attempts),
            "budget_charge_seconds": sum(a["budget_charge_seconds"] for a in attempts),
            "accounting_complete": all(a["accounting_complete"] for a in attempts),
            "parameter_fitting_performed": False,
            "validation_initials_fitted": False,
            "test_data_opened": False,
        }
        seal(finished, result)
        return read_seal(finished)
