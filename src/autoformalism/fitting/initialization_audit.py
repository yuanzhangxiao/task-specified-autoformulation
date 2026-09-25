"""Read-only compatibility checks of saved public initialization handoffs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def audit_saved_initialization(directory: Path) -> dict:
    """Check a historical freeze under current code without resuming its budget.

    Source differences are reported, not resealed. All other handoff fields must
    still agree. Boundary comparisons use saved guesses, not optimized values.
    No simulation, optimizer, LLM, model selection or filesystem mutation occurs.
    """
    frozen = json.loads((directory / "freeze.json").read_text())
    payload = {k: v for k, v in frozen.items() if k != "identity"}
    if public.content_sha256(payload) != frozen["identity"]:
        raise ValueError("saved fit freeze digest differs")
    request = PublicFitRequest.model_validate(frozen["request"])
    train = PublicSplit.model_validate(frozen["training"])
    val = PublicSplit.model_validate(frozen["validation"])
    current = public._bundle(request, train, val)

    def comparable(value: dict) -> dict:
        return {
            k: v for k, v in value.items() if k not in {"identity", "source_sha256"}
        }

    if comparable(frozen) != comparable(current):
        raise ValueError("saved content, lowering or profile differs beyond source")
    model, guesses, _ = public._lower(request)
    process_names = set(model.validated.process_expressions)
    collisions = {
        state: sorted(expression.symbols & process_names)
        for state, expression in model.validated.initial_condition_expressions.items()
        if state not in model.direct_state_observation_channels
        and expression.symbols & process_names
    }
    issue = public._capability(request, model)
    report = {
        "directory": str(directory),
        "task": request.source.task_id,
        "identity": frozen["identity"],
        "historical_source_sha256": frozen["source_sha256"],
        "audit_source_sha256": current["source_sha256"],
        "source_matches": frozen["source_sha256"] == current["source_sha256"],
        "initial_observation_process_name_collisions": collisions,
        "capability_supported": issue is None,
        "capability_message": issue,
        "fit_started_marker_exists": (directory / "started.json").exists(),
        "fit_result_exists": (directory / "result.json").exists(),
        "boundary_comparison": {"status": "not_run"},
    }
    if issue is not None or request.profile == "general-rollout-v1":
        return report
    system = SymbolicODE(model, allow_piecewise=True)
    missing = sorted(set(system.names) - set(guesses))
    if missing:
        report["boundary_comparison"] = {
            "status": "missing_saved_guesses",
            "parameters": missing,
        }
        return report
    theta = np.array([guesses[name] for name in system.names])
    rows = []
    for split in (train, val):
        for row in public.unpack_split(split).trajectories:
            try:
                production = system.initial_for(row, theta)
                symbolic = np.asarray(system.initial_symbolic(row, theta)).ravel()
                agree = bool(
                    np.all(np.isfinite(production))
                    and np.all(np.isfinite(symbolic))
                    and np.allclose(production, symbolic, rtol=1e-10, atol=1e-10)
                )
                result = {"agree": agree}
            except (ValueError, RuntimeError, KeyError) as error:
                result = {"agree": False, "error": f"{type(error).__name__}: {error}"}
            rows.append(
                {"split": split.name, "trajectory": row.trajectory_id, **result}
            )
    report["boundary_comparison"] = {
        "status": "agree" if all(row["agree"] for row in rows) else "disagreement",
        "at": "saved_parameter_guesses",
        "rows": rows,
    }
    return report
