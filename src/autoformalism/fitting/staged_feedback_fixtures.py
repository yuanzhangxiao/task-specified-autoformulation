"""Tiny training-only controls for staged fitting-feedback qualification.

These controls are independent of benchmark assets. Every fit uses the ordinary
bounded public bridge, including its sealed backend and deterministic resume.
Reviewer expectations and analytical arguments are returned separately from fit
inputs; they are never supplied to fitting or assessment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import poor_fit
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import scaling_recovery as recovery

PROTOCOL = "staged-feedback-diagnostic-fixtures-1"
_TITLES = {
    "piecewise_inactive_guard": "Inactive guard with a smooth local output map",
    "piecewise_clean_crossing": "Continuous flow crossing a drainage threshold",
    "piecewise_kink": "Observation kink at the retained coefficient",
    "piecewise_bound_limited": "Bound blocks a symmetric perturbation",
    "stationary_poor_retained": "Genuine stationary retained point with poor fit",
    "initializer_path": "Fitted latent initializer contributes to the output",
    "piecewise_initializer_path": "Fitted initializer through an inactive guard",
}


def roster() -> list[dict]:
    """Return reviewer-facing control IDs without fitting data or labels."""
    return [{"case_id": key, "title": title} for key, title in _TITLES.items()]


def _policy(*, piecewise: bool) -> bridge.Policy:
    """Reserve four starts with at most eighty real optimizer evaluations."""
    policy = bridge.Policy(allow_piecewise=piecewise)
    return policy.model_copy(
        update={
            "adapter": policy.adapter.model_copy(
                update={
                    "search": recovery.Policy(
                        maximum_blocks=4,
                        maximum_consecutive_blocks=1,
                        block_calls=20,
                        block_seconds=5,
                        check_seconds=5,
                        sensitivity_seconds=5,
                        poor_fit=poor_fit.Policy(),
                    )
                }
            )
        }
    )


def _case(case_id: str) -> tuple[dict, list[dict], dict]:
    """Build a public candidate and exact samples independently of the fitter."""
    time = np.linspace(0, 3, 13)
    rhs, observation, initial, guess = "a-0.2*max(x-10,0)", "x", 1.0, 1.5
    lower, upper = 0.0, 3.0
    values = initial + time
    evaluator = {
        "expected_accurate": True,
        "expected_diagnosis": "accurate_local_evidence",
        "expected_derivative_support": "supported",
        "expected_active_parameters": ["a"],
        "starting_point_scope": "generic coefficient guess; actual bounded fit",
    }
    if case_id == "piecewise_clean_crossing":
        initial = 9.0
        # x'=1 below 10; afterwards x'=1-0.2*(x-10), with crossing at t=1.
        values = np.where(time <= 1, initial + time, 15 - 5 * np.exp(-0.2 * (time - 1)))
        evaluator["analytic_argument"] = (
            "The continuous vector field crosses x=10 at t=1. Its output "
            "first derivative with respect to a exists across this crossing."
        )
    elif case_id == "piecewise_kink":
        rhs, observation, initial, guess = "1", "x+abs(a-1)", 0.0, 1.0
        values = time.copy()
        evaluator.update(
            expected_diagnosis="accurate_unresolved",
            expected_derivative_support="unsupported_kink",
            starting_point_scope="declared kink control; exact retained-point fit",
            analytic_argument=(
                "At a=1 the observation abs(a-1) has opposite one-sided "
                "derivatives. Agreement of AD with a central difference alone "
                "cannot establish a derivative or local rank confidence."
            ),
        )
    elif case_id == "piecewise_bound_limited":
        guess = 0.0
        values = np.full_like(time, initial)
        evaluator.update(
            expected_diagnosis="accurate_unresolved",
            expected_derivative_support="infeasible_direction",
            starting_point_scope="declared boundary control; exact retained-point fit",
            analytic_argument=(
                "The exact fit is at a=0, its declared lower bound. A symmetric "
                "parameter perturbation is infeasible even though x<10."
            ),
        )
    elif case_id == "stationary_poor_retained":
        rhs, initial, guess, lower = "a*a", 0.0, 0.0, -3.0
        values = -time
        evaluator.update(
            expected_accurate=False,
            expected_diagnosis="stationary_poor_fit",
            expected_derivative_support="smooth_local",
            starting_point_scope="declared zero coefficient; actual multistart search",
            analytic_argument=(
                "The candidate predicts a^2*t while observations are -t. "
                "The retained a=0 has nonzero residuals and an actual zero "
                "residual Jacobian. Its stationarity witness must come from "
                "the ordinary search, not an injected stop flag."
            ),
        )
    elif case_id in {"initializer_path", "piecewise_initializer_path"}:
        piecewise = case_id == "piecewise_initializer_path"
        rhs, initial = "z-0.2*max(x-10,0)" if piecewise else "z", 0.0
        values = 0.7 * time
        evaluator.update(
            expected_derivative_support="supported" if piecewise else "smooth_local",
            expected_active_parameters=["init_z_value"],
            analytic_argument=(
                "The latent state z is constant and x'=z. The fitted global "
                "initializer therefore contributes sensitivity t to x. The "
                "optional drainage guard remains inactive below x=10."
            ),
        )
    else:
        evaluator["analytic_argument"] = (
            "At the fitted a=1, x ranges from 1 to 4, strictly below the "
            "drainage threshold 10. Nearby two-sided perturbations are smooth."
        )
    candidate = {
        "candidate_id": f"synthetic_staged_{case_id}",
        "parent_candidate_id": None,
        "states": [
            {
                "name": "x",
                "kind": "latent" if case_id == "piecewise_kink" else "observed",
            }
        ],
        "state_equations": [{"state": "x", "rhs": rhs}],
        "observation_mappings": [{"channel": "y", "expression": observation}],
        "parameters": [
            {
                "name": "a",
                "role": "coefficient",
                "scope": "global",
                "bounds": {"lower": lower, "upper": upper},
            }
        ],
        "initial_conditions": [
            {"state": "x", "scope": "global", "fixed_value": initial}
        ],
    }
    rules = {
        "x": {
            "initial": {
                "mode": "known",
                "value": initial,
                "justification": "Synthetic control prescribes its prepared state.",
            }
        }
    }
    if case_id != "piecewise_kink":
        rules = {}
    guesses = {"a": guess}
    if case_id in {"initializer_path", "piecewise_initializer_path"}:
        candidate["states"].append({"name": "z", "kind": "latent"})
        candidate["state_equations"].append({"state": "z", "rhs": "0"})
        candidate["initial_conditions"].append(
            {"state": "z", "scope": "global", "fixed_value": 0.0}
        )
        candidate["parameters"] = []
        guesses = {}
        rules["z"] = {"initial": {"mode": "value", "guess": 1.5}}
    request = {
        "base_candidate": candidate,
        "context": {"targets": ["y"]},
        "initialization_plan": {"rules": rules},
        "parameter_guesses": guesses,
        "profile": "collocation-feasible-v1",
        "source": {
            "stage": "synthetic_control",
            "task_id": case_id,
            "artifact_sha256": public.content_sha256({"protocol": PROTOCOL}),
        },
    }
    rows = [
        {
            "trajectory_id": "train_0",
            "time": time.tolist(),
            "targets": {"y": values.tolist()},
            "external_inputs": {},
            "auxiliaries": {},
        }
    ]
    return request, rows, evaluator


def make(case_id: str) -> tuple[bridge.Inputs, bridge.Policy, dict]:
    """Separate training-only fit inputs from qualification-only expectations."""
    if case_id not in _TITLES:
        raise ValueError(f"unknown staged feedback fixture: {case_id}")
    request, rows, evaluator = _case(case_id)
    inputs = bridge.Inputs.model_validate(
        {
            "request": request,
            "training": {
                "name": "train",
                "rows": rows,
                "fingerprint": public.content_sha256(rows),
            },
        }
    )
    evaluator.update(
        protocol=PROTOCOL,
        case_id=case_id,
        benchmark_claim=False,
        reference_values_given_to_fitter=False,
        validation_or_test_data_opened=False,
        observation_noise="none",
    )
    return inputs, _policy(piecewise=case_id.startswith("piecewise_")), evaluator


def fit_fixture(case_id: str, folder: Path) -> dict:
    """Run the actual bridge with bounded checkpoints, returning no oracle labels."""
    inputs, policy, _ = make(case_id)
    result = bridge.run(inputs, policy, folder)
    if result.fit is None:
        raise ValueError("staged feedback fixture preparation failed")
    return {
        "problem": result.preparation["value"]["problem"],
        "fit": result.fit.model_dump(mode="json"),
        "backend": read_seal(folder / "fit/backend.json"),
    }
