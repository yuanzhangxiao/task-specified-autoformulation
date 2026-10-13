"""Analytic, training-only basin controls for fitting-feedback qualification.

These are newly synthesized controls, not benchmark observations or benchmark
claims. Reviewer labels and independent analytic witnesses are returned on a
separate evaluator channel and never enter the public request or fitter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import poor_fit
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import public_rollout as bridge
from autoformalism.fitting import scaling_recovery as recovery

PROTOCOL = "analytic-basin-feedback-fixtures-1"
_TITLES = {
    "adequate_drainage": "Adequate drainage candidate",
    "omitted_drainage": "Candidate omits drainage from an adequate specification",
    "ambiguous_transfer": "Accurate output with ambiguous parallel inflow gains",
    "budget_limited_drainage": "Adequate nonlinear drainage with a short search",
    "false_no_drainage_claim": "Explicit synthetic no-drainage claim contradicted",
}


def roster() -> list[dict]:
    """Return evaluator-facing control names, with no fit inputs or outcomes."""
    return [{"case_id": key, "title": value} for key, value in _TITLES.items()]


def _rows(*, nonlinear: bool) -> list[dict]:
    """Sample closed-form storage solutions; no numerical reference generator."""
    time = np.linspace(0, 4, 17)
    gain, drainage = 0.7, 0.8
    rows = []
    for index, (initial, inflow) in enumerate(((2.0, 0.1), (0.2, 0.9))):
        if nonlinear:
            equilibrium = np.sqrt(gain * inflow / drainage)
            decay = np.exp(-2 * drainage * equilibrium * time)
            numerator = initial + equilibrium + (initial - equilibrium) * decay
            denominator = initial + equilibrium - (initial - equilibrium) * decay
            depth = equilibrium * numerator / denominator
        else:
            equilibrium = gain * inflow / drainage
            depth = equilibrium + (initial - equilibrium) * np.exp(-drainage * time)
        rows.append(
            {
                "trajectory_id": f"train_{index}",
                "time": time.tolist(),
                "targets": {"h": depth.tolist()},
                "external_inputs": {"u": np.full_like(time, inflow).tolist()},
                "auxiliaries": {},
            }
        )
    return rows


def _policy(*, limited: bool) -> bridge.Policy:
    """Use at most forty calls, or eight calls for the uncertainty control."""
    policy = bridge.Policy()
    return policy.model_copy(
        update={
            "adapter": policy.adapter.model_copy(
                update={
                    "search": recovery.Policy(
                        maximum_blocks=4,
                        maximum_consecutive_blocks=1,
                        block_calls=2 if limited else 10,
                        block_seconds=5,
                        check_seconds=5,
                        sensitivity_seconds=5,
                        poor_fit=poor_fit.Policy(),
                    )
                }
            )
        }
    )


def no_drainage_certificate(training: dict) -> dict:
    """Evaluate the stated synthetic model-class inequality on observations.

    The explicit class is h'=g*u with g>=0 and piecewise-linear u>=0, hence
    every admissible h is nondecreasing. A decreasing pair contradicts that
    class irrespective of optimization. This is not inferred from a poor fit
    and makes no assertion about any natural-language benchmark specification.
    """
    if training.get("name") != "train":
        raise ValueError("certificate requires training observations only")
    violations = []
    for row in training["rows"]:
        time = np.asarray(row["time"], dtype=float)
        depth = np.asarray(row["targets"]["h"], dtype=float)
        inflow = np.asarray(row["external_inputs"]["u"], dtype=float)
        if (
            len(time) < 2
            or time.shape != depth.shape
            or time.shape != inflow.shape
            or not np.isfinite(np.r_[time, depth, inflow]).all()
            or np.any(np.diff(time) <= 0)
            or np.any(inflow < 0)
        ):
            raise ValueError("invalid monotonicity-certificate observations")
        for index in np.flatnonzero(np.diff(depth) < -1e-10):
            violations.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "time_pair": time[index : index + 2].tolist(),
                    "observed_depth_pair": depth[index : index + 2].tolist(),
                    "decrease": float(depth[index] - depth[index + 1]),
                }
            )
    return {
        "protocol": "synthetic-no-drainage-monotonicity-1",
        "explicit_claim": "h'=g*u; g>=0; u>=0; direct measured depth h",
        "implication": "h(t2)>=h(t1) whenever t2>t1",
        "evaluated_training_sha256": public.content_sha256(training),
        "contradicted": bool(violations),
        "violations": violations,
        "scope": "Only the explicitly defined synthetic model class.",
        "uses_fit_failure": False,
    }


def make(case_id: str) -> tuple[bridge.Inputs, bridge.Policy, dict]:
    """Separate allowed fit inputs from reviewer-only expectations and witnesses."""
    if case_id not in _TITLES:
        raise ValueError(f"unknown feedback fixture: {case_id}")
    limited = case_id == "budget_limited_drainage"
    omitted = case_id in {"omitted_drainage", "false_no_drainage_claim"}
    ambiguous = case_id == "ambiguous_transfer"
    names = ["g"] if omitted else ["g1", "g2"] if ambiguous else ["g", "k"]
    rhs = (
        "g*u"
        if omitted
        else "(g1+g2)*u-0.8*h"
        if ambiguous
        else "g*u-k*h*h"
        if limited
        else "g*u-k*h"
    )
    rows = _rows(nonlinear=limited)
    training = {
        "name": "train",
        "rows": rows,
        "fingerprint": public.content_sha256(rows),
    }
    # This generic start does not depend on the reference coefficients or labels.
    guesses = dict.fromkeys(names, 1.5)
    inputs = bridge.Inputs.model_validate(
        {
            "request": {
                "base_candidate": {
                    "candidate_id": "synthetic_basin_candidate",
                    "parent_candidate_id": None,
                    "states": [{"name": "h", "kind": "observed"}],
                    "state_equations": [{"state": "h", "rhs": rhs}],
                    "observation_mappings": [{"channel": "h", "expression": "h"}],
                    "parameters": [
                        {
                            "name": name,
                            "role": "nonnegative_coefficient",
                            "scope": "global",
                            "bounds": {"lower": 0.0, "upper": 5.0},
                        }
                        for name in names
                    ],
                    "initial_conditions": [
                        {"state": "h", "scope": "global", "fixed_value": 0.0}
                    ],
                },
                "context": {"targets": ["h"], "external_inputs": ["u"]},
                "initialization_plan": {"rules": {}},
                "parameter_guesses": guesses,
                "profile": "collocation-feasible-v1",
                "source": {
                    "stage": "synthetic_control",
                    "task_id": "synthetic_basin",
                    "artifact_sha256": public.content_sha256({"protocol": PROTOCOL}),
                },
            },
            "training": training,
        }
    )
    evaluator = {
        "protocol": PROTOCOL,
        "case_id": case_id,
        "synthetic_explicit_claim": (
            "h'=g*u with g>=0; no loss or drainage; directly observed depth"
            if case_id == "false_no_drainage_claim"
            else "h'=(g1+g2)*u-0.8*h; g1,g2>=0; known drainage rate 0.8"
            if ambiguous
            else "h'=g*u-k*h*h; g,k>=0; directly observed depth"
            if limited
            else "h'=g*u-k*h; g,k>=0; directly observed depth"
        ),
        "expected_search_status": "output_problem_solved"
        if omitted
        else "further_search_needed"
        if limited
        else "target_reached",
        "expected_diagnosis": "solved_restricted_problem_poor_fit"
        if omitted
        else "search_incomplete"
        if limited
        else "accurate_ambiguity"
        if ambiguous
        else "accurate_local_evidence",
        "expected_accurate": not (omitted or limited),
        "expected_rank": 1 if omitted or ambiguous else 2,
        "adequate_witness": {"g1": 0.2, "g2": 0.5}
        if ambiguous
        else {"g": 0.7, "k": 0.8},
        "adequate_witness_rhs": "g*u-k*h*h" if limited else "g*u-k*h",
        "expected_claim_compatibility": "contradicted"
        if case_id == "false_no_drainage_claim"
        else "compatible_witness_exists",
        "benchmark_claim": False,
        "observation_noise": "none",
        "reference_values_given_to_fitter": False,
        "validation_or_test_data_opened": False,
    }
    if ambiguous:
        evaluator["adequate_witness_rhs"] = rhs
        evaluator["ambiguity_witness"] = {"g1": 0.4, "g2": 0.3}
        evaluator["ambiguity_argument"] = "Only g1+g2 appears in any equation."
    if omitted:
        design = np.concatenate(
            [np.asarray(row["time"]) * row["external_inputs"]["u"][0] for row in rows]
        )
        change = np.concatenate(
            [np.asarray(row["targets"]["h"]) - row["targets"]["h"][0] for row in rows]
        )
        evaluator["analytic_candidate_optimum"] = {
            "g": float(np.clip(design @ change / (design @ design), 0, 5))
        }
    if case_id == "false_no_drainage_claim":
        evaluator["incompatibility_certificate"] = no_drainage_certificate(
            inputs.training.model_dump(mode="json")
        )
    return inputs, _policy(limited=limited), evaluator


def fit_fixture(case_id: str, folder: Path) -> dict:
    """Run a real bounded fit with sealed checkpoints; return no reviewer labels."""
    inputs, policy, _ = make(case_id)
    result = bridge.run(inputs, policy, folder)
    if result.fit is None:
        raise ValueError("analytic feedback fixture preparation failed")
    return {
        "problem": result.preparation["value"]["problem"],
        "fit": result.fit.model_dump(mode="json"),
        "backend": read_seal(folder / "fit/backend.json"),
    }
