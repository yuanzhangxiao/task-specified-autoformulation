#!/usr/bin/env python3
"""Real CPU fits with distinct dynamic outputs and an algebraic third output.

This synthetic software control is not a Dalla Man recovery experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autoformalism.data import Trajectory
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import sibling_fit
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def control(*, algebraic: bool = True):
    """Generate two training trajectories and an unseen insulin-like input schedule."""
    targets = ("Gp", "I", "U") if algebraic else ("Gp", "I")
    candidate = {
        "candidate_id": "multi_output_control",
        "parent_candidate_id": None,
        "states": [{"name": n, "kind": "latent"} for n in ("g", "i", "q")],
        "state_equations": [
            {"state": "g", "rhs": "-a*g + 0.2*i + meal"},
            {"state": "i", "rhs": "-b*i + insulin"},
            {"state": "q", "rhs": "0.4*(i-q)"},
        ],
        "observation_mappings": [
            {"channel": "Gp", "expression": "g"},
            {"channel": "I", "expression": "i"},
        ],
        "parameters": [
            {"name": n, "scope": "global", "role": "rate"} for n in ("a", "b")
        ],
        "initial_conditions": [
            {"state": n, "scope": "global", "fixed_value": 0} for n in ("g", "i", "q")
        ],
    }
    if algebraic:
        candidate["processes"] = [{"name": "disposal", "expression": "c*q*g+Uii"}]
        candidate["parameters"].append({"name": "c", "scope": "global", "role": "rate"})
        candidate["observation_mappings"].append(
            {"channel": "U", "expression": "disposal"}
        )
    # In the two-output case q is intentionally unobserved and has no fitted boundary.
    rules = {"q": {"initial": {"mode": "map", "expression": "0.5*I", "parameters": []}}}
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {
                "targets": targets,
                "external_inputs": ("meal", "insulin"),
                "auxiliaries": ("Uii",) if algebraic else (),
            },
            "initialization_plan": {"rules": rules},
            "parameter_guesses": {
                "a": 0.3,
                "b": 0.5,
                **({"c": 0.01} if algebraic else {}),
            },
            "profile": "collocation-multi-target-v1",
            "source": {
                "stage": "synthetic_control",
                "task_id": "multi-target-control",
                "artifact_sha256": public.content_sha256(candidate),
            },
        }
    )
    model, _, _ = public._lower(request)
    truth = {"a": 0.7, "b": 1.2, **({"c": 0.08} if algebraic else {})}
    time = np.linspace(0, 2, 21)

    def split(name: str, initials: tuple) -> PublicSplit:
        rows = []
        for index, (g0, i0) in enumerate(initials):
            observations = {"Gp": np.full_like(time, g0), "I": np.full_like(time, i0)}
            if algebraic:
                observations["U"] = np.full_like(time, 0.08 * 0.5 * i0 * g0 + 1)
            row = Trajectory(
                f"{name}_{index}",
                time,
                observations,
                {"Uii": np.ones_like(time)} if algebraic else {},
                {
                    "meal": np.full_like(time, 5 + 2 * index),
                    "insulin": 0.3 + 0.9 * (time >= 1)
                    if name == "val"
                    else np.full_like(time, 0.5 + index),
                },
                {},
                {},
            )
            result = simulate_trajectory(
                model,
                row,
                truth,
                {},
                FitConfig(
                    integration_method="Radau",
                    relative_tolerance=1e-10,
                    absolute_tolerance=1e-12,
                ),
            )
            assert result.success, result.message
            rows.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "time": time.tolist(),
                    "targets": {n: v.tolist() for n, v in result.predictions.items()},
                    "auxiliaries": {n: v.tolist() for n, v in row.auxiliaries.items()},
                    "external_inputs": {
                        n: v.tolist() for n, v in row.external_inputs.items()
                    },
                }
            )
        return PublicSplit.model_validate(
            {"name": name, "fingerprint": f"synthetic-{name}", "rows": rows}
        )

    return (
        request,
        split("train", ((100.0, 1.0), (170.0, 1.7))),
        split("val", ((60.0, 0.6),)),
        truth,
    )


def run(directory: Path) -> dict:
    """Exercise both fit dimensions, exact resume, and a real warm-started revision."""
    reports = []
    for algebraic in (False, True):
        request, train, val, truth = control(algebraic=algebraic)
        output = directory / ("three-targets" if algebraic else "two-targets")
        public.prepare_fit(request, train, val, output)
        result = public.execute_fit(output)
        assert result.status == "complete", result.message
        for metrics in (result.training, result.validation):
            assert set(metrics.per_target_normalized_mse) == set(
                request.context.targets
            )
            assert max(metrics.per_target_normalized_mse.values()) < 1e-7, metrics
        for name, value in truth.items():
            assert np.isclose(result.parameters[name], value, rtol=2e-3, atol=1e-4)
        saved = (output / "result.json").read_bytes()
        assert public.execute_fit(output) == result
        assert (output / "result.json").read_bytes() == saved
        backend = json.loads((output / "backend_result.json").read_text())
        assert backend["initializer"]["success"], backend["initializer"]
        if algebraic:
            # c appears only in U: this also detects an initializer dropping U.
            assert np.isclose(
                backend["initializer"]["parameters"]["c"], truth["c"], rtol=2e-3
            )
            raw = request.model_dump(mode="json")
            raw["base_candidate"]["processes"][0]["expression"] = "c*q*g+Uii+offset"
            raw["base_candidate"]["parameters"].append(
                {"name": "offset", "scope": "global", "role": "offset"}
            )
            child = PublicFitRequest.model_validate(raw)
            child_dir = output / "child"
            sibling_fit.prepare_child_fit(
                request,
                child,
                dict(result.parameters),
                train,
                val,
                child_dir,
                lineage={"control": "algebraic_revision"},
            )
            revised = sibling_fit.execute_child_fit(child_dir)
            assert revised.status == "complete", revised.message
            assert max(revised.validation.per_target_normalized_mse.values()) < 1e-7
            assert sibling_fit.execute_child_fit(child_dir) == revised
        reports.append(
            {
                "targets": request.context.targets,
                "training_per_target_nmse": dict(
                    result.training.per_target_normalized_mse
                ),
                "validation_per_target_nmse": dict(
                    result.validation.per_target_normalized_mse
                ),
                "initializer_success": True,
                "resume_unchanged": True,
            }
        )
    return {
        "status": "pass",
        "fits": reports,
        "multi_target_child_fit": "pass",
        "live_llm_calls": 0,
        "benchmark_data_opened": False,
        "test_data_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
