"""Two observation-equivalent mechanisms and an input-decoupling intervention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from autoformalism.data.scaling import TrainingScaler
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal.mechanisms import (
    MechanismEvaluationSpec,
    evaluate_mechanisms,
)
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit


def request(driver: str) -> PublicFitRequest:
    """Both models have identical complexity, ordinary starts and known preparation."""
    if driver not in {"u1", "u2"}:
        raise ValueError("unknown demo driver")
    candidate = {
        "candidate_id": f"demo_{driver}",
        "parent_candidate_id": None,
        "states": [{"name": "y", "kind": "observed"}, {"name": "z", "kind": "latent"}],
        "state_equations": [
            {"state": "y", "rhs": "z-y"},
            {"state": "z", "rhs": f"-a*z+b*{driver}"},
        ],
        "observation_mappings": [{"channel": "v01", "expression": "y"}],
        "parameters": [
            {"name": "a", "scope": "global", "role": "rate"},
            {"name": "b", "scope": "global", "role": "nonnegative_coefficient"},
        ],
        "initial_conditions": [
            {"state": s, "scope": "global", "fixed_value": 0} for s in ("y", "z")
        ],
    }
    return PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {"targets": ["v01"], "external_inputs": ["u1", "u2"]},
            "initialization_plan": {
                "rules": {
                    "z": {
                        "initial": {
                            "mode": "known",
                            "value": 0,
                            "justification": (
                                "This controlled experiment begins from a prescribed "
                                "unexcited hidden state."
                            ),
                        }
                    }
                }
            },
            "parameter_guesses": {"a": 0.8, "b": 0.8},
            "profile": "collocation-single-target-v2",
            "source": {
                "stage": "synthetic_control",
                "task_id": f"driver_{driver}",
                "artifact_sha256": public.content_sha256(candidate),
            },
        }
    )


def split(name: str, inputs: tuple[tuple[float, float], ...]) -> PublicSplit:
    """Exact generating solution for constant forcing; no hidden labels exposed."""
    time = np.linspace(0, 8, 81)
    rows = []
    for index, (u1, u2) in enumerate(inputs):
        a, b = 0.7, 1.4
        y = (
            b
            * u1
            / a
            * ((1 - np.exp(-time)) - (np.exp(-a * time) - np.exp(-time)) / (1 - a))
        )
        rows.append(
            {
                "trajectory_id": f"{name}_{index}",
                "time": time.tolist(),
                "targets": {"v01": y.tolist()},
                "external_inputs": {"u1": [u1] * len(time), "u2": [u2] * len(time)},
            }
        )
    return PublicSplit.model_validate(
        {"name": name, "fingerprint": f"controlled-demo-{name}", "rows": rows}
    )


def specification() -> MechanismEvaluationSpec:
    return MechanismEvaluationSpec.model_validate(
        {
            "benchmark_id": "controlled_driver_demo",
            "tier": "controlled",
            "required_mechanisms": [
                {
                    "id": "driver_memory",
                    "required_drivers": ["u1"],
                    "required_targets": ["v01"],
                    "requires_dynamic_memory": True,
                }
            ],
        }
    )


def verifier_audit() -> list[dict]:
    """Same candidate pool with and without the equation-based semantic gate."""
    base = request("u1")
    payloads = {
        "correct": base.base_candidate.model_dump(mode="json"),
        "wrong_driver": request("u2").base_candidate.model_dump(mode="json"),
    }
    disconnected = json.loads(json.dumps(payloads["correct"]))
    disconnected["state_equations"][0]["rhs"] = "-y"
    payloads["disconnected_memory"] = disconnected
    algebraic = json.loads(json.dumps(payloads["correct"]))
    algebraic["states"] = algebraic["states"][:1]
    algebraic["state_equations"] = [{"state": "y", "rhs": "b*u1-y"}]
    algebraic["initial_conditions"] = algebraic["initial_conditions"][:1]
    algebraic["parameters"] = [p for p in algebraic["parameters"] if p["name"] != "a"]
    payloads["no_persistent_memory"] = algebraic
    renamed = json.loads(
        json.dumps(payloads["correct"])
        .replace('"z"', '"w"')
        .replace("z-y", "w-y")
        .replace("-a*z", "-a*w")
    )
    payloads["renamed_latent"] = renamed
    scaled = json.loads(json.dumps(payloads["correct"]))
    scaled["state_equations"] = [
        {"state": "y", "rhs": "0.5*z-y"},
        {"state": "z", "rhs": "-a*z+2*b*u1"},
    ]
    payloads["scaled_latent"] = scaled
    rows = []
    for name, payload in payloads.items():
        candidate = base.base_candidate.__class__.model_validate(payload)
        compile_candidate(candidate, base.context)
        result = evaluate_mechanisms(candidate, specification())
        rows.append(
            {
                "case": name,
                "runtime_valid": True,
                "with_semantic_gate": result.mechanism_compliance_complete
                and result.mechanism_compliance == 1,
                "without_semantic_gate": True,
                "evidence": result.model_dump(mode="json"),
            }
        )
    return rows


def run(root: Path) -> dict:
    """Fit on correlated drivers, freeze both, then open decoupled interventions."""
    with public._lock(root):
        train = split("train", ((0.4, 0.4), (1.0, 1.0), (1.6, 1.6)))
        validation = split("val", ((0.7, 0.7), (1.3, 1.3)))
        identity = {
            "protocol": "review-controlled-driver-1",
            "source": public._source_identity(),
            "runtime": public._runtime(),
            "training": public.content_sha256(train),
            "validation": public.content_sha256(validation),
        }
        if (root / "plan.json").exists():
            if {
                k: v
                for k, v in sealed_read(root / "plan.json").items()
                if k != "artifact_sha256"
            } != identity:
                raise ValueError("controlled demo runtime changed")
        else:
            sealed_write(root / "plan.json", identity)
        if (root / "result.json").exists():
            value = sealed_read(root / "result.json")
            frozen = sealed_read(root / "evaluation_freeze.json")
            if any(
                value.get(k) != v for k, v in frozen.items() if k != "artifact_sha256"
            ):
                raise ValueError("demo result differs from its frozen fits")
            for driver in ("u1", "u2"):
                public.inspect_fit(root / driver)
                envelope = public._read(root / driver / "result.json")
                if (
                    envelope["sha256"] != public.content_sha256(envelope["result"])
                    or envelope["result"] != frozen["fits"][driver]
                ):
                    raise ValueError("demo fit differs from its frozen result")
            return value
        fits = {}
        for driver in ("u1", "u2"):
            directory = root / driver
            public.prepare_fit(request(driver), train, validation, directory)
            fits[driver] = public.execute_fit(directory).model_dump(mode="json")
        gate = verifier_audit()
        freeze = {
            "fits": fits,
            "verifier_audit": gate,
            "without_driver_specific_specification": (
                "both observationally compatible; do not choose confidently"
            ),
            "with_true_public_driver_requirement": (
                "u1 is admissible; u2 fails before interventions"
            ),
            "selection_uses_interventions": False,
        }
        if (root / "evaluation_freeze.json").exists():
            existing = sealed_read(root / "evaluation_freeze.json")
            if {k: v for k, v in existing.items() if k != "artifact_sha256"} != freeze:
                raise ValueError("demo fits changed after freeze")
        else:
            sealed_write(root / "evaluation_freeze.json", freeze)
        # These observations are generated only after the selections above are sealed.
        held_out = split("val", ((1.0, 0.0), (0.0, 1.0), (1.5, 0.2)))
        training = public.unpack_split(train)
        scale = TrainingScaler().fit(training).scales["target:v01"].standard_deviation
        rows = []
        for driver, fit in fits.items():
            predictions, errors = [], []
            if fit["parameters"] is not None:
                model, _, _ = public._lower(request(driver))
                for trajectory in public.unpack_split(held_out).trajectories:
                    simulation = simulate_trajectory(
                        model,
                        trajectory,
                        fit["parameters"],
                        {},
                        FitConfig(
                            integration_method="Radau",
                            allow_derivative_regression=False,
                            relative_tolerance=1e-8,
                            absolute_tolerance=1e-10,
                            maximum_wall_time_seconds=30,
                        ),
                        reset_observed_states=False,
                    )
                    if not simulation.success:
                        errors.append(simulation.message)
                        continue
                    predictions.append(simulation.predictions["v01"])
            target = [np.asarray(r.targets["v01"]) for r in held_out.rows]
            nmse = (
                float(
                    np.mean(
                        np.concatenate(
                            [
                                (p - y) / scale
                                for p, y in zip(predictions, target, strict=True)
                            ]
                        )
                        ** 2
                    )
                )
                if len(predictions) == len(target)
                else None
            )
            rows.append(
                {
                    "driver": driver,
                    "train_nmse": fit["training"]["normalized_mse"],
                    "validation_nmse": fit["validation"]["normalized_mse"],
                    "intervention_nmse": nmse,
                    "errors": errors,
                    "parameters": fit["parameters"],
                }
            )
        result = sealed_write(
            root / "result.json",
            {
                **freeze,
                "rows": rows,
                "interpretation": (
                    "A controlled verifier/identifiability demonstration, not "
                    "evidence that the LLM discovered either model."
                ),
                "independent_generating_families_added": 0,
                "wrong_specification_test": False,
            },
        )
        lines = [
            "# Controlled driver demonstration",
            "",
            "z'=-a*z+b*u; y'=z-y. Only y is observed.",
            (
                "Train/validation have u1=u2; interventions decouple them "
                "after freezing both fits."
            ),
            (
                "The correct public driver requirement distinguishes the "
                "models before intervention access."
            ),
            (
                "Without that requirement both remain compatible with the "
                "observational data."
            ),
            "",
            "| Driver | Train NMSE | Validation NMSE | Intervention NMSE |",
            "| --- | ---: | ---: | ---: |",
        ]
        lines += [
            (
                f"| {r['driver']} | {r['train_nmse']} | "
                f"{r['validation_nmse']} | {r['intervention_nmse']} |"
            )
            for r in rows
        ]
        (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
        return result
