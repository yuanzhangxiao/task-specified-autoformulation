"""Evaluator-assisted CSTR attainability diagnostics using the public fitter.

Known equations and reference-centred starts are diagnostic assistance, not
discovery results. No private trajectory is passed to the fitting objective.
"""

from __future__ import annotations

import math
from pathlib import Path
from time import monotonic

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.reference_audit import _digest
from autoformalism.config import DataConfig
from autoformalism.data import BenchmarkLoader
from autoformalism.data.registry import BenchmarkRegistry, continuous_phase_b_specs
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

SPEC_PATH = Path(
    "benchmark5_anonymous_nonlinear_process/private/system_specification.json"
)
STARTS = ("reference_near", "generic")
REPLAY_LIMIT_SECONDS = 240
REFERENCE_NMSE_LIMIT = 1e-3
FIT_NMSE_TARGET = 1e-2
SOLVER_AGREEMENT_LIMIT = 1e-4


def cstr_request(
    tier: str, start: str, specification: dict, *, conditional_auxiliary: bool = False
) -> tuple[PublicFitRequest, dict]:
    """Known Arrhenius balances with train-fitted shared affine initializers."""
    if tier not in {"easy", "hard"} or start not in STARTS:
        raise ValueError("expected easy/hard and a frozen diagnostic start")
    if conditional_auxiliary and tier != "easy":
        raise ValueError("conditional control requires the easy public auxiliaries")
    p, equilibrium = specification["parameters"], specification["equilibrium"]
    rate = p["k0"] * math.exp(-p["E_over_R"] / 350)
    truth = {
        "flow": p["flow_rate"],
        "activation": p["E_over_R"] / 350,
        "heat_rate": p["source_gain"] * rate,
        "exchange": p["exchange_rate"],
    }
    coupled = not conditional_auxiliary
    concentration = "c_internal" if tier == "easy" and coupled else "C"
    jacket = "j_internal" if tier == "easy" and coupled else "Tj"
    states = [concentration, "T", jacket] if coupled else ["T"]
    shape = f"exp(-activation*(350/max(T,250)-1))*max({concentration},0)"
    equations = [
        {
            "state": "T",
            "rhs": f"flow*(Tf-T) + heat_rate*({shape}) - exchange*(T-{jacket})",
        }
    ]
    rules = {}
    if coupled:
        truth.update(
            rate=rate,
            jacket_flow=p["secondary_flow_rate"],
            jacket_exchange=p["secondary_exchange_rate"],
        )
        equations.extend(
            [
                {
                    "state": concentration,
                    "rhs": f"flow*(Cf-{concentration}) - rate*({shape})",
                },
                {
                    "state": jacket,
                    "rhs": f"jacket_flow*(Tjf-{jacket}) + jacket_exchange*(T-{jacket})",
                },
            ]
        )
    if tier == "easy" and coupled:
        rules = {
            state: {"initial": {"mode": "map", "expression": channel, "parameters": []}}
            for state, channel in ((concentration, "C"), (jacket, "Tj"))
        }
    elif tier == "hard":
        # Only training initial designs inform this evaluator-only witness.
        # The single training offset is (dC,dT,dTj)=(.08,5,3).
        # No claim is made for unseen independently varied hidden initials.
        for state, slope in (("C", 0.08 / 5), ("Tj", 3 / 5)):
            intercept = equilibrium[state] - slope * (equilibrium["T"] - 350)
            rules[state] = {
                "initial": {
                    "mode": "map",
                    "expression": "a+b*(T-350)",
                    "parameters": [
                        {
                            "name": "a",
                            "role": "coefficient",
                            "guess": intercept
                            if start == "reference_near"
                            else (0.25 if state == "C" else 345),
                        },
                        {
                            "name": "b",
                            "role": "coefficient",
                            "guess": slope if start == "reference_near" else 0,
                        },
                    ],
                }
            }
    dynamic_truth = dict(truth)
    generic = {
        "flow": 0.5,
        "activation": 15,
        "heat_rate": 50,
        "exchange": 1,
        "rate": 0.5,
        "jacket_flow": 0.5,
        "jacket_exchange": 0.5,
    }
    representation = "coupled" if coupled else "conditional"
    candidate = {
        "candidate_id": f"cstr_reference_{tier}_{representation}",
        "parent_candidate_id": None,
        "states": [
            {"name": n, "kind": "observed" if n == "T" else "latent"} for n in states
        ],
        "state_equations": equations,
        "observation_mappings": [{"channel": "T", "expression": "T"}],
        "parameters": [
            {"name": n, "scope": "global", "role": "nonnegative_coefficient"}
            for n in dynamic_truth
        ],
        "initial_conditions": [
            {"state": n, "scope": "global", "fixed_value": 0} for n in states
        ],
    }
    request = PublicFitRequest.model_validate(
        {
            "base_candidate": candidate,
            "context": {
                "targets": ["T"],
                "auxiliaries": ["C", "Tj"] if tier == "easy" else [],
                "external_inputs": ["Cf", "Tf", "Tjf"],
            },
            "initialization_plan": {"rules": rules},
            "parameter_guesses": {
                n: 0.8 * v if start == "reference_near" else generic[n]
                for n, v in dynamic_truth.items()
            },
            "profile": "collocation-single-target-v2",
            "source": {
                "stage": "synthetic_control",
                "task_id": f"cstr_{tier}_{start}",
                "artifact_sha256": public.content_sha256(specification),
            },
        }
    )
    for state in rules if tier == "hard" else ():
        slope = 0.08 / 5 if state == "C" else 3 / 5
        truth[f"init_{state}_a"] = equilibrium[state] - slope * (equilibrium["T"] - 350)
        truth[f"init_{state}_b"] = slope
    return request, truth


def replay(
    request: PublicFitRequest, parameters: dict, train: PublicSplit, val: PublicSplit
) -> dict:
    """Bounded free rollout with two solvers and one training-derived scale."""
    model, _, _ = public._lower(request)
    if set(parameters) != set(model.parameter_names):
        raise ValueError("replay parameter vector is incomplete")
    scale = max(
        float(np.std(np.concatenate([r.targets["T"] for r in train.rows]))), 1e-12
    )
    deadline, rows = monotonic() + REPLAY_LIMIT_SECONDS, []
    for split in (train, val):
        for trajectory in public.unpack_split(split).trajectories:
            predictions, errors = [], []
            for method in ("Radau", "DOP853"):
                sim = simulate_trajectory(
                    model,
                    trajectory,
                    parameters,
                    {},
                    FitConfig(
                        integration_method=method,
                        relative_tolerance=1e-9,
                        absolute_tolerance=1e-11,
                    ),
                    deadline=deadline,
                    reset_observed_states=False,
                )
                if not sim.success:
                    errors.append(f"{method}: {sim.message}")
                else:
                    predictions.append(sim.predictions["T"])
            complete = len(predictions) == 2
            rows.append(
                {
                    "split": split.name,
                    "trajectory_id": trajectory.trajectory_id,
                    "samples": len(trajectory.time),
                    "complete": complete,
                    "errors": errors,
                    "nmse": float(
                        np.mean(
                            ((predictions[0] - trajectory.targets["T"]) / scale) ** 2
                        )
                    )
                    if complete
                    else None,
                    "solver_scaled_max_difference": float(
                        np.max(abs(predictions[0] - predictions[1])) / scale
                    )
                    if complete
                    else None,
                }
            )
    metrics = {}
    for split in ("train", "val"):
        selected = [r for r in rows if r["split"] == split]
        complete = all(r["complete"] for r in selected)
        metrics[split] = {
            "complete": complete,
            "nmse": sum(r["samples"] * r["nmse"] for r in selected)
            / sum(r["samples"] for r in selected)
            if complete
            else None,
        }
    return {
        "metrics": metrics,
        "rows": rows,
        "training_scale": scale,
        "complete": all(r["complete"] for r in rows),
        "test_data_opened": False,
        "validation_initials_fitted": False,
    }


def prepare(root: Path, public_root: Path, data_root: Path) -> dict:
    """Freeze four requests; keep all evaluator-only assistance outside public data."""
    release_root = public_root / "phase_b_continuous_inputs_v1"
    release = read_seal(release_root / "release_audit.json")
    specification = public._read(data_root / SPEC_PATH)
    if _digest(data_root / SPEC_PATH) != release["private_spec_sha256"][str(SPEC_PATH)]:
        raise ValueError("qualification reference differs from release")
    registry = BenchmarkRegistry(continuous_phase_b_specs())
    tasks = []
    with public._lock(root):
        plan = {
            "protocol": "phase-c-cstr-qualification-1",
            "release_sha256": public.content_sha256(release),
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "private_spec_sha256": public.content_sha256(specification),
            "reference_nmse_limit": REFERENCE_NMSE_LIMIT,
            "fit_nmse_target": FIT_NMSE_TARGET,
            "solver_agreement_limit": SOLVER_AGREEMENT_LIMIT,
            "assistance": (
                "known equations; reference_near uses 0.8 times true dynamic "
                "parameters and true shared initializer guesses; generic guesses "
                "are fixed"
            ),
            "scope": (
                "CSTR named easy/hard, train/validation only; "
                "no discovery or scientific uniqueness claim"
            ),
            "other_families": {
                "dalla_man": (
                    "pending exact public-interface transcription "
                    "of gastric bookkeeping"
                ),
                "alien_device": (
                    "pending qualification of public information "
                    "for hidden initial states"
                ),
            },
        }
        seal(root / "plan.json", plan)
        for tier in ("easy", "hard"):
            identifier = (
                "phase_b_cstr_controlled_reactor_mechanism_canonical_named_"
                f"{tier}_rates_v1"
            )
            entry = next(c for c in release["cells"] if c["benchmark_id"] == identifier)
            cell = release_root / identifier
            # Verify only development/prompt metadata here. Never read test.csv.
            for name in (
                "manifest.json",
                "proposer_prompt.txt",
                "judge_prompt.txt",
                "train.csv",
                "validation.csv",
            ):
                if _digest(cell / name) != entry["files"][name]:
                    raise ValueError(f"release development file changed: {cell / name}")
            dataset = BenchmarkLoader(registry).load_development(
                DataConfig(
                    root=public_root,
                    benchmark_id=identifier,
                    tier=tier,
                )
            )
            train, val = (
                public.pack_split(dataset.train),
                public.pack_split(dataset.validation),
            )
            for start in STARTS:
                task = f"cstr_{tier}_{start}"
                request, truth = cstr_request(tier, start, specification)
                directory = root / task
                seal(directory / "reference_parameters.json", truth)
                public.prepare_fit(request, train, val, directory / "fit")
                if tier == "easy" and start == STARTS[0]:
                    reduced, reduced_truth = cstr_request(
                        tier, start, specification, conditional_auxiliary=True
                    )
                    seal(
                        directory / "conditional_control.json",
                        {
                            "request": reduced.model_dump(mode="json"),
                            "parameters": reduced_truth,
                            "fit_performed": False,
                        },
                    )
                tasks.append({"task": task, "tier": tier, "start": start})
        seal(root / "tasks.json", {"tasks": tasks})
    return {"tasks": tasks, "live_llm_calls": 0, "test_data_opened": False}


def run_task(root: Path, index: int) -> dict:
    """Use the existing fit attempt ledger; interrupted fits get no fresh budget."""
    plan = read_seal(root / "plan.json")
    if (
        plan["runtime"] != public._runtime()
        or plan["source_sha256"] != public._source_identity()
    ):
        raise ValueError(
            "qualification runtime or source changed; use the frozen environment"
        )
    tasks = read_seal(root / "tasks.json")["tasks"]
    if index not in range(len(tasks)):
        raise ValueError("task index outside frozen roster")
    task = tasks[index]
    directory = root / task["task"]
    with public._lock(directory):
        public.inspect_fit(directory / "fit")  # verifies current source/runtime/data
        frozen = public._read(directory / "fit/freeze.json")
        request = PublicFitRequest.model_validate(frozen["request"])
        train = PublicSplit.model_validate(frozen["training"])
        val = PublicSplit.model_validate(frozen["validation"])
        control_replay = None
        if (directory / "conditional_control.json").exists():
            control = read_seal(directory / "conditional_control.json")
            path = directory / "conditional_replay.json"
            if not path.exists():
                seal(
                    path,
                    replay(
                        PublicFitRequest.model_validate(control["request"]),
                        control["parameters"],
                        train,
                        val,
                    ),
                )
            control_replay = read_seal(path)
        reference_path = directory / "reference_replay.json"
        if not reference_path.exists():
            seal(
                reference_path,
                replay(
                    request,
                    read_seal(directory / "reference_parameters.json"),
                    train,
                    val,
                ),
            )
        reference = read_seal(reference_path)
        attainable = (
            reference["complete"]
            and all(
                m["nmse"] <= REFERENCE_NMSE_LIMIT for m in reference["metrics"].values()
            )
            and all(
                r["solver_scaled_max_difference"] <= SOLVER_AGREEMENT_LIMIT
                for r in reference["rows"]
            )
        )
        if not attainable:
            result = {
                **task,
                "status": "reference_replay_failed",
                "reference": reference,
                "fit_performed": False,
            }
        else:
            fit = public.execute_fit(directory / "fit")
            result = {
                **task,
                "status": "complete",
                "reference": reference,
                "fit_performed": True,
                "fit": fit.model_dump(mode="json"),
            }
            if fit.status == "complete":
                path = directory / "fitted_replay.json"
                if not path.exists():
                    seal(path, replay(request, dict(fit.parameters), train, val))
                result["fitted_replay"] = read_seal(path)
        result["conditional_auxiliary_control"] = control_replay
        seal(directory / "qualification.json", result)
        return result


def report(root: Path) -> dict:
    """Report every predetermined start, retaining failures and missing records."""
    plan = read_seal(root / "plan.json")
    rows = []
    for task in read_seal(root / "tasks.json")["tasks"]:
        path = root / task["task"] / "qualification.json"
        value = read_seal(path) if path.exists() else {**task, "status": "missing"}
        fit = value.get("fit", {})
        checked = value.get("fitted_replay", {})
        target_met = (
            checked.get("complete", False)
            and all(m["nmse"] <= FIT_NMSE_TARGET for m in checked["metrics"].values())
            and all(
                r["solver_scaled_max_difference"] <= SOLVER_AGREEMENT_LIMIT
                for r in checked["rows"]
            )
        )
        rows.append(
            {
                **task,
                "status": value["status"],
                "fit_status": fit.get("status"),
                "low_error_witness_obtained": target_met,
                "reference": value.get("reference", {}).get("metrics"),
                "fitted_replay": value.get("fitted_replay", {}).get("metrics"),
                "conditional_auxiliary_control": (
                    value["conditional_auxiliary_control"]["metrics"]
                    if value.get("conditional_auxiliary_control")
                    else None
                ),
                "training": fit.get("training"),
                "validation": fit.get("validation"),
                "budget_exhausted": fit.get("budget_exhausted"),
                "native_optimizer_converged": fit.get("native_optimizer_converged"),
            }
        )
    summary = {
        "protocol": plan["protocol"],
        "status": "complete"
        if all(r["status"] != "missing" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "fit_nmse_target": FIT_NMSE_TARGET,
        "rows": rows,
        "other_families": plan["other_families"],
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": (
            "Assisted CSTR fitting diagnostic only. Both tiers use coupled balances. "
            "Easy also retains a reduced auxiliary-driven replay control. "
            "Hard initial maps cover the declared development "
            "designs, not arbitrary independent hidden initials. No proof of unique "
            "parameter recovery or whole-suite scientific validity."
        ),
    }
    with public._lock(root):
        public._write(root / "summary.json", summary)
    return summary
