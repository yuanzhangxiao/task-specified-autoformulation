"""Identifiable latent controls with shared initialization and isolated evaluation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import identifiable_cases as cases
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import NumericalCoordinates, training_coordinates
from autoformalism.fitting.feasibility import restart_points
from autoformalism.fitting.identifiable_refinement import ARMS, RefinementPolicy, refine
from autoformalism.fitting.matching_probe import (
    bounded_latent_start,
    observed_node_guess,
)
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.qualification import _report_lock, replay
from autoformalism.fitting.sensitivity_probe import SymbolicODE, SymbolicOracle
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-identifiable-fitting-1"
SETTINGS = FitConfig(relative_tolerance=1e-8, absolute_tolerance=1e-10)


class CampaignConfig(StrictSchema):
    """Frozen study, including endpoint recovery thresholds independent of stopping."""

    starts: int = Field(default=3, ge=1, le=8)
    initializer_seconds: float = Field(default=60, gt=0, le=600)
    refinement: RefinementPolicy = RefinementPolicy()
    replay_seconds: float = Field(default=120, gt=0, le=600)
    output_nmse_limit: float = Field(default=1e-6, gt=0)
    parameter_relative_limit: float = Field(default=0.01, gt=0)
    latent_nmse_limit: float = Field(default=1e-4, gt=0)


def prepare(root: Path, config: CampaignConfig) -> dict:
    """Freeze synthetic inputs, starts and physical node guesses before any fit."""
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, _ = verify(root)
            if plan["config"] != config.model_dump(mode="json"):
                raise ValueError("configuration differs from frozen campaign")
            return {
                "identity": public.content_sha256(plan),
                "tasks": len(plan["tasks"]),
                "gpus": 0,
            }
        inputs = cases.make_inputs()
        commons, tasks = {}, []
        for case_name, case in inputs["cases"].items():
            train = public.unpack_split(PublicSplit.model_validate(case["training"]))
            for seed in range(config.starts):
                request = cases.request(case_name, seed)
                model, start, _ = public._lower(request)
                system = SymbolicODE(model)
                vector = np.array([start[n] for n in system.names])
                key = f"{case_name}_s{seed}"
                commons[key] = {
                    "case": case_name,
                    "seed": seed,
                    "request": request.model_dump(mode="json"),
                    "start": start,
                    "coordinates": training_coordinates(model, train, start).model_dump(
                        mode="json"
                    ),
                    "nodes": {
                        r.trajectory_id: observed_node_guess(system, r, vector).tolist()
                        for r in train.trajectories
                    },
                }
                for arm in ARMS:
                    tasks.append({"task_id": f"{key}_{arm}", "common": key, "arm": arm})
        plan = {
            "protocol": PROTOCOL,
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "inputs_sha256": public.content_sha256(inputs),
            "config": config.model_dump(mode="json"),
            "commons": commons,
            "tasks": tasks,
            "test_data_opened": False,
            "live_llm_calls": 0,
            "gpus": 0,
        }
        seal(root / "inputs.json", inputs)
        seal(root / "plan.json", plan)
    report(root)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks), "gpus": 0}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Reject artifact, source and environment drift before granting numerical work."""
    plan, inputs = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] != PROTOCOL or plan["inputs_sha256"] != public.content_sha256(
        inputs
    ):
        raise ValueError("campaign identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("source/runtime differs from frozen campaign")
    return plan, inputs


def scale_for(train) -> dict:
    """Use only observed training samples to normalize the rollout residual."""
    return {
        c: max(
            float(np.std(np.concatenate([r.targets[c] for r in train.trajectories]))),
            1e-12,
        )
        for c in train.trajectories[0].targets
    }


def sensitivity_audit(
    request, training: PublicSplit, parameters: dict, directory: Path
) -> dict:
    """Local output sensitivity evidence; not a structural-identifiability proof."""
    model, _, _ = public._lower(request)
    system = SymbolicODE(model, allow_piecewise=True)
    train = public.unpack_split(training)
    oracle = SymbolicOracle(
        system,
        train,
        scale_for(train),
        SETTINGS,
        directory,
        monotonic() + 120,
        sensitivities=True,
    )
    residual = oracle(oracle.vector(parameters))
    matrix = oracle.last_jac
    if oracle.valid_calls == 0 or matrix is None or not np.isfinite(matrix).all():
        return {"available": False, "structural_identifiability": "not_certified"}
    norms = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix / np.maximum(norms, 1e-30), compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
    return {
        "available": True,
        "parameter_names": list(system.names),
        "column_norms": norms.tolist(),
        "column_normalized_singular_values": singular.tolist(),
        "singular_ratio": ratio,
        "full_local_rank": bool(np.all(norms > 0) and ratio > 1e-8),
        "training_nmse": float(np.mean(residual**2)),
        "piecewise_derivatives": system.has_piecewise,
        "structural_identifiability": "not_certified",
        "limitation": (
            "One parameter point and input design. Full numerical rank "
            "does not prove global uniqueness or noise robustness."
        ),
    }


def qualify(root: Path) -> dict:
    """Verify the reference and create one shared collocation pool per start."""
    plan, inputs = verify(root)
    config = CampaignConfig.model_validate(plan["config"])
    evidence = {}
    with public._lock(root / "qualification"):
        for name, case in inputs["cases"].items():
            path = root / "qualification" / f"{name}.json"
            if not path.exists():
                train, val = (
                    PublicSplit.model_validate(case[k])
                    for k in ("training", "validation")
                )
                req = cases.request(name, 0)
                checked = replay(
                    req, case["reference_parameters"], train, val, config.replay_seconds
                )
                audit = sensitivity_audit(
                    req,
                    train,
                    case["reference_parameters"],
                    root / "qualification" / name,
                )
                seal(
                    path,
                    {
                        "replay": checked,
                        "sensitivity": audit,
                        "identifiability": case["identifiability"],
                    },
                )
            evidence[name] = read_seal(path)
        passed = all(
            r["replay"]["complete"]
            and max(r["replay"]["metrics"].values()) <= 1e-8
            and r["replay"]["maximum_solver_difference"] <= 1e-4
            and r["identifiability"]["passed"]
            and r["sensitivity"].get("full_local_rank")
            for r in evidence.values()
        )
        seal(
            root / "qualification" / "result.json",
            {
                "passed": passed,
                "cases": evidence,
                "plan_sha256": public.content_sha256(plan),
            },
        )
    if not passed:
        raise ValueError("reference/identifiability gate failed; inspect qualification")
    for key, common in plan["commons"].items():
        directory = root / "common" / key
        with public._lock(directory):
            path = directory / "result.json"
            if path.exists():
                continue
            if (directory / "started.json").exists():
                raise ValueError(
                    f"interrupted shared initializer {key}; no automatic budget reset"
                )
            request = PublicFitRequest.model_validate(common["request"])
            model, start, _ = public._lower(request)
            train = public.unpack_split(
                PublicSplit.model_validate(inputs["cases"][common["case"]]["training"])
            )
            system = SymbolicODE(model)
            oracle = SymbolicOracle(
                system,
                train,
                scale_for(train),
                SETTINGS,
                directory / "layout",
                None,
                sensitivities=False,
            )
            seal(
                directory / "started.json",
                {"common_sha256": public.content_sha256(common)},
            )
            initializer = bounded_latent_start(
                system,
                training=train,
                lower=oracle.lower,
                upper=oracle.upper,
                start=oracle.vector(start),
                scale=scale_for(train),
                settings=SETTINGS,
                method="collocation_init",
                seconds=config.initializer_seconds,
                directory=directory / "collocation",
                node_start="rollout_or_observed",
                record_progress=True,
                numerical_coordinates=NumericalCoordinates.model_validate(
                    common["coordinates"]
                ),
                frozen_nodes=common["nodes"],
                checkpoint_pool_capacity=8,
            )
            pool = (initializer.get("progress") or {}).get("checkpoint_pool", [])
            points = restart_points(
                initializer,
                start,
                [
                    {"source": f"pool_{i}", "parameters": p["parameters"]}
                    for i, p in enumerate(pool)
                ],
            )
            seal(
                path,
                {
                    "common_sha256": public.content_sha256(common),
                    "node_sha256": public.content_sha256(common["nodes"]),
                    "initializer": initializer,
                    "points": points,
                },
            )
    return {
        "passed": True,
        "cases": len(evidence),
        "shared_initializers": len(plan["commons"]),
    }


def recovery_metrics(case: dict, parameters: dict) -> dict:
    """Private latent/parameter scoring after fitting; never selects a checkpoint."""
    truth = case["reference_parameters"]
    relative = {
        n: abs(parameters[n] - v) / max(abs(v), 1e-12) for n, v in truth.items()
    }
    train_truth = [cases.reference(r, truth)[:, 1] for r in case["training"]["rows"]]
    scale = max(float(np.std(np.concatenate(train_truth))), 1e-12)
    latent = {}
    for split in ("training", "validation"):
        residuals = [
            (cases.reference(r, parameters)[:, 1] - cases.reference(r, truth)[:, 1])
            / scale
            for r in case[split]["rows"]
        ]
        latent[split] = float(np.mean(np.concatenate(residuals) ** 2))
    return {
        "parameter_relative_errors": relative,
        "maximum_parameter_relative_error": max(relative.values()),
        "latent_nmse": latent,
        "evaluator_only": True,
    }


def run_task(root: Path, index: int) -> dict:
    """Resume completed artifacts, never restart an interrupted optimizer budget."""
    plan, inputs = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    config = CampaignConfig.model_validate(plan["config"])
    task = plan["tasks"][index]
    common = plan["commons"][task["common"]]
    shared = read_seal(root / "common" / task["common"] / "result.json")
    qualification = read_seal(root / "qualification" / "result.json")
    if (
        shared["common_sha256"] != public.content_sha256(common)
        or shared["node_sha256"] != public.content_sha256(common["nodes"])
        or not qualification["passed"]
        or qualification["plan_sha256"] != public.content_sha256(plan)
    ):
        raise ValueError("shared initializer/reference identity differs")
    directory = root / "results" / task["task_id"]
    with public._lock(directory):
        result_path, backend_path = (
            directory / "result.json",
            directory / "backend.json",
        )
        if result_path.exists():
            return read_seal(result_path)
        if (directory / "started.json").exists() and not backend_path.exists():
            result = {
                **task,
                "status": "interrupted",
                "budget_restarted": False,
                "recovery_passed": False,
            }
            seal(result_path, result)
            return result
        request = PublicFitRequest.model_validate(common["request"])
        case = inputs["cases"][common["case"]]
        train, val = (
            PublicSplit.model_validate(case[k]) for k in ("training", "validation")
        )
        if not backend_path.exists():
            seal(
                directory / "started.json",
                {
                    "plan_sha256": public.content_sha256(plan),
                    "shared_sha256": public.content_sha256(shared),
                },
            )
            model, _, _ = public._lower(request)
            dataset = public.unpack_split(train)
            try:
                backend = refine(
                    SymbolicODE(model),
                    dataset,
                    scale_for(dataset),
                    SETTINGS,
                    NumericalCoordinates.model_validate(common["coordinates"]),
                    shared["points"],
                    config.refinement,
                    task["arm"],
                    directory / "fit",
                )
            except (RuntimeError, ValueError, ArithmeticError) as error:
                backend = {
                    "parameters": None,
                    "stop_reason": "execution_failed",
                    "error": str(error)[-2000:],
                }
            seal(backend_path, backend)
        backend = read_seal(backend_path)
        parameters = backend.get("parameters")
        checked = recovery = None
        if parameters:
            path = directory / "replay.json"
            if not path.exists():
                seal(
                    path, replay(request, parameters, train, val, config.replay_seconds)
                )
            checked = read_seal(path)
            if checked["complete"]:
                try:
                    recovery = recovery_metrics(case, parameters)
                except (ValueError, RuntimeError, ArithmeticError) as error:
                    recovery = {"error": str(error)[-1000:]}
        accuracy = bool(
            checked
            and checked["complete"]
            and max(checked["metrics"].values()) <= config.output_nmse_limit
            and checked["maximum_solver_difference"] <= 1e-4
        )
        passed = bool(
            accuracy
            and recovery
            and "latent_nmse" in recovery
            and recovery["maximum_parameter_relative_error"]
            <= config.parameter_relative_limit
            and max(recovery["latent_nmse"].values()) <= config.latent_nmse_limit
        )
        result = {
            **task,
            "case": common["case"],
            "seed": common["seed"],
            "status": "complete"
            if checked and checked["complete"]
            else "no_complete_replay",
            "accuracy_passed": accuracy,
            "recovery_passed": passed,
            "recovery": recovery,
            "training_nmse": checked["metrics"]["train"] if checked else None,
            "validation_nmse": checked["metrics"]["val"] if checked else None,
            "parameters": parameters,
            "stop_reason": backend["stop_reason"],
            "actual_residual_calls": backend.get("actual_residual_calls"),
            "refinement_seconds": backend.get("seconds"),
            "budget_exhausted": backend.get("budget_exhausted"),
            "shared_sha256": public.content_sha256(shared),
            "backend_sha256": public.content_sha256(backend),
            "initializer_seconds_charged_per_arm": config.initializer_seconds,
            "validation_used_for_fitting": False,
        }
        seal(result_path, result)
        return result


def report(root: Path) -> dict:
    """Include every planned endpoint; execution completeness is not recovery."""
    plan, _ = verify(root, runtime=False)
    with _report_lock(root):
        rows = []
        for task in plan["tasks"]:
            path = root / "results" / task["task_id"] / "result.json"
            row = read_seal(path) if path.exists() else {**task, "status": "missing"}
            if row.get("backend_sha256") and row[
                "backend_sha256"
            ] != public.content_sha256(read_seal(path.parent / "backend.json")):
                raise ValueError("backend evidence differs from completed result")
            rows.append(row)
        summary = {
            "protocol": PROTOCOL,
            "status": "complete"
            if all(r["status"] != "missing" for r in rows)
            else "incomplete",
            "expected": len(rows),
            "recorded": sum(r["status"] != "missing" for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "recovery_passes": {
                a: sum(bool(r.get("recovery_passed")) for r in rows if r["arm"] == a)
                for a in ARMS
            },
            "rows": rows,
            "paired_comparisons": [
                {
                    "common": key,
                    "arms": {
                        row["arm"]: {
                            field: row.get(field)
                            for field in (
                                "status",
                                "recovery_passed",
                                "training_nmse",
                                "validation_nmse",
                                "actual_residual_calls",
                                "refinement_seconds",
                                "stop_reason",
                            )
                        }
                        for row in rows
                        if row["common"] == key
                    },
                }
                for key in plan["commons"]
            ],
            "test_data_opened": False,
            "live_llm_calls": 0,
            "limitation": (
                "Three assisted, identifiable synthetic controls, repeated starts; "
                "not arbitrary-model identifiability or benchmark success. "
                "No production default changes."
            ),
        }
        public._write(root / "summary.json", summary)
    return summary
