"""Opt-in Phase C strategy qualification, isolated from model construction.

Reference parameters and validation enter qualification/evaluation only; worker
payloads contain the training split, generic starts and public model restrictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.fitting_qualification_inputs import experiment_request
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import nonlinear_shape_case as shape
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.fitting.identifiable_campaign import sensitivity_audit
from autoformalism.fitting.matching_probe import observed_node_guess
from autoformalism.fitting.qualification import _report_lock, replay
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.transcription_fit import ARMS, StrategyPolicy, run
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-fitting-strategies-1"


class CampaignConfig(StrictSchema):
    """Frozen roster and thresholds; all starts are reported rather than selected."""

    starts: int = Field(default=3, ge=1, le=8)
    include_cstr: bool = True
    strategy: StrategyPolicy = StrategyPolicy()
    replay_seconds: float = Field(default=240, gt=0, le=1200)
    control_nmse: float = Field(default=1e-6, gt=0)
    cstr_nmse: float = Field(default=0.01, gt=0)
    parameter_relative: float = Field(default=0.01, gt=0)
    latent_nmse: float = Field(default=1e-4, gt=0)


def case_request(name, case, seed):
    """Choose a correct equation template and generic starts without truth draws."""
    if name.startswith("cstr"):
        return experiment_request(case, "joint", seed, True)[0]
    return shape.request(seed) if name == "shape" else controls.request(name, seed)


def prepare(
    root: Path, config: CampaignConfig, inputs_path: Path | None = None
) -> dict:
    """Freeze paired physical starts, immutable input bytes and numerical runtime."""
    source = None
    if config.include_cstr:
        if inputs_path is None:
            raise ValueError("CSTR requires sealed milestone-1 inputs")
        source = read_seal(inputs_path)
        if source["protocol"] != "phase-c-fitting-inputs-1" or source.get(
            "test_data_opened"
        ):
            raise ValueError("wrong input release/protocol")
    source_digest = public.content_sha256(source) if source else None
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, _ = verify(root)
            if (
                plan["config"] != config.model_dump(mode="json")
                or plan["source_inputs_sha256"] != source_digest
            ):
                raise ValueError("configuration/source inputs differ")
            return {
                "identity": public.content_sha256(plan),
                "tasks": len(plan["tasks"]),
                "gpus": 0,
            }
        inputs = controls.make_inputs()
        inputs["protocol"] = "phase-c-fitting-strategy-inputs-1"
        inputs["cases"]["shape"] = shape.make_case(inputs["cases"]["nonlinear"])
        if source:
            inputs["cases"].update(
                {n: source["cases"][n] for n in ("cstr_easy", "cstr_hard")}
            )
        commons, tasks = {}, []
        for name, case in inputs["cases"].items():
            train = public.unpack_split(PublicSplit.model_validate(case["training"]))
            for seed in range(config.starts):
                req = case_request(name, case, seed)
                public._check_data(
                    req,
                    PublicSplit.model_validate(case["training"]),
                    PublicSplit.model_validate(case["validation"]),
                )
                model, start, _ = public._lower(req)
                system = SymbolicODE(model, allow_piecewise=True)
                vector = np.array([start[n] for n in system.names])
                key = f"{name}_s{seed}"
                commons[key] = {
                    "case": name,
                    "seed": seed,
                    "request": req.model_dump(mode="json"),
                    "start": start,
                    "coordinates": training_coordinates(
                        model,
                        train,
                        start,
                        state_channels=case.get("state_channel_proxies"),
                    ).model_dump(mode="json"),
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
            "source_inputs_sha256": source_digest,
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


def verify(root: Path, *, runtime: bool = True):
    """Verify all immutable assets before execution; reporting tolerates new code."""
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


def qualify(root: Path) -> dict:
    """Require attainable reference rollouts and full local sensitivity rank.

    CSTR is a stress test with no global uniqueness claim. Synthetic controls also
    require their analytic sufficient-condition witness. No fitted seed is shared.
    """
    plan, inputs = verify(root)
    config = CampaignConfig.model_validate(plan["config"])
    with public._lock(root / "qualification"):
        records = {}
        for name, case in inputs["cases"].items():
            path = root / "qualification" / f"{name}.json"
            if not path.exists():
                train, val = (
                    PublicSplit.model_validate(case[k])
                    for k in ("training", "validation")
                )
                req = case_request(name, case, 0)
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
                        "identifiability": case.get(
                            "identifiability",
                            {"scope": "CSTR global uniqueness unproven"},
                        ),
                    },
                )
            records[name] = read_seal(path)
        passed = all(
            r["replay"]["complete"]
            and max(r["replay"]["metrics"].values()) <= 1e-6
            and r["replay"]["maximum_solver_difference"] <= 1e-4
            and r["sensitivity"].get("full_local_rank")
            and (n.startswith("cstr") or r["identifiability"].get("passed"))
            for n, r in records.items()
        )
        result = {
            "passed": passed,
            "cases": records,
            "plan_sha256": public.content_sha256(plan),
        }
        seal(root / "qualification/result.json", result)
    if not passed:
        raise ValueError("reference/excitation gate failed; inspect qualification")
    return {"passed": passed, "cases": len(records), "shared_fit_given_to_arms": False}


def worker_payload(plan, inputs, task):
    """Deliberately omit validation, private reference and excitation derivatives."""
    common = plan["commons"][task["common"]]
    return {k: common[k] for k in ("request", "coordinates", "nodes")} | {
        "training": inputs["cases"][common["case"]]["training"],
        "arm": task["arm"],
        "policy": plan["config"]["strategy"],
    }


def recovery_metrics(case, parameters, *, nonlinear_shape: bool):
    """Independent evaluator computes latent recovery after endpoint selection."""
    reference = shape.reference if nonlinear_shape else controls.reference
    truth = case["reference_parameters"]
    relative = {
        n: abs(parameters[n] - v) / max(abs(v), 1e-12) for n, v in truth.items()
    }
    scale = max(
        float(
            np.std(
                np.concatenate(
                    [reference(r, truth)[:, 1] for r in case["training"]["rows"]]
                )
            )
        ),
        1e-12,
    )
    latent = {
        split: float(
            np.mean(
                np.concatenate(
                    [
                        (reference(r, parameters)[:, 1] - reference(r, truth)[:, 1])
                        / scale
                        for r in case[split]["rows"]
                    ]
                )
                ** 2
            )
        )
        for split in ("training", "validation")
    }
    return {
        "maximum_parameter_relative_error": max(relative.values()),
        "parameter_relative_errors": relative,
        "latent_nmse": latent,
    }


def run_task(root: Path, index: int) -> dict:
    """Reuse completed results; an interrupted native budget is never restarted."""
    plan, inputs = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("invalid task index")
    gate = read_seal(root / "qualification/result.json")
    if not gate["passed"] or gate["plan_sha256"] != public.content_sha256(plan):
        raise ValueError("qualification not valid for this plan")
    config = CampaignConfig.model_validate(plan["config"])
    task = plan["tasks"][index]
    common = plan["commons"][task["common"]]
    case = inputs["cases"][common["case"]]
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        path, backend_path = folder / "result.json", folder / "backend.json"
        if path.exists():
            return read_seal(path)
        if not backend_path.exists():
            if (folder / "started.json").exists():
                result = {**task, "status": "interrupted", "budget_restarted": False}
                seal(path, result)
                return result
            seal(
                folder / "started.json",
                {
                    "plan_sha256": public.content_sha256(plan),
                    "common_sha256": public.content_sha256(common),
                },
            )
            backend = run(worker_payload(plan, inputs, task), folder / "fit")
            seal(backend_path, backend)
        backend = read_seal(backend_path)
        if backend.get("worker_payload_sha256") != public.content_sha256(
            worker_payload(plan, inputs, task)
        ):
            raise ValueError("backend worker payload differs")
        parameters, checked, recovery = backend.get("parameters"), None, None
        is_control = not common["case"].startswith("cstr")
        if parameters:
            replay_path = folder / "replay.json"
            if not replay_path.exists():
                checked = replay(
                    PublicFitRequest.model_validate(common["request"]),
                    parameters,
                    PublicSplit.model_validate(case["training"]),
                    PublicSplit.model_validate(case["validation"]),
                    config.replay_seconds,
                )
                seal(replay_path, checked)
            checked = read_seal(replay_path)
            if is_control and checked["complete"]:
                try:
                    recovery = recovery_metrics(
                        case, parameters, nonlinear_shape=common["case"] == "shape"
                    )
                except (ValueError, RuntimeError, ArithmeticError) as error:
                    recovery = {"error": str(error)[-1000:]}
        accurate = bool(
            checked
            and checked["complete"]
            and checked["maximum_solver_difference"] <= 1e-4
            and max(checked["metrics"].values())
            <= (config.control_nmse if is_control else config.cstr_nmse)
        )
        recovered = (
            bool(
                accurate
                and recovery
                and "latent_nmse" in recovery
                and max(recovery["latent_nmse"].values()) <= config.latent_nmse
                and recovery["maximum_parameter_relative_error"]
                <= config.parameter_relative
            )
            if is_control
            else None
        )
        result = {
            **task,
            "case": common["case"],
            "seed": common["seed"],
            "status": "complete"
            if checked and checked["complete"]
            else "no_complete_replay",
            "accuracy_passed": accurate,
            "recovery_passed": recovered,
            "recovery": recovery,
            "training_nmse": checked["metrics"]["train"] if checked else None,
            "validation_nmse": checked["metrics"]["val"] if checked else None,
            "parameters": parameters,
            "stop_reason": backend["stop_reason"],
            "seconds": backend["total_seconds"],
            "budget_exhausted": backend["budget_exhausted"],
            "actual_residual_calls": backend.get("actual_residual_calls"),
            "mesh_stage_timeouts": backend.get("mesh_stage_timeouts"),
            "common_sha256": public.content_sha256(common),
            "backend_sha256": public.content_sha256(backend),
            "validation_used_for_fitting": False,
            "global_identifiability_claimed": False,
        }
        seal(path, result)
    return result


def report(root: Path) -> dict:
    """All endpoints, separated by case, with explicit unavailable/interrupted rows."""
    plan, _ = verify(root, runtime=False)
    with _report_lock(root):
        rows, groups = [], defaultdict(list)
        for task in plan["tasks"]:
            common = plan["commons"][task["common"]]
            path = root / "results" / task["task_id"] / "result.json"
            row = read_seal(path) if path.exists() else {**task, "status": "missing"}
            if row.get("backend_sha256") and row[
                "backend_sha256"
            ] != public.content_sha256(read_seal(path.parent / "backend.json")):
                raise ValueError("backend digest differs")
            row = {"case": common["case"], "seed": common["seed"], **row}
            rows.append(row)
            groups[(row["case"], row["arm"])].append(row)
        summary = {
            "protocol": PROTOCOL,
            "plan_sha256": public.content_sha256(plan),
            "status": "complete"
            if all(r["status"] != "missing" for r in rows)
            else "incomplete",
            "expected": len(rows),
            "recorded": sum(r["status"] != "missing" for r in rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "rows": rows,
            "groups": [
                {
                    "case": name,
                    "arm": arm,
                    "expected": len(items),
                    "accuracy_passes": sum(
                        bool(r.get("accuracy_passed")) for r in items
                    ),
                    "recovery_passes": sum(
                        bool(r.get("recovery_passed")) for r in items
                    )
                    if not name.startswith("cstr")
                    else None,
                    "seconds": sum(r.get("seconds", 0) for r in items),
                }
                for (name, arm), items in sorted(groups.items())
            ],
            "test_data_opened": False,
            "live_llm_calls": 0,
            "limitation": (
                "Known equations, noiseless controls, finite budgets. "
                "CSTR global identifiability unproven. Strategy bundles "
                "differ in discretization and solver; no production promotion."
            ),
        }
        public._write(root / "summary.json", summary)
    return summary
