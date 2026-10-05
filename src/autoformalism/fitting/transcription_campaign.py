"""Opt-in Phase C strategy qualification, isolated from model construction.

Reference parameters and validation enter qualification/evaluation only; worker
payloads contain the training split, generic starts and public model restrictions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks import challenging_fitting_inputs as challenging
from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.benchmarks.fitting_qualification_inputs import experiment_request
from autoformalism.fitting import checkpoint_diagnostic as checkpoint
from autoformalism.fitting import identifiable_cases as controls
from autoformalism.fitting import nonlinear_shape_case as shape
from autoformalism.fitting import numerical_diagnostic as numerical
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting import screening_diagnostic as screening
from autoformalism.fitting.coordinates import training_coordinates
from autoformalism.fitting.identifiable_campaign import sensitivity_audit
from autoformalism.fitting.matching_probe import observed_node_guess
from autoformalism.fitting.qualification import _report_lock, replay
from autoformalism.fitting.reuse_diagnostic import ARMS as DIAGNOSTIC_ARMS
from autoformalism.fitting.reuse_diagnostic import ReuseDiagnosticPolicy
from autoformalism.fitting.sensitivity_probe import SymbolicODE
from autoformalism.fitting.transcription_fit import ARMS, REUSE_ARM, StrategyPolicy, run
from autoformalism.schemas.base import StrictSchema
from autoformalism.schemas.public_fitting import PublicFitRequest, PublicSplit

PROTOCOL = "phase-c-fitting-strategies-1"
FOLLOWUP_PROTOCOL = "phase-c-fitting-budget-reuse-1"
DIAGNOSTIC_PROTOCOL = "phase-c-fitting-reuse-diagnostic-1"
CHALLENGING_PROTOCOL = "phase-c-fitting-challenging-1"


class CampaignConfig(StrictSchema):
    """Frozen roster and thresholds; all starts are reported rather than selected."""

    starts: int = Field(default=3, ge=1, le=8)
    include_cstr: bool = True
    include_reuse: bool = False
    reuse_diagnostic: ReuseDiagnosticPolicy | None = None
    include_challenging: bool = False
    challenging_reuse: ReuseDiagnosticPolicy | None = None
    numerical_diagnostic: numerical.NumericalDiagnosticPolicy | None = None
    checkpoint_diagnostic: bool = False
    screening_diagnostic: screening.DiagnosticPolicy | None = None
    cases: (
        tuple[
            Literal[
                "linear",
                "nonlinear",
                "fast_slow",
                "shape",
                "cstr_easy",
                "cstr_hard",
                "basin_coupled",
                "alien_hard",
            ],
            ...,
        ]
        | None
    ) = None
    strategy: StrategyPolicy = StrategyPolicy()
    replay_seconds: float = Field(default=240, gt=0, le=1200)
    control_nmse: float = Field(default=1e-6, gt=0)
    cstr_nmse: float = Field(default=0.01, gt=0)
    parameter_relative: float = Field(default=0.01, gt=0)
    latent_nmse: float = Field(default=1e-4, gt=0)
    initial_absolute: float = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def validate_roster(self):
        """Reject empty/duplicate selections and unavailable CSTR inputs."""
        if self.screening_diagnostic is not None and (
            self.numerical_diagnostic is None or self.checkpoint_diagnostic
        ):
            raise ValueError("screening diagnostic requires separate numerical arms")
        if self.checkpoint_diagnostic and self.numerical_diagnostic is None:
            raise ValueError("checkpoint diagnostic requires fixed numerical arms")
        if self.include_reuse and self.reuse_diagnostic is not None:
            raise ValueError("historical reuse and isolated diagnostic are separate")
        if self.include_challenging and (
            self.include_cstr or self.include_reuse or self.reuse_diagnostic is not None
        ):
            raise ValueError("challenging campaign requires its separate input roster")
        if self.challenging_reuse is not None and not self.include_challenging:
            raise ValueError("challenging reuse requires challenging inputs")
        if self.numerical_diagnostic is not None:
            if self.include_reuse or self.reuse_diagnostic or self.challenging_reuse:
                raise ValueError("numerical diagnostic is separate from reuse arms")
            if (
                self.numerical_diagnostic.first_rollout_calls + 5
                > self.strategy.maximum_rollout_calls
            ):
                raise ValueError("reserve rollout calls for the second phase")
        diagnostic = self.reuse_diagnostic or self.challenging_reuse
        if diagnostic is not None and (
            2 * diagnostic.native_seconds + 7 * diagnostic.point_seconds
            >= self.strategy.seconds
        ):
            raise ValueError(
                "diagnostic requires time for two solves, screens and builds"
            )
        if self.cases is not None:
            if not self.cases or len(set(self.cases)) != len(self.cases):
                raise ValueError("cases must be nonempty and distinct")
            if not self.include_cstr and any(n.startswith("cstr") for n in self.cases):
                raise ValueError("CSTR case selection requires include_cstr")
            if any(
                (n in challenging.CELLS) != self.include_challenging for n in self.cases
            ):
                raise ValueError("case selection differs from challenging input mode")
        return self


def _config_exclusions(config):
    """Omit new opt-in fields from historical plan serialization."""
    return (
        ({"numerical_diagnostic"} if config.numerical_diagnostic is None else set())
        | (set() if config.checkpoint_diagnostic else {"checkpoint_diagnostic"})
        | ({"screening_diagnostic"} if config.screening_diagnostic is None else set())
    )


def case_request(name, case, seed):
    """Choose a correct equation template and generic starts without truth draws."""
    if name in challenging.CELLS:
        return challenging.request(case, seed)
    if name.startswith("cstr"):
        return experiment_request(case, "joint", seed, True)[0]
    return shape.request(seed) if name == "shape" else controls.request(name, seed)


def prepare(
    root: Path,
    config: CampaignConfig,
    inputs_path: Path | None = None,
    matched_source: Path | None = None,
) -> dict:
    """Freeze paired physical starts, immutable input bytes and numerical runtime."""
    source, predecessor, predecessor_inputs = None, None, None
    if config.include_challenging and matched_source is not None:
        raise ValueError("challenging campaign uses fresh generic starts")
    if matched_source is not None:
        predecessor, predecessor_inputs = verify(matched_source, runtime=False)
        if (
            not (config.include_reuse or config.reuse_diagnostic is not None)
            or predecessor["protocol"]
            not in (
                {PROTOCOL, FOLLOWUP_PROTOCOL}
                if config.reuse_diagnostic is not None
                else {PROTOCOL}
            )
            or predecessor.get("test_data_opened") is not False
        ):
            raise ValueError("matched source must be a supported development protocol")
    elif (
        config.include_cstr
        or config.include_challenging
        or (config.screening_diagnostic is not None and inputs_path is not None)
    ):
        if inputs_path is None:
            raise ValueError("CSTR requires sealed milestone-1 inputs")
        source = read_seal(inputs_path)
        if (
            source["protocol"]
            != (
                challenging.PROTOCOL
                if config.include_challenging
                else "phase-c-fitting-strategy-inputs-1"
                if config.screening_diagnostic is not None and not config.include_cstr
                else "phase-c-fitting-inputs-1"
            )
            or source.get("test_data_opened")
            or (
                config.include_challenging
                and source.get("test_data_opened") is not False
            )
        ):
            raise ValueError("wrong input release/protocol")
    source_digest = public.content_sha256(source) if source else None
    matched_digest = public.content_sha256(predecessor) if predecessor else None
    with public._lock(root):
        if (root / "plan.json").exists():
            plan, _ = verify(root)
            if (
                plan["config"]
                != config.model_dump(
                    mode="json",
                    exclude=_config_exclusions(config),
                )
                or plan["source_inputs_sha256"] != source_digest
                or plan.get("matched_source_plan_sha256") != matched_digest
            ):
                raise ValueError("configuration/source inputs differ")
            return {
                "identity": public.content_sha256(plan),
                "tasks": len(plan["tasks"]),
                "gpus": 0,
            }
        if predecessor_inputs is not None:
            inputs = deepcopy(predecessor_inputs)
            if not config.include_cstr:
                inputs["cases"] = {
                    n: c for n, c in inputs["cases"].items() if not n.startswith("cstr")
                }
        elif config.include_challenging or (
            config.screening_diagnostic is not None and source is not None
        ):
            if config.include_challenging and set(source["cases"]) != set(
                challenging.CELLS
            ):
                raise ValueError("challenging source roster differs")
            inputs = deepcopy(source)
        else:
            inputs = controls.make_inputs()
            inputs["protocol"] = "phase-c-fitting-strategy-inputs-1"
            inputs["cases"]["shape"] = shape.make_case(inputs["cases"]["nonlinear"])
        if source and config.include_cstr:
            inputs["cases"].update(
                {n: source["cases"][n] for n in ("cstr_easy", "cstr_hard")}
            )
        if config.cases is not None:
            inputs["cases"] = {n: inputs["cases"][n] for n in config.cases}
        commons, tasks = {}, []
        for name, case in inputs["cases"].items():
            train = public.unpack_split(PublicSplit.model_validate(case["training"]))
            for seed in range(config.starts):
                key = f"{name}_s{seed}"
                if predecessor and key not in predecessor["commons"]:
                    raise ValueError(f"matched source lacks requested start: {key}")
                req = (
                    PublicFitRequest.model_validate(
                        predecessor["commons"][key]["request"]
                    )
                    if predecessor
                    else case_request(name, case, seed)
                )
                frozen_common = (
                    inputs.get("frozen_start_commons", {}).get(key)
                    if config.screening_diagnostic is not None
                    else None
                )
                if frozen_common is not None:
                    if frozen_common["case"] != name or frozen_common["seed"] != seed:
                        raise ValueError("frozen ordinary start identity differs")
                    req = PublicFitRequest.model_validate(frozen_common["request"])
                public._check_data(
                    req,
                    PublicSplit.model_validate(case["training"]),
                    PublicSplit.model_validate(case["validation"]),
                )
                model, start, _ = public._lower(req)
                system = SymbolicODE(model, allow_piecewise=True)
                vector = np.array([start[n] for n in system.names])
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
                if predecessor:
                    # Preserve exact generic starts, observations and numerical
                    # coordinates across architectures. Never read fitted results.
                    commons[key] = deepcopy(predecessor["commons"][key])
                if frozen_common is not None:
                    commons[key] = deepcopy(frozen_common)
                arms = (
                    DIAGNOSTIC_ARMS
                    if config.reuse_diagnostic is not None
                    else (*ARMS, *((REUSE_ARM,) if config.include_reuse else ()))
                )
                if config.challenging_reuse is not None:
                    arms = (*arms, "cache_last_primal", "cache_screened_primal")
                if config.numerical_diagnostic is not None:
                    arms = (
                        tuple(checkpoint.ARMS)
                        if config.checkpoint_diagnostic
                        else numerical.ARMS
                    )
                if config.screening_diagnostic is not None:
                    commons[key]["assisted_start"] = screening.bind_start(
                        commons[key], inputs, config.screening_diagnostic
                    )
                    arms = tuple(screening.ARMS)
                for arm in arms:
                    tasks.append({"task_id": f"{key}_{arm}", "common": key, "arm": arm})
        plan = {
            "protocol": screening.PROTOCOL
            if config.screening_diagnostic is not None
            else checkpoint.PROTOCOL
            if config.checkpoint_diagnostic
            else numerical.PROTOCOL
            if config.numerical_diagnostic is not None
            else CHALLENGING_PROTOCOL
            if config.include_challenging
            else DIAGNOSTIC_PROTOCOL
            if config.reuse_diagnostic is not None
            else FOLLOWUP_PROTOCOL
            if config.include_reuse
            else PROTOCOL,
            "source_sha256": public._source_identity(),
            "runtime": public._runtime(),
            "source_inputs_sha256": source_digest,
            "inputs_sha256": public.content_sha256(inputs),
            "config": config.model_dump(
                mode="json",
                exclude=_config_exclusions(config),
            ),
            "commons": commons,
            "tasks": tasks,
            "test_data_opened": False,
            "live_llm_calls": 0,
            "gpus": 0,
        }
        if predecessor:
            plan["matched_source_plan_sha256"] = matched_digest
        seal(root / "inputs.json", inputs)
        seal(root / "plan.json", plan)
    report(root)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks), "gpus": 0}


def verify(root: Path, *, runtime: bool = True):
    """Verify all immutable assets before execution; reporting tolerates new code."""
    plan, inputs = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    if plan["protocol"] not in {
        PROTOCOL,
        FOLLOWUP_PROTOCOL,
        DIAGNOSTIC_PROTOCOL,
        CHALLENGING_PROTOCOL,
        numerical.PROTOCOL,
        checkpoint.PROTOCOL,
        screening.PROTOCOL,
    } or plan["inputs_sha256"] != public.content_sha256(inputs):
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
            and (
                n.startswith("cstr")
                or n in challenging.CELLS
                or r["identifiability"].get("passed")
            )
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
    payload = {k: common[k] for k in ("request", "coordinates", "nodes")} | {
        "training": inputs["cases"][common["case"]]["training"],
        "arm": task["arm"],
        "policy": plan["config"]["strategy"],
    }
    if plan["config"].get("reuse_diagnostic") is not None:
        payload["reuse_diagnostic"] = plan["config"]["reuse_diagnostic"]
    if task["arm"] in DIAGNOSTIC_ARMS and plan["config"].get("challenging_reuse"):
        payload["reuse_diagnostic"] = plan["config"]["challenging_reuse"]
    if plan["config"].get("numerical_diagnostic") is not None:
        payload["numerical_diagnostic"] = plan["config"]["numerical_diagnostic"]
        payload["seed"] = common["seed"]
    if plan["config"].get("checkpoint_diagnostic"):
        payload["arm"], payload["checkpoint_mode"] = checkpoint.ARMS[task["arm"]]
    if plan["config"].get("screening_diagnostic") is not None:
        base, method, assisted = screening.ARMS[task["arm"]]
        options = plan["config"]["screening_diagnostic"]
        payload.update(
            arm=base,
            checkpoint_mode="compact",
            screening={
                "method": method,
                "point_seconds": options["point_seconds"],
            },
        )
        if assisted:
            payload["assisted_start"] = common["assisted_start"]
            payload["screening_diagnostic"] = options
    return payload


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


def coefficient_metrics(case, request, parameters):
    """Post-selection truth comparison; separate coefficients and hidden initials.

    Relative error is undefined at a zero reference value. Keep absolute errors
    in that case rather than inventing a favorable denominator. This evaluator
    never supplies reference values to fitting or checkpoint selection.
    """
    if parameters is None:
        return {"available": False}
    system = SymbolicODE(
        public._lower(PublicFitRequest.model_validate(request))[0], allow_piecewise=True
    )
    truth = case["reference_parameters"]
    if set(parameters) != set(truth) or set(truth) != set(system.names):
        raise ValueError("coefficient evaluation parameter identities differ")
    if not all(np.isfinite(v) for v in [*parameters.values(), *truth.values()]):
        raise ValueError("coefficient evaluation requires finite values")
    initial = set(system.initial_parameter_names)
    dynamic = [n for n in system.names if n not in initial]
    absolute = {n: abs(parameters[n] - truth[n]) for n in system.names}
    relative = {
        n: absolute[n] / abs(truth[n]) if truth[n] != 0 else None for n in dynamic
    }
    finite_relative = [v for v in relative.values() if v is not None]
    return {
        "available": True,
        "coefficient_absolute_errors": {n: absolute[n] for n in dynamic},
        "coefficient_relative_errors": relative,
        "maximum_coefficient_relative_error": max(finite_relative)
        if finite_relative and len(finite_relative) == len(dynamic)
        else None,
        "initial_parameter_absolute_errors": {n: absolute[n] for n in sorted(initial)},
        "scope": (
            "post-selection dynamic coefficients; initials separate in physical units"
        ),
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
        if (
            plan["config"].get("screening_diagnostic") is not None
            and screening.ARMS[task["arm"]][2]
            and not common["assisted_start"]["eligible"]
        ):
            result = {
                **task,
                "status": "assisted_source_ineligible",
                "assistance": common["assisted_start"],
                "seconds": 0,
            }
            seal(path, result)
            return result
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
        is_control = common["case"] in {"linear", "nonlinear", "fast_slow", "shape"}
        if parameters:
            replay_path = folder / "replay.json"
            if not replay_path.exists():
                checked = replay(
                    PublicFitRequest.model_validate(common["request"]),
                    parameters,
                    PublicSplit.model_validate(case["training"]),
                    PublicSplit.model_validate(case["validation"]),
                    config.replay_seconds,
                    **(
                        {"journal": folder / "replay_progress.json"}
                        if config.checkpoint_diagnostic
                        or config.screening_diagnostic is not None
                        else {}
                    ),
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
            <= (
                config.control_nmse
                if is_control or common["case"] == "basin_coupled"
                else config.cstr_nmse
            )
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
            "evaluation_stop_reason": checked.get("stop_reason") if checked else None,
            "validation_used_for_fitting": False,
            "global_identifiability_claimed": False,
        }
        if plan["protocol"] in {
            FOLLOWUP_PROTOCOL,
            DIAGNOSTIC_PROTOCOL,
            CHALLENGING_PROTOCOL,
            numerical.PROTOCOL,
            checkpoint.PROTOCOL,
            screening.PROTOCOL,
        }:
            errors = coefficient_metrics(case, common["request"], parameters)
            coefficient_error = errors.get("maximum_coefficient_relative_error")
            result.update(
                coefficient_recovery=errors,
                coefficients_recovered=bool(
                    checked
                    and checked["complete"]
                    and checked["maximum_solver_difference"] <= 1e-4
                    and coefficient_error is not None
                    and coefficient_error <= config.parameter_relative
                ),
            )
            if plan["protocol"] in {
                CHALLENGING_PROTOCOL,
                numerical.PROTOCOL,
                checkpoint.PROTOCOL,
                screening.PROTOCOL,
            }:
                initial_errors = errors.get("initial_parameter_absolute_errors", {})
                result["initials_recovered"] = (
                    max(initial_errors.values()) <= config.initial_absolute
                    if initial_errors
                    else None
                )
        if "reuse_diagnostic" in worker_payload(plan, inputs, task):
            result["reuse_diagnostic"] = {
                "graph_builds": backend.get("graph_builds"),
                "attempts": backend.get("attempts", []),
                "partial_metadata": backend.get("partial_metadata", False),
                "screen_seconds": backend.get("screen_seconds"),
            }
        if "numerical_diagnostic" in worker_payload(plan, inputs, task):
            result["numerical_diagnostic"] = {
                "decision": backend.get("decision"),
                "completed_training_calls_by_phase": backend.get(
                    "completed_training_calls_by_phase", {}
                ),
                "stages": backend.get("stages", []),
                "partial_metadata": backend.get("partial_metadata", False),
            }
        if config.screening_diagnostic is not None:
            result["screening"] = backend.get("screening")
            result["assisted"] = backend.get("assisted")
            result["selected_source"] = backend.get("selected_source")
            assisted = backend.get("assisted") or {}
            phases = {p["phase"]: p for p in assisted.get("phases", [])}
            final = (phases.get("released") or {}).get("final")
            if final:
                result["released_final_coefficient_recovery"] = coefficient_metrics(
                    case, common["request"], final["parameters"]
                )
                attempts = (backend.get("screening") or {}).get("attempts", [])
                screened = next(
                    (a for a in attempts if a.get("source") == "released_final"), {}
                )
                result["released_final_training_nmse"] = screened.get("training_nmse")
                result["released_final_screen_status"] = screened.get(
                    "status", "not_started"
                )
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
            "protocol": plan["protocol"],
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
                    if name in {"linear", "nonlinear", "fast_slow", "shape"}
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
        if plan["protocol"] in {
            FOLLOWUP_PROTOCOL,
            DIAGNOSTIC_PROTOCOL,
            CHALLENGING_PROTOCOL,
            numerical.PROTOCOL,
            checkpoint.PROTOCOL,
            screening.PROTOCOL,
        }:
            for group in summary["groups"]:
                items = groups[(group["case"], group["arm"])]
                errors = [
                    r.get("coefficient_recovery", {}).get(
                        "maximum_coefficient_relative_error"
                    )
                    for r in items
                ]
                finite = [e for e in errors if e is not None and np.isfinite(e)]
                group.update(
                    coefficient_vectors_available=len(finite),
                    coefficient_passes=sum(
                        bool(r.get("coefficients_recovered")) for r in items
                    ),
                    median_maximum_coefficient_relative_error=float(np.median(finite))
                    if finite
                    else None,
                    worst_maximum_coefficient_relative_error=max(finite)
                    if finite
                    else None,
                )
                for split in ("training", "validation"):
                    values = [r.get(f"{split}_nmse") for r in items]
                    finite_scores = [
                        v for v in values if v is not None and np.isfinite(v)
                    ]
                    group[f"{split}_scores_available"] = len(finite_scores)
                    group[f"median_{split}_nmse"] = (
                        float(np.median(finite_scores)) if finite_scores else None
                    )
            summary["coefficient_reporting"] = (
                "Dynamic coefficient relative errors only; initial absolute errors "
                "are separate. Medians/worst use available finite vectors and their "
                "counts are explicit. Pass denominators include all planned starts."
            )
        if plan["protocol"] in {DIAGNOSTIC_PROTOCOL, CHALLENGING_PROTOCOL}:
            summary["limitation"] = (
                "Fixed-mesh, two-solve diagnostic on known equations. In-memory "
                "graph reuse, primal starts only, explicit zero duals. Native solve "
                "time includes lazy solver setup; formulation time is not all "
                "compilation cost. No production promotion or global "
                "identifiability claim."
            )
            for group in summary["groups"]:
                items = groups[(group["case"], group["arm"])]
                attempts = [
                    a
                    for r in items
                    for a in r.get("reuse_diagnostic", {}).get("attempts", [])
                ]
                group.update(
                    recorded_attempts=len(attempts),
                    graph_seconds=sum(a.get("graph_seconds", 0) for a in attempts),
                    native_seconds=sum(a.get("seconds", 0) for a in attempts),
                    native_timings_available=sum("seconds" in a for a in attempts),
                    second_start_sources=dict(
                        Counter(
                            a["start_source"] for a in attempts if a["attempt"] == 1
                        )
                    ),
                )
        if plan["protocol"] in {
            CHALLENGING_PROTOCOL,
            numerical.PROTOCOL,
            checkpoint.PROTOCOL,
            screening.PROTOCOL,
        }:
            summary["limitation"] = (
                "Known equations and fixed parameter blocks on unchanged development "
                "arrays. Alien latent coordinates are anchored by supplied internal "
                "couplings and nonlinear shapes; local rank is not global or practical "
                "identifiability. Strategy bundles share starts and total ceilings, "
                "but their native solve schedules differ. No production promotion."
            )
            for group in summary["groups"]:
                items = groups[(group["case"], group["arm"])]
                group["initial_recovery_passes"] = (
                    sum(r.get("initials_recovered") is True for r in items)
                    if group["case"] == "alien_hard"
                    else None
                )
        if plan["protocol"] == screening.PROTOCOL:
            summary["limitation"] += (
                " M9 generic-start arms compare bounded RK45/Radau screening. "
                "Assisted arms start from earlier training-fitted endpoints, with "
                "upstream cost recorded separately. Retaining that input is not "
                "generic-start recovery or evidence of collocation improvement."
            )
            for group in summary["groups"]:
                items = groups[(group["case"], group["arm"])]
                attempts = [
                    a
                    for r in items
                    for a in (r.get("screening") or {}).get("attempts", [])
                ]
                group["screen_statuses"] = dict(Counter(a["status"] for a in attempts))
                group["assisted_input_retained"] = sum(
                    bool((r.get("assisted") or {}).get("retained_assisted_input"))
                    for r in items
                )
                group["source_ineligible"] = sum(
                    r["status"] == "assisted_source_ineligible" for r in items
                )
                group["upstream_fit_seconds"] = sum(
                    (r.get("assisted") or {}).get("source", {}).get("source_seconds", 0)
                    for r in items
                )
        if plan["protocol"] == checkpoint.PROTOCOL:
            summary["limitation"] += (
                " M8 checkpoint-policy comparison at fixed equations and meshes. "
                "Compact records retain parameter/initial vectors each iteration; "
                "detailed node diagnostics only at native exit. Timed-out replay "
                "is unavailable evidence, not a partial-data score. No new optimizer."
            )
        if plan["protocol"] == numerical.PROTOCOL:
            summary["limitation"] += (
                " M7 fixed meshes: dense-output coarsening retains all observations "
                "and supplied forcing interpolation. Derivative profiles measure "
                "one initial point and consume native budget; interrupted phases "
                "have unknown final timing. Restart trigger uses training only. "
                "Continuation transfers parameters, not native trust-region state."
            )
        public._write(root / "summary.json", summary)
    return summary
