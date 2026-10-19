"""M43 isolated initialization graphs, joint training roots and prefix forecasts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "phase-c-initialization-qualification-1"
INPUT_PROTOCOL = "initialization-qualification-inputs-1"
CASES = ("shared_root", "trajectory_root")
SEEDS = (0, 1)


class Policy(StrictSchema):
    """Freeze numerical allowances, the causal observation window and checks."""

    protocol: Literal["phase-c-initialization-qualification-policy-1"] = (
        "phase-c-initialization-qualification-policy-1"
    )
    numerical: kernel.NumericalPolicy = Field(default_factory=kernel.NumericalPolicy)
    prefix: kernel.PrefixPolicy = Field(
        default_factory=lambda: kernel.PrefixPolicy(duration=1.0)
    )
    accuracy_tolerance: FiniteFloat = Field(default=1e-8, gt=0, le=1e-4)


def response(time, rate: float, theta: float) -> np.ndarray:
    """Evaluate a forced decay response, including equal-pole limits."""
    time = np.asarray(time, float)
    gap = rate - theta
    factor = time if gap == 0 else -np.expm1(-gap * time) / gap
    return np.exp(-theta * time) * factor


def analytic(time, y0: float, theta: float, phi: float, root: float) -> np.ndarray:
    """Independent closed-form output for the fixed three-state control."""
    return (
        y0 * np.exp(-theta * np.asarray(time))
        + phi * root * response(time, 1.0, theta)
        + root * response(time, 2.0, theta)
    )


def _split(name: str, initial: tuple, roots: tuple, truth: dict) -> dict:
    """Create actual public arrays without exporting hidden states to the fitter."""
    time = np.linspace(0, 4, 41).tolist()
    rows = [
        {
            "trajectory_id": f"{name}_{index}",
            "time": time,
            "targets": {
                "y": analytic(time, y0, truth["theta"], truth["phi"], root).tolist()
            },
        }
        for index, (y0, root) in enumerate(zip(initial, roots, strict=True))
    ]
    split = PublicSplit.model_validate(
        {"name": name, "fingerprint": "pending", "rows": rows}
    ).model_dump(mode="json")
    split["fingerprint"] = public.content_sha256({"name": name, "rows": split["rows"]})
    return split


def _problem(case: str, seed: int) -> kernel.PrefixFitProblem:
    """Build the declared equations and starts independently of generating values."""
    if case not in CASES or seed not in SEEDS:
        raise ValueError("unknown initialization qualification case/seed")
    shared = case == "shared_root"
    theta_start, phi_start, root_start = ((0.2, 0.5, 0.5), (0.6, 1.2, 1.1))[seed]
    return kernel.PrefixFitProblem.model_validate(
        {
            "base_candidate": {
                "candidate_id": "initialization_qualification_fixed_equations",
                "parent_candidate_id": None,
                "states": [
                    {"name": "x", "kind": "observed"},
                    {"name": "z", "kind": "latent"},
                    {"name": "r", "kind": "latent"},
                ],
                "parameters": [
                    {
                        "name": "theta",
                        "scope": "global",
                        "role": "rate",
                        "bounds": {"lower": 0.1, "upper": 0.8},
                    }
                ],
                "state_equations": [
                    {"state": "x", "rhs": "-theta*x+z+r"},
                    {"state": "z", "rhs": "-z"},
                    {"state": "r", "rhs": "-2*r"},
                ],
                "observation_mappings": [{"channel": "y", "expression": "x"}],
                "initial_conditions": [
                    {"state": name, "scope": "global", "fixed_value": 0}
                    for name in ("x", "z", "r")
                ],
            },
            "context": {"targets": ["y"]},
            "parameter_guesses": {"theta": theta_start},
            "graph": {
                "rules": {
                    "r": {
                        "initial": {
                            "mode": "value",
                            "guess": root_start,
                            "role": "nonnegative_coefficient",
                            "scope": "shared_training" if shared else "trajectory",
                        }
                    },
                    "z": {
                        "initial": {
                            "mode": "map",
                            "expression": "phi*root",
                            "parameters": [
                                {
                                    "name": "phi",
                                    "guess": phi_start,
                                    "role": "nonnegative_coefficient",
                                }
                            ],
                            "state_initials": {"root": "r"},
                        }
                    },
                }
            },
        }
    )


def fixture(case: str, seed: int) -> dict:
    """Make an opt-in known-equation control and isolated evaluator truth."""
    problem = _problem(case, seed)
    shared = case == "shared_root"
    truth = {"theta": 0.35, "phi": 0.7}
    train_roots = (0.8, 0.8, 0.8) if shared else (0.4, 1.2, 1.6)
    val_roots = (0.8, 0.8) if shared else (0.65, 1.45)
    return {
        "case": case,
        "seed": seed,
        "problem": problem.model_dump(mode="json"),
        "training": _split(
            "train", (0.2, 0.8, 1.4) if shared else (0.4,) * 3, train_roots, truth
        ),
        "validation": _split(
            "val", (0.5, 1.1) if shared else (0.4,) * 2, val_roots, truth
        ),
        "evaluation": truth
        | {
            "roots": {
                f"{split}_{index}": value
                for split, roots in (("train", train_roots), ("val", val_roots))
                for index, value in enumerate(roots)
            }
        },
    }


def make_inputs() -> dict:
    """Freeze four modest synthetic tasks without touching benchmark assets."""
    return {
        "protocol": INPUT_PROTOCOL,
        "records": {
            f"{case}_s{seed}": fixture(case, seed) for case in CASES for seed in SEEDS
        },
        "test_data_opened": False,
    }


def _validate_inputs(data: dict) -> list[dict]:
    """Validate public boundaries and content fingerprints without reading truth."""
    tasks = [
        {"task_id": f"{case}_s{seed}", "case": case, "seed": seed}
        for case in CASES
        for seed in SEEDS
    ]
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["records"]) != {task["task_id"] for task in tasks}
    ):
        raise ValueError("initialization qualification roster differs")
    for task in tasks:
        record = data["records"][task["task_id"]]
        problem = kernel.PrefixFitProblem.model_validate(record["problem"])
        if record["case"] != task["case"] or record["seed"] != task["seed"]:
            raise ValueError("initialization qualification metadata differs")
        if problem != _problem(task["case"], task["seed"]):
            raise ValueError("initialization qualification public model differs")
        for key, name, count in (("training", "train", 3), ("validation", "val", 2)):
            split = PublicSplit.model_validate(record[key])
            payload = split.model_dump(mode="json")
            if (
                split.name != name
                or len(split.rows) != count
                or split.fingerprint
                != public.content_sha256({"name": name, "rows": payload["rows"]})
            ):
                raise ValueError("initialization qualification split content differs")
            for index, row in enumerate(split.rows):
                if (
                    row.trajectory_id != f"{name}_{index}"
                    or row.time != tuple(np.linspace(0, 4, 41))
                    or set(row.targets) != {"y"}
                    or row.auxiliaries
                    or row.external_inputs
                    or row.fixed_covariates
                ):
                    raise ValueError("initialization qualification public grid differs")
    return tasks


def prepare(root: Path, inputs: Path | None, policy: Policy) -> dict:
    """Prepare immutable actual arrays once; resume never redraws acquisitions."""
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    data = read_seal(inputs) if inputs is not None else make_inputs()
    tasks = _validate_inputs(data)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": tasks,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "truth_used_for_selection": False,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(tasks)}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Verify source/runtime for work and portable sealed inputs for reports."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    if (
        plan["protocol"] != PROTOCOL
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["tasks"] != _validate_inputs(data)
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["truth_used_for_selection"] is not False
    ):
        raise ValueError("initialization qualification identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("initialization qualification source/runtime differs")
    return plan, data


def _operation(folder: Path, identity: dict, name: str, seconds: float, work) -> dict:
    """Use the existing durable no-retry operation and retain known progress."""
    print(f"M43: {identity['task']['task_id']} {name}", flush=True)
    path = folder / "operations" / name
    value = checks.operation(path, {"campaign": identity, "stage": name}, seconds, work)
    progress = (
        public._read(path / "progress.json")
        if (path / "progress.json").exists()
        else {}
    )
    return value | {"observed_progress": progress}


def _cost(operations: list[dict]) -> dict:
    """Keep unknown interrupted cost separate from observed lower bounds."""
    result = {
        "operations": len(operations),
        "accounting_complete": all(
            value["accounting_complete"] for value in operations
        ),
        "budget_charge_seconds": sum(
            value["budget_charge_seconds"] for value in operations
        ),
        "scope": (
            "Operation wall/CPU includes setup and checkpoint writes; native search "
            "and verification have separate allowances. Actual setup overhead may "
            "exceed the reserved search-plus-verification allowance."
        ),
        "checkpoint_calls_scope": (
            "Historical checkpoint calls are not uniform solver-call counts. "
            "Use the separate search and verification counters below."
        ),
    }
    for field in ("wall_seconds", "cpu_seconds", "calls"):
        values = [value.get(field) for value in operations]
        result[field + "_observed"] = sum(
            value
            if value is not None
            else operation.get("observed_progress", {}).get(field, 0)
            for value, operation in zip(values, operations, strict=True)
        )
        result[field + "_total"] = (
            sum(values) if all(value is not None for value in values) else None
        )
        result[field + "_unknown"] = sum(value is None for value in values)
    for field in ("verification_calls_started", "verification_calls_completed"):
        counts, observed = [], []
        for operation in operations:
            value = operation.get("value")
            terminal = value is not None and operation["status"] == "complete"
            count = (value or {}).get(field, 0) if terminal else None
            counts.append(count)
            observed.append(
                count
                if count is not None
                else operation.get("observed_progress", {}).get(field, 0)
            )
        result[field + "_observed"] = sum(observed)
        result[field + "_total"] = (
            sum(counts) if all(count is not None for count in counts) else None
        )
    for suffix, name in (("started", "calls"), ("completed", "completed_calls")):
        counts, observed = [], []
        for operation in operations:
            stage = operation["identity"]["stage"]
            if stage != "training" and not stage.endswith("_inference"):
                counts.append(0)
                observed.append(0)
                continue
            value = operation.get("value") or {}
            count = value.get("search", {}).get(name)
            counts.append(count)
            progress = operation.get("observed_progress", {})
            observed.append(
                count
                if count is not None
                else progress.get("search", {}).get(name, progress.get(name, 0))
            )
        result[f"search_calls_{suffix}_observed"] = sum(observed)
        result[f"search_calls_{suffix}_total"] = (
            sum(counts) if all(count is not None for count in counts) else None
        )
    return result


def _analytic_evaluation(record: dict, fit: dict, forecasts: list[dict]) -> dict:
    """Evaluate only sealed endpoints; truth never reaches fitting or inference."""
    shared = fit["shared_parameters"]
    theta, phi = shared["theta"], shared["init_z_phi"]
    training = PublicSplit.model_validate(record["training"])
    squared = []
    for row in training.rows:
        root = (shared | fit["training_roots"][row.trajectory_id])["init_r_value"]
        predicted = analytic(row.time, row.targets["y"][0], theta, phi, root)
        squared.extend(
            ((predicted - row.targets["y"]) / fit["training_scales"]["y"]) ** 2
        )
    forecast_checks = []
    for item in forecasts:
        prediction, inference = item.get("forecast"), item.get("inference")
        if prediction is None or not prediction.get("forecast_success"):
            continue
        row = next(
            row
            for row in record[item["split"]]["rows"]
            if row["trajectory_id"] == item["trajectory_id"]
        )
        root = (shared | inference["roots"])["init_r_value"]
        exact = analytic(row["time"], row["targets"]["y"][0], theta, phi, root)
        forecast_checks.append(
            {
                "trajectory_id": row["trajectory_id"],
                "maximum_analysis_difference": float(
                    np.max(abs(exact - prediction["predictions"]["y"]))
                ),
                "dependent_map_error": abs(prediction["states"]["z"][0] - phi * root),
            }
        )
    return {
        "training_analytic_normalized_mse": float(np.mean(squared)),
        "training_objective_difference": abs(
            float(np.mean(squared)) - fit["normalized_mse"]
        ),
        "forecast_analytic_checks": forecast_checks,
        "shared_parameter_errors_evaluator_only": {
            "theta": theta - record["evaluation"]["theta"],
            "phi": phi - record["evaluation"]["phi"],
        },
        "training_root_errors_evaluator_only": {
            row.trajectory_id: (shared | fit["training_roots"][row.trajectory_id])[
                "init_r_value"
            ]
            - record["evaluation"]["roots"][row.trajectory_id]
            for row in training.rows
        },
        "prefix_root_errors_evaluator_only": {
            item["trajectory_id"]: (shared | item["inference"]["roots"])["init_r_value"]
            - record["evaluation"]["roots"][item["trajectory_id"]]
            for item in forecasts
            if item.get("inference") and item["inference"].get("fit_success")
        },
        "confidence_probability": None,
    }


def _evaluate(record: dict, fit: dict, forecasts: list[dict]) -> dict:
    """Open suffix targets and evaluator references only after all forecasts seal."""
    scores = {}
    for item in forecasts:
        prediction = item.get("forecast")
        if prediction is None or not prediction.get("forecast_success"):
            scores[item["trajectory_id"]] = None
            continue
        row = next(
            row
            for row in PublicSplit.model_validate(record[item["split"]]).rows
            if row.trajectory_id == item["trajectory_id"]
        )
        scores[row.trajectory_id] = kernel.score_forecast(
            prediction, row, item["inference"]["cutoff"], fit["training_scales"]
        )
    return _analytic_evaluation(record, fit, forecasts) | {"scores": scores}


def _completed(folder: Path, identity: dict) -> dict:
    """Verify every operation and retained progress before a terminal no-op."""
    result = read_seal(folder / "result.json")
    if result["identity"] != identity:
        raise ValueError("initialization task identity differs")
    for name, stored in result["operations"].items():
        path = folder / "operations" / name
        saved = read_seal(path / "result.json")
        if (
            read_seal(path / "started.json") != saved["identity"]
            or saved["identity"]["campaign"] != identity
            or saved["identity"]["stage"] != name
        ):
            raise ValueError("initialization operation identity differs")
        progress = (
            public._read(path / "progress.json")
            if (path / "progress.json").exists()
            else {}
        )
        if saved | {"observed_progress": progress} != stored:
            raise ValueError("initialization operation artifact differs")
    if result["training_fit"] != result["operations"]["training"]["value"]:
        raise ValueError("initialization fitted endpoint differs")
    evaluation = result["operations"].get("evaluation", {}).get("value")
    if result["evaluation"] != evaluation:
        raise ValueError("initialization evaluator result differs")
    for item in result["forecasts"]:
        name = item["trajectory_id"]
        if (
            item["inference"] != result["operations"][name + "_inference"]["value"]
            or item["forecast"]
            != result["operations"].get(name + "_forecast", {}).get("value")
            or item["score"] != (evaluation or {}).get("scores", {}).get(name)
        ):
            raise ValueError("initialization forecast result differs")
    return result


def run_task(root: Path, index: int) -> dict:
    """Fit complete TRAIN, freeze shared parameters, then infer only prefix roots."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown initialization qualification task index")
    task = plan["tasks"][index]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    record, policy = (
        data["records"][task["task_id"]],
        Policy.model_validate(plan["policy"]),
    )
    problem = kernel.PrefixFitProblem.model_validate(record["problem"])
    training = PublicSplit.model_validate(record["training"])
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity)
        operations = {}
        allowance = policy.numerical.seconds + policy.numerical.verification_seconds
        operations["training"] = _operation(
            folder,
            identity,
            "training",
            allowance,
            lambda deadline, save: kernel.fit_training(
                problem, training, policy.numerical, checkpoint=save
            ),
        )
        fit = operations["training"]["value"]
        forecasts, evaluation = [], None
        if fit is not None and fit.get("fit_success"):
            for split in ("training", "validation"):
                for row in PublicSplit.model_validate(record[split]).rows:
                    prefix = kernel.make_prefix(row, policy.prefix)
                    name = f"{row.trajectory_id}_inference"
                    operations[name] = _operation(
                        folder,
                        identity,
                        name,
                        allowance,
                        lambda deadline, save, prefix=prefix: kernel.infer_prefix(
                            problem, fit, prefix, policy.numerical, checkpoint=save
                        ),
                    )
                    inference = operations[name]["value"]
                    item = {
                        "split": split,
                        "trajectory_id": row.trajectory_id,
                        "prefix_sha256": public.content_sha256(prefix),
                        "inference": inference,
                        "forecast": None,
                        "score": None,
                    }
                    if inference is not None and inference.get("fit_success"):
                        schedule = kernel.make_forecast_schedule(row, policy.prefix)

                        def forecast_work(
                            deadline, save, inference=inference, schedule=schedule
                        ):
                            return kernel.forecast(
                                problem,
                                fit,
                                inference,
                                schedule,
                                policy.numerical,
                                checkpoint=save,
                            )

                        name = f"{row.trajectory_id}_forecast"
                        operations[name] = _operation(
                            folder,
                            identity,
                            name,
                            policy.numerical.verification_seconds,
                            forecast_work,
                        )
                        prediction = operations[name]["value"]
                        if prediction is not None:
                            item["forecast"] = prediction
                    forecasts.append(item)

            def evaluate_work(deadline, save):
                save({"calls": 1, "completed_calls": 0})
                value = _evaluate(record, fit, forecasts)
                save({"calls": 1, "completed_calls": 1})
                return value

            operations["evaluation"] = _operation(
                folder,
                identity,
                "evaluation",
                policy.numerical.verification_seconds,
                evaluate_work,
            )
            evaluation = operations["evaluation"]["value"]
            for item in forecasts:
                item["score"] = (
                    (evaluation or {}).get("scores", {}).get(item["trajectory_id"])
                )
        scores = [item["score"] for item in forecasts]
        qualified = bool(
            fit
            and fit.get("fit_success")
            and fit["normalized_mse"] <= policy.accuracy_tolerance
            and evaluation is not None
            and evaluation["training_objective_difference"]
            <= policy.numerical.solver_tolerance
            and len(scores) == 5
            and len(evaluation["forecast_analytic_checks"]) == 5
            and all(
                score
                and score["available"]
                and score["normalized_mse"] <= policy.accuracy_tolerance
                for score in scores
            )
            and all(
                value["maximum_analysis_difference"]
                <= policy.numerical.solver_tolerance
                and value["dependent_map_error"] <= policy.numerical.solver_tolerance
                for value in evaluation["forecast_analytic_checks"]
            )
        )
        result = task | {
            "identity": identity,
            "status": "complete" if qualified else "incomplete",
            "qualification_passed": qualified,
            "training_fit": fit,
            "forecasts": forecasts,
            "evaluation": evaluation,
            "operations": operations,
            "cost": _cost(list(operations.values())),
            "test_data_opened": False,
            "truth_used_for_selection": False,
        }
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Separate full-training reconstruction and strict-future prefix forecasts."""
    plan, _ = verify(root, runtime=False)
    rows = []
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        if (folder / "result.json").exists():
            rows.append(_completed(folder, identity))
            continue
        operations = {}
        for started in sorted((folder / "operations").glob("*/started.json")):
            stage_identity = read_seal(started)
            if (
                stage_identity.get("campaign") != identity
                or stage_identity.get("stage") != started.parent.name
            ):
                raise ValueError("partial initialization operation identity differs")
            terminal = started.parent / "result.json"
            value = (
                read_seal(terminal)
                if terminal.exists()
                else {
                    "identity": stage_identity,
                    "status": "unfinished",
                    "value": None,
                    "wall_seconds": None,
                    "cpu_seconds": None,
                    "calls": None,
                    "accounting_complete": False,
                    "budget_charge_seconds": stage_identity["allowance_seconds"],
                }
            )
            if value["identity"] != stage_identity:
                raise ValueError("partial initialization result identity differs")
            progress = started.parent / "progress.json"
            operations[started.parent.name] = value | {
                "observed_progress": public._read(progress) if progress.exists() else {}
            }
        rows.append(task | {"status": "missing", "operations": operations})
    recorded = sum(row["status"] != "missing" for row in rows)
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "qualification_passed": all(
            row.get("qualification_passed", False) for row in rows
        ),
        "rows": rows,
        "training_reconstruction": [
            {
                "task_id": row["task_id"],
                "normalized_mse": (row.get("training_fit") or {}).get("normalized_mse"),
                "rank": (row.get("training_fit") or {}).get("rank_diagnostic"),
            }
            for row in rows
        ],
        "prefix_forecasts": {
            split: [
                {
                    "task_id": row["task_id"],
                    "trajectory_id": value["trajectory_id"],
                    "score": value["score"],
                    "rank": (value.get("inference") or {}).get("rank_diagnostic"),
                    "search_calls": (value.get("inference") or {})
                    .get("search", {})
                    .get("calls"),
                }
                for row in rows
                for value in row.get("forecasts", [])
                if value["split"] == split
            ]
            for split in ("training", "validation")
        },
        "cost": _cost(
            [
                operation
                for row in rows
                for operation in row.get("operations", {}).values()
            ]
        ),
        "missing_task_costs": len(rows) - recorded,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "confidence_probability": None,
        "limitation": (
            "Known-equation noisefree qualification only. Shared dynamics and map "
            "coefficients are fitted on complete TRAIN; separate roots are inferred "
            "from predeclared prefixes and only strict suffix samples are scored. "
            "Training-prefix forecasts use shared parameters learned from full TRAIN "
            "and are not independent held-out forecasts. Local rank is not global "
            "identification; no fault or confidence-probability claim is made."
        ),
    }
    if recorded != len(rows):
        summary["cost"]["accounting_complete"] = False
        for field in (
            "wall_seconds",
            "cpu_seconds",
            "calls",
            "verification_calls_started",
            "verification_calls_completed",
            "search_calls_started",
            "search_calls_completed",
        ):
            summary["cost"][field + "_total"] = None
    public._write(root / "summary.json", summary)
    return summary
