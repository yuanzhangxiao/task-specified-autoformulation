"""M44: select a prefix duration using validation on a common late horizon.

Synthetic known-equation qualification only. Generating roots and clean signals
are isolated in the final evaluator; none are used to select a duration or start.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import confidence_checks as checks
from autoformalism.fitting import initialization_qualification as previous
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_window_diagnostics as diagnostic
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "phase-c-prefix-window-selection-1"
INPUT_PROTOCOL = "prefix-window-inputs-1"
ACQUISITIONS = (("clean", 0, 0.0), ("noisy", 0, 0.01), ("noisy", 1, 0.01))


class Policy(StrictSchema):
    """Physical durations and numerical tolerances, frozen before observations."""

    protocol: Literal["phase-c-prefix-window-policy-1"] = (
        "phase-c-prefix-window-policy-1"
    )
    durations: tuple[FiniteFloat, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
    root_starts: tuple[FiniteFloat, FiniteFloat] = (0.5, 1.1)
    training_numerical: kernel.NumericalPolicy = Field(
        default_factory=kernel.NumericalPolicy
    )
    prefix_numerical: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            seconds=10, verification_seconds=5
        )
    )
    validation_nmse_limit: FiniteFloat = Field(default=0.01, gt=0)
    start_prediction_spread_limit: FiniteFloat = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if (
            not self.durations
            or self.durations != tuple(sorted(set(self.durations)))
            or self.durations[0] != 0
            or self.durations[-1] >= 4
            or any(
                not np.isclose(value * 10, round(value * 10))
                for value in self.durations
            )
        ):
            raise ValueError("durations must start at zero and follow grid before t=4")
        if len(set(self.root_starts)) != 2 or min(self.root_starts) < 0:
            raise ValueError("two distinct nonnegative root starts required")
        return self


def roster() -> list[dict]:
    """Three acquisitions, each repeated under two training optimization starts."""
    return [
        {
            "task_id": f"{case}_r{repeat}_s{seed}",
            "case": case,
            "repeat": repeat,
            "seed": seed,
            "noise_sigma": sigma,
        }
        for case, repeat, sigma in ACQUISITIONS
        for seed in (0, 1)
    ]


def _fingerprint(split: dict) -> dict:
    split = PublicSplit.model_validate(split).model_dump(mode="json")
    split["fingerprint"] = public.content_sha256(
        {"name": split["name"], "rows": split["rows"]}
    )
    return split


def make_inputs() -> dict:
    """Freeze measured acquisitions; exact first readings are an explicit control."""
    records = {}
    for task in roster():
        fixture = previous.fixture("trajectory_root", task["seed"])
        truth = fixture["evaluation"]
        validation = previous._split("val", (0.4,) * 4, (0.65, 0.95, 1.45, 1.8), truth)
        clean = {"training": fixture["training"], "validation": validation}
        measured = deepcopy(clean)
        rng = np.random.default_rng(20261019 + task["repeat"])
        for split in measured.values():
            for row in split["rows"]:
                values = np.asarray(row["targets"]["y"], float)
                # Time zero stays exact under the current observed-boundary contract.
                values[1:] += rng.normal(0, task["noise_sigma"], len(values) - 1)
                row["targets"]["y"] = values.tolist()
        records[task["task_id"]] = task | {
            "problem": fixture["problem"],
            "training": _fingerprint(measured["training"]),
            "validation": _fingerprint(measured["validation"]),
            "evaluation": {
                "theta": 0.35,
                "phi": 0.7,
                "validation_roots": dict(
                    zip(
                        (f"val_{i}" for i in range(4)),
                        (0.65, 0.95, 1.45, 1.8),
                        strict=True,
                    )
                ),
                "clean": clean,
            },
        }
    return {"protocol": INPUT_PROTOCOL, "records": records, "test_data_opened": False}


def _validate(data: dict) -> None:
    """Validate public shape and identity without consulting evaluator labels."""
    if (
        data["protocol"] != INPUT_PROTOCOL
        or data["test_data_opened"] is not False
        or set(data["records"]) != {task["task_id"] for task in roster()}
    ):
        raise ValueError("prefix-window input roster differs")
    for task in roster():
        record = data["records"][task["task_id"]]
        if any(record[key] != value for key, value in task.items()):
            raise ValueError("prefix-window acquisition metadata differs")
        if kernel.PrefixFitProblem.model_validate(
            record["problem"]
        ) != previous._problem("trajectory_root", task["seed"]):
            raise ValueError("prefix-window model differs")
        for key, name, count in (("training", "train", 3), ("validation", "val", 4)):
            split = PublicSplit.model_validate(record[key])
            if split.model_dump(mode="json") != _fingerprint(record[key]):
                raise ValueError("prefix-window split fingerprint differs")
            if split.name != name or len(split.rows) != count:
                raise ValueError("prefix-window split identity differs")
            for index, row in enumerate(split.rows):
                if (
                    row.trajectory_id != f"{name}_{index}"
                    or set(row.targets) != {"y"}
                    or row.time != tuple(np.linspace(0, 4, 41))
                    or row.targets["y"][0] != 0.4
                    or row.auxiliaries
                    or row.external_inputs
                    or row.fixed_covariates
                ):
                    raise ValueError("prefix-window public grid or boundary differs")
        pair = data["records"][task["task_id"].rsplit("_s", 1)[0] + "_s0"]
        if any(record[key] != pair[key] for key in ("training", "validation")):
            raise ValueError("training starts must share the same acquisition")


def prepare(root: Path, inputs: Path | None, policy: Policy) -> dict:
    """Freeze data, policy, code and runtime with deterministic no-redraw resume."""
    if inputs is None and (root / "inputs.json").exists():
        inputs = root / "inputs.json"
    data = read_seal(inputs) if inputs is not None else make_inputs()
    _validate(data)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "test_data_opened": False,
        "live_llm_calls": 0,
    }
    with public._lock(root):
        seal(root / "inputs.json", data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    _validate(data)
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster()
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
    ):
        raise ValueError("prefix-window plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("prefix-window source/runtime differs")
    return plan, data


def _operation(folder, identity, name, seconds, work):
    print(f"M44: {identity['task']['task_id']} {name}", flush=True)
    path = folder / "operations" / name
    value = checks.operation(path, {"campaign": identity, "stage": name}, seconds, work)
    progress = path / "progress.json"
    return value | {
        "observed_progress": public._read(progress) if progress.exists() else {}
    }


def _cost(operations: dict) -> dict:
    result = previous._cost(list(operations.values()))
    for field in ("diagnostic_calls_started", "diagnostic_calls_completed"):
        values = [
            (op.get("value") or {}).get(field, 0)
            if op["status"] == "complete"
            else None
            for name, op in operations.items()
            if name.endswith("_information")
        ]
        result[field + "_observed"] = sum(
            (op.get("value") or op.get("observed_progress") or {}).get(field, 0)
            for name, op in operations.items()
            if name.endswith("_information")
        )
        result[field + "_total"] = (
            sum(values) if all(v is not None for v in values) else None
        )
    return result


def _choose_prefix_start(attempts: list[dict]) -> int | None:
    """Choose by verified prefix loss alone; deterministic order breaks ties."""
    available = [
        (item["inference"]["normalized_mse"], index)
        for index, item in enumerate(attempts)
        if item.get("inference")
        and item["inference"].get("fit_success")
        and _valid_loss(item["inference"].get("normalized_mse"))
    ]
    return min(available)[1] if available else None


def _valid_loss(value) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and np.isfinite(value)
        and value >= 0
    )


def _complete_entries(duration: float, entries: list[dict], policy: Policy) -> bool:
    """Validate expected endpoint and scoring receipts before aggregating them."""
    try:
        if len(entries) != 4 or {e["trajectory_id"] for e in entries} != {
            f"val_{i}" for i in range(4)
        }:
            return False
        for item in entries:
            attempts = item["attempts"]
            if (
                len(attempts) != 2
                or [a["start"] for a in attempts] != list(policy.root_starts)
                or item["chosen_start"] is None
                or item["chosen_start"] != _choose_prefix_start(attempts)
            ):
                return False
            for a in attempts:
                inference, forecast, score = (
                    a["inference"],
                    a["forecast"],
                    a["common_score"],
                )
                if not inference or not forecast or not score:
                    return False
                times = np.asarray(forecast["time"], float)
                sample_count = int(np.sum(times > policy.durations[-1]))
                if (
                    not inference["fit_success"]
                    or not forecast["forecast_success"]
                    or not _valid_loss(inference["normalized_mse"])
                    or inference["cutoff"] != duration
                    or inference["root_start"] != {"init_r_value": a["start"]}
                    or forecast["trajectory_id"] != item["trajectory_id"]
                    or forecast["cutoff"] != duration
                    or forecast["inference_sha256"] != public.content_sha256(inference)
                    or not score["available"]
                    or not _valid_loss(score["normalized_mse"])
                    or score["forecast_cutoff"] != duration
                    or score["score_start"] != policy.durations[-1]
                    or score["forecast_sha256"] != public.content_sha256(forecast)
                    or sample_count == 0
                    or score["sample_count"] != sample_count
                    or score["horizon_sample_count"] != sample_count
                ):
                    return False
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def window_metrics(duration: float, entries: list[dict], policy: Policy) -> dict:
    """Keep failed attempts in the denominator; uncertainty remains advisory."""
    ready = _complete_entries(duration, entries, policy)
    losses, spreads, roots = [], [], []
    if ready:
        for item in entries:
            chosen = item["attempts"][item["chosen_start"]]
            losses.append(chosen["common_score"]["normalized_mse"])
            first, second = item["attempts"]
            times = np.asarray(first["forecast"]["time"], float)
            if not np.array_equal(times, second["forecast"]["time"]):
                ready = False
                break
            mask = times > policy.durations[-1]
            delta = np.asarray(first["forecast"]["predictions"]["y"]) - np.asarray(
                second["forecast"]["predictions"]["y"]
            )
            scale = first["inference"]["training_scales"]["y"]
            if (
                not mask.any()
                or not np.isfinite(delta).all()
                or not np.isfinite(scale)
                or scale <= 0
                or first["inference"]["training_scales"]
                != second["inference"]["training_scales"]
            ):
                ready = False
                break
            spreads.append(float(np.sqrt(np.mean((delta[mask] / scale) ** 2))))
            roots.append(
                abs(
                    first["inference"]["roots"]["init_r_value"]
                    - second["inference"]["roots"]["init_r_value"]
                )
            )
    mean = float(np.mean(losses)) if ready else None
    spread = max(spreads) if ready else None
    return {
        "duration": duration,
        "complete": ready,
        "validation_rows_expected": 4,
        "mean_validation_nmse": mean,
        "per_row_validation_nmse": losses,
        "maximum_start_prediction_spread": spread,
        "maximum_start_root_spread": max(roots) if ready else None,
        "adequate": bool(
            ready
            and mean <= policy.validation_nmse_limit
            and spread <= policy.start_prediction_spread_limit
        ),
    }


def select_duration(metrics: list[dict], policy: Policy) -> dict:
    """Select shortest adequate duration only after the entire grid is accounted."""
    complete = (
        len(metrics) == len(policy.durations)
        and [item["duration"] for item in metrics] == list(policy.durations)
        and all(item["complete"] for item in metrics)
    )
    acceptable = [item["duration"] for item in metrics if item["adequate"]]
    chosen = min(acceptable) if complete and acceptable else None
    return {
        "status": "unavailable"
        if not complete
        else "selected"
        if chosen is not None
        else "no_adequate_window",
        "duration": chosen,
        "scoring_start": policy.durations[-1],
        "selection_data": "validation_common_suffix",
        "criterion": "shortest_adequate_for_common_late_horizon",
        "root_information_used_as_gate": False,
        "unbiased_generalization_estimate": False,
    }


def _evaluate(record: dict, fit: dict, windows: list[dict], selection: dict) -> dict:
    """Generating values enter only this post-selection diagnostic."""
    values = []
    for window in windows:
        for item in window["trajectories"]:
            index = item["chosen_start"]
            if index is None:
                continue
            chosen = item["attempts"][index]
            if not chosen.get("forecast") or not chosen["forecast"].get(
                "forecast_success"
            ):
                continue
            name = item["trajectory_id"]
            clean = next(
                r
                for r in record["evaluation"]["clean"]["validation"]["rows"]
                if r["trajectory_id"] == name
            )
            prediction = chosen["forecast"]
            time = np.asarray(prediction["time"])
            mask = time > selection["scoring_start"]
            y = np.asarray(prediction["predictions"]["y"])
            root = chosen["inference"]["roots"]["init_r_value"]
            shared = fit["shared_parameters"]
            analytic = previous.analytic(
                time, 0.4, shared["theta"], shared["init_z_phi"], root
            )
            values.append(
                {
                    "duration": window["duration"],
                    "trajectory_id": name,
                    "root_error": root - record["evaluation"]["validation_roots"][name],
                    "clean_suffix_rmse": float(
                        np.sqrt(
                            np.mean(
                                (y[mask] - np.asarray(clean["targets"]["y"])[mask]) ** 2
                            )
                        )
                    ),
                    "maximum_analytic_difference": float(np.max(abs(y - analytic))),
                }
            )
    return {
        "selection_sha256": public.content_sha256(selection),
        "rows": values,
        "truth_used_for_selection": False,
        "confidence_probability": None,
    }


def run_task(root: Path, index: int) -> dict:
    """Fit once, independently infer prefixes, seal selection, then evaluate truth."""
    plan, data = verify(root)
    if not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown prefix-window task index")
    task = plan["tasks"][index]
    policy = Policy.model_validate(plan["policy"])
    record = data["records"][task["task_id"]]
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity, record, policy)
        operations, windows = {}, []

        def operation(name, allowance, work):
            operations[name] = _operation(folder, identity, name, allowance, work)
            return operations[name]["value"]

        problem = kernel.PrefixFitProblem.model_validate(record["problem"])
        training = PublicSplit.model_validate(record["training"])
        fit = operation(
            "training",
            policy.training_numerical.seconds
            + policy.training_numerical.verification_seconds,
            lambda deadline, save: kernel.fit_training(
                problem, training, policy.training_numerical, checkpoint=save
            ),
        )
        if fit and fit.get("fit_success"):
            for wi, duration in enumerate(policy.durations):
                entries = []
                prefix_policy = kernel.PrefixPolicy(duration=duration)
                for row in PublicSplit.model_validate(record["validation"]).rows:
                    prefix = kernel.make_prefix(row, prefix_policy)
                    schedule = kernel.make_forecast_schedule(row, prefix_policy)
                    attempts = []
                    for si, start in enumerate(policy.root_starts):
                        label = f"w{wi}_{row.trajectory_id}_s{si}"
                        inference = operation(
                            label + "_inference",
                            policy.prefix_numerical.seconds
                            + policy.prefix_numerical.verification_seconds,
                            lambda deadline,
                            save,
                            prefix=prefix,
                            start=start: kernel.infer_prefix(
                                problem,
                                fit,
                                prefix,
                                policy.prefix_numerical,
                                checkpoint=save,
                                root_guesses={"init_r_value": start},
                            ),
                        )
                        prediction, common, immediate = None, None, None
                        if inference and inference.get("fit_success"):
                            prediction = operation(
                                label + "_forecast",
                                policy.prefix_numerical.verification_seconds,
                                lambda deadline,
                                save,
                                inference=inference,
                                schedule=schedule: kernel.forecast(
                                    problem,
                                    fit,
                                    inference,
                                    schedule,
                                    policy.prefix_numerical,
                                    checkpoint=save,
                                ),
                            )
                            if prediction is not None:
                                common = diagnostic.common_horizon_score(
                                    prediction,
                                    row,
                                    policy.durations[-1],
                                    fit["training_scales"],
                                )
                                immediate = kernel.score_forecast(
                                    prediction,
                                    row,
                                    prefix.cutoff,
                                    fit["training_scales"],
                                )
                        attempts.append(
                            {
                                "label": label,
                                "start": start,
                                "inference": inference,
                                "forecast": prediction,
                                "common_score": common,
                                "own_suffix_score_descriptive_only": immediate,
                            }
                        )
                    chosen = _choose_prefix_start(attempts)
                    information = None
                    label = f"w{wi}_{row.trajectory_id}_information"
                    if chosen is not None:
                        inference = attempts[chosen]["inference"]
                        information = operation(
                            label,
                            policy.prefix_numerical.verification_seconds,
                            lambda deadline,
                            save,
                            prefix=prefix,
                            inference=inference: diagnostic.prefix_information(
                                problem,
                                fit,
                                prefix,
                                inference,
                                policy.prefix_numerical,
                                task["noise_sigma"],
                                checkpoint=save,
                            ),
                        )
                    entries.append(
                        {
                            "trajectory_id": row.trajectory_id,
                            "attempts": attempts,
                            "chosen_start": chosen,
                            "information_label": label,
                            "information": information,
                        }
                    )
                windows.append(
                    {
                        "duration": duration,
                        "trajectories": entries,
                        "metrics": window_metrics(duration, entries, policy),
                    }
                )
        selection = select_duration([item["metrics"] for item in windows], policy)
        # Commit the public-data decision before any generating values are consumed.
        seal(folder / "selection.json", selection)
        evaluation = operation(
            "evaluation",
            10,
            lambda deadline, save: _evaluate(record, fit, windows, selection),
        )
        result = task | {
            "identity": identity,
            "training_fit": fit,
            "windows": windows,
            "selection": selection,
            "evaluation": evaluation,
            "operations": operations,
            "cost": _cost(operations),
            "status": "complete"
            if selection["status"] != "unavailable"
            else "incomplete",
            "test_data_opened": False,
        }
        seal(folder / "result.json", result)
        return result


def _read_operations(folder: Path, identity: dict) -> dict:
    operations = {}
    for started in sorted((folder / "operations").glob("*/started.json")):
        item = read_seal(started)
        if item.get("campaign") != identity or item.get("stage") != started.parent.name:
            raise ValueError("prefix-window operation identity differs")
        path = started.parent / "result.json"
        value = (
            read_seal(path)
            if path.exists()
            else {
                "identity": item,
                "status": "unfinished",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "calls": None,
                "accounting_complete": False,
                "budget_charge_seconds": item["allowance_seconds"],
            }
        )
        if value["identity"] != item:
            raise ValueError("prefix-window operation receipt differs")
        progress = started.parent / "progress.json"
        operations[started.parent.name] = value | {
            "observed_progress": public._read(progress) if progress.exists() else {}
        }
    return operations


def _completed(folder: Path, identity: dict, record: dict, policy: Policy) -> dict:
    result = read_seal(folder / "result.json")
    if (
        result["identity"] != identity
        or result["operations"] != _read_operations(folder, identity)
        or result["training_fit"] != result["operations"]["training"]["value"]
        or result["selection"] != read_seal(folder / "selection.json")
        or result["evaluation"] != result["operations"]["evaluation"]["value"]
    ):
        raise ValueError("prefix-window terminal artifact differs")
    fit = result["training_fit"]
    expected = list(policy.durations) if fit and fit.get("fit_success") else []
    if [w["duration"] for w in result["windows"]] != expected:
        raise ValueError("prefix-window retained grid differs")
    observations = {
        row.trajectory_id: row
        for row in PublicSplit.model_validate(record["validation"]).rows
    }
    for wi, window in enumerate(result["windows"]):
        entries = window["trajectories"]
        if [entry["trajectory_id"] for entry in entries] != list(observations):
            raise ValueError("prefix-window retained validation roster differs")
        for entry in entries:
            row = observations[entry["trajectory_id"]]
            prefix_policy = kernel.PrefixPolicy(duration=window["duration"])
            prefix = kernel.make_prefix(row, prefix_policy)
            schedule = kernel.make_forecast_schedule(row, prefix_policy)
            if [a["start"] for a in entry["attempts"]] != list(policy.root_starts):
                raise ValueError("prefix-window retained starts differ")
            for si, attempt in enumerate(entry["attempts"]):
                if attempt["label"] != f"w{wi}_{row.trajectory_id}_s{si}":
                    raise ValueError("prefix-window retained attempt label differs")
                for field in ("inference", "forecast"):
                    saved = result["operations"].get(attempt["label"] + "_" + field)
                    if attempt[field] != (saved or {}).get("value"):
                        raise ValueError("prefix-window retained endpoint differs")
                inference, forecast = attempt["inference"], attempt["forecast"]
                if inference is not None and (
                    inference["fit_sha256"] != public.content_sha256(fit)
                    or inference["prefix_sha256"] != public.content_sha256(prefix)
                    or inference["root_start"] != {"init_r_value": attempt["start"]}
                    or inference["shared_parameters"] != fit["shared_parameters"]
                    or inference["training_scales"] != fit["training_scales"]
                ):
                    raise ValueError(
                        "prefix-window retained inference boundary differs"
                    )
                if forecast is not None:
                    if forecast["inference_sha256"] != public.content_sha256(
                        inference
                    ) or forecast["schedule_sha256"] != public.content_sha256(schedule):
                        raise ValueError(
                            "prefix-window retained forecast boundary differs"
                        )
                    common = diagnostic.common_horizon_score(
                        forecast, row, policy.durations[-1], fit["training_scales"]
                    )
                    own = kernel.score_forecast(
                        forecast, row, prefix.cutoff, fit["training_scales"]
                    )
                else:
                    common, own = None, None
                if (
                    attempt["common_score"] != common
                    or attempt["own_suffix_score_descriptive_only"] != own
                ):
                    raise ValueError("prefix-window retained score differs")
            if entry["chosen_start"] != _choose_prefix_start(entry["attempts"]):
                raise ValueError("prefix-window retained start selection differs")
            if entry["information_label"] != f"w{wi}_{row.trajectory_id}_information":
                raise ValueError("prefix-window retained information label differs")
            info = result["operations"].get(entry["information_label"])
            if entry["information"] != (info or {}).get("value"):
                raise ValueError("prefix-window retained diagnostic differs")
        if window["metrics"] != window_metrics(window["duration"], entries, policy):
            raise ValueError("prefix-window retained metrics differ")
    if result["selection"] != select_duration(
        [w["metrics"] for w in result["windows"]], policy
    ):
        raise ValueError("prefix-window retained selection differs")
    if result["cost"] != _cost(result["operations"]):
        raise ValueError("prefix-window retained cost differs")
    return result


def report(root: Path) -> dict:
    """Keep all runs visible and recommend a conservative duration per noise regime."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows, operations = [], {}
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        if (folder / "result.json").exists():
            row = _completed(folder, identity, data["records"][task["task_id"]], policy)
        else:
            row = task | {
                "status": "missing",
                "windows": [],
                "selection": None,
                "operations": _read_operations(folder, identity),
            }
        rows.append(row)
        operations.update(
            {
                task["task_id"] + "/" + key: value
                for key, value in row["operations"].items()
            }
        )
    selections = []
    for case in ("clean", "noisy"):
        members = [row for row in rows if row["case"] == case]
        metrics = []
        for index, duration in enumerate(policy.durations):
            found = [
                row["windows"][index]["metrics"]
                for row in members
                if len(row["windows"]) == len(policy.durations)
            ]
            complete = len(found) == len(members) and all(m["complete"] for m in found)
            metrics.append(
                {
                    "duration": duration,
                    "complete": complete,
                    "adequate": complete and all(m["adequate"] for m in found),
                }
            )
        selections.append(
            {
                "case": case,
                "training_runs": len(members),
                "aggregation": "adequate_in_every_acquisition_and_training_start",
                **select_duration(metrics, policy),
            }
        )
    recorded = sum(row["status"] != "missing" for row in rows)
    cost = _cost(operations)
    if recorded != len(rows):
        cost["accounting_complete"] = False
        for key in list(cost):
            if key.endswith("_total"):
                cost[key] = None
    summary = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete" if recorded == len(rows) else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "selection_counts": dict(Counter(s["status"] for s in selections)),
        "selections": selections,
        "rows": rows,
        "cost": cost,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Synthetic known-equation duration tuning on validation, "
        "not test "
        "performance. Three acquisitions repeated under two training starts. Exact t0 "
        "readings; root information is conditional on frozen shared coefficients. "
        "Common horizon begins after the longest prefix; earlier forecasts are "
        "descriptive only. No confidence probabilities or production defaults.",
    }
    public._write(root / "summary.json", summary)
    return summary
