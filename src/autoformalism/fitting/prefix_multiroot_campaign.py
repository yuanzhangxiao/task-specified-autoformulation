"""M46: multiple latent roots, duration tuning and frozen follow-up forecasts.

Synthetic controls only. No benchmark test payload or constructor is called.
Generating roots and clean signals are consumed only after selection is sealed.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import monotonic, process_time
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_execution as execute
from autoformalism.fitting import prefix_initialization as kernel
from autoformalism.fitting import prefix_multiroot_controls as controls
from autoformalism.fitting import prefix_verification as verification
from autoformalism.fitting import prefix_window_selection as previous
from autoformalism.fitting import public_fitting as public
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "phase-c-prefix-multiroot-1"


class Policy(StrictSchema):
    """Freeze all starts, horizons and adequacy criteria before any fitting."""

    protocol: Literal["phase-c-prefix-multiroot-policy-1"] = (
        "phase-c-prefix-multiroot-policy-1"
    )
    durations: tuple[FiniteFloat, ...] = (0, 0.1, 0.2, 0.5, 1, 2)
    root_starts: tuple[tuple[FiniteFloat, FiniteFloat], ...] = (
        (0.3, 1.3),
        (1.3, 0.3),
        (1, 1),
    )
    immediate_duration: FiniteFloat = Field(default=0.5, gt=0, le=1)
    validation_nmse_limit: FiniteFloat = Field(default=0.01, gt=0)
    start_prediction_spread_limit: FiniteFloat = Field(default=0.01, gt=0)
    training_numerical: kernel.NumericalPolicy = Field(
        default_factory=kernel.NumericalPolicy
    )
    prefix_numerical: kernel.NumericalPolicy = Field(
        default_factory=lambda: kernel.NumericalPolicy(
            seconds=10, verification_seconds=5
        )
    )
    execution: execute.SearchExecutionPolicy = Field(
        default_factory=execute.SearchExecutionPolicy
    )

    @model_validator(mode="after")
    def ordered(self):
        if (
            not self.durations
            or self.durations != tuple(sorted(set(self.durations)))
            or self.durations[0] != 0
            or self.durations[-1] + self.immediate_duration >= 4
            or any(t not in tuple(np.linspace(0, 4, 41)) for t in self.durations)
            or not np.isclose(
                self.immediate_duration * 10, round(self.immediate_duration * 10)
            )
        ):
            raise ValueError(
                "duration grid must use exact observation timestamps, "
                "start at zero and leave future data"
            )
        if (
            not 2 <= len(self.root_starts) <= 4
            or len(set(self.root_starts)) != len(self.root_starts)
            or min(min(s) for s in self.root_starts) < 0
        ):
            raise ValueError("distinct nonnegative two-root starts required")
        return self


def roster() -> list[dict]:
    """Two training starts for each of six fixed acquisition/scenario pairs."""
    return [
        {
            "task_id": f"{scenario}_{case}_s{seed}",
            "scenario": scenario,
            "case": case,
            "noise_sigma": sigma,
            "seed": seed,
        }
        for scenario in controls.SCENARIOS
        for case, sigma in (("clean", 0.0), ("noisy", 0.01))
        for seed in (0, 1)
    ]


def make_inputs() -> dict:
    """Generate reproducible qualification arrays, not released benchmark data."""
    return {
        "protocol": "prefix-multiroot-inputs-1",
        "records": {
            t["task_id"]: controls.fixture(
                t["scenario"], seed=t["seed"], noise_sigma=t["noise_sigma"]
            )
            for t in roster()
        },
    }


def _validate(data: dict) -> None:
    if data["protocol"] != "prefix-multiroot-inputs-1" or set(data["records"]) != {
        t["task_id"] for t in roster()
    }:
        raise ValueError("multiroot input roster differs")
    for task in roster():
        r = data["records"][task["task_id"]]
        if (
            r["scenario"] != task["scenario"]
            or r["seed"] != task["seed"]
            or r["noise_sigma"] != task["noise_sigma"]
            or r["repeat"] != 0
            or r["test_data_opened"] is not False
            or r["followup_scope"] != "new_synthetic_qualification_trajectories"
        ):
            raise ValueError("multiroot acquisition identity differs")
        if kernel.PrefixFitProblem.model_validate(r["problem"]) != controls.problem(
            task["scenario"], task["seed"]
        ):
            raise ValueError("multiroot declared model differs")
        identifiers = set()
        for key, count, name in (
            ("training", 3, "train"),
            ("validation", 4, "val"),
            ("followup", 4, "val"),
        ):
            split = PublicSplit.model_validate(r[key])
            if split.name != name or len(split.rows) != count:
                raise ValueError("multiroot split identity differs")
            if controls.fingerprint(r[key]) != r[key]:
                raise ValueError("multiroot observation fingerprint differs")
            for row in split.rows:
                if (
                    row.trajectory_id in identifiers
                    or row.time != tuple(np.linspace(0, 4, 41))
                    or set(row.targets) != {"y"}
                    or row.targets["y"][0] != 0.4
                    or row.auxiliaries
                    or row.external_inputs
                    or row.fixed_covariates
                ):
                    raise ValueError("multiroot grid/boundary/row identity differs")
                identifiers.add(row.trajectory_id)
        pair = data["records"][task["task_id"].rsplit("_s", 1)[0] + "_s0"]
        if any(r[k] != pair[k] for k in ("training", "validation", "followup")):
            raise ValueError("training starts must share identical observations")


def prepare(root: Path, policy: Policy) -> dict:
    """Freeze the complete experiment once; resume never redraws observations."""
    path = root / "inputs.json"
    data = read_seal(path) if path.exists() else make_inputs()
    _validate(data)
    plan = {
        "protocol": PROTOCOL,
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "production_default_changed": False,
        "actual_construction_export_qualified": False,
    }
    with public._lock(root):
        seal(path, data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Check sealed data/source identity without consulting generating truth."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    Policy.model_validate(plan["policy"])
    _validate(data)
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster()
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["production_default_changed"] is not False
        or plan["actual_construction_export_qualified"] is not False
    ):
        raise ValueError("multiroot plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("multiroot source/runtime differs")
    return plan, data


def _operation(folder, identity, name, seconds, work):
    """One durable attempt; interrupt reservations never silently become retries."""
    print(f"M46: {identity['task']['task_id']} {name}", flush=True)
    directory = folder / "operations" / name
    bound = {"campaign": identity, "stage": name, "allowance_seconds": seconds}
    with public._lock(directory):
        start, path = directory / "started.json", directory / "result.json"
        if start.exists() and read_seal(start) != bound:
            raise ValueError("multiroot operation identity differs")
        if path.exists():
            saved = read_seal(path)
            if saved["identity"] != bound:
                raise ValueError("multiroot operation receipt differs")
            return saved
        if start.exists():
            result = {
                "status": "interrupted",
                "value": None,
                "wall_seconds": None,
                "cpu_seconds": None,
                "budget_charge_seconds": seconds,
                "accounting_complete": False,
            }
        else:
            seal(start, bound)
            begun, cpu, progress = monotonic(), process_time(), {}

            def save(value):
                progress.update(value)
                public._write(directory / "progress.json", progress)

            interrupt = None
            try:
                result = {"status": "complete", "value": work(save)}
            except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
                result = {
                    "status": "unavailable",
                    "value": None,
                    "error": str(error)[-1000:],
                }
            except BaseException as error:
                interrupt = error
                receipt = getattr(error, "verification_receipt", None)
                if receipt is not None:
                    save(receipt)
                result = {"status": "interrupted", "value": receipt}
            wall = monotonic() - begun
            result.update(
                wall_seconds=wall,
                cpu_seconds=process_time() - cpu,
                budget_charge_seconds=wall,
                accounting_complete=True,
            )
            if interrupt is not None:
                seal(path, result | {"identity": bound, "budget_restarted": False})
                raise interrupt
        result |= {"identity": bound, "budget_restarted": False}
        seal(path, result)
        return result


def _read_operations(folder: Path, identity: dict) -> dict:
    return previous._read_operations(folder, identity)


def _cost(operations: dict) -> dict:
    """Unknown partial counters stay unknown; observed values are lower bounds."""
    result = {
        "operations": len(operations),
        "accounting_complete": all(
            o["accounting_complete"] for o in operations.values()
        ),
        "budget_charge_seconds": sum(
            o["budget_charge_seconds"] for o in operations.values()
        ),
        "scope": "Operation CPU/wall includes setup and progress writes; excludes "
        "final receipt sealing, report writing and scheduler wait. Search counts "
        "are oracle calls, not individual trajectories or RHS evaluations.",
    }
    for field in ("wall_seconds", "cpu_seconds"):
        values = [o.get(field) for o in operations.values()]
        result[field + "_observed"] = sum(v for v in values if v is not None)
        result[field + "_total"] = (
            sum(values) if all(v is not None for v in values) else None
        )
    result["stage_costs"] = {}
    for stage in ("training", "inference", "forecast", "information"):
        group = [
            o
            for o in operations.values()
            if o["identity"]["stage"] == stage
            or o["identity"]["stage"].endswith("_" + stage)
        ]
        result["stage_costs"][stage] = {
            "operations": len(group),
            **{
                field + "_observed": sum(o.get(field) or 0 for o in group)
                for field in ("wall_seconds", "cpu_seconds")
            },
        }
    # Subcategories are nested in operation totals, never additional charges.
    for field in (
        "callback_wall_seconds",
        "callback_cpu_seconds",
        "numerical_active_wall_seconds",
    ):
        observed, complete = [], True
        for op in operations.values():
            value = op.get("value") or op.get("observed_progress") or {}
            snapshots = (
                [value.get("search_timing"), value.get("verification_timing")]
                if "search_timing" in value
                else [value.get("timing")]
            )
            observed.extend(s[field] for s in snapshots if s and field in s)
            complete &= op["status"] == "complete"
        result[field + "_observed"] = sum(observed)
        result[field + "_total"] = sum(observed) if complete else None
    for field in (
        "search_calls_attempted",
        "search_calls_started",
        "search_calls_completed",
        "search_calls_failed",
        "verification_calls_attempted",
        "verification_calls_started",
        "verification_calls_completed",
        "verification_calls_failed",
        "diagnostic_calls_attempted",
        "diagnostic_calls_started",
        "diagnostic_calls_completed",
    ):
        values, observed = [], []
        for op in operations.values():
            value = op.get("value") or {}
            progress = op.get("observed_progress") or {}
            terminal = op["status"] == "complete"
            values.append(value.get(field, 0) if terminal else None)
            observed.append(value.get(field, progress.get(field, 0)))
        result[field + "_observed"] = sum(observed)
        result[field + "_total"] = (
            sum(values) if all(v is not None for v in values) else None
        )
        result["accounting_complete"] &= all(v is not None for v in values)
    return result


def _scores(prediction, row, duration, fit, policy):
    """Use equal late horizon and equal-length immediate windows."""
    if not prediction or not prediction.get("forecast_success"):
        return None
    if (
        prediction["trajectory_id"] != row.trajectory_id
        or tuple(prediction["time"]) != row.time
        or prediction["cutoff"] != duration
        or set(prediction["predictions"]) != {"y"}
    ):
        raise ValueError("multiroot score identity differs")
    time = np.asarray(row.time)
    values = np.asarray(prediction["predictions"]["y"])
    scale = fit["training_scales"]["y"]
    if (
        values.shape != time.shape
        or not np.isfinite(values).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("invalid forecast values/scale")
    error = (values - np.asarray(row.targets["y"])) / scale
    masks = {
        "common": time > policy.durations[-1],
        "immediate": (time > duration)
        & (time <= duration + policy.immediate_duration + 1e-12),
        "own_suffix": time > duration,
    }
    return {
        "forecast_sha256": public.content_sha256(prediction),
        "scores": {
            k: {
                "normalized_mse": float(np.mean(error[mask] ** 2)),
                "sample_count": int(mask.sum()),
            }
            for k, mask in masks.items()
        },
    }


def _window_metrics(duration: float, entries: list, policy: Policy) -> dict:
    ready = len(entries) == 4 and all(
        e["chosen_start"] is not None
        and len(e["attempts"]) == len(policy.root_starts)
        and all(a["scores"] is not None for a in e["attempts"])
        for e in entries
    )
    scores = {}
    root_spread = None
    if ready:
        root_spread = max(
            float(
                np.linalg.norm(
                    np.array([a["inference"]["roots"][n] for n in controls.ROOT_NAMES])
                    - np.array(
                        [b["inference"]["roots"][n] for n in controls.ROOT_NAMES]
                    )
                )
            )
            for e in entries
            for a in e["attempts"]
            for b in e["attempts"]
        )
        for horizon in ("common", "immediate", "own_suffix"):
            losses, spreads = [], []
            for e in entries:
                selected = e["attempts"][e["chosen_start"]]
                losses.append(selected["scores"]["scores"][horizon]["normalized_mse"])
                time = np.asarray(selected["forecast"]["time"])
                mask = (
                    time > policy.durations[-1]
                    if horizon == "common"
                    else (time > duration)
                    & (
                        (time <= duration + policy.immediate_duration + 1e-12)
                        if horizon == "immediate"
                        else True
                    )
                )
                predictions = [
                    np.asarray(a["forecast"]["predictions"]["y"])[mask]
                    / a["inference"]["training_scales"]["y"]
                    for a in e["attempts"]
                ]
                spreads.extend(
                    float(np.sqrt(np.mean((a - b) ** 2)))
                    for a in predictions
                    for b in predictions
                )
            scores[horizon] = {
                "mean_nmse": float(np.mean(losses)),
                "per_row_nmse": losses,
                "maximum_start_prediction_spread": max(spreads),
            }
    return {
        "duration": duration,
        "complete": ready,
        "rows_expected": 4,
        "scores": scores,
        "maximum_start_root_l2_spread": root_spread,
        "adequate": ready
        and all(
            scores[h]["mean_nmse"] <= policy.validation_nmse_limit
            and scores[h]["maximum_start_prediction_spread"]
            <= policy.start_prediction_spread_limit
            for h in ("common", "immediate")
        ),
    }


def select_duration(metrics: list[dict], policy: Policy) -> dict:
    """Validation only; root identifiability is reported, not used as a veto."""
    complete = [m["duration"] for m in metrics] == list(policy.durations) and all(
        m["complete"] for m in metrics
    )
    adequate = [m["duration"] for m in metrics if m["adequate"]]
    duration = min(adequate) if complete and adequate else None
    return {
        "status": "unavailable"
        if not complete
        else "selected"
        if duration is not None
        else "no_adequate_window",
        "duration": duration,
        "criterion": "shortest_adequate_common_and_immediate",
        "selection_data": "validation_only",
        "common_start": policy.durations[-1],
        "immediate_duration": policy.immediate_duration,
        "root_information_used_as_gate": False,
        "independent_per_training_run": True,
    }


def _value(operations: dict, key: str):
    op = operations.get(key)
    return op.get("value") if op and op["status"] == "complete" else None


def _windows(record, fit, operations, policy, split_key, durations):
    """Rebuild and validate each endpoint against its exact boundary and source."""
    if not fit or not fit.get("fit_success"):
        return []
    problem = kernel.PrefixFitProblem.model_validate(record["problem"])
    result = []
    for wi, duration in enumerate(durations):
        entries = []
        prefix_policy = kernel.PrefixPolicy(duration=duration)
        for row in PublicSplit.model_validate(record[split_key]).rows:
            prefix = kernel.make_prefix(row, prefix_policy)
            schedule = kernel.make_forecast_schedule(row, prefix_policy)
            attempts = []
            for si, start in enumerate(policy.root_starts):
                name = f"{split_key}_w{wi}_{row.trajectory_id}_s{si}"
                inf = _value(operations, name + "_inference")
                forecast = _value(operations, name + "_forecast")
                roots = dict(zip(controls.ROOT_NAMES, start, strict=True))
                if inf is not None and (
                    inf["problem_sha256"] != public.content_sha256(problem)
                    or inf["fit_sha256"] != public.content_sha256(fit)
                    or inf["prefix_sha256"] != public.content_sha256(prefix)
                    or inf["root_start"] != roots
                    or inf["cutoff"] != duration
                    or inf["shared_parameters"] != fit["shared_parameters"]
                    or inf["training_scales"] != fit["training_scales"]
                ):
                    raise ValueError("multiroot inference identity differs")
                if forecast is not None and (
                    not inf
                    or not inf["fit_success"]
                    or forecast["inference_sha256"] != public.content_sha256(inf)
                    or forecast["schedule_sha256"] != public.content_sha256(schedule)
                ):
                    raise ValueError("multiroot forecast identity differs")
                attempts.append(
                    {
                        "label": name,
                        "start": list(start),
                        "inference": inf,
                        "forecast": forecast,
                        "scores": _scores(forecast, row, duration, fit, policy),
                    }
                )
            chosen = previous._choose_prefix_start(attempts)
            info = _value(
                operations, f"{split_key}_w{wi}_{row.trajectory_id}_information"
            )
            if info is not None and (
                chosen is None
                or info["inference_sha256"]
                != public.content_sha256(attempts[chosen]["inference"])
            ):
                raise ValueError("multiroot information endpoint differs")
            entries.append(
                {
                    "trajectory_id": row.trajectory_id,
                    "attempts": attempts,
                    "chosen_start": chosen,
                    "information": info,
                }
            )
        result.append(
            {
                "duration": duration,
                "trajectories": entries,
                "metrics": _window_metrics(duration, entries, policy),
            }
        )
    return result


def _evaluation(record, fit, windows, followup, selection):
    """Retrospective root/clean error; labels never choose roots or durations."""
    if not fit or not fit.get("fit_success"):
        return {"available": False, "truth_used_for_selection": False}
    # Reference schema is owned by the controls module.
    return controls.evaluate(record, fit, windows, followup, selection)


def _assemble(record, identity, operations, policy, selection):
    fit = _value(operations, "training")
    if fit is not None and (
        fit["problem_sha256"] != public.content_sha256(record["problem"])
        or fit["training_sha256"] != public.content_sha256(record["training"])
    ):
        raise ValueError("multiroot training identity differs")
    windows = _windows(record, fit, operations, policy, "validation", policy.durations)
    expected = select_duration([w["metrics"] for w in windows], policy)
    if selection != expected:
        raise ValueError("multiroot retained selection differs")
    followup = _windows(
        record,
        fit,
        operations,
        policy,
        "followup",
        (selection["duration"],) if selection["status"] == "selected" else (),
    )
    names = {"training"}
    diagnostic_available, diagnostic_expected = 0, 0
    for split_key, group in (("validation", windows), ("followup", followup)):
        for wi, window in enumerate(group):
            for entry in window["trajectories"]:
                for attempt in entry["attempts"]:
                    names.add(attempt["label"] + "_inference")
                    if attempt["inference"] and attempt["inference"].get("fit_success"):
                        names.add(attempt["label"] + "_forecast")
                if entry["chosen_start"] is not None:
                    names.add(f"{split_key}_w{wi}_{entry['trajectory_id']}_information")
                    diagnostic_expected += 1
                    diagnostic_available += bool(
                        entry["information"]
                        and entry["information"]["information"]["available"]
                        and entry["information"]["optimum_check"]["available"]
                    )
    if set(operations) != names:
        raise ValueError("multiroot operation roster differs")
    complete = selection["status"] != "unavailable" and (
        selection["status"] != "selected" or followup[0]["metrics"]["complete"]
    )
    return {
        "identity": identity,
        "status": "complete" if complete else "incomplete",
        "training_fit": fit,
        "windows": windows,
        "selection": selection,
        "followup": followup,
        "diagnostics_expected": diagnostic_expected,
        "diagnostics_available": diagnostic_available,
        "evaluation": _evaluation(record, fit, windows, followup, selection),
        "operations": operations,
        "cost": _cost(operations),
        "test_data_opened": False,
    }


def _portable_analytic_path(path: tuple[str | int, ...]) -> bool:
    """Identify only postselection analytic-versus-numerical discrepancy leaves."""
    return (
        len(path) in (6, 7)
        and path[0] == "evaluation"
        and path[1] in ("validation", "followup")
        and type(path[2]) is int
        and path[2] >= 0
        and path[3] == "rows"
        and type(path[4]) is int
        and path[4] >= 0
        and (
            (len(path) == 6 and path[5] == "analytic_max_absolute_difference")
            or (
                len(path) == 7
                and path[5] == "all_start_analytic_max_absolute_differences"
                and type(path[6]) is int
                and path[6] >= 0
            )
        )
    )


def _portable_result_matches(
    saved: object,
    expected: object,
    path: tuple[str | int, ...] = (),
) -> bool:
    """Compare reconstructed JSON, allowing only platform analytic roundoff.

    The two postselection discrepancy fields may differ by at most 1e-15
    absolute (zero relative tolerance), covering the observed 4.44e-16
    cross-platform drift. This never relaxes scores, selections, parameters,
    receipt identities or hashes. Types, container shapes and all other values
    remain exact; nonfinite numbers and negative absolute errors are invalid.
    """
    if type(saved) is not type(expected):
        return False
    if _portable_analytic_path(path):
        return saved is None or (
            type(saved) is float
            and type(expected) is float
            and np.isfinite(saved)
            and np.isfinite(expected)
            and min(saved, expected) >= 0
            and abs(saved - expected) <= 1e-15
        )
    if isinstance(saved, dict) and isinstance(expected, dict):
        return (
            saved.keys() == expected.keys()
            and all(type(key) is str for key in saved)
            and all(
                _portable_result_matches(value, expected[key], (*path, key))
                for key, value in saved.items()
            )
        )
    if isinstance(saved, list) and isinstance(expected, list):
        return len(saved) == len(expected) and all(
            _portable_result_matches(left, right, (*path, index))
            for index, (left, right) in enumerate(zip(saved, expected, strict=True))
        )
    if type(saved) is float and not np.isfinite(saved):
        return False
    return (
        saved is None or type(saved) in (bool, int, float, str)
    ) and saved == expected


def _completed(folder, identity, record, policy):
    selection = read_seal(folder / "selection.json")
    operations = _read_operations(folder, identity)
    result = read_seal(folder / "result.json")
    expected = _assemble(record, identity, operations, policy, selection)
    if not _portable_result_matches(result, expected):
        raise ValueError("multiroot terminal artifact differs")
    return result


def _diagnostic(problem, fit, prefix, inf, policy, sigma, save):
    """Journal reservations outside bounded analytic work; count returned calls."""
    budget = verification._Budget(
        verification.ExecutionPolicy(
            numerical_seconds=2 * policy.prefix_numerical.verification_seconds,
            overall_seconds=2 * policy.prefix_numerical.verification_seconds + 20,
        )
    )
    result = {
        "inference_sha256": public.content_sha256(inf),
        "diagnostic_calls_attempted": 0,
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "integration_calls": 0,
    }

    def checkpoint():
        result["timing"] = budget.snapshot()
        budget.callback(lambda: save(result))

    for name, function in (
        (
            "information",
            lambda: controls.prefix_information(
                problem, fit, prefix, inf, policy.prefix_numerical, sigma
            ),
        ),
        (
            "optimum_check",
            lambda: controls.conditional_linear_optimum(
                problem, fit, prefix, inf, policy.prefix_numerical
            ),
        ),
    ):
        result["diagnostic_calls_attempted"] += 1
        checkpoint()
        value = budget.numerical(lambda deadline, function=function: function())
        result[name] = value
        for field in ("diagnostic_calls_started", "diagnostic_calls_completed"):
            result[field] += value[field]
        checkpoint()
        budget.check()
    result["timing"] = budget.snapshot()
    return result


def run_task(root: Path, index: int) -> dict:
    """Train once, tune on validation, freeze duration, assess fresh trajectories."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("unknown multiroot task index")
    task = plan["tasks"][index]
    record = data["records"][task["task_id"]]
    policy = Policy.model_validate(plan["policy"])
    identity = {"plan_sha256": public.content_sha256(plan), "task": task}
    folder = root / "results" / task["task_id"]
    with public._lock(folder):
        if (folder / "result.json").exists():
            return _completed(folder, identity, record, policy)
        operations = {}

        def operation(name, seconds, work):
            operations[name] = _operation(folder, identity, name, seconds, work)
            return _value(operations, name)

        problem = kernel.PrefixFitProblem.model_validate(record["problem"])
        execution = policy.execution
        search_wall = (
            execution.search_overall_seconds + execution.verification_overall_seconds
        )
        fit = operation(
            "training",
            search_wall,
            lambda save: execute.fit_training(
                problem,
                PublicSplit.model_validate(record["training"]),
                policy.training_numerical,
                execution,
                checkpoint=save,
            ),
        )

        def run_windows(split_key, durations):
            if not fit or not fit.get("fit_success"):
                return
            for wi, duration in enumerate(durations):
                pp = kernel.PrefixPolicy(duration=duration)
                for row in PublicSplit.model_validate(record[split_key]).rows:
                    prefix = kernel.make_prefix(row, pp)
                    schedule = kernel.make_forecast_schedule(row, pp)
                    attempts = []
                    for si, start in enumerate(policy.root_starts):
                        label = f"{split_key}_w{wi}_{row.trajectory_id}_s{si}"
                        inf = operation(
                            label + "_inference",
                            search_wall,
                            lambda save,
                            start=start,
                            prefix=prefix: execute.infer_prefix(
                                problem,
                                fit,
                                prefix,
                                policy.prefix_numerical,
                                execution,
                                checkpoint=save,
                                root_guesses=dict(
                                    zip(controls.ROOT_NAMES, start, strict=True)
                                ),
                            ),
                        )
                        if inf and inf.get("fit_success"):
                            operation(
                                label + "_forecast",
                                execution.verification_overall_seconds,
                                lambda save,
                                inf=inf,
                                schedule=schedule: verification.forecast(
                                    problem,
                                    fit,
                                    inf,
                                    schedule,
                                    policy.prefix_numerical,
                                    verification.ExecutionPolicy(
                                        numerical_seconds=policy.prefix_numerical.verification_seconds,
                                        overall_seconds=execution.verification_overall_seconds,
                                    ),
                                    checkpoint=save,
                                ),
                            )
                        attempts.append({"inference": inf})
                    chosen = previous._choose_prefix_start(attempts)
                    if chosen is not None:
                        inf = attempts[chosen]["inference"]
                        operation(
                            f"{split_key}_w{wi}_{row.trajectory_id}_information",
                            2 * policy.prefix_numerical.verification_seconds + 20,
                            lambda save, prefix=prefix, inf=inf: _diagnostic(
                                problem,
                                fit,
                                prefix,
                                inf,
                                policy,
                                task["noise_sigma"],
                                save,
                            ),
                        )

        run_windows("validation", policy.durations)
        windows = _windows(
            record, fit, operations, policy, "validation", policy.durations
        )
        selection = select_duration([w["metrics"] for w in windows], policy)
        # No follow-up targets or generating values influence this durable choice.
        seal(folder / "selection.json", selection)
        if selection["status"] == "selected":
            run_windows("followup", (selection["duration"],))
        operations = _read_operations(folder, identity)
        result = _assemble(record, identity, operations, policy, selection)
        seal(folder / "result.json", result)
        return result


def report(root: Path) -> dict:
    """Recompute selections/scores and retain missing runs and unknown costs."""
    plan, data = verify(root, runtime=False)
    policy = Policy.model_validate(plan["policy"])
    rows, operations = [], {}
    for task in plan["tasks"]:
        folder = root / "results" / task["task_id"]
        identity = {"plan_sha256": public.content_sha256(plan), "task": task}
        result = (
            _completed(folder, identity, data["records"][task["task_id"]], policy)
            if (folder / "result.json").exists()
            else {
                "status": "missing",
                "selection": None,
                "operations": _read_operations(folder, identity),
            }
        )
        rows.append(task | result)
        operations.update(
            {task["task_id"] + "/" + k: v for k, v in result["operations"].items()}
        )
    recorded = sum(r["status"] != "missing" for r in rows)
    cost = _cost(operations)
    if recorded != len(rows):
        cost["accounting_complete"] = False
        for k in list(cost):
            if k.endswith("_total"):
                cost[k] = None
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "all_results_recorded": recorded == len(rows),
        "expected": len(rows),
        "recorded": recorded,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "selection_counts": dict(
            Counter((r["selection"] or {}).get("status", "missing") for r in rows)
        ),
        "rows": rows,
        "cost": cost,
        "actual_construction_export_qualified": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Known-structure synthetic controls with exact initial observed "
        "readings. New follow-up trajectories use the sealed validation-selected "
        "duration separately for each training run; neither training start is "
        "selected by follow-up accuracy. Root information is conditional on fitted "
        "shared parameters, not confidence probability. No production changes.",
    }
    public._write(root / "summary.json", result)
    return result
