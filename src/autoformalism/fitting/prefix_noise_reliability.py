"""M47: paired prefix policies across independent observation-noise acquisitions.

These known-equation controls do not change released datasets or fitting defaults.
Both choices are sealed before any follow-up inference, scoring or truth access.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from autoformalism.benchmarks.audited_release import read_seal, seal
from autoformalism.fitting import prefix_multiroot_campaign as base
from autoformalism.fitting import prefix_multiroot_controls as controls
from autoformalism.fitting import prefix_predictive_support as support
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.prefix_multiroot_campaign import Policy as MultirootPolicy
from autoformalism.schemas.base import FiniteFloat, StrictSchema
from autoformalism.schemas.public_fitting import PublicSplit

PROTOCOL = "phase-c-prefix-noise-reliability-1"
ARMS = ("baseline", "noise_aware")
CLEAN_REPEAT = 99


class Policy(StrictSchema):
    """Freeze acquisitions and both rules without inspecting follow-up outcomes."""

    protocol: Literal["phase-c-prefix-noise-reliability-policy-1"] = (
        "phase-c-prefix-noise-reliability-policy-1"
    )
    base: MultirootPolicy = Field(default_factory=MultirootPolicy)
    noise_repeats: tuple[int, ...] = (100, 101, 102, 103, 104)
    objective_rise_units: FiniteFloat = Field(default=4, gt=0, le=100)
    support_rms_limit: FiniteFloat = Field(default=0.1, gt=0)
    support_seconds: FiniteFloat = Field(default=5, gt=0, le=60)

    @model_validator(mode="after")
    def valid_repeats(self):
        if (
            not 1 <= len(self.noise_repeats) <= 20
            or self.noise_repeats != tuple(sorted(set(self.noise_repeats)))
            or any(
                type(r) is not int or not 100 <= r <= 2**32 - 1
                for r in self.noise_repeats
            )
        ):
            raise ValueError("require 1..20 distinct ordered acquisition IDs >=100")
        return self


def roster(policy: Policy) -> list[dict]:
    """Paired optimizer starts are not independent acquisition replications."""
    return [
        {
            "task_id": f"{scenario}_{case}_r{repeat}_s{seed}",
            "scenario": scenario,
            "case": case,
            "noise_sigma": sigma,
            "repeat": repeat,
            "seed": seed,
        }
        for scenario in controls.SCENARIOS
        for case, sigma, repeat in [
            ("clean", 0.0, CLEAN_REPEAT),
            *(("noisy", 0.01, r) for r in policy.noise_repeats),
        ]
        for seed in (0, 1)
    ]


def _fixture(task: dict) -> dict:
    """New fixed root roster, paired noise across scenarios, exact initial reading.

    Root configurations are drawn once from a declared seed, before experiments.
    Repeats vary observation noise only. Starts share every observed sample.
    """
    record = controls.fixture(task["scenario"], task["seed"], repeat=task["repeat"])
    roots_rng = np.random.default_rng(np.random.SeedSequence([20261020, 47, 1]))
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([20261020, 47, task["repeat"], 2])
    )
    record["noise_sigma"] = task["noise_sigma"]
    record["evaluation"]["roots"] = {}
    for key, label, count in (
        ("training", "train", 3),
        ("validation", "val", 4),
        ("followup", "followup", 4),
    ):
        roots = roots_rng.uniform(0.2, 2.0, size=(count, 2))
        clean = controls._split(label, roots, controls.RATES[task["scenario"]])
        measured = deepcopy(clean)
        for row, pair in zip(measured["rows"], roots, strict=True):
            values = np.asarray(row["targets"]["y"], float)
            values[1:] += noise_rng.normal(0, task["noise_sigma"], size=len(values) - 1)
            row["targets"]["y"] = values.tolist()
            record["evaluation"]["roots"][row["trajectory_id"]] = dict(
                zip(controls.ROOT_NAMES, pair.tolist(), strict=True)
            )
        record[key] = controls.fingerprint(measured)
        record["evaluation"]["clean"][key] = clean
    return record


def make_inputs(policy: Policy) -> dict:
    """Generate a new qualification acquisition; never read M46 follow-up data."""
    return {
        "protocol": "prefix-noise-reliability-inputs-1",
        "records": {t["task_id"]: _fixture(t) for t in roster(policy)},
    }


def _validate(data: dict, policy: Policy) -> None:
    """Verify public inputs and pairing; evaluator labels cannot select a policy."""
    tasks = roster(policy)
    if data["protocol"] != "prefix-noise-reliability-inputs-1" or set(
        data["records"]
    ) != {t["task_id"] for t in tasks}:
        raise ValueError("noise reliability input roster differs")
    for task in tasks:
        record = data["records"][task["task_id"]]
        if any(
            record[k] != task[k] for k in ("scenario", "seed", "repeat", "noise_sigma")
        ):
            raise ValueError("noise acquisition identity differs")
        if (
            record["test_data_opened"] is not False
            or record["followup_scope"] != "new_synthetic_qualification_trajectories"
            or base.kernel.PrefixFitProblem.model_validate(record["problem"])
            != controls.problem(task["scenario"], task["seed"])
        ):
            raise ValueError("noise control declaration differs")
        ids = set()
        for key, count, name in (
            ("training", 3, "train"),
            ("validation", 4, "val"),
            ("followup", 4, "val"),
        ):
            split = PublicSplit.model_validate(record[key])
            if (
                split.name != name
                or len(split.rows) != count
                or controls.fingerprint(record[key]) != record[key]
            ):
                raise ValueError("noise observation fingerprint or split differs")
            for row in split.rows:
                if (
                    row.trajectory_id in ids
                    or row.time != tuple(np.linspace(0, 4, 41))
                    or set(row.targets) != {"y"}
                    or row.targets["y"][0] != 0.4
                    or row.auxiliaries
                    or row.external_inputs
                    or row.fixed_covariates
                ):
                    raise ValueError("noise observation contract differs")
                ids.add(row.trajectory_id)
        pair = data["records"][task["task_id"].rsplit("_s", 1)[0] + "_s0"]
        if any(record[k] != pair[k] for k in ("training", "validation", "followup")):
            raise ValueError("optimizer starts must share identical observations")


def prepare(root: Path, policy: Policy) -> dict:
    """Seal acquisition arrays, policy and source once; never redraw on resume."""
    path = root / "inputs.json"
    data = read_seal(path) if path.exists() else make_inputs(policy)
    _validate(data, policy)
    plan = {
        "protocol": PROTOCOL,
        "policy": policy.model_dump(mode="json"),
        "tasks": roster(policy),
        "inputs_sha256": public.content_sha256(data),
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "test_data_opened": False,
        "live_llm_calls": 0,
        "production_default_changed": False,
        "actual_construction_export_qualified": False,
        "acquisitions": "New fixed roots; independently redrawn train/tuning/follow-up "
        "noise for each repeat, paired across scenarios and optimizer starts.",
    }
    with public._lock(root):
        seal(path, data)
        seal(root / "plan.json", plan)
    return {"identity": public.content_sha256(plan), "tasks": len(plan["tasks"])}


def verify(root: Path, *, runtime: bool = True) -> tuple[dict, dict]:
    """Validate immutable execution identity without regenerating evaluator truth."""
    plan, data = read_seal(root / "plan.json"), read_seal(root / "inputs.json")
    policy = Policy.model_validate(plan["policy"])
    _validate(data, policy)
    if (
        plan["protocol"] != PROTOCOL
        or plan["tasks"] != roster(policy)
        or plan["inputs_sha256"] != public.content_sha256(data)
        or plan["test_data_opened"] is not False
        or plan["live_llm_calls"] != 0
        or plan["production_default_changed"] is not False
        or plan["actual_construction_export_qualified"] is not False
    ):
        raise ValueError("noise reliability plan identity differs")
    if runtime and (
        plan["source_sha256"] != public._source_identity()
        or plan["runtime"] != public._runtime()
    ):
        raise ValueError("noise reliability source/runtime differs")
    return plan, data


def _horizons(time, duration: float, policy: Policy) -> dict[str, list[float]]:
    """Future timestamps only, never future measurements."""
    time = np.asarray(time)
    return {
        "common": time[time > policy.base.durations[-1]].tolist(),
        "immediate": time[
            (time > duration)
            & (time <= duration + policy.base.immediate_duration + 1e-12)
        ].tolist(),
    }


def _support_diagnostic(model, fit, prefix, inference, times, sigma, policy, save):
    """Charge analytical diagnostics and persistence in separate timing fields."""
    budget = base.verification._Budget(
        base.verification.ExecutionPolicy(
            numerical_seconds=policy.support_seconds,
            overall_seconds=policy.support_seconds + 20,
        )
    )
    result = {
        "inference_sha256": public.content_sha256(inference),
        "diagnostic_calls_attempted": 1,
        "diagnostic_calls_started": 0,
        "diagnostic_calls_completed": 0,
        "integration_calls": 0,
    }

    def checkpoint():
        result["timing"] = budget.snapshot()
        budget.callback(lambda: save(result))

    checkpoint()
    value = budget.numerical(
        lambda deadline: support.predictive_support(
            model,
            fit,
            prefix,
            inference,
            policy.base.prefix_numerical,
            sigma,
            _horizons(times, prefix.cutoff, policy),
            objective_rise_units=policy.objective_rise_units,
            maximum_seconds=max(1e-12, deadline - monotonic()),
        )
    )
    result["support"] = value
    for key in ("diagnostic_calls_started", "diagnostic_calls_completed"):
        result[key] = value[key]
    checkpoint()
    budget.check()
    result["timing"] = budget.snapshot()
    return result


def _windows(record, fit, operations, policy, split, durations):
    """Use unchanged prefix estimates/scoring, attaching separate support receipts."""
    windows = base._windows(record, fit, operations, policy.base, split, durations)
    for wi, window in enumerate(windows):
        for entry in window["trajectories"]:
            receipt = base._value(
                operations, f"{split}_w{wi}_{entry['trajectory_id']}_support"
            )
            if receipt is not None and (
                entry["chosen_start"] is None
                or receipt["inference_sha256"]
                != public.content_sha256(
                    entry["attempts"][entry["chosen_start"]]["inference"]
                )
            ):
                raise ValueError("predictive support endpoint differs")
            if receipt is not None:
                inf = entry["attempts"][entry["chosen_start"]]["inference"]
                model = base.kernel.PrefixFitProblem.model_validate(record["problem"])
                row = next(
                    r
                    for r in PublicSplit.model_validate(record[split]).rows
                    if r.trajectory_id == entry["trajectory_id"]
                )
                prefix = base.kernel.make_prefix(
                    row, base.kernel.PrefixPolicy(duration=window["duration"])
                )
                expected = {
                    "problem_sha256": public.content_sha256(model),
                    "fit_sha256": public.content_sha256(fit),
                    "prefix_sha256": public.content_sha256(prefix),
                    "inference_sha256": public.content_sha256(inf),
                    "numerical_policy_sha256": public.content_sha256(
                        policy.base.prefix_numerical
                    ),
                    "horizon_times_sha256": public.content_sha256(
                        _horizons(row.time, window["duration"], policy)
                    ),
                    "known_noise_sigma": record["noise_sigma"],
                    "objective_rise_units": policy.objective_rise_units,
                    "future_observations_used": False,
                    "truth_used": False,
                    "roots_replaced": False,
                }
                if any(receipt["support"].get(k) != v for k, v in expected.items()):
                    raise ValueError("predictive support identity differs")
            entry["predictive_support"] = receipt
        values = [e["predictive_support"] for e in window["trajectories"]]
        complete = len(values) == 4 and all(
            v and v["support"]["available"] for v in values
        )
        bounds = {
            h: (
                max(
                    v["support"]["horizons"][h]["normalized_rms_upper_bound"]
                    for v in values
                )
                if complete
                and all(
                    v["support"]["horizons"][h]["normalized_rms_upper_bound"]
                    is not None
                    for v in values
                )
                else None
            )
            for h in ("common", "immediate")
        }
        window["support_metrics"] = {
            "complete": complete,
            "maximum_rms_bound": bounds,
            "adequate": complete
            and all(
                b is not None and b <= policy.support_rms_limit for b in bounds.values()
            ),
        }
    return windows


def select_policies(windows: list[dict], policy: Policy) -> dict:
    """Freeze the baseline and added uncertainty rule before follow-up scoring."""
    metrics = [w["metrics"] for w in windows]
    baseline = base.select_duration(metrics, policy.base)
    strengthened = [
        m
        | {
            "complete": m["complete"] and w["support_metrics"]["complete"],
            "adequate": m["adequate"] and w["support_metrics"]["adequate"],
        }
        for m, w in zip(metrics, windows, strict=True)
    ]
    aware = base.select_duration(strengthened, policy.base) | {
        "criterion": "baseline_plus_conditional_predictive_support",
        "objective_rise_units": policy.objective_rise_units,
        "support_rms_limit": policy.support_rms_limit,
        "confidence_probability": None,
        "shared_parameter_uncertainty_included": False,
    }
    return {"baseline": baseline, "noise_aware": aware}


def _followup_durations(selections: dict) -> tuple[float, ...]:
    """Deduplicate identical frozen choices without sharing observed outcomes."""
    return tuple(
        sorted(
            {s["duration"] for s in selections.values() if s["status"] == "selected"}
        )
    )


def _operation_names(fit, windows, followup) -> set[str]:
    names = {"training"}
    if not fit or not fit.get("fit_success"):
        return names
    for split, group in (("validation", windows), ("followup", followup)):
        for wi, window in enumerate(group):
            for entry in window["trajectories"]:
                for attempt in entry["attempts"]:
                    names.add(attempt["label"] + "_inference")
                    if attempt["inference"] and attempt["inference"].get("fit_success"):
                        names.add(attempt["label"] + "_forecast")
                if entry["chosen_start"] is not None:
                    name = f"{split}_w{wi}_{entry['trajectory_id']}"
                    names.update((name + "_information", name + "_support"))
    return names


def _cost(operations: dict) -> dict:
    result = base._cost(operations)
    values = [o for name, o in operations.items() if name.endswith("_support")]
    result["stage_costs"]["predictive_support"] = {
        "operations": len(values),
        **{
            f + "_observed": sum(o.get(f) or 0 for o in values)
            for f in ("cpu_seconds", "wall_seconds")
        },
    }
    result["pairing"] = (
        "Fitting/inference shared by both rules, charged once; "
        "duplicate selected durations reuse follow-up work."
    )
    return result


def _assemble(record, identity, operations, policy, selections):
    """Reconstruct every decision; never let retrospective truth alter a choice."""
    fit = base._value(operations, "training")
    if fit is not None and (
        fit["problem_sha256"] != public.content_sha256(record["problem"])
        or fit["training_sha256"] != public.content_sha256(record["training"])
    ):
        raise ValueError("noise training identity differs")
    windows = _windows(
        record, fit, operations, policy, "validation", policy.base.durations
    )
    if selections != select_policies(windows, policy):
        raise ValueError("noise retained selections differ")
    followup = _windows(
        record, fit, operations, policy, "followup", _followup_durations(selections)
    )
    if set(operations) != _operation_names(fit, windows, followup):
        raise ValueError("noise operation roster differs")
    outcomes, evaluations = {}, {}
    for arm, choice in selections.items():
        selected = [w for w in followup if w["duration"] == choice["duration"]]
        metrics = selected[0]["metrics"] if selected else None
        outcomes[arm] = {
            "selection": choice,
            "followup": metrics,
            "forecast_adequate": metrics["adequate"] if metrics else None,
            "followup_support": selected[0]["support_metrics"] if selected else None,
        }
        evaluations[arm] = base._evaluation(record, fit, windows, selected, choice)
    complete = all(
        o["selection"]["status"] != "unavailable"
        and (
            o["selection"]["status"] != "selected"
            or bool(o["followup"] and o["followup"]["complete"])
        )
        for o in outcomes.values()
    )
    entries = [
        e
        for w in windows + followup
        for e in w["trajectories"]
        if e["chosen_start"] is not None
    ]
    return {
        "identity": identity,
        "status": "complete" if complete else "incomplete",
        "training_fit": fit,
        "windows": windows,
        "selections": selections,
        "followup": followup,
        "outcomes": outcomes,
        "evaluations": evaluations,
        "diagnostics_expected": len(entries),
        "diagnostics_available": sum(
            bool(
                e["information"]
                and e["information"]["information"]["available"]
                and e["information"]["optimum_check"]["available"]
            )
            for e in entries
        ),
        "predictive_support_available": sum(
            bool(
                e["predictive_support"]
                and e["predictive_support"]["support"]["available"]
            )
            for e in entries
        ),
        "operations": operations,
        "cost": _cost(operations),
        "test_data_opened": False,
    }


def _completed(folder, identity, record, policy):
    saved = read_seal(folder / "result.json")
    expected = _assemble(
        record,
        identity,
        base._read_operations(folder, identity),
        policy,
        read_seal(folder / "selections.json"),
    )
    # Only derived analytic-vs-numerical discrepancy fields receive the M46
    # portability allowance. Scores, selections, ledgers and identities stay exact.
    if not isinstance(saved.get("evaluations"), dict) or set(
        saved["evaluations"]
    ) != set(ARMS):
        raise ValueError("noise terminal evaluator roster differs")
    for arm in ARMS:
        if not base._portable_result_matches(
            {"evaluation": saved["evaluations"][arm]},
            {"evaluation": expected["evaluations"][arm]},
        ):
            raise ValueError("noise terminal evaluator differs")
    if not base._portable_result_matches(
        {k: v for k, v in saved.items() if k != "evaluations"},
        {k: v for k, v in expected.items() if k != "evaluations"},
    ):
        raise ValueError("noise terminal artifact differs")
    return saved


def run_task(root: Path, index: int) -> dict:
    """Execute one paired experiment, preserving all durable attempt charges."""
    plan, data = verify(root)
    if type(index) is not int or not 0 <= index < len(plan["tasks"]):
        raise ValueError("noise task index out of range")
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
            print(f"M47: {task['task_id']} {name}", flush=True)
            operations[name] = base._operation(folder, identity, name, seconds, work)
            return base._value(operations, name)

        model = base.kernel.PrefixFitProblem.model_validate(record["problem"])
        p, execution = policy.base, policy.base.execution
        search_wall = (
            execution.search_overall_seconds + execution.verification_overall_seconds
        )
        fit = operation(
            "training",
            search_wall,
            lambda save: base.execute.fit_training(
                model,
                PublicSplit.model_validate(record["training"]),
                p.training_numerical,
                execution,
                checkpoint=save,
            ),
        )

        def run_windows(split, durations):
            if not fit or not fit.get("fit_success"):
                return
            for wi, duration in enumerate(durations):
                pp = base.kernel.PrefixPolicy(duration=duration)
                for row in PublicSplit.model_validate(record[split]).rows:
                    prefix = base.kernel.make_prefix(row, pp)
                    schedule = base.kernel.make_forecast_schedule(row, pp)
                    attempts = []
                    for si, start in enumerate(p.root_starts):
                        label = f"{split}_w{wi}_{row.trajectory_id}_s{si}"
                        inf = operation(
                            label + "_inference",
                            search_wall,
                            lambda save,
                            start=start,
                            prefix=prefix: base.execute.infer_prefix(
                                model,
                                fit,
                                prefix,
                                p.prefix_numerical,
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
                                schedule=schedule: base.verification.forecast(
                                    model,
                                    fit,
                                    inf,
                                    schedule,
                                    p.prefix_numerical,
                                    base.verification.ExecutionPolicy(
                                        numerical_seconds=p.prefix_numerical.verification_seconds,
                                        overall_seconds=execution.verification_overall_seconds,
                                    ),
                                    checkpoint=save,
                                ),
                            )
                        attempts.append({"inference": inf})
                    chosen = base.previous._choose_prefix_start(attempts)
                    if chosen is not None:
                        inf = attempts[chosen]["inference"]
                        name = f"{split}_w{wi}_{row.trajectory_id}"
                        operation(
                            name + "_information",
                            2 * p.prefix_numerical.verification_seconds + 20,
                            lambda save, prefix=prefix, inf=inf: base._diagnostic(
                                model,
                                fit,
                                prefix,
                                inf,
                                p,
                                task["noise_sigma"],
                                save,
                            ),
                        )
                        operation(
                            name + "_support",
                            policy.support_seconds + 20,
                            lambda save,
                            prefix=prefix,
                            inf=inf,
                            times=row.time: _support_diagnostic(
                                model,
                                fit,
                                prefix,
                                inf,
                                times,
                                task["noise_sigma"],
                                policy,
                                save,
                            ),
                        )

        run_windows("validation", p.durations)
        windows = _windows(record, fit, operations, policy, "validation", p.durations)
        selections = select_policies(windows, policy)
        # Both rules are frozen before reading any follow-up observations.
        seal(folder / "selections.json", selections)
        run_windows("followup", _followup_durations(selections))
        result = _assemble(
            record,
            identity,
            base._read_operations(folder, identity),
            policy,
            selections,
        )
        seal(folder / "result.json", result)
        return result


def _groups(rows: list[dict]) -> list[dict]:
    """Keep complete denominators and separate starts from acquisition repeats."""
    groups = []
    for scenario in controls.SCENARIOS:
        for case in ("clean", "noisy"):
            runs = [r for r in rows if r["scenario"] == scenario and r["case"] == case]
            for arm in ARMS:
                outcomes = [r.get("outcomes", {}).get(arm) for r in runs]
                scored = [
                    o
                    for o in outcomes
                    if o and o["followup"] and o["followup"]["complete"]
                ]
                evaluated = [
                    r["evaluations"][arm]["followup"][0]
                    for r in runs
                    if r.get("evaluations", {}).get(arm, {}).get("followup")
                    and r["evaluations"][arm]["followup"][0]["complete"]
                ]
                root_errors = [
                    e["root_errors"]
                    for evaluation in evaluated
                    for e in evaluation["rows"]
                ]
                groups.append(
                    {
                        "scenario": scenario,
                        "case": case,
                        "policy": arm,
                        "expected_runs": len(runs),
                        "independent_acquisitions": len({r["repeat"] for r in runs}),
                        "recorded_runs": sum(r["status"] != "missing" for r in runs),
                        "selection_counts": dict(
                            Counter(
                                o["selection"]["status"] if o else "missing"
                                for o in outcomes
                            )
                        ),
                        "selected_duration_counts": dict(
                            Counter(
                                str(o["selection"]["duration"])
                                for o in outcomes
                                if o and o["selection"]["status"] == "selected"
                            )
                        ),
                        "scored_runs": len(scored),
                        "forecast_adequate_runs": sum(
                            o["forecast_adequate"] for o in scored
                        ),
                        "mean_followup_nmse": {
                            h: float(
                                np.mean(
                                    [
                                        o["followup"]["scores"][h]["mean_nmse"]
                                        for o in scored
                                    ]
                                )
                            )
                            if scored
                            else None
                            for h in ("common", "immediate")
                        },
                        "root_evaluated_runs": len(evaluated),
                        "root_evaluated_trajectories": len(root_errors),
                        "mean_root_l2_error": float(
                            np.mean([e["root_l2_error"] for e in root_errors])
                        )
                        if root_errors
                        else None,
                        "maximum_root_l2_error": max(
                            (e["root_l2_error"] for e in root_errors), default=None
                        ),
                        "maximum_absolute_root_sum_error": max(
                            (abs(e["sum_root_error"]) for e in root_errors),
                            default=None,
                        ),
                        "interpretation": "Two optimizer starts per acquisition; "
                        "counts are not independent success probabilities.",
                    }
                )
    return groups


def report(root: Path) -> dict:
    """Portable immutable reconstruction, with missing runs kept in denominators."""
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
                "operations": base._read_operations(folder, identity),
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
        for key in list(cost):
            if key.endswith("_total"):
                cost[key] = None
    result = {
        "protocol": PROTOCOL,
        "plan_sha256": public.content_sha256(plan),
        "status": "complete"
        if all(r["status"] == "complete" for r in rows)
        else "incomplete",
        "expected": len(rows),
        "recorded": recorded,
        "all_results_recorded": recorded == len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "selection_counts": {
            a: dict(
                Counter(
                    r.get("selections", {}).get(a, {}).get("status", "missing")
                    for r in rows
                )
            )
            for a in ARMS
        },
        "policies": list(ARMS),
        "groups": _groups(rows),
        "rows": rows,
        "cost": cost,
        "actual_construction_export_qualified": False,
        "test_data_opened": False,
        "live_llm_calls": 0,
        "limitation": "Known-structure synthetic controls and exact initial readings; "
        "Fresh independently drawn noise with fixed new root configurations. "
        "Support bounds condition on fitted shared theta and relax root bounds. "
        "No confidence probability, universal policy or production qualification. "
        "Shared numerical work is charged once; follow-up never selects either rule.",
    }
    public._write(root / "summary.json", result)
    return result
