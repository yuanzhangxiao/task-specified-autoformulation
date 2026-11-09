"""Freeze the returned T2 cohort and score eight-minute-prefix validation forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import monotonic

import numpy as np
from scipy.optimize import least_squares

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.expressions import compile_candidate
from autoformalism.fitting import prefix_initialization as prefix
from autoformalism.fitting import public_fitting as public
from autoformalism.fitting.fitter import _parameter_variable
from autoformalism.fitting.initialization_graph import apply_initialization_graph
from autoformalism.fitting.models import FitConfig
from autoformalism.fitting.simulation import simulate_trajectory
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/iclr_2027_phase_c_method_freeze_v1.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def runtime() -> dict:
    return {
        "packages": public._runtime(),
        "library": public._source_identity(),
        "evaluator": sha(Path(__file__)),
        "manifest": sha(CONFIG),
    }


def compiled(problem: PrefixFitProblem):
    """Certify that suffix execution uses only generated state and external inputs."""
    base = compile_candidate(problem.base_candidate, problem.context)
    if problem.context.lagged_targets or set(base.validated.forcing_symbols) - set(
        problem.context.external_inputs
    ):
        raise ValueError("Expected a free-running model with external forcing only")
    return apply_initialization_graph(base, problem.graph)


def freeze_fit(record: dict, training_ids: list[str]) -> dict:
    """Keep the shared coefficients and root medians, discarding row-specific fits."""
    problem = PrefixFitProblem.model_validate(record["problem"])
    model, guesses, audit = compiled(problem)
    roots = sorted(audit["trajectory_root_parameters"].values())
    frozen = record["best_sampled_admissible"]
    if (
        public.content_sha256(problem) != record["model_sha256"]
        or not frozen["sampled_domain_passed"]
        or public.content_sha256(frozen["parameters"]) != frozen["parameter_sha256"]
    ):
        raise ValueError("Different model, coefficient vector or sampled-domain result")
    point = frozen["parameters"]
    shared = sorted(set(model.parameter_names) - set(roots))
    expected = {f"trajectory:{i}:{n}" for n in roots for i in training_ids}
    if set(point) != expected | set(shared):
        raise ValueError("Incomplete shared coefficients or training roots")
    return {
        "task": record["task"],
        "problem": record["problem"],
        "model_sha256": record["model_sha256"],
        "parameter_sha256": frozen["parameter_sha256"],
        "training_nmse": frozen["training_nmse"],
        "shared": {n: point[n] for n in shared},
        "starts": {
            "graph_default": {n: guesses[n] for n in roots},
            "training_root_median": {
                n: median(point[f"trajectory:{i}:{n}"] for i in training_ids)
                for n in roots
            },
        },
    }


def prepare(root: Path, construction: Path, fits: Path) -> dict:
    """Seal all endpoints, inputs and scoring choices without opening validation."""
    design = json.loads(CONFIG.read_text())
    method = design["revised_method"]
    experiment = design["public_evaluation"]
    cp = sealed_read(construction / "plan.json")
    execution = sealed_read(construction / "execution.json")
    fp = read_seal(fits / "plan.json")
    inputs = read_seal(fits / "inputs.json")
    exported = read_seal(fits / "fitted-models.json")
    if (
        cp["artifact_sha256"] != method["construction_plan_sha256"]
        or cp["runtime_source"] != method["construction_runtime_sha256"]
        or execution["artifact_sha256"] != method["construction_execution_sha256"]
        or public.content_sha256(fp) != method["fit_plan_sha256"]
        or fp["source_sha256"] != method["fit_source_sha256"]
        or public.content_sha256(inputs) != fp["inputs_sha256"]
        or public.content_sha256(exported) != method["fit_export_sha256"]
        or exported["plan_sha256"] != method["fit_plan_sha256"]
        or exported["prefix_forecast_evaluated"]
        or exported["validation_or_test_access"]
        or public.content_sha256(inputs["training"]) != experiment["training_sha256"]
        or exported["training_sha256"] != experiment["training_sha256"]
        or exported["prefix_policy"]["duration"] != experiment["prefix_minutes"]
        or [r["task"] for r in exported["models"]] != method["tasks"]
    ):
        raise ValueError("Training sources or original experiment identities differ")
    training = PublicSplit.model_validate(inputs["training"])
    if training.name != "train":
        raise ValueError("A training split is required for the freeze")
    first = training.rows[0]
    scales = {
        n: float(np.std(np.concatenate([r.targets[n] for r in training.rows])))
        for n in first.targets
    }
    if set(scales) != {"Gp", "I", "U"} or min(scales.values()) <= 0:
        raise ValueError("Unexpected T2 channels or scales")
    ids = [r.trajectory_id for r in training.rows]
    models = [freeze_fit(record, ids) for record in exported["models"]]
    control = next(m for m in models if m["task"] == experiment["parent_control"])
    if control["shared"]["basal_floor"] != 0:
        raise ValueError("The parent-equivalent control must have zero basal amplitude")
    with public._lock(root / "freeze-lock"):
        sealed_write(root / "models.json", {"models": models})
        return sealed_write(
            root / "plan.json",
            {
                "design": design,
                "runtime": runtime(),
                "models_sha256": public.content_sha256(models),
                "prefix": exported["prefix_policy"],
                "target_scales": scales,
                "training_ids": ids,
                "test_data_opened": False,
            },
        )


def spread(values: list[float]) -> dict:
    if not values:
        return {"median": None, "mad": None}
    center = median(values)
    return {"median": center, "mad": median(abs(x - center) for x in values)}


def rollout(
    model, trajectory, params: dict, method: str, policy: dict, deadline: float
):
    settings = FitConfig(
        integration_method=method,
        relative_tolerance=policy["rtol"],
        absolute_tolerance=policy["atol"],
        maximum_integration_step=policy["maximum_step_minutes"],
        allow_derivative_regression=False,
    )
    sim = simulate_trajectory(
        model,
        trajectory,
        params,
        {},
        settings,
        reset_observed_states=False,
        deadline=deadline,
    )
    if not sim.success:
        raise ValueError(f"{method} rollout failed: {sim.message}")
    return sim.predictions


def verify_curve(model, trajectory, params, scales, policy) -> tuple[dict, float]:
    outputs = [
        rollout(
            model,
            trajectory,
            params,
            solver,
            policy,
            monotonic() + policy["verification_seconds_per_solver"],
        )
        for solver in policy["verification_solvers"]
    ]
    first, second = outputs
    difference = max(
        float(np.max(np.abs(first[n] - second[n]))) / scales[n] for n in scales
    )
    if (
        not np.isfinite(difference)
        or difference > policy["maximum_standardized_solver_difference"]
    ):
        raise ValueError(f"Independent solver difference {difference:g}")
    return first, difference


def infer_roots(model, frozen, observed: prefix.PrefixView, scales, policy) -> dict:
    """Choose a fresh root vector by verified prefix loss with deterministic ties."""
    if not isinstance(observed, prefix.PrefixView):
        raise TypeError("Only prefix observations may enter root inference")
    tr = prefix._trajectory(observed)
    roots = sorted(frozen["starts"]["graph_default"])
    specs = {s.name: s for s in model.validated.candidate.parameters}
    variables = [_parameter_variable(model, specs[n], FitConfig()) for n in roots]
    bounds = ([v.lower for v in variables], [v.upper for v in variables])
    numerical = policy["optimizer_tolerance"]

    def residual(predictions):
        return np.concatenate(
            [(predictions[n] - observed.targets[n]) / scales[n] for n in sorted(scales)]
        )

    attempts = []
    for name in policy["starts_in_tie_order"]:
        start = frozen["starts"][name]
        if set(start) != set(roots):
            raise ValueError("Inconsistent training-only root guesses")
        deadline = monotonic() + policy["prefix_seconds_per_start"]
        calls = 0

        def fun(vector, deadline=deadline):
            nonlocal calls
            calls += 1
            params = frozen["shared"] | dict(zip(roots, vector, strict=True))
            return residual(rollout(model, tr, params, "DOP853", policy, deadline))

        try:
            if roots:
                opt = least_squares(
                    fun,
                    [start[n] for n in roots],
                    bounds=bounds,
                    method="trf",
                    x_scale="jac",
                    max_nfev=policy["prefix_max_nfev_per_start"],
                    ftol=numerical,
                    gtol=numerical,
                    xtol=numerical,
                )
                values = dict(zip(roots, map(float, opt.x), strict=True))
                reason = str(opt.message)
            else:
                values, reason = {}, "no independent roots"
            checked, gap = verify_curve(
                model, tr, frozen["shared"] | values, scales, policy
            )
            loss = float(np.mean(residual(checked) ** 2))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite prefix error")
            attempts.append(
                {
                    "start": name,
                    "roots": values,
                    "prefix_nmse": loss,
                    "solver_gap": gap,
                    "stop": reason,
                    "calls": calls,
                }
            )
        except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
            attempts.append({"start": name, "error": str(error), "calls": calls})
    usable = [a for a in attempts if "roots" in a]
    return {
        "attempts": attempts,
        "selected": min(usable, key=lambda a: a["prefix_nmse"]) if usable else None,
    }


def forecast_row(frozen, observed, schedule, scales, policy) -> dict:
    """Forecast from a typed prefix and a target-free schedule. Scoring runs later."""
    if not isinstance(schedule, prefix.ForecastSchedule) or not isinstance(
        observed, prefix.PrefixView
    ):
        raise TypeError("Expected typed prefix and target-free forcing schedule")
    if prefix._boundary(observed) != prefix._boundary(
        schedule
    ) or prefix._prefix_forcing(observed, observed.time[-1]) != prefix._prefix_forcing(
        schedule, observed.time[-1]
    ):
        raise ValueError("Different prefix and forecast origins or forcing")
    problem = PrefixFitProblem.model_validate(frozen["problem"])
    model, _, _ = compiled(problem)
    inference = infer_roots(model, frozen, observed, scales, policy)
    selected = inference["selected"]
    result = {
        "inference": inference,
        "forecast_success": False,
        "trajectory_id": schedule.trajectory_id,
        "time": list(schedule.time),
        "cutoff": schedule.cutoff,
    }
    if selected is None:
        return {**result, "reason": "no verified prefix estimate"}
    try:
        curves, gap = verify_curve(
            model,
            prefix._trajectory(schedule),
            frozen["shared"] | selected["roots"],
            scales,
            policy,
        )
        return {
            **result,
            "forecast_success": True,
            "solver_gap": gap,
            "predictions": {k: v.tolist() for k, v in curves.items()},
        }
    except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as error:
        return {**result, "reason": str(error)}


def summary(plan, rows) -> dict:
    method = plan["design"]["revised_method"]
    e = plan["design"]["public_evaluation"]
    models = {}
    for task in [*method["tasks"], "persistence"]:
        available = [r for r in rows if r["task"] == task and r["score"]["available"]]
        planned = sum(r["task"] == task for r in rows)
        count = sum(r["score"]["sample_count"] for r in available)
        per = (
            {
                n: sum(
                    r["score"]["per_target_normalized_mse"][n]
                    * r["score"]["sample_count"]
                    for r in available
                )
                / count
                for n in plan["target_scales"]
            }
            if available and len(available) == planned
            else {}
        )
        models[task] = {
            "available": len(available),
            "planned": planned,
            "pooled_nmse": float(np.mean(list(per.values()))) if per else None,
            "per_target": per,
            "trajectory": spread([r["score"]["normalized_mse"] for r in available]),
            "trajectories_with_negative_target": sum(
                any(
                    v < -1e-8 * plan["target_scales"][n] for n, v in r["minima"].items()
                )
                for r in available
            ),
        }
    values = {(r["task"], r["id"]): r["score"]["normalized_mse"] for r in rows}
    child, control = e["primary_child"], e["parent_control"]
    differences = {}
    for r in rows:
        if r["task"] == child:
            a, b = values[(child, r["id"])], values[(control, r["id"])]
            differences[r["id"]] = a - b if a is not None and b is not None else None
    return {
        "models": models,
        "paired_child_minus_control": differences,
        "paired_spread": spread([d for d in differences.values() if d is not None]),
    }


def run(root: Path, validation_file: Path) -> dict:
    """Execute every frozen model/trajectory once, reusing its completed forecast."""
    plan = sealed_read(root / "plan.json")
    frozen = sealed_read(root / "models.json")["models"]
    if (
        plan["runtime"] != runtime()
        or plan["models_sha256"] != public.content_sha256(frozen)
        or plan["design"] != json.loads(CONFIG.read_text())
    ):
        raise ValueError("Frozen model roster, evaluator or method contract differs")
    e = plan["design"]["public_evaluation"]
    if (
        validation_file.name != "val.json"
        or sha(validation_file) != e["validation_file_sha256"]
    ):
        raise ValueError("Expected the pinned public validation file")
    val = PublicSplit.model_validate(read_seal(validation_file))
    ids = [r.trajectory_id for r in val.rows]
    if (
        val.name != "val"
        or len(set(ids)) != len(ids)
        or set(ids) & set(plan["training_ids"])
        or not ids
    ):
        raise ValueError("Expected distinct nontraining validation trajectories")
    policy = PrefixPolicy.model_validate(plan["prefix"])
    scales = plan["target_scales"]
    rows = []
    with public._lock(root / "forecast-lock"):
        split_id = public.content_sha256(val)
        for index, row in enumerate(val.rows):
            observed = prefix.make_prefix(row, policy)
            schedule = prefix.make_forecast_schedule(row, policy)
            for task, record in [(m["task"], m) for m in frozen] + [
                ("persistence", None)
            ]:
                path = root / "rows" / task / f"{index:02d}.json"
                started = path.with_suffix(".started.json")
                identity = {
                    "plan": plan["artifact_sha256"],
                    "validation": split_id,
                    "id": row.trajectory_id,
                    "task": task,
                }
                if path.exists():
                    result = sealed_read(path)
                    if any(result.get(k) != v for k, v in identity.items()):
                        raise ValueError("Different forecast receipt")
                else:
                    pred = {
                        "time": list(row.time),
                        "cutoff": observed.cutoff,
                        "trajectory_id": row.trajectory_id,
                        "forecast_success": False,
                    }
                    if started.exists():
                        if any(
                            sealed_read(started).get(k) != v
                            for k, v in identity.items()
                        ):
                            raise ValueError("Different forecast reservation")
                        pred["reason"] = "interrupted"
                    else:
                        sealed_write(started, identity)
                        if record is None:
                            pred.update(
                                forecast_success=True,
                                predictions={
                                    n: [vs[-1]] * len(schedule.time)
                                    for n, vs in observed.targets.items()
                                },
                            )
                        else:
                            pred = forecast_row(record, observed, schedule, scales, e)
                    score = prefix.score_forecast(pred, row, observed.cutoff, scales)
                    mask = np.array(schedule.time) > observed.cutoff
                    minima = (
                        {
                            n: float(np.min(np.array(v)[mask]))
                            for n, v in pred.get("predictions", {}).items()
                        }
                        if score["available"]
                        else {}
                    )
                    result = sealed_write(
                        path,
                        {
                            **identity,
                            "forecast": pred,
                            "score": score,
                            "minima": minima,
                        },
                    )
                rows.append(result)
        return sealed_write(
            root / "summary.json",
            {
                "plan": plan["artifact_sha256"],
                "validation": split_id,
                **summary(plan, rows),
                "official_test_data_opened": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("freeze")
    prep.add_argument("--construction", type=Path, required=True)
    prep.add_argument("--fits", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--validation-file", type=Path, required=True)
    for stage in (prep, score):
        stage.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "freeze":
        result = prepare(args.root, args.construction, args.fits)
        print(
            json.dumps(
                {
                    "plan": result["artifact_sha256"],
                    "tasks": result["design"]["revised_method"]["tasks"],
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(run(args.root, args.validation_file), indent=2))
