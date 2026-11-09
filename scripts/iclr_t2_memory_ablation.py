"""Measure the validation contribution of the frozen T2 meal-memory coupling."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from autoformalism.benchmarks.audited_release import read_seal
from autoformalism.fitting import prefix_initialization as prefix
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal.mechanism_checks import effective_trees
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas.prefix_initialization import PrefixFitProblem, PrefixPolicy
from autoformalism.schemas.public_fitting import PublicSplit
from scripts import iclr_frozen_t2_forecast as frozen

FROZEN_ROSTER = "5ceedcd94edf88288fc3a32a26c064643b512ea5675ae1fd6506645a7b755bc0"
CHILD = "c15-3"
CONTROL = "c15-4-zero-basal"
MEAL = "meal_rate_g_per_min"


def common_equations(child: PrefixFitProblem, control: PrefixFitProblem) -> dict:
    """Check the zero-gain restriction on every common equation and initializer."""
    candidate = child.base_candidate
    base = control.base_candidate
    ct = effective_trees(candidate, {"gain_M": 0.0})
    bt = effective_trees(base, {"basal_floor": 0.0})
    rules = child.graph.rules
    child_initials = [i for i in candidate.initial_conditions if i.state != "M"]
    child_params = {p.name: p for p in candidate.parameters}
    base_params = {p.name: p for p in base.parameters}
    # Dropping zero * tanh(I/eps) is safe here because the scale is positive.
    if (
        child.context != control.context
        or set(ct) != set(bt) | {"M"}
        or set(rules) != set(control.graph.rules) | {"M"}
        or set(child_params) - set(base_params) != {"k_M_rise", "tau_M", "gain_M"}
        or set(base_params) - set(child_params) != {"eps", "basal_floor"}
        or base_params["eps"].domain.value != "positive"
        or child_params["tau_M"].domain.value != "positive"
        or any(
            child_params[n] != base_params[n]
            for n in child_params.keys() & base_params.keys()
        )
        or candidate.constraints != base.constraints
        or [s for s in candidate.states if s.name != "M"] != list(base.states)
        or child_initials != list(base.initial_conditions)
        or any(rules[n] != r for n, r in control.graph.rules.items())
        or rules["M"].initial.mode != "map"
        or rules["M"].initial.expression != "0"
        or any(ast.dump(ct[n]) != ast.dump(bt[n]) for n in bt)
        or any(
            any(
                isinstance(node, ast.Name) and node.id == "M"
                for node in ast.walk(bt[n])
            )
            for n in bt
        )
    ):
        raise ValueError("Cannot establish the frozen nested memory/control relation")
    return {
        "common_expressions": len(bt),
        "insulin_at_zero_gain": ast.unparse(ct["I"]),
        "extra_state_rhs": ast.unparse(ct["M"]),
        "extra_initial": "0",
        "scope": (
            "Common equations, parameter domains and initial laws agree "
            "when both extra terms are disabled."
        ),
    }


def run(primary: Path, val_path: Path, output: Path) -> dict:
    result = sealed_read(primary / "summary.json")
    plan = sealed_read(primary / "plan.json")
    models = sealed_read(primary / "models.json")["models"]
    if (
        result["plan"] != plan["artifact_sha256"]
        or plan["design"] != json.loads(frozen.CONFIG.read_text())
        or plan["models_sha256"] != FROZEN_ROSTER
        or plan["models_sha256"] != public.content_sha256(models)
        or plan["runtime"] != frozen.runtime()
    ):
        raise ValueError("Requires the completed unchanged primary forecast experiment")
    e = plan["design"]["public_evaluation"]
    if (
        val_path.name != "val.json"
        or frozen.sha(val_path) != e["validation_file_sha256"]
    ):
        raise ValueError("Requires the frozen public validation split")
    data = read_seal(val_path)
    if public.content_sha256(data) != result["validation"]:
        raise ValueError("Different validation receipt")
    val = PublicSplit.model_validate(data)
    cohort = {x["task"]: x for x in models}
    trained = cohort[CHILD]
    child = PrefixFitProblem.model_validate(trained["problem"])
    parent = PrefixFitProblem.model_validate(cohort[CONTROL]["problem"])
    if (
        cohort[CONTROL]["shared"]["basal_floor"] != 0
        or trained["shared"]["gain_M"] == 0
    ):
        raise ValueError("Unexpected intact and control gain values")
    model, _, _ = frozen.compiled(child)
    policies = PrefixPolicy.model_validate(plan["prefix"])
    scales = plan["target_scales"]
    with public._lock(output / "audit-lock"):
        protocol = sealed_write(
            output / "plan.json",
            {
                "protocol": "retrospective-t2-frozen-coupling-ablation-1",
                "source_summary": result["artifact_sha256"],
                "source_code": frozen.sha(Path(__file__)),
                "model": trained["model_sha256"],
                "parameters": trained["parameter_sha256"],
                "nested_equations": common_equations(child, parent),
                "change": {"gain_M": 0.0},
                "roots": "reuse each intact child prefix estimate",
                "optimizer_calls": 0,
                "llm_calls": 0,
                "test_data_opened": False,
            },
        )
        rows = []
        for index, row in enumerate(val.rows):
            source = sealed_read(primary / "rows" / CHILD / f"{index:02d}.json")
            if (
                source["plan"] != plan["artifact_sha256"]
                or source["id"] != row.trajectory_id
                or source["task"] != CHILD
            ):
                raise ValueError("Different intact model/trajectory receipt")
            path = output / "rows" / f"{index:02d}.json"
            identity = {
                "plan": protocol["artifact_sha256"],
                "source": source["artifact_sha256"],
                "id": row.trajectory_id,
            }
            if path.exists():
                record = sealed_read(path)
                if any(record[k] != v for k, v in identity.items()):
                    raise ValueError("Different ablation receipt")
            else:
                intact = source["forecast"]
                schedule = prefix.make_forecast_schedule(row, policies)
                meals = np.asarray(schedule.external_inputs[MEAL])
                has_meal = bool(np.any(meals))
                try:
                    if not intact["forecast_success"]:
                        raise ValueError("Intact forecast unavailable")
                    params = (
                        trained["shared"]
                        | intact["inference"]["selected"]["roots"]
                        | {"gain_M": 0.0}
                    )
                    curves, gap = frozen.verify_curve(
                        model, prefix._trajectory(schedule), params, scales, e
                    )
                    pred = {
                        "forecast_success": True,
                        "time": list(schedule.time),
                        "cutoff": schedule.cutoff,
                        "trajectory_id": row.trajectory_id,
                        "predictions": {k: v.tolist() for k, v in curves.items()},
                    }
                    score = prefix.score_forecast(pred, row, schedule.cutoff, scales)
                    delta = max(
                        float(np.max(np.abs(curves[n] - intact["predictions"][n])))
                        / scales[n]
                        for n in scales
                    )
                    record = {
                        "status": "complete",
                        "has_meal": has_meal,
                        "score": score,
                        "intact_nmse": source["score"]["normalized_mse"],
                        "solver_gap": gap,
                        "max_standardized_change": delta,
                        "predictions": pred["predictions"],
                    }
                    if (
                        not has_meal
                        and delta > e["maximum_standardized_solver_difference"]
                    ):
                        raise ValueError("Zero-meal negative control disagrees")
                except (ValueError, RuntimeError, ArithmeticError, TimeoutError) as exc:
                    record = {
                        "status": "unavailable",
                        "has_meal": has_meal,
                        "reason": str(exc),
                    }
                record = sealed_write(path, identity | record)
            rows.append(record)
        complete = [r for r in rows if r["status"] == "complete"]
        scores = [r["score"]["normalized_mse"] for r in complete]
        total = sum(r["score"]["sample_count"] for r in complete)
        pooled = (
            sum(
                r["score"]["normalized_mse"] * r["score"]["sample_count"]
                for r in complete
            )
            / total
            if len(complete) == len(rows) and total
            else None
        )
        return sealed_write(
            output / "summary.json",
            {
                "plan": protocol["artifact_sha256"],
                "complete": len(complete),
                "planned": len(rows),
                "silenced_pooled_nmse": pooled,
                "silenced_trajectory": frozen.spread(scores),
                "rows": [
                    {
                        k: v
                        for k, v in r.items()
                        if k not in ("predictions", "plan", "source")
                    }
                    for r in rows
                ],
                "fixed_roots_and_other_coefficients": True,
                "official_test_data_opened": False,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    print(json.dumps(run(a.primary, a.validation_file, a.output), indent=2))
