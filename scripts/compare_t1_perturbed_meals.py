#!/usr/bin/env python3
"""Frozen, exploratory meal probes for two Full and three Sol T1 endpoints.

This is post-hoc diagnostic evaluation, not fitting or a benchmark test update.
Physical probes regenerate all public auxiliaries; a separate conditional probe
holds them fixed and has no physical reference accuracy score.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from autoformalism.data import TrainingScaler
from autoformalism.expressions import ValidationContext, compile_candidate
from autoformalism.fitting import public_fitting as public
from autoformalism.rebuttal import dalla_man as reference
from autoformalism.rebuttal.prefit_replay import sealed_read, sealed_write
from autoformalism.schemas import CandidateModel
from autoformalism.schemas.public_fitting import PublicSplit
from scripts.probe_t1_interventions import effect_error
from scripts.replay_t1_curves import SETTINGS, replay_one, write_csv

CELL = "phase_b_anonymous_system_t1_perturbed_obfuscated_easy"
NAMED_CELL = "phase_b_dalla_man_t1_perturbed_named_easy"
IDS = (
    "perturbed_full_r12_seed0",
    "perturbed_full_r12_seed1",
    *(f"anonymous_system_t1_perturbed_obfuscated_easy_sol{i}" for i in range(3)),
)
ALIASES = {
    "Gp": "v01",
    "EGP": "v02",
    "Uii": "v03",
    "E": "v04",
    "Gt": "v05",
    "meal_event_g": "u01",
}


def identity() -> dict:
    """Bind resume to both trusted numerical code and analysis code."""
    return {
        "source_sha256": public._source_identity(),
        "runtime": public._runtime(),
        "scripts": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "compare_t1_perturbed_meals.py",
                "replay_t1_curves.py",
                "probe_t1_interventions.py",
            )
        },
    }


def cases() -> list[dict]:
    """Fix every requested schedule, including necessary matched controls."""
    rows = []
    for horizon in (300, 900):
        rows.append(
            {
                "id": f"fasting_{horizon}",
                "duration": horizon,
                "meals": [],
                "control": None,
                "comparison": None,
                "is_control": True,
            }
        )
    for grams in (30, 60, 120):
        rows.append(
            {
                "id": f"single_{grams}",
                "duration": 300,
                "meals": [[60, grams]],
                "control": "fasting_300",
                "comparison": None,
                "is_control": False,
            }
        )
    for gap in (15, 30, 60, 120):
        rows.append(
            {
                "id": f"split_gap{gap}",
                "duration": 300,
                "meals": [[60, 30], [60 + gap, 30]],
                "control": "fasting_300",
                "comparison": "single_60",
                "is_control": False,
            }
        )
    rows.append(
        {
            "id": "single_60_long",
            "duration": 900,
            "meals": [[60, 60]],
            "control": "fasting_900",
            "comparison": None,
            "is_control": True,
        }
    )
    rows.append(
        {
            "id": "washout_pair",
            "duration": 900,
            "meals": [[60, 60], [540, 60]],
            "control": "fasting_900",
            "comparison": "single_60_long",
            "is_control": False,
        }
    )
    return rows


def freeze(models_path: Path, public_path: Path, root: Path) -> dict:
    """Freeze exact previously discussed endpoints without score-based selection."""
    raw = json.loads(models_path.read_text())
    selected = [m for m in raw if m["label"] in IDS]
    if len(selected) != 5 or {m["label"] for m in selected} != set(IDS):
        raise ValueError("require both Full seeds and all three Sol repetitions")
    models = []
    for m in sorted(selected, key=lambda x: IDS.index(x["label"])):
        if m["cell"] != CELL or m["context"]["lagged_targets"]:
            raise ValueError("wrong cell or measured-target forcing")
        candidate = CandidateModel.model_validate(m["candidate"])
        compiled = compile_candidate(
            candidate, ValidationContext.model_validate(m["context"])
        )
        if set(compiled.parameter_names) != set(m["parameters"]):
            raise ValueError("frozen parameter inventory differs")
        models.append(
            {
                "id": m["label"],
                "candidate": m["candidate"],
                "context": m["context"],
                "parameters": m["parameters"],
                "initials": m.get("initials", {}),
                "candidate_sha256": public.content_sha256(candidate),
                "parameter_sha256": public.content_sha256(m["parameters"]),
            }
        )
    public_plan = sealed_read(public_path)
    cell = public_plan["cells"][NAMED_CELL]
    train = public.unpack_split(PublicSplit.model_validate(cell["training"]))
    scale = TrainingScaler().fit(train).scales["target:Gp"].standard_deviation
    plan = {
        "protocol": "t1-perturbed-frozen-meal-comparison-1",
        "identity": identity(),
        "models_file_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
        "public_plan_sha256": public_plan["artifact_sha256"],
        "models": models,
        "cases": cases(),
        "training_scale": scale,
        "scale_source": (
            "Named perturbed T1 public training; exact physical channel alias"
        ),
        "reference_variant": "perturbed_b1",
        "reference_parameters": asdict(reference.DallaManParameters()),
        "reference_solvers": ["Radau", "DOP853"],
        "reference_rtol": 1e-10,
        "reference_atol": 1e-12,
        "reference_max_step": 0.5,
        "agreement_tolerance": 2e-5,
        "dt": 1.0,
        "replay_settings": SETTINGS.model_dump(mode="json"),
        "conditional_probe": {
            "pulse": "single_60",
            "auxiliaries": "fasting_300",
            "physical_accuracy_score": False,
        },
        "metrics": [
            "absolute_mse",
            "training_variance_normalized_mse",
            "response_relative_squared_error_against_matched_fasting",
            "schedule_difference_relative_squared_error",
        ],
        "input_semantics": (
            "Reference: exact stomach jumps. Models: original linearly interpolated "
            "one-minute public meal pulses; no model reset."
        ),
        "scope": (
            "Post-hoc equation-directed exploratory stress test; all five frozen "
            "models and all declared cases reported. "
            "No claim of untouched test evaluation."
        ),
        "fitting_performed": False,
        "test_trajectories_opened": False,
        "live_llm_calls": 0,
        "reference_used_for_diagnostic": True,
    }
    return sealed_write(root / "plan.json", plan)


def generate_reference(case: dict, plan: dict, method: str) -> dict:
    """Regenerate target and all permitted auxiliaries from perturbed physics."""

    def solve(*args, **kwargs):
        kwargs.update(
            method=method,
            rtol=plan["reference_rtol"],
            atol=plan["reference_atol"],
            max_step=plan["reference_max_step"],
        )
        return solve_ivp(*args, **kwargs)

    with patch.object(reference, "solve_ivp", solve):
        result = reference.simulate_dalla_man(
            meals=tuple(tuple(x) for x in case["meals"]),
            duration=case["duration"],
            dt=plan["dt"],
            variant=plan["reference_variant"],
            parameters=reference.DallaManParameters(**plan["reference_parameters"]),
        )
    return {
        "trajectory_id": case["id"],
        "time": result.time.tolist(),
        "targets": {"v01": result.states[:, reference.STATE_INDEX["Gp"]].tolist()},
        "auxiliaries": {
            "v02": result.derived["EGP"].tolist(),
            "v03": result.derived["Uii"].tolist(),
            "v04": result.derived["E"].tolist(),
            "v05": result.states[:, reference.STATE_INDEX["Gt"]].tolist(),
        },
        "external_inputs": {"u01": result.meal_event_g.tolist()},
    }


def conditional_row(pulse: dict, control: dict) -> dict:
    """Change only input forcing; the control target is not intervention truth."""
    if pulse["time"] != control["time"]:
        raise ValueError("conditional probe grids differ")
    row = copy.deepcopy(control)
    row["trajectory_id"] = "conditional_single60"
    row["external_inputs"] = copy.deepcopy(pulse["external_inputs"])
    return row


def trajectory(row: dict):
    """Construct a public diagnostic trajectory with a content-bound identity."""
    return public.unpack_split(
        PublicSplit.model_validate(
            {
                "name": "val",
                "fingerprint": public.content_sha256(row),
                "rows": [row],
            }
        )
    ).trajectories[0]


def run(root: Path) -> dict:
    """Resume exact frozen rollouts and expose failures without partial scores."""
    with public._lock(root):
        plan = sealed_read(root / "plan.json")
        if plan["identity"] != identity():
            raise ValueError("source/runtime changed after freeze")
        refs = {}
        for case in plan["cases"]:
            path = root / "references" / f"{case['id']}.json"
            if path.exists():
                refs[case["id"]] = sealed_read(path)["row"]
                continue
            a, b = [
                generate_reference(case, plan, method)
                for method in plan["reference_solvers"]
            ]
            diff = {
                key: float(np.max(np.abs(np.asarray(values) - b[role][key])))
                for role in ("targets", "auxiliaries")
                for key, values in a[role].items()
            }
            if max(diff.values()) > plan["agreement_tolerance"]:
                raise ValueError(f"independent reference solvers disagree: {diff}")
            refs[case["id"]] = sealed_write(
                path, {"row": a, "solver_max_differences": diff}
            )["row"]
            print("reference", case["id"], "verified", flush=True)
        refs["conditional_single60"] = conditional_row(
            refs["single_60"], refs["fasting_300"]
        )
        rows, curves, conditional = [], [], []
        for job in plan["models"]:
            model = compile_candidate(
                CandidateModel.model_validate(job["candidate"]),
                ValidationContext.model_validate(job["context"]),
            )
            records = {
                name: replay_one(
                    model,
                    trajectory(row),
                    job["parameters"],
                    job["initials"],
                    "v01",
                    plan["training_scale"],
                    root / "replays" / job["id"] / f"{name}.json",
                )
                for name, row in refs.items()
                if name != "conditional_single60"
            }
            for case in plan["cases"]:
                r = records[case["id"]]
                response = (
                    effect_error(r, records[case["control"]])
                    if case["control"]
                    else None
                )
                difference = (
                    effect_error(r, records[case["comparison"]])
                    if case["comparison"]
                    else None
                )
                rows.append(
                    {
                        "model_id": job["id"],
                        "case_id": case["id"],
                        "success": r["success"],
                        "nmse": r["nmse"],
                        "mse": r["nmse"] * plan["training_scale"] ** 2
                        if r["success"]
                        else None,
                        "response": response,
                        "schedule_difference": difference,
                    }
                )
                for i, time in enumerate(r["time"]):
                    curves.append(
                        {
                            "model_id": job["id"],
                            "case_id": case["id"],
                            "time_min": time,
                            "reference": r["observed"][i],
                            "predicted": r["predicted"][i] if r["success"] else None,
                            "reference_delta": response["reference_delta"][i]
                            if response and response["status"] == "complete"
                            else None,
                            "predicted_delta": response["predicted_delta"][i]
                            if response and response["status"] == "complete"
                            else None,
                        }
                    )
            cp = root / "conditional" / f"{job['id']}.json"
            if cp.exists():
                cr = sealed_read(cp)
            else:
                # The target values provide an initial observation only. There is no
                # physically consistent truth for this conditional forcing experiment.
                temp = root / "conditional-raw" / f"{job['id']}.json"
                cr = replay_one(
                    model,
                    trajectory(refs["conditional_single60"]),
                    job["parameters"],
                    job["initials"],
                    "v01",
                    plan["training_scale"],
                    temp,
                )
                cr = {
                    k: v
                    for k, v in cr.items()
                    if k not in {"artifact_sha256", "observed", "nmse"}
                }
                cr["scope"] = "Conditional input response; no physical accuracy score"
                base = records["fasting_300"]
                cr["delta"] = (
                    (np.asarray(cr["predicted"]) - base["predicted"]).tolist()
                    if cr["success"] and base["success"]
                    else None
                )
                cr = sealed_write(cp, cr)
                temp.unlink()
                temp.with_suffix(".started.json").unlink()
            conditional.append(
                {
                    "model_id": job["id"],
                    "success": cr["success"],
                    "min_delta": min(cr["delta"]) if cr["delta"] is not None else None,
                    "max_delta": max(cr["delta"]) if cr["delta"] is not None else None,
                }
            )
            print("model", job["id"], "completed", flush=True)
        write_csv(root / "curves.csv", curves)
        write_csv(
            root / "metrics.csv",
            [
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"response", "schedule_difference"}
                }
                | {
                    "response_relative_squared_error": (row["response"] or {}).get(
                        "relative_squared_error"
                    ),
                    "schedule_difference_relative_squared_error": (
                        row["schedule_difference"] or {}
                    ).get("relative_squared_error"),
                }
                for row in rows
            ],
        )
        return sealed_write(
            root / "summary.json",
            {
                "plan_sha256": plan["artifact_sha256"],
                "status": "complete",
                "models": len(plan["models"]),
                "cases": len(plan["cases"]),
                "rows": rows,
                "conditional": conditional,
                "fitting_performed": False,
                "test_trajectories_opened": False,
                "live_llm_calls": 0,
                "scope": plan["scope"],
            },
        )


def main() -> None:
    """Run freeze and replay separately so the design precedes new outcomes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--public-plan", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.models is None or args.public_plan is None:
            parser.error("freeze requires --models and --public-plan")
        result = freeze(args.models, args.public_plan, args.root)
    else:
        result = run(args.root)
    print(
        json.dumps(
            {k: result[k] for k in ("artifact_sha256", "status") if k in result},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
